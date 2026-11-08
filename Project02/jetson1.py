import cv2
import numpy as np
import json
import time
import threading
import subprocess
import wave
import sounddevice as sd
import os

from collections import deque
from scipy.io.wavfile import write
from scipy.signal import resample_poly
from piper import PiperVoice
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# 환경 변수 로드 (.env 파일이 있는 경우 인증 토큰 등 불러오기)
load_dotenv()


# =====================================================
# 1. Jetson 기본 설정 (수정 반영)
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

def project_path(path):
    """프로젝트 루트 기준 경로를 절대 경로로 변환한다."""
    p = Path(path)
    return p if p.is_absolute() else BASE_DIR / p


# [수정 반영] 퀀타이즈 적용된 TensorRT 엔진 파일 경로 (기존 .pt 대신 .engine 사용 시)
YOLO_MODEL_PATH = project_path("shoplifting_wights.engine")

# 웹사이트 예제(05-LLM_and_Gemma.ipynb) 기준 Hugging Face 모델 ID 또는 로컬 GGUF 경로 설정
MODEL_ID = "google/gemma-3-4b-it"
HF_TOKEN = os.getenv("HF_TOKEN")

# 로컬 Llama 모델을 사용할 경우의 경로
GEMMA_MODEL_PATH = project_path(
    "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
)

# LLM 구동 방식 선택 ("api" 또는 "local")
LLM_MODE = "api"

# 카메라 설정
USE_CSI_CAMERA = False
VIDEO_SOURCE = 0

# CSI 카메라 해상도 / FPS
CSI_WIDTH = 1280
CSI_HEIGHT = 720
CSI_FPS = 30

# 결과 저장
OUTPUT_PATH = project_path("outputs/processed.mp4")
LOG_PATH = project_path("outputs/conversation.jsonl")
TRANSCRIPT_PATH = project_path("outputs/conversation.txt")

# 화면 / 추론 설정
WIDTH = 960
YOLO_IMAGE_SIZE = 640
YOLO_DEVICE = 0       # Jetson CUDA GPU

# [수정 반영] 퀀타이즈 모델(.engine) 사용 시 내부 정밀도가 지정되므로 half는 False로 설정
YOLO_HALF = False     
YOLO_CONF = 0.05

SHOP_CONF = 0.60
NORMAL_CONF = 0.80

FRAME_WINDOW = 12
MIN_DETECTIONS = 5

EVENT_COOLDOWN = 60
ENABLE_VOICE = True

# Gemma 메모리 사용량 조절 (Local 모드용)
GEMMA_GPU_LAYERS = 20
GEMMA_CONTEXT = 1024
GEMMA_BATCH = 64
GEMMA_THREADS = 4

# 오디오 설정
MIC_DEVICE = None

# [수정 반영] 스피커 필수 지정: 오디오 장치 목록(index) 확인 후 번호 입력 (예: 2)
# None일 경우 시스템 기본 출력 장치 사용
SPEAKER_DEVICE = None 

RECORD_SECONDS = 5
RECORD_SAMPLE_RATE = 44100
WHISPER_SAMPLE_RATE = 16000

AUDIO_FILE = project_path("src/audio/input.wav")

# [수정 반영] 상위 폴더 구조에 맞춘 Whisper 실행 파일 및 모델 경로
WHISPER_PATH = project_path("../whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = project_path("../whisper.cpp/models/ggml-base.bin")

PIPER_MODEL = project_path("src/models/Piper/ko_KR-kss-medium.onnx")
OUTPUT_FILE = project_path("src/audio/response.wav")

# 얼굴 검출 모델
FACE_MODEL = project_path(
    "src/models/face_detection_yunet_2023mar.onnx"
)

# 한글 폰트 후보
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

WINDOW_NAME = "Local Shoplifting Detection System"

# 최근 4개 발화를 영상 자막으로 표시
subtitle_lines = deque(maxlen=4)
subtitle_lock = threading.Lock()

# 종료 / 대화 중복 실행 방지
stop_event = threading.Event()
voice_busy = threading.Event()


# =====================================================
# 2. 폴더 및 파일 사전 검사
# =====================================================

(project_path("outputs")).mkdir(parents=True, exist_ok=True)
(project_path("src/audio")).mkdir(parents=True, exist_ok=True)


def require_file(path, description):
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"{description} 파일을 찾을 수 없습니다: {path}"
        )


require_file(YOLO_MODEL_PATH, "YOLO 퀀타이즈 모델(.engine)")
require_file(WHISPER_PATH, "Whisper 실행 파일")
require_file(WHISPER_MODEL, "Whisper 모델")
require_file(PIPER_MODEL, "Piper 음성 모델")
require_file(FACE_MODEL, "YuNet 얼굴 검출 모델")

if LLM_MODE == "local":
    require_file(GEMMA_MODEL_PATH, "Gemma 모델")


# =====================================================
# 3. 한글 폰트
# =====================================================

def load_korean_font(size):
    for font_path in FONT_CANDIDATES:
        if Path(font_path).is_file():
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                pass

    print(
        "[경고] 한글 폰트를 찾지 못했습니다. "
        "자막 한글이 깨지면 Noto CJK 폰트를 설치하세요.",
        flush=True
    )
    return ImageFont.load_default()


STATUS_FONT = load_korean_font(20)
SUBTITLE_FONT = load_korean_font(18)


# =====================================================
# 4. 얼굴 모자이크 설정
# =====================================================

if not hasattr(cv2, "FaceDetectorYN"):
    raise RuntimeError(
        "현재 OpenCV에 FaceDetectorYN이 없습니다. "
        "YuNet을 지원하는 OpenCV 빌드가 필요합니다."
    )

face_detector = cv2.FaceDetectorYN.create(
    str(FACE_MODEL),
    "",
    (320, 320),
    0.7,
    0.3,
    5000
)


def apply_face_mosaic(frame):
    """YuNet으로 얼굴을 검출하고 모자이크 처리한다."""
    height, width = frame.shape[:2]
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(frame)

    if faces is None:
        return frame

    for face in faces:
        x, y, w, h = face[:4].astype(int)

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)

        if x2 <= x1 or y2 <= y1:
            continue

        face_region = frame[y1:y2, x1:x2]

        small = cv2.resize(
            face_region,
            (max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)),
            interpolation=cv2.INTER_LINEAR
        )

        mosaic = cv2.resize(
            small,
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_NEAREST
        )

        frame[y1:y2, x1:x2] = mosaic

    return frame


# =====================================================
# 5. YOLO 모델
# =====================================================

print("[초기화] YOLO 퀀타이즈 모델 로드 중...", flush=True)

model = YOLO(str(YOLO_MODEL_PATH))

print("[초기화] YOLO 로드 성공", flush=True)
print("[YOLO 모델 경로]", YOLO_MODEL_PATH, flush=True)
print("[YOLO 클래스]", model.names, flush=True)


def get_class_id(target_name):
    for class_id, name in model.names.items():
        if name.lower() == target_name.lower():
            return int(class_id)
    return None


SHOP_ID = get_class_id("shoplifting")
NORMAL_ID = get_class_id("not_shoplifting")

if SHOP_ID is None:
    raise ValueError("shoplifting 클래스가 모델에 없습니다.")

if NORMAL_ID is None:
    raise ValueError("not_shoplifting 클래스가 모델에 없습니다.")


# =====================================================
# 6. 로그 및 자막 관리
# =====================================================

log_lock = threading.Lock()

ui_status = {
    "status": "대기 중"
}
ui_status_lock = threading.Lock()


def set_ui_status(status=None):
    with ui_status_lock:
        if status is not None:
            ui_status["status"] = status


def save_log(category, content):
    """JSONL 형식으로 로그를 저장한다."""
    data = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "category": category,
        "content": content
    }

    with log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def add_conversation(speaker, message):
    """영상 자막 표시와 TXT 대화 기록 저장을 수행한다."""
    if not message:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with subtitle_lock:
        subtitle_lines.append((speaker, message))

    with log_lock:
        with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {speaker}: {message}\n")

    print(f"[대화 저장] {speaker}: {message}", flush=True)


def draw_status(frame):
    """상태 표시와 최근 대화 자막을 영상에 그린다."""
    with ui_status_lock:
        status = ui_status["status"]

    with subtitle_lock:
        lines = list(subtitle_lines)

    image = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    draw = ImageDraw.Draw(image)

    frame_width, frame_height = image.size

    # 상단 상태 표시
    draw.rectangle(
        (10, 45, frame_width - 10, 85),
        fill=(0, 0, 0)
    )
    draw.text(
        (20, 52),
        f"상태: {status}",
        font=STATUS_FONT,
        fill=(255, 255, 0)
    )

    # 하단 대화 자막
    if lines:
        line_height = 30
        box_height = len(lines) * line_height + 20
        top = frame_height - box_height

        draw.rectangle(
            (0, top, frame_width, frame_height),
            fill=(20, 20, 20)
        )

        y = top + 10

        for speaker, message in lines:
            text = f"{speaker}: {message}"

            while (
                text and
                draw.textlength(text, font=SUBTITLE_FONT)
                > frame_width - 40
            ):
                text = text[:-1]

            if text != f"{speaker}: {message}" and len(text) > 3:
                text = text[:-3] + "..."

            draw.text(
                (20, y),
                text,
                font=SUBTITLE_FONT,
                fill=(255, 255, 255)
            )
            y += line_height

    return cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )


# =====================================================
# 7. LLM 모델 초기화 및 호출
# =====================================================

llm_client = None
local_llm = None

if LLM_MODE == "api":
    print("[초기화] Hugging Face InferenceClient 설정 중...", flush=True)
    if not HF_TOKEN:
        print("[경고] HF_TOKEN이 환경 변수 또는 .env 파일에 설정되지 않았습니다.", flush=True)
    llm_client = InferenceClient(api_key=HF_TOKEN)
    print("[초기화] InferenceClient 설정 완료", flush=True)
else:
    print("[초기화] Gemma 로컬 모델(llama_cpp) 로드 중...", flush=True)
    from llama_cpp import Llama
    local_llm = Llama(
        model_path=str(GEMMA_MODEL_PATH),
        n_gpu_layers=GEMMA_GPU_LAYERS,
        n_ctx=GEMMA_CONTEXT,
        n_batch=GEMMA_BATCH,
        n_threads=GEMMA_THREADS,
        n_threads_batch=GEMMA_THREADS,
        verbose=False
    )
    print("[초기화] Gemma 로컬 로드 성공", flush=True)


def ask_gemma(messages):
    if LLM_MODE == "api":
        try:
            response = llm_client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=180,
                temperature=0.7
            )
            content = response.choices[0].message.content
            return (content or "").strip()
        except Exception as e:
            print("[Hugging Face API 오류]", repr(e), flush=True)
            return "죄송합니다. 시스템 통신 중 문제가 발생했습니다."
    else:
        response = local_llm.create_chat_completion(
            messages=messages,
            max_tokens=180,
            temperature=0.7
        )
        content = response["choices"][0]["message"]["content"]
        return (content or "").strip()


SYSTEM_PROMPT = """
너는 매장에서 고객을 응대하는 AI 안내원이야.

규칙:
1. 항상 정중하고 자연스러운 한국어를 사용한다.
2. 고객을 절도범이라고 단정하거나 비난하지 않는다.
3. 상품 확인이 필요하면 정중하게 도움을 제안한다.
4. 고객의 대답에 맞춰 짧고 자연스럽게 응답한다.
5. 불필요하게 반복해서 질문하지 않는다.
6. 문제가 해결되면 정중하게 대화를 마무리한다.
"""


# =====================================================
# 8. 오디오 장치 확인
# =====================================================

def print_audio_devices():
    print("\n========== 오디오 장치 목록 ==========", flush=True)
    try:
        devices = sd.query_devices()
        for index, device in enumerate(devices):
            print(
                f"[{index}] {device['name']} | "
                f"입력 {device['max_input_channels']}ch | "
                f"출력 {device['max_output_channels']}ch",
                flush=True
            )
    except Exception as e:
        print("[오디오 장치 조회 실패]", repr(e), flush=True)
    print("======================================\n", flush=True)


print_audio_devices()


# =====================================================
# 9. STT / TTS
# =====================================================

def speech_to_text():
    print("[STT 1] 녹음 시작", flush=True)
    set_ui_status("녹음 중... 말씀해 주세요.")

    try:
        audio = sd.rec(
            int(RECORD_SECONDS * RECORD_SAMPLE_RATE),
            samplerate=RECORD_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            device=MIC_DEVICE
        )

        sd.wait()
        print("[STT 2] 녹음 완료", flush=True)

        audio_16k = resample_poly(
            audio[:, 0],
            WHISPER_SAMPLE_RATE,
            RECORD_SAMPLE_RATE
        )

        audio_16k = np.clip(
            audio_16k,
            -32768,
            32767
        ).astype(np.int16)

        AUDIO_FILE.parent.mkdir(parents=True, exist_ok=True)

        write(
            str(AUDIO_FILE),
            WHISPER_SAMPLE_RATE,
            audio_16k
        )

        print("[STT 3] WAV 저장 완료", flush=True)

        set_ui_status("음성 인식 중...")

        output_txt = AUDIO_FILE.with_suffix(".txt")
        output_base = AUDIO_FILE.with_suffix("")

        if output_txt.exists():
            output_txt.unlink()

        command = [
            str(WHISPER_PATH),
            "-m", str(WHISPER_MODEL),
            "-f", str(AUDIO_FILE),
            "-l", "ko",
            "-nt",
            "-otxt",
            "-of", str(output_base)
        ]

        print("[STT 4] Whisper 실행:", " ".join(command), flush=True)

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(BASE_DIR)
        )

        print("[Whisper 종료 코드]", result.returncode, flush=True)

        if result.stdout:
            print("[Whisper stdout]", result.stdout, flush=True)
        if result.stderr:
            print("[Whisper stderr]", result.stderr, flush=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"Whisper 실행 실패: 종료 코드 {result.returncode}"
            )

        if not output_txt.exists():
            print("[Whisper] 결과 TXT가 생성되지 않았습니다.", flush=True)
            return ""

        question = output_txt.read_text(
            encoding="utf-8-sig"
        ).strip()

        print("[Whisper 인식 결과]", repr(question), flush=True)

        return question

    except Exception as e:
        print("[음성 인식 오류]", repr(e), flush=True)
        return ""

    finally:
        set_ui_status("음성 인식 완료")


print("[초기화] Piper 음성 모델 로드 중...", flush=True)
voice = PiperVoice.load(str(PIPER_MODEL))
print("[초기화] Piper 로드 성공", flush=True)


def text_to_speech(text):
    if not text:
        return

    set_ui_status("AI 말하는 중...")
    print("[TTS] 음성 생성 중...", flush=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(OUTPUT_FILE), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    with wave.open(str(OUTPUT_FILE), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(
            f"지원하지 않는 WAV 샘플 크기: {sample_width}"
        )

    audio_data = np.frombuffer(frames, dtype=np.int16)

    if channels > 1:
        audio_data = audio_data.reshape(-1, channels)

    # [수정 반영] 설정된 SPEAKER_DEVICE로 소리 출력
    sd.play(
        audio_data,
        samplerate=sample_rate,
        device=SPEAKER_DEVICE
    )
    sd.wait()

    print("[TTS] 음성 재생 완료", flush=True)
    set_ui_status("고객 응답 대기 중...")


# =====================================================
# 10. AI 음성 대화
# =====================================================

def conversation_worker():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    try:
        answer = (
            "고객님, 혹시 계산 전인 상품이 있으신가요? "
            "확인을 도와드리겠습니다."
        )

        messages.append({
            "role": "assistant",
            "content": answer
        })

        print("[AI 안내]", answer, flush=True)
        save_log("AI", answer)
        add_conversation("AI", answer)

        text_to_speech(answer)

        # 최대 3회 음성 대화
        for _ in range(3):
            if stop_event.is_set():
                break

            question = speech_to_text()
            print("[DEBUG] STT 반환:", repr(question), flush=True)

            if not question:
                print("[Whisper] 음성이 인식되지 않았습니다.", flush=True)
                continue

            print("[고객]", question, flush=True)
            save_log("USER", question)
            add_conversation("고객", question)

            messages.append({
                "role": "user",
                "content": question
            })

            set_ui_status("AI 답변 생성 중...")
            print("[DEBUG] Gemma 호출 시작", flush=True)

            answer = ask_gemma(messages)

            if not answer:
                answer = "죄송합니다. 다시 한 번 말씀해 주시겠어요?"

            print("[DEBUG] Gemma 응답 완료:", answer, flush=True)

            messages.append({
                "role": "assistant",
                "content": answer
            })

            save_log("AI", answer)
            add_conversation("AI", answer)

            text_to_speech(answer)

    except Exception as e:
        print("[음성 대화 오류]", repr(e), flush=True)
        save_log("ERROR", repr(e))

    finally:
        set_ui_status("대화 종료")
        voice_busy.clear()


# =====================================================
# 11. Jetson 카메라 입력
# =====================================================

def create_csi_pipeline():
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){CSI_WIDTH}, "
        f"height=(int){CSI_HEIGHT}, format=(string)NV12, "
        f"framerate=(fraction){CSI_FPS}/1 ! "
        f"nvvidconv ! "
        f"video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1"
    )


def open_camera():
    if USE_CSI_CAMERA:
        pipeline = create_csi_pipeline()
        print("[카메라] CSI 카메라 파이프라인 사용", flush=True)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        print("[카메라] USB 카메라 사용:", VIDEO_SOURCE, flush=True)
        cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(
            "카메라를 열 수 없습니다. USB 카메라 번호 또는 "
            "CSI 카메라/GStreamer 설정을 확인하세요."
        )

    return cap


# =====================================================
# 12. 영상 입력 및 출력
# =====================================================

cap = open_camera()

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)
cv2.resizeWindow(
    WINDOW_NAME,
    1280,
    900
)

writer = None

detection_history = deque(maxlen=FRAME_WINDOW)
last_event_time = 0

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or fps > 120:
    fps = 25

print("[시스템] 실시간 감지를 시작합니다. Q 키로 종료하세요.", flush=True)


# =====================================================
# 13. 메인 탐지 루프
# =====================================================

try:
    while not stop_event.is_set():
        ret, frame = cap.read()

        if not ret:
            print("[카메라] 프레임을 가져오지 못했습니다.", flush=True)
            time.sleep(0.05)
            continue

        height, width = frame.shape[:2]
        if width > WIDTH:
            new_height = int(height * WIDTH / width)
            frame = cv2.resize(
                frame,
                (WIDTH, new_height),
                interpolation=cv2.INTER_AREA
            )

        display_frame = apply_face_mosaic(frame.copy())

        try:
            results = model.predict(
                frame,
                conf=YOLO_CONF,
                imgsz=YOLO_IMAGE_SIZE,
                device=YOLO_DEVICE,
                half=YOLO_HALF,
                verbose=False
            )
        except Exception as e:
            print("[YOLO 추론 오류]", repr(e), flush=True)
            set_ui_status("YOLO 추론 오류")
            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        boxes = results[0].boxes

        shop_detected = False
        max_shop_conf = 0.0

        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                if class_id == SHOP_ID:
                    if confidence < SHOP_CONF:
                        continue

                    shop_detected = True
                    max_shop_conf = max(
                        max_shop_conf,
                        confidence
                    )

                    color = (0, 0, 255)
                    label = "SHOPLIFTING CHECK"

                elif class_id == NORMAL_ID:
                    if confidence < NORMAL_CONF:
                        continue

                    color = (0, 255, 0)
                    label = "NOT SHOPLIFTING"

                else:
                    continue

                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    display_frame,
                    f"{label} {confidence * 100:.1f}%",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        detection_history.append(shop_detected)

        detection_count = sum(detection_history)
        stable_detection = detection_count >= MIN_DETECTIONS

        current_time = time.time()

        if stable_detection:
            cv2.putText(
                display_frame,
                "AI DETECTION - CHECK REQUIRED",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            if current_time - last_event_time >= EVENT_COOLDOWN:
                last_event_time = current_time

                save_log(
                    "DETECTION",
                    {
                        "class": "shoplifting",
                        "confidence": max_shop_conf,
                        "message": "AI detection; human review required"
                    }
                )

                if ENABLE_VOICE and not voice_busy.is_set():
                    print("[EVENT] 음성 대화 시작!", flush=True)
                    voice_busy.set()

                    threading.Thread(
                        target=conversation_worker,
                        daemon=True
                    ).start()

        else:
            cv2.putText(
                display_frame,
                "MONITORING",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        display_frame = draw_status(display_frame)

        if writer is None and OUTPUT_PATH:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(OUTPUT_PATH),
                fourcc,
                fps,
                (
                    display_frame.shape[1],
                    display_frame.shape[0]
                )
            )

            if not writer.isOpened():
                print(
                    "[경고] 영상 저장 파일을 열 수 없습니다. "
                    "영상 저장을 비활성화합니다.",
                    flush=True
                )
                writer = None

        if writer is not None:
            writer.write(display_frame)

        cv2.imshow(WINDOW_NAME, display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

finally:
    stop_event.set()

    if voice_busy.is_set():
        print("[종료] 진행 중인 음성 대화 종료 대기...", flush=True)

    cap.release()

    if writer is not None:
        writer.release()

    try:
        sd.stop()
    except Exception:
        pass

    cv2.destroyAllWindows()

    print("[시스템] 프로그램 종료", flush=True)
    print("[저장 위치] 대화 기록:", TRANSCRIPT_PATH, flush=True)
    print("[저장 위치] JSON 로그:", LOG_PATH, flush=True)
    print("[저장 위치] 처리 영상:", OUTPUT_PATH, flush=True)
