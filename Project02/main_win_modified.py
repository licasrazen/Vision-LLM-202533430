
import cv2
import numpy as np
import json
import time
import threading
import subprocess
import wave
import sounddevice as sd

from collections import deque
from scipy.io.wavfile import write
from scipy.signal import resample_poly
from piper import PiperVoice
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from llama_cpp import Llama

# =====================================================
# 1. 기본 설정
# =====================================================

YOLO_MODEL_PATH = "shoplifting_wights.pt"
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"

# 0 = 웹캠
# 동영상 테스트 시 "res/inout1.mp4"
VIDEO_SOURCE = 0

OUTPUT_PATH = "outputs/processed.mp4"
LOG_PATH = "outputs/conversation.jsonl"
TRANSCRIPT_PATH = "outputs/conversation.txt"

WIDTH = 960

SHOP_CONF = 0.60
NORMAL_CONF = 0.80

FRAME_WINDOW = 12
MIN_DETECTIONS = 5

EVENT_COOLDOWN = 60
ENABLE_VOICE = True

# 최근 4개 발화를 영상 자막으로 표시
subtitle_lines = deque(maxlen=4)
subtitle_lock = threading.Lock()

# =====================================================
# 2. 폴더 생성
# =====================================================

Path("outputs").mkdir(parents=True, exist_ok=True)
Path("src/audio").mkdir(parents=True, exist_ok=True)

# =====================================================
# 3. 얼굴 모자이크 설정
# =====================================================

FACE_MODEL = "face_detection_yunet_2023mar.onnx"

face_detector = cv2.FaceDetectorYN.create(
    FACE_MODEL,
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
# 4. YOLO 모델
# =====================================================

model = YOLO(YOLO_MODEL_PATH)

print("YOLO 로드 성공")
print("YOLO 모델 경로:", YOLO_MODEL_PATH)
print("클래스:", model.names)


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
# 5. 로그 및 자막 관리
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
    """기존 JSONL 로그 저장"""
    data = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "category": category,
        "content": content
    }

    with log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def add_conversation(speaker, message):
    """자막 표시와 TXT 대화 기록 저장을 동시에 수행한다."""
    if not message:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 영상 자막에 추가
    with subtitle_lock:
        subtitle_lines.append((speaker, message))

    # TXT 파일에 전체 대화 누적 저장
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

    font = ImageFont.truetype(
        "C:/Windows/Fonts/malgun.ttf",
        20
    )

    subtitle_font = ImageFont.truetype(
        "C:/Windows/Fonts/malgun.ttf",
        18
    )

    frame_width, frame_height = image.size

    # -----------------------------
    # 상단 상태 표시
    # -----------------------------

    draw.rectangle(
        (10, 45, frame_width - 10, 85),
        fill=(0, 0, 0)
    )

    draw.text(
        (20, 52),
        f"상태: {status}",
        font=font,
        fill=(255, 255, 0)
    )

    # -----------------------------
    # 하단 대화 자막
    # -----------------------------

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
            # 너무 긴 문장은 화면 폭에 맞게 자른다.
            text = f"{speaker}: {message}"

            while (
                draw.textlength(text, font=subtitle_font)
                > frame_width - 40
            ):
                text = text[:-1]

            draw.text(
                (20, y),
                text,
                font=subtitle_font,
                fill=(255, 255, 255)
            )

            y += line_height

    return cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )


# =====================================================
# 6. Gemma 로컬 모델
# =====================================================

llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=2048,
    n_batch=32,
    n_ubatch=32,
    verbose=False
)


def ask_gemma(messages):
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=180,
        temperature=0.7
    )

    return response["choices"][0]["message"]["content"].strip()


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
# 7. STT / TTS
# =====================================================

MIC_DEVICE = 13

AUDIO_FILE = "src/audio/input.wav"
WHISPER_PATH = r"whisper.cpp\build\bin\Release\whisper-cli.exe"
WHISPER_MODEL = r"whisper.cpp\ggml-base.bin"

PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
OUTPUT_FILE = "src/audio/response.wav"

RECORD_SECONDS = 5


def speech_to_text():
    print("[DEBUG 1] 녹음 시작", flush=True)
    set_ui_status("녹음 중... 말씀해 주세요.")

    try:
        # 1. 마이크 녹음
        audio = sd.rec(
            int(RECORD_SECONDS * 44100),
            samplerate=44100,
            channels=1,
            dtype="int16",
            device=MIC_DEVICE
        )

        print("[DEBUG 2] 녹음 요청 완료", flush=True)

        sd.wait()

        print("[DEBUG 3] 녹음 완료", flush=True)

        # 2. 44.1kHz -> 16kHz 변환
        audio_16k = resample_poly(
            audio[:, 0],
            160,
            441
        )

        audio_16k = np.clip(
            audio_16k,
            -32768,
            32767
        ).astype(np.int16)

        # 3. WAV 저장
        audio_path = Path(AUDIO_FILE)
        audio_path.parent.mkdir(parents=True, exist_ok=True)

        write(
            str(audio_path),
            16000,
            audio_16k
        )

        print("[DEBUG 4] WAV 저장 완료", flush=True)

        # 4. Whisper 실행 준비
        set_ui_status("음성 인식 중...")

        output_txt = audio_path.with_suffix(".txt")
        output_base = output_txt.with_suffix("")

        if output_txt.exists():
            output_txt.unlink()

        print("[DEBUG 5] Whisper 실행 시작", flush=True)

        command = [
            WHISPER_PATH,
            "-m", WHISPER_MODEL,
            "-f", str(audio_path),
            "-l", "ko",
            "-nt",
            "-otxt",
            "-of", str(output_base)
        ]

        # 5. Whisper 실행
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        print("[Whisper 종료 코드]", result.returncode, flush=True)
        print("[Whisper stdout]", repr(result.stdout), flush=True)
        print("[Whisper stderr]", result.stderr, flush=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"Whisper 실행 실패: 종료 코드 {result.returncode}"
            )

        print("[DEBUG 6] Whisper 실행 완료", flush=True)

        # 6. TXT 파일에서 인식 결과 읽기
        if not output_txt.exists():
            print(
                "[Whisper 오류] 결과 TXT 파일이 생성되지 않았습니다.",
                flush=True
            )
            return ""

        question = output_txt.read_text(
            encoding="utf-8-sig"
        ).strip()

        print("[Whisper 원본 출력]", repr(question), flush=True)

        if not question:
            print("[Whisper] 음성이 인식되지 않았습니다.", flush=True)
            return ""

        print("[Whisper 인식 성공]", question, flush=True)

        return question

    except Exception as e:
        print("[음성 인식 오류]", repr(e), flush=True)
        return ""

    finally:
        set_ui_status("음성 인식 완료")


voice = PiperVoice.load(PIPER_MODEL)


def text_to_speech(text):
    set_ui_status("AI 말하는 중...")
    print("[TTS] 음성 생성 중...")

    with wave.open(OUTPUT_FILE, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    with wave.open(OUTPUT_FILE, "rb") as wav_file:
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

    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()

    print("[TTS] 음성 재생 완료")
    set_ui_status("고객 응답 대기 중...")


# =====================================================
# 8. AI 음성 대화
# =====================================================

voice_busy = threading.Event()


def conversation_worker():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    try:
        # 1. 첫 안내
        answer = (
            "고객님, 혹시 계산 전인 상품이 있으신가요? "
            "확인을 도와드리겠습니다."
        )

        messages.append({
            "role": "assistant",
            "content": answer
        })

        print("[AI 안내]", answer)

        save_log("AI", answer)
        add_conversation("AI", answer)

        text_to_speech(answer)

        # 2. 최대 3회 음성 대화
        for _ in range(3):
            question = speech_to_text()

            print("[DEBUG] STT 반환:", repr(question))

            if not question:
                print("[Whisper] 음성이 인식되지 않았습니다.")
                continue

            # 고객 발화 저장 및 자막 표시
            print("[고객]", question)

            save_log("USER", question)
            add_conversation("고객", question)

            messages.append({
                "role": "user",
                "content": question
            })

            # 3. Gemma 답변 생성
            set_ui_status("AI 답변 생성 중...")
            print("[DEBUG] Gemma 호출 시작")

            answer = ask_gemma(messages)

            print("[DEBUG] Gemma 응답 완료:", answer)

            messages.append({
                "role": "assistant",
                "content": answer
            })

            # AI 답변 저장 및 자막 표시
            print("[Gemma]", answer)

            save_log("AI", answer)
            add_conversation("AI", answer)

            # 4. AI 음성 재생
            text_to_speech(answer)

    except Exception as e:
        print("[음성 대화 오류]", e)
        save_log("ERROR", str(e))

    finally:
        set_ui_status("대화 종료")
        voice_busy.clear()


# =====================================================
# 9. 영상 입력 및 출력
# =====================================================

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError("영상 입력 장치를 열 수 없습니다.")

cv2.namedWindow(
    "Local Shoplifting Detection System",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Local Shoplifting Detection System",
    1280,
    900
)

writer = None

detection_history = deque(maxlen=FRAME_WINDOW)
last_event_time = 0

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25


# =====================================================
# 10. 메인 탐지 루프
# =====================================================

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 화면 크기 조절
    height, width = frame.shape[:2]

    if width > WIDTH:
        new_height = int(height * WIDTH / width)
        frame = cv2.resize(frame, (WIDTH, new_height))

    # YOLO 추론
    results = model.predict(
        frame,
        conf=0.05,
        imgsz=960,
        verbose=False
    )

    boxes = results[0].boxes

    shop_detected = False
    max_shop_conf = 0.0

    # 모자이크 처리용 프레임
    display_frame = apply_face_mosaic(frame.copy())

    # YOLO 결과 표시
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

                print("[YOLO] shoplifting:", confidence)

                shop_detected = True
                max_shop_conf = max(max_shop_conf, confidence)

                color = (0, 0, 255)
                label = "SHOPLIFTING CHECK"

            elif class_id == NORMAL_ID:
                print("[YOLO] not_shoplifting:", confidence)

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

    # =================================================
    # 11. 연속 프레임 판정
    # =================================================

    detection_history.append(shop_detected)

    print(
        f"[DEBUG] shop_detected={shop_detected}, "
        f"history={sum(detection_history)}/{FRAME_WINDOW}, "
        f"stable={sum(detection_history) >= MIN_DETECTIONS}"
    )

    stable_detection = (
        sum(detection_history) >= MIN_DETECTIONS
    )

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

            # 음성 대화 중복 실행 방지
            if ENABLE_VOICE and not voice_busy.is_set():
                print("[EVENT] 음성 대화 시작!")

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

    # =================================================
    # 12. 상태 및 대화 자막 표시
    # =================================================

    display_frame = draw_status(display_frame)

    # =================================================
    # 13. 화면 및 영상 저장
    # =================================================

    if writer is None and OUTPUT_PATH:
        writer = cv2.VideoWriter(
            OUTPUT_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (
                display_frame.shape[1],
                display_frame.shape[0]
            )
        )

    if writer is not None:
        writer.write(display_frame)

    cv2.imshow(
        "Local Shoplifting Detection System",
        display_frame
    )

    # Q 키로 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =====================================================
# 14. 종료
# =====================================================

cap.release()

if writer is not None:
    writer.release()

cv2.destroyAllWindows()

print("프로그램 종료")
print("대화 기록 저장 위치:", TRANSCRIPT_PATH)
print("JSON 로그 저장 위치:", LOG_PATH)
