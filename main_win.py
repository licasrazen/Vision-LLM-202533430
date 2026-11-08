
import cv2
import numpy as np
import json
import time
import threading
import subprocess
import urllib.request
import wave
import sounddevice as sd
import numpy as np

from scipy.io.wavfile import write
from piper import PiperVoice
from pathlib import Path
from datetime import datetime
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# =====================================================
# 1. 기본 설정
# =====================================================

MODEL_PATH = "shoplifting_wights.pt"

# 0 = 웹캠
# 동영상 테스트 시 "res/inout1.mp4"
VIDEO_SOURCE = 0

OUTPUT_PATH = "outputs/processed.mp4"
LOG_PATH = "outputs/conversation.jsonl"

WIDTH = 960

SHOP_CONF = 0.40
NORMAL_CONF = 0.80

FRAME_WINDOW = 12
MIN_DETECTIONS = 3

EVENT_COOLDOWN = 60
ENABLE_VOICE = True


# =====================================================
# 2. 얼굴 모자이크 설정
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

        # 화면 밖으로 나가는 좌표 보정
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)

        if x2 <= x1 or y2 <= y1:
            continue

        face_region = frame[y1:y2, x1:x2]

        small = cv2.resize(
            face_region,
            (max(1, (x2-x1)//12), max(1, (y2-y1)//12)),
            interpolation=cv2.INTER_LINEAR
        )

        mosaic = cv2.resize(
            small,
            (x2-x1, y2-y1),
            interpolation=cv2.INTER_NEAREST
        )

        frame[y1:y2, x1:x2] = mosaic

    return frame


# =====================================================
# 3. YOLO 모델
# =====================================================

model = YOLO(MODEL_PATH)

print("YOLO 로드 성공")
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
# 4. 로그 저장
# =====================================================

Path("outputs").mkdir(exist_ok=True)
Path("src/audio").mkdir(parents=True, exist_ok=True)

log_lock = threading.Lock()

ui_status = {
    "status": "대기 중",
    "user_text": ""
}

ui_status_lock = threading.Lock()


def set_ui_status(status=None, user_text=None):
    with ui_status_lock:
        if status is not None:
            ui_status["status"] = status

        if user_text is not None:
            ui_status["user_text"] = user_text

def draw_status(frame):

    with ui_status_lock:
        status = ui_status["status"]
        user_text = ui_status["user_text"]

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(
        "C:/Windows/Fonts/malgun.ttf", 22
    )

    draw.rectangle((10, 45, 950, 135), fill=(0, 0, 0))

    draw.text(
        (20, 52),
        status,
        font=font,
        fill=(255, 255, 0)
    )

    draw.text(
        (20, 88),
        "고객: " + user_text[-55:],
        font=font,
        fill=(255, 255, 255)
    )

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def save_log(category, content):

    data = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "category": category,
        "content": content
    }

    with log_lock:

        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


# =====================================================
# 5. Gemma 로컬 서버
# =====================================================


from llama_cpp import Llama

MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
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
너는 매장에서 고객을 응대하는  AI 안내원이야.

규칙:
1. 항상 정중하고 자연스러운 한국어를 사용한다.
2. 고객을 절도범이라고 단정하거나 비난하지 않는다.
3. 상품 확인이 필요하면 정중하게 도움을 제안한다.
4. 고객의 대답에 맞춰 짧고 자연스럽게 응답한다.
5. 불필요하게 반복해서 질문하지 않는다.
6. 문제가 해결되면 정중하게 대화를 마무리한다.
"""




# =====================================================
# 6. STT / TTS
# =====================================================

MIC_DEVICE = "plughw:3,0"

AUDIO_FILE = "src/audio/input.wav"
WHISPER_PATH = r"whisper.cpp\build-cpu\bin\whisper-cli.exe"
WHISPER_MODEL = r"whisper.cpp\models\ggml-base.bin"

PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"

OUTPUT_FILE = "src/audio/response.wav"

RECORD_SECONDS = 5


def speech_to_text():

    set_ui_status("녹음 중... 말씀해 주세요.")

    print("[마이크] 녹음 중입니다. 말씀해 주세요.")

    audio = sd.rec(
        int(RECORD_SECONDS * 16000),
        samplerate=16000,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    write(AUDIO_FILE, 16000, audio)

    set_ui_status("음성 인식 중...")
    print("[Whisper] 음성 인식 중...")

    result = subprocess.run(
        [
            WHISPER_PATH,
            "-m", WHISPER_MODEL,
            "-f", AUDIO_FILE,
            "-l", "ko",
            "--no-timestamps"
        ],
        text=True,
        capture_output=True,
        check=True
    )

    question = result.stdout.strip()

    if question:
        print("[사용자 STT]", question)
        set_ui_status("고객 음성 인식 완료", question)
    else:
        print("[Whisper] 음성이 인식되지 않았습니다.")
        set_ui_status("음성이 인식되지 않았습니다.")

    return question


PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"

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
        raise ValueError(f"지원하지 않는 WAV 샘플 크기: {sample_width}")

    audio_data = np.frombuffer(frames, dtype=np.int16)

    if channels > 1:
        audio_data = audio_data.reshape(-1, channels)

    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()

    print("[TTS] 음성 재생 완료")
    set_ui_status("고객 응답 대기 중...")

# =====================================================
# 7. AI 음성 대화
# =====================================================

voice_busy = threading.Event()


def conversation_worker():

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    try:
        # 1. 감지 즉시 첫 안내
        answer = "고객님, 혹시 계산 전인 상품이 있으신가요? 확인을 도와드리겠습니다."

        messages.append({
            "role": "assistant",
            "content": answer
        })

        print("[AI 안내]", answer)
        save_log("AI", answer)

        # 안내 음성 먼저 재생
        text_to_speech(answer)

        # 2. 최대 3회 음성 대화
        for _ in range(3):

            # 녹음 및 음성 인식
            question = speech_to_text()

            if not question:
                print("[Whisper] 음성이 인식되지 않았습니다.")
                continue

            print("[사용자]", question)
            save_log("USER", question)

            messages.append({
                "role": "user",
                "content": question
            })

            # 3. Gemma 답변 생성
            answer = ask_gemma(messages)

            messages.append({
                "role": "assistant",
                "content": answer
            })

            print("[Gemma]", answer)
            save_log("AI", answer)

            # 4. 답변 음성 재생
            text_to_speech(answer)

    except Exception as e:
        print("[음성 대화 오류]", e)
        save_log("ERROR", str(e))

    finally:
        voice_busy.clear()

# =====================================================
# 8. 영상 입력 및 출력
# =====================================================

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError("영상 입력 장치를 열 수 없습니다.")

writer = None

detection_history = deque(maxlen=FRAME_WINDOW)

last_event_time = 0

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25


# =====================================================
# 9. 메인 탐지 루프
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

    # 원본 프레임에서 YOLO 추론
    results = model.predict(
        frame,
        conf=0.25,
        verbose=False
    )

    boxes = results[0].boxes

    shop_detected = False
    max_shop_conf = 0.0

    # 모자이크 및 표시용 프레임
    display_frame = apply_face_mosaic(frame.copy())

    # YOLO 결과 표시
    if boxes is not None:

        for box in boxes:

            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = (
                box.xyxy[0].cpu().numpy().astype(int)
            )

            if class_id == SHOP_ID:
                print(
    f"[YOLO] class={class_id}, "
    f"name={model.names[class_id]}, "
    f"confidence={confidence:.3f}"
)

                color = (0, 0, 255)

                if confidence >= SHOP_CONF:

                    shop_detected = True
                    max_shop_conf = max(
                        max_shop_conf,
                        confidence
                    )

                label = "SHOPLIFTING CHECK"

            elif class_id == NORMAL_ID:
                print(
        f"[YOLO] class={class_id}, "
        f"name={model.names[class_id]}, "
        f"confidence={confidence:.3f}"
    )

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
    # 10. 연속 프레임 판정
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
    # 11. 화면 및 영상 저장
    # =================================================

    if writer is None and OUTPUT_PATH:

        writer = cv2.VideoWriter(
            OUTPUT_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (display_frame.shape[1], display_frame.shape[0])
        )

    if writer is not None:
        writer.write(display_frame)

    cv2.imshow("Local Shoplifting Detection System", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =====================================================
# 12. 종료
# =====================================================

cap.release()

if writer is not None:
    writer.release()

cv2.destroyAllWindows()

print("프로그램 종료")
