
import cv2
import numpy as np
import sounddevice as sd
import mediapipe as mp
import threading
import time
import math

from collections import deque, Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# 1. CONFIGURATION
# =========================================================

MODEL_PATH = "src/models/MediaPipe/hand_landmarker.task"

SAMPLE_RATE = 44100
BLOCK_SIZE = 512
SCOPE_SIZE = 2048

BASE_FREQ_MIN = 110.0
BASE_FREQ_MAX = 440.0


RATIOS = {
    0: (1, 1),
    1: (1, 1),
    2: (1, 2),
    3: (2, 3),
    4: (3, 4),
    5: (4, 5),
    6: (5, 6),
}


HISTORY_SIZE = 5
STABLE_FRAMES = 2
LOST_HAND_GRACE_FRAMES = 8
POSITION_ALPHA = 0.18

# BGR colors
CYAN = (220, 245, 130)
BLUE = (255, 110, 75)
PURPLE = (220, 100, 230)
GREEN = (100, 255, 150)
WHITE = (225, 230, 230)
DIM = (65, 75, 80)
MAGENTA = (220, 100, 255)

control_lock = threading.Lock()
scope_lock = threading.Lock()

control = {
    "count": 1,
    "base_freq": 220.0,
    "gate": 0.0,
}

scope_audio = np.zeros(SCOPE_SIZE, dtype=np.float32)
scope_x = np.zeros(SCOPE_SIZE, dtype=np.float32)
scope_y = np.zeros(SCOPE_SIZE, dtype=np.float32)

phase1 = 0.0
phase2 = 0.0
envelope = 0.0
filter_state = 0.0

# Tracking trails
TRAIL_LENGTH = 24
trails = [deque(maxlen=TRAIL_LENGTH) for _ in range(42)]

# Hand tracking model
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.7,
)

detector = vision.HandLandmarker.create_from_options(options)


# =========================================================
# 2. FINGER COUNT
# =========================================================


def count_fingers(hand, handedness):
    def point(i):
        return np.array(
            [hand[i].x, hand[i].y, hand[i].z],
            dtype=float
        )

    def angle(a, b, c):
        v1 = a - b
        v2 = c - b
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)

        if denom < 1e-8:
            return 0.0

        cosine = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    wrist = point(0)
    palm_size = np.linalg.norm(point(9) - wrist)

    if palm_size < 1e-6:
        return 0

    count = 0

    # 검지, 중지, 약지, 새끼손가락
    fingers = [
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ]

    for mcp, pip, dip, tip in fingers:
        pip_angle = angle(point(mcp), point(pip), point(dip))

        # 손목에서 손가락 끝까지의 거리도 함께 확인
        tip_dist = np.linalg.norm(point(tip) - wrist) / palm_size
        pip_dist = np.linalg.norm(point(pip) - wrist) / palm_size

        extended = (
            pip_angle > 150
            and tip_dist > pip_dist + 0.18
        )

        if extended:
            count += 1

    # 엄지: 손바닥 중심에서 얼마나 바깥으로 뻗었는지 확인
    thumb_mcp = point(2)
    thumb_ip = point(3)
    thumb_tip = point(4)
    index_mcp = point(5)

    thumb_angle = angle(point(1), thumb_mcp, thumb_ip)

    tip_from_palm = np.linalg.norm(thumb_tip - point(0))
    ip_from_palm = np.linalg.norm(thumb_ip - point(0))

    tip_from_index = np.linalg.norm(thumb_tip - index_mcp)
    ip_from_index = np.linalg.norm(thumb_ip - index_mcp)

    thumb_extended = (
        tip_from_palm > ip_from_palm + 0.12 * palm_size
        and tip_from_index > ip_from_index + 0.10 * palm_size
        and thumb_angle > 135
    )

    if thumb_extended:
        count += 1

    return count




# =========================================================
# 3. SOFT SYNTH AUDIO
# =========================================================

def audio_callback(outdata, frames, time_info, status):
    global phase1, phase2, envelope, filter_state

    with control_lock:
        count = control["count"]
        base_freq = control["base_freq"]
        gate = control["gate"]

    numerator, denominator = RATIOS[count]

    f1 = float(np.clip(base_freq, 80, 700))
    f2 = float(np.clip(base_freq * denominator / numerator, 80, 1400))

    t = np.arange(frames)
    inc1 = 2 * np.pi * f1 / SAMPLE_RATE
    inc2 = 2 * np.pi * f2 / SAMPLE_RATE

    p1 = phase1 + inc1 * t
    p2 = phase2 + inc2 * t

    tone1 = (
        np.sin(p1)
        + 0.10 * np.sin(2 * p1)
        + 0.025 * np.sin(3 * p1)
    ) / 1.125

    tone2 = (
        np.sin(p2)
        + 0.10 * np.sin(2 * p2)
        + 0.025 * np.sin(3 * p2)
    ) / 1.125

    mixed = 0.5 * tone1 + 0.5 * tone2

    # Smooth attack and release
    attack_step = 1.0 / (0.06 * SAMPLE_RATE)
    release_step = 1.0 / (0.25 * SAMPLE_RATE)

    env = np.empty(frames, dtype=np.float32)

    for i in range(frames):
        if gate > 0.5:
            envelope = min(1.0, envelope + attack_step)
        else:
            envelope = max(0.0, envelope - release_step)

        env[i] = envelope

    mixed *= env

    # Gentle low-pass filter
    cutoff = 3500.0
    dt = 1.0 / SAMPLE_RATE
    rc = 1.0 / (2 * np.pi * cutoff)
    alpha = dt / (rc + dt)

    filtered = np.empty(frames, dtype=np.float32)

    for i in range(frames):
        filter_state += alpha * (mixed[i] - filter_state)
        filtered[i] = filter_state

    sound = (0.35 * filtered).astype(np.float32)

    outdata[:, 0] = sound
    outdata[:, 1] = sound

    phase1 = (phase1 + inc1 * frames) % (2 * np.pi)
    phase2 = (phase2 + inc2 * frames) % (2 * np.pi)

    n = min(frames, SCOPE_SIZE)

    with scope_lock:
        scope_audio[:-n] = scope_audio[n:]
        scope_audio[-n:] = sound[-n:]

        scope_x[:-n] = scope_x[n:]
        scope_y[:-n] = scope_y[n:]

        scope_x[-n:] = np.sin(p1[-n:])
        scope_y[-n:] = np.sin(p2[-n:])


# =========================================================
# 4. RETRO HUD DRAWING HELPERS
# =========================================================

def glow_line(img, p1, p2, color, thickness=1):
    # Subtle glow, then a crisp line
    cv2.line(img, p1, p2, color, thickness + 3, cv2.LINE_AA)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def label(img, text, pos, color=CYAN, scale=0.42):
    cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_PLAIN,
        scale * 1.5,
        color,
        1,
        cv2.LINE_AA,
    )


def panel(img, x, y, w, h, title):
    cv2.rectangle(img, (x, y), (x + w, y + h), DIM, 1)
    cv2.line(img, (x, y), (x + 50, y), CYAN, 1)
    label(img, title, (x + 8, y + 16), CYAN)


def draw_corner_marks(img, x, y, w, h, size=12, color=BLUE):
    for sx, sy in [
        (x, y),
        (x + w, y),
        (x, y + h),
        (x + w, y + h),
    ]:
        dx = size if sx == x else -size
        dy = size if sy == y else -size

        cv2.line(img, (sx, sy), (sx + dx, sy), color, 1)
        cv2.line(img, (sx, sy), (sx, sy + dy), color, 1)


# =========================================================
# 5. CAMERA + AUDIO
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    detector.close()
    raise RuntimeError("웹캠을 열 수 없습니다.")

# Request a consistent working resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

stream = sd.OutputStream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    channels=2,
    dtype="float32",
    callback=audio_callback,
)

finger_history = deque(maxlen=HISTORY_SIZE)

current_count = 1
candidate_count = None
candidate_frames = 0

current_base_freq = 220.0
smoothed_x = None

lost_hand_frames = 0
last_timestamp_ms = 0
start_time = time.monotonic()

# Deterministic background star field
rng = np.random.default_rng(12)
star_positions = None

print("RETRO TRACKING SYNTH")
print("Open hand to control sound | Q to quit")

stream.start()

try:
    while True:
        ret, camera_frame = cap.read()

        if not ret:
            break

        camera_frame = cv2.flip(camera_frame, 1)
        h, w = camera_frame.shape[:2]

        if star_positions is None:
            star_positions = [
                (
                    int(rng.integers(0, w)),
                    int(rng.integers(0, h)),
                    int(rng.integers(1, 3)),
                )
                for _ in range(220)
            ]

        # Dark technical display, not a camera-video background
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Tiny static stars / sensor noise
        for sx, sy, radius in star_positions:
            frame[sy, sx] = (radius * 3, radius * 3, radius * 4)

        # Low-intensity technical grid
        for gx in range(0, w, 40):
            cv2.line(frame, (gx, 0), (gx, h), (10, 15, 17), 1)

        for gy in range(0, h, 40):
            cv2.line(frame, (0, gy), (w, gy), (10, 15, 17), 1)

        rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        timestamp_ms = max(
            int(time.monotonic() * 1000),
            last_timestamp_ms + 1,
        )
        last_timestamp_ms = timestamp_ms

        result = detector.detect_for_video(image, timestamp_ms)

        total_fingers = 0
        first_index_x = None
        detected_any = bool(result.hand_landmarks)

        # Keep separate trails for each detected hand
        active_trail_ids = set()

        for idx, hand in enumerate(result.hand_landmarks):
            handedness = result.handedness[idx][0].category_name
            fingers = count_fingers(hand, handedness)
            total_fingers += fingers

            points = [
                (int(p.x * w), int(p.y * h))
                for p in hand
            ]

            # First detected hand controls the base frequency
            if first_index_x is None:
                first_index_x = hand[8].x

            # Hand-specific color
            line_color = CYAN if idx == 0 else PURPLE
            node_color = GREEN if idx == 0 else MAGENTA

            # Joint skeleton
            for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                a = points[connection.start]
                b = points[connection.end]
                glow_line(frame, a, b, line_color, 1)

            # Draw all landmarks with special fingertip nodes
            for landmark_id, point_xy in enumerate(points):
                x, y = point_xy

                if landmark_id in [4, 8, 12, 16, 20]:
                    cv2.circle(frame, (x, y), 7, node_color, 1, cv2.LINE_AA)
                    cv2.circle(frame, (x, y), 2, WHITE, -1)
                    label(
                        frame,
                        f"J{landmark_id:02d}",
                        (x + 7, y - 7),
                        node_color,
                        0.36,
                    )
                else:
                    cv2.circle(frame, (x, y), 2, line_color, -1)

            # Persistent-looking trajectory for index fingertip
            trail_id = idx * 21 + 8
            active_trail_ids.add(trail_id)
            trails[trail_id].append(points[8])

            trail = list(trails[trail_id])

            for i in range(1, len(trail)):
                fade = i / len(trail)
                color = (
                    int(line_color[0] * fade),
                    int(line_color[1] * fade),
                    int(line_color[2] * fade),
                )
                cv2.line(
                    frame,
                    trail[i - 1],
                    trail[i],
                    color,
                    1,
                    cv2.LINE_AA,
                )

            # Bounding box around hand
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            bx1 = max(0, min(xs) - 14)
            by1 = max(0, min(ys) - 14)
            bx2 = min(w - 1, max(xs) + 14)
            by2 = min(h - 1, max(ys) + 14)

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), DIM, 1)
            draw_corner_marks(frame, bx1, by1, bx2 - bx1, by2 - by1)

            label(
                frame,
                f"ID:0{idx + 1}  F:{fingers:02d}",
                (bx1, max(18, by1 - 7)),
                line_color,
                0.52,
            )

            # Coordinates for the index fingertip
            ix, iy = points[8]
            label(
                frame,
                f"X:{ix:03d} Y:{iy:03d}",
                (ix + 12, iy + 16),
                WHITE,
                0.38,
            )

        # -------------------------------------------------
        # Stable finger count
        # -------------------------------------------------

        if detected_any:
            lost_hand_frames = 0

            observed = int(np.clip(total_fingers, 0, 6))


            finger_history.append(observed)

            if len(finger_history) == HISTORY_SIZE:
                stable_count = Counter(
                    finger_history
                ).most_common(1)[0][0]

                if stable_count == candidate_count:
                    candidate_frames += 1
                else:
                    candidate_count = stable_count
                    candidate_frames = 1

                if candidate_frames >= STABLE_FRAMES:
                    current_count = stable_count

            if first_index_x is not None:
                if smoothed_x is None:
                    smoothed_x = first_index_x
                else:
                    smoothed_x += POSITION_ALPHA * (
                        first_index_x - smoothed_x
                    )

                target_freq = float(np.interp(
                    smoothed_x,
                    [0.0, 1.0],
                    [BASE_FREQ_MIN, BASE_FREQ_MAX],
                ))

                current_base_freq += 0.15 * (
                    target_freq - current_base_freq
                )

            gate = 1.0

        else:
            lost_hand_frames += 1

            if lost_hand_frames <= LOST_HAND_GRACE_FRAMES:
                gate = 1.0
            else:
                gate = 0.0
                finger_history.clear()
                candidate_count = None
                candidate_frames = 0
                smoothed_x = None

        with control_lock:
            control["count"] = current_count
            control["base_freq"] = current_base_freq
            control["gate"] = gate

        # -------------------------------------------------
        # HUD HEADER
        # -------------------------------------------------

        elapsed = time.monotonic() - start_time
        numerator, denominator = RATIOS[current_count]

        label(frame, "RETRO TRACKING SYSTEM", (20, 28), WHITE, 0.85)
        label(frame, "HAND INPUT / SYNTH ENGINE", (20, 47), CYAN, 0.5)

        label(frame, f"TC: {elapsed:08.2f}", (w - 175, 28), WHITE, 0.55)
        label(frame, f"FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}", (w - 175, 47), GREEN, 0.5)

        cv2.line(frame, (20, 60), (w - 20, 60), DIM, 1)

        # -------------------------------------------------
        # LEFT DATA PANEL
        # -------------------------------------------------

        panel(frame, 12, 72, 190, 190, "TRACKING DATA")

        label(frame, f"HANDS: {len(result.hand_landmarks):02d}", (24, 112), GREEN)
        label(frame, f"FINGERS: {current_count:02d}", (24, 138), WHITE)
        label(frame, f"RATIO: {numerator}:{denominator}", (24, 164), PURPLE)
        label(frame, f"BASE: {current_base_freq:06.1f} Hz", (24, 190), CYAN)
        label(frame, f"NOTE A: {current_base_freq:.1f}", (24, 216), WHITE)
        label(
            frame,
            f"NOTE B: {current_base_freq * denominator / numerator:.1f}",
            (24, 242),
            WHITE,
        )

        # -------------------------------------------------
        # RIGHT DATA PANEL
        # -------------------------------------------------

        panel(frame, w - 220, 72, 208, 190, "SIGNAL STATUS")

        label(frame, f"TRACK: {'LOCKED' if detected_any else 'SEARCH'}",
              (w - 208, 112), GREEN if detected_any else MAGENTA)
        label(frame, f"INPUT: {'ACTIVE' if gate else 'IDLE'}",
              (w - 208, 1138), CYAN)
        label(frame, f"SR: {SAMPLE_RATE}", (w - 208, 164), WHITE)
        label(frame, f"BLOCK: {BLOCK_SIZE}", (w - 208, 190), WHITE)
        label(frame, "MODE: VIDEO", (w - 208, 216), PURPLE)
        label(frame, "AUDIO: STEREO", (w - 208, 242), GREEN)

        # -------------------------------------------------
        # CENTRAL TRACKING TARGET
        # -------------------------------------------------

        cx, cy = w // 2, h // 2

        # Small central reference reticle
        cv2.circle(frame, (cx, cy), 12, DIM, 1)
        cv2.circle(frame, (cx, cy), 2, CYAN, 1)
        cv2.line(frame, (cx - 36, cy), (cx - 14, cy), BLUE, 1)
        cv2.line(frame, (cx + 14, cy), (cx + 36, cy), BLUE, 1)
        cv2.line(frame, (cx, cy - 36), (cx, cy - 14), BLUE, 1)
        cv2.line(frame, (cx, cy + 14), (cx, cy + 36), BLUE, 1)

        # -------------------------------------------------
        # LOWER OSCILLOSCOPE PANEL
        # -------------------------------------------------

        panel_h = 180
        panel_y = h - panel_h - 20

        panel(frame, 18, panel_y, w - 36, panel_h, "AUDIO SIGNAL / OSCILLOSCOPE")

        with scope_lock:
            audio = scope_audio.copy()
            lx = scope_x.copy()
            ly = scope_y.copy()

        plot_x = 32
        plot_y = panel_y + 32
        plot_w = w - 64
        plot_h = 130

        mid_y = plot_y + plot_h // 2

        cv2.line(
            frame,
            (plot_x, mid_y),
            (plot_x + plot_w, mid_y),
            DIM,
            1,
        )

        # Downsample waveform to screen width
        
        # 최근 오디오 구간을 확대해서 표시
        visible_samples = 1024
        recent_audio = audio[-visible_samples:]

        indices = np.linspace(
            0,
            visible_samples - 1,
            plot_w,
        ).astype(int)

        wave = recent_audio[indices]

        # 파형 진폭을 화면에 맞게 확대
        wave_scale = max(float(np.max(np.abs(wave))), 0.015)
        normalized_wave = wave / wave_scale

        wave_points = np.column_stack((
            plot_x + np.arange(plot_w),
            mid_y - normalized_wave * (plot_h * 0.42),
        )).astype(np.int32)

        cv2.polylines(
            frame,
            [wave_points],
            False,
            GREEN,
            1,
            cv2.LINE_AA,
        )


        # Small waveform peak indicator
        peak = float(np.max(np.abs(audio)))
        label(
            frame,
            f"PEAK: {peak:.3f}",
            (w - 130, panel_y + 17),
            CYAN,
            0.4,
        )

        # -------------------------------------------------
        # SMALL LISSAJOUS INSET
        # -------------------------------------------------

        inset_w = 125
        inset_h = 125
        inset_x = w - inset_w - 24
        inset_y = 280

        panel(frame, inset_x, inset_y, inset_w, inset_h, "XY PHASE")

        lx_plot = inset_x + inset_w // 2
        ly_plot = inset_y + 72

        liss_points = np.column_stack((
            lx_plot + lx * 30,
            ly_plot + ly * 30,
        )).astype(np.int32)

        cv2.polylines(
            frame,
            [liss_points],
            False,
            PURPLE,
            1,
            cv2.LINE_AA,
        )

        # -------------------------------------------------
        # FOOTER / TECHNICAL LABELS
        # -------------------------------------------------

        cv2.line(frame, (20, h - 18), (w - 20, h - 18), DIM, 1)

        label(
            frame,
            "SYSTEM: ONLINE",
            (20, h - 5),
            GREEN,
            0.45,
        )

        label(
            frame,
            "HAND LANDMARK TRACKING // SYNTHESIS",
            (w // 2 - 130, h - 5),
            DIM,
            0.45,
        )

        label(
            frame,
            "Q: EXIT",
            (w - 80, h - 5),
            WHITE,
            0.45,
        )


        cv2.namedWindow("RETRO TRACKING SYNTH", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RETRO TRACKING SYNTH", 1280, 720)
        cv2.imshow("RETRO TRACKING SYNTH", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    try:
        stream.stop()
        stream.close()
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
