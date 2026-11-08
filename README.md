<<<<<<< HEAD
# 무인 경비 보안 시스템
### YOLO 기반 이상행동 감지 및 로컬 멀티모달 AI 고객 응대 시스템

## 1. 파일 구조 (필수)

프로젝트의 기본 디렉터리 구조는 다음과 같습니다.

```text
Project/
│
├── main_win.py                    # Windows 실행 코드
├── main_jetson.py                 # Jetson 실행 코드
├── shoplifting_wights.pt          # YOLO 학습 모델
├── face_detection_yunet_2023mar.onnx  # YuNet 얼굴 검출 모델
│
├── src/
│   ├── models/
│   │   ├── Gemma4/
│   │   │   └── google_gemma-4-E2B-it-Q4_K_M.gguf
│   │   │
│   │   └── Piper/
│   │       ├── ko_KR-kss-medium.onnx
│   │       └── ko_KR-kss-medium.onnx.json
│   │
│   └── audio/
│       ├── input.wav              # STT 입력 음성
│       └── response.wav           # TTS 출력 음성
│
├── whisper.cpp/
│   ├── build/
│   │   └── bin/
│   │       └── Release/           # Windows whisper 실행 파일 위치
│   │           └── whisper-cli.exe
│   ├── models/
│   │   └── ggml-base.bin          # Whisper 음성 인식 모델
│   └── ggml-base.bin              # Windows 코드 경로에 맞춰 배치할 경우
│
├── output/
│   ├── recordings/                # 녹화 영상
│   └── logs/                      # 대화 기록 및 이벤트 로그
│
├── requirements-windows.txt
└── requirements-jetson.txt
```

---

## 2. 실행 방식 (필수)

### 2-1. 개발 환경

| 구분 | Windows | Jetson |
|---|---|---|
| 운영체제 | Windows 10/11 | Linux Ubuntu |
| 하드웨어 | 일반 PC 및 NVIDIA GPU | Jetson Orin Nano 8GB |
| Python | Python 3.10 권장 | Python 3.10 권장 |
| 객체 탐지 | YOLO / Ultralytics | YOLO / CUDA |
| 언어 모델 | Gemma GGUF | Gemma GGUF |
| 음성 인식 | Whisper.cpp | Whisper.cpp |
| 음성 합성 | Piper TTS | Piper TTS |
| 얼굴 검출 | YuNet | YuNet |
| 영상 처리 | OpenCV | OpenCV / GStreamer |

모든 AI 모델은 로컬에 저장하고 실행한다. 인터넷 연결은 최초 패키지 및 모델 설치·다운로드에 필요하며, 필요한 파일을 준비한 이후에는 외부 AI API를 호출하지 않고 추론할 수 있도록 구성한다.

### 2-2. Windows 실행 방법

**① Python 가상환경 생성**

Python 3.10 설치 후 프로젝트 폴더에서 다음 명령어를 실행한다.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

PowerShell 실행 정책으로 활성화가 되지 않는 경우 다음 명령어를 사용할 수 있다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**② Python 패키지 설치**

프로젝트 코드에서 사용하는 외부 패키지는 다음과 같다.

```powershell
python -m pip install numpy scipy opencv-python pillow ultralytics sounddevice
python -m pip install llama-cpp-python piper-tts
```

필요한 경우 PyTorch 및 torchvision을 별도로 설치한다.

```powershell
python -m pip install torch torchvision
```

단, PyTorch와 llama-cpp-python은 GPU 사용 여부 및 CUDA 버전에 따라 설치 방법이 달라진다. 위 명령어는 기본 설치 예시이며, NVIDIA GPU 가속을 사용하려면 호환되는 CUDA 빌드가 필요하다.

**③ Whisper.cpp 설치**

Whisper는 Python 패키지가 아니라 별도로 빌드하여 사용하는 실행 프로그램이다.

필요한 도구:
- Git
- CMake
- C/C++ 컴파일러 (Visual Studio Build Tools 등)

프로젝트에 Whisper.cpp를 준비한 뒤 빌드한다.

```powershell
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build
cmake --build build --config Release
```

빌드가 완료되면 다음 파일이 생성되어야 한다.

```text
whisper.cpp/build/bin/Release/whisper-cli.exe
```

Whisper 음성 인식 모델도 별도로 다운로드해야 한다.

```text
whisper.cpp/models/ggml-base.bin
```

코드에서 지정한 경로에 맞춰 실행 파일과 모델 파일을 배치한다.

**④ AI 모델 다운로드**

다음 모델은 Python 패키지 설치만으로 자동 생성되지 않으므로 직접 준비해야 한다.

| 모델 | 역할 | 파일 |
|---|---|---|
| YOLO | 절도 의심 행동 분류 | `shoplifting_wights.pt` |
| YuNet | 얼굴 검출 및 모자이크 | `face_detection_yunet_2023mar.onnx` |
| Gemma | 로컬 AI 응답 생성 | `google_gemma-4-E2B-it-Q4_K_M.gguf` |
| Whisper | 한국어 음성 인식 | `ggml-base.bin` |
| Piper | 한국어 음성 합성 | `ko_KR-kss-medium.onnx` 및 JSON 설정 파일 |

Gemma는 Hugging Face 등의 모델 저장소에서 GGUF 파일을 준비한다. Piper는 모델 파일과 해당 모델의 JSON 설정 파일을 함께 준비한다.

**⑤ 실행**

```powershell
python main_win.py
```

프로그램이 실행되면 카메라 영상을 표시하고 YOLO 모델을 통해 이상행동을 감지한다. 의심 상황이 일정 조건을 만족하면 고객 안내 및 음성 대화 기능이 실행된다.

종료는 `Q` 키를 사용한다.

### 2-3. Jetson 실행 방법

**① JetPack 및 CUDA 환경 준비**

Jetson Orin Nano 8GB에서는 일반 PC와 동일한 방식으로 AI 패키지를 설치하면 안 된다.

먼저 NVIDIA JetPack에 맞는 환경을 구성해야 한다.

필수 구성:
- NVIDIA JetPack
- CUDA
- cuDNN
- Jetson 호환 PyTorch
- Jetson 호환 torchvision
- OpenCV 및 GStreamer
- Python 3.10

Jetson의 CUDA와 PyTorch는 JetPack 버전별 호환성이 있으므로 NVIDIA 공식 설치 안내를 기준으로 구성한다.

**② Python 가상환경 생성**

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

`--system-site-packages` 옵션은 JetPack에서 시스템에 설치한 CUDA 관련 Python 패키지를 가상환경에서 활용하기 위한 것이다.

**③ Python 패키지 설치**

```bash
python -m pip install numpy scipy pillow sounddevice
python -m pip install ultralytics
python -m pip install piper-tts
```

추가로 다음 패키지를 준비한다.

- `opencv-python` 또는 JetPack에 포함된 OpenCV
- `llama-cpp-python`
- CUDA 지원이 가능한 PyTorch 및 torchvision

Jetson에서는 OpenCV를 무조건 `pip install opencv-python`으로 재설치하지 않는다. 기존 JetPack OpenCV와 충돌하거나 GStreamer 기능이 빠질 수 있으므로 현재 설치된 OpenCV의 CUDA/GStreamer 지원 여부를 먼저 확인한다.

**④ llama-cpp-python CUDA 빌드**

Gemma GGUF 모델을 GPU에서 실행하기 위해서는 CUDA 지원으로 빌드된 llama-cpp-python이 필요하다.

Jetson의 CUDA 환경에 맞춰 CMake CUDA 옵션을 활성화하여 설치한다. CPU 전용 패키지가 설치되면 GPU 가속을 사용할 수 없으므로 설치 후 CUDA 지원 여부를 확인한다.

**⑤ Whisper.cpp 빌드**

필요한 도구를 설치한다.

```bash
sudo apt update
sudo apt install -y git cmake build-essential
```

Whisper.cpp를 내려받고 빌드한다.

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build
cmake --build build -j4
```

빌드 후 실행 파일과 모델을 준비한다.

```text
whisper.cpp/build/bin/whisper-cli
whisper.cpp/models/ggml-base.bin
```

Jetson에서는 CPU 기반 Whisper 실행부터 확인한 뒤, 필요에 따라 가속 설정을 조정한다.

**⑥ 카메라 및 오디오 설정**

Jetson은 USB 카메라와 CSI 카메라를 모두 사용할 수 있도록 구성한다.

CSI 카메라를 사용하는 경우 GStreamer의 `nvarguscamerasrc` 파이프라인을 이용한다.

```python
pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, "
    "height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink drop=true max-buffers=1"
)
```

USB 카메라를 사용하는 경우 OpenCV의 VideoCapture 장치 번호를 설정한다.

마이크와 스피커는 `sounddevice`에서 인식하는 장치 번호를 확인하고 코드에 반영한다.

**⑦ 실행**

```bash
python3 main_jetson.py
```

Jetson에서는 YOLO, Gemma, Whisper가 동시에 메모리를 사용할 수 있으므로 모델 크기, GPU 할당량, 입력 해상도 및 추론 주기를 조정해야 한다.

특히 8GB 메모리 환경에서는 Gemma의 GPU 레이어 수와 YOLO 추론 설정에 따라 메모리 부족이 발생할 수 있다.

---

## 3. 창의성 및 확장성 (프로젝트에만 해당)

### 3-1. 아이디어 설명

**무인 매장의 보안 공백을 해결하는 로컬 AI 기반 지능형 경비 시스템**

무인 편의점, 스터디카페, 물류 창고 등 무인 또는 최소 인력으로 운영되는 사업장이 증가하고 있다.

그러나 무인 매장은 24시간 운영되는 반면 점주가 24시간 CCTV 화면을 확인하는 것은 현실적으로 어렵다.

고객이 상품을 결제하지 않고 반출하거나 기물을 파손하더라도 점주는 다음 날 매출과 재고의 차이를 확인하고 CCTV 영상을 다시 살펴본 뒤에야 문제를 인지하는 경우가 있다.

또한 키오스크 주변에서 반복적으로 배회하거나 상품 진열대에서 의심스러운 행동을 하는 고객이 있어도 실시간으로 확인하지 못하면 대응 시점을 놓치기 쉽다.

기존 CCTV는 사고가 발생한 뒤 영상을 확인하는 데에는 유용하지만, 사고 발생 전후의 이상 상황을 자동으로 분석하고 점주에게 즉시 알려주는 기능에는 한계가 있다.

본 프로젝트는 이러한 문제를 해결하기 위해 YOLO 기반 객체 및 행동 감지와 로컬 언어 모델을 결합하여 매장 내 이상 상황을 감지하고, 점주와 고객 간의 음성·텍스트 기반 대응까지 지원하는 지능형 보안 시스템을 제안한다.

**주요 타겟 시장**
- 무인 편의점 및 무인 아이스크림 매장 점주
- 스터디카페 및 무인 독서실 관리자
- 무인 카페 및 소규모 무인 매장 운영자
- 물류 창고 및 재고 관리 담당자
- 유인 매장의 보안 및 시설 관리자

### 3-2. 문제점 및 시장 필요성

기존 무인 매장 운영에서 발생하는 주요 문제는 다음과 같다.

1. **사후 확인 중심의 보안**

   상품이 사라진 뒤 재고와 매출을 비교하고 CCTV를 다시 확인해야 하므로 즉각적인 대응이 어렵다.

2. **실시간 이상행동 감지의 한계**

   키오스크 주변 배회, 상품을 집어 들고 결제하지 않는 행동, 기물 파손 등의 상황을 점주가 직접 확인해야 한다.

3. **CCTV 사각지대 및 설치 위치 문제**

   단순히 카메라 대수를 늘리는 것만으로는 모든 행동을 정확하게 확인하기 어렵다. 출입구, 계산대, 상품 진열대, 사각지대 등을 고려하여 설치 위치를 결정해야 한다.

4. **보안 인력 및 장비 비용**

   모든 매장을 사람이 실시간으로 관제하기 어렵고, 상품마다 도난방지 태그나 스티커를 부착하는 방식은 상품 수가 많은 무인 매장에서 추가적인 인력 및 물적 비용을 발생시킨다.

5. **오탐지 문제**

   고객이 자신의 소지품을 가방에 넣거나 구매한 상품을 다시 진열하는 행동을 절도라고 판단하면 불필요한 경보가 발생한다.

### 3-3. 핵심 해결 방법

본 프로젝트는 단순히 물건을 주머니에 넣는 행동만으로 절도를 판단하지 않는다.

사람의 위치, 자세, 상품과의 상호작용, 행동의 순서 및 매장 내 상황을 종합적으로 분석하는 구조를 목표로 한다.

**① YOLO 기반 객체 및 이상행동 감지**

YOLO 모델을 활용하여 영상 내 사람과 이상행동을 감지한다.

현재 구현에서는 학습된 `shoplifting`, `not_shoplifting` 클래스를 기반으로 절도 의심 상황을 분류한다.

**② State Machine 기반 행동 흐름 분석 (확장 기능)**

향후 YOLOv8 Pose 모델을 도입하여 사람의 자세와 관절 위치를 분석하고, 행동의 시간적 순서를 기억하는 상태 머신을 구현할 계획이다.

```text
사람이 매장 진입
      ↓
상품 진열대 접근
      ↓
상품 접촉 및 픽업
      ↓
결제 여부 확인
      ↓
가방 투입 또는 상품 반출
      ↓
행동 및 상황 종합 분석
      ↓
정상 / 의심 상황 분류
```

각 행동을 독립적으로 판단하는 것이 아니라 이전 상태와 현재 상태를 연결하여 분석한다.

예를 들어 상품을 가방에 넣었다는 사실만으로 절도라고 판단하지 않는다. 상품을 집어 든 이후 결제가 이루어졌는지, 구매한 상품인지, 실제로 매장 밖으로 반출했는지 등의 정보를 함께 확인해야 한다.

**③ Context Understanding (상황 맥락 이해)**

동일한 동작이라도 상황에 따라 의미가 달라질 수 있다.

따라서 다양한 매장 환경과 정상적인 쇼핑 행동 데이터를 학습시켜 오탐지를 줄이는 것이 중요하다.

향후 Vision-Language Model을 활용하여 특정 행동의 전후 맥락을 분석하고, 단일 행동 감지 모델의 한계를 보완할 계획이다.

**④ 상품 및 결제 내역 대조 (확장 기능)**

실제 절도 판단의 정확도를 높이기 위해 매장 진열대의 상품 정보와 결제 데이터를 연동하는 기능을 제안한다.

```text
진열대 상품 정보
       +
키오스크 바코드 스캔 및 결제 내역
       +
카메라 기반 상품 반출 추적
       ↓
결제 여부 및 상품 이동 정보 대조
       ↓
불일치 상황 발생 시 점주 알림
```

이를 통해 결제하지 않은 상품의 반출 가능성을 탐지하고 점주의 재고 확인 및 CCTV 확인 부담을 줄이는 것을 목표로 한다.

단, 실제 상품과 바코드 결제 내역을 정확히 연결하려면 상품 인식, 가림 현상, 다중 고객, 결제 취소 등의 예외 상황을 처리해야 한다.

### 3-4. 비식별화 및 개인정보 보호

본 프로젝트의 중요한 차별점은 로컬 기반 영상 처리 구조이다.

일반적인 클라우드 기반 영상 분석은 CCTV 영상을 외부 서버로 전송하여 분석하는 구조를 사용할 수 있다. 이 경우 영상 전송 비용과 개인정보 처리에 대한 관리 부담이 발생한다.

본 프로젝트는 매장 내부의 엣지 디바이스에서 AI 추론을 수행하는 온프레미스 구조를 지향한다.

```text
매장 CCTV
    ↓
매장 내부 Jetson
    ↓
YOLO 영상 분석
    ↓
YuNet 얼굴 검출
    ↓
얼굴 모자이크 처리
    ↓
의심 상황 메타데이터 생성
    ↓
로컬 Gemma 및 음성 처리
    ↓
점주 알림 및 텍스트 로그 저장
```

- 영상 분석 및 AI 추론을 매장 내부에서 수행한다.
- 얼굴 검출 후 모자이크 처리하여 영상의 식별 가능성을 낮춘다.
- 외부 클라우드 API에 영상이나 음성을 전송하지 않는 구조를 목표로 한다.
- 대화 내용은 STT를 통해 텍스트로 변환하고 로컬 파일로 저장한다.
- 점주가 상황을 확인할 수 있도록 이벤트 기록과 대화 기록을 관리한다.

이러한 구조는 클라우드 추론 API 사용료를 줄이고 영상 데이터의 외부 전송을 최소화할 수 있다.

다만 얼굴 모자이크만으로 개인정보 보호 의무가 모두 해결되는 것은 아니다. CCTV 운영 목적, 안내판, 영상 보관 기간, 접근 권한 및 음성 수집에 대한 별도 법적 검토가 필요하다.

### 3-5. 실시간 음성 대응 시스템

절도 또는 기물 파손 의심 정황이 감지되면 점주에게 알림을 전달하고, 필요에 따라 고객에게 음성 안내를 제공한다.

**기본 동작 흐름**

```text
절도 / 파손 의심 상황 감지
          ↓
점주에게 이벤트 알림
          ↓
고객에게 음성 안내
          ↓
고객 발화 녹음
          ↓
Whisper STT
          ↓
고객 발화 텍스트 표시
          ↓
Gemma 로컬 AI 응답 생성
          ↓
Piper TTS 음성 출력
          ↓
대화 내용 TXT 파일 저장
```

고객에게는 범죄자로 단정하는 표현 대신 중립적인 안내를 제공한다.

예시:

> 고객님, 혹시 계산 전인 상품이 있으신가요? 확인을 도와드리겠습니다.

고객이 응답하면 Whisper가 음성을 텍스트로 변환하고 Gemma가 상황에 맞는 답변을 생성한다. 생성된 답변은 Piper TTS를 통해 음성으로 출력한다.

또한 고객의 발화와 AI 응답을 모두 텍스트로 기록하여 점주가 사후에 대화 내용을 확인할 수 있도록 한다.

음성 안내와 대화 기록은 점주의 대응을 지원하기 위한 기능이며, 저장된 텍스트가 그 자체로 법적 증거력을 보장하는 것은 아니다. 음성 수집·녹음 및 기록 보관은 관련 법령과 고지·동의 요건을 별도로 검토해야 한다.

### 3-6. 구현한 내용

현재 프로젝트에서 구현한 주요 기능은 다음과 같다.

| 기능 | 사용 기술 | 구현 내용 |
|---|---|---|
| 이상행동 감지 | YOLO / Ultralytics | 학습된 `shoplifting`, `not_shoplifting` 모델을 통한 분류 |
| 오탐 방지 | Detection History | 여러 프레임의 탐지 결과를 누적하여 이벤트 발생 여부 판단 |
| 얼굴 비식별화 | YuNet / OpenCV | 얼굴 검출 및 모자이크 처리 |
| 로컬 AI | Gemma / llama.cpp | 외부 API 없이 로컬 언어 모델 실행 |
| 음성 인식 | Whisper.cpp | 고객의 음성을 한국어 텍스트로 변환 |
| 음성 합성 | Piper TTS | AI 응답을 한국어 음성으로 출력 |
| 고객 응대 | State 기반 대화 흐름 | 의심 상황 발생 후 음성 안내 및 최대 3회 대화 |
| 대화 기록 | TXT / 로그 | 고객 발화와 AI 응답을 텍스트 파일로 저장 |
| 영상 처리 | OpenCV | 실시간 영상 출력 및 녹화 |
| 하드웨어 확장 | NVIDIA Jetson | 현장 로컬 AI 처리를 위한 Jetson 실행 환경 구성 |

현재 구현된 Detection History는 여러 프레임의 탐지 결과를 누적하는 방식이다. 상품 픽업부터 결제 여부, 가방 투입까지의 전체 행동 순서를 추적하는 State Machine은 향후 고도화 항목이다.

### 3-7. 확장성

**① 기존 CCTV 관제 시스템 연동**

기존 CCTV를 교체하지 않고 별도의 엣지 디바이스를 연결하여 AI 분석 기능을 추가하는 방향으로 확장할 수 있다.

**② YOLOv8 Pose 기반 행동 분석**

사람의 관절과 자세를 추적하여 상품을 집는 행동, 가방에 넣는 행동, 기물을 파손하는 행동 등을 분석한다.

**③ 바코드 및 키오스크 연동**

POS 또는 키오스크 결제 데이터를 연동하여 실제 상품 반출 정보와 결제 내역을 비교한다.

**④ 다중 매장 통합 관제**

여러 매장의 엣지 디바이스에서 생성된 이벤트 메타데이터를 점주가 한곳에서 확인할 수 있도록 확장한다.

**⑤ 유인 매장 및 물류 창고 적용**

무인 매장뿐 아니라 직원이 근무하는 매장의 사각지대, 물류 창고의 물품 반출 및 시설 파손 감지에도 활용할 수 있다.

**⑥ 방문객 분석**

개인 식별이 아닌 방문객 수, 시간대별 혼잡도, 매장 내 동선 등의 통계 정보를 분석하는 기능으로 확장할 수 있다. 이 경우에도 데이터 처리 목적과 개인정보 보호 요건을 고려해야 한다.

### 3-8. 기타 창의성 및 확장성 내용

**기존 도난 방지 시스템과의 차별점**

| 구분 | 기존 방식 | 제안 시스템 |
|---|---|---|
| 감지 방식 | CCTV 사후 확인 또는 단순 경보 | AI 기반 실시간 이상행동 분석 |
| 판단 기준 | 특정 행동 또는 단일 감지 이벤트 | 행동 흐름 및 상황 맥락 분석 지향 |
| 영상 처리 | 관제 서버 및 클라우드 활용 가능 | 매장 내부 엣지 디바이스 처리 |
| 고객 대응 | 점주가 직접 전화하거나 방문 | AI 음성 안내 및 대화 지원 |
| 기록 | CCTV 영상 중심 | 이벤트 메타데이터 및 대화 텍스트 |
| 확장성 | 별도 장비 및 관제 시스템 의존 | 기존 CCTV와 연동하는 방향으로 확장 |


**예상 리스크 및 대응 방안**

| 리스크 | 대응 방안 |
|---|---|
| 오탐지 증가 | 프레임 누적, 행동 순서 분석, 정상 행동 데이터 추가 학습 |
| 미탐지 발생 | 카메라 설치 위치 최적화 및 학습 데이터 다양화 |
| 개인정보 보호 문제 | 비식별화, 로컬 처리, 접근 권한 및 보관 정책 관리 |
| Jetson 메모리 부족 | 모델 경량화, 추론 주기 조정, GPU 메모리 관리 |
| 고객 불만 및 오해 | 범죄자로 단정하지 않는 중립적인 음성 안내 |
| 결제 정보 연동 문제 | POS·키오스크 연동 규격 및 상품 식별 체계 확보 |

---

## 4. 기타 참고 사항

### 4-1. 사용 기술 및 라이브러리

**Python 기본 내장 모듈**

별도 `pip install`이 필요하지 않은 모듈이다.

```python
import json
import time
import threading
import subprocess
import urllib.request
import wave

from collections import deque
from threading import Lock
from pathlib import Path
from datetime import datetime
```

**외부 Python 패키지**

| 패키지 | 용도 |
|---|---|
| `opencv-python` 또는 JetPack OpenCV | 영상 처리 및 카메라 제어 |
| `numpy` | 영상 및 배열 데이터 처리 |
| `scipy` | 오디오 리샘플링 및 WAV 처리 |
| `sounddevice` | 마이크 녹음 및 스피커 출력 |
| `Pillow` | 이미지 및 자막 처리 |
| `ultralytics` | YOLO 모델 실행 |
| `torch` | 딥러닝 모델 실행 |
| `torchvision` | PyTorch 관련 이미지 및 모델 기능 |
| `llama-cpp-python` | GGUF 로컬 언어 모델 실행 |
| `piper-tts` | 로컬 음성 합성 |

**별도 설치 및 모델 파일**

| 항목 | 용도 |
|---|---|
| Whisper.cpp | 음성 인식 실행 프로그램 |
| `ggml-base.bin` | Whisper 음성 인식 모델 |
| Gemma GGUF | 로컬 언어 모델 |
| Piper ONNX 모델 | 한국어 음성 합성 모델 |
| YuNet ONNX 모델 | 얼굴 검출 모델 |
| YOLO `.pt` 파일 | 학습된 절도 의심 행동 분류 모델 |
| CMake / C++ 빌드 도구 | Whisper.cpp 빌드 |
| CUDA / cuDNN | NVIDIA GPU 가속 |

### 4-2. 주요 기술적 한계

현재 YOLO 모델은 학습 데이터에 기반하여 절도 의심 여부를 분류한다. 따라서 모든 종류의 절도나 기물 파손을 정확하게 판별할 수 있는 것은 아니다.

특히 실제 매장에서 사용하려면 다음 기능의 추가 개발이 필요하다.

- 사람과 상품의 개별 추적
- 상품 픽업 및 반출 행동 분석
- POS 및 바코드 결제 정보 연동
- 여러 사람의 행동이 겹치는 상황 처리
- 가방이나 주머니에 상품이 가려지는 경우 처리
- 다양한 조명 및 카메라 각도에서의 성능 검증

AI가 생성한 의심 이벤트는 범죄 사실의 확정이 아닌 점주의 확인이 필요한 경고로 취급한다.

### 4-3. 참고 자료

1. ADT캡스 - 무인점포 절도 대응 및 보안 관련 자료  
   https://www.adtcaps.co.kr/caps-story/blog/unmanned-store-theft-response

2. LG유플러스 비즈니스 - 무인매장 CCTV와 실시간 대응의 필요성  
   https://blog.naver.com/lguplusbusiness/224340380945

3. AI 탐구노트 - AI를 이용한 도난 감지 및 무인화 매장 확산을 위한 필요 요소  
   https://42morrow.tistory.com/entry/AI를-이용한-도난-감지-무인화-매장-확산을-위한-필요-요소

4. Ultralytics - YOLO 공식 문서  
   https://docs.ultralytics.com/

5. llama.cpp - 로컬 GGUF 모델 실행  
   https://github.com/ggml-org/llama.cpp

6. Whisper.cpp - 로컬 음성 인식  
   https://github.com/ggml-org/whisper.cpp

7. Piper - 로컬 음성 합성  
   https://github.com/OHF-Voice/piper1-gpl

8. NVIDIA Jetson - 개발자 문서  
   https://developer.nvidia.com/embedded/jetson-linux

9. 찾기쉬운 생활법령정보 - 무인상점 CCTV 설치 및 관리  
   https://easylaw.go.kr/CSP/OnhunqueansInfoRetrieve.laf?category=91&curPage=1&onhunqnaAstSeq=91&onhunqueAstSeq=442&onhunqueSeq=6269&pageType=20&sch=&sortType=11
=======
장예원
>>>>>>> befc0d41a2f2d05268e5b128ab52a6bf858ed653
