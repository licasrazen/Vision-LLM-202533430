# MediaPipe를 활용한 핸드 트래킹 기반 신시사이저 및 리사주 도형 시각화 시스템

## 파일 구조 (필수)

- **mathaudio_jetson.py**: 젯슨 실행용
- **mathaudio_win.py**: 노트북 웹캠용

<br>

## 실행 방식 (필수)

가상환경에서 실행할 경우
py -3.10 -m venv --system-site-packages .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

python -m pip install jupyter ipykernel

python -m ipykernel install --user --name=vision_llm --display-name "Vision-LLM (Python3.10)"

설치 패키지 목록

python -m pip install --upgrade pip setuptools wheel pip uninstall matplotlib-inline pip install "matplotlib-inline<0.2" pip install --no-deps \     scikit-image==0.19.3 \     imageio==2.19.3 \     tifffile==2022.5.4 \     PyWavelets==1.3.0 \     networkx==2.8.8

mkdir -p src/models/MediaPipe

pip install --no-deps mediapipe
pip install absl-py flatbuffers sounddevice

wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -O src/models/MediaPipe/hand_landmarker.task

wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task -O src/models/MediaPipe/pose_landmarker_full.task

pip install sounddevice

설치하고 실행

<br>

## 창의성 및 확장성 (프로젝트에만 해당)

### 아이디어 설명 여기에

MediaPipe를 활용한 핸드 트래킹 기반 신시사이저 및 리사주 도형 시각화 시스템을 구현하였다. 카메라로 인식한 손의 움직임과 손가락 수를 음악 생성에 연결하고, 생성되는 소리를 오실로스코프 파형과 리사주 도형으로 시각화하는 프로젝트이다. 단순히 손동작을 인식하는 데 그치지 않고, 손의 움직임에 따라 소리와 시각적 요소가 실시간으로 변화하는 인터랙티브 시스템을 목표로 하였다.

### 구현한 내용 여기에

손 추적: MediaPipe를 활용하여 카메라 영상에서 손의 관절 좌표와 손가락을 인식한다.

음악 생성: Python의 오디오 처리 라이브러리를 활용하여 신시사이저 사운드를 생성하고, 손가락 수에 따른 두 음의 주파수 비율을 적용하였다.

실시간 주파수 제어: 검지 끝의 좌우 위치를 이용해 기본 주파수를 조절하여 손의 움직임이 음높이에 반영되도록 하였다.

파형 시각화: 오실로스코프 형태의 그래프를 통해 생성되는 소리의 파형을 실시간으로 표시한다.

리사주 도형 시각화: 두 음의 주파수 관계에 따라 변화하는 리사주 도형을 표시하여 소리의 변화를 시각적으로 확인할 수 있도록 하였다.

SF 스타일 인터페이스: 추적 정보, 손가락 수, 주파수, 음의 비율 등의 데이터를 HUD 형태의 패널에 표시하고, 손의 랜드마크와 이동 궤적을 색상으로 구분하였다.

### 확장성 여기에

양손 독립 제어: 현재의 손가락 수 합산 방식을 개선하여 왼손과 오른손이 각각 다른 음의 주파수와 음정을 제어하도록 확장할 수 있다.

다양한 악기 음색: 사인파 중심의 합성음에서 벗어나 사각파, 톱니파 등의 파형이나 다양한 신시사이저 음색을 추가할 수 있다.

추가 제스처 활용: 손의 높이, 두 손 사이의 거리, 손가락을 펴고 접는 동작 등을 활용해 음량이나 음색, 효과를 조절할 수 있다.

시각화 기능 확장: 주파수 스펙트럼, 파형의 확대 및 축소, 다양한 리사주 패턴과 시각 효과를 추가할 수 있다.

실시간 연주 시스템: 여러 제어 요소를 조합하여 사용자가 직접 음악을 연주하고 시각 효과를 생성하는 인터랙티브 퍼포먼스 도구로 발전시킬 수 있다.

### 기타 창의성 및 확장성 내용

음향 신호 처리, 컴퓨터 비전, 수학적 시각화를 하나의 시스템에 결합했다. 특히 소리의 변화를 청각적으로만 전달하는 것이 아니라 파형과 리사주 도형으로 함께 표현하여, 주파수 관계와 음향 신호의 특성을 직관적으로 관찰할 수 있도록 하였다. 또한 손 추적 데이터를 시각적 인터페이스에 반영해 기능적인 제어와 디자인 요소를 통합하였다.

### 필요한 만큼 카테고리 추가

내용 기재

<br>

## 기타 참고 사항 (과제 & 프로젝트 모두 해당)

프로젝트를 구현하는 과정에서 손가락 인식이 불안정해지거나 손의 움직임에 비해 주파수 및 시각화가 자연스럽게 반응하지 않는 문제를 해결하고자 하였습니다. 이를 개선하기 위해 손의 좌표와 주파수 값이 부드럽게 변화하도록 처리하고, 손 인식 결과를 안정화하며, 소리의 시작과 종료가 자연스럽게 연결되도록 구현하였습니다.

AI를 활용하여 코드의 구조와 라이브러리의 사용법을 파악하고, 기능 개선 방향을 검토하였습니다. 단순히 코드를 실행하는 데 그치지 않고, 손 추적 결과가 주파수 제어와 음향 생성, 파형 시각화에 어떻게 연결되는지 확인하며 프로젝트를 구성하였습니다. 향후에는 양손을 독립적인 입력으로 활용하는 기능과 더욱 다양한 음향 제어 기능을 구현하여 프로젝트의 완성도를 높이고자 합니다.
