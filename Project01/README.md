# 0. 가상환경 세팅할 경우 실행
py -3.10 -m venv --system-site-packages .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

python -m pip install jupyter ipykernel

python -m ipykernel install --user --name=vision_llm --display-name "Vision-LLM (Python3.10)"

# 1. 윈도우 기본 설치 목록
python -m pip install --upgrade pip setuptools wheel

python -m pip uninstall matplotlib-inline
python -m pip install "matplotlib-inline<0.2"

python -m pip install --no-deps scikit-image==0.19.3
python -m pip install --no-deps imageio==2.19.3
python -m pip install --no-deps tifffile==2022.5.4
python -m pip install --no-deps PyWavelets==1.3.0
python -m pip install --no-deps networkx==2.8.8

python -m pip install --no-deps mediapipe
python -m pip install absl-py flatbuffers sounddevice

pip install sounddevice

New-Item -ItemType Directory -Force "src\models\MediaPipe"

curl.exe -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -o "src\models\MediaPipe\hand_landmarker.task"

curl.exe -L "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -o "src\models\MediaPipe\pose_landmarker_full.task"

# 2. 젯슨 기본 설치 목록
python -m pip install --upgrade pip setuptools wheel pip uninstall matplotlib-inline pip install "matplotlib-inline<0.2" pip install --no-deps \     scikit-image==0.19.3 \     imageio==2.19.3 \     tifffile==2022.5.4 \     PyWavelets==1.3.0 \     networkx==2.8.8

mkdir -p src/models/MediaPipe

pip install --no-deps mediapipe
pip install absl-py flatbuffers sounddevice

wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -O src/models/MediaPipe/hand_landmarker.task

wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task -O src/models/MediaPipe/pose_landmarker_full.task