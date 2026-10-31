# CNN 및 GPU 가속 연산과 객체 탐지 알고리즘 구현

## 파일 구조 (필수)

- **assignment03_1.py**: NumPy와 CuPy를 이용한 CPU/GPU 연산 시간 비교 및 PyTorch 기반 MNIST, CIFAR-10 CNN 모델 학습
- **assignment03_2.py**: Bounding Box 좌표 변환, IoU 계산, Confidence Thresholding 및 NMS(Non-Maximum Suppression) 구현

## 실행 방식 (필수)

Python 환경에서 필요한 라이브러리를 설치한 후 각 파일을 실행합니다.

**필요 라이브러리**
- NumPy
- CuPy
- PyTorch
- torchvision
- OpenCV
- Matplotlib
- scikit-image

**라이브러리 설치**
```bash
pip install numpy cupy-cuda12x torch torchvision matplotlib opencv-python scikit-image
```

**실행**
```bash
python assignment03_1.py
python assignment03_2.py
```

- `assignment03_1.py` 실행 시 NumPy와 CuPy의 Motion Trail 및 2D Convolution 연산 시간을 비교한 뒤 MNIST와 CIFAR-10 데이터셋을 이용해 CNN 모델을 학습합니다.
- `assignment03_2.py` 실행 시 Bounding Box의 IoU를 계산하고 Confidence Thresholding과 NMS를 적용하여 중복된 Bounding Box를 제거합니다.
- 데이터셋은 코드 실행 과정에서 다운로드될 수 있습니다.
- `assignment03_1.py`의 CIFAR-10 학습 모델은 `src/models/CIFAR10/CIFAR10_CNN.pth`에 저장됩니다.



## 기타 참고 사항
1. CNN 모델 구현

MNIST 데이터셋을 이용하여 손글씨 숫자 0~9를 분류하는 CNN 모델을 구현하였습니다.

CIFAR-10 데이터셋을 이용하여 10개 클래스를 분류하는 CNN 모델을 구현하였습니다.

MNIST 모델은 두 개의 Convolution 계층과 Max Pooling 계층을 사용하였으며, CIFAR-10 모델은 ConvBlock과 Batch Normalization, Dropout을 적용하였습니다.

CIFAR-10 학습 과정에서 Random Crop과 Random Horizontal Flip을 적용하여 데이터 증강을 수행하였습니다.

CrossEntropyLoss를 손실 함수로 사용하고, MNIST에는 Adam, CIFAR-10에는 AdamW Optimizer를 적용하였습니다.

학습 및 평가 과정에서 Loss와 Accuracy를 계산하였습니다.

2. CPU 및 GPU 연산 비교

NumPy와 CuPy를 이용하여 Motion Trail 연산을 각각 구현하고 실행 시간을 비교하였습니다.

2D Convolution 연산을 CPU와 GPU 환경에서 수행하고 실행 시간을 측정하였습니다.

GPU 연산 시간 측정 시 CUDA 동기화를 적용하여 비동기 실행에 따른 측정 오차를 줄이고자 하였습니다.

3. 객체 탐지 알고리즘 구현

Bounding Box 좌표를 (x, y, w, h) 형식에서 (x1, y1, x2, y2) 형식으로 변환하였습니다.

두 Bounding Box의 교집합과 합집합을 이용하여 IoU를 계산하였습니다.

Confidence Thresholding을 적용하여 기준 점수보다 낮은 Bounding Box를 제거하였습니다.

NMS를 통해 Confidence Score가 높은 Bounding Box를 우선적으로 유지하고, 중복되는 Bounding Box를 제거하였습니다.

4. 추가 설명

PyTorch 모델은 CUDA 사용 가능 여부에 따라 GPU 또는 CPU를 선택하여 학습하도록 구현하였습니다.

CIFAR-10 학습이 완료되면 모델 가중치, Optimizer 상태, 테스트 정확도 및 클래스 정보를 저장하도록 구현하였습니다.

MNIST 모델은 학습 및 평가를 수행하지만 별도의 모델 파일 저장 기능은 구현하지 않았습니다.
