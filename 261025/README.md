# CNN 및 GPU 가속 연산과 객체 탐지 알고리즘 구현

## 파일 구조 (필수)

- **assignment03_1.py**: NumPy와 CuPy를 이용한 CPU/GPU 연산 시간 비교 및 PyTorch 기반 MNIST, CIFAR-10 CNN 모델 학습
- **assignment03_2.py**: Bounding Box 좌표 변환, IoU 계산, Confidence Thresholding 및 NMS(Non-Maximum Suppression) 구현
- **CIFAR10_CNN.pth**: 과제 assignment03_1.py를 실행하면 나오는 파일

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
###**assignment03_1.py**
**1. GPU 가속 (잔상효과, Convolution2D)**
NumPy를 사용하여 잔상 효과 함수 정의: np.zeros()와 np.zeros_like()를 통해 이미지와 동일한 크기의 0 배열을 생성하여 오른쪽으로 shift 픽셀만큼 이동하고, 거리별 가중치를 곱하여 결과 배열에 누적하였습니다.

CuPy를 사용하여 잔상 효과 함수 정의: np만 cp로 바꾸었습니다.

2차원 Convolution (CPU 버전): convolution2d 함수를 이용하여 입력 이미지에 커널을 적용하고, 이미지 영역과 커널의 합성곱을 통해 2차원 Convolution 연산을 구현하였습니다.

CuPy를 사용하여 2차원 convolution 함수 정의: 역시 np만 cp로 바꾸었습니다.

x, y방향 경계선 필터를 만들고 잔상 효과와 컨볼루션2D의 연산을 비교하였습니다. 

**2. MNIST 데이터셋 main_gpu() 함수 및 모델 구현**
클래스 SimpleCNN을 이용해 CNN 모델을 정의하고, 최종적으로 추출한 특징을 10개 숫자 클래스(0~9)로 분류했습니다. 그리고 train_one_epoch로 학습 데이터를 입력하고, 가중치를 업데이트 했습니다. evaluate로는 평가 데이터를 입력해서 정답을 비교하고 손실과 정확도를 계산했습니다.

main_gpu 함수를 이용하여 MNIST 데이터셋을 불러오고 DataLoader를 통해 학습 및 테스트 데이터를 배치 단위로 구성하였습니다. CUDA 사용 가능 여부에 따라 CPU 또는 GPU를 선택하고, CNN 모델을 Adam 최적화 알고리즘과 CrossEntropyLoss를 이용하여 학습하였습니다. 에포크별 Loss, Accuracy 및 전체 학습 시간을 측정하였습니다.


**3. CIFAR-10 데이터셋 모델 학습 파트 + 저장한 모델 파일 (.pth)**
CIFAR-10 데이터셋을 이용하여 10개 클래스의 이미지 분류 CNN 모델을 구현하였습니다. Random Crop과 Random Horizontal Flip을 적용하여 학습 데이터를 증강하고, 5 Epoch 동안 모델을 학습 및 평가하였습니다. 학습 시간과 Loss 및 Accuracy를 측정하고, 학습된 모델을 .pth 파일로 저장하였습니다.

###**assignment03_2.py**
**1. IoU 구현**
OpenCV의 `boundingRect()`를 활용하여 BBox 좌표를 `(x, y, w, h)`에서 `(x1, y1, x2, y2)` 형식으로 변환하였습니다. 두 BBox의 교집합 및 합집합 면적을 계산하여 IoU를 산출하였습니다. 허블 딥 필드 이미지를 활용하여 랜덤한 위치와 크기의 BBox 및 신뢰도 점수를 생성하였습니다. Confidence Thresholding을 구현하여 설정한 임계값보다 낮은 신뢰도 점수를 가진 BBox를 제거하였습니다. BBox와 신뢰도 점수를 이미지에 시각화하여 결과를 확인하였습니다.

 **2. NMS 구현**
 신뢰도 점수를 내림차순으로 정렬하고, 최고 점수 BBox를 선택했습니다. IoU 비교를 통한 중복 BBox는 NMS로 제거하고, 적용 전후 Bounding Box 개수 및 결과를 시각화하였습니다.
