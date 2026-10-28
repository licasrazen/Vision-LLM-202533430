# PyTorch CNN 모델 학습 및 CuPy GPU 가속 연산 성능 비교

## 파일 구조 (필수)

* **`cnn_cupy_experiment.py`**: SimpleCNN 모델 학습, Feature Map 시각화, NumPy(CPU) 대 CuPy(GPU) 잔상 효과 및 2D Convolution 연산 속도 비교 코드가 포함된 메인 스크립트.

## 실행 방식 (필수)

가상환경 생성 및 활성화:

```bash
py -3.10 -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1

```

필수 패키지 설치:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision matplotlib numpy cupy-cuda12x

```

스크립트 실행:

```bash
python cnn_cupy_experiment.py

```

## 기타 참고 사항
젯슨에서만 돌아가는 것들이 있어서 일단 초안으로 코드만 채워놓았습니다.
