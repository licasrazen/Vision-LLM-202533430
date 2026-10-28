import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cupy as cp



SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
TRAIL_LENGTH = 32

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="src/datasets", train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root="src/datasets", train=False, transform=transform, download=True
)

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=1000, shuffle=False, num_workers=0
)


class SimpleCNN(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(
        in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1
    )
    self.conv2 = nn.Conv2d(
        in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
    )
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(16 * 7 * 7, 64)
    self.fc2 = nn.Linear(64, 10)

  def forward(self, x, return_features=False):
    conv1_output = self.conv1(x)
    relu1_output = F.relu(conv1_output)
    pool1_output = self.pool(relu1_output)
    conv2_output = self.conv2(pool1_output)
    relu2_output = F.relu(conv2_output)
    pool2_output = self.pool(relu2_output)
    flattened = torch.flatten(pool2_output, start_dim=1)
    fc1_output = F.relu(self.fc1(flattened))
    logits = self.fc2(fc1_output)

    if return_features:
      features = {
          "conv1": conv1_output,
          "relu1": relu1_output,
          "pool1": pool1_output,
          "conv2": conv2_output,
          "relu2": relu2_output,
          "pool2": pool2_output,
      }
      return logits, features
    return logits


def train_one_epoch(model, data_loader, criterion, optimizer, device):
  model.train()
  total_loss, correct_count, sample_count = 0.0, 0, 0
  for images, labels in data_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    batch_size = images.size(0)
    total_loss += loss.item() * batch_size
    predictions = logits.argmax(dim=1)
    correct_count += (predictions == labels).sum().item()
    sample_count += batch_size

  return total_loss / sample_count, correct_count / sample_count


def evaluate(model, data_loader, criterion, device):
  model.eval()
  total_loss, correct_count, sample_count = 0.0, 0, 0
  with torch.no_grad():
    for images, labels in data_loader:
      images, labels = images.to(device), labels.to(device)
      logits = model(images)
      loss = criterion(logits, labels)

      batch_size = images.size(0)
      total_loss += loss.item() * batch_size
      predictions = logits.argmax(dim=1)
      correct_count += (predictions == labels).sum().item()
      sample_count += batch_size

  return total_loss / sample_count, correct_count / sample_count


def convolution2d(image, kernel, bias=-0.2, stride=1, padding=1):
  ih, iw = image.shape
  kh, kw = kernel.shape
  padded = np.pad(image, ((padding, padding), (padding, padding)), mode="constant")
  oh = (padded.shape[0] - kh) // stride + 1
  ow = (padded.shape[1] - kw) // stride + 1
  output = np.zeros((oh, ow), dtype=np.float32)

  for y in range(oh):
    for x in range(ow):
      region = padded[y * stride : y * stride + kh, x * stride : x * stride + kw]
      output[y, x] = np.sum(region * kernel) + bias
  return output


model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 3
train_loss_history, train_accuracy_history = [], []
test_loss_history, test_accuracy_history = [], []

for epoch in range(epochs):
  t_loss, t_acc = train_one_epoch(
      model, train_loader, criterion, optimizer, device
  )
  v_loss, v_acc = evaluate(model, test_loader, criterion, device)
  train_loss_history.append(t_loss)
  train_accuracy_history.append(t_acc)
  test_loss_history.append(v_loss)
  test_accuracy_history.append(v_acc)


# ==========================================
# 원본 코드 및 TODO 그대로 유지한 구현부
# ==========================================

# TODO: plt.plot(x축_데이터, y축_데이터, marker="o", label="데이터_이름")을 사용하여 그래프 생성
plt.figure(figsize=(8, 5))

# TODO: train_loss_history 그래프 생성
plt.plot(
    range(1, epochs + 1),
    train_loss_history,
    marker="o",
    label="Train Loss",
)
# TODO: test_loss_history 그래프 생성
plt.plot(
    range(1, epochs + 1),
    test_loss_history,
    marker="s",
    label="Test Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Loss")
plt.grid()
plt.legend()
plt.show()


# TODO: plt.plot(x축_데이터, y축_데이터, marker="o", label="데이터_이름")을 사용하여 그래프 생성
plt.figure(figsize=(8, 5))

# TODO: train_accuracy_history 그래프 생성 (백분율)
plt.plot(
    range(1, epochs + 1),
    [a * 100 for a in train_accuracy_history],
    marker="o",
    label="Train Accuracy",
)
# TODO: test_accuracy_history 그래프 생성 (백분율)
plt.plot(
    range(1, epochs + 1),
    [a * 100 for a in test_accuracy_history],
    marker="s",
    label="Test Accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("CNN Accuracy")
plt.grid()
plt.legend()
plt.show()


# TODO: CNN 각 층의 Feature Map 시각화
# TODO: cmap="viridis" 사용


# TODO: 이미지 선택
image, true_label = test_dataset[0]

# TODO: 역전파(파라미터 수정) 없이 순전파 수행
model.eval()
feature_maps = {}


def get_hook(name):

  def hook(model, input, output):
    feature_maps[name] = output.detach()

  return hook


model.conv1.register_forward_hook(get_hook("conv1"))
model.relu1.register_forward_hook(get_hook("relu1"))
model.pool1.register_forward_hook(get_hook("pool1"))
model.conv2.register_forward_hook(get_hook("conv2"))
model.relu2.register_forward_hook(get_hook("relu2"))
model.pool2.register_forward_hook(get_hook("pool2"))

with torch.no_grad():
  # 배치 차원 추가 및 디바이스 이동 후 순전파 실행
  input_tensor = image.unsqueeze(0).to(device)
  output = model(input_tensor)


# Feature Map 시각화 보조 함수
def show_feature_maps(fmap, num_plots=8, cmap="viridis"):
  fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 1.5, 2))
  for i in range(num_plots):
    ax = axes[i] if num_plots > 1 else axes
    ax.imshow(fmap[0, i].cpu().numpy(), cmap=cmap)
    ax.axis("off")
  plt.show()


# TODO: Conv1 Feature Maps 8개 시각화
show_feature_maps(feature_maps["conv1"], num_plots=8, cmap="viridis")

# TODO: Conv2 Feature Maps 16개 시각화
show_feature_maps(feature_maps["conv2"], num_plots=16, cmap="viridis")

# TODO: 6개 Layer별 대표 Feature Map 4개씩 비교
layer_names = ["conv1", "relu1", "pool1", "conv2", "relu2", "pool2"]

fig, axes = plt.subplots(
    len(layer_names), 4, figsize=(10, len(layer_names) * 2)
)
for row, name in enumerate(layer_names):
  fmap = feature_maps[name]
  for col in range(4):
    ax = axes[row, col]
    if col < fmap.shape[1]:
      ax.imshow(fmap[0, col].cpu().numpy(), cmap="viridis")
    ax.axis("off")
    if col == 0:
      ax.set_ylabel(name, fontsize=10, rotation=0, labelpad=30, va="center")

plt.tight_layout()
plt.show()


# TODO: NumPy를 사용하여 잔상 효과 함수 정의


def motion_trail_np(image, weights):

  """NumPy를 이용해 이미지 잔상 효과를 생성한다.

  출력 픽셀을 Python 반복문으로 하나씩 계산한다.
  """

  # TODO: 필요한 변수 정의 및 출력 배열 생성
  output = np.zeros_like(image, dtype=np.float32)

  for output_y in range(image.shape[0]):
    for output_x in range(image.shape[1]):

      for shift in range(TRAIL_LENGTH):
        # TODO: x값 shift
        input_x = output_x - shift

        # TODO: shift한 픽셀을 누적
        # TODO: 출력 배열에 누적 픽셀 저장
        if 0 <= input_x < image.shape[1]:
          output[output_y, output_x] += image[output_y, input_x] * weights[shift]

  return output


# TODO: CuPy를 사용하여 잔상 효과 함수 정의


def motion_trail_cp(image, weights):
  """CuPy를 이용해 이미지 잔상 효과를 생성한다.

  이동 거리는 Python에서 순차적으로 반복하지만, 각 이동 거리의 이미지 전체 계산은 GPU에서 병렬 처리한다.
  """

  # TODO: 필요한 변수 정의 및 출력 배열 생성
  output = cp.zeros_like(image, dtype=cp.float32)

  for shift in range(TRAIL_LENGTH):
    # TODO: 배열 slicing
    if shift == 0:
      input_region = image[:, :]
      output_region = output[:, :]
    else:
      input_region = image[:, :-shift]
      output_region = output[:, shift:]

    # TODO: GPU가 병렬 연산할 output_region 계산 코드
    output_region += input_region * weights[shift]

  return output


# 이동 거리가 멀어질수록 가중치를 작게 설정
weights_np = np.linspace(1.0, 0.05, TRAIL_LENGTH, dtype=np.float32)

# 전체 가중치의 합을 1로 정규화
weights_np /= np.sum(weights_np)

input_np = image.squeeze(0).numpy().astype(np.float32)

cpu_start = time.perf_counter()

output_np = motion_trail_np(image=input_np, weights=weights_np)

cpu_end = time.perf_counter()

cpu_time = cpu_end - cpu_start


# TODO: NumPy 배열을 CuPy 배열로 변환

upload_start = time.perf_counter()

input_cp = cp.asarray(input_np)
weights_cp = cp.asarray(weights_np)

cp.cuda.Stream.null.synchronize()

upload_end = time.perf_counter()

upload_time = upload_end - upload_start


warmup_image = cp.zeros((64, 64), dtype=cp.float32)
warmup_weights = cp.asarray(weights_np[:4])

_ = motion_trail_cp(image=warmup_image, weights=warmup_weights)

cp.cuda.Stream.null.synchronize()


cp.cuda.Stream.null.synchronize()

gpu_start = time.perf_counter()

output_cp = motion_trail_cp(image=input_cp, weights=weights_cp)

cp.cuda.Stream.null.synchronize()

gpu_end = time.perf_counter()

gpu_time = gpu_end - gpu_start


# TODO: output_cp를 CPU 메모리로 복사

download_start = time.perf_counter()

output_cp_np = output_cp.get()

download_end = time.perf_counter()

download_time = download_end - download_start

compute_speedup = cpu_time / gpu_time

gpu_total_time = upload_time + gpu_time + download_time
total_speedup = cpu_time / gpu_total_time


print("========== NumPy CPU ==========")
print(f"Processing time: {cpu_time:.4f} seconds")
print()

print("========== CuPy GPU ==========")
print(f"CPU to GPU: {upload_time:.4f} seconds")
print(f"GPU processing: {gpu_time:.4f} seconds")
print(f"GPU to CPU: {download_time:.4f} seconds")
print(f"Total GPU time: {gpu_total_time:.4f} seconds")
print()

print("========== Comparison ==========")
print(f"Compute-only speedup: {compute_speedup:.2f}x")
print(f"Including transfers: {total_speedup:.2f}x")


plt.figure(figsize=(18, 12))

plt.subplot(1, 3, 1), plt.imshow(input_np, cmap="gray"), plt.title(
    "Original Image"
)
plt.subplot(1, 3, 2), plt.imshow(output_np, cmap="gray"), plt.title(
    "Motion Trail (CPU)"
)
plt.subplot(1, 3, 3), plt.imshow(output_cp_np, cmap="gray"), plt.title(
    "Motion Trail (GPU)"
)

for ax in plt.gcf().axes:
  ax.axis("off")


plt.figure(figsize=(12, 8))
plt.imshow(output_cp_np, cmap="gray")
plt.axis("off")


plt.imshow(input_np, cmap="gray")
plt.axis("off")


# TODO: CuPy를 사용하여 2차원 convolution 함수 정의


def convolution2d_cp(image, kernel, bias=-0.2, stride=1, padding=1):
  """CuPy를 이용한 2차원 convolution.

  커널의 위치는 Python 반복문으로 순차 처리하고, 각 위치에 해당하는 출력 배열 전체 연산은 GPU에서 처리한다.
  """
  kh, kw = kernel.shape
  h, w = image.shape

  if padding > 0:
    image_padded = cp.pad(
        image, ((padding, padding), (padding, padding)), mode="constant"
    )
  else:
    image_padded = image

  oh = (h + 2 * padding - kh) // stride + 1
  ow = (w + 2 * padding - kw) // stride + 1

  output = cp.full((oh, ow), bias, dtype=cp.float32)

  for ky in range(kh):
    for kx in range(kw):
      img_slice = image_padded[
          ky : ky + oh * stride : stride, kx : kx + ow * stride : stride
      ]
      output += img_slice * kernel[ky, kx]

  return output


# TODO: 기존에 정의한 convolution2d 함수를 사용하여 convolution을 CPU로 처리
# TODO: 변수 재사용 가능

kernel_dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
kernel_dy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

cpu_start = time.perf_counter()

output_dx_np = convolution2d(input_np, kernel_dx)
output_dy_np = convolution2d(input_np, kernel_dy)

cpu_end = time.perf_counter()
cpu_time = cpu_end - cpu_start

relu_dx_np = np.maximum(0, output_dx_np)
relu_dy_np = np.maximum(0, output_dy_np)


# TODO: convolution을 GPU로 처리 (변환-워밍업-처리 파이프라인 구현)
# TODO: 변수 재사용 가능

gpu_start = time.perf_counter()

output_dx_cp = convolution2d_cp(input_cp, cp.asarray(kernel_dx))
output_dy_cp = convolution2d_cp(input_cp, cp.asarray(kernel_dy))

cp.cuda.Stream.null.synchronize()
gpu_end = time.perf_counter()
gpu_time = gpu_end - gpu_start

relu_dx_cp_np = cp.asnumpy(cp.maximum(0, output_dx_cp))
relu_dy_cp_np = cp.asnumpy(cp.maximum(0, output_dy_cp))


# TODO: CPU/GPU 결과 비교 및 출력
print(f"CPU 연산 시간: {cpu_time:.4f} 초")
print(f"GPU 연산 시간: {gpu_time:.4f} 초")
print(f"성능 향상 (Speedup): {cpu_time / gpu_time:.2f}배")

dx_diff = np.max(np.abs(relu_dx_np - relu_dx_cp_np))
dy_diff = np.max(np.abs(relu_dy_np - relu_dy_cp_np))

print(f"X방향 연산 최대 오차: {dx_diff}")
print(f"Y방향 연산 최대 오차: {dy_diff}")

if np.allclose(relu_dx_np, relu_dx_cp_np, atol=1e-5) and np.allclose(
    relu_dx_np, relu_dy_cp_np, atol=1e-5
):
  print("결과 검증 성공: CPU와 GPU의 결과가 일치합니다.")
else:
  print("결과 검증 경고: CPU와 GPU의 결과에 차이가 있습니다.")


plt.figure(figsize=(18, 8))

plt.subplot(2, 3, 1), plt.imshow(input_np, cmap="gray"), plt.title(
    "Input Image"
)
plt.subplot(2, 3, 2), plt.imshow(relu_dx_np, cmap="gray"), plt.title(
    "Vertical Convolution (CPU)"
)
plt.subplot(2, 3, 3), plt.imshow(relu_dx_cp_np, cmap="gray"), plt.title(
    "Vertical Convolution (GPU)"
)
plt.subplot(2, 3, 4), plt.imshow(input_np, cmap="gray"), plt.title(
    "Input Image"
)
plt.subplot(2, 3, 5), plt.imshow(relu_dy_np, cmap="gray"), plt.title(
    "Horizontal Convolution (CPU)"
)
plt.subplot(2, 3, 6), plt.imshow(relu_dy_cp_np, cmap="gray"), plt.title(
    "Horizontal Convolution (GPU)"
)

for ax in plt.gcf().axes:
  ax.axis("off")