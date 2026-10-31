# ==============================================================================
# 1. GPU 가속 (잔상효과, Convolution2D)
# ==============================================================================

import time
import numpy as np
import cupy as cp

IMAGE_PATH = "src/images/seagull.jpg"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TRAIL_LENGTH = 32

# NumPy를 사용하여 잔상 효과 함수 정의
def motion_trail_np(image, weights):
    """
    NumPy를 이용해 이미지 잔상 효과를 생성한다.
    출력 픽셀을 Python 반복문으로 하나씩 계산한다.
    """
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.float32)

    for shift in range(TRAIL_LENGTH):
        weight = weights[shift]
        if shift == 0:
            output += image * weight
        else:
            # 오른쪽으로 shift 픽셀만큼 이동
            shifted_image = np.zeros_like(image)
            shifted_image[:, shift:] = image[:, :-shift]
            output += shifted_image * weight

    return output

# CuPy를 사용하여 잔상 효과 함수 정의 (TODO 구현 포함)
def motion_trail_cp(image, weights):
    """
    CuPy를 이용해 이미지 잔상 효과를 생성한다.
    이동 거리는 Python에서 순차적으로 반복하지만,
    각 이동 거리의 이미지 전체 계산은 GPU에서 병렬 처리한다.
    """
    height, width = image.shape
    output = cp.zeros((height, width), dtype=cp.float32)

    for shift in range(TRAIL_LENGTH):
        weight = weights[shift]
        if shift == 0:
            output += image * weight
        else:
            shifted_image = cp.zeros_like(image)
            shifted_image[:, shift:] = image[:, :-shift]
            output += shifted_image * weight

    return output

# 2차원 Convolution (CPU 버전)
def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    h, w = image.shape
    kh, kw = kernel.shape
    out_h = (h - kh + 2 * padding) // stride + 1
    out_w = (w - kw + 2 * padding) // stride + 1
    
    if padding > 0:
        padded = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    else:
        padded = image

    output = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            region = padded[i*stride:i*stride+kh, j*stride:j*stride+kw]
            output[i, j] = np.sum(region * kernel) + bias
    return output

# CuPy를 사용하여 2차원 convolution 함수 정의 (TODO 구현 완료)
def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """
    CuPy를 이용한 2차원 convolution.
    커널의 위치는 Python 반복문으로 순차 처리하고,
    각 위치에 해당하는 출력 배열 전체 연산은 GPU에서 처리한다.
    """
    h, w = image.shape
    kh, kw = kernel.shape
    out_h = (h - kh + 2 * padding) // stride + 1
    out_w = (w - kw + 2 * padding) // stride + 1
    
    if padding > 0:
        padded = cp.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    else:
        padded = image

    output = cp.zeros((out_h, out_w), dtype=cp.float32)
    for i in range(out_h):
        for j in range(out_w):
            region = padded[i*stride:i*stride+kh, j*stride:j*stride+kw]
            output[i, j] = cp.sum(region * kernel) + bias
    return output

# 필터 및 테스트 실행 코드
dx_edge_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
dy_edge_filter = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
dx_bias = -0.2
dy_bias = -0.2

def run_gpu_accelerations():
    # 이미지 가상 생성 (테스트용)
    input_np = np.random.rand(720, 1280).astype(np.float32)
    
    # 1) 잔상 효과 연산 비교
    weights_np = np.linspace(1.0, 0.05, TRAIL_LENGTH, dtype=np.float32)
    weights_np /= np.sum(weights_np)
    
    t0 = time.perf_counter()
    output_np = motion_trail_np(input_np, weights_np)
    cpu_time = time.perf_counter() - t0

    input_cp = cp.asarray(input_np)
    weights_cp = cp.asarray(weights_np)
    
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    output_cp = motion_trail_cp(input_cp, weights_cp)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - t1

    print(f"[Motion Trail] CPU Time: {cpu_time:.4f}s, GPU Time: {gpu_time:.4f}s")

    # 2) Convolution2D 연산 비교
    t0 = time.perf_counter()
    output_dx_np = convolution2d(input_np, dx_edge_filter, bias=dx_bias)
    cpu_conv_time = time.perf_counter() - t0

    kernel_dx_cp = cp.asarray(dx_edge_filter)
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    output_dx_cp = convolution2d_cp(input_cp, kernel_dx_cp, bias=dx_bias)
    cp.cuda.Stream.null.synchronize()
    gpu_conv_time = time.perf_counter() - t1

    print(f"[Conv2D] CPU Time: {cpu_conv_time:.4f}s, GPU Time: {gpu_conv_time:.4f}s")


# ==============================================================================
# 2. MNIST 데이터셋 main_gpu() 함수 및 모델 구현
# ==============================================================================

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SEED = 42
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
LEARNING_RATE = 0.001
EPOCHS = 3
NUM_WORKERS = 0
DATA_ROOT = "src/datasets"

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_count += torch.sum(preds == labels).item()
        sample_count += images.size(0)

    return total_loss / sample_count, correct_count / sample_count

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_count += torch.sum(preds == labels).item()
            sample_count += images.size(0)

    return total_loss / sample_count, correct_count / sample_count

def main_gpu(): 
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=DATA_ROOT,
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root=DATA_ROOT,
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    training_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        if device.type == "cuda":
            torch.cuda.synchronize() 
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        epoch_time = time.perf_counter() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: {test_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    total_training_time = time.perf_counter() - training_start_time
    print(f"Total GPU/CPU training time: {total_training_time:.2f}s")

    return total_training_time


# ==============================================================================
# 3. CIFAR-10 데이터셋 모델 학습 파트 + 저장한 모델 파일 (.pth)
# ==============================================================================

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.block(x)

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_cifar10_complete():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    train_dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    EPOCHS = 5
    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        epoch_time = time.perf_counter() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: {test_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    total_training_time = time.perf_counter() - training_start_time
    print(f"Total GPU CIFAR-10 training time: {total_training_time:.2f}s")

    # 모델 및 상태 저장 (.pth)
    import os
    os.makedirs("src/models/CIFAR10", exist_ok=True)

    save_data = {
        "model_name": "CIFAR10CNN",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_accuracy": test_accuracy_history[-1],
        "epochs": EPOCHS,
        "class_names": CIFAR10_CLASSES,
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
    }

    torch.save(save_data, "src/models/CIFAR10/CIFAR10_CNN.pth")
    print("Saved CIFAR-10 model to src/models/CIFAR10/CIFAR10_CNN.pth")

if __name__ == "__main__":
    print("Running GPU Acceleration Tasks...")
    run_gpu_accelerations()
    
    print("\nRunning MNIST GPU Training...")
    main_gpu()
    
    print("\nRunning CIFAR-10 Training & Saving Model...")
    train_cifar10_complete()