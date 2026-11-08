
from pathlib import Path
import random
import shutil

# 데이터셋 경로
ROOT = Path("src")

train_images = ROOT / "train" / "images"
train_labels = ROOT / "train" / "labels"

valid_images = ROOT / "valid" / "images"
valid_labels = ROOT / "valid" / "labels"

# 검증 데이터 비율
VALID_RATIO = 0.2

# 이미지 확장자
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    random.seed(42)

    # 폴더 생성
    valid_images.mkdir(parents=True, exist_ok=True)
    valid_labels.mkdir(parents=True, exist_ok=True)

    # 이미지 목록 수집
    images = [
        p for p in train_images.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # 이미지와 라벨 파일명 확인
    pairs = []
    unmatched = []

    for image in images:
        label = train_labels / f"{image.stem}.txt"

        if label.exists():
            pairs.append((image, label))
        else:
            unmatched.append(image.name)

    if unmatched:
        raise RuntimeError(
            f"라벨이 없는 이미지가 {len(unmatched)}장 있습니다. "
            f"예시: {unmatched[:5]}"
        )

    if not pairs:
        raise RuntimeError("분리할 이미지가 없습니다.")

    # 무작위로 섞기
    random.shuffle(pairs)

    # 20%를 검증 데이터로 분리
    valid_count = max(1, int(len(pairs) * VALID_RATIO))
    valid_pairs = pairs[:valid_count]

    # 이미지와 라벨을 함께 이동
    for image, label in valid_pairs:
        shutil.move(str(image), str(valid_images / image.name))
        shutil.move(str(label), str(valid_labels / label.name))

    print("데이터 분리 완료")
    print(f"학습 데이터: {len(pairs) - valid_count}장")
    print(f"검증 데이터: {valid_count}장")
    print(f"테스트 데이터: 기존 데이터 유지")

if __name__ == "__main__":
    main()
