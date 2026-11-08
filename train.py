from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    model.train(
    data="src/data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=2
)

if __name__ == '__main__':
    main()