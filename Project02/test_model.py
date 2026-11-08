from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source=0,
    conf=0.65,
    iou=0.5,
    show=True,
    save=True
)