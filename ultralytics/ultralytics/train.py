from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.train(data='data.yaml', epochs=18, imgsz=1024, device=-1)