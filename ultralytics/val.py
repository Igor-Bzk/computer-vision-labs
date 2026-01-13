from ultralytics import YOLO

model = YOLO("/home/bezmaternykh/Bureau/Deep Learning/all_classes/yolo11n_img1024_ep20/weights/best.pt")
# model = YOLO("runs/detect/train12/weights/best.pt")
model.val(data="data_new.yaml", save_json=True, save_hybrid=True)