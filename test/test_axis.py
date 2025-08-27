from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("image.jpg")

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])     # 类别 ID
        conf = float(box.conf[0])    # 置信度
        xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

        print(f"Class: {model.names[cls_id]}, Conf: {conf:.2f}, BBox: {xyxy}")
        print("Central:", [int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)])
