from ultralytics import YOLO

# 加载预训练模型 (YOLOv8n - nano 版，速度快)
model = YOLO("yolov8n.pt")

# 对图片做目标检测
results = model("image.jpg")

# 显示结果
results[0].show()
