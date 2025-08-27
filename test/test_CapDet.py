import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Config 
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start Streaming
pipeline.start(config)

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Result after the model
        results = model(color_image)
        annotated = results[0].plot()

        # Display
        cv2.imshow("Color", annotated)
        if cv2.waitKey(1) == 27: #ESC to quit
            break

finally:
    pipeline.stop()

