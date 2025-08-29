import pyrealsense2 as rs
import numpy as np 
import cv2
from ultralytics import YOLO

def find_person(image, model):
    central_coordinate = None
    for box in image[0].boxes:
        class_id = int(box.cls[0])
        if model.names[class_id].lower() == "person":
            xyxy = box.xyxy[0].cpu().numpy()
            central_coordinate = [int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)]
    return central_coordinate

def person_distance(coordinate, window_size, image):
    cx, cy = coordinate[0], coordinate[1]
    h, w, _= image.shape
    xs = np.clip(np.arange(cx - window_size//2, cx + window_size//2 + 1), 0, w-1)
    ys = np.clip(np.arange(cy - window_size//2, cy + window_size//2 + 1), 0, h-1)
    dvals = []
    for yy in ys:
        for xx in xs:
            d = depth_frame.get_distance(int(xx), int(yy))
            if d > 0:
                dvals.append(d)
    roi_distance = float(np.mean(dvals)) if dvals else 0.0
    return roi_distance

if __name__ == "__main__":
    # Load the YOLO model
    model = YOLO("yolov8n.pt")

    # Configure the pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start the pipeline and get the depth scale
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale() # Scale: meter

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # YOLO process the image
            results = model(color_image)
            annotated = results[0].plot()

            # Find the central coordinate of box person
            central_coordinate = find_person(image = results, model = model)
            if not central_coordinate:
                print("No image captured.")
                continue

            # Find the distance of person (ROI mean)
            roi_distance = person_distance(coordinate = central_coordinate, window_size = 5, image = color_image)

            # Display with caption
            vis = annotated.copy()
            txt = f"ROI distance of person is {roi_distance:.2f}m"
            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
            cv2.imshow('annotated', vis)
            if cv2.waitKey(1) == 27:
                break
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()