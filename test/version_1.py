import pyrealsense2 as rs
import numpy as np 
import cv2
import argparse
from ultralytics import YOLO
from math import atan2, sqrt

def find_target(image, model, target="person"):
    central_coordinate = None
    for box in image[0].boxes:
        class_id = int(box.cls[0])
        if model.names[class_id].lower() == target:
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

def get_polar(coordinate, depth, Z):
    u, v = coordinate

    # Get intrinsics of the depth stream
    dprofile = depth.profile.as_video_stream_profile()
    intr = dprofile.get_intrinsics()
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

    # Deproject pixel to 3D camera coordinates
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z

    # Polar coordinates
    r = sqrt(X*X + Y*Y + Z*Z)
    azimuth = atan2(X, Z)
    elevation = atan2(-Y, sqrt(X*X + Z*Z))

    return r, azimuth, elevation

def user_input():
    # Create the parser
    parser = argparse.ArgumentParser(description="User Input")

    # Add arguments
    parser.add_argument("-d", "--detect", type=str, help="The class to be detected.", default="person")

    # Parse the arguments
    args = parser.parse_args()
    return args.detect

if __name__ == "__main__":
    # User input
    detect = user_input()

    # Load the YOLO model
    model = YOLO("yolov8n.pt")

    # Configure the pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start the pipeline and get the depth scale
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # YOLO process the image
            results = model(color_image)
            annotated = results[0].plot()

            # Find the central coordinate of box person
            central_coordinate = find_target(image = results, model = model, target=detect)
            if not central_coordinate:
                print("No target captured.")
                continue

            # Find the distance of person (ROI mean)
            roi_distance = person_distance(coordinate = central_coordinate, window_size = 5, image = color_image)

            # Convert to the Polar coordinate
            r, azimuth, elevation = get_polar(central_coordinate, depth_frame, roi_distance)

            # Display with caption
            vis = annotated.copy()
            txt = f"ROI distance of {detect} is {r:.2f}m, azimuth: {azimuth:.2f}, elevation: {elevation:.2f}"
            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv2.imshow('Camera', vis)
            if cv2.waitKey(1) == 27:
                break
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()