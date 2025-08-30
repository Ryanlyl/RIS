import pyrealsense2 as rs
import numpy as np
from math import atan2, sqrt

# --- Setup streams ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# Align depth to color so pixel coords line up with YOLO
align = rs.align(rs.stream.color)

try:
    # Wait for a coherent pair
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color = aligned.get_color_frame()
    depth = aligned.get_depth_frame()
    if not color or not depth:
        raise RuntimeError("No frames")

    # Get intrinsics of the (aligned) depth stream
    dprofile = depth.profile.as_video_stream_profile()
    intr = dprofile.get_intrinsics()
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

    # Suppose YOLO gives you bbox center (u, v)
    u, v = 320, 240  # <-- replace with your YOLO center

    # Median depth from a small window to reduce noise
    win = 3
    zs = []
    for dv in range(-win, win+1):
        for du in range(-win, win+1):
            uu = int(np.clip(u+du, 0, depth.get_width()-1))
            vv = int(np.clip(v+dv, 0, depth.get_height()-1))
            z = depth.get_distance(uu, vv)  # meters
            if z > 0:
                zs.append(z)
    if not zs:
        raise RuntimeError("No valid depth at target")
    Z = float(np.median(zs))

    # Deproject pixel to 3D camera coordinates
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z

    # Polar coordinates
    r = sqrt(X*X + Y*Y + Z*Z)            # full 3D range
    azimuth = atan2(X, Z)                # yaw (left/right)
    elevation = atan2(-Y, sqrt(X*X+Z*Z)) # pitch (up/down)

    print(f"Camera-frame Cartesian: X={X:.3f} m, Y={Y:.3f} m, Z={Z:.3f} m")
    print(f"Polar: r={r:.3f} m, azimuth={azimuth:.3f} rad, elevation={elevation:.3f} rad")

finally:
    pipeline.stop()
