import pyrealsense2 as rs
import numpy as np
import cv2

# pipeline and configuration
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# start the pipeline and get the depth scale
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()  # depth scale -> meter

"""
Align the depth to color
in order to get the pixel distance from the color image
"""
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        """
        Display the color image
        Compute the average of central small window to make the value stable
        """
        color = np.asanyarray(color_frame.get_data())
        h, w, _ = color.shape
        cx, cy = w // 2, h // 2

        """
        Get the meter scale depth from depth_frame (considered inner scale)
        Note: get_distance is in meter, no need to multiple depth_scale again
        """
        center_dist_m = depth_frame.get_distance(cx, cy)

        """
        calculate the average in a 5x5 window (filter the zero/cavity)
        """
        win = 5
        xs = np.clip(np.arange(cx - win//2, cx + win//2 + 1), 0, w-1)
        ys = np.clip(np.arange(cy - win//2, cy + win//2 + 1), 0, h-1)
        dvals = []
        for yy in ys:
            for xx in xs:
                d = depth_frame.get_distance(int(xx), int(yy))
                if d > 0:  # delete the invalid depth
                    dvals.append(d)
        roi_mean_m = float(np.mean(dvals)) if dvals else 0.0

        # display with caption
        vis = color.copy()
        cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)
        txt = f"center: {center_dist_m:.3f} m | ROI mean: {roi_mean_m:.3f} m"
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
        cv2.imshow('color', vis)
        if cv2.waitKey(1) == 27:  # ESC to exit
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
