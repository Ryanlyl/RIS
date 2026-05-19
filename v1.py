#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v1.py
PC-side RIS camera tracking + STM32 USART controller UI.

Functions:
1. Tkinter UI for RIS / Tx / tracking / USART parameters.
2. Builds fixed binary USART frames for STM32-side RIS calculation.
3. Supports test/demo target sending without camera.
4. Optional RealSense + YOLO person tracking if required packages are installed.
5. Single target rule: Rx1=(az, el, r), Rx2=(-az, el, r).
6. Two-or-more target rule: Rx1=target0, Rx2=target1.
7. Opens USART, sends frames, continuously receives MCU text feedback,
   and displays feedback in a text box.

USART frame format, little endian:
Header:
  magic      4 bytes: b'RIS1'
  seq        uint16
  payloadlen uint16 = 72
Payload:
  nx         uint16
  ny         uint16
  bits       uint8
  mode       uint8     0=auto, 1=near_field, 2=far_field
  tx_count   uint8
  rx_count   uint8
  freq_hz    uint32
  dx_m       float32
  dy_m       float32
  le_hold_us uint16
  flags      uint16
  tx[2]      each: az float32, el float32, range float32
  rx[2]      each: az float32, el float32, range float32
CRC:
  crc16-ccitt-false over header + payload, uint16 little endian
Total frame length: 82 bytes.
"""

from __future__ import annotations

import math
import os
import queue
import struct
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

try:
    import serial
    from serial.tools import list_ports
except Exception:  # pyserial not installed
    serial = None
    list_ports = None

# Optional camera dependencies. The UI can still run without these.
try:
    import cv2
    import numpy as np
    import pyrealsense2 as rs
    from ultralytics import YOLO
    from scipy.optimize import linear_sum_assignment
    CAMERA_DEPS_OK = True
except Exception:
    cv2 = None
    np = None
    rs = None
    YOLO = None
    linear_sum_assignment = None
    CAMERA_DEPS_OK = False

APP_VERSION = "V1_FULL_CAMERA_USART_RIS_2026_05_16"
MAGIC = b"RIS1"
PAYLOAD_LEN = 72
FRAME_LEN = 8 + PAYLOAD_LEN + 2

MODE_TO_ID = {
    "auto": 0,
    "near_field": 1,
    "far_field": 2,
}


@dataclass
class TargetInfo:
    track_id: int
    azimuth: float
    elevation: float
    distance: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    speed_3d: float = 0.0
    score: float = 1.0
    box: Optional[List[int]] = None
    timestamp: float = 0.0


@dataclass
class Terminal:
    az_deg: float
    el_deg: float
    r_m: float


# -----------------------------
# Protocol helpers
# -----------------------------
def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def build_frame(
    *,
    seq: int,
    nx: int,
    ny: int,
    bits: int,
    mode: str,
    freq_hz: float,
    dx_m: float,
    dy_m: float,
    le_hold_us: int,
    txs: List[Terminal],
    rxs: List[Terminal],
    flags: int = 0,
) -> bytes:
    """Build one STM32 USART control frame."""
    if not (0 <= seq <= 0xFFFF):
        seq = seq & 0xFFFF

    mode_id = MODE_TO_ID.get(mode, 0)
    txs = list(txs[:2])
    rxs = list(rxs[:2])

    while len(txs) < 2:
        txs.append(Terminal(0.0, 0.0, 0.0))
    while len(rxs) < 2:
        rxs.append(Terminal(0.0, 0.0, 0.0))

    tx_count = max(1, min(2, len([t for t in txs if t.r_m > 0.0]) or 1))
    rx_count = max(1, min(2, len([r for r in rxs if r.r_m > 0.0]) or 1))

    payload = struct.pack(
        "<HHBBBBIffHH",
        int(nx),
        int(ny),
        int(bits),
        int(mode_id),
        int(tx_count),
        int(rx_count),
        int(round(freq_hz)),
        float(dx_m),
        float(dy_m),
        int(le_hold_us),
        int(flags),
    )

    for t in txs:
        payload += struct.pack("<fff", float(t.az_deg), float(t.el_deg), float(t.r_m))
    for r in rxs:
        payload += struct.pack("<fff", float(r.az_deg), float(r.el_deg), float(r.r_m))

    if len(payload) != PAYLOAD_LEN:
        raise RuntimeError(f"payload length mismatch: {len(payload)} != {PAYLOAD_LEN}")

    header = MAGIC + struct.pack("<HH", seq & 0xFFFF, PAYLOAD_LEN)
    crc = crc16_ccitt_false(header + payload)
    return header + payload + struct.pack("<H", crc)


def frame_to_hex(frame: bytes) -> str:
    return " ".join(f"{b:02X}" for b in frame)


def list_serial_port_names() -> List[str]:
    if list_ports is None:
        return []
    return [p.device for p in list_ports.comports()]


# -----------------------------
# Optional simple camera tracker
# -----------------------------
class CoordinateConverter:
    """Same camera-to-RIS coordinate converter as camera.py."""

    def __init__(self, intrinsics, offset_y: float = 0.26):
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.cx = intrinsics.ppx
        self.cy = intrinsics.ppy
        self.offset_y = offset_y

    def pixel_to_world(self, u: int, v: int, depth_m: float):
        """Pixel + depth -> RIS-centered 3D coordinate in meters.

        Camera coordinates: X right, Y down, Z forward.
        RIS coordinates: origin shifted down by offset_y; Y positive upward.
        """
        if depth_m <= 0.0:
            return None
        x_cam = (u - self.cx) * depth_m / self.fx
        y_cam = (v - self.cy) * depth_m / self.fy
        z_cam = depth_m
        x_ms = x_cam
        y_ms = -(y_cam - self.offset_y)
        z_ms = z_cam
        return np.array([x_ms, y_ms, z_ms], dtype=float)

    def world_to_angles(self, pos_3d):
        x, y, z = pos_3d
        dist = float(np.linalg.norm(pos_3d))
        if dist < 1e-4:
            return 0.0, 0.0, 0.0
        horiz = math.sqrt(float(x) ** 2 + float(z) ** 2)
        az = math.degrees(math.atan2(float(x), float(z)))
        el = math.degrees(math.atan2(float(y), horiz))
        return dist, az, el


class KalmanFilter3D:
    """3D constant-velocity Kalman filter copied from the standalone camera tracker."""

    def __init__(self, pos_3d, dt: float = 1 / 30.0):
        self.dt = dt
        self.x = np.zeros((6, 1))
        self.x[:3] = pos_3d.reshape(3, 1)
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.Q = np.diag([0.01, 0.01, 0.05, 0.1, 0.1, 0.5])
        self.R = np.diag([0.02, 0.02, 0.05])
        self.P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_position(), self.get_velocity()

    def update(self, pos_3d):
        z = pos_3d.reshape(3, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def get_position(self):
        return self.x[:3, 0].copy()

    def get_velocity(self):
        return self.x[3:, 0].copy()

    def update_dt(self, dt: float):
        self.dt = dt
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

    def predict_future(self, n_frames: int):
        x_tmp = self.x.copy()
        positions = []
        for _ in range(n_frames):
            x_tmp = self.F @ x_tmp
            positions.append(x_tmp[:3, 0].copy())
        return positions


class CameraTrack:
    _id = 0

    def __init__(self, box, pos_3d, score: float, dt: float = 1 / 30.0):
        self.id = CameraTrack._id
        CameraTrack._id += 1
        self.kf = KalmanFilter3D(pos_3d, dt)
        self.box = box
        self.pos = pos_3d.copy()
        self.vel = np.zeros(3)
        self.score = score
        self.age = 0
        self.hits = 1
        self.last_time = time.time()
        self.trace_3d = [pos_3d.copy()]
        self.trace_px = [self._ctr()]
        np.random.seed(self.id * 7)
        self.color = tuple(np.random.randint(50, 255, 3).tolist())

    def predict(self, now: float):
        dt = now - self.last_time
        dt = float(np.clip(dt, 1 / 60.0, 1 / 10.0))
        self.kf.update_dt(dt)
        self.pos, self.vel = self.kf.predict()
        self.age += 1

    def update(self, box, pos_3d, score: float, now: float):
        dt = now - self.last_time
        dt = float(np.clip(dt, 1 / 60.0, 1 / 10.0))
        self.kf.update_dt(dt)
        self.kf.update(pos_3d)
        self.pos = self.kf.get_position()
        self.vel = self.kf.get_velocity()
        self.box = box
        self.score = score
        self.age = 0
        self.hits += 1
        self.last_time = now
        self.trace_3d.append(self.pos.copy())
        self.trace_px.append(self._ctr())
        if len(self.trace_3d) > 60:
            self.trace_3d.pop(0)
            self.trace_px.pop(0)

    def _ctr(self):
        return ((self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2)

    @property
    def speed_3d(self):
        return float(np.linalg.norm(self.vel))

    @property
    def vx(self):
        return float(self.vel[0])

    @property
    def vy(self):
        return float(self.vel[1])

    @property
    def vz(self):
        return float(self.vel[2])


def iou_2d(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter + 1e-6)


def dist_3d(pos_a, pos_b) -> float:
    return float(np.linalg.norm(pos_a - pos_b))


class Tracker3D:
    """Hungarian tracker using the same IoU + 3D-distance cost as camera.py."""

    def __init__(self, iou_thr: float = 0.3, dist_thr: float = 1.0, max_age: int = 10, min_hits: int = 2):
        self.tracks: List[CameraTrack] = []
        self.iou_thr = iou_thr
        self.dist_thr = dist_thr
        self.max_age = max_age
        self.min_hits = min_hits

    def _cost_matrix(self, dets, det_pos3d):
        n_t = len(self.tracks)
        n_d = len(dets)
        cost = np.ones((n_t, n_d))
        for i, t in enumerate(self.tracks):
            for j, d in enumerate(dets):
                iou_cost = 1.0 - iou_2d(t.box, d)
                if det_pos3d[j] is not None:
                    d3 = dist_3d(t.pos, det_pos3d[j])
                    d_cost = min(d3 / self.dist_thr, 1.0)
                    cost[i, j] = 0.5 * iou_cost + 0.5 * d_cost
                else:
                    cost[i, j] = iou_cost
        return cost

    def update(self, dets, scores, det_pos3d, now: float):
        for t in self.tracks:
            t.predict(now)
        matched_t, matched_d = set(), set()
        if self.tracks and dets:
            cost = self._cost_matrix(dets, det_pos3d)
            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                ok_iou = iou_2d(self.tracks[r].box, dets[c]) >= self.iou_thr
                ok_dist = det_pos3d[c] is not None and dist_3d(self.tracks[r].pos, det_pos3d[c]) < self.dist_thr
                if ok_iou or ok_dist:
                    pos = det_pos3d[c] if det_pos3d[c] is not None else self.tracks[r].pos
                    self.tracks[r].update(dets[c], pos, scores[c], now)
                    matched_t.add(r)
                    matched_d.add(c)
        for j in range(len(dets)):
            if j not in matched_d:
                pos = det_pos3d[j] if det_pos3d[j] is not None else np.array([0.0, 0.0, 1.0])
                self.tracks.append(CameraTrack(dets[j], pos, scores[j]))
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return [t for t in self.tracks if t.hits >= self.min_hits or t.age == 0]


def sample_depth(depth_img, box, scale: float) -> float:
    x1, y1, x2, y2 = box
    h, w = depth_img.shape
    cx1 = max(0, x1 + (x2 - x1) // 3)
    cx2 = min(w - 1, x1 + (x2 - x1) * 2 // 3)
    cy1 = max(0, y1 + (y2 - y1) // 3)
    cy2 = min(h - 1, y1 + (y2 - y1) * 2 // 3)
    roi = depth_img[cy1:cy2, cx1:cx2].astype(float)
    vals = roi[roi > 0]
    return float(np.median(vals)) * scale if len(vals) else 0.0


def draw_camera_frame(frame, tracks, converter: CoordinateConverter):
    """Draw exactly the same visual contents as the standalone camera.py window.

    Includes the MetaSurface crosshair, bounding boxes, historical track trails,
    future Kalman predictions, velocity arrows, ID/az/el/r labels, and target count.
    """
    h, w = frame.shape[:2]

    # Meta-surface center crosshair.
    cv2.drawMarker(frame, (w // 2, h // 2), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(
        frame,
        "MetaSurface",
        (w // 2 + 12, h // 2 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )

    for t in tracks:
        x1, y1, x2, y2 = t.box
        c = t.color

        # Detection box.
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

        # Historical pixel track.
        for k in range(1, len(t.trace_px)):
            alpha = k / len(t.trace_px)
            tc = tuple(int(v * alpha) for v in c)
            cv2.line(frame, t.trace_px[k - 1], t.trace_px[k], tc, 2)

        # Future Kalman trajectory.
        future = t.kf.predict_future(5)
        prev_px = t._ctr()
        for fp in future:
            if fp[2] > 0.1:
                fu = int(fp[0] * converter.fx / fp[2] + converter.cx)
                fv = int(-fp[1] * converter.fy / fp[2] + converter.cy + converter.offset_y * converter.fy / fp[2])
                fu = int(np.clip(fu, 0, w - 1))
                fv = int(np.clip(fv, 0, h - 1))
                cv2.line(frame, prev_px, (fu, fv), c, 1)
                prev_px = (fu, fv)

        # Velocity arrow in image plane.
        cx_ = (x1 + x2) // 2
        cy_ = (y1 + y2) // 2
        if t.pos[2] > 0.1:
            arrow_x = int(t.vx / t.pos[2] * converter.fx * 0.5)
            arrow_y = int(-t.vy / t.pos[2] * converter.fy * 0.5)
            cv2.arrowedLine(frame, (cx_, cy_), (cx_ + arrow_x, cy_ + arrow_y), c, 2, tipLength=0.4)

        # Information label.
        dist, az, el = converter.world_to_angles(t.pos)
        vz_str = f"vz:{t.vz:+.2f}m/s"
        vz_color = (0, 100, 255) if t.vz > 0.1 else (255, 100, 0) if t.vz < -0.1 else (200, 200, 200)
        lines = [
            f"ID:{t.id}  {t.score:.2f}",
            f"Dist:{dist:.2f}m",
            f"Az:{az:+.1f} El:{el:+.1f}",
            f"V:{t.speed_3d:.2f}m/s",
            vz_str,
        ]
        lh, pw, bw = 20, 4, 165
        bh = len(lines) * lh + pw * 2
        lx = max(0, min(x1, w - bw))
        ly = max(0, y1 - bh - 2)
        cv2.rectangle(frame, (lx, ly), (lx + bw, ly + bh), c, -1)
        for i, line in enumerate(lines):
            color_ = vz_color if i == 4 else (255, 255, 255)
            cv2.putText(
                frame,
                line,
                (lx + 3, ly + pw + (i + 1) * lh - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color_,
                1,
                cv2.LINE_AA,
            )

    cv2.putText(frame, f"Targets:{len(tracks)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


class SimpleRealSenseTracker:
    """RealSense + YOLO + Kalman + Hungarian tracker with camera.py-compatible display."""

    def __init__(self, confidence: float, log_func):
        if not CAMERA_DEPS_OK:
            raise RuntimeError(
                "Camera dependencies are missing. Install: ultralytics opencv-python "
                "pyrealsense2 scipy numpy"
            )
        self.confidence = confidence
        self.log = log_func
        self.model = YOLO("yolov8n.pt")
        self.tracker = Tracker3D(iou_thr=0.3, dist_thr=1.0, max_age=10, min_hits=2)

    def run_loop(self, stop_event: threading.Event, callback, update_interval_s: float):
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(cfg)
        align = rs.align(rs.stream.color)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        converter = CoordinateConverter(intr, offset_y=0.26)
        self.log("[CAMERA] 3D tracking started; press ESC in camera window to stop")

        last_send = 0.0
        try:
            while not stop_event.is_set():
                now = time.time()
                frames = align.process(pipeline.wait_for_frames())
                cf = frames.get_color_frame()
                df = frames.get_depth_frame()
                if not cf or not df:
                    continue

                color = np.asanyarray(cf.get_data())
                depth = np.asanyarray(df.get_data())

                # YOLO person detection, exactly as in camera.py.
                result = self.model(color, conf=self.confidence, classes=[0], verbose=False)[0]
                boxes, scores = [], []
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(box.conf[0]))

                det_pos3d = []
                for b in boxes:
                    dm = sample_depth(depth, b, depth_scale)
                    u = (b[0] + b[2]) // 2
                    v = (b[1] + b[3]) // 2
                    p = converter.pixel_to_world(u, v, dm)
                    det_pos3d.append(p)

                tracks = self.tracker.update(boxes, scores, det_pos3d, now)

                targets: List[TargetInfo] = []
                for t in tracks:
                    dist, az, el = converter.world_to_angles(t.pos)
                    targets.append(
                        TargetInfo(
                            track_id=t.id,
                            azimuth=az,
                            elevation=el,
                            distance=dist,
                            vx=t.vx,
                            vy=t.vy,
                            vz=t.vz,
                            speed_3d=t.speed_3d,
                            score=t.score,
                            box=t.box,
                            timestamp=now,
                        )
                    )

                cv2.imshow("3D Tracking", draw_camera_frame(color.copy(), tracks, converter))
                if cv2.waitKey(1) == 27:
                    stop_event.set()
                    break

                if time.time() - last_send >= max(0.02, update_interval_s):
                    last_send = time.time()
                    callback(targets)
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
            self.log("[CAMERA] stopped")


# -----------------------------
# Tkinter application
# -----------------------------
class MainApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"RIS PC Camera -> STM32 USART Controller - {APP_VERSION}")
        self.root.geometry("1050x780")
        self.root.minsize(950, 680)

        self.vars: dict[str, tk.Variable] = {}
        self.seq = 0
        self.serial_obj = None
        self.serial_reader_running = False
        self.serial_thread: Optional[threading.Thread] = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.stop_event: Optional[threading.Event] = None
        self.camera_thread: Optional[threading.Thread] = None

        self._build_ui()
        self.log_mcu(f"[APP] Loaded {APP_VERSION}")
        if serial is None:
            self.log_mcu("[WARN] pyserial not installed. Run: python -m pip install pyserial")
        if not CAMERA_DEPS_OK:
            self.log_mcu("[WARN] Camera dependencies missing; Test send still works.")

    def _add_entry(self, parent, row: int, label: str, key: str, default):
        tk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        var = tk.StringVar(value=str(default))
        self.vars[key] = var
        ent = ttk.Entry(parent, textvariable=var, width=18)
        ent.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        parent.columnconfigure(1, weight=1)
        return ent

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=8, pady=8)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(4, weight=1)

        ris_box = ttk.LabelFrame(main, text="RIS hardware / STM32 calculation")
        ris_box.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self._add_entry(ris_box, 0, "Frequency Hz", "freq_hz", 3.5e9)
        self._add_entry(ris_box, 1, "Nx", "nx", 20)
        self._add_entry(ris_box, 2, "Ny", "ny", 20)
        self._add_entry(ris_box, 3, "dx m", "dx_m", 0.0428)
        self._add_entry(ris_box, 4, "dy m", "dy_m", 0.0431)
        self._add_entry(ris_box, 5, "Bits", "bits", 2)
        self._add_entry(ris_box, 6, "LE hold time us", "le_hold_us", 10)

        tx_box = ttk.LabelFrame(main, text="Tx settings")
        tx_box.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        self._add_entry(tx_box, 0, "Tx1 az deg", "tx_az_deg", 0.0)
        self._add_entry(tx_box, 1, "Tx1 el deg", "tx_el_deg", 0.0)
        self._add_entry(tx_box, 2, "Tx1 range m", "tx_r_m", 2.5)
        self.vars["use_second_tx"] = tk.BooleanVar(value=False)
        tk.Checkbutton(tx_box, text="Use Tx2", variable=self.vars["use_second_tx"]).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )
        self._add_entry(tx_box, 4, "Tx2 az deg", "tx2_az_deg", 5.7)
        self._add_entry(tx_box, 5, "Tx2 el deg", "tx2_el_deg", -11.3)
        self._add_entry(tx_box, 6, "Tx2 range m", "tx2_r_m", 2.0)

        rx_box = ttk.LabelFrame(main, text="Tracking / USART settings")
        rx_box.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._add_entry(rx_box, 0, "Update interval s", "update_interval_s", 0.20)
        self._add_entry(rx_box, 1, "YOLO confidence", "yolo_confidence", 0.5)
        self._add_entry(rx_box, 2, "Output dir", "output_dir", ".")

        tk.Label(rx_box, text="USART port").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        self.vars["serial_port"] = tk.StringVar(value="")
        self.port_combo = ttk.Combobox(
            rx_box,
            textvariable=self.vars["serial_port"],
            values=list_serial_port_names(),
            width=15,
            state="normal",
        )
        self.port_combo.grid(row=3, column=1, sticky="ew", padx=4, pady=2)
        ports = list_serial_port_names()
        if ports:
            self.vars["serial_port"].set(ports[0])
        else:
            self.vars["serial_port"].set("COM3")
        ttk.Button(rx_box, text="Refresh ports", command=self.refresh_ports).grid(
            row=3, column=2, sticky="ew", padx=4, pady=2
        )

        tk.Label(rx_box, text="Baudrate").grid(row=4, column=0, sticky="w", padx=4, pady=2)
        self.vars["baudrate"] = tk.StringVar(value="115200")
        ttk.Combobox(
            rx_box,
            textvariable=self.vars["baudrate"],
            values=["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"],
            width=15,
            state="normal",
        ).grid(row=4, column=1, sticky="ew", padx=4, pady=2)

        tk.Label(rx_box, text="Mode").grid(row=5, column=0, sticky="w", padx=4, pady=2)
        self.vars["force_mode"] = tk.StringVar(value="auto")
        ttk.Combobox(
            rx_box,
            textvariable=self.vars["force_mode"],
            values=["auto", "near_field", "far_field"],
            width=15,
            state="readonly",
        ).grid(row=5, column=1, sticky="ew", padx=4, pady=2)

        self.vars["enable_usart"] = tk.BooleanVar(value=True)
        tk.Checkbutton(rx_box, text="Enable USART send", variable=self.vars["enable_usart"]).grid(
            row=6, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )
        self.vars["save_last_frame"] = tk.BooleanVar(value=True)
        tk.Checkbutton(rx_box, text="Save last frame to output dir", variable=self.vars["save_last_frame"]).grid(
            row=7, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )

        btn_box = ttk.LabelFrame(main, text="Control")
        btn_box.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        ttk.Button(btn_box, text="Start camera + USART", command=self.start_camera).grid(
            row=0, column=0, sticky="ew", padx=4, pady=4
        )
        ttk.Button(btn_box, text="Stop camera", command=self.stop_camera).grid(
            row=0, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Button(btn_box, text="Test send demo targets", command=self.test_send).grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=4
        )
        ttk.Button(btn_box, text="Open USART", command=self.open_usart).grid(
            row=2, column=0, sticky="ew", padx=4, pady=4
        )
        ttk.Button(btn_box, text="Close USART", command=self.close_usart).grid(
            row=2, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Button(btn_box, text="Clear log", command=self.clear_mcu_log).grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=4, pady=4
        )
        btn_box.columnconfigure(0, weight=1)
        btn_box.columnconfigure(1, weight=1)

        note = ttk.LabelFrame(main, text="Single / dual target rule")
        note.grid(row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        tk.Label(
            note,
            text="Single ID: Rx1=(az, el, r), Rx2=(-az, el, r).  Two IDs: Rx1=target0, Rx2=target1.",
            anchor="w",
        ).pack(fill="x", padx=6, pady=4)

        tk.Label(main, text="MCU response / USART log", anchor="w").grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=4, pady=(8, 2)
        )
        self.mcu_text = scrolledtext.ScrolledText(main, width=100, height=18, state="disabled")
        self.mcu_text.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)

    # -----------------------------
    # UI helpers and logs
    # -----------------------------
    def log_mcu(self, msg: str):
        def _append():
            self.mcu_text.configure(state="normal")
            self.mcu_text.insert("end", msg + "\n")
            self.mcu_text.see("end")
            self.mcu_text.configure(state="disabled")
            print(msg)
        if threading.current_thread() is threading.main_thread():
            _append()
        else:
            self.log_queue.put(msg)

    def poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.mcu_text.configure(state="normal")
                self.mcu_text.insert("end", msg + "\n")
                self.mcu_text.see("end")
                self.mcu_text.configure(state="disabled")
                print(msg)
        except queue.Empty:
            pass
        if self.serial_reader_running:
            self.root.after(50, self.poll_log_queue)

    def clear_mcu_log(self):
        self.mcu_text.configure(state="normal")
        self.mcu_text.delete("1.0", "end")
        self.mcu_text.configure(state="disabled")

    def refresh_ports(self):
        ports = list_serial_port_names()
        self.port_combo["values"] = ports
        if ports:
            self.vars["serial_port"].set(ports[0])
        self.log_mcu(f"[USART] ports: {ports if ports else 'none'}")

    def _get_float(self, key: str) -> float:
        return float(self.vars[key].get())

    def _get_int(self, key: str) -> int:
        return int(float(self.vars[key].get()))

    # -----------------------------
    # Serial operations
    # -----------------------------
    def open_usart(self):
        if serial is None:
            self.log_mcu("[USART ERROR] pyserial not installed. Run: python -m pip install pyserial")
            return
        port = self.vars["serial_port"].get().strip()
        baud = self._get_int("baudrate")
        if not port:
            self.log_mcu("[USART ERROR] no port selected")
            return
        try:
            if self.serial_obj is not None and self.serial_obj.is_open:
                self.serial_obj.close()
            self.serial_obj = serial.Serial(
                port=port,
                baudrate=baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.05,
            )
            self.log_mcu(f"[USART] opened {port} @ {baud}")
            self.start_serial_reader()
        except Exception as e:
            self.log_mcu(f"[USART OPEN ERROR] {e}")

    def close_usart(self):
        self.serial_reader_running = False
        try:
            if self.serial_obj is not None and self.serial_obj.is_open:
                self.serial_obj.close()
                self.log_mcu("[USART] closed")
        except Exception as e:
            self.log_mcu(f"[USART CLOSE ERROR] {e}")

    def start_serial_reader(self):
        if self.serial_obj is None or not self.serial_obj.is_open:
            self.log_mcu("[USART ERROR] serial not open")
            return
        if self.serial_reader_running:
            return
        self.serial_reader_running = True
        self.serial_thread = threading.Thread(target=self.serial_reader_loop, daemon=True)
        self.serial_thread.start()
        self.root.after(50, self.poll_log_queue)

    def serial_reader_loop(self):
        buffer = bytearray()
        while self.serial_reader_running:
            try:
                if self.serial_obj is None or not self.serial_obj.is_open:
                    time.sleep(0.05)
                    continue
                n = self.serial_obj.in_waiting
                if n:
                    data = self.serial_obj.read(n)
                    for b in data:
                        if b in (10, 13):  # LF or CR
                            if buffer:
                                text = buffer.decode(errors="ignore").strip()
                                if text:
                                    self.log_queue.put("[MCU] " + text)
                                buffer.clear()
                        else:
                            buffer.append(b)
                else:
                    time.sleep(0.01)
            except Exception as e:
                self.log_queue.put(f"[USART READ ERROR] {e}")
                time.sleep(0.1)

    def send_frame(self, frame: bytes, label: str = "frame"):
        self._save_last_frame(frame)
        if not bool(self.vars["enable_usart"].get()):
            self.log_mcu(f"[USART disabled] built {label}, bytes={len(frame)}")
            self.log_mcu(frame_to_hex(frame))
            return
        if self.serial_obj is None or not self.serial_obj.is_open:
            self.log_mcu("[USART ERROR] serial is not open. Click Open USART first.")
            return
        try:
            self.serial_obj.write(frame)
            self.serial_obj.flush()
            self.log_mcu(f"[PC -> MCU] {label}; sent seq={self.seq}, bytes={len(frame)}")
        except Exception as e:
            self.log_mcu(f"[USART SEND ERROR] {e}")

    # -----------------------------
    # Frame assembly
    # -----------------------------
    def current_txs(self) -> List[Terminal]:
        txs = [Terminal(self._get_float("tx_az_deg"), self._get_float("tx_el_deg"), self._get_float("tx_r_m"))]
        if bool(self.vars["use_second_tx"].get()):
            txs.append(Terminal(self._get_float("tx2_az_deg"), self._get_float("tx2_el_deg"), self._get_float("tx2_r_m")))
        return txs

    @staticmethod
    def targets_to_rxs(targets: List[TargetInfo]) -> Tuple[List[Terminal], str]:
        valid = [t for t in targets if t.distance > 0.0]
        valid.sort(key=lambda t: t.track_id)
        if len(valid) == 0:
            # Demo fallback target.
            return [Terminal(10.0, -20.0, 1.5), Terminal(-10.0, -20.0, 1.5)], "fallback_demo_no_target"
        if len(valid) == 1:
            t = valid[0]
            return [
                Terminal(t.azimuth, t.elevation, t.distance),
                Terminal(-t.azimuth, t.elevation, t.distance),
            ], f"single_id_{t.track_id}_direct_and_mirror"
        t0, t1 = valid[0], valid[1]
        return [
            Terminal(t0.azimuth, t0.elevation, t0.distance),
            Terminal(t1.azimuth, t1.elevation, t1.distance),
        ], f"two_ids_{t0.track_id}_{t1.track_id}"

    def build_frame_from_targets(self, targets: List[TargetInfo]) -> Tuple[bytes, str]:
        txs = self.current_txs()
        rxs, label = self.targets_to_rxs(targets)
        frame = build_frame(
            seq=self.seq,
            nx=self._get_int("nx"),
            ny=self._get_int("ny"),
            bits=self._get_int("bits"),
            mode=self.vars["force_mode"].get(),
            freq_hz=self._get_float("freq_hz"),
            dx_m=self._get_float("dx_m"),
            dy_m=self._get_float("dy_m"),
            le_hold_us=self._get_int("le_hold_us"),
            txs=txs,
            rxs=rxs,
        )
        self.seq = (self.seq + 1) & 0xFFFF
        return frame, label

    def _save_last_frame(self, frame: bytes):
        if not bool(self.vars["save_last_frame"].get()):
            return
        try:
            out_dir = self.vars["output_dir"].get().strip() or "."
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "last_usart_frame.bin"), "wb") as f:
                f.write(frame)
            with open(os.path.join(out_dir, "last_usart_frame.hex"), "w", encoding="utf-8") as f:
                f.write(frame_to_hex(frame) + "\n")
        except Exception as e:
            self.log_mcu(f"[SAVE ERROR] {e}")

    # -----------------------------
    # Button actions
    # -----------------------------
    def test_send(self):
        targets = [TargetInfo(track_id=0, azimuth=10.0, elevation=-20.0, distance=1.5, score=1.0, timestamp=time.time())]
        try:
            frame, label = self.build_frame_from_targets(targets)
            self.send_frame(frame, label)
            self.log_mcu("[TEST] target ID0 => Rx1=(10,-20,1.5), Rx2=(-10,-20,1.5)")
        except Exception as e:
            messagebox.showerror("Test send error", str(e))
            self.log_mcu(f"[TEST SEND ERROR] {e}")

    def start_camera(self):
        if self.camera_thread is not None and self.camera_thread.is_alive():
            self.log_mcu("[CAMERA] already running")
            return
        if not CAMERA_DEPS_OK:
            self.log_mcu("[CAMERA ERROR] missing dependencies. Test send can still be used.")
            return
        self.stop_event = threading.Event()
        confidence = self._get_float("yolo_confidence")
        interval = self._get_float("update_interval_s")
        try:
            tracker = SimpleRealSenseTracker(confidence=confidence, log_func=self.log_mcu)
        except Exception as e:
            self.log_mcu(f"[CAMERA ERROR] {e}")
            return

        def on_targets(targets: List[TargetInfo]):
            if targets:
                desc = "; ".join(
                    f"ID{t.track_id}:az={t.azimuth:+.1f},el={t.elevation:+.1f},r={t.distance:.2f}"
                    for t in targets[:2]
                )
                self.log_mcu(f"[TRACK] {desc}")
            frame, label = self.build_frame_from_targets(targets)
            self.send_frame(frame, label)

        self.camera_thread = threading.Thread(
            target=tracker.run_loop,
            args=(self.stop_event, on_targets, interval),
            daemon=True,
        )
        self.camera_thread.start()
        self.log_mcu("[CAMERA] starting thread")

    def stop_camera(self):
        if self.stop_event is not None:
            self.stop_event.set()
        self.log_mcu("[CAMERA] stop requested")

    def on_close(self):
        self.stop_camera()
        self.close_usart()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = MainApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
