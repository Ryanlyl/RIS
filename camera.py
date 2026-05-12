# track_realsense_3d.py
from ultralytics import YOLO
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import pyrealsense2 as rs
import math
import time
from typing import Callable, List


# ────────────────────────────────────────
# 数据结构
# ────────────────────────────────────────
class TargetInfo:
    def __init__(self, track_id, azimuth, elevation, distance,
                 vx, vy, vz, speed_3d,
                 score, box, timestamp):
        self.track_id  = track_id
        self.azimuth   = azimuth     # 方位角（度）
        self.elevation = elevation   # 俯仰角（度）
        self.distance  = distance    # 直线距离（米）
        self.vx        = vx          # X方向速度（米/秒）
        self.vy        = vy          # Y方向速度（米/秒）
        self.vz        = vz          # Z方向速度（米/秒，景深方向）
        self.speed_3d  = speed_3d    # 三维合速度（米/秒）
        self.score     = score
        self.box       = box
        self.timestamp = timestamp

    def __repr__(self):
        return (f"ID:{self.track_id:2d} | "
                f"Az:{self.azimuth:+6.1f}° El:{self.elevation:+5.1f}° | "
                f"Dist:{self.distance:.2f}m | "
                f"V3D:{self.speed_3d:.2f}m/s "
                f"(vx:{self.vx:+.2f} vy:{self.vy:+.2f} vz:{self.vz:+.2f})")


# ────────────────────────────────────────
# 坐标转换（像素+深度 → 世界坐标）
# ────────────────────────────────────────
class CoordinateConverter:
    def __init__(self, intrinsics, offset_y=0.26):
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.cx = intrinsics.ppx
        self.cy = intrinsics.ppy
        self.offset_y = offset_y   # 超表面中心在摄像头下方0.26m

    def pixel_to_world(self, u, v, depth_m):
        """
        像素坐标 + 深度 → 超表面坐标系3D坐标（米）

        相机坐标系:  X右 Y下 Z前
        超表面坐标系: 原点下移0.26m，Y轴向上为正
        """
        if depth_m <= 0:
            return None

        # 反投影到相机坐标系
        x_cam = (u - self.cx) * depth_m / self.fx
        y_cam = (v - self.cy) * depth_m / self.fy
        z_cam = depth_m

        # 转超表面坐标系（原点下移0.26m，Y轴翻转）
        x_ms =  x_cam
        y_ms = -(y_cam - self.offset_y)   # Y轴翻转：向上为正
        z_ms =  z_cam

        return np.array([x_ms, y_ms, z_ms])

    def world_to_angles(self, pos_3d):
        """
        3D世界坐标 → 方位角/俯仰角/距离
        """
        x, y, z = pos_3d
        dist  = np.linalg.norm(pos_3d)
        if dist < 1e-4:
            return 0.0, 0.0, 0.0
        horiz = math.sqrt(x**2 + z**2)
        az    = math.degrees(math.atan2(x, z))    # 水平角，右为正
        el    = math.degrees(math.atan2(y, horiz)) # 仰角，上为正
        return dist, az, el


# ────────────────────────────────────────
# ★ 三维卡尔曼滤波器
#
# 状态向量 (6×1):
#   [x, y, z, vx, vy, vz]
#   单位：米，米/秒
#
# 观测向量 (3×1):
#   [x, y, z]  ← 从深度图直接获得
#
# 运动模型：匀速运动
#   x(t+1) = x(t) + vx * dt
# ────────────────────────────────────────
class KalmanFilter3D:
    def __init__(self, pos_3d: np.ndarray, dt=1/30.0):
        """
        pos_3d: 初始3D位置 [x, y, z]（米）
        dt:     帧间时间（秒），30fps → 1/30
        """
        self.dt = dt

        # ── 状态向量 x: [x, y, z, vx, vy, vz] ──
        self.x = np.zeros((6, 1))
        self.x[:3] = pos_3d.reshape(3, 1)

        # ── 状态转移矩阵 F（匀速运动模型）────
        #   [ 1  0  0  dt  0  0  ]
        #   [ 0  1  0  0  dt  0  ]
        #   [ 0  0  1  0   0  dt ]
        #   [ 0  0  0  1   0  0  ]
        #   [ 0  0  0  0   1  0  ]
        #   [ 0  0  0  0   0  1  ]
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # ── 观测矩阵 H: 只观测位置 ────────────
        # z_obs = H * x = [x, y, z]
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # ── 过程噪声 Q（运动不确定性）─────────
        # 位置噪声小，速度噪声大
        self.Q = np.diag([
            0.01,  # x位置噪声 (m²)
            0.01,  # y位置噪声
            0.05,  # z位置噪声（深度方向噪声更大）
            0.1,   # vx速度噪声 (m²/s²)
            0.1,   # vy速度噪声
            0.5,   # vz速度噪声（深度方向速度更难估）
        ])

        # ── 观测噪声 R（深度测量不确定性）──────
        # RealSense 在近距离精度约±2mm，远距离约±2cm
        self.R = np.diag([
            0.02,  # x测量噪声 (m²)
            0.02,  # y测量噪声
            0.05,  # z测量噪声（深度方向精度更低）
        ])

        # ── 初始误差协方差 P ──────────────────
        self.P = np.diag([
            1.0,   # x初始不确定性
            1.0,   # y
            1.0,   # z
            10.0,  # vx初始速度未知，不确定性大
            10.0,  # vy
            10.0,  # vz
        ])

    def predict(self):
        """
        预测下一帧的位置和速度
        即使当前帧没有检测到目标，也能给出估计
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_position(), self.get_velocity()

    def update(self, pos_3d: np.ndarray):
        """
        用新的3D观测值修正状态
        pos_3d: [x, y, z]（米）
        """
        z = pos_3d.reshape(3, 1)

        # 新息（观测值 - 预测值）
        y = z - self.H @ self.x

        # 新息协方差
        S = self.H @ self.P @ self.H.T + self.R

        # 卡尔曼增益
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 状态更新
        self.x = self.x + K @ y

        # 协方差更新（Joseph形式，数值稳定）
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def get_position(self):
        """返回滤波后的3D位置（米）"""
        return self.x[:3, 0].copy()

    def get_velocity(self):
        """返回滤波后的3D速度（米/秒）"""
        return self.x[3:, 0].copy()

    def update_dt(self, dt):
        """动态更新帧间时间（适应帧率波动）"""
        self.dt = dt
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

    def predict_future(self, n_frames: int):
        """
        预测未来n帧的位置（用于画预测轨迹）
        不改变内部状态
        """
        x_tmp = self.x.copy()
        positions = []
        for _ in range(n_frames):
            x_tmp = self.F @ x_tmp
            positions.append(x_tmp[:3, 0].copy())
        return positions


# ────────────────────────────────────────
# 追踪轨迹
# ────────────────────────────────────────
class Track:
    _id = 0

    def __init__(self, box, pos_3d, score, dt=1/30.0):
        self.id    = Track._id; Track._id += 1
        self.kf    = KalmanFilter3D(pos_3d, dt)   # ★ 3D卡尔曼
        self.box   = box
        self.pos   = pos_3d.copy()    # 滤波后的3D位置
        self.vel   = np.zeros(3)      # 滤波后的3D速度
        self.score = score
        self.age   = 0
        self.hits  = 1
        self.last_time = time.time()
        self.trace_3d  = [pos_3d.copy()]   # 3D轨迹历史
        self.trace_px  = [self._ctr()]     # 像素轨迹（画图用）
        np.random.seed(self.id * 7)
        self.color = tuple(np.random.randint(50,255,3).tolist())

    def predict(self, now):
        """根据实际时间差预测"""
        dt = now - self.last_time
        dt = np.clip(dt, 1/60.0, 1/10.0)  # 限制在合理范围
        self.kf.update_dt(dt)
        self.pos, self.vel = self.kf.predict()
        self.age += 1

    def update(self, box, pos_3d, score, now):
        dt = now - self.last_time
        dt = np.clip(dt, 1/60.0, 1/10.0)
        self.kf.update_dt(dt)
        self.kf.update(pos_3d)

        self.pos   = self.kf.get_position()
        self.vel   = self.kf.get_velocity()
        self.box   = box
        self.score = score
        self.age   = 0
        self.hits += 1
        self.last_time = now

        self.trace_3d.append(self.pos.copy())
        self.trace_px.append(self._ctr())
        if len(self.trace_3d) > 60:
            self.trace_3d.pop(0)
            self.trace_px.pop(0)

    def _ctr(self):
        return ((self.box[0]+self.box[2])//2,
                (self.box[1]+self.box[3])//2)

    @property
    def speed_3d(self):
        return float(np.linalg.norm(self.vel))

    @property
    def vx(self): return float(self.vel[0])

    @property
    def vy(self): return float(self.vel[1])

    @property
    def vz(self): return float(self.vel[2])   # ★ 景深方向速度


# ────────────────────────────────────────
# IoU（用于2D匹配）
# ────────────────────────────────────────
def iou_2d(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    ua=(a[2]-a[0])*(a[3]-a[1]); ub=(b[2]-b[0])*(b[3]-b[1])
    return inter/(ua+ub-inter+1e-6)


def dist_3d(pos_a, pos_b):
    """3D欧氏距离（米），用于辅助匹配"""
    return float(np.linalg.norm(pos_a - pos_b))


# ────────────────────────────────────────
# ★ 追踪器（3D版）
# ────────────────────────────────────────
class Tracker3D:
    """
    匹配策略：IoU(像素) + 3D距离 联合打分
    近距离目标用3D距离更准，远距离或无深度时用IoU
    """
    def __init__(self, iou_thr=0.3, dist_thr=1.0,
                 max_age=10, min_hits=2):
        self.tracks   = []
        self.iou_thr  = iou_thr
        self.dist_thr = dist_thr   # 3D距离阈值（米）
        self.max_age  = max_age
        self.min_hits = min_hits

    def _cost_matrix(self, dets, det_pos3d):
        """
        代价矩阵 = IoU代价 * 0.5 + 3D距离代价 * 0.5
        两者加权，比单独用IoU更鲁棒
        """
        n_t = len(self.tracks)
        n_d = len(dets)
        cost = np.ones((n_t, n_d))

        for i, t in enumerate(self.tracks):
            for j, d in enumerate(dets):
                # IoU代价（越大越好→取负）
                iou_cost  = 1.0 - iou_2d(t.box, d)

                # 3D距离代价（归一化到[0,1]）
                if det_pos3d[j] is not None:
                    d3  = dist_3d(t.pos, det_pos3d[j])
                    d_cost = min(d3 / self.dist_thr, 1.0)
                    cost[i, j] = 0.5 * iou_cost + 0.5 * d_cost
                else:
                    cost[i, j] = iou_cost

        return cost

    def update(self, dets, scores, det_pos3d, now):
        """
        dets:      [[x1,y1,x2,y2], ...]  检测框
        scores:    [0.9, ...]
        det_pos3d: [np.array([x,y,z]) or None, ...]  3D位置
        now:       当前时间戳
        """
        # Step1: 所有轨迹先预测
        for t in self.tracks:
            t.predict(now)

        # Step2: 匈牙利匹配
        matched_t, matched_d = set(), set()

        if self.tracks and dets:
            cost = self._cost_matrix(dets, det_pos3d)
            row_idx, col_idx = linear_sum_assignment(cost)

            for r, c in zip(row_idx, col_idx):
                # 验证匹配质量（IoU或3D距离至少一个满足）
                ok_iou  = iou_2d(self.tracks[r].box, dets[c]) >= self.iou_thr
                ok_dist = (det_pos3d[c] is not None and
                           dist_3d(self.tracks[r].pos, det_pos3d[c]) < self.dist_thr)

                if ok_iou or ok_dist:
                    pos = det_pos3d[c] if det_pos3d[c] is not None \
                          else self.tracks[r].pos
                    self.tracks[r].update(dets[c], pos, scores[c], now)
                    matched_t.add(r); matched_d.add(c)

        # Step3: 未匹配检测 → 新轨迹
        for j in range(len(dets)):
            if j not in matched_d:
                pos = det_pos3d[j] if det_pos3d[j] is not None \
                      else np.array([0., 0., 1.])
                self.tracks.append(Track(dets[j], pos, scores[j]))

        # Step4: 清理过期轨迹
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        # Step5: 只输出稳定轨迹
        return [t for t in self.tracks
                if t.hits >= self.min_hits or t.age == 0]


# ────────────────────────────────────────
# 检测器
# ────────────────────────────────────────
MODEL = YOLO("yolov8n.pt")

def detect(frame, confidence=0.5):
    res = MODEL(frame, conf=confidence, classes=[0], verbose=False)[0]
    boxes, scores = [], []
    for box in res.boxes:
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
        boxes.append([x1,y1,x2,y2])
        scores.append(float(box.conf[0]))
    return boxes, scores


# ────────────────────────────────────────
# 深度采样
# ────────────────────────────────────────
def sample_depth(depth_img, box, scale):
    x1,y1,x2,y2 = box
    H,W = depth_img.shape
    cx1=max(0,x1+(x2-x1)//3);  cx2=min(W-1,x1+(x2-x1)*2//3)
    cy1=max(0,y1+(y2-y1)//3);  cy2=min(H-1,y1+(y2-y1)*2//3)
    roi = depth_img[cy1:cy2, cx1:cx2].astype(float)
    v   = roi[roi>0]
    return float(np.median(v))*scale if len(v) else 0.0


# ────────────────────────────────────────
# 可视化
# ────────────────────────────────────────
def draw(frame, tracks, converter: CoordinateConverter):
    h, w = frame.shape[:2]

    # 超表面中心标记
    cv2.drawMarker(frame,(w//2,h//2),(0,255,255),cv2.MARKER_CROSS,20,2)
    cv2.putText(frame,"MetaSurface",(w//2+12,h//2-8),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1)

    for t in tracks:
        x1,y1,x2,y2 = t.box; c = t.color

        # 检测框
        cv2.rectangle(frame,(x1,y1),(x2,y2),c,2)

        # 历史轨迹（像素）
        for k in range(1, len(t.trace_px)):
            alpha = k / len(t.trace_px)
            tc    = tuple(int(v*alpha) for v in c)
            cv2.line(frame, t.trace_px[k-1], t.trace_px[k], tc, 2)

        # 预测轨迹（未来5帧，虚线效果）
        future = t.kf.predict_future(5)
        # 把3D预测位置投影回像素
        prev_px = t._ctr()
        for fp in future:
            if fp[2] > 0.1:  # z>0才有意义
                fu = int(fp[0] * converter.fx / fp[2] + converter.cx)
                fv = int(-fp[1] * converter.fy / fp[2] + converter.cy
                          + converter.offset_y * converter.fy / fp[2])
                fu = np.clip(fu, 0, w-1)
                fv = np.clip(fv, 0, h-1)
                cv2.line(frame, prev_px, (fu,fv), c, 1)
                prev_px = (fu, fv)

        # 速度箭头（像素空间）
        cx_ = (x1+x2)//2; cy_ = (y1+y2)//2
        # 把3D速度投影到图像平面（简化）
        if t.pos[2] > 0.1:
            arrow_x = int(t.vx / t.pos[2] * converter.fx * 0.5)
            arrow_y = int(-t.vy / t.pos[2] * converter.fy * 0.5)
            cv2.arrowedLine(frame,
                            (cx_, cy_),
                            (cx_+arrow_x, cy_+arrow_y),
                            c, 2, tipLength=0.4)

        # 信息标签
        dist, az, el = converter.world_to_angles(t.pos)
        vz_str = f"vz:{t.vz:+.2f}m/s"
        # vz>0 目标靠近，vz<0 目标远离
        vz_color = (0,100,255) if t.vz > 0.1 else \
                   (255,100,0) if t.vz < -0.1 else (200,200,200)

        lines = [
            f"ID:{t.id}  {t.score:.2f}",
            f"Dist:{dist:.2f}m",
            f"Az:{az:+.1f} El:{el:+.1f}",
            f"V:{t.speed_3d:.2f}m/s",
            vz_str,
        ]
        lh,pw,bw = 20,4,165
        bh = len(lines)*lh+pw*2
        lx = max(0,min(x1,w-bw)); ly = max(0,y1-bh-2)
        cv2.rectangle(frame,(lx,ly),(lx+bw,ly+bh),c,-1)
        for i,ln in enumerate(lines):
            color_ = vz_color if i == 4 else (255,255,255)
            cv2.putText(frame,ln,(lx+3,ly+pw+(i+1)*lh-3),
                        cv2.FONT_HERSHEY_SIMPLEX,0.48,
                        color_,1,cv2.LINE_AA)

    cv2.putText(frame,f"Targets:{len(tracks)}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    return frame


# ────────────────────────────────────────
# 示例回调
# ────────────────────────────────────────
def my_custom_function(targets: List[TargetInfo]):
    for t in targets:
        print(t)


# ────────────────────────────────────────
# 主程序
# ────────────────────────────────────────
def run(callbacks: List[Callable[[List[TargetInfo]], None]] = None):
    tracker   = Tracker3D(iou_thr=0.3, dist_thr=1.0, max_age=10, min_hits=2)
    callbacks = callbacks or []

    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16,  30)
    cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
    profile  = pipeline.start(cfg)

    scale     = profile.get_device().first_depth_sensor().get_depth_scale()
    intr      = (profile.get_stream(rs.stream.color)
                        .as_video_stream_profile().get_intrinsics())
    converter = CoordinateConverter(intr, offset_y=0.26)
    align     = rs.align(rs.stream.color)

    print("3D追踪已启动，按 ESC 退出")

    try:
        while True:
            now    = time.time()
            frames = align.process(pipeline.wait_for_frames())
            cf     = frames.get_color_frame()
            df     = frames.get_depth_frame()
            if not cf or not df: continue

            color = np.asanyarray(cf.get_data())
            depth = np.asanyarray(df.get_data())

            # 检测
            boxes, scores = detect(color)

            # 每个框的3D位置
            det_pos3d = []
            for b in boxes:
                dm = sample_depth(depth, b, scale)
                u  = (b[0]+b[2])//2
                v  = (b[1]+b[3])//2
                p  = converter.pixel_to_world(u, v, dm)
                det_pos3d.append(p)   # None if dm<=0

            # 3D追踪
            tracks = tracker.update(boxes, scores, det_pos3d, now)

            # 打包参数 → 回调
            target_list = []
            for t in tracks:
                dist, az, el = converter.world_to_angles(t.pos)
                target_list.append(TargetInfo(
                    track_id  = t.id,
                    azimuth   = az,
                    elevation = el,
                    distance  = dist,
                    vx        = t.vx,
                    vy        = t.vy,
                    vz        = t.vz,
                    speed_3d  = t.speed_3d,
                    score     = t.score,
                    box       = t.box,
                    timestamp = now,
                ))

            for cb in callbacks:
                cb(target_list)

            cv2.imshow("3D Tracking", draw(color.copy(), tracks, converter))
            if cv2.waitKey(1) == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(callbacks=[my_custom_function])