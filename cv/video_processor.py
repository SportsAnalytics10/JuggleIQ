import cv2
import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from collections import Counter, deque


# MediaPipe 33-point pose indices
POSE_LEFT_HIP,   POSE_LEFT_KNEE,   POSE_LEFT_ANKLE  = 23, 25, 27
POSE_RIGHT_HIP,  POSE_RIGHT_KNEE,  POSE_RIGHT_ANKLE = 24, 26, 28
POSE_LEFT_FOOT,  POSE_RIGHT_FOOT                    = 31, 32

LEG_CONNECTIONS = [
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]
LEG_LANDMARK_INDICES = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


class VideoProcessor:
    def __init__(self, conf=0.45, model_name="yolov8m.pt"):
        print("VideoProcessor initialized.")
        self.model = YOLO(model_name)
        self.conf  = conf
        self.pose_landmarker = self._create_pose_landmarker()
        self.reset_tracking()

    # ------------------------------------------------------------------
    # Pose landmarker
    # ------------------------------------------------------------------

    def _create_pose_landmarker(self):
        model_path = Path(__file__).resolve().parent.parent / "pose_landmarker.task"
        if not model_path.exists():
            MODEL_URL = (
                "https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            )
            print("Downloading pose_landmarker.task ...")
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print("Download complete.")

        base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
        pose_options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        return vision.PoseLandmarker.create_from_options(pose_options)

    def detect_pose(self, frame, timestamp_ms):
        h, w = frame.shape[:2]
        rgb  = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self.pose_landmarker.detect_for_video(mp_image, int(timestamp_ms))
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None
        pl = result.pose_landmarks[0]
        return [(int(lm.x * w), int(lm.y * h), float(lm.z)) for lm in pl]

    # ------------------------------------------------------------------
    # Reset tracking
    # ------------------------------------------------------------------

    def reset_tracking(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.processNoiseCov[2, 2] = 5.0
        self.kf.processNoiseCov[3, 3] = 5.0
        self.kf.measurementNoiseCov   = np.eye(2, dtype=np.float32) * 0.5

        self.kalman_initialized          = False
        self.kalman_predicted_this_frame = False
        self.lost_frames                 = 0
        self.MAX_LOST                    = 12
        self.last_radius                 = 20

        self.smooth_cx = None
        self.smooth_cy = None
        self.ALPHA     = 0.55

        self.fps         = 30.0
        self.MAX_JUMP_PX = 150

        # Trail history (capped)
        self.ball_history          = []
        self.trail_len             = 25

        # Detected-only history — kept full for peak/drift analysis
        self.ball_history_detected = []

        # ── Touch detection ──────────────────────────────────────────
        self.touches            = []
        self.touch_flash_frames = 6
        self.last_touch_t       = -1e9

        # FIX 1: raised thresholds — less noise, more reliable
        self.min_touch_interval = 0.20   # was 0.15
        self.vy_min_flip        = 180.0  # was 100 — must be real velocity

        # FIX 2: velocity smoothing window — avoids single-frame noise
        self.vy_window          = deque(maxlen=3)  # smooth over 3 detected frames
        self.det_window         = deque(maxlen=3)  # matching point window

        # FIX 3: confirmation buffer — ball must rise for N frames
        self.MIN_RISE_FRAMES    = 2      # must see upward motion for 2+ frames
        self.rise_count         = 0      # consecutive rising frames counter
        self.in_rise            = False  # currently tracking a rise?
        self.touch_candidate    = None   # pending touch waiting for confirmation

        # F1
        self.last_landmarks     = None

        # F2
        self.FOOT_PROXIMITY_PX  = 400

        # F3
        self.peaks              = []
        self.drifts             = []
        self.intervals          = []

        # F4
        self.knee_angles_left   = []
        self.knee_angles_right  = []

    # ------------------------------------------------------------------
    # Kalman helpers
    # ------------------------------------------------------------------

    def _init_kalman(self, cx, cy, h_frame):
        if cy < h_frame * 0.15:
            return
        if cy > h_frame * 0.90:
            return
        self.kf.statePre  = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        self.kalman_initialized = True
        self.lost_frames = 0

    def _kalman_predict(self):
        pred = self.kf.predict()
        pred = np.array(pred).flatten()
        return int(pred[0]), int(pred[1])

    def _kalman_state(self):
        flat = np.array(self.kf.statePost).flatten()
        return int(flat[0]), int(flat[1])

    def _kalman_correct(self, cx, cy):
        meas = np.array([[np.float32(cx)], [np.float32(cy)]], dtype=np.float32)
        self.kf.correct(meas)

    def _smooth(self, cx, cy):
        if self.smooth_cx is None:
            self.smooth_cx = cx
            self.smooth_cy = cy
        else:
            self.smooth_cx = int(self.ALPHA * cx + (1 - self.ALPHA) * self.smooth_cx)
            self.smooth_cy = int(self.ALPHA * cy + (1 - self.ALPHA) * self.smooth_cy)
        return self.smooth_cx, self.smooth_cy

    # ------------------------------------------------------------------
    # Jump rejection
    # ------------------------------------------------------------------

    def _is_valid_jump(self, cx, cy):
        for p in reversed(self.ball_history):
            if p["x"] is not None and p["y"] is not None:
                dist = np.sqrt((cx - p["x"]) ** 2 + (cy - p["y"]) ** 2)
                return dist <= self.MAX_JUMP_PX
        return True

    # ------------------------------------------------------------------
    # Ball history append
    # ------------------------------------------------------------------

    def _append_ball_point(self, t, frame_idx, cx, cy, status):
        if cx is not None and cy is not None:
            if self._is_valid_jump(cx, cy):
                self.ball_history.append({
                    "t": t, "x": cx, "y": cy, "status": status
                })
                if status == "detected":
                    self.ball_history_detected.append({
                        "t":         t,
                        "frame_idx": frame_idx,
                        "x":         cx,
                        "y":         cy
                    })
            else:
                self.ball_history.append({
                    "t": t, "x": None, "y": None, "status": None
                })
        else:
            self.ball_history.append({
                "t": t, "x": None, "y": None, "status": None
            })

        if len(self.ball_history) > self.trail_len:
            self.ball_history.pop(0)

    # ------------------------------------------------------------------
    # F1: identify foot
    # ------------------------------------------------------------------

    def _identify_foot(self, ball_cx, ball_cy):
        if self.last_landmarks is None:
            return "U"
        lms = self.last_landmarks

        def get_lm(idx):
            if idx < len(lms):
                return np.array([lms[idx][0], lms[idx][1]], dtype=np.float32)
            return None

        ball        = np.array([ball_cx, ball_cy], dtype=np.float32)
        left_ankle  = get_lm(POSE_LEFT_ANKLE)
        left_foot   = get_lm(POSE_LEFT_FOOT)
        right_ankle = get_lm(POSE_RIGHT_ANKLE)
        right_foot  = get_lm(POSE_RIGHT_FOOT)

        def side_dist(a, b):
            pts = [p for p in [a, b] if p is not None]
            if not pts:
                return float("inf")
            return float(np.linalg.norm(ball - np.mean(pts, axis=0)))

        ld = side_dist(left_ankle, left_foot)
        rd = side_dist(right_ankle, right_foot)
        if ld == float("inf") and rd == float("inf"):
            return "U"
        return "L" if ld < rd else "R"

    # ------------------------------------------------------------------
    # F2: proximity gate
    # ------------------------------------------------------------------

    def _is_ball_near_foot(self, ball_cx, ball_cy):
        if self.last_landmarks is None:
            return True
        lms  = self.last_landmarks
        ball = np.array([ball_cx, ball_cy], dtype=np.float32)
        min_dist = float("inf")
        for idx in [POSE_LEFT_ANKLE, POSE_LEFT_FOOT,
                    POSE_RIGHT_ANKLE, POSE_RIGHT_FOOT]:
            if idx < len(lms):
                lm   = np.array([lms[idx][0], lms[idx][1]], dtype=np.float32)
                dist = float(np.linalg.norm(ball - lm))
                min_dist = min(min_dist, dist)
                if dist <= self.FOOT_PROXIMITY_PX:
                    return True
        print(f"    [F2] min dist: {min_dist:.1f}px → rejected")
        return False

    # ------------------------------------------------------------------
    # F4: knee angle
    # ------------------------------------------------------------------

    def _compute_knee_angle(self, landmarks_px, side="left"):
        if landmarks_px is None:
            return None
        if side == "left":
            hip_idx, knee_idx, ankle_idx = (
                POSE_LEFT_HIP, POSE_LEFT_KNEE, POSE_LEFT_ANKLE)
        else:
            hip_idx, knee_idx, ankle_idx = (
                POSE_RIGHT_HIP, POSE_RIGHT_KNEE, POSE_RIGHT_ANKLE)

        if any(i >= len(landmarks_px) for i in [hip_idx, knee_idx, ankle_idx]):
            return None

        hip   = np.array(landmarks_px[hip_idx][:2],   dtype=np.float32)
        knee  = np.array(landmarks_px[knee_idx][:2],  dtype=np.float32)
        ankle = np.array(landmarks_px[ankle_idx][:2], dtype=np.float32)
        v1    = hip   - knee
        v2    = ankle - knee
        cos_a = np.clip(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6),
            -1.0, 1.0
        )
        return round(float(np.degrees(np.arccos(cos_a))), 1)

    # ------------------------------------------------------------------
    # FIX: improved touch detection
    # ------------------------------------------------------------------

    def _update_touch_detection(self, det_point):
        """
        Improved touch detection:
        1. Smooth velocity over a 3-point window (not single frame)
        2. Require MIN_RISE_FRAMES consecutive rising frames for confirmation
        3. Higher vy_min_flip threshold (180 px/s)
        4. Proximity gate still applied
        """
        # Add to sliding windows
        self.vy_window.append(det_point)
        self.det_window.append(det_point)

        # Need at least 2 points to compute velocity
        if len(self.vy_window) < 2:
            return

        # FIX: compute smoothed velocity using oldest and newest in window
        p_old = self.vy_window[0]
        p_new = self.vy_window[-1]
        dt    = p_new["t"] - p_old["t"]

        if dt <= 1e-6:
            return

        # Smoothed vy over window (pixels/sec)
        vy = (p_new["y"] - p_old["y"]) / dt

        # ── Rising phase tracking ─────────────────────────────────────
        # In OpenCV y increases downward
        # vy < 0 means ball is moving UP (rising after a touch)
        # vy > 0 means ball is moving DOWN (falling)

        if vy < -self.vy_min_flip:
            # Ball is rising fast enough to be a real touch response
            if not self.in_rise:
                # Just started rising — save candidate touch point
                self.in_rise         = True
                self.rise_count      = 1
                self.touch_candidate = p_old  # touch happened just before rise
            else:
                self.rise_count += 1

            # FIX: only confirm touch after MIN_RISE_FRAMES consecutive rises
            if (self.rise_count == self.MIN_RISE_FRAMES and
                    self.touch_candidate is not None):

                t1        = self.touch_candidate["t"]
                far_enough = (t1 - self.last_touch_t) >= self.min_touch_interval
                near_foot  = self._is_ball_near_foot(
                    self.touch_candidate["x"],
                    self.touch_candidate["y"]
                )

                if far_enough and near_foot:
                    foot = self._identify_foot(
                        self.touch_candidate["x"],
                        self.touch_candidate["y"]
                    )
                    self.touches.append({
                        "frame_idx": self.touch_candidate["frame_idx"],
                        "t":         self.touch_candidate["t"],
                        "x":         self.touch_candidate["x"],
                        "y":         self.touch_candidate["y"],
                        "foot":      foot
                    })
                    self.last_touch_t = t1
                    print(f"  ✅ Touch #{len(self.touches)} [{foot}] "
                          f"confirmed at t={t1:.2f}s "
                          f"(vy={vy:.1f} over {self.rise_count} frames)")
                else:
                    if not far_enough:
                        print(f"  ✗ Rejected — too soon after last touch")
                    if not near_foot:
                        print(f"  ✗ Rejected — ball too far from foot")

                # Reset rise tracking after decision
                self.touch_candidate = None

        elif vy > self.vy_min_flip:
            # Ball is falling — reset rise tracking
            if self.in_rise:
                self.in_rise    = False
                self.rise_count = 0

    # ------------------------------------------------------------------
    # YOLO ball detection
    # ------------------------------------------------------------------

    def _yolo_candidates(self, frame):
        h_frame, w_frame = frame.shape[:2]
        results    = self.model.predict(
            frame, conf=self.conf, classes=[32], verbose=False
        )
        boxes      = results[0].boxes
        candidates = []

        if boxes is None or len(boxes) == 0:
            return candidates

        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            cx    = (x1 + x2) / 2.0
            cy    = (y1 + y2) / 2.0
            w     = x2 - x1
            h     = y2 - y1
            score = float(b.conf[0].cpu().numpy())

            if cy < h_frame * 0.15:
                continue
            if cy > h_frame * 0.90:
                continue

            min_size = 0.02 * min(h_frame, w_frame)
            if w < min_size or h < min_size:
                continue
            if w > 0.45 * w_frame or h > 0.45 * h_frame:
                continue

            aspect = w / h if h > 0 else 0
            if not (0.55 < aspect < 1.8):
                continue

            r = int(0.25 * (w + h))
            candidates.append((score, cx, cy, r))

        return candidates

    def detect_ball(self, frame):
        h_frame    = frame.shape[0]
        candidates = self._yolo_candidates(frame)
        self.kalman_predicted_this_frame = False

        if candidates:
            score, cx, cy, r = max(candidates, key=lambda t: t[0])
            cx, cy = int(cx), int(cy)

            if not self.kalman_initialized:
                self._init_kalman(cx, cy, h_frame)
            else:
                self._kalman_predict()
                self.kalman_predicted_this_frame = True
                self._kalman_correct(cx, cy)

            self.lost_frames = 0
            self.last_radius = int(r)
            cx, cy = self._smooth(cx, cy)
            return cx, cy, int(r), "detected"

        if not self.kalman_initialized:
            return None, None, None, None

        self.lost_frames += 1
        if self.lost_frames > self.MAX_LOST:
            self.reset_tracking()
            return None, None, None, None

        if not self.kalman_predicted_this_frame:
            cx, cy = self._kalman_predict()
            self.kalman_predicted_this_frame = True
        else:
            cx, cy = self._kalman_state()

        cx, cy = self._smooth(cx, cy)
        return cx, cy, int(self.last_radius), "predicted"

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_trail(self, frame):
        if len(self.ball_history) < 2:
            return
        recent = [p for p in self.ball_history if p["x"] is not None]
        if len(recent) < 2:
            return
        pts = np.array(
            [(p["x"], p["y"]) for p in recent], dtype=np.int32
        ).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False,
                      color=(255, 255, 255), thickness=2)
        total = len(recent)
        for i, p in enumerate(recent):
            alpha  = (i + 1) / total
            radius = max(2, int(6 * alpha))
            color  = (0, 165, 255) if p["status"] == "detected" else (0, 0, 255)
            cv2.circle(frame, (p["x"], p["y"]), radius, color, -1)

    def _draw_pose(self, frame, landmarks_px):
        if landmarks_px is None:
            return
        for (i, j) in LEG_CONNECTIONS:
            if i < len(landmarks_px) and j < len(landmarks_px):
                cv2.line(frame, landmarks_px[i][:2],
                         landmarks_px[j][:2], (0, 255, 0), 2)
        for i in LEG_LANDMARK_INDICES:
            if i < len(landmarks_px):
                lx, ly, _ = landmarks_px[i]
                cv2.circle(frame, (lx, ly), 4, (255, 0, 255), -1)
        for idx, name in [
            (POSE_LEFT_HIP,    "L hip"),
            (POSE_LEFT_KNEE,   "L knee"),
            (POSE_LEFT_ANKLE,  "L ankle"),
            (POSE_RIGHT_HIP,   "R hip"),
            (POSE_RIGHT_KNEE,  "R knee"),
            (POSE_RIGHT_ANKLE, "R ankle"),
        ]:
            if idx < len(landmarks_px):
                x, y = landmarks_px[idx][0], landmarks_px[idx][1]
                cv2.circle(frame, (x, y), 8, (0, 255, 255), 2)
                cv2.putText(frame, name, (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def _draw_foot_chart(self, frame):
        h, w  = frame.shape[:2]
        feet  = [t["foot"] for t in self.touches]
        left  = feet.count("L")
        right = feet.count("R")
        unk   = feet.count("U")
        total = len(feet)
        if total == 0:
            return

        chart_x   = 30
        chart_y   = h - 180
        bar_width = 40
        max_bar_h = 120
        gap       = 20

        cv2.rectangle(frame,
                      (chart_x - 10, chart_y - 30),
                      (chart_x + 3 * (bar_width + gap) + 10, h - 20),
                      (0, 0, 0), -1)
        cv2.rectangle(frame,
                      (chart_x - 10, chart_y - 30),
                      (chart_x + 3 * (bar_width + gap) + 10, h - 20),
                      (255, 255, 255), 1)
        cv2.putText(frame, "L vs R", (chart_x, chart_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        def draw_bar(x_offset, count, color, label):
            bar_h  = int((count / total) * max_bar_h) if total > 0 else 0
            top    = chart_y + (max_bar_h - bar_h)
            bottom = chart_y + max_bar_h
            bx     = chart_x + x_offset
            cv2.rectangle(frame, (bx, top), (bx + bar_width, bottom), color, -1)
            cv2.putText(frame, str(count), (bx + 10, top - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, label, (bx + 10, bottom + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        draw_bar(0,                     left,  (0, 200, 0),     "L")
        draw_bar(bar_width + gap,       right, (200, 100, 0),   "R")
        draw_bar(2 * (bar_width + gap), unk,   (120, 120, 120), "?")

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------

    def analyze_frame(self, frame, frame_idx, timestamp_ms=0):
        cv2.putText(
            frame, f"JuggleIQ | Frame: {frame_idx}",
            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )

        cx, cy, r, status = self.detect_ball(frame)
        t = frame_idx / self.fps
        self._append_ball_point(t, frame_idx, cx, cy, status)

        landmarks_px = self.detect_pose(frame, timestamp_ms)
        if landmarks_px is not None:
            self.last_landmarks = landmarks_px
            la = self._compute_knee_angle(landmarks_px, "left")
            ra = self._compute_knee_angle(landmarks_px, "right")
            if la is not None:
                self.knee_angles_left.append(la)
            if ra is not None:
                self.knee_angles_right.append(ra)

        # Touch detection — detected frames only
        if status == "detected" and len(self.ball_history_detected) > 0:
            self._update_touch_detection(self.ball_history_detected[-1])

        self._draw_trail(frame)
        self._draw_pose(frame, landmarks_px)

        if status == "detected":
            cv2.circle(frame, (cx, cy), r, (0, 165, 255), 4)
            cv2.circle(frame, (cx, cy), 8, (0, 165, 255), -1)
            cv2.putText(frame, "BALL", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        elif status == "predicted":
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), 4)
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
            cv2.putText(frame, "BALL (pred)", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        touch_count = len(self.touches)
        cv2.putText(frame, f"Touches: {touch_count}",
                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if self.touches:
            last_touch   = self.touches[-1]
            frames_since = frame_idx - last_touch["frame_idx"]
            if frames_since <= self.touch_flash_frames:
                tx, ty      = last_touch["x"], last_touch["y"]
                foot        = last_touch["foot"]
                flash_color = {
                    "L": (0, 200, 0),
                    "R": (200, 100, 0),
                    "U": (180, 180, 180)
                }.get(foot, (255, 255, 255))
                cv2.circle(frame, (tx, ty), 30, flash_color, 3)
                cv2.putText(
                    frame, f"TOUCH #{touch_count} [{foot}]",
                    (tx + 15, ty - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, flash_color, 2
                )

        self._draw_foot_chart(frame)
        return frame

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_foot_summary(self):
        feet   = [t["foot"] for t in self.touches]
        total  = len(feet)
        if total == 0:
            return {"L": 0, "R": 0, "U": 0, "total": 0,
                    "L_pct": 0.0, "R_pct": 0.0}
        counts = Counter(feet)
        return {
            "L":     counts.get("L", 0),
            "R":     counts.get("R", 0),
            "U":     counts.get("U", 0),
            "total": total,
            "L_pct": round(counts.get("L", 0) / total * 100, 1),
            "R_pct": round(counts.get("R", 0) / total * 100, 1),
        }

    def _calculate_best_streak(self):
        MAX_TOUCH_GAP  = 3.0
        if not self.touches:
            return 0, []
        best_streak    = 1
        current_streak = 1
        streaks        = []
        for i in range(1, len(self.touches)):
            gap = self.touches[i]["t"] - self.touches[i - 1]["t"]
            if gap < MAX_TOUCH_GAP:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1
            best_streak = max(best_streak, current_streak)
        streaks.append(current_streak)
        return best_streak, streaks

    def _compute_peak_and_drift(self):
        if len(self.touches) < 2:
            return
        self.peaks     = []
        self.drifts    = []
        self.intervals = []
        detected       = self.ball_history_detected
        for i in range(1, len(self.touches)):
            t_prev = self.touches[i - 1]
            t_curr = self.touches[i]
            self.intervals.append(round(t_curr["t"] - t_prev["t"], 3))
            self.drifts.append(abs(t_curr["x"] - t_prev["x"]))
            between = [
                p for p in detected
                if t_prev["t"] <= p["t"] <= t_curr["t"]
            ]
            if between:
                self.peaks.append(
                    max(p["y"] for p in between) -
                    min(p["y"] for p in between)
                )
            else:
                self.peaks.append(0)

    def _compute_rhythm_score(self):
        if len(self.intervals) < 2:
            return 0.0
        arr  = np.array(self.intervals)
        mean = np.mean(arr)
        if mean == 0:
            return 0.0
        cv = np.std(arr) / mean
        return round(max(0.0, min(100.0, (1 - cv) * 100)), 1)

    def _compute_skill_score(self):
        touch_count  = len(self.touches)
        rhythm_score = self._compute_rhythm_score()
        touch_score  = min(100.0, (touch_count / 50) * 100)

        if self.peaks:
            peak_arr   = np.array(self.peaks, dtype=np.float32)
            peak_mean  = float(np.mean(peak_arr))
            peak_cv    = float(np.std(peak_arr) / peak_mean) if peak_mean > 0 else 1.0
            peak_score = round(max(0.0, min(100.0, (1 - peak_cv) * 100)), 1)
        else:
            peak_mean  = 0.0
            peak_score = 0.0

        if self.drifts:
            avg_drift   = float(np.mean(self.drifts))
            drift_score = round(max(0.0, min(100.0, (1 - avg_drift / 300) * 100)), 1)
        else:
            avg_drift   = 0.0
            drift_score = 0.0

        final_score = round(
            touch_score  * 0.40 +
            rhythm_score * 0.30 +
            peak_score   * 0.15 +
            drift_score  * 0.15, 1
        )

        tips = []
        if touch_count < 5:
            tips.append("Keep the ball up longer — aim for 10+ touches")
        elif touch_count < 15:
            tips.append("Good effort! Try to chain more consecutive touches")

        if rhythm_score < 40:
            tips.append("Work on consistent timing — try to keep a steady beat")
        elif rhythm_score < 70:
            tips.append("Rhythm is developing — focus on even intervals")

        if peak_mean < 80:
            tips.append("Kick the ball higher for better control")

        if avg_drift > 150:
            tips.append("Reduce sideways drift — keep the ball centered")
        elif avg_drift > 80:
            tips.append("Good drift control — try to keep it even tighter")

        if final_score >= 80:
            tips.append("Excellent session! You have strong ball control")
        elif final_score >= 60:
            tips.append("Solid performance — keep practicing!")

        if not tips:
            tips.append("Keep it up!")

        return final_score, tips, {
            "touch_score":  round(touch_score,  1),
            "rhythm_score": round(rhythm_score, 1),
            "peak_score":   peak_score,
            "drift_score":  drift_score,
            "avg_peak_px":  round(peak_mean, 1),
            "avg_drift_px": round(avg_drift, 1),
        }

    def _compute_stiffness_feedback(self):
        result = {
            "avg_left_knee_angle":  None,
            "avg_right_knee_angle": None,
            "avg_knee_angle":       None,
            "stiffness_label":      "unknown",
            "stiffness_tip":        None
        }
        angles = []
        if self.knee_angles_left:
            avg_left = round(float(np.mean(self.knee_angles_left)), 1)
            result["avg_left_knee_angle"] = avg_left
            angles.append(avg_left)
        if self.knee_angles_right:
            avg_right = round(float(np.mean(self.knee_angles_right)), 1)
            result["avg_right_knee_angle"] = avg_right
            angles.append(avg_right)
        if not angles:
            return result

        avg_angle = round(float(np.mean(angles)), 1)
        result["avg_knee_angle"] = avg_angle

        if avg_angle > 160:
            result["stiffness_label"] = "very_stiff"
            result["stiffness_tip"]   = (
                "Your legs are very straight — bend your knees more "
                "for better balance and ball control"
            )
        elif avg_angle > 140:
            result["stiffness_label"] = "moderately_stiff"
            result["stiffness_tip"]   = (
                "Try to keep your knees slightly more bent — "
                "it gives you better reaction time"
            )
        elif avg_angle > 120:
            result["stiffness_label"] = "good"
            result["stiffness_tip"]   = (
                "Good knee bend — you have a solid athletic stance"
            )
        else:
            result["stiffness_label"] = "very_bent"
            result["stiffness_tip"]   = (
                "Knees very bent — make sure you stay comfortable "
                "and not overly crouched"
            )
        return result

    def get_results(self):
        """Returns full results dict — ready for FastAPI."""
        summary                      = self.get_foot_summary()
        best_streak, all_streaks     = self._calculate_best_streak()
        self._compute_peak_and_drift()
        skill_score, tips, breakdown = self._compute_skill_score()
        stiffness                    = self._compute_stiffness_feedback()

        if stiffness["stiffness_tip"]:
            tips.append(stiffness["stiffness_tip"])

        return {
            "touch_count":     summary["total"],
            "left_foot":       summary["L"],
            "right_foot":      summary["R"],
            "unknown":         summary["U"],
            "left_pct":        summary.get("L_pct", 0.0),
            "right_pct":       summary.get("R_pct", 0.0),
            "best_streak":     best_streak,
            "all_streaks":     all_streaks,
            "skill_score":     skill_score,
            "coaching_tips":   tips,
            "score_breakdown": breakdown,
            "rhythm_score":    breakdown["rhythm_score"],
            "avg_peak_px":     breakdown["avg_peak_px"],
            "avg_drift_px":    breakdown["avg_drift_px"],
            "intervals":       self.intervals,
            "knee_feedback":   stiffness,
            "touches": [
                {
                    "touch_num": i + 1,
                    "foot":      t["foot"],
                    "t":         round(t["t"], 2),
                    "frame_idx": t["frame_idx"],
                    "x":         t["x"],
                    "y":         t["y"]
                }
                for i, t in enumerate(self.touches)
            ]
        }

    # ------------------------------------------------------------------
    # Video I/O
    # ------------------------------------------------------------------

    def process_video(self, input_path, output_path):
        self.reset_tracking()
        self.pose_landmarker = self._create_pose_landmarker()

        input_path  = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path.resolve()}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.fps = fps
        print(f"Video Info -> FPS: {fps:.2f}, Size: {width}x{height}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_idx  = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            writer.write(self.analyze_frame(frame, frame_idx, timestamp_ms))
            frame_idx += 1

        cap.release()
        writer.release()

        duration = time.time() - start_time
        print(f"Processed {frame_idx} frames in {duration:.2f}s")
        print(f"Saved → {output_path.resolve()}")

        results = self.get_results()

        print(f"\n{'='*40}")
        print(f"Touch count   : {results['touch_count']}")
        print(f"Left  foot    : {results['left_foot']} ({results['left_pct']}%)")
        print(f"Right foot    : {results['right_foot']} ({results['right_pct']}%)")
        print(f"Unknown       : {results['unknown']}")
        print(f"Best streak   : {results['best_streak']}")
        print(f"Skill score   : {results['skill_score']} / 100")
        print(f"Rhythm score  : {results['rhythm_score']} / 100")
        print(f"Avg peak      : {results['avg_peak_px']} px")
        print(f"Avg drift     : {results['avg_drift_px']} px")
        print(f"Knee angle    : {results['knee_feedback']['avg_knee_angle']}° "
              f"({results['knee_feedback']['stiffness_label']})")
        print(f"\nCoaching tips:")
        for tip in results['coaching_tips']:
            print(f"  → {tip}")
        print(f"{'='*40}\n")

        return results


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    vp = VideoProcessor(conf=0.45, model_name="yolov8m.pt")

    BASE_DIR   = Path(__file__).resolve().parent
    input_dir  = BASE_DIR / "data" / "input_videos"
    output_dir = BASE_DIR / "data" / "output_videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    videos = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in video_exts
    ]

    if not videos:
        print("No videos found in input directory.")
        raise SystemExit(0)

    for video_path in videos:
        out_path = output_dir / f"{video_path.stem}_F4_annotated.mp4"
        print(f"\nProcessing: {video_path.name}")
        try:
            results = vp.process_video(video_path, out_path)
            print(f"Saved → {out_path.name}")
        except Exception as e:
            print(f"Failed on {video_path.name}: {e}")