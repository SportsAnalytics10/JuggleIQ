import cv2
import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO


class VideoProcessor:
    def __init__(self, conf=0.45, model_name="yolov8m.pt"):
        print("VideoProcessor initialized.")
        self.model = YOLO(model_name)
        self.conf = conf
        self.reset_tracking()

    def reset_tracking(self):
        """Reset all tracking state between videos."""
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
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        self.kalman_initialized          = False
        self.kalman_predicted_this_frame = False
        self.lost_frames                 = 0
        self.MAX_LOST                    = 12
        self.last_radius                 = 20

        self.smooth_cx = None
        self.smooth_cy = None
        self.ALPHA     = 0.55

        # FIX 1: default fps so analyze_frame never crashes
        self.fps = 30.0

        # Trajectory trail
        self.ball_history  = []
        self.trail_len     = 25

        # Jump rejection threshold (pixels)
        self.MAX_JUMP_PX   = 150

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
        """Always safe — flattens 2D output before reading."""
        pred = self.kf.predict()
        pred = np.array(pred).flatten()
        return int(pred[0]), int(pred[1])

    def _kalman_state(self):
        """Safely read statePost without predicting."""
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
    # Jump rejection helper
    # ------------------------------------------------------------------

    def _is_valid_jump(self, cx, cy):
        """
        Returns True if (cx, cy) is within MAX_JUMP_PX of the
        last valid point in history. Always True if no history yet.
        """
        # Find last valid point
        for p in reversed(self.ball_history):
            if p["x"] is not None and p["y"] is not None:
                last_x, last_y = p["x"], p["y"]
                dist = np.sqrt((cx - last_x) ** 2 + (cy - last_y) ** 2)
                return dist <= self.MAX_JUMP_PX  # True = valid, False = crazy jump
        return True  # no history yet — always accept first point

    # ------------------------------------------------------------------
    # YOLO detection
    # ------------------------------------------------------------------

    def _yolo_candidates(self, frame):
        h_frame, w_frame = frame.shape[:2]
        results = self.model.predict(
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

    # ------------------------------------------------------------------
    # Main detection logic
    # ------------------------------------------------------------------

    def detect_ball(self, frame):
        """
        Returns (cx, cy, radius, status)
        'detected'  → YOLO confirmed
        'predicted' → Kalman filling gap
        None        → completely lost
        """
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
    # Trajectory trail  (Milestone C)
    # ------------------------------------------------------------------

    def _draw_trail(self, frame):
        """
        Draw last N valid positions as a fading trajectory trail.
        Orange dots = detected, Red dots = predicted, White line = path
        """
        if len(self.ball_history) < 2:
            return

        recent = [p for p in self.ball_history if p["x"] is not None]
        if len(recent) < 2:
            return

        # White connecting polyline drawn first (behind dots)
        pts = np.array(
            [(p["x"], p["y"]) for p in recent], dtype=np.int32
        ).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=(255, 255, 255), thickness=2)

        # Fading dots — older = smaller, newer = bigger
        total = len(recent)
        for i, p in enumerate(recent):
            alpha  = (i + 1) / total
            radius = max(2, int(6 * alpha))
            color  = (0, 165, 255) if p["status"] == "detected" else (0, 0, 255)
            cv2.circle(frame, (p["x"], p["y"]), radius, color, -1)

    # ------------------------------------------------------------------
    # Metrics helper — detected points only
    # ------------------------------------------------------------------

    def get_detected_history(self):
        """
        Returns only 'detected' points from ball_history.
        Use this for computing touches, peaks, and any metrics.
        Predicted points are excluded to prevent false readings.
        """
        return [
            p for p in self.ball_history
            if p["status"] == "detected" and p["x"] is not None
        ]

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------

    def analyze_frame(self, frame, frame_idx):
        cv2.putText(
            frame, f"JuggleIQ | Frame: {frame_idx}",
            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )

        cx, cy, r, status = self.detect_ball(frame)
        t = frame_idx / self.fps

        # ── Jump rejection before appending ─────────────────────────
        if cx is not None and cy is not None:
            if self._is_valid_jump(cx, cy):
                # Valid point — append with real position
                self.ball_history.append({
                    "t":      t,
                    "x":      cx,
                    "y":      cy,
                    "status": status
                })
            else:
                # Crazy jump detected — append None so trail gaps correctly
                self.ball_history.append({
                    "t":      t,
                    "x":      None,
                    "y":      None,
                    "status": None
                })
        else:
            # Ball completely lost
            self.ball_history.append({
                "t":      t,
                "x":      None,
                "y":      None,
                "status": None
            })

        # Cap history to trail_len — prevents memory leak
        if len(self.ball_history) > self.trail_len:
            self.ball_history.pop(0)

        # Draw trail behind ball circle
        self._draw_trail(frame)

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

        return frame

    # ------------------------------------------------------------------
    # Video I/O
    # ------------------------------------------------------------------

    def process_video(self, input_path, output_path):
        self.reset_tracking()

        input_path  = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path.resolve()}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Set real fps before loop starts
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
            writer.write(self.analyze_frame(frame, frame_idx))
            frame_idx += 1

        cap.release()
        writer.release()

        duration = time.time() - start_time
        print(f"Processed {frame_idx} frames in {duration:.2f}s")
        print(f"Saved → {output_path.resolve()}")


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
        out_path = output_dir / f"{video_path.stem}_C_trail_annotated.mp4"
        print(f"\nProcessing: {video_path.name}")
        try:
            vp.process_video(video_path, out_path)
            print(f"Saved → {out_path.name}")
        except Exception as e:
            print(f"Failed on {video_path.name}: {e}")