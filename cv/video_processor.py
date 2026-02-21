# import cv2
# import time
# from pathlib import Path
# import numpy as np
# from ultralytics import YOLO
# import math


# class VideoProcessor:
#     def __init__(self, conf=0.4):
#         print("VideoProcessor initialized.")
#         self.model = YOLO("yolov8m.pt")
#         self.conf = conf

#         # --- Kalman Filter Setup ---
#         self.kf = cv2.KalmanFilter(4, 2)  # 4 state vars (x,y,vx,vy), 2 measurements (x,y)
#         self.kf.measurementMatrix = np.array(
#             [[1, 0, 0, 0],
#              [0, 1, 0, 0]], np.float32)
#         self.kf.transitionMatrix = np.array(
#             [[1, 0, 1, 0],
#              [0, 1, 0, 1],
#              [0, 0, 1, 0],
#              [0, 0, 0, 1]], np.float32)
#         self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.03
#         self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

#         self.kalman_initialized = False
#         self.lost_frames = 0
#         self.MAX_LOST = 15  # how many frames to keep predicting before giving up
#         self.last_radius = 20  # fallback radius when predicting

#     def _init_kalman(self, cx, cy):
#         """Initialize Kalman state at first detection."""
#         self.kf.statePre  = np.array([[cx], [cy], [0], [0]], np.float32)
#         self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
#         self.kalman_initialized = True
#         self.lost_frames = 0


# ## Ball Detection
#     def detect_ball(self, frame):
#         """
#         Returns (cx, cy, radius, status)
#         status = 'detected'  → YOLO found ball  (green circle)
#         status = 'predicted' → Kalman predicting (yellow circle)
#         status = None        → completely lost
#         """
#         results = self.model.predict(frame, conf=self.conf, classes=[32], verbose=False)
#         boxes = results[0].boxes

#         h_frame, w_frame = frame.shape[:2]
#         candidates = []

#         if boxes is not None and len(boxes) > 0:
#             for b in boxes:
#                 x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
#                 cx = (x1 + x2) / 2.0
#                 cy = (y1 + y2) / 2.0
#                 w  = x2 - x1
#                 h  = y2 - y1
#                 score = float(b.conf[0].cpu().numpy())

#                 # Min size filter
#                 min_size = 0.02 * min(h_frame, w_frame)
#                 if w < min_size or h < min_size:
#                     continue

#                 # Aspect ratio — ball is roughly circular
#                 aspect = w / h if h > 0 else 0
#                 if not (0.6 < aspect < 1.6):
#                     continue

#                 # Max size filter
#                 if w > 0.5 * w_frame or h > 0.5 * h_frame:
#                     continue

#                 r = int(0.25 * (w + h))
#                 candidates.append((score, cx, cy, r))

#         # --- YOLO found something ---
#         if candidates:
#             best = max(candidates, key=lambda t: t[0])
#             score, cx, cy, r = best

#             if not self.kalman_initialized:
#                 self._init_kalman(cx, cy)
#             else:
#                 self.kf.predict()
#                 self.kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))

#             self.lost_frames = 0
#             self.last_radius = r
#             return int(cx), int(cy), r, 'detected'

#         # --- YOLO missed — fall back to Kalman ---
#         if not self.kalman_initialized:
#             return None, None, None, None  # never seen the ball yet

#         self.lost_frames += 1

#         if self.lost_frames > self.MAX_LOST:
#             self.kalman_initialized = False  # give up, reset
#             return None, None, None, None

#         # Kalman predicts where ball should be
#         # pred = self.kf.predict()
#         # cx = int(pred[0])
#         # cy = int(pred[1])
#         # return cx, cy, self.last_radius, 'predicted'
#         pred = self.kf.predict()
#         cx = int(pred[0, 0].item())
#         cy = int(pred[1, 0].item())
#         return cx, cy, int(self.last_radius), 'predicted'
    
#     def reset_tracking(self):
#         self.kalman_initialized = False
#         self.lost_frames = 0
#         self.last_radius = 20

#     def process_video(self, input_path, output_path):
#         self.reset_tracking()


# ## Pose Detection of the Player
#     def detect_pose(self, frame):
#         # Milestone A: no pose yet
#         return None


# ## Frame analyzing 
#     def analyze_frame(self, frame, frame_idx):
#         """
#         This is where future CV logic will go.
#         For now, we just draw basic overlay text.
#         """
#         cv2.putText(
#             frame,
#             f"AltinhaAI | Frame: {frame_idx}",
#             (30, 50),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1.0,
#             (0, 255, 0),
#             2
#         )

#         cx, cy, r, status = self.detect_ball(frame)

#         if status == 'detected':
#             # Solid green — YOLO confirmed
#             cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
#             cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
#             cv2.putText(frame, "BALL", (cx + 10, cy - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         elif status == 'predicted':
#             # Yellow dashed — Kalman predicting, YOLO missed
#             cv2.circle(frame, (cx, cy), r, (0, 255, 255), 2)
#             cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
#             cv2.putText(frame, "BALL (pred)", (cx + 10, cy - 10),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#         return frame

#     def process_video(self, input_path, output_path):
#         input_path = Path(input_path)
#         output_path = Path(output_path)
#         output_path.parent.mkdir(parents=True, exist_ok=True)

#         cap = cv2.VideoCapture(str(input_path))
#         if not cap.isOpened():
#             raise RuntimeError(f"Cannot open video: {input_path.resolve()}")

#         fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         print(f"Video Info -> FPS: {fps:.2f}, Size: {width}x{height}")

#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

#         frame_idx = 0
#         start_time = time.time()

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             processed_frame = self.analyze_frame(frame, frame_idx)
#             writer.write(processed_frame)

#             frame_idx += 1

#         cap.release()
#         writer.release()

#         duration = time.time() - start_time
#         print(f"Processed {frame_idx} frames in {duration:.2f} seconds")
#         print(f"Saved output to: {output_path.resolve()}")



# if __name__ == "__main__":
#     # from pathlib import Path

#     # Initialize processor
#     vp = VideoProcessor()

#     # Base directory = this cv folder
#     BASE_DIR = Path(__file__).resolve().parent

#     # Input / Output paths
#     input_dir = BASE_DIR / "data" / "input_videos"
#     output_dir = BASE_DIR / "data" / "output_videos"

#     # Create output directory if missing
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # Debug prints (VERY useful)
#     print("Running from:", BASE_DIR)
#     print("Input folder:", input_dir)
#     print("Output folder:", output_dir)

#     # Allowed video formats
#     video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

#     # Safety checks
#     if not input_dir.exists():
#         raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

#     videos = [
#         p for p in input_dir.iterdir()
#         if p.is_file() and p.suffix.lower() in video_exts
#     ]

#     if not videos:
#         print("No videos found in input directory.")
#         exit()

#     # Process each video
#     for video_path in sorted(videos):
#         out_path = output_dir / f"{video_path.stem}_A_annotated.mp4"

#         print(f"\nProcessing: {video_path.name}")

#         try:
#             vp.process_video(video_path, out_path)
#             print(f"Saved → {out_path}")
#         except Exception as e:
#             print(f"Failed on {video_path.name}: {e}")
    # from pathlib import Path

    # vp = VideoProcessor()

    # input_dir = Path("data/input_videos")
    # output_dir = Path("Hacklytics/data/output_videos")
    # output_dir.mkdir(parents=True, exist_ok=True)

    # video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    # for video_path in sorted(input_dir.iterdir()):
    #     if video_path.suffix.lower() not in video_exts:
    #         continue

    #     out_path = output_dir / f"{video_path.stem}_A_annotated.mp4"

    #     try:
    #         vp.process_video(video_path, out_path)
    #     except Exception as e:
    #         print(f"Failed on {video_path.name}: {e}")


import cv2
import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO


class VideoProcessor:
    def __init__(self, conf=0.4, model_name="yolov8n.pt"):
        """
        conf: YOLO confidence threshold
        model_name: use yolov8n.pt for speed, yolov8m.pt for a bit more accuracy (slower)
        """
        print("VideoProcessor initialized.")
        self.model = YOLO(model_name)
        self.conf = conf

        # --- Kalman Filter Setup ---
        # State: [x, y, vx, vy]^T  (4)
        # Measurement: [x, y]^T   (2)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32
        )
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        self.reset_tracking()

    def reset_tracking(self):
        """Reset tracking state (important when processing multiple videos)."""
        self.kalman_initialized = False
        self.lost_frames = 0
        self.MAX_LOST = 15
        self.last_radius = 20

    def _init_kalman(self, cx, cy):
        """Initialize Kalman state at first detection."""
        self.kf.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        self.kalman_initialized = True
        self.lost_frames = 0

    def detect_ball(self, frame):
        """
        Returns (cx, cy, radius, status)
        status = 'detected'  → YOLO found ball (green)
        status = 'predicted' → Kalman predicted (yellow)
        status = None        → completely lost
        """
        h_frame, w_frame = frame.shape[:2]

        # COCO class id for "sports ball" is 32
        results = self.model.predict(frame, conf=self.conf, classes=[32], verbose=False)
        boxes = results[0].boxes

        candidates = []

        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1
                score = float(b.conf[0].cpu().numpy())

                # Min size filter (avoid tiny false positives)
                min_size = 0.02 * min(h_frame, w_frame)
                if w < min_size or h < min_size:
                    continue

                # Aspect ratio filter (ball roughly square-ish box)
                aspect = w / h if h > 0 else 0
                if not (0.6 < aspect < 1.6):
                    continue

                # Max size filter (avoid giant boxes)
                if w > 0.5 * w_frame or h > 0.5 * h_frame:
                    continue

                r = int(0.25 * (w + h))
                candidates.append((score, cx, cy, r))

        # --- YOLO found candidates ---
        if candidates:
            score, cx, cy, r = max(candidates, key=lambda t: t[0])

            if not self.kalman_initialized:
                self._init_kalman(cx, cy)
            else:
                # predict then correct with measurement
                self.kf.predict()
                meas = np.array([[np.float32(cx)], [np.float32(cy)]], dtype=np.float32)
                self.kf.correct(meas)

            self.lost_frames = 0
            self.last_radius = int(r)
            return int(cx), int(cy), int(r), "detected"

        # --- YOLO missed: use Kalman prediction ---
        if not self.kalman_initialized:
            return None, None, None, None  # never seen ball yet

        self.lost_frames += 1
        if self.lost_frames > self.MAX_LOST:
            self.kalman_initialized = False
            return None, None, None, None

        pred = self.kf.predict()

        # ✅ FIX: pred elements are arrays; convert to Python scalars safely
        cx = int(pred[0, 0].item())
        cy = int(pred[1, 0].item())
        return cx, cy, int(self.last_radius), "predicted"

    def analyze_frame(self, frame, frame_idx):
        # Debug overlay
        cv2.putText(
            frame,
            f"AltinhaAI | Frame: {frame_idx}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        cx, cy, r, status = self.detect_ball(frame)

        if status == "detected":
            cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
            cv2.putText(frame, "BALL", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        elif status == "predicted":
            cv2.circle(frame, (cx, cy), r, (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
            cv2.putText(frame, "BALL (pred)", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    def process_video(self, input_path, output_path):
        # ✅ Important for multi-video: reset tracker per video
        self.reset_tracking()

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path.resolve()}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video Info -> FPS: {fps:.2f}, Size: {width}x{height}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_idx = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.analyze_frame(frame, frame_idx)
            writer.write(processed_frame)

            frame_idx += 1

        cap.release()
        writer.release()

        duration = time.time() - start_time
        print(f"Processed {frame_idx} frames in {duration:.2f} seconds")
        print(f"Saved output to: {output_path.resolve()}")


if __name__ == "__main__":
    vp = VideoProcessor(conf=0.4, model_name="yolov8n.pt")  # change to yolov8m.pt if you want

    BASE_DIR = Path(__file__).resolve().parent
    input_dir = BASE_DIR / "data" / "input_videos"
    output_dir = BASE_DIR / "data" / "output_videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running from:", BASE_DIR)
    print("Input folder:", input_dir)
    print("Output folder:", output_dir)

    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    videos = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in video_exts]
    if not videos:
        print("No videos found in input directory.")
        raise SystemExit(0)

    for video_path in sorted(videos):
        out_path = output_dir / f"{video_path.stem}_B_ball_annotated.mp4"
        print(f"\nProcessing: {video_path.name}")

        try:
            vp.process_video(video_path, out_path)
            print(f"Saved → {out_path}")
        except Exception as e:
            print(f"Failed on {video_path.name}: {e}")