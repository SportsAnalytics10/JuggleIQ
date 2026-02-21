# import cv2
# import time
# from pathlib import Path
# import numpy as np
# from ultralytics import YOLO


# class VideoProcessor:
#     def __init__(self, conf=0.4, model_name="yolov8n.pt"):
#         """
#         conf: YOLO confidence threshold
#         model_name: use yolov8n.pt for speed, yolov8m.pt for a bit more accuracy (slower)
#         """
#         print("VideoProcessor initialized.")
#         self.model = YOLO(model_name)
#         self.conf = conf

#         # --- Kalman Filter Setup ---
#         # State: [x, y, vx, vy]^T  (4)
#         # Measurement: [x, y]^T   (2)
#         self.kf = cv2.KalmanFilter(4, 2)
#         self.kf.measurementMatrix = np.array(
#             [[1, 0, 0, 0],
#              [0, 1, 0, 0]], dtype=np.float32
#         )
#         self.kf.transitionMatrix = np.array(
#             [[1, 0, 1, 0],
#              [0, 1, 0, 1],
#              [0, 0, 1, 0],
#              [0, 0, 0, 1]], dtype=np.float32
#         )
#         self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
#         self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

#         self.reset_tracking()

#     def reset_tracking(self):
#         """Reset tracking state (important when processing multiple videos)."""
#         self.kalman_initialized = False
#         self.lost_frames = 0
#         self.MAX_LOST = 5
#         self.last_radius = 20

#     def _init_kalman(self, cx, cy):
#         """Initialize Kalman state at first detection."""
#         self.kf.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
#         self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
#         self.kalman_initialized = True
#         self.lost_frames = 0

#     # def detect_ball(self, frame, run_detector: bool = True):
#     #     """
#     #     Returns (cx, cy, radius, status)
#     #     status = 'detected'  → YOLO found ball (green)
#     #     status = 'predicted' → Kalman predicted (yellow)
#     #     status = None        → completely lost
#     #     """
#     #     h_frame, w_frame = frame.shape[:2]

#     #     # COCO class id for "sports ball" is 32
#     #     results = self.model.predict(frame, conf=self.conf, classes=[32], verbose=False)
#     #     boxes = results[0].boxes

#     #     candidates = []

#     #     if boxes is not None and len(boxes) > 0:
#     #         for b in boxes:
#     #             x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
#     #             cx = (x1 + x2) / 2.0
#     #             cy = (y1 + y2) / 2.0
#     #             w = x2 - x1
#     #             h = y2 - y1
#     #             score = float(b.conf[0].cpu().numpy())

#     #             # Min size filter (avoid tiny false positives)
#     #             min_size = 0.02 * min(h_frame, w_frame)
#     #             if w < min_size or h < min_size:
#     #                 continue

#     #             # Aspect ratio filter (ball roughly square-ish box)
#     #             aspect = w / h if h > 0 else 0
#     #             if not (0.6 < aspect < 1.6):
#     #                 continue

#     #             # Max size filter (avoid giant boxes)
#     #             if w > 0.5 * w_frame or h > 0.5 * h_frame:
#     #                 continue

#     #             r = int(0.25 * (w + h))
#     #             candidates.append((score, cx, cy, r))

#     #     # --- YOLO found candidates ---
#     #     # if candidates:
#     #     #     score, cx, cy, r = max(candidates, key=lambda t: t[0])

#     #     #     if not self.kalman_initialized:
#     #     #         self._init_kalman(cx, cy)
#     #     #     else:
#     #     #         # predict then correct with measurement
#     #     #         self.kf.predict()
#     #     #         meas = np.array([[np.float32(cx)], [np.float32(cy)]], dtype=np.float32)
#     #     #         self.kf.correct(meas)

#     #     #     self.lost_frames = 0
#     #     #     self.last_radius = int(r)
#     #     #     return int(cx), int(cy), int(r), "detected"

#     #     if candidates:
#     #         if self.kalman_initialized:
#     #             pred = self.kf.predict()
#     #             px = float(pred[0, 0])
#     #             py = float(pred[1, 0])

#     #             # choose candidate closest to predicted (with confidence tie-break)
#     #             candidates.sort(key=lambda t: ( (t[1]-px)**2 + (t[2]-py)**2, -t[0] ))
#     #             score, cx, cy, r = candidates[0]

#     #             # GATE: if detection is too far, treat as miss
#     #             dist2 = (cx - px) ** 2 + (cy - py) ** 2
#     #             if dist2 > (0.15 * min(w_frame, h_frame)) ** 2:  # ~15% of frame size
#     #                 candidates = []  # force miss handling
#     #         else:
#     #             score, cx, cy, r = max(candidates, key=lambda t: t[0])

#     #     # --- YOLO missed: use Kalman prediction ---
#     #     if not self.kalman_initialized:
#     #         return None, None, None, None  # never seen ball yet

#     #     self.lost_frames += 1
#     #     if self.lost_frames > self.MAX_LOST:
#     #         self.kalman_initialized = False
#     #         return None, None, None, None

#     #     pred = self.kf.predict()

#     #     # ✅ FIX: pred elements are arrays; convert to Python scalars safely
#     #     cx = int(pred[0, 0].item())
#     #     cy = int(pred[1, 0].item())
#     #     return cx, cy, int(self.last_radius), "predicted"

#     def detect_ball(self, frame, run_detector: bool = True):
#         """
#         Returns (cx, cy, radius, status)
#         """

#         h_frame, w_frame = frame.shape[:2]

#         # =========================================================
#         # 🔥 1️⃣ IF WE DO NOT WANT TO RUN YOLO → ONLY PREDICT
#         # =========================================================
#         if not run_detector:
#             if not self.kalman_initialized:
#                 return None, None, None, None

#             self.lost_frames += 1
#             if self.lost_frames > self.MAX_LOST:
#                 self.kalman_initialized = False
#                 return None, None, None, None

#             pred = self.kf.predict()
#             cx = int(pred[0, 0].item())
#             cy = int(pred[1, 0].item())
#             return cx, cy, int(self.last_radius), "predicted"

#         # =========================================================
#         # 🔥 2️⃣ OTHERWISE RUN YOLO (DETECTION STEP)
#         # =========================================================

#         results = self.model.predict(frame, conf=self.conf, classes=[32], verbose=False)
#         boxes = results[0].boxes

#         candidates = []

#         if boxes is not None and len(boxes) > 0:
#             for b in boxes:
#                 x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
#                 cx = (x1 + x2) / 2.0
#                 cy = (y1 + y2) / 2.0
#                 w = x2 - x1
#                 h = y2 - y1
#                 score = float(b.conf[0].cpu().numpy())

#                 min_size = 0.02 * min(h_frame, w_frame)
#                 if w < min_size or h < min_size:
#                     continue

#                 aspect = w / h if h > 0 else 0
#                 if not (0.6 < aspect < 1.6):
#                     continue

#                 if w > 0.5 * w_frame or h > 0.5 * h_frame:
#                     continue

#                 r = int(0.25 * (w + h))
#                 candidates.append((score, cx, cy, r))

#         # =========================================================
#         # 🔥 3️⃣ IF YOLO FOUND CANDIDATE → CORRECT KALMAN
#         # =========================================================
#         if candidates:
#             if self.kalman_initialized:
#                 pred = self.kf.predict()
#                 px = float(pred[0, 0])
#                 py = float(pred[1, 0])

#                 candidates.sort(key=lambda t: ((t[1]-px)**2 + (t[2]-py)**2, -t[0]))
#                 score, cx, cy, r = candidates[0]

#                 dist2 = (cx - px) ** 2 + (cy - py) ** 2
#                 if dist2 > (0.15 * min(w_frame, h_frame)) ** 2:
#                     candidates = []

#             else:
#                 score, cx, cy, r = max(candidates, key=lambda t: t[0])

#         if candidates:
#             if not self.kalman_initialized:
#                 self._init_kalman(cx, cy)
#             else:
#                 meas = np.array([[np.float32(cx)], [np.float32(cy)]], dtype=np.float32)
#                 self.kf.correct(meas)

#             self.lost_frames = 0
#             self.last_radius = int(r)
#             return int(cx), int(cy), int(r), "detected"

#         # =========================================================
#         # 🔥 4️⃣ YOLO MISSED → FALL BACK TO KALMAN
#         # =========================================================

#         if not self.kalman_initialized:
#             return None, None, None, None

#         self.lost_frames += 1
#         if self.lost_frames > self.MAX_LOST:
#             self.kalman_initialized = False
#             return None, None, None, None

#         pred = self.kf.predict()
#         cx = int(pred[0, 0].item())
#         cy = int(pred[1, 0].item())
#         return cx, cy, int(self.last_radius), "predicted"

#     # def analyze_frame(self, frame, frame_idx):
#     def analyze_frame(self, frame, frame_idx, run_detector: bool):
#         # Debug overlay
#         cv2.putText(
#             frame,
#             f"AltinhaAI | Frame: {frame_idx}",
#             (30, 50),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1.0,
#             (0, 255, 0),
#             2
#         )

#         # cx, cy, r, status = self.detect_ball(frame)
#         cx, cy, r, status = self.detect_ball(frame, run_detector=run_detector)

#         if status == "detected":
#             cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
#             cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
#             cv2.putText(frame, "BALL", (cx + 10, cy - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         elif status == "predicted":
#             cv2.circle(frame, (cx, cy), r, (0, 255, 255), 2)
#             cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
#             cv2.putText(frame, "BALL (pred)", (cx + 10, cy - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#         return frame

#     def process_video(self, input_path, output_path, target_fps: int = 10):
#         """
#         Processes input video frame-by-frame and writes an annotated output video.
#         Runs YOLO only every 'stride' frames (approx target_fps). Kalman predicts in between.

#         target_fps: desired YOLO detection rate (e.g., 10). Output video FPS remains same as input.
#         """
#         # Reset per video (important when processing multiple videos)
#         self.reset_tracking()

#         input_path = Path(input_path)
#         output_path = Path(output_path)
#         output_path.parent.mkdir(parents=True, exist_ok=True)

#         cap = cv2.VideoCapture(str(input_path))
#         if not cap.isOpened():
#             raise RuntimeError(f"Cannot open video: {input_path.resolve()}")

#         fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         # Decide how often to run YOLO
#         stride = max(1, int(round(fps / float(target_fps))))
#         print(f"Video Info -> FPS: {fps:.2f}, Size: {width}x{height}")
#         print(f"Running YOLO every {stride} frames (~{fps/stride:.1f} FPS)")

#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
#         if not writer.isOpened():
#             cap.release()
#             raise RuntimeError(f"Cannot open VideoWriter for: {output_path.resolve()}")

#         frame_idx = 0
#         start_time = time.time()

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             run_detector = (frame_idx % stride == 0)

#             # IMPORTANT: analyze_frame must accept run_detector now
#             processed_frame = self.analyze_frame(frame, frame_idx, run_detector)
#             writer.write(processed_frame)

#             frame_idx += 1

#         cap.release()
#         writer.release()

#         duration = time.time() - start_time
#         print(f"Processed {frame_idx} frames in {duration:.2f} seconds")
#         print(f"Saved output to: {output_path.resolve()}")

#     # def process_video(self, input_path, output_path):
#     #     # ✅ Important for multi-video: reset tracker per video
#     #     self.reset_tracking()

#     #     input_path = Path(input_path)
#     #     output_path = Path(output_path)
#     #     output_path.parent.mkdir(parents=True, exist_ok=True)
#     #     run_detector = (frame_idx % stride == 0)
#     #     processed_frame = self.analyze_frame(frame, frame_idx, run_detector)

#     #     cap = cv2.VideoCapture(str(input_path))
#     #     if not cap.isOpened():
#     #         raise RuntimeError(f"Cannot open video: {input_path.resolve()}")

#     #     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     #     print(f"Video Info -> FPS: {fps:.2f}, Size: {width}x{height}")

#     #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     #     writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

#     #     frame_idx = 0
#     #     start_time = time.time()

#     #     while True:
#     #         ret, frame = cap.read()
#     #         if not ret:
#     #             break

#     #         processed_frame = self.analyze_frame(frame, frame_idx)
#     #         writer.write(processed_frame)

#     #         frame_idx += 1

#     #     cap.release()
#     #     writer.release()

#     #     duration = time.time() - start_time
#     #     print(f"Processed {frame_idx} frames in {duration:.2f} seconds")
#     #     print(f"Saved output to: {output_path.resolve()}")


# if __name__ == "__main__":
#     vp = VideoProcessor(conf=0.2, model_name="yolov8s.pt")  # change to yolov8m.pt if you want

#     BASE_DIR = Path(__file__).resolve().parent
#     input_dir = BASE_DIR / "data" / "input_videos"
#     output_dir = BASE_DIR / "data" / "output_videos"
#     output_dir.mkdir(parents=True, exist_ok=True)

#     print("Running from:", BASE_DIR)
#     print("Input folder:", input_dir)
#     print("Output folder:", output_dir)

#     video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

#     if not input_dir.exists():
#         raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

#     videos = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in video_exts]
#     if not videos:
#         print("No videos found in input directory.")
#         raise SystemExit(0)

#     for video_path in sorted(videos):
#         out_path = output_dir / f"{video_path.stem}_B_ball_annotated.mp4"
#         print(f"\nProcessing: {video_path.name}")

#         try:
#             vp.process_video(video_path, out_path)
#             print(f"Saved → {out_path}")
#         except Exception as e:
#             print(f"Failed on {video_path.name}: {e}")


# import cv2
# import time
# from pathlib import Path
# import numpy as np
# from ultralytics import YOLO


# class VideoProcessor:
#     def __init__(self, conf=0.45, model_name="yolov8m.pt"):
#         print("VideoProcessor initialized.")
#         self.model = YOLO(model_name)
#         self.conf = conf
#         self.reset_tracking()

#     def reset_tracking(self):
#         """Reset all tracking state between videos."""
#         # Kalman filter
#         self.kf = cv2.KalmanFilter(4, 2)
#         self.kf.measurementMatrix = np.array(
#             [[1, 0, 0, 0],
#              [0, 1, 0, 0]], dtype=np.float32)
#         self.kf.transitionMatrix = np.array(
#             [[1, 0, 1, 0],
#              [0, 1, 0, 1],
#              [0, 0, 1, 0],
#              [0, 0, 0, 1]], dtype=np.float32)

#         # FIX 1: Higher process noise so Kalman reacts to fast ball movement
#         self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
#         self.kf.processNoiseCov[2, 2] = 5.0  # vx changes fast
#         self.kf.processNoiseCov[3, 3] = 5.0  # vy changes fast
#         self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

#         self.kalman_initialized = False
#         self.kalman_predicted_this_frame = False  # FIX 2: avoid double predict
#         self.lost_frames = 0
#         self.MAX_LOST = 12
#         self.last_radius = 20

#         # FIX 3: Exponential smoothing on top of Kalman to reduce jitter
#         self.smooth_cx = None
#         self.smooth_cy = None
#         self.ALPHA = 0.55  # lower = smoother, higher = more responsive

#     def _init_kalman(self, cx, cy):
#         self.kf.statePre  = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
#         self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
#         self.kalman_initialized = True
#         self.lost_frames = 0

#     def _smooth(self, cx, cy):
#         """Exponential moving average to reduce circle jitter."""
#         if self.smooth_cx is None:
#             self.smooth_cx = cx
#             self.smooth_cy = cy
#         else:
#             self.smooth_cx = int(self.ALPHA * cx + (1 - self.ALPHA) * self.smooth_cx)
#             self.smooth_cy = int(self.ALPHA * cy + (1 - self.ALPHA) * self.smooth_cy)
#         return self.smooth_cx, self.smooth_cy

#     def _yolo_candidates(self, frame):
#         """Run YOLO and return filtered ball candidates."""
#         h_frame, w_frame = frame.shape[:2]
#         results = self.model.predict(frame, conf=self.conf, classes=[32], verbose=False)
#         boxes = results[0].boxes
#         candidates = []

#         if boxes is None or len(boxes) == 0:
#             return candidates

#         for b in boxes:
#             x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
#             cx = (x1 + x2) / 2.0
#             cy = (y1 + y2) / 2.0
#             w  = x2 - x1
#             h  = y2 - y1
#             score = float(b.conf[0].cpu().numpy())

#             # Min size — ignore tiny blobs
#             min_size = 0.02 * min(h_frame, w_frame)
#             if w < min_size or h < min_size:
#                 continue

#             # Aspect ratio — ball must be roughly circular
#             aspect = w / h if h > 0 else 0
#             if not (0.55 < aspect < 1.8):
#                 continue

#             # Max size — ball can't dominate the frame
#             if w > 0.45 * w_frame or h > 0.45 * h_frame:
#                 continue

#             # FIX: Reject detections near bottom 10% of frame (ground false positives)
#             if cy > h_frame * 0.90:
#                 continue

#             r = int(0.25 * (w + h))
#             candidates.append((score, cx, cy, r))

#         return candidates

#     def detect_ball(self, frame):
#         """
#         Returns (cx, cy, radius, status)
#         'detected'  → YOLO confirmed  (green)
#         'predicted' → Kalman only     (yellow)
#         None        → completely lost
#         """
#         candidates = self._yolo_candidates(frame)
#         self.kalman_predicted_this_frame = False

#         if candidates:
#             # Pick highest confidence detection
#             score, cx, cy, r = max(candidates, key=lambda t: t[0])

#             if not self.kalman_initialized:
#                 self._init_kalman(cx, cy)
#             else:
#                 # FIX 2: Only predict if we haven't already this frame
#                 self.kf.predict()
#                 self.kalman_predicted_this_frame = True
#                 meas = np.array([[np.float32(cx)], [np.float32(cy)]], dtype=np.float32)
#                 self.kf.correct(meas)

#             self.lost_frames = 0
#             self.last_radius = int(r)
#             cx, cy = self._smooth(int(cx), int(cy))
#             return cx, cy, int(r), "detected"

#         # --- YOLO missed ---
#         if not self.kalman_initialized:
#             return None, None, None, None

#         self.lost_frames += 1
#         if self.lost_frames > self.MAX_LOST:
#             # Ball lost too long — full reset including smoothing
#             self.reset_tracking()
#             return None, None, None, None

#         # FIX 2: Only predict if not already predicted this frame
#         if not self.kalman_predicted_this_frame:
#             pred = self.kf.predict()
#             self.kalman_predicted_this_frame = True
#         else:
#             pred = self.kf.statePost

#         cx = int(pred[0, 0].item())
#         cy = int(pred[1, 0].item())
#         cx, cy = self._smooth(cx, cy)
#         return cx, cy, int(self.last_radius), "predicted"

#     def analyze_frame(self, frame, frame_idx):
#         h, w = frame.shape[:2]

#         cv2.putText(frame, f"AltinhaAI | Frame: {frame_idx}",
#                     (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

#         cx, cy, r, status = self.detect_ball(frame)

#         if status == "detected":
#             cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
#             cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
#             cv2.putText(frame, "BALL", (cx + 10, cy - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         elif status == "predicted":
#             cv2.circle(frame, (cx, cy), r, (0, 255, 255), 2)
#             cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
#             cv2.putText(frame, "BALL (pred)", (cx + 10, cy - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#         return frame

#     def process_video(self, input_path, output_path):
#         self.reset_tracking()

#         input_path  = Path(input_path)
#         output_path = Path(output_path)
#         output_path.parent.mkdir(parents=True, exist_ok=True)

#         cap = cv2.VideoCapture(str(input_path))
#         if not cap.isOpened():
#             raise RuntimeError(f"Cannot open video: {input_path.resolve()}")

#         fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
#         width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         print(f"Video Info -> FPS: {fps:.2f}, Size: {width}x{height}")

#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

#         frame_idx  = 0
#         start_time = time.time()

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             writer.write(self.analyze_frame(frame, frame_idx))
#             frame_idx += 1

#         cap.release()
#         writer.release()
#         duration = time.time() - start_time
#         print(f"Processed {frame_idx} frames in {duration:.2f}s")
#         print(f"Saved → {output_path.resolve()}")


# if __name__ == "__main__":
#     # yolov8m.pt is the minimum viable model for sports ball detection
#     vp = VideoProcessor(conf=0.45, model_name="yolov8m.pt")

#     BASE_DIR   = Path(__file__).resolve().parent
#     input_dir  = BASE_DIR / "data" / "input_videos"
#     output_dir = BASE_DIR / "data" / "output_videos"
#     output_dir.mkdir(parents=True, exist_ok=True)

#     video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
#     videos = [p for p in sorted(input_dir.iterdir())
#               if p.is_file() and p.suffix.lower() in video_exts]

#     if not videos:
#         print("No videos found.")
#         raise SystemExit(0)

#     for video_path in videos:
#         out_path = output_dir / f"{video_path.stem}_B_ball_annotated.mp4"
#         print(f"\nProcessing: {video_path.name}")
#         try:
#             vp.process_video(video_path, out_path)
#         except Exception as e:
#             print(f"Failed on {video_path.name}: {e}")

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

        # Higher velocity noise = reacts faster to direction changes
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.processNoiseCov[2, 2] = 5.0
        self.kf.processNoiseCov[3, 3] = 5.0
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        self.kalman_initialized     = False
        self.kalman_predicted_this_frame = False
        self.lost_frames            = 0
        self.MAX_LOST               = 12
        self.last_radius            = 20

        # Exponential smoothing state
        self.smooth_cx = None
        self.smooth_cy = None
        self.ALPHA     = 0.55  # 0 = very smooth, 1 = no smoothing

    # ------------------------------------------------------------------
    # Kalman helpers
    # ------------------------------------------------------------------

    def _init_kalman(self, cx, cy, h_frame):
        """
        Initialize Kalman only if position is in a reasonable area.
        Rejects false positives in top 15% and bottom 10% of frame.
        """
        if cy < h_frame * 0.15:
            return  # likely sky / lights — skip
        if cy > h_frame * 0.90:
            return  # likely ground — skip

        self.kf.statePre  = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        self.kalman_initialized = True
        self.lost_frames = 0

    def _kalman_predict(self):
        """
        Safe Kalman predict — always returns plain Python ints.
        Flattens the result so it works across all numpy/OpenCV versions.
        """
        pred = self.kf.predict()
        pred = np.array(pred).flatten()
        return int(pred[0]), int(pred[1])

    def _kalman_correct(self, cx, cy):
        """Safe Kalman correct."""
        meas = np.array([[np.float32(cx)], [np.float32(cy)]], dtype=np.float32)
        self.kf.correct(meas)

    def _smooth(self, cx, cy):
        """Exponential moving average to reduce circle jitter."""
        if self.smooth_cx is None:
            self.smooth_cx = cx
            self.smooth_cy = cy
        else:
            self.smooth_cx = int(self.ALPHA * cx + (1 - self.ALPHA) * self.smooth_cx)
            self.smooth_cy = int(self.ALPHA * cy + (1 - self.ALPHA) * self.smooth_cy)
        return self.smooth_cx, self.smooth_cy

    # ------------------------------------------------------------------
    # YOLO detection
    # ------------------------------------------------------------------

    def _yolo_candidates(self, frame):
        """
        Run YOLO and return filtered ball candidates.
        Each candidate: (score, cx, cy, radius)
        """
        h_frame, w_frame = frame.shape[:2]
        results = self.model.predict(
            frame, conf=self.conf, classes=[32], verbose=False
        )
        boxes = results[0].boxes
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

            # --- Spatial rejection ---
            # Reject top 15% of frame (sky, lights, crowd)
            if cy < h_frame * 0.15:
                continue
            # Reject bottom 10% of frame (ground, shoes)
            if cy > h_frame * 0.90:
                continue

            # --- Size filters ---
            min_size = 0.02 * min(h_frame, w_frame)
            if w < min_size or h < min_size:
                continue
            if w > 0.45 * w_frame or h > 0.45 * h_frame:
                continue

            # --- Shape filter: ball must be roughly circular ---
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
        'detected'  → YOLO confirmed        → draw green
        'predicted' → Kalman filling gap    → draw yellow
        None        → completely lost       → draw nothing
        """
        h_frame = frame.shape[0]
        candidates = self._yolo_candidates(frame)
        self.kalman_predicted_this_frame = False

        # ── YOLO found the ball ──────────────────────────────────────
        if candidates:
            score, cx, cy, r = max(candidates, key=lambda t: t[0])
            cx, cy = int(cx), int(cy)

            if not self.kalman_initialized:
                self._init_kalman(cx, cy, h_frame)
            else:
                # predict → correct (standard Kalman cycle)
                self._kalman_predict()
                self.kalman_predicted_this_frame = True
                self._kalman_correct(cx, cy)

            self.lost_frames  = 0
            self.last_radius  = int(r)
            cx, cy = self._smooth(cx, cy)
            return cx, cy, int(r), "detected"

        # ── YOLO missed ──────────────────────────────────────────────
        if not self.kalman_initialized:
            return None, None, None, None

        self.lost_frames += 1

        if self.lost_frames > self.MAX_LOST:
            # Lost too long — full reset so stale state doesn't persist
            self.reset_tracking()
            return None, None, None, None

        # Predict only if we haven't already this frame
        if not self.kalman_predicted_this_frame:
            cx, cy = self._kalman_predict()
            self.kalman_predicted_this_frame = True
        else:
            flat   = np.array(self.kf.statePost).flatten()
            cx, cy = int(flat[0]), int(flat[1])

        cx, cy = self._smooth(cx, cy)
        return cx, cy, int(self.last_radius), "predicted"

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------

    def analyze_frame(self, frame, frame_idx):
        cv2.putText(
            frame, f"JuggleIQ | Frame: {frame_idx}",
            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )

        cx, cy, r, status = self.detect_ball(frame)

        if status == "detected":
            # Solid green circle — YOLO confirmed
            # cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
            # cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            # cv2.putText(frame, "BALL", (cx + 10, cy - 10),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.circle(frame, (cx, cy), r, (0, 165, 255), 4)   # outer circle thick
            cv2.circle(frame, (cx, cy), 8, (0, 165, 255), -1)  # center dot
            cv2.putText(frame, "BALL", (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        elif status == "predicted":
            # Yellow circle — Kalman filling in gap
            # cv2.circle(frame, (cx, cy), r, (0, 255, 255), 2)
            # cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            # cv2.putText(frame, "BALL (pred)", (cx + 10, cy - 10),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.circle(frame, (cx, cy), r, (0, 0, 255), 4)     # outer circle thick
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)    # center dot
            cv2.putText(frame, "BALL (pred)", (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    # ------------------------------------------------------------------
    # Video I/O
    # ------------------------------------------------------------------

    def process_video(self, input_path, output_path):
        # Always reset tracker at start of each video
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
        out_path = output_dir / f"{video_path.stem}_B_ball_annotated.mp4"
        print(f"\nProcessing: {video_path.name}")
        try:
            vp.process_video(video_path, out_path)
            print(f"Saved → {out_path.name}")
        except Exception as e:
            print(f"Failed on {video_path.name}: {e}")