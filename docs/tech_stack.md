AltinhaAI Tech Stack (Hacklytics MVP)
Product/UI

Streamlit (Python) — main web UI (drills, start/stop, results page)

streamlit-webrtc — webcam capture / recording inside Streamlit

Computer Vision / ML

MediaPipe Pose — leg/foot landmarks (hip/knee/ankle/foot)

YOLO (Ultralytics YOLOv8/YOLO11 depending on what you use) — soccer ball detection

OpenCV — video I/O, frame processing, drawing overlays, output video rendering

NumPy — core math for trajectory + angles

(Optional) FilterPy or a simple Kalman filter — smoothing ball track

Backend Architecture

Single Python app (Streamlit runs UI + calls analyze_video() locally)

For hackathon speed, keep it monorepo + single runtime.

(Optional if you split services) FastAPI — separate analysis service if you want clean separation

Data / Storage

Local file storage during demo (write input.mp4, annotated.mp4, results.json)

(Optional) SQLite for saving sessions locally

Dev / Tooling

Python 3.10+

uv or pip + venv for dependencies

Git + GitHub

Cursor (development + prompting with @docs/PRD.md)