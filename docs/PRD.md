
# Product Requirements Document (PRD)
Product Name: AltinhaAI
Event: Hacklytics 2026
Owner: Gabriel Dos Santos
Version: Hackathon MVP (Upload + Record Mode)  

1. Overview

AltinhaAI is a computer vision-powered soccer juggling coach.
Users record themselves juggling a soccer ball, and the system analyzes their technique and provides actionable feedback.

The MVP will support:

Record session (10–20 seconds)

Automatic ball + pose tracking

Touch detection

Coaching feedback per touch

Drill progression (basic progression logic)

Annotated replay + session summary

This is a post-record analysis app (not real-time feedback).

2. Problem Statement

Learning to juggle (altinha) is difficult without feedback.
Players struggle with:

Overpowering touches

Poor contact placement

Inconsistent height

Poor rhythm timing

Stiff or collapsed leg technique

Most players do not have access to coaching for micro-corrections.

AltinhaAI provides structured, objective feedback using computer vision.

3. Goals (Hackathon Scope)
Primary Goals

Detect soccer ball trajectory from video

Detect leg landmarks using pose estimation

Identify touch events

Compute:

Power consistency

Lateral drift

Height consistency

Knee stiffness proxy

Generate 1–3 coaching suggestions

Display annotated video with overlay

Non-Goals (Out of Scope for MVP)

Real-time feedback

User accounts

Long-term progress storage

ML model training from scratch

Mobile native apps

4. User Flow
Step 1 – Setup

User sees:

“Choose Drill”

Camera preview

Start Button

Step 2 – Record

User presses “Start”

App records 10–20 seconds

User presses “Stop” OR auto-stops

Step 3 – Analyze

Show loading indicator

Backend runs CV pipeline

Generate:

Touch list

Metrics

Annotated video

Step 4 – Results

Display:

Annotated video

Streak count

Power consistency score

Height band performance

Drift statistics

Top 3 Coaching Tips

5. Drill System (MVP)
Drill 1 – Right Foot Pop Control

Goal: Drop ball → pop straight up consistently

Metrics:

Peak height in target band

Minimal lateral drift

Touch consistency

Drill 2 – Left Foot Pop Control

Same metrics, left foot only

Drill 3 – 5 Juggles Right Foot

Count valid touches

Height consistency threshold

Drill 4 – Alternating

Detect R/L sequence

Check alternation pattern

6. Technical Architecture
Frontend

Streamlit

streamlit-webrtc for recording

Display:

video player

metric cards

coaching text

optional charts

Backend (Python)

MediaPipe Pose

YOLO (ball detection)

OpenCV (video I/O + drawing overlays)

Optional: Kalman filter for smoothing

7. Computer Vision Pipeline
Step 1 – Frame Extraction

Input: recorded video
Process at 10–15 FPS

Step 2 – Pose Estimation

Extract:

Hip

Knee

Ankle

Foot

Compute:

Knee angle

Ankle distance to ball

Step 3 – Ball Detection

Use YOLO to detect bounding box center per frame.

Track ball trajectory over time.

Step 4 – Touch Detection

A touch occurs when:

Ball distance to foot < threshold

Ball velocity changes direction OR magnitude sharply

Store touch timestamps.

8. Metrics & Coaching Logic
A. Power

Compute vertical velocity immediately after touch.

If:

Peak height > upper threshold → “Too much power”

Peak height < lower threshold → “Too little power”

B. Drift (Contact Side)

Measure horizontal displacement 0.3–0.5s after touch.

If drift right:

“You struck the left side of the ball.”

If drift left:

“You struck the right side of the ball.”

C. Height Consistency

Compute standard deviation of peak heights.

High variance → “Work on consistent touch strength.”

D. Stiffness Proxy

Measure knee angle change during contact window.

If minimal change:

“Leg too stiff — allow slight knee bend.”

If excessive collapse:

“Stabilize knee for better control.”

9. Output Requirements

The analysis function must return:

{
  "touch_count": int,
  "streak": int,
  "avg_peak_height": float,
  "height_std": float,
  "avg_drift": float,
  "power_feedback": str,
  "drift_feedback": str,
  "form_feedback": str,
  "annotated_video_path": str
}
10. UI Requirements

Results page must show:

Annotated video

Drill name

Touch count

Best streak

3 coaching tips

“Try Again” button

11. Success Criteria (Hackathon)

Demo must:

Successfully detect ≥ 3 touches

Correctly identify drift direction

Generate believable coaching feedback

Play annotated overlay video

Not crash during demo

12. Stretch Goals

If time permits:

Live mode (limited real-time drill)

Skill rating score

Ball height zone overlay during playback

Replay with touch markers

Store session stats locally

13. Risks & Mitigation

Risk: Ball detection failure
Mitigation: Add OpenCV color fallback

Risk: Lighting issues
Mitigation: Ask user to use bright ball, plain background

Risk: Processing too slow
Mitigation: Downsample frames to 10 FPS

14. Hackathon Pitch Angle

AltinhaAI is:

A computer vision skill coach

Democratizes micro-feedback

Transforms solo practice into guided training

Scalable to all ball sports