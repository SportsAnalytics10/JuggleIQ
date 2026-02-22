# JuggleIQ — Dev Post & Project Overview

**Event:** Hacklytics 2026  
**Tagline:** CV-powered soccer juggling coach — upload, analyze, improve.

---

## 1. Project Overview

**JuggleIQ** (AltinhaAI) is a **computer vision–powered soccer juggling coach**. Users upload a short video of themselves juggling; the system runs ball detection, pose estimation, and touch detection, then returns **actionable metrics and coaching tips**. The frontend presents results with charts and a **Figma Make–ready** export so users can build custom animated dashboards from their session data.

### Problem

Learning to juggle (altinha) is hard without feedback. Players struggle with:

- Overpowering or weak touches  
- Poor contact placement and lateral drift  
- Inconsistent rhythm and height  
- Stiff or collapsed leg technique  

Most players don’t have access to coaching for these micro-corrections.

### Solution

- **Backend:** Python CV pipeline (YOLO ball detection, MediaPipe Pose, Kalman filter, touch logic) processes uploaded video and returns a rich JSON result plus an optional annotated video.  
- **Frontend:** React app for upload, results (metrics, charts, coaching tips), and **Download session JSON** for use in Figma Make or other tools.  
- **Figma Make:** Session JSON is designed as a small (&lt;5MB) dataset so users can build **unique animated dashboards** (gauges, touch timeline, foot donut, rhythm chart, etc.) in Figma Make.

---

## 2. What We Built

### Backend (Python / Kaggle)

- **FastAPI** app hosted on **Kaggle** (exposed via **ngrok**).
- **Endpoints:**
  - `POST /analyze` — upload video (mp4, mov, avi, mkv, webm); returns full analysis JSON and `annotated_video_url`.
  - `GET /download/{job_id}` — returns the annotated MP4 (inline so it can play in browser).
  - `GET /health` — health check.
- **CV pipeline (`video_processor.py`):**
  - YOLO (Ultralytics) for soccer ball detection.
  - MediaPipe Pose for leg/hip/knee/ankle/foot landmarks.
  - Kalman filter + smoothing for ball trajectory.
  - Touch detection (velocity flip, proximity to feet, rise confirmation).
  - Left/right foot identification, knee angle (stiffness proxy), peak height, drift, rhythm intervals.
  - Output: annotated video (ball trail, pose skeleton, touch markers, live touch count, L/R foot chart) + full metrics and coaching tips.

### Frontend (React / Vite)

- **Single-page app** with:
  - **Sample videos** — drill videos from `drills/` (or `public/samples/`) for quick tryout.
  - **Upload & Analyze** — file picker, POST to `/analyze`, loading state, auto-scroll to results.
  - **Results:**
    - **Download session JSON** — full API response plus `annotated_video_full_url` for dashboards.
    - **Metric cards** — touch count, best streak, skill score, rhythm score, avg drift, knee stiffness.
    - **Coaching tips** — bullet list from `coaching_tips`.
    - **Knee tip** — from `knee_feedback.stiffness_tip`.
    - **Charts (Recharts):** foot usage (pie), rhythm intervals (line), score breakdown (radar).
    - **Touch timeline** — table of `touches[]` (foot, time, frame, x/y).
  - **Figma Make block** — CTA to open Figma Make dashboard + **animation ideas** derived from the API response (tap touch, timeline, score gauge, foot donut, rhythm chart, stiffness badge, coaching tips).
  - **Resources** — YouTube links, Figma Make dashboard link.

- **Config:** API base URL in `frontend/src/config.js` (default: ngrok URL; overridable via `VITE_API_BASE`).

---

## 3. Tech Stack

| Layer      | Technologies |
|-----------|--------------|
| Backend   | Python 3.10+, FastAPI, OpenCV, Ultralytics YOLO, MediaPipe, NumPy |
| Frontend  | React 19, Vite 7, Recharts |
| Hosting   | Kaggle (backend), ngrok (public URL), static frontend (e.g. Vercel/Netlify) |
| Dataset   | Session JSON (&lt;5MB) for Figma Make |

---

## 4. API Response Shape (Summary)

The `/analyze` response includes (among others):

- `touch_count`, `left_foot`, `right_foot`, `unknown`, `left_pct`, `right_pct`
- `best_streak`, `all_streaks`
- `skill_score`, `rhythm_score`, `avg_peak_px`, `avg_drift_px`
- `coaching_tips[]`, `score_breakdown` (touch_score, rhythm_score, peak_score, drift_score)
- `knee_feedback` (avg_left_knee_angle, avg_right_knee_angle, stiffness_label, stiffness_tip)
- `intervals[]` (time between touches in seconds)
- `touches[]` (touch_num, foot, t, frame_idx, x, y)
- `job_id`, `original_filename`, `annotated_video_url`

The frontend adds `annotated_video_full_url` when downloading the session JSON.

---

## 5. Figma Make Implementation

We designed the flow so users can **skip the in-app video** and instead build a **unique animated dashboard** in Figma Make using the same API result.

### 5.1 Dataset

- **Source:** “Download session JSON” in the app (or any client that calls `POST /analyze` and saves the response).
- **Format:** Single JSON object per session (one row / one “record” in Figma Make terms, or you can treat the whole object as the dataset).
- **Size:** Well under 5MB (Figma Make limit).
- **Fields useful for animations:**  
  `touch_count`, `skill_score`, `rhythm_score`, `left_foot`, `right_foot`, `best_streak`, `coaching_tips`, `knee_feedback`, `score_breakdown`, `intervals`, `touches`, `annotated_video_full_url`.

### 5.2 Workflow

1. User uploads video in the JuggleIQ app and clicks **Analyze**.
2. User clicks **Download session JSON** and gets one JSON file per session.
3. In **Figma Make**, create a new project and **import this JSON as the dataset** (or paste/link the structure).
4. Build frames and components that **bind to dataset fields** and add animations (see below).
5. When the user **taps the Figma dashboard**, they see their analysis with **unique animations** (gauges, timelines, tips, etc.) instead of a generic static view.

### 5.3 Animation Ideas (from API response)

These are implemented as **suggestions in the app** and can be built in Figma Make:

| Idea | Data used | Implementation hint |
|------|-----------|----------------------|
| **Tap a touch** | `touches[]` (foot, t, x, y) | On tap, reveal foot (L/R), time, and coordinates with a highlight or pulse. |
| **Touch timeline** | `touches[]` | Staggered reveal of touches (e.g. dots or steps) with labels; one-by-one or scroll. |
| **Skill score gauge** | `skill_score` | Animate a needle or bar from 0 to `skill_score` when the frame loads. |
| **Foot usage donut** | `left_foot`, `right_foot` | Build the donut segments with a short draw/animate-in. |
| **Rhythm chart** | `intervals[]` | Animate a line or bar chart that “draws” in sequence. |
| **Knee stiffness badge** | `knee_feedback.stiffness_label`, `stiffness_tip` | Show label + tip with a subtle entrance animation. |
| **Coaching tips** | `coaching_tips[]` | Cycle or reveal tips with type-on or fade-in. |
| **Video link** | `annotated_video_full_url` | “Watch analysis video” button that opens the URL in browser. |

### 5.4 Why This Is a Good Use of Figma Make

- **Small, structured dataset** — one JSON per session, easy to import and bind.
- **Rich but flat fields** — numbers, short strings, and arrays map well to text, progress, and lists in Figma.
- **Clear “story”** — touches, scores, and tips give a natural flow for animations (e.g. score → tips → timeline).
- **Sponsor alignment** — uses Figma Make as the **presentation layer** for hackathon deliverables (animated analysis dashboard).

---

## 6. How to Run

### Backend (Kaggle + ngrok)

- Run the FastAPI app in a Kaggle notebook/environment where `video_processor.py` and models (e.g. YOLO, pose landmarker) are available.
- Expose the server with ngrok and set the frontend `API_BASE` (or `VITE_API_BASE`) to that ngrok URL.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

- Set `VITE_API_BASE` in `.env` if the API is not at the default ngrok URL.
- For production: `npm run build` and deploy the `dist/` folder (e.g. Vercel, Netlify).

### Figma Make

- No local run: use the exported session JSON in Figma Make as described in **Section 5**.

---

## 7. Repo Structure (high level)

```
JuggleIQ/
├── cv/
│   ├── main.py              # FastAPI app, /analyze, /download, /health
│   └── video_processor.py   # YOLO, MediaPipe, touch detection, metrics
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Upload, results, charts, Figma Make block
│   │   ├── config.js        # API_BASE, getAnnotatedVideoUrl
│   │   └── App.css
│   └── public/samples/      # Sample drill videos
├── drills/                  # Source drill videos
└── docs/
    ├── PRD.md
    ├── tech_stack.md
    └── DEVPOST_OVERVIEW.md  # This file
```

---

## 8. Summary

We built an **end-to-end juggling analysis flow**: upload video → CV pipeline → JSON + optional annotated video → **React results page** with metrics, charts, and coaching tips, plus **one-click export** of session JSON for **Figma Make**. The Figma Make implementation gives a clear path to **unique animated dashboards** (gauges, touch timeline, foot donut, rhythm chart, stiffness badge, coaching tips) so users see their analysis in a custom, animated way when they tap the Figma dashboard.

---

*JuggleIQ — Hacklytics 2026*
