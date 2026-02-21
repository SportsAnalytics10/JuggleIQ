# AltinhaAI Task List

Track progress for Hacklytics 2026. See [PRD.md](PRD.md) and [tech_stack.md](tech_stack.md) for details.

---

## Milestone 0: Skeleton (30–60 min)

**Goal:** Run the app and produce any output artifact.

- [ ] Create folders: `analysis/`, `outputs/`
- [ ] Streamlit app scaffold (`app.py`)
- [ ] Record/upload + save video to `outputs/input.mp4`
- [ ] `analyze_video()` stub: saves input, returns fake result, copies to `outputs/annotated.mp4`
- [ ] Results page renders output video + placeholder tips

**Demoable:** AltinhaAI records/uploads and returns results.

---

## Milestone 1: Ball (1–3 hrs)

**Goal:** Ball trajectory + overlay works.

- [ ] YOLO ball detection (or OpenCV fallback first)
- [ ] Draw ball center + trail onto frames
- [ ] Export `outputs/annotated.mp4`
- [ ] Show simple metric: "Ball detected % of frames"

**Demoable:** We track the ball and render an annotated replay.

---

## Milestone 2: Pose + Touch (2–5 hrs)

**Goal:** Detect touches and give ONE believable correction.

- [ ] MediaPipe Pose: extract ankles/knees
- [ ] Touch detection: ball close to ankle + velocity change
- [ ] Count touches + best streak
- [ ] Add one tip (power too high/low via peak height)

**Demoable:** We detect touches and coach power.

---

## Milestone 3: Drills + Polish (remaining time)

**Goal:** Product feel + progression.

- [ ] Drill selection (Right foot / Left foot / Alternating)
- [ ] Add drift-based tip (contact side)
- [ ] Add stiffness proxy tip (knee bend)
- [ ] Pretty results cards + "Try again" loop
- [ ] Add 2–3 pre-recorded demo videos for backup

**Demoable:** Progressive coaching app.

---

## What to Do First (Today)

Make the skeleton Streamlit app that can save a video and display an output video (even if it's just the same file).

**Only then add ball detection.**
