import sys
sys.path.insert(0, "/kaggle/working")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import uuid
from pathlib import Path
from video_processor import VideoProcessor

app = FastAPI(title="JuggleIQ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading models — this takes ~30 seconds...")
vp = VideoProcessor(conf=0.45, model_name="yolov8m.pt")
print("Models ready!")

BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "videos"
WORK_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "JuggleIQ API is running"
    }


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Upload one video file.
    Returns full analysis JSON:
      - touch_count, left_foot, right_foot, best_streak
      - skill_score, rhythm_score, avg_peak_px, avg_drift_px
      - coaching_tips, knee_feedback, score_breakdown
      - per-touch detail list
      - annotated_video_url to download the processed video
    """
    # Validate file type
    allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    suffix  = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}"
        )

    # Create unique job
    job_id      = str(uuid.uuid4())[:8]
    input_path  = WORK_DIR / f"{job_id}_input{suffix}"
    output_path = WORK_DIR / f"{job_id}_output.mp4"

    # Save uploaded video to disk
    try:
        with open(input_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    print(f"[{job_id}] Processing: {file.filename}")

    try:
        # Run full video analysis
        results = vp.process_video(str(input_path), str(output_path))

        # Add job metadata
        results["job_id"]              = job_id
        results["original_filename"]   = file.filename
        results["annotated_video_url"] = f"/download/{job_id}"

        print(f"[{job_id}] Done — {results['touch_count']} touches detected")
        return results

    except Exception as e:
        print(f"[{job_id}] ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    finally:
        # Always clean up raw input file
        if input_path.exists():
            input_path.unlink()


@app.get("/download/{job_id}")
def download_annotated_video(job_id: str):
    """
    Download the annotated output video by job_id.
    The job_id comes from the /analyze response.
    """
    output_path = WORK_DIR / f"{job_id}_output.mp4"
    if not output_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Video not found for job_id: {job_id}"
        )
    return FileResponse(
        str(output_path),
        media_type="video/mp4",
        filename=f"juggleiq_{job_id}_annotated.mp4",
        content_disposition_type="inline",
    )


@app.delete("/cleanup/{job_id}")
def cleanup_video(job_id: str):
    """
    Delete the annotated output video after frontend has downloaded it.
    Call this after downloading to save disk space.
    """
    output_path = WORK_DIR / f"{job_id}_output.mp4"
    if output_path.exists():
        output_path.unlink()
        return {"status": "deleted", "job_id": job_id}
    return {"status": "not_found", "job_id": job_id}
