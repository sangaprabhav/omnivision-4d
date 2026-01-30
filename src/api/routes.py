"""
API Routes for Omnivision-4D
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, FileResponse
import tempfile
import os
import json
from datetime import datetime
from typing import Optional
import asyncio

from src.core.annotator import annotator
from src.config import settings

router = APIRouter()

# In-memory job store (replace with Redis in production)
jobs_db = {}

@router.post("/jobs", response_model=dict)
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: Optional[str] = Form("Track robot movements and output 4D trajectories"),
    webhook_url: Optional[str] = Form(None)
):
    """
    Submit a video for 4D annotation.
    Returns job_id immediately. Poll GET /jobs/{job_id} for results.
    """
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
    
    # Validate file
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be video (MP4, MOV, AVI)")
    
    # Check file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > 500:
        raise HTTPException(413, f"File too large: {size_mb:.1f}MB (max 500MB)")
    
    # Save to temp
    temp_path = f"/tmp/{job_id}_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)
    
    # Create job record
    jobs_db[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "filename": file.filename,
        "size_mb": round(size_mb, 2),
        "created_at": datetime.utcnow().isoformat(),
        "prompt": prompt,
        "webhook_url": webhook_url,
        "result": None,
        "error": None
    }
    
    # Process in background (or sync for now)
    async def process_job():
        try:
            jobs_db[job_id]["status"] = "processing"
            result = annotator.process(temp_path, prompt, job_id)
            
            if result["status"] == "success":
                jobs_db[job_id]["status"] = "completed"
                jobs_db[job_id]["result"] = result
                jobs_db[job_id]["completed_at"] = datetime.utcnow().isoformat()
                
                # TODO: Send webhook if webhook_url provided
                if webhook_url:
                    pass  # Implement webhook call
                
            else:
                jobs_db[job_id]["status"] = "failed"
                jobs_db[job_id]["error"] = result.get("error", "Unknown error")
                
        except Exception as e:
            jobs_db[job_id]["status"] = "failed"
            jobs_db[job_id]["error"] = str(e)
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Run immediately (for Colab/single-worker) or use background tasks
    if settings.ENV == "development":
        await process_job()
    else:
        background_tasks.add_task(process_job)
    
    return {
        "job_id": job_id,
        "status": jobs_db[job_id]["status"],
        "message": "Job created. Poll GET /jobs/{job_id} for results."
    }

@router.get("/jobs/{job_id}", response_model=dict)
async def get_job_status(job_id: str):
    """Get job status and results"""
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    job = jobs_db[job_id].copy()
    
    # Don't return full result in list view (too large)
    if job.get("result") and "objects" in job["result"]:
        job["result_summary"] = {
            "total_objects": len(job["result"]["objects"]),
            "processing_time": job["result"]["annotation_metadata"]["fusion_timestamp"]
        }
        del job["result"]  # Full result available via download endpoint
    
    return job

@router.get("/jobs/{job_id}/download")
async def download_result(job_id: str):
    """Download full JSON result"""
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    job = jobs_db[job_id]
    if job["status"] != "completed":
        raise HTTPException(400, f"Job not completed (status: {job['status']})")
    
    # Save to temp file for download
    output_path = f"/tmp/{job_id}_result.json"
    with open(output_path, "w") as f:
        json.dump(job["result"], f, indent=2)
    
    return FileResponse(
        output_path,
        media_type="application/json",
        filename=f"omnivision_4d_{job_id}.json"
    )

@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a queued/processing job"""
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    if jobs_db[job_id]["status"] in ["completed", "failed"]:
        raise HTTPException(400, "Cannot cancel completed/failed job")
    
    jobs_db[job_id]["status"] = "cancelled"
    return {"message": "Job cancelled", "job_id": job_id}

@router.get("/jobs")
async def list_jobs(limit: int = 10, offset: int = 0):
    """List recent jobs"""
    jobs_list = list(jobs_db.values())
    jobs_list.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "total": len(jobs_list),
        "jobs": jobs_list[offset:offset+limit]
    }