from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import tempfile
import os
from datetime import datetime

from src.core.annotator import annotator
from src.config import settings

app = FastAPI(title="Omnivision-4D API", version="1.0.0")
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

@app.post("/annotate", response_model=dict)
async def create_annotation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = "Track robot movements",
    token: str = Depends(verify_token)
):
    """
    Submit video for 4D annotation.
    Returns immediately with job_id. Check /status/{job_id} for results.
    """
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + os.urandom(4).hex()
    
    # Save temp file
    temp_path = f"/tmp/{job_id}_{file.filename}"
    with open(temp_path, "wb") as buffer:
        content = await file.read()
        if len(content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        buffer.write(content)
    
    # Process synchronously for now (async queue later)
    result = annotator.process(temp_path, prompt, job_id)
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result

@app.get("/health")
async def health_check():
    """K8s/GCP health check"""
    return {
        "status": "healthy", 
        "gpu": torch.cuda.is_available() if 'torch' in globals() else False,
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {"message": "Omnivision-4D API", "docs": "/docs"}