# main.py
import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx
import uuid
import logging
from datetime import datetime
import uvicorn
from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "YOUR_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "fswz4ju3asche1")
RUNPOD_API_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./runpod_jobs.db")

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class JobRecord(Base):
    __tablename__ = "job_records"
    
    id = Column(String, primary_key=True, index=True)
    status = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    prompt = Column(Text)
    result = Column(Text, nullable=True)

Base.metadata.create_all(bind=engine)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
class RunpodRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to send to RunPod")

class RunpodResponse(BaseModel):
    id: str
    status: str

class WebhookPayload(BaseModel):
    id: str
    status: str
    output: Optional[Dict[str, Any]] = None

# Initialize FastAPI
app = FastAPI(
    title="RunPod API Integration",
    description="A FastAPI service for interfacing with RunPod AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background task to submit job to RunPod
async def submit_runpod_job(job_id: str, prompt: str, db: Session):
    try:
        webhook_url = f"{os.getenv('PUBLIC_URL', 'https://your-api-domain.com')}/webhook"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {RUNPOD_API_KEY}'
        }
        
        data = {
            'input': {"prompt": prompt},
            'webhook': webhook_url
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(RUNPOD_API_URL, headers=headers, json=data, timeout=30.0)
            
            if response.status_code == 200:
                response_data = response.json()
                runpod_job_id = response_data.get('id')
                
                # Update the job record with RunPod's job ID and status
                db_job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
                if db_job:
                    db_job.runpod_id = runpod_job_id
                    db_job.status = "SUBMITTED"
                    db.commit()
                
                logger.info(f"Job {job_id} submitted to RunPod with ID {runpod_job_id}")
            else:
                # Update job status to failed
                db_job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
                if db_job:
                    db_job.status = "FAILED"
                    db_job.result = f"Failed to submit to RunPod: {response.text}"
                    db.commit()
                
                logger.error(f"Failed to submit job {job_id} to RunPod: {response.text}")
                
    except Exception as e:
        # Update job status to failed
        db_job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if db_job:
            db_job.status = "FAILED"
            db_job.result = f"Error submitting to RunPod: {str(e)}"
            db.commit()
            
        logger.error(f"Error submitting job {job_id} to RunPod: {str(e)}")

@app.post("/generate", response_model=RunpodResponse)
async def generate(
    request: RunpodRequest, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    # Generate a unique ID for this job
    job_id = str(uuid.uuid4())
    
    # Create a new job record
    new_job = JobRecord(
        id=job_id,
        status="PENDING",
        prompt=request.prompt
    )
    db.add(new_job)
    db.commit()
    
    # Submit the job to RunPod asynchronously
    background_tasks.add_task(submit_runpod_job, job_id, request.prompt, db)
    
    return RunpodResponse(id=job_id, status="PENDING")

@app.get("/job/{job_id}")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    result = {
        "id": job.id,
        "status": job.status,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }
    
    if job.result:
        result["result"] = job.result
    
    return result

@app.post("/webhook")
async def webhook_handler(payload: WebhookPayload, db: Session = Depends(get_db)):
    logger.info(f"Received webhook callback for job: {payload.id}")
    
    # Find the job by RunPod's job ID
    job = db.query(JobRecord).filter(JobRecord.runpod_id == payload.id).first()
    
    if not job:
        logger.warning(f"Received webhook for unknown job ID: {payload.id}")
        return JSONResponse(content={"message": "Job not found"}, status_code=404)
    
    # Update job status and result
    job.status = payload.status
    
    if payload.output:
        job.result = str(payload.output)
    
    db.commit()
    
    return {"message": "Webhook processed successfully"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# For local development
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)