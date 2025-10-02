"""Job management and processing API routes"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import json

from ..db import get_db
from ..models import Job, Episode, JobStatus
from ..schemas import JobCreate, JobOut
from ..services.jobs import (
    process_render_episode_job,
    process_assemble_episode_job, 
    process_voiceover_job
)

router = APIRouter()

# Try to import RQ for job queue, fall back to synchronous processing
try:
    from rq import Queue
    import redis
    from ..settings import get_settings
    
    settings = get_settings()
    redis_conn = redis.from_url(settings.REDIS_URL)
    job_queue = Queue(connection=redis_conn)
    RQ_AVAILABLE = True
except:
    RQ_AVAILABLE = False
    job_queue = None

@router.get("/", response_model=List[JobOut])
def list_jobs(status: JobStatus = None, kind: str = None, db: Session = Depends(get_db)):
    """List all jobs, optionally filtered by status or kind"""
    query = db.query(Job)
    if status:
        query = query.filter(Job.status == status)
    if kind:
        query = query.filter(Job.kind == kind)
    jobs = query.order_by(Job.created_at.desc()).all()
    return jobs

@router.post("/", response_model=JobOut)
def create_job(job: JobCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Create and optionally queue a new job"""
    db_job = Job(**job.dict())
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    
    # Process job asynchronously
    if RQ_AVAILABLE and job_queue:
        # Queue job with RQ
        rq_job = job_queue.enqueue(
            _process_job_wrapper,
            db_job.id,
            job_timeout='30m'
        )
        # Store RQ job ID in metadata
        if not db_job.result:
            db_job.result = {}
        db_job.result["rq_job_id"] = rq_job.id
        db.commit()
    else:
        # Process synchronously in background
        background_tasks.add_task(_process_job_sync, db_job.id)
    
    return db_job

@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: int, db: Session = Depends(get_db)):
    """Get a specific job"""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.post("/{job_id}/retry", response_model=JobOut)
def retry_job(job_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Retry a failed job"""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status not in [JobStatus.FAILED, JobStatus.COMPLETE]:
        raise HTTPException(status_code=400, detail="Job can only be retried if failed or complete")
    
    # Reset job status
    job.status = JobStatus.PENDING
    job.started_at = None
    job.completed_at = None
    if job.result:
        job.result["retry_count"] = job.result.get("retry_count", 0) + 1
    
    db.commit()
    
    # Process job again
    if RQ_AVAILABLE and job_queue:
        rq_job = job_queue.enqueue(_process_job_wrapper, job.id, job_timeout='30m')
        if not job.result:
            job.result = {}
        job.result["rq_job_id"] = rq_job.id
        db.commit()
    else:
        background_tasks.add_task(_process_job_sync, job.id)
    
    return job

# Convenience endpoints for common job types

@router.post("/render_episode/{episode_id}")
def queue_render_episode(episode_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Queue a job to render all scenes in an episode"""
    # Verify episode exists
    episode = db.query(Episode).filter(Episode.id == episode_id).first()
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    # Create job
    job = Job(
        kind="render_episode",
        payload={"episode_id": episode_id}
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Process job
    if RQ_AVAILABLE and job_queue:
        rq_job = job_queue.enqueue(_process_job_wrapper, job.id, job_timeout='60m')
        if not job.result:
            job.result = {}
        job.result["rq_job_id"] = rq_job.id
        db.commit()
    else:
        background_tasks.add_task(_process_job_sync, job.id)
    
    return {"job_id": job.id, "message": "Episode rendering job queued"}

@router.post("/assemble/{episode_id}")
def queue_assemble_episode(episode_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Queue a job to assemble an episode from rendered scenes"""
    episode = db.query(Episode).filter(Episode.id == episode_id).first()
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    job = Job(
        kind="assemble_episode",
        payload={"episode_id": episode_id}
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    if RQ_AVAILABLE and job_queue:
        rq_job = job_queue.enqueue(_process_job_wrapper, job.id, job_timeout='30m')
        if not job.result:
            job.result = {}
        job.result["rq_job_id"] = rq_job.id
        db.commit()
    else:
        background_tasks.add_task(_process_job_sync, job.id)
    
    return {"job_id": job.id, "message": "Episode assembly job queued"}

@router.post("/mux_vo/{episode_id}")
def queue_voiceover_mux(episode_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Queue a job to add voiceover to an assembled episode"""
    episode = db.query(Episode).filter(Episode.id == episode_id).first()
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    job = Job(
        kind="voiceover_mux",
        payload={"episode_id": episode_id}
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    if RQ_AVAILABLE and job_queue:
        rq_job = job_queue.enqueue(_process_job_wrapper, job.id, job_timeout='30m')
        if not job.result:
            job.result = {}
        job.result["rq_job_id"] = rq_job.id
        db.commit()
    else:
        background_tasks.add_task(_process_job_sync, job.id)
    
    return {"job_id": job.id, "message": "Voiceover mux job queued"}

# Job processing functions

def _process_job_wrapper(job_id: int):
    """Wrapper for RQ job processing"""
    from ..db import SessionLocal
    
    db = SessionLocal()
    try:
        return _process_job_sync(job_id, db)
    finally:
        db.close()

def _process_job_sync(job_id: int, db: Session = None):
    """Process a job synchronously"""
    if db is None:
        from ..db import SessionLocal
        db = SessionLocal()
        should_close = True
    else:
        should_close = False
    
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return {"error": "Job not found"}
        
        job.status = JobStatus.PROCESSING
        from datetime import datetime
        job.started_at = datetime.utcnow()
        db.commit()
        
        try:
            # Process based on job kind
            if job.kind == "render_episode":
                result = process_render_episode_job(job.payload["episode_id"], db)
            elif job.kind == "assemble_episode":
                result = process_assemble_episode_job(job.payload["episode_id"], db)
            elif job.kind == "voiceover_mux":
                result = process_voiceover_job(job.payload["episode_id"], db)
            else:
                result = {"success": False, "error": f"Unknown job kind: {job.kind}"}
            
            # Update job with result
            job.result = result
            job.status = JobStatus.COMPLETE if result.get("success") else JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            db.commit()
            
            return result
            
        except Exception as e:
            job.result = {"success": False, "error": str(e)}
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            db.commit()
            return {"success": False, "error": str(e)}
    
    finally:
        if should_close:
            db.close()

