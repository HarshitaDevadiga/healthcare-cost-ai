from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import schemas
from ..database import Job, Patient, get_db

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=schemas.JobOut)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found.")
    return job


@router.get("/{job_id}/results", response_model=list[schemas.PatientOut])
def get_job_results(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found.")
    patients = (
        db.query(Patient)
        .filter(Patient.source == job_id)
        .order_by(Patient.predicted_cost.desc())
        .all()
    )
    return patients


@router.get("", response_model=list[schemas.JobOut])
def list_jobs(db: Session = Depends(get_db)):
    return db.query(Job).order_by(Job.created_at.desc()).limit(20).all()
