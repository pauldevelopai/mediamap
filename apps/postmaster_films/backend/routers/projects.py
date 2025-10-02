"""Project management API routes"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from ..db import get_db
from ..models import Project
from ..schemas import ProjectCreate, ProjectOut

router = APIRouter()

@router.get("/", response_model=List[ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    """List all projects"""
    projects = db.query(Project).all()
    return projects

@router.post("/", response_model=ProjectOut)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    """Create a new project"""
    # Check if project name already exists
    existing = db.query(Project).filter(Project.name == project.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Project name already exists")
    
    db_project = Project(**project.dict())
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: int, db: Session = Depends(get_db)):
    """Get a specific project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@router.put("/{project_id}", response_model=ProjectOut)
def update_project(project_id: int, project_update: ProjectCreate, db: Session = Depends(get_db)):
    """Update a project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Check if new name conflicts with existing project
    if project_update.name != project.name:
        existing = db.query(Project).filter(Project.name == project_update.name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Project name already exists")
    
    for field, value in project_update.dict().items():
        setattr(project, field, value)
    
    db.commit()
    db.refresh(project)
    return project

@router.delete("/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    """Delete a project and all its episodes"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db.delete(project)
    db.commit()
    return {"message": "Project deleted successfully"}

