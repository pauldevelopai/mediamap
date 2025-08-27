"""Episode management API routes"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from ..db import get_db
from ..models import Episode, Scene, Project, EpisodeStatus
from ..schemas import EpisodeCreate, EpisodeCreateFromScript, EpisodeOut, EpisodeUpdate, BudgetInfo
from ..services.shotlist import script_to_scenes
from ..router_budget import get_budget_info

router = APIRouter()

@router.get("/", response_model=List[EpisodeOut])
def list_episodes(project_id: int = None, db: Session = Depends(get_db)):
    """List all episodes, optionally filtered by project"""
    query = db.query(Episode)
    if project_id:
        query = query.filter(Episode.project_id == project_id)
    episodes = query.all()
    return episodes

@router.post("/", response_model=EpisodeOut)
def create_episode(episode: EpisodeCreate, db: Session = Depends(get_db)):
    """Create a new episode"""
    # Verify project exists
    project = db.query(Project).filter(Project.id == episode.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Create episode
    episode_data = episode.dict()
    scenes_data = episode_data.pop("scenes", [])
    
    db_episode = Episode(**episode_data)
    db.add(db_episode)
    db.flush()  # Get episode ID
    
    # Create scenes
    for scene_data in scenes_data:
        scene_data["episode_id"] = db_episode.id
        db_scene = Scene(**scene_data)
        db.add(db_scene)
    
    db.commit()
    db.refresh(db_episode)
    return db_episode

@router.post("/from_script", response_model=EpisodeOut)
def create_episode_from_script(episode: EpisodeCreateFromScript, db: Session = Depends(get_db)):
    """Create episode by auto-generating scenes from script"""
    # Verify project exists
    project = db.query(Project).filter(Project.id == episode.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Create episode
    db_episode = Episode(
        project_id=episode.project_id,
        title=episode.title,
        budget_usd=episode.budget_usd,
        script_text=episode.script_text
    )
    db.add(db_episode)
    db.flush()  # Get episode ID
    
    # Generate scenes from script
    scenes_data = script_to_scenes(episode.script_text)
    
    for scene_data in scenes_data:
        db_scene = Scene(
            episode_id=db_episode.id,
            index=scene_data["index"],
            description=scene_data["description"],
            duration_sec=scene_data["duration_sec"],
            scene_type=scene_data["scene_type"]
        )
        db.add(db_scene)
    
    db.commit()
    db.refresh(db_episode)
    return db_episode

@router.get("/{episode_id}", response_model=EpisodeOut)
def get_episode(episode_id: int, db: Session = Depends(get_db)):
    """Get a specific episode with all scenes"""
    episode = db.query(Episode).filter(Episode.id == episode_id).first()
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode

@router.put("/{episode_id}", response_model=EpisodeOut)
def update_episode(episode_id: int, episode_update: EpisodeUpdate, db: Session = Depends(get_db)):
    """Update an episode"""
    episode = db.query(Episode).filter(Episode.id == episode_id).first()
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    # Update only provided fields
    update_data = episode_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(episode, field, value)
    
    db.commit()
    db.refresh(episode)
    return episode

@router.delete("/{episode_id}")
def delete_episode(episode_id: int, db: Session = Depends(get_db)):
    """Delete an episode and all its scenes"""
    episode = db.query(Episode).filter(Episode.id == episode_id).first()
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    db.delete(episode)
    db.commit()
    return {"message": "Episode deleted successfully"}

@router.get("/{episode_id}/budget", response_model=BudgetInfo)
def get_episode_budget(episode_id: int, db: Session = Depends(get_db)):
    """Get budget information for an episode"""
    episode = db.query(Episode).filter(Episode.id == episode_id).first()
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    budget_info = get_budget_info(episode.budget_usd, episode.veo_spend_usd)
    return budget_info

@router.post("/{episode_id}/regenerate_scenes")
def regenerate_scenes_from_script(episode_id: int, db: Session = Depends(get_db)):
    """Regenerate scenes from the episode's script"""
    episode = db.query(Episode).filter(Episode.id == episode_id).first()
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    if not episode.script_text:
        raise HTTPException(status_code=400, detail="Episode has no script text")
    
    # Delete existing scenes
    db.query(Scene).filter(Scene.episode_id == episode_id).delete()
    
    # Generate new scenes
    scenes_data = script_to_scenes(episode.script_text)
    
    for scene_data in scenes_data:
        db_scene = Scene(
            episode_id=episode_id,
            index=scene_data["index"],
            description=scene_data["description"],
            duration_sec=scene_data["duration_sec"],
            scene_type=scene_data["scene_type"]
        )
        db.add(db_scene)
    
    # Reset episode status
    episode.status = EpisodeStatus.DRAFT
    episode.veo_spend_usd = 0.0
    episode.final_video_path = None
    
    db.commit()
    
    return {"message": f"Regenerated {len(scenes_data)} scenes from script"}

