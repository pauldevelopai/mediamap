"""Scene management API routes"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from ..db import get_db
from ..models import Scene, Episode
from ..schemas import SceneCreate, SceneOut, SceneUpdate

router = APIRouter()

@router.get("/", response_model=List[SceneOut])
def list_scenes(episode_id: int = None, db: Session = Depends(get_db)):
    """List all scenes, optionally filtered by episode"""
    query = db.query(Scene)
    if episode_id:
        query = query.filter(Scene.episode_id == episode_id)
    scenes = query.order_by(Scene.episode_id, Scene.index).all()
    return scenes

@router.post("/", response_model=SceneOut)
def create_scene(scene: SceneCreate, db: Session = Depends(get_db)):
    """Create a new scene"""
    # Verify episode exists
    episode = db.query(Episode).filter(Episode.id == scene.episode_id).first()
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    db_scene = Scene(**scene.dict())
    db.add(db_scene)
    db.commit()
    db.refresh(db_scene)
    return db_scene

@router.get("/{scene_id}", response_model=SceneOut)
def get_scene(scene_id: int, db: Session = Depends(get_db)):
    """Get a specific scene"""
    scene = db.query(Scene).filter(Scene.id == scene_id).first()
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    return scene

@router.put("/{scene_id}", response_model=SceneOut)
def update_scene(scene_id: int, scene_update: SceneUpdate, db: Session = Depends(get_db)):
    """Update a scene"""
    scene = db.query(Scene).filter(Scene.id == scene_id).first()
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    # Update only provided fields
    update_data = scene_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(scene, field, value)
    
    db.commit()
    db.refresh(scene)
    return scene

@router.delete("/{scene_id}")
def delete_scene(scene_id: int, db: Session = Depends(get_db)):
    """Delete a scene"""
    scene = db.query(Scene).filter(Scene.id == scene_id).first()
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    db.delete(scene)
    db.commit()
    return {"message": "Scene deleted successfully"}

@router.post("/{scene_id}/reorder")
def reorder_scene(scene_id: int, new_index: int, db: Session = Depends(get_db)):
    """Reorder a scene within its episode"""
    scene = db.query(Scene).filter(Scene.id == scene_id).first()
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    old_index = scene.index
    episode_id = scene.episode_id
    
    # Get all scenes in the episode
    scenes = db.query(Scene).filter(Scene.episode_id == episode_id).order_by(Scene.index).all()
    
    # Validate new index
    if new_index < 0 or new_index >= len(scenes):
        raise HTTPException(status_code=400, detail="Invalid new index")
    
    if old_index == new_index:
        return {"message": "Scene already at target position"}
    
    # Reorder scenes
    if old_index < new_index:
        # Moving forward: shift scenes backward
        for s in scenes:
            if old_index < s.index <= new_index:
                s.index -= 1
    else:
        # Moving backward: shift scenes forward
        for s in scenes:
            if new_index <= s.index < old_index:
                s.index += 1
    
    # Set new index for target scene
    scene.index = new_index
    
    db.commit()
    
    return {"message": f"Scene moved from index {old_index} to {new_index}"}

@router.post("/{scene_id}/duplicate", response_model=SceneOut)
def duplicate_scene(scene_id: int, db: Session = Depends(get_db)):
    """Duplicate a scene"""
    original = db.query(Scene).filter(Scene.id == scene_id).first()
    if not original:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    # Get next available index in episode
    max_index = db.query(Scene).filter(Scene.episode_id == original.episode_id).count()
    
    # Create duplicate
    duplicate = Scene(
        episode_id=original.episode_id,
        index=max_index,
        description=f"Copy of: {original.description}",
        duration_sec=original.duration_sec,
        scene_type=original.scene_type,
        prompt=original.prompt,
        ref_image_path=original.ref_image_path,
        meta=original.meta.copy() if original.meta else None
    )
    
    db.add(duplicate)
    db.commit()
    db.refresh(duplicate)
    
    return duplicate

