"""Pydantic schemas for Postmaster Films API"""

from pydantic import BaseModel, validator
from typing import Optional, List, Any, Dict
from datetime import datetime
from .models import SceneType, EpisodeStatus, JobStatus

# Base schemas
class ProjectBase(BaseModel):
    name: str
    client: Optional[str] = None
    notes: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass

class ProjectOut(ProjectBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Scene schemas
class SceneBase(BaseModel):
    index: int
    description: str
    duration_sec: int = 5
    scene_type: SceneType = SceneType.FILLER
    prompt: Optional[str] = None
    ref_image_path: Optional[str] = None

class SceneCreate(SceneBase):
    episode_id: int

class SceneUpdate(BaseModel):
    description: Optional[str] = None
    duration_sec: Optional[int] = None
    scene_type: Optional[SceneType] = None
    prompt: Optional[str] = None
    ref_image_path: Optional[str] = None

class SceneOut(SceneBase):
    id: int
    episode_id: int
    model_route: Optional[str] = None
    output_video_path: Optional[str] = None
    cost_usd: float = 0.0
    meta: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Episode schemas
class EpisodeBase(BaseModel):
    title: str
    budget_usd: float = 50.0
    script_text: Optional[str] = None

class EpisodeCreate(EpisodeBase):
    project_id: int
    scenes: List[SceneBase] = []

class EpisodeCreateFromScript(BaseModel):
    project_id: int
    title: str
    budget_usd: float = 50.0
    script_text: str

class EpisodeUpdate(BaseModel):
    title: Optional[str] = None
    budget_usd: Optional[float] = None
    script_text: Optional[str] = None
    status: Optional[EpisodeStatus] = None

class EpisodeOut(EpisodeBase):
    id: int
    project_id: int
    veo_spend_usd: float = 0.0
    status: EpisodeStatus
    final_video_path: Optional[str] = None
    vo_audio_path: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    scenes: List[SceneOut] = []
    
    class Config:
        from_attributes = True

# Asset schemas
class AssetBase(BaseModel):
    kind: str
    label: str
    path: str
    meta: Optional[Dict[str, Any]] = None

class AssetCreate(AssetBase):
    pass

class AssetOut(AssetBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Job schemas
class JobBase(BaseModel):
    kind: str
    payload: Dict[str, Any]

class JobCreate(JobBase):
    pass

class JobOut(JobBase):
    id: int
    status: JobStatus
    result: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

# Response schemas
class BudgetInfo(BaseModel):
    total_budget_usd: float
    veo_spend_usd: float
    remaining_budget_usd: float
    veo_seconds_available: int

class RenderProgress(BaseModel):
    episode_id: int
    total_scenes: int
    completed_scenes: int
    current_scene: Optional[int] = None
    estimated_completion: Optional[datetime] = None
