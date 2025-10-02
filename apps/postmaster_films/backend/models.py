"""SQLAlchemy models for Postmaster Films"""

from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, JSON, Text, Enum, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .db import Base
import enum

class SceneType(str, enum.Enum):
    HERO = "HERO"
    FILLER = "FILLER"

class EpisodeStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    RENDERING = "RENDERING" 
    ASSEMBLING = "ASSEMBLING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"

class JobStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"

class Project(Base):
    """Video project containing multiple episodes"""
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    client = Column(String, index=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    episodes = relationship("Episode", back_populates="project", cascade="all, delete-orphan")

class Episode(Base):
    """Individual episode within a project"""
    __tablename__ = "episodes"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    title = Column(String, index=True, nullable=False)
    budget_usd = Column(Float, default=50.0)
    veo_spend_usd = Column(Float, default=0.0)
    status = Column(Enum(EpisodeStatus), default=EpisodeStatus.DRAFT)
    script_text = Column(Text, nullable=True)
    final_video_path = Column(String, nullable=True)
    vo_audio_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    project = relationship("Project", back_populates="episodes")
    scenes = relationship("Scene", back_populates="episode", cascade="all, delete-orphan")

class Scene(Base):
    """Individual scene within an episode"""
    __tablename__ = "scenes"
    
    id = Column(Integer, primary_key=True, index=True)
    episode_id = Column(Integer, ForeignKey("episodes.id", ondelete="CASCADE"), nullable=False)
    index = Column(Integer, nullable=False)  # Order within episode
    description = Column(Text, nullable=False)
    duration_sec = Column(Integer, default=5)
    scene_type = Column(Enum(SceneType), default=SceneType.FILLER)
    prompt = Column(Text, nullable=True)  # Generated or custom prompt
    ref_image_path = Column(String, nullable=True)  # Reference image for generation
    model_route = Column(String, nullable=True)  # "veo" or "animdiff"
    output_video_path = Column(String, nullable=True)  # Generated video file
    cost_usd = Column(Float, default=0.0)  # Actual cost for this scene
    meta = Column(JSON, nullable=True)  # Additional metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    episode = relationship("Episode", back_populates="scenes")

class Asset(Base):
    """Reusable assets like reference frames, backgrounds, LoRAs"""
    __tablename__ = "assets"
    
    id = Column(Integer, primary_key=True, index=True)
    kind = Column(String, nullable=False)  # "ref_frame", "background", "lora", "style"
    label = Column(String, index=True, nullable=False)
    path = Column(String, nullable=False)
    meta = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Job(Base):
    """Background jobs for video processing"""
    __tablename__ = "jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    kind = Column(String, nullable=False)  # "render_scene", "assemble_episode", "tts", "extract_frame"
    payload = Column(JSON, nullable=False)  # Job parameters
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    result = Column(JSON, nullable=True)  # Job output/error info
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

