"""
AIMAP Data Models
Enhanced models for multi-sector AI adoption tracking
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, JSON, Text, DateTime
from sqlalchemy.orm import relationship

# Use existing db instance from main models
try:
    from ..models import db
except ImportError:
    from models import db

class Organisation(db.Model):
    """Enhanced organisation model for multi-sector tracking"""
    __tablename__ = 'organisations'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, index=True)
    sector = Column(String(100), nullable=False, default="Media", index=True)
    subsector = Column(String(100), nullable=True, index=True)
    region = Column(String(100), nullable=True, index=True)
    country = Column(String(100), nullable=True, index=True)
    size_band = Column(String(50), nullable=True, index=True)  # startup, small, medium, large, enterprise
    client_tag = Column(String(100), nullable=True, index=True)
    contact = Column(String(255), nullable=True)
    ai_tools = Column(JSON, nullable=True)  # List of detected AI tools
    notes = Column(Text, nullable=True)
    website_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    metrics = relationship("Metrics", back_populates="organisation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Organisation {self.name} ({self.sector})>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'sector': self.sector,
            'subsector': self.subsector,
            'region': self.region,
            'country': self.country,
            'size_band': self.size_band,
            'client_tag': self.client_tag,
            'contact': self.contact,
            'ai_tools': self.ai_tools or [],
            'notes': self.notes,
            'website_url': self.website_url,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Metrics(db.Model):
    """AI adoption metrics and scoring"""
    __tablename__ = 'metrics'
    
    id = Column(Integer, primary_key=True)
    organisation_id = Column(Integer, ForeignKey('organisations.id'), nullable=False)
    ai_adoption_score = Column(Float, nullable=True)  # 0-100 score
    maturity_stage = Column(String(50), nullable=True)  # Exploring, Piloting, Scaling, Optimizing
    signals = Column(JSON, nullable=True)  # Raw signals data
    benchmark_bucket = Column(String(200), nullable=True, index=True)  # sector:region:size_band
    period = Column(String(10), nullable=False, index=True)  # YYYY-MM format
    source_tag = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    organisation = relationship("Organisation", back_populates="metrics")
    
    def __repr__(self):
        return f'<Metrics {self.organisation_id} {self.period} score={self.ai_adoption_score}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'organisation_id': self.organisation_id,
            'ai_adoption_score': self.ai_adoption_score,
            'maturity_stage': self.maturity_stage,
            'signals': self.signals or {},
            'benchmark_bucket': self.benchmark_bucket,
            'period': self.period,
            'source_tag': self.source_tag,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
