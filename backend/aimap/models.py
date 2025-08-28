"""
AIMAP Data Models
Extended models for comprehensive data management
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, JSON, Text, Enum, DateTime, Date
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import enum

Base = declarative_base()

class Organisation(Base):
    __tablename__ = "organisations"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True)
    sector = Column(String, index=True)
    subsector = Column(String, index=True)
    region = Column(String, index=True)
    country = Column(String, index=True)
    size_band = Column(String, index=True)
    client_tag = Column(String, index=True)
    contact = Column(String)
    ai_tools = Column(JSON)
    notes = Column(Text)
    website_url = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    metrics = relationship("Metrics", back_populates="organisation")
    people = relationship("Person", back_populates="organisation")
    leads = relationship("Lead", back_populates="organisation")
    research_reports = relationship("ResearchReport", back_populates="organisation")

class Metrics(Base):
    __tablename__ = "metrics"
    
    id = Column(Integer, primary_key=True)
    organisation_id = Column(Integer, ForeignKey("organisations.id", ondelete="CASCADE"))
    ai_adoption_score = Column(Float)
    maturity_stage = Column(String)
    signals = Column(JSON)
    benchmark_bucket = Column(String, index=True)
    period = Column(String, index=True)  # YYYY-MM format
    source_tag = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    organisation = relationship("Organisation", back_populates="metrics")

class Person(Base):
    """People/Contacts associated with organizations"""
    __tablename__ = "people"
    
    id = Column(Integer, primary_key=True)
    organisation_id = Column(Integer, ForeignKey("organisations.id", ondelete="CASCADE"))
    first_name = Column(String)
    last_name = Column(String)
    email = Column(String, index=True)
    phone = Column(String)
    title = Column(String)
    department = Column(String)
    role = Column(String)  # Decision maker, influencer, user, etc.
    linkedin_url = Column(String)
    notes = Column(Text)
    is_primary_contact = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    organisation = relationship("Organisation", back_populates="people")
    interactions = relationship("Interaction", back_populates="person")

class LeadStatus(str, enum.Enum):
    PROSPECT = "Prospect"
    CONTACTED = "Contacted"
    QUALIFIED = "Qualified"
    PROPOSAL = "Proposal"
    NEGOTIATION = "Negotiation"
    CLOSED_WON = "Closed Won"
    CLOSED_LOST = "Closed Lost"

class Lead(Base):
    """Lead/Prospect management"""
    __tablename__ = "leads"
    
    id = Column(Integer, primary_key=True)
    organisation_id = Column(Integer, ForeignKey("organisations.id", ondelete="CASCADE"))
    status = Column(String, default=LeadStatus.PROSPECT)
    source = Column(String)  # Website, referral, cold outreach, etc.
    priority = Column(String)  # High, Medium, Low
    estimated_value = Column(Float)
    probability = Column(Float)  # 0-100%
    expected_close_date = Column(Date)
    assigned_to = Column(String)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    organisation = relationship("Organisation", back_populates="leads")
    activities = relationship("LeadActivity", back_populates="lead")

class LeadActivity(Base):
    """Activities and interactions with leads"""
    __tablename__ = "lead_activities"
    
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer, ForeignKey("leads.id", ondelete="CASCADE"))
    activity_type = Column(String)  # Call, email, meeting, proposal, etc.
    description = Column(Text)
    outcome = Column(String)
    next_action = Column(String)
    scheduled_date = Column(DateTime)
    completed_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    lead = relationship("Lead", back_populates="activities")

class Interaction(Base):
    """General interactions with people"""
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey("people.id", ondelete="CASCADE"))
    interaction_type = Column(String)  # Call, email, meeting, etc.
    subject = Column(String)
    notes = Column(Text)
    outcome = Column(String)
    follow_up_required = Column(Boolean, default=False)
    follow_up_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    person = relationship("Person", back_populates="interactions")

class ResearchReport(Base):
    """Research reports and documents"""
    __tablename__ = "research_reports"
    
    id = Column(Integer, primary_key=True)
    organisation_id = Column(Integer, ForeignKey("organisations.id", ondelete="CASCADE"))
    title = Column(String)
    description = Column(Text)
    report_type = Column(String)  # Industry analysis, case study, white paper, etc.
    file_path = Column(String)
    file_size = Column(Integer)
    file_type = Column(String)
    tags = Column(JSON)  # Array of tags
    ai_insights = Column(JSON)  # AI-extracted insights
    summary = Column(Text)
    author = Column(String)
    publication_date = Column(Date)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    organisation = relationship("Organisation", back_populates="research_reports")

class CustomData(Base):
    """Flexible custom data storage"""
    __tablename__ = "custom_data"
    
    id = Column(Integer, primary_key=True)
    data_type = Column(String, index=True)  # e.g., 'competitor', 'market_trend', 'tool_review'
    title = Column(String)
    content = Column(JSON)  # Flexible JSON storage
    tags = Column(JSON)
    metadata = Column(JSON)  # Additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ConsultingProject(Base):
    """Consulting projects and engagements"""
    __tablename__ = "consulting_projects"
    
    id = Column(Integer, primary_key=True)
    organisation_id = Column(Integer, ForeignKey("organisations.id", ondelete="CASCADE"))
    project_name = Column(String)
    project_type = Column(String)  # Strategy, implementation, assessment, etc.
    status = Column(String)  # Planning, Active, Completed, On Hold
    start_date = Column(Date)
    end_date = Column(Date)
    budget = Column(Float)
    actual_cost = Column(Float)
    description = Column(Text)
    objectives = Column(JSON)
    deliverables = Column(JSON)
    team_members = Column(JSON)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    organisation = relationship("Organisation")
    milestones = relationship("ProjectMilestone", back_populates="project")

class ProjectMilestone(Base):
    """Project milestones and deliverables"""
    __tablename__ = "project_milestones"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("consulting_projects.id", ondelete="CASCADE"))
    milestone_name = Column(String)
    description = Column(Text)
    due_date = Column(Date)
    completed_date = Column(Date)
    status = Column(String)  # Not Started, In Progress, Completed, Delayed
    deliverables = Column(JSON)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship("ConsultingProject", back_populates="milestones")
