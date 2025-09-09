"""
AIMAP Data Models (Flask SQLAlchemy)
Uses the application's shared db instance so queries work via .query
"""
from datetime import datetime
import enum

try:
    # Prefer absolute import when running via module
    from backend.models import db as _app_db
except Exception:  # pragma: no cover - fallback for direct script execution
    from models import db as _app_db

# Re-export db so other modules can import from aimap.models
db = _app_db


class Organisation(db.Model):
    __tablename__ = "organisations"
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, unique=True, index=True)
    sector = db.Column(db.String, index=True)
    subsector = db.Column(db.String, index=True)
    region = db.Column(db.String, index=True)
    country = db.Column(db.String, index=True)
    size_band = db.Column(db.String, index=True)
    client_tag = db.Column(db.String, index=True)
    contact = db.Column(db.String)
    ai_tools = db.Column(db.JSON)
    notes = db.Column(db.Text)
    website_url = db.Column(db.String)
    tags = db.Column(db.String)  # Comma-separated tags
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    metrics = db.relationship("Metrics", back_populates="organisation", cascade="all, delete-orphan")
    # people = db.relationship("Person", back_populates="organisation", cascade="all, delete-orphan")  # Disabled - using people_management table
    leads = db.relationship("Lead", back_populates="organisation", cascade="all, delete-orphan")
    research_reports = db.relationship("ResearchReport", back_populates="organisation", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "sector": self.sector,
            "subsector": self.subsector,
            "region": self.region,
            "country": self.country,
            "size_band": self.size_band,
            "client_tag": self.client_tag,
            "contact": self.contact,
            "ai_tools": self.ai_tools,
            "notes": self.notes,
            "website_url": self.website_url,
            "tags": self.tags.split(',') if self.tags else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Metrics(db.Model):
    __tablename__ = "metrics"
    
    id = db.Column(db.Integer, primary_key=True)
    organisation_id = db.Column(db.Integer, db.ForeignKey("organisations.id", ondelete="CASCADE"))
    ai_adoption_score = db.Column(db.Float)
    maturity_stage = db.Column(db.String)
    signals = db.Column(db.JSON)
    benchmark_bucket = db.Column(db.String, index=True)
    period = db.Column(db.String, index=True)  # YYYY-MM format
    source_tag = db.Column(db.String)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    organisation = db.relationship("Organisation", back_populates="metrics")

    def to_dict(self):
        return {
            "id": self.id,
            "organisation_id": self.organisation_id,
            "ai_adoption_score": self.ai_adoption_score,
            "maturity_stage": self.maturity_stage,
            "signals": self.signals,
            "benchmark_bucket": self.benchmark_bucket,
            "period": self.period,
            "source_tag": self.source_tag,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# class Person(db.Model):  # DISABLED - Using PersonManagement in backend.models instead
#     """People/Contacts associated with organizations"""
#     __tablename__ = "people"
#     
#     id = db.Column(db.Integer, primary_key=True)
#     organisation_id = db.Column(db.Integer, db.ForeignKey("organisations.id", ondelete="CASCADE"))
#     first_name = db.Column(db.String)
#     last_name = db.Column(db.String)
#     email = db.Column(db.String, index=True)
#     phone = db.Column(db.String)
#     title = db.Column(db.String)
#     department = db.Column(db.String)
#     role = db.Column(db.String)  # Decision maker, influencer, user, etc.
#     linkedin_url = db.Column(db.String)
#     notes = db.Column(db.Text)
#     is_primary_contact = db.Column(db.Boolean, default=False)
#     created_at = db.Column(db.DateTime, default=datetime.utcnow)
#     updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
#     
#     # Relationships
#     organisation = db.relationship("Organisation", back_populates="people")
#     interactions = db.relationship("Interaction", back_populates="person", cascade="all, delete-orphan")


class LeadStatus(str, enum.Enum):
    PROSPECT = "Prospect"
    CONTACTED = "Contacted"
    QUALIFIED = "Qualified"
    PROPOSAL = "Proposal"
    NEGOTIATION = "Negotiation"
    CLOSED_WON = "Closed Won"
    CLOSED_LOST = "Closed Lost"


class Lead(db.Model):
    """Lead/Prospect management"""
    __tablename__ = "leads"
    
    id = db.Column(db.Integer, primary_key=True)
    organisation_id = db.Column(db.Integer, db.ForeignKey("organisations.id", ondelete="CASCADE"))
    status = db.Column(db.String, default=LeadStatus.PROSPECT)
    source = db.Column(db.String)  # Website, referral, cold outreach, etc.
    priority = db.Column(db.String)  # High, Medium, Low
    estimated_value = db.Column(db.Float)
    probability = db.Column(db.Float)  # 0-100%
    expected_close_date = db.Column(db.Date)
    assigned_to = db.Column(db.String)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    organisation = db.relationship("Organisation", back_populates="leads")
    activities = db.relationship("LeadActivity", back_populates="lead", cascade="all, delete-orphan")


class LeadActivity(db.Model):
    """Activities and interactions with leads"""
    __tablename__ = "lead_activities"
    
    id = db.Column(db.Integer, primary_key=True)
    lead_id = db.Column(db.Integer, db.ForeignKey("leads.id", ondelete="CASCADE"))
    activity_type = db.Column(db.String)  # Call, email, meeting, proposal, etc.
    description = db.Column(db.Text)
    outcome = db.Column(db.String)
    next_action = db.Column(db.String)
    scheduled_date = db.Column(db.DateTime)
    completed_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    lead = db.relationship("Lead", back_populates="activities")


class Interaction(db.Model):
    """General interactions with people"""
    __tablename__ = "interactions"
    
    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey("people.id", ondelete="CASCADE"))
    interaction_type = db.Column(db.String)  # Call, email, meeting, etc.
    subject = db.Column(db.String)
    notes = db.Column(db.Text)
    outcome = db.Column(db.String)
    follow_up_required = db.Column(db.Boolean, default=False)
    follow_up_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    # person = db.relationship("Person", back_populates="interactions")  # Disabled - using people_management table


class ResearchReport(db.Model):
    """Research reports and documents"""
    __tablename__ = "research_reports"
    
    id = db.Column(db.Integer, primary_key=True)
    organisation_id = db.Column(db.Integer, db.ForeignKey("organisations.id", ondelete="CASCADE"))
    title = db.Column(db.String)
    description = db.Column(db.Text)
    report_type = db.Column(db.String)  # Industry analysis, case study, white paper, etc.
    file_path = db.Column(db.String)
    file_size = db.Column(db.Integer)
    file_type = db.Column(db.String)
    tags = db.Column(db.JSON)  # Array of tags
    ai_insights = db.Column(db.JSON)  # AI-extracted insights
    summary = db.Column(db.Text)
    author = db.Column(db.String)
    publication_date = db.Column(db.Date)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    organisation = db.relationship("Organisation", back_populates="research_reports")


class CustomData(db.Model):
    """Flexible custom data storage"""
    __tablename__ = "custom_data"
    
    id = db.Column(db.Integer, primary_key=True)
    data_type = db.Column(db.String, index=True)  # e.g., 'competitor', 'market_trend', 'tool_review'
    title = db.Column(db.String)
    content = db.Column(db.JSON)  # Flexible JSON storage
    tags = db.Column(db.JSON)
    custom_metadata = db.Column(db.JSON)  # Additional metadata (renamed to avoid conflict)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ConsultingProject(db.Model):
    """Consulting projects and engagements"""
    __tablename__ = "consulting_projects"
    
    id = db.Column(db.Integer, primary_key=True)
    organisation_id = db.Column(db.Integer, db.ForeignKey("organisations.id", ondelete="CASCADE"))
    project_name = db.Column(db.String)
    project_type = db.Column(db.String)  # Strategy, implementation, assessment, etc.
    status = db.Column(db.String)  # Planning, Active, Completed, On Hold
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    budget = db.Column(db.Float)
    actual_cost = db.Column(db.Float)
    description = db.Column(db.Text)
    objectives = db.Column(db.JSON)
    deliverables = db.Column(db.JSON)
    team_members = db.Column(db.JSON)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    organisation = db.relationship("Organisation")
    milestones = db.relationship("ProjectMilestone", back_populates="project", cascade="all, delete-orphan")


class ProjectMilestone(db.Model):
    """Project milestones and deliverables"""
    __tablename__ = "project_milestones"
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("consulting_projects.id", ondelete="CASCADE"))
    milestone_name = db.Column(db.String)
    description = db.Column(db.Text)
    due_date = db.Column(db.Date)
    completed_date = db.Column(db.Date)
    status = db.Column(db.String)  # Not Started, In Progress, Completed, Delayed
    deliverables = db.Column(db.JSON)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    project = db.relationship("ConsultingProject", back_populates="milestones")
