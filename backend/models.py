from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    # Add location fields
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    location_name = db.Column(db.String(200))
    
    # Relationships
    analyses = db.relationship('MediaAnalysis', backref='user', lazy=True)
    @property
    def chats(self):
        from models import Chat
        return Chat.query.filter_by(user_id=self.id).all()

class MediaAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    media_url = db.Column(db.String(500), nullable=False)
    analysis_result = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

class Chat(db.Model):
    __tablename__ = 'chats'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    title = db.Column(db.String(255), nullable=True)  # Optional title based on content
    fact_sheet = db.Column(db.Text, nullable=True)  # Stores extracted company info
    strategies = db.Column(db.Text, nullable=True)  # Stores generated strategies
    
    # Relationships
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'title': self.title,
            'fact_sheet': self.fact_sheet,
            'strategies': self.strategies,
            'messages': [message.to_dict() for message in self.messages]
        }

class Message(db.Model):
    __tablename__ = 'messages'
    
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chats.id'), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # 'user', 'assistant', 'system'
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    chat = relationship("Chat", back_populates="messages")
    
    def to_dict(self):
        return {
            'id': self.id,
            'role': self.role,
            'content': self.content,
            'created_at': self.created_at.isoformat()
        }

class Lesson(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    order = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class UserLesson(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    lesson_id = db.Column(db.Integer, db.ForeignKey('lesson.id'), nullable=False)
    completed = db.Column(db.Boolean, default=False)
    last_accessed = db.Column(db.DateTime, default=datetime.utcnow)

class OrganizationInfo(db.Model):
    __tablename__ = 'organization_info'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # Make it user-specific
    org_info = db.Column(db.Text)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

class OrganizationFact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    fact = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Translation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    original_text = db.Column(db.Text, nullable=False)
    translated_text = db.Column(db.Text, nullable=False)
    source_language = db.Column(db.String(10))
    target_language = db.Column(db.String(10))
    rating = db.Column(db.Integer)  # Store user rating 1-10
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class TranslationFeedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    translation_id = db.Column(db.Integer, db.ForeignKey('translation.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    corrected_text = db.Column(db.Text, nullable=False)
    source_language = db.Column(db.String(10))
    target_language = db.Column(db.String(10))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Location(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'

    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.String(50), nullable=False)
    recipient_id = db.Column(db.String(50), nullable=False)
    message_text = db.Column(db.String(2000), nullable=False)
    timestamp = db.Column(db.DateTime, server_default=func.now())
    is_user_message = db.Column(db.Boolean, default=True)

    def __repr__(self):
        return f"<ChatMessage(message_text='{self.message_text}')>"

class LoginEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    login_time = db.Column(db.DateTime, nullable=False, default=datetime.now)
    method = db.Column(db.String(20), nullable=False)  # password, 2fa, etc.
    success = db.Column(db.Boolean, default=True)
    failure_reason = db.Column(db.String(255), nullable=True)
    
    user = db.relationship('User', backref=db.backref('login_events', lazy=True))

class Feedback(db.Model):
    __tablename__ = 'feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    username = db.Column(db.String(80), nullable=False)  # Store username for reference
    feedback_type = db.Column(db.String(20), nullable=False)  # bug, feature, improvement, general
    subject = db.Column(db.String(255), nullable=False)
    message = db.Column(db.Text, nullable=False)
    allow_followup = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='new')  # new, reviewed, in_progress, resolved
    admin_notes = db.Column(db.Text, nullable=True)  # For admin to add notes
    
    # Relationship
    user = db.relationship('User', backref=db.backref('feedback_submissions', lazy=True)) 

class NotionIntegration(db.Model):
    """Model for Notion integration settings"""
    id = db.Column(db.Integer, primary_key=True)
    notion_token = db.Column(db.String(500), nullable=False)
    workspace_id = db.Column(db.String(100), nullable=False)
    database_id = db.Column(db.String(100), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<NotionIntegration {self.workspace_id}>'

class News(db.Model):
    """Model for storing user's personalized news"""
    __tablename__ = 'news'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text, nullable=True)
    url = db.Column(db.String(1000), nullable=False)
    source_name = db.Column(db.String(200), nullable=True)
    published_at = db.Column(db.DateTime, nullable=True)
    search_terms = db.Column(db.Text, nullable=True)  # Store the search terms used
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('news_articles', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'url': self.url,
            'source': {'name': self.source_name},
            'publishedAt': self.published_at.isoformat() if self.published_at else None,
            'search_terms': self.search_terms,
            'created_at': self.created_at.isoformat()
        }
    
    def __repr__(self):
        return f'<News {self.title[:50]}...>'

class SavedStrategy(db.Model):
    """Model for storing user's saved AI strategies"""
    __tablename__ = 'saved_strategies'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100), nullable=True)  # e.g., 'content_creation', 'automation', 'analytics'
    priority = db.Column(db.String(20), default='medium')  # low, medium, high
    status = db.Column(db.String(20), default='draft')  # draft, active, completed, archived
    notes = db.Column(db.Text, nullable=True)  # User notes about the strategy
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('saved_strategies', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'category': self.category,
            'priority': self.priority,
            'status': self.status,
            'notes': self.notes,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    def __repr__(self):
        return f'<SavedStrategy {self.title[:50]}...>'

class SavedNews(db.Model):
    """Model for storing user's saved news articles"""
    __tablename__ = 'saved_news'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text, nullable=True)
    url = db.Column(db.String(1000), nullable=False)
    source_name = db.Column(db.String(200), nullable=True)
    published_at = db.Column(db.DateTime, nullable=True)
    notes = db.Column(db.Text, nullable=True)  # User notes about the article
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('saved_news', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'url': self.url,
            'source_name': self.source_name,
            'published_at': self.published_at.isoformat() if self.published_at else None,
            'notes': self.notes,
            'created_at': self.created_at.isoformat()
        }
    
    def __repr__(self):
        return f'<SavedNews {self.title[:50]}...>' 


# ---- Crawling models ----
class CrawlSource(db.Model):
    __tablename__ = 'crawl_sources'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    url = db.Column(db.String(1000), nullable=False)
    source_type = db.Column(db.String(50), nullable=False, default='website')  # website, rss, newsletter
    description = db.Column(db.Text, nullable=True)
    crawl_frequency = db.Column(db.String(50), nullable=False, default='daily')
    config = db.Column(db.Text, nullable=True)  # JSON string for custom config
    is_active = db.Column(db.Boolean, default=True)
    last_crawled = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    jobs = db.relationship('CrawlJob', backref='source', lazy=True)
    content_items = db.relationship('CrawledContent', backref='source', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'url': self.url,
            'source_type': self.source_type,
            'description': self.description,
            'crawl_frequency': self.crawl_frequency,
            'config': self.config,
            'is_active': self.is_active,
            'last_crawled': self.last_crawled.isoformat() if self.last_crawled else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f'<CrawlSource {self.name}>'


class CrawlJob(db.Model):
    __tablename__ = 'crawl_jobs'

    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.Integer, db.ForeignKey('crawl_sources.id'), nullable=False)
    status = db.Column(db.String(50), default='pending')  # pending, running, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    items_found = db.Column(db.Integer, default=0)
    items_processed = db.Column(db.Integer, default=0)
    error_message = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f'<CrawlJob {self.id} status={self.status}>'


class CrawledContent(db.Model):
    __tablename__ = 'crawled_content'

    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.Integer, db.ForeignKey('crawl_sources.id'), nullable=False)
    title = db.Column(db.String(500), nullable=False)
    content = db.Column(db.Text, nullable=False)
    url = db.Column(db.String(1000), nullable=True)
    published_date = db.Column(db.DateTime, nullable=True)
    content_type = db.Column(db.String(50), default='article')  # strategy, use_case, code_example, article
    tags = db.Column(db.Text, nullable=True)  # JSON array as string
    summary = db.Column(db.Text, nullable=True)
    relevance_score = db.Column(db.Float, default=0.0)
    is_processed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<CrawledContent {self.title[:50]}...>'


# ---- Implementation planning & reporting ----
class ImplementationPlan(db.Model):
    __tablename__ = 'implementation_plans'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    summary = db.Column(db.Text, nullable=True)
    objectives = db.Column(db.Text, nullable=True)  # JSON or markdown
    tasks = db.Column(db.Text, nullable=True)       # JSON or markdown
    status = db.Column(db.String(50), default='draft')  # draft, active, completed, archived
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('implementation_plans', lazy=True))

    def __repr__(self):
        return f'<ImplementationPlan {self.title[:50]}...>'


class DailyReport(db.Model):
    __tablename__ = 'daily_reports'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    plan_id = db.Column(db.Integer, db.ForeignKey('implementation_plans.id'), nullable=True)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    content = db.Column(db.Text, nullable=False)
    progress = db.Column(db.Text, nullable=True)
    blockers = db.Column(db.Text, nullable=True)
    next_steps = db.Column(db.Text, nullable=True)
    metrics = db.Column(db.Text, nullable=True)  # JSON as string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('daily_reports', lazy=True))
    plan = db.relationship('ImplementationPlan', backref=db.backref('reports', lazy=True))

    def __repr__(self):
        return f'<DailyReport {self.id} {self.date}>'


class CheatSheet(db.Model):
    __tablename__ = 'cheatsheets'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100), nullable=True)
    tags = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('cheatsheets', lazy=True))

    def __repr__(self):
        return f'<CheatSheet {self.title[:50]}...>'

class UserSession(db.Model):
    """Track admin user sessions and memory access"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_token = db.Column(db.String(255), unique=True, nullable=False)
    login_time = db.Column(db.DateTime, default=datetime.utcnow)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    
    # Memory and access control
    accessible_memories = db.Column(db.JSON)  # List of memory IDs this session can access
    access_level = db.Column(db.String(50), default='full')  # full, limited, read_only
    session_notes = db.Column(db.Text)  # Notes about this session's purpose
    
    user = db.relationship('User', backref='sessions')

class Memory(db.Model):
    """Store session-specific memories and information"""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('user_session.id'), nullable=False)
    memory_type = db.Column(db.String(50))  # conversation, analysis, strategy, insight
    content = db.Column(db.Text)
    memory_metadata = db.Column(db.JSON)  # Additional context
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    accessed_at = db.Column(db.DateTime, default=datetime.utcnow)
    importance_score = db.Column(db.Float, default=0.0)  # 0-1 score for memory importance
    
    session = db.relationship('UserSession', backref='memories')

class Client(db.Model):
    """Consulting clients and businesses"""
    __tablename__ = 'client'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    website = db.Column(db.String(200))
    industry = db.Column(db.String(100))
    status = db.Column(db.String(50))  # Active, Proposal, Inactive, Completed
    engagement_type = db.Column(db.String(200))
    notes = db.Column(db.Text)
    last_contact = db.Column(db.DateTime)
    contact_person = db.Column(db.String(200))
    email = db.Column(db.String(200))
    phone = db.Column(db.String(50))
    tags = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Newsroom(db.Model):
    """Media organizations and newsrooms"""
    __tablename__ = 'newsroom'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    website = db.Column(db.String(200))
    type = db.Column(db.String(50))  # National, Regional, Digital-First, International
    location = db.Column(db.String(200))
    ai_readiness = db.Column(db.String(50))  # High, Medium, Low
    last_analysis = db.Column(db.DateTime)
    notes = db.Column(db.Text)
    client_id = db.Column(db.Integer, db.ForeignKey('client.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to client
    client = db.relationship('Client', backref='newsrooms')

class AIPrototype(db.Model):
    """AI prototypes being developed by newsrooms"""
    __tablename__ = 'ai_prototypes'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    newsroom_id = db.Column(db.Integer, db.ForeignKey('newsroom.id'), nullable=True)
    newsroom_name = db.Column(db.String(200))  # Denormalized for performance
    
    # Prototype details
    category = db.Column(db.String(100))  # Content Generation, Fact-Checking, Personalization, Analytics, etc.
    technology_stack = db.Column(db.String(500))  # AI models, frameworks used
    stage = db.Column(db.String(50))  # Ideation, Development, Testing, Production, Completed
    
    # Progress tracking
    progress_percentage = db.Column(db.Integer, default=0)
    start_date = db.Column(db.DateTime)
    target_completion = db.Column(db.DateTime)
    actual_completion = db.Column(db.DateTime)
    
    # Metrics
    success_metrics = db.Column(db.Text)  # JSON string of metrics
    current_results = db.Column(db.Text)  # Current performance data
    challenges = db.Column(db.Text)  # Current challenges faced
    
    # Collaboration
    team_size = db.Column(db.Integer)
    external_partners = db.Column(db.String(500))  # External collaborators
    budget = db.Column(db.Float)  # Budget allocated
    
    # Status and notes
    status = db.Column(db.String(50), default='Active')  # Active, Paused, Completed, Cancelled
    notes = db.Column(db.Text)
    lessons_learned = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to newsroom
    newsroom = db.relationship('Newsroom', backref='ai_prototypes')

class PrototypeUpdate(db.Model):
    """Updates and progress reports for AI prototypes"""
    __tablename__ = 'prototype_updates'
    
    id = db.Column(db.Integer, primary_key=True)
    prototype_id = db.Column(db.Integer, db.ForeignKey('ai_prototypes.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text)
    update_type = db.Column(db.String(50))  # Progress, Milestone, Challenge, Success, Lesson
    
    # Progress data
    progress_percentage = db.Column(db.Integer)
    metrics_data = db.Column(db.Text)  # JSON string of metrics
    
    # Media and attachments
    attachments = db.Column(db.Text)  # JSON string of file paths/URLs
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.String(200))  # Who created the update
    
    # Relationship
    prototype = db.relationship('AIPrototype', backref='updates')

class ResearchProject(db.Model):
    """Research projects and studies"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    category = db.Column(db.String(100))  # AI Trends, Media Analysis, Market Research, Case Studies
    status = db.Column(db.String(50))  # In Progress, Completed, On Hold
    start_date = db.Column(db.DateTime)
    end_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DailyInsight(db.Model):
    """Daily insights and analysis"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text)
    category = db.Column(db.String(100))  # AI, Media, Tech, Industry
    source = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class HighlanderChat(db.Model):
    """Highlander AI chat conversations"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_id = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    context = db.Column(db.Text)  # JSON string for additional context
    category = db.Column(db.String(100))  # Client Analysis, Business Strategy, etc.
    processed = db.Column(db.Boolean, default=False)  # For later processing
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class NewsroomImplementationExperience(db.Model):
    """Track newsroom experiences with AIMAP implementations"""
    __tablename__ = 'newsroom_implementation_experiences'
    
    id = db.Column(db.Integer, primary_key=True)
    newsroom_id = db.Column(db.Integer, db.ForeignKey('newsroom.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Implementation Details
    implementation_type = db.Column(db.String(100), nullable=False)  # AI Strategy, Content Analysis, Workflow Optimization, etc.
    implementation_date = db.Column(db.Date, nullable=False)
    implementation_duration_weeks = db.Column(db.Integer)
    
    # Experience Details
    experience_summary = db.Column(db.Text, nullable=False)
    challenges_faced = db.Column(db.Text)
    solutions_found = db.Column(db.Text)
    outcomes_achieved = db.Column(db.Text)
    
    # Success Metrics
    success_rating = db.Column(db.Integer)  # 1-5 scale
    time_saved_hours_per_week = db.Column(db.Float)
    cost_savings_percentage = db.Column(db.Float)
    quality_improvement_rating = db.Column(db.Integer)  # 1-5 scale
    
    # Recommendations
    would_recommend = db.Column(db.Boolean, default=True)
    recommendations_for_others = db.Column(db.Text)
    suggestions_for_improvement = db.Column(db.Text)
    
    # Status
    status = db.Column(db.String(50), default='Submitted')  # Submitted, Reviewed, Followed Up
    admin_notes = db.Column(db.Text)
    follow_up_required = db.Column(db.Boolean, default=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    newsroom = db.relationship('Newsroom', backref='implementation_experiences')
    user = db.relationship('User', backref='implementation_experiences')
    
    def to_dict(self):
        return {
            'id': self.id,
            'newsroom_id': self.newsroom_id,
            'newsroom_name': self.newsroom.name if self.newsroom else None,
            'user_id': self.user_id,
            'user_name': self.user.username if self.user else None,
            'implementation_type': self.implementation_type,
            'implementation_date': self.implementation_date.isoformat() if self.implementation_date else None,
            'implementation_duration_weeks': self.implementation_duration_weeks,
            'experience_summary': self.experience_summary,
            'challenges_faced': self.challenges_faced,
            'solutions_found': self.solutions_found,
            'outcomes_achieved': self.outcomes_achieved,
            'success_rating': self.success_rating,
            'time_saved_hours_per_week': self.time_saved_hours_per_week,
            'cost_savings_percentage': self.cost_savings_percentage,
            'quality_improvement_rating': self.quality_improvement_rating,
            'would_recommend': self.would_recommend,
            'recommendations_for_others': self.recommendations_for_others,
            'suggestions_for_improvement': self.suggestions_for_improvement,
            'status': self.status,
            'admin_notes': self.admin_notes,
            'follow_up_required': self.follow_up_required,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class ImplementationChatSession(db.Model):
    """Track chat sessions specifically for implementation experience sharing"""
    __tablename__ = 'implementation_chat_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    newsroom_id = db.Column(db.Integer, db.ForeignKey('newsroom.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_id = db.Column(db.String(100), nullable=False)
    
    # Session Details
    session_type = db.Column(db.String(100), default='Implementation Experience')  # Implementation Experience, Follow-up, etc.
    implementation_experience_id = db.Column(db.Integer, db.ForeignKey('newsroom_implementation_experiences.id'), nullable=True)
    
    # Chat Content
    messages = db.relationship('ImplementationChatMessage', back_populates='session', cascade='all, delete-orphan')
    
    # Status
    status = db.Column(db.String(50), default='Active')  # Active, Completed, Archived
    admin_reviewed = db.Column(db.Boolean, default=False)
    admin_notes = db.Column(db.Text)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    newsroom = db.relationship('Newsroom', backref='implementation_chat_sessions')
    user = db.relationship('User', backref='implementation_chat_sessions')
    implementation_experience = db.relationship('NewsroomImplementationExperience', backref='chat_sessions')
    
    def to_dict(self):
        return {
            'id': self.id,
            'newsroom_id': self.newsroom_id,
            'newsroom_name': self.newsroom.name if self.newsroom else None,
            'user_id': self.user_id,
            'user_name': self.user.username if self.user else None,
            'session_id': self.session_id,
            'session_type': self.session_type,
            'implementation_experience_id': self.implementation_experience_id,
            'status': self.status,
            'admin_reviewed': self.admin_reviewed,
            'admin_notes': self.admin_notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'message_count': len(self.messages)
        }

class ImplementationChatMessage(db.Model):
    """Individual messages in implementation experience chat sessions"""
    __tablename__ = 'implementation_chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('implementation_chat_sessions.id'), nullable=False)
    
    # Message Details
    sender_type = db.Column(db.String(50), nullable=False)  # user, ai, system
    message_content = db.Column(db.Text, nullable=False)
    message_type = db.Column(db.String(50), default='text')  # text, experience_form, follow_up_question
    
    # Context
    context_data = db.Column(db.Text)  # JSON string for additional context
    related_experience_id = db.Column(db.Integer, db.ForeignKey('newsroom_implementation_experiences.id'), nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    session = db.relationship('ImplementationChatSession', back_populates='messages')
    related_experience = db.relationship('NewsroomImplementationExperience', backref='chat_messages')
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'sender_type': self.sender_type,
            'message_content': self.message_content,
            'message_type': self.message_type,
            'context_data': self.context_data,
            'related_experience_id': self.related_experience_id,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class TrainingWorkshop(db.Model):
    """Training workshops conducted for newsrooms"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    category = db.Column(db.String(100))  # AI Basics, Advanced AI, Implementation, Strategy
    duration_hours = db.Column(db.Float, default=1.0)
    max_participants = db.Column(db.Integer)
    materials_url = db.Column(db.String(500))  # Link to training materials
    notes = db.Column(db.Text)  # Additional notes about the training
    status = db.Column(db.String(50), default='Scheduled')  # Scheduled, Completed, Cancelled
    scheduled_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    attendees = db.relationship('TrainingAttendance', back_populates='workshop', cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'category': self.category,
            'duration_hours': self.duration_hours,
            'max_participants': self.max_participants,
            'materials_url': self.materials_url,
            'notes': self.notes,
            'status': self.status,
            'scheduled_date': self.scheduled_date.isoformat() if self.scheduled_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'attendee_count': len(self.attendees)
        }

class TrainingAttendance(db.Model):
    """Track who attended which training workshops"""
    id = db.Column(db.Integer, primary_key=True)
    workshop_id = db.Column(db.Integer, db.ForeignKey('training_workshop.id'), nullable=False)
    newsroom_id = db.Column(db.Integer, db.ForeignKey('newsroom.id'), nullable=True)
    attendee_name = db.Column(db.String(200), nullable=False)
    attendee_email = db.Column(db.String(200))
    attendee_role = db.Column(db.String(100))  # Editor, Reporter, Manager, etc.
    attendance_status = db.Column(db.String(50), default='Registered')  # Registered, Attended, No-show, Cancelled
    feedback_rating = db.Column(db.Integer)  # 1-5 rating
    feedback_comments = db.Column(db.Text)
    certificate_issued = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    workshop = db.relationship('TrainingWorkshop', back_populates='attendees')
    newsroom = db.relationship('Newsroom', backref='training_attendances')
    
    def to_dict(self):
        return {
            'id': self.id,
            'workshop_id': self.workshop_id,
            'newsroom_id': self.newsroom_id,
            'attendee_name': self.attendee_name,
            'attendee_email': self.attendee_email,
            'attendee_role': self.attendee_role,
            'attendance_status': self.attendance_status,
            'feedback_rating': self.feedback_rating,
            'feedback_comments': self.feedback_comments,
            'certificate_issued': self.certificate_issued,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'newsroom_name': self.newsroom.name if self.newsroom else None
        }

# ---- AI Tools Models ----
class AITool(db.Model):
    """Comprehensive AI tools database"""
    __tablename__ = 'ai_tools'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    website_url = db.Column(db.String(500), nullable=True)
    company = db.Column(db.String(200), nullable=True)
    category = db.Column(db.String(100), nullable=False)  # Text Generation, Image Generation, Video, Audio, etc.
    subcategory = db.Column(db.String(100), nullable=True)  # More specific categorization
    pricing_model = db.Column(db.String(100), nullable=True)  # Free, Freemium, Subscription, Pay-per-use
    pricing_details = db.Column(db.Text, nullable=True)
    
    # Data Safety & Privacy
    data_safety_score = db.Column(db.Float, default=0.0)  # 0-10 score
    data_safety_assessment = db.Column(db.Text, nullable=True)
    privacy_policy_url = db.Column(db.String(500), nullable=True)
    data_retention_policy = db.Column(db.Text, nullable=True)
    gdpr_compliant = db.Column(db.Boolean, default=False)
    ccpa_compliant = db.Column(db.Boolean, default=False)
    data_encryption = db.Column(db.Boolean, default=False)
    data_localization = db.Column(db.String(100), nullable=True)  # Where data is stored
    
    # Technical Details
    api_available = db.Column(db.Boolean, default=False)
    api_documentation_url = db.Column(db.String(500), nullable=True)
    integration_options = db.Column(db.Text, nullable=True)  # JSON array of integration methods
    supported_languages = db.Column(db.Text, nullable=True)  # JSON array of supported languages
    model_type = db.Column(db.String(100), nullable=True)  # GPT, Claude, Custom, etc.
    
    # Usage & Popularity
    user_count = db.Column(db.String(100), nullable=True)  # Approximate user count
    rating = db.Column(db.Float, default=0.0)  # User rating 0-5
    review_count = db.Column(db.Integer, default=0)
    
    # MediaMap Assessment
    recommendation_score = db.Column(db.Float, default=0.0)  # 0-10 score for newsroom use
    recommendation_reason = db.Column(db.Text, nullable=True)
    use_cases = db.Column(db.Text, nullable=True)  # JSON array of specific use cases
    limitations = db.Column(db.Text, nullable=True)
    alternatives = db.Column(db.Text, nullable=True)  # JSON array of alternative tools
    
    # Status & Updates
    status = db.Column(db.String(50), default='Active')  # Active, Discontinued, Beta, etc.
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    reviews = db.relationship('AIToolReview', back_populates='tool', cascade='all, delete-orphan')
    use_cases_rel = db.relationship('AIToolUseCase', back_populates='tool', cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'website_url': self.website_url,
            'company': self.company,
            'category': self.category,
            'subcategory': self.subcategory,
            'pricing_model': self.pricing_model,
            'pricing_details': self.pricing_details,
            'data_safety_score': self.data_safety_score,
            'data_safety_assessment': self.data_safety_assessment,
            'privacy_policy_url': self.privacy_policy_url,
            'data_retention_policy': self.data_retention_policy,
            'gdpr_compliant': self.gdpr_compliant,
            'ccpa_compliant': self.ccpa_compliant,
            'data_encryption': self.data_encryption,
            'data_localization': self.data_localization,
            'api_available': self.api_available,
            'api_documentation_url': self.api_documentation_url,
            'integration_options': self.integration_options,
            'supported_languages': self.supported_languages,
            'model_type': self.model_type,
            'user_count': self.user_count,
            'rating': self.rating,
            'review_count': self.review_count,
            'recommendation_score': self.recommendation_score,
            'recommendation_reason': self.recommendation_reason,
            'use_cases': self.use_cases,
            'limitations': self.limitations,
            'alternatives': self.alternatives,
            'status': self.status,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<AITool {self.name}>'


class AIToolReview(db.Model):
    """User reviews and assessments of AI tools"""
    __tablename__ = 'ai_tool_reviews'
    
    id = db.Column(db.Integer, primary_key=True)
    tool_id = db.Column(db.Integer, db.ForeignKey('ai_tools.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    reviewer_name = db.Column(db.String(200), nullable=False)
    reviewer_organization = db.Column(db.String(200), nullable=True)
    rating = db.Column(db.Float, nullable=False)  # 1-5 rating
    review_text = db.Column(db.Text, nullable=True)
    
    # Specific assessments
    ease_of_use = db.Column(db.Integer)  # 1-5 rating
    data_safety = db.Column(db.Integer)  # 1-5 rating
    cost_effectiveness = db.Column(db.Integer)  # 1-5 rating
    output_quality = db.Column(db.Integer)  # 1-5 rating
    customer_support = db.Column(db.Integer)  # 1-5 rating
    
    # Use case specific
    use_case = db.Column(db.String(200), nullable=True)
    industry = db.Column(db.String(100), nullable=True)
    team_size = db.Column(db.String(50), nullable=True)  # Small, Medium, Large
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tool = db.relationship('AITool', back_populates='reviews')
    user = db.relationship('User', backref='ai_tool_reviews')
    
    def to_dict(self):
        return {
            'id': self.id,
            'tool_id': self.tool_id,
            'tool_name': self.tool.name if self.tool else None,
            'user_id': self.user_id,
            'reviewer_name': self.reviewer_name,
            'reviewer_organization': self.reviewer_organization,
            'rating': self.rating,
            'review_text': self.review_text,
            'ease_of_use': self.ease_of_use,
            'data_safety': self.data_safety,
            'cost_effectiveness': self.cost_effectiveness,
            'output_quality': self.output_quality,
            'customer_support': self.customer_support,
            'use_case': self.use_case,
            'industry': self.industry,
            'team_size': self.team_size,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class AIToolUseCase(db.Model):
    """Specific use cases and implementations for AI tools"""
    __tablename__ = 'ai_tool_use_cases'
    
    id = db.Column(db.Integer, primary_key=True)
    tool_id = db.Column(db.Integer, db.ForeignKey('ai_tools.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    industry = db.Column(db.String(100), nullable=True)
    organization_type = db.Column(db.String(100), nullable=True)  # Newsroom, Agency, etc.
    implementation_details = db.Column(db.Text, nullable=True)
    results = db.Column(db.Text, nullable=True)
    challenges = db.Column(db.Text, nullable=True)
    lessons_learned = db.Column(db.Text, nullable=True)
    
    # Metrics
    time_saved = db.Column(db.String(100), nullable=True)  # e.g., "50%", "2 hours per day"
    cost_savings = db.Column(db.String(100), nullable=True)
    quality_improvement = db.Column(db.String(100), nullable=True)
    
    # Contact info for case study
    contact_name = db.Column(db.String(200), nullable=True)
    contact_email = db.Column(db.String(200), nullable=True)
    contact_organization = db.Column(db.String(200), nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tool = db.relationship('AITool', back_populates='use_cases_rel')
    
    def to_dict(self):
        return {
            'id': self.id,
            'tool_id': self.tool_id,
            'tool_name': self.tool.name if self.tool else None,
            'title': self.title,
            'description': self.description,
            'industry': self.industry,
            'organization_type': self.organization_type,
            'implementation_details': self.implementation_details,
            'results': self.results,
            'challenges': self.challenges,
            'lessons_learned': self.lessons_learned,
            'time_saved': self.time_saved,
            'cost_savings': self.cost_savings,
            'quality_improvement': self.quality_improvement,
            'contact_name': self.contact_name,
            'contact_email': self.contact_email,
            'contact_organization': self.contact_organization,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class AIToolCategory(db.Model):
    """Categories and tags for organizing AI tools"""
    __tablename__ = 'ai_tool_categories'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.Text, nullable=True)
    parent_category = db.Column(db.String(100), nullable=True)  # For hierarchical categories
    icon = db.Column(db.String(100), nullable=True)  # Bootstrap icon name
    color = db.Column(db.String(20), nullable=True)  # CSS color code
    sort_order = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'parent_category': self.parent_category,
            'icon': self.icon,
            'color': self.color,
            'sort_order': self.sort_order,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class AIToolRecommendation(db.Model):
    """Curated recommendations for specific use cases and organizations"""
    __tablename__ = 'ai_tool_recommendations'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    target_audience = db.Column(db.String(200), nullable=True)  # e.g., "Small newsrooms", "Digital-first publishers"
    use_case = db.Column(db.String(200), nullable=True)
    budget_range = db.Column(db.String(100), nullable=True)  # e.g., "Free", "$10-50/month", "$100+/month"
    
    # Recommended tools (JSON array of tool IDs with priority)
    recommended_tools = db.Column(db.Text, nullable=True)  # JSON array of {tool_id, priority, reason}
    
    # Alternative recommendations
    alternatives = db.Column(db.Text, nullable=True)  # JSON array of alternative tool sets
    
    # Implementation guidance
    implementation_steps = db.Column(db.Text, nullable=True)
    timeline = db.Column(db.String(100), nullable=True)
    estimated_cost = db.Column(db.String(100), nullable=True)
    training_requirements = db.Column(db.Text, nullable=True)
    
    # Status
    status = db.Column(db.String(50), default='Active')  # Active, Draft, Archived
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'target_audience': self.target_audience,
            'use_case': self.use_case,
            'budget_range': self.budget_range,
            'recommended_tools': self.recommended_tools,
            'alternatives': self.alternatives,
            'implementation_steps': self.implementation_steps,
            'timeline': self.timeline,
            'estimated_cost': self.estimated_cost,
            'training_requirements': self.training_requirements,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# ---- Consulting Models ----
class ConsultingClient(db.Model):
    """Consulting clients and their organizations"""
    __tablename__ = 'consulting_clients'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    organization = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), nullable=False)
    phone = db.Column(db.String(50), nullable=True)
    website = db.Column(db.String(500), nullable=True)
    
    # Organization details
    industry = db.Column(db.String(100), nullable=True)  # Media, Technology, Non-Profit, etc.
    organization_size = db.Column(db.String(50), nullable=True)  # Small (1-10), Medium (11-50), Large (50+)
    location = db.Column(db.String(200), nullable=True)
    timezone = db.Column(db.String(50), nullable=True)
    
    # Consulting relationship
    engagement_type = db.Column(db.String(100), nullable=True)  # One-time, Ongoing, Retainer
    contract_value = db.Column(db.Float, nullable=True)  # Total contract value
    start_date = db.Column(db.DateTime, nullable=True)
    end_date = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(50), default='Active')  # Active, Completed, Paused, Cancelled
    
    # Contact person details
    contact_person = db.Column(db.String(200), nullable=True)
    contact_role = db.Column(db.String(100), nullable=True)
    contact_email = db.Column(db.String(200), nullable=True)
    
    # Notes and tracking
    notes = db.Column(db.Text, nullable=True)
    goals = db.Column(db.Text, nullable=True)  # Client's primary goals
    challenges = db.Column(db.Text, nullable=True)  # Main challenges they face
    success_metrics = db.Column(db.Text, nullable=True)  # How success will be measured
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sessions = db.relationship('ConsultingSession', back_populates='client', cascade='all, delete-orphan')
    progress_reports = db.relationship('ConsultingProgressReport', back_populates='client', cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'organization': self.organization,
            'email': self.email,
            'phone': self.phone,
            'website': self.website,
            'industry': self.industry,
            'organization_size': self.organization_size,
            'location': self.location,
            'timezone': self.timezone,
            'engagement_type': self.engagement_type,
            'contract_value': self.contract_value,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'status': self.status,
            'contact_person': self.contact_person,
            'contact_role': self.contact_role,
            'contact_email': self.contact_email,
            'notes': self.notes,
            'goals': self.goals,
            'challenges': self.challenges,
            'success_metrics': self.success_metrics,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'session_count': len(self.sessions),
            'total_hours': sum(session.duration_hours for session in self.sessions if session.duration_hours)
        }


class ConsultingSession(db.Model):
    """Individual consulting/mentoring sessions"""
    __tablename__ = 'consulting_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.Integer, db.ForeignKey('consulting_clients.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    
    # Session details
    session_type = db.Column(db.String(100), nullable=False)  # Strategy, Implementation, Training, Review, Q&A
    session_date = db.Column(db.DateTime, nullable=False)
    duration_hours = db.Column(db.Float, nullable=True)
    session_notes = db.Column(db.Text, nullable=True)
    
    # Recording and materials
    recording_url = db.Column(db.String(500), nullable=True)  # Link to uploaded recording
    recording_file_path = db.Column(db.String(500), nullable=True)  # Local file path
    recording_duration = db.Column(db.Integer, nullable=True)  # Duration in seconds
    materials_shared = db.Column(db.Text, nullable=True)  # JSON array of shared materials
    
    # Session outcomes
    topics_covered = db.Column(db.Text, nullable=True)  # JSON array of topics discussed
    action_items = db.Column(db.Text, nullable=True)  # JSON array of action items
    next_steps = db.Column(db.Text, nullable=True)
    
    # Client feedback
    client_satisfaction = db.Column(db.Integer, nullable=True)  # 1-5 rating
    client_feedback = db.Column(db.Text, nullable=True)
    client_questions = db.Column(db.Text, nullable=True)  # Questions asked during session
    
    # Session status
    status = db.Column(db.String(50), default='Scheduled')  # Scheduled, Completed, Cancelled, Rescheduled
    follow_up_required = db.Column(db.Boolean, default=False)
    follow_up_date = db.Column(db.DateTime, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    client = db.relationship('ConsultingClient', back_populates='sessions')
    progress_entries = db.relationship('ConsultingProgressEntry', back_populates='session', cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'client_id': self.client_id,
            'client_name': self.client.name if self.client else None,
            'title': self.title,
            'description': self.description,
            'session_type': self.session_type,
            'session_date': self.session_date.isoformat() if self.session_date else None,
            'duration_hours': self.duration_hours,
            'session_notes': self.session_notes,
            'recording_url': self.recording_url,
            'recording_file_path': self.recording_file_path,
            'recording_duration': self.recording_duration,
            'materials_shared': self.materials_shared,
            'topics_covered': self.topics_covered,
            'action_items': self.action_items,
            'next_steps': self.next_steps,
            'client_satisfaction': self.client_satisfaction,
            'client_feedback': self.client_feedback,
            'client_questions': self.client_questions,
            'status': self.status,
            'follow_up_required': self.follow_up_required,
            'follow_up_date': self.follow_up_date.isoformat() if self.follow_up_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ConsultingProgressReport(db.Model):
    """Periodic progress reports for consulting clients"""
    __tablename__ = 'consulting_progress_reports'
    
    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.Integer, db.ForeignKey('consulting_clients.id'), nullable=False)
    report_date = db.Column(db.DateTime, nullable=False)
    report_period = db.Column(db.String(50), nullable=True)  # Weekly, Monthly, Quarterly
    
    # Progress assessment
    goals_progress = db.Column(db.Text, nullable=True)  # JSON object with goal progress
    achievements = db.Column(db.Text, nullable=True)  # JSON array of achievements
    challenges_faced = db.Column(db.Text, nullable=True)  # JSON array of challenges
    lessons_learned = db.Column(db.Text, nullable=True)  # JSON array of lessons learned
    
    # Metrics and KPIs
    key_metrics = db.Column(db.Text, nullable=True)  # JSON object with metric values
    improvement_areas = db.Column(db.Text, nullable=True)  # JSON array of areas needing improvement
    success_stories = db.Column(db.Text, nullable=True)  # JSON array of success stories
    
    # Recommendations
    recommendations = db.Column(db.Text, nullable=True)  # JSON array of recommendations
    next_quarter_goals = db.Column(db.Text, nullable=True)  # JSON array of future goals
    
    # Client feedback
    client_comments = db.Column(db.Text, nullable=True)
    overall_satisfaction = db.Column(db.Integer, nullable=True)  # 1-5 rating
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    client = db.relationship('ConsultingClient', back_populates='progress_reports')
    
    def to_dict(self):
        return {
            'id': self.id,
            'client_id': self.client_id,
            'client_name': self.client.name if self.client else None,
            'report_date': self.report_date.isoformat() if self.report_date else None,
            'report_period': self.report_period,
            'goals_progress': self.goals_progress,
            'achievements': self.achievements,
            'challenges_faced': self.challenges_faced,
            'lessons_learned': self.lessons_learned,
            'key_metrics': self.key_metrics,
            'improvement_areas': self.improvement_areas,
            'success_stories': self.success_stories,
            'recommendations': self.recommendations,
            'next_quarter_goals': self.next_quarter_goals,
            'client_comments': self.client_comments,
            'overall_satisfaction': self.overall_satisfaction,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ConsultingProgressEntry(db.Model):
    """Individual progress entries linked to sessions"""
    __tablename__ = 'consulting_progress_entries'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('consulting_sessions.id'), nullable=False)
    entry_date = db.Column(db.DateTime, nullable=False)
    
    # Progress tracking
    category = db.Column(db.String(100), nullable=False)  # Knowledge, Skills, Implementation, Results
    metric_name = db.Column(db.String(200), nullable=False)
    metric_value = db.Column(db.String(200), nullable=True)
    metric_unit = db.Column(db.String(50), nullable=True)  # %, hours, $, etc.
    
    # Assessment
    current_level = db.Column(db.String(50), nullable=True)  # Beginner, Intermediate, Advanced, Expert
    target_level = db.Column(db.String(50), nullable=True)
    progress_notes = db.Column(db.Text, nullable=True)
    
    # Evidence
    evidence = db.Column(db.Text, nullable=True)  # Description of evidence supporting progress
    supporting_files = db.Column(db.Text, nullable=True)  # JSON array of file paths
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    session = db.relationship('ConsultingSession', back_populates='progress_entries')
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'entry_date': self.entry_date.isoformat() if self.entry_date else None,
            'category': self.category,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'metric_unit': self.metric_unit,
            'current_level': self.current_level,
            'target_level': self.target_level,
            'progress_notes': self.progress_notes,
            'evidence': self.evidence,
            'supporting_files': self.supporting_files,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ConsultingSuccessMetric(db.Model):
    """Success metrics and KPIs for consulting engagements"""
    __tablename__ = 'consulting_success_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.Integer, db.ForeignKey('consulting_clients.id'), nullable=False)
    metric_name = db.Column(db.String(200), nullable=False)
    metric_description = db.Column(db.Text, nullable=True)
    
    # Metric details
    metric_type = db.Column(db.String(50), nullable=False)  # Quantitative, Qualitative, Binary
    unit = db.Column(db.String(50), nullable=True)  # %, $, hours, etc.
    baseline_value = db.Column(db.Float, nullable=True)
    target_value = db.Column(db.Float, nullable=True)
    current_value = db.Column(db.Float, nullable=True)
    
    # Tracking
    measurement_frequency = db.Column(db.String(50), nullable=True)  # Daily, Weekly, Monthly, Quarterly
    last_measured = db.Column(db.DateTime, nullable=True)
    next_measurement = db.Column(db.DateTime, nullable=True)
    
    # Status
    status = db.Column(db.String(50), default='Active')  # Active, Achieved, Paused, Cancelled
    priority = db.Column(db.String(20), default='Medium')  # Low, Medium, High, Critical
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'client_id': self.client_id,
            'metric_name': self.metric_name,
            'metric_description': self.metric_description,
            'metric_type': self.metric_type,
            'unit': self.unit,
            'baseline_value': self.baseline_value,
            'target_value': self.target_value,
            'current_value': self.current_value,
            'measurement_frequency': self.measurement_frequency,
            'last_measured': self.last_measured.isoformat() if self.last_measured else None,
            'next_measurement': self.next_measurement.isoformat() if self.next_measurement else None,
            'status': self.status,
            'priority': self.priority,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class PersonManagement(db.Model):
    """People management for AI consulting projects"""
    __tablename__ = 'people_management'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    title = db.Column(db.String(200), nullable=True)
    role = db.Column(db.String(100), nullable=False)  # AI Expert, Consultant, Client Contact, Team Member
    organization = db.Column(db.String(200), nullable=True)
    
    # Contact Information
    email = db.Column(db.String(200), nullable=True)
    phone = db.Column(db.String(50), nullable=True)
    linkedin_url = db.Column(db.String(500), nullable=True)
    
    # Expertise and Skills
    expertise = db.Column(db.String(500), nullable=True)  # Comma-separated expertise areas
    ai_skills = db.Column(db.String(500), nullable=True)  # Comma-separated AI skills
    industry_experience = db.Column(db.String(500), nullable=True)  # Comma-separated industries
    
    # Project Information
    current_projects = db.Column(db.String(500), nullable=True)  # Comma-separated project IDs
    availability = db.Column(db.String(50), default='Available')  # Available, Busy, Unavailable
    hourly_rate = db.Column(db.Float, nullable=True)
    
    # Status and Metadata
    status = db.Column(db.String(50), default='Active')  # Active, Inactive, Former
    notes = db.Column(db.Text, nullable=True)
    tags = db.Column(db.String(500), nullable=True)  # Comma-separated tags
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships (simplified for now)
    # project_assignments = db.relationship('ProjectAssignment', back_populates='person', cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'title': self.title,
            'role': self.role,
            'organization': self.organization,
            'email': self.email,
            'phone': self.phone,
            'linkedin_url': self.linkedin_url,
            'expertise': self.expertise.split(',') if self.expertise else [],
            'ai_skills': self.ai_skills.split(',') if self.ai_skills else [],
            'industry_experience': self.industry_experience.split(',') if self.industry_experience else [],
            'current_projects': self.current_projects.split(',') if self.current_projects else [],
            'availability': self.availability,
            'hourly_rate': self.hourly_rate,
            'status': self.status,
            'notes': self.notes,
            'tags': self.tags.split(',') if self.tags else [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class Project(db.Model):
    """AI consulting projects management"""
    __tablename__ = 'projects'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    type = db.Column(db.String(100), nullable=False)  # AI Implementation, Strategy, Training, Research, Consulting
    status = db.Column(db.String(50), default='Planning')  # Planning, Active, On Hold, Completed, Cancelled
    
    # Client Information
    client_id = db.Column(db.Integer, db.ForeignKey('consulting_clients.id'), nullable=True)
    client_name = db.Column(db.String(200), nullable=True)  # Fallback if client not in system
    
    # Timeline
    start_date = db.Column(db.DateTime, nullable=True)
    end_date = db.Column(db.DateTime, nullable=True)
    estimated_hours = db.Column(db.Float, nullable=True)
    actual_hours = db.Column(db.Float, nullable=True)
    
    # Project Details
    objectives = db.Column(db.Text, nullable=True)
    deliverables = db.Column(db.Text, nullable=True)
    success_metrics = db.Column(db.Text, nullable=True)
    risks_and_challenges = db.Column(db.Text, nullable=True)
    
    # AI Specific Fields
    ai_technologies = db.Column(db.String(500), nullable=True)  # Comma-separated AI tools/technologies
    ai_maturity_level = db.Column(db.String(50), nullable=True)  # Beginner, Intermediate, Advanced, Expert
    data_requirements = db.Column(db.Text, nullable=True)
    
    # Financial
    budget = db.Column(db.Float, nullable=True)
    actual_cost = db.Column(db.Float, nullable=True)
    billing_type = db.Column(db.String(50), nullable=True)  # Hourly, Fixed, Retainer
    
    # Metadata
    tags = db.Column(db.String(500), nullable=True)  # Comma-separated tags
    priority = db.Column(db.String(20), default='Medium')  # Low, Medium, High, Critical
    notes = db.Column(db.Text, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships (simplified for now)
    # assignments = db.relationship('ProjectAssignment', back_populates='project', cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'type': self.type,
            'status': self.status,
            'client_id': self.client_id,
            'client_name': self.client_name,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'estimated_hours': self.estimated_hours,
            'actual_hours': self.actual_hours,
            'objectives': self.objectives,
            'deliverables': self.deliverables,
            'success_metrics': self.success_metrics,
            'risks_and_challenges': self.risks_and_challenges,
            'ai_technologies': self.ai_technologies.split(',') if self.ai_technologies else [],
            'ai_maturity_level': self.ai_maturity_level,
            'data_requirements': self.data_requirements,
            'budget': self.budget,
            'actual_cost': self.actual_cost,
            'billing_type': self.billing_type,
            'tags': self.tags.split(',') if self.tags else [],
            'priority': self.priority,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ProjectAssignment(db.Model):
    """Many-to-many relationship between People and Projects with role information"""
    __tablename__ = 'project_assignments'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    person_id = db.Column(db.Integer, db.ForeignKey('people.id'), nullable=False)
    
    # Assignment Details
    role = db.Column(db.String(100), nullable=False)  # Project Manager, AI Expert, Consultant, Developer
    responsibilities = db.Column(db.Text, nullable=True)
    start_date = db.Column(db.DateTime, nullable=True)
    end_date = db.Column(db.DateTime, nullable=True)
    hours_allocated = db.Column(db.Float, nullable=True)
    hourly_rate = db.Column(db.Float, nullable=True)
    
    # Status
    status = db.Column(db.String(50), default='Active')  # Active, Completed, On Hold, Removed
    performance_rating = db.Column(db.Integer, nullable=True)  # 1-5 rating
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships (simplified for now)
    # project = db.relationship('Project', back_populates='assignments')
    # person = db.relationship('Person', back_populates='project_assignments')
    
    def to_dict(self):
        return {
            'id': self.id,
            'project_id': self.project_id,
            'person_id': self.person_id,
            'role': self.role,
            'responsibilities': self.responsibilities,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'hours_allocated': self.hours_allocated,
            'hourly_rate': self.hourly_rate,
            'status': self.status,
            'performance_rating': self.performance_rating,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ProjectTemplate(db.Model):
    """Pre-built AI project templates for different industries and use cases"""
    __tablename__ = 'project_templates'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    category = db.Column(db.String(100), nullable=False)  # Industry, Use Case, Technology
    industry = db.Column(db.String(100), nullable=True)
    
    # Template Structure
    phases = db.Column(db.Text, nullable=True)  # JSON array of project phases
    deliverables = db.Column(db.Text, nullable=True)  # JSON array of deliverables
    timeline = db.Column(db.String(100), nullable=True)  # Estimated timeline
    estimated_hours = db.Column(db.Float, nullable=True)
    
    # AI Specific
    ai_technologies = db.Column(db.String(500), nullable=True)  # Comma-separated AI tools
    ai_maturity_requirements = db.Column(db.String(100), nullable=True)
    data_requirements = db.Column(db.Text, nullable=True)
    
    # Success Metrics
    success_metrics = db.Column(db.Text, nullable=True)  # JSON array of metrics
    risk_factors = db.Column(db.Text, nullable=True)  # JSON array of risks
    
    # Usage
    usage_count = db.Column(db.Integer, default=0)
    rating = db.Column(db.Float, default=0.0)
    tags = db.Column(db.String(500), nullable=True)  # Comma-separated tags
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'industry': self.industry,
            'phases': self.phases,
            'deliverables': self.deliverables,
            'timeline': self.timeline,
            'estimated_hours': self.estimated_hours,
            'ai_technologies': self.ai_technologies.split(',') if self.ai_technologies else [],
            'ai_maturity_requirements': self.ai_maturity_requirements,
            'data_requirements': self.data_requirements,
            'success_metrics': self.success_metrics,
            'risk_factors': self.risk_factors,
            'usage_count': self.usage_count,
            'rating': self.rating,
            'tags': self.tags.split(',') if self.tags else [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }