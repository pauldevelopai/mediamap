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