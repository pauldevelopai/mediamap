"""Database configuration for Postmaster Films"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from .settings import get_settings

settings = get_settings()
engine = create_engine(
    settings.DATABASE_URL, 
    future=True,
    # SQLite specific configuration
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

