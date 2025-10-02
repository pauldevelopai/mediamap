#!/usr/bin/env python3
import os
import sys
import sqlite3
from pathlib import Path

# Add the backend directory to Python path
basedir = os.path.abspath(os.path.dirname(__file__))
backend_dir = os.path.join(basedir, 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from flask import Flask
from models import db, User, HighlanderChat, PromptTemplate, PromptVersion
from werkzeug.security import generate_password_hash

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///opt/mediamap/instance/media_analysis.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    return app

def fix_database():
    app = create_app()
    with app.app_context():
        print("🔧 Creating missing tables...")
        
        # Create all tables
        db.create_all()
        
        # Create missing tables manually if needed
        try:
            db.engine.execute("""
                CREATE TABLE IF NOT EXISTS highlander_chat (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id VARCHAR(100) NOT NULL,
                    message TEXT NOT NULL,
                    response TEXT NOT NULL,
                    context TEXT,
                    category VARCHAR(100),
                    processed BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            print("✅ highlander_chat table created")
        except Exception as e:
            print(f"⚠️ highlander_chat table: {e}")
        
        try:
            db.engine.execute("""
                CREATE TABLE IF NOT EXISTS prompt_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name VARCHAR(200) NOT NULL UNIQUE,
                    description TEXT,
                    category VARCHAR(100) NOT NULL,
                    prompt_type VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    llm_provider VARCHAR(50) NOT NULL,
                    model_name VARCHAR(100),
                    usage_context VARCHAR(200),
                    variables TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    version VARCHAR(20) DEFAULT '1.0',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_by INTEGER,
                    FOREIGN KEY (created_by) REFERENCES users (id)
                )
            """)
            print("✅ prompt_templates table created")
        except Exception as e:
            print(f"⚠️ prompt_templates table: {e}")
        
        try:
            db.engine.execute("""
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_id INTEGER NOT NULL,
                    version_number VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_by INTEGER,
                    FOREIGN KEY (prompt_id) REFERENCES prompt_templates (id),
                    FOREIGN KEY (created_by) REFERENCES users (id)
                )
            """)
            print("✅ prompt_versions table created")
        except Exception as e:
            print(f"⚠️ prompt_versions table: {e}")
        
        # Create HealthPIN tables
        try:
            db.engine.execute("""
                CREATE TABLE IF NOT EXISTS healthpin_patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    phone_number VARCHAR(20) NOT NULL UNIQUE,
                    whatsapp_id VARCHAR(50) UNIQUE,
                    first_name VARCHAR(100) NOT NULL,
                    last_name VARCHAR(100) NOT NULL,
                    date_of_birth DATE,
                    gender VARCHAR(10),
                    language_preference VARCHAR(10) DEFAULT 'en',
                    city VARCHAR(100),
                    province VARCHAR(100),
                    country VARCHAR(100) DEFAULT 'South Africa',
                    preferred_specialties TEXT,
                    cultural_preferences TEXT,
                    accessibility_needs TEXT,
                    emergency_contact_name VARCHAR(200),
                    emergency_contact_phone VARCHAR(20),
                    family_members TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            print("✅ healthpin_patients table created")
        except Exception as e:
            print(f"⚠️ healthpin_patients table: {e}")
        
        try:
            db.engine.execute("""
                CREATE TABLE IF NOT EXISTS healthpin_doctors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    phone_number VARCHAR(20) NOT NULL UNIQUE,
                    whatsapp_id VARCHAR(50) UNIQUE,
                    first_name VARCHAR(100) NOT NULL,
                    last_name VARCHAR(100) NOT NULL,
                    medical_license VARCHAR(100) UNIQUE,
                    specialties TEXT,
                    qualifications TEXT,
                    experience_years INTEGER,
                    languages TEXT,
                    consultation_fee DECIMAL(10,2),
                    availability_schedule TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            print("✅ healthpin_doctors table created")
        except Exception as e:
            print(f"⚠️ healthpin_doctors table: {e}")
        
        try:
            db.engine.execute("""
                CREATE TABLE IF NOT EXISTS healthpin_health_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    doctor_id INTEGER,
                    record_type VARCHAR(50) NOT NULL,
                    title VARCHAR(200) NOT NULL,
                    content TEXT NOT NULL,
                    diagnosis TEXT,
                    treatment_plan TEXT,
                    medications TEXT,
                    follow_up_date DATE,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES healthpin_patients (id),
                    FOREIGN KEY (doctor_id) REFERENCES healthpin_doctors (id)
                )
            """)
            print("✅ healthpin_health_records table created")
        except Exception as e:
            print(f"⚠️ healthpin_health_records table: {e}")
        
        try:
            db.engine.execute("""
                CREATE TABLE IF NOT EXISTS healthpin_doctor_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    doctor_id INTEGER NOT NULL,
                    match_score DECIMAL(5,2),
                    match_reasons TEXT,
                    status VARCHAR(50) DEFAULT 'pending',
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES healthpin_patients (id),
                    FOREIGN KEY (doctor_id) REFERENCES healthpin_doctors (id)
                )
            """)
            print("✅ healthpin_doctor_matches table created")
        except Exception as e:
            print(f"⚠️ healthpin_doctor_matches table: {e}")
        
        try:
            db.engine.execute("""
                CREATE TABLE IF NOT EXISTS healthpin_family_notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    family_member_phone VARCHAR(20) NOT NULL,
                    notification_type VARCHAR(50) NOT NULL,
                    message TEXT NOT NULL,
                    status VARCHAR(50) DEFAULT 'pending',
                    sent_at DATETIME,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES healthpin_patients (id)
                )
            """)
            print("✅ healthpin_family_notifications table created")
        except Exception as e:
            print(f"⚠️ healthpin_family_notifications table: {e}")
        
        try:
            db.engine.execute("""
                CREATE TABLE IF NOT EXISTS healthpin_consultations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    doctor_id INTEGER NOT NULL,
                    consultation_type VARCHAR(50) NOT NULL,
                    scheduled_date DATETIME,
                    status VARCHAR(50) DEFAULT 'scheduled',
                    notes TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES healthpin_patients (id),
                    FOREIGN KEY (doctor_id) REFERENCES healthpin_doctors (id)
                )
            """)
            print("✅ healthpin_consultations table created")
        except Exception as e:
            print(f"⚠️ healthpin_consultations table: {e}")
        
        try:
            db.engine.execute("""
                CREATE TABLE IF NOT EXISTS healthpin_health_news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title VARCHAR(200) NOT NULL,
                    content TEXT NOT NULL,
                    category VARCHAR(100),
                    language VARCHAR(10) DEFAULT 'en',
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("✅ healthpin_health_news table created")
        except Exception as e:
            print(f"⚠️ healthpin_health_news table: {e}")
        
        # Create admin user if it doesn't exist
        try:
            admin_user = User.query.filter_by(username='admin').first()
            if not admin_user:
                admin_user = User(
                    username='admin',
                    email='admin@aimap.ai',
                    password_hash=generate_password_hash('admin123'),
                    is_admin=True
                )
                db.session.add(admin_user)
                db.session.commit()
                print("✅ Admin user created")
            else:
                print("✅ Admin user exists")
        except Exception as e:
            print(f"⚠️ Admin user: {e}")
        
        # Create default prompt templates
        try:
            if PromptTemplate.query.count() == 0:
                default_prompts = [
                    PromptTemplate(
                        name='MediaMap System Prompt',
                        description='Default system prompt for MediaMap AI',
                        category='system',
                        prompt_type='system_message',
                        content='You are MediaMap AI, a specialized assistant for media industry analysis, business insights, and strategic planning.',
                        llm_provider='openai',
                        model_name='gpt-4',
                        usage_context='MediaMap chat interface',
                        variables='{"user_name": "User name", "context": "Business context"}',
                        is_active=True,
                        version='1.0'
                    ),
                    PromptTemplate(
                        name='HealthPIN System Prompt',
                        description='Default system prompt for HealthPIN AI',
                        category='system',
                        prompt_type='system_message',
                        content='You are HealthPIN AI, a specialized medical assistant for healthcare analysis and clinical insights.',
                        llm_provider='openai',
                        model_name='gpt-4',
                        usage_context='HealthPIN chat interface',
                        variables='{"user_name": "User name", "context": "Clinical context"}',
                        is_active=True,
                        version='1.0'
                    )
                ]
                
                for prompt in default_prompts:
                    db.session.add(prompt)
                
                db.session.commit()
                print("✅ Default prompt templates created")
            else:
                print("✅ Prompt templates exist")
        except Exception as e:
            print(f"⚠️ Prompt templates: {e}")
        
        print("🎉 Database fix completed!")

if __name__ == '__main__':
    fix_database()
