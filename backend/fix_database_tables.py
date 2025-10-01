#!/usr/bin/env python3
"""
Database Fix Script for Lightsail Instance
==========================================

This script fixes missing database tables and ensures all required tables exist.
Run this on the Lightsail instance to resolve database issues.
"""

import os
import sys
import sqlite3
from pathlib import Path

# Add the backend directory to Python path
basedir = os.path.abspath(os.path.dirname(__file__))
if basedir not in sys.path:
    sys.path.insert(0, basedir)

from flask import Flask
from models import db, User, MediaAnalysis, Chat, Message, UserSection, Lesson, UserLesson, OrganizationInfo, OrganizationFact, Translation, TranslationFeedback, Location, Feedback, NotionIntegration, News, SavedStrategy, SavedNews, ImplementationPlan, DailyReport, CheatSheet, HighlanderChat, PromptTemplate, PromptVersion

def create_app():
    """Create Flask app for database operations"""
    app = Flask(__name__)
    
    # Database configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/aimap.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    return app

def fix_database_tables():
    """Fix missing database tables"""
    app = create_app()
    
    with app.app_context():
        print("🔧 Fixing database tables...")
        
        try:
            # Create all tables
            db.create_all()
            print("✅ All database tables created successfully")
            
            # Check if tables exist
            inspector = db.inspect(db.engine)
            existing_tables = inspector.get_table_names()
            
            required_tables = [
                'users', 'media_analysis', 'chats', 'messages', 'user_sections',
                'lessons', 'user_lessons', 'organization_info', 'organization_facts',
                'translations', 'translation_feedback', 'locations', 'feedback',
                'notion_integrations', 'news', 'saved_strategies', 'saved_news',
                'implementation_plans', 'daily_reports', 'cheat_sheets',
                'highlander_chat', 'prompt_templates', 'prompt_versions'
            ]
            
            missing_tables = []
            for table in required_tables:
                if table not in existing_tables:
                    missing_tables.append(table)
            
            if missing_tables:
                print(f"⚠️  Missing tables: {missing_tables}")
                print("🔧 Creating missing tables...")
                
                # Create missing tables individually
                for table in missing_tables:
                    try:
                        if table == 'highlander_chat':
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
                        elif table == 'prompt_templates':
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
                        elif table == 'prompt_versions':
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
                        
                        print(f"✅ Created table: {table}")
                    except Exception as e:
                        print(f"❌ Error creating table {table}: {e}")
                
                db.session.commit()
                print("✅ All missing tables created")
            else:
                print("✅ All required tables exist")
            
            # Verify tables exist
            inspector = db.inspect(db.engine)
            final_tables = inspector.get_table_names()
            print(f"📊 Total tables in database: {len(final_tables)}")
            
            # Check for admin user
            admin_user = User.query.filter_by(username='admin').first()
            if not admin_user:
                print("🔧 Creating admin user...")
                from werkzeug.security import generate_password_hash
                
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
            
            # Create default prompt templates
            create_default_prompts()
            
            print("🎉 Database fix completed successfully!")
            
        except Exception as e:
            print(f"❌ Error fixing database: {e}")
            db.session.rollback()
            raise

def create_default_prompts():
    """Create default prompt templates"""
    try:
        # Check if prompts already exist
        existing_prompts = PromptTemplate.query.count()
        if existing_prompts > 0:
            print("✅ Prompt templates already exist")
            return
        
        print("🔧 Creating default prompt templates...")
        
        default_prompts = [
            {
                'name': 'MediaMap System Prompt',
                'description': 'Default system prompt for MediaMap AI',
                'category': 'system',
                'prompt_type': 'system_message',
                'content': 'You are MediaMap AI, a specialized assistant for media industry analysis, business insights, and strategic planning. You provide expert advice on media trends, ROI optimization, and industry best practices.',
                'llm_provider': 'openai',
                'model_name': 'gpt-4',
                'usage_context': 'MediaMap chat interface',
                'variables': '{"user_name": "User name", "context": "Business context"}',
                'is_active': True,
                'version': '1.0'
            },
            {
                'name': 'HealthPIN System Prompt',
                'description': 'Default system prompt for HealthPIN AI',
                'category': 'system',
                'prompt_type': 'system_message',
                'content': 'You are HealthPIN AI, a specialized medical assistant for healthcare analysis and clinical insights. You provide evidence-based medical information, patient care guidance, and healthcare industry analysis.',
                'llm_provider': 'openai',
                'model_name': 'gpt-4',
                'usage_context': 'HealthPIN chat interface',
                'variables': '{"user_name": "User name", "context": "Clinical context"}',
                'is_active': True,
                'version': '1.0'
            },
            {
                'name': 'Highlander Business Analysis',
                'description': 'Prompt for Highlander AI business analysis',
                'category': 'analysis',
                'prompt_type': 'user_prompt',
                'content': 'Analyze the following business data and provide strategic insights: {data}',
                'llm_provider': 'openai',
                'model_name': 'gpt-4',
                'usage_context': 'Highlander AI analysis',
                'variables': '{"data": "Business data to analyze"}',
                'is_active': True,
                'version': '1.0'
            }
        ]
        
        for prompt_data in default_prompts:
            prompt = PromptTemplate(**prompt_data)
            db.session.add(prompt)
        
        db.session.commit()
        print("✅ Default prompt templates created")
        
    except Exception as e:
        print(f"❌ Error creating default prompts: {e}")
        db.session.rollback()

def check_database_health():
    """Check database health and report issues"""
    app = create_app()
    
    with app.app_context():
        print("🔍 Checking database health...")
        
        try:
            # Check table counts
            tables_to_check = [
                ('users', User),
                ('highlander_chat', HighlanderChat),
                ('prompt_templates', PromptTemplate),
                ('chats', Chat),
                ('messages', Message)
            ]
            
            for table_name, model_class in tables_to_check:
                try:
                    count = model_class.query.count()
                    print(f"📊 {table_name}: {count} records")
                except Exception as e:
                    print(f"❌ Error checking {table_name}: {e}")
            
            print("✅ Database health check completed")
            
        except Exception as e:
            print(f"❌ Error during health check: {e}")

if __name__ == '__main__':
    print("🚀 Starting database fix for Lightsail instance...")
    print("=" * 50)
    
    try:
        fix_database_tables()
        print("\n" + "=" * 50)
        check_database_health()
        print("\n🎉 Database fix completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Database fix failed: {e}")
        sys.exit(1)
