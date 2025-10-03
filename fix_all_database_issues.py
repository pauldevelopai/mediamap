#!/usr/bin/env python3
"""
Complete Database Fix Script for Lightsail
=========================================

This script fixes both issues:
1. Missing 'organisations' table error
2. HealthPIN doctor scraping database setup

Run this on the Lightsail instance to fix all database issues.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_paths():
    """Setup Python paths for imports"""
    script_dir = Path(__file__).parent
    backend_dir = script_dir / 'backend'
    
    # Add paths
    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(backend_dir))
    
    logger.info(f"Added paths: {script_dir}, {backend_dir}")

def create_flask_app():
    """Create and configure Flask app"""
    from flask import Flask
    from backend.models import db
    
    app = Flask(__name__)
    
    # Database configuration
    db_path = os.path.join(os.path.dirname(__file__), 'instance', 'aimap.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'fix-database-key'
    
    db.init_app(app)
    
    logger.info(f"Flask app configured with database: {db_path}")
    return app, db

def fix_organisations_table(app, db):
    """Fix the missing organisations table issue"""
    logger.info("🔧 Fixing organisations table...")
    
    with app.app_context():
        try:
            # Import all models to ensure they're registered
            from backend.models import User, MediaAnalysis, Chat
            from backend.aimap.models import (
                Organisation, Lead, LeadActivity, Interaction,
                ResearchReport, CustomData, ConsultingProject, ProjectMilestone
            )
            
            logger.info("✅ All models imported successfully")
            
            # Create all tables
            db.create_all()
            logger.info("✅ Database tables created/verified")
            
            # Verify organisations table exists
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            logger.info(f"Available tables: {', '.join(tables)}")
            
            if 'organisations' in tables:
                logger.info("✅ organisations table verified")
                
                # Test a simple query
                orgs = Organisation.query.limit(5).all()
                logger.info(f"✅ organisations table query successful - found {len(orgs)} records")
                
                return True
            else:
                logger.error("❌ organisations table NOT found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error fixing organisations table: {e}")
            import traceback
            traceback.print_exc()
            return False

def fix_healthpin_database(app, db):
    """Fix HealthPIN database tables"""
    logger.info("🏥 Fixing HealthPIN database...")
    
    with app.app_context():
        try:
            # Import HealthPIN models
            from backend.healthpin.models import (
                Patient, Doctor, HealthRecord, DoctorMatch
            )
            
            logger.info("✅ HealthPIN models imported successfully")
            
            # Create HealthPIN tables
            db.create_all()
            logger.info("✅ HealthPIN database tables created/verified")
            
            # Verify HealthPIN tables exist
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            healthpin_tables = [t for t in tables if 'healthpin' in t]
            logger.info(f"HealthPIN tables: {', '.join(healthpin_tables)}")
            
            expected_tables = ['healthpin_patients', 'healthpin_doctors', 'healthpin_health_records', 'healthpin_doctor_matches']
            missing_tables = [t for t in expected_tables if t not in tables]
            
            if missing_tables:
                logger.warning(f"⚠️ Missing HealthPIN tables: {', '.join(missing_tables)}")
            else:
                logger.info("✅ All HealthPIN tables verified")
            
            # Test doctor table specifically
            doctors = Doctor.query.limit(5).all()
            logger.info(f"✅ HealthPIN doctors table query successful - found {len(doctors)} records")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error fixing HealthPIN database: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_healthpin_agent():
    """Test if HealthPIN agent is working"""
    logger.info("🧪 Testing HealthPIN agent...")
    
    try:
        from backend.agents.agent_manager import AgentManager
        
        # Initialize agent manager
        agent_manager = AgentManager()
        
        if 'healthpin' in agent_manager.agents:
            agent = agent_manager.agents['healthpin']
            logger.info("✅ HealthPIN agent found in agent manager")
            
            # Check if scraping method exists
            if hasattr(agent, 'scrape_doctors_south_africa'):
                logger.info("✅ Doctor scraping method available")
                
                # Test with a small limit to verify it works
                logger.info("🔍 Testing doctor scraping (limit=1)...")
                result = agent.scrape_doctors_south_africa(limit=1)
                
                if result.get('success'):
                    logger.info(f"✅ Doctor scraping test successful: {result}")
                else:
                    logger.warning(f"⚠️ Doctor scraping test failed: {result}")
                
                return True
            else:
                logger.error("❌ Doctor scraping method not available")
                return False
        else:
            logger.error("❌ HealthPIN agent not found in agent manager")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing HealthPIN agent: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main fix function"""
    logger.info("🚀 Starting database fix process...")
    
    # Setup paths
    setup_paths()
    
    # Create Flask app
    app, db = create_flask_app()
    
    # Fix organisations table
    org_success = fix_organisations_table(app, db)
    
    # Fix HealthPIN database
    healthpin_success = fix_healthpin_database(app, db)
    
    # Test HealthPIN agent
    agent_success = test_healthpin_agent()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("FIX SUMMARY")
    logger.info("="*50)
    logger.info(f"✅ Organisations table: {'SUCCESS' if org_success else 'FAILED'}")
    logger.info(f"✅ HealthPIN database: {'SUCCESS' if healthpin_success else 'FAILED'}")
    logger.info(f"✅ HealthPIN agent: {'SUCCESS' if agent_success else 'FAILED'}")
    
    if org_success and healthpin_success:
        logger.info("\n🎉 All database issues fixed successfully!")
        logger.info("💡 You can now:")
        logger.info("   - Search for organisations without errors")
        logger.info("   - Use HealthPIN doctor scraping functionality")
        logger.info("   - Access all HealthPIN features")
        return True
    else:
        logger.error("\n❌ Some issues remain - check logs above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
