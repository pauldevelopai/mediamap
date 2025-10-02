#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.app import app
from backend.models import db, CrawlSource, CrawledContent, CrawlJob

def migrate_crawling_db():
    """Create the new crawling tables in the database"""
    with app.app_context():
        print("Creating crawling database tables...")
        db.create_all()
        print("✅ Crawling database tables created successfully!")
        
        # Check if tables were created
        try:
            sources = CrawlSource.query.all()
            content = CrawledContent.query.all()
            jobs = CrawlJob.query.all()
            print(f"✅ Tables verified: {len(sources)} sources, {len(content)} content items, {len(jobs)} jobs")
        except Exception as e:
            print(f"❌ Error verifying tables: {e}")

if __name__ == "__main__":
    migrate_crawling_db() 