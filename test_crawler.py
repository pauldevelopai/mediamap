#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.app import app
from backend.models import db, CrawlSource, CrawledContent, CrawlJob
from backend.crawler_service import CrawlManager
import json

def test_crawler():
    """Test the crawling functionality"""
    with app.app_context():
        print("Testing crawling functionality...")
        
        # Test 1: Create a test source
        print("\n1. Creating test source...")
        test_source = CrawlSource(
            name="Test Website",
            url="https://example.com",
            source_type="website",
            description="Test source for crawling",
            crawl_frequency="manual",
            is_active=True
        )
        db.session.add(test_source)
        db.session.commit()
        print(f"✅ Created source: {test_source.name} (ID: {test_source.id})")
        
        # Test 2: Create a crawl job
        print("\n2. Creating crawl job...")
        crawl_manager = CrawlManager(db, os.getenv('OPENAI_API_KEY'))
        job_id = crawl_manager.create_crawl_job(test_source.id)
        print(f"✅ Created job ID: {job_id}")
        
        # Test 3: Get crawl stats
        print("\n3. Getting crawl stats...")
        stats = crawl_manager.get_crawl_stats()
        print(f"✅ Stats: {stats}")
        
        # Test 4: Test content analysis
        print("\n4. Testing content analysis...")
        from backend.crawler_service import ContentCrawler
        crawler = ContentCrawler(os.getenv('OPENAI_API_KEY'))
        
        test_content = """
        This is a test article about AI strategies for media businesses.
        It discusses automation, content creation, and analytics.
        """
        
        analysis = crawler.analyze_content(test_content)
        print(f"✅ Content analysis: {analysis}")
        
        # Cleanup
        print("\n5. Cleaning up...")
        # Delete the job first due to foreign key constraint
        job = CrawlJob.query.get(job_id)
        if job:
            db.session.delete(job)
        db.session.delete(test_source)
        db.session.commit()
        print("✅ Test completed successfully!")

if __name__ == "__main__":
    test_crawler() 