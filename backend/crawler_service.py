#!/usr/bin/env python3

import requests
import json
import re
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import feedparser
import time
import logging
from typing import List, Dict, Optional, Tuple
import openai
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentCrawler:
    """Service for crawling content from various sources"""
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def crawl_website(self, url: str, max_pages: int = 10) -> List[Dict]:
        """Crawl a website for content"""
        try:
            logger.info(f"Crawling website: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Look for common article selectors
            selectors = [
                'article', '.post', '.entry', '.content', '.article',
                '[class*="post"]', '[class*="article"]', '[class*="entry"]'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    logger.info(f"Found {len(elements)} elements with selector: {selector}")
                    break
            
            if not elements:
                # Fallback: look for any div with substantial text
                elements = soup.find_all('div', class_=re.compile(r'post|article|entry|content'))
            
            for element in elements[:max_pages]:
                try:
                    # Extract title
                    title_elem = element.find(['h1', 'h2', 'h3', 'h4'])
                    title = title_elem.get_text(strip=True) if title_elem else "Untitled"
                    
                    # Extract content
                    content_elem = element.find(['p', 'div'])
                    content = content_elem.get_text(strip=True) if content_elem else ""
                    
                    # Extract link
                    link_elem = element.find('a')
                    link = urljoin(url, link_elem['href']) if link_elem and link_elem.get('href') else url
                    
                    if title and content and len(content) > 100:  # Minimum content length
                        articles.append({
                            'title': title,
                            'content': content,
                            'url': link,
                            'published_date': None,  # Would need more sophisticated parsing
                            'source_type': 'website'
                        })
                
                except Exception as e:
                    logger.warning(f"Error processing element: {e}")
                    continue
            
            logger.info(f"Extracted {len(articles)} articles from {url}")
            return articles
            
        except Exception as e:
            logger.error(f"Error crawling website {url}: {e}")
            return []
    
    def crawl_rss(self, rss_url: str) -> List[Dict]:
        """Crawl RSS feed for content"""
        try:
            logger.info(f"Crawling RSS feed: {rss_url}")
            feed = feedparser.parse(rss_url)
            
            articles = []
            for entry in feed.entries:
                try:
                    # Extract content
                    content = ""
                    if hasattr(entry, 'content'):
                        content = entry.content[0].value
                    elif hasattr(entry, 'summary'):
                        content = entry.summary
                    elif hasattr(entry, 'description'):
                        content = entry.description
                    
                    # Clean HTML from content
                    if content:
                        soup = BeautifulSoup(content, 'html.parser')
                        content = soup.get_text(strip=True)
                    
                    # Parse published date
                    published_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_date = datetime(*entry.published_parsed[:6])
                    
                    articles.append({
                        'title': entry.title,
                        'content': content,
                        'url': entry.link,
                        'published_date': published_date,
                        'source_type': 'rss'
                    })
                
                except Exception as e:
                    logger.warning(f"Error processing RSS entry: {e}")
                    continue
            
            logger.info(f"Extracted {len(articles)} articles from RSS feed")
            return articles
            
        except Exception as e:
            logger.error(f"Error crawling RSS feed {rss_url}: {e}")
            return []
    
    def crawl_newsletter_archive(self, archive_url: str) -> List[Dict]:
        """Crawl newsletter archive (Substack, etc.)"""
        try:
            logger.info(f"Crawling newsletter archive: {archive_url}")
            
            # This is a simplified version - would need to be customized per platform
            if 'substack.com' in archive_url:
                return self._crawl_substack(archive_url)
            else:
                # Fallback to general website crawling
                return self.crawl_website(archive_url)
                
        except Exception as e:
            logger.error(f"Error crawling newsletter archive {archive_url}: {e}")
            return []
    
    def _crawl_substack(self, substack_url: str) -> List[Dict]:
        """Crawl Substack newsletter"""
        try:
            response = self.session.get(substack_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Look for Substack post elements
            post_elements = soup.find_all('div', class_=re.compile(r'post|entry'))
            
            for element in post_elements:
                try:
                    # Extract title
                    title_elem = element.find(['h1', 'h2', 'h3'])
                    title = title_elem.get_text(strip=True) if title_elem else "Untitled"
                    
                    # Extract content preview
                    content_elem = element.find(['p', 'div'])
                    content = content_elem.get_text(strip=True) if content_elem else ""
                    
                    # Extract link
                    link_elem = element.find('a')
                    link = urljoin(substack_url, link_elem['href']) if link_elem and link_elem.get('href') else substack_url
                    
                    if title and content:
                        articles.append({
                            'title': title,
                            'content': content,
                            'url': link,
                            'published_date': None,
                            'source_type': 'newsletter'
                        })
                
                except Exception as e:
                    logger.warning(f"Error processing Substack element: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error crawling Substack {substack_url}: {e}")
            return []
    
    def analyze_content(self, content: str) -> Dict:
        """Use AI to analyze content and extract strategies, use cases, and code examples"""
        if not self.openai_api_key:
            logger.warning("OpenAI API key not available, skipping content analysis")
            return {
                'content_type': 'article',
                'tags': [],
                'summary': content[:200] + "..." if len(content) > 200 else content,
                'relevance_score': 0.5
            }
        
        try:
            prompt = f"""
            Analyze the following content and extract:
            1. Content type (strategy, use_case, code_example, article)
            2. Relevant tags (AI, automation, media, business, etc.)
            3. Brief summary
            4. Relevance score (0.0-1.0) for media business AI applications
            
            Content:
            {content[:2000]}  # Limit content length for API
            
            Respond in JSON format:
            {{
                "content_type": "strategy|use_case|code_example|article",
                "tags": ["tag1", "tag2"],
                "summary": "Brief summary",
                "relevance_score": 0.85
            }}
            """
            
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI content analyzer specializing in media business and AI applications."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            try:
                result = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                # Fallback if AI doesn't return valid JSON
                logger.warning("AI response not in valid JSON format, using fallback")
                result = {
                    'content_type': 'article',
                    'tags': [],
                    'summary': response.choices[0].message.content[:200] + "..." if len(response.choices[0].message.content) > 200 else response.choices[0].message.content,
                    'relevance_score': 0.5
                }
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing content with AI: {e}")
            return {
                'content_type': 'article',
                'tags': [],
                'summary': content[:200] + "..." if len(content) > 200 else content,
                'relevance_score': 0.5
            }
    
    def crawl_source(self, source: Dict) -> List[Dict]:
        """Crawl a specific source based on its type"""
        source_type = source.get('source_type', 'website')
        url = source.get('url')
        
        if not url:
            logger.error("No URL provided for source")
            return []
        
        try:
            if source_type == 'rss':
                articles = self.crawl_rss(url)
            elif source_type == 'newsletter':
                articles = self.crawl_newsletter_archive(url)
            else:  # website
                articles = self.crawl_website(url)
            
            # Analyze each article with AI
            for article in articles:
                analysis = self.analyze_content(article['content'])
                article.update(analysis)
                article['is_processed'] = True
            
            return articles
            
        except Exception as e:
            logger.error(f"Error crawling source {url}: {e}")
            return []


class CrawlManager:
    """Manager for coordinating crawl operations"""
    
    def __init__(self, db_session, openai_api_key: str = None):
        self.db = db_session
        self.crawler = ContentCrawler(openai_api_key)
    
    def create_crawl_job(self, source_id: int) -> int:
        """Create a new crawl job"""
        from backend.models import CrawlJob
        
        job = CrawlJob(
            source_id=source_id,
            status='pending',
            created_at=datetime.utcnow()
        )
        self.db.session.add(job)
        self.db.session.commit()
        return job.id
    
    def run_crawl_job(self, job_id: int) -> bool:
        """Run a specific crawl job"""
        from backend.models import CrawlJob, CrawlSource, CrawledContent
        
        try:
            job = CrawlJob.query.get(job_id)
            if not job:
                logger.error(f"Crawl job {job_id} not found")
                return False
            
            source = CrawlSource.query.get(job.source_id)
            if not source:
                logger.error(f"Source {job.source_id} not found")
                return False
            
            # Update job status
            job.status = 'running'
            job.started_at = datetime.utcnow()
            self.db.session.commit()
            
            # Crawl the source
            articles = self.crawler.crawl_source(source.to_dict())
            
            # Save crawled content
            items_processed = 0
            for article in articles:
                try:
                    # Check if content already exists (basic deduplication)
                    existing = CrawledContent.query.filter_by(
                        source_id=source.id,
                        title=article['title']
                    ).first()
                    
                    if not existing:
                        crawled_content = CrawledContent(
                            source_id=source.id,
                            title=article['title'],
                            content=article['content'],
                            url=article.get('url'),
                            published_date=article.get('published_date'),
                            content_type=article.get('content_type', 'article'),
                            tags=json.dumps(article.get('tags', [])),
                            summary=article.get('summary'),
                            relevance_score=article.get('relevance_score', 0.0),
                            is_processed=article.get('is_processed', False)
                        )
                        self.db.session.add(crawled_content)
                        items_processed += 1
                
                except Exception as e:
                    logger.error(f"Error saving crawled content: {e}")
                    continue
            
            # Update job status
            job.status = 'completed'
            job.completed_at = datetime.utcnow()
            job.items_found = len(articles)
            job.items_processed = items_processed
            
            # Update source last_crawled
            source.last_crawled = datetime.utcnow()
            
            self.db.session.commit()
            logger.info(f"Crawl job {job_id} completed: {items_processed} items processed")
            return True
            
        except Exception as e:
            logger.error(f"Error running crawl job {job_id}: {e}")
            
            # Update job status to failed
            job = CrawlJob.query.get(job_id)
            if job:
                job.status = 'failed'
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                self.db.session.commit()
            
            return False
    
    def get_crawl_stats(self) -> Dict:
        """Get crawling statistics"""
        from backend.models import CrawlSource, CrawledContent, CrawlJob
        
        total_sources = CrawlSource.query.count()
        active_sources = CrawlSource.query.filter_by(is_active=True).count()
        total_content = CrawledContent.query.count()
        recent_jobs = CrawlJob.query.filter(
            CrawlJob.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        ).count()
        
        return {
            'total_sources': total_sources,
            'active_sources': active_sources,
            'total_content': total_content,
            'recent_jobs': recent_jobs
        } 