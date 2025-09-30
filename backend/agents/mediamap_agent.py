"""
MediaMap AI Agent
================

AI agent that continuously collects media industry data and learns
business patterns, trends, and insights for the MediaMap section.
"""

import os
import json
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
import re
from bs4 import BeautifulSoup
import feedparser

from .base_agent import BaseAgent, AgentConfig, DataPoint

logger = logging.getLogger(__name__)

class MediaMapAgent(BaseAgent):
    """AI agent for MediaMap section - collects media industry data and learns business patterns"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Media-specific data sources
        self.media_sources = {
            "news_feeds": [
                "https://feeds.feedburner.com/oreilly/radar",
                "https://www.niemanlab.org/feed/",
                "https://www.poynter.org/feed/",
                "https://www.journalism.co.uk/feed/",
                "https://www.mediapost.com/rss/"
            ],
            "industry_news": [
                "https://www.mediapost.com/",
                "https://www.niemanlab.org/",
                "https://www.poynter.org/",
                "https://www.journalism.co.uk/"
            ],
            "social_media": [
                "twitter_media_hashtags",
                "linkedin_media_groups"
            ]
        }
        
        # Media industry keywords for relevance scoring
        self.media_keywords = [
            "media", "journalism", "news", "publishing", "broadcasting",
            "digital media", "content creation", "audience engagement",
            "revenue", "subscription", "advertising", "monetization",
            "AI", "artificial intelligence", "automation", "workflow",
            "analytics", "data", "insights", "performance", "metrics"
        ]
        
        logger.info(f"📰 MediaMap agent initialized with {len(self.media_sources)} data source categories")

    # -----------------------------
    # Data Cleaning for MediaMap
    # -----------------------------
    def clean_existing_data(self, dry_run: bool = False) -> Dict[str, Any]:
        """Clean and deduplicate existing collected data stored in the agent's data file.

        - Deduplicate by normalized URL and normalized title
        - Normalize titles (trim, fix whitespace)
        - Canonicalize URLs (lowercase scheme/host, remove fragments)
        - Drop entries with too-short content
        """
        try:
            # Load existing
            if not os.path.exists(self.data_file):
                return {"success": True, "kept": 0, "removed": 0, "deduped": 0}

            with open(self.data_file, 'r') as f:
                items = json.load(f) or []

            original_count = len(items)

            def normalize_title(title: str) -> str:
                if not title:
                    return ""
                t = re.sub(r"\s+", " ", title).strip()
                return t

            def canonical_url(url: str) -> str:
                if not url:
                    return ""
                try:
                    p = urlparse(url)
                    # Normalize scheme/host, drop fragment
                    scheme = (p.scheme or 'https').lower()
                    netloc = (p.netloc or '').lower()
                    path = p.path or '/'
                    query = f"?{p.query}" if p.query else ''
                    return f"{scheme}://{netloc}{path}{query}"
                except Exception:
                    return url

            seen_urls = set()
            seen_titles = set()
            cleaned = []
            deduped = 0
            removed_short = 0

            for it in items:
                title = normalize_title(it.get('content') or it.get('title') or '')
                url = canonical_url(it.get('metadata', {}).get('url') or it.get('link') or '')

                # Filter short content
                content = (it.get('content') or '').strip()
                if len(content) < 50 and len(title) < 20:
                    removed_short += 1
                    continue

                # Deduplicate by URL first, fallback to title
                key = None
                if url:
                    key = ('url', url)
                elif title:
                    key = ('title', title.lower())

                if key:
                    if key in seen_urls or key in seen_titles:
                        deduped += 1
                        continue
                    if key[0] == 'url':
                        seen_urls.add(key)
                    else:
                        seen_titles.add(key)

                # Write back normalized fields
                it['content'] = content if content else title
                if 'metadata' not in it:
                    it['metadata'] = {}
                if url:
                    it['metadata']['url'] = url
                # Keep normalized title as part of content/metadata for usability
                it['metadata']['normalized_title'] = title

                cleaned.append(it)

            kept = len(cleaned)
            removed = original_count - kept

            if not dry_run:
                with open(self.data_file, 'w') as f:
                    json.dump(cleaned, f, indent=2)

            return {
                "success": True,
                "original": original_count,
                "kept": kept,
                "removed": removed,
                "deduped": deduped,
                "removed_short": removed_short,
                "dry_run": dry_run
            }
        except Exception as e:
            logger.error(f"Error cleaning MediaMap data: {e}")
            return {"success": False, "error": str(e)}

    def run_learning_cycle(self):
        # Clean before regular cycle to keep store healthy
        try:
            self.clean_existing_data(dry_run=False)
        except Exception as e:
            logger.warning(f"MediaMap clean failed in cycle: {e}")
        return super().run_learning_cycle()
    
    def _collect_from_source(self, source: str) -> List[Dict[str, Any]]:
        """Collect data from media industry sources"""
        # Check if it's an RSS feed (most common case)
        if source.endswith('/feed/') or source.endswith('/rss') or 'feedburner.com' in source or 'rss' in source.lower():
            return self._collect_from_rss_feed(source)
        elif source in self.media_sources["news_feeds"]:
            return self._collect_from_rss_feed(source)
        elif source in self.media_sources["industry_news"]:
            return self._collect_from_news_site(source)
        elif source in self.media_sources["social_media"]:
            return self._collect_from_social_media(source)
        else:
            # Default to RSS feed collection for unknown sources
            logger.info(f"Treating unknown source as RSS feed: {source}")
            return self._collect_from_rss_feed(source)
    
    def _collect_from_rss_feed(self, feed_url: str) -> List[Dict[str, Any]]:
        """Collect data from RSS feeds"""
        try:
            logger.info(f"📡 Fetching RSS feed: {feed_url}")
            feed = feedparser.parse(feed_url)
            articles = []
            
            if not feed.entries:
                logger.warning(f"⚠️ No entries found in RSS feed: {feed_url}")
                return []
            
            for entry in feed.entries[:10]:  # Limit to 10 most recent
                article = {
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", "") or entry.get("description", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "source": feed_url,
                    "type": "rss_feed",
                    "author": entry.get("author", ""),
                    "tags": [tag.get("term", "") for tag in entry.get("tags", [])]
                }
                articles.append(article)
                logger.info(f"📰 Collected: {article['title'][:50]}...")
            
            logger.info(f"✅ Collected {len(articles)} articles from RSS feed: {feed_url}")
            return articles
            
        except Exception as e:
            logger.error(f"❌ Error collecting from RSS feed {feed_url}: {e}")
            return []
    
    def _collect_from_news_site(self, site_url: str) -> List[Dict[str, Any]]:
        """Collect data from news websites"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(site_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Look for article links and titles
            article_links = soup.find_all('a', href=True)
            
            for link in article_links[:15]:  # Limit to 15 articles
                href = link.get('href')
                title = link.get_text(strip=True)
                
                if href and title and len(title) > 20:
                    # Make sure it's a full URL
                    if href.startswith('/'):
                        href = urljoin(site_url, href)
                    
                    article = {
                        "title": title,
                        "link": href,
                        "source": site_url,
                        "type": "news_site",
                        "published": datetime.utcnow().isoformat()
                    }
                    articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting from news site {site_url}: {e}")
            return []
    
    def _collect_from_social_media(self, source: str) -> List[Dict[str, Any]]:
        """Collect data from social media (placeholder for API integration)"""
        # This would integrate with Twitter API, LinkedIn API, etc.
        # For now, return sample data
        if source == "twitter_media_hashtags":
            return [
                {
                    "content": "AI is transforming media workflows and content creation",
                    "source": "twitter",
                    "type": "social_media",
                    "hashtags": ["#AI", "#MediaTech", "#DigitalTransformation"],
                    "published": datetime.utcnow().isoformat()
                }
            ]
        elif source == "linkedin_media_groups":
            return [
                {
                    "content": "Media companies are investing heavily in AI and automation",
                    "source": "linkedin",
                    "type": "social_media",
                    "published": datetime.utcnow().isoformat()
                }
            ]
        
        return []
    
    def _process_data_item(self, item: Dict[str, Any], source: str) -> Optional[DataPoint]:
        """Process a media industry data item"""
        try:
            # Extract content
            content = item.get("title", "") + " " + item.get("summary", "") + " " + item.get("content", "")
            content = content.strip()
            
            if not content or len(content) < 50:
                return None
            
            # Calculate relevance score based on media keywords
            relevance_score = self._calculate_relevance_score(content)
            
            if relevance_score < 0.3:
                return None
            
            # Determine category
            category = self._categorize_content(content)
            
            # Extract metadata
            metadata = {
                "source_type": item.get("type", "unknown"),
                "url": item.get("link", ""),
                "published": item.get("published", ""),
                "hashtags": item.get("hashtags", []),
                "word_count": len(content.split()),
                "relevance_keywords": self._extract_relevant_keywords(content)
            }
            
            return DataPoint(
                source=source,
                content=content,
                metadata=metadata,
                timestamp=datetime.utcnow(),
                relevance_score=relevance_score,
                category=category
            )
            
        except Exception as e:
            logger.error(f"Error processing data item: {e}")
            return None
    
    def _calculate_relevance_score(self, content: str) -> float:
        """Calculate relevance score for media industry content"""
        content_lower = content.lower()
        score = 0.0
        
        # Check for media keywords
        for keyword in self.media_keywords:
            if keyword in content_lower:
                score += 0.1
        
        # Boost score for specific high-value terms
        high_value_terms = ["AI", "artificial intelligence", "automation", "revenue", "subscription"]
        for term in high_value_terms:
            if term.lower() in content_lower:
                score += 0.2
        
        # Normalize score
        return min(score, 1.0)
    
    def _categorize_content(self, content: str) -> str:
        """Categorize content based on its focus"""
        content_lower = content.lower()
        
        if any(term in content_lower for term in ["AI", "artificial intelligence", "automation", "machine learning"]):
            return "AI_Technology"
        elif any(term in content_lower for term in ["revenue", "monetization", "subscription", "advertising"]):
            return "Business_Model"
        elif any(term in content_lower for term in ["audience", "engagement", "analytics", "metrics"]):
            return "Audience_Analytics"
        elif any(term in content_lower for term in ["workflow", "process", "efficiency", "productivity"]):
            return "Operations"
        elif any(term in content_lower for term in ["content", "creation", "publishing", "distribution"]):
            return "Content_Strategy"
        else:
            return "General_Media"
    
    def _extract_relevant_keywords(self, content: str) -> List[str]:
        """Extract relevant keywords from content"""
        content_lower = content.lower()
        relevant_keywords = []
        
        for keyword in self.media_keywords:
            if keyword in content_lower:
                relevant_keywords.append(keyword)
        
        return relevant_keywords
    
    def _extract_insights(self, data_point: DataPoint) -> List[Dict[str, Any]]:
        """Extract insights from media industry data"""
        insights = []
        
        # Analyze content for business insights
        content = data_point.content.lower()
        
        # AI/Technology insights
        if any(term in content for term in ["AI", "artificial intelligence", "automation"]):
            insights.append({
                "type": "AI_Adoption",
                "insight": "Media companies are increasingly adopting AI technologies",
                "confidence": 0.8,
                "category": data_point.category,
                "timestamp": datetime.utcnow().isoformat(),
                "source": data_point.source
            })
        
        # Revenue model insights
        if any(term in content for term in ["subscription", "paywall", "membership"]):
            insights.append({
                "type": "Revenue_Model",
                "insight": "Subscription-based revenue models are gaining traction in media",
                "confidence": 0.7,
                "category": data_point.category,
                "timestamp": datetime.utcnow().isoformat(),
                "source": data_point.source
            })
        
        # Audience engagement insights
        if any(term in content for term in ["engagement", "audience", "analytics"]):
            insights.append({
                "type": "Audience_Engagement",
                "insight": "Data-driven audience engagement strategies are becoming critical",
                "confidence": 0.6,
                "category": data_point.category,
                "timestamp": datetime.utcnow().isoformat(),
                "source": data_point.source
            })
        
        return insights
    
    def _update_patterns(self, data_point: DataPoint) -> Dict[str, List[Dict[str, Any]]]:
        """Update patterns based on media industry data"""
        patterns = {}
        
        # Trend patterns
        if data_point.category == "AI_Technology":
            patterns["AI_Trends"] = [{
                "pattern": "AI adoption in media",
                "frequency": 1,
                "last_seen": datetime.utcnow().isoformat(),
                "confidence": data_point.relevance_score
            }]
        
        # Business model patterns
        if data_point.category == "Business_Model":
            patterns["Revenue_Trends"] = [{
                "pattern": "Revenue model evolution",
                "frequency": 1,
                "last_seen": datetime.utcnow().isoformat(),
                "confidence": data_point.relevance_score
            }]
        
        # Content strategy patterns
        if data_point.category == "Content_Strategy":
            patterns["Content_Trends"] = [{
                "pattern": "Content strategy evolution",
                "frequency": 1,
                "last_seen": datetime.utcnow().isoformat(),
                "confidence": data_point.relevance_score
            }]
        
        return patterns
    
    def get_media_insights(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get media-specific insights"""
        insights = self.get_insights()
        
        if category:
            return [insight for insight in insights if insight.get("category") == category]
        
        return insights
    
    def get_industry_trends(self) -> Dict[str, Any]:
        """Get current media industry trends"""
        patterns = self.get_patterns()
        
        trends = {
            "AI_Adoption": patterns.get("AI_Trends", []),
            "Revenue_Models": patterns.get("Revenue_Trends", []),
            "Content_Strategies": patterns.get("Content_Trends", []),
            "Audience_Engagement": patterns.get("Audience_Trends", [])
        }
        
        return trends
    
    def get_business_recommendations(self) -> List[str]:
        """Get business recommendations based on learned patterns"""
        insights = self.get_media_insights()
        recommendations = []
        
        # Analyze insights for recommendations
        ai_insights = [i for i in insights if i.get("type") == "AI_Adoption"]
        if len(ai_insights) > 2:
            recommendations.append("Consider implementing AI tools for content creation and workflow automation")
        
        revenue_insights = [i for i in insights if i.get("type") == "Revenue_Model"]
        if len(revenue_insights) > 1:
            recommendations.append("Evaluate subscription-based revenue models for sustainable growth")
        
        audience_insights = [i for i in insights if i.get("type") == "Audience_Engagement"]
        if len(audience_insights) > 1:
            recommendations.append("Invest in audience analytics and engagement tools")
        
        return recommendations




