"""
HealthPIN AI Agent
=================

AI agent that continuously collects healthcare data and learns
clinical patterns, patient care insights, and medical trends.
"""

import os
import json
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import feedparser

from .base_agent import BaseAgent, AgentConfig, DataPoint

logger = logging.getLogger(__name__)

class HealthPINAgent(BaseAgent):
    """AI agent for HealthPIN section - collects healthcare data and learns clinical patterns"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Healthcare-specific data sources
        self.healthcare_sources = {
            "medical_news": [
                "https://www.medicalnewstoday.com/rss",
                "https://www.healthline.com/rss",
                "https://www.webmd.com/rss",
                "https://www.mayoclinic.org/rss"
            ],
            "research_feeds": [
                "https://pubmed.ncbi.nlm.nih.gov/rss/search/",
                "https://www.nejm.org/rss",
                "https://www.thelancet.com/rss"
            ],
            "healthcare_tech": [
                "https://www.healthcareitnews.com/rss",
                "https://www.mobihealthnews.com/rss",
                "https://www.himss.org/rss"
            ],
            "patient_care": [
                "https://www.ama-assn.org/rss",
                "https://www.aha.org/rss",
                "https://www.jointcommission.org/rss"
            ]
        }
        
        # Healthcare keywords for relevance scoring
        self.healthcare_keywords = [
            "healthcare", "medical", "patient", "clinical", "diagnosis", "treatment",
            "EHR", "electronic health record", "telemedicine", "digital health",
            "AI", "artificial intelligence", "machine learning", "analytics",
            "patient care", "outcomes", "safety", "quality", "protocols",
            "medication", "therapy", "surgery", "prevention", "wellness",
            "chronic disease", "mental health", "pediatrics", "geriatrics"
        ]
        
        # Medical specialties for categorization
        self.medical_specialties = [
            "cardiology", "oncology", "neurology", "orthopedics", "pediatrics",
            "geriatrics", "psychiatry", "dermatology", "endocrinology", "gastroenterology",
            "pulmonology", "nephrology", "urology", "gynecology", "ophthalmology"
        ]
        
        logger.info(f"🏥 HealthPIN agent initialized with {len(self.healthcare_sources)} data source categories")
    
    def _collect_from_source(self, source: str) -> List[Dict[str, Any]]:
        """Collect data from healthcare sources"""
        if source in self.healthcare_sources["medical_news"]:
            return self._collect_from_rss_feed(source)
        elif source in self.healthcare_sources["research_feeds"]:
            return self._collect_from_research_feed(source)
        elif source in self.healthcare_sources["healthcare_tech"]:
            return self._collect_from_tech_feed(source)
        elif source in self.healthcare_sources["patient_care"]:
            return self._collect_from_care_feed(source)
        else:
            logger.warning(f"Unknown healthcare source type: {source}")
            return []
    
    def _collect_from_rss_feed(self, feed_url: str) -> List[Dict[str, Any]]:
        """Collect data from medical news RSS feeds"""
        try:
            feed = feedparser.parse(feed_url)
            articles = []
            
            for entry in feed.entries[:10]:  # Limit to 10 most recent
                article = {
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "source": feed_url,
                    "type": "medical_news"
                }
                articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting from medical RSS feed {feed_url}: {e}")
            return []
    
    def _collect_from_research_feed(self, feed_url: str) -> List[Dict[str, Any]]:
        """Collect data from medical research feeds"""
        try:
            feed = feedparser.parse(feed_url)
            articles = []
            
            for entry in feed.entries[:5]:  # Limit to 5 most recent research articles
                article = {
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "source": feed_url,
                    "type": "research",
                    "authors": entry.get("authors", []),
                    "journal": entry.get("journal", "")
                }
                articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting from research feed {feed_url}: {e}")
            return []
    
    def _collect_from_tech_feed(self, feed_url: str) -> List[Dict[str, Any]]:
        """Collect data from healthcare technology feeds"""
        try:
            feed = feedparser.parse(feed_url)
            articles = []
            
            for entry in feed.entries[:8]:  # Limit to 8 most recent
                article = {
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "source": feed_url,
                    "type": "healthcare_tech"
                }
                articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting from tech feed {feed_url}: {e}")
            return []
    
    def _collect_from_care_feed(self, feed_url: str) -> List[Dict[str, Any]]:
        """Collect data from patient care and policy feeds"""
        try:
            feed = feedparser.parse(feed_url)
            articles = []
            
            for entry in feed.entries[:6]:  # Limit to 6 most recent
                article = {
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "source": feed_url,
                    "type": "patient_care"
                }
                articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting from care feed {feed_url}: {e}")
            return []
    
    def _process_data_item(self, item: Dict[str, Any], source: str) -> Optional[DataPoint]:
        """Process a healthcare data item"""
        try:
            # Extract content
            content = item.get("title", "") + " " + item.get("summary", "") + " " + item.get("content", "")
            content = content.strip()
            
            if not content or len(content) < 50:
                return None
            
            # Calculate relevance score based on healthcare keywords
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
                "authors": item.get("authors", []),
                "journal": item.get("journal", ""),
                "word_count": len(content.split()),
                "relevance_keywords": self._extract_relevant_keywords(content),
                "medical_specialties": self._identify_specialties(content)
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
            logger.error(f"Error processing healthcare data item: {e}")
            return None
    
    def _calculate_relevance_score(self, content: str) -> float:
        """Calculate relevance score for healthcare content"""
        content_lower = content.lower()
        score = 0.0
        
        # Check for healthcare keywords
        for keyword in self.healthcare_keywords:
            if keyword in content_lower:
                score += 0.1
        
        # Boost score for specific high-value terms
        high_value_terms = ["AI", "artificial intelligence", "EHR", "patient outcomes", "clinical decision"]
        for term in high_value_terms:
            if term.lower() in content_lower:
                score += 0.2
        
        # Boost score for medical specialties
        for specialty in self.medical_specialties:
            if specialty in content_lower:
                score += 0.15
        
        # Normalize score
        return min(score, 1.0)
    
    def _categorize_content(self, content: str) -> str:
        """Categorize content based on healthcare focus"""
        content_lower = content.lower()
        
        if any(term in content_lower for term in ["AI", "artificial intelligence", "machine learning", "digital health"]):
            return "Healthcare_Technology"
        elif any(term in content_lower for term in ["patient care", "clinical", "treatment", "diagnosis"]):
            return "Clinical_Care"
        elif any(term in content_lower for term in ["EHR", "electronic health record", "health information"]):
            return "Health_IT"
        elif any(term in content_lower for term in ["research", "study", "clinical trial", "evidence"]):
            return "Medical_Research"
        elif any(term in content_lower for term in ["policy", "regulation", "compliance", "quality"]):
            return "Healthcare_Policy"
        elif any(term in content_lower for term in ["outcomes", "analytics", "metrics", "performance"]):
            return "Healthcare_Analytics"
        else:
            return "General_Healthcare"
    
    def _extract_relevant_keywords(self, content: str) -> List[str]:
        """Extract relevant healthcare keywords from content"""
        content_lower = content.lower()
        relevant_keywords = []
        
        for keyword in self.healthcare_keywords:
            if keyword in content_lower:
                relevant_keywords.append(keyword)
        
        return relevant_keywords
    
    def _identify_specialties(self, content: str) -> List[str]:
        """Identify medical specialties mentioned in content"""
        content_lower = content.lower()
        specialties = []
        
        for specialty in self.medical_specialties:
            if specialty in content_lower:
                specialties.append(specialty)
        
        return specialties
    
    def _extract_insights(self, data_point: DataPoint) -> List[Dict[str, Any]]:
        """Extract insights from healthcare data"""
        insights = []
        
        # Analyze content for clinical insights
        content = data_point.content.lower()
        
        # AI/Technology insights
        if any(term in content for term in ["AI", "artificial intelligence", "machine learning"]):
            insights.append({
                "type": "AI_Healthcare",
                "insight": "AI technologies are transforming healthcare delivery and patient care",
                "confidence": 0.8,
                "category": data_point.category,
                "timestamp": datetime.utcnow().isoformat(),
                "source": data_point.source,
                "specialties": data_point.metadata.get("medical_specialties", [])
            })
        
        # EHR/Health IT insights
        if any(term in content for term in ["EHR", "electronic health record", "health information"]):
            insights.append({
                "type": "Health_IT",
                "insight": "Electronic health records and health IT systems are evolving rapidly",
                "confidence": 0.7,
                "category": data_point.category,
                "timestamp": datetime.utcnow().isoformat(),
                "source": data_point.source
            })
        
        # Patient care insights
        if any(term in content for term in ["patient care", "clinical outcomes", "quality"]):
            insights.append({
                "type": "Patient_Care",
                "insight": "Patient care quality and outcomes are being enhanced through technology",
                "confidence": 0.6,
                "category": data_point.category,
                "timestamp": datetime.utcnow().isoformat(),
                "source": data_point.source
            })
        
        # Telemedicine insights
        if any(term in content for term in ["telemedicine", "remote care", "virtual health"]):
            insights.append({
                "type": "Telemedicine",
                "insight": "Telemedicine and remote care are becoming standard practice",
                "confidence": 0.7,
                "category": data_point.category,
                "timestamp": datetime.utcnow().isoformat(),
                "source": data_point.source
            })
        
        return insights
    
    def _update_patterns(self, data_point: DataPoint) -> Dict[str, List[Dict[str, Any]]]:
        """Update patterns based on healthcare data"""
        patterns = {}
        
        # Clinical patterns
        if data_point.category == "Clinical_Care":
            patterns["Clinical_Trends"] = [{
                "pattern": "Clinical care evolution",
                "frequency": 1,
                "last_seen": datetime.utcnow().isoformat(),
                "confidence": data_point.relevance_score,
                "specialties": data_point.metadata.get("medical_specialties", [])
            }]
        
        # Technology patterns
        if data_point.category == "Healthcare_Technology":
            patterns["Tech_Trends"] = [{
                "pattern": "Healthcare technology adoption",
                "frequency": 1,
                "last_seen": datetime.utcnow().isoformat(),
                "confidence": data_point.relevance_score
            }]
        
        # Research patterns
        if data_point.category == "Medical_Research":
            patterns["Research_Trends"] = [{
                "pattern": "Medical research developments",
                "frequency": 1,
                "last_seen": datetime.utcnow().isoformat(),
                "confidence": data_point.relevance_score
            }]
        
        return patterns
    
    def get_healthcare_insights(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get healthcare-specific insights"""
        insights = self.get_insights()
        
        if category:
            return [insight for insight in insights if insight.get("category") == category]
        
        return insights
    
    def get_clinical_trends(self) -> Dict[str, Any]:
        """Get current clinical and healthcare trends"""
        patterns = self.get_patterns()
        
        trends = {
            "Clinical_Care": patterns.get("Clinical_Trends", []),
            "Technology": patterns.get("Tech_Trends", []),
            "Research": patterns.get("Research_Trends", []),
            "Policy": patterns.get("Policy_Trends", [])
        }
        
        return trends
    
    def get_specialty_insights(self, specialty: str) -> List[Dict[str, Any]]:
        """Get insights for a specific medical specialty"""
        insights = self.get_healthcare_insights()
        
        specialty_insights = []
        for insight in insights:
            specialties = insight.get("specialties", [])
            if specialty in specialties:
                specialty_insights.append(insight)
        
        return specialty_insights
    
    def get_clinical_recommendations(self) -> List[str]:
        """Get clinical recommendations based on learned patterns"""
        insights = self.get_healthcare_insights()
        recommendations = []
        
        # Analyze insights for recommendations
        ai_insights = [i for i in insights if i.get("type") == "AI_Healthcare"]
        if len(ai_insights) > 2:
            recommendations.append("Consider implementing AI tools for clinical decision support and patient care")
        
        ehr_insights = [i for i in insights if i.get("type") == "Health_IT"]
        if len(ehr_insights) > 1:
            recommendations.append("Evaluate EHR optimization and health IT integration opportunities")
        
        telemedicine_insights = [i for i in insights if i.get("type") == "Telemedicine"]
        if len(telemedicine_insights) > 1:
            recommendations.append("Develop telemedicine capabilities for improved patient access")
        
        care_insights = [i for i in insights if i.get("type") == "Patient_Care"]
        if len(care_insights) > 1:
            recommendations.append("Focus on patient-centered care models and outcome measurement")
        
        return recommendations
    
    def get_patient_care_protocols(self) -> List[Dict[str, Any]]:
        """Get patient care protocol recommendations"""
        insights = self.get_healthcare_insights("Clinical_Care")
        
        protocols = []
        for insight in insights:
            if insight.get("type") == "Patient_Care":
                protocols.append({
                    "protocol": insight.get("insight", ""),
                    "confidence": insight.get("confidence", 0.0),
                    "source": insight.get("source", ""),
                    "timestamp": insight.get("timestamp", "")
                })
        
        return protocols




