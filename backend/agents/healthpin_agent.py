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

    def scrape_doctors_south_africa(self, limit: Optional[int] = None, progress_cb: Optional[Any] = None) -> Dict[str, Any]:
        """Scrape doctors in South Africa using Overpass (OpenStreetMap) API.

        Creates or updates Doctor records with a synthetic license key based on OSM element id.
        """
        try:
            overpass_endpoints = [
                "https://overpass-api.de/api/interpreter",
                "https://overpass.kumi.systems/api/interpreter",
                "https://overpass.openstreetmap.ru/api/interpreter"
            ]
            query = (
                "[out:json][timeout:180];"
                "area[\"ISO3166-1\"=\"ZA\"]->.searchArea;"
                "("
                "  node[\"amenity\"=\"doctors\"](area.searchArea);"
                "  node[\"healthcare\"=\"doctor\"](area.searchArea);"
                "  way[\"amenity\"=\"doctors\"](area.searchArea);"
                "  way[\"healthcare\"=\"doctor\"](area.searchArea);"
                "  relation[\"amenity\"=\"doctors\"](area.searchArea);"
                "  relation[\"healthcare\"=\"doctor\"](area.searchArea);"
                ");"
                "out center tags;"
            )

            last_error = None
            data = None
            for endpoint in overpass_endpoints:
                try:
                    resp = requests.post(endpoint, data={"data": query}, timeout=240)
                    if resp.status_code == 429:
                        last_error = f"Rate limited by Overpass endpoint: {endpoint}"
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    if not isinstance(data, dict) or 'elements' not in data:
                        last_error = f"Unexpected response from {endpoint}"
                        continue
                    break
                except Exception as e:
                    last_error = str(e)
                    continue
            if data is None:
                return {"success": False, "error": last_error or "Overpass query failed"}

            elements = data.get("elements", [])
            if limit and isinstance(limit, int) and limit > 0:
                elements = elements[:limit]

            total = max(1, len(elements))
            if progress_cb:
                progress_cb(10, {"stage": "fetched", "total": total})

            created = 0
            updated = 0
            skipped = 0

            # Import here to avoid circular imports at module load
            try:
                from backend.healthpin.models import Doctor
                from backend.models import db
            except ImportError:
                from healthpin.models import Doctor
                from models import db

            processed = 0
            for el in elements:
                tags = el.get("tags", {})
                if not tags:
                    skipped += 1
                    processed += 1
                    if progress_cb and processed % 50 == 0:
                        pct = 10 + int((processed / total) * 80)
                        progress_cb(min(90, pct), {"processed": processed, "created": created, "updated": updated, "skipped": skipped})
                    continue

                # Build synthetic unique license from OSM id
                osm_type = el.get("type", "node")
                osm_id = el.get("id")
                license_key = f"OSM-{osm_type}-{osm_id}"

                # Name and title parsing
                name = tags.get("name") or ""
                title, first_name, last_name = self._clean_name(name)

                # Location
                city = self._clean_city(tags.get("addr:city") or tags.get("addr:suburb") or "")
                province = self._clean_province(tags.get("addr:province") or tags.get("addr:state") or "")
                address_parts = [
                    tags.get("addr:housenumber"),
                    tags.get("addr:street"),
                    tags.get("addr:suburb"),
                    city,
                    province,
                    tags.get("addr:postcode"),
                ]
                address = ", ".join([p for p in address_parts if p]) or None

                # Contact
                phone = self._normalize_phone_za(tags.get("phone") or tags.get("contact:phone"))
                email = self._normalize_email(tags.get("email") or tags.get("contact:email"))

                # Coordinates
                lat = el.get("lat") or (el.get("center", {}) or {}).get("lat")
                lon = el.get("lon") or (el.get("center", {}) or {}).get("lon")

                # Specialties
                specialties = []
                for key in [
                    "healthcare:speciality",
                    "healthcare:specialty",
                    "specialty",
                    "medical:specialty",
                ]:
                    if tags.get(key):
                        specialties = [s.strip() for s in str(tags.get(key)).replace(";", ",").split(",") if s.strip()]
                        break

                # Practice name if different from person name
                practice_name = None
                if tags.get("operator"):
                    practice_name = tags.get("operator")
                elif tags.get("name") and not title:
                    # Might be a facility rather than a person
                    practice_name = tags.get("name")

                # Find existing doctor
                existing = Doctor.query.filter_by(medical_license=license_key).first()
                if existing:
                    existing.first_name = first_name or existing.first_name
                    existing.last_name = last_name or existing.last_name
                    existing.title = title or existing.title
                    existing.specialties = specialties or existing.specialties
                    existing.practice_name = practice_name or existing.practice_name
                    existing.city = city or existing.city
                    existing.province = province or existing.province
                    existing.address = address or existing.address
                    existing.phone = phone or existing.phone
                    existing.email = email or existing.email
                    existing.latitude = lat or existing.latitude
                    existing.longitude = lon or existing.longitude
                    updated += 1
                else:
                    doc = Doctor(
                        first_name=first_name,
                        last_name=last_name,
                        title=title,
                        medical_license=license_key,
                        specialties=specialties or [],
                        qualifications=[],
                        years_experience=None,
                        practice_name=practice_name,
                        practice_type=None,
                        languages_spoken=[],
                        city=city or "",
                        province=province or "",
                        address=address,
                        latitude=lat,
                        longitude=lon,
                        phone=phone,
                        email=email,
                        whatsapp_available=False,
                        consultation_fee=None,
                        accepts_medical_aid=False,
                        availability_schedule={},
                        patient_ratings={},
                        cultural_sensitivity_score=0.0,
                        accessibility_score=0.0,
                        communication_style=None,
                        is_verified=False,
                        is_active=True,
                    )
                    db.session.add(doc)
                    created += 1

                processed += 1
                if progress_cb and processed % 50 == 0:
                    pct = 10 + int((processed / total) * 80)
                    progress_cb(min(90, pct), {"processed": processed, "created": created, "updated": updated, "skipped": skipped})

            db.session.commit()

            if progress_cb:
                progress_cb(100, {"processed": processed, "created": created, "updated": updated, "skipped": skipped})
            return {
                "success": True,
                "created": created,
                "updated": updated,
                "skipped": skipped,
                "total_processed": len(elements)
            }
        except Exception as e:
            logger.error(f"Error scraping SA doctors: {e}")
            return {"success": False, "error": str(e)}

    def clean_doctor_data(self, dry_run: bool = False) -> Dict[str, Any]:
        """Normalize, dedupe, and validate existing doctor records."""
        try:
            try:
                from backend.healthpin.models import Doctor
                from backend.models import db
            except ImportError:
                from healthpin.models import Doctor
                from models import db

            doctors = Doctor.query.all()
            seen_licenses = set()
            updated = 0
            removed = 0

            for d in doctors:
                # Normalize names/title
                title, first, last = self._clean_name(f"{d.title or ''} {d.first_name or ''} {d.last_name or ''}".strip())
                d.title = title or d.title
                d.first_name = first or d.first_name
                d.last_name = last or d.last_name

                # Normalize phone/email
                d.phone = self._normalize_phone_za(d.phone)
                d.email = self._normalize_email(d.email)

                # Normalize city/province
                d.city = self._clean_city(d.city)
                d.province = self._clean_province(d.province)

                # Deduplicate by license
                if d.medical_license:
                    if d.medical_license in seen_licenses:
                        if not dry_run:
                            db.session.delete(d)
                        removed += 1
                        continue
                    seen_licenses.add(d.medical_license)

                updated += 1

            if not dry_run:
                db.session.commit()

            return {"success": True, "updated": updated, "removed": removed, "dry_run": dry_run}
        except Exception as e:
            logger.error(f"Error cleaning doctor data: {e}")
            return {"success": False, "error": str(e)}

    # ----------------------------
    # Cleaning/normalization utils
    # ----------------------------
    def _normalize_phone_za(self, phone: Optional[str]) -> Optional[str]:
        if not phone:
            return None
        p = str(phone)
        for ch in [" ", "(", ")", "-", "."]:
            p = p.replace(ch, "")
        if p.startswith("+27"):
            return p
        if p.startswith("27"):
            return "+" + p
        if p.startswith("0") and len(p) >= 10:
            return "+27" + p[1:]
        # Fallback: if digits length 9-12, assume missing +
        digits = ''.join([c for c in p if c.isdigit()])
        if len(digits) >= 9:
            if digits.startswith("27"):
                return "+" + digits
            return "+27" + digits[-9:]
        return None

    def _normalize_email(self, email: Optional[str]) -> Optional[str]:
        if not email:
            return None
        e = email.strip().lower()
        return e if "@" in e and "." in e.split("@")[-1] else None

    def _clean_name(self, full_name: str) -> (Optional[str], str, str):
        if not full_name:
            return None, "Unknown", ""
        n = full_name.strip()
        title = None
        lower = n.lower()
        if lower.startswith("dr. "):
            title = "Dr."
            n = n[4:].strip()
        elif lower.startswith("dr "):
            title = "Dr."
            n = n[3:].strip()
        parts = [p for p in n.split(" ") if p]
        first = parts[0].capitalize() if parts else "Unknown"
        last = " ".join([p.capitalize() for p in parts[1:]]) if len(parts) > 1 else ""
        return title, first, last

    def _clean_city(self, city: Optional[str]) -> Optional[str]:
        return city.strip().title() if city else None

    def _clean_province(self, province: Optional[str]) -> Optional[str]:
        if not province:
            return None
        p = province.strip().title()
        # Basic normalization for common SA provinces
        mapping = {
            "Kwazulu-Natal": "KwaZulu-Natal",
            "Wes-Kaap": "Western Cape",
            "Oos-Kaap": "Eastern Cape",
        }
        return mapping.get(p, p)

    # Run scrape as part of regular learning cycle so it runs in background
    def run_learning_cycle(self):
        try:
            self.scrape_doctors_south_africa()
        except Exception as e:
            logger.error(f"Doctor directory scrape failed in cycle: {e}")
        # Continue with normal collection/learning
        return super().run_learning_cycle()
    
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




