"""
Organization Discovery API Service
Real integrations with external APIs for finding organizations
"""

import requests
import json
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiscoveryAPIService:
    """Service for discovering organizations through various APIs"""
    
    def __init__(self):
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.linkedin_api_key = os.getenv('LINKEDIN_API_KEY')
        self.twitter_api_key = os.getenv('TWITTER_API_KEY')
        self.crunchbase_api_key = os.getenv('CRUNCHBASE_API_KEY')
        
    def discover_from_news(self, keywords: str, sector: str = None, max_results: int = 25) -> List[Dict]:
        """Discover organizations from news articles using News API"""
        if not self.news_api_key:
            logger.warning("News API key not configured")
            return []
            
        try:
            # Build query with sector-specific keywords
            query = keywords
            if sector:
                query += f" AND {sector}"
                
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'apiKey': self.news_api_key,
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': min(max_results, 100),
                'from': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            organizations = []
            
            for article in data.get('articles', []):
                # Extract organization mentions from article
                org_info = self._extract_organization_from_article(article, sector)
                if org_info:
                    organizations.append(org_info)
                    
            logger.info(f"Found {len(organizations)} organizations from news")
            return organizations[:max_results]
            
        except Exception as e:
            logger.error(f"Error fetching from News API: {e}")
            return []
    
    def discover_from_linkedin(self, keywords: str, sector: str = None, max_results: int = 25) -> List[Dict]:
        """Discover organizations from LinkedIn using LinkedIn Marketing API"""
        if not self.linkedin_api_key:
            logger.warning("LinkedIn API key not configured")
            return []
            
        try:
            # LinkedIn Marketing API v2 for company discovery
            url = "https://api.linkedin.com/v2/adTargetingEntities"
            headers = {
                'Authorization': f'Bearer {self.linkedin_api_key}',
                'Content-Type': 'application/json',
                'X-Restli-Protocol-Version': '2.0.0'
            }
            
            params = {
                'q': 'criteria',
                'type': 'COMPANY',
                'keywords': keywords,
                'count': min(max_results, 100)
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            organizations = []
            
            for company in data.get('elements', []):
                org_info = self._extract_organization_from_linkedin(company, sector)
                if org_info:
                    organizations.append(org_info)
                    
            logger.info(f"Found {len(organizations)} organizations from LinkedIn")
            return organizations[:max_results]
            
        except Exception as e:
            logger.error(f"Error fetching from LinkedIn API: {e}")
            return []
    
    def _extract_organization_from_linkedin(self, company_data: Dict, sector: str = None) -> Optional[Dict]:
        """Extract organization information from LinkedIn data"""
        try:
            properties = company_data.get('targetingCriteria', {})
            
            company_name = properties.get('companyName', 'Unknown')
            industry = properties.get('industry', 'Unknown')
            company_size = properties.get('companySize', 'Unknown')
            
            # Calculate relevance score
            relevance_score = self._calculate_linkedin_relevance(properties, sector)
            
            return {
                'name': company_name,
                'source': 'LinkedIn',
                'score': f"{relevance_score}%",
                'signals': self._extract_linkedin_signals(properties),
                'sector': sector or industry,
                'estimatedSize': company_size,
                'url': f"https://linkedin.com/company/{company_data.get('id', '')}",
                'industry': industry,
                'followers': properties.get('followerCount', 0)
            }
            
        except Exception as e:
            logger.error(f"Error extracting organization from LinkedIn: {e}")
            return None
    
    def _calculate_linkedin_relevance(self, properties: Dict, target_sector: str = None) -> int:
        """Calculate relevance score for LinkedIn organizations"""
        score = 60  # Base score for LinkedIn data
        
        if target_sector:
            industry = properties.get('industry', '').lower()
            if target_sector.lower() in industry:
                score += 25
        
        # Boost score for companies with followers
        followers = properties.get('followerCount', 0)
        if followers > 10000:
            score += 15
        elif followers > 1000:
            score += 10
        
        return min(score, 100)
    
    def _extract_linkedin_signals(self, properties: Dict) -> str:
        """Extract business signals from LinkedIn data"""
        signals = []
        
        industry = properties.get('industry', '').lower()
        if any(tech in industry for tech in ['technology', 'software', 'artificial intelligence']):
            signals.append('Technology Company')
        
        company_size = properties.get('companySize', '')
        if '1000+' in company_size or '500+' in company_size:
            signals.append('Large Enterprise')
        
        return ', '.join(signals) if signals else 'Professional Network Company'
    
    def discover_from_twitter(self, keywords: str, sector: str = None, max_results: int = 25) -> List[Dict]:
        """Discover organizations from Twitter using Twitter API v2"""
        if not self.twitter_api_key:
            logger.warning("Twitter API key not configured")
            return []
            
        try:
            # Twitter API v2 for company mentions and hashtags
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {
                'Authorization': f'Bearer {self.twitter_api_key}',
                'Content-Type': 'application/json'
            }
            
            query = f"{keywords} (company OR organization OR business)"
            if sector:
                query += f" {sector}"
            
            params = {
                'query': query,
                'max_results': min(max_results, 100),
                'tweet.fields': 'author_id,created_at,public_metrics',
                'expansions': 'author_id',
                'user.fields': 'name,username,description,verified'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            organizations = []
            
            # Extract company mentions from tweets
            for tweet in data.get('data', []):
                org_info = self._extract_organization_from_tweet(tweet, sector)
                if org_info:
                    organizations.append(org_info)
                    
            logger.info(f"Found {len(organizations)} organizations from Twitter")
            return organizations[:max_results]
            
        except Exception as e:
            logger.error(f"Error fetching from Twitter API: {e}")
            return []
    
    def _extract_organization_from_tweet(self, tweet_data: Dict, sector: str = None) -> Optional[Dict]:
        """Extract organization information from Twitter data"""
        try:
            text = tweet_data.get('text', '')
            
            # Extract company names from tweet text
            company_names = self._extract_company_names(text)
            if not company_names:
                return None
                
            company_name = company_names[0]
            
            # Calculate relevance score
            relevance_score = self._calculate_twitter_relevance(text, sector)
            
            return {
                'name': company_name,
                'source': 'Twitter',
                'score': f"{relevance_score}%",
                'signals': self._extract_twitter_signals(text),
                'sector': sector or self._detect_sector_from_content(text),
                'estimatedSize': self._estimate_company_size(text),
                'url': f"https://twitter.com/user/status/{tweet_data.get('id')}",
                'tweet_text': text[:200] + '...' if len(text) > 200 else text,
                'engagement': tweet_data.get('public_metrics', {}).get('retweet_count', 0)
            }
            
        except Exception as e:
            logger.error(f"Error extracting organization from tweet: {e}")
            return None
    
    def _calculate_twitter_relevance(self, text: str, target_sector: str = None) -> int:
        """Calculate relevance score for Twitter mentions"""
        score = 50  # Base score for Twitter data
        
        if target_sector:
            sector_keywords = {
                'Media': ['news', 'journalism', 'media', 'broadcasting'],
                'Communications': ['pr', 'public relations', 'communications', 'marketing'],
                'Technology': ['technology', 'software', 'ai', 'machine learning']
            }
            
            if target_sector in sector_keywords:
                for keyword in sector_keywords[target_sector]:
                    if keyword in text.lower():
                        score += 20
                        break
        
        # Boost score for AI-related content
        ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'automation']
        for keyword in ai_keywords:
            if keyword in text.lower():
                score += 15
                break
        
        return min(score, 100)
    
    def _extract_twitter_signals(self, text: str) -> str:
        """Extract business signals from Twitter content"""
        signals = []
        
        text_lower = text.lower()
        if any(word in text_lower for word in ['ai', 'artificial intelligence', 'machine learning']):
            signals.append('AI Technology Focus')
        
        if any(word in text_lower for word in ['digital transformation', 'innovation', 'tech adoption']):
            signals.append('Digital Innovation')
        
        if any(word in text_lower for word in ['hiring', 'careers', 'job openings']):
            signals.append('Growth & Hiring')
        
        return ', '.join(signals) if signals else 'Social Media Presence'
    
    def discover_from_crunchbase(self, keywords: str, sector: str = None, max_results: int = 25) -> List[Dict]:
        """Discover organizations from Crunchbase"""
        if not self.crunchbase_api_key:
            logger.warning("Crunchbase API key not configured")
            return []
            
        try:
            # Crunchbase API integration
            url = "https://api.crunchbase.com/v3.1/organizations"
            params = {
                'user_key': self.crunchbase_api_key,
                'name': keywords,
                'limit': max_results
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            organizations = []
            
            for org in data.get('data', {}).get('items', []):
                org_info = self._extract_organization_from_crunchbase(org, sector)
                if org_info:
                    organizations.append(org_info)
                    
            logger.info(f"Found {len(organizations)} organizations from Crunchbase")
            return organizations[:max_results]
            
        except Exception as e:
            logger.error(f"Error fetching from Crunchbase API: {e}")
            return []
    
    def _extract_organization_from_article(self, article: Dict, sector: str = None) -> Optional[Dict]:
        """Extract organization information from a news article"""
        try:
            # Extract company names from title and description
            title = article.get('title', '')
            description = article.get('description', '')
            content = article.get('content', '')
            
            # Simple company name extraction (in production, use NLP)
            company_names = self._extract_company_names(title + " " + description + " " + content)
            
            if not company_names:
                return None
                
            # Get the most prominent company name
            company_name = company_names[0]
            
            # Determine sector from content analysis
            detected_sector = sector or self._detect_sector_from_content(content)
            
            # Calculate relevance score based on keyword matches
            relevance_score = self._calculate_relevance_score(title, description, content, sector)
            
            return {
                'name': company_name,
                'source': 'News',
                'score': f"{relevance_score}%",
                'signals': self._extract_ai_signals(content),
                'sector': detected_sector,
                'estimatedSize': self._estimate_company_size(content),
                'url': article.get('url'),
                'publishedAt': article.get('publishedAt'),
                'title': title,
                'description': description
            }
            
        except Exception as e:
            logger.error(f"Error extracting organization from article: {e}")
            return None
    
    def _extract_organization_from_crunchbase(self, org_data: Dict, sector: str = None) -> Optional[Dict]:
        """Extract organization information from Crunchbase data"""
        try:
            properties = org_data.get('properties', {})
            
            company_name = properties.get('name', 'Unknown')
            detected_sector = sector or properties.get('category_groups', [{}])[0].get('name', 'Unknown')
            
            # Calculate relevance score
            relevance_score = self._calculate_crunchbase_relevance(properties, sector)
            
            return {
                'name': company_name,
                'source': 'Crunchbase',
                'score': f"{relevance_score}%",
                'signals': self._extract_crunchbase_signals(properties),
                'sector': detected_sector,
                'estimatedSize': self._estimate_crunchbase_size(properties),
                'url': properties.get('homepage_url'),
                'founded': properties.get('founded_on'),
                'funding': properties.get('total_funding_usd'),
                'employees': properties.get('num_employees_enum')
            }
            
        except Exception as e:
            logger.error(f"Error extracting organization from Crunchbase: {e}")
            return None
    
    def _extract_company_names(self, text: str) -> List[str]:
        """Extract company names from text (simplified version)"""
        # This is a simplified version - in production, use proper NLP
        import re
        
        # Common company indicators
        company_indicators = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Group|Media|Communications|PR|Agency)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Times|Post|Journal|News|Broadcasting|Network)\b'
        ]
        
        companies = []
        for pattern in company_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            companies.extend(matches)
        
        # Remove duplicates and return
        return list(set(companies))
    
    def _detect_sector_from_content(self, content: str) -> str:
        """Detect sector from content analysis"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['news', 'journalism', 'media', 'broadcasting']):
            return 'Media'
        elif any(word in content_lower for word in ['pr', 'public relations', 'communications', 'marketing']):
            return 'Communications'
        elif any(word in content_lower for word in ['technology', 'software', 'ai', 'machine learning']):
            return 'Technology'
        else:
            return 'Other'
    
    def _calculate_relevance_score(self, title: str, description: str, content: str, target_sector: str = None) -> int:
        """Calculate relevance score based on content analysis"""
        score = 50  # Base score
        
        # Boost score for sector matches
        if target_sector:
            sector_keywords = {
                'Media': ['news', 'journalism', 'media', 'broadcasting', 'publishing'],
                'Communications': ['pr', 'public relations', 'communications', 'marketing', 'advertising'],
                'Technology': ['technology', 'software', 'ai', 'machine learning', 'digital']
            }
            
            if target_sector in sector_keywords:
                for keyword in sector_keywords[target_sector]:
                    if keyword in (title + description + content).lower():
                        score += 20
                        break
        
        # Boost score for AI-related content
        ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'automation', 'digital transformation']
        for keyword in ai_keywords:
            if keyword in (title + description + content).lower():
                score += 15
                break
        
        # Cap score at 100
        return min(score, 100)
    
    def _extract_ai_signals(self, content: str) -> str:
        """Extract AI adoption signals from content"""
        content_lower = content.lower()
        signals = []
        
        ai_indicators = {
            'AI-powered content': ['ai-powered', 'artificial intelligence content', 'automated content'],
            'Machine learning': ['machine learning', 'ml algorithms', 'predictive analytics'],
            'Automation': ['automated', 'automation', 'robotic process automation'],
            'Digital transformation': ['digital transformation', 'digital innovation', 'tech adoption']
        }
        
        for signal, keywords in ai_indicators.items():
            if any(keyword in content_lower for keyword in keywords):
                signals.append(signal)
        
        return ', '.join(signals[:3]) if signals else 'AI adoption signals detected'
    
    def _estimate_company_size(self, content: str) -> str:
        """Estimate company size from content"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['fortune 500', 'global', 'multinational', 'enterprise']):
            return 'Large'
        elif any(word in content_lower for word in ['startup', 'small', 'local', 'regional']):
            return 'Small'
        else:
            return 'Medium'
    
    def _calculate_crunchbase_relevance(self, properties: Dict, target_sector: str = None) -> int:
        """Calculate relevance score for Crunchbase organizations"""
        score = 60  # Base score for Crunchbase data
        
        if target_sector:
            category_groups = properties.get('category_groups', [])
            for category in category_groups:
                if target_sector.lower() in category.get('name', '').lower():
                    score += 25
                    break
        
        # Boost score for funded companies
        if properties.get('total_funding_usd', 0) > 0:
            score += 15
        
        return min(score, 100)
    
    def _extract_crunchbase_signals(self, properties: Dict) -> str:
        """Extract AI signals from Crunchbase data"""
        signals = []
        
        # Check for technology categories
        category_groups = properties.get('category_groups', [])
        for category in category_groups:
            category_name = category.get('name', '').lower()
            if any(tech in category_name for tech in ['artificial intelligence', 'machine learning', 'software']):
                signals.append('AI/ML Technology Company')
                break
        
        # Check for recent funding (indicates growth)
        if properties.get('total_funding_usd', 0) > 1000000:
            signals.append('Well-funded growth company')
        
        return ', '.join(signals) if signals else 'Technology company signals'
    
    def _estimate_crunchbase_size(self, properties: Dict) -> str:
        """Estimate company size from Crunchbase data"""
        employees = properties.get('num_employees_enum', '')
        
        if '10001+' in employees or '1000-10000' in employees:
            return 'Large'
        elif '1-10' in employees or '11-50' in employees:
            return 'Small'
        else:
            return 'Medium'
    
    def start_discovery_scan(self, keywords: str, sector: str = None, sources: List[str] = None, max_results: int = 100) -> Dict:
        """Start a comprehensive discovery scan across multiple sources"""
        if not sources:
            sources = ['news', 'crunchbase', 'twitter', 'linkedin']
        
        all_organizations = []
        scan_results = {
            'total_found': 0,
            'sources_scanned': [],
            'organizations': [],
            'scan_time': datetime.now().isoformat(),
            'status': 'completed'
        }
        
        try:
            # Check if we have API keys configured
            has_api_keys = any([
                self.news_api_key,
                self.linkedin_api_key,
                self.twitter_api_key,
                self.crunchbase_api_key
            ])
            
            if not has_api_keys:
                logger.info("No API keys configured, using sample data")
                # Provide sample data when APIs aren't configured
                sample_orgs = self._get_sample_organizations(keywords, sector)
                all_organizations.extend(sample_orgs)
                scan_results['sources_scanned'].append('Sample Data')
            else:
                # Scan News API
                if 'news' in sources:
                    logger.info("Scanning News API...")
                    news_orgs = self.discover_from_news(keywords, sector, max_results // len(sources))
                    all_organizations.extend(news_orgs)
                    scan_results['sources_scanned'].append('News API')
                
                # Scan Twitter API
                if 'twitter' in sources:
                    logger.info("Scanning Twitter API...")
                    twitter_orgs = self.discover_from_twitter(keywords, sector, max_results // len(sources))
                    all_organizations.extend(twitter_orgs)
                    scan_results['sources_scanned'].append('Twitter API')
                
                # Scan LinkedIn API
                if 'linkedin' in sources:
                    logger.info("Scanning LinkedIn API...")
                    linkedin_orgs = self.discover_from_linkedin(keywords, sector, max_results // len(sources))
                    all_organizations.extend(linkedin_orgs)
                    scan_results['sources_scanned'].append('LinkedIn API')
                
                # Scan Crunchbase
                if 'crunchbase' in sources:
                    logger.info("Scanning Crunchbase...")
                    crunchbase_orgs = self.discover_from_crunchbase(keywords, sector, max_results // len(sources))
                    all_organizations.extend(crunchbase_orgs)
                    scan_results['sources_scanned'].append('Crunchbase')
            
            # Remove duplicates based on company name
            unique_organizations = self._remove_duplicates(all_organizations)
            
            # Sort by relevance score
            unique_organizations.sort(key=lambda x: int(x['score'].replace('%', '')), reverse=True)
            
            scan_results['total_found'] = len(unique_organizations)
            scan_results['organizations'] = unique_organizations[:max_results]
            
            logger.info(f"Discovery scan completed. Found {len(unique_organizations)} unique organizations")
            
        except Exception as e:
            logger.error(f"Error during discovery scan: {e}")
            scan_results['status'] = 'error'
            scan_results['error'] = str(e)
            
        return scan_results
    
    def _get_sample_organizations(self, keywords: str, sector: str = None) -> List[Dict]:
        """Get sample organizations for demonstration when APIs aren't configured"""
        sample_orgs = [
            {
                'name': 'The Daily Chronicle',
                'website': 'https://daily-chronicle.com',
                'sector': 'Media',
                'location': 'New York, NY',
                'size': 'Medium',
                'description': 'Leading digital news organization with AI-powered content generation',
                'ai_signals': 'AI content generation, automated fact-checking, personalized news feeds',
                'score': '95%',
                'source': 'Sample Data',
                'last_updated': datetime.now().isoformat(),
                'relevance_reasons': ['AI implementation', 'Digital transformation', 'Content automation']
            },
            {
                'name': 'TechNews Media Group',
                'website': 'https://technews-media.com',
                'sector': 'Technology',
                'location': 'San Francisco, CA',
                'size': 'Large',
                'description': 'Technology-focused media company with advanced AI tools',
                'ai_signals': 'Machine learning algorithms, automated reporting, AI-driven analytics',
                'score': '92%',
                'source': 'Sample Data',
                'last_updated': datetime.now().isoformat(),
                'relevance_reasons': ['AI adoption', 'Technology focus', 'Innovation leadership']
            },
            {
                'name': 'Community News Network',
                'website': 'https://community-news.net',
                'sector': 'Media',
                'location': 'Chicago, IL',
                'size': 'Small',
                'description': 'Local news organization exploring AI for community engagement',
                'ai_signals': 'AI-powered community insights, automated local reporting',
                'score': '88%',
                'source': 'Sample Data',
                'last_updated': datetime.now().isoformat(),
                'relevance_reasons': ['Community focus', 'AI exploration', 'Local media innovation']
            },
            {
                'name': 'Digital First Media',
                'website': 'https://digitalfirst.media',
                'sector': 'Media',
                'location': 'Austin, TX',
                'size': 'Medium',
                'description': 'Digital-first newsroom with comprehensive AI integration',
                'ai_signals': 'AI content curation, automated social media, predictive analytics',
                'score': '90%',
                'source': 'Sample Data',
                'last_updated': datetime.now().isoformat(),
                'relevance_reasons': ['Digital transformation', 'AI integration', 'Innovation focus']
            },
            {
                'name': 'Media Innovation Lab',
                'website': 'https://media-innovation-lab.org',
                'sector': 'Non-Profit',
                'location': 'Boston, MA',
                'size': 'Small',
                'description': 'Non-profit organization supporting AI adoption in media',
                'ai_signals': 'AI research, media technology consulting, innovation programs',
                'score': '85%',
                'source': 'Sample Data',
                'last_updated': datetime.now().isoformat(),
                'relevance_reasons': ['AI research', 'Media support', 'Innovation programs']
            }
        ]
        
        # Filter by keywords if provided
        if keywords:
            keywords_lower = keywords.lower()
            # Split keywords into individual words for more flexible matching
            keyword_words = keywords_lower.split()
            filtered_orgs = []
            
            for org in sample_orgs:
                # Check if any keyword word matches in any field
                org_text = f"{org['name']} {org['description']} {org['ai_signals']}".lower()
                
                # If any keyword word is found in the organization text, include it
                if any(word in org_text for word in keyword_words if len(word) > 2):
                    filtered_orgs.append(org)
                # Also include if the full keyword phrase is found
                elif keywords_lower in org_text:
                    filtered_orgs.append(org)
            
            # If no matches found with keywords, return all sample orgs (for demo purposes)
            if not filtered_orgs:
                # For demo purposes, always return some organizations
                return sample_orgs[:5]
            
            return filtered_orgs[:10]  # Return up to 10 matches
        
        return sample_orgs[:10]
    
    def _remove_duplicates(self, organizations: List[Dict]) -> List[Dict]:
        """Remove duplicate organizations based on name similarity"""
        seen_names = set()
        unique_organizations = []
        
        for org in organizations:
            name = org['name'].lower().strip()
            
            # Check if we've seen a similar name
            is_duplicate = False
            for seen_name in seen_names:
                if self._names_are_similar(name, seen_name):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_names.add(name)
                unique_organizations.append(org)
        
        return unique_organizations
    
    def _names_are_similar(self, name1: str, name2: str) -> bool:
        """Check if two company names are similar (simplified version)"""
        # Remove common suffixes
        suffixes = [' inc', ' corp', ' llc', ' ltd', ' company', ' group']
        for suffix in suffixes:
            name1 = name1.replace(suffix, '')
            name2 = name2.replace(suffix, '')
        
        # Simple similarity check
        return name1 == name2 or name1 in name2 or name2 in name1
