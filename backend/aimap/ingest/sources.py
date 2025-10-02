"""
AIMAP Data Sources
Web scraping and data collection from various sources
"""
import re
import time
import requests
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from ..config import AI_TOOLS_PATTERNS, USER_AGENT, REQUEST_TIMEOUT, REQUEST_DELAY

class WebScraper:
    """Base web scraper with common functionality"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
    
    def get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Safely fetch and parse a web page"""
        try:
            time.sleep(REQUEST_DELAY)  # Rate limiting
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text from soup"""
        if not soup:
            return ""
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        return soup.get_text(separator=' ', strip=True)
    
    def detect_ai_tools(self, text: str) -> List[str]:
        """Detect AI tools mentioned in text"""
        if not text:
            return []
        
        text_lower = text.lower()
        found_tools = []
        
        for pattern in AI_TOOLS_PATTERNS:
            if pattern.lower() in text_lower:
                found_tools.append(pattern)
        
        return list(set(found_tools))  # Remove duplicates

class MediaScraper(WebScraper):
    """Scraper for media organizations"""
    
    def scrape_organization(self, base_url: str, org_name: str) -> Dict:
        """Scrape media organization for AI adoption signals"""
        signals = {
            'total_ai_tools': 0,
            'transcription_tools': 0,
            'genai_copydesk_tools': 0,
            'personalization_signals': 0,
            'training_mentions': 0,
            'policy_documents': 0,
            'automation_mentions': 0,
            'governance_mentions': 0,
            'detected_tools': []
        }
        
        # Scrape main pages
        pages_to_check = [
            base_url,
            urljoin(base_url, '/about'),
            urljoin(base_url, '/technology'),
            urljoin(base_url, '/careers'),
            urljoin(base_url, '/newsroom'),
            urljoin(base_url, '/press')
        ]
        
        all_text = ""
        for url in pages_to_check:
            soup = self.get_page(url)
            if soup:
                text = self.extract_text(soup)
                all_text += " " + text
        
        # Analyze text for signals
        text_lower = all_text.lower()
        
        # Detect AI tools
        detected_tools = self.detect_ai_tools(all_text)
        signals['detected_tools'] = detected_tools
        signals['total_ai_tools'] = len(detected_tools)
        
        # Specific signal detection
        signals['transcription_tools'] = self._count_transcription_signals(text_lower)
        signals['genai_copydesk_tools'] = self._count_genai_signals(text_lower)
        signals['personalization_signals'] = self._count_personalization_signals(text_lower)
        signals['training_mentions'] = self._count_training_signals(text_lower)
        signals['policy_documents'] = self._count_policy_signals(text_lower)
        signals['automation_mentions'] = self._count_automation_signals(text_lower)
        signals['governance_mentions'] = self._count_governance_signals(text_lower)
        
        return signals
    
    def _count_transcription_signals(self, text: str) -> int:
        patterns = ['transcription', 'speech-to-text', 'subtitle', 'caption']
        return sum(1 for pattern in patterns if pattern in text)
    
    def _count_genai_signals(self, text: str) -> int:
        patterns = ['generative ai', 'content generation', 'ai writing', 'automated writing']
        return sum(1 for pattern in patterns if pattern in text)
    
    def _count_personalization_signals(self, text: str) -> int:
        patterns = ['personalization', 'recommendation engine', 'ai recommendation']
        return sum(1 for pattern in patterns if pattern in text)
    
    def _count_training_signals(self, text: str) -> int:
        patterns = ['ai training', 'staff training', 'digital literacy', 'ai workshop']
        return sum(1 for pattern in patterns if pattern in text)
    
    def _count_policy_signals(self, text: str) -> int:
        patterns = ['ai policy', 'ai ethics', 'ai guidelines', 'responsible ai']
        return sum(1 for pattern in patterns if pattern in text)
    
    def _count_automation_signals(self, text: str) -> int:
        patterns = ['automation', 'automated', 'workflow automation', 'process automation']
        return sum(1 for pattern in patterns if pattern in text)
    
    def _count_governance_signals(self, text: str) -> int:
        patterns = ['ai governance', 'ai oversight', 'ai committee', 'ai strategy']
        return sum(1 for pattern in patterns if pattern in text)

class CommunicationsScraper(WebScraper):
    """Scraper for communications/PR organizations"""
    
    def scrape_organization(self, base_url: str, org_name: str) -> Dict:
        """Scrape communications organization for AI adoption signals"""
        signals = {
            'total_ai_tools': 0,
            'press_workflow_ai': 0,
            'content_automation_tools': 0,
            'media_generation_tools': 0,
            'ai_analytics_tools': 0,
            'ai_disclosure_policy': 0,
            'training_mentions': 0,
            'detected_tools': []
        }
        
        # Discover press and careers pages
        press_url = self.discover_press_page(base_url)
        jobs_url = self.discover_jobs_page(base_url)
        
        # Scrape main pages
        pages_to_check = [
            base_url,
            urljoin(base_url, '/about'),
            urljoin(base_url, '/services'),
            urljoin(base_url, '/capabilities'),
            urljoin(base_url, '/careers'),
            urljoin(base_url, '/team')
        ]
        
        if press_url:
            pages_to_check.append(press_url)
        if jobs_url:
            pages_to_check.append(jobs_url)
        
        all_text = ""
        for url in pages_to_check:
            soup = self.get_page(url)
            if soup:
                text = self.extract_text(soup)
                all_text += " " + text
        
        # Analyze text for signals
        text_lower = all_text.lower()
        
        # Detect AI tools
        detected_tools = self.detect_ai_tools(all_text)
        signals['detected_tools'] = detected_tools
        signals['total_ai_tools'] = len(detected_tools)
        
        # Specific signal detection
        signals['press_workflow_ai'] = self._count_press_workflow_signals(text_lower)
        signals['content_automation_tools'] = self._count_content_automation_signals(text_lower)
        signals['media_generation_tools'] = self._count_media_generation_signals(text_lower)
        signals['ai_analytics_tools'] = self._count_analytics_signals(text_lower)
        signals['ai_disclosure_policy'] = self._count_disclosure_signals(text_lower)
        signals['training_mentions'] = self._count_training_signals(text_lower)
        
        return signals
    
    def discover_press_page(self, base_url: str) -> Optional[str]:
        """Discover press/news page URL"""
        soup = self.get_page(base_url)
        if not soup:
            return None
        
        # Look for press/news links
        press_keywords = ['press', 'news', 'media', 'newsroom']
        for link in soup.find_all('a', href=True):
            href = link['href'].lower()
            text = link.get_text().lower()
            
            if any(keyword in href or keyword in text for keyword in press_keywords):
                return urljoin(base_url, link['href'])
        
        return None
    
    def discover_jobs_page(self, base_url: str) -> Optional[str]:
        """Discover careers/jobs page URL"""
        soup = self.get_page(base_url)
        if not soup:
            return None
        
        # Look for careers/jobs links
        job_keywords = ['careers', 'jobs', 'opportunities', 'join', 'hiring']
        for link in soup.find_all('a', href=True):
            href = link['href'].lower()
            text = link.get_text().lower()
            
            if any(keyword in href or keyword in text for keyword in job_keywords):
                return urljoin(base_url, link['href'])
        
        return None
    
    def _count_press_workflow_signals(self, text: str) -> int:
        patterns = ['ai press', 'automated press', 'ai content creation', 'ai copywriting']
        return sum(1 for pattern in patterns if pattern in text)
    
    def _count_content_automation_signals(self, text: str) -> int:
        patterns = ['content automation', 'social media automation', 'campaign automation']
        return sum(1 for pattern in patterns if pattern in text)
    
    def _count_media_generation_signals(self, text: str) -> int:
        patterns = ['ai image', 'ai video', 'ai voice', 'generative media', 'synthetic media']
        return sum(1 for pattern in patterns if pattern in text)
    
    def _count_analytics_signals(self, text: str) -> int:
        patterns = ['ai analytics', 'predictive analytics', 'sentiment analysis', 'ai insights']
        return sum(1 for pattern in patterns if pattern in text)
    
    def _count_disclosure_signals(self, text: str) -> int:
        patterns = ['ai disclosure', 'ai transparency', 'ai ethics policy']
        return sum(1 for pattern in patterns if pattern in text)
    
    def _count_training_signals(self, text: str) -> int:
        patterns = ['ai training', 'team training', 'ai skills', 'ai workshop']
        return sum(1 for pattern in patterns if pattern in text)

def get_scraper_for_sector(sector: str) -> WebScraper:
    """Get appropriate scraper for sector"""
    if sector.lower() == 'media':
        return MediaScraper()
    elif sector.lower() == 'communications':
        return CommunicationsScraper()
    else:
        return WebScraper()  # Generic scraper
