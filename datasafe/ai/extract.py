"""
IOC (Indicators of Compromise) extraction using regex patterns
Extracts IPs, hashes, URLs, CVEs, emails, domains
"""
import re
from typing import Dict, List, Set
import logging

logger = logging.getLogger(__name__)

class IOCExtractor:
    """Extract various types of IOCs from text"""
    
    def __init__(self):
        # Compile regex patterns for better performance
        self.patterns = {
            'ipv4': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'ipv6': re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),
            'domain': re.compile(r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.([a-zA-Z]{2,})\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'url': re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),
            'md5': re.compile(r'\b[a-fA-F0-9]{32}\b'),
            'sha1': re.compile(r'\b[a-fA-F0-9]{40}\b'),
            'sha256': re.compile(r'\b[a-fA-F0-9]{64}\b'),
            'cve': re.compile(r'CVE-\d{4}-\d{4,}', re.IGNORECASE),
            'file_hash': re.compile(r'\b[a-fA-F0-9]{32,64}\b'),
        }
        
        # Common false positive domains to filter out
        self.domain_whitelist = {
            'example.com', 'example.org', 'example.net',
            'localhost', 'test.com', 'google.com', 'microsoft.com'
        }
    
    def extract_ips(self, text: str) -> List[str]:
        """Extract IPv4 and IPv6 addresses"""
        ipv4_matches = self.patterns['ipv4'].findall(text)
        ipv6_matches = self.patterns['ipv6'].findall(text)
        
        # Filter out private/reserved IPs for IPv4
        valid_ipv4 = []
        for ip in ipv4_matches:
            parts = ip.split('.')
            if len(parts) == 4:
                first_octet = int(parts[0])
                # Skip private ranges (10.x.x.x, 172.16-31.x.x, 192.168.x.x) and localhost
                if not (first_octet == 10 or 
                       (first_octet == 172 and 16 <= int(parts[1]) <= 31) or
                       (first_octet == 192 and int(parts[1]) == 168) or
                       first_octet == 127):
                    valid_ipv4.append(ip)
        
        return list(set(valid_ipv4 + ipv6_matches))
    
    def extract_domains(self, text: str) -> List[str]:
        """Extract domain names, filtering out common false positives"""
        domains = self.patterns['domain'].findall(text)
        # domains will be tuples from the regex groups, so we need to join them
        domain_strings = [f"{d[0]}.{d[1]}" for d in domains if isinstance(d, tuple)]
        
        # Filter out whitelisted domains
        filtered_domains = [d for d in domain_strings if d.lower() not in self.domain_whitelist]
        
        return list(set(filtered_domains))
    
    def extract_emails(self, text: str) -> List[str]:
        """Extract email addresses"""
        emails = self.patterns['email'].findall(text)
        return list(set(emails))
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs"""
        urls = self.patterns['url'].findall(text)
        return list(set(urls))
    
    def extract_hashes(self, text: str) -> Dict[str, List[str]]:
        """Extract file hashes (MD5, SHA1, SHA256)"""
        hashes = {
            'md5': list(set(self.patterns['md5'].findall(text))),
            'sha1': list(set(self.patterns['sha1'].findall(text))),
            'sha256': list(set(self.patterns['sha256'].findall(text)))
        }
        return hashes
    
    def extract_cves(self, text: str) -> List[str]:
        """Extract CVE identifiers"""
        cves = self.patterns['cve'].findall(text)
        return list(set(cves))
    
    def extract_all(self, text: str) -> Dict[str, List[str]]:
        """
        Extract all types of IOCs from text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing all extracted IOCs by type
        """
        try:
            iocs = {
                'ips': self.extract_ips(text),
                'domains': self.extract_domains(text),
                'emails': self.extract_emails(text),
                'urls': self.extract_urls(text),
                'cves': self.extract_cves(text),
                **self.extract_hashes(text)  # Unpacks md5, sha1, sha256
            }
            
            # Log extraction summary
            total_iocs = sum(len(v) for v in iocs.values())
            if total_iocs > 0:
                logger.info(f"Extracted {total_iocs} IOCs from text")
            
            return iocs
            
        except Exception as e:
            logger.error(f"IOC extraction failed: {e}")
            return {
                'ips': [], 'domains': [], 'emails': [], 'urls': [],
                'cves': [], 'md5': [], 'sha1': [], 'sha256': []
            }

# Global extractor instance
_extractor = None

def get_extractor() -> IOCExtractor:
    """Get or create global extractor instance"""
    global _extractor
    if _extractor is None:
        _extractor = IOCExtractor()
    return _extractor

def extract_iocs(text: str) -> Dict[str, List[str]]:
    """
    Convenience function to extract IOCs from text
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary containing all extracted IOCs by type
    """
    extractor = get_extractor()
    return extractor.extract_all(text)
