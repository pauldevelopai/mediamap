#!/usr/bin/env python3
"""
DataSafe Hugging Face Integration Demo
Shows how to integrate the HF pipeline with existing DataSafe scrapers
"""
import sys
import os
import json
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up minimal logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def demo_scraper_integration():
    """
    Demo showing how to integrate the HF pipeline with your existing scrapers
    """
    print("🔗 DataSafe Scraper Integration Demo")
    print("=" * 50)
    
    # This simulates how your existing scraper would work
    from datasafe.pipeline.normalize import RawItem, normalize
    
    # Example 1: PhishTank scraper integration
    print("\n📰 Example 1: PhishTank Integration")
    phishtank_data = {
        'phish_id': '12345',
        'url': 'http://fake-bank.malicious-site.com/login',
        'target': 'Standard Bank',
        'description': 'Phishing page mimicking Standard Bank login portal',
        'verification_time': '2025-08-12T10:30:00Z',
        'details': '''This phishing site perfectly replicates the Standard Bank online banking interface.
        The attackers registered the domain fake-bank.malicious-site.com and are harvesting
        credentials from unsuspecting users. The site exploits CVE-2023-5678 to bypass browser security.
        Contact email for reporting: abuse@phishtank.org'''
    }
    
    # Convert to DataSafe format
    raw_item = RawItem(
        source="PhishTank",
        title=f"Phishing attack targeting {phishtank_data['target']}",
        body=phishtank_data['details'],
        url=f"https://phishtank.org/phish_detail.php?phish_id={phishtank_data['phish_id']}",
        published_at=phishtank_data['verification_time']
    )
    
    # Process through DataSafe HF pipeline
    threat_record = normalize(raw_item)
    
    print(f"   🎯 Processed: {threat_record.title}")
    print(f"   🏢 Sector: {threat_record.sector} (confidence: {threat_record.sector_confidence:.2f})")
    print(f"   ⚠️  Threat: {threat_record.threat_type} (confidence: {threat_record.threat_confidence:.2f})")
    print(f"   🚨 Severity: {threat_record.severity}")
    print(f"   📊 IOCs: {sum(len(v) for v in threat_record.iocs.values())} indicators found")
    
    # Example 2: News feed integration
    print("\n📰 Example 2: Security News Feed")
    news_article = {
        'title': 'Major Healthcare Provider Suffers Ransomware Attack',
        'content': '''A prominent healthcare provider in the EU reported a significant ransomware incident
        affecting multiple facilities. The BlackCat ransomware group claimed responsibility and demanded
        $2 million in cryptocurrency. Initial infection vector appears to be a phishing email containing
        malicious payload (SHA256: 7d865e959b2466918c9863afca942d0fb89d7c9ac0c99bafc3749504ded97730).
        The attack exploited CVE-2024-7777 in the hospital network infrastructure. Command and control
        communication was traced to IP address 198.51.100.250.''',
        'source_url': 'https://cybersecurity-news.com/healthcare-ransomware-2025',
        'published': '2025-08-11T15:45:00Z',
        'author': 'Security Research Team'
    }
    
    raw_item = RawItem(
        source="CyberSecurity News",
        title=news_article['title'],
        body=news_article['content'],
        url=news_article['source_url'],
        published_at=news_article['published']
    )
    
    threat_record = normalize(raw_item)
    
    print(f"   🎯 Processed: {threat_record.title}")
    print(f"   🏢 Sector: {threat_record.sector} (confidence: {threat_record.sector_confidence:.2f})")
    print(f"   ⚠️  Threat: {threat_record.threat_type} (confidence: {threat_record.threat_confidence:.2f})")
    print(f"   🚨 Severity: {threat_record.severity}")
    print(f"   📊 IOCs: {sum(len(v) for v in threat_record.iocs.values())} indicators found")
    if threat_record.iocs['sha256']:
        print(f"      - SHA256 hashes: {threat_record.iocs['sha256']}")
    if threat_record.iocs['ips']:
        print(f"      - IP addresses: {threat_record.iocs['ips']}")
    if threat_record.iocs['cves']:
        print(f"      - CVEs: {threat_record.iocs['cves']}")

def demo_batch_processing():
    """
    Demo showing batch processing of multiple threat intelligence items
    """
    print("\n\n🔄 Batch Processing Demo")
    print("=" * 30)
    
    from datasafe.pipeline.normalize import RawItem, batch_normalize
    
    # Simulate multiple scraped items
    scraped_items = [
        RawItem(
            source="ThreatDB",
            title="Malware campaign distributing banking trojans",
            body="New banking trojan variant detected targeting European banks. Malware hash: a1b2c3d4e5f6789012345678901234567890abcdef. C2 server: 203.0.113.100",
            url="https://threatdb.com/report/001",
            published_at="2025-08-12T09:00:00Z"
        ),
        RawItem(
            source="OSINT Feed",
            title="Supply chain attack on software library",
            body="Malicious code injected into popular npm package. Affects versions 2.1.0-2.1.5. Backdoor communicates with malicious-cdn.attacker.com",
            url="https://osint-feed.org/supply-chain-001",
            published_at="2025-08-12T08:30:00Z"
        ),
        RawItem(
            source="Vulnerability Scanner",
            title="Critical RCE vulnerability in web framework",
            body="CVE-2025-1234 allows remote code execution in WebApp Framework v3.2. CVSS score 9.8. Immediate patching recommended.",
            url="https://vuln-scanner.net/cve-2025-1234",
            published_at="2025-08-12T07:15:00Z"
        )
    ]
    
    # Process all items in batch
    print(f"   📋 Processing {len(scraped_items)} items...")
    threat_records = batch_normalize(scraped_items)
    
    print(f"   ✅ Successfully processed {len(threat_records)} items")
    
    # Summary of results
    severity_counts = {}
    sector_counts = {}
    threat_counts = {}
    
    for record in threat_records:
        severity_counts[record.severity] = severity_counts.get(record.severity, 0) + 1
        sector_counts[record.sector] = sector_counts.get(record.sector, 0) + 1
        threat_counts[record.threat_type] = threat_counts.get(record.threat_type, 0) + 1
    
    print(f"\n   📊 Processing Summary:")
    print(f"      Severity Distribution: {severity_counts}")
    print(f"      Sector Distribution: {sector_counts}")
    print(f"      Threat Type Distribution: {threat_counts}")

def demo_api_integration():
    """
    Demo showing how to integrate with DataSafe Flask API
    """
    print("\n\n🌐 API Integration Example")
    print("=" * 30)
    
    print("   💡 With DataSafe running locally, you can:")
    print("      GET  /api/datasafe/high-severity-threats - Get recent critical threats")
    print("      POST /api/datasafe/process-unprocessed   - Process unprocessed content")
    print("      POST /api/datasafe/process-content/<id>  - Process specific content")
    print()
    print("   📝 Example API usage (with curl):")
    print("      curl http://localhost:5000/api/datasafe/high-severity-threats")
    print("      curl -X POST http://localhost:5000/api/datasafe/process-unprocessed \\")
    print("           -H 'Content-Type: application/json' \\")
    print("           -d '{\"limit\": 100}'")
    print()
    print("   🎛️  Admin dashboard available at:")
    print("      http://localhost:5000/admin/datasafe-hf")

def main():
    """Main demo function"""
    print("🚀 DataSafe Hugging Face Integration - Complete Demo")
    print("=" * 60)
    
    try:
        demo_scraper_integration()
        demo_batch_processing()
        demo_api_integration()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("\n📚 Integration Guide:")
        print("1. Import: from datasafe.pipeline.normalize import RawItem, normalize")
        print("2. Convert your scraped data to RawItem format")
        print("3. Call normalize(raw_item) to get ThreatRecord")
        print("4. Use the admin dashboard to monitor processing")
        print("5. Configure model settings via environment variables")
        
        print("\n🔧 Environment Variables (optional):")
        print("   export DS_HF_ZERO_SHOT_MODEL=facebook/bart-large-mnli")
        print("   export DS_HF_SUMMARY_MODEL=facebook/bart-large-cnn")
        print("   export DS_HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2")
        print("   export DS_ZERO_SHOT_CONF=0.55")
        print("   export DS_DEDUP_SIM=0.9")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
