"""
Example runner demonstrating the DataSafe Hugging Face integration pipeline
"""
import logging
import json
from typing import List

from .normalize import RawItem, normalize, batch_normalize

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def create_sample_data() -> List[RawItem]:
    """Create sample threat intelligence data for testing"""
    
    samples = [
        RawItem(
            source="PhishTank",
            title="Fake banking login targeting SA users",
            body="""A sophisticated phishing campaign has been discovered targeting Standard Bank customers in South Africa. 
            The malicious website http://std-bank-login.co.za perfectly mimics the legitimate banking portal.
            The attackers are using compromised email addresses including admin@fake-stdbank.com to send 
            convincing phishing emails. The campaign appears to exploit CVE-2023-12345 vulnerability.
            Malicious file hash detected: a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456.
            Infrastructure linked to IP addresses 192.168.1.100 and 203.0.113.50.""",
            url="https://phishtank.org/phish_detail.php?phish_id=12345",
            published_at="2025-08-10T09:15:00Z"
        ),
        
        RawItem(
            source="ThreatFeed",
            title="Ransomware attack on healthcare provider",
            body="""A major healthcare provider in the EU was hit by a sophisticated ransomware attack last week.
            The BlackCat ransomware variant was deployed across the network, encrypting critical patient data.
            Initial access was gained through a spear-phishing email containing malicious attachment 
            (MD5: d41d8cd98f00b204e9800998ecf8427e). The attack leveraged CVE-2024-8888 in the hospital's
            VPN software. Command and control server identified at evil-c2.darkweb.onion (185.220.100.240).
            Patient records and billing systems were completely compromised.""",
            url="https://threatfeed.io/report/12345",
            published_at="2025-08-09T14:30:00Z"
        ),
        
        RawItem(
            source="OSINT_Blog",
            title="Supply chain compromise in software dependency",
            body="""Security researchers discovered a supply chain attack affecting the popular npm package 'web-utils'.
            The compromised version 2.1.5 contains malicious code that exfiltrates environment variables.
            The backdoor communicates with attacker-controlled domain collector.suspicious-cdn.net.
            Affected organizations should check for network connections to IP 198.51.100.42.
            The malicious payload also drops a secondary implant with SHA256 hash:
            7d865e959b2466918c9863afca942d0fb89d7c9ac0c99bafc3749504ded97730.""",
            url="https://osint-blog.com/supply-chain-attack-npm",
            published_at="2025-08-08T11:20:00Z"
        )
    ]
    
    return samples

def demonstrate_single_item():
    """Demonstrate processing a single threat intelligence item"""
    logger.info("=== Single Item Processing Demo ===")
    
    # Create a sample phishing report
    raw = RawItem(
        source="SecurityTeam",
        title="Credential harvesting campaign targeting financial sector",
        body="""Our security team identified a large-scale credential harvesting operation 
        targeting employees in the financial services sector. The campaign uses fake Microsoft 365
        login pages hosted on compromised domains including fake-office365.badsite.com and
        ms-login.malicious-domain.org. Phishing emails originate from compromised accounts
        like noreply@legitimate-company.com. The operation appears linked to IP address 
        203.0.113.25 and uses CVE-2023-9999 exploit kit.""",
        url="https://internal-security.company.com/reports/2025-001",
        published_at="2025-08-10T10:00:00Z"
    )
    
    # Process the item
    record = normalize(raw)
    
    # Display results
    print(f"\n📊 Processing Results:")
    print(f"   Title: {record.title}")
    print(f"   Summary: {record.summary}")
    print(f"   Sector: {record.sector} (confidence: {record.sector_confidence:.2f})")
    print(f"   Threat: {record.threat_type} (confidence: {record.threat_confidence:.2f})")
    print(f"   Severity: {record.severity}")
    print(f"   IOCs Found:")
    for ioc_type, values in record.iocs.items():
        if values:
            print(f"     {ioc_type}: {values}")

def demonstrate_batch_processing():
    """Demonstrate batch processing of multiple threat intelligence items"""
    logger.info("=== Batch Processing Demo ===")
    
    # Get sample data
    samples = create_sample_data()
    
    # Process all items
    records = batch_normalize(samples)
    
    # Display summary
    print(f"\n📊 Batch Processing Results ({len(records)} items):")
    print("-" * 80)
    
    for i, record in enumerate(records, 1):
        print(f"\n{i}. {record.title[:60]}...")
        print(f"   Source: {record.source}")
        print(f"   Sector: {record.sector} | Threat: {record.threat_type} | Severity: {record.severity}")
        
        # Count IOCs
        total_iocs = sum(len(v) for v in record.iocs.values())
        print(f"   IOCs: {total_iocs} indicators extracted")
        
        # Show summary preview
        print(f"   Summary: {record.summary[:100]}...")

def demonstrate_json_export():
    """Demonstrate exporting processed records to JSON"""
    logger.info("=== JSON Export Demo ===")
    
    samples = create_sample_data()
    records = batch_normalize(samples)
    
    # Convert to JSON-serializable format
    json_records = []
    for record in records:
        json_record = {
            'source': record.source,
            'title': record.title,
            'summary': record.summary,
            'classifications': {
                'sector': record.sector,
                'sector_confidence': record.sector_confidence,
                'threat_type': record.threat_type,
                'threat_confidence': record.threat_confidence,
                'severity': record.severity
            },
            'iocs': record.iocs,
            'metadata': {
                'url': record.url,
                'published_at': record.published_at,
                'processed_at': record.processed_at
            }
        }
        json_records.append(json_record)
    
    # Save to file
    with open('datasafe_processed_threats.json', 'w') as f:
        json.dump(json_records, f, indent=2)
    
    print(f"\n💾 Exported {len(json_records)} processed threat records to 'datasafe_processed_threats.json'")

def main():
    """Main demo function"""
    print("🚀 DataSafe Hugging Face Integration Demo")
    print("=" * 50)
    
    try:
        # Run demonstrations
        demonstrate_single_item()
        print("\n" + "=" * 50)
        
        demonstrate_batch_processing()
        print("\n" + "=" * 50)
        
        demonstrate_json_export()
        print("\n" + "=" * 50)
        
        print("✅ Demo completed successfully!")
        print("\nNext steps:")
        print("1. Integrate with your existing scrapers using normalize() function")
        print("2. Configure model settings via environment variables")
        print("3. Add persistence layer for processed records")
        print("4. Set up monitoring and alerting for high-severity threats")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    main()
