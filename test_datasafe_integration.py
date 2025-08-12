#!/usr/bin/env python3
"""
Test script for DataSafe Hugging Face integration
"""
import sys
import os
import logging

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic functionality without heavy ML models"""
    print("🧪 Testing DataSafe Integration...")
    
    try:
        # Test IOC extraction (no ML required)
        from datasafe.ai.extract import extract_iocs
        
        test_text = """
        Malicious domain: evil.badsite.com
        C&C server: 203.0.113.50
        Email: attacker@malicious.org
        CVE: CVE-2023-12345
        Hash: a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456
        """
        
        iocs = extract_iocs(test_text)
        print("✅ IOC Extraction Test:")
        for ioc_type, values in iocs.items():
            if values:
                print(f"   {ioc_type}: {values}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def test_with_sample_data():
    """Test with sample threat intelligence data"""
    try:
        # Import classes
        from datasafe.pipeline.normalize import RawItem, normalize
        
        print("\n🔬 Testing with Sample Data...")
        
        # Create sample data
        raw = RawItem(
            source="TestSource",
            title="Sample phishing campaign",
            body="""A phishing campaign targeting banking customers has been identified.
            The attackers use fake websites hosted on compromised domains including
            fake-bank.malicious-site.com. Emails are sent from admin@fake-bank.com
            directing users to harvest credentials. The campaign exploits CVE-2023-9999
            and uses malware with hash d41d8cd98f00b204e9800998ecf8427e.""",
            url="https://example.com/report/123",
            published_at="2025-08-12T10:00:00Z"
        )
        
        print("📊 Processing sample threat intelligence...")
        
        # Process (this will use ML models if available, fallback if not)
        record = normalize(raw)
        
        print(f"✅ Sample Processing Results:")
        print(f"   Title: {record.title}")
        print(f"   Sector: {record.sector} (confidence: {record.sector_confidence:.2f})")
        print(f"   Threat: {record.threat_type} (confidence: {record.threat_confidence:.2f})")
        print(f"   Severity: {record.severity}")
        print(f"   Summary: {record.summary[:100]}...")
        
        # Show IOCs
        total_iocs = sum(len(v) for v in record.iocs.values())
        print(f"   IOCs Found: {total_iocs} indicators")
        for ioc_type, values in record.iocs.items():
            if values:
                print(f"     {ioc_type}: {values}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_existing_scraper():
    """Test integration with existing DataSafe scraper functionality"""
    try:
        print("\n🔗 Testing Integration with Existing Scraper...")
        
        # This is how you would integrate with existing scraper
        from datasafe.pipeline.normalize import RawItem, normalize
        
        # Simulate data from existing scraper
        scraped_data = {
            'source': 'PhishTank',
            'title': 'Banking phish targeting European users',
            'content': '''Recent analysis reveals a sophisticated phishing operation 
            targeting major European banks. The campaign uses domains like 
            secure-bank-login.fake-domain.org and compromised infrastructure 
            at IP address 198.51.100.42. Phishing emails contain CVE-2024-1234 exploits.''',
            'url': 'https://phishtank.org/phish_detail.php?phish_id=67890',
            'timestamp': '2025-08-12T11:30:00Z'
        }
        
        # Convert to RawItem format
        raw_item = RawItem(
            source=scraped_data['source'],
            title=scraped_data['title'],
            body=scraped_data['content'],
            url=scraped_data['url'],
            published_at=scraped_data['timestamp']
        )
        
        # Process through DataSafe pipeline
        threat_record = normalize(raw_item)
        
        print("✅ Integration Test Results:")
        print(f"   Processed: {threat_record.source} -> {threat_record.threat_type}")
        print(f"   Severity: {threat_record.severity}")
        print(f"   IOCs extracted: {sum(len(v) for v in threat_record.iocs.values())}")
        
        # This processed record could now be:
        # 1. Stored in database
        # 2. Sent to alerting system
        # 3. Added to threat intelligence feeds
        # 4. Used for similarity analysis
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 DataSafe Hugging Face Integration Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_basic_functionality():
        tests_passed += 1
    
    if test_with_sample_data():
        tests_passed += 1
    
    if test_integration_with_existing_scraper():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"📈 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! DataSafe Hugging Face integration is ready.")
        print("\n📋 Next Steps:")
        print("1. Configure environment variables for model selection:")
        print("   export DS_HF_ZERO_SHOT_MODEL=facebook/bart-large-mnli")
        print("   export DS_HF_SUMMARY_MODEL=facebook/bart-large-cnn") 
        print("   export DS_HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2")
        print("2. Integrate with your existing scraper using the normalize() function")
        print("3. Set up database persistence for processed ThreatRecord objects")
        print("4. Configure alerting for high-severity threats")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
