"""
DataSafe Hugging Face Integration for existing scraper functionality
This module provides integration between the existing DataSafe scraper and the new HF pipeline
"""
import sys
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# Add parent directory to path for datasafe imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasafe.pipeline.normalize import RawItem, NormalizedThreat, normalize
from .models import db, CrawledContent, CrawlSource

logger = logging.getLogger(__name__)

class DataSafeProcessor:
    """Integrates Hugging Face pipeline with existing DataSafe scraper"""
    
    def __init__(self):
        self.processed_count = 0
        self.failed_count = 0
    
    def process_crawled_content(self, content_id: int) -> Optional[NormalizedThreat]:
        """
        Process a single CrawledContent record through the HF pipeline
        
        Args:
            content_id: ID of the CrawledContent record
            
        Returns:
            NormalizedThreat or None if processing fails
        """
        try:
            # Fetch the crawled content
            content = CrawledContent.query.get(content_id)
            if not content:
                logger.error(f"CrawledContent with ID {content_id} not found")
                return None
            
            # Get source information
            source = CrawlSource.query.get(content.source_id)
            source_name = source.name if source else "Unknown"
            
            # Convert to RawItem format
            raw_item = RawItem(
                source=source_name,
                title=content.title,
                body=content.content,
                url=content.url,
                published_at=content.published_date.isoformat() if content.published_date else None
            )
            
            # Process through HF pipeline
            logger.info(f"Processing content ID {content_id} from {source_name}")
            threat_record = normalize(raw_item)
            
            # Update the content record
            content.is_processed = True
            content.summary = threat_record.summary
            content.relevance_score = threat_record.threat_confidence
            content.content_type = threat_record.threat_type
            
            # You could store additional HF results in tags as JSON
            import json
            content.tags = json.dumps({
                'sector': threat_record.sector,
                'sector_confidence': threat_record.sector_confidence,
                'threat_type': threat_record.threat_type,
                'threat_confidence': threat_record.threat_confidence,
                'severity': threat_record.severity,
                'iocs_count': sum(len(v) for v in threat_record.iocs.values()),
                'processed_by': 'huggingface_pipeline'
            })
            
            db.session.commit()
            self.processed_count += 1
            
            logger.info(f"Successfully processed content ID {content_id}: "
                       f"{threat_record.threat_type} / {threat_record.severity}")
            
            return threat_record
            
        except Exception as e:
            logger.error(f"Failed to process content ID {content_id}: {e}")
            self.failed_count += 1
            return None
    
    def process_batch(self, content_ids: List[int]) -> List[NormalizedThreat]:
        """
        Process multiple CrawledContent records
        
        Args:
            content_ids: List of CrawledContent IDs
            
        Returns:
            List of successfully processed NormalizedThreat objects
        """
        logger.info(f"Starting batch processing of {len(content_ids)} items")
        
        records = []
        for content_id in content_ids:
            record = self.process_crawled_content(content_id)
            if record:
                records.append(record)
        
        logger.info(f"Batch processing complete: {len(records)} successful, "
                   f"{len(content_ids) - len(records)} failed")
        
        return records
    
    def process_unprocessed_content(self, limit: int = 50) -> List[NormalizedThreat]:
        """
        Process unprocessed CrawledContent records
        
        Args:
            limit: Maximum number of records to process
            
        Returns:
            List of processed NormalizedThreat objects
        """
        # Get unprocessed content
        unprocessed = CrawledContent.query.filter_by(is_processed=False).limit(limit).all()
        
        if not unprocessed:
            logger.info("No unprocessed content found")
            return []
        
        logger.info(f"Found {len(unprocessed)} unprocessed content items")
        
        content_ids = [item.id for item in unprocessed]
        return self.process_batch(content_ids)
    
    def get_high_severity_threats(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recently processed high-severity threats
        
        Args:
            hours: Look back this many hours
            
        Returns:
            List of dictionaries with threat information
        """
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        recent_content = CrawledContent.query.filter(
            CrawledContent.is_processed == True,
            CrawledContent.created_at >= cutoff,
            CrawledContent.tags.isnot(None)
        ).all()
        
        high_severity = []
        
        for content in recent_content:
            try:
                import json
                tags = json.loads(content.tags or '{}')
                
                if tags.get('severity') in ['High', 'Critical']:
                    source = CrawlSource.query.get(content.source_id)
                    high_severity.append({
                        'id': content.id,
                        'title': content.title,
                        'source': source.name if source else 'Unknown',
                        'severity': tags.get('severity'),
                        'threat_type': tags.get('threat_type'),
                        'confidence': tags.get('threat_confidence', 0),
                        'url': content.url,
                        'created_at': content.created_at.isoformat(),
                        'summary': content.summary
                    })
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse tags for content {content.id}: {e}")
        
        return sorted(high_severity, key=lambda x: x['confidence'], reverse=True)

# Flask route integration functions
def setup_datasafe_routes(app):
    """Add DataSafe HF integration routes to Flask app"""
    
    @app.route('/api/datasafe/process-unprocessed', methods=['POST'])
    def process_unprocessed():
        """API endpoint to process unprocessed crawled content"""
        try:
            from flask import request, jsonify
            from flask_login import login_required
            
            data = request.get_json() or {}
            limit = data.get('limit', 50)
            
            processor = DataSafeProcessor()
            records = processor.process_unprocessed_content(limit)
            
            return jsonify({
                'success': True,
                'processed': len(records),
                'failed': processor.failed_count,
                'message': f'Processed {len(records)} items through HF pipeline'
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/datasafe/high-severity-threats')
    def get_high_severity_threats():
        """API endpoint to get recent high-severity threats"""
        try:
            from flask import request, jsonify
            from flask_login import login_required
            
            hours = request.args.get('hours', 24, type=int)
            
            processor = DataSafeProcessor()
            threats = processor.get_high_severity_threats(hours)
            
            return jsonify({
                'success': True,
                'threats': threats,
                'count': len(threats)
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/datasafe/process-content/<int:content_id>', methods=['POST'])
    def process_specific_content(content_id):
        """API endpoint to process specific content by ID"""
        try:
            from flask import jsonify
            from flask_login import login_required
            
            processor = DataSafeProcessor()
            record = processor.process_crawled_content(content_id)
            
            if record:
                return jsonify({
                    'success': True,
                    'threat_record': {
                        'title': record.title,
                        'sector': record.sector,
                        'threat_type': record.threat_type,
                        'severity': record.severity,
                        'summary': record.summary,
                        'iocs_count': sum(len(v) for v in record.iocs.values())
                    }
                })
            else:
                return jsonify({'success': False, 'error': 'Processing failed'}), 500
                
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

# Utility functions for scraper integration
def process_scraped_item(source_name: str, title: str, content: str, 
                        url: str = None, published_at: str = None) -> NormalizedThreat:
    """
    Convenience function to process a scraped item directly
    
    Args:
        source_name: Name of the source
        title: Title of the content
        content: Main content/body text
        url: Optional URL
        published_at: Optional publish timestamp
        
    Returns:
        Processed NormalizedThreat
    """
    raw_item = RawItem(
        source=source_name,
        title=title,
        body=content,
        url=url,
        published_at=published_at
    )
    
    return normalize(raw_item)

def save_processed_threat(threat_record: NormalizedThreat, source_id: int = None) -> int:
    """
    Save a processed threat record to the database
    
    Args:
        threat_record: Processed NormalizedThreat object
        source_id: Optional CrawlSource ID
        
    Returns:
        ID of the created CrawledContent record
    """
    import json
    
    content = CrawledContent(
        source_id=source_id,
        title=threat_record.title,
        content=threat_record.original_body,
        url=threat_record.url,
        published_date=datetime.fromisoformat(threat_record.published_at) if threat_record.published_at else None,
        content_type=threat_record.threat_type,
        tags=json.dumps({
            'sector': threat_record.sector,
            'sector_confidence': threat_record.sector_confidence,
            'threat_type': threat_record.threat_type,
            'threat_confidence': threat_record.threat_confidence,
            'severity': threat_record.severity,
            'iocs': threat_record.iocs,
            'processed_by': 'huggingface_pipeline'
        }),
        summary=threat_record.summary,
        relevance_score=threat_record.threat_confidence,
        is_processed=True,
        created_at=datetime.utcnow()
    )
    
    db.session.add(content)
    db.session.commit()
    
    return content.id
