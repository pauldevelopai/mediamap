"""
AIMAP Ingestion Pipeline
Main pipeline for processing and scoring organizations
"""
from typing import Dict, List, Optional
from datetime import datetime
from ..models import Organisation, Metrics, db
from ..scoring.engine import ScoringEngine
from .sources import get_scraper_for_sector
from .transformers import normalize_signals, deduplicate_tools
from ..config import DEFAULT_PERIOD

class IngestionPipeline:
    """Main ingestion pipeline for AIMAP"""
    
    def __init__(self):
        self.scoring_engine = ScoringEngine()
    
    def process_organisation(self, org: Organisation, period: str = None, dry_run: bool = False) -> Dict:
        """Process a single organisation through the ingestion pipeline"""
        if period is None:
            period = DEFAULT_PERIOD
        
        print(f"Processing {org.name} ({org.sector})")
        
        # Skip if no website URL
        if not org.website_url:
            return {
                'status': 'skipped',
                'reason': 'No website URL',
                'organisation': org.name
            }
        
        try:
            # Step 1: Scrape data
            scraper = get_scraper_for_sector(org.sector)
            raw_signals = scraper.scrape_organization(org.website_url, org.name)
            
            # Step 2: Transform and normalize
            signals = normalize_signals(raw_signals)
            ai_tools = deduplicate_tools(signals.get('detected_tools', []))
            
            # Step 3: Score the organisation
            score, maturity_stage = self.scoring_engine.score_organisation(org, signals, period)
            
            # Step 4: Create benchmark bucket
            benchmark_bucket = self.scoring_engine.create_benchmark_bucket(org)
            
            if dry_run:
                return {
                    'status': 'success_dry_run',
                    'organisation': org.name,
                    'score': score,
                    'maturity_stage': maturity_stage,
                    'signals': signals,
                    'ai_tools': ai_tools,
                    'benchmark_bucket': benchmark_bucket
                }
            
            # Step 5: Update organisation with detected tools
            org.ai_tools = ai_tools
            org.updated_at = datetime.utcnow()
            
            # Step 6: Upsert metrics
            existing_metric = Metrics.query.filter_by(
                organisation_id=org.id,
                period=period
            ).first()
            
            if existing_metric:
                existing_metric.ai_adoption_score = score
                existing_metric.maturity_stage = maturity_stage
                existing_metric.signals = signals
                existing_metric.benchmark_bucket = benchmark_bucket
                existing_metric.updated_at = datetime.utcnow()
                metric = existing_metric
            else:
                metric = Metrics(
                    organisation_id=org.id,
                    ai_adoption_score=score,
                    maturity_stage=maturity_stage,
                    signals=signals,
                    benchmark_bucket=benchmark_bucket,
                    period=period,
                    source_tag='web_scraping'
                )
                db.session.add(metric)
            
            db.session.commit()
            
            return {
                'status': 'success',
                'organisation': org.name,
                'score': score,
                'maturity_stage': maturity_stage,
                'tools_detected': len(ai_tools),
                'benchmark_bucket': benchmark_bucket
            }
            
        except Exception as e:
            db.session.rollback()
            return {
                'status': 'error',
                'organisation': org.name,
                'error': str(e)
            }
    
    def process_sector(self, sector: str, period: str = None, dry_run: bool = False) -> List[Dict]:
        """Process all organisations in a sector"""
        if period is None:
            period = DEFAULT_PERIOD
        
        orgs = Organisation.query.filter_by(sector=sector).all()
        results = []
        
        for org in orgs:
            result = self.process_organisation(org, period, dry_run)
            results.append(result)
        
        return results
    
    def process_all(self, period: str = None, dry_run: bool = False) -> List[Dict]:
        """Process all organisations"""
        if period is None:
            period = DEFAULT_PERIOD
        
        orgs = Organisation.query.all()
        results = []
        
        for org in orgs:
            result = self.process_organisation(org, period, dry_run)
            results.append(result)
        
        return results
    
    def process_by_name(self, org_name: str, period: str = None, dry_run: bool = False) -> Dict:
        """Process a single organisation by name"""
        if period is None:
            period = DEFAULT_PERIOD
        
        org = Organisation.query.filter_by(name=org_name).first()
        if not org:
            return {
                'status': 'error',
                'error': f'Organisation "{org_name}" not found'
            }
        
        return self.process_organisation(org, period, dry_run)
