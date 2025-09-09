#!/usr/bin/env python3
"""
AIMAP CLI Tool
Command-line interface for AIMAP operations
"""
import sys
import os
import click
import random
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from aimap.models import Organisation, Metrics, db
from aimap.ingest.pipeline import IngestionPipeline
from aimap.scoring.engine import ScoringEngine
from aimap.config import DEFAULT_PERIOD
from flask import Flask
from models import db as main_db

# Initialize Flask app for database context
app = Flask(__name__)
# Use the same database path as the main application
import os
basedir = os.path.abspath(os.path.dirname(__file__))
instance_dir = os.path.join(basedir, '..', 'backend', 'instance')
os.makedirs(instance_dir, exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(instance_dir, "media_analysis.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
main_db.init_app(app)

@click.group()
def cli():
    """AIMAP CLI - AI Adoption Intelligence Platform"""
    pass

@cli.command()
@click.option('--sector', required=True, help='Sector to seed (Media, Communications)')
@click.option('--n', default=10, help='Number of organisations to create')
def seed_demo(sector, n):
    """Seed demo data for testing"""
    with app.app_context():
        click.echo(f"Seeding {n} demo organisations for {sector} sector...")
        
        # Sample data
        if sector == "Media":
            names = [
                "Metro Daily News", "City Broadcasting Corp", "Regional Times",
                "Valley Voice Media", "Central News Network", "Local Leader Press",
                "Community Herald", "Town Tribune", "County Chronicle",
                "Digital News Today", "Morning Gazette", "Evening Standard",
                "Weekend Review", "Sports Weekly", "Tech Report Daily"
            ]
            subsectors = ["Newspaper", "TV/Radio", "Digital", "Magazine"]
        elif sector == "Communications":
            names = [
                "Premier PR Agency", "Strategic Communications Ltd", "Brand Voice Partners",
                "Public Affairs Group", "Creative Communications Co", "Media Relations Inc",
                "Corporate Comms Hub", "Digital PR Solutions", "Crisis Communications LLC",
                "Reputation Management Pro", "Social Media Experts", "Content Strategy Co",
                "Influencer Relations Agency", "Event Communications", "B2B PR Specialists"
            ]
            subsectors = ["PR Agency", "Corporate Communications", "Digital Marketing", "Crisis Communications"]
        else:
            click.echo(f"Unknown sector: {sector}")
            return
        
        regions = ["North America", "Europe", "Asia Pacific", "Latin America"]
        countries = ["United States", "United Kingdom", "Canada", "Australia", "Germany", "France"]
        size_bands = ["startup", "small", "medium", "large", "enterprise"]
        client_tags = ["tier1", "tier2", "prospect", "lead"]
        
        created_count = 0
        for i in range(min(n, len(names))):
            name = names[i]
            
            # Check if organisation already exists
            existing = Organisation.query.filter_by(name=name).first()
            if existing:
                click.echo(f"  Skipping {name} (already exists)")
                continue
            
            org = Organisation(
                name=name,
                sector=sector,
                subsector=random.choice(subsectors),
                region=random.choice(regions),
                country=random.choice(countries),
                size_band=random.choice(size_bands),
                client_tag=random.choice(client_tags),
                website_url=f"https://{name.lower().replace(' ', '').replace(',', '')}.com",
                notes=f"Demo organisation for {sector} sector testing"
            )
            
            db.session.add(org)
            created_count += 1
            click.echo(f"  Created: {name}")
        
        try:
            db.session.commit()
            click.echo(f"✅ Successfully created {created_count} organisations")
        except Exception as e:
            db.session.rollback()
            click.echo(f"❌ Error creating organisations: {e}")

@cli.command()
@click.option('--org', help='Specific organisation name')
@click.option('--all', 'process_all', is_flag=True, help='Process all organisations')
@click.option('--sector', help='Process all organisations in sector')
@click.option('--period', default=DEFAULT_PERIOD, help='Period (YYYY-MM format)')
@click.option('--dry-run', is_flag=True, help='Dry run without saving to database')
def ingest(org, process_all, sector, period, dry_run):
    """Run ingestion pipeline"""
    with app.app_context():
        pipeline = IngestionPipeline()
        
        if org:
            click.echo(f"Processing organisation: {org}")
            result = pipeline.process_by_name(org, period, dry_run)
            results = [result]
        elif sector:
            click.echo(f"Processing all organisations in {sector} sector")
            results = pipeline.process_sector(sector, period, dry_run)
        elif process_all:
            click.echo("Processing all organisations")
            results = pipeline.process_all(period, dry_run)
        else:
            click.echo("Please specify --org, --sector, or --all")
            return
        
        # Display results
        success_count = 0
        error_count = 0
        
        for result in results:
            status = result['status']
            org_name = result['organisation']
            
            if status == 'success':
                score = result.get('score', 'N/A')
                stage = result.get('maturity_stage', 'N/A')
                tools = result.get('tools_detected', 0)
                click.echo(f"  ✅ {org_name}: Score={score}, Stage={stage}, Tools={tools}")
                success_count += 1
            elif status == 'success_dry_run':
                score = result.get('score', 'N/A')
                stage = result.get('maturity_stage', 'N/A')
                tools = len(result.get('ai_tools', []))
                click.echo(f"  🔍 {org_name} (DRY RUN): Score={score}, Stage={stage}, Tools={tools}")
                success_count += 1
            elif status == 'skipped':
                reason = result.get('reason', 'Unknown')
                click.echo(f"  ⏭️  {org_name}: Skipped - {reason}")
            else:
                error = result.get('error', 'Unknown error')
                click.echo(f"  ❌ {org_name}: Error - {error}")
                error_count += 1
        
        click.echo(f"\n📊 Summary: {success_count} successful, {error_count} errors")
        if dry_run:
            click.echo("🔍 This was a dry run - no data was saved")

@cli.command()
@click.option('--period', required=True, help='Period (YYYY-MM format)')
@click.option('--org', help='Specific organisation name')
@click.option('--sector', help='Process all organisations in sector')
def score(period, org, sector):
    """Run scoring for organisations"""
    with app.app_context():
        scoring_engine = ScoringEngine()
        
        if org:
            click.echo(f"Scoring organisation: {org}")
            org_obj = Organisation.query.filter_by(name=org).first()
            if not org_obj:
                click.echo(f"❌ Organisation '{org}' not found")
                return
            
            metric = Metrics.query.filter_by(
                organisation_id=org_obj.id,
                period=period
            ).first()
            
            if not metric:
                click.echo(f"❌ No data found for {org} in period {period}")
                return
            
            orgs_to_score = [org_obj]
            metrics_to_score = [metric]
        else:
            query = Organisation.query
            if sector:
                query = query.filter(Organisation.sector == sector)
                click.echo(f"Scoring all organisations in {sector} sector for {period}")
            else:
                click.echo(f"Scoring all organisations for {period}")
            
            orgs_to_score = query.all()
            
            # Get corresponding metrics
            metrics_to_score = []
            for org_obj in orgs_to_score:
                metric = Metrics.query.filter_by(
                    organisation_id=org_obj.id,
                    period=period
                ).first()
                if metric and metric.signals:
                    metrics_to_score.append((org_obj, metric))
        
        success_count = 0
        error_count = 0
        
        if org:
            # Single organisation
            try:
                score_val, maturity_stage = scoring_engine.score_organisation(org_obj, metric.signals or {}, period)
                benchmark_bucket = scoring_engine.create_benchmark_bucket(org_obj)
                
                metric.ai_adoption_score = score_val
                metric.maturity_stage = maturity_stage
                metric.benchmark_bucket = benchmark_bucket
                
                db.session.commit()
                
                click.echo(f"  ✅ {org_obj.name}: Score={score_val:.1f}, Stage={maturity_stage}")
                success_count += 1
            except Exception as e:
                click.echo(f"  ❌ {org_obj.name}: Error - {e}")
                error_count += 1
        else:
            # Multiple organisations
            for org_obj, metric in metrics_to_score:
                try:
                    score_val, maturity_stage = scoring_engine.score_organisation(org_obj, metric.signals or {}, period)
                    benchmark_bucket = scoring_engine.create_benchmark_bucket(org_obj)
                    
                    metric.ai_adoption_score = score_val
                    metric.maturity_stage = maturity_stage
                    metric.benchmark_bucket = benchmark_bucket
                    
                    click.echo(f"  ✅ {org_obj.name}: Score={score_val:.1f}, Stage={maturity_stage}")
                    success_count += 1
                except Exception as e:
                    click.echo(f"  ❌ {org_obj.name}: Error - {e}")
                    error_count += 1
            
            if success_count > 0:
                try:
                    db.session.commit()
                    click.echo(f"\n💾 Saved {success_count} scores to database")
                except Exception as e:
                    db.session.rollback()
                    click.echo(f"❌ Error saving to database: {e}")
        
        click.echo(f"\n📊 Summary: {success_count} successful, {error_count} errors")

@cli.command()
@click.option('--org', required=True, help='Organisation name')
@click.option('--fmt', type=click.Choice(['pptx', 'pdf']), default='pdf', help='Report format')
@click.option('--period', default=DEFAULT_PERIOD, help='Period (YYYY-MM format)')
@click.option('--out', default='./reports/', help='Output directory')
@click.option('--logo', help='Path to logo file')
def report(org, fmt, period, out, logo):
    """Generate report for organisation"""
    with app.app_context():
        # Find organisation
        org_obj = Organisation.query.filter_by(name=org).first()
        if not org_obj:
            click.echo(f"❌ Organisation '{org}' not found")
            return
        
        # Check if metrics exist
        metric = Metrics.query.filter_by(
            organisation_id=org_obj.id,
            period=period
        ).first()
        
        if not metric:
            click.echo(f"❌ No metrics found for {org} in period {period}")
            return
        
        try:
            if fmt == 'pptx':
                # Lazy import to avoid requiring python-pptx for non-report commands
                from aimap.reports.pptx_export import PPTXReportGenerator  # type: ignore
                generator = PPTXReportGenerator()
                filepath = generator.generate_report(org_obj, period, logo)
            else:  # pdf
                from aimap.reports.pdf_export import PDFReportGenerator  # type: ignore
                generator = PDFReportGenerator()
                filepath = generator.generate_report(org_obj, period, logo)
            
            click.echo(f"✅ Report generated: {filepath}")
            
        except Exception as e:
            click.echo(f"❌ Error generating report: {e}")

@cli.command()
def status():
    """Show AIMAP database status"""
    with app.app_context():
        try:
            org_count = Organisation.query.count()
            metrics_count = Metrics.query.count()
            
            click.echo("📊 AIMAP Database Status")
            click.echo(f"  Organisations: {org_count}")
            click.echo(f"  Metrics Records: {metrics_count}")
            
            # Count by sector
            from sqlalchemy import func
            sector_counts = db.session.query(
                Organisation.sector,
                func.count(Organisation.id)
            ).group_by(Organisation.sector).all()
            
            if sector_counts:
                click.echo("\n  By Sector:")
                for sector, count in sector_counts:
                    click.echo(f"    {sector}: {count}")
            
            # Recent metrics
            recent_metrics = Metrics.query.order_by(
                Metrics.created_at.desc()
            ).limit(5).all()
            
            if recent_metrics:
                click.echo("\n  Recent Scores:")
                for metric in recent_metrics:
                    org_name = metric.organisation.name
                    score = metric.ai_adoption_score or 0
                    stage = metric.maturity_stage or 'Unknown'
                    click.echo(f"    {org_name}: {score:.1f} ({stage})")
            
        except Exception as e:
            click.echo(f"❌ Error accessing database: {e}")

if __name__ == '__main__':
    cli()
