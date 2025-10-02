"""
AIMAP PDF Export
Generate PDF reports for organisations using HTML templates
"""
import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML, CSS
from typing import Dict, List, Optional
from datetime import datetime
from backend.aimap.models import Organisation, Metrics
from ..scoring.engine import ScoringEngine
from ..config import REPORTS_ROOT

class PDFReportGenerator:
    """Generate PDF reports for organisations"""
    
    def __init__(self):
        self.scoring_engine = ScoringEngine()
        self.template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(str(self.template_dir)))
    
    def generate_report(self, org: Organisation, period: str, logo_path: Optional[str] = None) -> str:
        """Generate a comprehensive PDF report for an organisation"""
        
        # Get latest metrics
        metrics = Metrics.query.filter_by(
            organisation_id=org.id,
            period=period
        ).first()
        
        if not metrics:
            raise ValueError(f"No metrics found for {org.name} in period {period}")
        
        # Get benchmark data
        benchmark_data = self.scoring_engine.get_peer_benchmarks(metrics.benchmark_bucket, period)
        
        # Get recommendations
        if org.sector in self.scoring_engine.sector_adapters:
            adapter = self.scoring_engine.sector_adapters[org.sector]
            features = adapter.extract_features(metrics.signals or {})
            gaps = self.scoring_engine.identify_gaps(features, org.sector)
            recommendations = self.scoring_engine.get_recommendations(org, features, gaps)
        else:
            recommendations = ["Contact AIMAP team for sector-specific recommendations"]
        
        # Prepare template context
        context = {
            'org': org,
            'metrics': metrics,
            'period': period,
            'benchmark_data': benchmark_data,
            'recommendations': recommendations,
            'generated_at': datetime.now().strftime("%B %d, %Y"),
            'logo_path': logo_path
        }
        
        # Render HTML
        template = self.env.get_template('organisation_report.html')
        html_content = template.render(context)
        
        # Generate PDF
        filename = f"{org.name.replace(' ', '_')}_{period}_report.pdf"
        filepath = REPORTS_ROOT / filename
        
        # Create CSS for styling
        css_content = self._get_report_css()
        
        # Convert HTML to PDF
        HTML(string=html_content).write_pdf(
            str(filepath),
            stylesheets=[CSS(string=css_content)]
        )
        
        return str(filepath)
    
    def _get_report_css(self) -> str:
        """Get CSS styling for PDF reports"""
        return """
        @page {
            size: A4;
            margin: 1in;
        }
        
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
        }
        
        .header {
            text-align: center;
            border-bottom: 2px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #007acc;
            margin: 0;
            font-size: 28px;
        }
        
        .header h2 {
            color: #666;
            margin: 10px 0 0 0;
            font-weight: normal;
            font-size: 18px;
        }
        
        .section {
            margin-bottom: 30px;
            page-break-inside: avoid;
        }
        
        .section h3 {
            color: #007acc;
            border-bottom: 1px solid #007acc;
            padding-bottom: 5px;
            margin-bottom: 15px;
        }
        
        .scorecard {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .score-large {
            font-size: 36px;
            font-weight: bold;
            color: #007acc;
            text-align: center;
            margin: 10px 0;
        }
        
        .maturity-stage {
            background-color: #007acc;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            display: inline-block;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .benchmark-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        
        .benchmark-table th,
        .benchmark-table td {
            border: 1px solid #dee2e6;
            padding: 10px;
            text-align: left;
        }
        
        .benchmark-table th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        
        .recommendations ul,
        .signals ul {
            padding-left: 20px;
        }
        
        .recommendations li,
        .signals li {
            margin-bottom: 8px;
        }
        
        .ai-tools {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 10px 0;
        }
        
        .ai-tool {
            background-color: #e3f2fd;
            color: #1976d2;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 12px;
            font-weight: bold;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            color: #666;
            font-size: 12px;
        }
        
        .page-break {
            page-break-before: always;
        }
        """
