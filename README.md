# AIMAP - AI Adoption Intelligence Platform

AIMAP is a comprehensive AI adoption intelligence platform that tracks, scores, and benchmarks AI implementation across multiple sectors, starting with Media and Communications/PR.

## Quick Start

### Local Development

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Database setup
alembic upgrade head

# Seed demo data
python scripts/aimap_cli.py seed-demo --sector Media --n 10
python scripts/aimap_cli.py seed-demo --sector Communications --n 10

# Run ingestion
python scripts/aimap_cli.py ingest --all --sector Communications

# Generate scores
python scripts/aimap_cli.py score --period 2025-08 --sector Communications

# Start the application
python -m backend.app
```

### Docker

```bash
docker-compose up --build
```

## Features

- **Multi-sector AI adoption tracking** (Media, Communications/PR)
- **Intelligent scoring engine** with sector-specific benchmarks
- **Predictive analytics** with ML-powered forecasting models
- **Risk assessment** for organizations falling behind peers
- **ROI estimation** for AI investment scenarios
- **AI consulting intelligence** with comprehensive strategy generation
- **Process library** with real implementation strategies and tool recommendations
- **Success tracking** with measurement frameworks and progress monitoring
- **Automated ingestion** from company websites and public sources
- **Export capabilities** (PPTX reports, PDF summaries)
- **Peer benchmarking** by sector, region, and size
- **Security intelligence** via integrated DataSafe module

## API Endpoints

### Organisations
- `GET /api/organisations` - List organisations with filters
- `GET /api/organisations/{id}` - Get organisation details
- `POST /api/organisations` - Create new organisation

### Reports & Analytics
- `POST /api/reports/{id}/pptx` - Generate PowerPoint report
- `POST /api/reports/{id}/pdf` - Generate PDF report
- `GET /api/benchmarks` - Get benchmark data

### Ingestion & Scoring
- `POST /api/ingest/run` - Run data ingestion
- `POST /api/score/run` - Calculate AI adoption scores

### Machine Learning & Predictions
- `GET /api/ml/status` - Get ML models status
- `POST /api/ml/initialize` - Initialize and train ML models
- `GET /api/ml/predict/{id}` - Get comprehensive predictions for organization
- `POST /api/ml/roi/{id}` - Estimate ROI for investment scenario
- `GET /api/ml/sector-insights/{sector}` - Get sector-wide predictive insights
- `GET /api/ml/recommendations/{id}` - Get AI investment recommendations
- `GET /api/ml/risk-assessment/{id}` - Get detailed risk assessment

### AI Consulting Intelligence
- `GET /api/consulting/processes` - Get available AI processes
- `POST /api/consulting/strategy/{id}` - Generate comprehensive AI strategy
- `POST /api/consulting/package/{id}` - Generate complete consulting package
- `GET /api/consulting/insights/{id}` - Get consulting insights and recommendations
- `GET /api/consulting/recommendations/{id}` - Get AI process recommendations
- `POST /api/consulting/success-plan/{id}` - Create success tracking plan
- `POST /api/consulting/track-progress/{id}` - Track progress against success plan
- `GET /api/consulting/sectors` - Get available sectors for consulting
- `GET /api/consulting/deliverables/{id}` - Get consulting deliverables

## CLI Commands

```bash
# Seed demo data
python scripts/aimap_cli.py seed-demo --sector Media --n 10
python scripts/aimap_cli.py seed-demo --sector Communications --n 10

# Run ingestion
python scripts/aimap_cli.py ingest --all --sector Communications

# Generate scores
python scripts/aimap_cli.py score --period 2025-08 --sector Media

# Generate reports
python scripts/aimap_cli.py report --org "Test Company" --fmt pdf

# Check status
python scripts/aimap_cli.py status
```

## Configuration

Set environment variables:
- `OPENAI_API_KEY` - For AI-powered features
- `DATABASE_URL` - Custom database connection (optional)

## Testing

```bash
# Run all tests
python -m pytest backend/tests/ -v

# Run specific test
python -m pytest backend/tests/test_scoring.py -v
```
