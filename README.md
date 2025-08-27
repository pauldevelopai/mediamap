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
