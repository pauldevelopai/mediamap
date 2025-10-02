.PHONY: help install test clean dev build docker-build docker-run seed-demo run-ingestion

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -r requirements.txt

test: ## Run tests
	python -m pytest backend/tests/ -v

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

dev: ## Run development server
	export FLASK_APP=backend/app.py && export FLASK_ENV=development && export PYTHONPATH=backend && flask run --host=127.0.0.1 --port=5000

build: ## Build for production
	echo "Building AIMAP..."

docker-build: ## Build Docker image
	docker build -t aimap:latest .

docker-run: ## Run Docker container
	docker-compose up -d

docker-down: ## Stop Docker containers
	docker-compose down

migrate: ## Run database migration
	cd backend && export PYTHONPATH=. && alembic upgrade head

seed-demo: ## Seed demo data
	python scripts/aimap_cli.py seed-demo --sector Media --n 10
	python scripts/aimap_cli.py seed-demo --sector Communications --n 10

run-ingestion: ## Run ingestion for all organisations
	python scripts/aimap_cli.py ingest --all

run-scoring: ## Run scoring for current period
	python scripts/aimap_cli.py score --period 2025-08

generate-report: ## Generate sample report (requires ORG_ID)
	@if [ -z "$(ORG_ID)" ]; then echo "Usage: make generate-report ORG_ID=1"; exit 1; fi
	python scripts/aimap_cli.py report --org-id $(ORG_ID) --fmt pdf

status: ## Show database status
	python scripts/aimap_cli.py status

setup: install migrate seed-demo ## Complete setup for new installation

lint: ## Run linter
	flake8 backend/aimap/ --max-line-length=100 --ignore=E501,W503

quick-test: ## Run quick tests
	python -m pytest backend/tests/test_scoring.py -v
