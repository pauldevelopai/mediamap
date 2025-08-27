"""
AIMAP Pydantic Schemas
Data validation and API schemas
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class OrganisationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    sector: str = Field(default="Media", max_length=100)
    subsector: Optional[str] = Field(None, max_length=100)
    region: Optional[str] = Field(None, max_length=100)
    country: Optional[str] = Field(None, max_length=100)
    size_band: Optional[str] = Field(None, max_length=50)
    client_tag: Optional[str] = Field(None, max_length=100)
    contact: Optional[str] = Field(None, max_length=255)
    website_url: Optional[str] = Field(None, max_length=500)
    notes: Optional[str] = None

class OrganisationUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    sector: Optional[str] = Field(None, max_length=100)
    subsector: Optional[str] = Field(None, max_length=100)
    region: Optional[str] = Field(None, max_length=100)
    country: Optional[str] = Field(None, max_length=100)
    size_band: Optional[str] = Field(None, max_length=50)
    client_tag: Optional[str] = Field(None, max_length=100)
    contact: Optional[str] = Field(None, max_length=255)
    website_url: Optional[str] = Field(None, max_length=500)
    notes: Optional[str] = None

class OrganisationResponse(BaseModel):
    id: int
    name: str
    sector: str
    subsector: Optional[str]
    region: Optional[str]
    country: Optional[str]
    size_band: Optional[str]
    client_tag: Optional[str]
    contact: Optional[str]
    ai_tools: List[str]
    notes: Optional[str]
    website_url: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]

class MetricsCreate(BaseModel):
    organisation_id: int
    signals: Dict[str, Any]
    period: str = Field(..., regex=r'^\d{4}-\d{2}$')
    source_tag: Optional[str] = Field(None, max_length=100)

class MetricsResponse(BaseModel):
    id: int
    organisation_id: int
    ai_adoption_score: Optional[float]
    maturity_stage: Optional[str]
    signals: Dict[str, Any]
    benchmark_bucket: Optional[str]
    period: str
    source_tag: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]

class OrganisationWithMetrics(OrganisationResponse):
    latest_metrics: Optional[MetricsResponse]
    metrics_history: List[MetricsResponse]

class IngestRequest(BaseModel):
    sector: Optional[str] = None
    organisation: Optional[str] = None
    dry_run: bool = False

class ScoreRequest(BaseModel):
    period: str = Field(..., regex=r'^\d{4}-\d{2}$')
    sector: Optional[str] = None
    organisation: Optional[str] = None

class BenchmarkData(BaseModel):
    bucket: str
    median_score: float
    p25_score: float
    p75_score: float
    count: int

class ReportRequest(BaseModel):
    organisation_id: int
    include_logo: bool = False
    logo_path: Optional[str] = None
