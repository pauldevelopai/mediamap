from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from ..storage.sqlite_store import query_recent

app=FastAPI(title="DataSafe API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

@app.get("/threats/recent")
def recent(limit:int=Query(200,ge=1,le=1000), severity:str|None=None, min_risk:int=0):
    return {"items": query_recent(limit=limit, severity=severity, min_risk=min_risk)}
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import json

from ..storage.sqlite_store import ensure_schema, query_recent, get_stats

app = FastAPI(title="DataSafe API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    ensure_schema()

@app.get("/threats/recent")
def threats_recent(
    sector: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    min_risk: Optional[int] = Query(None),
    limit: int = Query(200, ge=1, le=500),
):
    items = query_recent(limit=limit, sector=sector, severity=severity, min_risk=min_risk)
    return {"items": items, "count": len(items)}

@app.get("/stats/summary")
def stats_summary():
    return get_stats()


