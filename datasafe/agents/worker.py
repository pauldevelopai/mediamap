from ..collectors import rss_media, openphish, urlhaus
from ..pipeline.normalize import RawItem, normalize
from ..pipeline.risk import compute_risk
from ..storage.sqlite_store import ensure_schema, insert_record

def ingest(limit=200):
    ensure_schema()
    collected=[]
    for fn in (rss_media.collect, openphish.collect, urlhaus.collect):
        try:
            collected += fn(limit=limit//3)
        except Exception as e:
            print("[WARN]", fn.__name__, e)
    saved=0
    for d in collected:
        rec=normalize(RawItem(d["source"], d["title"], d["body"], d.get("url"), d.get("published_at")))
        if not rec: continue
        rec["risk"]=compute_risk(rec)
        try:
            insert_record(rec); saved+=1
        except Exception as e:
            print("[WARN] save failed", e)
    print(f"[ingest] collected={len(collected)} saved={saved}")
from typing import List, Dict, Any
from dataclasses import asdict, is_dataclass
import os

from ..collectors import rss_generic  # keep existing generic
from ..collectors.rss_media import fetch as fetch_media_rss
from ..collectors.openphish import fetch as fetch_openphish
from ..collectors.urlhaus import collect as fetch_urlhaus
from ..collectors.social_x import fetch_from_jsonl
from ..collectors.abuse_reports import fetch_from_folder

from ..pipeline.normalize import RawItem, normalize
from ..pipeline.enrich import enrich_features
from ..pipeline.risk import compute as compute_risk
from ..storage.sqlite_store import ensure_schema, insert_record


def _as_dict(obj):
    if is_dataclass(obj):
        return asdict(obj)
    return obj

def ingest() -> List[Dict[str, Any]]:
    ensure_schema()

    collected: List[Dict[str, Any]] = []

    # Media RSS
    try:
        collected += fetch_media_rss()
    except Exception:
        pass

    # OpenPhish
    try:
        collected += fetch_openphish()
    except Exception:
        pass

    # URLhaus
    try:
        collected += fetch_urlhaus()
    except Exception:
        pass

    # Social X JSONL (optional)
    x_path = os.getenv("DS_X_JSONL")
    if x_path:
        try:
            collected += fetch_from_jsonl(x_path)
        except Exception:
            pass

    # Abuse reports folder (optional)
    abuse_folder = os.getenv("DS_ABUSE_FOLDER")
    if abuse_folder:
        try:
            collected += fetch_from_folder(abuse_folder)
        except Exception:
            pass

    results: List[Dict[str, Any]] = []

    for item in collected:
        raw = RawItem(
            source=item.get("source", ""),
            title=item.get("title", ""),
            body=item.get("body", ""),
            url=item.get("url"),
            published_at=item.get("published_at"),
        )
        normalized = normalize(raw)
        if not normalized:
            continue
        nd = _as_dict(normalized)
        nd = enrich_features(nd)
        risk = compute_risk(nd)
        nd["risk"] = risk
        insert_record({
            "source": nd.get("source"),
            "title": nd.get("title"),
            "summary": nd.get("summary"),
            "threats": nd.get("threats", []),
            "sectors": nd.get("sectors", []),
            "iocs": nd.get("iocs", {}),
            "url": nd.get("url"),
            "published_at": nd.get("published_at"),
            "severity": nd.get("severity", "Low"),
            "risk": risk,
        })
        results.append(nd)

    return results


