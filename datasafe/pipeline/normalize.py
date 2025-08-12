from dataclasses import dataclass
from typing import Dict, List, Optional
from ..ai.extract import extract_iocs
from ..ai.classify import classify_text
from ..ai.summarize import summarize_text
from ..config import FINANCE_KEYWORDS, KEEP_NON_FINANCE

@dataclass
class RawItem:
    source: str
    title: str
    body: str
    url: Optional[str] = None
    published_at: Optional[str] = None

@dataclass
class NormalizedThreat:
    source: str
    title: str
    summary: str
    threats: List[str]
    sectors: List[str]
    iocs: Dict[str, List[str]]
    url: Optional[str]
    published_at: Optional[str]
    severity: str

def _heuristic_severity(threats: List[str], iocs: Dict[str, List[str]]) -> str:
    score = 0
    if "Ransomware" in threats: score += 2
    if "Phishing" in threats: score += 1
    if iocs.get("cve"): score += 1
    if len(iocs.get("url", [])) >= 3: score += 1
    return "Critical" if score >= 3 else ("High" if score == 2 else "Medium" if score == 1 else "Low")

def relevant_to_finance(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in FINANCE_KEYWORDS)

def normalize(raw: RawItem) -> Optional[NormalizedThreat]:
    text = f"{raw.title or ''}\n\n{raw.body or ''}".strip()
    if not text:
        return None
    iocs = extract_iocs(text)
    classifications = classify_text(text)
    threats = [classifications['threat'][0]] if classifications['threat'][1] > 0.5 else []
    sectors = [classifications['sector'][0]] if classifications['sector'][1] > 0.5 else []
    finance_like = ("Finance" in sectors) or relevant_to_finance(text)
    if not (finance_like or KEEP_NON_FINANCE):
        return None
    summary = summarize_text(text)
    # severity after summary to avoid extra compute if filtered
    severity = _heuristic_severity(threats, iocs)
    return NormalizedThreat(
        source=raw.source, title=raw.title or "", summary=summary,
        threats=threats, sectors=sectors, iocs=iocs,
        url=raw.url, published_at=raw.published_at, severity=severity
    )
