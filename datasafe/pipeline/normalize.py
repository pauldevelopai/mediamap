from dataclasses import dataclass
from typing import Optional, List, Dict
from ..ai.extract import extract_iocs
from ..ai.classify import classify_text
from ..ai.summarize import summarize
from ..config import MEDIA_KEYWORDS, ACTION_PLAYBOOK

@dataclass
class RawItem:
    source:str; title:str; body:str
    url:Optional[str]=None; published_at:Optional[str]=None

def _media_relevant(text:str)->bool:
    t=(text or "").lower()
    return any(k in t for k in MEDIA_KEYWORDS)

def _severity(threats:List[str], iocs:Dict[str,List[str]])->str:
    score=0
    if "Ransomware" in threats: score+=2
    if "Phishing" in threats: score+=1
    if iocs.get("cve"): score+=1
    if len(iocs.get("url",[]))>=3: score+=1
    return "Critical" if score>=3 else ("High" if score==2 else "Medium" if score==1 else "Low")

def _actions(threats:List[str])->List[str]:
    seen=set(); out=[]
    for t in threats:
        for a in ACTION_PLAYBOOK.get(t, []):
            if a not in seen:
                seen.add(a); out.append(a)
    return out

def normalize(raw:RawItem):
    text=f"{raw.title or ''}\n\n{raw.body or ''}".strip()
    if not text: return None
    iocs=extract_iocs(text)
    threats,sectors=classify_text(text)
    media_like=("Media" in sectors) or _media_relevant(text)
    if not media_like: return None
    summary=summarize(text)
    severity=_severity(threats,iocs)
    actions=_actions(threats)
    return {
      "source":raw.source,"title":raw.title,"summary":summary,"threats":threats,"sectors":sectors,
      "iocs":iocs,"url":raw.url,"published_at":raw.published_at,"severity":severity,"actions":actions
    }
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable
from ..ai.extract import extract_iocs
from ..ai.classify import classify_text
from ..ai.summarize import summarize_text
from ..config import FINANCE_KEYWORDS, KEEP_NON_FINANCE, MEDIA_KEYWORDS, KEEP_NON_MEDIA

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

@dataclass
class ThreatRecord:
    # Compatibility class for existing backend integration
    title: str
    summary: str
    threats: List[str]
    sectors: List[str]
    iocs: Dict[str, List[str]]
    severity: str
    url: Optional[str]
    published_at: Optional[str]
    source: str
    # Additional fields expected by legacy integration
    threat_type: str = ""
    sector: str = ""
    threat_confidence: float = 0.0
    sector_confidence: float = 0.0
    original_body: str = ""

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

def relevant_to_media(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in MEDIA_KEYWORDS)

def normalize(raw: RawItem) -> Optional[ThreatRecord]:
    text = f"{raw.title or ''}\n\n{raw.body or ''}".strip()
    if not text:
        return None
    iocs = extract_iocs(text)
    threats, sectors = classify_text(text)
    finance_like = ("Finance" in sectors) or relevant_to_finance(text)
    media_like = ("Media" in sectors) or relevant_to_media(text)
    if not (finance_like or media_like or KEEP_NON_FINANCE or KEEP_NON_MEDIA):
        return None
    summary = summarize_text(text)
    # severity after summary to avoid extra compute if filtered
    severity = _heuristic_severity(threats, iocs)
    # Choose single primary labels for compatibility fields
    threat_type = threats[0] if threats else ""
    sector = sectors[0] if sectors else ("Finance" if finance_like else ("Media" if media_like else "Other"))
    return ThreatRecord(
        source=raw.source,
        title=raw.title or "",
        summary=summary,
        threats=threats,
        sectors=sectors,
        iocs=iocs,
        url=raw.url,
        published_at=raw.published_at,
        severity=severity,
        threat_type=threat_type,
        sector=sector,
        threat_confidence=0.0,
        sector_confidence=0.0,
        original_body=raw.body or "",
    )

def batch_normalize(items: Iterable[RawItem]) -> List[ThreatRecord]:
    results: List[ThreatRecord] = []
    for it in items:
        rec = normalize(it)
        if rec:
            results.append(rec)
    return results