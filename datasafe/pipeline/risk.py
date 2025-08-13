DIGITAL_RISK_WEIGHTS = {
  "phishing": 1.0,
  "malware": 1.5,
  "credential_leak": 2.5,
  "ddos": 1.5
}

PHYSICAL_RISK_WEIGHTS = {
  "harassment": 2.0,
  "doxxing": 2.5
}

def compute_risk(rec:dict)->dict:
    th=set(rec.get("threats",[])); iocs=rec.get("iocs",{})
    score = 0
    score += DIGITAL_RISK_WEIGHTS.get("phishing",1) if "Phishing" in th else 0
    score += DIGITAL_RISK_WEIGHTS.get("malware",1.5) if "Vulnerability" in th else 0
    score += DIGITAL_RISK_WEIGHTS.get("credential_leak",2.5) if iocs.get("email") else 0
    score += DIGITAL_RISK_WEIGHTS.get("ddos",1.5) if "DDoS" in th else 0
    score = min(int(score*20), 100)
    pscore = 0
    pscore += PHYSICAL_RISK_WEIGHTS.get("harassment",2) if "Harassment" in th else 0
    pscore += PHYSICAL_RISK_WEIGHTS.get("doxxing",2.5) if "Doxxing" in th else 0
    pscore = min(int(pscore*20),100)
    return {"digital":score,"physical":pscore}
from typing import Dict, Any, List
from ..config import DIGITAL_RISK_WEIGHTS, PHYSICAL_RISK_WEIGHTS
from ..ai.toxicity import toxicity_score

def compute(normalized: Dict[str, Any]) -> Dict[str, Any]:
    threats: List[str] = normalized.get("threats", [])
    iocs = normalized.get("iocs", {})
    summary = normalized.get("summary", "")

    digital = 0.0
    reasons: List[str] = []

    if "Phishing" in threats:
        digital += DIGITAL_RISK_WEIGHTS.get("phishing", 1)
        reasons.append("Phishing label detected")
    if "Ransomware" in threats or "Malware" in threats:
        digital += DIGITAL_RISK_WEIGHTS.get("malware", 1.5)
        reasons.append("Malware/ransomware signal")
    if any(c.lower().startswith("cve-") for c in iocs.get("cves", [])):
        digital += DIGITAL_RISK_WEIGHTS.get("cve_critical", 2)
        reasons.append("CVE present")
    if "DDoS" in threats:
        digital += DIGITAL_RISK_WEIGHTS.get("ddos", 1.5)
        reasons.append("DDoS label")
    if any("password" in s.lower() or "credential" in s.lower() for s in [summary]):
        digital += DIGITAL_RISK_WEIGHTS.get("credential_leak", 2.5)
        reasons.append("Credentials mentioned in summary")

    tox = toxicity_score(summary or (normalized.get("title") or ""))
    physical = 0.0
    if tox > 0.6:
        physical += PHYSICAL_RISK_WEIGHTS.get("targeted_harassment", 2)
        reasons.append("Toxic/harassing language")
    if any(k in (summary or "").lower() for k in ["phone", "address", "id number", "home"]):
        physical += PHYSICAL_RISK_WEIGHTS.get("doxxing", 2.5)
        reasons.append("Possible doxxing content")

    # Scale to 0..100
    digital_score = int(min(100, round(digital * 20)))
    physical_score = int(min(100, round(physical * 20)))

    return {"digital": digital_score, "physical": physical_score, "explanations": reasons}


