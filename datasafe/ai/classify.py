from typing import List, Tuple
from transformers import pipeline
from ..config import HF_ZERO_SHOT_MODEL, ZERO_SHOT_CONFIDENCE

_clf = None
THREAT_LABELS = ["Phishing","Ransomware","Vulnerability","Data leak","Fraud","DDoS"]
SECTOR_LABELS = ["Finance","Healthcare","Retail","Telecoms","Energy","Media","Other"]

def _get():
    global _clf
    if _clf is None:
        _clf = pipeline("zero-shot-classification", model=HF_ZERO_SHOT_MODEL)
    return _clf

def classify_text(text: str) -> Tuple[List[str], List[str]]:
    clf = _get()
    t = clf(text, THREAT_LABELS, multi_label=True)
    s = clf(text, SECTOR_LABELS, multi_label=True)
    threats = [lab for lab,score in zip(t["labels"], t["scores"]) if score >= ZERO_SHOT_CONFIDENCE]
    sectors = [lab for lab,score in zip(s["labels"], s["scores"]) if score >= ZERO_SHOT_CONFIDENCE]
    return threats, sectors
