from typing import Optional
from transformers import pipeline

_tox = None

def _get():
    global _tox
    if _tox is None:
        # Lightweight zero-shot for threatening language
        _tox = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return _tox

_CANDIDATES = [
    "threatening language",
    "harassment",
    "doxxing",
    "incitement of violence",
]

def toxicity_score(text: str) -> float:
    """Return 0..1 score approximating toxicity/harassment signals."""
    clf = _get()
    res = clf(text, _CANDIDATES, multi_label=True)
    # Take max score across candidates
    return float(max(res["scores"]) if res and "scores" in res else 0.0)


