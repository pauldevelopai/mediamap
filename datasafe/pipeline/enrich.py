import tldextract

def domain_features(url:str)->dict:
    if not url: return {}
    ext=tldextract.extract(url)
    return {"domain":".".join([p for p in [ext.domain, ext.suffix] if p])}
from typing import Dict, Any
import tldextract

def enrich_features(record: Dict[str, Any]) -> Dict[str, Any]:
    feats = {}
    urls = record.get("iocs", {}).get("urls") or record.get("iocs", {}).get("url") or []
    urls = urls if isinstance(urls, list) else []
    if urls:
        first = urls[0]
        feats["https"] = first.startswith("https://")
        ext = tldextract.extract(first)
        feats["domain"] = ".".join(x for x in [ext.domain, ext.suffix] if x)
        feats["punycode"] = record.get("url", "").startswith("http://xn--") or first.startswith("http://xn--") or first.startswith("https://xn--")
    record["enrich"] = feats
    return record


