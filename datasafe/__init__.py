"""
DataSafe Hugging Face Integration Package
"""
__version__ = "1.0.0"
__author__ = "DataSafe Team"

from .pipeline.normalize import RawItem, NormalizedThreat, normalize
from .ai.classify import classify_text
from .ai.extract import extract_iocs
from .ai.summarize import summarize_text
try:
    from .ai.embeddings import calculate_similarity, find_duplicates
    _has_embeddings = True
except ImportError:
    _has_embeddings = False

__all__ = [
    'RawItem',
    'NormalizedThreat', 
    'normalize',
    'classify_text',
    'extract_iocs',
    'summarize_text'
]

if _has_embeddings:
    __all__.extend(['calculate_similarity', 'find_duplicates'])
