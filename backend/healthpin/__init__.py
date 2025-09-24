"""
HealthPIN Package
================

HealthPIN is a WhatsApp-first, AI-driven health companion designed to make 
healthcare in Africa more personal, portable, and predictive.

Core Features:
- HealthFind: AI-powered doctor matching
- HealthBank: Personal health record management  
- FamilyHealth: Family notification system
- HealthNews: Personalized health content
"""

try:
    from .routes import healthpin_bp
except ImportError as e:
    print(f"HealthPIN import error: {e}")
    healthpin_bp = None

__all__ = [
    'healthpin_bp'
]
