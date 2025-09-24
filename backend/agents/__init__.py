"""
AI Agents Package
================

This package contains AI agents that continuously collect data and learn
in the background for both MediaMap and HealthPIN sections.

Agents:
- MediaMapAgent: Collects media industry data and learns business patterns
- HealthPINAgent: Collects healthcare data and learns clinical patterns
- BaseAgent: Common functionality for all agents
"""

from .base_agent import BaseAgent
from .mediamap_agent import MediaMapAgent
from .healthpin_agent import HealthPINAgent

__all__ = ['BaseAgent', 'MediaMapAgent', 'HealthPINAgent']




