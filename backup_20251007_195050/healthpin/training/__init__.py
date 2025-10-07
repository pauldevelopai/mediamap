"""
HealthPIN Training Module
========================

This module provides HealthPIN-specific AI training capabilities that extend
the existing AIMAP training infrastructure.

Components:
- HealthPINDataCollector: Collects HealthPIN-specific training data
- HealthPINModelTrainer: Trains HealthPIN models
- HealthPINModelManager: Manages trained HealthPIN models
- Training configuration and utilities
"""

from .healthpin_trainer import HealthPINModelTrainer, HealthPINDataCollector
from .healthpin_model_manager import HealthPINModelManager, get_healthpin_model_manager
from .train_healthpin import main as train_healthpin_model, quick_train

__all__ = [
    'HealthPINModelTrainer',
    'HealthPINDataCollector', 
    'HealthPINModelManager',
    'get_healthpin_model_manager',
    'train_healthpin_model',
    'quick_train'
]
