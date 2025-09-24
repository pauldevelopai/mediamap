"""
Model Factory for MediaMap and HealthPIN
========================================

This module provides a factory pattern to get the correct model manager
for different use cases:
- MediaMap model for Highlander AI (business/media)
- HealthPIN model for Doc AI (healthcare)
"""

import os
import logging
from typing import Optional, Dict, Any
from backend.training.model_manager import HighlanderModelManager
from backend.healthpin.training.healthpin_model_manager import HealthPINModelManager

logger = logging.getLogger(__name__)

# Global model manager instances
_mediamap_manager = None
_healthpin_manager = None

def get_mediamap_model_manager() -> HighlanderModelManager:
    """Get the MediaMap model manager for Highlander AI"""
    global _mediamap_manager
    
    if _mediamap_manager is None:
        try:
            _mediamap_manager = HighlanderModelManager(
                models_dir="./models",
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                use_custom_model=True
            )
            logger.info("✅ MediaMap model manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MediaMap model manager: {e}")
            # Fallback to basic manager
            _mediamap_manager = HighlanderModelManager(
                models_dir="./models",
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                use_custom_model=False
            )
    
    return _mediamap_manager

def get_healthpin_model_manager() -> HealthPINModelManager:
    """Get the HealthPIN model manager for Doc AI"""
    global _healthpin_manager
    
    if _healthpin_manager is None:
        try:
            _healthpin_manager = HealthPINModelManager(
                models_dir="./healthpin_models",
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                use_custom_model=True
            )
            logger.info("✅ HealthPIN model manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize HealthPIN model manager: {e}")
            # Fallback to basic manager
            _healthpin_manager = HealthPINModelManager(
                models_dir="./healthpin_models",
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                use_custom_model=False
            )
    
    return _healthpin_manager

def get_model_manager(model_type: str = "mediamap") -> Any:
    """
    Get the appropriate model manager based on type
    
    Args:
        model_type: "mediamap" for Highlander AI, "healthpin" for Doc AI
    
    Returns:
        The appropriate model manager instance
    """
    if model_type.lower() == "healthpin":
        return get_healthpin_model_manager()
    else:
        return get_mediamap_model_manager()

def get_all_model_status() -> Dict[str, Any]:
    """Get status of all model managers"""
    try:
        mediamap_manager = get_mediamap_model_manager()
        healthpin_manager = get_healthpin_model_manager()
        
        return {
            "mediamap": {
                "status": "loaded" if mediamap_manager.is_model_loaded else "not_loaded",
                "model_info": mediamap_manager.get_model_info(),
                "performance": mediamap_manager.get_performance_metrics()
            },
            "healthpin": {
                "status": "loaded" if healthpin_manager.is_model_loaded else "not_loaded", 
                "model_info": healthpin_manager.get_model_info(),
                "performance": healthpin_manager.get_performance_metrics()
            }
        }
    except Exception as e:
        logger.error(f"❌ Error getting model status: {e}")
        return {
            "mediamap": {"status": "error", "error": str(e)},
            "healthpin": {"status": "error", "error": str(e)}
        }

def reset_model_managers():
    """Reset all model managers (useful for testing)"""
    global _mediamap_manager, _healthpin_manager
    _mediamap_manager = None
    _healthpin_manager = None
    logger.info("🔄 Model managers reset")
