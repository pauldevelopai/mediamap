"""Budget management and model routing logic"""

from .settings import get_settings
from .models import SceneType

settings = get_settings()

def choose_route(scene_type: str, remaining_veo_budget_usd: float, duration_sec: int) -> str:
    """
    Determine which model to route to based on scene type and budget.
    
    Args:
        scene_type: "HERO" or "FILLER"
        remaining_veo_budget_usd: Available Veo budget remaining
        duration_sec: Duration of the scene in seconds
        
    Returns:
        "veo" if HERO scene with sufficient budget, else "animdiff"
    """
    if scene_type == SceneType.HERO.value:
        needed_budget = duration_sec * settings.VEO_PRICE_PER_SEC
        if remaining_veo_budget_usd >= needed_budget and settings.USE_VEO:
            return "veo"
    
    return "animdiff"

def calculate_scene_cost(model_route: str, duration_sec: int) -> float:
    """Calculate the cost for generating a scene"""
    if model_route == "veo":
        return duration_sec * settings.VEO_PRICE_PER_SEC
    else:
        # AnimateDiff/SVD is free (open source)
        return 0.0

def get_budget_info(episode_budget: float, veo_spend: float) -> dict:
    """Get budget information for an episode"""
    remaining = episode_budget - veo_spend
    available_seconds = int(remaining / settings.VEO_PRICE_PER_SEC) if settings.VEO_PRICE_PER_SEC > 0 else 0
    
    return {
        "total_budget_usd": episode_budget,
        "veo_spend_usd": veo_spend,
        "remaining_budget_usd": remaining,
        "veo_seconds_available": available_seconds
    }

