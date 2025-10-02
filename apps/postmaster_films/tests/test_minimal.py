"""Minimal tests for Postmaster Films core functionality"""

import pytest
from apps.postmaster_films.backend.router_budget import choose_route, calculate_scene_cost, get_budget_info
from apps.postmaster_films.backend.services.shotlist import script_to_scenes, estimate_total_duration
from apps.postmaster_films.backend.services.prompts import build_prompt, load_templates

def test_budget_routing():
    """Test budget-based model routing logic"""
    # HERO scene with sufficient budget should route to Veo
    assert choose_route("HERO", 10.0, 5) in ("veo", "animdiff")
    
    # FILLER scene should always route to AnimateDiff
    assert choose_route("FILLER", 100.0, 5) == "animdiff"
    
    # HERO scene with insufficient budget should route to AnimateDiff
    assert choose_route("HERO", 1.0, 10) == "animdiff"

def test_scene_cost_calculation():
    """Test scene cost calculation"""
    # Veo costs $0.40 per second
    assert calculate_scene_cost("veo", 5) == 2.0
    assert calculate_scene_cost("veo", 10) == 4.0
    
    # AnimateDiff is free
    assert calculate_scene_cost("animdiff", 5) == 0.0
    assert calculate_scene_cost("animdiff", 10) == 0.0

def test_budget_info():
    """Test budget information calculation"""
    budget_info = get_budget_info(50.0, 10.0)
    
    assert budget_info["total_budget_usd"] == 50.0
    assert budget_info["veo_spend_usd"] == 10.0
    assert budget_info["remaining_budget_usd"] == 40.0
    assert budget_info["veo_seconds_available"] == 100  # 40.0 / 0.40

def test_script_to_scenes():
    """Test script parsing into scenes"""
    test_script = """This is the first paragraph of the script.
It talks about the opening scene.

This is the second paragraph.
It describes the main action sequence.

And this is the third paragraph.
The conclusion of our story."""
    
    scenes = script_to_scenes(test_script)
    
    assert len(scenes) == 3
    assert scenes[0]["index"] == 0
    assert scenes[1]["index"] == 1
    assert scenes[2]["index"] == 2
    
    # Each scene should have required fields
    for scene in scenes:
        assert "description" in scene
        assert "duration_sec" in scene
        assert "scene_type" in scene
        assert scene["duration_sec"] >= 3  # Minimum duration
        assert scene["duration_sec"] <= 12  # Maximum duration
        assert scene["scene_type"] in ["HERO", "FILLER"]

def test_duration_estimation():
    """Test scene duration estimation"""
    scenes = [
        {"duration_sec": 5},
        {"duration_sec": 3},
        {"duration_sec": 7}
    ]
    
    total = estimate_total_duration(scenes)
    assert total == 15

def test_prompt_building():
    """Test prompt generation"""
    # Basic prompt building
    prompt = build_prompt("A reporter stands in front of a building", is_hero=True)
    assert "reporter stands in front of a building" in prompt.lower()
    assert len(prompt) > 20  # Should have style information added
    
    # Hero scene should have additional enhancements
    hero_prompt = build_prompt("Action scene", is_hero=True)
    filler_prompt = build_prompt("Action scene", is_hero=False)
    assert len(hero_prompt) > len(filler_prompt)

def test_template_loading():
    """Test prompt template loading"""
    templates = load_templates()
    
    assert "styles" in templates
    assert "cinematic" in templates["styles"]
    assert len(templates["styles"]["cinematic"]) > 10

def test_scene_types():
    """Test scene type assignment"""
    # Test with action keywords
    action_script = "This is a dramatic and intense action sequence with key revelations."
    scenes = script_to_scenes(action_script)
    
    # Should detect action keywords and assign HERO type
    assert any(scene["scene_type"] == "HERO" for scene in scenes)
    
    # Test with neutral content
    neutral_script = "The person walks down the hallway and enters the room."
    scenes = script_to_scenes(neutral_script)
    
    # Should default to FILLER type
    assert scenes[0]["scene_type"] == "FILLER"

if __name__ == "__main__":
    pytest.main([__file__])

