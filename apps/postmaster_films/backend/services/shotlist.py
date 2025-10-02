"""Script to scene conversion services"""

import re
from typing import List, Dict

def script_to_scenes(script_text: str) -> List[Dict]:
    """
    Convert a script into a list of scenes with estimated durations.
    
    This is a naive implementation that splits on paragraphs and estimates
    duration based on word count. In production, you'd use an LLM for better
    scene segmentation and duration estimation.
    
    Args:
        script_text: Raw script content
        
    Returns:
        List of scene dictionaries with index, description, and duration_sec
    """
    scenes = []
    
    # Split script into paragraphs, filtering out empty ones
    paragraphs = [p.strip() for p in script_text.split("\n\n") if p.strip()]
    
    for i, paragraph in enumerate(paragraphs):
        # Clean up the paragraph text
        description = re.sub(r'\s+', ' ', paragraph).strip()
        
        # Estimate duration based on word count
        word_count = len(description.split())
        # Rough estimate: 150-200 words per minute of speech
        # Minimum 3 seconds, maximum 12 seconds per scene
        duration_sec = max(3, min(12, int(word_count / 2.5)))
        
        # Determine scene type based on content keywords
        scene_type = "HERO" if any(keyword in description.lower() for keyword in [
            "action", "climax", "reveal", "dramatic", "intense", "key", "important", "critical"
        ]) else "FILLER"
        
        scenes.append({
            "index": i,
            "description": description,
            "duration_sec": duration_sec,
            "scene_type": scene_type
        })
    
    return scenes

def estimate_total_duration(scenes: List[Dict]) -> int:
    """Calculate total duration of all scenes"""
    return sum(scene.get("duration_sec", 5) for scene in scenes)

def optimize_scene_durations(scenes: List[Dict], target_duration_sec: int = None) -> List[Dict]:
    """
    Optimize scene durations to fit a target episode length.
    
    Args:
        scenes: List of scene dictionaries
        target_duration_sec: Desired total episode duration
        
    Returns:
        Optimized scenes with adjusted durations
    """
    if not target_duration_sec:
        return scenes
    
    current_duration = estimate_total_duration(scenes)
    if current_duration == 0:
        return scenes
    
    # Scale factor to reach target duration
    scale_factor = target_duration_sec / current_duration
    
    optimized_scenes = []
    for scene in scenes:
        new_duration = max(2, min(15, int(scene["duration_sec"] * scale_factor)))
        optimized_scene = scene.copy()
        optimized_scene["duration_sec"] = new_duration
        optimized_scenes.append(optimized_scene)
    
    return optimized_scenes

