"""Prompt generation and template management"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

def get_templates_path() -> Path:
    """Get path to prompt templates file"""
    return Path(__file__).parent.parent / "samples" / "prompt_templates.json"

def load_templates() -> Dict:
    """Load prompt templates from JSON file"""
    templates_path = get_templates_path()
    
    # Create default templates if file doesn't exist
    if not templates_path.exists():
        default_templates = {
            "styles": {
                "cinematic": "A cinematic, high dynamic range, shallow depth of field, natural film grain, realistic motion physics. CAMERA: tracked dolly-in, gentle parallax. LIGHTING: golden hour, soft key, practicals.",
                "documentary": "Documentary style, handheld camera movement, natural lighting, realistic colors, authentic atmosphere.",
                "commercial": "Professional commercial quality, smooth camera movements, perfect lighting, vibrant colors, polished aesthetic.",
                "news": "News broadcast style, steady camera, bright even lighting, professional presentation, clear visibility."
            },
            "subjects": {
                "person": "A professional-looking person in business attire",
                "newsroom": "A modern newsroom with multiple screens and desks",
                "office": "A contemporary office environment with modern furniture",
                "studio": "A professional TV studio with cameras and lighting"
            },
            "actions": {
                "talking": "speaking confidently to camera",
                "presenting": "presenting information with gestures",
                "working": "working at a desk with computer",
                "walking": "walking purposefully through the space"
            }
        }
        
        # Ensure directory exists
        templates_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(templates_path, 'w') as f:
            json.dump(default_templates, f, indent=2)
    
    try:
        with open(templates_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return minimal default if loading fails
        return {
            "styles": {
                "cinematic": "Cinematic style with professional lighting and camera movement."
            }
        }

def build_prompt(scene_desc: str, is_hero: bool = False, style: str = "cinematic", 
                custom_style: Optional[str] = None) -> str:
    """
    Build a video generation prompt from scene description and style.
    
    Args:
        scene_desc: Description of the scene action/content
        is_hero: Whether this is a hero scene (gets priority treatment)
        style: Style template to use
        custom_style: Custom style override
        
    Returns:
        Complete prompt for video generation
    """
    templates = load_templates()
    
    # Use custom style if provided, otherwise get from templates
    if custom_style:
        style_prompt = custom_style
    else:
        style_prompt = templates.get("styles", {}).get(style, "Professional video quality.")
    
    # Clean up scene description
    action = scene_desc.replace("\n", " ").strip()
    
    # Add hero scene enhancements
    if is_hero:
        style_prompt += " HERO SCENE: Enhanced dramatic lighting, dynamic camera movement, increased visual impact."
    
    # Combine style and action
    full_prompt = f"{style_prompt} ACTION: {action}"
    
    return full_prompt

def generate_continuity_prompt(previous_scene_path: Optional[str], current_scene_desc: str, 
                             style: str = "cinematic") -> str:
    """
    Generate a prompt that maintains visual continuity with the previous scene.
    
    Args:
        previous_scene_path: Path to previous scene video (for reference frame extraction)
        current_scene_desc: Description of current scene
        style: Style template to use
        
    Returns:
        Prompt with continuity instructions
    """
    base_prompt = build_prompt(current_scene_desc, style=style)
    
    if previous_scene_path:
        continuity_instruction = " CONTINUITY: Maintain consistent lighting, color palette, and visual style from previous scene."
        base_prompt += continuity_instruction
    
    return base_prompt

def optimize_prompt_length(prompt: str, max_length: int = 500) -> str:
    """
    Optimize prompt length for model constraints.
    
    Args:
        prompt: Original prompt
        max_length: Maximum allowed characters
        
    Returns:
        Optimized prompt within length limits
    """
    if len(prompt) <= max_length:
        return prompt
    
    # Try to preserve the most important parts
    parts = prompt.split(". ")
    optimized = parts[0]  # Always keep the first sentence
    
    for part in parts[1:]:
        if len(optimized + ". " + part) <= max_length:
            optimized += ". " + part
        else:
            break
    
    return optimized

