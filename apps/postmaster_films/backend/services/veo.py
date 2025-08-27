"""Veo 3 Fast video generation - supports both Gemini API and Vertex AI"""

import os
import time
import pathlib
import requests
import json
import base64
from typing import Optional
from ..settings import get_settings

settings = get_settings()

def generate_video(prompt: str, duration_sec: int, ref_image_path: Optional[str], out_dir: str) -> str:
    """
    Generate video using Veo 3 Fast. Tries multiple access methods:
    1. Direct Gemini API (if GEMINI_API_KEY set)
    2. Vertex AI SDK (if GOOGLE_CLOUD_PROJECT set)
    
    Args:
        prompt: Text prompt for video generation
        duration_sec: Duration in seconds (1-8 seconds for Veo 3 Fast)
        ref_image_path: Optional reference image path
        out_dir: Output directory for generated video
        
    Returns:
        Path to generated video file
        
    Raises:
        ValueError: If Veo is disabled or no API access configured
        RuntimeError: If video generation fails
    """
    if not settings.USE_VEO:
        raise ValueError("Veo 3 Fast is disabled. Set USE_VEO=true in environment.")
    
    # Clamp duration to Veo limits
    duration_sec = max(1, min(duration_sec, settings.VEO_MAX_DURATION))
    
    # Ensure output directory exists
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    errors = []
    
    # Try Gemini API first (simpler, direct access)
    if settings.GEMINI_API_KEY:
        try:
            return _generate_via_gemini_api(prompt, duration_sec, ref_image_path, out_dir)
        except Exception as e:
            errors.append(f"Gemini API: {str(e)}")
    
    # Try Vertex AI as fallback
    if settings.GOOGLE_CLOUD_PROJECT:
        try:
            return _generate_via_vertex_ai(prompt, duration_sec, ref_image_path, out_dir)
        except Exception as e:
            errors.append(f"Vertex AI: {str(e)}")
    
    # No valid configuration found
    if not settings.GEMINI_API_KEY and not settings.GOOGLE_CLOUD_PROJECT:
        raise ValueError(
            "No Veo access configured. Set either GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT. "
            "Both require Google approval for Veo 3 Fast preview access."
        )
    
    # All methods failed
    error_summary = "; ".join(errors)
    raise RuntimeError(f"All Veo generation methods failed: {error_summary}")


def _generate_via_gemini_api(prompt: str, duration_sec: int, ref_image_path: Optional[str], out_dir: str) -> str:
    """Generate video via direct Gemini API call"""
    timestamp = int(time.time())
    out_path = out_dir / f"veo_gemini_{timestamp}.mp4"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.VEO_MODEL_ID}:generateContent?key={settings.GEMINI_API_KEY}"
    
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [{
            "parts": [{
                "text": f"Create a {duration_sec}-second video: {prompt}. Style: cinematic, {settings.VEO_ASPECT_RATIO} aspect ratio, high quality."
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "candidate_count": 1
        }
    }
    
    # Add reference image if provided
    if ref_image_path and pathlib.Path(ref_image_path).exists():
        with open(ref_image_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
            payload["contents"][0]["parts"].append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": img_data
                }
            })
    
    response = requests.post(url, headers=headers, json=payload, timeout=180)
    
    if response.status_code != 200:
        raise RuntimeError(f"Gemini API error {response.status_code}: {response.text}")
    
    result = response.json()
    
    # Extract video from response
    if "candidates" in result and result["candidates"]:
        candidate = result["candidates"][0]
        if "content" in candidate and "parts" in candidate["content"]:
            for part in candidate["content"]["parts"]:
                if "videoData" in part:
                    video_data = base64.b64decode(part["videoData"])
                    with open(out_path, 'wb') as f:
                        f.write(video_data)
                    return str(out_path)
                elif "fileData" in part and "mimeType" in part["fileData"]:
                    if "video" in part["fileData"]["mimeType"]:
                        # Handle file URI response
                        file_uri = part["fileData"]["fileUri"]
                        video_response = requests.get(file_uri, timeout=60)
                        video_response.raise_for_status()
                        with open(out_path, 'wb') as f:
                            f.write(video_response.content)
                        return str(out_path)
    
    raise ValueError(f"No video data in Gemini response: {result}")


def _generate_via_vertex_ai(prompt: str, duration_sec: int, ref_image_path: Optional[str], out_dir: str) -> str:
    """Generate video via Vertex AI SDK"""
    try:
        from google.cloud import aiplatform
        from google.cloud.aiplatform.preview.vision_models import VideoGenerationModel
    except ImportError:
        raise ImportError("google-cloud-aiplatform required for Vertex AI access")
    
    timestamp = int(time.time())
    out_path = out_dir / f"veo_vertex_{timestamp}.mp4"
    
    # Initialize Vertex AI
    aiplatform.init(
        project=settings.GOOGLE_CLOUD_PROJECT,
        location=settings.GOOGLE_CLOUD_REGION
    )
    
    # Load Veo model
    model = VideoGenerationModel.from_pretrained("veo-3-fast")
    
    # Prepare generation config
    generation_config = {
        "max_frames": duration_sec * 24,  # 24 fps
        "aspect_ratio": settings.VEO_ASPECT_RATIO,
        "motion_level": "medium"
    }
    
    # Add reference image if provided
    reference_image = None
    if ref_image_path and pathlib.Path(ref_image_path).exists():
        from google.cloud.aiplatform.preview.vision_models import Image
        reference_image = Image.load_from_file(ref_image_path)
    
    # Generate video
    if reference_image:
        response = model.generate_videos(
            prompt=prompt,
            reference_image=reference_image,
            **generation_config
        )
    else:
        response = model.generate_videos(
            prompt=prompt,
            **generation_config
        )
    
    # Save generated video
    if response and len(response) > 0:
        video = response[0]
        video.save(location=str(out_path))
        return str(out_path)
    
    raise ValueError("No video generated by Vertex AI")






def estimate_generation_time(duration_sec: int) -> int:
    """
    Estimate generation time for Veo video.
    
    Args:
        duration_sec: Video duration in seconds
        
    Returns:
        Estimated generation time in seconds
    """
    # Veo 3 Fast is approximately 1:1 ratio or faster
    # Add some buffer time
    return max(duration_sec * 1.2, 30)

def validate_prompt(prompt: str) -> tuple[bool, str]:
    """
    Validate prompt for Veo generation.
    
    Args:
        prompt: Input prompt
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not prompt or not prompt.strip():
        return False, "Prompt cannot be empty"
    
    if len(prompt) > 2000:
        return False, "Prompt too long (max 2000 characters)"
    
    # Check for potentially problematic content
    prohibited_terms = ["violence", "explicit", "illegal"]
    prompt_lower = prompt.lower()
    for term in prohibited_terms:
        if term in prompt_lower:
            return False, f"Prompt contains prohibited content: {term}"
    
    return True, ""
