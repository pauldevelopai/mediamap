"""AnimateDiff/Stable Video Diffusion generation via ComfyUI"""

import os
import time
import json
import pathlib
import requests
from typing import Optional, Dict, Any
from ..settings import get_settings

settings = get_settings()

def generate_video(prompt: str, duration_sec: int, ref_image_path: Optional[str], out_dir: str) -> str:
    """
    Generate video using AnimateDiff/SVD via ComfyUI REST API.
    
    Args:
        prompt: Text prompt for video generation
        duration_sec: Duration in seconds
        ref_image_path: Optional reference image path
        out_dir: Output directory for generated video
        
    Returns:
        Path to generated video file
    """
    if not _is_comfyui_available():
        raise ConnectionError(f"ComfyUI server not available at {settings.COMFYUI_BASE_URL}")
    
    # Ensure output directory exists
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique output filename
    timestamp = int(time.time())
    out_path = out_dir / f"animdiff_{timestamp}.mp4"
    
    return _generate_via_comfyui(prompt, duration_sec, ref_image_path, out_path)

def _is_comfyui_available() -> bool:
    """Check if ComfyUI server is available"""
    try:
        response = requests.get(f"{settings.COMFYUI_BASE_URL}/system_stats", timeout=5)
        return response.status_code == 200
    except:
        return False

def _generate_via_comfyui(prompt: str, duration_sec: int, ref_image_path: Optional[str], 
                         out_path: pathlib.Path) -> str:
    """
    Generate video via ComfyUI REST API using AnimateDiff workflow.
    """
    try:
        # Load AnimateDiff workflow template
        workflow = _create_animatediff_workflow(prompt, duration_sec, ref_image_path)
        
        # Submit workflow to ComfyUI
        prompt_id = _submit_workflow(workflow)
        
        # Poll for completion
        result = _wait_for_completion(prompt_id)
        
        # Download generated video
        video_url = _get_output_url(result)
        _download_video(video_url, out_path)
        
        return str(out_path)
        
    except Exception as e:
        raise RuntimeError(f"ComfyUI generation failed: {str(e)}")

def _create_animatediff_workflow(prompt: str, duration_sec: int, ref_image_path: Optional[str]) -> Dict[str, Any]:
    """Create AnimateDiff workflow configuration"""
    # Calculate frame count (assuming 8 fps for AnimateDiff)
    frame_count = max(8, duration_sec * 8)
    
    workflow = {
        "3": {
            "inputs": {
                "seed": int(time.time()),
                "steps": 20,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler"
        },
        "4": {
            "inputs": {
                "ckpt_name": "animatediff_v3_sd15_mm.ckpt"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "5": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": frame_count
            },
            "class_type": "EmptyLatentImage"
        },
        "6": {
            "inputs": {
                "text": prompt,
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "7": {
            "inputs": {
                "text": "blurry, low quality, distorted",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEDecode"
        },
        "9": {
            "inputs": {
                "filename_prefix": "animdiff_output",
                "images": ["8", 0]
            },
            "class_type": "SaveImage"
        }
    }
    
    return workflow

def _submit_workflow(workflow: Dict[str, Any]) -> str:
    """Submit workflow to ComfyUI and return prompt ID"""
    url = f"{settings.COMFYUI_BASE_URL}/prompt"
    payload = {"prompt": workflow}
    
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    
    result = response.json()
    return result["prompt_id"]

def _wait_for_completion(prompt_id: str, timeout: int = 300) -> Dict[str, Any]:
    """Poll ComfyUI for workflow completion"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        url = f"{settings.COMFYUI_BASE_URL}/history/{prompt_id}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            history = response.json()
            if prompt_id in history:
                result = history[prompt_id]
                if result.get("status", {}).get("status_str") == "success":
                    return result
                elif result.get("status", {}).get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI workflow failed: {result}")
        
        time.sleep(2)
    
    raise TimeoutError(f"Workflow {prompt_id} did not complete within {timeout} seconds")

def _get_output_url(result: Dict[str, Any]) -> str:
    """Extract output video URL from ComfyUI result"""
    outputs = result.get("outputs", {})
    
    for node_id, node_output in outputs.items():
        if "images" in node_output:
            # Get the first image/video output
            images = node_output["images"]
            if images:
                filename = images[0]["filename"]
                return f"{settings.COMFYUI_BASE_URL}/view?filename={filename}"
    
    raise ValueError("No video output found in ComfyUI result")

def _download_video(url: str, output_path: pathlib.Path) -> None:
    """Download video from ComfyUI output URL"""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)



def get_available_models() -> Dict[str, Any]:
    """
    Get list of available models from ComfyUI.
    
    Returns:
        Dictionary of available models and their capabilities
    """
    if not _is_comfyui_available():
        return {
            "animatediff": {"available": False, "reason": "ComfyUI not accessible"},
            "svd": {"available": False, "reason": "ComfyUI not accessible"}
        }
    
    # In production, query ComfyUI for available models
    return {
        "animatediff": {
            "available": True,
            "max_duration": 16,
            "resolutions": ["512x512", "768x512", "1024x576"]
        },
        "svd": {
            "available": True,
            "max_duration": 4,
            "resolutions": ["1024x576"]
        }
    }

def estimate_generation_time(duration_sec: int, model: str = "animatediff") -> int:
    """
    Estimate generation time for AnimateDiff/SVD.
    
    Args:
        duration_sec: Video duration in seconds
        model: Model type being used
        
    Returns:
        Estimated generation time in seconds
    """
    # AnimateDiff typically takes 2-5x real time
    # SVD is faster but limited to shorter clips
    if model == "svd":
        return max(duration_sec * 2, 60)
    else:  # animatediff
        return max(duration_sec * 3, 90)

def validate_parameters(prompt: str, duration_sec: int, model: str = "animatediff") -> tuple[bool, str]:
    """
    Validate parameters for AnimateDiff/SVD generation.
    
    Args:
        prompt: Input prompt
        duration_sec: Video duration
        model: Model type
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not prompt or not prompt.strip():
        return False, "Prompt cannot be empty"
    
    if model == "svd" and duration_sec > 4:
        return False, "SVD model limited to 4 seconds maximum"
    
    if model == "animatediff" and duration_sec > 16:
        return False, "AnimateDiff model limited to 16 seconds maximum"
    
    if duration_sec < 1:
        return False, "Duration must be at least 1 second"
    
    return True, ""
