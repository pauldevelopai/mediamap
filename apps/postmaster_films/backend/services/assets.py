"""Asset management for reference frames and reusable content"""

import pathlib
import cv2
import shutil
from typing import Optional, List, Dict
from ..settings import get_settings

settings = get_settings()

def extract_last_frame(video_path: str, out_path: str) -> str:
    """
    Extract the last frame from a video for use as reference in next scene.
    
    Args:
        video_path: Input video file path
        out_path: Output image file path
        
    Returns:
        Path to extracted frame image
    """
    if not pathlib.Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Ensure output directory exists
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    try:
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError("Video has no frames")
        
        # Seek to last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Could not read last frame")
        
        # Save the frame
        success = cv2.imwrite(out_path, frame)
        if not success:
            raise ValueError(f"Could not save frame to {out_path}")
        
        return out_path
        
    finally:
        cap.release()

def extract_first_frame(video_path: str, out_path: str) -> str:
    """
    Extract the first frame from a video.
    
    Args:
        video_path: Input video file path
        out_path: Output image file path
        
    Returns:
        Path to extracted frame image
    """
    if not pathlib.Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    try:
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")
        
        # Save the frame
        success = cv2.imwrite(out_path, frame)
        if not success:
            raise ValueError(f"Could not save frame to {out_path}")
        
        return out_path
        
    finally:
        cap.release()

def create_asset_library(base_path: str) -> None:
    """
    Create asset library directory structure.
    
    Args:
        base_path: Base path for asset library
    """
    base = pathlib.Path(base_path)
    
    # Create directory structure
    directories = [
        "ref_frames",
        "backgrounds", 
        "styles",
        "templates",
        "audio/music",
        "audio/sfx"
    ]
    
    for directory in directories:
        (base / directory).mkdir(parents=True, exist_ok=True)

def save_reference_frame(video_path: str, scene_id: int, episode_id: int, 
                        frame_type: str = "last") -> str:
    """
    Save a reference frame from a scene for future use.
    
    Args:
        video_path: Source video file path
        scene_id: Scene identifier
        episode_id: Episode identifier
        frame_type: "first" or "last"
        
    Returns:
        Path to saved reference frame
    """
    assets_dir = pathlib.Path(settings.MEDIA_ROOT) / "assets" / "ref_frames"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with metadata
    filename = f"ep{episode_id}_scene{scene_id}_{frame_type}_frame.jpg"
    out_path = assets_dir / filename
    
    if frame_type == "last":
        return extract_last_frame(video_path, str(out_path))
    else:
        return extract_first_frame(video_path, str(out_path))

def get_reference_frames(episode_id: Optional[int] = None) -> List[Dict]:
    """
    Get list of available reference frames.
    
    Args:
        episode_id: Optional filter by episode
        
    Returns:
        List of reference frame metadata
    """
    assets_dir = pathlib.Path(settings.MEDIA_ROOT) / "assets" / "ref_frames"
    if not assets_dir.exists():
        return []
    
    frames = []
    for frame_file in assets_dir.glob("*.jpg"):
        # Parse filename: ep{id}_scene{id}_{type}_frame.jpg
        parts = frame_file.stem.split("_")
        if len(parts) >= 4:
            try:
                ep_id = int(parts[0][2:])  # Remove "ep" prefix
                scene_id = int(parts[1][5:])  # Remove "scene" prefix
                frame_type = parts[2]
                
                # Filter by episode if specified
                if episode_id is not None and ep_id != episode_id:
                    continue
                
                frames.append({
                    "path": str(frame_file),
                    "episode_id": ep_id,
                    "scene_id": scene_id,
                    "frame_type": frame_type,
                    "filename": frame_file.name
                })
            except (ValueError, IndexError):
                # Skip files that don't match naming pattern
                continue
    
    return sorted(frames, key=lambda x: (x["episode_id"], x["scene_id"]))

def copy_asset_to_project(asset_path: str, project_id: int, asset_type: str) -> str:
    """
    Copy an asset to a project-specific directory.
    
    Args:
        asset_path: Source asset file path
        project_id: Project identifier
        asset_type: Type of asset (ref_frame, background, etc.)
        
    Returns:
        Path to copied asset in project directory
    """
    if not pathlib.Path(asset_path).exists():
        raise FileNotFoundError(f"Asset not found: {asset_path}")
    
    # Create project asset directory
    project_dir = pathlib.Path(settings.MEDIA_ROOT) / "projects" / str(project_id) / "assets" / asset_type
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy file with original name
    source_file = pathlib.Path(asset_path)
    dest_path = project_dir / source_file.name
    
    shutil.copy2(asset_path, dest_path)
    return str(dest_path)

def create_style_template(name: str, style_prompt: str, metadata: Optional[Dict] = None) -> str:
    """
    Create a reusable style template.
    
    Args:
        name: Template name
        style_prompt: Style prompt text
        metadata: Optional metadata dictionary
        
    Returns:
        Path to saved template file
    """
    templates_dir = pathlib.Path(settings.MEDIA_ROOT) / "assets" / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    template_data = {
        "name": name,
        "style_prompt": style_prompt,
        "metadata": metadata or {},
        "created_at": str(pathlib.Path().stat().st_ctime)
    }
    
    # Save as JSON file
    import json
    template_file = templates_dir / f"{name.lower().replace(' ', '_')}.json"
    
    with open(template_file, 'w') as f:
        json.dump(template_data, f, indent=2)
    
    return str(template_file)

def get_style_templates() -> List[Dict]:
    """
    Get list of available style templates.
    
    Returns:
        List of style template data
    """
    templates_dir = pathlib.Path(settings.MEDIA_ROOT) / "assets" / "templates"
    if not templates_dir.exists():
        return []
    
    templates = []
    for template_file in templates_dir.glob("*.json"):
        try:
            import json
            with open(template_file, 'r') as f:
                template_data = json.load(f)
                template_data["file_path"] = str(template_file)
                templates.append(template_data)
        except (json.JSONDecodeError, KeyError):
            # Skip invalid template files
            continue
    
    return sorted(templates, key=lambda x: x.get("name", ""))

def cleanup_project_assets(project_id: int) -> None:
    """
    Clean up assets for a specific project.
    
    Args:
        project_id: Project identifier
    """
    project_dir = pathlib.Path(settings.MEDIA_ROOT) / "projects" / str(project_id)
    if project_dir.exists():
        shutil.rmtree(project_dir)

