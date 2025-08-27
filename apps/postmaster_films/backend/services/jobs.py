"""Job processing services for video generation and assembly"""

import os
import pathlib
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from ..settings import get_settings
from ..models import Scene, Episode, Job, JobStatus
from ..router_budget import choose_route, calculate_scene_cost
from .prompts import build_prompt, generate_continuity_prompt
from . import veo as veo_svc
from . import animdiff as ad_svc
from .ffmpeg_tools import concat_videos, create_title_card
from .audio import tts_to_wav, mux_audio
from .assets import save_reference_frame

settings = get_settings()

def render_scene(scene: Scene, episode: Episode, db: Session, out_dir: str) -> str:
    """
    Render a single scene using the appropriate model.
    
    Args:
        scene: Scene database object
        episode: Episode database object  
        db: Database session
        out_dir: Output directory for rendered video
        
    Returns:
        Path to rendered video file
    """
    # Calculate remaining budget
    remaining_budget = episode.budget_usd - episode.veo_spend_usd
    
    # Choose model route based on scene type and budget
    route = choose_route(scene.scene_type.value, remaining_budget, scene.duration_sec)
    scene.model_route = route
    
    # Generate or use existing prompt
    if not scene.prompt:
        is_hero = scene.scene_type.value == "HERO"
        scene.prompt = build_prompt(scene.description, is_hero)
    
    # Create output directory
    scene_dir = pathlib.Path(out_dir) / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate video based on route
    try:
        if route == "veo":
            video_path = veo_svc.generate_video(
                scene.prompt, 
                scene.duration_sec, 
                scene.ref_image_path, 
                str(scene_dir)
            )
            # Update budget spend
            cost = calculate_scene_cost(route, scene.duration_sec)
            episode.veo_spend_usd += cost
            scene.cost_usd = cost
            
        else:  # animdiff
            video_path = ad_svc.generate_video(
                scene.prompt,
                scene.duration_sec,
                scene.ref_image_path,
                str(scene_dir)
            )
            scene.cost_usd = 0.0  # Open source model is free
        
        # Update scene with output path
        scene.output_video_path = video_path
        
        # Save reference frame for continuity
        try:
            ref_frame_path = save_reference_frame(
                video_path, scene.id, episode.id, "last"
            )
            # Store reference frame path in scene metadata
            if not scene.meta:
                scene.meta = {}
            scene.meta["reference_frame"] = ref_frame_path
        except Exception as e:
            print(f"Failed to save reference frame: {e}")
        
        # Commit changes to database
        db.commit()
        
        return video_path
        
    except Exception as e:
        print(f"Scene rendering failed: {e}")
        # Create error placeholder video
        error_path = scene_dir / f"error_scene_{scene.id}.mp4"
        create_title_card(
            f"Error rendering scene {scene.index}: {str(e)[:100]}",
            str(error_path),
            scene.duration_sec
        )
        scene.output_video_path = str(error_path)
        db.commit()
        return str(error_path)

def assemble_episode(episode: Episode, db: Session, out_dir: str) -> str:
    """
    Assemble all scenes into a complete episode video.
    
    Args:
        episode: Episode database object
        db: Database session
        out_dir: Output directory for final video
        
    Returns:
        Path to assembled episode video
    """
    # Get all scenes ordered by index
    scenes = sorted(episode.scenes, key=lambda s: s.index)
    
    if not scenes:
        raise ValueError("No scenes found for episode")
    
    # Collect video paths, ensuring all scenes are rendered
    video_paths = []
    missing_scenes = []
    
    for scene in scenes:
        if scene.output_video_path and pathlib.Path(scene.output_video_path).exists():
            video_paths.append(scene.output_video_path)
        else:
            missing_scenes.append(scene.index)
    
    if missing_scenes:
        raise ValueError(f"Missing rendered scenes: {missing_scenes}")
    
    # Create output directory
    episodes_dir = pathlib.Path(out_dir) / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    safe_title = "".join(c for c in episode.title if c.isalnum() or c in (' ', '-', '_')).strip()
    output_filename = f"ep_{episode.id}_{safe_title.replace(' ', '_')}.mp4"
    output_path = episodes_dir / output_filename
    
    try:
        # Create title card if episode has a title
        if episode.title:
            title_card_path = episodes_dir / f"title_ep_{episode.id}.mp4"
            create_title_card(episode.title, str(title_card_path), duration=3)
            video_paths.insert(0, str(title_card_path))
        
        # Concatenate all videos
        final_path = concat_videos(video_paths, str(output_path))
        
        # Update episode with final video path
        episode.final_video_path = final_path
        episode.status = "COMPLETE"
        db.commit()
        
        return final_path
        
    except Exception as e:
        print(f"Episode assembly failed: {e}")
        episode.status = "ERROR"
        db.commit()
        raise

def generate_voiceover_and_mux(episode: Episode, db: Session, out_dir: str) -> str:
    """
    Generate voiceover for episode script and mux with video.
    
    Args:
        episode: Episode database object
        db: Database session
        out_dir: Output directory
        
    Returns:
        Path to final video with voiceover
    """
    if not episode.final_video_path:
        raise ValueError("Episode must be assembled before adding voiceover")
    
    if not episode.script_text:
        raise ValueError("Episode must have script text for voiceover")
    
    # Create audio directory
    audio_dir = pathlib.Path(out_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate audio filename
    audio_filename = f"vo_ep_{episode.id}.wav"
    audio_path = audio_dir / audio_filename
    
    try:
        # Generate voiceover audio
        tts_to_wav(episode.script_text, str(audio_path))
        
        # Update episode with audio path
        episode.vo_audio_path = str(audio_path)
        
        # Create final output path
        episodes_dir = pathlib.Path(out_dir) / "episodes"
        final_filename = pathlib.Path(episode.final_video_path).stem + "_with_vo.mp4"
        final_path = episodes_dir / final_filename
        
        # Mux audio with video
        mux_audio(episode.final_video_path, str(audio_path), str(final_path))
        
        # Update episode paths
        episode.final_video_path = str(final_path)
        db.commit()
        
        return str(final_path)
        
    except Exception as e:
        print(f"Voiceover generation failed: {e}")
        db.commit()
        raise

def process_render_episode_job(episode_id: int, db: Session) -> Dict[str, Any]:
    """
    Process a job to render all scenes in an episode.
    
    Args:
        episode_id: Episode identifier
        db: Database session
        
    Returns:
        Job result dictionary
    """
    episode = db.query(Episode).filter(Episode.id == episode_id).first()
    if not episode:
        return {"success": False, "error": f"Episode {episode_id} not found"}
    
    episode.status = "RENDERING"
    db.commit()
    
    try:
        # Set up output directory
        media_root = pathlib.Path(settings.MEDIA_ROOT)
        project_dir = media_root / "projects" / str(episode.project_id)
        
        rendered_scenes = []
        failed_scenes = []
        
        # Render each scene
        for scene in sorted(episode.scenes, key=lambda s: s.index):
            try:
                video_path = render_scene(scene, episode, db, str(project_dir))
                rendered_scenes.append({
                    "scene_id": scene.id,
                    "index": scene.index,
                    "video_path": video_path,
                    "cost_usd": scene.cost_usd
                })
            except Exception as e:
                failed_scenes.append({
                    "scene_id": scene.id,
                    "index": scene.index,
                    "error": str(e)
                })
        
        # Update episode status
        if failed_scenes:
            episode.status = "ERROR"
        else:
            episode.status = "ASSEMBLING"
        
        db.commit()
        
        return {
            "success": len(failed_scenes) == 0,
            "episode_id": episode_id,
            "rendered_scenes": rendered_scenes,
            "failed_scenes": failed_scenes,
            "total_cost_usd": episode.veo_spend_usd
        }
        
    except Exception as e:
        episode.status = "ERROR"
        db.commit()
        return {"success": False, "error": str(e)}

def process_assemble_episode_job(episode_id: int, db: Session) -> Dict[str, Any]:
    """
    Process a job to assemble an episode from rendered scenes.
    
    Args:
        episode_id: Episode identifier
        db: Database session
        
    Returns:
        Job result dictionary
    """
    episode = db.query(Episode).filter(Episode.id == episode_id).first()
    if not episode:
        return {"success": False, "error": f"Episode {episode_id} not found"}
    
    try:
        # Set up output directory
        media_root = pathlib.Path(settings.MEDIA_ROOT)
        project_dir = media_root / "projects" / str(episode.project_id)
        
        # Assemble episode
        final_path = assemble_episode(episode, db, str(project_dir))
        
        return {
            "success": True,
            "episode_id": episode_id,
            "final_video_path": final_path,
            "total_duration_sec": sum(s.duration_sec for s in episode.scenes)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def process_voiceover_job(episode_id: int, db: Session) -> Dict[str, Any]:
    """
    Process a job to add voiceover to an assembled episode.
    
    Args:
        episode_id: Episode identifier
        db: Database session
        
    Returns:
        Job result dictionary
    """
    episode = db.query(Episode).filter(Episode.id == episode_id).first()
    if not episode:
        return {"success": False, "error": f"Episode {episode_id} not found"}
    
    try:
        # Set up output directory
        media_root = pathlib.Path(settings.MEDIA_ROOT)
        project_dir = media_root / "projects" / str(episode.project_id)
        
        # Generate voiceover and mux
        final_path = generate_voiceover_and_mux(episode, db, str(project_dir))
        
        return {
            "success": True,
            "episode_id": episode_id,
            "final_video_path": final_path,
            "audio_path": episode.vo_audio_path
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

