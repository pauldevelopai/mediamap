"""FFmpeg video processing and assembly tools"""

import os
import pathlib
import subprocess
import tempfile
from typing import List, Optional, Tuple

def concat_videos(video_paths: List[str], out_path: str) -> str:
    """
    Concatenate multiple videos into a single output video.
    
    Args:
        video_paths: List of input video file paths
        out_path: Output video file path
        
    Returns:
        Path to the concatenated video
    """
    if not video_paths:
        raise ValueError("No video paths provided")
    
    # Ensure output directory exists
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file list for ffmpeg concat
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for video_path in video_paths:
            # Ensure path exists
            if not pathlib.Path(video_path).exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            # Use forward slashes for cross-platform compatibility
            f.write(f"file '{pathlib.Path(video_path).as_posix()}'\n")
        list_file = f.name
    
    try:
        # Use ffmpeg concat demuxer for lossless concatenation
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0', '-i', list_file,
            '-c', 'copy', '-y', out_path
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out_path
        
    except subprocess.CalledProcessError as e:
        # If concat fails, try re-encoding approach
        print(f"Concat failed, trying re-encode: {e}")
        return _concat_with_reencoding(video_paths, out_path)
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(list_file)
        except:
            pass

def _concat_with_reencoding(video_paths: List[str], out_path: str) -> str:
    """
    Concatenate videos with re-encoding (fallback method).
    
    Args:
        video_paths: List of input video file paths
        out_path: Output video file path
        
    Returns:
        Path to the concatenated video
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for video_path in video_paths:
            f.write(f"file '{pathlib.Path(video_path).as_posix()}'\n")
        list_file = f.name
    
    try:
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0', '-i', list_file,
            '-c:v', 'libx264', '-c:a', 'aac', '-preset', 'fast',
            '-y', out_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out_path
        
    finally:
        try:
            os.unlink(list_file)
        except:
            pass

def add_fade_transitions(video_paths: List[str], out_path: str, fade_duration: float = 0.5) -> str:
    """
    Concatenate videos with fade transitions between clips.
    
    Args:
        video_paths: List of input video file paths
        out_path: Output video file path
        fade_duration: Duration of fade transition in seconds
        
    Returns:
        Path to the video with transitions
    """
    if len(video_paths) < 2:
        # No transitions needed for single video
        return concat_videos(video_paths, out_path)
    
    # Build complex filter for crossfade transitions
    filter_parts = []
    input_args = []
    
    # Add all input files
    for i, video_path in enumerate(video_paths):
        input_args.extend(['-i', video_path])
    
    # Build crossfade filter chain
    current_label = "[0:v]"
    for i in range(1, len(video_paths)):
        output_label = f"[v{i}]" if i < len(video_paths) - 1 else ""
        filter_parts.append(
            f"{current_label}[{i}:v]xfade=transition=fade:duration={fade_duration}:offset=0{output_label}"
        )
        current_label = f"[v{i}]"
    
    filter_complex = ";".join(filter_parts)
    
    # Ensure output directory exists
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    cmd = ['ffmpeg'] + input_args + [
        '-filter_complex', filter_complex,
        '-c:a', 'aac', '-y', out_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out_path
    except subprocess.CalledProcessError:
        # Fallback to simple concatenation
        print("Fade transitions failed, falling back to simple concat")
        return concat_videos(video_paths, out_path)

def resize_video(input_path: str, output_path: str, width: int, height: int) -> str:
    """
    Resize video to specified dimensions.
    
    Args:
        input_path: Input video file path
        output_path: Output video file path
        width: Target width in pixels
        height: Target height in pixels
        
    Returns:
        Path to resized video
    """
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-i', input_path,
        '-vf', f'scale={width}:{height}',
        '-c:a', 'copy', '-y', output_path
    ]
    
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path

def extract_frame(video_path: str, timestamp: str, output_path: str) -> str:
    """
    Extract a frame from video at specified timestamp.
    
    Args:
        video_path: Input video file path
        timestamp: Timestamp in format "HH:MM:SS" or seconds
        output_path: Output image file path
        
    Returns:
        Path to extracted frame image
    """
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-i', video_path, '-ss', timestamp,
        '-vframes', '1', '-y', output_path
    ]
    
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path

def get_video_info(video_path: str) -> dict:
    """
    Get video information using ffprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    import json
    probe_data = json.loads(result.stdout)
    
    # Extract relevant info
    video_stream = next((s for s in probe_data['streams'] if s['codec_type'] == 'video'), None)
    if not video_stream:
        raise ValueError("No video stream found")
    
    return {
        'duration': float(probe_data['format']['duration']),
        'width': int(video_stream['width']),
        'height': int(video_stream['height']),
        'fps': eval(video_stream['r_frame_rate']),  # Convert fraction to float
        'codec': video_stream['codec_name'],
        'bitrate': int(probe_data['format'].get('bit_rate', 0))
    }

def create_title_card(text: str, output_path: str, duration: int = 3, 
                     width: int = 1280, height: int = 720) -> str:
    """
    Create a title card video with text.
    
    Args:
        text: Text to display
        output_path: Output video file path
        duration: Duration in seconds
        width: Video width
        height: Video height
        
    Returns:
        Path to created title card video
    """
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Escape text for ffmpeg
    escaped_text = text.replace("'", "\\'").replace(":", "\\:")
    
    cmd = [
        'ffmpeg', '-f', 'lavfi',
        '-i', f'color=c=black:s={width}x{height}:d={duration}',
        '-vf', f"drawtext=text='{escaped_text}':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2",
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-y', output_path
    ]
    
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path

