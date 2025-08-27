"""Audio processing and TTS services"""

import os
import pathlib
import subprocess
from typing import Optional
from elevenlabs import ElevenLabs
from ..settings import get_settings

settings = get_settings()

def tts_to_wav(text: str, out_wav: str, voice_id: Optional[str] = None) -> str:
    """
    Convert text to speech using ElevenLabs TTS API.
    
    Args:
        text: Text to convert to speech
        out_wav: Output WAV file path
        voice_id: Optional voice ID override
        
    Returns:
        Path to generated audio file
    """
    if not settings.USE_TTS or not settings.ELEVENLABS_API_KEY:
        raise ValueError("ElevenLabs TTS requires API key. Set ELEVENLABS_API_KEY in environment.")
    
    # Ensure output directory exists
    pathlib.Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize ElevenLabs client
        client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
        
        # Use provided voice ID or default from settings
        voice = voice_id or settings.ELEVENLABS_VOICE_ID
        
        # Generate audio
        response = client.text_to_speech.convert(
            voice_id=voice,
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings={
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True
            }
        )
        
        # Save audio data to file
        with open(out_wav, 'wb') as f:
            for chunk in response:
                f.write(chunk)
        
        return out_wav
        
    except Exception as e:
        error_msg = f"ElevenLabs TTS generation failed: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg)



def mux_audio(video_path: str, audio_path: str, out_path: str, 
              audio_fade_in: float = 0.5, audio_fade_out: float = 0.5) -> str:
    """
    Combine video and audio tracks into final output.
    
    Args:
        video_path: Input video file path
        audio_path: Input audio file path
        out_path: Output video file path
        audio_fade_in: Audio fade-in duration in seconds
        audio_fade_out: Audio fade-out duration in seconds
        
    Returns:
        Path to final video with audio
    """
    # Ensure output directory exists
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Build audio filter with fades
    audio_filter = f"afade=in:st=0:d={audio_fade_in},afade=out:st=-{audio_fade_out}:d={audio_fade_out}"
    
    cmd = [
        'ffmpeg', '-i', video_path, '-i', audio_path,
        '-af', audio_filter,
        '-shortest',  # End when shortest input ends
        '-c:v', 'copy',  # Copy video without re-encoding
        '-c:a', 'aac',   # Encode audio as AAC
        '-y', out_path
    ]
    
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path

def adjust_audio_level(audio_path: str, out_path: str, volume_db: float = 0.0) -> str:
    """
    Adjust audio volume level.
    
    Args:
        audio_path: Input audio file path
        out_path: Output audio file path
        volume_db: Volume adjustment in decibels
        
    Returns:
        Path to adjusted audio file
    """
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-i', audio_path,
        '-af', f'volume={volume_db}dB',
        '-y', out_path
    ]
    
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path

def extract_audio(video_path: str, out_path: str) -> str:
    """
    Extract audio track from video.
    
    Args:
        video_path: Input video file path
        out_path: Output audio file path
        
    Returns:
        Path to extracted audio file
    """
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # Uncompressed audio
        '-y', out_path
    ]
    
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path

def add_background_music(video_path: str, music_path: str, out_path: str, 
                        music_volume: float = 0.3) -> str:
    """
    Add background music to video.
    
    Args:
        video_path: Input video file path
        music_path: Background music file path
        out_path: Output video file path
        music_volume: Music volume (0.0 to 1.0)
        
    Returns:
        Path to video with background music
    """
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-i', video_path, '-i', music_path,
        '-filter_complex', f'[1:a]volume={music_volume}[a1];[0:a][a1]amix=inputs=2:duration=first[a]',
        '-map', '0:v', '-map', '[a]',
        '-c:v', 'copy', '-c:a', 'aac',
        '-y', out_path
    ]
    
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path

def estimate_speech_duration(text: str, words_per_minute: int = 150) -> float:
    """
    Estimate speech duration from text.
    
    Args:
        text: Input text
        words_per_minute: Speaking rate
        
    Returns:
        Estimated duration in seconds
    """
    words = len(text.split())
    duration_minutes = words / words_per_minute
    return duration_minutes * 60
