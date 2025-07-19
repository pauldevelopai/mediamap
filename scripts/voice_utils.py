"""
Voice utilities for Highlander application

This module provides text-to-speech synthesis functionality using ElevenLabs API.
"""
import requests
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variables
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
DEFAULT_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default voice ID

def synthesize_speech(text: str) -> str:
    """
    Convert text to speech using ElevenLabs API
    
    Args:
        text: Text to convert to speech
        
    Returns:
        Base64 encoded audio data
    """
    if not ELEVENLABS_API_KEY:
        # Return empty string if API key is not set
        print("Warning: ELEVENLABS_API_KEY not set. Using mock TTS.")
        return mock_synthesize_speech(text)
    
    try:
        payload = {
            "text": text,
            "voice_settings": {
                "stability": 0.7,
                "similarity_boost": 0.7
            }
        }
        
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{DEFAULT_VOICE_ID}",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            json=payload
        )
        
        if response.status_code == 200:
            audio_bytes = response.content
            return base64.b64encode(audio_bytes).decode()
        else:
            print(f"Error from ElevenLabs API: {response.status_code} - {response.text}")
            return mock_synthesize_speech(text)
            
    except Exception as e:
        print(f"Error synthesizing speech: {e}")
        return mock_synthesize_speech(text)

def mock_synthesize_speech(text: str) -> str:
    """
    Mock TTS function for testing or when API key is not available
    
    Args:
        text: Text to convert to speech
        
    Returns:
        Base64 encoded placeholder
    """
    # Return a placeholder base64 string (empty audio)
    return "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="
