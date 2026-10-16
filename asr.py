"""Automatic Speech Recognition (ASR) module"""

import os
import tempfile
from typing import Optional


def transcribe_audio_whisper(audio_bytes: bytes) -> Optional[str]:
    """
    Transcribe audio using OpenAI Whisper model.

    Args:
        audio_bytes: Raw audio data in bytes

    Returns:
        Transcribed text, or None if transcription fails
    """
    try:
        import whisper

        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        try:
            # Load Whisper model (base model for balance of speed and accuracy)
            model = whisper.load_model("base")

            # Transcribe audio
            result = model.transcribe(temp_audio_path)
            transcript = result["text"].strip()

            return transcript if transcript else None

        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    except ImportError:
        print("Whisper not available, falling back to speech_recognition")
        return None
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return None


def transcribe_audio_sr(audio_bytes: bytes) -> Optional[str]:
    """
    Transcribe audio using speech_recognition library.
    This is the fallback method if Whisper is not available.

    Args:
        audio_bytes: Raw audio data in bytes

    Returns:
        Transcribed text, or None if transcription fails
    """
    try:
        import speech_recognition as sr

        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        try:
            # Initialize recognizer
            recognizer = sr.Recognizer()

            # Load audio file
            with sr.AudioFile(temp_audio_path) as source:
                audio_data = recognizer.record(source)

            # Transcribe using Google Speech Recognition (free)
            transcript = recognizer.recognize_google(audio_data)

            return transcript.strip() if transcript else None

        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    except ImportError:
        print("speech_recognition library not available")
        return None
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return None


def transcribe_audio(audio_bytes: bytes, prefer_whisper: bool = True) -> Optional[str]:
    """
    Transcribe audio using available ASR methods.
    Tries Whisper first (if preferred), then falls back to speech_recognition.

    Args:
        audio_bytes: Raw audio data in bytes
        prefer_whisper: Whether to try Whisper first (default: True)

    Returns:
        Transcribed text, or None if all methods fail
    """
    if not audio_bytes:
        return None

    # Try Whisper first if preferred
    if prefer_whisper:
        transcript = transcribe_audio_whisper(audio_bytes)
        if transcript:
            return transcript

        # Fallback to speech_recognition
        transcript = transcribe_audio_sr(audio_bytes)
        if transcript:
            return transcript

    # Try speech_recognition first if Whisper not preferred
    else:
        transcript = transcribe_audio_sr(audio_bytes)
        if transcript:
            return transcript

        # Fallback to Whisper
        transcript = transcribe_audio_whisper(audio_bytes)
        if transcript:
            return transcript

    return None


def transcribe_from_file(audio_file_path: str) -> Optional[str]:
    """
    Transcribe audio from a file path.

    Args:
        audio_file_path: Path to audio file

    Returns:
        Transcribed text, or None if transcription fails
    """
    try:
        with open(audio_file_path, 'rb') as f:
            audio_bytes = f.read()

        return transcribe_audio(audio_bytes)

    except Exception as e:
        print(f"Error reading audio file: {e}")
        return None
