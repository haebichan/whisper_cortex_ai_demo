# services/transcribe.py
import os
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

BACKEND = os.getenv("TRANSCRIBE_BACKEND", "faster-whisper").lower()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")

# Lazy singletons
_faster_model = None
_openai_client = None

def transcribe(path_to_wav: str) -> str:
    """
    Returns the transcribed text from an audio file at path_to_wav.
    Backend selected by TRANSCRIBE_BACKEND env.
    """
    if BACKEND == "openai":
        return _transcribe_openai(path_to_wav)
    elif BACKEND == "whisper":
        return _transcribe_whisper(path_to_wav)
    else:
        return _transcribe_faster_whisper(path_to_wav)

def _transcribe_faster_whisper(path: str) -> str:
    global _faster_model
    from faster_whisper import WhisperModel
    if _faster_model is None:
        _faster_model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="auto")
    segments, info = _faster_model.transcribe(path, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments)

def _transcribe_openai(path: str) -> str:
    global _openai_client
    from openai import OpenAI
    if _openai_client is None:
        _openai_client = OpenAI()
    with open(path, "rb") as f:
        resp = _openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    # OpenAI returns .text
    return getattr(resp, "text", "").strip()

def _transcribe_whisper(path: str) -> str:
    import whisper  # PyTorch Whisper
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(path)
    return result.get("text", "").strip()
