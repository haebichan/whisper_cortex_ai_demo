"""Configuration management for the Speech-to-Text application."""

import os
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    chunk_duration: int = 3
    sample_rate: int = 16000
    channels: int = 1
    min_chunk_length_ms: int = 500
    min_audio_length_samples: int = 8000  # 0.5 seconds at 16kHz


@dataclass
class WhisperConfig:
    """Whisper model configuration."""
    model_size: str = "base"
    language: Optional[str] = None
    fp16: bool = True
    
    @property
    def available_models(self) -> List[str]:
        return ["tiny", "base", "small", "medium", "large"]
    
    @property
    def available_languages(self) -> List[Optional[str]]:
        return [None, "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"]


@dataclass
class TwilioConfig:
    """Twilio configuration for TURN servers."""
    account_sid: Optional[str] = None
    auth_token: Optional[str] = None
    
    def __post_init__(self):
        self.account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        self.auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    
    @property
    def is_configured(self) -> bool:
        return bool(self.account_sid and self.auth_token)


@dataclass
class AppConfig:
    """Main application configuration."""
    debug: bool = False
    webrtc_audio_receiver_size: int = 1024
    
    def __post_init__(self):
        self.debug = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]


# Default configurations
DEFAULT_AUDIO_CONFIG = AudioConfig()
DEFAULT_WHISPER_CONFIG = WhisperConfig()
DEFAULT_TWILIO_CONFIG = TwilioConfig()
DEFAULT_APP_CONFIG = AppConfig()