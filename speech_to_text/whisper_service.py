"""Whisper model management and transcription service."""

import logging
from typing import Optional

import numpy as np
import streamlit as st
import torch
import whisper

from .config import WhisperConfig, DEFAULT_WHISPER_CONFIG

logger = logging.getLogger(__name__)


class WhisperService:
    """Manages Whisper model loading and transcription."""
    
    def __init__(self, config: WhisperConfig = DEFAULT_WHISPER_CONFIG):
        self.config = config
        self._model = None
    
    @st.cache_resource
    def load_model(_self, model_size: str):
        """Load and cache the Whisper model."""
        try:
            return whisper.load_model(model_size)
        except Exception as e:
            logger.error(f"Failed to load Whisper model {model_size}: {e}")
            raise
    
    def get_model(self):
        """Get the current Whisper model, loading it if necessary."""
        if self._model is None:
            self._model = self.load_model(self.config.model_size)
        return self._model
    
    def transcribe(self, audio_array: np.ndarray, language: Optional[str] = None) -> str:
        """
        Transcribe an audio array using Whisper.
        
        Args:
            audio_array: Audio data as numpy array
            language: Language code (None for auto-detection)
            
        Returns:
            Transcribed text
        """
        try:
            model = self.get_model()
            
            result = model.transcribe(
                audio_array,
                language=language or self.config.language,
                fp16=self.config.fp16 and torch.cuda.is_available()
            )
            
            return result["text"].strip()
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    def update_config(self, new_config: WhisperConfig) -> None:
        """Update the configuration and reload model if necessary."""
        if new_config.model_size != self.config.model_size:
            self._model = None  # Force model reload
        self.config = new_config