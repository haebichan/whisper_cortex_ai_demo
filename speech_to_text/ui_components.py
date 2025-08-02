"""UI components for the Speech-to-Text application."""

import streamlit as st
from typing import Tuple

from .config import WhisperConfig, AudioConfig


class UIComponents:
    """Manages UI components and user interactions."""
    
    @staticmethod
    def render_header() -> None:
        """Render the main header and description."""
        st.header("Real Time Speech-to-Text with OpenAI Whisper")
        st.markdown(
            """
        This demo app is using [OpenAI Whisper](https://github.com/openai/whisper),
        a robust speech recognition model.

        Choose your preferred model size below. Larger models are more accurate but slower.
        """
        )
    
    @staticmethod
    def render_model_selection(whisper_config: WhisperConfig) -> str:
        """Render model selection UI and return selected model."""
        return st.selectbox(
            "Choose Whisper model size",
            whisper_config.available_models,
            index=whisper_config.available_models.index(whisper_config.model_size),
            help="tiny: fastest, base: good balance, large: most accurate"
        )
    
    @staticmethod
    def render_audio_settings(audio_config: AudioConfig, whisper_config: WhisperConfig) -> Tuple[int, str]:
        """Render audio settings sidebar and return settings."""
        st.sidebar.header("Audio Settings")
        
        chunk_duration = st.sidebar.slider(
            "Audio chunk duration (seconds)", 
            1, 10, audio_config.chunk_duration
        )
        
        language = st.sidebar.selectbox(
            "Language (auto-detect if None)",
            whisper_config.available_languages,
            index=0 if whisper_config.language is None else whisper_config.available_languages.index(whisper_config.language),
            help="Leave as None for auto-detection"
        )
        
        return chunk_duration, language
    
    @staticmethod
    def render_app_mode_selection() -> str:
        """Render app mode selection and return selected mode."""
        sound_only_page = "Sound only (sendonly)"
        with_video_page = "With video (sendrecv)"
        
        return st.selectbox(
            "Choose the app mode", 
            [sound_only_page, with_video_page]
        )
    
    @staticmethod
    def create_status_placeholders() -> Tuple[st.empty, st.empty]:
        """Create status indicator and text output placeholders."""
        status_indicator = st.empty()
        text_output = st.empty()
        return status_indicator, text_output
    
    @staticmethod
    def display_transcription_results(text_output: st.empty, current_text: str, full_transcript: str) -> None:
        """Display transcription results."""
        if current_text:
            text_output.markdown(f"**Current:** {current_text}")
        
        if full_transcript:
            st.text_area("Full Transcript:", full_transcript, height=150)
    
    @staticmethod
    def display_status(status_indicator: st.empty, message: str) -> None:
        """Display status message."""
        status_indicator.write(message)
    
    @staticmethod
    def display_error(message: str) -> None:
        """Display error message."""
        st.error(message)