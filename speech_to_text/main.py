"""Main application entry point for the Speech-to-Text application."""

from pathlib import Path

from .audio_processor import AudioProcessor
from .config import (
    AudioConfig,
    WhisperConfig,
    DEFAULT_APP_CONFIG,
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_WHISPER_CONFIG,
    DEFAULT_TWILIO_CONFIG
)
from .logger_setup import setup_logging
from .network import NetworkManager
from .ui_components import UIComponents
from .webrtc_handlers import AudioOnlyHandler, VideoAudioHandler
from .whisper_service import WhisperService

HERE = Path(__file__).parent


class SpeechToTextApp:
    """Main Speech-to-Text application."""
    
    def __init__(self):
        # Setup logging
        setup_logging(DEFAULT_APP_CONFIG)
        
        # Initialize services
        self.network_manager = NetworkManager(DEFAULT_TWILIO_CONFIG)
        self.whisper_service = WhisperService(DEFAULT_WHISPER_CONFIG)
        self.ui_components = UIComponents()
        
        # Initialize configurations
        self.audio_config = DEFAULT_AUDIO_CONFIG
        self.whisper_config = DEFAULT_WHISPER_CONFIG
    
    def run(self) -> None:
        """Run the main application."""
        # Render UI components
        self.ui_components.render_header()
        
        # Get user configurations
        model_size = self.ui_components.render_model_selection(self.whisper_config)
        chunk_duration, language = self.ui_components.render_audio_settings(
            self.audio_config, self.whisper_config
        )
        app_mode = self.ui_components.render_app_mode_selection()
        
        # Update configurations based on user input
        self._update_configurations(model_size, chunk_duration, language)
        
        # Run appropriate handler based on selected mode
        self._run_selected_mode(app_mode, language)
    
    def _update_configurations(self, model_size: str, chunk_duration: int, language: str) -> None:
        """Update configurations based on user input."""
        # Update audio config
        self.audio_config = AudioConfig(chunk_duration=chunk_duration)
        
        # Update whisper config
        whisper_config = WhisperConfig(model_size=model_size, language=language)
        self.whisper_service.update_config(whisper_config)
        self.whisper_config = whisper_config
    
    def _run_selected_mode(self, app_mode: str, language: str) -> None:
        """Run the selected application mode."""
        # Create fresh audio processor for each session
        audio_processor = AudioProcessor(self.audio_config)
        
        if app_mode == "Sound only (sendonly)":
            handler = AudioOnlyHandler(
                self.whisper_service,
                self.network_manager,
                audio_processor,
                self.ui_components,
                DEFAULT_APP_CONFIG
            )
            handler.run(language)
        
        elif app_mode == "With video (sendrecv)":
            handler = VideoAudioHandler(
                self.whisper_service,
                self.network_manager,
                audio_processor,
                self.ui_components,
                DEFAULT_APP_CONFIG
            )
            handler.run(language)


def main() -> None:
    """Main entry point."""
    app = SpeechToTextApp()
    app.run()


if __name__ == "__main__":
    main()