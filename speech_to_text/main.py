"""Main application entry point for the Speech-to-Text + Cortex Search application."""

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
from .cortex_service import CortexSearchService
from .logger_setup import setup_logging
from .network import NetworkManager
from .ui_components import UIComponents
from .webrtc_handlers import AudioOnlyHandler, VideoAudioHandler
from .whisper_service import WhisperService

import streamlit as st

HERE = Path(__file__).parent


class VoiceCortexSearchApp:
    """Main Voice-Powered Cortex Search application."""
    
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
        
        # Initialize Cortex Search (will be configured from UI)
        self.cortex_service = None
    
    def run(self) -> None:
        """Run the main application."""
        # Render UI components
        self.ui_components.render_header()
        
        # Create two columns for main content and controls
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # Get user configurations in sidebar-style column
            model_size = self.ui_components.render_model_selection(self.whisper_config)
            chunk_duration, language = self.ui_components.render_audio_settings(
                self.audio_config, self.whisper_config
            )
            search_service, chunk_limit, auto_search = self.ui_components.render_cortex_settings()
            app_mode = self.ui_components.render_app_mode_selection()
            
            # Test Cortex connection
            self._initialize_cortex_service(search_service, chunk_limit)
        
        with col1:
            # Voice interface comes first - this is the main interaction
            st.subheader("🎤 Voice Interface")
            
            # Update configurations based on user input
            self._update_configurations(model_size, chunk_duration, language)
            
            # Run appropriate handler based on selected mode
            self._run_selected_mode(app_mode, language, auto_search)
            
            # Add some spacing
            st.markdown("---")
            
            # Optional manual search (hidden by default)
            manual_query = self.ui_components.render_manual_search()
            if manual_query:
                self._process_manual_query(manual_query)
            
            # Conversation area at the bottom
            st.subheader("💬 Conversation History")
            self.ui_components.create_conversation_area()
            
            # Conversation controls
            self.ui_components.render_conversation_controls()
    
    def _initialize_cortex_service(self, search_service: str, chunk_limit: int) -> None:
        """Initialize Cortex Search service and test connection."""
        try:
            self.cortex_service = CortexSearchService(search_service, chunk_limit)
            is_connected, message = self.cortex_service.test_connection()
            self.ui_components.display_cortex_connection_status(is_connected, message)
            
            if not is_connected:
                st.sidebar.warning("⚠️ Make sure you're running this in a Snowflake environment with access to Cortex Search.")
                
        except Exception as e:
            self.ui_components.display_cortex_connection_status(False, f"Initialization failed: {str(e)}")
            self.cortex_service = None
    
    def _process_manual_query(self, query: str) -> None:
        """Process a manual text query through Cortex Search."""
        # Only process if we haven't already processed this exact query in this session
        if "last_processed_query" not in st.session_state:
            st.session_state.last_processed_query = ""
        
        if st.session_state.last_processed_query == query:
            return  # Don't process the same query again
        
        if self.cortex_service and self.cortex_service.is_available():
            with st.spinner("Searching..."):
                response, success = self.cortex_service.search_and_summarize(query)
                
                if success:
                    # Add to conversation
                    self.ui_components.add_message_to_conversation("user", query)
                    self.ui_components.add_message_to_conversation("assistant", response)
                    st.session_state.last_processed_query = query
                    st.rerun()
                else:
                    st.error(f"Search failed: {response}")
        else:
            st.error("Cortex Search service is not available")
    
    def _update_configurations(self, model_size: str, chunk_duration: int, language: str) -> None:
        """Update configurations based on user input."""
        # Update audio config
        self.audio_config = AudioConfig(chunk_duration=chunk_duration)
        
        # Update whisper config
        whisper_config = WhisperConfig(model_size=model_size, language=language)
        self.whisper_service.update_config(whisper_config)
        self.whisper_config = whisper_config
    
    def _run_selected_mode(self, app_mode: str, language: str, auto_search: bool) -> None:
        """Run the selected application mode."""
        # Create fresh audio processor for each session
        audio_processor = AudioProcessor(self.audio_config)
        
        # Ensure we have a Cortex service (even if not connected)
        if self.cortex_service is None:
            self.cortex_service = CortexSearchService()
        
        if app_mode == "Sound only (sendonly)":
            handler = AudioOnlyHandler(
                self.whisper_service,
                self.network_manager,
                audio_processor,
                self.ui_components,
                self.cortex_service,
                auto_search,
                DEFAULT_APP_CONFIG
            )
            handler.run(language)
        
        elif app_mode == "With video (sendrecv)":
            handler = VideoAudioHandler(
                self.whisper_service,
                self.network_manager,
                audio_processor,
                self.ui_components,
                self.cortex_service,
                auto_search,
                DEFAULT_APP_CONFIG
            )
            handler.run(language)


def main() -> None:
    """Main entry point."""
    import streamlit as st
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="Voice-Powered Cortex Search",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    app = VoiceCortexSearchApp()
    app.run()


if __name__ == "__main__":
    main()