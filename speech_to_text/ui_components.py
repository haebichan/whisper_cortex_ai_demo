"""UI components for the Speech-to-Text + Cortex Search application."""

import streamlit as st
from typing import Tuple, List, Dict, Any
from datetime import datetime

from .config import WhisperConfig, AudioConfig


class UIComponents:
    """Manages UI components and user interactions."""
    
    @staticmethod
    def render_header() -> None:
        """Render the main header and description."""
        st.header("🎤 Voice-Powered Cortex Search")
        st.markdown(
            """
        **Speak your question and get intelligent answers!**
        
        This application combines:
        - **OpenAI Whisper** for speech-to-text conversion
        - **Snowflake Cortex Search** for document retrieval
        - **LLM summarization** for intelligent responses
        
        Simply speak your question, and the system will transcribe it, search relevant documents, 
        and provide you with a comprehensive answer.
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
        st.sidebar.header("🔧 Audio Settings")
        
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
    def render_cortex_settings() -> Tuple[str, int, bool]:
        """Render Cortex Search settings sidebar and return settings."""
        st.sidebar.header("🔍 Cortex Search Settings")
        
        search_service = st.sidebar.text_input(
            "Cortex Search Service Name",
            value="haebi_cortex_search_service",
            help="Name of your Cortex Search service"
        )
        
        chunk_limit = st.sidebar.slider(
            "Document chunk limit",
            1, 10, 2,
            help="Number of document chunks to retrieve"
        )
        
        auto_search = st.sidebar.checkbox(
            "Auto-search on transcription",
            value=True,
            help="Automatically search when speech is transcribed"
        )
        
        return search_service, chunk_limit, auto_search
    
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
    def create_conversation_area() -> None:
        """Create the conversation display area."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display conversation history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    @staticmethod
    def add_message_to_conversation(role: str, content: str) -> None:
        """Add a message to the conversation."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        st.session_state.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    @staticmethod
    def create_status_placeholders() -> Tuple[st.empty, st.empty, st.empty]:
        """Create status indicator, transcription, and search result placeholders."""
        # Create containers for better organization
        status_container = st.container()
        results_container = st.container()
        
        with status_container:
            status_indicator = st.empty()
            transcription_output = st.empty()
        
        with results_container:
            search_output = st.empty()
            
        return status_indicator, transcription_output, search_output
    
    @staticmethod
    def display_transcription_results(transcription_output: st.empty, current_text: str) -> None:
        """Display transcription results."""
        if current_text:
            with transcription_output.container():
                st.success(f"🎤 **Transcribed:** {current_text}")
    
    @staticmethod
    def display_search_results(search_output: st.empty, query: str, response: str, success: bool) -> None:
        """Display search results."""
        with search_output.container():
            if success:
                st.info(f"🔍 **Query:** {query}")
                st.success(f"📝 **Answer:** {response}")
                
                # Add to conversation
                UIComponents.add_message_to_conversation("user", query)
                UIComponents.add_message_to_conversation("assistant", response)
            else:
                st.error(f"❌ **Search failed:** {response}")
    
    @staticmethod
    def display_status(status_indicator: st.empty, message: str, status_type: str = "info") -> None:
        """Display status message with different types."""
        status_icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "listening": "🎤",
            "processing": "⚙️",
            "searching": "🔍"
        }
        
        icon = status_icons.get(status_type, "ℹ️")
        status_indicator.write(f"{icon} {message}")
    
    @staticmethod
    def display_error(message: str) -> None:
        """Display error message."""
        st.error(f"❌ {message}")
    
    @staticmethod
    def display_cortex_connection_status(is_connected: bool, message: str) -> None:
        """Display Cortex Search connection status."""
        if is_connected:
            st.sidebar.success(f"✅ Cortex Search: Connected")
        else:
            st.sidebar.error(f"❌ Cortex Search: {message}")
    
    @staticmethod
    def render_manual_search() -> str:
        """Render manual search input as an optional fallback."""
        # Use session state to track if manual search is shown
        if "show_manual_search" not in st.session_state:
            st.session_state.show_manual_search = False
        
        # Toggle button
        if not st.session_state.show_manual_search:
            if st.button("📝 Need to type instead? Click here"):
                st.session_state.show_manual_search = True
                st.rerun()
            return ""
        
        # Show manual search form
        st.subheader("💬 Manual Search")
        
        # Use a form to prevent automatic resubmission
        with st.form("manual_search_form", clear_on_submit=True):
            manual_query = st.text_input(
                "Type your question here:",
                placeholder="What is the warranty policy?",
                help="You can also type your question instead of speaking"
            )
            col1, col2 = st.columns([1, 1])
            with col1:
                submitted = st.form_submit_button("🔍 Search")
            with col2:
                if st.form_submit_button("🎤 Back to Voice"):
                    st.session_state.show_manual_search = False
                    st.rerun()
            
            if submitted and manual_query:
                return manual_query
            
        return ""
    
    @staticmethod
    def render_conversation_controls() -> None:
        """Render conversation control buttons."""
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()