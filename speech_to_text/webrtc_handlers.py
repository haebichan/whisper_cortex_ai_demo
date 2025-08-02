"""WebRTC stream handlers for audio processing with Cortex Search integration."""

import queue
import threading
import time
from collections import deque
from typing import List

import av
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from .audio_processor import AudioProcessor, create_silent_audio_frame
from .config import AppConfig, DEFAULT_APP_CONFIG
from .cortex_service import CortexSearchService
from .network import NetworkManager
from .ui_components import UIComponents
from .whisper_service import WhisperService


class AudioOnlyHandler:
    """Handles audio-only WebRTC streaming, transcription, and Cortex Search."""
    
    def __init__(
        self, 
        whisper_service: WhisperService,
        network_manager: NetworkManager,
        audio_processor: AudioProcessor,
        ui_components: UIComponents,
        cortex_service: CortexSearchService,
        auto_search: bool = True,
        app_config: AppConfig = DEFAULT_APP_CONFIG
    ):
        self.whisper_service = whisper_service
        self.network_manager = network_manager
        self.audio_processor = audio_processor
        self.ui_components = ui_components
        self.cortex_service = cortex_service
        self.auto_search = auto_search
        self.app_config = app_config
    
    def run(self, language: str) -> None:
        """Run the audio-only speech-to-text with Cortex Search application."""
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text-cortex",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=self.app_config.webrtc_audio_receiver_size,
            rtc_configuration=self.network_manager.get_rtc_configuration(),
            media_stream_constraints={"video": False, "audio": True},
        )

        status_indicator, transcription_output, search_output = self.ui_components.create_status_placeholders()
        
        if not webrtc_ctx.state.playing:
            self.ui_components.display_status(status_indicator, "Click 'START' to begin voice input", "info")
            return

        self._process_audio_stream(webrtc_ctx, status_indicator, transcription_output, search_output, language)
    
    def _process_audio_stream(self, webrtc_ctx, status_indicator, transcription_output, search_output, language: str) -> None:
        """Process the audio stream for transcription and search."""
        self.ui_components.display_status(status_indicator, "Loading Whisper model...", "processing")
        
        try:
            model = self.whisper_service.get_model()
            self.ui_components.display_status(status_indicator, "Ready! Start speaking...", "listening")
        except Exception as e:
            self.ui_components.display_error(f"Failed to load model: {e}")
            return
        
        while True:
            if webrtc_ctx.audio_receiver:
                try:
                    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                except queue.Empty:
                    time.sleep(0.1)
                    continue

                self.ui_components.display_status(status_indicator, "Listening... Speak your question!", "listening")

                # Process audio frames
                self.audio_processor.add_audio_frames(audio_frames)

                # Transcribe if ready
                if self.audio_processor.should_transcribe():
                    self._transcribe_and_search(status_indicator, transcription_output, search_output, language)
            else:
                self.ui_components.display_status(status_indicator, "Audio receiver not available", "error")
                break
    
    def _transcribe_and_search(self, status_indicator, transcription_output, search_output, language: str) -> None:
        """Transcribe the current audio chunk and perform Cortex Search."""
        audio_array = self.audio_processor.get_audio_for_transcription()
        if audio_array is not None:
            self.ui_components.display_status(status_indicator, "Transcribing speech...", "processing")
            
            # Transcribe the audio
            text = self.whisper_service.transcribe(audio_array, language)
            
            if text:
                # Display transcription
                self.ui_components.display_transcription_results(transcription_output, text)
                
                # Perform Cortex Search if auto-search is enabled
                if self.auto_search and self.cortex_service.is_available():
                    self.ui_components.display_status(status_indicator, "Searching knowledge base...", "searching")
                    
                    response, success = self.cortex_service.search_and_summarize(text)
                    self.ui_components.display_search_results(search_output, text, response, success)
                    
                    if success:
                        self.ui_components.display_status(status_indicator, "Search completed! Ready for next question.", "success")
                    else:
                        self.ui_components.display_status(status_indicator, "Search failed. Ready for next question.", "warning")
                elif not self.cortex_service.is_available():
                    self.ui_components.display_status(status_indicator, "Cortex Search unavailable. Transcription only.", "warning")
                else:
                    self.ui_components.display_status(status_indicator, "Transcription completed. Auto-search disabled.", "success")
            
            self.audio_processor.reset_chunk()


class VideoAudioHandler:
    """Handles video+audio WebRTC streaming, transcription, and Cortex Search."""
    
    def __init__(
        self, 
        whisper_service: WhisperService,
        network_manager: NetworkManager,
        audio_processor: AudioProcessor,
        ui_components: UIComponents,
        cortex_service: CortexSearchService,
        auto_search: bool = True,
        app_config: AppConfig = DEFAULT_APP_CONFIG
    ):
        self.whisper_service = whisper_service
        self.network_manager = network_manager
        self.audio_processor = audio_processor
        self.ui_components = ui_components
        self.cortex_service = cortex_service
        self.auto_search = auto_search
        self.app_config = app_config
        
        # Thread-safe frame queue
        self.frames_deque_lock = threading.Lock()
        self.frames_deque = deque([])
    
    async def _queued_audio_frames_callback(self, frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
        """Callback to queue audio frames and return silent frames."""
        with self.frames_deque_lock:
            self.frames_deque.extend(frames)

        # Return silent frames
        return [create_silent_audio_frame(frame) for frame in frames]
    
    def run(self, language: str) -> None:
        """Run the video+audio speech-to-text with Cortex Search application."""
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text-cortex-w-video",
            mode=WebRtcMode.SENDRECV,
            queued_audio_frames_callback=self._queued_audio_frames_callback,
            rtc_configuration=self.network_manager.get_rtc_configuration(),
            media_stream_constraints={"video": True, "audio": True},
        )

        status_indicator, transcription_output, search_output = self.ui_components.create_status_placeholders()

        if not webrtc_ctx.state.playing:
            self.ui_components.display_status(status_indicator, "Click 'START' to begin video chat", "info")
            return

        self._process_video_audio_stream(webrtc_ctx, status_indicator, transcription_output, search_output, language)
    
    def _process_video_audio_stream(self, webrtc_ctx, status_indicator, transcription_output, search_output, language: str) -> None:
        """Process the video+audio stream for transcription and search."""
        self.ui_components.display_status(status_indicator, "Loading Whisper model...", "processing")
        
        try:
            model = self.whisper_service.get_model()
            self.ui_components.display_status(status_indicator, "Ready! Start speaking...", "listening")
        except Exception as e:
            self.ui_components.display_error(f"Failed to load model: {e}")
            return

        while True:
            if webrtc_ctx.state.playing:
                # Get audio frames from the queue
                audio_frames = self._get_queued_frames()
                
                if len(audio_frames) == 0:
                    time.sleep(0.1)
                    continue

                self.ui_components.display_status(status_indicator, "Listening... Speak your question!", "listening")

                # Process audio frames
                self.audio_processor.add_audio_frames(audio_frames)

                # Transcribe if ready
                if self.audio_processor.should_transcribe():
                    self._transcribe_and_search(status_indicator, transcription_output, search_output, language)
            else:
                self.ui_components.display_status(status_indicator, "Stopped.", "info")
                break
    
    def _get_queued_frames(self) -> List[av.AudioFrame]:
        """Get all queued audio frames."""
        audio_frames = []
        with self.frames_deque_lock:
            while len(self.frames_deque) > 0:
                frame = self.frames_deque.popleft()
                audio_frames.append(frame)
        return audio_frames
    
    def _transcribe_and_search(self, status_indicator, transcription_output, search_output, language: str) -> None:
        """Transcribe the current audio chunk and perform Cortex Search."""
        audio_array = self.audio_processor.get_audio_for_transcription()
        if audio_array is not None:
            self.ui_components.display_status(status_indicator, "Transcribing speech...", "processing")
            
            # Transcribe the audio
            text = self.whisper_service.transcribe(audio_array, language)
            
            if text:
                # Display transcription
                self.ui_components.display_transcription_results(transcription_output, text)
                
                # Perform Cortex Search if auto-search is enabled
                if self.auto_search and self.cortex_service.is_available():
                    self.ui_components.display_status(status_indicator, "Searching knowledge base...", "searching")
                    
                    response, success = self.cortex_service.search_and_summarize(text)
                    self.ui_components.display_search_results(search_output, text, response, success)
                    
                    if success:
                        self.ui_components.display_status(status_indicator, "Search completed! Ready for next question.", "success")
                    else:
                        self.ui_components.display_status(status_indicator, "Search failed. Ready for next question.", "warning")
                elif not self.cortex_service.is_available():
                    self.ui_components.display_status(status_indicator, "Cortex Search unavailable. Transcription only.", "warning")
                else:
                    self.ui_components.display_status(status_indicator, "Transcription completed. Auto-search disabled.", "success")
            
            self.audio_processor.reset_chunk()