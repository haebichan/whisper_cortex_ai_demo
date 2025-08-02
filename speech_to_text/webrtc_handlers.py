"""WebRTC stream handlers for audio processing."""

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
from .network import NetworkManager
from .ui_components import UIComponents
from .whisper_service import WhisperService


class AudioOnlyHandler:
    """Handles audio-only WebRTC streaming and transcription."""
    
    def __init__(
        self, 
        whisper_service: WhisperService,
        network_manager: NetworkManager,
        audio_processor: AudioProcessor,
        ui_components: UIComponents,
        app_config: AppConfig = DEFAULT_APP_CONFIG
    ):
        self.whisper_service = whisper_service
        self.network_manager = network_manager
        self.audio_processor = audio_processor
        self.ui_components = ui_components
        self.app_config = app_config
    
    def run(self, language: str) -> None:
        """Run the audio-only speech-to-text application."""
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=self.app_config.webrtc_audio_receiver_size,
            rtc_configuration=self.network_manager.get_rtc_configuration(),
            media_stream_constraints={"video": False, "audio": True},
        )

        status_indicator, text_output = self.ui_components.create_status_placeholders()
        
        if not webrtc_ctx.state.playing:
            self.ui_components.display_status(status_indicator, "Click 'START' to begin recording")
            return

        self._process_audio_stream(webrtc_ctx, status_indicator, text_output, language)
    
    def _process_audio_stream(self, webrtc_ctx, status_indicator, text_output, language: str) -> None:
        """Process the audio stream for transcription."""
        self.ui_components.display_status(status_indicator, "Loading Whisper model...")
        
        try:
            # Model is loaded through the service
            model = self.whisper_service.get_model()
            self.ui_components.display_status(status_indicator, "Model loaded. Listening...")
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

                self.ui_components.display_status(status_indicator, "🎤 Listening... Say something!")

                # Process audio frames
                self.audio_processor.add_audio_frames(audio_frames)

                # Transcribe if ready
                if self.audio_processor.should_transcribe():
                    self._transcribe_current_chunk(text_output, language)
            else:
                self.ui_components.display_status(status_indicator, "Audio receiver not available")
                break
    
    def _transcribe_current_chunk(self, text_output, language: str) -> None:
        """Transcribe the current audio chunk."""
        audio_array = self.audio_processor.get_audio_for_transcription()
        if audio_array is not None:
            text = self.whisper_service.transcribe(audio_array, language)
            
            if text:
                self.audio_processor.add_to_transcript(text)
                self.ui_components.display_transcription_results(
                    text_output, 
                    text, 
                    self.audio_processor.get_full_transcript()
                )
            
            self.audio_processor.reset_chunk()


class VideoAudioHandler:
    """Handles video+audio WebRTC streaming and transcription."""
    
    def __init__(
        self, 
        whisper_service: WhisperService,
        network_manager: NetworkManager,
        audio_processor: AudioProcessor,
        ui_components: UIComponents,
        app_config: AppConfig = DEFAULT_APP_CONFIG
    ):
        self.whisper_service = whisper_service
        self.network_manager = network_manager
        self.audio_processor = audio_processor
        self.ui_components = ui_components
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
        """Run the video+audio speech-to-text application."""
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text-w-video",
            mode=WebRtcMode.SENDRECV,
            queued_audio_frames_callback=self._queued_audio_frames_callback,
            rtc_configuration=self.network_manager.get_rtc_configuration(),
            media_stream_constraints={"video": True, "audio": True},
        )

        status_indicator, text_output = self.ui_components.create_status_placeholders()

        if not webrtc_ctx.state.playing:
            self.ui_components.display_status(status_indicator, "Click 'START' to begin")
            return

        self._process_video_audio_stream(webrtc_ctx, status_indicator, text_output, language)
    
    def _process_video_audio_stream(self, webrtc_ctx, status_indicator, text_output, language: str) -> None:
        """Process the video+audio stream for transcription."""
        self.ui_components.display_status(status_indicator, "Loading Whisper model...")
        
        try:
            model = self.whisper_service.get_model()
            self.ui_components.display_status(status_indicator, "Model loaded. Listening...")
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

                self.ui_components.display_status(status_indicator, "🎤 Listening... Say something!")

                # Process audio frames
                self.audio_processor.add_audio_frames(audio_frames)

                # Transcribe if ready
                if self.audio_processor.should_transcribe():
                    self._transcribe_current_chunk(text_output, language)
            else:
                self.ui_components.display_status(status_indicator, "Stopped.")
                break
    
    def _get_queued_frames(self) -> List[av.AudioFrame]:
        """Get all queued audio frames."""
        audio_frames = []
        with self.frames_deque_lock:
            while len(self.frames_deque) > 0:
                frame = self.frames_deque.popleft()
                audio_frames.append(frame)
        return audio_frames
    
    def _transcribe_current_chunk(self, text_output, language: str) -> None:
        """Transcribe the current audio chunk."""
        audio_array = self.audio_processor.get_audio_for_transcription()
        if audio_array is not None:
            text = self.whisper_service.transcribe(audio_array, language)
            
            if text:
                self.audio_processor.add_to_transcript(text)
                self.ui_components.display_transcription_results(
                    text_output, 
                    text, 
                    self.audio_processor.get_full_transcript()
                )
            
            self.audio_processor.reset_chunk()