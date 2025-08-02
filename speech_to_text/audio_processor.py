"""Audio processing utilities for speech-to-text conversion."""

import logging
import time
from typing import List, Optional

import av
import numpy as np
import pydub
import torch

from .config import AudioConfig, DEFAULT_AUDIO_CONFIG

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio processing and accumulation for transcription."""
    
    def __init__(self, config: AudioConfig = DEFAULT_AUDIO_CONFIG):
        self.config = config
        self.sound_chunk = pydub.AudioSegment.empty()
        self.last_transcription_time = time.time()
        self.full_transcript = ""
    
    def add_audio_frame(self, audio_frame: av.AudioFrame) -> None:
        """Add an audio frame to the current chunk."""
        sound = pydub.AudioSegment(
            data=audio_frame.to_ndarray().tobytes(),
            sample_width=audio_frame.format.bytes,
            frame_rate=audio_frame.sample_rate,
            channels=len(audio_frame.layout.channels),
        )
        self.sound_chunk += sound
    
    def add_audio_frames(self, audio_frames: List[av.AudioFrame]) -> None:
        """Add multiple audio frames to the current chunk."""
        for frame in audio_frames:
            self.add_audio_frame(frame)
    
    def should_transcribe(self) -> bool:
        """Check if we should transcribe the current audio chunk."""
        current_time = time.time()
        chunk_length_seconds = len(self.sound_chunk) / 1000.0  # pydub uses milliseconds
        
        return (
            chunk_length_seconds >= self.config.chunk_duration or 
            current_time - self.last_transcription_time >= self.config.chunk_duration
        ) and len(self.sound_chunk) > self.config.min_chunk_length_ms
    
    def get_audio_for_transcription(self) -> Optional[np.ndarray]:
        """
        Get the current audio chunk prepared for transcription.
        Returns None if the chunk is too short.
        """
        if len(self.sound_chunk) == 0:
            return None
        
        try:
            # Ensure audio is in the right format for Whisper
            audio_chunk = self.sound_chunk.set_channels(self.config.channels).set_frame_rate(self.config.sample_rate)
            audio_array = np.array(audio_chunk.get_array_of_samples(), dtype=np.float32)
            audio_array = audio_array / 32768.0  # Normalize to [-1, 1]
            
            # Only return if we have enough audio
            if len(audio_array) < self.config.min_audio_length_samples:
                return None
            
            return audio_array
        
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return None
    
    def reset_chunk(self) -> None:
        """Reset the current audio chunk after transcription."""
        self.sound_chunk = pydub.AudioSegment.empty()
        self.last_transcription_time = time.time()
    
    def add_to_transcript(self, text: str) -> None:
        """Add transcribed text to the full transcript."""
        if text:
            self.full_transcript += text + " "
    
    def get_full_transcript(self) -> str:
        """Get the complete transcript."""
        return self.full_transcript
    
    def clear_transcript(self) -> None:
        """Clear the full transcript."""
        self.full_transcript = ""


def create_silent_audio_frame(original_frame: av.AudioFrame) -> av.AudioFrame:
    """Create a silent audio frame with the same properties as the original."""
    input_array = original_frame.to_ndarray()
    new_frame = av.AudioFrame.from_ndarray(
        np.zeros(input_array.shape, dtype=input_array.dtype),
        layout=original_frame.layout.name,
    )
    new_frame.sample_rate = original_frame.sample_rate
    return new_frame