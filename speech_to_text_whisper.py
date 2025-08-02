import logging
import logging.handlers
import queue
import threading
import time
import os
from collections import deque
from pathlib import Path
from typing import List
import tempfile

import av
import numpy as np
import pydub
import streamlit as st
from twilio.rest import Client
import whisper
import torch

from streamlit_webrtc import WebRtcMode, webrtc_streamer

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


# This code is based on https://github.com/whitphx/streamlit-webrtc/blob/c1fe3c783c9e8042ce0c95d789e833233fd82e74/sample_utils/turn.py
@st.cache_data  # type: ignore
def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.
    """
    # Ref: https://www.twilio.com/docs/stun-turn/api
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)
    token = client.tokens.create()
    return token.ice_servers


@st.cache_resource
def load_whisper_model(model_size="base"):
    """Load and cache the Whisper model"""
    return whisper.load_model(model_size)


def main():
    st.header("Real Time Speech-to-Text with OpenAI Whisper")
    st.markdown(
        """
This demo app is using [OpenAI Whisper](https://github.com/openai/whisper),
a robust speech recognition model.

Choose your preferred model size below. Larger models are more accurate but slower.
"""
    )

    # Model selection
    model_size = st.selectbox(
        "Choose Whisper model size",
        ["tiny", "base", "small", "medium", "large"],
        index=1,  # Default to "base"
        help="tiny: fastest, base: good balance, large: most accurate"
    )

    # Audio processing settings
    st.sidebar.header("Audio Settings")
    chunk_duration = st.sidebar.slider("Audio chunk duration (seconds)", 1, 10, 3)
    language = st.sidebar.selectbox(
        "Language (auto-detect if None)",
        [None, "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
        help="Leave as None for auto-detection"
    )

    sound_only_page = "Sound only (sendonly)"
    with_video_page = "With video (sendrecv)"
    app_mode = st.selectbox("Choose the app mode", [sound_only_page, with_video_page])

    if app_mode == sound_only_page:
        app_sst(model_size, chunk_duration, language)
    elif app_mode == with_video_page:
        app_sst_with_video(model_size, chunk_duration, language)


def transcribe_audio_chunk(audio_chunk, model, language=None):
    """Transcribe an audio chunk using Whisper"""
    try:
        # Convert audio chunk to numpy array
        if len(audio_chunk) == 0:
            return ""
        
        # Ensure audio is in the right format for Whisper
        audio_chunk = audio_chunk.set_channels(1).set_frame_rate(16000)
        audio_array = np.array(audio_chunk.get_array_of_samples(), dtype=np.float32)
        audio_array = audio_array / 32768.0  # Normalize to [-1, 1]
        
        # Only transcribe if we have enough audio (at least 0.5 seconds)
        if len(audio_array) < 8000:  # 0.5 seconds at 16kHz
            return ""
        
        # Transcribe with Whisper
        result = model.transcribe(
            audio_array,
            language=language,
            fp16=torch.cuda.is_available()
        )
        
        return result["text"].strip()
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""


def app_sst(model_size: str, chunk_duration: int, language: str):
    """Speech-to-text app with audio only"""
    
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": False, "audio": True},
    )

    status_indicator = st.empty()
    text_output = st.empty()
    
    if not webrtc_ctx.state.playing:
        status_indicator.write("Click 'START' to begin recording")
        return

    status_indicator.write("Loading Whisper model...")
    
    try:
        model = load_whisper_model(model_size)
        status_indicator.write("Model loaded. Listening...")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # Initialize variables for audio accumulation
    sound_chunk = pydub.AudioSegment.empty()
    last_transcription_time = time.time()
    full_transcript = ""
    
    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                continue

            status_indicator.write("🎤 Listening... Say something!")

            # Process new audio frames
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            # Transcribe when we have enough audio or enough time has passed
            current_time = time.time()
            chunk_length_seconds = len(sound_chunk) / 1000.0  # pydub uses milliseconds
            
            if (chunk_length_seconds >= chunk_duration or 
                current_time - last_transcription_time >= chunk_duration):
                
                if len(sound_chunk) > 500:  # At least 0.5 seconds
                    text = transcribe_audio_chunk(sound_chunk, model, language)
                    
                    if text:
                        full_transcript += text + " "
                        text_output.markdown(f"**Current:** {text}")
                        st.text_area("Full Transcript:", full_transcript, height=150)
                    
                    # Reset for next chunk
                    sound_chunk = pydub.AudioSegment.empty()
                    last_transcription_time = current_time
        else:
            status_indicator.write("Audio receiver not available")
            break


def app_sst_with_video(model_size: str, chunk_duration: int, language: str):
    """Speech-to-text app with video"""
    
    frames_deque_lock = threading.Lock()
    frames_deque: deque = deque([])

    async def queued_audio_frames_callback(frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
        with frames_deque_lock:
            frames_deque.extend(frames)

        # Return empty frames to be silent
        new_frames = []
        for frame in frames:
            input_array = frame.to_ndarray()
            new_frame = av.AudioFrame.from_ndarray(
                np.zeros(input_array.shape, dtype=input_array.dtype),
                layout=frame.layout.name,
            )
            new_frame.sample_rate = frame.sample_rate
            new_frames.append(new_frame)

        return new_frames

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text-w-video",
        mode=WebRtcMode.SENDRECV,
        queued_audio_frames_callback=queued_audio_frames_callback,
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": True, "audio": True},
    )

    status_indicator = st.empty()
    text_output = st.empty()

    if not webrtc_ctx.state.playing:
        status_indicator.write("Click 'START' to begin")
        return

    status_indicator.write("Loading Whisper model...")
    
    try:
        model = load_whisper_model(model_size)
        status_indicator.write("Model loaded. Listening...")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # Initialize variables
    sound_chunk = pydub.AudioSegment.empty()
    last_transcription_time = time.time()
    full_transcript = ""

    while True:
        if webrtc_ctx.state.playing:
            # Get audio frames from the queue
            audio_frames = []
            with frames_deque_lock:
                while len(frames_deque) > 0:
                    frame = frames_deque.popleft()
                    audio_frames.append(frame)

            if len(audio_frames) == 0:
                time.sleep(0.1)
                continue

            status_indicator.write("🎤 Listening... Say something!")

            # Process audio frames
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            # Transcribe when we have enough audio
            current_time = time.time()
            chunk_length_seconds = len(sound_chunk) / 1000.0
            
            if (chunk_length_seconds >= chunk_duration or 
                current_time - last_transcription_time >= chunk_duration):
                
                if len(sound_chunk) > 500:  # At least 0.5 seconds
                    text = transcribe_audio_chunk(sound_chunk, model, language)
                    
                    if text:
                        full_transcript += text + " "
                        text_output.markdown(f"**Current:** {text}")
                        st.text_area("Full Transcript:", full_transcript, height=150)
                    
                    # Reset for next chunk
                    sound_chunk = pydub.AudioSegment.empty()
                    last_transcription_time = current_time
        else:
            status_indicator.write("Stopped.")
            break


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()