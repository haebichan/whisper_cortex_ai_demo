# app.py
import os
import tempfile
import hashlib
from dotenv import load_dotenv

import streamlit as st
from audio_recorder_streamlit import audio_recorder

from utils.audio import save_wav_bytes, analyze_wav
from services.transcribe import transcribe
from services.rag import ask_with_search, warmup

load_dotenv()
st.set_page_config(page_title="Voice → Whisper → Cortex", page_icon="🎙️", layout="centered")

# ----- CSS -----
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
[data-testid="stChatMessage"] { margin-bottom: .6rem; }
.bottom-bar { border-top: 1px solid #eee; padding-top: .4rem; margin-top: .6rem; }

/* typing bubble */
.typing {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 8px 10px; border-radius: 12px; background: #f5f5f5;
}
.typing .dot {
  width: 8px; height: 8px; border-radius: 50%; background: #999;
  animation: blink 1.2s infinite ease-in-out;
}
.typing .dot:nth-child(2) { animation-delay: .2s; }
.typing .dot:nth-child(3) { animation-delay: .4s; }
@keyframes blink { 0%, 80%, 100% { opacity: .2 } 40% { opacity: 1 } }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* keep buttons on a single line + let them size naturally */
.stButton > button {
  white-space: nowrap;
  width: auto !important;
  max-width: none !important;
  display: inline-block;
}
</style>
""", unsafe_allow_html=True)

st.title("🎙️ Voice → Whisper → Cortex")

SILENCE_THRESH = float(os.getenv("SILENCE_THRESH", "0.01"))
MIN_DURATION_S = float(os.getenv("MIN_DURATION_S", "0.5"))

try:
    warmup()
except Exception:
    pass

# ---------- state ----------
ss = st.session_state
ss.setdefault("messages", [])                 # [{'role':'user'|'assistant','content': str}]
ss.setdefault("last_audio_sig", None)
ss.setdefault("auto_process", True)
ss.setdefault("pending_audio_bytes", None)
ss.setdefault("pending_query", None)
ss.setdefault("proc_stage", None)
ss.setdefault("typing_msg_index", None)
status_area = st.empty()

def fingerprint(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

# ---------- Clear Conversation ----------
clear_col, _ = st.columns([1, 6])
with clear_col:
    # Clear Conversation - full width
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        ss.messages.clear()
        ss.pending_audio_bytes = None
        ss.pending_query = None
        ss.proc_stage = None
        st.rerun()


# ---------- pending runner ----------
def run_pending():
    if ss.pending_query is not None and ss.proc_stage is None:
        ss.messages.append({"role": "user", "content": ss.pending_query})
        ss.proc_stage = "answering"
        st.rerun()

    if ss.proc_stage == "answering" and ss.pending_query is not None:
        q = ss.pending_query
        with status_area.container():
            with st.spinner("Searching & reasoning in Snowflake…"):
                answer, meta = ask_with_search(q)
        ss.messages.append({"role": "assistant", "content": answer})
        ss.pending_query = None
        ss.proc_stage = None
        st.rerun()

    if ss.pending_audio_bytes is not None and ss.proc_stage is None:
        with status_area.container():
            with st.spinner("Processing recording…"):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav_path, _ = save_wav_bytes(ss.pending_audio_bytes, tmp.name)
                stats = analyze_wav(wav_path)
                voice_ok = (stats["duration_s"] >= MIN_DURATION_S) and (
                    stats["rms"] >= SILENCE_THRESH or stats["peak"] >= (SILENCE_THRESH * 3)
                )
                if not voice_ok:
                    st.error("Looks silent or too short. Try speaking closer to the mic.")
                    ss.pending_audio_bytes = None
                    st.rerun()
                text = transcribe(wav_path).strip()
                if not text:
                    st.error("Transcription came back empty.")
                    ss.pending_audio_bytes = None
                    st.rerun()
        ss.pending_audio_bytes = None
        ss.pending_query = text
        st.rerun()

# ---------- chat history ----------
for m in ss.messages:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"], unsafe_allow_html=True)

# ---------- typing bubble (placed right after last message) ----------
if ss.pending_query is not None and ss.proc_stage == "answering":
    with st.chat_message("assistant"):
        st.markdown('<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>', unsafe_allow_html=True)

# ---------- bottom bar ----------
st.markdown('<div class="bottom-bar"></div>', unsafe_allow_html=True)
left, right = st.columns([6, 1], vertical_alignment="center")

with left:
    user_text = st.chat_input("Type a message or use the mic…")
    if user_text:
        ss.pending_query = user_text.strip()
        st.rerun()

with right:
    audio_bytes = audio_recorder(
        text = "Click for Voice",
        sample_rate=48_000,
        pause_threshold=1.0,
        icon_size="2x",
    )

opt1, opt2 = st.columns([2, 5])
with opt1:
    ss.auto_process = st.checkbox(
        "Auto transcribe",
        value=ss.auto_process,
        help="Automatically transcribe & search when recording stops.",
    )
with opt2:
    if audio_bytes:
        st.caption("🎤 recorded — processing…" if ss.auto_process else "🎤 recorded — click to process")

if audio_bytes:
    sig = fingerprint(audio_bytes)
    if sig != ss.last_audio_sig:
        ss.last_audio_sig = sig
        if ss.auto_process:
            ss.pending_audio_bytes = audio_bytes
            st.rerun()
        else:
            with status_area.container():
                st.audio(audio_bytes, format="audio/wav")
                if st.button("Transcribe this recording"):
                    ss.pending_audio_bytes = audio_bytes
                    st.rerun()

# run pending *after* UI so bubble is placed correctly
run_pending()
