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
st.set_page_config(page_title="Ask Snowflake — With Your Voice", page_icon="🎙️", layout="centered")

# =========================
# Global CSS (place BEFORE hero so it styles it)
# =========================
st.markdown("""
<style>
:root{
  --bg:#f7f8fb; --surface:#fff; --muted:#5b6675;
  --brand:#00A3E0; --brand-2:#29B5E8;
  --radius:14px; --shadow:0 10px 25px rgba(0,0,0,.08);
}
.block-container{padding-top:0!important;max-width:860px}
html,body{background:var(--bg)}

/* Hero */
.hero{
  position:relative;margin:12px 0 14px;padding:22px;
  border-radius:calc(var(--radius)+4px);
  background:linear-gradient(135deg,rgba(0,163,224,.20),rgba(41,181,232,.20));
  border:1px solid rgba(255,255,255,.7);box-shadow:var(--shadow);backdrop-filter:blur(6px)
}
.hero h1{margin:0;font-size:1.38rem;letter-spacing:.2px;color:#0f172a}
.hero .sub{color:var(--muted);font-size:.96rem;margin-top:8px}
.hero::after{
  content:"";position:absolute;right:18px;top:18px;width:90px;height:90px;border-radius:20px;
  background:radial-gradient(circle at 30% 30%,rgba(255,255,255,.9),rgba(255,255,255,.2) 60%);
  mix-blend-mode:screen;opacity:.55;pointer-events:none
}

/* Badge */
.badge{
  display:inline-flex;align-items:center;gap:6px;background:#ffffffcc;
  border:1px solid #e6edf5;color:#0f172a;padding:6px 10px;border-radius:999px;
  font-size:.80rem;box-shadow:var(--shadow);margin-bottom:8px
}

/* Chat bubbles */
[data-testid="stChatMessage"]>div{
  border-radius:var(--radius);box-shadow:var(--shadow);
  border:1px solid #eef0f5;padding:14px 16px!important
}
[data-testid="stChatMessage-user"]>div{background:#fff}
[data-testid="stChatMessage-assistant"]>div{background:#fafbff}

/* Typing bubble */
.typing{
  display:inline-flex;align-items:center;gap:6px;padding:8px 12px;border-radius:999px;
  background:#eef1f8;border:1px solid #e6e9f2
}
.typing .dot{
  width:8px;height:8px;border-radius:50%;background:#9aa3b2;
  animation:blink 1.2s infinite ease-in-out
}
.typing .dot:nth-child(2){animation-delay:.2s}
.typing .dot:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,80%,100%{opacity:.25}40%{opacity:1}}

/* Sticky bottom bar */
.bottom-shell{
  position:sticky;bottom:0;z-index:9;
  background:linear-gradient(180deg,rgba(247,248,251,0),var(--bg) 40%);
  padding-top:10px
}
.bottom-bar{
  display:flex;gap:10px;align-items:center;background:var(--surface);
  border:1px solid #e9edf3;border-radius:calc(var(--radius)+2px);
  padding:10px;box-shadow:var(--shadow)
}
.bottom-bar .stChatInput{flex:1}
.bottom-bar .stChatInput>div{margin:0!important}

/* Buttons */
.stButton>button{
  white-space:nowrap;background:linear-gradient(135deg,var(--brand),var(--brand-2));
  border:0;color:#fff;padding:10px 14px;border-radius:12px;box-shadow:var(--shadow)
}
.stButton>button:hover{filter:brightness(1.02)}
.clear-row{display:flex;justify-content:flex-end;margin:10px 0 6px}
.clear-row .stButton>button{
  background:#f3f4f6;color:#374151;border:1px solid #e5e7eb
}

/* Ensure button text never wraps weirdly */
.stButton>button{width:auto!important;max-width:none!important;display:inline-block}
</style>
""", unsafe_allow_html=True)

# =========================
# Hero header
# =========================
st.markdown("""
<div class="hero">
  <div class="badge">❄️ Powered by Snowflake Cortex</div>
  <h1>🎙️ Ask Snowflake — With Your Voice</h1>
  <div class="sub">Whisper transcription → Cortex Search retrieval → Cortex LLM reasoning — in real time.</div>
</div>
""", unsafe_allow_html=True)

# =========================
# Behavior knobs
# =========================
SILENCE_THRESH = float(os.getenv("SILENCE_THRESH", "0.01"))
MIN_DURATION_S = float(os.getenv("MIN_DURATION_S", "0.5"))

# Warm Snowflake (non-fatal)
try:
  warmup()
except Exception:
  pass

# =========================
# State
# =========================
ss = st.session_state
ss.setdefault("messages", [])                 # [{'role':'user'|'assistant','content': str}]
ss.setdefault("last_audio_sig", None)
ss.setdefault("auto_process", True)
ss.setdefault("pending_audio_bytes", None)    # queued audio blob
ss.setdefault("pending_query", None)          # queued text
ss.setdefault("proc_stage", None)             # None | "answering"
status_area = st.container()

def fingerprint(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

# =========================
# Clear Conversation (single line, no wrapping)
# =========================
st.markdown('<div class="clear-row">', unsafe_allow_html=True)
if st.button("🗑️ Clear Conversation", key="clear_chat"):
    ss.messages.clear()
    ss.pending_audio_bytes = None
    ss.pending_query = None
    ss.proc_stage = None
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Pending runner
#   - Ensures outputs render above bottom bar
#   - Keeps typing bubble right under the last message
# =========================
def run_pending():
    # If we have text pending: add user message, then answer
    if ss.pending_query is not None and ss.proc_stage is None:
        ss.messages.append({"role": "user", "content": ss.pending_query})
        ss.proc_stage = "answering"
        st.rerun()

    if ss.proc_stage == "answering" and ss.pending_query is not None:
        q = ss.pending_query
        with status_area:
            with st.spinner("🔎 Searching & reasoning in Snowflake…"):
                answer, meta = ask_with_search(q)
            st.caption(f"debug: {meta.get('search_debug','')}")
        ss.messages.append({"role": "assistant", "content": answer})
        ss.pending_query = None
        ss.proc_stage = None
        st.rerun()

    # If audio pending: analyze + transcribe → becomes a text query
    if ss.pending_audio_bytes is not None and ss.proc_stage is None:
        with status_area:
            with st.spinner("🎧 Processing recording…"):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav_path, _ = save_wav_bytes(ss.pending_audio_bytes, tmp.name)
                stats = analyze_wav(wav_path)
                voice_ok = (stats["duration_s"] >= MIN_DURATION_S) and (
                    stats["rms"] >= SILENCE_THRESH or stats["peak"] >= (SILENCE_THRESH * 3)
                )
                if not voice_ok:
                    st.error("It sounded too short or too quiet. Try closer to the mic.")
                    ss.pending_audio_bytes = None
                    st.rerun()
                st.caption(f"Audio {stats['duration_s']:.2f}s · RMS {stats['rms']:.4f} · Peak {stats['peak']:.4f}")
                st.caption("Transcribing with Whisper…")
                text = transcribe(wav_path).strip()
                if not text:
                    st.error("Transcription came back empty.")
                    ss.pending_audio_bytes = None
                    st.rerun()
                st.caption(f"Transcript: “{text[:110]}{'…' if len(text) > 110 else ''}”")
        ss.pending_audio_bytes = None
        ss.pending_query = text
        st.rerun()

# =========================
# Render chat history
# =========================
for m in ss.messages:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"], unsafe_allow_html=True)

# Typing indicator (right under the last message)
if ss.pending_query is not None and ss.proc_stage == "answering":
    with st.chat_message("assistant"):
        st.markdown('<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>', unsafe_allow_html=True)

# =========================
# Bottom bar (sticky): chat input + mic
# =========================
st.markdown('<div class="bottom-shell">', unsafe_allow_html=True)
st.markdown('<div class="bottom-bar">', unsafe_allow_html=True)

left, right = st.columns([1, 0.18], vertical_alignment="center")
with left:
    user_text = st.chat_input("Type a message…")
    if user_text:
        ss.pending_query = user_text.strip()
        st.rerun()

with right:
    audio_bytes = audio_recorder(
        text="Click for Voice",   # customize this label
        sample_rate=48_000,
        pause_threshold=1.0,      # auto-stop after ~1s silence
        icon_size="2x",
    )

st.markdown('</div>', unsafe_allow_html=True)   # end .bottom-bar

# Options row under the bar
opt1, opt2 = st.columns([0.32, 0.68])
with opt1:
    ss.auto_process = st.checkbox(
        "Auto transcribe",
        value=ss.auto_process,
        help="When on, we’ll transcribe & search right after you stop speaking."
    )
with opt2:
    if audio_bytes:
        st.caption("🎤 recorded — processing…" if ss.auto_process else "🎤 recorded — click to process")

st.markdown('</div>', unsafe_allow_html=True)   # end .bottom-shell

# =========================
# Queue new audio (don’t process inline)
# =========================
def _fp(b: bytes) -> str: return hashlib.md5(b).hexdigest()
if audio_bytes:
    sig = _fp(audio_bytes)
    if sig != ss.last_audio_sig:
        ss.last_audio_sig = sig
        if ss.auto_process:
            ss.pending_audio_bytes = audio_bytes
            st.rerun()
        else:
            with status_area:
                st.audio(audio_bytes, format="audio/wav")
                if st.button("Transcribe this recording", key="manual_transcribe"):
                    ss.pending_audio_bytes = audio_bytes
                    st.rerun()

# Run pending after UI so the typing bubble sits under the last message
run_pending()
