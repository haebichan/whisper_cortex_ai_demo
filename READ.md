# Voice → Whisper → Cortex (Streamlit)

A tiny Streamlit app that records mic audio (no WebRTC), transcribes with Whisper, and sends the text to Snowflake Cortex. Results display in a simple chat UI.

## 1) Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
