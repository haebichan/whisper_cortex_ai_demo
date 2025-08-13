# utils/audio.py
import numpy as np
import soundfile as sf

def save_wav_bytes(wav_bytes: bytes, out_path: str) -> tuple[str, int]:
    """Writes WAV bytes to out_path and returns (path, sample_rate)."""
    with open(out_path, "wb") as f:
        f.write(wav_bytes)
    data, sr = sf.read(out_path, dtype="float32", always_2d=False)
    return out_path, sr

def analyze_wav(path: str) -> dict:
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    s = np.clip(data, -1.0, 1.0)
    rms = float(np.sqrt((s**2).mean())) if s.size else 0.0
    peak = float(np.abs(s).max()) if s.size else 0.0
    secs = (len(s) / sr) if sr else 0.0
    return {"duration_s": secs, "rms": rms, "peak": peak, "sr": sr}
