"""
Speech-to-text: OpenAI Whisper API (primary); Deepgram as fallback.
Handles browser MediaRecorder blobs (webm/wav), normalizes audio with pydub.
"""
import io
import logging
import tempfile
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DEEPGRAM_API_KEY,
    OPENAI_API_KEY,
    STT_FALLBACK_DEEPGRAM,
    STT_OPENAI_MODEL,
)

logger = logging.getLogger(__name__)


def preprocess_audio(audio_bytes: bytes) -> bytes:
    """Normalize volume and trim leading/trailing silence using pydub."""
    if not audio_bytes:
        return audio_bytes
    try:
        from pydub import AudioSegment
        # Assume webm or wav; pydub needs ffmpeg for webm
        try:
            seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
        except Exception:
            try:
                seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
            except Exception:
                seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        # Normalize to -20 dBFS
        seg = seg.normalize()
        # Trim silence: 500ms min, -40 dBFS threshold
        seg = seg.strip_silence(silence_len=500, silence_thresh=-40, padding=100)
        out = io.BytesIO()
        seg.export(out, format="wav")
        return out.getvalue()
    except Exception as e:
        logger.warning("Audio preprocessing failed, using original: %s", e)
        return audio_bytes


def transcribe_audio(audio_bytes: bytes, format: str = "webm") -> dict[str, Any]:
    """
    Transcribe audio blob from browser MediaRecorder API.
    Returns: { "text": str, "confidence": float, "language": str, "duration_seconds": float }
    """
    audio_bytes = preprocess_audio(audio_bytes)
    if not audio_bytes:
        return {"text": "", "confidence": 0.0, "language": "en", "duration_seconds": 0.0}

    # Try OpenAI Whisper first
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            with tempfile.NamedTemporaryFile(suffix="." + format, delete=False) as f:
                f.write(audio_bytes)
                path = f.name
            try:
                with open(path, "rb") as af:
                    resp = client.audio.transcriptions.create(
                        model=STT_OPENAI_MODEL,
                        file=af,
                        response_format="verbose_json",
                    )
                text = getattr(resp, "text", "") or ""
                duration = getattr(resp, "duration", None)
                if duration is None and audio_bytes:
                    try:
                        from pydub import AudioSegment
                        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
                        duration = len(seg) / 1000.0
                    except Exception:
                        duration = 0.0
                return {
                    "text": text.strip(),
                    "confidence": 0.95 if text else 0.0,
                    "language": getattr(resp, "language", "en") or "en",
                    "duration_seconds": float(duration) if duration else 0.0,
                }
            finally:
                Path(path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Whisper transcription failed: %s", e)
            if STT_FALLBACK_DEEPGRAM and DEEPGRAM_API_KEY:
                return _transcribe_deepgram(audio_bytes)
            return {"text": "", "confidence": 0.0, "language": "en", "duration_seconds": 0.0}

    if STT_FALLBACK_DEEPGRAM and DEEPGRAM_API_KEY:
        return _transcribe_deepgram(audio_bytes)
    return {"text": "", "confidence": 0.0, "language": "en", "duration_seconds": 0.0}


def _transcribe_deepgram(audio_bytes: bytes) -> dict[str, Any]:
    """Fallback: Deepgram Nova-2."""
    try:
        import httpx
        resp = httpx.post(
            "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true",
            headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            content=audio_bytes,
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        channel = (data.get("results") or {}).get("channels") or []
        alt = (channel[0] or {}).get("alternatives") or []
        best = alt[0] if alt else {}
        return {
            "text": (best.get("transcript") or "").strip(),
            "confidence": float(best.get("confidence", 0.9)),
            "language": "en",
            "duration_seconds": float((data.get("metadata") or {}).get("duration") or 0),
        }
    except Exception as e:
        logger.exception("Deepgram fallback failed: %s", e)
        return {"text": "", "confidence": 0.0, "language": "en", "duration_seconds": 0.0}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test with empty/silence (no API key required for path test)
    result = transcribe_audio(b"", format="webm")
    print("Empty input result:", result)
    print("Voice STT __main__ OK (run with real audio + OPENAI_API_KEY for full test)")
