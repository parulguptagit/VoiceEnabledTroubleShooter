"""
Text-to-speech: OpenAI TTS (tts-1-hd, voice: nova). Optional ElevenLabs fallback.
Pre-processes text through audio_adapter.adapt_for_voice before synthesis.
"""
import io
import logging
from pathlib import Path
from typing import AsyncGenerator

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ELEVENLABS_API_KEY,
    OPENAI_API_KEY,
    TTS_FALLBACK_ELEVENLABS,
    TTS_OPENAI_MODEL,
    TTS_OPENAI_VOICE,
    TTS_SPEED,
)

logger = logging.getLogger(__name__)


def _adapt_for_voice(text: str) -> str:
    """Delegate to audio_adapter to avoid circular import at module load."""
    from modules.audio_adapter import adapt_for_voice
    return adapt_for_voice(text)


def synthesize(text: str, voice: str = TTS_OPENAI_VOICE, speed: float = TTS_SPEED) -> bytes:
    """
    Convert text to MP3 audio bytes. Text is pre-processed via audio_adapter.adapt_for_voice.
    Returns: MP3 audio bytes.
    """
    text = _adapt_for_voice(text)
    if not text.strip():
        return b""
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.audio.speech.create(
                model=TTS_OPENAI_MODEL,
                voice=voice,
                input=text,
                speed=speed,
            )
            return resp.content
        except Exception as e:
            logger.warning("OpenAI TTS failed: %s", e)
            if TTS_FALLBACK_ELEVENLABS and ELEVENLABS_API_KEY:
                return _synthesize_elevenlabs(text)
            return b""
    if TTS_FALLBACK_ELEVENLABS and ELEVENLABS_API_KEY:
        return _synthesize_elevenlabs(text)
    return b""


def _synthesize_elevenlabs(text: str) -> bytes:
    """Fallback: ElevenLabs TTS."""
    try:
        import httpx
        resp = httpx.post(
            "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            },
            json={"text": text, "model_id": "eleven_monolingual_v1"},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.exception("ElevenLabs TTS failed: %s", e)
        return b""


async def stream_synthesize(text: str) -> AsyncGenerator[bytes, None]:
    """Streaming TTS for long responses — yield MP3 chunks as they're generated."""
    # OpenAI TTS API doesn't support true streaming; we yield full response in one chunk
    # For real streaming you'd use a provider that supports it (e.g. ElevenLabs streaming)
    text = _adapt_for_voice(text)
    if not text.strip():
        return
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.audio.speech.create(
                model=TTS_OPENAI_MODEL,
                voice=TTS_OPENAI_VOICE,
                input=text,
                speed=TTS_SPEED,
            )
            yield resp.content
        except Exception as e:
            logger.warning("OpenAI TTS stream failed: %s", e)
    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = "Step 2 of 5. Go to Settings, then Battery. Tap Battery Health."
    out = synthesize(sample)
    print("TTS bytes length:", len(out))
    print("Voice TTS __main__ OK (set OPENAI_API_KEY for real synthesis)")
