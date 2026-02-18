"""Tests for voice STT and TTS."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import voice_stt, voice_tts


def test_transcribe_empty_returns_dict():
    result = voice_stt.transcribe_audio(b"", format="webm")
    assert "text" in result
    assert "confidence" in result
    assert result["text"] == ""


def test_synthesize_empty_returns_empty_bytes():
    out = voice_tts.synthesize("")
    assert out == b""


def test_synthesize_returns_bytes():
    # May be empty if no API key
    out = voice_tts.synthesize("Hello")
    assert isinstance(out, bytes)
