"""Tests for LLM agent and context."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import context_manager, llm_agent


def test_detect_frustration():
    assert llm_agent.detect_frustration("that didn't work") is True
    assert llm_agent.detect_frustration("I tried that, it still drains fast") is True
    assert llm_agent.detect_frustration("What should I do?") is False


def test_build_context_prompt():
    rag = [{"content": "Battery tips", "source_file": "battery.md"}]
    web = [{"title": "Apple", "url": "https://apple.com", "content": "Support"}]
    out = llm_agent.build_context_prompt(rag, web, "Image: settings screen")
    assert "Battery tips" in out
    assert "Apple" in out
    assert "Image" in out or "settings" in out


def test_conversation_context():
    ctx = context_manager.ConversationContext("test-session")
    ctx.add_turn("user", "Battery drains")
    ctx.add_turn("assistant", "Check Settings > Battery")
    hist = ctx.get_history(max_tokens=1000)
    assert len(hist) == 2
    assert ctx.detect_issue_category() == "battery"
    assert ctx.export_session()["session_id"] == "test-session"
