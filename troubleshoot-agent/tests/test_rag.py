"""Tests for RAG engine: ingest, retrieve, rerank."""
import os
import sys
from pathlib import Path

import pytest

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import rag_engine


@pytest.fixture
def sample_doc(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text("""
# Battery drain
If your iPhone battery drains fast, go to Settings > Battery.
Check Battery Health. Turn on Low Power Mode.
""", encoding="utf-8")
    return str(path)


def test_retrieve_empty_returns_list():
    """Without ingestion, retrieve may return empty or use existing DB."""
    results = rag_engine.retrieve("iPhone battery", top_k=2)
    assert isinstance(results, list)


def test_ingest_document(sample_doc):
    """Ingest a single document returns chunk count."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    n = rag_engine.ingest_document(sample_doc, metadata={"source_file": "sample.md"})
    assert n >= 1


def test_rerank_results():
    """Rerank preserves list structure."""
    results = [
        {"content": "Battery settings", "relevance_score": 0.5},
        {"content": "WiFi settings", "relevance_score": 0.3},
    ]
    out = rag_engine.rerank_results(results, "battery drain")
    assert len(out) == 2
    assert "relevance_score" in out[0]
