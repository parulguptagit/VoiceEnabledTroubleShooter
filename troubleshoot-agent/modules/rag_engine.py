"""
RAG engine: document ingestion, chunking, embedding, vector search, and reranking.

Technology choices (inline):
- Vector DB: ChromaDB — zero-infrastructure local persistence; ideal for dev; swap to Pinecone/Weaviate for production scale.
- Embedding: OpenAI text-embedding-3-small — best cost/quality for short troubleshooting chunks; 1536 dims; 62% cheaper than ada-002.
- Chunking: Recursive character splitter with semantic boundary detection at sentence ends.
"""
import logging
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# Import config from parent package (run from troubleshoot-agent root)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    TOP_K_RAG,
)

logger = logging.getLogger(__name__)

# Lazy-initialized globals
_embeddings: OpenAIEmbeddings | None = None
_client: OpenAI | None = None
_chroma_client: chromadb.PersistentClient | None = None
_collection_name = "troubleshoot_docs"
_reranker = None

# Scoring constants (documented in spec)
BOOST_SUPPORT_APPLE = 0.15
BOOST_RECENT_6MO = 0.10
PENALTY_SHORT_CHUNK = 0.05
MIN_TOKENS_FOR_PENALTY = 50


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    return _embeddings


def _get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _get_chroma() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
    return _chroma_client


def _get_collection():
    return _get_chroma().get_or_create_collection(
        name=_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed via OpenAI API (Chroma can use OpenAI embedding function)."""
    client = _get_openai_client()
    out = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [d.embedding for d in out.data]


def _approx_tokens(text: str) -> int:
    """Rough token count (~4 chars per token for English)."""
    return max(0, len(text) // 4)


def _parse_updated_date(metadata: dict) -> str | None:
    """Return 'updated' date string from metadata if present."""
    if not metadata:
        return None
    return metadata.get("updated") or metadata.get("updated_date")


def _is_recent_6mo(updated_str: str | None) -> bool:
    """True if updated within last 6 months (YYYY-MM-DD)."""
    if not updated_str:
        return False
    try:
        from datetime import datetime, timedelta
        dt = datetime.strptime(updated_str[:10], "%Y-%m-%d")
        return dt >= datetime.utcnow() - timedelta(days=180)
    except Exception:
        return False


def _source_domain(source_file: str) -> str:
    """Extract domain from source_file if it looks like a URL; else return as-is."""
    if "support.apple.com" in source_file:
        return "support.apple.com"
    if "discussions.apple.com" in source_file:
        return "discussions.apple.com"
    if "apple.com" in source_file:
        return "apple.com"
    return source_file


def ingest_document(filepath: str, metadata: dict | None = None) -> int:
    """
    Load, chunk, embed, and store a single document.
    Returns number of chunks created.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {filepath}")

    raw = path.read_text(encoding="utf-8", errors="replace")
    # Strip YAML frontmatter for chunking
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            raw = parts[2].strip()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(raw)
    if not chunks:
        logger.warning("No chunks produced for %s", filepath)
        return 0

    meta = metadata or {}
    meta.setdefault("source_file", path.name)
    meta.setdefault("updated", meta.get("updated", ""))

    collection = _get_collection()
    ids = [f"{path.name}_{i}" for i in range(len(chunks))]
    embeddings = _embed_texts(chunks)
    metadatas = []
    for i, c in enumerate(chunks):
        m = {**meta, "chunk_index": i, "content_preview": c[:100]}
        metadatas.append(m)

    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    logger.info("Ingested %s: %d chunks", filepath, len(chunks))
    return len(chunks)


def ingest_directory(dir_path: str) -> dict[str, Any]:
    """Batch ingest all .md, .txt, and .pdf files in a directory."""
    base = Path(dir_path)
    if not base.is_dir():
        return {"error": f"Not a directory: {dir_path}", "files": 0, "chunks": 0}

    total_chunks = 0
    files_ingested = []
    for ext in ("*.md", "*.txt", "*.pdf"):
        for path in base.glob(ext):
            try:
                if path.suffix.lower() == ".pdf":
                    # PDF: read with PyPDF2 or similar if available; else skip
                    try:
                        import pypdf
                        reader = pypdf.PdfReader(path)
                        raw = "\n".join(p.extract_text() or "" for p in reader.pages)
                        # Write to temp txt and ingest as text
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
                            f.write(raw)
                            tmp = f.name
                        n = ingest_document(tmp, metadata={"source_file": path.name})
                        Path(tmp).unlink(missing_ok=True)
                    except ImportError:
                        logger.warning("PDF support requires pypdf; skipping %s", path)
                        continue
                else:
                    n = ingest_document(str(path), metadata={"source_file": path.name})
                total_chunks += n
                files_ingested.append({"path": str(path), "chunks": n})
            except Exception as e:
                logger.exception("Failed to ingest %s: %s", path, e)

    return {"files": len(files_ingested), "chunks": total_chunks, "details": files_ingested}


def retrieve(
    query: str,
    top_k: int = TOP_K_RAG,
    filter_metadata: dict | None = None,
) -> list[dict]:
    """
    Embed query, search ChromaDB, return ranked results with:
    - content (str)
    - source_file (str)
    - relevance_score (float) — base cosine + boosts/penalties
    - has_images (bool)
    - image_urls (list[str])
    """
    collection = _get_collection()
    query_embedding = _embed_texts([query])[0]
    n_results = min(top_k * 2, collection.count())  # Fetch extra for rerank/score adjustments
    n_results = max(n_results, 1)

    where = None
    if filter_metadata:
        where = {"$and": [{"metadata." + k: v} for k, v in filter_metadata.items()]}

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    docs = result["documents"][0] if result["documents"] else []
    metadatas = result["metadatas"][0] if result["metadatas"] else []
    distances = result["distances"][0] if result["distances"] else []

    # Chroma cosine distance: 0 = identical, 2 = opposite. Convert to similarity: 1 - (d/2)
    out = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
        sim = 1.0 - (dist / 2.0) if dist is not None else 0.0
        source_file = (meta or {}).get("source_file", "unknown")
        domain = _source_domain(source_file)
        if domain == "support.apple.com":
            sim += BOOST_SUPPORT_APPLE
        updated = _parse_updated_date(meta or {})
        if _is_recent_6mo(updated):
            sim += BOOST_RECENT_6MO
        if _approx_tokens(doc or "") < MIN_TOKENS_FOR_PENALTY:
            sim -= PENALTY_SHORT_CHUNK
        sim = max(0.0, min(1.0, sim))
        has_images = (meta or {}).get("has_images", False)
        image_urls = (meta or {}).get("image_urls") or []
        if isinstance(image_urls, str):
            image_urls = [image_urls] if image_urls else []
        out.append({
            "content": doc or "",
            "source_file": source_file,
            "relevance_score": round(sim, 4),
            "has_images": bool(has_images),
            "image_urls": list(image_urls),
            "metadata": meta or {},
        })
    out.sort(key=lambda x: x["relevance_score"], reverse=True)
    return out[:top_k]


def rerank_results(results: list[dict], query: str) -> list[dict]:
    """Cross-encoder reranking using cross-encoder/ms-marco-MiniLM-L-6-v2."""
    if not results:
        return results
    try:
        from sentence_transformers import CrossEncoder
        global _reranker
        if _reranker is None:
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, r["content"]) for r in results]
        scores = _reranker.predict(pairs)
        for i, r in enumerate(results):
            r["relevance_score"] = float(scores[i])
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results
    except Exception as e:
        logger.warning("Reranking failed, keeping original order: %s", e)
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test: ensure config has OPENAI_API_KEY for embed
    if not OPENAI_API_KEY:
        print("Set OPENAI_API_KEY in .env to run RAG tests")
    else:
        # Ingest one doc if knowledge_base exists
        kb = Path(__file__).resolve().parent.parent / "knowledge_base" / "docs"
        if kb.is_dir():
            md_files = list(kb.glob("*.md"))
            if md_files:
                n = ingest_document(str(md_files[0]), metadata={"source_file": md_files[0].name})
                print(f"Ingested {md_files[0].name}: {n} chunks")
        results = retrieve("iPhone battery draining fast", top_k=3)
        for r in results:
            print(r["source_file"], r["relevance_score"], r["content"][:80])
        print("RAG engine __main__ OK")
