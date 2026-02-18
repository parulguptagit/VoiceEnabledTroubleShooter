"""
CLI script to batch-load documents from knowledge_base/docs into the vector DB.
Run from project root: python knowledge_base/ingest.py
"""
import logging
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import rag_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    docs_dir = Path(__file__).parent / "docs"
    if not docs_dir.is_dir():
        logger.error("Directory not found: %s", docs_dir)
        sys.exit(1)
    result = rag_engine.ingest_directory(str(docs_dir))
    logger.info("Ingest result: %s", result)
    print("Ingested", result.get("files", 0), "files,", result.get("chunks", 0), "chunks")
    return 0 if result.get("chunks", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
