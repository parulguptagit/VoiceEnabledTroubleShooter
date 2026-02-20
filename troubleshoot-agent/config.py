"""
Central configuration for the iPhone Troubleshooting Agent.
Loads secrets from .env; defines model names, vector DB, and feature flags.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from this package's directory so it works regardless of CWD
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)

# --- Embedding & LLM ---
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI; 1536 dims, cost-effective
LLM_MODEL = "claude-opus-4-6"    # Anthropic Claude via anthropic SDK

# --- Vector DB (ChromaDB: zero-infra, persistent; swap to Pinecone/Weaviate for prod) ---
VECTOR_DB = "chromadb"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(Path(__file__).parent / "chroma_store"))

# --- Chunking (recursive splitter; semantic boundaries at sentence ends) ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# --- Retrieval ---
TOP_K_RAG = 5
TOP_K_WEB = 3
RAG_SCORE_THRESHOLD = 0.75  # Below this → trigger web search fallback

# --- Web search (Tavily: RAG/agent-friendly, clean parsed results) ---
WEB_SEARCH_PRIORITY_DOMAINS = [
    "support.apple.com",
    "discussions.apple.com",
    "apple.com",
]

# --- API keys (from .env) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")  # Optional STT fallback
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")  # Optional TTS fallback

# --- Voice: STT (Whisper primary; Deepgram fallback) ---
STT_PROVIDER = "openai"  # "openai" | "deepgram"
STT_OPENAI_MODEL = "whisper-1"
STT_FALLBACK_DEEPGRAM = bool(DEEPGRAM_API_KEY)

# --- Voice: TTS (OpenAI TTS primary; ElevenLabs fallback) ---
TTS_PROVIDER = "openai"  # "openai" | "elevenlabs"
TTS_OPENAI_MODEL = "tts-1-hd"
TTS_OPENAI_VOICE = "nova"
TTS_SPEED = 0.95
TTS_FALLBACK_ELEVENLABS = bool(ELEVENLABS_API_KEY)

# --- Session ---
SESSION_EXPIRE_MINUTES = 30
