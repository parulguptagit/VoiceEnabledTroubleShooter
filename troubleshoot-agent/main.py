"""
FastAPI entry point for the iPhone Troubleshooting Agent.
CORS, request logging, global exception handler, and all API routes.
"""
import base64
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root is on path when running as uvicorn main:app
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RAG_SCORE_THRESHOLD, TOP_K_RAG, TOP_K_WEB
from modules import context_manager, image_handler, llm_agent, rag_engine, voice_stt, voice_tts, web_search

# Guardrail: only allow web search for iPhone/Apple device troubleshooting
IPHONE_RELATED_KEYWORDS = (
    "iphone", "apple", "ios", "ipad", "watch", "airpods",
    "battery", "storage", "wifi", "bluetooth", "screen", "crash", "overheat",
    "settings", "update", "restore", "backup", "face id", "touch id",
    "not working", "issue", "problem", "fix", "troubleshoot", "help",
)


def is_iphone_related_query(message: str) -> bool:
    """True if the query is about iPhone/Apple device troubleshooting (guardrail for web search)."""
    if not message or not message.strip():
        return False
    text = message.lower().strip()
    return any(kw in text for kw in IPHONE_RELATED_KEYWORDS)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory session store (session_id -> ConversationContext)
sessions: dict[str, context_manager.ConversationContext] = {}
# In-memory document list for GET /api/documents (optional: could query Chroma)
ingested_docs: list[dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    sessions.clear()


app = FastAPI(title="ARIA iPhone Troubleshooting Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    logger.info("%s %s %.3fs", request.method, request.url.path, duration)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )


# --- Request/response models ---
class ChatBody(BaseModel):
    session_id: str
    message: str
    image_base64: str | None = None


# --- Endpoints ---

@app.post("/api/upload-documents")
async def upload_documents(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accept multipart file (PDF, TXT, MD), call rag_engine.ingest_document."""
    if not file.filename:
        raise HTTPException(400, "Missing filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, "Only PDF, TXT, MD allowed")
    contents = await file.read()
    # Write to temp file for ingest_document (expects filepath)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(contents)
        path = tmp.name
    try:
        n = rag_engine.ingest_document(path, metadata={"source_file": file.filename})
        ingested_docs.append({"filename": file.filename, "chunks_created": n})
        return {"success": True, "chunks_created": n, "filename": file.filename}
    finally:
        Path(path).unlink(missing_ok=True)


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> dict[str, Any]:
    """Accept audio blob (webm/wav), return transcription."""
    fmt = "webm"
    if audio.filename and ".wav" in audio.filename.lower():
        fmt = "wav"
    data = await audio.read()
    result = voice_stt.transcribe_audio(data, format=fmt)
    return {"text": result.get("text", ""), "confidence": result.get("confidence", 0)}


@app.post("/api/chat")
async def chat(body: ChatBody) -> dict[str, Any]:
    """Orchestrate: context → RAG → web (if needed) → LLM → TTS. Return text + audio_base64 + sources + steps (Think/Act/Observe)."""
    session_id = body.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = context_manager.ConversationContext(session_id)
    ctx = sessions[session_id]
    message = (body.message or "").strip()
    image_description: str | None = None
    steps: list[dict[str, str]] = []  # [{ "phase": "think"|"act"|"observe", "text": "..." }]

    if body.image_base64:
        try:
            raw = base64.b64decode(body.image_base64)
            img_result = image_handler.analyze_image(raw, message)
            image_description = f"{img_result.get('description', '')} Issue: {img_result.get('issue_detected', '')}. Focus: {img_result.get('suggested_focus', '')}"
            ctx.uploaded_images.append(body.image_base64[:50])
        except Exception as e:
            logger.warning("Image analysis skipped: %s", e)

    ctx.add_turn("user", message)
    if llm_agent.detect_frustration(message):
        ctx.frustration_signals += 1

    # ——— Think: Checking RAG ———
    steps.append({"phase": "think", "text": "Checking knowledge base (RAG) for relevant docs…"})
    rag_results = rag_engine.retrieve(message, top_k=TOP_K_RAG)
    rag_results = rag_engine.rerank_results(rag_results, message)
    rag_scores = [r["relevance_score"] for r in rag_results]
    top_score = float(rag_scores[0]) if rag_scores else 0.0
    if not rag_results:
        steps.append({"phase": "observe", "text": "No matching documents in knowledge base."})
    elif top_score < RAG_SCORE_THRESHOLD:
        steps.append({"phase": "observe", "text": f"No strong match (best score {top_score:.2f}). Will try web search if query is iPhone-related."})
    else:
        steps.append({"phase": "observe", "text": f"Found {len(rag_results)} relevant chunk(s) (best score {top_score:.2f})."})

    # ——— Act/Observe: Web search (guardrail: only for iPhone-related queries) ———
    web_results: list[dict] = []
    if web_search.is_web_search_needed(rag_scores, threshold=RAG_SCORE_THRESHOLD):
        if is_iphone_related_query(message):
            steps.append({"phase": "act", "text": "Searching web (support.apple.com, apple.com, discussions.apple.com)…"})
            web_results = web_search.search(message, top_k=TOP_K_WEB)
            steps.append({"phase": "observe", "text": f"Found {len(web_results)} web result(s)."})
        else:
            steps.append({"phase": "observe", "text": "Web search skipped (only allowed for iPhone/Apple device troubleshooting)."})

    steps.append({"phase": "act", "text": "Generating response…"})
    history = ctx.get_history()
    llm_out = llm_agent.run(
        user_message=message,
        conversation_history=history,
        rag_context=rag_results,
        web_context=web_results,
        image_description=image_description,
    )
    final_text = llm_out["text"]
    final_sources = llm_out.get("sources", [])

    # ——— If response sounds like "I don't have that" and we didn't use web yet, try web search and retry ———
    if llm_agent.sounds_like_no_knowledge(final_text) and not web_results and is_iphone_related_query(message):
        steps.append({"phase": "observe", "text": "Answer not in knowledge base. Trying web search…"})
        steps.append({"phase": "act", "text": "Searching support.apple.com, apple.com for more info…"})
        web_results = web_search.search(message, top_k=TOP_K_WEB)
        steps.append({"phase": "observe", "text": f"Found {len(web_results)} web result(s)."})
        steps.append({"phase": "act", "text": "Generating response using web results…"})
        llm_out_2 = llm_agent.run(
            user_message=message,
            conversation_history=history,
            rag_context=rag_results,
            web_context=web_results,
            image_description=image_description,
        )
        final_text = llm_out_2["text"]
        final_sources = llm_out_2.get("sources", [])

    ctx.add_turn("assistant", final_text)
    ctx.current_issue = ctx.detect_issue_category()
    ctx.step_counter += 1
    ctx.steps_attempted.append(final_text[:80])

    # TTS
    audio_base64 = ""
    try:
        audio_bytes = voice_tts.synthesize(final_text)
        if audio_bytes:
            audio_base64 = base64.standard_b64encode(audio_bytes).decode("ascii")
    except Exception as e:
        logger.warning("TTS failed: %s", e)

    return {
        "text": final_text,
        "audio_base64": audio_base64,
        "sources": final_sources,
        "session_id": session_id,
        "steps": steps,
    }


@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accept image file, return analyze_image result."""
    data = await file.read()
    result = image_handler.analyze_image(data, "")
    return {"description": result.get("description", ""), "issue_detected": result.get("issue_detected", "")}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    """Return full conversation history for the session."""
    if session_id not in sessions:
        return {"session_id": session_id, "history": []}
    ctx = sessions[session_id]
    return {"session_id": session_id, "history": ctx.get_history(max_tokens=8000)}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    """Clear session from memory."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "deleted", "session_id": session_id}


@app.get("/api/documents")
async def list_documents() -> dict[str, Any]:
    """List ingested documents (from upload history)."""
    return {"documents": ingested_docs}


# Serve static frontend
static_dir = Path(__file__).parent / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")
    @app.get("/")
    async def root():
        from fastapi.responses import FileResponse
        return FileResponse(static_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)
