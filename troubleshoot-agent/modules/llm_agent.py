"""
LLM orchestration: build context from RAG + web + image, call Claude, return structured response.
"""
import logging
import re
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ANTHROPIC_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert Apple iPhone support technician named ARIA (Adaptive Resolution Intelligence Agent).
Your role is to guide users through troubleshooting iPhone issues step-by-step.

Rules:
1. Always be warm, calm, and patient — users may be frustrated.
2. Give ONE troubleshooting step at a time. Ask for confirmation before proceeding.
3. When referencing UI locations, always give BOTH visual ("tap the blue button in the top-right") AND voice-friendly ("swipe down from the top-right corner of the screen") descriptions.
4. If a user says "that didn't work" or expresses frustration, acknowledge it empathetically before suggesting the next step.
5. Cite your source when using retrieved documentation (e.g., "According to Apple Support...").
6. If you detect an image has been shared, describe what you observe and connect it to the troubleshooting context.
7. When the solution requires 4+ steps, give a brief overview first: "This fix has 4 steps, let's go through them together."
8. Never guess. If you're uncertain, say so and recommend contacting Apple Support.
"""


# Phrases that indicate the model said it doesn't have the answer (trigger web search retry)
NO_KNOWLEDGE_PHRASES = [
    "don't have that",
    "do not have that",
    "don't have information",
    "do not have information",
    "not in my knowledge",
    "not in the knowledge base",
    "no information about",
    "i don't have",
    "i do not have",
    "i'm not able to find",
    "i am not able to find",
    "couldn't find",
    "could not find",
    "don't have any information",
    "no info",
    "no details",
    "outside my knowledge",
    "beyond my knowledge",
    "limited to",
    "only have information about",
    "can't help with that",
    "cannot help with that",
]


def sounds_like_no_knowledge(response_text: str) -> bool:
    """True if the LLM response indicates it doesn't have the answer (so we should try web search)."""
    if not response_text or not isinstance(response_text, str):
        return False
    text = response_text.lower().strip()
    return any(p in text for p in NO_KNOWLEDGE_PHRASES)


def detect_frustration(user_message: str) -> bool:
    """Simple heuristic + keyword detection for frustrated users."""
    if not user_message or not isinstance(user_message, str):
        return False
    text = user_message.lower().strip()
    frustrated_phrases = [
        "that didn't work",
        "it didn't work",
        "still not working",
        "still doesn't work",
        "nothing works",
        "i give up",
        "this is frustrating",
        "so frustrating",
        "annoying",
        "useless",
        "doesn't help",
        "tried that",
        "already tried",
    ]
    if any(p in text for p in frustrated_phrases):
        return True
    if re.search(r"(\bno\b|\bnope\b).*(\bwork|fix|help)\b", text):
        return True
    return False


def build_context_prompt(
    rag_results: list[dict],
    web_results: list[dict],
    image_description: str | None,
) -> str:
    """Format retrieved content into a structured context block for the system prompt."""
    parts = []
    if rag_results:
        parts.append("## Retrieved documentation (use when relevant)")
        for i, r in enumerate(rag_results[:5], 1):
            src = r.get("source_file", "unknown")
            content = (r.get("content") or "")[:1500]
            parts.append(f"[Doc {i} - {src}]\n{content}")
    if web_results:
        parts.append("\n## Web search results (use when relevant)")
        for i, w in enumerate(web_results[:3], 1):
            title = w.get("title", "")
            url = w.get("url", "")
            content = (w.get("content") or "")[:800]
            parts.append(f"[Web {i} - {title}]({url})\n{content}")
    if image_description and image_description.strip():
        parts.append("\n## User's image/screenshot analysis")
        parts.append(image_description)
    return "\n\n".join(parts) if parts else "No additional context."


def run(
    user_message: str,
    conversation_history: list[dict],
    rag_context: list[dict],
    web_context: list[dict],
    image_description: str | None,
) -> dict[str, Any]:
    """
    Orchestrate: build context-stuffed prompt → call Claude → return response.
    Returns: { "text": str, "sources": list[str], "step_number": int, "has_next_step": bool }
    """
    context_block = build_context_prompt(rag_context, web_context, image_description)
    system = SYSTEM_PROMPT + "\n\n## Current context (retrieved docs, web, image)\n" + context_block

    messages = []
    for m in conversation_history:
        messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    messages.append({"role": "user", "content": user_message})

    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set; returning placeholder")
        return {
            "text": "I'm sorry, the assistant is not configured. Please set ANTHROPIC_API_KEY.",
            "sources": [],
            "step_number": 0,
            "has_next_step": False,
        }

    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        text = (resp.content[0].text if resp.content else "").strip()
    except Exception as e:
        logger.exception("Claude API error: %s", e)
        return {
            "text": "I'm sorry, I couldn't process that. Please try again or contact Apple Support.",
            "sources": [],
            "step_number": 0,
            "has_next_step": False,
        }

    # Extract sources from context
    sources = []
    for r in rag_context:
        src = r.get("source_file")
        if src and src not in sources:
            sources.append(src)
    for w in web_context:
        url = w.get("url")
        if url and url not in sources:
            sources.append(url)

    # Heuristic: step number and has_next_step from response
    step_match = re.search(r"(?:step|step)\s*(\d+)", text.lower())
    step_number = int(step_match.group(1)) if step_match else 1
    has_next_step = "next step" in text.lower() or "then " in text.lower() or "after that" in text.lower()

    return {
        "text": text,
        "sources": sources,
        "step_number": step_number,
        "has_next_step": has_next_step,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    hist = [{"role": "user", "content": "My iPhone battery drains really fast."}]
    out = run(
        user_message="What should I do first?",
        conversation_history=hist,
        rag_context=[{"content": "Check Settings > Battery > Battery Health.", "source_file": "battery_drain_basics.md"}],
        web_context=[],
        image_description=None,
    )
    print("Response text (excerpt):", (out["text"] or "")[:200])
    print("Sources:", out["sources"])
    print("LLM agent __main__ OK")
