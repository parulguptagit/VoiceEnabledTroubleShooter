"""
Demo script simulating the full conversation from the spec.
Run from project root with API keys set: python tests/demo_session.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RAG_SCORE_THRESHOLD
from modules import context_manager, llm_agent, rag_engine, web_search


def run_turn(ctx: context_manager.ConversationContext, user_msg: str) -> str:
    """One conversation turn: RAG → web (if needed) → LLM; return ARIA response text."""
    rag_results = rag_engine.retrieve(user_msg, top_k=5)
    rag_results = rag_engine.rerank_results(rag_results, user_msg)
    rag_scores = [r["relevance_score"] for r in rag_results]
    web_results = []
    if web_search.is_web_search_needed(rag_scores, threshold=RAG_SCORE_THRESHOLD):
        web_results = web_search.search(user_msg, top_k=3)
    history = ctx.get_history()
    out = llm_agent.run(
        user_message=user_msg,
        conversation_history=history,
        rag_context=rag_results,
        web_context=web_results,
        image_description=None,
    )
    ctx.add_turn("user", user_msg)
    ctx.add_turn("assistant", out["text"])
    ctx.current_issue = ctx.detect_issue_category()
    ctx.step_counter += 1
    ctx.steps_attempted.append(out["text"][:80])
    return out["text"]


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run demo. Optional: OPENAI_API_KEY for RAG, TAVILY_API_KEY for web.")
        return 1
    ctx = context_manager.ConversationContext("demo-session")
    turns = [
        ("My iPhone battery is draining really fast",),
        ("It shows 87%",),
        ("Yeah, Instagram is using 34%",),
        ("I tried that, it still drains fast",),
    ]
    print("=== ARIA Demo Session ===\n")
    for i, (user,) in enumerate(turns, 1):
        print(f"User: {user}")
        aria = run_turn(ctx, user)
        print(f"ARIA: {aria[:400]}{'...' if len(aria) > 400 else ''}\n")
    print("Issue category:", ctx.detect_issue_category())
    print("Should escalate:", ctx.should_escalate())
    return 0


if __name__ == "__main__":
    sys.exit(main())
