"""
Web search via Tavily API (RAG/agent-oriented, clean parsed results).
Ranking by domain priority, recency, and content length.
"""
import logging
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlparse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TAVILY_API_KEY, TOP_K_WEB, WEB_SEARCH_PRIORITY_DOMAINS

logger = logging.getLogger(__name__)

# Domain priority: apple.com = 1.0, discussions.apple.com = 0.85, other = 0.5
DOMAIN_SCORES = {
    "apple.com": 1.0,
    "discussions.apple.com": 0.85,
    "support.apple.com": 1.0,
}
DEFAULT_DOMAIN_SCORE = 0.5


def _get_domain(url: str) -> str:
    try:
        parsed = urlparse(url or "")
        netloc = (parsed.netloc or "").lower()
        for d in WEB_SEARCH_PRIORITY_DOMAINS:
            if d in netloc or netloc.endswith("." + d):
                return d
        return netloc or "other"
    except Exception:
        return "other"


def _domain_priority_score(domain: str) -> float:
    return DOMAIN_SCORES.get(domain, DEFAULT_DOMAIN_SCORE)


def _parse_date(published: str | None) -> datetime | None:
    if not published:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(published[:19].replace("Z", ""), fmt)
        except ValueError:
            continue
    return None


def search(query: str, top_k: int = TOP_K_WEB) -> list[dict]:
    """
    Search the web via Tavily, restricted to Apple domains (support.apple.com, apple.com,
    discussions.apple.com) so we get authoritative answers when RAG has no good match.
    Each result contains: title, url, content, domain, priority_score, published_date.
    """
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set; skipping web search")
        return []
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            max_results=max(top_k, 5),
            include_domains=WEB_SEARCH_PRIORITY_DOMAINS or None,
        )
        results = getattr(response, "results", None) or []
        out = []
        for r in results:
            url = r.get("url", "")
            content = r.get("content", "") or r.get("snippet", "") or ""
            domain = _get_domain(url)
            out.append({
                "title": r.get("title", ""),
                "url": url,
                "content": content,
                "domain": domain,
                "priority_score": _domain_priority_score(domain),
                "published_date": r.get("published_date") or "",
            })
        return rank_results(out)[:top_k]
    except Exception as e:
        logger.exception("Tavily search failed: %s", e)
        return []


def rank_results(results: list[dict]) -> list[dict]:
    """
    Rank by composite score:
    - Domain priority: apple.com = 1.0, discussions.apple.com = 0.85, other = 0.5
    - Recency: last 30 days = +0.2, last year = +0.1
    - Content length: >200 words = +0.1
    """
    now = datetime.utcnow()
    for r in results:
        base = r.get("priority_score", DEFAULT_DOMAIN_SCORE)
        pub = _parse_date(r.get("published_date"))
        if pub:
            delta = now - pub
            if delta < timedelta(days=30):
                base += 0.2
            elif delta < timedelta(days=365):
                base += 0.1
        words = len((r.get("content") or "").split())
        if words > 200:
            base += 0.1
        r["composite_score"] = round(base, 4)
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results


def is_web_search_needed(rag_scores: list[float], threshold: float = 0.85) -> bool:
    """Return True if no RAG results or top RAG score below threshold — then we search web (e.g. support.apple.com, apple.com)."""
    if not rag_scores:
        return True
    return float(rag_scores[0]) < threshold


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if TAVILY_API_KEY:
        res = search("iPhone battery drain fix", top_k=3)
        for r in res:
            print(r.get("title"), r.get("composite_score"), r.get("url"))
    else:
        print("Set TAVILY_API_KEY to run web search test")
    print("is_web_search_needed([0.8]):", is_web_search_needed([0.8]))
    print("is_web_search_needed([0.6]):", is_web_search_needed([0.6]))
