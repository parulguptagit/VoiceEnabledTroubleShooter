"""
Image analysis via Claude Vision: describe screenshot, infer issue, extract UI text.
"""
import base64
import logging
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ANTHROPIC_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)


def _encode_image(image_bytes: bytes) -> str:
    return base64.standard_b64encode(image_bytes).decode("ascii")


def analyze_image(image_bytes: bytes, user_context: str = "") -> dict[str, Any]:
    """
    Send image + context to Claude Vision. Returns:
    - description: what is visible
    - issue_detected: inferred iPhone issue (if any)
    - relevant_elements: UI elements, error messages, battery %, etc.
    - suggested_focus: what the troubleshooting agent should address
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set")
        return {
            "description": "",
            "issue_detected": "",
            "relevant_elements": [],
            "suggested_focus": "",
        }
    try:
        import anthropic
        client = anthropic.Anthropic()
        b64 = _encode_image(image_bytes)
        media_type = "image/png"
        if image_bytes[:3] == b"\xff\xd8\xff":
            media_type = "image/jpeg"
        prompt = f"""Analyze this iPhone screenshot or photo for troubleshooting.

User context: {user_context or "No additional context."}

Provide a JSON object with these exact keys (no markdown, no code fence):
- "description": 1-2 sentences of what is visible (screen, app, settings, error message, etc.)
- "issue_detected": inferred iPhone issue category if any (e.g. battery, WiFi, storage, crash, overheating, Bluetooth, screen) or "none"
- "relevant_elements": list of strings (UI elements, labels, error text, battery %, toggles)
- "suggested_focus": one sentence on what the support agent should address first"""
        msg = client.messages.create(
            model=LLM_MODEL,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": b64},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        text = (msg.content[0].text if msg.content else "").strip()
        # Parse JSON from response (handle optional markdown wrapper)
        import json
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"description": text[:200], "issue_detected": "", "relevant_elements": [], "suggested_focus": ""}
        return {
            "description": data.get("description", ""),
            "issue_detected": data.get("issue_detected", ""),
            "relevant_elements": data.get("relevant_elements") or [],
            "suggested_focus": data.get("suggested_focus", ""),
        }
    except Exception as e:
        logger.exception("Image analysis failed: %s", e)
        return {
            "description": "",
            "issue_detected": "",
            "relevant_elements": [],
            "suggested_focus": str(e),
        }


def extract_text_from_screenshot(image_bytes: bytes) -> str:
    """OCR-style extraction of error messages, settings labels, etc. via Claude Vision."""
    result = analyze_image(image_bytes, user_context="Extract all visible text: labels, error messages, buttons, and settings values.")
    parts = [result.get("description", ""), result.get("suggested_focus", "")]
    elements = result.get("relevant_elements") or []
    return " ".join(parts + elements).strip()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # No image bytes = minimal response
    out = analyze_image(b"", "")
    print(out)
    print("Image handler __main__ OK (use real image + ANTHROPIC_API_KEY for full test)")
