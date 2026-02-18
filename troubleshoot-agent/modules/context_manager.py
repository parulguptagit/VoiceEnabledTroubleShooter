"""
Multi-turn conversation memory: history, issue category, steps attempted, escalation.
"""
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Approximate chars per token for truncation
CHARS_PER_TOKEN = 4


class ConversationContext:
    """
    Per-session state for the troubleshooting agent.
    Tracks history, current issue, steps attempted, and frustration signals.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: list[dict] = []  # OpenAI/Anthropic message format: { "role", "content" }
        self.current_issue: str = ""
        self.steps_attempted: list[str] = []
        self.step_counter: int = 0
        self.frustration_signals: int = 0
        self.uploaded_images: list[str] = []

    def add_turn(self, role: str, content: str) -> None:
        """Append a message to history."""
        if isinstance(content, str) and content.strip():
            self.history.append({"role": role, "content": content.strip()})

    def get_history(self, max_tokens: int = 4000) -> list[dict]:
        """Return recent history, truncating oldest messages to stay under max_tokens."""
        max_chars = max_tokens * CHARS_PER_TOKEN
        total = 0
        out = []
        for msg in reversed(self.history):
            c = (msg.get("content") or "")
            total += len(c) + 50  # overhead per message
            if total > max_chars and out:
                break
            out.append(msg)
        out.reverse()
        return out

    def detect_issue_category(self) -> str:
        """
        Infer category from conversation: battery, wifi, storage, crash, overheat, bluetooth, screen.
        """
        text = " ".join(
            (m.get("content") or "") for m in self.history
        ).lower()
        if re.search(r"battery|drain|charging|low power|percent", text):
            return "battery"
        if re.search(r"wifi|wi-fi|network|internet|connection|router", text):
            return "wifi"
        if re.search(r"storage|full|space|icloud|offload", text):
            return "storage"
        if re.search(r"crash|crashes|freeze|force quit|reinstall", text):
            return "crashes"
        if re.search(r"overheat|hot|warming|temperature", text):
            return "overheating"
        if re.search(r"bluetooth|bluetooth|airpods|pairing", text):
            return "bluetooth"
        if re.search(r"screen|display|touch|calibrat|true tone", text):
            return "screen"
        return "general"

    def should_escalate(self) -> bool:
        """True if 3+ steps failed or high frustration."""
        failed = sum(1 for s in self.steps_attempted if "not resolved" in s.lower() or "didn't work" in s.lower())
        return failed >= 3 or self.frustration_signals >= 2

    def export_session(self) -> dict[str, Any]:
        """Export for debugging/logging."""
        return {
            "session_id": self.session_id,
            "current_issue": self.current_issue,
            "step_counter": self.step_counter,
            "steps_attempted": self.steps_attempted.copy(),
            "frustration_signals": self.frustration_signals,
            "uploaded_images_count": len(self.uploaded_images),
            "history_length": len(self.history),
        }


if __name__ == "__main__":
    ctx = ConversationContext("test-session")
    ctx.add_turn("user", "My iPhone battery is draining really fast")
    ctx.add_turn("assistant", "Let's check Battery Health. Go to Settings > Battery.")
    ctx.add_turn("user", "That didn't work")
    ctx.frustration_signals = 1
    ctx.steps_attempted = ["Check Battery Health", "Limit background app", "not resolved"]
    print("Issue category:", ctx.detect_issue_category())
    print("Should escalate:", ctx.should_escalate())
    print("Export:", ctx.export_session())
    print("Context manager __main__ OK")
