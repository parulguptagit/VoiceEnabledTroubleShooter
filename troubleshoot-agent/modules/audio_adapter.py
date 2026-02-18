"""
Adapt UI/visual instructions for voice delivery: coordinates → directions,
button labels → spoken form, step numbers → spoken, URLs simplified, etc.
"""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def _coordinates_to_directions(text: str) -> str:
    """e.g. 'top-right' → 'upper right corner', 'bottom-left' → 'lower left corner'."""
    replacements = [
        (r"\btop-right\b", "upper right corner"),
        (r"\btop right\b", "upper right corner"),
        (r"\btop-left\b", "upper left corner"),
        (r"\bbottom-right\b", "lower right corner"),
        (r"\bbottom-left\b", "lower left corner"),
        (r"\bmid(?:dle)?-?right\b", "middle of the screen on the right"),
        (r"\bmid(?:dle)?-?left\b", "middle of the screen on the left"),
    ]
    for pat, repl in replacements:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text


def _button_labels_to_spoken(text: str) -> str:
    """'tap ⚙️' / 'tap Settings' → 'tap the Settings icon'."""
    # Common icon → spoken
    icon_map = [
        (r"tap\s*⚙️", "tap the Settings icon"),
        (r"tap\s*⚙", "tap the Settings icon"),
        (r"tap\s*Settings\s*(icon)?", "tap the Settings icon"),
        (r"tap\s*🔋", "tap the Battery icon"),
        (r"tap\s*📶", "tap the Wi-Fi icon"),
        (r"tap\s*🔵", "tap the blue button"),
        (r"\(⚙️\)", "(the Settings icon)"),
    ]
    for pat, repl in icon_map:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text


def _step_numbers_to_spoken(text: str) -> str:
    """'Step 3:' → 'For the third step,'."""
    def repl(m):
        num = m.group(1)
        ordinals = {"1": "first", "2": "second", "3": "third", "4": "fourth", "5": "fifth"}
        ord_str = ordinals.get(num, num)
        return f"For the {ord_str} step,"
    text = re.sub(r"Step\s+(\d+)\s*:", repl, text, flags=re.IGNORECASE)
    return text


def _urls_simplify(text: str) -> str:
    """'Visit support.apple.com/en-us/HT201487' → 'visit Apple Support online'."""
    text = re.sub(
        r"https?://[^\s\)]+",
        "visit Apple Support online",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?:visit|go to)\s+support\.apple\.com[^\s\)]*",
        "visit Apple Support online",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _paths_to_spoken(text: str) -> str:
    """'Settings > General > Transfer' → 'go to Settings, then General, then Transfer'."""
    text = re.sub(
        r"([A-Za-z][A-Za-z0-9\s]*)\s*>\s*([A-Za-z][A-Za-z0-9\s]*)\s*>\s*([A-Za-z][A-Za-z0-9\s]*)",
        r"go to \1, then \2, then \3",
        text,
    )
    text = re.sub(
        r"([A-Za-z][A-Za-z0-9\s]*)\s*>\s*([A-Za-z][A-Za-z0-9\s]*)",
        r"go to \1, then \2",
        text,
    )
    return text


def _lists_to_sequential(text: str) -> str:
    """Bullet points → 'First... Then... Finally...'."""
    lines = text.split("\n")
    out = []
    bullet = re.compile(r"^[\s]*[-*•]\s+", re.MULTILINE)
    ordinals = ["First", "Then", "Next", "After that", "Finally"]
    idx = 0
    for line in lines:
        if bullet.match(line):
            pref = ordinals[idx % len(ordinals)] if idx < len(ordinals) else "Then"
            idx += 1
            line = pref + ", " + bullet.sub("", line)
        out.append(line)
    return "\n".join(out)


def _ellipsis_and_special(text: str) -> str:
    """Ellipsis → comma for pause; remove or replace problematic chars."""
    text = text.replace("...", ", ")
    text = re.sub(r"\s+", " ", text)
    return text


def _emoji_to_words(text: str) -> str:
    """Remove or replace emoji with word equivalent."""
    emoji_map = [
        ("📸", ""),
        ("🔋", "Battery"),
        ("📶", "Wi-Fi"),
        ("⚙️", "Settings"),
        ("✅", "done"),
        ("❌", "not"),
        ("⚠️", "Note:"),
    ]
    for emoji, word in emoji_map:
        text = text.replace(emoji, word)
    # Strip any remaining common emoji (single codepoints)
    text = re.sub(r"[\U0001F300-\U0001F9FF]", "", text)
    return text


def adapt_for_voice(text: str) -> str:
    """
    Transform UI-reference text to be voice-friendly.
    Applies: coordinates → directions, button labels → spoken, step numbers,
    URLs simplified, paths → sequential speech, lists → First/Then/Finally,
    ellipsis/special chars, emoji removal.
    """
    if not text or not isinstance(text, str):
        return text
    text = _coordinates_to_directions(text)
    text = _button_labels_to_spoken(text)
    text = _step_numbers_to_spoken(text)
    text = _urls_simplify(text)
    text = _paths_to_spoken(text)
    text = _lists_to_sequential(text)
    text = _ellipsis_and_special(text)
    text = _emoji_to_words(text)
    return text.strip()


def inject_empathy(text: str, frustration_detected: bool) -> str:
    """Prepend an empathetic phrase if user is frustrated."""
    if not frustration_detected or not text:
        return text
    preambles = [
        "I understand that can be frustrating. ",
        "I hear you — let's try something else. ",
    ]
    import random
    prefix = random.choice(preambles)
    if not text.strip().lower().startswith(("i understand", "i hear")):
        return prefix + text
    return text


def add_step_preview(text: str, total_steps: int, current_step: int) -> str:
    """Add 'Step 2 of 5:' style prefix for multi-step flows."""
    if total_steps <= 1 or current_step < 1:
        return text
    prefix = f"Step {current_step} of {total_steps}. "
    if text.strip().lower().startswith("step "):
        return text
    return prefix + text


if __name__ == "__main__":
    sample = "Step 3: Tap ⚙️ at top-right. Go to Settings > General > Battery. Visit https://support.apple.com/HT201487"
    print(adapt_for_voice(sample))
    print(inject_empathy("Let's try resetting network settings.", True))
    print(add_step_preview("Open Settings and tap Battery.", 5, 2))
    print("Audio adapter __main__ OK")
