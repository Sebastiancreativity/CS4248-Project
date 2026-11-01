"""Shared cleaning for new and cached translation outputs."""
import re

PREFIX = re.compile(
    r"^(?:(?:sure|certainly|of course)[,!\.]\s*)?"
    r"(?:(?:here\s*(?:is|['’]s)\s*)?(?:the\s+|a\s+(?:possible\s+)?)?"
    r"(?:english\s+translation|translation(?:\s+(?:in|into|to)\s+english)?|english)"
    r"(?:\s+is)?|英文(?:翻译)?|译文)\s*[:：]\s*",
    re.I,
)


def clean(raw, src):
    # Drop recognized introduction-only lines before selecting the translation.
    # For example, Tower emits "Translation in English:\n<translation>".
    lines = [PREFIX.sub("", l.strip()) for l in raw.strip().split("\n")]
    lines = [l for l in lines if l]
    out = lines[0] if lines else ""
    if len(out) > 1 and out[0] == out[-1] and out[0] in "\"'“”" and src[:1] not in "\"“'‘「":
        out = out[1:-1].strip()
    return out
