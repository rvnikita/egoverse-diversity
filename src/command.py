"""Utterance -> Command. Rule-based on purpose: zero deps, zero latency, zero API keys.

The only hard part of parsing "pick up the small red block on the left" is pulling out
the noun phrase, because that phrase is what gets handed to an open-vocabulary detector
verbatim. OWLv2 handles "small red block" fine, so we do not need to understand it —
we just need to stop chopping it up.

Swap in an LLM later via `parse(text, llm=...)` if the demo needs compound commands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Ordered longest-first so "pick up" wins over "pick".
VERBS: dict[str, tuple[str, ...]] = {
    "pick": ("pick up", "pick", "grab", "grasp", "get", "take", "lift", "fetch"),
    "place": ("put down", "place", "drop", "release", "let go", "open the gripper"),
    "point": ("point at", "point to", "find", "locate", "look at", "where is", "show me"),
    "home": ("go home", "home", "reset", "stand by"),
    "stop": ("stop", "halt", "freeze", "abort", "cancel"),
}

_FILLER = re.compile(
    r"^\s*(?:hey\s+\w+\s*,?\s*|ok(?:ay)?\s+\w+\s*,?\s*|please\s+|can you\s+|could you\s+|"
    r"would you\s+|i want you to\s+|i'?d like you to\s+)+",
    re.I,
)
_ARTICLES = re.compile(r"^(?:the|a|an|that|this|my)\s+", re.I)
# Repeat the group: "...for me please" needs both stripped, not just the last one.
_TRAILING = re.compile(
    r"(?:\s*(?:please|now|for me|for us|thanks|thank you|ok(?:ay)?))+[.!?]*\s*$", re.I
)
_PRONOUNS = {"it", "that", "this", "them", "those", "these", ""}

# Spatial qualifiers we strip from the label but keep as a modifier, because a detector
# cannot use them but a disambiguation step can.
_SPATIAL = {
    "left": ("on the left", "to the left", "leftmost", "left one", "left"),
    "right": ("on the right", "to the right", "rightmost", "right one", "right"),
    "near": ("closest", "nearest", "in front", "closer"),
    "far": ("farthest", "furthest", "in the back", "further"),
}


@dataclass
class Command:
    verb: str  # one of VERBS keys, or "unknown"
    target: str = ""  # noun phrase to feed the detector, e.g. "red block"
    modifiers: list[str] = field(default_factory=list)  # e.g. ["left"]
    raw: str = ""

    def __bool__(self) -> bool:
        return self.verb != "unknown"

    def describe(self) -> str:
        """Short spoken confirmation — what TTS says back before moving."""
        if self.verb == "home":
            return "Going to the home position."
        if self.verb == "stop":
            return "Stopping."
        if self.verb == "place":
            return "Releasing."
        where = {
            "left": " on the left",
            "right": " on the right",
            "near": " closest to you",
            "far": " furthest away",
        }.get(self.modifiers[0] if self.modifiers else "", "")
        thing = f"the {self.target}" if self.target else "it"
        if self.verb == "point":
            return f"Looking for {thing}{where}."
        return f"Picking up {thing}{where}."


def parse(text: str) -> Command:
    raw = text or ""
    s = _TRAILING.sub("", _FILLER.sub("", raw.strip())).strip().rstrip(".!?,")
    if not s:
        return Command("unknown", raw=raw)

    low = s.lower()

    # "put it down" / "set that down" — the verb is split around a pronoun, so the
    # plain phrase table misses it.
    if re.search(r"\b(?:put|set|lay)\s+(?:it|that|this|them)?\s*down\b", low):
        return Command("place", raw=raw)

    verb = "unknown"
    rest = s
    for canonical, phrases in VERBS.items():
        for phrase in sorted(phrases, key=len, reverse=True):
            if low.startswith(phrase):
                verb = canonical
                rest = s[len(phrase) :]
                break
            # also allow the verb mid-sentence: "now grab the cup"
            m = re.search(rf"\b{re.escape(phrase)}\b", low)
            if m:
                verb = canonical
                rest = s[m.end() :]
                break
        if verb != "unknown":
            break

    if verb in ("home", "stop"):
        return Command(verb, raw=raw)

    modifiers: list[str] = []
    rest_low = rest.lower()
    for canonical, phrases in _SPATIAL.items():
        for phrase in sorted(phrases, key=len, reverse=True):
            if re.search(rf"\b{re.escape(phrase)}\b", rest_low):
                modifiers.append(canonical)
                rest = re.sub(rf"\b{re.escape(phrase)}\b", " ", rest, flags=re.I)
                rest_low = rest.lower()
                break

    target = re.sub(r"\s+", " ", rest).strip().strip(",.")
    target = _ARTICLES.sub("", target).strip()
    # drop a dangling preposition left behind by stripping a spatial phrase
    target = re.sub(r"\s+(?:on|to|in|at|of|from)$", "", target, flags=re.I).strip()

    # "grab it" carries no label; the caller reuses the previous target.
    if target.lower() in _PRONOUNS:
        target = ""

    if verb == "unknown" and target:
        # A bare noun phrase ("the red block") is a reasonable "point at it".
        verb = "point"

    return Command(verb, target, modifiers, raw=raw)


if __name__ == "__main__":
    samples = [
        "Hey robot, please pick up the red block",
        "grab the small blue cube on the left",
        "point at the coffee cup",
        "where is the screwdriver?",
        "put it down",
        "go home",
        "STOP",
        "could you get that yellow lego brick for me please",
        "the red block",
        "",
    ]
    for s in samples:
        c = parse(s)
        print(f"{s!r:55} -> verb={c.verb:8} target={c.target!r:22} mods={c.modifiers}")
