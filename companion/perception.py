import re

from .items import item_stimuli
from .models import Perception, Stimulus

DEFAULT_LEXICON: dict[str, Stimulus] = {
    "chocolate cake": Stimulus(domain="taste", value="sweet"),
    "chocolate": Stimulus(domain="taste", value="sweet"),
    "candy": Stimulus(domain="taste", value="sweet"),
    "cake": Stimulus(domain="taste", value="sweet"),
    "honey": Stimulus(domain="taste", value="sweet"),
    "pretzel": Stimulus(domain="taste", value="salty"),
    "chips": Stimulus(domain="taste", value="salty"),
    "salty": Stimulus(domain="taste", value="salty"),
    "sweet": Stimulus(domain="taste", value="sweet"),
    "kitten": Stimulus(domain="entity", value="cat"),
    "cat": Stimulus(domain="entity", value="cat"),
    "dog": Stimulus(domain="entity", value="dog"),
    "spider": Stimulus(domain="entity", value="spider"),
    "climbing": Stimulus(domain="activity", value="climbing"),
    "mountain": Stimulus(domain="activity", value="climbing"),
    "rain": Stimulus(domain="topic", value="weather"),
}


def parse_method(raw: str) -> tuple[str | None, list[str]]:
    if not raw.startswith("/"):
        return (None, [])
    parts = raw[1:].split()
    if not parts:
        return (None, [])
    return (parts[0], parts[1:])


def extract_stimuli(text, lexicon=DEFAULT_LEXICON) -> list[Stimulus]:
    lower = text.lower()
    seen: set[str] = set()
    out: list[Stimulus] = []
    for kw in sorted(lexicon, key=len, reverse=True):
        if re.search(r"\b" + re.escape(kw) + r"\b", lower):
            s = lexicon[kw]
            if s.key() not in seen:
                seen.add(s.key())
                out.append(s)
    return out


def perceive(raw, time_gap_hours=0.0, lexicon=None, items=None) -> Perception:
    lex = lexicon if lexicon is not None else DEFAULT_LEXICON
    method, args = parse_method(raw)
    stimuli = extract_stimuli(raw, lex)
    # Items (M7): a mentioned item expands into item/tag/category stimuli.
    # Optional — without a registry this function is exactly the M1 spec.
    if items is not None:
        for item in items.match_text(raw):
            stimuli.extend(item_stimuli(item))
    if method:
        stimuli.append(Stimulus(domain="action", value=method))
        for a in args:
            stimuli.extend(extract_stimuli(a, lex))
    seen: set[str] = set()
    deduped: list[Stimulus] = []
    for s in stimuli:
        if s.key() not in seen:
            seen.add(s.key())
            deduped.append(s)
    return Perception(
        raw_input=raw,
        method=method,
        method_args=args,
        stimuli=deduped,
        time_gap_hours=time_gap_hours,
    )
