"""Relationship dynamics (Milestone 5): the emergent layer."""
from __future__ import annotations

import time

from .models import Memory, RelationshipVector, VersionedModel, VoiceProfile, voice_delta


class PhaseSpec(VersionedModel):
    name: str
    dimension: str
    enter: float            # threshold to ENTER the phase
    exit: float             # threshold to LEAVE it (hysteresis gap)
    enter_above: bool       # True: enter at >= enter, exit at < exit
                            # False: enter at <= enter, exit at > exit


PHASES: list[PhaseSpec] = [
    PhaseSpec(name="high_trust", dimension="trust",
              enter=0.70, exit=0.60, enter_above=True),
    PhaseSpec(name="breached_trust", dimension="trust",
              enter=0.30, exit=0.40, enter_above=False),
    PhaseSpec(name="high_intimacy", dimension="intimacy",
              enter=0.60, exit=0.50, enter_above=True),
    PhaseSpec(name="resentment", dimension="resentment",
              enter=0.40, exit=0.30, enter_above=True),
]

PHASE_VOICE: dict[str, VoiceProfile] = {
    "high_trust": voice_delta(temperature=0.15, humor=0.2),
    "breached_trust": voice_delta(temperature=-0.3, formality=0.3, humor=-0.3),
    "high_intimacy": voice_delta(temperature=0.1, humor=0.2, verbosity=0.1),
    "resentment": voice_delta(temperature=-0.2, verbosity=-0.3, humor=-0.3),
}

PHASE_NOTES: dict[str, str] = {
    "high_trust": "You feel safe with them; you may be more open than usual.",
    "breached_trust": "You do not trust them right now. Keep your distance; do not explain why.",
    "high_intimacy": "You feel close to them; shared references and warmth come easily.",
    "resentment": "You are nursing resentment. Let it show as brevity, not accusations.",
}

WOUND_TAGS = {"pain", "fear"}
WOUND_SALIENCE = 0.7
WOUND_WINDOW_DAYS = 3.0
WOUND_AMPLIFIER = 1.5

RESENTMENT_AMPLIFIER = 0.3


def update_phases(active: list[str], rel: RelationshipVector) -> list[str]:
    """Hysteretic phase transitions. Pure; idempotent; returns sorted names."""
    current = set(active)
    for p in PHASES:
        value = getattr(rel, p.dimension)
        if p.name in current:
            still = value >= p.exit if p.enter_above else value <= p.exit
            if not still:
                current.discard(p.name)
        else:
            enters = value >= p.enter if p.enter_above else value <= p.enter
            if enters:
                current.add(p.name)
    return sorted(current)


def phase_voice_delta(phases: list[str]) -> VoiceProfile:
    delta = voice_delta()
    for name in phases:
        delta = delta + PHASE_VOICE[name]
    return delta


def phase_notes(phases: list[str]) -> list[str]:
    return [PHASE_NOTES[p] for p in phases]


def wound_amplifier(memories: list[Memory], now: float | None = None) -> float:
    """WOUND_AMPLIFIER if any fresh, high-salience, painful memory exists."""
    now = now or time.time()
    for m in memories:
        age_days = (now - m.created_at) / 86400.0
        if (age_days <= WOUND_WINDOW_DAYS and m.salience >= WOUND_SALIENCE
                and WOUND_TAGS & set(m.emotional_tags)):
            return WOUND_AMPLIFIER
    return 1.0
