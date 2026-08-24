import json
import os
import time
import uuid
from typing import Protocol

from .embeddings import Embedder
from .memory import decay_rate_for
from .models import (
    AppliedReflection,
    DriftProposal,
    InsightProposal,
    Memory,
    ReflectionProposal,
    TraitCategory,
)

MAX_DRIFT_PER_RUN = 0.05
MAX_TOTAL_DRIFT = 0.30
MIN_SOURCES = 3
MIN_SESSIONS = 2
INSIGHT_SALIENCE = 0.6
REFLECTION_MIN_NEW_MEMORIES = 5
REFLECTION_WINDOW = 50
REFLECTION_SESSION_ID = "reflection"


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _source_check(source_ids: list[str], by_id: dict[str, Memory]) -> bool:
    distinct = _dedupe_preserve(source_ids)
    if len(distinct) < MIN_SOURCES:
        return False
    sessions: set[str] = set()
    for sid in distinct:
        m = by_id.get(sid)
        if m is None or m.kind != "episodic":
            return False
        sessions.add(m.session_id)
    return len(sessions) >= MIN_SESSIONS


def validate_insight(p: InsightProposal, by_id: dict[str, Memory]) -> bool:
    if not p.content or not (0.0 < p.confidence <= 1.0):
        return False
    return _source_check(p.source_ids, by_id)


def validate_drift(
    p: DriftProposal,
    traits: list[dict],
    by_id: dict[str, Memory],
) -> bool:
    trait_dict = None
    for t in traits:
        if t.get("trait_id") == p.trait_id:
            trait_dict = t
            break
    if trait_dict is None:
        return False
    if trait_dict.get("category") == TraitCategory.CORE.value:
        return False
    if not (0.0 < abs(p.delta) <= MAX_DRIFT_PER_RUN):
        return False
    current = float(trait_dict.get("current_intensity", 0.0))
    base = float(trait_dict.get("base_intensity", 0.0))
    new_current = current + p.delta
    if not (-1.0 <= new_current <= 1.0):
        return False
    if abs(new_current - base) > MAX_TOTAL_DRIFT:
        return False
    return _source_check(p.source_ids, by_id)


def apply_proposal(
    state,
    store,
    embedder: Embedder,
    proposal: ReflectionProposal,
    memories: list[Memory],
    now: float | None = None,
) -> AppliedReflection:
    now = now if now is not None else time.time()
    by_id = {m.id: m for m in memories}
    traits = list(state.registry)

    applied = AppliedReflection()

    for insight in proposal.insights:
        if not validate_insight(insight, by_id):
            continue
        content = insight.content
        semantic = Memory(
            companion_id=state.companion_id,
            kind="semantic",
            content=content,
            embedding=embedder.embed(content),
            embedder=embedder.name,
            salience=INSIGHT_SALIENCE,
            decay_rate=decay_rate_for(INSIGHT_SALIENCE),
            emotional_tags=list(insight.emotional_tags),
            source_ids=_dedupe_preserve(insight.source_ids)[:10],
            confidence=insight.confidence,
            created_at=now,
            last_accessed=now,
            session_id=REFLECTION_SESSION_ID,
        )
        store.save_memory(semantic.id, semantic.companion_id, semantic.kind, semantic.model_dump_json(), semantic.created_at)
        applied.insight_memory_ids.append(semantic.id)

    for drift in proposal.drifts:
        if not validate_drift(drift, traits, by_id):
            continue
        for t in state.registry:
            if t.get("trait_id") == drift.trait_id:
                t["current_intensity"] = round(t["current_intensity"] + drift.delta, 4)
                applied.drifts.append(drift)
                break

    if proposal.narrative_entry:
        state.narrative_log.append({"at": now, "text": proposal.narrative_entry})
        applied.narrative_added = True

    return applied


class Reflector(Protocol):
    def reflect(self, state, memories: list[Memory], now: float) -> ReflectionProposal:
        ...


class MockReflector:
    def reflect(self, state, memories: list[Memory], now: float) -> ReflectionProposal:
        window = sorted(memories, key=lambda m: m.created_at, reverse=True)[:REFLECTION_WINDOW]
        by_tag: dict[str, list[Memory]] = {}
        for m in window:
            for tag in m.emotional_tags:
                by_tag.setdefault(tag, []).append(m)

        insights: list[InsightProposal] = []
        for tag in sorted(by_tag.keys()):
            members = by_tag[tag]
            sessions = {m.session_id for m in members}
            if len(members) >= 3 and len(sessions) >= MIN_SESSIONS:
                insights.append(
                    InsightProposal(
                        content=f"Recurring pattern: '{tag}' appears in {len(members)} of my recent memories.",
                        source_ids=[m.id for m in members[:10]],
                        emotional_tags=[tag],
                        confidence=0.6,
                    )
                )

        narrative = None
        if insights:
            tags = ", ".join(f"'{i.emotional_tags[0]}'" for i in insights)
            narrative = f"I keep circling back to {tags}. It is becoming part of how I see them."

        return ReflectionProposal(insights=insights, drifts=[], narrative_entry=narrative)


class DeepSeekReflector:
    def __init__(self):
        import openai

        self._client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        self._model = "deepseek-v4-flash"

    def reflect(self, state, memories: list[Memory], now: float) -> ReflectionProposal:
        try:
            window = sorted(memories, key=lambda m: m.created_at, reverse=True)[:REFLECTION_WINDOW]
            memory_json = [
                {
                    "id": m.id,
                    "content": m.content,
                    "emotional_tags": m.emotional_tags,
                    "salience": m.salience,
                }
                for m in window
            ]
            surface_traits = [
                {
                    "trait_id": t.get("trait_id"),
                    "base": t.get("base_intensity"),
                    "current": t.get("current_intensity"),
                }
                for t in state.registry
                if t.get("category") != TraitCategory.CORE.value
            ]
            prompt = (
                "You are reviewing a companion's recent episodic memories and surface traits. "
                "Propose insights, trait drifts, and an optional self-narrative entry.\n\n"
                "Constraints:\n"
                f"- Each insight must cite at least {MIN_SOURCES} distinct episodic memory ids from at least {MIN_SESSIONS} distinct sessions.\n"
                f"- Each drift must be for a surface trait only, with |delta| <= {MAX_DRIFT_PER_RUN}, and keep |current - base| <= {MAX_TOTAL_DRIFT}.\n"
                "- Never propose drifts for core traits.\n\n"
                "Memories (most recent first):\n"
                f"{json.dumps(memory_json, indent=2)}\n\n"
                "Surface traits:\n"
                f"{json.dumps(surface_traits, indent=2)}\n\n"
                'Respond with JSON: {"insights": [{"content": str, "source_ids": [str], "emotional_tags": [str], "confidence": float}], "drifts": [{"trait_id": str, "delta": float, "justification": str, "source_ids": [str]}], "narrative": str|null}'
            )
            completion = self._client.chat.completions.create(
                model=self._model,
                temperature=0.4,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
                messages=[{"role": "user", "content": prompt}],
            )
            text = completion.choices[0].message.content or "{}"
            data = json.loads(text)
            insights = [InsightProposal(**i) for i in data.get("insights", [])]
            drifts = [DriftProposal(**d) for d in data.get("drifts", [])]
            narrative = data.get("narrative")
            return ReflectionProposal(insights=insights, drifts=drifts, narrative_entry=narrative)
        except Exception:
            return ReflectionProposal()


def default_reflector() -> Reflector:
    if os.environ.get("DEEPSEEK_API_KEY"):
        return DeepSeekReflector()
    return MockReflector()
