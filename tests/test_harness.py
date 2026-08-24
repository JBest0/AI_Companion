import json

import pytest

from chat import format_trace, load_session
from companion import (
    CompanionSession,
    CompanionState,
    HashEmbedder,
    MockLLM,
    Store,
    Trait,
    TraitCategory,
    TraitRegistry,
    Trigger,
    VoiceProfile,
    dislikes,
    likes,
    load_character,
)
from companion.loop import FALLBACK_LINES
from companion.models import VoiceProfile as _VoiceProfile


def felinophobia() -> Trait:
    return Trait(
        trait_id="felinophobia",
        category=TraitCategory.CORE,
        description="Childhood trauma involving a feral cat.",
        triggers=[Trigger(domain="entity", values=["cat"])],
        base_intensity=-0.9,
        current_intensity=-0.9,
        curve="steep",
        archetypes_negative=["disgusted_rejection"],
    )


@pytest.fixture
def registry() -> TraitRegistry:
    return TraitRegistry(
        [
            likes("taste", "sweet", intensity=0.7),
            dislikes("taste", "salty", intensity=-0.7),
            felinophobia(),
        ]
    )


class CapturingLLM:
    def __init__(self):
        self.system = ""
        self.user = ""

    def generate(self, *, system: str, user: str, voice: _VoiceProfile) -> str:
        self.system = system
        self.user = user
        return "[captured]"


class FailingLLM:
    def generate(self, *, system: str, user: str, voice: _VoiceProfile) -> str:
        raise ConnectionError("model unavailable")


def test_persona_loads():
    char = load_character("characters/kira.yaml")
    state = CompanionState.create(
        companion_id="kira",
        name=char["name"],
        registry=char["registry"],
        voice_baseline=char["voice_baseline"],
        affect_baseline=char["mood_baseline"],
        backstory=char.get("persona", {}).get("backstory", ""),
        speaking_style=char.get("persona", {}).get("speaking_style", ""),
    )
    assert "border town" in state.backstory
    assert "Economical" in state.speaking_style


def test_prompt_order(registry):
    store = Store(":memory:")
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
        backstory="Test backstory.",
        speaking_style="Test style.",
    )
    llm = CapturingLLM()
    session = CompanionSession(state, store, llm, HashEmbedder())
    session.open()
    session.turn("/hug")
    system = llm.system
    assert "BACKSTORY: Test backstory." in system
    assert "SPEAKING STYLE: Test style." in system
    backstory_pos = system.find("BACKSTORY:")
    archetype_pos = system.find("ARCHETYPE:")
    assert 0 < backstory_pos < archetype_pos


def test_persona_optional(registry):
    store = Store(":memory:")
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )
    llm = CapturingLLM()
    session = CompanionSession(state, store, llm, HashEmbedder())
    session.open()
    session.turn("/hug")
    system = llm.system
    assert "BACKSTORY:" not in system
    assert "SPEAKING STYLE:" not in system


def test_old_states_hydrate(registry):
    old = {
        "schema_version": 1,
        "companion_id": "kira",
        "name": "Kira",
        "registry": [t.model_dump(mode="json") for t in registry.all()],
        "voice_baseline": {},
        "affect": {},
        "affect_baseline": {},
        "relationship": {},
        "session_log": [],
        "last_session_end": None,
    }
    store = Store(":memory:")
    store.save_state("kira", json.dumps(old))
    hydrated = CompanionState.hydrate("kira", store)
    assert hydrated.backstory == ""
    assert hydrated.speaking_style == ""


def test_fallback_counts(registry):
    store = Store(":memory:")
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )
    session = CompanionSession(state, store, FailingLLM(), HashEmbedder())
    session.open()
    valence_before = state.affect.valence
    response, trace = session.turn("/gift cat")
    assert response == FALLBACK_LINES["severe_negative"]
    assert trace.fallback is True
    assert state.affect.valence < valence_before
    assert len(store.load_memories("kira", kind="episodic")) == 1


def test_fallback_band(registry):
    store = Store(":memory:")
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )
    session = CompanionSession(state, store, FailingLLM(), HashEmbedder())
    session.open()
    response, trace = session.turn("I brought you some chocolate cake!")
    assert response == FALLBACK_LINES["warm_positive"]
    assert trace.fallback is True


def test_mock_unaffected(registry):
    store = Store(":memory:")
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
        backstory="A test backstory.",
    )
    session = CompanionSession(state, store, MockLLM(), HashEmbedder())
    session.open()
    response, trace = session.turn("/gift cat")
    assert "disgusted_rejection" in response
    assert trace.fallback is False


def test_trace_summary(registry):
    store = Store(":memory:")
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )
    session = CompanionSession(state, store, MockLLM(), HashEmbedder())
    session.open()
    response, trace = session.turn("/gift cat")
    summary = format_trace(trace)
    assert "archetype=disgusted_rejection" in summary
    assert "fallback=False" in summary
