import pydantic
import pytest

from companion import (
    AffectState,
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
    evaluate,
    likes,
    perceive,
)


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


def test_core_trait_lock():
    with pytest.raises(pydantic.ValidationError):
        Trait(
            trait_id="broken",
            category=TraitCategory.CORE,
            triggers=[Trigger(domain="entity", values=["cat"])],
            base_intensity=-0.9,
            current_intensity=-0.5,
        )


def test_helper_signs():
    assert likes("taste", "sweet").current_intensity > 0
    assert dislikes("taste", "salty").current_intensity < 0
    with pytest.raises(AssertionError):
        likes("taste", "sweet", intensity=-0.3)


def test_lexicon_perception():
    p = perceive("I brought you chocolate cake")
    assert any(s.key() == "taste:sweet" for s in p.stimuli)


def test_method_parsing():
    p = perceive("/gift cat")
    assert p.method == "gift"
    assert p.method_args == ["cat"]
    keys = [s.key() for s in p.stimuli]
    assert "entity:cat" in keys
    assert "action:gift" in keys


def test_positive_routing(registry):
    act = evaluate(perceive("have some chocolate"), registry, trust=0.5)
    assert act.impact > 0.3
    assert act.archetype in {"warm_positive", "delight"}


def test_negative_routing(registry):
    act = evaluate(perceive("want a salty pretzel?"), registry, trust=0.5)
    assert act.impact < -0.3


def test_emergent_ambivalence(registry):
    act = evaluate(perceive("chocolate and something salty"), registry, trust=0.0)
    assert act.ambivalent is True
    assert act.director_notes


def test_steep_core_archetype(registry):
    act = evaluate(perceive("/gift cat"), registry, trust=0.5)
    assert act.impact == pytest.approx(-(0.9**0.5) * 0.85, abs=0.001)
    assert act.archetype == "disgusted_rejection"
    assert act.hard_constraints


def test_trust_buffer(registry):
    impact_lo = evaluate(perceive("/gift cat"), registry, trust=0.0).impact
    impact_hi = evaluate(perceive("/gift cat"), registry, trust=1.0).impact
    assert impact_hi > impact_lo


def test_baseline_decay():
    baseline = AffectState(valence=-0.4, arousal=0.3)
    affect = AffectState(valence=0.8, arousal=0.3)
    affect.decay_toward(baseline, hours=12)
    assert affect.valence == pytest.approx(0.2, abs=1e-9)

    affect2 = AffectState(valence=0.8, arousal=0.3)
    affect2.decay_toward(baseline, hours=1000)
    assert affect2.valence == pytest.approx(-0.4, abs=1e-3)


def test_state_roundtrip(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=[likes("taste", "sweet", intensity=0.7), dislikes("taste", "salty", intensity=-0.7), felinophobia()],
        voice_baseline=VoiceProfile(temperature=0.3),
    )
    state.affect.valence = 0.9
    state.checkpoint(store)

    hydrated = CompanionState.hydrate("kira", store)
    assert hydrated is not None
    assert hydrated.affect.valence == pytest.approx(0.9)
    assert len(hydrated.trait_registry()) == 3
    assert hydrated.voice_baseline.temperature == pytest.approx(0.3)


def test_end_to_end(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=[likes("taste", "sweet", intensity=0.7), dislikes("taste", "salty", intensity=-0.7), felinophobia()],
    )
    session = CompanionSession(state, store, MockLLM(), HashEmbedder())
    session.open()

    response, trace = session.turn("/gift cat")
    assert "disgusted_rejection" in response
    assert trace.activation.impact < -0.6
    assert trace.affect_after.valence < trace.affect_before.valence
    assert trace.relationship_after.trust < trace.relationship_before.trust

    assert len(store.load_traces("kira", limit=50)) >= 1

    hydrated = CompanionState.hydrate("kira", store)
    assert hydrated is not None
    assert hydrated.relationship.trust < 0.5
