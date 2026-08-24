import pytest

from companion import (
    CompanionSession,
    CompanionState,
    HashEmbedder,
    Memory,
    MethodRegistry,
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
    outcome_for,
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


def test_unknown_method():
    methods = MethodRegistry()
    error = methods.validate("dance", [])
    assert error is not None
    assert "Unknown method" in error
    assert "/gift" in error


def test_arg_counts():
    methods = MethodRegistry()
    assert methods.validate("gift", []) is not None
    assert methods.validate("gift", ["cat"]) is None
    assert methods.validate("hug", ["me"]) is not None
    assert methods.validate("hug", []) is None


def test_vacuum_fill(registry):
    act = evaluate(perceive("/hug"), registry, trust=0.5, methods=MethodRegistry(), companion_name="Kira")
    assert act.impact == pytest.approx(0.6)
    assert act.archetype == "warm_positive"
    assert any(c.trait_id == "method:hug" for c in act.contributions)


def test_traits_rule_domain(registry):
    act = evaluate(perceive("/gift cat"), registry, trust=0.5, methods=MethodRegistry(), companion_name="Kira")
    assert act.impact == pytest.approx(-(0.9**0.5) * 0.85, abs=0.001)
    assert not any(c.trait_id.startswith("method:") for c in act.contributions)
    assert act.archetype == "disgusted_rejection"


def test_unknown_item_gift(registry):
    act = evaluate(perceive("/gift rock"), registry, trust=0.5, methods=MethodRegistry(), companion_name="Kira")
    assert act.impact == pytest.approx(0.3)
    assert act.archetype == "neutral"


def test_self_directed_insult(registry):
    act = evaluate(perceive("/insult me"), registry, trust=0.0, methods=MethodRegistry(), companion_name="Kira")
    assert act.impact == pytest.approx(-0.8)
    assert act.archetype == "severe_negative"


def test_other_target_insult(registry):
    act = evaluate(perceive("/insult spiders"), registry, trust=0.5, methods=MethodRegistry(), companion_name="Kira")
    assert act.impact == pytest.approx(0.0)
    assert not any(c.trait_id.startswith("method:") for c in act.contributions)


def test_name_as_self(registry):
    act = evaluate(perceive("/insult kira"), registry, trust=0.5, methods=MethodRegistry(), companion_name="Kira")
    assert any(c.trait_id == "method:insult" for c in act.contributions)


def test_social_traits():
    reg = TraitRegistry(
        [
            likes("taste", "sweet", intensity=0.7),
            dislikes("taste", "salty", intensity=-0.7),
            felinophobia(),
            dislikes("social", "hug", intensity=-0.6),
        ]
    )
    act = evaluate(perceive("/hug"), reg, trust=0.5, methods=MethodRegistry(), companion_name="Kira")
    assert act.impact == pytest.approx(-0.6 * (1 - 0.3 * 0.5), abs=0.001)
    assert not any(c.trait_id.startswith("method:") for c in act.contributions)


def test_outcome_mapping():
    assert outcome_for(0.7) == "accepted"
    assert outcome_for(-0.7) == "rejected"
    assert outcome_for(0.1) == "acknowledged"


def test_relationship_dims(tmp_path, registry):
    store = Store(str(tmp_path / "test.db"))
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )
    session = CompanionSession(state, store, MockLLM(), HashEmbedder())
    session.open()
    session.turn("/hug")
    session.turn("/insult me")
    session.close()

    hydrated = CompanionState.hydrate("kira", store)
    assert hydrated.relationship.intimacy > 0.1
    assert hydrated.relationship.resentment > 0.0
    assert hydrated.relationship.trust < 0.5


def test_invalid_is_not_a_turn(tmp_path, registry):
    store = Store(str(tmp_path / "test.db"))
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )
    session = CompanionSession(state, store, MockLLM(), HashEmbedder())
    session.open()
    trust_before = state.relationship.trust
    response, trace = session.turn("/dance")
    assert trace is None
    assert state.relationship.trust == trust_before
    assert len(store.load_memories("kira", kind="episodic")) == 0


def test_outcome_in_memory(tmp_path, registry):
    store = Store(str(tmp_path / "test.db"))
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )
    session = CompanionSession(state, store, MockLLM(), HashEmbedder())
    session.open()
    session.turn("/gift cat")
    session.close()

    memories = [Memory.model_validate_json(row) for row in store.load_memories("kira", kind="episodic")]
    assert len(memories) == 1
    assert "(rejected)" in memories[0].content
    assert set(memories[0].emotional_tags) == {"pain", "fear"}
