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
    perceive,
)
from companion.dynamics import (
    PHASES,
    WOUND_AMPLIFIER,
    WOUND_SALIENCE,
    WOUND_TAGS,
    WOUND_WINDOW_DAYS,
    phase_notes,
    phase_voice_delta,
    update_phases,
    wound_amplifier,
)
from companion.models import RelationshipVector

NOW = 1_750_000_000.0
embedder = HashEmbedder()


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


def wound(salience: float = WOUND_SALIENCE, tags: set[str] | None = None, age_days: float = 0.0) -> Memory:
    import time

    tags = tags or {"pain"}
    created = time.time() - age_days * 86400
    return Memory(
        id="wound-1",
        companion_id="k",
        kind="episodic",
        content="they hurt me",
        embedding=embedder.embed("they hurt me"),
        embedder=embedder.name,
        salience=salience,
        decay_rate=0.0,
        emotional_tags=list(tags),
        created_at=created,
        last_accessed=created,
        session_id="s1",
    )


def make_session(tmp_path, registry, db_name: str = "test.db", trust: float = 0.5, resentment: float = 0.0):
    store = Store(str(tmp_path / db_name))
    state = CompanionState.create(
        companion_id="k",
        name="Kira",
        registry=registry.all(),
    )
    state.relationship.trust = trust
    state.relationship.resentment = resentment
    session = CompanionSession(state, store, MockLLM(), embedder)
    session.open()
    return session, state, store


def test_61_phase_enters_at_threshold():
    rel = RelationshipVector(trust=0.70)
    assert update_phases([], rel) == ["high_trust"]
    rel.trust = 0.69
    assert update_phases([], rel) == []


def test_62_hysteresis_holds_inside_the_gap():
    rel = RelationshipVector(trust=0.65)
    active = update_phases(["high_trust"], rel)
    assert active == ["high_trust"]
    rel.trust = 0.59
    assert update_phases(["high_trust"], rel) == []
    rel.trust = 0.65
    assert update_phases([], rel) == []


def test_63_breached_trust_enter_below_exit_above():
    rel = RelationshipVector(trust=0.30)
    assert update_phases([], rel) == ["breached_trust"]
    rel.trust = 0.35
    assert update_phases(["breached_trust"], rel) == ["breached_trust"]
    rel.trust = 0.45
    assert update_phases(["breached_trust"], rel) == []


def test_64_multiple_phases_sorted():
    rel = RelationshipVector(trust=0.25, resentment=0.50)
    active = update_phases([], rel)
    assert active == ["breached_trust", "resentment"]
    dims = {p.dimension for p in PHASES}
    assert dims == {"trust", "intimacy", "resentment"}


def test_65_phase_voice_delta_sums():
    breached = phase_voice_delta(["breached_trust"])
    assert breached.temperature == pytest.approx(-0.3)
    assert breached.formality == pytest.approx(0.3)
    assert breached.humor == pytest.approx(-0.3)

    combined = phase_voice_delta(["high_trust", "high_intimacy"])
    assert combined.temperature == pytest.approx(0.25)

    empty = phase_voice_delta([])
    assert empty.temperature == pytest.approx(0.0)
    assert empty.verbosity == pytest.approx(0.0)
    assert empty.humor == pytest.approx(0.0)
    assert empty.formality == pytest.approx(0.0)
    assert empty.metaphor_density == pytest.approx(0.0)


def test_66_phase_notes_exact_strings():
    notes = phase_notes(["high_trust", "breached_trust"])
    assert notes == [
        "You feel safe with them; you may be more open than usual.",
        "You do not trust them right now. Keep your distance; do not explain why.",
    ]
    assert phase_notes([]) == []


def test_67_wound_amplifier_conditions():
    assert wound_amplifier([wound()]) == WOUND_AMPLIFIER
    assert wound_amplifier([wound(age_days=WOUND_WINDOW_DAYS + 0.1)]) == 1.0
    assert wound_amplifier([wound(salience=WOUND_SALIENCE - 0.1)]) == 1.0
    assert wound_amplifier([wound(tags={"joy"})]) == 1.0
    assert wound_amplifier([wound(tags={"fear"})]) == WOUND_AMPLIFIER
    assert wound_amplifier([]) == 1.0


def test_68_resentment_amplifies_negative_impact(registry):
    methods = MethodRegistry()
    neutral = evaluate(perceive("/insult me"), registry, trust=0.0, methods=methods, resentment=0.0)
    bitter = evaluate(perceive("/insult me"), registry, trust=0.0, methods=methods, resentment=0.5)
    assert neutral.impact == pytest.approx(-0.8)
    assert bitter.impact == pytest.approx(-0.8 * 1.15, abs=1e-3)


def test_69_resentment_leaves_positive_impact_alone(registry):
    methods = MethodRegistry()
    neutral = evaluate(perceive("/hug"), registry, trust=0.5, methods=methods, resentment=0.0)
    bitter = evaluate(perceive("/hug"), registry, trust=0.5, methods=methods, resentment=0.9)
    assert neutral.impact == pytest.approx(0.6)
    assert bitter.impact == pytest.approx(0.6)


def test_70_fresh_wound_cuts_deeper(tmp_path, registry):
    session_wound, state_wound, _ = make_session(tmp_path, registry, db_name="w.db")
    wound_mem = wound()
    session_wound._save_memory(wound_mem)
    _, trace_wound = session_wound.turn("/insult me")
    trust_wound = trace_wound.relationship_after.trust

    session_plain, _, _ = make_session(tmp_path, registry, db_name="p.db")
    _, trace_plain = session_plain.turn("/insult me")
    trust_plain = trace_plain.relationship_after.trust

    assert trust_wound == pytest.approx(0.3125, abs=1e-3)
    assert trust_plain == pytest.approx(0.33, abs=1e-3)
    assert trust_wound < trust_plain


def test_71_phase_voice_reaches_the_turn(tmp_path, registry):
    session, state, _ = make_session(tmp_path, registry, trust=0.20)
    state.active_phases = update_phases([], state.relationship)
    _, trace = session.turn("hello there")
    assert trace.active_phases == ["breached_trust"]
    assert trace.voice_after.humor == pytest.approx(trace.voice_before.humor - 0.3, abs=1e-6)
    assert trace.voice_after.temperature == pytest.approx(trace.voice_before.temperature - 0.3, abs=1e-6)


def test_72_phases_update_after_the_deltas(tmp_path, registry):
    session, state, _ = make_session(tmp_path, registry, trust=0.35)
    _, trace = session.turn("/insult me")
    assert trace.active_phases == []
    assert state.active_phases == ["breached_trust"]


def test_73_identity_at_default_relationship(tmp_path, registry):
    session, state, _ = make_session(tmp_path, registry)
    _, trace = session.turn("I brought you some chocolate cake!")
    assert trace.activation.impact == pytest.approx(0.70)
    assert trace.active_phases == []
    assert state.active_phases == []
