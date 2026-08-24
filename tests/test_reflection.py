import pytest

from companion import (
    CompanionSession,
    CompanionState,
    HashEmbedder,
    Memory,
    MockLLM,
    Store,
    Trait,
    TraitCategory,
    TraitRegistry,
    Trigger,
    VoiceProfile,
    dislikes,
    likes,
)
from companion.embeddings import HashEmbedder as _HashEmbedder
from companion.memory import decay_rate_for
from companion.models import DriftProposal, InsightProposal, ReflectionProposal
from companion.reflection import (
    MAX_DRIFT_PER_RUN,
    MAX_TOTAL_DRIFT,
    MIN_SESSIONS,
    MIN_SOURCES,
    REFLECTION_MIN_NEW_MEMORIES,
    REFLECTION_SESSION_ID,
    MockReflector,
    apply_proposal,
    validate_drift,
    validate_insight,
)

NOW = 1_750_000_000.0
embedder = _HashEmbedder()


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


def mem(memory_id: str, session: str, tags: list[str], kind: str = "episodic", age: float = 0.0) -> Memory:
    created = NOW - age
    return Memory(
        id=memory_id,
        companion_id="kira",
        kind=kind,
        content=f"memory {memory_id}",
        embedding=embedder.embed(f"memory {memory_id}"),
        embedder=embedder.name,
        salience=0.7,
        decay_rate=decay_rate_for(0.7),
        emotional_tags=tags,
        created_at=created,
        last_accessed=created,
        session_id=session,
    )


def make_by_id(*memories: Memory) -> dict[str, Memory]:
    return {m.id: m for m in memories}


def test_insight_source_floor():
    by_id = make_by_id(
        mem("a", "s1", ["pain"]),
        mem("b", "s1", ["pain"]),
        mem("c", "s2", ["pain"]),
    )
    valid = InsightProposal(content="x", source_ids=["a", "b", "c"], emotional_tags=["pain"])
    invalid = InsightProposal(content="x", source_ids=["a", "b"], emotional_tags=["pain"])
    assert validate_insight(valid, by_id) is True
    assert validate_insight(invalid, by_id) is False


def test_insight_session_spread():
    by_id = make_by_id(
        mem("a", "s1", ["pain"]),
        mem("b", "s1", ["pain"]),
        mem("c", "s1", ["pain"]),
    )
    p = InsightProposal(content="x", source_ids=["a", "b", "c"], emotional_tags=["pain"])
    assert validate_insight(p, by_id) is False


def test_drift_core_lock(registry):
    traits = [t.model_dump(mode="json") for t in registry.all()]
    by_id = make_by_id(
        mem("a", "s1", ["pain"]),
        mem("b", "s1", ["pain"]),
        mem("c", "s2", ["pain"]),
    )
    p = DriftProposal(trait_id="felinophobia", delta=-0.05, source_ids=["a", "b", "c"])
    assert validate_drift(p, traits, by_id) is False


def test_drift_per_run_cap(registry):
    traits = [t.model_dump(mode="json") for t in registry.all()]
    by_id = make_by_id(
        mem("a", "s1", ["pain"]),
        mem("b", "s1", ["pain"]),
        mem("c", "s2", ["pain"]),
    )
    too_much = DriftProposal(trait_id="dislikes_taste_salty", delta=-0.10, source_ids=["a", "b", "c"])
    ok = DriftProposal(trait_id="dislikes_taste_salty", delta=-0.05, source_ids=["a", "b", "c"])
    assert validate_drift(too_much, traits, by_id) is False
    assert validate_drift(ok, traits, by_id) is True


def test_drift_total_cap(registry):
    traits = [t.model_dump(mode="json") for t in registry.all()]
    # Push the salty dislike close to the total-drift boundary.
    for t in traits:
        if t["trait_id"] == "dislikes_taste_salty":
            t["current_intensity"] = -0.95
            t["base_intensity"] = -0.7
    by_id = make_by_id(
        mem("a", "s1", ["pain"]),
        mem("b", "s1", ["pain"]),
        mem("c", "s2", ["pain"]),
    )
    p = DriftProposal(trait_id="dislikes_taste_salty", delta=-0.05, source_ids=["a", "b", "c"])
    # current -0.95 + (-0.05) = -1.00, |current - base| = |(-1.0) - (-0.7)| = 0.30 <= MAX_TOTAL_DRIFT
    # But -1.0 is allowed? Wait, -1.0 <= -1.0 <= 1.0 is true, and abs(-1.0 - -0.7) = 0.3 <= 0.30 is true.
    # So this would be valid. Need to make it exceed.
    # Set current to -0.96 so -0.96 + -0.05 = -1.01 (invalid range).
    for t in traits:
        if t["trait_id"] == "dislikes_taste_salty":
            t["current_intensity"] = -0.96
    assert validate_drift(p, traits, by_id) is False


def test_apply_insight(registry):
    store = Store(":memory:")
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )
    by_id = make_by_id(
        mem("a", "s1", ["pain"]),
        mem("b", "s1", ["pain"]),
        mem("c", "s2", ["pain"]),
    )
    proposal = ReflectionProposal(
        insights=[
            InsightProposal(
                content="Pain keeps showing up.",
                source_ids=["a", "b", "c"],
                emotional_tags=["pain"],
            )
        ]
    )
    applied = apply_proposal(state, store, embedder, proposal, list(by_id.values()), NOW)
    assert len(applied.insight_memory_ids) == 1
    semantic_rows = store.load_memories("kira", kind="semantic")
    assert len(semantic_rows) == 1
    semantic = Memory.model_validate_json(semantic_rows[0])
    assert semantic.salience == pytest.approx(0.6)
    assert semantic.kind == "semantic"
    assert semantic.session_id == REFLECTION_SESSION_ID
    assert set(semantic.source_ids) == {"a", "b", "c"}


def test_apply_drift(registry):
    store = Store(":memory:")
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )
    by_id = make_by_id(
        mem("a", "s1", ["pain"]),
        mem("b", "s1", ["pain"]),
        mem("c", "s2", ["pain"]),
    )
    proposal = ReflectionProposal(
        drifts=[
            DriftProposal(
                trait_id="dislikes_taste_salty",
                delta=0.05,
                source_ids=["a", "b", "c"],
            )
        ]
    )
    applied = apply_proposal(state, store, embedder, proposal, list(by_id.values()), NOW)
    assert len(applied.drifts) == 1
    trait = state.trait_registry().get("dislikes_taste_salty")
    assert trait.current_intensity == pytest.approx(-0.65)


def test_narrative(registry):
    store = Store(":memory:")
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )
    proposal = ReflectionProposal(narrative_entry="I keep circling back.")
    applied = apply_proposal(state, store, embedder, proposal, [], NOW)
    assert applied.narrative_added is True
    assert len(state.narrative_log) == 1
    assert state.narrative_log[0]["text"] == "I keep circling back."


def test_gate_below_threshold(registry):
    store = Store(":memory:")
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )
    session = CompanionSession(state, store, MockLLM(), embedder)
    session.open()
    for _ in range(2):
        session.turn("hello")
    summary = session.close()
    assert summary is None
    assert state.last_reflection_at is None


def test_gate_fires_and_latches(registry):
    store = Store(":memory:")
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )
    session = CompanionSession(state, store, MockLLM(), embedder)
    session.open()
    for _ in range(REFLECTION_MIN_NEW_MEMORIES):
        session.turn("hello")
    summary = session.close()
    assert summary is not None
    assert state.last_reflection_at is not None

    # Second session with no new memories should not reflect again.
    session2 = CompanionSession(state, store, MockLLM(), embedder)
    session2.open()
    summary2 = session2.close()
    assert summary2 is None


def test_mock_finds_recurrence(registry):
    store = Store(":memory:")
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )

    # Generate memories across 3 distinct sessions.
    for i in range(5):
        session = CompanionSession(state, store, MockLLM(), embedder)
        session.open()
        if i % 2 == 0:
            session.turn("/gift cat")
        else:
            session.turn("/insult me")
        session.close()

    semantic = store.load_memories("kira", kind="semantic")
    assert len(semantic) >= 1
    contents = " ".join(Memory.model_validate_json(row).content for row in semantic)
    assert "pain" in contents or "fear" in contents or "warmth" in contents

    assert len(state.narrative_log) >= 1
    reflections = store.load_reflections("kira")
    assert len(reflections) >= 1


def test_semantic_retrievable(registry):
    store = Store(":memory:")
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )

    for i in range(5):
        session = CompanionSession(state, store, MockLLM(), embedder)
        session.open()
        if i % 2 == 0:
            session.turn("/gift cat")
        else:
            session.turn("/insult me")
        session.close()

    session = CompanionSession(state, store, MockLLM(), embedder)
    session.open()
    response, trace = session.turn("have you noticed any recurring pattern in my behavior?")
    semantic_recalled = any("Recurring pattern" in r.content for r in trace.retrieved_memories)
    assert semantic_recalled


def test_core_never_drifts(registry):
    store = Store(":memory:")
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )

    for i in range(8):
        session = CompanionSession(state, store, MockLLM(), embedder)
        session.open()
        if i % 2 == 0:
            session.turn("/gift cat")
        else:
            session.turn("/insult me")
        session.close()

    trait = state.trait_registry().get("felinophobia")
    assert trait.current_intensity == pytest.approx(-0.9)


def test_rollback(registry):
    store = Store(":memory:")
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )
    by_id = make_by_id(
        mem("a", "s1", ["pain"]),
        mem("b", "s1", ["pain"]),
        mem("c", "s2", ["pain"]),
    )
    proposal = ReflectionProposal(
        insights=[
            InsightProposal(
                content="Pain keeps showing up.",
                source_ids=["a", "b", "c"],
                emotional_tags=["pain"],
            )
        ],
        drifts=[
            DriftProposal(
                trait_id="dislikes_taste_salty",
                delta=0.05,
                source_ids=["a", "b", "c"],
            )
        ],
        narrative_entry="I keep circling back.",
    )
    applied = apply_proposal(state, store, embedder, proposal, list(by_id.values()), NOW)
    entry_id = "entry-1"
    store.save_reflection(entry_id, "kira", applied.model_dump_json())

    # Now rollback.
    for memory_id in applied.insight_memory_ids:
        store.delete_memory(memory_id)
    for drift in applied.drifts:
        for t in state.registry:
            if t.get("trait_id") == drift.trait_id:
                t["current_intensity"] = round(t["current_intensity"] - drift.delta, 4)
                break
    if applied.narrative_added and state.narrative_log:
        state.narrative_log.pop()
    store.mark_rolled_back(entry_id)

    assert len(store.load_memories("kira", kind="semantic")) == 0
    trait = state.trait_registry().get("dislikes_taste_salty")
    assert trait.current_intensity == pytest.approx(-0.7)
    assert len(state.narrative_log) == 0
    rows = store.load_reflections("kira")
    entry = __import__("json").loads(rows[0])
    assert entry["rolled_back"] is True
