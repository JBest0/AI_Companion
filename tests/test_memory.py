import pytest

from companion import (
    AffectState,
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
    TurnTrace,
    VoiceProfile,
    cosine,
    decay_rate_for,
    effective_salience,
    emotional_tags_for,
    evaluate,
    perceive,
    retrieve,
    write_salience,
)

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
    from companion import dislikes, likes

    return TraitRegistry(
        [
            likes("taste", "sweet", intensity=0.7),
            dislikes("taste", "salty", intensity=-0.7),
            felinophobia(),
        ]
    )


def make_memory(
    content,
    salience=0.5,
    tags=None,
    age_days=0,
    access_count=0,
    last_accessed_age_days=None,
) -> Memory:
    created_at = NOW - age_days * 86400
    last_accessed = NOW - (last_accessed_age_days if last_accessed_age_days is not None else age_days) * 86400
    return Memory(
        companion_id="kira",
        kind="episodic",
        content=content,
        embedding=embedder.embed(content),
        embedder=embedder.name,
        salience=salience,
        decay_rate=decay_rate_for(salience),
        emotional_tags=tags or [],
        created_at=created_at,
        last_accessed=last_accessed,
        access_count=access_count,
    )


def test_cosine_properties():
    a = embedder.embed("the cat sat on the mat")
    b = embedder.embed("the cat sat on the mat")
    assert cosine(a, b) == pytest.approx(1.0, abs=1e-9)
    assert cosine(a, [1.0, 2.0]) == 0.0
    assert cosine([], []) == 0.0


def test_decay_math():
    assert decay_rate_for(0.8) == pytest.approx(0.000333)
    m = make_memory("x", salience=0.8, age_days=30)
    assert effective_salience(m, NOW) == pytest.approx(0.8 * (1 - 0.000333) ** 30, abs=1e-6)


def test_anchor_never_decays():
    m = make_memory("x", salience=0.95, age_days=3650)
    assert decay_rate_for(0.95) == 0.0
    assert effective_salience(m, NOW) == pytest.approx(0.95, abs=1e-6)


def test_rehearsal_boost():
    m = make_memory("x", salience=0.5, access_count=5)
    assert effective_salience(m, NOW) == pytest.approx(0.5 * (1 + 0.1 * 5), abs=1e-6)


def test_write_mapping():
    assert write_salience(-0.84) == 0.84
    assert write_salience(0.0) == 0.1
    assert decay_rate_for(0.84) == pytest.approx(0.000333)


def test_write_time_tags(registry):
    act = evaluate(perceive("/gift cat"), registry, trust=0.5)
    assert emotional_tags_for(act) == ["pain", "fear"]

    act = evaluate(perceive("chocolate and something salty"), registry, trust=0.0)
    assert act.ambivalent is True
    assert emotional_tags_for(act) == ["conflict"]


def test_retrieval_ranking():
    cat_mem = make_memory("User gave me a cat", salience=0.7, tags=["pain", "fear"])
    choc_mem = make_memory("User gave me chocolate", salience=0.7, tags=["warmth"])
    retrieved = retrieve(
        [cat_mem, choc_mem],
        embedder.embed("remember the cat"),
        AffectState(),
        NOW,
        embedder.name,
    )
    assert len(retrieved) >= 1
    assert retrieved[0].memory_id == cat_mem.id


def test_resonance_boost():
    cat_mem = make_memory("User gave me a cat", salience=0.5, tags=["pain", "fear"])
    query = embedder.embed("remember the cat")
    neutral = retrieve([cat_mem], query, AffectState(), NOW, embedder.name)
    negative = retrieve([cat_mem], query, AffectState(valence=-0.5), NOW, embedder.name)
    assert len(neutral) == 1
    assert len(negative) == 1
    assert negative[0].score > neutral[0].score


def test_threshold_and_cap():
    memories = [make_memory(f"unrelated topic number {i}", salience=0.1) for i in range(10)]
    retrieved = retrieve(
        memories,
        embedder.embed("quantum physics"),
        AffectState(),
        NOW,
        embedder.name,
    )
    assert len(retrieved) <= 5
    assert all(r.score >= 0.15 for r in retrieved)


def test_cross_session_recall(tmp_path, registry):
    store = Store(str(tmp_path / "test.db"))
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
        voice_baseline=VoiceProfile(),
    )
    session = CompanionSession(state, store, MockLLM(), HashEmbedder())
    session.open()
    session.turn("/gift cat")
    session.close()

    state2 = CompanionState.hydrate("kira", store)
    session2 = CompanionSession(state2, store, MockLLM(), HashEmbedder())
    session2.open()
    response, trace = session2.turn("Do you remember the cat I gave you?")
    assert trace.retrieved_memories
    assert "/gift cat" in trace.retrieved_memories[0].content


def test_no_self_recall(tmp_path, registry):
    store = Store(str(tmp_path / "test.db"))
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )
    session = CompanionSession(state, store, MockLLM(), HashEmbedder())
    session.open()
    response, trace = session.turn("/gift cat")
    assert trace.retrieved_memories == []


def test_rehearsal_persists(tmp_path, registry):
    store = Store(str(tmp_path / "test.db"))
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )
    session = CompanionSession(state, store, MockLLM(), HashEmbedder())
    session.open()
    session.turn("/gift cat")
    session.turn("Do you remember the cat I gave you?")
    session.close()

    memories = [Memory.model_validate_json(row) for row in store.load_memories("kira", kind="episodic")]
    cat_mem = next(m for m in memories if "/gift cat" in m.content)
    assert cat_mem.access_count >= 1

    traces = [TurnTrace.model_validate_json(row) for row in store.load_traces("kira")]
    recall_trace = next(t for t in traces if "remember the cat" in t.user_input)
    assert recall_trace.retrieved_memories
    top = recall_trace.retrieved_memories[0]
    assert top.breakdown["salience"] > cat_mem.salience * 0.99
