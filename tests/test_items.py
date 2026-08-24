import math
from pathlib import Path

import pytest

from companion import (
    CompanionSession,
    CompanionState,
    HashEmbedder,
    ItemRegistry,
    MethodRegistry,
    MockLLM,
    Store,
    Trait,
    TraitCategory,
    TraitRegistry,
    Trigger,
    dislikes,
    evaluate,
    item_stimuli,
    likes,
    load_items,
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


def vegetarian() -> Trait:
    return Trait(
        trait_id="vegetarian",
        category=TraitCategory.SURFACE,
        description="Raised above a bakery on bread, fruit and sweets; the idea of eating flesh genuinely revolts her.",
        triggers=[Trigger(domain="tag", values=["meat"])],
        base_intensity=-0.75,
        current_intensity=-0.75,
        curve="steep",
        archetypes_negative=["disgusted_rejection", "cold_withdrawal"],
        voice_modifiers={"temperature": -0.4, "humor": -0.5, "verbosity": -0.2},
    )


@pytest.fixture
def items(tmp_path):
    yaml_path = tmp_path / "items.yaml"
    yaml_path.write_text(
        """
categories: [food, toy, weapon]
items:
  - id: steak
    name: Steak
    category: food
    tags: [luxurious, savory, meat]

  - id: plush_cat
    name: Plush cat
    category: toy
    tags: [soft, comforting, cute]
    aliases: [stuffed cat]

  - id: sword
    name: Sword
    category: weapon
    tags: [iron, weapon, potentially_dangerous]
""",
        encoding="utf-8",
    )
    return load_items(yaml_path)


@pytest.fixture
def registry() -> TraitRegistry:
    return TraitRegistry(
        [
            likes("tag", "sweet", intensity=0.7),
            vegetarian(),
            felinophobia(),
            dislikes("tag", "potentially_dangerous", intensity=-0.6),
        ]
    )


def test_79_load_items_parses(items):
    steak = items.get("steak")
    assert steak is not None
    assert steak.name == "Steak"
    assert steak.category == "food"
    assert steak.tags == ["luxurious", "savory", "meat"]
    assert steak.aliases == []

    plush = items.get("plush_cat")
    assert plush is not None
    assert plush.name == "Plush cat"
    assert plush.category == "toy"
    assert plush.tags == ["soft", "comforting", "cute"]
    assert plush.aliases == ["stuffed cat"]


def test_80_unknown_item_is_none(items):
    assert items.get("spoon") is None


def test_81_item_stimuli_expansion(items):
    steak = items.get("steak")
    keys = [s.key() for s in item_stimuli(steak)]
    assert keys == [
        "item:steak",
        "tag:luxurious",
        "tag:savory",
        "tag:meat",
        "category:food",
    ]


def test_82_match_text_alias_and_boundaries(items):
    found = items.match_text("I got a stuffed cat!")
    assert len(found) == 1
    assert found[0].item_id == "plush_cat"

    assert items.match_text("steakhouse") == []

    found2 = items.match_text("I bought steak and a sword")
    ids = {it.item_id for it in found2}
    assert ids == {"steak", "sword"}


def test_82b_load_items_validates(tmp_path):
    bad_cat = tmp_path / "bad_cat.yaml"
    bad_cat.write_text(
        "categories: [food]\nitems:\n  - id: steak\n    name: Steak\n    category: weapon\n    tags: [meat]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="undeclared category"):
        load_items(bad_cat)

    bad_id = tmp_path / "bad_id.yaml"
    bad_id.write_text(
        "categories: [food]\nitems:\n  - id: 'plush cat'\n    name: 'Plush cat'\n    category: food\n    tags: [soft]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="single token"):
        load_items(bad_id)


def test_83_perceive_without_items_is_identity():
    p = perceive("/gift cat")
    assert [s.key() for s in p.stimuli] == ["entity:cat", "action:gift"]


def test_84_perceive_expands_items(items):
    p = perceive("/gift steak", items=items)
    keys = [s.key() for s in p.stimuli]
    assert "item:steak" in keys
    assert "tag:meat" in keys
    assert "category:food" in keys
    assert "action:gift" in keys


def test_85_free_text_mentions_expand(items):
    p = perceive("I made steak for dinner tonight.", items=items)
    assert p.method is None
    assert any(s.key() == "tag:meat" for s in p.stimuli)


def test_86_vegetarian_meets_steak(registry, items):
    p = perceive("/gift steak", items=items)
    act = evaluate(p, registry, trust=0.5, methods=MethodRegistry(), companion_name="Kira")
    expected = -(0.75 ** 0.5) * 0.85
    assert act.impact == pytest.approx(expected, abs=1e-3)
    assert act.archetype == "disgusted_rejection"
    assert not any(c.trait_id.startswith("method:") for c in act.contributions)
    assert any(c.trait_id == "vegetarian" for c in act.contributions)


def test_87_plush_cat_is_not_a_cat(registry, items):
    p = perceive("/gift plush_cat", items=items)
    act = evaluate(p, registry, trust=0.5, methods=MethodRegistry(), companion_name="Kira")
    assert not any(c.trait_id == "felinophobia" for c in act.contributions)
    assert act.impact == pytest.approx(0.30)
    assert act.archetype == "neutral"


def test_88_tag_conflict_is_emergent_ambivalence(items):
    reg = TraitRegistry(
        [
            likes("tag", "luxurious", intensity=0.7),
            dislikes("tag", "meat", intensity=-0.72),
        ]
    )
    p = perceive("/gift steak", items=items)
    act = evaluate(p, reg, trust=0.0)
    assert act.ambivalent is True
    assert abs(act.impact) < 0.2


def test_89_items_wire_through_the_session(registry, items, tmp_path):
    store = Store(str(tmp_path / "test.db"))
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )
    state.relationship.trust = 0.5
    session = CompanionSession(state, store, MockLLM(), HashEmbedder(), items=items)
    session.open()
    response, trace = session.turn("/gift steak")
    session.close()

    assert trace is not None
    assert any(c.trait_id == "vegetarian" for c in trace.activation.contributions)
    assert trace.relationship_after.trust < 0.5


def test_90_free_text_item_never_moves_relationship(registry, items, tmp_path):
    store = Store(str(tmp_path / "test.db"))
    state = CompanionState.create(
        companion_id="kira",
        name="Kira",
        registry=registry.all(),
    )
    state.relationship.trust = 0.5
    session = CompanionSession(state, store, MockLLM(), HashEmbedder(), items=items)
    session.open()
    response, trace = session.turn("I made steak for dinner tonight.")
    session.close()

    assert trace is not None
    assert trace.activation.impact < -0.5
    assert trace.relationship_after.trust == pytest.approx(0.5)
