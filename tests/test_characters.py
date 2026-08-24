import time
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from companion import (
    CharacterManager,
    CharacterSpec,
    CompanionState,
    Store,
    likes,
    load_character,
    load_items,
    spec_warnings,
    slugify,
    template_spec,
)
from companion.characters import CHARACTER_TEMPLATES, SLUG_RE
from companion.models import AffectState, Trait, TraitCategory, Trigger, VoiceProfile


@pytest.fixture
def cdir(tmp_path):
    chars = tmp_path / "characters"
    chars.mkdir()
    (chars / "rex.yaml").write_text(
        "name: Rex\n"
        "mood_baseline: {valence: 0.0, arousal: 0.2}\n"
        "voice_baseline:\n"
        "  temperature: 0.5\n"
        "likes:\n"
        "  - { domain: entity, values: [dog], intensity: 0.5 }\n",
        encoding="utf-8",
    )
    return chars


@pytest.fixture
def manager(cdir, tmp_path):
    return CharacterManager(cdir, Store(tmp_path / "t.db"))


def test_91_slug_rules():
    assert slugify("Captain Mira") == "captain-mira"
    assert slugify("!!!") == ""
    for bad in ("Kira", "1abc", "a" * 33, "-x", "a"):
        assert not SLUG_RE.match(bad)
        with pytest.raises(ValueError):
            CharacterSpec.from_yaml_dict({"name": "X"}, bad)
    for good in ("ab", "k2", "captain-mira"):
        assert SLUG_RE.match(good)
        CharacterSpec.from_yaml_dict({"name": "X"}, good)


def test_92_kira_loads_through_spec():
    char = load_character(Path(__file__).parent.parent / "characters" / "kira.yaml")
    registry = char["registry"]
    trait_ids = [t.trait_id for t in registry]
    assert trait_ids == [
        "likes_taste_sweet",
        "likes_activity_climbing",
        "likes_entity_dog",
        "dislikes_taste_salty",
        "dislikes_topic_weather",
        "dislikes_tag_potentially_dangerous",
        "vegetarian",
        "felinophobia",
    ]
    felinophobia = next(t for t in registry if t.trait_id == "felinophobia")
    assert felinophobia.category == TraitCategory.CORE
    assert felinophobia.curve == "steep"
    assert felinophobia.salience_class == "high"
    vegetarian = next(t for t in registry if t.trait_id == "vegetarian")
    assert vegetarian.voice_modifiers.temperature == pytest.approx(-0.4)
    assert len(char["definition_hash"]) == 12
    int(char["definition_hash"], 16)  # hex


def test_93_create_writes_and_roundtrips(manager):
    spec = CharacterSpec(
        char_id="nova",
        name="Nova",
        avatar="🌟",
        mood_baseline=AffectState(valence=0.3, arousal=0.4),
        voice_baseline=VoiceProfile(temperature=0.7, humor=0.2),
        backstory="A stargazer.",
        speaking_style="Thoughtful, slow.",
        likes=[{"domain": "topic", "values": ["weather"], "intensity": 0.4}],
        traits=[Trait(
            trait_id="loves_stars",
            category=TraitCategory.SURFACE,
            triggers=[Trigger(domain="topic", values=["stars"])],
            base_intensity=0.6,
            current_intensity=0.6,
            curve="steep",
        )],
    )
    manager.create(spec)
    loaded = manager.load("nova")
    assert loaded.model_dump() == spec.model_dump()


def test_94_duplicate_id_rejected(manager):
    spec = CharacterSpec(char_id="rex", name="Rex Two", likes=[])
    with pytest.raises(ValueError, match="already exists"):
        manager.create(spec)
    manager.archive("rex")
    with pytest.raises(ValueError, match="already exists"):
        manager.create(spec)


def test_95_preference_signs_enforced():
    with pytest.raises(ValidationError):
        CharacterSpec(char_id="x", name="X", likes=[
            {"domain": "taste", "values": ["sweet"], "intensity": -0.5}
        ])
    with pytest.raises(ValidationError):
        CharacterSpec(char_id="x", name="X", dislikes=[
            {"domain": "taste", "values": ["salty"], "intensity": 0.5}
        ])
    with pytest.raises(ValidationError):
        CharacterSpec(char_id="x", name="X", likes=[
            {"domain": "taste", "values": ["sweet"], "intensity": 0.0}
        ])


def test_96_reserved_domains_and_bad_curve():
    with pytest.raises(ValidationError):
        CharacterSpec(char_id="x", name="X", traits=[Trait(
            trait_id="bad",
            triggers=[Trigger(domain="social", values=["hug"])],
            base_intensity=0.5,
            current_intensity=0.5,
        )])
    with pytest.raises(ValidationError):
        CharacterSpec(char_id="x", name="X", traits=[Trait(
            trait_id="bad",
            triggers=[Trigger(domain="taste", values=["sweet"])],
            base_intensity=0.5,
            current_intensity=0.5,
            curve="spiky",
        )])
    with pytest.raises(ValidationError):
        CharacterSpec(char_id="x", name="X", likes=[
            {"domain": "weather", "values": ["rain"], "intensity": 0.5}
        ])
    with pytest.raises(ValidationError):
        CharacterSpec(char_id="x", name="X", traits=[Trait(
            trait_id="bad",
            triggers=[Trigger(domain="entity", values=["Cat"])],
            base_intensity=0.5,
            current_intensity=0.5,
        )])


def test_97_drifted_trait_rejected():
    with pytest.raises(ValidationError, match="base_intensity"):
        CharacterSpec(char_id="xx", name="X", traits=[Trait(
            trait_id="drifted",
            category=TraitCategory.SURFACE,
            triggers=[Trigger(domain="taste", values=["sweet"])],
            base_intensity=0.5,
            current_intensity=0.6,
        )])


def test_98_warnings_not_errors(manager):
    spec = CharacterSpec(
        char_id="dragonfan",
        name="Dragon Fan",
        traits=[Trait(
            trait_id="dragon_lover",
            triggers=[Trigger(domain="entity", values=["dragon"])],
            base_intensity=0.5,
            current_intensity=0.5,
        )],
    )
    warns = spec_warnings(spec)
    assert len(warns) == 1
    assert "entity:dragon" in warns[0]

    kira_path = Path(__file__).parent.parent / "characters" / "kira.yaml"
    kira_spec = CharacterSpec.from_yaml_dict(
        yaml.safe_load(kira_path.read_text(encoding="utf-8")), "kira")
    items = load_items(Path(__file__).parent.parent / "items.yaml")
    assert spec_warnings(kira_spec, items=items) == []


def test_99_archive_restore_purge(manager, tmp_path):
    store = manager.store
    # seed state and memory for rex
    rex_spec = manager.load("rex")
    rex_state = CompanionState.create(
        "rex", rex_spec.name, rex_spec.to_registry(),
        definition_hash=rex_spec.definition_hash())
    rex_state.checkpoint(store)
    store.save_memory("m1", "rex", "episodic", "{}", time.time())
    # seed kira state
    kira_spec = CharacterSpec.from_yaml_dict(
        yaml.safe_load((Path(__file__).parent.parent / "characters" / "kira.yaml")
                       .read_text(encoding="utf-8")), "kira")
    kira_state = CompanionState.create(
        "kira", kira_spec.name, kira_spec.to_registry(),
        definition_hash=kira_spec.definition_hash())
    kira_state.checkpoint(store)

    manager.archive("rex")
    summaries = manager.list(include_archived=True)
    rex_sum = next(s for s in summaries if s.char_id == "rex")
    assert rex_sum.archived is True

    manager.restore("rex")
    assert manager.exists("rex")

    manager.purge("rex")
    assert not manager.exists("rex")
    assert store.load_state("rex") is None
    assert store.count_memories("rex") == 0
    assert store.load_state("kira") is not None

    with pytest.raises(ValueError):
        manager.purge("nobody")


def test_100_duplicate_character(manager):
    new_id = manager.duplicate("rex", "rex-two", "Rex Two")
    assert new_id == "rex-two"
    assert manager.exists("rex-two")
    spec = manager.load("rex-two")
    assert spec.name == "Rex Two"
    assert [t.trait_id for t in spec.to_registry()] == [
        t.trait_id for t in manager.load("rex").to_registry()
    ]
    assert spec.definition_hash() != manager.load("rex").definition_hash()

    with pytest.raises(ValueError, match="already exists"):
        manager.duplicate("rex", "rex-two", "Rex Two Again")


def test_101_definition_hash_stability(tmp_path):
    spec1 = CharacterSpec(char_id="xx", name="X",
                          likes=[{"domain": "taste", "values": ["sweet"],
                                  "intensity": 0.5}])
    spec2 = CharacterSpec(char_id="xx", name="X",
                          likes=[{"domain": "taste", "values": ["sweet"],
                                  "intensity": 0.5}])
    assert spec1.definition_hash() == spec2.definition_hash()

    spec3 = CharacterSpec(char_id="xx", name="X",
                          backstory="different",
                          likes=[{"domain": "taste", "values": ["sweet"],
                                  "intensity": 0.5}])
    assert spec3.definition_hash() != spec1.definition_hash()

    spec4 = CharacterSpec(char_id="yy", name="X",
                          likes=[{"domain": "taste", "values": ["sweet"],
                                  "intensity": 0.5}])
    assert spec4.definition_hash() != spec1.definition_hash()

    # YAML comment does not affect hash
    path = tmp_path / "xx.yaml"
    path.write_text("# comment\nname: X\nlikes:\n  - {domain: taste, values: [sweet], intensity: 0.5}\n")
    spec5 = CharacterSpec.from_yaml_dict(
        yaml.safe_load(path.read_text(encoding="utf-8")), "xx")
    assert spec5.definition_hash() == spec1.definition_hash()


def test_102_templates_are_valid():
    for t in CHARACTER_TEMPLATES:
        spec = template_spec(t, "test-id", "Test")
        assert spec.char_id == "test-id"
        assert spec.name == "Test"
    sunny = template_spec("sunny_friend", "test-id", "Test")
    assert sunny.voice_baseline.temperature == pytest.approx(0.8)
    assert spec_warnings(sunny) == []
    grumpy = template_spec("grumpy_mentor", "test-id", "Test")
    assert spec_warnings(grumpy) == []

    with pytest.raises(ValueError):
        template_spec("nope", "test-id", "Test")
