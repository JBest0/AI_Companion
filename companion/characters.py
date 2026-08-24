"""Character definitions (Milestone 8): the character manager.

A character *definition* is a YAML file in characters/ — name, baselines,
persona, preferences, traits. A character *instance* is the CompanionState
row (plus memories, traces, reflections) created from that definition on
first meeting. Definitions are edited freely; instances are NEVER migrated
to an edited definition — a mismatch surfaces as `definition_changed` and
the user chooses: keep talking, or restart as new.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml
from pydantic import Field, field_validator, model_validator

from .methods import DEFAULT_METHODS
from .models import (AffectState, Trait, TraitCategory, VersionedModel,
                     VoiceProfile, voice_delta)
from .perception import DEFAULT_LEXICON
from .traits import dislikes, likes

# ---- naming & format rules ----------------------------------------------
SLUG_RE = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")   # char_id == filename stem
TRAIT_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
VALUE_RE = re.compile(r"^(\*|[a-z][a-z0-9_]*)$")   # trigger/preference values
NAME_MAX = 40
PROSE_MAX = 4000          # backstory / speaking_style, chars each
AVATAR_MAX = 8            # codepoints; one emoji

ARCHIVE_DIR = ".archive"

# Domains a character file may author triggers/preferences against.
# "social" is reserved: evaluate() adds it internally for method valence
# (M3) — an authored trait on it would double-count the method.
AUTHORABLE_DOMAINS = frozenset({
    "taste", "entity", "activity", "topic",
    "tag", "item", "category", "action", "*",
})
RESERVED_DOMAINS = frozenset({"social"})
CURVES = frozenset({"linear", "steep", "threshold"})
SALIENCE_CLASSES = frozenset({"low", "medium", "high"})


def slugify(name: str) -> str:
    """'Captain Mira' -> 'captain-mira'. '' when nothing usable remains."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower()).strip("-_")
    return slug if SLUG_RE.match(slug) else ""


class Preference(VersionedModel):
    """The likes:/dislikes: shorthand entries in a character file."""

    domain: str
    values: list[str]
    intensity: float

    @field_validator("domain")
    @classmethod
    def _domain_authorable(cls, v: str) -> str:
        if v in RESERVED_DOMAINS or v not in AUTHORABLE_DOMAINS:
            raise ValueError(f"unknown or reserved domain: {v!r}")
        return v

    @field_validator("values")
    @classmethod
    def _values_wellformed(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("values must be non-empty")
        for val in v:
            if not VALUE_RE.match(val):
                raise ValueError(
                    f"value {val!r} must be lowercase snake_case or '*'; "
                    "trigger matching is exact, anything else can never fire")
        return v

    @field_validator("intensity")
    @classmethod
    def _intensity_bounds(cls, v: float) -> float:
        if v == 0.0 or not (-1.0 <= v <= 1.0):
            raise ValueError("intensity must be non-zero within [-1.0, 1.0]")
        return v


def _trait_from_dict(raw: dict) -> Trait:
    """Exactly the conversions the pre-M8 load_character applied."""
    t = dict(raw)
    if isinstance(t.get("voice_modifiers"), dict):
        t["voice_modifiers"] = voice_delta(**t["voice_modifiers"])
    if isinstance(t.get("category"), str):
        t["category"] = TraitCategory(t["category"])
    return Trait(**t)


def _pref_dict(p: Preference) -> dict:
    return {"domain": p.domain, "values": list(p.values),
            "intensity": p.intensity}


def _trait_dict(t: Trait) -> dict:
    """Clean YAML shape: required fields always, defaults omitted."""
    d: dict = {"trait_id": t.trait_id, "category": t.category.value}
    if t.description:
        d["description"] = t.description
    d["triggers"] = [{"domain": tr.domain, "values": list(tr.values)}
                     for tr in t.triggers]
    d["base_intensity"] = t.base_intensity
    d["current_intensity"] = t.current_intensity
    if t.curve != "linear":
        d["curve"] = t.curve
    if t.archetypes_positive:
        d["archetypes_positive"] = list(t.archetypes_positive)
    if t.archetypes_negative:
        d["archetypes_negative"] = list(t.archetypes_negative)
    vm = {k: v for k, v in t.voice_modifiers.model_dump().items()
          if k != "schema_version" and v != 0.0}
    if vm:
        d["voice_modifiers"] = vm
    if t.salience_class != "medium":
        d["salience_class"] = t.salience_class
    return d


class CharacterSpec(VersionedModel):
    """A character definition: everything a fresh instance is created from."""

    char_id: str
    name: str
    avatar: str = ""
    mood_baseline: AffectState = Field(default_factory=AffectState)
    voice_baseline: VoiceProfile = Field(default_factory=VoiceProfile)
    backstory: str = ""
    speaking_style: str = ""
    likes: list[Preference] = Field(default_factory=list)
    dislikes: list[Preference] = Field(default_factory=list)
    traits: list[Trait] = Field(default_factory=list)

    @field_validator("char_id")
    @classmethod
    def _id_is_slug(cls, v: str) -> str:
        if not SLUG_RE.match(v):
            raise ValueError(
                f"char_id {v!r} must match {SLUG_RE.pattern} "
                "(it becomes the YAML filename and the companion_id)")
        return v

    @field_validator("name")
    @classmethod
    def _name_ok(cls, v: str) -> str:
        v = v.strip()
        if not (1 <= len(v) <= NAME_MAX):
            raise ValueError(f"name must be 1..{NAME_MAX} characters")
        return v

    @field_validator("avatar")
    @classmethod
    def _avatar_ok(cls, v: str) -> str:
        if v and (len(v) > AVATAR_MAX or any(c.isspace() for c in v)):
            raise ValueError("avatar must be a single emoji (or empty)")
        return v

    @field_validator("backstory", "speaking_style")
    @classmethod
    def _prose_bounded(cls, v: str) -> str:
        if len(v) > PROSE_MAX:
            raise ValueError(f"persona prose is capped at {PROSE_MAX} chars")
        return v

    @field_validator("mood_baseline")
    @classmethod
    def _mood_bounds(cls, v: AffectState) -> AffectState:
        if not (-1.0 <= v.valence <= 1.0):
            raise ValueError("mood valence must be within [-1.0, 1.0]")
        if not (0.0 <= v.arousal <= 1.0):
            raise ValueError("mood arousal must be within [0.0, 1.0]")
        return v

    @field_validator("voice_baseline")
    @classmethod
    def _voice_bounds(cls, v: VoiceProfile) -> VoiceProfile:
        for f in ("temperature", "formality", "metaphor_density"):
            if not (0.0 <= getattr(v, f) <= 1.0):
                raise ValueError(f"voice {f} must be within [0.0, 1.0]")
        for f in ("verbosity", "humor"):
            if not (-1.0 <= getattr(v, f) <= 1.0):
                raise ValueError(f"voice {f} must be within [-1.0, 1.0]")
        return v

    @model_validator(mode="after")
    def _cross_field(self) -> "CharacterSpec":
        for p in self.likes:
            if p.intensity <= 0:
                raise ValueError("likes need intensity > 0")
        for p in self.dislikes:
            if p.intensity >= 0:
                raise ValueError("dislikes need intensity < 0")
        seen: set[str] = set()
        for p, kind in [*( (p, "likes") for p in self.likes),
                        *( (p, "dislikes") for p in self.dislikes)]:
            auto = f"{kind}_{p.domain}_{'_'.join(p.values)}"
            if auto in seen:
                raise ValueError(f"duplicate generated trait_id: {auto}")
            seen.add(auto)
        for t in self.traits:
            if not TRAIT_ID_RE.match(t.trait_id):
                raise ValueError(f"bad trait_id: {t.trait_id!r}")
            if t.trait_id in seen:
                raise ValueError(f"duplicate trait_id: {t.trait_id}")
            seen.add(t.trait_id)
            if t.curve not in CURVES:
                raise ValueError(
                    f"trait {t.trait_id!r}: curve must be one of "
                    f"{sorted(CURVES)} (a typo silently becomes linear)")
            if t.salience_class not in SALIENCE_CLASSES:
                raise ValueError(
                    f"trait {t.trait_id!r}: salience_class must be one of "
                    f"{sorted(SALIENCE_CLASSES)}")
            for tr in t.triggers:
                if tr.domain in RESERVED_DOMAINS or tr.domain not in AUTHORABLE_DOMAINS:
                    raise ValueError(
                        f"trait {t.trait_id!r}: unknown or reserved trigger "
                        f"domain {tr.domain!r}")
                for val in tr.values:
                    if not VALUE_RE.match(val):
                        raise ValueError(
                            f"trait {t.trait_id!r}: bad trigger value {val!r}")
            for a in (*t.archetypes_positive, *t.archetypes_negative):
                if not TRAIT_ID_RE.match(a):
                    raise ValueError(
                        f"trait {t.trait_id!r}: archetype {a!r} must be "
                        "lowercase snake_case (it is rendered into the prompt)")
            if t.current_intensity != t.base_intensity:
                raise ValueError(
                    f"trait {t.trait_id!r}: definitions are fresh — "
                    "current_intensity must equal base_intensity "
                    "(drift is reflection's job, M4)")
        return self

    # ---- conversion ----

    def to_registry(self) -> list[Trait]:
        """likes + dislikes + authored traits, in file order — identical to
        the pre-M8 load_character registry."""
        reg = [likes(p.domain, *p.values, intensity=p.intensity)
               for p in self.likes]
        reg += [dislikes(p.domain, *p.values, intensity=p.intensity)
                for p in self.dislikes]
        reg += list(self.traits)
        return reg

    def definition_hash(self) -> str:
        """Content hash over the whole parsed spec (char_id and name
        included). Formatting and comments in the YAML do NOT affect it."""
        payload = json.dumps(self.model_dump(mode="json"),
                             sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def from_yaml_dict(cls, data: dict, char_id: str) -> "CharacterSpec":
        if not isinstance(data, dict):
            raise ValueError(f"{char_id}: character file must be a mapping")
        persona = data.get("persona") or {}
        return cls(
            char_id=char_id,
            name=data.get("name", ""),
            avatar=data.get("avatar", "") or "",
            mood_baseline=AffectState(**(data.get("mood_baseline") or {})),
            voice_baseline=VoiceProfile(**(data.get("voice_baseline") or {})),
            backstory=persona.get("backstory", "") or "",
            speaking_style=persona.get("speaking_style", "") or "",
            likes=[Preference(**e) for e in (data.get("likes") or [])],
            dislikes=[Preference(**e) for e in (data.get("dislikes") or [])],
            traits=[_trait_from_dict(t) for t in (data.get("traits") or [])],
        )

    def to_yaml_dict(self) -> dict:
        d: dict = {"name": self.name}
        if self.avatar:
            d["avatar"] = self.avatar
        d["mood_baseline"] = {"valence": self.mood_baseline.valence,
                              "arousal": self.mood_baseline.arousal}
        v = self.voice_baseline
        d["voice_baseline"] = {"temperature": v.temperature,
                               "verbosity": v.verbosity,
                               "humor": v.humor,
                               "formality": v.formality,
                               "metaphor_density": v.metaphor_density}
        if self.backstory or self.speaking_style:
            d["persona"] = {"backstory": self.backstory,
                            "speaking_style": self.speaking_style}
        if self.likes:
            d["likes"] = [_pref_dict(p) for p in self.likes]
        if self.dislikes:
            d["dislikes"] = [_pref_dict(p) for p in self.dislikes]
        if self.traits:
            d["traits"] = [_trait_dict(t) for t in self.traits]
        return d


def load_character(path) -> dict:
    """character YAML -> the dict session factories have always consumed,
    plus a new 'definition_hash' key (M8). Callers are unchanged."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec = CharacterSpec.from_yaml_dict(data or {}, char_id=path.stem)
    return {
        "name": spec.name,
        "mood_baseline": spec.mood_baseline,
        "voice_baseline": spec.voice_baseline,
        "registry": spec.to_registry(),
        "persona": {"backstory": spec.backstory,
                    "speaking_style": spec.speaking_style},
        "definition_hash": spec.definition_hash(),
    }


def spec_warnings(spec: CharacterSpec, lexicon=None, items=None,
                  methods=None) -> list[str]:
    """Non-blocking authoring hints: values that can never fire because
    nothing produces the matching stimulus. Warnings, never errors."""
    lex = DEFAULT_LEXICON if lexicon is None else lexicon
    method_names = {m.name for m in
                    (DEFAULT_METHODS if methods is None else methods)}
    lex_values: dict[str, set[str]] = {}
    for s in lex.values():
        lex_values.setdefault(s.domain, set()).add(s.value)
    tag_values = item_ids = categories = None
    if items is not None:
        tag_values = {t for it in items.all() for t in it.tags}
        item_ids = {it.item_id for it in items.all()}
        categories = {it.category for it in items.all()}

    warns: list[str] = []

    def check(owner: str, domain: str, value: str) -> None:
        if domain == "*" or value == "*":
            return
        known: set[str] | None = None
        if domain in ("taste", "entity", "activity", "topic"):
            known = lex_values.get(domain, set())
        elif domain == "tag":
            known = tag_values
        elif domain == "item":
            known = item_ids
        elif domain == "category":
            known = categories
        elif domain == "action":
            known = method_names
        if known is not None and value not in known:
            warns.append(
                f"{owner}: {domain}:{value} matches nothing — "
                "it will never fire")

    for p in (*spec.likes, *spec.dislikes):
        for v in p.values:
            check(f"{p.domain} preference", p.domain, v)
    for t in spec.traits:
        for tr in t.triggers:
            for v in tr.values:
                check(t.trait_id, tr.domain, v)
    return warns


class CharacterSummary(VersionedModel):
    char_id: str
    name: str = ""
    avatar: str = ""
    archived: bool = False
    valid: bool = True
    load_error: str = ""
    has_save: bool = False
    last_active: float | None = None
    definition_hash: str = ""


class CharacterManager:
    """CRUD over characters/*.yaml. Definitions only — instances live in
    the Store. Every method raises ValueError with a readable message;
    the HTTP layer maps those to 4xx."""

    def __init__(self, characters_dir, store=None):
        self.dir = Path(characters_dir)
        self.store = store

    def _live(self, char_id: str) -> Path:
        return self.dir / f"{char_id}.yaml"

    def _archived(self, char_id: str) -> Path:
        return self.dir / ARCHIVE_DIR / f"{char_id}.yaml"

    def exists(self, char_id: str) -> bool:
        return self._live(char_id).is_file()

    def load(self, char_id: str) -> CharacterSpec:
        path = self._live(char_id)
        if not path.is_file():
            raise ValueError(f"no character named {char_id!r}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return CharacterSpec.from_yaml_dict(data or {}, char_id)

    def list(self, include_archived: bool = True) -> list[CharacterSummary]:
        """Never raises: an invalid file still appears, flagged, so the UI
        can offer purge."""
        metas = self.store.list_state_meta() if self.store is not None else {}
        out: list[CharacterSummary] = []
        folders = [(self.dir, False)]
        if include_archived:
            folders.append((self.dir / ARCHIVE_DIR, True))
        for folder, archived in folders:
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.yaml")):
                cid = path.stem
                s = CharacterSummary(char_id=cid, archived=archived)
                try:
                    spec = CharacterSpec.from_yaml_dict(
                        yaml.safe_load(path.read_text(encoding="utf-8")) or {},
                        cid)
                    s.name = spec.name
                    s.avatar = spec.avatar
                    s.definition_hash = spec.definition_hash()
                except Exception as e:  # noqa: BLE001 - report, don't crash
                    s.valid = False
                    s.name = cid
                    s.load_error = f"{type(e).__name__}: {e}"
                if not archived and cid in metas:
                    s.has_save = True
                    s.last_active = metas[cid]
                out.append(s)
        return out

    def create(self, spec: CharacterSpec) -> str:
        if self._live(spec.char_id).exists() or self._archived(spec.char_id).exists():
            raise ValueError(f"character {spec.char_id!r} already exists")
        self._write(self._live(spec.char_id), spec)
        return spec.char_id

    def update(self, char_id: str, spec: CharacterSpec) -> None:
        if spec.char_id != char_id:
            raise ValueError(
                "spec.char_id must match the file being updated — "
                "rename via duplicate() + archive()")
        if not self._live(char_id).is_file():
            raise ValueError(f"no character named {char_id!r}")
        self._write(self._live(char_id), spec)

    def archive(self, char_id: str) -> None:
        src = self._live(char_id)
        if not src.is_file():
            raise ValueError(f"no character named {char_id!r}")
        dst = self._archived(char_id)
        dst.parent.mkdir(exist_ok=True)
        src.rename(dst)

    def restore(self, char_id: str) -> None:
        src = self._archived(char_id)
        if not src.is_file():
            raise ValueError(f"no archived character named {char_id!r}")
        if self._live(char_id).exists():
            raise ValueError(f"character {char_id!r} already exists")
        src.rename(self._live(char_id))

    def purge(self, char_id: str) -> None:
        """Final. Removes the definition file (live or archived) AND every
        DB row for the id: state, traces, memories, reflections."""
        removed = False
        for p in (self._live(char_id), self._archived(char_id)):
            if p.exists():
                p.unlink()
                removed = True
        if not removed:
            raise ValueError(f"no character named {char_id!r}")
        if self.store is not None:
            self.store.purge_companion(char_id)

    def duplicate(self, char_id: str, new_id: str, new_name: str) -> str:
        spec = self.load(char_id)
        clone = CharacterSpec.model_validate(
            spec.model_dump(mode="json")
            | {"char_id": new_id, "name": new_name})
        return self.create(clone)

    def _write(self, path: Path, spec: CharacterSpec) -> None:
        header = ("# Character definition (Milestone 8) — "
                  f"the id is the filename: {spec.char_id}\n")
        path.write_text(
            header + yaml.safe_dump(spec.to_yaml_dict(),
                                    sort_keys=False, allow_unicode=True),
            encoding="utf-8")


# ---- templates (the creator's starting points) ---------------------------
# Each is a partial character-file dict; name/char_id are filled in by the
# creator. All three must validate as CharacterSpecs (test 102).

CHARACTER_TEMPLATES: dict[str, dict] = {
    "blank": {
        "name": "",
        "mood_baseline": {"valence": 0.0, "arousal": 0.2},
        "voice_baseline": {"temperature": 0.5, "verbosity": 0.0, "humor": 0.0,
                           "formality": 0.5, "metaphor_density": 0.2},
    },
    "sunny_friend": {
        "name": "",
        "avatar": "☀️",
        "mood_baseline": {"valence": 0.4, "arousal": 0.5},
        "voice_baseline": {"temperature": 0.8, "verbosity": 0.3, "humor": 0.6,
                           "formality": 0.2, "metaphor_density": 0.3},
        "persona": {
            "backstory": ("An old friend who is genuinely happy to see you, "
                          "every single time."),
            "speaking_style": ("Warm and quick to laugh; generous with "
                               "encouragement; short exclamations."),
        },
        "likes": [
            {"domain": "taste", "values": ["sweet"], "intensity": 0.6},
            {"domain": "entity", "values": ["dog"], "intensity": 0.5},
        ],
        "dislikes": [
            {"domain": "entity", "values": ["spider"], "intensity": -0.4},
        ],
    },
    "grumpy_mentor": {
        "name": "",
        "avatar": "🦉",
        "mood_baseline": {"valence": -0.2, "arousal": 0.2},
        "voice_baseline": {"temperature": 0.3, "verbosity": -0.3, "humor": -0.1,
                           "formality": 0.7, "metaphor_density": 0.4},
        "persona": {
            "backstory": ("Decades of hard-won experience and little patience "
                          "for nonsense; secretly invested in your progress."),
            "speaking_style": ("Terse and exacting; praise is rare, dry, and "
                               "means something."),
        },
        "likes": [
            {"domain": "taste", "values": ["salty"], "intensity": 0.3},
        ],
        "dislikes": [
            {"domain": "taste", "values": ["sweet"], "intensity": -0.3},
        ],
    },
}


def template_spec(template: str, char_id: str, name: str) -> CharacterSpec:
    if template not in CHARACTER_TEMPLATES:
        raise ValueError(f"unknown template {template!r}; "
                         f"have {sorted(CHARACTER_TEMPLATES)}")
    data = dict(CHARACTER_TEMPLATES[template])
    data["name"] = name
    return CharacterSpec.from_yaml_dict(data, char_id)
