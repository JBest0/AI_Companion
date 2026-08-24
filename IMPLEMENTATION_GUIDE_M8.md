# Implementation Guide — Milestone 8: Character Manager & Creator

> **For the coding agent:** every decision in this document is already made.
> Do not redesign, rename, reorder, or "improve" anything. Implement exactly
> what is specified, run the tests, match the demo contract. If a detail is
> not specified here, copy it from the existing codebase — it was decided in
> an earlier milestone.

---

## 1. What this milestone is

The engine has been multi-character all along without knowing it: every
table in `store.py` is keyed by `companion_id`, and `chat.py kira` already
parameterizes the character. What is missing is everything *around* that:
`server.py` and `gui.py` hardcode `"kira"`, nothing enumerates
`characters/*.yaml`, and creating a character means hand-writing an
intricate YAML with no validation until it explodes at load time.

M8 adds the missing layer, built on one explicit distinction:

```
definition   = characters/<id>.yaml   (name, baselines, persona, traits)
instance     = companion_state row + memories + traces + reflections
               (created from the definition ONCE, at first meeting)
```

A **definition** is edited freely. An **instance** is a lived-in companion:
its trait registry was snapshotted at creation and may since have drifted
through reflection (M4). An instance is therefore **never migrated** to an
edited definition. Instead, the instance carries a `definition_hash`; when
the file's hash differs, the UI shows a badge and selecting that character
offers exactly two choices: **keep talking** (old instance, nothing
changes) or **restart as new** (purge the instance, create fresh from the
current definition). There is no third option and there never will be —
merging a new definition into drifted state has undefined semantics.

The user-facing surface:

```
web GUI:   avatar chip strip (switch) + full creator/editor modal
Tkinter:   character combobox in the config bar (switch only)
REPL:      python chat.py            -> picker when >1 character
           python chat.py --list     -> table of characters
```

### Design locks (do not revisit)

1. **The filename is the identity.** `char_id` == YAML filename stem ==
   `companion_id` in the DB. The id is never stored inside the file (two
   sources of truth would drift). `char_id` matches
   `^[a-z][a-z0-9_-]{1,31}$`.
2. **Definitions live in YAML; the DB stores instances only.** Same rule
   as `items.yaml` (M7): content is a diffable file, state is a database.
3. **Instances are never migrated** to an edited definition (see above).
   Editing is always safe by construction: nothing about the running
   companion changes until the user explicitly chooses *restart as new*.
4. **Legacy instances never nag.** States created before M8 have
   `definition_hash == ""`; the `definition_changed` badge requires BOTH
   hashes non-empty and unequal.
5. **The pipeline is untouched.** No changes to `loop.py`,
   `constraint.py`, `voice.py`, `memory.py`, `dynamics.py`,
   `reflection.py`, `perception.py`, `items.py`, or any M1–M7 constant.
   All 93 existing tests pass unmodified; the demo contract (M7 §5) is
   unchanged, character for character.
6. **Everything stays deterministic and offline.** No LLM anywhere in the
   manager, the creator, or switching. Tests use `MockLLM` only.
7. **Transport stays boring.** `server.py` remains stdlib-only; the web
   GUI remains vanilla JS with no build step; no new Python dependencies.
8. **Switching is non-destructive and confirm-free.** `checkpoint()` runs
   after every turn, so switching away loses nothing: close the old
   session (reflection included, per M4), open the new one. Creating a
   character auto-selects it.
9. **Delete is soft by default.** Archive moves the file to
   `characters/.archive/` (a rename — undo is free, and the DB instance
   is untouched, so a restore brings the companion back with full
   memory). Permanent purge is a separate, typed-confirm action that
   removes the file AND all four tables' rows for that id.

---

## 2. New module: `companion/characters.py` (complete source)

```python
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
```

PyYAML and pydantic are already dependencies. No new packages.

Note what is deliberately NOT validated: archetype *names* beyond their
snake_case format. Band archetypes come from `band_archetype()`, but trait
archetype lists are free-form prompt strings (MockLLM renders any of them;
`characters/kira.yaml` uses `disgusted_rejection`, `cold_withdrawal`,
`trauma_flashback`, which appear in no registry). A fixed allowlist would
invent a rule the existing content does not satisfy.

---

## 3. Edits to existing files

### 3.1 `companion/traits.py`

`load_character` moves to `companion/characters.py` (its new home — every
caller imports it from the package root, so no caller changes). Delete the
function and shrink the import line to what remains in use:

```python
from .models import Stimulus, Trait, Trigger
```

(`yaml`, `AffectState`, `TraitCategory`, `VoiceProfile`, `voice_delta` and
`VersionedModel` were only used by `load_character`.) `likes`, `dislikes`
and `TraitRegistry` are untouched.

### 3.2 `companion/state.py`

One new field on `CompanionState` (default keeps pre-M8 databases
hydrating cleanly — this is the legacy-instance rule, lock 4):

```python
    definition_hash: str = ""   # M8: hash of the definition this instance
                                # was created from; "" = legacy, never badged
```

and `create()` gains a trailing keyword:

```python
    def create(cls, companion_id, name, registry, voice_baseline=None,
               affect_baseline=None, backstory="", speaking_style="",
               definition_hash: str = ""):
        ...
        return cls(
            ...
            speaking_style=speaking_style,
            definition_hash=definition_hash,
            ...)
```

Nothing else in the file changes.

### 3.3 `companion/store.py`

Four new methods, appended after `load_reflections`:

```python
    def list_state_meta(self) -> dict[str, float]:
        """companion_id -> updated_at, for every saved instance."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT companion_id, updated_at FROM companion_state"
            ).fetchall()
            return {r[0]: r[1] for r in rows}

    def count_traces(self, companion_id: str) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM turn_traces WHERE companion_id = ?",
                (companion_id,)).fetchone()[0]

    def count_memories(self, companion_id: str) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM memories WHERE companion_id = ?",
                (companion_id,)).fetchone()[0]

    def purge_companion(self, companion_id: str) -> None:
        """Every row for this id, all four tables. Used by restart-as-new
        and by permanent purge. Irreversible."""
        with self._conn() as conn:
            for table in ("companion_state", "turn_traces", "memories",
                          "reflection_log"):
                conn.execute(f"DELETE FROM {table} WHERE companion_id = ?",
                             (companion_id,))
            conn.commit()
```

### 3.4 `companion/__init__.py`

Replace `load_character` in the `.traits` import (it now lives in
`.characters`) and add the new export line:

```python
from .traits import TraitRegistry, dislikes, likes
from .characters import (CHARACTER_TEMPLATES, CharacterManager,
                         CharacterSpec, CharacterSummary, Preference,
                         load_character, slugify, spec_warnings,
                         template_spec)
```

Add to `__all__` (append, do not reorder existing entries):
`"Preference"`, `"CharacterSpec"`, `"CharacterSummary"`,
`"CharacterManager"`, `"CHARACTER_TEMPLATES"`, `"template_spec"`,
`"spec_warnings"`, `"slugify"`. (`"load_character"` is already listed.)

### 3.5 `server.py`

This is the bulk of the milestone: the server stops owning one hardcoded
companion and becomes a multi-character shell. Transport only — all
personality logic stays in the package.

**(a) Imports and constants.** Add `import re` and `import threading` to
the stdlib imports. Change the `from companion import ...` block: remove
`load_character`, add `CharacterManager, CharacterSpec, slugify,
spec_warnings`. Add a second import line:

```python
from companion.characters import CHARACTER_TEMPLATES, SLUG_RE
```

Bump `MAX_BODY = 16_384` to `MAX_BODY = 65_536` (a creator payload with a
full backstory and several traits can exceed 16 KB; this is transport,
not a personality constant).

**(b) Globals.** Keep `_session`. Add:

```python
_active_id: str | None = None
_manager: CharacterManager | None = None
_switch_lock = threading.Lock()
```

**(c) Config gains `active_character`.** `_default_config()` returns
`{"provider": "mock", "model": "", "base_url": "", "active_character": "kira"}`.
`_normalize_config` gains a fallback parameter and normalizes the new key:

```python
def _normalize_config(cfg: dict, active_fallback: str = "kira") -> dict:
    provider = cfg.get("provider", "mock")
    model = (cfg.get("model") or "").strip()
    base_url = (cfg.get("base_url") or "").strip()
    if provider == "mock":
        model, base_url = "", ""
    elif provider in PROVIDER_DEFAULTS and not model:
        model = PROVIDER_DEFAULTS[provider]
    active = (cfg.get("active_character") or "").strip() or active_fallback
    if not SLUG_RE.match(active):
        active = active_fallback
    return {"provider": provider, "model": model, "base_url": base_url,
            "active_character": active}
```

`save_config` preserves the current active character when the caller only
sent provider fields (the existing `/api/config` POST does exactly that):

```python
def save_config(cfg: dict) -> dict:
    global _config
    _config = _normalize_config(
        cfg, active_fallback=load_config().get("active_character", "kira"))
    CONFIG_FILE.write_text(json.dumps(_config, indent=2), "utf-8")
    return _config
```

`config_payload()` is NOT changed (test 79 pins its shape; the active
character is exposed via `/api/characters`).

**(d) Manager and active-character accessors:**

```python
def get_manager() -> CharacterManager:
    global _manager
    if _manager is None:
        _manager = CharacterManager(ROOT / "characters", Store(DB))
    return _manager


def active_character() -> str:
    global _active_id
    if _active_id is None:
        _active_id = load_config()["active_character"]
    return _active_id
```

**(e) `get_session()` rewrite** — the hardcoded `"kira"` becomes the
active character, and creation goes through the validated spec:

```python
def get_session() -> CompanionSession:
    global _session
    if _session is None:
        char_id = active_character()
        store = Store(DB)
        state = CompanionState.hydrate(char_id, store)
        if state is None:
            spec = get_manager().load(char_id)
            state = CompanionState.create(
                char_id, spec.name, spec.to_registry(),
                voice_baseline=spec.voice_baseline,
                affect_baseline=spec.mood_baseline,
                backstory=spec.backstory,
                speaking_style=spec.speaking_style,
                definition_hash=spec.definition_hash())
        _session = CompanionSession(state, store, build_llm(load_config())[0],
                                    default_embedder(),
                                    items=load_items(ROOT / "items.yaml"))
    return _session
```

**(f) The switch, with the M4 close semantics intact:**

```python
def select_character(char_id: str, restart: bool = False) -> dict:
    """Close the current companion (reflection included), open char_id.

    restart=True purges char_id's saved instance first — the ONLY way an
    edited definition takes effect ('restart as new', design lock 3).
    """
    global _session, _active_id
    with _switch_lock:
        if not get_manager().exists(char_id):
            raise KeyError(char_id)
        if _session is not None and _active_id == char_id and not restart:
            return {"session": _session, "gap": 0.0, "reflection": None}
        reflection = None
        if _session is not None:
            reflection = _session.close()   # M4: maybe-reflect + checkpoint
            _session = None
        if restart:
            Store(DB).purge_companion(char_id)
        _active_id = char_id
        save_config({**load_config(), "active_character": char_id})
        s = get_session()
        gap = s.open()
        return {"session": s, "gap": gap, "reflection": reflection}
```

**(g) Character payloads:**

```python
def _stored_definition_hash(store: Store, char_id: str) -> str:
    raw = store.load_state(char_id)
    if raw is None:
        return ""
    try:
        return json.loads(raw).get("definition_hash", "") or ""
    except ValueError:
        return ""


def characters_payload() -> dict:
    manager = get_manager()
    store = Store(DB)
    active = active_character()
    chars = []
    for s in manager.list(include_archived=True):
        stored = _stored_definition_hash(store, s.char_id) if s.has_save else ""
        chars.append({
            "char_id": s.char_id, "name": s.name, "avatar": s.avatar,
            "archived": s.archived, "valid": s.valid,
            "load_error": s.load_error,
            "has_save": s.has_save, "last_active": s.last_active,
            "is_active": (not s.archived) and s.char_id == active,
            # legacy instances (stored hash "") are never badged (lock 4)
            "definition_changed": bool(s.definition_hash and stored
                                       and s.definition_hash != stored),
        })
    return {"characters": chars, "active": active}
```

**(h) Routing.** In `do_GET`, after the `/api/config` branch and before
the final 404:

```python
        if self.path == "/api/characters":
            return self._send_json(characters_payload())
        if self.path == "/api/characters/templates":
            return self._send_json({"templates": CHARACTER_TEMPLATES})
        m = re.fullmatch(r"/api/characters/([a-z][a-z0-9_-]{1,31})", self.path)
        if m:
            try:
                spec = get_manager().load(m.group(1))
            except ValueError as e:
                return self._send_json({"error": str(e)}, 404)
            store = Store(DB)
            return self._send_json({
                "spec": json.loads(spec.model_dump_json()),
                "warnings": spec_warnings(
                    spec, items=load_items(ROOT / "items.yaml")),
                "has_save": store.load_state(m.group(1)) is not None,
                "definition_changed": bool(
                    spec.definition_hash()
                    and _stored_definition_hash(store, m.group(1))
                    and spec.definition_hash()
                        != _stored_definition_hash(store, m.group(1))),
            })
```

In `do_POST`, before the `/api/turn` fall-through (i.e. before
`if self.path != "/api/turn"`), add:

```python
        if self.path == "/api/characters" or self.path.startswith("/api/characters/"):
            return self._post_character()
```

**(i) `Handler._post_character()`** — complete:

```python
    def _read_body(self) -> dict | None:
        """Parsed JSON body, or None after sending the error response."""
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._send_json({"error": "input too long"}, 413)
            return None
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, UnicodeDecodeError):
            self._send_json({"error": "bad json"}, 400)
            return None

    def _validation_error(self, e) -> None:
        fields = [{"loc": ".".join(str(x) for x in err["loc"]),
                   "msg": err["msg"]} for err in e.errors()]
        self._send_json({"error": "validation", "fields": fields}, 422)

    def _post_character(self) -> None:
        from pydantic import ValidationError

        body = self._read_body()
        if body is None:
            return
        manager = get_manager()
        rest = self.path[len("/api/characters"):].strip("/")
        parts = rest.split("/") if rest else []

        # POST /api/characters — create; the new character is selected
        # immediately (design lock 8).
        if not parts:
            try:
                spec = CharacterSpec.model_validate(body)
            except ValidationError as e:
                return self._validation_error(e)
            try:
                char_id = manager.create(spec)
            except ValueError as e:
                status = 409 if "already exists" in str(e) else 400
                return self._send_json({"error": str(e)}, status)
            sel = select_character(char_id)
            return self._send_json({"ok": True, "char_id": char_id,
                                    "state": state_snapshot(sel["session"])})

        char_id = parts[0]

        # POST /api/characters/<id> — update the definition. The running
        # instance is untouched (lock 3); the badge appears on next list.
        if len(parts) == 1:
            try:
                spec = CharacterSpec.model_validate(body)
            except ValidationError as e:
                return self._validation_error(e)
            if spec.char_id != char_id:
                return self._send_json(
                    {"error": "spec.char_id must match the URL"}, 400)
            try:
                manager.update(char_id, spec)
            except ValueError as e:
                return self._send_json({"error": str(e)}, 404)
            return self._send_json({"ok": True})

        action = parts[1]

        if action == "select":
            try:
                sel = select_character(char_id,
                                       restart=bool(body.get("restart")))
            except KeyError:
                return self._send_json(
                    {"error": f"no character named {char_id!r}"}, 404)
            except ValueError as e:   # file exists but fails validation
                return self._send_json({"error": str(e)}, 400)
            return self._send_json({
                "ok": True, "gap": sel["gap"],
                "reflection": sel["reflection"],
                "state": state_snapshot(sel["session"])})

        if action == "archive":
            if char_id == active_character():
                return self._send_json(
                    {"error": "cannot archive the active character — "
                              "switch to someone else first"}, 400)
            try:
                manager.archive(char_id)
            except ValueError as e:
                return self._send_json({"error": str(e)}, 404)
            return self._send_json({"ok": True})

        if action == "restore":
            try:
                manager.restore(char_id)
            except ValueError as e:
                return self._send_json({"error": str(e)}, 400)
            return self._send_json({"ok": True})

        if action == "duplicate":
            new_id = slugify(str(body.get("name", ""))) or str(
                body.get("char_id", ""))
            try:
                created = manager.duplicate(
                    char_id, new_id, str(body.get("name", "")).strip())
            except ValueError as e:
                status = 409 if "already exists" in str(e) else 400
                return self._send_json({"error": str(e)}, status)
            return self._send_json({"ok": True, "char_id": created})

        if action == "purge":
            if char_id == active_character():
                return self._send_json(
                    {"error": "cannot purge the active character — "
                              "switch to someone else first"}, 400)
            if body.get("confirm") != char_id:
                return self._send_json(
                    {"error": "purge requires confirm == char_id"}, 400)
            try:
                manager.purge(char_id)
            except ValueError as e:
                return self._send_json({"error": str(e)}, 404)
            return self._send_json({"ok": True})

        return self._send_json({"error": "not found"}, 404)
```

**(j) Turns and switching are mutually exclusive.** In the `/api/turn`
branch, wrap session use:

```python
        with _switch_lock:
            s = get_session()
            response, trace = s.turn(user_input)
```

(The response payload below it is unchanged.)

**(k) `main()`** — follows the configured active character and closes
whichever session is active on shutdown:

```python
def main() -> None:
    s = get_session()
    gap = s.open()
    print(f"{s.state.name} is awake (gap since last session: {gap:.2f}h)")
    cfg = load_config()
    print(f"LLM provider: {cfg['provider']}"
          + (f" ({cfg['model']})" if cfg["model"] else ""))
    print(f"GUI: http://{HOST}:{PORT}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if _session is not None:
            summary = _session.close()
            if summary:
                print(f"reflection on close: {summary}")
        print("session closed.")
```

`state_snapshot`, `memories_payload`, and every pre-existing endpoint are
unchanged — they all read through `get_session()`, so they follow the
active character automatically.

### 3.6 `chat.py`

Add `CharacterManager` to the `from companion import ...` list. In
`load_session`, pass the hash through on creation:

```python
        state = CompanionState.create(
            companion_id=character,
            name=char["name"],
            registry=char["registry"],
            voice_baseline=char["voice_baseline"],
            affect_baseline=char["mood_baseline"],
            backstory=char.get("persona", {}).get("backstory", ""),
            speaking_style=char.get("persona", {}).get("speaking_style", ""),
            definition_hash=char["definition_hash"],   # M8
        )
```

Replace `main()` with:

```python
def main(argv=None):
    load_dotenv()
    argv = argv if argv is not None else sys.argv[1:]
    db_path = "./companion.db"
    if "--db" in argv:
        idx = argv.index("--db")
        if idx + 1 < len(argv):
            db_path = argv[idx + 1]

    manager = CharacterManager(ROOT / "characters", Store(db_path))

    if "--list" in argv:
        for c in manager.list(include_archived=False):
            status = "met" if c.has_save else "new"
            flag = "" if c.valid else f"  INVALID: {c.load_error}"
            print(f"{c.char_id:20} {c.name:20} {status}{flag}")
        return

    positional = [a for a in argv if not a.startswith("--")
                  and a != db_path]
    character = positional[0] if positional else None
    if character is None:
        chars = [c for c in manager.list(include_archived=False) if c.valid]
        if not chars:
            print("no characters found in characters/ — create one in the "
                  "web GUI (python server.py) or add a YAML file.")
            return
        if len(chars) == 1:
            character = chars[0].char_id
        else:
            for i, c in enumerate(chars, 1):
                print(f"  {i}. {c.name} ({c.char_id})")
            try:
                choice = input("talk to [1]: ").strip() or "1"
            except EOFError:
                return
            if choice.isdigit() and 1 <= int(choice) <= len(chars):
                character = chars[int(choice) - 1].char_id
            else:
                character = choice
    if not manager.exists(character):
        print(f"no character named {character!r}; "
              f"have: {', '.join(c.char_id for c in manager.list(False))}")
        return

    session = load_session(character, db_path)
    # ... everything below this line is unchanged ...
```

### 3.7 `gui.py` (Tkinter — switcher only; the creator is web-only, §7)

Delete the `CHARACTER = "kira"` global. `load_session` takes the character:

```python
def load_session(db_path: str, char_id: str) -> CompanionSession:
    """Hydrate or create the session, like chat.py but config-driven for the LLM."""
    store = Store(db_path)
    state = CompanionState.hydrate(char_id, store)
    if state is None:
        char = load_character(ROOT / "characters" / f"{char_id}.yaml")
        persona = char["persona"]
        state = CompanionState.create(
            companion_id=char_id,
            name=char["name"],
            registry=char["registry"],
            voice_baseline=char["voice_baseline"],
            affect_baseline=char["mood_baseline"],
            backstory=persona.get("backstory", ""),
            speaking_style=persona.get("speaking_style", ""),
            definition_hash=char["definition_hash"],   # M8
        )
    llm, _ = build_llm(load_config())
    return CompanionSession(state, store, llm, default_embedder())
```

In `ConfigBar.__init__`, prepend a character combobox before the Provider
label (same row, same style):

```python
        ttk.Label(self, text="Character").grid(row=0, column=8, padx=(12, 4))
        self.character = tk.StringVar()
        self.char_menu = ttk.Combobox(self, textvariable=self.character,
                                      state="readonly", width=12)
        self.char_menu.grid(row=0, column=9)
        self.char_menu.bind("<<ComboboxSelected>>",
                            lambda e: self.on_character(self.character.get()))
        self.on_character = lambda cid: None   # replaced by App
```

(`on_apply`/`_apply` etc. are unchanged.) In `App.__init__`, replace
`self.session = load_session(db_path)` with:

```python
        self.db_path = db_path
        self.active = load_config().get("active_character", "kira")
        self.manager = CharacterManager(ROOT / "characters", Store(db_path))
        self.session = load_session(db_path, self.active)
        self.switching = False
```

and after `self.config_bar.bind_session(self.session)` add:

```python
        ids = [c.char_id for c in self.manager.list(include_archived=False)
               if c.valid]
        self.config_bar.char_menu["values"] = ids
        self.config_bar.character.set(self.active)
        self.config_bar.on_character = self._switch_character
```

New method on `App`:

```python
    def _switch_character(self, char_id):
        if self.switching or char_id == self.active:
            return
        self.switching = True
        self.chat.send_btn.configure(state="disabled")

        def work():
            try:
                self.session.close()          # M4 close semantics
                self.session = load_session(self.db_path, char_id)
                self.active = char_id
                save_config({**load_config(), "active_character": char_id})
                gap = self.session.open()
                self.q.put(("switched", char_id, gap))
            except Exception as e:  # noqa: BLE001
                self.q.put(("error", str(e)))

        threading.Thread(target=work, daemon=True).start()
```

In `_poll`, handle the new queue item before the `set_ready` fall-through
(and add `self.switching = False`):

```python
                elif item[0] == "switched":
                    _, char_id, gap = item
                    self.config_bar.bind_session(self.session)
                    self.root.title(f"Companion — {self.session.state.name}")
                    self.chat.transcript.configure(state="normal")
                    self.chat.transcript.delete("1.0", "end")
                    self.chat.transcript.insert(
                        "end", f"{self.session.state.name} is awake "
                               f"(gap {gap:.2f}h). Type /help for methods.\n\n")
                    self.chat.transcript.configure(state="disabled")
                    self.switching = False
                    self._refresh_all()
```

### 3.8 `demo.py`

One line, inside `CompanionState.create(...)`:

```python
            definition_hash=char["definition_hash"],   # M8
```

`golden.py` and `reflect.py` are NOT touched.

### 3.9 Web GUI

Three files; all edits are additive except the cache-buster bump. The
config modal's `.modal`/`.modal-card`/`.modal-actions`/`.ghost`/`.warn`
styles are reused as-is.

**`web/index.html`** — bump both cache-busters to `?v=4`. Insert the
character strip as the first child of `<aside id="left">`:

```html
    <div id="char-strip" role="tablist" aria-label="Characters"></div>
```

Add an edit button inside `#left-head`, between the `<h1>` and
`#settings-btn`:

```html
      <button id="char-edit-btn" title="Edit this character"
              aria-label="Edit this character">✎</button>
```

Append two modals before the `<script>` tag — the restart choice:

```html
<div id="restart-modal" class="modal hidden">
  <div class="modal-card">
    <h2 id="restart-title">Definition changed</h2>
    <p id="restart-text" class="dim"></p>
    <div class="modal-actions">
      <button id="restart-cancel" class="ghost">Cancel</button>
      <button id="restart-keep" class="ghost">Keep talking</button>
      <button id="restart-new">Restart as new</button>
    </div>
  </div>
</div>
```

and the creator/editor (complete):

```html
<div id="char-modal" class="modal hidden">
  <div class="modal-card wide">
    <h2 id="ch-title">New character</h2>

    <div class="form-row" id="ch-template-row">
      <label for="ch-template">Start from</label>
      <select id="ch-template">
        <option value="blank">Blank</option>
        <option value="sunny_friend">Sunny friend</option>
        <option value="grumpy_mentor">Grumpy mentor</option>
      </select>
    </div>

    <div class="form-row">
      <label for="ch-name">Name</label>
      <input id="ch-name" type="text" maxlength="40" autocomplete="off"
             placeholder="Captain Mira">
    </div>
    <div class="form-row">
      <label for="ch-id">id <span class="dim">(filename, auto)</span></label>
      <input id="ch-id" type="text" autocomplete="off"
             placeholder="captain-mira">
    </div>
    <div class="form-row">
      <label for="ch-avatar">Avatar <span class="dim">(one emoji)</span></label>
      <input id="ch-avatar" type="text" maxlength="8" autocomplete="off"
             placeholder="🧗">
    </div>

    <div class="form-row col">
      <label for="ch-backstory">Backstory <span class="dim">(injected verbatim into the prompt)</span></label>
      <textarea id="ch-backstory" rows="4" maxlength="4000"></textarea>
    </div>
    <div class="form-row col">
      <label for="ch-style">Speaking style</label>
      <textarea id="ch-style" rows="2" maxlength="4000"></textarea>
    </div>

    <h3>Mood baseline</h3>
    <div class="form-row"><label>gloomy ↔ sunny</label>
      <input id="ch-valence" type="range" min="-1" max="1" step="0.1">
      <span id="ch-valence-v" class="num"></span></div>
    <div class="form-row"><label>calm ↔ excitable</label>
      <input id="ch-arousal" type="range" min="0" max="1" step="0.1">
      <span id="ch-arousal-v" class="num"></span></div>

    <h3>Voice baseline</h3>
    <div class="form-row"><label>warmth</label>
      <input id="ch-vtemp" type="range" min="0" max="1" step="0.05">
      <span id="ch-vtemp-v" class="num"></span></div>
    <div class="form-row"><label>verbosity</label>
      <input id="ch-vverb" type="range" min="-1" max="1" step="0.1">
      <span id="ch-vverb-v" class="num"></span></div>
    <div class="form-row"><label>humor</label>
      <input id="ch-vhumor" type="range" min="-1" max="1" step="0.1">
      <span id="ch-vhumor-v" class="num"></span></div>
    <div class="form-row"><label>formality</label>
      <input id="ch-vform" type="range" min="0" max="1" step="0.05">
      <span id="ch-vform-v" class="num"></span></div>
    <div class="form-row"><label>metaphor density</label>
      <input id="ch-vmeta" type="range" min="0" max="1" step="0.05">
      <span id="ch-vmeta-v" class="num"></span></div>

    <details id="ch-advanced">
      <summary>Deep personality (preferences &amp; traits)</summary>

      <h3>Likes</h3>
      <div id="ch-likes"></div>
      <button type="button" id="ch-add-like" class="ghost">+ like</button>

      <h3>Dislikes</h3>
      <div id="ch-dislikes"></div>
      <button type="button" id="ch-add-dislike" class="ghost">+ dislike</button>

      <h3>Traits</h3>
      <div id="ch-traits"></div>
      <button type="button" id="ch-add-trait" class="ghost">+ trait</button>
    </details>

    <div id="ch-warnings" class="dim"></div>
    <div id="ch-errors" class="warn hidden"></div>

    <div class="modal-actions">
      <button id="ch-cancel" class="ghost">Cancel</button>
      <button id="ch-save">Save</button>
    </div>

    <div id="ch-danger" class="hidden">
      <h3>Danger zone</h3>
      <div class="modal-actions">
        <button id="ch-archive" class="ghost">Archive</button>
        <button id="ch-duplicate" class="ghost">Duplicate</button>
      </div>
      <div class="form-row">
        <label for="ch-purge-confirm">Type the id to delete permanently</label>
        <input id="ch-purge-confirm" type="text" autocomplete="off">
        <button id="ch-purge" disabled>Delete permanently</button>
      </div>
    </div>
  </div>
</div>
```

**`web/app.js`** — append this section verbatim (it uses the existing `$`
and `escapeHtml` helpers), and add one line to the boot IIFE:
`loadCharacters();` before `reportDiag();`.

```javascript
/* ---------- characters (M8) ---------- */
const SLUG_JS = (s) => {
  const slug = s.trim().toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-").replace(/^[-_]+|[-_]+$/g, "");
  return /^[a-z][a-z0-9_-]{1,31}$/.test(slug) ? slug : "";
};
const PREF_DOMAINS = ["taste", "entity", "activity", "topic", "tag",
                      "item", "category"];
let _templates = null;          // lazy-fetched from /api/characters/templates
let _editingChar = null;        // char_id in edit mode, null when creating
let _idTouched = false;         // user overrode the auto-slug

async function loadCharacters() {
  const res = await fetch("/api/characters").then(r => r.json());
  const strip = $("#char-strip");
  strip.innerHTML = "";
  for (const c of res.characters.filter(c => !c.archived)) {
    const b = document.createElement("button");
    b.className = "chip" + (c.is_active ? " active" : "") +
                  (c.valid ? "" : " invalid");
    b.textContent = c.avatar || (c.name || "?").slice(0, 1);
    b.title = c.name +
      (c.valid ? "" : " — invalid: " + c.load_error) +
      (c.definition_changed ? " — definition changed since you met" : "");
    if (c.definition_changed) {
      const dot = document.createElement("span");
      dot.className = "badge-dot";
      b.appendChild(dot);
    }
    b.onclick = () => selectCharacter(c);
    strip.appendChild(b);
  }
  const add = document.createElement("button");
  add.className = "chip add";
  add.textContent = "+";
  add.title = "New character";
  add.onclick = () => openCreator(null);
  strip.appendChild(add);
}

function askRestart(name) {
  return new Promise((resolve) => {
    $("#restart-text").textContent =
      `${name}'s definition was edited since you met. ` +
      `Keep talking (nothing changes), or restart as new — ` +
      `a fresh companion from the updated definition; the current ` +
      `relationship, memories and history are erased.`;
    $("#restart-modal").classList.remove("hidden");
    const done = (v) => {
      $("#restart-modal").classList.add("hidden");
      resolve(v);
    };
    $("#restart-keep").onclick = () => done(false);
    $("#restart-new").onclick = () => done(true);
    $("#restart-cancel").onclick = () => done(null);
  });
}

async function selectCharacter(c) {
  if (c.is_active) return;
  let restart = false;
  if (c.definition_changed && c.has_save) {
    restart = await askRestart(c.name);
    if (restart === null) return;
  }
  const res = await fetch(`/api/characters/${c.char_id}/select`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({restart}),
  }).then(r => r.json());
  if (res.error) { window.__guiErrors.push(res.error); return; }
  location.reload();   // boot rebuilds state, history, panels
}

/* ----- creator / editor ----- */
function sliderFill(spec) {
  $("#ch-name").value = spec.name || "";
  if (!_idTouched) $("#ch-id").value = spec.char_id || "";
  $("#ch-avatar").value = spec.avatar || "";
  $("#ch-backstory").value = spec.backstory || "";
  $("#ch-style").value = spec.speaking_style || "";
  $("#ch-valence").value = spec.mood_baseline?.valence ?? 0;
  $("#ch-arousal").value = spec.mood_baseline?.arousal ?? 0.2;
  const v = spec.voice_baseline || {};
  $("#ch-vtemp").value = v.temperature ?? 0.5;
  $("#ch-vverb").value = v.verbosity ?? 0;
  $("#ch-vhumor").value = v.humor ?? 0;
  $("#ch-vform").value = v.formality ?? 0.5;
  $("#ch-vmeta").value = v.metaphor_density ?? 0.2;
  $("#ch-likes").innerHTML = "";
  $("#ch-dislikes").innerHTML = "";
  $("#ch-traits").innerHTML = "";
  for (const p of spec.likes || []) addPrefRow("#ch-likes", +1, p);
  for (const p of spec.dislikes || []) addPrefRow("#ch-dislikes", -1, p);
  for (const t of spec.traits || []) addTraitRow(t);
  // fields the form does not edit are preserved on save, keyed by trait_id
  window._origTraits = Object.fromEntries(
    (spec.traits || []).map(t => [t.trait_id, t]));
  updateRangeLabels();
}

function addPrefRow(container, sign, p) {
  const row = document.createElement("div");
  row.className = "pref-row";
  row.innerHTML =
    `<select class="pref-domain">${PREF_DOMAINS.map(d =>
       `<option ${p && p.domain === d ? "selected" : ""}>${d}</option>`).join("")}
     </select>
     <input class="pref-values" type="text" placeholder="values, comma-separated"
            value="${p ? escapeHtml(p.values.join(", ")) : ""}">
     <input class="pref-intensity" type="range" min="0.1" max="1" step="0.05"
            value="${p ? Math.abs(p.intensity) : 0.5}">
     <button type="button" class="ghost row-del">×</button>`;
  row.querySelector(".row-del").onclick = () => row.remove();
  $(container).appendChild(row);
}

function addTraitRow(t) {
  const row = document.createElement("div");
  row.className = "trait-row";
  row.innerHTML = `
    <div class="trait-grid">
      <input class="t-id" type="text" placeholder="trait_id"
             value="${t ? escapeHtml(t.trait_id) : ""}">
      <select class="t-domain">${PREF_DOMAINS.map(d =>
        `<option ${t && t.triggers?.[0]?.domain === d ? "selected" : ""}>${d}</option>`).join("")}
      </select>
      <input class="t-values" type="text" placeholder="trigger values, comma-sep"
             value="${t ? escapeHtml((t.triggers?.[0]?.values || []).join(", ")) : ""}">
      <select class="t-curve">${["linear", "steep", "threshold"].map(c =>
        `<option ${t && t.curve === c ? "selected" : ""}>${c}</option>`).join("")}
      </select>
      <select class="t-category">
        <option value="surface" ${t && t.category === "surface" ? "selected" : ""}>surface</option>
        <option value="core" ${t && t.category === "core" ? "selected" : ""}>core</option>
      </select>
      <input class="t-intensity" type="range" min="-1" max="1" step="0.05"
             value="${t ? t.base_intensity : -0.5}">
      <input class="t-archetypes" type="text" placeholder="negative archetypes, comma-sep"
             value="${t ? escapeHtml((t.archetypes_negative || []).join(", ")) : ""}">
      <input class="t-desc" type="text" placeholder="description"
             value="${t ? escapeHtml(t.description || "") : ""}">
      <button type="button" class="ghost row-del">×</button>
    </div>`;
  row.querySelector(".row-del").onclick = () => row.remove();
  $("#ch-traits").appendChild(row);
}

function updateRangeLabels() {
  for (const id of ["#ch-valence", "#ch-arousal", "#ch-vtemp", "#ch-vverb",
                    "#ch-vhumor", "#ch-vform", "#ch-vmeta"]) {
    $(id + "-v").textContent = (+$(id).value).toFixed(2);
  }
}

async function openCreator(charId) {
  if (_templates === null) {
    _templates = (await fetch("/api/characters/templates")
                  .then(r => r.json())).templates;
  }
  _editingChar = charId;
  _idTouched = false;
  $("#ch-errors").classList.add("hidden");
  $("#ch-warnings").textContent = "";
  $("#ch-template-row").style.display = charId ? "none" : "";
  $("#ch-danger").classList.toggle("hidden", !charId);
  $("#ch-title").textContent = charId ? "Edit character" : "New character";
  $("#ch-id").disabled = !!charId;
  $("#ch-purge-confirm").value = "";
  $("#ch-purge").disabled = true;

  if (charId) {
    const res = await fetch(`/api/characters/${charId}`).then(r => r.json());
    sliderFill(res.spec);
    $("#ch-id").value = charId;
    _idTouched = true;
    if (res.warnings.length) {
      $("#ch-warnings").textContent = "⚠ " + res.warnings.join(" · ");
    }
  } else {
    sliderFill({..._templates[$("#ch-template").value] || _templates.blank});
    $("#ch-id").value = "";
  }
  $("#char-modal").classList.remove("hidden");
  $("#ch-name").focus();
}

function collectSpec() {
  const num = (id) => parseFloat($(id).value);
  const prefs = (container, sign) =>
    [...$(container).querySelectorAll(".pref-row")]
      .map(r => ({
        domain: r.querySelector(".pref-domain").value,
        values: r.querySelector(".pref-values").value
          .split(",").map(s => s.trim()).filter(Boolean),
        intensity: sign * parseFloat(r.querySelector(".pref-intensity").value),
      }))
      .filter(p => p.values.length);
  const traits = [...$("#ch-traits").querySelectorAll(".trait-row")]
    .map(r => {
      const intensity = parseFloat(r.querySelector(".t-intensity").value);
      const t = {
        trait_id: r.querySelector(".t-id").value.trim(),
        category: r.querySelector(".t-category").value,
        description: r.querySelector(".t-desc").value.trim(),
        triggers: [{
          domain: r.querySelector(".t-domain").value,
          values: r.querySelector(".t-values").value
            .split(",").map(s => s.trim()).filter(Boolean),
        }],
        base_intensity: intensity,
        current_intensity: intensity,
        curve: r.querySelector(".t-curve").value,
        archetypes_negative: r.querySelector(".t-archetypes").value
          .split(",").map(s => s.trim()).filter(Boolean),
      };
      const orig = (window._origTraits || {})[t.trait_id];
      if (orig) {   // fields with no form control survive the edit
        t.voice_modifiers = orig.voice_modifiers;
        t.archetypes_positive = orig.archetypes_positive || [];
        t.salience_class = orig.salience_class || "medium";
      }
      return t;
    })
    .filter(t => t.trait_id);
  return {
    char_id: $("#ch-id").value.trim(),
    name: $("#ch-name").value.trim(),
    avatar: $("#ch-avatar").value.trim(),
    mood_baseline: {valence: num("#ch-valence"), arousal: num("#ch-arousal")},
    voice_baseline: {
      temperature: num("#ch-vtemp"), verbosity: num("#ch-vverb"),
      humor: num("#ch-vhumor"), formality: num("#ch-vform"),
      metaphor_density: num("#ch-vmeta"),
    },
    backstory: $("#ch-backstory").value,
    speaking_style: $("#ch-style").value,
    likes: prefs("#ch-likes", +1),
    dislikes: prefs("#ch-dislikes", -1),
    traits,
  };
}

function showSpecErrors(data) {
  const box = $("#ch-errors");
  const msgs = data.fields
    ? data.fields.map(f => `${f.loc}: ${f.msg}`)
    : [data.error || "unknown error"];
  box.textContent = msgs.join("\n");
  box.classList.remove("hidden");
}

async function saveCharacter() {
  const spec = collectSpec();
  const url = _editingChar ? `/api/characters/${_editingChar}`
                           : "/api/characters";
  const res = await fetch(url, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(spec),
  });
  const data = await res.json();
  if (!res.ok || data.error) { showSpecErrors(data); return; }
  $("#char-modal").classList.add("hidden");
  location.reload();   // create auto-selects server-side; edit updates the badge
}

/* ----- wiring ----- */
$("#ch-name").addEventListener("input", () => {
  if (!_idTouched) $("#ch-id").value = SLUG_JS($("#ch-name").value);
});
$("#ch-id").addEventListener("input", () => { _idTouched = true; });
$("#ch-template").addEventListener("change", () => {
  const name = $("#ch-name").value, id = $("#ch-id").value;
  sliderFill({..._templates[$("#ch-template").value]});
  $("#ch-name").value = name;
  $("#ch-id").value = id;
});
for (const id of ["#ch-valence", "#ch-arousal", "#ch-vtemp", "#ch-vverb",
                  "#ch-vhumor", "#ch-vform", "#ch-vmeta"]) {
  $(id).addEventListener("input", updateRangeLabels);
}
$("#ch-add-like").addEventListener("click", () => addPrefRow("#ch-likes", +1, null));
$("#ch-add-dislike").addEventListener("click", () => addPrefRow("#ch-dislikes", -1, null));
$("#ch-add-trait").addEventListener("click", () => addTraitRow(null));
$("#ch-save").addEventListener("click", saveCharacter);
$("#ch-cancel").addEventListener("click", () =>
  $("#char-modal").classList.add("hidden"));
$("#char-modal").addEventListener("click", (e) => {
  if (e.target === $("#char-modal")) $("#char-modal").classList.add("hidden");
});
$("#char-edit-btn").addEventListener("click", async () => {
  const res = await fetch("/api/characters").then(r => r.json());
  if (res.active) openCreator(res.active);
});

$("#ch-archive").addEventListener("click", async () => {
  const res = await fetch(`/api/characters/${_editingChar}/archive`,
                          {method: "POST"}).then(r => r.json());
  if (res.error) { showSpecErrors(res); return; }
  $("#char-modal").classList.add("hidden");
  location.reload();
});
$("#ch-duplicate").addEventListener("click", async () => {
  const res = await fetch(`/api/characters/${_editingChar}/duplicate`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({name: $("#ch-name").value.trim() + " copy"}),
  }).then(r => r.json());
  if (res.error) { showSpecErrors(res); return; }
  $("#char-modal").classList.add("hidden");
  location.reload();
});
$("#ch-purge-confirm").addEventListener("input", () => {
  $("#ch-purge").disabled =
    $("#ch-purge-confirm").value.trim() !== _editingChar;
});
$("#ch-purge").addEventListener("click", async () => {
  const res = await fetch(`/api/characters/${_editingChar}/purge`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({confirm: $("#ch-purge-confirm").value.trim()}),
  }).then(r => r.json());
  if (res.error) { showSpecErrors(res); return; }
  $("#char-modal").classList.add("hidden");
  location.reload();
});
```

**`web/style.css`** — append:

```css
/* characters (M8) */
#char-strip { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.chip { width: 36px; height: 36px; border-radius: 50%; background: #0d0f13;
        color: var(--text); border: 2px solid var(--border); font-size: 17px;
        display: flex; align-items: center; justify-content: center;
        padding: 0; position: relative; cursor: pointer; }
.chip.active { border-color: var(--accent); }
.chip.invalid { border-color: var(--bad); }
.chip.add { border-style: dashed; color: var(--dim); }
.chip .badge-dot { position: absolute; top: -2px; right: -2px; width: 9px;
                   height: 9px; border-radius: 50%; background: var(--warm); }
#char-edit-btn { background: none; color: var(--dim); padding: 6px;
                 border-radius: 6px; }
#char-edit-btn:hover { color: var(--text); background: #0d0f13; }
.modal-card.wide { max-width: 560px; max-height: 86vh; overflow-y: auto; }
.form-row { display: flex; align-items: center; gap: 8px; margin: 8px 0; }
.form-row.col { flex-direction: column; align-items: stretch; }
.form-row label { width: 150px; font-size: 12px; color: var(--dim);
                  flex-shrink: 0; }
.form-row.col label { width: auto; }
.form-row input[type="text"], .form-row textarea, .form-row select {
  flex: 1; background: #0d0f13; color: var(--text); border: 1px solid
  var(--border); border-radius: 6px; padding: 7px 9px; font: inherit; }
.form-row input[type="range"] { flex: 1; }
.pref-row { display: flex; gap: 6px; margin: 4px 0; align-items: center; }
.pref-row select { width: 110px; }
.pref-row input[type="text"] { flex: 1; }
.trait-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
              margin: 6px 0; padding: 8px; border: 1px solid var(--border);
              border-radius: 6px; }
#ch-advanced { margin-top: 14px; }
#ch-advanced summary { cursor: pointer; color: var(--dim); }
#ch-danger { margin-top: 18px; border-top: 1px solid var(--border);
             padding-top: 10px; }
#ch-purge { background: var(--bad); color: #fff; }
#ch-purge:disabled { opacity: 0.4; cursor: not-allowed; }
```

---

## 4. Demo contract

`rm -f companion.db* && python demo.py` → **must match the M7 contract (§5
of the M7 guide) exactly, character for character**: sessions 1–5
identity-proof M1–M5, session 6 exercises items & tags. M8 changes no
personality code;
the only behavioral delta is that the persisted `kira` state row now
carries a `definition_hash`. Verify it matches the file:

```bash
python -c "
import json, sqlite3
from companion import CharacterSpec
row = sqlite3.connect('companion.db').execute(
    \"SELECT data FROM companion_state WHERE companion_id='kira'\").fetchone()
stored = json.loads(row[0])['definition_hash']
import yaml
spec = CharacterSpec.from_yaml_dict(
    yaml.safe_load(open('characters/kira.yaml')), 'kira')
assert stored == spec.definition_hash(), (stored, spec.definition_hash())
print('definition_hash matches:', stored)
"
```

REPL contract:
- `python chat.py --list` prints one line per character (`char_id`, name,
  `new`/`met`).
- `python chat.py` with only `kira.yaml` present starts Kira directly —
  no picker (the pre-M8 commands in CLAUDE.md keep working unchanged).
- `python chat.py nonexistent` prints the available ids and exits without
  touching the DB.

Manual GUI smoke test (web):
1. `+` → template *Sunny friend* → name `Nova` → Save. The strip shows
   Nova's chip active; `/api/state`-driven panels show Nova at trust 0.5.
2. Talk to Nova, switch back to Kira via her chip — Kira's chat history,
   bars and memories are exactly as left (switching loses nothing).
3. Edit Kira (✎), change one word in her backstory, Save → her chip grows
   a badge dot. Click her chip → the restart modal appears → *Keep
   talking*: state unchanged, badge persists (it is honest, not pushy).
4. Create a scratch character, switch to Kira, archive the scratch one →
   chip disappears; restore is covered by tests, not the smoke test.

---

## 5. Test spec (20 new tests, 91–110)

Names `test_91_...` … `test_110_...`, in two new files. All previous
suites pass unmodified: **93 + 20 = 113 total**.

### 5.1 `tests/test_characters.py` (91–102)

Fixtures: `cdir` — a `tmp_path` characters dir containing one minimal
valid `rex.yaml` (name `Rex`, one like: entity dog +0.5); `manager` —
`CharacterManager(cdir, Store(tmp_path / "t.db"))`. Tests that need the
real Kira read `characters/kira.yaml` from the project root (same pattern
as `test_harness.py`).

| # | name | assertion |
|---|---|---|
| 91 | `test_91_slug_rules` | `slugify("Captain Mira") == "captain-mira"`; `slugify("!!!") == ""`; `CharacterSpec` rejects `Kira`, `1abc`, `"a"*33`, `-x`, `a` (single char); accepts `ab`, `k2`, `captain-mira` |
| 92 | `test_92_kira_loads_through_spec` | repo `kira.yaml` → `load_character` registry trait_ids in file order: `likes_taste_sweet, likes_activity_climbing, likes_entity_dog, dislikes_taste_salty, dislikes_topic_weather, dislikes_tag_potentially_dangerous, vegetarian, felinophobia`; `felinophobia` category `CORE`, curve `steep`, salience_class `high`; `vegetarian` voice_modifiers.temperature `−0.4`; `definition_hash` is 12 hex chars |
| 93 | `test_93_create_writes_and_roundtrips` | build a spec in code (avatar, persona, one like, one full trait) → `manager.create` → file exists → `manager.load(id).model_dump() == spec.model_dump()` |
| 94 | `test_94_duplicate_id_rejected` | second `create` with same id → ValueError `already exists`; after `archive`, `create` with the same id still → ValueError (archived files block reuse) |
| 95 | `test_95_preference_signs_enforced` | like with `intensity=-0.5` → ValidationError; dislike with `+0.5` → ValidationError; `intensity=0` → ValidationError |
| 96 | `test_96_reserved_domains_and_bad_curve` | trait trigger domain `social` → ValidationError; curve `spiky` → ValidationError; preference domain `weather` → ValidationError; trigger value `"Cat"` → ValidationError |
| 97 | `test_97_drifted_trait_rejected` | surface trait with `current_intensity != base_intensity` → ValidationError mentioning `base_intensity` (drift belongs to reflection) |
| 98 | `test_98_warnings_not_errors` | trait triggering `entity:dragon` loads fine; `spec_warnings` returns one string containing `entity:dragon`; `spec_warnings(kira_spec, items=load_items("items.yaml")) == []` — Kira's values all fire |
| 99 | `test_99_archive_restore_purge` | seed store: checkpoint a `rex` state + one memory + one `kira` state; `archive("rex")` → live file gone, `list()` shows `archived=True`; `restore("rex")` → live again; `purge("rex")` → file gone, `load_state("rex")` None, `count_memories("rex") == 0`, `load_state("kira")` intact; `purge("nobody")` → ValueError |
| 100 | `test_100_duplicate_character` | `duplicate("rex", "rex-two", "Rex Two")` → file exists; loaded spec: name `Rex Two`, same registry trait_ids as rex; hashes differ (char_id is hashed); `duplicate` onto an existing id → ValueError |
| 101 | `test_101_definition_hash_stability` | same spec built twice → same hash; changed backstory → different hash; appended YAML comment → same hash (formatting-independent); changed char_id → different hash (locked semantics) |
| 102 | `test_102_templates_are_valid` | all of `CHARACTER_TEMPLATES` validate via `template_spec(t, "test-id", "Test")`; `sunny_friend` voice temperature 0.8; `spec_warnings(...) == []` for both non-blank templates; `template_spec("nope", ...)` → ValueError |

### 5.2 `tests/test_gui_characters.py` (103–110)

Copy the `_ORIG_ENVIRON` scrub header from `tests/test_gui.py` verbatim
(importing `server` loads the developer's `.env`). Fixture `guim`:

```python
@pytest.fixture()
def guim(tmp_path):
    chars = tmp_path / "characters"
    chars.mkdir()
    shutil.copy(Path(server.__file__).parent / "characters" / "kira.yaml",
                chars / "kira.yaml")
    store = Store(tmp_path / "gui.db")
    server._manager = CharacterManager(chars, store)
    server._active_id = "kira"
    server._session = None
    orig_config_file = server.CONFIG_FILE
    server.CONFIG_FILE = tmp_path / "config.json"
    server._config = None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", store, chars
    httpd.shutdown()
    if server._session is not None:
        server._session.close()
        server._session = None
    server._manager = None
    server._active_id = None
    server.CONFIG_FILE = orig_config_file
    server._config = None
```

Helper `post_json(base, path, body) -> (status, dict)` — like
`post_turn` but returns the status code and converts `HTTPError` into
`(e.code, parsed_body)` instead of raising (422/409/400 are payloads
here, not failures).

| # | name | assertion |
|---|---|---|
| 103 | `test_103_characters_listed` | GET `/api/characters` → exactly kira: `is_active` True, `has_save` False, `definition_changed` False, `valid` True; after POST `/api/turn` `"hello"` → `has_save` True, `last_active` not None |
| 104 | `test_104_create_validates_and_auto_selects` | POST `/api/characters` valid spec (`char_id` nova) → 200, `ok`, response `state.name == "Nova"`; GET list → nova `is_active` True, kira False; `nova.yaml` exists on disk; re-POST same body → 409; POST with `name: ""` → 422 with non-empty `fields` |
| 105 | `test_105_select_isolates_histories` | as kira: POST turn `"I brought you some chocolate cake!"` (impact ≈ +0.70); create `ezra` (blank spec, name `Ezra`) — auto-selected; GET `/api/state` → name `Ezra`, trust 0.5; GET `/api/memories` → `[]`; `store.count_memories("kira") == 1` |
| 106 | `test_106_switch_closes_session` | after any turn as kira and a switch away: `json.loads(store.load_state("kira"))["last_session_end"]` is not None (the M4 close ran) |
| 107 | `test_107_restart_purges_instance` | as kira: POST turn `/insult me` → trust < 0.5; POST `/api/characters/kira/select` `{"restart": true}` → 200; GET `/api/state` → trust 0.5; `count_traces("kira") == 0`, `count_memories("kira") == 0` |
| 108 | `test_108_active_character_persisted` | after creating nova: `json.loads(server.CONFIG_FILE.read_text())["active_character"] == "nova"`; a fresh `load_config()` round-trips it |
| 109 | `test_109_archive_restore_purge_endpoints` | create `tempc` (auto-selected); select kira; archive tempc → 200, list shows `archived`; select tempc → 404; restore tempc → 200; archive tempc again → 200; purge with wrong `confirm` → 400; purge with `confirm == "tempc"` → 200, absent from list entirely; archive kira while active → 400 |
| 110 | `test_110_new_character_traits_drive_turns` | create `dogfan` with like entity dog +0.9 (auto-selected); POST `/api/turn` `/gift dog` → `error` False, impact ≈ +0.90 (abs 1e-3), archetype `delight` — the gift social valence is vacuum-suppressed by the trait (M3 rule, unchanged) |

---

## 6. Definition of done

1. `companion/characters.py` exists exactly as specified in §2.
2. The §3 edits are the only changes to existing files. In particular:
   `loop.py`, `constraint.py`, `voice.py`, `memory.py`, `dynamics.py`,
   `reflection.py`, `perception.py`, `items.py`, `golden.py`,
   `reflect.py` are byte-identical to M7.
3. `python -m pytest tests/ -q` → **113 passed**, nothing old modified.
4. `rm -f companion.db* && python demo.py` matches §4 (the M7 numbers,
   plus the `definition_hash` equality check).
5. The REPL contract in §4 holds; the four manual GUI smoke steps behave
   as described.

---

## 7. Explicit non-goals (do NOT build these)

- **No LLM-assisted drafting.** A "describe your character, AI fills the
  form" button is M-later, will only ever draft prose/baselines (never
  traits), and needs its own determinism story. Not here.
- **No instance migration.** Restart-as-new is the only path from an
  edited definition to a running companion. This is permanent, not a v1
  shortcut (see §1, lock 3).
- **No `voice_modifiers` / `archetypes_positive` / `salience_class` in the
  web form.** The trait rows expose id, trigger, intensity, curve,
  category, negative archetypes, description. The other fields are
  preserved on save (merged back by `trait_id`, §3.9) and remain
  YAML-authorable.
- **No multi-user, no per-tab active character, no auth.** The active
  character is server-global, exactly like the single session before M8.
- **No avatar images or uploads** — one emoji. No character
  import/export formats (character cards et al.).
- **No Tkinter/REPL creator.** `gui.py` gets a switcher, `chat.py` gets a
  picker and `--list`. Creation is web-GUI-only (plus hand-written YAML,
  which remains first-class).
- **No per-character item files, no new trigger domains, no span
  parsing** — M7's rules stand.
- **No changes to any M1–M7 constant or formula**, and no renaming or
  removing any existing public API (`load_character` keeps its
  package-root export; it just lives in `characters.py` now).
- **No migration of pre-M8 databases.** Legacy states hydrate with
  `definition_hash == ""` and simply never show the badge (lock 4).
- **CLAUDE.md is maintained by the maintainer**, not this milestone.

---

## 8. Anti-deviation clause

The definition/instance split, the filename-as-identity rule, the
never-migrate rule, and the auto-select-on-create behavior are what make
the feature *simple* — every alternative (DB-stored definitions, in-place
registry merges, per-tab selection) was considered and rejected for cause.
If a number in §4/§5 disagrees with your build, your build is wrong — not
the guide. Do not patch tests to match your code. If something is
genuinely impossible as specified, stop and report the conflict with a
minimal reproduction instead of improvising.
