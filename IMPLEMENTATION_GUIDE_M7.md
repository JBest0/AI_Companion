# Implementation Guide — Milestone 7: Items & Tags (World Content Pipeline)

> **For the coding agent:** every decision in this document is already made.
> Do not redesign, rename, reorder, or "improve" anything. Implement exactly
> what is specified, run the tests, match the demo contract. If a detail is
> not specified here, copy it from the existing codebase — it was decided in
> an earlier milestone.

---

## 1. What this milestone is

Until now the companion could only react to raw trigger words: `/gift cat`
works because "cat" is literally in the lexicon. There is no concept of an
*object* with *properties*. M7 adds the content pipeline:

```
items.yaml:  steak  ->  category food, tags [luxurious, savory, meat]
kira.yaml:   vegetarian  ->  dislikes tag:meat (steep, -0.75)
result:      /gift steak  ->  disgusted_rejection, no special-case code
```

An **Item** is a named thing with a category and flat tags. Perception
expands a mentioned item into stimuli on three new domains:

| stimulus | example | purpose |
|---|---|---|
| `item:<id>` | `item:steak` | this specific thing |
| `tag:<tag>` | `tag:meat`, `tag:savory` | its properties (the workhorse) |
| `category:<category>` | `category:food` | broad classes |

Characters author traits against any of the three domains — the constraint
engine, ambivalence, phases, wounds: everything downstream is unchanged and
just works. A vegan companion and a steak never meet in code; they meet in
the stimulus sum.

### Design locks (do not revisit)

1. **The item registry is optional everywhere.** `perceive()` and
   `CompanionSession()` take `items=None`; without it, behavior is
   byte-identical to M1–M6. All 78 existing tests pass unmodified.
2. **Tags are not entities.** A `plush_cat`'s tags are
   `[soft, comforting, cute]` — never `cat`. The felinophobia trigger keys
   on `entity:cat`, so a plush toy must NOT trigger it. This distinction is
   the soul of the milestone; test 87 guards it.
3. **Item ids are single tokens** (`plush_cat`, not `plush cat`) because
   `/gift` splits args on whitespace. Aliases may be multi-word phrases for
   free-text matching.
4. **Items are world content, shared across characters** — one
   `items.yaml` at project root. Character-specific stances live in
   character YAMLs, never in the item file.
5. **Free-text mentions expand too** ("I made steak for dinner"), not only
   method args — but per the M1/M3 rule, only self-directed *methods* move
   the relationship. Mentioning steak disgusts her; it doesn't betray her.

---

## 2. New module: `companion/items.py` (complete source)

```python
"""Items & tags (Milestone 7): the world-content pipeline."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from .models import Stimulus, VersionedModel


class Item(VersionedModel):
    item_id: str                 # single token, used in /gift <item_id>
    name: str
    category: str                # must be declared in the file's categories
    tags: list[str]
    aliases: list[str] = []      # extra names (may be multi-word phrases)


class ItemRegistry:
    def __init__(self, items: list[Item] | None = None):
        self._items: dict[str, Item] = {}
        for it in items or []:
            self.add(it)

    def add(self, item: Item) -> None:
        if item.item_id in self._items:
            raise ValueError(f"duplicate item_id: {item.item_id}")
        self._items[item.item_id] = item

    def all(self) -> list[Item]:
        return list(self._items.values())

    def get(self, item_id: str) -> Item | None:
        return self._items.get(item_id)

    def _patterns(self) -> list[tuple[re.Pattern, Item]]:
        """Longest alias first, so 'plush cat' wins over 'cat'."""
        pats = []
        for it in self._items.values():
            for phrase in [it.item_id, it.name, *it.aliases]:
                pats.append((re.compile(rf"\b{re.escape(phrase.lower())}\b"), it))
        pats.sort(key=lambda p: len(p[0].pattern), reverse=True)
        return pats

    def match_text(self, text: str) -> list[Item]:
        """All items mentioned in free text or method args. Deduped."""
        lowered = text.lower()
        found: dict[str, Item] = {}
        for pat, item in self._patterns():
            if item.item_id not in found and pat.search(lowered):
                found[item.item_id] = item
        return list(found.values())


def item_stimuli(item: Item) -> list[Stimulus]:
    """The expansion: one item becomes item/tag/category stimuli."""
    return ([Stimulus(domain="item", value=item.item_id)]
            + [Stimulus(domain="tag", value=t) for t in item.tags]
            + [Stimulus(domain="category", value=item.category)])


def load_items(path: str | Path) -> ItemRegistry:
    """items.yaml -> ItemRegistry. Validates categories and duplicate ids."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    categories = set(data.get("categories", []))
    items = []
    for raw in data.get("items", []):
        item = Item(item_id=raw["id"], name=raw["name"],
                    category=raw["category"], tags=list(raw.get("tags", [])),
                    aliases=list(raw.get("aliases", [])))
        if " " in item.item_id:
            raise ValueError(f"item_id must be a single token: {item.item_id!r}")
        if categories and item.category not in categories:
            raise ValueError(
                f"item {item.item_id!r}: undeclared category {item.category!r}")
        items.append(item)
    return ItemRegistry(items)
```

PyYAML is already a dependency (M1's `load_character` uses it). No new
packages.

---

## 3. New content file: `items.yaml` (project root, complete)

```yaml
# World content (Milestone 7): items the user can mention or /gift.
# An item = id + category + flat tags. Characters react to the TAGS (and
# categories, and specific item ids), so authoring a new item needs no new
# code — a vegan companion and a steak meet in the constraint engine.
#
# Rules:
#   - id is a single token (this is what /gift takes: /gift steak)
#   - category must be declared below
#   - tags are lowercase snake_case; aliases may be multi-word phrases

categories: [food, toy, weapon]

items:
  - id: steak
    name: Steak
    category: food
    tags: [luxurious, savory, meat]

  - id: chocolate_bar
    name: Chocolate bar
    category: food
    tags: [sweet, comforting]
    aliases: [chocolate]

  - id: plush_cat
    name: Plush cat
    category: toy
    tags: [soft, comforting, cute]
    aliases: [stuffed cat]
    # NOTE: tags say nothing about real cats — a plush cat does NOT carry
    # the entity:cat stimulus, so Kira's phobia stays silent. By design.

  - id: sword
    name: Sword
    category: weapon
    tags: [iron, weapon, potentially_dangerous]

  - id: teddy_bear
    name: Teddy bear
    category: toy
    tags: [soft, comforting, childlike]
```

---

## 4. Edits to existing files

### 4.1 `companion/perception.py`

One import and one optional parameter. Nothing else changes:

```python
from .items import item_stimuli
```

```python
def perceive(raw: str, time_gap_hours: float = 0.0,
             lexicon: dict[str, Stimulus] | None = None,
             items=None) -> Perception:
    method, args = parse_method(raw)
    stimuli = extract_stimuli(raw, lexicon)
    # Items (M7): a mentioned item expands into item/tag/category stimuli.
    # Optional — without a registry this function is exactly the M1 spec.
    if items is not None:
        for item in items.match_text(raw):
            stimuli.extend(item_stimuli(item))
    # ... rest unchanged (method/action expansion, dedupe, return)
```

Note the order: item expansion happens on the **raw input** (so aliases
match natural phrasing), before the method-arg lexicon pass; the existing
dedupe by `s.key()` absorbs any overlap.

### 4.2 `companion/loop.py`

`CompanionSession.__init__` gains a trailing `items=None` parameter stored
as `self.items`, and the `perceive` call becomes:

```python
perception = perceive(user_input, time_gap_hours=gap, items=self.items)
```

### 4.3 `companion/__init__.py`

Export and add to `__all__`: `Item`, `ItemRegistry`, `item_stimuli`,
`load_items` (from `.items`).

### 4.4 `characters/kira.yaml`

Two additions. A simple-tier dislike (tag domain):

```yaml
dislikes:
  # ... existing entries unchanged ...
  # M7: tag-domain preferences react to ITEM properties (see items.yaml).
  - { domain: tag, values: [potentially_dangerous], intensity: -0.6 }
```

And one full-form trait, placed before `felinophobia` in `traits:`:

```yaml
  # The "vegan meets steak" pattern: one tag trait, any meat item triggers it.
  - trait_id: vegetarian
    category: surface
    description: "Raised above a bakery on bread, fruit and sweets; the idea of eating flesh genuinely revolts her."
    triggers:
      - { domain: tag, values: [meat] }
    base_intensity: -0.75
    current_intensity: -0.75
    curve: steep
    archetypes_negative: [disgusted_rejection, cold_withdrawal]
    voice_modifiers: { temperature: -0.4, humor: -0.5, verbosity: -0.2 }
```

These do not alter any M1–M6 contract: none of the old demo/test inputs
produce `tag:` stimuli, so nothing old fires them.

### 4.5 `chat.py` and `server.py`

Both session factories load the world file:

```python
# chat.py, in load_session(), replacing the bare CompanionSession(...) return:
items_file = ROOT / "items.yaml"
items = load_items(items_file) if items_file.exists() else None
return CompanionSession(state, store, default_llm(), default_embedder(),
                        items=items)
```

```python
# server.py, in get_session():
_session = CompanionSession(state, store, default_llm(),
                            default_embedder(),
                            items=load_items(ROOT / "items.yaml"))
```

(Add `load_items` to the `from companion import ...` list in both.)

### 4.6 `demo.py`

Sessions 1–5 keep running **without** items (that is the identity proof, on
display in every demo run). Change `get_session()` to accept
`items=None` and pass it through, then append session 6:

```python
# --- session 6: items & tags ---
session = get_session(items=load_items(ROOT / "items.yaml"))
session.open()
print("\n=== session 6 opened (items & tags) ===")
for inp in [
    "/gift steak",            # vegetarian tag-trait fires -> she is NOT pleased
    "/gift plush_cat",        # no phobia: it's a toy, not an entity:cat
    "/gift sword",            # wary of dangerous things
    "I made steak for dinner tonight.",   # free text, no method: mood only
]:
    resp, trace = session.turn(inp)
    show(inp, resp, trace)
session.close()
print("\n=== session 6 closed ===")
```

---

## 5. Demo contract (verified against the reference build)

`rm -f companion.db* && python demo.py`. **Sessions 1–5 match the M5
contract exactly, character for character** (tolerances as in M5). Session 6
continues from session 5's ending state (trust 0.41, resentment 0.07, no
active phases — breached_trust cleared on the final repair turn):

| input | archetype | impact | trust | notes |
|---|---|---|---|---|
| `/gift steak` | disgusted_rejection | −0.78 ± 0.02 | 0.41 → 0.21 | `vegetarian:-0.78`; gift social suppressed (traits rule); wound amplifier active (fresh pain memories) |
| `/gift plush_cat` | neutral | +0.30 ± 0.02 | 0.21 → 0.24 | only `method:gift:+0.30` (vacuum rule); **no felinophobia contribution**; `breached_trust` now active (entered after the steak) |
| `/gift sword` | mild_dislike | −0.57 ± 0.02 | 0.24 → 0.03 | `dislikes_tag_potentially_dangerous:-0.57` |
| `I made steak for dinner tonight.` | disgusted_rejection | −0.88 ± 0.02 | **0.03 → 0.03** | free text: mood moves, relationship does not (no method) |

The last row is a contract point, not an oversight: mentioning meat is an
insult to her values, not an act against her.

---

## 6. Test spec: `tests/test_items.py` (13 new tests, 79–90)

Names `test_79_...` … `test_90_...` (82 gets a sibling `test_82b_...`).
All previous suites pass unmodified: **78 + 13 = 91 total**.

Fixtures: an `items` fixture writing the three-item YAML (steak, plush_cat
with alias "stuffed cat", sword) to `tmp_path` and loading it; a `registry`
fixture with likes sweet +0.7, the vegetarian trait (−0.75 steep,
`archetypes_negative=["disgusted_rejection"]`), felinophobia (M1 form), and
`dislikes("tag", "potentially_dangerous", intensity=-0.6)`.

| # | name | assertion |
|---|---|---|
| 79 | `test_79_load_items_parses` | steak fields exact; plush_cat aliases `["stuffed cat"]` |
| 80 | `test_80_unknown_item_is_none` | `get("spoon")` → None |
| 81 | `test_81_item_stimuli_expansion` | exact key order: `item:steak, tag:luxurious, tag:savory, tag:meat, category:food` |
| 82 | `test_82_match_text_alias_and_boundaries` | "I got a stuffed cat!" → plush_cat; "steakhouse" → no match; steak+sword both found |
| 82b | `test_82b_load_items_validates` | undeclared category → ValueError; multi-word id → ValueError |
| 83 | `test_83_perceive_without_items_is_identity` | `/gift cat` stimuli exactly `["entity:cat", "action:gift"]` |
| 84 | `test_84_perceive_expands_items` | `/gift steak` contains item/tag/category + `action:gift` |
| 85 | `test_85_free_text_mentions_expand` | free-text steak mention, `method is None`, `tag:meat` present |
| 86 | `test_86_vegetarian_meets_steak` | impact −(0.75^0.5)×0.85 ± 1e-3 at trust 0.5; archetype disgusted_rejection; no `method:` contribution |
| 87 | `test_87_plush_cat_is_not_a_cat` | no felinophobia contribution; impact +0.30 (gift vacuum); archetype neutral. **Pass `methods=MethodRegistry(), companion_name="Kira"` — without the methods registry there is no social charge (M3 rule)** |
| 88 | `test_88_tag_conflict_is_emergent_ambivalence` | registry of `likes tag:luxurious +0.7` + `dislikes tag:meat −0.72`; steak at trust 0.0 → `ambivalent is True`, \|impact\| < 0.2 |
| 89 | `test_89_items_wire_through_the_session` | session with items: `/gift steak` → vegetarian contribution; trust < 0.5 |
| 90 | `test_90_free_text_item_never_moves_relationship` | "I made steak for dinner tonight." → impact < −0.5, trust stays 0.5 |

---

## 7. Definition of done

1. `companion/items.py` and `items.yaml` exist exactly as specified.
2. The §4 edits are the only changes to existing files.
3. `python -m pytest tests/ -q` → **91 passed**, nothing old modified.
4. `rm -f companion.db* && python demo.py` matches §5 (sessions 1–5
   unchanged, session 6 as tabled).
5. `python chat.py kira` and `server.py` both accept `/gift steak` and show
   the vegetarian reaction; the GUI trace inspector shows the
   `tag:meat`-driven contribution.

---

## 8. Explicit non-goals (do NOT build these)

- No inventory system: the companion does not "keep" gifts. (Deliberate —
  an inventory is M-later and will need its own state + decay rules.)
- No item effects over time, no consumables, no equipment.
- No span-consumption parsing (an alias match does not suppress lexicon
  hits inside the matched span — that is why plush_cat's id is a single
  token; document, don't fix).
- No tag hierarchies or ontologies (meat ⊂ food etc.) — flat tags only.
- No per-character item files, no GUI item editor, no `/give` aliases
  beyond the existing `/gift`.
- No LLM involvement in item matching — it is pure regex, deterministic.

---

## 9. Anti-deviation clause

The domain names (`item`, `tag`, `category`), the expansion order
(item → tags → category), the single-token id rule, and the optional-everywhere
wiring are what keep 78 old tests green while adding a whole content
pipeline. If a number in §5/§6 disagrees with your build, your build is
wrong — not the guide. Do not patch tests to match your code. If something
is genuinely impossible as specified, stop and report the conflict with a
minimal reproduction instead of improvising.
