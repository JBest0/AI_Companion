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
