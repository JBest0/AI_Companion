import yaml

from .models import AffectState, Stimulus, Trait, TraitCategory, Trigger, VoiceProfile, voice_delta


def likes(
    domain: str,
    *values: str,
    intensity: float = 0.5,
    trait_id: str | None = None,
    **kw,
) -> Trait:
    assert intensity > 0
    tid = trait_id if trait_id is not None else f"likes_{domain}_{'_'.join(values)}"
    return Trait(
        trait_id=tid,
        triggers=[Trigger(domain=domain, values=list(values))],
        base_intensity=intensity,
        current_intensity=intensity,
        **kw,
    )


def dislikes(
    domain: str,
    *values: str,
    intensity: float = -0.5,
    trait_id: str | None = None,
    **kw,
) -> Trait:
    assert intensity < 0
    tid = trait_id if trait_id is not None else f"dislikes_{domain}_{'_'.join(values)}"
    return Trait(
        trait_id=tid,
        triggers=[Trigger(domain=domain, values=list(values))],
        base_intensity=intensity,
        current_intensity=intensity,
        **kw,
    )


class TraitRegistry:
    def __init__(self, traits: list[Trait] | None = None):
        self._traits: list[Trait] = []
        if traits:
            for t in traits:
                self.add(t)

    def add(self, trait: Trait):
        if self.get(trait.trait_id) is not None:
            raise ValueError(f"duplicate trait_id: {trait.trait_id}")
        self._traits.append(trait)

    def all(self) -> list[Trait]:
        return list(self._traits)

    def get(self, trait_id: str) -> Trait | None:
        for t in self._traits:
            if t.trait_id == trait_id:
                return t
        return None

    def __len__(self) -> int:
        return len(self._traits)

    def matching(self, stimuli: list[Stimulus]) -> list[tuple[Trait, float]]:
        out = []
        for t in self._traits:
            r = t.relevance(stimuli)
            if r > 0:
                out.append((t, r))
        return out


def load_character(path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    mood = data.get("mood_baseline")
    mood_baseline = AffectState(**mood) if mood else AffectState()

    voice = data.get("voice_baseline")
    voice_baseline = VoiceProfile(**voice) if voice else VoiceProfile()

    registry: list[Trait] = []
    for entry in data.get("likes", []) or []:
        registry.append(likes(entry["domain"], *entry["values"], intensity=entry["intensity"]))
    for entry in data.get("dislikes", []) or []:
        registry.append(dislikes(entry["domain"], *entry["values"], intensity=entry["intensity"]))
    for entry in data.get("traits", []) or []:
        t = dict(entry)
        if isinstance(t.get("voice_modifiers"), dict):
            t["voice_modifiers"] = voice_delta(**t["voice_modifiers"])
        if isinstance(t.get("category"), str):
            t["category"] = TraitCategory(t["category"])
        registry.append(Trait(**t))

    return {
        "name": data["name"],
        "mood_baseline": mood_baseline,
        "voice_baseline": voice_baseline,
        "registry": registry,
        "persona": data.get("persona", {}),
    }
