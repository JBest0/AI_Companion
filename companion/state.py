import time

from pydantic import Field

from .models import AffectState, RelationshipVector, Trait, VersionedModel, VoiceProfile
from .traits import TraitRegistry


class CompanionState(VersionedModel):
    companion_id: str
    name: str
    backstory: str = ""
    speaking_style: str = ""
    registry: list[dict] = Field(default_factory=list)
    voice_baseline: VoiceProfile
    affect: AffectState
    affect_baseline: AffectState
    relationship: RelationshipVector = Field(default_factory=RelationshipVector)
    active_phases: list[str] = Field(default_factory=list)
    session_log: list[dict] = Field(default_factory=list)
    narrative_log: list[dict] = Field(default_factory=list)
    last_session_end: float | None = None
    last_reflection_at: float | None = None
    definition_hash: str = ""   # M8: hash of the definition this instance
                                # was created from; "" = legacy, never badged

    def trait_registry(self) -> TraitRegistry:
        return TraitRegistry([Trait.model_validate(d) for d in self.registry])

    @classmethod
    def create(
        cls,
        companion_id: str,
        name: str,
        registry: list[Trait],
        voice_baseline: VoiceProfile | None = None,
        affect_baseline: AffectState | None = None,
        backstory: str = "",
        speaking_style: str = "",
        definition_hash: str = "",
    ) -> CompanionState:
        voice_baseline = voice_baseline if voice_baseline is not None else VoiceProfile()
        affect_baseline = affect_baseline if affect_baseline is not None else AffectState()
        return cls(
            companion_id=companion_id,
            name=name,
            backstory=backstory,
            speaking_style=speaking_style,
            registry=[t.model_dump(mode="json") for t in registry],
            voice_baseline=voice_baseline,
            affect=affect_baseline.model_copy(deep=True),
            affect_baseline=affect_baseline.model_copy(deep=True),
            definition_hash=definition_hash,
        )

    def begin_session(self, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        if self.last_session_end is not None:
            gap = max(0.0, (now - self.last_session_end) / 3600.0)
            self.affect.decay_toward(self.affect_baseline, gap)
            return gap
        return 0.0

    def end_session(self, now: float | None = None):
        self.last_session_end = now if now is not None else time.time()

    def checkpoint(self, store):
        store.save_state(self.companion_id, self.model_dump_json())

    @classmethod
    def hydrate(cls, companion_id: str, store) -> CompanionState | None:
        data = store.load_state(companion_id)
        if data is None:
            return None
        return cls.model_validate_json(data)
