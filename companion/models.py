import time
import uuid
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class VersionedModel(BaseModel):
    schema_version: int = 1


class Stimulus(VersionedModel):
    domain: str
    value: str

    def key(self) -> str:
        return f"{self.domain}:{self.value}"


class VoiceProfile(VersionedModel):
    temperature: float = 0.5
    verbosity: float = 0.0
    humor: float = 0.0
    formality: float = 0.5
    metaphor_density: float = 0.2

    def clamped(self) -> VoiceProfile:
        return VoiceProfile(
            temperature=min(1.0, max(0.0, self.temperature)),
            verbosity=min(1.0, max(-1.0, self.verbosity)),
            humor=min(1.0, max(-1.0, self.humor)),
            formality=min(1.0, max(0.0, self.formality)),
            metaphor_density=min(1.0, max(0.0, self.metaphor_density)),
        )

    def __add__(self, other: VoiceProfile) -> VoiceProfile:
        return VoiceProfile(
            temperature=self.temperature + other.temperature,
            verbosity=self.verbosity + other.verbosity,
            humor=self.humor + other.humor,
            formality=self.formality + other.formality,
            metaphor_density=self.metaphor_density + other.metaphor_density,
        )

    def scaled(self, k: float) -> VoiceProfile:
        return VoiceProfile(
            temperature=self.temperature * k,
            verbosity=self.verbosity * k,
            humor=self.humor * k,
            formality=self.formality * k,
            metaphor_density=self.metaphor_density * k,
        )


def voice_delta(**kw: float) -> VoiceProfile:
    base = {
        "temperature": 0.0,
        "verbosity": 0.0,
        "humor": 0.0,
        "formality": 0.0,
        "metaphor_density": 0.0,
    }
    base.update(kw)
    return VoiceProfile(**base)


class TraitCategory(str, Enum):
    CORE = "core"
    SURFACE = "surface"


class Trigger(VersionedModel):
    domain: str
    values: list[str]

    def matches(self, s: Stimulus) -> bool:
        return (self.domain == "*" or self.domain == s.domain) and (
            "*" in self.values or s.value in self.values
        )


class Trait(VersionedModel):
    trait_id: str
    triggers: list[Trigger]
    base_intensity: float
    current_intensity: float
    curve: str = "linear"
    archetypes_positive: list[str] = Field(default_factory=list)
    archetypes_negative: list[str] = Field(default_factory=list)
    voice_modifiers: VoiceProfile = Field(default_factory=voice_delta)
    salience_class: str = "medium"
    category: TraitCategory = TraitCategory.SURFACE
    description: str = ""

    @model_validator(mode="after")
    def _validate(self):
        if not (-1.0 <= self.base_intensity <= 1.0 and -1.0 <= self.current_intensity <= 1.0):
            raise ValueError("intensities must be within [-1.0, 1.0]")
        if self.category == TraitCategory.CORE and self.current_intensity != self.base_intensity:
            raise ValueError("core traits are immutable: current_intensity must equal base_intensity")
        return self

    def relevance(self, stimuli: list[Stimulus]) -> float:
        return 1.0 if any(trig.matches(s) for trig in self.triggers for s in stimuli) else 0.0


class AffectState(VersionedModel):
    valence: float = 0.0
    arousal: float = 0.2

    def apply_impact(self, impact: float):
        if impact >= 0:
            self.valence += impact * 0.5 * (1.0 - self.valence)
        else:
            self.valence += impact * 0.5 * (1.0 + self.valence)
        self.arousal = min(1.0, self.arousal + abs(impact) * 0.4)
        self.valence = min(1.0, max(-1.0, self.valence))

    def decay_toward(self, baseline: AffectState, hours: float, half_life_hours: float = 12.0):
        f = 0.5 ** (hours / half_life_hours)
        self.valence = baseline.valence + (self.valence - baseline.valence) * f
        self.arousal = baseline.arousal + (self.arousal - baseline.arousal) * f


class RelationshipVector(VersionedModel):
    trust: float = 0.5
    intimacy: float = 0.1
    playfulness: float = 0.3
    resentment: float = 0.0
    safety: float = 0.5

    def apply_delta(self, dim: str, delta: float):
        delta = min(0.15, max(-0.15, delta))
        cur = getattr(self, dim)
        if delta > 0:
            delta *= 0.5 * (1.0 - cur)
        else:
            delta *= 1.0 + 0.5 * (1.0 - cur)
        setattr(self, dim, min(1.0, max(0.0, cur + delta)))


class Perception(VersionedModel):
    raw_input: str
    method: str | None = None
    method_args: list[str] = Field(default_factory=list)
    stimuli: list[Stimulus] = Field(default_factory=list)
    time_gap_hours: float = 0.0


class TraitContribution(VersionedModel):
    trait_id: str
    relevance: float
    impact: float


class Activation(VersionedModel):
    archetype: str = "neutral"
    ambivalent: bool = False
    impact: float = 0.0
    contributions: list[TraitContribution] = Field(default_factory=list)
    voice_deltas: VoiceProfile = Field(default_factory=voice_delta)
    hard_constraints: list[str] = Field(default_factory=list)
    director_notes: list[str] = Field(default_factory=list)
    suppress: list[str] = Field(default_factory=list)


class Memory(VersionedModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    companion_id: str
    kind: str = "episodic"
    content: str
    embedding: list[float] = Field(default_factory=list)
    embedder: str = "hash256"
    salience: float
    decay_rate: float
    emotional_tags: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)
    access_count: int = 0
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    session_id: str = ""


class RetrievedMemory(VersionedModel):
    memory_id: str
    content: str
    created_at: float
    score: float
    breakdown: dict[str, float] = Field(default_factory=dict)


class InsightProposal(VersionedModel):
    content: str
    source_ids: list[str]
    emotional_tags: list[str] = Field(default_factory=list)
    confidence: float = 0.6


class DriftProposal(VersionedModel):
    trait_id: str
    delta: float
    justification: str = ""
    source_ids: list[str] = Field(default_factory=list)


class ReflectionProposal(VersionedModel):
    insights: list[InsightProposal] = Field(default_factory=list)
    drifts: list[DriftProposal] = Field(default_factory=list)
    narrative_entry: str | None = None


class AppliedReflection(VersionedModel):
    """What validation let through — also the rollback record."""

    insight_memory_ids: list[str] = Field(default_factory=list)
    drifts: list[DriftProposal] = Field(default_factory=list)
    narrative_added: bool = False


class TurnTrace(VersionedModel):
    turn_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = Field(default_factory=time.time)
    user_input: str
    perception: Perception
    activation: Activation
    voice_before: VoiceProfile
    voice_after: VoiceProfile
    affect_before: AffectState
    affect_after: AffectState
    relationship_before: RelationshipVector
    relationship_after: RelationshipVector
    response: str = ""
    retrieved_memories: list[RetrievedMemory] = Field(default_factory=list)
    active_phases: list[str] = Field(default_factory=list)
    fallback: bool = False
    latency_ms: dict[str, int] = Field(default_factory=dict)
