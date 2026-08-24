import math

from .methods import MethodRegistry
from .models import Activation, Perception, Stimulus, TraitContribution, voice_delta

TRUST_BUFFER = 0.3
AMBIVALENCE_NET = 0.2
AMBIVALENCE_SIDE = 0.3
DOMINANCE = 0.6

AMBIVALENT_NOTE = (
    "You feel genuinely pulled in both directions. Let BOTH show — do not resolve the tension cleanly."
)

BAND_CONSTRAINTS = {
    "severe_negative": [
        "Do not accept, enjoy, or warm up to the stimulus.",
        "Do not use humor.",
    ],
    "mild_dislike": [
        "Show reluctance; do not pretend to enjoy it.",
    ],
}


def apply_curve(x: float, curve: str) -> float:
    if curve == "steep":
        return math.copysign(abs(x) ** 0.5, x)
    if curve == "threshold":
        return x if abs(x) >= 0.3 else 0.0
    return x


def band_archetype(net: float) -> str:
    if net > 0.7:
        return "delight"
    if net > 0.3:
        return "warm_positive"
    if net > -0.3:
        return "neutral"
    if net > -0.7:
        return "mild_dislike"
    return "severe_negative"


def evaluate(
    perception: Perception,
    registry,
    trust: float,
    methods: MethodRegistry | None = None,
    companion_name: str = "",
    resentment: float = 0.0,
) -> Activation:
    RESENTMENT_AMP = 0.3
    stimuli = list(perception.stimuli)
    social_spec = None
    if methods is not None and perception.method:
        spec = methods.get(perception.method)
        if spec is not None and methods.social_applies(spec, perception.method_args, companion_name):
            social_spec = spec
            stimuli.append(Stimulus(domain="social", value=spec.name))

    contributions: list[TraitContribution] = []
    for trait, relevance in registry.matching(stimuli):
        eff = apply_curve(trait.current_intensity * relevance, trait.curve)
        if eff < 0:
            eff *= (1.0 - TRUST_BUFFER * trust) * (1.0 + RESENTMENT_AMP * resentment)
        contributions.append(
            TraitContribution(trait_id=trait.trait_id, relevance=relevance, impact=round(eff, 4))
        )

    trait_net = sum(c.impact for c in contributions)
    if social_spec is not None and abs(trait_net) < 0.3:
        method_impact = social_spec.social_valence
        if method_impact < 0:
            method_impact *= (1.0 - TRUST_BUFFER * trust) * (1.0 + RESENTMENT_AMP * resentment)
        contributions.append(
            TraitContribution(
                trait_id=f"method:{social_spec.name}",
                relevance=1.0,
                impact=round(method_impact, 4),
            )
        )

    pos = sum(c.impact for c in contributions if c.impact > 0)
    neg = sum(c.impact for c in contributions if c.impact < 0)
    net = pos + neg

    ambivalent = (pos >= AMBIVALENCE_SIDE) and (neg <= -AMBIVALENCE_SIDE) and (abs(net) < AMBIVALENCE_NET)

    archetype = band_archetype(net)
    largest = max(contributions, key=lambda c: abs(c.impact), default=None)
    if largest is not None and abs(largest.impact) >= DOMINANCE:
        trait = registry.get(largest.trait_id)
        if trait is not None:
            if largest.impact < 0 and trait.archetypes_negative:
                archetype = trait.archetypes_negative[0]
            elif largest.impact >= 0 and trait.archetypes_positive:
                archetype = trait.archetypes_positive[0]

    deltas = voice_delta()
    for c in contributions:
        trait = registry.get(c.trait_id)
        if trait is not None:
            deltas = deltas + trait.voice_modifiers.scaled(abs(c.impact))

    hard_constraints = list(BAND_CONSTRAINTS.get(band_archetype(net), []))

    director_notes: list[str] = []
    if ambivalent:
        director_notes.append(AMBIVALENT_NOTE)

    return Activation(
        archetype=archetype,
        ambivalent=ambivalent,
        impact=round(net, 4),
        contributions=contributions,
        voice_deltas=deltas,
        hard_constraints=hard_constraints,
        director_notes=director_notes,
    )
