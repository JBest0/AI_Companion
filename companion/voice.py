from .models import AffectState, VoiceProfile


def mood_voice_offset(affect: AffectState) -> VoiceProfile:
    return VoiceProfile(
        temperature=affect.valence * 0.4,
        verbosity=affect.arousal * -0.3,
        humor=affect.valence * 0.5,
        formality=0.5 + (-affect.valence * 0.3 if affect.valence < 0 else 0.0),
        metaphor_density=0.0,
    )


def compose_voice(
    baseline: VoiceProfile,
    affect: AffectState,
    trait_deltas: VoiceProfile,
    phase_delta: VoiceProfile | None = None,
) -> VoiceProfile:
    total = baseline + mood_voice_offset(affect) + trait_deltas
    if phase_delta is not None:
        total = total + phase_delta
    return total.clamped()


def voice_to_prompt(v: VoiceProfile) -> str:
    def word(value: float, low: str, high: str) -> str:
        if value < -0.33:
            return low
        if value > 0.33:
            return high
        return "moderate"

    return (
        f"Voice: temperature {word(v.temperature, 'cold, distant', 'warm')} ({v.temperature:.2f}), "
        f"verbosity {word(v.verbosity, 'very clipped', 'expansive')} ({v.verbosity:.2f}), "
        f"humor {word(v.humor, 'no humor at all', 'playful humor')} ({v.humor:.2f}), "
        f"formality {v.formality:.2f}, "
        f"metaphor_density {v.metaphor_density:.2f}."
    )
