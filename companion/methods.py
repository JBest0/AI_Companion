from .models import Stimulus, VersionedModel

SELF_TOKENS = {"me", "you", "yourself", "u"}


class MethodSpec(VersionedModel):
    name: str
    min_args: int = 0
    max_args: int = 0
    social_valence: float = 0.0
    targeted: bool = False
    relationship_dims: list[str] = ["trust"]


DEFAULT_METHODS = [
    MethodSpec(name="gift", min_args=1, max_args=10, social_valence=0.3),
    MethodSpec(name="hug", min_args=0, max_args=0, social_valence=0.6, relationship_dims=["trust", "intimacy"]),
    MethodSpec(name="insult", min_args=1, max_args=1, social_valence=-0.8, targeted=True, relationship_dims=["trust", "resentment"]),
    MethodSpec(name="praise", min_args=1, max_args=1, social_valence=0.4, targeted=True, relationship_dims=["trust", "playfulness"]),
    MethodSpec(name="comfort", min_args=0, max_args=0, social_valence=0.5),
    MethodSpec(name="challenge", min_args=1, max_args=50, social_valence=-0.2),
    MethodSpec(name="leave", min_args=0, max_args=0, social_valence=-0.4),
]


class MethodRegistry:
    def __init__(self, specs: list[MethodSpec] | None = None):
        self._specs: dict[str, MethodSpec] = {}
        for spec in specs if specs is not None else DEFAULT_METHODS:
            self._specs[spec.name] = spec

    def get(self, name: str) -> MethodSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return sorted(self._specs.keys())

    def validate(self, name: str, args: list[str]) -> str | None:
        spec = self.get(name)
        if spec is None:
            available = ", ".join(f"/{n}" for n in self.names())
            return f"Unknown method '/{name}'. Available: {available}."
        n = len(args)
        if n < spec.min_args or n > spec.max_args:
            if spec.min_args == spec.max_args:
                return f"'/{name}' expects {spec.min_args} argument(s), got {n}."
            return f"'/{name}' expects {spec.min_args}-{spec.max_args} argument(s), got {n}."
        return None

    def social_applies(self, spec: MethodSpec, args: list[str], companion_name: str) -> bool:
        if not spec.targeted:
            return True
        if not args:
            return False
        target = args[0].lower()
        return target in SELF_TOKENS or target == companion_name.lower()


def outcome_for(impact: float) -> str:
    if impact > 0.3:
        return "accepted"
    if impact < -0.3:
        return "rejected"
    return "acknowledged"
