import time
import uuid

from .constraint import band_archetype, evaluate
from .dynamics import phase_notes, phase_voice_delta, update_phases, wound_amplifier
from .embeddings import Embedder
from .llm import LLM
from .memory import apply_rehearsal, episodic_memory, relative_time, retrieve
from .methods import MethodRegistry
from .models import Memory, TurnTrace, voice_delta
from .perception import perceive
from .reflection import (
    REFLECTION_MIN_NEW_MEMORIES,
    REFLECTION_SESSION_ID,
    Reflector,
    apply_proposal,
    default_reflector,
)
from .state import CompanionState
from .store import Store
from .voice import compose_voice, voice_to_prompt

SESSION_LOG_TAIL = 20
RELATIONSHIP_COUPLE = 0.2

FALLBACK_LINES = {
    "severe_negative": "*turns away, jaw tight, and says nothing*",
    "mild_dislike": "*gives a short, noncommittal shrug*",
    "neutral": "*nods slowly, distracted*",
    "warm_positive": "*smiles faintly, at a loss for words*",
    "delight": "*laughs, caught off guard, and doesn't answer*",
}


class CompanionSession:
    def __init__(
        self,
        state: CompanionState,
        store: Store,
        llm: LLM,
        embedder: Embedder,
        methods: MethodRegistry | None = None,
        reflector: Reflector | None = None,
        items=None,
    ):
        self.state = state
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.methods = methods if methods is not None else MethodRegistry()
        self.reflector = reflector if reflector is not None else default_reflector()
        self.items = items
        self._session_gap_hours = 0.0
        self._turn_count = 0
        self._session_id = ""

    def open(self) -> float:
        gap = self.state.begin_session()
        self._session_gap_hours = gap
        self._turn_count = 0
        self._session_id = uuid.uuid4().hex
        self.state.checkpoint(self.store)
        return gap

    def close(self) -> dict | None:
        self.state.end_session()
        summary = self._maybe_reflect()
        self.state.checkpoint(self.store)
        return summary

    def _load_memories(self) -> list[Memory]:
        rows = self.store.load_memories(self.state.companion_id)
        return [Memory.model_validate_json(row) for row in rows]

    def _save_memory(self, m: Memory) -> None:
        self.store.save_memory(
            m.id,
            m.companion_id,
            m.kind,
            m.model_dump_json(),
            m.created_at,
        )

    def _maybe_reflect(self) -> dict | None:
        all_memories = self._load_memories()
        episodic = [m for m in all_memories if m.kind == "episodic"]
        last = self.state.last_reflection_at
        if last is None:
            new_memories = episodic
        else:
            new_memories = [m for m in episodic if m.created_at > last]
        if len(new_memories) < REFLECTION_MIN_NEW_MEMORIES:
            return None

        now = time.time()
        proposal = self.reflector.reflect(self.state, episodic, now)
        applied = apply_proposal(self.state, self.store, self.embedder, proposal, episodic, now)
        self.state.last_reflection_at = now

        has_any = bool(applied.insight_memory_ids or applied.drifts or applied.narrative_added)
        if has_any:
            self.store.save_reflection(
                uuid.uuid4().hex,
                self.state.companion_id,
                applied.model_dump_json(),
            )
        return {
            "insights": len(applied.insight_memory_ids),
            "drifts": len(applied.drifts),
            "narrative": applied.narrative_added,
        }

    def turn(self, user_input: str) -> tuple[str, TurnTrace | None]:
        started = time.time()
        now = time.time()

        from .perception import parse_method

        method_name, method_args = parse_method(user_input)
        if method_name:
            error = self.methods.validate(method_name, method_args)
            if error:
                return error, None

        voice_before = compose_voice(self.state.voice_baseline, self.state.affect, voice_delta())
        affect_before = self.state.affect.model_copy(deep=True)
        relationship_before = self.state.relationship.model_copy(deep=True)
        phases_before = list(self.state.active_phases)

        gap = self._session_gap_hours if self._turn_count == 0 else 0.0
        perception = perceive(user_input, time_gap_hours=gap, items=self.items)
        activation = evaluate(
            perception,
            self.state.trait_registry(),
            self.state.relationship.trust,
            methods=self.methods,
            companion_name=self.state.name,
            resentment=self.state.relationship.resentment,
        )

        memories = self._load_memories()
        query_text = user_input + " " + " ".join(s.value for s in perception.stimuli)
        retrieved = retrieve(
            memories,
            self.embedder.embed(query_text),
            self.state.affect,
            now,
            self.embedder.name,
            suppress=set(activation.suppress),
        )

        retrieved_ids = {r.memory_id for r in retrieved}
        for m in memories:
            if m.id in retrieved_ids:
                apply_rehearsal(m, now)
                self._save_memory(m)

        voice_after = compose_voice(
            self.state.voice_baseline,
            self.state.affect,
            activation.voice_deltas,
            phase_delta=phase_voice_delta(phases_before),
        )

        lines = [
            f"You are {self.state.name}, a companion with a persistent personality.",
        ]
        if self.state.backstory:
            lines.append(f"BACKSTORY: {self.state.backstory}")
        if self.state.speaking_style:
            lines.append(f"SPEAKING STYLE: {self.state.speaking_style}")
        if self.state.narrative_log:
            lines.append(f"SELF-NARRATIVE: {self.state.narrative_log[-1]['text']}")
        lines.append(f"ARCHETYPE: {activation.archetype}")
        lines.append(voice_to_prompt(voice_after))
        if retrieved:
            lines.append("MEMORIES (what you remember, oldest first):")
            for r in sorted(retrieved, key=lambda x: x.created_at):
                lines.append(f"- [{relative_time(r.created_at, now)}] {r.content}")
        if activation.ambivalent:
            lines.append("You are conflicted: show BOTH sides of your reaction.")
        for c in activation.hard_constraints:
            lines.append(f"HARD CONSTRAINT: {c}")
        for n in activation.director_notes:
            lines.append(f"DIRECTOR'S NOTE (never mention this directly): {n}")
        for n in phase_notes(phases_before):
            lines.append(f"DIRECTOR'S NOTE (never mention this directly): {n}")
        system = "\n".join(lines)

        if self.state.session_log:
            history = "\n".join(
                f"{e['role']}: {e['content']}" for e in self.state.session_log[-SESSION_LOG_TAIL:]
            )
            user = history + "\nuser: " + user_input
        else:
            user = user_input

        fallback = False
        try:
            response = self.llm.generate(system=system, user=user, voice=voice_after)
        except Exception:
            response = FALLBACK_LINES[band_archetype(activation.impact)]
            fallback = True

        self.state.affect.apply_impact(activation.impact)
        if perception.method and abs(activation.impact) >= 0.3:
            spec = self.methods.get(perception.method)
            if spec is not None and self.methods.social_applies(spec, perception.method_args, self.state.name):
                amplifier = wound_amplifier(memories, now) if activation.impact < 0 else 1.0
                for dim in spec.relationship_dims:
                    delta = activation.impact * RELATIONSHIP_COUPLE * amplifier
                    if dim == "resentment":
                        delta = -delta
                    self.state.relationship.apply_delta(dim, delta)

        # phases re-evaluate AFTER the deltas: a betrayal lands next turn
        self.state.active_phases = update_phases(phases_before, self.state.relationship)

        self._save_memory(
            episodic_memory(
                self.state.companion_id,
                user_input,
                activation,
                self.embedder,
                now,
                method=perception.method,
                session_id=self._session_id,
            )
        )

        self.state.session_log.append({"role": "user", "content": user_input})
        self.state.session_log.append({"role": "companion", "content": response})
        self.state.checkpoint(self.store)
        self._turn_count += 1

        trace = TurnTrace(
            user_input=user_input,
            perception=perception,
            activation=activation,
            voice_before=voice_before,
            voice_after=voice_after,
            affect_before=affect_before,
            affect_after=self.state.affect.model_copy(deep=True),
            relationship_before=relationship_before,
            relationship_after=self.state.relationship.model_copy(deep=True),
            response=response,
            retrieved_memories=retrieved,
            active_phases=phases_before,
            fallback=fallback,
            latency_ms={"total": int((time.time() - started) * 1000)},
        )
        self.store.save_trace(trace.turn_id, self.state.companion_id, trace.model_dump_json())
        return response, trace
