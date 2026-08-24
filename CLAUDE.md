# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

**Milestones 1–8 are complete and verified**: 103 tests pass, `demo.py` output matches the contract in `IMPLEMENTATION_GUIDE_M7.md` §5, `reflect.py` behaves per `IMPLEMENTATION_GUIDE_M4.md` §6, and the M8 character manager + creator are covered by `tests/test_characters.py` and `tests/test_config.py`. The guide is the authoritative, final spec — every decision in it is fixed. Do not invent alternatives, add features, rename things, or leave TODOs. If something is unspecified, it is out of scope.

**Not a web app — permanently.** This is a local, single-process, single-user desktop application. A web app is impossible by design, not by effort: exactly one live session per process with a server-global active character (M8 lock: no multi-user, no per-tab active character, no auth), all state is local files at fixed paths, and the GUI is a native Tkinter process. The web path (`server.py`, `web/`, `refresh_server.py`) was **deleted** — there is no server to run. `gui.py` is the GUI and the character creator; `companion/config.py` is the provider/config layer. See `NO_WEB_APP.md`. This supersedes M8 §7's "creator is web-GUI-only" line.

## Commands

Run from the project root (`C:\Users\Administrator\Documents\AI_Companion`):

```bash
python -m pytest tests/ -q          # full suite (must stay at 103 passed)
python -m pytest tests/test_core.py -k test_trust_buffer -q   # single M1 test
python -m pytest tests/test_memory.py -k test_cross_session_recall -q   # single M2 test
python -m pytest tests/test_methods.py -k test_vacuum_fill -q   # single M3 test
python -m pytest tests/test_harness.py -k test_fallback_counts -q   # single M3.5 test
python -m pytest tests/test_reflection.py -k test_mock_finds_recurrence -q   # single M4 test
python -m pytest tests/test_dynamics.py -k test_wound_amplifier -q   # single M5 test
python demo.py                      # six-session demo (sessions 1-5 identity-proof M5; session 6 exercises items & tags); writes companion.db in root
python gui.py                       # Tkinter GUI + character creator; writes companion.db in root
python chat.py                      # interactive REPL; writes companion.db in root
python golden.py                    # DeepSeek golden-scenario check; needs DEEPSEEK_API_KEY
python reflect.py --log             # view applied reflection log
python reflect.py --rollback-last   # undo the latest active reflection
```

Notes:
- The environment is Python 3.14 (the guide targets 3.12; the code uses 3.12+ syntax and runs cleanly on 3.14).
- Only dependencies: `pydantic` v2, `pyyaml`, `pytest`. Nothing else (unless a later milestone adds it).
- `demo.py` creates a fresh `companion.db` on first run; delete `companion.db*` to re-test the demo from a clean state. The expected demo values only hold on a fresh database.
- API keys can be placed in a `.env` file (copy `.env.example`) or exported directly. `gui.py`, `chat.py` and `golden.py` load `.env` automatically via `companion.env.load_dotenv()` (also called by `companion/config.py` at import). Environment variables take precedence over `.env` values.
- `openai` is optional and only imported lazily inside `OpenAILLM` / `golden.py`. With no `OPENAI_API_KEY`, `default_llm()` returns `MockLLM` (deterministic — always used by tests/demo).

## Architecture

A **code-driven companion pipeline**: the LLM renders a reaction, it never decides one. All personality lives in deterministic code.

Per-turn flow (`companion/loop.py`, exact order, do not reorder):

```
user_input → validate method → perceive() → evaluate(resentment) → retrieve memories → rehearsal update
           → compose_voice(phase_delta) → build prompt (persona + self-narrative + archetype + voice + memories + constraints + phase notes)
           → llm.generate() → fallback if needed → apply affect impact → relationship update (wound-amplified if negative)
           → update phases → write episodic memory → append to session_log → checkpoint state → save TurnTrace
```

Invalid slash-commands return `(error_string, None)`: no LLM call, no state change, no memory, no trace.

Session close (`CompanionSession.close()`) ends the session, runs `_maybe_reflect()` if enough new episodic memories exist, checkpoints state, and returns a reflection summary `{"insights": n, "drifts": n, "narrative": bool}` or `None`.

The layers:

1. **World content (M7)** — `items.py` + `items.yaml`.
   - `items.py`: `Item`, `ItemRegistry`, `item_stimuli()`, `load_items()`. Items have a single-token `item_id`, a declared `category`, and flat `tags`. Mentioning an item expands to `item:`, `tag:`, and `category:` stimuli in `perceive()`.
   - `items.yaml`: shared world file (food/toy/weapon). Character-specific stances live in character YAMLs, never here.
   - Tags are not entities: a `plush_cat` is tagged `[soft, comforting, cute]` — never `cat` — so it does not trigger felinophobia.
2. **Pure, deterministic computation** — `perception.py` → `constraint.py` → `voice.py`.
   - `perception.py`: lexicon word-boundary keyword matching plus optional item expansion (no LLM classifier — that's a later milestone).
   - `constraint.py`: the "personality math". Constants `TRUST_BUFFER`, `AMBIVALENCE_NET`, `AMBIVALENCE_SIDE`, `DOMINANCE` plus the curve/archetype-band formulas are tuned as a set — **never change the constants or formulas**. If a test fails, the implementation is wrong, not the constant.
   - `voice.py`: affect → voice offset, composition, and prompt rendering.
3. **Models & state** — `models.py` (all Pydantic, all inherit `VersionedModel`) and `state.py`.
   - `Trait` splits `base_intensity` (never mutated) from `current_intensity`. Core traits are structurally immutable: a validator raises if `current_intensity != base_intensity` for `category == CORE`. Nothing mutates `current_intensity` yet (trait drift is a later milestone).
   - `CompanionState` serializes its whole registry as JSON and round-trips through SQLite.
3. **Memory** — `embeddings.py` + `memory.py`.
   - `embeddings.py`: `Embedder` protocol and the sole `HashEmbedder` (256 dims, deterministic, offline). No external embedding provider.
   - `memory.py`: episodic memory creation, decay-at-retrieval, ensemble scoring (similarity/salience/recency/resonance), and `retrieve`. Stored salience is never mutated; rehearsal only touches `access_count` and `last_accessed`.
4. **Methods** — `methods.py`.
   - `MethodSpec` / `MethodRegistry`: explicit slash-commands (`/gift`, `/hug`, `/insult`, etc.) with arg validation, social valence, and targeted/self-directed rules.
   - `evaluate()` adds a `social` stimulus when a method applies; the **vacuum rule** adds the method's `social_valence` only when matched traits have no strong opinion (`|trait_net| < 0.3`).
5. **Dynamics** — `companion/dynamics.py`.
   - `PhaseSpec` / `PHASES`: hysteretic relationship phases (`high_trust`, `breached_trust`, `high_intimacy`, `resentment`) keyed to dimension thresholds with separate enter/exit values.
   - `update_phases()` recomputes active phases after each turn's relationship deltas; `phase_voice_delta()` and `phase_notes()` inject phase-appropriate voice and director notes into the prompt.
   - `wound_amplifier()` scans recent high-salience memories tagged `pain`/`fear` and multiplies negative relationship deltas by `WOUND_AMPLIFIER` (1.5x).
6. **Reflection** — `companion/reflection.py` + `reflect.py`.
   - `companion/reflection.py`: validators and `apply_proposal()` enforce the safety bounds (drift caps, core lock, source/session floors). `MockReflector` is the offline default; `DeepSeekReflector` calls `deepseek-v4-flash` when `DEEPSEEK_API_KEY` is set.
   - `reflect.py`: audit log viewer and rollback CLI for applied reflections.
7. **Harness / interface** — `gui.py` + `chat.py` + `golden.py`.
   - `gui.py`: the only GUI — a Tkinter window (dashboard | chat | inspector) driven by a worker thread + `queue.Queue`, plus the in-window character creator (create / edit / duplicate / archive / purge) via `CharacterEditor`. Switching characters applies M8's "restart as new" semantics when a saved instance predates its edited definition.
   - `chat.py`: interactive REPL that hydrates/creates a session, prints trace summaries, and persists state.
   - `golden.py`: pre-flight check replaying the betrayal cascade against the real DeepSeek model (`deepseek-chat` via the `openai` client) on a fresh `golden.db` with companion id `golden-kira`.
8. **Persistence & LLM adapter** — `store.py` (stdlib `sqlite3`, WAL; four tables: `companion_state`, `turn_traces`, `memories`, `reflection_log`) and `llm.py` (`LLM` Protocol; `MockLLM` default; `OpenAILLM` lazily imports `openai`, model `gpt-4o-mini`). **Provider/config** lives in `companion/config.py` (was `server.py`): `load_config`/`save_config` (config.json holds provider/model/base_url/active_character), `build_llm`, `config_warning`, and `_OpenAICompatLLM`. API keys always come from the environment (`.env`), never config.json.

Key invariants:
- **Exactly one LLM call per turn** (only for valid turns), behind the `LLM` Protocol.
- The archetype is decided by `evaluate()` (band of net impact, with a dominance override for a strong trait's archetype list). `MockLLM` just parses the `ARCHETYPE:` line and renders a fixed flavor line.
- Negative trait/method impacts in `evaluate()` are amplified by `(1 - TRUST_BUFFER * trust) * (1 + 0.3 * resentment)`; positive impacts are unchanged.
- Relationship dimensions move **only on method turns** when the method is self-directed (`social_applies()` is true) AND `|impact| >= 0.3`. The affected dimensions come from `MethodSpec.relationship_dims`; `resentment` is inverted relative to the impact. If the impact is negative and any retrieved memory is a fresh (`<= 3` days), high-salience (`>= 0.7`) memory tagged `pain` or `fear`, the relationship delta is multiplied by `WOUND_AMPLIFIER` (1.5x).
- Phases are evaluated with hysteresis **after** relationship deltas are applied, using the thresholds and enter/exit rules in `companion/dynamics.py`. Active phases are recorded on the `TurnTrace` and included as director notes in the prompt.
- `Activation.voice_deltas`, `hard_constraints`, and `director_notes` are assembled in code and injected into the prompt; the LLM never supplies them.
- Persona is injected verbatim in the system prompt: `BACKSTORY` (if non-empty) and `SPEAKING STYLE` (if non-empty) appear **before** `ARCHETYPE:`. MockLLM ignores everything except the archetype line.
- LLM failures are caught and replaced with a canned in-character fallback line keyed by `band_archetype(activation.impact)`. The turn still counts: affect/relationship updates, memory write, session log, checkpoint, and trace (`fallback=True`) all proceed normally.
- Retrieval runs **before** generation and **before** the affect update, scored against the pre-turn mood (`s.affect`). This preserves the "no self-recall" guarantee: a turn can only retrieve memories written in earlier turns.
- Exactly **one episodic memory is written per turn**, after the response, using the deterministic template in `memory.py`. Method turns append the outcome (`accepted`/`rejected`/`acknowledged`) to the memory content. Memories carry a `session_id`; reflection-produced semantic memories use `session_id="reflection"`.
- Reflection is gated by `>= 5` new episodic memories since `last_reflection_at`. The reflector proposes insights (semantic memories) and trait drifts; validators enforce: `|delta| <= 0.05` per run, `|current - base| <= 0.30` total, surface traits only, `>= 3` source memories from `>= 2` sessions. Core traits are structurally rejected. `last_reflection_at` advances even if nothing is applied.
- Retrieval loads both episodic and semantic memories (no `kind` filter); semantic memories are scored like any other memory.
- The memory scoring weights (`0.45/0.25/0.15/0.15`), thresholds (`MIN_SCORE=0.15`, `MAX_RESULTS=5`, `WORD_BUDGET=600`), decay table, and rehearsal boost are tuned as a set — do not change them.
- The method social valences (`−0.8 … +0.6`), the `0.3` vacuum threshold, and the dimension assignments are tuned as a set — do not change them.
- The reflection bounds (`0.05`, `0.30`, `>= 3` sources, `>= 2` sessions, insight salience `0.6`, gate `>= 5`) are tuned as a set — do not change them.

## Integration points for future milestones (do not implement yet)

The M4 guide's §9 lists non-goals: memory consolidation/demotion/deletion of episodes, reflection-triggered methods or proactive outreach, numeric contracts for the DeepSeekReflector path (only MockReflector is contract-tested), and changes to any M1–M3 constant. The M3.5 guide's §8 non-goals still stand: output filter / regeneration loop, streaming responses, web UI, multi-user support, and persona-driven trait derivation (backstory is prose, not parsed). (Web UI and multi-user support are no longer mere non-goals — the web path was deleted and they are permanently ruled out; see the "Not a web app" section above.) The M3 non-goals still stand: implicit method parsing from free text, methods that close sessions (`/leave` is a stimulus, not a session command), per-companion method registries, sentiment alignment for targeted methods, phase-shift announcements, and new relationship dimensions. The M2 non-goals still stand: second-pass retrieval on the draft response, setting `activation.suppress` (the field exists and retrieval honors it, but nothing produces it yet), an LLM-based memory writer, vector databases/ANN indexes/embedding caches, and changes to `constraint.py` formulas or M1 constants. Also still future: an LLM perception classifier.

## Naming

`__init__.py` re-exports the public API (list in guide §8). Downstream milestones import these names — **ask before renaming or removing any of them**.
