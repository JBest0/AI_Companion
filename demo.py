"""Four-session demo for the companion pipeline (Milestone 5).

An existing companion.db keeps its persisted state (including any old
persona-less state). Delete companion.db* to re-test from a clean state and
pick up persona changes from the YAML.
"""

from pathlib import Path

from companion import (
    CompanionSession,
    CompanionState,
    HashEmbedder,
    Store,
    default_llm,
    load_character,
    load_items,
)

ROOT = Path(__file__).parent


def show_input_response(text, response, trace):
    print(f"input: {text}")
    if trace is None:
        print(f"  error: {response}")
        return
    act = trace.activation
    contributions = ", ".join(f"{c.trait_id}={c.impact}" for c in act.contributions)
    print(f"  archetype={act.archetype} ambivalent={act.ambivalent} impact={act.impact:.4f}")
    print(f"  contributions: {contributions}")
    print(
        f"  valence {trace.affect_before.valence:+.2f} -> {trace.affect_after.valence:+.2f} | "
        f"trust {trace.relationship_before.trust:.2f} -> {trace.relationship_after.trust:.2f} | "
        f"resentment {trace.relationship_before.resentment:.2f} -> {trace.relationship_after.resentment:.2f}"
    )
    if trace.active_phases:
        print(
            f"  phases={trace.active_phases} | voice "
            f"temp={trace.voice_after.temperature:+.2f} "
            f"formality={trace.voice_after.formality:+.2f} "
            f"humor={trace.voice_after.humor:+.2f} "
            f"verbosity={trace.voice_after.verbosity:+.2f}"
        )
    if trace.retrieved_memories:
        print("  retrieved memories:")
        for r in trace.retrieved_memories:
            breakdown = " ".join(f"{k}={v:.4f}" for k, v in r.breakdown.items())
            print(f"    - [{r.score:.4f}] {breakdown} | {r.content}")
    else:
        print("  retrieved memories: none")
    print(f"  response: {response}")


def run_session(get_session, inputs, session_label):
    session = get_session()
    session.open()

    print(f"\n=== {session_label} ===")
    for text in inputs:
        response, trace = session.turn(text)
        show_input_response(text, response, trace)

    summary = session.close()
    return summary


def main():
    char = load_character("characters/kira.yaml")
    store = Store("companion.db")

    state = CompanionState.hydrate("kira", store)
    if state is None:
        state = CompanionState.create(
            companion_id="kira",
            name=char["name"],
            registry=char["registry"],
            voice_baseline=char["voice_baseline"],
            affect_baseline=char["mood_baseline"],
            backstory=char.get("persona", {}).get("backstory", ""),
            speaking_style=char.get("persona", {}).get("speaking_style", ""),
        )

    def get_session(items=None):
        return CompanionSession(state, store, default_llm(), HashEmbedder(),
                                items=items)

    session1_inputs = [
        "I brought you some chocolate cake!",
        "Want some of my salty pretzel?",
        "I brought you chocolate but also this salty pretzel",
        "How are you doing today?",
        "/gift cat",
    ]
    run_session(get_session, session1_inputs, "Session 1")

    state = CompanionState.hydrate("kira", store)
    session2_inputs = [
        "Do you remember the cat I gave you?",
        "I still think chocolate is the best.",
    ]
    run_session(get_session, session2_inputs, "Session 2")

    state = CompanionState.hydrate("kira", store)
    session3_inputs = [
        "/comfort",
        "/hug",
        "/insult me",
        "/gift chocolate",
        "/dance",
        "/gift cat",
    ]
    summary = run_session(get_session, session3_inputs, "Session 3")
    print(f"reflection: {summary}")

    state = CompanionState.hydrate("kira", store)
    session4_inputs = [
        "have you been thinking about us?",
    ]
    run_session(get_session, session4_inputs, "Session 4")

    state = CompanionState.hydrate("kira", store)
    session = get_session()
    session.open()
    print("\n=== session 5 opened (the bond: repair arc) ===")
    for inp in [
        "I'm sorry about the cat. Truly.",
        "/comfort",
        "/hug",
        "/gift chocolate",
        "/hug",
        "/comfort",
        "/gift chocolate",
        "/hug",
        "/comfort",
        "Thanks for not giving up on me.",
    ]:
        resp, trace = session.turn(inp)
        show_input_response(inp, resp, trace)
    session.close()
    print("\n=== session 5 closed ===")

    # --- session 6: items & tags ---
    state = CompanionState.hydrate("kira", store)
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
        show_input_response(inp, resp, trace)
    session.close()
    print("\n=== session 6 closed ===")


if __name__ == "__main__":
    main()
