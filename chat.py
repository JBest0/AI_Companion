import sys

from pathlib import Path

from companion import (
    CompanionSession,
    CompanionState,
    Store,
    default_embedder,
    default_llm,
    load_character,
    load_dotenv,
    load_items,
)

ROOT = Path(__file__).parent


def load_session(character: str, db_path: str) -> CompanionSession:
    store = Store(db_path)
    state = CompanionState.hydrate(character, store)
    if state is None:
        char = load_character(f"characters/{character}.yaml")
        state = CompanionState.create(
            companion_id=character,
            name=char["name"],
            registry=char["registry"],
            voice_baseline=char["voice_baseline"],
            affect_baseline=char["mood_baseline"],
            backstory=char.get("persona", {}).get("backstory", ""),
            speaking_style=char.get("persona", {}).get("speaking_style", ""),
        )
    items_file = ROOT / "items.yaml"
    items = load_items(items_file) if items_file.exists() else None
    return CompanionSession(state, store, default_llm(), default_embedder(),
                            items=items)


def format_trace(trace) -> str:
    act = trace.activation
    if act.contributions:
        traits = ",".join(
            f"{c.trait_id}:{c.impact:+.2f}" for c in act.contributions
        )
    else:
        traits = "-"
    return (
        f"[trace] archetype={act.archetype} impact={act.impact:+.2f} "
        f"ambivalent={act.ambivalent} traits=[{traits}] "
        f"recalled={len(trace.retrieved_memories)} "
        f"valence={trace.affect_before.valence:+.2f}->{trace.affect_after.valence:+.2f} "
        f"trust={trace.relationship_before.trust:.2f}->{trace.relationship_after.trust:.2f} "
        f"fallback={trace.fallback}"
    )


def main(argv=None):
    load_dotenv()
    argv = argv if argv is not None else sys.argv[1:]
    character = argv[0] if argv else "kira"
    db_path = "./companion.db"
    if "--db" in argv:
        idx = argv.index("--db")
        if idx + 1 < len(argv):
            db_path = argv[idx + 1]

    session = load_session(character, db_path)
    gap = session.open()
    name = session.state.name
    if gap >= 0.5:
        print(f"{name} is here. (last seen {gap:.1f}h ago)")
    else:
        print(f"{name} is here.")

    last_trace = None
    try:
        while True:
            try:
                user_input = input("you: ")
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.strip() == "/quit":
                break
            if user_input.strip() == "/trace":
                if last_trace is None:
                    print("[trace] no turns yet")
                else:
                    print(format_trace(last_trace))
                continue
            response, trace = session.turn(user_input)
            print(f"{name}: {response}")
            if trace is not None:
                last_trace = trace
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        print(f"{name}: *session closed*")


if __name__ == "__main__":
    main()
