import sys

from pathlib import Path

from companion import (
    CharacterManager,
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
            definition_hash=char["definition_hash"],   # M8
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
    db_path = "./companion.db"
    if "--db" in argv:
        idx = argv.index("--db")
        if idx + 1 < len(argv):
            db_path = argv[idx + 1]

    manager = CharacterManager(ROOT / "characters", Store(db_path))

    if "--list" in argv:
        for c in manager.list(include_archived=False):
            status = "met" if c.has_save else "new"
            flag = "" if c.valid else f"  INVALID: {c.load_error}"
            print(f"{c.char_id:20} {c.name:20} {status}{flag}")
        return

    positional = [a for a in argv if not a.startswith("--")
                  and a != db_path]
    character = positional[0] if positional else None
    if character is None:
        chars = [c for c in manager.list(include_archived=False) if c.valid]
        if not chars:
            print("no characters found in characters/ — create one in the "
                  "GUI (python gui.py) or add a YAML file.")
            return
        if len(chars) == 1:
            character = chars[0].char_id
        else:
            for i, c in enumerate(chars, 1):
                print(f"  {i}. {c.name} ({c.char_id})")
            try:
                choice = input("talk to [1]: ").strip() or "1"
            except EOFError:
                return
            if choice.isdigit() and 1 <= int(choice) <= len(chars):
                character = chars[int(choice) - 1].char_id
            else:
                character = choice
    if not manager.exists(character):
        print(f"no character named {character!r}; "
              f"have: {', '.join(c.char_id for c in manager.list(False))}")
        return

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
