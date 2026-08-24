"""Audit log viewer and rollback tool for applied reflections."""

import argparse
import time

from companion import AppliedReflection, CompanionState, Store, load_dotenv


def format_entry(index: int, entry: dict) -> str:
    applied_raw = entry["applied"]
    applied = AppliedReflection.model_validate_json(applied_raw)
    dt = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry["created_at"]))
    status = "active" if not entry["rolled_back"] else "ROLLED BACK"
    return (
        f"[{index}] {dt} {status} — "
        f"insights={len(applied.insight_memory_ids)} "
        f"drifts={len(applied.drifts)} "
        f"narrative={applied.narrative_added}"
    )


def log_view(store: Store, companion_id: str) -> None:
    rows = store.load_reflections(companion_id)
    if not rows:
        print("no reflections logged")
        return
    for i, row in enumerate(rows):
        entry = __import__("json").loads(row)
        print(format_entry(i, entry))


def rollback_last(store: Store, companion_id: str) -> None:
    rows = store.load_reflections(companion_id)
    target = None
    for row in reversed(rows):
        entry = __import__("json").loads(row)
        if not entry["rolled_back"]:
            target = entry
            break
    if target is None:
        print("nothing to roll back")
        return

    state = CompanionState.hydrate(companion_id, store)
    if state is None:
        print("companion state not found")
        return

    applied = AppliedReflection.model_validate_json(target["applied"])

    for memory_id in applied.insight_memory_ids:
        store.delete_memory(memory_id)

    for drift in applied.drifts:
        for t in state.registry:
            if t.get("trait_id") == drift.trait_id:
                t["current_intensity"] = round(t["current_intensity"] - drift.delta, 4)
                break

    if applied.narrative_added and state.narrative_log:
        state.narrative_log.pop()

    state.checkpoint(store)
    store.mark_rolled_back(target["id"])
    print(
        f"rolled back {target['id'][:8]}: "
        f"-{len(applied.insight_memory_ids)} insights, "
        f"-{len(applied.drifts)} drifts, "
        f"narrative={applied.narrative_added}"
    )


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Reflection audit and rollback")
    parser.add_argument("--companion", default="kira", help="companion id")
    parser.add_argument("--db", default="./companion.db", help="database path")
    parser.add_argument("--log", action="store_true", help="show reflection log")
    parser.add_argument("--rollback-last", action="store_true", help="roll back the latest active reflection")
    args = parser.parse_args()

    store = Store(args.db)
    if args.rollback_last:
        rollback_last(store, args.companion)
    if args.log or not args.rollback_last:
        log_view(store, args.companion)


if __name__ == "__main__":
    main()
