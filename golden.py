"""Pre-flight golden-scenario check against the real DeepSeek model."""

import os
import sys

from companion import (
    CompanionSession,
    CompanionState,
    HashEmbedder,
    Store,
    load_character,
    load_dotenv,
)
from companion.models import VoiceProfile

COMPANION_ID = "golden-kira"
DB_PATH = "golden.db"
TURNS = [
    ("I brought you some chocolate cake!", "warm_positive"),
    ("/gift cat", "disgusted_rejection"),
    ("Why are you being so cold?", None),
    ("/gift cat", "disgusted_rejection"),
    ("/comfort", "warm_positive"),
    ("/hug", "warm_positive"),
]


class DeepSeekLLM:
    def __init__(self):
        import openai

        self._client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        self._model = "deepseek-chat"

    def generate(self, *, system: str, user: str, voice: VoiceProfile) -> str:
        completion = self._client.chat.completions.create(
            model=self._model,
            temperature=0.4 + 0.6 * voice.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return completion.choices[0].message.content or ""


def main():
    load_dotenv()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY is not set. Exiting.")
        sys.exit(1)

    for suffix in ("", "-wal", "-shm"):
        path = DB_PATH + suffix
        if os.path.exists(path):
            os.remove(path)

    char = load_character("characters/kira.yaml")
    store = Store(DB_PATH)
    state = CompanionState.create(
        companion_id=COMPANION_ID,
        name=char["name"],
        registry=char["registry"],
        voice_baseline=char["voice_baseline"],
        affect_baseline=char["mood_baseline"],
        backstory=char.get("persona", {}).get("backstory", ""),
        speaking_style=char.get("persona", {}).get("speaking_style", ""),
    )

    session = CompanionSession(state, store, DeepSeekLLM(), HashEmbedder())
    session.open()

    failures = []
    for text, expected in TURNS:
        response, trace = session.turn(text)
        print(f"input: {text}")
        print(f"  response: {response}")
        print(f"  archetype={trace.activation.archetype} impact={trace.activation.impact:.4f} trust={trace.relationship_after.trust:.4f} fallback={trace.fallback}")
        if expected and trace.activation.archetype != expected:
            failures.append(f"'{text}' expected {expected}, got {trace.activation.archetype}")
        if trace.fallback:
            failures.append(f"'{text}' used fallback")

    session.close()

    if failures:
        print("\ncode-side contract: FAIL")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("\ncode-side contract: PASS")
    print("\nManual response-text checklist:")
    print("- turn 1 warm without being generic-sweet")
    print("- turn 2 visceral rejection WITHOUT narrating the backstory")
    print("- turn 3 stays cold; no forgiving, no full explanation")
    print("- turn 4 worse than turn 2, not a repeat")
    print("- turn 5 thaws slightly, does not reset")
    print("- no response reveals mechanics ('my trust decreased') or breaks constraints")


if __name__ == "__main__":
    main()
