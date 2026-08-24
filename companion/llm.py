import os
from typing import Protocol

from .models import VoiceProfile


class LLM(Protocol):
    def generate(self, *, system: str, user: str, voice: VoiceProfile) -> str:
        ...


class MockLLM:
    BAND_FLAVORS = {
        "severe_negative": "recoils, every line of their body refusing this",
        "mild_dislike": "hesitates, clearly not thrilled",
        "neutral": "considers it with an even expression",
        "warm_positive": "brightens visibly",
        "delight": "lights up completely, barely containing themselves",
    }

    def generate(self, *, system: str, user: str, voice: VoiceProfile) -> str:
        archetype = "neutral"
        for line in system.splitlines():
            if line.startswith("ARCHETYPE: "):
                archetype = line[len("ARCHETYPE: "):]
                break
        flavor = self.BAND_FLAVORS.get(archetype, "reacts with " + archetype.replace("_", " "))
        prefix = "*coldly* " if voice.temperature < 0.25 else ""
        return f"{prefix}[{archetype}] The companion {flavor}."


class OpenAILLM:
    def __init__(self):
        import openai

        self._client = openai.OpenAI()
        self._model = "gpt-4o-mini"

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


def default_llm() -> LLM:
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAILLM()
    return MockLLM()
