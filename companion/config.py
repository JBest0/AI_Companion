"""Provider / LLM configuration for the desktop GUI.

The config layer that once lived in server.py: which provider to talk to
(mock / openai / deepseek / custom), which model and base URL, and how to
build the LLM object from config.json + .env. Purely local — nothing here
touches the network at import time, and API keys are never persisted
(they always come from the environment, loaded from .env at import).

This is the only config home. There is no web server any more — the GUI
(gui.py) and tests import from here directly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .characters import SLUG_RE
from .env import load_dotenv
from .llm import LLM, MockLLM
from .models import VoiceProfile

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config.json"

# config.json holds only provider/model/base_url. API keys always come from
# the environment (loaded from .env at startup); nothing secret is persisted.
PROVIDERS = ("mock", "openai", "deepseek", "custom")
PROVIDER_DEFAULTS = {"openai": "gpt-4o-mini", "deepseek": "deepseek-chat"}
_PLACEHOLDER_KEYS = {"your-openai-key-here", "your-deepseek-key-here"}

_config: dict | None = None

load_dotenv(ROOT / ".env")


def _default_config() -> dict:
    return {"provider": "mock", "model": "", "base_url": "",
            "active_character": "kira"}


def _normalize_config(cfg: dict, active_fallback: str = "kira") -> dict:
    provider = cfg.get("provider", "mock")
    model = (cfg.get("model") or "").strip()
    base_url = (cfg.get("base_url") or "").strip()
    if provider == "mock":
        model, base_url = "", ""
    elif provider in PROVIDER_DEFAULTS and not model:
        model = PROVIDER_DEFAULTS[provider]
    active = (cfg.get("active_character") or "").strip() or active_fallback
    if not SLUG_RE.match(active):
        active = active_fallback
    return {"provider": provider, "model": model, "base_url": base_url,
            "active_character": active}


def load_config() -> dict:
    global _config
    if _config is None:
        try:
            _config = _normalize_config(json.loads(CONFIG_FILE.read_text("utf-8")))
        except (OSError, ValueError, TypeError):
            _config = _default_config()
    return _config


def save_config(cfg: dict) -> dict:
    global _config
    _config = _normalize_config(
        cfg, active_fallback=load_config().get("active_character", "kira"))
    CONFIG_FILE.write_text(json.dumps(_config, indent=2), "utf-8")
    return _config


def _key_present(name: str) -> bool:
    value = os.environ.get(name, "")
    return bool(value) and value not in _PLACEHOLDER_KEYS


class _OpenAICompatLLM:
    """OpenAI-compatible chat client; provider, model and base_url configurable."""

    def __init__(self, api_key: str, base_url: str | None, model: str,
                 timeout: float = 45.0):
        import openai

        self._client = openai.OpenAI(api_key=api_key, base_url=base_url,
                                     timeout=timeout)
        self._model = model
        self.last_error: str | None = None

    def generate(self, *, system: str, user: str, voice: VoiceProfile) -> str:
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                temperature=0.4 + 0.6 * voice.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            self.last_error = None
            return completion.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            # loop.py catches the raise and shows a canned fallback line; we
            # keep the real reason here so the GUI can display it instead of a
            # generic "LLM offline".
            self.last_error = f"{type(e).__name__}: {e}"
            raise

    def probe(self, voice: VoiceProfile | None = None) -> str:
        """Send one no-op call to confirm the provider/model actually works."""
        probe_voice = voice or VoiceProfile(temperature=0.0)
        try:
            self.generate(
                system="Reply with a single word: OK",
                user="ping",
                voice=probe_voice,
            )
            return ""
        except Exception as e:  # noqa: BLE001
            return self.last_error or f"{type(e).__name__}: {e}"


def config_warning(cfg: dict) -> str:
    provider = cfg["provider"]
    if provider == "openai" and not _key_present("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is not set in .env — responses will fall back until it is."
    if provider == "deepseek" and not _key_present("DEEPSEEK_API_KEY"):
        return "DEEPSEEK_API_KEY is not set in .env — responses will fall back until it is."
    if provider == "custom":
        if not cfg["model"]:
            return "Custom provider needs a model."
        if not cfg["base_url"]:
            return "Custom provider needs a base URL."
    return ""


def build_llm(cfg: dict) -> tuple[LLM, str]:
    provider = cfg["provider"]
    try:
        if provider == "mock":
            return MockLLM(), ""
        if provider == "openai":
            if not _key_present("OPENAI_API_KEY"):
                return MockLLM(), config_warning(cfg)
            return _OpenAICompatLLM(os.environ["OPENAI_API_KEY"], None, cfg["model"]), ""
        if provider == "deepseek":
            if not _key_present("DEEPSEEK_API_KEY"):
                return MockLLM(), config_warning(cfg)
            return _OpenAICompatLLM(os.environ["DEEPSEEK_API_KEY"],
                                    "https://api.deepseek.com", cfg["model"]), ""
        if provider == "custom":
            if not cfg["model"] or not cfg["base_url"]:
                return MockLLM(), config_warning(cfg)
            return _OpenAICompatLLM(os.environ.get("OPENAI_API_KEY") or "EMPTY",
                                    cfg["base_url"], cfg["model"]), ""
        return MockLLM(), f"Unknown provider '{provider}'."
    except ImportError:
        return MockLLM(), "The 'openai' package is not installed; using mock until it is."
