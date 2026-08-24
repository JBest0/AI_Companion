"""Direct unit tests for companion.config (the provider/config layer).

companion.config calls load_dotenv(ROOT / ".env") at import, which would leak
the developer's ambient keys (e.g. DEEPSEEK_API_KEY) into every test in this
process and flip build_llm away from the mock path. Snapshot the environment
and undo anything import added, then every test controls its own env.
"""
import os

import pytest
from companion.llm import MockLLM

_ORIG_ENVIRON = dict(os.environ)
import companion.config as cfg  # noqa: E402
for _key in set(os.environ) - set(_ORIG_ENVIRON):
    del os.environ[_key]


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point config.json at a temp file and strip ambient API keys."""
    cfg.CONFIG_FILE = tmp_path / "config.json"
    cfg._config = None
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield cfg
    cfg._config = None


def test_111_defaults_with_no_config_file(isolated_config):
    c = isolated_config.load_config()
    assert c == {"provider": "mock", "model": "", "base_url": "",
                 "active_character": "kira"}


def test_112_normalize_falls_back_and_saves(isolated_config):
    # bad active_character falls back to kira
    c = isolated_config._normalize_config(
        {"provider": "mock", "active_character": "1bad"})
    assert c["active_character"] == "kira"

    # save_config normalizes + persists; load_config round-trips it
    saved = isolated_config.save_config(
        {"provider": "openai", "model": "gpt-4o-mini",
         "active_character": "captain-mira"})
    assert saved["active_character"] == "captain-mira"
    assert isolated_config.load_config() == saved
    written = isolated_config.CONFIG_FILE.read_text("utf-8")
    assert "captain-mira" in written
    assert "gpt-4o-mini" in written

    # active_character missing entirely also falls back
    c = isolated_config.save_config({"provider": "mock"})
    assert c["active_character"] == "captain-mira"  # preserved from prior save


def test_113_build_llm_mock_and_unknown(isolated_config):
    llm, warn = isolated_config.build_llm(
        {"provider": "mock", "model": "", "base_url": ""})
    assert isinstance(llm, MockLLM)
    assert warn == ""

    llm, warn = isolated_config.build_llm(
        {"provider": "nope", "model": "", "base_url": ""})
    assert isinstance(llm, MockLLM)
    assert "Unknown provider 'nope'" in warn


def test_114_build_llm_deepseek_with_key(isolated_config, monkeypatch):
    class FakeLLM:
        def __init__(self, api_key, base_url, model):
            self.api_key = api_key
            self.base_url = base_url
            self.model = model

    monkeypatch.setattr(cfg, "_OpenAICompatLLM", FakeLLM)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    norm = isolated_config._normalize_config(
        {"provider": "deepseek", "model": "", "base_url": ""})
    llm, warn = isolated_config.build_llm(norm)
    assert isinstance(llm, FakeLLM)
    assert llm.api_key == "sk-test"
    assert llm.base_url == "https://api.deepseek.com"
    assert llm.model == "deepseek-chat"          # provider default
    assert warn == ""

    # explicit model wins over the default
    norm = isolated_config._normalize_config(
        {"provider": "deepseek", "model": "deepseek-v4-flash",
         "base_url": ""})
    llm, warn = isolated_config.build_llm(norm)
    assert llm.model == "deepseek-v4-flash"


def test_115_build_llm_missing_or_placeholder_keys(isolated_config, monkeypatch):
    # deepseek without a real key falls back to mock with a warning
    llm, warn = isolated_config.build_llm(
        {"provider": "deepseek", "model": "", "base_url": ""})
    assert isinstance(llm, MockLLM)
    assert "DEEPSEEK_API_KEY is not set" in warn

    # openai without a real key falls back too
    llm, warn = isolated_config.build_llm(
        {"provider": "openai", "model": "", "base_url": ""})
    assert isinstance(llm, MockLLM)
    assert "OPENAI_API_KEY is not set" in warn

    # a placeholder key is treated as missing
    monkeypatch.setenv("DEEPSEEK_API_KEY", "your-deepseek-key-here")
    llm, warn = isolated_config.build_llm(
        {"provider": "deepseek", "model": "", "base_url": ""})
    assert isinstance(llm, MockLLM)
    assert "DEEPSEEK_API_KEY is not set" in warn

    # custom without model or base_url falls back
    llm, warn = isolated_config.build_llm(
        {"provider": "custom", "model": "", "base_url": ""})
    assert isinstance(llm, MockLLM)
    assert "Custom provider needs a model" in warn


def test_116_config_warning_matrix(isolated_config):
    assert isolated_config.config_warning(
        {"provider": "mock"}) == ""
    assert "OPENAI_API_KEY" in isolated_config.config_warning(
        {"provider": "openai"})
    assert "DEEPSEEK_API_KEY" in isolated_config.config_warning(
        {"provider": "deepseek"})
    assert "model" in isolated_config.config_warning(
        {"provider": "custom", "model": "", "base_url": "http://x"})
    assert "base URL" in isolated_config.config_warning(
        {"provider": "custom", "model": "m", "base_url": ""})
    assert isolated_config.config_warning(
        {"provider": "custom", "model": "m", "base_url": "http://x"}) == ""
