import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from companion import (
    CompanionSession,
    CompanionState,
    HashEmbedder,
    MockLLM,
    Store,
    TraitRegistry,
    dislikes,
    likes,
)

# server.py loads the developer's real .env at import time, which would leak
# ambient keys (e.g. DEEPSEEK_API_KEY) into every test in this process and
# flip default_reflector() away from MockReflector. Snapshot and undo it so
# the suite never relies on the ambient environment.
_ORIG_ENVIRON = dict(os.environ)
import server  # noqa: E402
for _key in set(os.environ) - set(_ORIG_ENVIRON):
    del os.environ[_key]


@pytest.fixture()
def gui(tmp_path):
    """Running server + injected fresh session; yields (base_url, session)."""
    registry = TraitRegistry([likes("taste", "sweet", intensity=0.7),
                              dislikes("taste", "salty", intensity=-0.7)])
    store = Store(tmp_path / "gui.db")
    state = CompanionState.create("k", "Kira", registry.all())
    server._session = CompanionSession(state, store, MockLLM(), HashEmbedder())
    server._session.open()
    orig_config_file = server.CONFIG_FILE
    server.CONFIG_FILE = tmp_path / "config.json"
    server._config = None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)  # port 0 = ephemeral
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", server._session
    httpd.shutdown()
    server._session.close()
    server._session = None
    server.CONFIG_FILE = orig_config_file
    server._config = None


def get_json(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_turn(base: str, text: str) -> dict:
    req = urllib.request.Request(
        base + "/api/turn",
        data=json.dumps({"input": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_74_state_snapshot_shape(gui):
    base, _ = gui
    st = get_json(base, "/api/state")
    assert set(st.keys()) == {
        "name", "affect", "relationship", "active_phases", "turn_count", "narrative",
    }
    assert st["name"] == "Kira"
    assert st["relationship"]["trust"] == pytest.approx(0.5)
    assert st["active_phases"] == []


def test_75_memories_payload(gui):
    base, _ = gui
    post_turn(base, "I brought you some chocolate cake!")
    mems = get_json(base, "/api/memories")
    assert len(mems) == 1
    m = mems[0]
    assert m["kind"] == "episodic"
    assert m["salience"] == pytest.approx(0.7)
    assert "effective_salience" in m
    again = get_json(base, "/api/memories")
    assert again[0]["salience"] == pytest.approx(0.7)


def test_76_turn_round_trip(gui):
    base, _ = gui
    res = post_turn(base, "I brought you some chocolate cake!")
    assert res["error"] is False
    assert res["trace"]["activation"]["impact"] == pytest.approx(0.70, abs=0.001)
    assert res["state"]["relationship"]["trust"] == pytest.approx(0.5)
    traces = get_json(base, "/api/traces")
    assert len(traces) == 1
    assert traces[0]["user_input"] == "I brought you some chocolate cake!"
    assert traces[0]["response"] == res["response"]
    detail = get_json(base, "/api/trace/" + res["trace"]["turn_id"])
    assert detail["turn_id"] == res["trace"]["turn_id"]


def test_77_invalid_method_is_not_a_turn(gui):
    base, _ = gui
    res = post_turn(base, "/dance")
    assert res["error"] is True
    assert res["trace"] is None
    assert "Unknown method" in res["response"]
    assert get_json(base, "/api/traces") == []


def test_78_static_traversal_blocked(gui):
    base, _ = gui
    with urllib.request.urlopen(base + "/") as resp:
        page = resp.read().decode("utf-8")
    assert "<title>Companion</title>" in page
    for path in ("/static/../server.py", "/static/%2e%2e/server.py"):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + path)
        assert e.value.code == 404


def post_config(base: str, payload: dict) -> dict:
    req = urllib.request.Request(
        base + "/api/config",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_79_config_defaults(gui):
    base, _ = gui
    cfg = get_json(base, "/api/config")
    assert cfg["provider"] == "mock"
    assert cfg["model"] == ""
    assert cfg["base_url"] == ""
    assert "warning" in cfg
    assert set(cfg["available"]) == {"OPENAI_API_KEY", "DEEPSEEK_API_KEY"}


def test_80_config_switches_llm(gui, monkeypatch):
    base, session = gui
    built = []

    class StubLLM:
        def __init__(self, api_key, base_url, model):
            built.append((api_key, base_url, model))

        def generate(self, **kw):
            return "stub"

    monkeypatch.setattr(server, "_OpenAICompatLLM", StubLLM)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    res = post_config(base, {"provider": "deepseek", "model": "", "base_url": ""})
    assert res["ok"] is True
    assert res["config"]["provider"] == "deepseek"
    assert res["config"]["model"] == "deepseek-chat"
    assert built == [("sk-test", "https://api.deepseek.com", "deepseek-chat")]
    assert isinstance(session.llm, StubLLM)


def test_81_unknown_provider_rejected(gui):
    base, _ = gui
    req = urllib.request.Request(
        base + "/api/config",
        data=json.dumps({"provider": "wat", "model": "", "base_url": ""}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 400
