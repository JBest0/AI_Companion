"""Local web GUI server (Milestone 6).

    python3 server.py            # then open http://localhost:8765

Stdlib only — no new dependencies. The server owns ONE global
CompanionSession (opened at startup, closed — with reflection — on
Ctrl+C) and exposes it over a small JSON API. All personality logic stays
in the companion package; this file is transport, nothing else.
"""
from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from companion import (AffectState, CompanionSession, CompanionState, LLM,
                       MockLLM, Store, VoiceProfile, default_embedder,
                       effective_salience, load_character, load_items)
from companion.env import load_dotenv
from companion.models import Memory, TurnTrace

load_dotenv(Path(__file__).parent / ".env")

ROOT = Path(__file__).parent
DB = ROOT / "companion.db"
WEB = ROOT / "web"
HOST = "127.0.0.1"
PORT = 8765
TRACE_LIMIT = 50
MAX_BODY = 16_384

CONTENT_TYPES = {".html": "text/html; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8"}

_session: CompanionSession | None = None

# ---- provider configuration (GUI config menu) ----
# config.json holds only provider/model/base_url. API keys always come from
# the environment (loaded from .env at startup); nothing secret is persisted.

CONFIG_FILE = ROOT / "config.json"
PROVIDERS = ("mock", "openai", "deepseek", "custom")
PROVIDER_DEFAULTS = {"openai": "gpt-4o-mini", "deepseek": "deepseek-chat"}
_PLACEHOLDER_KEYS = {"your-openai-key-here", "your-deepseek-key-here"}

_config: dict | None = None

# Last browser self-report posted by the GUI (diagnostics: real computed
# layout/errors from whatever browser loads the page).
_last_diag: dict | None = None


def _default_config() -> dict:
    return {"provider": "mock", "model": "", "base_url": ""}


def _normalize_config(cfg: dict) -> dict:
    provider = cfg.get("provider", "mock")
    model = (cfg.get("model") or "").strip()
    base_url = (cfg.get("base_url") or "").strip()
    if provider == "mock":
        model, base_url = "", ""
    elif provider in PROVIDER_DEFAULTS and not model:
        model = PROVIDER_DEFAULTS[provider]
    return {"provider": provider, "model": model, "base_url": base_url}


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
    _config = _normalize_config(cfg)
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


def _available() -> dict:
    return {"OPENAI_API_KEY": _key_present("OPENAI_API_KEY"),
            "DEEPSEEK_API_KEY": _key_present("DEEPSEEK_API_KEY")}


def config_payload() -> dict:
    cfg = load_config()
    _, warning = build_llm(cfg)
    return {"provider": cfg["provider"], "model": cfg["model"],
            "base_url": cfg["base_url"], "warning": warning,
            "available": _available()}


def get_session() -> CompanionSession:
    global _session
    if _session is None:
        store = Store(DB)
        state = CompanionState.hydrate("kira", store)
        if state is None:
            char = load_character(ROOT / "characters" / "kira.yaml")
            persona = char["persona"]
            state = CompanionState.create(
                "kira", char["name"], char["registry"],
                voice_baseline=char["voice_baseline"],
                affect_baseline=char["mood_baseline"],
                backstory=persona.get("backstory", ""),
                speaking_style=persona.get("speaking_style", ""))
        _session = CompanionSession(state, store, build_llm(load_config())[0],
                                    default_embedder(),
                                    items=load_items(ROOT / "items.yaml"))
    return _session


def state_snapshot(s: CompanionSession) -> dict:
    st = s.state
    return {"name": st.name,
            "affect": st.affect.model_dump(mode="json"),
            "relationship": st.relationship.model_dump(mode="json"),
            "active_phases": list(st.active_phases),
            "turn_count": s._turn_count,
            "narrative": st.narrative_log[-1]["text"] if st.narrative_log else ""}


def memories_payload(s: CompanionSession) -> list[dict]:
    now = time.time()
    out = []
    for raw in s.store.load_memories(s.state.companion_id):
        m = Memory.model_validate_json(raw)
        out.append({"id": m.id, "kind": m.kind, "content": m.content,
                    "salience": m.salience,
                    "effective_salience": round(effective_salience(m, now), 4),
                    "emotional_tags": m.emotional_tags,
                    "access_count": m.access_count,
                    "created_at": m.created_at,
                    "session_id": m.session_id,
                    "source_ids": m.source_ids})
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "CompanionGUI/6"

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file() or path.suffix not in CONTENT_TYPES:
            self._send_json({"error": "not found"}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES[path.suffix])
        self.send_header("Content-Length", str(len(body)))
        # Without this the browser heuristically caches index.html and keeps
        # rendering an old DOM (e.g. from mid-build) until a hard refresh.
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:     # keep the console clean
        pass

    def do_GET(self) -> None:
        s = get_session()
        if self.path == "/":
            return self._send_file(WEB / "index.html")
        if self.path.startswith("/static/"):
            name = self.path[len("/static/"):].split("?", 1)[0]  # drop cache-busting ?v=
            if "/" in name or ".." in name:
                return self._send_json({"error": "not found"}, 404)
            return self._send_file(WEB / name)
        if self.path == "/api/state":
            return self._send_json(state_snapshot(s))
        if self.path == "/api/traces":
            traces = [json.loads(raw) for raw in
                      s.store.load_traces(s.state.companion_id, TRACE_LIMIT)]
            return self._send_json([
                {"turn_id": t["turn_id"], "user_input": t["user_input"],
                 "response": t["response"],
                 "archetype": t["activation"]["archetype"],
                 "impact": t["activation"]["impact"],
                 "active_phases": t.get("active_phases", []),
                 "fallback": t.get("fallback", False)}
                for t in traces])
        if self.path.startswith("/api/trace/"):
            turn_id = self.path[len("/api/trace/"):]
            for raw in s.store.load_traces(s.state.companion_id, 10_000):
                if json.loads(raw)["turn_id"] == turn_id:
                    return self._send_json(json.loads(raw))
            return self._send_json({"error": "trace not found"}, 404)
        if self.path == "/api/memories":
            return self._send_json(memories_payload(s))
        if self.path == "/api/reflections":
            out = []
            for raw in s.store.load_reflections(s.state.companion_id):
                entry = json.loads(raw)
                applied = entry.get("applied")
                if isinstance(applied, str):
                    try:
                        applied = json.loads(applied)
                    except ValueError:
                        applied = {}
                entry["applied"] = applied or {}
                out.append(entry)
            return self._send_json(out)
        if self.path == "/api/config":
            return self._send_json(config_payload())
        if self.path == "/api/diag":
            return self._send_json({"report": _last_diag})
        self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.path == "/api/diag":
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._send_json({"error": "input too long"}, 413)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, UnicodeDecodeError):
                return self._send_json({"error": "bad json"}, 400)
            global _last_diag
            _last_diag = body
            return self._send_json({"ok": True})
        if self.path == "/api/config":
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._send_json({"error": "input too long"}, 413)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, UnicodeDecodeError):
                return self._send_json({"error": "bad json"}, 400)
            provider = body.get("provider")
            if provider not in PROVIDERS:
                return self._send_json({"error": f"unknown provider '{provider}'"}, 400)
            cfg = save_config({"provider": provider, "model": body.get("model", ""),
                               "base_url": body.get("base_url", "")})
            llm, warning = build_llm(cfg)
            get_session().llm = llm
            payload = {"provider": cfg["provider"], "model": cfg["model"],
                       "base_url": cfg["base_url"], "warning": warning,
                       "available": _available()}
            return self._send_json({"ok": True, "config": payload})
        if self.path != "/api/turn":
            return self._send_json({"error": "not found"}, 404)
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._send_json({"error": "input too long"}, 413)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            user_input = str(body.get("input", "")).strip()
        except (ValueError, UnicodeDecodeError):
            return self._send_json({"error": "bad json"}, 400)
        if not user_input:
            return self._send_json({"error": "empty input"}, 400)
        s = get_session()
        response, trace = s.turn(user_input)
        self._send_json({
            "response": response,
            "error": trace is None,          # invalid method -> error message
            "trace": json.loads(trace.model_dump_json()) if trace else None,
            "state": state_snapshot(s)})


def main() -> None:
    s = get_session()
    gap = s.open()
    print(f"{s.state.name} is awake (gap since last session: {gap:.2f}h)")
    cfg = load_config()
    print(f"LLM provider: {cfg['provider']}"
          + (f" ({cfg['model']})" if cfg["model"] else ""))
    print(f"GUI: http://{HOST}:{PORT}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        summary = s.close()
        if summary:
            print(f"reflection on close: {summary}")
        print("session closed.")


if __name__ == "__main__":
    main()
