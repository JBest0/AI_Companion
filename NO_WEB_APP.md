# Not a web app — permanently

This project is a **local, single-process, single-user desktop application**.
It can never be a web app — not because of effort, but by design. Every
architecture lock below is permanent. Do not treat any of them as "future
work".

## The engine is not built for a browser

- **Exactly one live session per process, with a server-global active
  character.** The M8 lock is explicit: no multi-user, no per-tab active
  character, no auth. The active character is a process-wide value. A web app
  needs per-user sessions and per-user active characters — that would require
  rewriting the engine's core model, which is off-limits by design.
- **Deterministic and offline by default.** The personality pipeline is
  pure code: the LLM renders a reaction, it never decides one. A browser app
  would add latency, transport, and a whole class of non-determinism the
  engine was designed without.

## All state is local

Everything lives on the machine at a fixed path:

- `characters/*.yaml` — character definitions, read off disk
- one SQLite database (`companion.db`) at a fixed project path
- `config.json` — provider/model/base-url/active-character
- `.env` — API keys, read from disk at startup (never persisted)

A browser app would need per-user databases, concurrent access to the same
SQLite file, and network transport for the state machine. None of that exists
or is planned. This is a desktop app; `companion.db` is a local file.

## The GUI is a native process

`gui.py` is a Tkinter window: a worker thread plus a `queue.Queue` drives
turns into a live window. The character creator (create / edit / duplicate /
archive / purge) lives in the same Tkinter window. This is the interface —
there is no other one.

An earlier web render was built (a browser GUI over the engine) and worked,
but it broke at the **paint level** and was never viable. The engine's design
locks make a real multi-user web app permanently out of scope. The maintainer
decided: **not a web app, no matter what.**

## The web path is gone

`server.py`, the `web/` directory, `refresh_server.py`, and the HTTP tests
were deleted. There is no server to run. `python server.py` and the `web/`
directory **no longer exist**. The provider/config layer that once lived in
`server.py` now lives in `companion/config.py`, imported directly by the
desktop GUI.

The interfaces are:

```bash
python gui.py    # the Tkinter GUI + character creator
python chat.py   # the interactive REPL
```
