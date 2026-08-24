"""Desktop GUI (Tkinter) — a native shell around the same CompanionSession.

Reuses the exact pipeline, state, memory and config layer that the web GUI
used (server.py), but renders it as a desktop window with no browser, no
JavaScript, no CSS and no GPU compositing — so it cannot hit the paint-level
problem that broke the web render.

    python gui.py                 # opens the window; companion.db in project root
    python gui.py --db my.db      # use a different database file

Layout mirrors the web app: three columns = dashboard | chat | inspector.
Provider/model/base-url config (mock / openai / deepseek / custom) is set from
a bar at the top; keys come from .env / env vars exactly as in server.py.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from companion import (
    CompanionSession,
    CompanionState,
    Memory,
    Store,
    default_embedder,
    load_character,
    load_dotenv,
)

# Reuse the tested provider/config layer from the web server so the desktop
# app behaves identically for provider selection and LLM construction.
from server import (  # noqa: E402
    PROVIDERS,
    build_llm,
    config_warning,
    load_config,
    save_config,
)

ROOT = Path(__file__).resolve().parent
DB = ROOT / "companion.db"
CHARACTER = "kira"

load_dotenv(ROOT / ".env")


def load_session(db_path: str) -> CompanionSession:
    """Hydrate or create the session, like chat.py but config-driven for the LLM."""
    store = Store(db_path)
    state = CompanionState.hydrate(CHARACTER, store)
    if state is None:
        char = load_character(ROOT / "characters" / f"{CHARACTER}.yaml")
        persona = char["persona"]
        state = CompanionState.create(
            companion_id=CHARACTER,
            name=char["name"],
            registry=char["registry"],
            voice_baseline=char["voice_baseline"],
            affect_baseline=char["mood_baseline"],
            backstory=persona.get("backstory", ""),
            speaking_style=persona.get("speaking_style", ""),
        )
    llm, _ = build_llm(load_config())
    return CompanionSession(state, store, llm, default_embedder())


class ConfigBar(ttk.Frame):
    """Provider/model/base-url bar; mirrors the web app's config modal."""

    def __init__(self, master, on_apply):
        super().__init__(master, padding=6)
        self.on_apply = on_apply

        ttk.Label(self, text="Provider").grid(row=0, column=0, padx=(0, 4))
        self.provider = tk.StringVar(value="mock")
        self._menu = ttk.Combobox(self, textvariable=self.provider, state="readonly",
                                  width=10)
        self._menu["values"] = list(PROVIDERS)
        self._menu.grid(row=0, column=1, padx=(0, 12))
        self._menu.bind("<<ComboboxSelected>>", lambda e: self._on_change())

        ttk.Label(self, text="Model").grid(row=0, column=2, padx=(0, 4))
        self.model = tk.StringVar()
        self.model_entry = ttk.Entry(self, textvariable=self.model, width=18)
        self.model_entry.grid(row=0, column=3, padx=(0, 12))

        ttk.Label(self, text="Base URL").grid(row=0, column=4, padx=(0, 4))
        self.base_url = tk.StringVar()
        self.base_url_entry = ttk.Entry(self, textvariable=self.base_url, width=22)
        self.base_url_entry.grid(row=0, column=5, padx=(0, 12))

        ttk.Button(self, text="Apply", command=self._apply).grid(row=0, column=6, padx=(0, 12))

        self.status = ttk.Label(self, text="", foreground="#e0af68")
        self.status.grid(row=0, column=7, sticky="w")

        self._refresh_from_config()

    def _refresh_from_config(self):
        cfg = load_config()
        self.provider.set(cfg["provider"])
        self.model.set(cfg["model"])
        self.base_url.set(cfg["base_url"])
        self._on_change()

    def _on_change(self):
        p = self.provider.get()
        # show base_url only for custom, mirroring web behavior
        state = "normal" if p == "custom" else "disabled"
        self.base_url_entry.configure(state=state)
        self._show_warning(p)

    def _show_warning(self, provider: str):
        cfg = {"provider": provider, "model": self.model.get(),
               "base_url": self.base_url.get()}
        self.status.configure(text=config_warning(cfg))

    def _apply(self):
        provider = self.provider.get()
        if provider not in PROVIDERS:
            self.status.configure(text=f"Unknown provider '{provider}'")
            return
        cfg = save_config({"provider": provider, "model": self.model.get(),
                           "base_url": self.base_url.get()})
        llm, warning = build_llm(cfg)
        probe = ""
        if hasattr(llm, "probe") and provider != "mock":
            self.status.configure(text="probing provider…")
            self.update_idletasks()
            probe = llm.probe()
        if probe:
            # A real call failed (e.g. DeepSeek rejected the model name).
            # Keep the previous LLM so the user isn't silently served fallback lines.
            self.status.configure(
                text=f"provider {provider} unusable: {probe[:120]}")
            return
        self._session.llm = llm
        self.status.configure(text=warning or f"provider → {provider}")

    def bind_session(self, session):
        self._session = session


class Dashboard(ttk.Frame):
    """Left column: mood + relationship + phases + narrative."""

    BARS = ("trust", "intimacy", "resentment", "valence", "arousal")

    def __init__(self, master):
        super().__init__(master, padding=8)
        self.name = ttk.Label(self, text="…", font=("Segoe UI", 16, "bold"))
        self.name.pack(anchor="w")
        self.narrative = ttk.Label(self, text="", wraplength=230, justify="left")
        self.narrative.pack(anchor="w", pady=(2, 8))

        ttk.Label(self, text="RELATIONSHIP", foreground="#8a90a0",
                  font=("Segoe UI", 9)).pack(anchor="w")
        self._bars = {}
        for key in ("trust", "intimacy", "resentment"):
            self._bar_row(key)
        ttk.Separator(self).pack(fill="x", pady=8)

        ttk.Label(self, text="MOOD", foreground="#8a90a0",
                  font=("Segoe UI", 9)).pack(anchor="w")
        for key in ("valence", "arousal"):
            self._bar_row(key)
        ttk.Separator(self).pack(fill="x", pady=8)

        ttk.Label(self, text="PHASES", foreground="#8a90a0",
                  font=("Segoe UI", 9)).pack(anchor="w")
        self.phases = ttk.Label(self, text="none", wraplength=230, justify="left")
        self.phases.pack(anchor="w")

    def _bar_row(self, key):
        row = ttk.Frame(self)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=key, width=9, foreground="#8a90a0").pack(side="left")
        bar = ttk.Progressbar(row, maximum=1.0, length=90, mode="determinate")
        bar.pack(side="left", padx=4, fill="x", expand=True)
        val = ttk.Label(row, text="", width=5, anchor="e")
        val.pack(side="right")
        self._bars[key] = (bar, val)

    def update(self, st):
        self.name.configure(text=st.name)
        self.narrative.configure(
            text=st.narrative_log[-1]["text"] if st.narrative_log else "")
        for key, (bar, val) in self._bars.items():
            v = getattr(st.relationship if key != "arousal" and key != "valence"
                        else st.affect, key)
            if key == "valence":
                bar.configure(value=(v + 1.0) / 2.0)   # center-anchor
            else:
                bar.configure(value=max(0.0, min(1.0, v)))
            val.configure(text=f"{v:+.2f}" if key == "valence" else f"{v:.2f}")
        self.phases.configure(text=", ".join(st.active_phases) or "none")


class Chat(ttk.Frame):
    """Center column: transcript + composer."""

    def __init__(self, master, on_send):
        super().__init__(master)
        self.on_send = on_send
        self.transcript = scrolledtext.ScrolledText(
            self, wrap="word", state="disabled", font=("Segoe UI", 11),
            background="#14161a", foreground="#d8dce4", insertbackground="#d8dce4")
        self.transcript.tag_configure("user", foreground="#7aa2f7")
        self.transcript.tag_configure("meta", foreground="#8a90a0",
                                      font=("Segoe UI", 9))
        self.transcript.pack(fill="both", expand=True, padx=6, pady=6)
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=6, pady=(0, 6))
        self.input = ttk.Entry(bottom)
        self.input.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.input.bind("<Return>", lambda e: self._send())
        self.send_btn = ttk.Button(bottom, text="Send", command=self._send)
        self.send_btn.pack(side="right")

    def _send(self):
        text = self.input.get().strip()
        if not text:
            return
        self.input.delete(0, "end")
        self.append("user", text)
        self.send_btn.configure(state="disabled")
        self.on_send(text)

    def append(self, role, text, meta=None):
        self.transcript.configure(state="normal")
        tag = "user" if role == "user" else "meta"
        self.transcript.insert("end", f"{'You' if role == 'user' else role}  ›  "
                              f"{text}\n", tag)
        if meta:
            self.transcript.insert("end", f"    {meta}\n", "meta")
        self.transcript.insert("end", "\n")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def set_ready(self):
        self.send_btn.configure(state="normal")
        self.input.focus_set()


class Inspector(ttk.Frame):
    """Right column: Traces / Memories / Reflections tabs."""

    COL_TRACES = ("impact", "archetype", "input", "phases")
    COL_MEM = ("kind", "salience", "content", "accessed")
    COL_REFL = ("when", "insights", "drifts", "narrative")

    def __init__(self, master):
        super().__init__(master)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        self._build_traces()
        self._build_memories()
        self._build_reflections()

    def _tree(self, columns, widths):
        tree = ttk.Treeview(self.notebook, columns=columns, show="headings", height=20)
        for col, width in zip(columns, widths):
            tree.heading(col, text=col)
            tree.column(col, width=width, anchor="w")
        vsb = ttk.Scrollbar(self.notebook, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        return tree, vsb

    def _build_traces(self):
        fr = ttk.Frame(self.notebook)
        self.notebook.add(fr, text="Traces")
        self.traces, vsb = self._tree(self.COL_TRACES, (50, 80, 150, 100))
        self.traces.pack(fill="both", expand=True)

    def _build_memories(self):
        fr = ttk.Frame(self.notebook)
        self.notebook.add(fr, text="Memories")
        self.mems, vsb = self._tree(self.COL_MEM, (60, 60, 200, 60))
        self.mems.pack(fill="both", expand=True)

    def _build_reflections(self):
        fr = ttk.Frame(self.notebook)
        self.notebook.add(fr, text="Reflections")
        self.refls, vsb = self._tree(self.COL_REFL, (150, 60, 60, 60))
        self.refls.pack(fill="both", expand=True)

    def update_traces(self, traces):
        self.traces.delete(*self.traces.get_children())
        for t in reversed(traces):
            sign = "+" if t["impact"] >= 0 else ""
            self.traces.insert("", "end", values=(
                f"{sign}{t['impact']:.2f}", t["archetype"],
                t["user_input"], ", ".join(t.get("active_phases", []))))

    def update_memories(self, memories, now):
        self.mems.delete(*self.mems.get_children())
        for m in reversed(memories):
            self.mems.insert("", "end", values=(
                m["kind"], f"{m['salience']:.2f}", m["content"],
                f"x{m['access_count']}"))

    def update_reflections(self, reflections):
        self.refls.delete(*self.refls.get_children())
        for r in reversed(reflections):
            ap = r.get("applied", {})
            if isinstance(ap, str):
                ap = {}
            self.refls.insert("", "end", values=(
                r["created_at"], len(ap.get("insight_memory_ids", [])),
                len(ap.get("drifts", [])), ap.get("narrative_added", False)))


class App:
    def __init__(self, root, db_path):
        self.root = root
        self.session = load_session(db_path)
        self.q: "queue.Queue" = queue.Queue()
        self.root.title(f"Companion — {self.session.state.name}")
        self.root.geometry("1240x700")

        self.config_bar = ConfigBar(root, None)
        self.config_bar.pack(fill="x")

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0, minsize=270)
        body.columnconfigure(1, weight=1, minsize=360)
        body.columnconfigure(2, weight=0, minsize=300)
        body.rowconfigure(0, weight=1)

        self.dashboard = Dashboard(body)
        self.dashboard.grid(row=0, column=0, sticky="nsw")
        self.chat = Chat(body, self._handle_send)
        self.chat.grid(row=0, column=1, sticky="nsew")
        self.inspector = Inspector(body)
        self.inspector.grid(row=0, column=2, sticky="nse")

        self.config_bar.bind_session(self.session)

        gap = self.session.open()
        self.chat.transcript.configure(state="normal")
        self.chat.transcript.insert(
            "end", f"{self.session.state.name} is awake (gap {gap:.2f}h). "
                   f"Type /help for methods.\n\n")
        self.chat.transcript.configure(state="disabled")
        self._refresh_all()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(80, self._poll)

    # ---- sending ----
    def _handle_send(self, text):
        threading.Thread(target=self._worker, args=(text,), daemon=True).start()

    def _worker(self, text):
        try:
            response, trace = self.session.turn(text)
            llm_err = getattr(self.session.llm, "last_error", None)
            self.q.put(("turn", text, response, trace, llm_err))
        except Exception as e:  # noqa: BLE001
            self.q.put(("error", str(e)))

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "turn":
                    _, text, response, trace, llm_err = item
                    meta = None
                    if trace is not None:
                        a = trace.activation
                        sign = "+" if a.impact >= 0 else ""
                        meta = f"[{a.archetype}] impact {sign}{a.impact:.2f}"
                        if trace.fallback:
                            reason = llm_err or "LLM offline"
                            meta += f" · fallback — {reason[:120]}"
                    self.chat.append("companion", response, meta)
                    self._refresh_all()
                elif item[0] == "error":
                    self.chat.append("error", item[1])
                self.chat.set_ready()
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    # ---- refresh ----
    def _refresh_all(self):
        st = self.session.state
        self.dashboard.update(st)

        traces = []
        for raw in self.session.store.load_traces(st.companion_id, 50):
            t = json.loads(raw)
            traces.append({
                "user_input": t["user_input"],
                "archetype": t["activation"]["archetype"],
                "impact": t["activation"]["impact"],
                "active_phases": t.get("active_phases", []),
                "fallback": t.get("fallback", False),
            })
        self.inspector.update_traces(traces)

        mems = []
        for raw in self.session.store.load_memories(st.companion_id):
            m = Memory.model_validate_json(raw)
            mems.append({"kind": m.kind, "content": m.content,
                         "salience": m.salience, "access_count": m.access_count})
        self.inspector.update_memories(mems, time.time())

        refls = []
        for raw in self.session.store.load_reflections(st.companion_id):
            entry = json.loads(raw)
            ap = entry.get("applied")
            if isinstance(ap, str):
                ap = json.loads(ap)
            refls.append({"created_at": entry["created_at"], "applied": ap or {}})
        self.inspector.update_reflections(refls)

    def _on_close(self):
        try:
            summary = self.session.close()
            if summary:
                messagebox.showinfo(
                    "Companion",
                    f"Reflection on close: insights={summary['insights']} "
                    f"drifts={summary['drifts']} narrative={summary['narrative']}")
        except Exception as e:  # noqa: BLE001
            messagebox.showwarning("Companion", f"Reflection failed: {e}")
        self.root.destroy()


def main(argv=None):
    load_dotenv()
    argv = argv if argv is not None else sys.argv[1:]
    db_path = str(DB)
    if "--db" in argv:
        idx = argv.index("--db")
        if idx + 1 < len(argv):
            db_path = argv[idx + 1]
    root = tk.Tk()
    App(root, db_path)
    root.mainloop()


if __name__ == "__main__":
    main()
