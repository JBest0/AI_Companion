"""Desktop GUI (Tkinter) — the only front-end for the companion.

A native shell around CompanionSession: no browser, no JavaScript, no CSS,
no server. Renders the engine as a desktop window, so the paint-level problem
that broke the earlier web render cannot occur.

    python gui.py                 # opens the window; companion.db in project root
    python gui.py --db my.db      # use a different database file

Three columns = dashboard | chat | inspector. Provider/model/base-url config
(mock / openai / deepseek / custom) is set from a bar at the top; keys come
from .env / env vars. The same bar switches characters, and the ＋ New / ✎
Edit buttons open the Tkinter character creator (create / edit / duplicate /
archive / purge).
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, simpledialog, ttk

from companion import (
    CHARACTER_TEMPLATES,
    CharacterManager,
    CharacterSpec,
    CompanionSession,
    CompanionState,
    Memory,
    Store,
    default_embedder,
    load_character,
    load_dotenv,
)
from companion.models import AffectState, Trait, TraitCategory, Trigger, VoiceProfile

# Provider/config layer. Was server.py; now companion.config so the desktop
# app is fully self-contained — there is no HTTP anywhere in the project.
from companion.config import (
    PROVIDERS,
    build_llm,
    config_warning,
    load_config,
    save_config,
)

ROOT = Path(__file__).resolve().parent
DB = ROOT / "companion.db"

load_dotenv(ROOT / ".env")


def load_session(db_path: str, char_id: str) -> CompanionSession:
    """Hydrate or create the session, like chat.py but config-driven for the LLM."""
    store = Store(db_path)
    state = CompanionState.hydrate(char_id, store)
    if state is None:
        char = load_character(ROOT / "characters" / f"{char_id}.yaml")
        persona = char["persona"]
        state = CompanionState.create(
            companion_id=char_id,
            name=char["name"],
            registry=char["registry"],
            voice_baseline=char["voice_baseline"],
            affect_baseline=char["mood_baseline"],
            backstory=persona.get("backstory", ""),
            speaking_style=persona.get("speaking_style", ""),
            definition_hash=char["definition_hash"],   # M8
        )
    llm, _ = build_llm(load_config())
    return CompanionSession(state, store, llm, default_embedder())


class CharacterEditor(tk.Toplevel):
    """Create or edit a character definition (Tkinter creator for M8+).

    Mirrors the web creator's fields: name, avatar, persona prose, mood and
    voice baselines, likes/dislikes, and traits. Saving validates through
    CharacterSpec and writes the YAML via CharacterManager.
    """

    PREF_DOMAINS = ("taste", "entity", "activity", "topic", "tag",
                    "item", "category")
    CURVES = ("linear", "steep", "threshold")
    CATEGORIES = ("surface", "core")

    def __init__(self, master, manager: CharacterManager, on_save, char_id=None):
        super().__init__(master)
        self.manager = manager
        self.on_save = on_save
        self.char_id = char_id
        self._spec = None
        self._likes: list[dict] = []
        self._dislikes: list[dict] = []
        self._traits: list[dict] = []

        self.title("Edit character" if char_id else "New character")
        self.geometry("520x720")
        self.minsize(460, 540)
        self.transient(master)
        self.grab_set()

        self._build_ui()
        self._load_spec()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _build_ui(self):
        # scrolled canvas so the form fits on smaller screens
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self._frame = ttk.Frame(canvas, padding=12)
        self._frame.bind("<Configure>",
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._frame, anchor="nw", width=500)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(
            int(-1 * (e.delta / 120)), "units"))

        # template (new only)
        self._template_row = self._row(self._frame, "Template")
        self._template = ttk.Combobox(self._template_row, state="readonly",
                                      values=list(CHARACTER_TEMPLATES.keys()),
                                      width=20)
        self._template.set("blank")
        self._template.pack(side="left", fill="x", expand=True)
        self._template.bind("<<ComboboxSelected>>", lambda e: self._apply_template())

        # name
        row, self._name_var, _ = self._entry_row(self._frame, "Name")

        # id (new only; shown disabled when editing)
        row, self._char_id, self._char_id_entry = self._entry_row(
            self._frame, "Id (filename)")

        # avatar
        row, self._avatar, _ = self._entry_row(self._frame, "Avatar (one emoji)")

        # persona
        row, self._backstory = self._text_row(self._frame, "Backstory", 4)
        row, self._style = self._text_row(self._frame, "Speaking style", 3)

        # mood
        ttk.Label(self._frame, text="Mood baseline",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 4))
        row, self._valence = self._slider_row(self._frame, "gloomy ↔ sunny", -1, 1, 0.1)
        row, self._arousal = self._slider_row(self._frame, "calm ↔ excitable", 0, 1, 0.1)

        # voice
        ttk.Label(self._frame, text="Voice baseline",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 4))
        row, self._vtemp = self._slider_row(self._frame, "warmth", 0, 1, 0.05)
        row, self._vverb = self._slider_row(self._frame, "verbosity", -1, 1, 0.1)
        row, self._vhumor = self._slider_row(self._frame, "humor", -1, 1, 0.1)
        row, self._vform = self._slider_row(self._frame, "formality", 0, 1, 0.05)
        row, self._vmeta = self._slider_row(self._frame, "metaphor density", 0, 1, 0.05)

        # likes / dislikes / traits
        ttk.Label(self._frame, text="Personality",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 4))
        self._likes_list, self._likes_frame = self._list_section(
            self._frame, "Likes", self._add_like, self._del_like)
        self._dislikes_list, self._dislikes_frame = self._list_section(
            self._frame, "Dislikes", self._add_dislike, self._del_dislike)
        self._traits_list, self._traits_frame = self._list_section(
            self._frame, "Traits", self._add_trait, self._del_trait)

        # errors
        self._errors = ttk.Label(self._frame, text="", foreground="#f7768e",
                                 wraplength=460, justify="left")
        self._errors.pack(fill="x", pady=(10, 0))

        # buttons
        btns = ttk.Frame(self._frame)
        btns.pack(fill="x", pady=(12, 0))
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(btns, text="Save", command=self._save).pack(side="right")

        # danger zone (edit only)
        self._danger = ttk.LabelFrame(self._frame, text="Danger zone", padding=8)
        ttk.Button(self._danger, text="Archive",
                   command=self._archive).pack(side="left", padx=(0, 6))
        ttk.Button(self._danger, text="Duplicate",
                   command=self._duplicate).pack(side="left", padx=(0, 6))
        ttk.Button(self._danger, text="Purge…",
                   command=self._purge).pack(side="left")

    @staticmethod
    def _row(parent, label):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=16).pack(side="left")
        return row

    def _entry_row(self, parent, label):
        row = self._row(parent, label)
        var = tk.StringVar()
        entry = ttk.Entry(row, textvariable=var)
        entry.pack(side="left", fill="x", expand=True)
        return row, var, entry

    def _text_row(self, parent, label, height):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=16).pack(side="left", anchor="n")
        txt = tk.Text(row, height=height, wrap="word",
                      background="#14161a", foreground="#d8dce4",
                      insertbackground="#d8dce4")
        txt.pack(side="left", fill="x", expand=True)
        return row, txt

    def _slider_row(self, parent, label, min_v, max_v, step):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=16).pack(side="left")
        var = tk.DoubleVar(value=(min_v + max_v) / 2)
        scale = ttk.Scale(row, from_=min_v, to=max_v, variable=var,
                          orient="horizontal", length=220)
        scale.pack(side="left", fill="x", expand=True, padx=(0, 6))
        val = ttk.Label(row, text=f"{var.get():.2f}", width=5)
        val.pack(side="left")
        var.trace_add("write", lambda *a, v=var, l=val:
                      l.configure(text=f"{v.get():.2f}"))
        return row, var

    def _list_section(self, parent, label, add_cmd, del_cmd):
        frame = ttk.LabelFrame(parent, text=label, padding=6)
        frame.pack(fill="x", pady=4)
        lst = tk.Listbox(frame, height=4, background="#14161a",
                         foreground="#d8dce4")
        lst.pack(side="left", fill="both", expand=True)
        btns = ttk.Frame(frame)
        btns.pack(side="right", fill="y", padx=(6, 0))
        ttk.Button(btns, text="+", width=3, command=add_cmd).pack(pady=(0, 3))
        ttk.Button(btns, text="−", width=3, command=del_cmd).pack()
        return lst, frame

    def _load_spec(self):
        if self.char_id:
            self._spec = self.manager.load(self.char_id)
            self._template_row.pack_forget()
            self._char_id.set(self.char_id)
            self._char_id_entry.configure(state="disabled")
            self._danger.pack(fill="x", pady=(14, 0))
            self._fill_form()
        else:
            # No spec yet: a CharacterSpec requires a valid id + name, which
            # the user hasn't typed. Fill the form from the raw template dict
            # instead, and let _save build (and validate) the spec.
            self._spec = None
            self._danger.pack_forget()
            self._fill_from_dict(CHARACTER_TEMPLATES[self._template.get()])

    def _fill_form(self):
        s = self._spec
        self._name_var.set(s.name)
        self._char_id.set(s.char_id)
        self._avatar.set(s.avatar)
        self._backstory.delete("1.0", "end")
        self._backstory.insert("1.0", s.backstory)
        self._style.delete("1.0", "end")
        self._style.insert("1.0", s.speaking_style)
        self._valence.set(s.mood_baseline.valence)
        self._arousal.set(s.mood_baseline.arousal)
        self._vtemp.set(s.voice_baseline.temperature)
        self._vverb.set(s.voice_baseline.verbosity)
        self._vhumor.set(s.voice_baseline.humor)
        self._vform.set(s.voice_baseline.formality)
        self._vmeta.set(s.voice_baseline.metaphor_density)
        self._likes = [{"domain": p.domain, "values": list(p.values),
                        "intensity": p.intensity} for p in s.likes]
        self._dislikes = [{"domain": p.domain, "values": list(p.values),
                           "intensity": p.intensity} for p in s.dislikes]
        self._traits = []
        for t in s.traits:
            self._traits.append({
                "trait_id": t.trait_id,
                "category": t.category.value,
                "description": t.description,
                "triggers": [{"domain": tr.domain, "values": list(tr.values)}
                             for tr in t.triggers],
                "base_intensity": t.base_intensity,
                "current_intensity": t.current_intensity,
                "curve": t.curve,
                "archetypes_negative": list(t.archetypes_negative),
                "archetypes_positive": list(t.archetypes_positive),
                "voice_modifiers": t.voice_modifiers.model_dump(),
                "salience_class": t.salience_class,
            })
        self._refresh_lists()

    def _apply_template(self):
        # Templates are partial character-file dicts and can't build a spec
        # until the user supplies a valid name and id, so fill the form
        # directly and preserve whatever they already typed.
        name = self._name_var.get()
        cid = self._char_id.get()
        self._fill_from_dict(CHARACTER_TEMPLATES[self._template.get()])
        self._name_var.set(name)
        self._char_id.set(cid)

    def _fill_from_dict(self, data):
        """Populate the form from a raw character-file/template dict. Used for
        new characters, where there is no CharacterSpec yet."""
        self._name_var.set(data.get("name", ""))
        self._char_id.set(data.get("char_id", ""))
        self._avatar.set(data.get("avatar") or "")
        self._backstory.delete("1.0", "end")
        self._backstory.insert("1.0",
                               (data.get("persona") or {}).get("backstory", ""))
        self._style.delete("1.0", "end")
        self._style.insert("1.0",
                           (data.get("persona") or {}).get("speaking_style", ""))
        mb = data.get("mood_baseline") or {}
        self._valence.set(mb.get("valence", 0.0))
        self._arousal.set(mb.get("arousal", 0.2))
        vb = data.get("voice_baseline") or {}
        self._vtemp.set(vb.get("temperature", 0.5))
        self._vverb.set(vb.get("verbosity", 0.0))
        self._vhumor.set(vb.get("humor", 0.0))
        self._vform.set(vb.get("formality", 0.5))
        self._vmeta.set(vb.get("metaphor_density", 0.2))
        self._likes = [dict(p) for p in (data.get("likes") or [])]
        self._dislikes = [dict(p) for p in (data.get("dislikes") or [])]
        self._traits = [dict(t) for t in (data.get("traits") or [])]
        self._refresh_lists()

    def _refresh_lists(self):
        for lst, data, fmt in (
            (self._likes_list, self._likes,
             lambda d: f"{d['domain']}: {', '.join(d['values'])} ({d['intensity']:+.2f})"),
            (self._dislikes_list, self._dislikes,
             lambda d: f"{d['domain']}: {', '.join(d['values'])} ({d['intensity']:+.2f})"),
            (self._traits_list, self._traits,
             lambda d: f"{d['trait_id']} ({d['category']}) {d['base_intensity']:+.2f}"),
        ):
            lst.delete(0, "end")
            for item in data:
                lst.insert("end", fmt(item))

    def _add_like(self):
        self._edit_pref(self._likes, +1)

    def _add_dislike(self):
        self._edit_pref(self._dislikes, -1)

    def _del_like(self):
        self._del_selected(self._likes_list, self._likes)

    def _del_dislike(self):
        self._del_selected(self._dislikes_list, self._dislikes)

    def _add_trait(self):
        self._edit_trait()

    def _del_trait(self):
        self._del_selected(self._traits_list, self._traits)

    def _del_selected(self, lst, data):
        sel = lst.curselection()
        if not sel:
            return
        idx = sel[0]
        data.pop(idx)
        self._refresh_lists()

    def _edit_pref(self, data, sign):
        dialog = tk.Toplevel(self)
        dialog.title("Preference")
        dialog.transient(self)
        dialog.grab_set()
        dialog.geometry("360x160")

        ttk.Label(dialog, text="Domain").grid(row=0, column=0, padx=6, pady=6)
        domain = ttk.Combobox(dialog, state="readonly", values=self.PREF_DOMAINS)
        domain.set(self.PREF_DOMAINS[0])
        domain.grid(row=0, column=1, padx=6, pady=6)

        ttk.Label(dialog, text="Values (comma)").grid(row=1, column=0, padx=6, pady=6)
        values = ttk.Entry(dialog)
        values.grid(row=1, column=1, padx=6, pady=6, sticky="ew")

        ttk.Label(dialog, text="Intensity").grid(row=2, column=0, padx=6, pady=6)
        intensity = tk.DoubleVar(value=0.5)
        ttk.Scale(dialog, from_=0.1, to=1.0, variable=intensity,
                  orient="horizontal").grid(row=2, column=1, padx=6, pady=6, sticky="ew")

        def ok():
            vals = [v.strip() for v in values.get().split(",") if v.strip()]
            if not vals:
                return
            data.append({"domain": domain.get(), "values": vals,
                         "intensity": sign * abs(float(intensity.get()))})
            self._refresh_lists()
            dialog.destroy()

        ttk.Button(dialog, text="OK", command=ok).grid(row=3, column=1, padx=6,
                                                        pady=6, sticky="e")
        dialog.columnconfigure(1, weight=1)
        dialog.wait_window(dialog)

    def _edit_trait(self):
        dialog = tk.Toplevel(self)
        dialog.title("Trait")
        dialog.transient(self)
        dialog.grab_set()
        dialog.geometry("420x280")

        ttk.Label(dialog, text="trait_id").grid(row=0, column=0, padx=6, pady=4)
        tid = ttk.Entry(dialog)
        tid.grid(row=0, column=1, padx=6, pady=4, sticky="ew")

        ttk.Label(dialog, text="Domain").grid(row=1, column=0, padx=6, pady=4)
        domain = ttk.Combobox(dialog, state="readonly", values=self.PREF_DOMAINS)
        domain.set(self.PREF_DOMAINS[0])
        domain.grid(row=1, column=1, padx=6, pady=4, sticky="ew")

        ttk.Label(dialog, text="Values (comma)").grid(row=2, column=0, padx=6, pady=4)
        values = ttk.Entry(dialog)
        values.grid(row=2, column=1, padx=6, pady=4, sticky="ew")

        ttk.Label(dialog, text="Intensity").grid(row=3, column=0, padx=6, pady=4)
        intensity = tk.DoubleVar(value=-0.5)
        ttk.Scale(dialog, from_=-1.0, to=1.0, variable=intensity,
                  orient="horizontal").grid(row=3, column=1, padx=6, pady=4, sticky="ew")

        ttk.Label(dialog, text="Curve").grid(row=4, column=0, padx=6, pady=4)
        curve = ttk.Combobox(dialog, state="readonly", values=self.CURVES)
        curve.set("linear")
        curve.grid(row=4, column=1, padx=6, pady=4, sticky="ew")

        ttk.Label(dialog, text="Category").grid(row=5, column=0, padx=6, pady=4)
        category = ttk.Combobox(dialog, state="readonly", values=self.CATEGORIES)
        category.set("surface")
        category.grid(row=5, column=1, padx=6, pady=4, sticky="ew")

        ttk.Label(dialog, text="Description").grid(row=6, column=0, padx=6, pady=4)
        desc = ttk.Entry(dialog)
        desc.grid(row=6, column=1, padx=6, pady=4, sticky="ew")

        def ok():
            vals = [v.strip() for v in values.get().split(",") if v.strip()]
            if not vals:
                return
            self._traits.append({
                "trait_id": tid.get().strip(),
                "category": category.get(),
                "description": desc.get().strip(),
                "triggers": [{"domain": domain.get(), "values": vals}],
                "base_intensity": float(intensity.get()),
                "current_intensity": float(intensity.get()),
                "curve": curve.get(),
                "archetypes_negative": [],
                "archetypes_positive": [],
                "voice_modifiers": {},
                "salience_class": "medium",
            })
            self._refresh_lists()
            dialog.destroy()

        ttk.Button(dialog, text="OK", command=ok).grid(row=7, column=1, padx=6,
                                                        pady=8, sticky="e")
        dialog.columnconfigure(1, weight=1)
        dialog.wait_window(dialog)

    def _collect_spec(self) -> CharacterSpec:
        data = {
            "char_id": self._char_id.get().strip(),
            "name": self._name_var.get().strip(),
            "avatar": self._avatar.get().strip(),
            "mood_baseline": {
                "valence": float(self._valence.get()),
                "arousal": float(self._arousal.get()),
            },
            "voice_baseline": {
                "temperature": float(self._vtemp.get()),
                "verbosity": float(self._vverb.get()),
                "humor": float(self._vhumor.get()),
                "formality": float(self._vform.get()),
                "metaphor_density": float(self._vmeta.get()),
            },
            "backstory": self._backstory.get("1.0", "end-1c"),
            "speaking_style": self._style.get("1.0", "end-1c"),
            "likes": self._likes,
            "dislikes": self._dislikes,
            "traits": self._traits,
        }
        return CharacterSpec.from_yaml_dict(data, data["char_id"])

    def _save(self):
        self._errors.configure(text="")
        try:
            spec = self._collect_spec()
        except Exception as e:  # noqa: BLE001
            self._errors.configure(text=f"Validation error: {e}")
            return
        try:
            if self.char_id:
                self.manager.update(self.char_id, spec)
                char_id, action = self.char_id, "update"
            else:
                self.manager.create(spec)
                char_id, action = spec.char_id, "create"
        except Exception as e:  # noqa: BLE001
            self._errors.configure(text=f"Save failed: {e}")
            return
        self.on_save(char_id, action)
        self.destroy()

    def _archive(self):
        # web parity: only the active character is ever editable, and the
        # engine needs a live companion, so the active one cannot be removed.
        if self.char_id == load_config().get("active_character"):
            messagebox.showerror(
                "Archive", "cannot archive the active character — "
                           "switch to someone else first")
            return
        if not messagebox.askyesno("Archive", f"Archive {self.char_id}?\n\n"
                                   "The file moves to characters/.archive and the DB is untouched."):
            return
        try:
            self.manager.archive(self.char_id)
            self.on_save(None, "archive")
            self.destroy()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Archive failed", str(e))

    def _duplicate(self):
        name = self._name_var.get().strip() + " copy"
        new_id = self._char_id.get().strip() + "_copy"
        try:
            created = self.manager.duplicate(self.char_id, new_id, name)
            self.on_save(created, "duplicate")
            self.destroy()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Duplicate failed", str(e))

    def _purge(self):
        # web parity: the active character cannot be permanently removed.
        if self.char_id == load_config().get("active_character"):
            messagebox.showerror(
                "Purge", "cannot purge the active character — "
                         "switch to someone else first")
            return
        confirm = simpledialog.askstring(
            "Purge character", f"Type '{self.char_id}' to permanently delete the file and DB rows:")
        if confirm != self.char_id:
            return
        try:
            self.manager.purge(self.char_id)
            self.on_save(None, "purge")
            self.destroy()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Purge failed", str(e))


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

        ttk.Label(self, text="Character").grid(row=0, column=8, padx=(12, 4))
        self.character = tk.StringVar()
        self.char_menu = ttk.Combobox(self, textvariable=self.character,
                                      state="readonly", width=12)
        self.char_menu.grid(row=0, column=9)
        self.char_menu.bind("<<ComboboxSelected>>",
                            lambda e: self.on_character(self.character.get()))
        self.on_character = lambda cid: None   # replaced by App

        self.edit_btn = ttk.Button(self, text="✎ Edit", width=6,
                                   command=lambda: self.on_edit())
        self.edit_btn.grid(row=0, column=10, padx=(8, 4))
        self.on_edit = lambda: None            # replaced by App

        self.new_btn = ttk.Button(self, text="＋ New", width=7,
                                  command=lambda: self.on_new())
        self.new_btn.grid(row=0, column=11, padx=(0, 8))
        self.on_new = lambda: None             # replaced by App

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
        self.db_path = db_path
        self.active = load_config().get("active_character", "kira")
        self.manager = CharacterManager(ROOT / "characters", Store(db_path))
        self.session = load_session(db_path, self.active)
        self.switching = False
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

        ids = [c.char_id for c in self.manager.list(include_archived=False)
               if c.valid]
        self.config_bar.char_menu["values"] = ids
        self.config_bar.character.set(self.active)
        self.config_bar.on_character = self._switch_character
        self.config_bar.on_new = lambda: self._open_editor(None)
        self.config_bar.on_edit = lambda: self._open_editor(self.active)

        gap = self.session.open()
        self.chat.transcript.configure(state="normal")
        self.chat.transcript.insert(
            "end", f"{self.session.state.name} is awake (gap {gap:.2f}h). "
                   f"Type /help for methods.\n\n")
        self.chat.transcript.configure(state="disabled")
        self._refresh_all()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(80, self._poll)

    def _switch_character(self, char_id):
        if self.switching or char_id == self.active:
            return
        restart = self._restart_choice(char_id)
        if restart is None:                       # user cancelled the dialog
            self.config_bar.character.set(self.active)
            return
        self.switching = True
        self.chat.send_btn.configure(state="disabled")
        self._set_editing_buttons(False)

        def work():
            try:
                self.session.close()          # M4 close semantics
                if restart:
                    self.manager.store.purge_companion(char_id)
                self.session = load_session(self.db_path, char_id)
                self.active = char_id
                save_config({**load_config(), "active_character": char_id})
                gap = self.session.open()
                self.q.put(("switched", char_id, gap))
            except Exception as e:  # noqa: BLE001
                self.q.put(("error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    # ---- character creator wiring ----
    def _open_editor(self, char_id):
        CharacterEditor(self.root, self.manager,
                        on_save=self._on_editor_save, char_id=char_id)

    def _on_editor_save(self, char_id, action):
        if action == "create" and char_id:
            self._refresh_char_menu()
            self._switch_character(char_id)      # M8 lock 8: auto-select
        elif action == "duplicate":
            self._refresh_char_menu()
            self.config_bar.status.configure(text=f"duplicated → {char_id}")
        elif action == "update":
            self._refresh_char_menu()
            self.config_bar.status.configure(
                text="definition saved — instance unchanged; "
                     "restart as new to apply")
        else:                                     # archive / purge
            self._refresh_char_menu()
            self.config_bar.status.configure(
                text=f"character {char_id or ''} removed")

    def _refresh_char_menu(self):
        ids = [c.char_id for c in self.manager.list(include_archived=False)
               if c.valid]
        self.config_bar.char_menu["values"] = ids
        self.config_bar.character.set(self.active)

    def _set_editing_buttons(self, enabled):
        state = "normal" if enabled else "disabled"
        self.config_bar.edit_btn.configure(state=state)
        self.config_bar.new_btn.configure(state=state)

    # ---- restart-as-new on switch (M8 definition vs instance) ----
    def _restart_choice(self, char_id):
        """If char_id has a saved instance whose stored definition_hash differs
        from the file's, ask Keep talking / Restart as new / Cancel. Returns
        'keep' | 'restart' | None (cancel). Legacy instances (stored hash '')
        are never badged (M8 lock 4)."""
        summary = next(
            (c for c in self.manager.list(include_archived=False)
             if c.char_id == char_id), None)
        if summary is None or not summary.valid or not summary.has_save:
            return "keep"
        stored = self._stored_definition_hash(char_id)
        if not (summary.definition_hash and stored
                and summary.definition_hash != stored):
            return "keep"
        return self._ask_restart(summary.name or char_id)

    def _stored_definition_hash(self, char_id):
        raw = self.manager.store.load_state(char_id)
        if raw is None:
            return ""
        try:
            return json.loads(raw).get("definition_hash", "") or ""
        except ValueError:
            return ""

    def _ask_restart(self, name):
        result = {"value": None}

        def finish(value):
            result["value"] = value
            dlg.destroy()

        dlg = tk.Toplevel(self.root)
        dlg.title("Definition changed")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)
        dlg.bind("<Escape>", lambda e: finish(None))
        ttk.Label(dlg, wraplength=380, justify="left",
                  text=(f"{name}'s definition was edited since you met. Keep "
                        "talking leaves the current companion and memories "
                        "untouched; restart as new replaces them with a fresh "
                        "companion from the updated definition.")).pack(
                        padx=16, pady=(16, 12))
        btns = ttk.Frame(dlg)
        btns.pack(pady=(0, 12))
        ttk.Button(btns, text="Cancel",
                   command=lambda: finish(None)).pack(side="left", padx=4)
        ttk.Button(btns, text="Keep talking",
                   command=lambda: finish("keep")).pack(side="left", padx=4)
        ttk.Button(btns, text="Restart as new",
                   command=lambda: finish("restart")).pack(side="left", padx=4)
        dlg.wait_window(dlg)
        return result["value"]

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
                elif item[0] == "switched":
                    _, char_id, gap = item
                    self.config_bar.bind_session(self.session)
                    self.root.title(f"Companion — {self.session.state.name}")
                    self.chat.transcript.configure(state="normal")
                    self.chat.transcript.delete("1.0", "end")
                    self.chat.transcript.insert(
                        "end", f"{self.session.state.name} is awake "
                               f"(gap {gap:.2f}h). Type /help for methods.\n\n")
                    self.chat.transcript.configure(state="disabled")
                    self.switching = False
                    self.config_bar.character.set(self.active)
                    self._set_editing_buttons(True)
                    self._refresh_all()
                elif item[0] == "error":
                    self.switching = False
                    self._set_editing_buttons(True)
                    self.config_bar.character.set(self.active)
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
