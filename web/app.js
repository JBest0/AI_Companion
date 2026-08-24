/* Companion GUI — vanilla JS, no build step. Talks to server.py. */

const $ = (sel) => document.querySelector(sel);

/* ---------- diagnostics (dev aid) ---------- */
window.__guiErrors = [];
window.addEventListener("error", (e) => window.__guiErrors.push("error: " + String(e.message)));
window.addEventListener("unhandledrejection", (e) =>
  window.__guiErrors.push("rejection: " + String((e.reason && e.reason.message) || e.reason)));

function reportDiag() {
  try {
    const rect = (sel) => {
      const el = document.querySelector(sel);
      if (!el) return null;
      const b = el.getBoundingClientRect();
      return { x: Math.round(b.x), w: Math.round(b.width),
               onScreen: b.width > 0 && b.x >= -1 && b.x + b.width <= innerWidth + 1 };
    };
    const app = document.querySelector("#app");
    const diag = {
      url: location.href, time: new Date().toISOString(),
      innerWidth, innerHeight, dpr: devicePixelRatio,
      grid: app ? getComputedStyle(app).gridTemplateColumns : null,
      css: [...document.styleSheets].map((s) => s.href).filter(Boolean),
      left: rect("#left"), chat: rect("#chat-wrap"), right: rect("#right"),
      asides: document.querySelectorAll("aside").length,
      msgs: document.querySelectorAll("#chat .msg").length,
      errors: window.__guiErrors.slice(0, 5),
    };
    fetch("/api/diag", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(diag) }).catch(() => {});
  } catch (e) { /* never break the page for diagnostics */ }
}

/* ---------- state panel ---------- */
function renderState(st) {
  $("#companion-name").textContent = st.name;
  $("#narrative").textContent = st.narrative || "";

  const bars = [
    ["trust", st.relationship.trust],
    ["intimacy", st.relationship.intimacy],
    ["resentment", st.relationship.resentment],
    ["arousal", st.affect.arousal],
  ];
  for (const [k, v] of bars) {
    $("#bar-" + k).style.width = (v * 100).toFixed(1) + "%";
    $("#val-" + k).textContent = v.toFixed(2);
  }
  // valence is -1..1, anchored at the center of its bar
  const v = st.affect.valence;
  const fill = $("#bar-valence");
  fill.style.width = Math.abs(v * 50).toFixed(1) + "%";
  fill.style.left = v >= 0 ? "50%" : (50 + v * 50).toFixed(1) + "%";
  fill.style.background = v >= 0 ? "var(--good)" : "var(--bad)";
  $("#val-valence").textContent = v.toFixed(2);

  const ph = $("#phases");
  ph.innerHTML = "";
  if (!st.active_phases.length) {
    ph.innerHTML = '<span class="dim">none</span>';
  } else {
    for (const p of st.active_phases) {
      const b = document.createElement("span");
      b.className = "badge " + p;
      b.textContent = p;
      ph.appendChild(b);
    }
  }
}

/* ---------- chat ---------- */
function addMessage(role, text, metaHtml) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.textContent = text;
  if (metaHtml) {
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML = metaHtml;
    div.appendChild(meta);
  }
  $("#chat").appendChild(div);
  $("#chat").scrollTop = $("#chat").scrollHeight;
  return div;
}

$("#composer").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("#input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  addMessage("user", text);
  const pending = addMessage("companion", "…");

  const res = await fetch("/api/turn", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({input: text}),
  }).then(r => r.json());

  pending.remove();
  if (res.error && !res.trace) {
    addMessage("error", res.response);           // invalid method etc.
    return;
  }
  const a = res.trace.activation;
  const sign = a.impact >= 0 ? "+" : "";
  let meta = `<b>${a.archetype}</b> impact ${sign}${a.impact.toFixed(2)}`;
  if (res.trace.fallback) meta += " · <i>fallback line (LLM offline)</i>";
  const msg = addMessage("companion", res.response, meta);
  msg.style.cursor = "pointer";
  msg.title = "click to inspect this turn";
  msg.onclick = () => showTrace(res.trace.turn_id);

  renderState(res.state);
  loadTraces();
  loadMemories();
});

/* ---------- traces ---------- */
async function loadTraces() {
  const list = await fetch("/api/traces").then(r => r.json());
  const el = $("#trace-list");
  el.innerHTML = "";
  for (const t of list) {
    const div = document.createElement("div");
    div.className = "trace-item";
    const cls = t.impact >= 0 ? "pos" : "neg";
    const sign = t.impact >= 0 ? "+" : "";
    div.innerHTML =
      `<span class="impact ${cls}">${sign}${t.impact.toFixed(2)}</span> ` +
      `${t.archetype}` +
      (t.active_phases.length ? ` · ${t.active_phases.join(", ")}` : "") +
      `<div class="input">${escapeHtml(t.user_input)}</div>`;
    div.onclick = () => showTrace(t.turn_id);
    el.appendChild(div);
  }
}

async function showTrace(turnId) {
  document.querySelector('[data-tab="traces"]').click();
  const t = await fetch("/api/trace/" + turnId).then(r => r.json());
  const d = $("#trace-detail");
  $("#trace-list").classList.add("hidden");
  d.classList.remove("hidden");

  const rows = (obj) => Object.entries(obj)
    .filter(([k]) => k !== "schema_version")
    .map(([k, v]) => `<tr><td>${k}</td><td>${typeof v === "number" ? (+v).toFixed(3) : v}</td></tr>`)
    .join("");

  const a = t.activation;
  const contribs = a.contributions.map(c =>
    `<tr><td>${c.trait_id}</td><td>${(c.impact >= 0 ? "+" : "") + c.impact.toFixed(3)}</td></tr>`).join("");
  const mems = t.retrieved_memories.map(m =>
    `<tr><td title="${escapeHtml(m.content)}">${escapeHtml(m.content.slice(0, 42))}…</td>
     <td>${m.score.toFixed(3)}</td></tr>`).join("");

  d.innerHTML = `
    <button class="back">← all traces</button>
    <h3>Turn</h3>
    <table>
      <tr><td>input</td><td>${escapeHtml(t.user_input)}</td></tr>
      <tr><td>archetype</td><td>${a.archetype}</td></tr>
      <tr><td>impact</td><td>${a.impact.toFixed(3)}${a.ambivalent ? " (ambivalent)" : ""}</td></tr>
      <tr><td>phases during turn</td><td>${(t.active_phases || []).join(", ") || "—"}</td></tr>
      <tr><td>fallback</td><td>${t.fallback}</td></tr>
      <tr><td>latency</td><td>${t.latency_ms.total} ms</td></tr>
    </table>
    <h3>Contributions</h3>
    <table>${contribs || '<tr><td class="dim">none</td><td></td></tr>'}</table>
    <h3>Voice before → after</h3>
    <table>${rows(t.voice_after)}</table>
    <h3>Affect after</h3>
    <table>${rows(t.affect_after)}</table>
    <h3>Relationship after</h3>
    <table>${rows(t.relationship_after)}</table>
    <h3>Retrieved memories</h3>
    <table>${mems || '<tr><td class="dim">none</td><td></td></tr>'}</table>`;
  d.querySelector(".back").onclick = () => {
    d.classList.add("hidden");
    $("#trace-list").classList.remove("hidden");
  };
}

/* ---------- memories ---------- */
async function loadMemories() {
  const list = await fetch("/api/memories").then(r => r.json());
  const el = $("#memory-list");
  el.innerHTML = "";
  for (const m of list.slice().reverse()) {     // newest first
    const div = document.createElement("div");
    div.className = "mem";
    const when = new Date(m.created_at * 1000).toLocaleString();
    div.innerHTML =
      `<span class="kind ${m.kind}">${m.kind}</span> ` +
      `<span class="dim">${when} · accessed ×${m.access_count}</span>` +
      `<div>${escapeHtml(m.content)}</div>` +
      (m.emotional_tags.length
        ? `<div class="tags">${m.emotional_tags.map(t => `<span class="tag">${t}</span>`).join("")}</div>`
        : "") +
      `<div class="sal-bar"><div class="sal-fill" style="width:${(m.effective_salience * 100).toFixed(0)}%"></div></div>` +
      `<div class="dim">salience ${m.salience.toFixed(2)} → now ${m.effective_salience.toFixed(2)}</div>`;
    el.appendChild(div);
  }
}

/* ---------- reflections ---------- */
async function loadReflections() {
  const list = await fetch("/api/reflections").then(r => r.json());
  const el = $("#reflection-list");
  el.innerHTML = list.length ? "" : '<span class="dim">No reflections yet — she dreams when a session closes with enough new memories.</span>';
  for (const r of list) {
    const ap = r.applied && typeof r.applied === "object" ? r.applied : {};
    const div = document.createElement("div");
    div.className = "refl" + (r.rolled_back ? " rolled" : "");
    const when = new Date(r.created_at * 1000).toLocaleString();
    div.innerHTML =
      `<div class="dim">${when}${r.rolled_back ? " · ROLLED BACK" : ""}</div>` +
      `<div>insights: ${ap.insight_memory_ids.length} · drifts: ${ap.drifts.length} · narrative: ${ap.narrative_added}</div>` +
      ap.drifts.map(d =>
        `<div class="dim">${d.trait_id}: ${d.current} → ${d.proposed}</div>`).join("");
    el.appendChild(div);
  }
}

/* ---------- config ---------- */
async function openConfig() {
  const cfg = await fetch("/api/config").then(r => r.json());
  window._cfgAvailable = cfg.available;
  $("#cfg-provider").value = cfg.provider;
  $("#cfg-model").value = cfg.model || "";
  $("#cfg-base-url").value = cfg.base_url || "";
  updateConfigWarnings();
  $("#config-modal").classList.remove("hidden");
}

function updateConfigWarnings() {
  const p = $("#cfg-provider").value;
  const custom = p === "custom";
  $("#cfg-base-url-label").style.display = custom ? "" : "none";
  $("#cfg-base-url").style.display = custom ? "" : "none";

  let msg = "";
  if (p === "openai" && window._cfgAvailable && !window._cfgAvailable.OPENAI_API_KEY) {
    msg = "OPENAI_API_KEY is not set — add it to .env to use OpenAI.";
  } else if (p === "deepseek" && window._cfgAvailable && !window._cfgAvailable.DEEPSEEK_API_KEY) {
    msg = "DEEPSEEK_API_KEY is not set — add it to .env to use DeepSeek.";
  } else if (custom && (!$("#cfg-model").value.trim() || !$("#cfg-base-url").value.trim())) {
    msg = "Custom provider needs a model and a base URL.";
  }
  const warn = $("#cfg-warning");
  warn.textContent = msg;
  warn.classList.toggle("hidden", !msg);
}

function closeConfig() {
  $("#config-modal").classList.add("hidden");
}

$("#settings-btn").addEventListener("click", openConfig);
$("#cfg-provider").addEventListener("change", updateConfigWarnings);
$("#cfg-model").addEventListener("input", updateConfigWarnings);
$("#cfg-base-url").addEventListener("input", updateConfigWarnings);
$("#cfg-cancel").addEventListener("click", closeConfig);
$("#config-modal").addEventListener("click", (e) => {
  if (e.target === $("#config-modal")) closeConfig();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("#config-modal").classList.contains("hidden")) closeConfig();
});

$("#cfg-save").addEventListener("click", async () => {
  const body = {
    provider: $("#cfg-provider").value,
    model: $("#cfg-model").value.trim(),
    base_url: $("#cfg-base-url").value.trim(),
  };
  const res = await fetch("/api/config", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  }).then(r => r.json());

  const warn = $("#cfg-warning");
  if (res.error) {
    warn.textContent = res.error;
    warn.classList.remove("hidden");
    return;
  }
  window._cfgAvailable = res.config.available;
  warn.textContent = res.config.warning || "";
  warn.classList.toggle("hidden", !res.config.warning);
  if (!res.config.warning) closeConfig();
});

/* ---------- tabs & boot ---------- */
document.querySelectorAll("#tabs button").forEach(btn =>
  btn.addEventListener("click", () => {
    document.querySelectorAll("#tabs button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    $("#panel-" + btn.dataset.tab).classList.add("active");
  }));

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, c =>
    ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
}

(async () => {
  const st = await fetch("/api/state").then(r => r.json());
  renderState(st);

  // restore the recent conversation from traces (newest-first -> flip)
  const history = await fetch("/api/traces").then(r => r.json());
  for (const t of history.slice().reverse()) {
    addMessage("user", t.user_input);
    const sign = t.impact >= 0 ? "+" : "";
    let meta = `<b>${t.archetype}</b> impact ${sign}${t.impact.toFixed(2)}`;
    if (t.fallback) meta += " · <i>fallback line (LLM offline)</i>";
    const msg = addMessage("companion", t.response, meta);
    msg.style.cursor = "pointer";
    msg.title = "click to inspect this turn";
    msg.onclick = () => showTrace(t.turn_id);
  }

  loadTraces();
  loadMemories();
  loadReflections();
  $("#input").focus();

  // self-report layout to the server (right after boot, and again later to
  // catch any post-load "disappear")
  reportDiag();
  setTimeout(reportDiag, 4000);
})();
