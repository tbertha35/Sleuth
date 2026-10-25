// Playground page: prompt a model and watch the tokens arrive; see what it
// "thinks" comes next; compare two models (or two snapshots) side by side.
//
// Models are loaded inside the dashboard's process on first use and stay
// loaded (at most two), so only the first request is slow. Unload frees them,
// including the GPU for jobs.
import {
  $, api, el, fillSelect, guarded, hasToken, labeled, post, postStream, statusLine, visible,
} from "./common.js";

const st = { runs: [], mode: "text", chats: [[], []], running: null, outputs: ["", ""] };
const setStatus = statusLine($("play-status"));
const SETTINGS = [
  // [id, label, default, step, help]
  ["max_new_tokens", "Max new tokens", 200, 1, "stop after this many tokens"],
  ["temperature", "Temperature", 0.8, 0.05, "0 = greedy (always the most likely token)"],
  ["top_k", "Top-k", 0, 1, "keep only the k most likely tokens (0 = off)"],
  ["top_p", "Top-p", 1.0, 0.01, "keep the smallest set reaching this probability (1 = off)"],
  ["min_p", "Min-p", 0.05, 0.01, "drop tokens below this fraction of the best one's probability"],
  ["repetition_penalty", "Repetition penalty", 1.0, 0.05, "above 1 discourages recently used tokens"],
  ["repeat_last_n", "Penalty window", 64, 1, "how many recent tokens the penalty looks at (0 = all)"],
  ["no_repeat_ngram", "No-repeat n-gram", 0, 1, "never repeat a phrase of this many tokens (3-4 breaks loops; 0 = off)"],
  ["seed", "Seed", "", 1, "same seed + same settings = same text (empty = random)"],
];

export function initPlayground() {
  const box = $("play-settings");
  for (const [id, label, def, step, help] of SETTINGS) {
    box.append(labeled(label, el("input", { type: "number", id: `ps-${id}`, value: def, step, min: 0 }), help));
  }
  for (const slot of [0, 1]) {
    $(`pm-model-${slot}`).addEventListener("change", guarded(setStatus, () => loadSteps(slot)));
  }
  $("play-compare").addEventListener("change", () => {
    for (const n of document.querySelectorAll(".slot-1")) n.hidden = !$("play-compare").checked;
    renderOutputs();
  });
  for (const b of document.querySelectorAll("[data-mode]")) {
    b.addEventListener("click", () => setMode(b.dataset.mode));
  }
  $("play-go").addEventListener("click", guarded(setStatus, go));
  $("play-stop").addEventListener("click", () => st.running?.abort());
  $("play-clear").addEventListener("click", () => { st.chats = [[], []]; st.outputs = ["", ""]; renderOutputs(); });
  $("play-next").addEventListener("click", guarded(setStatus, nextTokens));
  $("play-append").addEventListener("click", () => {
    $("play-prompt").value += st.outputs[0];
    st.outputs = ["", ""];
    renderOutputs();
  });
  $("play-unload").addEventListener("click", guarded(setStatus, async () => {
    renderLoaded(await post("playground/unload"));
    setStatus("Unloaded: the GPU is free for jobs again.");
  }));
  const submitKeys = (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); guarded(setStatus, go)(); } };
  $("play-prompt").addEventListener("keydown", submitKeys);
  $("play-user").addEventListener("keydown", submitKeys);
  setMode("text");
}

export async function showPlayground(runs) {
  st.runs = runs.filter((r) => !r.stale);
  for (const slot of [0, 1]) {
    const input = $(`pm-model-${slot}`);
    $(`pm-list`).replaceChildren(...st.runs.map((r) => el("option", { value: r.id })));
    if (!input.value && st.runs.length) {
      input.value = (st.runs.find((r) => !r.external) || st.runs[0]).id;
      await loadSteps(slot);
    }
  }
  renderLoaded(await api("playground"));
  if (!hasToken()) setStatus("Read-only: open the link that `sleuth ui` printed (it ends in ?token=…) to use the Playground.", true);
}

function setMode(mode) {
  st.mode = mode;
  for (const b of document.querySelectorAll("[data-mode]")) b.setAttribute("aria-pressed", String(b.dataset.mode === mode));
  $("play-text").hidden = mode !== "text";
  $("play-chat").hidden = mode !== "chat";
  $("play-go").textContent = mode === "chat" ? "Send" : "Generate";
  $("play-append").hidden = mode !== "text";
  renderOutputs();
}

async function loadSteps(slot) {
  const name = $(`pm-model-${slot}`).value.trim();
  const run = st.runs.find((r) => r.id === name && !r.external);
  const sel = $(`pm-step-${slot}`);
  if (!run) { fillSelect(sel, [["", "–"]]); sel.disabled = true; return; }
  const detail = await api(`runs/${encodeURIComponent(name)}`);
  sel.disabled = false;
  const snaps = [...detail.snapshots].reverse();
  fillSelect(sel, [["", `latest (${snaps[0] ?? "–"})`], ...snaps.map((s) => [s, `step ${s}`])]);
}

function modelSpec(slot) {
  return {
    model: $(`pm-model-${slot}`).value.trim(), step: $(`pm-step-${slot}`).value || null,
    device: $(`pm-device-${slot}`).value, dtype: $(`pm-dtype-${slot}`).value,
  };
}

function settings() {
  const out = {};
  for (const [id] of SETTINGS) {
    const v = $(`ps-${id}`).value;
    if (v !== "") out[id] = Number(v);
  }
  return out;
}

const slots = () => ($("play-compare").checked ? [0, 1] : [0]);

// ------------------------------------------------------------------ generate
async function go() {
  if (st.running) return;
  let user = null;
  if (st.mode === "chat") {
    user = $("play-user").value.trim();
    if (!user) return;
    $("play-user").value = "";
    for (const s of slots()) st.chats[s].push({ role: "user", content: user });
  }
  st.outputs = ["", ""];
  renderOutputs();
  const ctrl = new AbortController();
  st.running = ctrl;
  $("play-stop").disabled = false;
  $("play-go").disabled = true;
  try {
    // One model after the other: they share the GPU, and streaming both at
    // once would only halve each one's speed.
    for (const s of slots()) {
      if (ctrl.signal.aborted) break;
      await generateInto(s, ctrl);
    }
  } finally {
    st.running = null;
    $("play-stop").disabled = true;
    $("play-go").disabled = false;
    renderLoaded(await api("playground").catch(() => null));
  }
}

async function generateInto(slot, ctrl) {
  const body = { model: modelSpec(slot), mode: st.mode, ...settings() };
  if (st.mode === "chat") {
    const system = $("play-system").value.trim();
    body.messages = (system ? [{ role: "system", content: system }] : []).concat(st.chats[slot]);
    body.think = $("play-think").checked;
  } else {
    body.prompt = $("play-prompt").value;
  }
  const stats = $(`play-stats-${slot}`);
  stats.textContent = "…";
  try {
    await postStream("playground/generate", body, (name, data) => {
      if (name === "token") { st.outputs[slot] += data.text; renderOutput(slot); }
      else if (name === "status") stats.textContent = data.text;
      else if (name === "done") stats.textContent = statsText(data);
      else if (name === "error") { stats.textContent = ""; throw new Error(data.error); }
    }, ctrl.signal);
  } catch (e) {
    if (e.name !== "AbortError") { setStatus(String(e.message || e), true); }
    else stats.textContent = "stopped";
    if (e.name !== "AbortError" && st.mode === "chat") st.chats[slot].pop(); // the user turn got no answer
  }
  if (st.mode === "chat" && st.outputs[slot]) st.chats[slot].push({ role: "assistant", content: st.outputs[slot] });
  if (st.mode === "chat") { st.outputs[slot] = ""; renderOutputs(); }
}

function statsText(d) {
  return `${d.model}: ${d.tokens} tokens (${d.stop_reason.replace(/_/g, " ")}), prompt ${d.prompt_tokens} tokens in ` +
    `${d.prompt_ms} ms, ${d.tok_per_s} tok/s` + (d.peak_gb !== undefined ? `, peak GPU memory ${d.peak_gb} GiB` : "");
}

// ------------------------------------------------------------------ output
function renderOutputs() {
  for (const s of [0, 1]) renderOutput(s);
}

function renderOutput(slot) {
  const box = $(`play-out-${slot}`);
  if (st.mode === "text") {
    box.className = "out";
    box.replaceChildren(el("span", { class: "prompt-echo", text: $("play-prompt").value }), st.outputs[slot]);
    return;
  }
  box.className = "out chat";
  const msgs = st.chats[slot].map((m) => el("div", { class: `bubble ${m.role}` },
    el("div", { class: "role", text: m.role }), m.content));
  if (st.outputs[slot]) {
    msgs.push(el("div", { class: "bubble assistant" }, el("div", { class: "role", text: "assistant" }), st.outputs[slot]));
  }
  box.replaceChildren(...msgs);
  box.scrollTop = box.scrollHeight;
}

function renderLoaded(state) {
  if (!state) return;
  const list = $("play-loaded");
  list.replaceChildren(...(state.models.length ? state.models.map((m) =>
    el("span", { class: "pill", title: `${m.format}, ${(m.params / 1e6).toFixed(1)}M params, loaded in ${m.load_seconds} s` +
      (m.chat_format ? `, chat format ${m.chat_format}` : ", no chat format (base model)") },
    `${m.label} · ${m.device}`)) : [el("span", { class: "muted", text: "no model loaded yet (the first request loads it)" })]));
  $("play-unload").disabled = !state.models.length;
  $("play-gpu").textContent = state.gpu_holder === "playground"
    ? "The Playground holds the GPU: GPU jobs wait until you unload." : "";
}

// ------------------------------------------------------------------ next token
async function nextTokens() {
  const box = $("play-next-out");
  box.replaceChildren(el("p", { class: "hint", text: "Asking…" }));
  const cols = [];
  for (const s of slots()) {
    const body = { model: modelSpec(s), mode: st.mode, top: 10 };
    if (st.mode === "chat") {
      const system = $("play-system").value.trim();
      const pending = $("play-user").value.trim();
      body.messages = (system ? [{ role: "system", content: system }] : []).concat(st.chats[s],
        pending ? [{ role: "user", content: pending }] : []);
      if (!body.messages.length) throw new Error("type a message first");
    } else {
      body.prompt = $("play-prompt").value + st.outputs[s];
    }
    cols.push(nextTable(await post("playground/next", body), s));
  }
  box.replaceChildren(el("div", { class: "two-col" }, ...cols));
}

function nextTable(r, slot) {
  const t = el("table", { class: "next" });
  t.append(el("tr", {}, el("th", { text: "#" }), el("th", { text: "token" }), el("th", { text: "probability" }),
    el("th", { text: "logit" })));
  r.candidates.forEach((c, i) => {
    const bar = el("span", { class: "bar" });
    bar.style.width = `${Math.max(1, c.prob * 100)}%`;
    const tr = el("tr", { title: st.mode === "text" ? "click to add this token to the prompt" : "" },
      el("td", { text: String(i + 1) }), el("td", {}, el("code", { text: visible(c.text, 30) })),
      el("td", { class: "probcell" }, el("span", { class: "barwrap" }, bar), ` ${(c.prob * 100).toFixed(1)}%`),
      el("td", { text: c.logit.toFixed(2) }));
    if (st.mode === "text") {
      tr.classList.add("clickable");
      tr.addEventListener("click", () => {
        if (st.outputs[slot]) { $("play-prompt").value += st.outputs[slot]; st.outputs = ["", ""]; }
        $("play-prompt").value += c.text;
        renderOutputs();
        guarded(setStatus, nextTokens)();
      });
    }
    t.append(tr);
  });
  return el("div", {},
    el("h3", { text: r.model }),
    el("p", { class: "hint", text: `after “${visible(r.last_tokens.join(""), 40)}” (${r.prompt_tokens} tokens). ` +
      `Entropy ${r.entropy.toFixed(2)} nats: 0 = certain, ${Math.log(10).toFixed(1)} ≈ torn between 10 tokens.` }),
    el("div", { class: "table-wrap" }, t));
}
