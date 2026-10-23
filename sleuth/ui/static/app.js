// Sleuth dashboard: fetches JSON from the local server and draws it.
import { emptyState, fmtNum, heatmap, lineChart, ordinalColors, tokens } from "./charts.js";

const $ = (id) => document.getElementById(id);

// Per-layer metrics produced by analyzer/neurons.py, with plain-language help.
const METRICS = {
  frac_selective: ["Category-selective neurons", "Share of neurons whose firing concentrates in one test category (e.g. fires on code but rarely on stories). Rising = neurons specializing."],
  fire_rate_mean: ["Mean firing rate", "Average fraction of tokens on which a neuron fires (|activation| above its layer's RMS)."],
  frac_low_activity: ["Nearly-dead neurons", "Share of neurons that fire on fewer than 1% of tokens. Some is normal (sparse features); a large share can mean wasted capacity."],
  frac_high_activity: ["Always-on neurons", "Share of neurons firing on more than half of all tokens."],
  act_rms: ["MLP activation size (RMS)", "Typical magnitude of the MLP hidden activations in each layer."],
  mlp_out_rms: ["MLP output size", "How much each layer's MLP writes into the residual stream."],
  attn_out_rms: ["Attention output size", "How much each layer's attention writes into the residual stream."],
  resid_rms: ["Residual stream size", "Magnitude of the running residual stream after each layer. Usually grows with depth."],
};

const state = { runs: [], run: null, metrics: null, timeline: null, ckpt: null };

async function api(path) {
  const r = await fetch(`/api/${path}`);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

function setStatus(msg, isError = false) {
  const s = $("status");
  s.textContent = msg;
  s.classList.toggle("error", isError);
}

function fillSelect(sel, options, value) {
  sel.replaceChildren();
  for (const [v, label] of options) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = label;
    sel.appendChild(o);
  }
  if (value !== undefined && options.some(([v]) => String(v) === String(value))) sel.value = value;
}

const stepDir = (s) => `step_${String(s).padStart(7, "0")}`;

// ------------------------------------------------------------------ loading
async function loadRuns() {
  state.runs = await api("runs");
  if (!state.runs.length) {
    setStatus("No runs yet. Train one with: python -m sleuth train configs/smoke-cpu.toml");
    return;
  }
  const wanted = new URLSearchParams(location.search).get("run");
  fillSelect($("run-select"), state.runs.map((r) => [r.id, `${r.id}  (step ${r.step}/${r.max_steps})`]),
    wanted || state.runs[0].id);
  await loadRun($("run-select").value);
}

async function loadRun(id) {
  setStatus("Loading…");
  const [run, metrics] = await Promise.all([api(`runs/${id}`), api(`runs/${id}/metrics`)]);
  state.run = run;
  state.metrics = metrics;
  state.timeline = null;
  state.ckpt = null;
  const suites = run.analyses.filter((a) => a.view === "neuron_stats");
  fillSelect($("suite-select"), suites.map((a) => [a.suite, a.suite]));
  renderTiles();
  renderLoss();
  if (suites.length) {
    await loadSuite(suites[0].suite);
    setStatus("");
  } else {
    for (const id of ["timeline-chart", "timeline-heat", "neuron-heat"]) {
      emptyState($(id), `No analysis yet. Run: python -m sleuth analyze runs/${run.id} --suite suites/basic`);
    }
    $("layer-table").replaceChildren();
    setStatus("");
  }
}

async function loadSuite(suite) {
  const run = state.run;
  state.timeline = await api(`runs/${run.id}/analysis/neuron_stats/${suite}/timeline`);
  const t = state.timeline;
  const metricOpts = Object.entries(METRICS).map(([k, [label]]) => [k, label]);
  for (const c of t.categories) metricOpts.push([`cat:${c}`, `Firing rate on “${c}” tests`]);
  fillSelect($("metric-select"), metricOpts, $("metric-select").value || "frac_selective");
  fillSelect($("ckpt-select"), t.steps.map((s, i) => [s, `step ${s}  (${fmtNum(t.tokens[i])} tokens)`]),
    t.steps[t.steps.length - 1]);
  fillSelect($("cat-select"), [["all", "all categories"], ...t.categories.map((c) => [c, c])]);
  renderTimeline();
  await loadCheckpoint();
}

async function loadCheckpoint() {
  const run = state.run, suite = $("suite-select").value, step = $("ckpt-select").value;
  if (!step) return;
  state.ckpt = await api(`runs/${run.id}/analysis/neuron_stats/${suite}/${stepDir(step)}`);
  renderNeurons();
}

// ------------------------------------------------------------------ rendering
function renderTiles() {
  const { run, metrics } = state;
  const last = metrics.train[metrics.train.length - 1] || {};
  const lastVal = metrics.val[metrics.val.length - 1] || {};
  const p = run.params;
  const tiles = [
    ["Parameters", p ? fmtNum(p.non_embedding) : "–", p ? `non-embedding (+${fmtNum(p.embedding)} embedding)` : ""],
    ["Step", `${last.step ?? 0}`, `of ${run.config.train.max_steps}`],
    ["Tokens seen", fmtNum(last.tokens ?? 0), `${fmtNum(last.tok_per_s ?? 0)} tok/s last`],
    ["Train loss", last.loss !== undefined ? last.loss.toFixed(3) : "–", "nats/token"],
    ["Val bits/byte", lastVal.val_bpb !== undefined ? lastVal.val_bpb.toFixed(3) : "–", "lower is better"],
    ["Snapshots", `${run.snapshots.length}`, run.env ? `${run.env.hostname}${run.env.gpus?.length ? " · " + run.env.gpus[0] : " · CPU"}` : ""],
  ];
  const root = $("tiles");
  root.replaceChildren();
  for (const [label, value, sub] of tiles) {
    const d = document.createElement("div");
    d.className = "tile";
    const l = document.createElement("div"); l.className = "label"; l.textContent = label;
    const v = document.createElement("div"); v.className = "value"; v.textContent = value;
    const s = document.createElement("div"); s.className = "sub"; s.textContent = sub;
    d.append(l, v, s);
    root.appendChild(d);
  }
}

function renderLoss() {
  const T = tokens();
  const { train, val } = state.metrics;
  lineChart($("loss-chart"), [
    { name: "train loss", color: T.series[0], points: train.map((r) => [r.step, r.loss]) },
    { name: "validation loss", color: T.series[1], points: val.map((r) => [r.step, r.val_loss]) },
  ], { xLabel: "step", yLabel: "loss", yFormat: (v) => v.toFixed(2), empty: "No metrics logged yet" });
}

function metricSeries(key) {
  const t = state.timeline;
  if (key.startsWith("cat:")) return t.fire_rate_by_category[key.slice(4)];
  return t.metrics[key];
}

function renderTimeline() {
  const t = state.timeline;
  if (!t) return;
  const key = $("metric-select").value;
  const [label, help] = key.startsWith("cat:")
    ? [`Firing rate on “${key.slice(4)}”`, "Average firing rate of each layer's neurons, on this category's tests only."]
    : METRICS[key];
  $("metric-help").textContent = help;
  const vals = metricSeries(key);  // [checkpoint][layer]
  const L = t.n_layers;
  const colors = ordinalColors(L);
  const isFrac = key.startsWith("frac") || key.startsWith("cat:") || key === "fire_rate_mean";
  const fmt = isFrac ? (v) => `${(v * 100).toFixed(1)}%` : fmtNum;
  lineChart($("timeline-chart"),
    Array.from({ length: L }, (_, l) => ({
      name: `layer ${l}`, color: colors[l], points: t.steps.map((s, i) => [s, vals[i][l]]),
    })),
    { xLabel: "step", yLabel: label, logX: $("logx").checked, yFormat: fmt });
  const matrix = Array.from({ length: L }, (_, l) => t.steps.map((_, i) => vals[i][l]));
  heatmap($("timeline-heat"), matrix, {
    rowLabels: matrix.map((_, l) => `L${l}`), colLabels: t.steps.map(String),
    xLabel: "checkpoint (step)", format: fmt, legendLabel: label,
    tipTitle: (r, c) => `layer ${r}, step ${t.steps[c]}`,
  });
}

function renderNeurons() {
  const d = state.ckpt;
  if (!d) return;
  const cat = $("cat-select").value;
  const cats = d.summary.categories;
  let rows = cat === "all" ? d.arrays.fire_rate : d.arrays.fire_rate_by_category.map((byCat) => byCat[cats.indexOf(cat)]);
  const sort = $("sort-neurons").checked;
  const order = rows.map((row) => {
    const idx = row.map((_, i) => i);
    return sort ? idx.sort((a, b) => row[b] - row[a]) : idx;
  });
  const matrix = rows.map((row, l) => order[l].map((i) => row[i]));
  heatmap($("neuron-heat"), matrix, {
    rowLabels: matrix.map((_, l) => `L${l}`),
    colLabels: matrix[0].map((_, i) => (sort ? `#${i + 1}` : String(i))),
    xLabel: sort ? "neurons, most active first" : "neuron index",
    format: (v) => `${(v * 100).toFixed(1)}%`, legendLabel: "firing rate", domain: [0, Math.max(1e-6, ...matrix.flat())],
    tipTitle: (r, c) => `layer ${r}, neuron ${order[r][c]}`,
  });
  renderTable(d.summary);
}

function renderTable(summary) {
  const cols = [
    ["layer", "layer", (v) => v],
    ["fire_rate_mean", "mean firing", (v) => `${(v * 100).toFixed(1)}%`],
    ["frac_low_activity", "nearly dead", (v) => `${(v * 100).toFixed(1)}%`],
    ["frac_high_activity", "always on", (v) => `${(v * 100).toFixed(1)}%`],
    ["frac_selective", "selective", (v) => `${(v * 100).toFixed(1)}%`],
    ["act_rms", "act RMS", (v) => v.toFixed(3)],
    ["attn_out_rms", "attn out", (v) => v.toFixed(3)],
    ["mlp_out_rms", "MLP out", (v) => v.toFixed(3)],
    ["resid_rms", "residual", (v) => v.toFixed(3)],
  ];
  const table = $("layer-table");
  table.replaceChildren();
  const head = table.insertRow();
  for (const [, label] of cols) { const th = document.createElement("th"); th.textContent = label; head.appendChild(th); }
  for (const lay of summary.layers) {
    const tr = table.insertRow();
    for (const [k, , f] of cols) tr.insertCell().textContent = f(lay[k]);
  }
}

function renderAll() {
  if (!state.run) return;
  renderLoss();
  renderTimeline();
  renderNeurons();
}

// ------------------------------------------------------------------ wiring
function guard(fn) {
  return (...args) => fn(...args).catch((e) => setStatus(String(e.message || e), true));
}

$("run-select").addEventListener("change", guard(async (e) => {
  history.replaceState(null, "", `?run=${encodeURIComponent(e.target.value)}`);
  await loadRun(e.target.value);
}));
$("suite-select").addEventListener("change", guard((e) => loadSuite(e.target.value)));
$("ckpt-select").addEventListener("change", guard(loadCheckpoint));
$("metric-select").addEventListener("change", renderTimeline);
$("logx").addEventListener("change", renderTimeline);
$("cat-select").addEventListener("change", renderNeurons);
$("sort-neurons").addEventListener("change", renderNeurons);
$("refresh").addEventListener("click", guard(async () => {
  const keep = $("run-select").value;
  await loadRuns();
  if (keep) { $("run-select").value = keep; await loadRun(keep); }
}));
$("theme").addEventListener("click", () => {
  const dark = getComputedStyle(document.documentElement).getPropertyValue("--surface").trim() === "#1a1a19";
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("sleuth-theme", document.documentElement.dataset.theme); } catch { /* storage may be blocked */ }
  renderAll();
});
try {
  const saved = localStorage.getItem("sleuth-theme");
  if (saved) document.documentElement.dataset.theme = saved;
} catch { /* ignore */ }
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
let resizeTimer;
addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderAll, 150); });

guard(loadRuns)();
