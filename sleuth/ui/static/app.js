// Sleuth dashboard: fetches JSON from the local server and draws it.
//
// Layout: training curves, then one tab per analyzer view. Each tab shows
// (1) any of the view's timeline series across checkpoints and (2) a detail
// panel for the selected checkpoint, drawn by that view's renderer below.
import { emptyState, fmtNum, heatmap, lineChart, ordinalColors, tokens } from "./charts.js";

const $ = (id) => document.getElementById(id);

const VIEWS = {
  neuron_stats: { title: "Neurons", def: "frac_selective", render: renderNeurons },
  attention: { title: "Attention heads", def: "induction_score", render: renderAttention },
  logit_lens: { title: "Logit lens", def: "top1_accuracy", render: renderLogitLens },
  ablation: { title: "Ablation", def: "mlp_delta_loss", render: renderAblation },
};

const state = { runs: [], run: null, metrics: null, suite: null, view: null, timeline: null, ckpt: null };

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

/** Small DOM builder: el("div", {class: "x"}, child, "text", ...). Text always via textContent. */
function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else e.setAttribute(k, v);
  }
  for (const c of children) e.append(c);
  return e;
}

function selectEl(id, options, value, onChange) {
  const s = el("select", { id });
  fillSelect(s, options, value);
  s.addEventListener("change", onChange);
  return s;
}

const stepDir = (s) => `step_${String(s).padStart(7, "0")}`;
const pct = (v) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "–");
const num = (v) => (!Number.isFinite(v) ? "–" : Math.abs(v) < 10 ? v.toFixed(3) : fmtNum(v));
const fmtFor = (f) => (f === "pct" ? pct : num);
/** Show a token with its whitespace visible: spaces as ·, newlines as ⏎. */
const tok = (s, max = 10) => {
  const t = s.replace(/ /g, "·").replace(/\n/g, "⏎").replace(/\t/g, "⇥");
  return t.length > max ? t.slice(0, max - 1) + "…" : t;
};

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
  Object.assign(state, { run, metrics, timeline: null, ckpt: null });
  // External models have no training history: hide what would be empty.
  $("training-card").hidden = Boolean(run.config.external);
  renderTiles();
  renderLoss();
  const suites = [...new Set(run.analyses.map((a) => a.suite))];
  fillSelect($("suite-select"), suites.map((s) => [s, s]), state.suite);
  if (!suites.length) {
    $("tabs").replaceChildren();
    for (const i of ["timeline-chart", "timeline-heat"]) {
      emptyState($(i), `No analysis yet. Run: python -m sleuth analyze runs/${run.id}`);
    }
    $("detail").replaceChildren();
    setStatus("");
    return;
  }
  await selectSuite($("suite-select").value);
  setStatus("");
}

function availableViews() {
  return state.run.analyses.filter((a) => a.suite === state.suite).map((a) => a.view);
}

async function selectSuite(suite) {
  state.suite = suite;
  const avail = availableViews();
  const view = avail.includes(state.view) ? state.view : Object.keys(VIEWS).find((v) => avail.includes(v));
  await selectView(view);
}

async function selectView(view) {
  state.view = view;
  renderTabs();
  const t = await api(`runs/${state.run.id}/analysis/${view}/${state.suite}/timeline`);
  state.timeline = t;
  const keys = Object.keys(t.series);
  const prev = $("metric-select").value;
  fillSelect($("metric-select"), keys.map((k) => [k, t.series[k].label]),
    keys.includes(prev) ? prev : VIEWS[view].def);
  const prevStep = $("ckpt-select").value;
  fillSelect($("ckpt-select"), t.steps.map((s, i) => [s, `step ${s} (${fmtNum(t.tokens[i])} tok)`]),
    t.steps.map(String).includes(prevStep) ? prevStep : t.steps[t.steps.length - 1]);
  $("timeline-title").textContent = `${VIEWS[view].title} across checkpoints`;
  $("timeline-card").hidden = t.steps.length < 2;  // a "timeline" of one checkpoint is just a dot
  renderTimeline();
  await loadCheckpoint();
}

async function loadCheckpoint() {
  const step = $("ckpt-select").value;
  if (!step || !state.view) return;
  state.ckpt = await api(`runs/${state.run.id}/analysis/${state.view}/${state.suite}/${stepDir(step)}`);
  renderDetail();
}

// ------------------------------------------------------------------ shared rendering
function renderTabs() {
  const avail = availableViews();
  const nav = $("tabs");
  nav.replaceChildren();
  for (const [v, info] of Object.entries(VIEWS)) {
    const b = el("button", { class: "tab", role: "tab", type: "button", text: info.title });
    b.setAttribute("aria-selected", String(v === state.view));
    if (!avail.includes(v)) {
      b.disabled = true;
      b.title = `Not analyzed yet: python -m sleuth analyze runs/<run> --views ${v}`;
    }
    b.addEventListener("click", guard(() => selectView(v)));
    nav.appendChild(b);
  }
}

function renderTiles() {
  const { run, metrics } = state;
  const last = metrics.train[metrics.train.length - 1] || {};
  const lastVal = metrics.val[metrics.val.length - 1] || {};
  const p = run.params;
  const ext = run.config.external;
  if (ext) {
    const tiles = [
      ["Parameters", p ? fmtNum(p.total) : "–", p ? `${fmtNum(p.non_embedding)} non-embedding` : ""],
      ["Source", ext.format.toUpperCase(), ext.info?.architecture || ext.info?.model_type || ""],
      ["Layers × heads", `${run.config.model.n_layers} × ${run.config.model.n_heads}`, `width ${run.config.model.d_model}`],
    ];
    $("tiles").replaceChildren(...tiles.map(([l, v, s]) =>
      el("div", { class: "tile" }, el("div", { class: "label", text: l }), el("div", { class: "value", text: v }),
        el("div", { class: "sub", text: s }))));
    return;
  }
  const tiles = [
    ["Parameters", p ? fmtNum(p.non_embedding) : "–", p ? `non-embedding (+${fmtNum(p.embedding)} embedding)` : ""],
    ["Step", `${last.step ?? 0}`, `of ${run.config.train.max_steps}`],
    ["Tokens seen", fmtNum(last.tokens ?? 0), `${fmtNum(last.tok_per_s ?? 0)} tok/s last`],
    ["Train loss", last.loss !== undefined ? last.loss.toFixed(3) : "–", "nats/token"],
    ["Val bits/byte", lastVal.val_bpb !== undefined ? lastVal.val_bpb.toFixed(3) : "–", "lower is better"],
    ["Snapshots", `${run.snapshots.length}`, run.env?.gpus?.length ? run.env.gpus[0] : "CPU"],
  ];
  $("tiles").replaceChildren(...tiles.map(([l, v, s]) =>
    el("div", { class: "tile" }, el("div", { class: "label", text: l }), el("div", { class: "value", text: v }),
      el("div", { class: "sub", text: s }))));
}

function renderLoss() {
  if ($("training-card").hidden) return;
  const T = tokens();
  const { train, val } = state.metrics;
  lineChart($("loss-chart"), [
    { name: "train loss", color: T.series[0], points: train.map((r) => [r.step, r.loss]) },
    { name: "validation loss", color: T.series[1], points: val.map((r) => [r.step, r.val_loss]) },
  ], { xLabel: "step", yLabel: "loss", yFormat: (v) => v.toFixed(2), empty: "No metrics logged yet" });
}

function renderTimeline() {
  const t = state.timeline;
  if (!t || $("timeline-card").hidden) return;
  const key = $("metric-select").value;
  const ser = t.series[key];
  if (!ser) return;
  $("metric-help").textContent = ser.help;
  const fmt = fmtFor(ser.fmt);
  const R = ser.rows.length;
  const col = (r) => t.steps.map((_, i) => ser.values[i][r]);
  // Too many rows (e.g. every head) make an unreadable line chart: show the 8
  // rows with the largest values; the heatmap still shows all of them.
  let shown = [...Array(R).keys()];
  if (R > 12) {
    shown = shown.map((r) => [r, Math.max(...col(r).map((v) => Math.abs(v ?? 0)))])
      .sort((a, b) => b[1] - a[1]).slice(0, 8).map(([r]) => r).sort((a, b) => a - b);
  }
  $("lines-title").textContent = R > 12 ? `Top 8 of ${R} rows (largest values)` : (R === 1 ? "Value" : "One line per row");
  const colors = R === 1 ? [tokens().series[0]] : ordinalColors(shown.length);
  lineChart($("timeline-chart"),
    shown.map((r, i) => ({ name: ser.rows[r], color: colors[i], points: t.steps.map((s, j) => [s, ser.values[j][r]]) })),
    { xLabel: "step", yLabel: ser.label, logX: $("logx").checked, yFormat: fmt });
  const matrix = ser.rows.map((_, r) => col(r));
  heatmap($("timeline-heat"), matrix, {
    rowLabels: ser.rows, colLabels: t.steps.map(String), xLabel: "checkpoint (step)", format: fmt,
    legendLabel: ser.label, diverging: key.includes("delta"), marginLeft: 52,
    tipTitle: (r, c) => `${ser.rows[r]}, step ${t.steps[c]}`,
  });
}

function renderDetail() {
  const box = $("detail");
  box.replaceChildren();
  if (state.ckpt) VIEWS[state.view].render(box, state.ckpt, $("ckpt-select").value);
}

function table(rows, cols) {
  const t = el("table");
  const head = t.insertRow();
  for (const c of cols) head.appendChild(el("th", { text: c.label }));
  for (const r of rows) {
    const tr = t.insertRow();
    for (const c of cols) tr.insertCell().textContent = c.get(r);
  }
  return el("div", { class: "table-wrap" }, t);
}

function cardHead(title, ...controls) {
  return el("div", { class: "card-head" }, el("h2", { text: title }), el("div", { class: "controls" }, ...controls));
}

const labeled = (text, control) => el("label", {}, `${text} `, control);

// ------------------------------------------------------------------ view: neurons
function renderNeurons(box, d, step) {
  const cats = d.summary.categories;
  const catSel = selectEl("cat-select", [["all", "all categories"], ...cats.map((c) => [c, c])], "all", draw);
  const sort = el("input", { type: "checkbox", id: "sort-neurons" });
  sort.checked = true;
  sort.addEventListener("change", draw);
  const chart = el("div", { class: "chart tall" });
  box.append(
    cardHead(`Neurons at step ${step}`, labeled("Category", catSel), el("label", {}, sort, " sort neurons by rate")),
    el("p", { class: "hint", text: "Firing rate = fraction of tokens on which |activation| exceeds that layer's RMS. Each row is a layer; each column a neuron." }),
    chart,
    el("details", {}, el("summary", { text: "Table view" }), table(d.summary.layers, [
      { label: "layer", get: (l) => l.layer },
      { label: "mean firing", get: (l) => pct(l.fire_rate_mean) },
      { label: "nearly dead", get: (l) => pct(l.frac_low_activity) },
      { label: "always on", get: (l) => pct(l.frac_high_activity) },
      { label: "selective", get: (l) => pct(l.frac_selective) },
      { label: "act RMS", get: (l) => num(l.act_rms) },
      { label: "attn out", get: (l) => num(l.attn_out_rms) },
      { label: "MLP out", get: (l) => num(l.mlp_out_rms) },
      { label: "residual", get: (l) => num(l.resid_rms) },
    ])));
  function draw() {
    const cat = catSel.value;
    const rows = cat === "all" ? d.arrays.fire_rate : d.arrays.fire_rate_by_category.map((byCat) => byCat[cats.indexOf(cat)]);
    const order = rows.map((row) => {
      const idx = row.map((_, i) => i);
      return sort.checked ? idx.sort((a, b) => row[b] - row[a]) : idx;
    });
    const matrix = rows.map((row, l) => order[l].map((i) => row[i]));
    heatmap(chart, matrix, {
      rowLabels: matrix.map((_, l) => `L${l}`),
      colLabels: matrix[0].map((_, i) => (sort.checked ? `#${i + 1}` : String(i))),
      xLabel: sort.checked ? "neurons, most active first" : "neuron index",
      format: pct, legendLabel: "firing rate", domain: [0, Math.max(1e-6, ...matrix.flat())],
      tipTitle: (r, c) => `layer ${r}, neuron ${order[r][c]}`,
    });
  }
  draw();
}

// ------------------------------------------------------------------ view: attention
function renderAttention(box, d, step) {
  const s = d.summary, a = d.arrays;
  const L = s.n_layers, Hn = s.n_heads;
  const metrics = [["induction_score", "Induction score"], ["prev_token", "Previous-token attention"],
    ["bos_share", "Attention to <|bos|>"], ["self", "Attention to itself"], ["entropy", "Entropy (0-1)"],
    ["duplicate_score", "Attention to earlier copy"]];
  const mSel = selectEl("head-metric", metrics, "induction_score", drawGrid);
  const best = s.heads.reduce((b, h) => (h.induction_score > b.induction_score ? h : b), s.heads[0]);
  const lSel = selectEl("pat-layer", [...Array(L).keys()].map((l) => [l, `layer ${l}`]), best.layer, drawPattern);
  const hSel = selectEl("pat-head", [...Array(Hn).keys()].map((h) => [h, `head ${h}`]), best.head, drawPattern);
  const ind = s.induction;
  const grid = el("div", { class: "chart" });
  const pattern = el("div", { class: "chart square" });
  box.append(
    cardHead(`Attention heads at step ${step}`, labeled("Color by", mSel)),
    el("div", { class: "facts" },
      el("span", {}, "induction gain ", el("strong", { text: `${ind.induction_gain.toFixed(2)} nats/token` })),
      el("span", {}, "random sequence loss: first copy ", el("strong", { text: ind.loss_first_copy.toFixed(2) }),
        ", repeat ", el("strong", { text: ind.loss_second_copy.toFixed(2) }),
        ind.token_pool ? ` (${ind.pool_size} ${ind.token_pool}; uniform guess ${ind.uniform_loss.toFixed(2)})` : ""),
      ind.any_token ? el("span", {}, "with any token: gain ", el("strong", { text: `${ind.induction_gain_any_token.toFixed(2)}` }),
        ` (${ind.any_token.loss_first_copy.toFixed(2)} → ${ind.any_token.loss_second_copy.toFixed(2)})`) : "",
      el("span", {}, "top induction heads ", el("strong", { text: s.top_induction_heads.slice(0, 3).join(", ") }))),
    el("p", { class: "hint", text: "Each cell is one head (row = layer, column = head). Click a head to see its attention pattern below." }),
    grid,
    el("div", { class: "card-head" }, el("h3", { text: "Attention pattern" }), el("div", { class: "controls" }, lSel, hSel)),
    el("p", { class: "hint", text: s.example ? `Test “${s.example.case_id}”. Row = the token doing the looking; column = the token it looks at. Each row sums to 100%.` : "" }),
    pattern);
  function drawGrid() {
    const key = mSel.value;
    heatmap(grid, a[key], {
      rowLabels: [...Array(L).keys()].map((l) => `L${l}`), colLabels: [...Array(Hn).keys()].map((h) => `H${h}`),
      xLabel: "head", format: pct, legendLabel: metrics.find(([k]) => k === key)[1], domain: [0, 1],
      tipTitle: (r, c) => `layer ${r}, head ${c}`,
      onClick: (r, c) => { lSel.value = r; hSel.value = c; drawPattern(); },
    });
  }
  function drawPattern() {
    if (!s.example || !a.example_pattern) return emptyState(pattern, "No example pattern saved");
    const toks = s.example.tokens;
    const m = a.example_pattern[+lSel.value][+hSel.value];
    heatmap(pattern, m, {
      rowLabels: toks.map((x) => tok(x, 8)), colLabels: toks.map((x) => tok(x, 8)), marginLeft: 64,
      xLabel: "key (looked at)", format: pct, legendLabel: "attention", domain: [0, 1],
      tipTitle: (r, c) => `“${tok(toks[r], 20)}” (pos ${r}) → “${tok(toks[c], 20)}” (pos ${c})`,
    });
  }
  drawGrid();
  drawPattern();
}

// ------------------------------------------------------------------ view: logit lens
function renderLogitLens(box, d, step) {
  const s = d.summary, a = d.arrays;
  const grid = el("div", { class: "chart square" });
  const cols = [{ label: "layer", get: (i) => s.rows[i] }, { label: "loss", get: (i) => num(s.loss[i]) },
    { label: "top-1", get: (i) => pct(s.top1_accuracy[i]) },
    ...s.categories.map((c) => ({ label: `top-1 ${c}`, get: (i) => pct(s.top1_by_category[c][i]) }))];
  box.append(
    cardHead(`Logit lens at step ${step}`),
    el("p", { class: "hint", text: "What each layer would predict if the model stopped there. Rows: embeddings, then after each layer (the last row is the real output). Color = probability given to the true next token. Hover to see each layer's top guess." }),
    s.emb_top1_is_input === undefined ? "" : el("p", { class: "hint", text: `About the “emb” row: it's the input embedding sent straight to the output layer, before any layer has run. With tied embeddings that mostly predicts the *current* token again (here ${pct(s.emb_top1_is_input)} of positions), which is almost never the next token, so its loss (${num(s.loss[0])}) is far above a uniform guess (${num(s.uniform_loss)}). Read it as a starting point, not a prediction.` }),
    grid,
    el("details", { open: "" }, el("summary", { text: "Per-layer accuracy (whole suite)" }),
      table([...s.rows.keys()], cols)));
  if (!s.example) return emptyState(grid, "No example saved");
  const toks = s.example.tokens;
  heatmap(grid, a.example_true_p, {
    rowLabels: s.rows, colLabels: toks.slice(1).map((x) => tok(x, 8)), marginLeft: 52,
    xLabel: `true next token (test “${s.example.case_id}”)`, format: pct, legendLabel: "p(true next token)",
    domain: [0, 1],
    tipTitle: (r, c) => `${s.rows[r]}, after “${tok(toks[c], 20)}”, true next: “${tok(toks[c + 1], 20)}”`,
    tipRows: (r, c) => [
      { value: pct(a.example_true_p[r][c]), label: "p(true next token)" },
      { value: `“${tok(s.example.top_tokens[r][c], 20)}”`, label: `top guess (${pct(a.example_top_p[r][c])})` },
    ],
  });
}

// ------------------------------------------------------------------ view: ablation
function renderAblation(box, d, step) {
  const s = d.summary, a = d.arrays;
  const L = s.n_layers, Hn = s.n_heads;
  const keys = [...s.categories, "overall"];
  const measures = [["delta_loss", "Δ loss (nats/token)"], ["delta_expected_loss", "Δ loss on expected answers"],
    ["drop_exact", "Drop in exact-match answers"]];
  const mSel = selectEl("abl-measure", measures, "delta_loss", draw);
  const cSel = selectEl("abl-cat", keys.map((k) => [k, k]), "overall", draw);
  const heads = el("div", { class: "chart" });
  const layers = el("div", { class: "chart" });
  layers.style.height = "160px";
  const topBox = el("div");
  const base = s.baseline;
  box.append(
    cardHead(`Ablation at step ${step} (${s.method})`, labeled("Measure", mSel), labeled("Category", cSel)),
    el("p", { class: "hint", text: `Each part of the model is replaced by its ${s.method === "mean" ? "average output" : "zero"} and the suite is re-run. Red = the model gets worse without it (that part matters); blue = it gets better; gray = no effect or no data.` }),
    el("div", { class: "facts" }, ...keys.map((k) => el("span", {}, `${k}: loss `, el("strong", { text: num(base[k].loss) }),
      base[k].exact_match !== null ? `, exact ${pct(base[k].exact_match)}` : ""))),
    el("h3", { text: "Single heads" }), heads,
    el("h3", { text: "Whole layers" }), layers,
    topBox);
  function values(measure) {
    const src = measure === "drop_exact" ? a.delta_exact : a[measure];
    const j = keys.indexOf(cSel.value);
    return src.map((row) => (row[j] === null ? NaN : measure === "drop_exact" ? -row[j] : row[j]));
  }
  function draw() {
    const v = values(mSel.value);
    const fmt = mSel.value === "drop_exact" ? pct : num;
    const label = measures.find(([k]) => k === mSel.value)[1];
    const headM = [...Array(L).keys()].map((l) => v.slice(l * Hn, (l + 1) * Hn));
    const layerM = [v.slice(L * Hn, L * Hn + L), v.slice(L * Hn + L)];
    const lim = Math.max(1e-9, ...v.filter(Number.isFinite).map(Math.abs));
    heatmap(heads, headM, {
      rowLabels: [...Array(L).keys()].map((l) => `L${l}`), colLabels: [...Array(Hn).keys()].map((h) => `H${h}`),
      xLabel: "head", format: fmt, legendLabel: label, diverging: true, domain: [-lim, lim],
      tipTitle: (r, c) => `remove layer ${r}, head ${c}`,
    });
    heatmap(layers, layerM, {
      rowLabels: ["attn", "MLP"], colLabels: [...Array(L).keys()].map((l) => `L${l}`), xLabel: "layer",
      format: fmt, legendLabel: label, diverging: true, domain: [-lim, lim],
      tipTitle: (r, c) => `remove layer ${c} ${r === 0 ? "attention" : "MLP"}`,
    });
    // The ranking follows the selected measure and category (it used to be
    // fixed to overall Δ loss, whatever the selectors said).
    const ranked = v.map((x, i) => [x, i]).filter(([x]) => Number.isFinite(x)).sort((p, q) => q[0] - p[0]).slice(0, 12);
    topBox.replaceChildren(el("details", { open: "" },
      el("summary", { text: `Most important parts (${label}, ${cSel.value})` }),
      table(ranked, [{ label: "component", get: ([, i]) => s.components[i] }, { label, get: ([x]) => fmt(x) }])));
  }
  draw();
}

// ------------------------------------------------------------------ wiring
function guard(fn) {
  return (...args) => Promise.resolve(fn(...args)).catch((e) => setStatus(String(e.message || e), true));
}

function renderAll() {
  if (!state.run) return;
  renderLoss();
  renderTimeline();
  renderDetail();
}

$("run-select").addEventListener("change", guard(async (e) => {
  history.replaceState(null, "", `?run=${encodeURIComponent(e.target.value)}`);
  await loadRun(e.target.value);
}));
$("suite-select").addEventListener("change", guard((e) => selectSuite(e.target.value)));
$("ckpt-select").addEventListener("change", guard(loadCheckpoint));
$("metric-select").addEventListener("change", renderTimeline);
$("logx").addEventListener("change", renderTimeline);
$("refresh").addEventListener("click", guard(async () => {
  const keep = $("run-select").value;
  await loadRuns();
  if (keep && state.runs.some((r) => r.id === keep)) { $("run-select").value = keep; await loadRun(keep); }
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
