// Jobs page: start Sleuth commands, watch the queue, read live logs.
//
// The forms are built from /api/jobs/specs, the server's allow-list of
// commands and fields, so the page and the server can't disagree about what
// may be run. The server validates everything again anyway.
import {
  $, api, el, fillSelect, fmtDuration, fmtTime, guarded, hasToken, post, statusLine,
} from "./common.js";

const st = { specs: [], jobs: [], runs: [], selected: null, source: null, timer: null, visible: false };
const setStatus = statusLine($("jobs-status"));

const STATUS_HELP = {
  queued: "waiting for its lane (one GPU job and one CPU job run at a time)",
  running: "running", stopping: "asked to stop; finishing the current step",
  done: "finished successfully", failed: "exited with an error (see the log)",
  stopped: "stopped from the dashboard", cancelled: "removed from the queue before it started",
  orphaned: "the dashboard restarted while it ran",
};

export async function initJobs() {
  st.specs = await api("jobs/specs");
  fillSelect($("job-command"), st.specs.map((s) => [s.name, s.title]), "train");
  $("job-command").addEventListener("change", renderForm);
  $("job-form").addEventListener("submit", guarded(setStatus, submit));
  if (!hasToken()) {
    setStatus("Read-only: open the link that `sleuth ui` printed (it ends in ?token=…) to start or stop jobs.", true);
  }
}

export async function showJobs(runs) {
  st.visible = true;
  st.runs = runs;
  renderForm();
  await refresh();
}

export function hideJobs() {
  st.visible = false;
  clearTimeout(st.timer);
}

/** Open a job's log (also used by other pages, e.g. "Run as a job"). */
export function openLog(id) {
  st.selected = id;
  streamLog(id);
  renderTable();
}

// ------------------------------------------------------------------ the form
function spec() {
  return st.specs.find((s) => s.name === $("job-command").value);
}

function renderForm() {
  const s = spec();
  const box = $("job-fields");
  box.replaceChildren();
  $("job-help").textContent = `${s.help} Runs on the ${s.lane.toUpperCase()} lane.`;
  for (const f of s.fields) box.append(fieldControl(f));
}

const ourRuns = () => st.runs.filter((r) => !r.stale && !r.external);
const modelChoices = () => st.runs.filter((r) => !r.stale);

function fieldControl(f) {
  const id = `jf-${f.name}`;
  let control;
  switch (f.kind) {
    case "bool":
      control = el("input", { type: "checkbox", id, checked: f.default === true });
      return el("label", { class: "check", title: f.help || undefined }, control, ` ${f.label}`);
    case "int": case "float":
      control = el("input", { type: "number", id, step: f.kind === "int" ? "1" : "any", value: f.default ?? "",
        min: f.min ?? undefined, max: f.max ?? undefined, placeholder: f.required ? "required" : "" });
      break;
    case "choice": case "config":
      control = el("select", { id });
      fillSelect(control, (f.choices || []).map((c) => [c, c]), f.default);
      break;
    case "run":
      control = el("select", { id });
      fillSelect(control, ourRuns().map((r) => [r.id, r.id]));
      break;
    case "model": {
      const list = el("datalist", { id: `${id}-list` },
        ...modelChoices().map((r) => el("option", { value: r.id })));
      control = el("input", { type: "text", id, list: `${id}-list`, placeholder: "a run, or a .gguf / model folder path",
        value: ourRuns()[0]?.id || "" });
      return el("div", { class: "field" }, el("span", { class: "flabel", text: f.label }), control, list,
        f.help ? el("span", { class: "fhelp", text: f.help }) : "");
    }
    case "suite": {
      const def = new Set(f.default || []);
      control = el("div", { id, class: "checks" }, ...(f.choices || []).map((c) =>
        el("label", { class: "check" }, el("input", { type: "checkbox", value: c, checked: def.has(c) }), ` ${c}`)));
      break;
    }
    case "set":
      control = el("textarea", { id, rows: "3", placeholder: "train.max_steps=2000\ntrain.lr=3e-4" });
      break;
    default: // text, path
      control = (f.multiline || f.multi)
        ? el("textarea", { id, rows: f.multi ? "2" : "4", placeholder: f.multi ? "one per line" : "" })
        : el("input", { type: "text", id, value: f.default ?? "" });
  }
  return el("div", { class: "field" }, el("span", { class: "flabel", text: f.label + (f.required ? " *" : "") }),
    control, f.help ? el("span", { class: "fhelp", text: f.help }) : "");
}

function formValues() {
  const values = {};
  for (const f of spec().fields) {
    const c = $(`jf-${f.name}`);
    if (f.kind === "bool") values[f.name] = c.checked;
    else if (f.kind === "suite") values[f.name] = [...c.querySelectorAll("input:checked")].map((i) => i.value);
    else if (f.multi) values[f.name] = c.value.split("\n").map((x) => x.trim()).filter(Boolean);
    else if (c.value !== "") values[f.name] = c.value;
  }
  return values;
}

async function submit(e) {
  e.preventDefault();
  const job = await post("jobs", { command: spec().name, values: formValues() });
  setStatus(job.status === "queued" ? `Queued ${job.id}.` : `Started ${job.id}.`);
  openLog(job.id);
  await refresh();
}

// ------------------------------------------------------------------ the list
async function refresh() {
  clearTimeout(st.timer);
  if (!st.visible) return;
  try {
    st.jobs = await api("jobs");
    renderTable();
  } catch (e) {
    setStatus(String(e.message || e), true);
  }
  // Poll fast while something is happening, slowly otherwise.
  const active = st.jobs.some((j) => !j.finished);
  st.timer = setTimeout(refresh, active ? 2000 : 10000);
}

function renderTable() {
  const tbody = $("jobs-body");
  tbody.replaceChildren();
  $("jobs-empty").hidden = st.jobs.length > 0;
  const now = Date.now() / 1000;
  for (const j of st.jobs) {
    const dur = j.started ? (j.ended || now) - j.started : null;
    const actions = el("div", { class: "row-actions" },
      el("button", { type: "button", text: "Log", onclick: () => openLog(j.id) }));
    if (!j.finished) {
      actions.append(el("button", {
        type: "button", class: "danger", text: j.status === "stopping" ? "Kill" : (j.status === "queued" ? "Cancel" : "Stop"),
        title: j.status === "stopping" ? "End the process now (no checkpoint)" : "Training saves a checkpoint first",
        onclick: guarded(setStatus, async () => { await post(`jobs/${j.id}/stop`); await refresh(); }),
      }));
    }
    const run = j.run ? el("a", { href: `?run=${encodeURIComponent(j.run)}#runs`, text: j.run }) : "–";
    const tr = el("tr", { class: j.id === st.selected ? "selected" : "" },
      el("td", {}, el("span", { class: `badge ${j.status}`, text: j.status, title: STATUS_HELP[j.status] || "" })),
      el("td", { class: "jobcell" }, el("div", { text: j.title }), el("code", { text: j.display }),
        j.note ? el("div", { class: "note", text: j.note }) : ""),
      el("td", { text: j.lane.toUpperCase() }),
      el("td", { text: fmtTime(j.started || j.created) }),
      el("td", { text: dur === null ? "–" : fmtDuration(dur) }),
      el("td", { text: j.exit_code ?? "–" }),
      el("td", {}, run),
      el("td", {}, actions));
    tbody.append(tr);
  }
}

// ------------------------------------------------------------------ the log
function streamLog(id) {
  if (st.source) st.source.close();
  const pre = $("job-log");
  pre.textContent = "";
  $("log-title").textContent = `Log: ${id}`;
  $("log-card").hidden = false;
  // Start with the last ~200 KB: a long training log can be many MB.
  const src = new EventSource(`/api/jobs/${encodeURIComponent(id)}/log?offset=-200000`);
  st.source = src;
  src.addEventListener("log", (ev) => {
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8;
    pre.append(JSON.parse(ev.data).text);
    if (atBottom) pre.scrollTop = pre.scrollHeight;
  });
  src.addEventListener("end", (ev) => {
    src.close();
    const j = JSON.parse(ev.data).job;
    $("log-title").textContent = `Log: ${id} (${j.status}${j.exit_code !== null ? `, exit code ${j.exit_code}` : ""})`;
    refresh();
  });
  src.onerror = () => { /* EventSource reconnects by itself, resuming from the last offset */ };
}
