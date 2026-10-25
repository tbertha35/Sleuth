// Helpers shared by every page of the dashboard: fetching, the session
// token, and a tiny DOM builder. Text always goes in via textContent (it can
// come from file names, model output or logs), never as HTML.

export const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ session token
// `sleuth ui` prints a link ending in ?token=...; the page keeps the token and
// sends it with every request that *does* something (POST). It's removed from
// the address bar so it doesn't end up in bookmarks or screenshots.
const TOKEN_KEY = "sleuth-token";
let token = null;
{
  const params = new URLSearchParams(location.search);
  if (params.has("token")) {
    token = params.get("token");
    try { localStorage.setItem(TOKEN_KEY, token); } catch { /* storage may be blocked */ }
    params.delete("token");
    const q = params.toString();
    history.replaceState(null, "", location.pathname + (q ? `?${q}` : "") + location.hash);
  } else {
    try { token = localStorage.getItem(TOKEN_KEY); } catch { token = null; }
  }
}
export const hasToken = () => Boolean(token);

export class ApiError extends Error {
  constructor(msg, status, data) { super(msg); this.status = status; this.data = data; }
}

export async function api(path) {
  const r = await fetch(`/api/${path}`);
  const j = await r.json();
  if (!r.ok) throw new ApiError(j.error || `HTTP ${r.status}`, r.status, j);
  return j;
}

function postInit(body, signal) {
  return {
    method: "POST", signal, body: JSON.stringify(body ?? {}),
    headers: { "Content-Type": "application/json", "X-Sleuth-Token": token || "" },
  };
}

export async function post(path, body) {
  const r = await fetch(`/api/${path}`, postInit(body));
  const j = await r.json();
  if (!r.ok) throw new ApiError(j.error || `HTTP ${r.status}`, r.status, j);
  return j;
}

/** POST that answers with Server-Sent Events; calls onEvent(name, data) for each.
 *  Abort with the signal (that's the Stop button: the server notices and stops). */
export async function postStream(path, body, onEvent, signal) {
  const r = await fetch(`/api/${path}`, postInit(body, signal));
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error || msg; } catch { /* not JSON */ }
    throw new ApiError(msg, r.status);
  }
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i);
      buf = buf.slice(i + 2);
      let name = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) name = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) onEvent(name, JSON.parse(data));
    }
  }
}

// ------------------------------------------------------------------ DOM
/** el("div", {class: "x"}, child, "text", ...). Text always via textContent. */
export function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k === "value") e.value = v;
    else if (k === "checked") e.checked = Boolean(v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children) if (c !== null && c !== undefined && c !== "") e.append(c);
  return e;
}

export function fillSelect(sel, options, value) {
  sel.replaceChildren();
  for (const [v, label] of options) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = label;
    sel.appendChild(o);
  }
  if (value !== undefined && options.some(([v]) => String(v) === String(value))) sel.value = value;
}

export function selectEl(id, options, value, onChange) {
  const s = el("select", { id });
  fillSelect(s, options, value);
  if (onChange) s.addEventListener("change", onChange);
  return s;
}

export const labeled = (text, control, hint) =>
  el("label", { title: hint || undefined }, `${text} `, control);

export function statusLine(node) {
  return (msg, isError = false) => {
    node.textContent = msg;
    node.classList.toggle("error", isError);
  };
}

/** Run an async handler; report failures in a status line instead of the console. */
export function guarded(setStatus, fn) {
  return (...args) => Promise.resolve().then(() => fn(...args)).catch((e) => setStatus(String(e.message || e), true));
}

// ------------------------------------------------------------------ formatting
export function fmtBytes(n) {
  if (!Number.isFinite(n)) return "–";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1000 && i < u.length - 1) { n /= 1000; i++; }
  return `${n < 10 && i ? n.toFixed(1) : Math.round(n)} ${u[i]}`;
}

export function fmtTime(t) {
  if (!t) return "–";
  const d = new Date(t * 1000);
  const today = new Date().toDateString() === d.toDateString();
  return today ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function fmtDuration(s) {
  if (!Number.isFinite(s) || s < 0) return "–";
  if (s < 60) return `${Math.round(s)} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
  return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
}

/** Show a token with its whitespace visible: spaces as ·, newlines as ⏎. */
export const visible = (s, max = 24) => {
  const t = s.replace(/ /g, "·").replace(/\n/g, "⏎").replace(/\t/g, "⇥");
  return t.length > max ? t.slice(0, max - 1) + "…" : t;
};
