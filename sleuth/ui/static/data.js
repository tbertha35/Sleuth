// Data page: what's in the data folder (raw text, fine-tuning sets, shards,
// tokenizers), with sizes and, for shards, token counts and the source mix.
import { $, api, el, fmtBytes, fmtTime, guarded, statusLine } from "./common.js";

const setStatus = statusLine($("data-status"));

export function initData() {
  $("data-refresh").addEventListener("click", guarded(setStatus, showData));
}

export async function showData() {
  setStatus("Reading the data folder…");
  const d = await api("data");
  setStatus(d.exists ? "" : `No data folder at ${d.root} yet.`);
  $("data-root").textContent = d.root;
  const box = $("data-sections");
  box.replaceChildren(...d.sections.map(section));
}

const tokens = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : `${n ?? "–"}`);

function section(s) {
  const head = el("div", { class: "card-head" }, el("h2", { text: `${s.name}/` }), el("span", { class: "hint", text: s.title }));
  if (!s.entries.length) {
    return el("section", { class: "card" }, head, el("p", { class: "hint", text: s.exists ? "(empty)" : "(no such folder)" }));
  }
  const t = el("table");
  const cols = ["name", "size", "files", "modified"].concat(s.name === "shards" ? ["tokens (train / val)", "vocab", "source mix"] :
    s.name === "tokenizers" ? ["vocab"] : []);
  t.append(el("tr", {}, ...cols.map((c) => el("th", { text: c }))));
  for (const e of s.entries) {
    const tr = el("tr", {}, el("td", { text: e.name + (e.dir ? "/" : "") }), el("td", { text: fmtBytes(e.bytes) }),
      el("td", { text: e.files.toLocaleString() }), el("td", { text: fmtTime(e.modified) }));
    if (s.name === "shards") {
      const m = e.shards;
      tr.append(el("td", { text: m ? `${tokens(m.train_tokens)} / ${tokens(m.val_tokens)}` : "no meta.json" }),
        el("td", { text: m?.vocab_size ?? "–" }), el("td", {}, m ? mix(m.sources) : "–"));
    } else if (s.name === "tokenizers") {
      tr.append(el("td", { text: e.vocab_size ?? "–" }));
    }
    t.append(tr);
  }
  return el("section", { class: "card" }, head, el("div", { class: "table-wrap" }, t));
}

/** The source mix as a stacked bar plus labels (only when there's more than one source). */
function mix(sources) {
  if (!sources.length) return "–";
  if (sources.length === 1) return sources[0].name;
  const bar = el("span", { class: "mixbar" });
  sources.forEach((s, i) => {
    const seg = el("span", { class: `seg s${i % 6}`, title: `${s.name}: ${(s.share * 100).toFixed(1)}% of the text` });
    seg.style.width = `${s.share * 100}%`;
    bar.append(seg);
  });
  return el("div", { class: "mix" }, bar, el("div", { class: "mixlabels" },
    ...sources.map((s, i) => el("span", {}, el("span", { class: `dot s${i % 6}` }), ` ${s.name} ${(s.share * 100).toFixed(0)}%`))));
}
