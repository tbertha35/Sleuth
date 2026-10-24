// Tiny canvas chart helpers: a line chart with a crosshair, and a heatmap.
// No libraries: everything the dashboard draws is in this file.
//
// Conventions: colors come from CSS tokens (style.css), so light/dark
// switch in one place; text is always ink, never a series color; labels go
// into the DOM with textContent (they can come from file names).

export function tokens() {
  const s = getComputedStyle(document.documentElement);
  const g = (n) => s.getPropertyValue(n).trim();
  const list = (n) => g(n).split(",").map((x) => x.trim()).filter(Boolean);
  return {
    ink: g("--ink"), ink2: g("--ink-2"), muted: g("--muted"), grid: g("--grid"), axis: g("--axis"),
    surface: g("--surface"), series: [g("--series-1"), g("--series-2"), g("--series-3")],
    seq: list("--seq"), ordinal: list("--ordinal"), div: list("--div"),
  };
}

// ---------------------------------------------------------------- colors
function hexToRgb(h) {
  const n = parseInt(h.replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Color at t in [0,1] along a ramp of hex stops, as [r,g,b]. */
export function rampRgb(ramp, t) {
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * (ramp.length - 1);
  const i = Math.min(ramp.length - 2, Math.floor(x));
  const f = x - i;
  const a = hexToRgb(ramp[i]), b = hexToRgb(ramp[i + 1]);
  return [0, 1, 2].map((k) => Math.round(a[k] + (b[k] - a[k]) * f));
}
const rgbStr = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

/** n ordered colors for ordinal series (e.g. layers: shallow -> deep). */
export function ordinalColors(n) {
  const ramp = tokens().ordinal;
  if (n === 1) return [ramp[Math.floor(ramp.length / 2)]];
  return Array.from({ length: n }, (_, i) => rgbStr(rampRgb(ramp, i / (n - 1))));
}

// ---------------------------------------------------------------- tooltip
const tipEl = () => document.getElementById("tooltip");

export function showTip(evt, title, rows) {
  const tip = tipEl();
  tip.replaceChildren();
  const t = document.createElement("div");
  t.className = "title";
  t.textContent = title;
  tip.appendChild(t);
  for (const r of rows) {
    const row = document.createElement("div");
    row.className = "row";
    if (r.color) {
      const k = document.createElement("span");
      k.className = "key";
      k.style.background = r.color;
      row.appendChild(k);
    }
    const v = document.createElement("strong");
    v.textContent = r.value;
    const l = document.createElement("span");
    l.textContent = r.label;
    row.append(v, l);
    tip.appendChild(row);
  }
  tip.hidden = false;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > innerWidth - 4) x = evt.clientX - w - pad;
  if (y + h > innerHeight - 4) y = evt.clientY - h - pad;
  tip.style.left = `${Math.max(4, x)}px`;
  tip.style.top = `${Math.max(4, y)}px`;
}
export const hideTip = () => { tipEl().hidden = true; };

// ---------------------------------------------------------------- layout
function setup(el, legendItems) {
  el.replaceChildren();
  const canvas = document.createElement("canvas");
  el.appendChild(canvas);
  let legend = null;
  if (legendItems && legendItems.length) {
    legend = document.createElement("div");
    legend.className = "legend";
    for (const it of legendItems) {
      const span = document.createElement("span");
      if (it.ramp) {
        const lo = document.createElement("span");
        lo.textContent = it.lo;
        const bar = document.createElement("span");
        bar.className = "ramp";
        bar.style.background = `linear-gradient(90deg, ${it.ramp.join(",")})`;
        const hi = document.createElement("span");
        hi.textContent = it.hi;
        span.append(it.label ? `${it.label}: ` : "", lo, bar, hi);
      } else {
        const k = document.createElement("span");
        k.className = "key";
        k.style.background = it.color;
        span.append(k, it.label);
      }
      legend.appendChild(span);
    }
    el.appendChild(legend);
  }
  const W = el.clientWidth;
  const H = el.clientHeight - (legend ? legend.offsetHeight + 6 : 0);
  const dpr = window.devicePixelRatio || 1;
  canvas.style.height = `${H}px`;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return { canvas, ctx, W, H };
}

export function emptyState(el, msg) {
  el.replaceChildren();
  const d = document.createElement("div");
  d.className = "empty";
  d.textContent = msg;
  el.appendChild(d);
}

function niceTicks(min, max, count = 5) {
  if (!(max > min)) { max = min + 1; }
  const span = max - min;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / count)));
  const err = (count * step0) / span;
  const step = step0 * (err <= 0.15 ? 10 : err <= 0.35 ? 5 : err <= 0.75 ? 2 : 1);
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(+v.toPrecision(12));
  return ticks;
}

export function fmtNum(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(a >= 1e10 ? 0 : 1) + "B";
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M";
  if (a >= 1e4) return (v / 1e3).toFixed(a >= 1e5 ? 0 : 1) + "k";
  if (a >= 100 || Number.isInteger(v)) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  if (a === 0) return "0";
  return v.toPrecision(2);
}

// ---------------------------------------------------------------- line chart
/**
 * series: [{name, color, points: [[x, y], ...]}]
 * opts: {xLabel, yLabel, logX, yFormat, xFormat}
 */
export function lineChart(el, series, opts = {}) {
  series = series.filter((s) => s.points.length);
  if (!series.length) return emptyState(el, opts.empty || "No data yet");
  const legendItems = series.length >= 2 ? series.map((s) => ({ label: s.name, color: s.color })) : null;
  const { canvas, ctx, W, H } = setup(el, legendItems);
  if (W < 10 || H < 10) return;  // hidden or collapsed: nothing to draw into
  const T = tokens();
  const m = { l: opts.yLabel ? 64 : 48, r: 16, t: 10, b: 36 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const fx = opts.logX ? (v) => Math.log10(Math.max(0, v) + 1) : (v) => v;
  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
  const ys = series.flatMap((s) => s.points.map((p) => p[1])).filter(Number.isFinite);
  let x0 = fx(xs[0]), x1 = fx(xs[xs.length - 1]);
  if (x1 === x0) x1 = x0 + 1;
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  const padY = (y1 - y0) * 0.06 || Math.abs(y1) * 0.1 || 1;
  y0 -= padY; y1 += padY;
  if (opts.yMin !== undefined) y0 = Math.max(opts.yMin, y0);
  const yt = niceTicks(y0, y1, Math.max(3, Math.floor(ph / 50)));
  y0 = Math.min(y0, yt[0]); y1 = Math.max(y1, yt[yt.length - 1]);
  const X = (v) => m.l + ((fx(v) - x0) / (x1 - x0)) * pw;
  const Y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * ph;
  const yFmt = opts.yFormat || fmtNum, xFmt = opts.xFormat || fmtNum;

  let xt;
  if (opts.logX) {
    xt = [0];
    for (let p = 1; p <= xs[xs.length - 1]; p *= 10) xt.push(p);
  } else {
    xt = niceTicks(xs[0], xs[xs.length - 1], Math.max(3, Math.floor(pw / 90)));
  }

  function draw(hoverX) {
    ctx.clearRect(0, 0, W, H);
    ctx.font = "11px system-ui, sans-serif";
    // grid + y labels
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const v of yt) {
      const y = Math.round(Y(v)) + 0.5;
      ctx.strokeStyle = T.grid; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + pw, y); ctx.stroke();
      ctx.fillStyle = T.muted; ctx.fillText(yFmt(v), m.l - 6, y);
    }
    // x axis
    ctx.strokeStyle = T.axis;
    ctx.beginPath(); ctx.moveTo(m.l, m.t + ph + 0.5); ctx.lineTo(m.l + pw, m.t + ph + 0.5); ctx.stroke();
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = T.muted;
    for (const v of xt) {
      const x = X(v);
      if (x < m.l - 1 || x > m.l + pw + 1) continue;
      const label = xFmt(v), half = ctx.measureText(label).width / 2;
      ctx.fillText(label, Math.min(W - half - 1, Math.max(half + 1, x)), m.t + ph + 6);
    }
    if (opts.xLabel) { ctx.fillStyle = T.ink2; ctx.fillText(opts.xLabel, m.l + pw / 2, m.t + ph + 21); }
    if (opts.yLabel) {
      ctx.save(); ctx.translate(10, m.t + ph / 2); ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = "middle"; ctx.fillStyle = T.ink2; ctx.fillText(opts.yLabel, 0, 0); ctx.restore();
    }
    // lines
    ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.lineCap = "round";
    for (const s of series) {
      ctx.strokeStyle = s.color;
      ctx.beginPath();
      s.points.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
      ctx.stroke();
      if (s.points.length === 1) {
        ctx.fillStyle = s.color;
        ctx.beginPath(); ctx.arc(X(s.points[0][0]), Y(s.points[0][1]), 4, 0, 2 * Math.PI); ctx.fill();
      }
    }
    // crosshair
    if (hoverX !== undefined) {
      const hx = Math.round(X(hoverX)) + 0.5;
      ctx.strokeStyle = T.axis; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(hx, m.t); ctx.lineTo(hx, m.t + ph); ctx.stroke();
      for (const s of series) {
        const p = s.points.find((q) => q[0] === hoverX);
        if (!p) continue;
        ctx.fillStyle = s.color; ctx.strokeStyle = T.surface; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      }
    }
  }
  draw();

  canvas.addEventListener("pointermove", (evt) => {
    const r = canvas.getBoundingClientRect();
    const px = evt.clientX - r.left;
    if (px < m.l - 10 || px > m.l + pw + 10) { draw(); hideTip(); return; }
    let best = xs[0], bd = Infinity;
    for (const x of xs) { const d = Math.abs(X(x) - px); if (d < bd) { bd = d; best = x; } }
    draw(best);
    const rows = [];
    for (const s of series) {
      const p = s.points.find((q) => q[0] === best);
      if (p) rows.push({ color: s.color, value: yFmt(p[1]), label: s.name });
    }
    showTip(evt, `${opts.xLabel || "x"} ${xFmt(best)}`, rows);
  });
  canvas.addEventListener("pointerleave", () => { draw(); hideTip(); });
}

// ---------------------------------------------------------------- heatmap
/**
 * matrix: rows x cols numbers. opts: {rowLabels, colLabels, xLabel, yLabel,
 * format, domain: [min, max], legendLabel, tipTitle(r, c)}
 */
export function heatmap(el, matrix, opts = {}) {
  if (!matrix.length || !matrix[0].length) return emptyState(el, opts.empty || "No data yet");
  const T = tokens();
  const fmt = opts.format || fmtNum;
  const flat = matrix.flat().filter(Number.isFinite);
  if (!flat.length) return emptyState(el, opts.empty || "No values to show");
  // Diverging (e.g. "change vs. baseline"): symmetric around 0, two hues with
  // a neutral middle. Sequential (magnitudes): one hue, light -> dark.
  let [lo, hi] = opts.domain || [Math.min(...flat), Math.max(...flat)];
  if (opts.diverging && !opts.domain) { const a = Math.max(Math.abs(lo), Math.abs(hi)) || 1; lo = -a; hi = a; }
  const ramp = opts.diverging ? T.div : T.seq;
  const { canvas, ctx, W, H } = setup(el, [{ ramp, lo: fmt(lo), hi: fmt(hi), label: opts.legendLabel }]);
  if (W < 10 || H < 10) return;  // hidden or collapsed: nothing to draw into
  const nr = matrix.length, nc = matrix[0].length;
  const rowLabels = opts.rowLabels || matrix.map((_, i) => String(i));
  const colLabels = opts.colLabels || matrix[0].map((_, i) => String(i));
  const m = { l: opts.marginLeft || 44, r: 10, t: 6, b: opts.xLabel ? 36 : 22 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const cw = pw / nc, rh = ph / nr;
  const t = (v) => (hi > lo ? (v - lo) / (hi - lo) : 0.5);
  const missing = hexToRgb(T.grid.startsWith("#") ? T.grid : "#888888");
  const color = (v) => (Number.isFinite(v) ? rampRgb(ramp, t(v)) : missing);  // no value -> neutral grid tone
  const sparse = cw >= 6 && rh >= 6;  // big cells get a surface gap and an outline on hover

  // Render cells once into an offscreen image, then reuse it while hovering.
  const img = document.createElement("canvas");
  const dpr = window.devicePixelRatio || 1;
  img.width = canvas.width; img.height = canvas.height;
  const ictx = img.getContext("2d");
  ictx.scale(dpr, dpr);
  if (sparse) {
    for (let r = 0; r < nr; r++) for (let c = 0; c < nc; c++) {
      ictx.fillStyle = rgbStr(color(matrix[r][c]));
      ictx.fillRect(m.l + c * cw + 1, m.t + r * rh + 1, cw - 2, rh - 2);
    }
  } else {
    // Dense (e.g. thousands of neurons): one pixel per cell, scaled up.
    const px = document.createElement("canvas");
    px.width = nc; px.height = nr;
    const pctx = px.getContext("2d");
    const data = pctx.createImageData(nc, nr);
    for (let r = 0; r < nr; r++) for (let c = 0; c < nc; c++) {
      const [R, G, B] = color(matrix[r][c]);
      const o = 4 * (r * nc + c);
      data.data[o] = R; data.data[o + 1] = G; data.data[o + 2] = B; data.data[o + 3] = 255;
    }
    pctx.putImageData(data, 0, 0);
    ictx.imageSmoothingEnabled = false;
    // Thin surface lines between rows keep layers visually separate.
    for (let r = 0; r < nr; r++) ictx.drawImage(px, 0, r, nc, 1, m.l, m.t + r * rh + 1, pw, Math.max(1, rh - 2));
  }
  ictx.font = "11px system-ui, sans-serif";
  ictx.fillStyle = T.muted;
  ictx.textAlign = "right"; ictx.textBaseline = "middle";
  const rowEvery = Math.ceil(14 / rh);
  rowLabels.forEach((l, r) => { if (r % rowEvery === 0) ictx.fillText(l, m.l - 6, m.t + (r + 0.5) * rh); });
  ictx.textAlign = "center"; ictx.textBaseline = "top";
  const nTicks = Math.max(2, Math.min(nc, Math.floor(pw / 70)));
  for (let i = 0; i < nTicks; i++) {
    const c = Math.round((i * (nc - 1)) / Math.max(1, nTicks - 1));
    const half = ictx.measureText(colLabels[c]).width / 2;
    ictx.fillText(colLabels[c], Math.min(W - half - 1, Math.max(half + 1, m.l + (c + 0.5) * cw)), m.t + ph + 6);
  }
  if (opts.xLabel) { ictx.fillStyle = T.ink2; ictx.fillText(opts.xLabel, m.l + pw / 2, m.t + ph + 21); }
  if (opts.yLabel) {
    ictx.save(); ictx.translate(10, m.t + ph / 2); ictx.rotate(-Math.PI / 2);
    ictx.textBaseline = "middle"; ictx.fillStyle = T.ink2; ictx.fillText(opts.yLabel, 0, 0); ictx.restore();
  }

  function draw(hr, hc) {
    ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    ctx.restore();
    if (hr !== undefined) {
      ctx.strokeStyle = T.ink; ctx.lineWidth = 1.5;
      if (sparse) ctx.strokeRect(m.l + hc * cw + 0.5, m.t + hr * rh + 0.5, cw - 1, rh - 1);
      else ctx.strokeRect(m.l - 0.5, m.t + hr * rh + 0.5, pw + 1, rh);  // highlight the row
    }
  }
  draw();
  canvas.addEventListener("pointermove", (evt) => {
    const r = canvas.getBoundingClientRect();
    const c = Math.floor((evt.clientX - r.left - m.l) / cw), row = Math.floor((evt.clientY - r.top - m.t) / rh);
    if (c < 0 || c >= nc || row < 0 || row >= nr) { draw(); hideTip(); return; }
    draw(row, c);
    const title = opts.tipTitle ? opts.tipTitle(row, c) : `${rowLabels[row]}, ${colLabels[c]}`;
    const rows = opts.tipRows ? opts.tipRows(row, c) : [{ value: fmt(matrix[row][c]), label: opts.legendLabel || "value" }];
    showTip(evt, title, rows);
  });
  canvas.addEventListener("pointerleave", () => { draw(); hideTip(); });
  if (opts.onClick) {
    canvas.style.cursor = "pointer";
    canvas.addEventListener("click", (evt) => {
      const r = canvas.getBoundingClientRect();
      const c = Math.floor((evt.clientX - r.left - m.l) / cw), row = Math.floor((evt.clientY - r.top - m.t) / rh);
      if (c >= 0 && c < nc && row >= 0 && row < nr) opts.onClick(row, c);
    });
  }
}
