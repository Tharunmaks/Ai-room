"use strict";
/* Core helpers: DOM, API, icons, safe markdown, toasts and confetti. No dependencies. */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const store = {
  get(k, d = null) { try { const v = localStorage.getItem("ar." + k); return v === null ? d : v; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("ar." + k, v); } catch (e) { /* private mode */ } },
};

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts, body: opts.body === undefined ? undefined : JSON.stringify(opts.body) });
  let data = {};
  try { data = await r.json(); } catch (e) { /* empty body */ }
  if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : r.statusText || "Request failed");
  return data;
}

/* ── icons ── */
const ICONS = {
  classroom: '<path d="M3 21h18M5 21V9l7-5 7 5v12M9 21v-6h6v6"/>',
  subjects: '<rect x="3" y="3" width="7" height="7" rx="1.6"/><rect x="14" y="3" width="7" height="7" rx="1.6"/><rect x="3" y="14" width="7" height="7" rx="1.6"/><rect x="14" y="14" width="7" height="7" rx="1.6"/>',
  teacher: '<path d="M22 9 12 4 2 9l10 5 10-5z"/><path d="M6 11.5V16c0 1.4 2.7 3 6 3s6-1.6 6-3v-4.5"/><path d="M22 9v6"/>',
  library: '<path d="M4 19V5a1 1 0 0 1 1-1h3v16H5a1 1 0 0 1-1-1z"/><path d="M8 4h4v16H8"/><path d="m14 6.5 3.4-1 3.1 12.5-3.4 1z"/>',
  notebook: '<path d="M6 3h11a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6z"/><path d="M6 3v18M10 8h6M10 12h6"/>',
  report: '<path d="M4 20V11M10 20V4M16 20v-7M22 20H2"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
  plus: '<path d="M12 5v14M5 12h14"/>', play: '<path d="m7 4 13 8-13 8z"/>', ff: '<path d="m3 5 8 7-8 7zM12 5l8 7-8 7z"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2.5"/>', send: '<path d="m22 2-7 20-4-9-9-4zM22 2 11 13"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>', upload: '<path d="M12 16V4M7 9l5-5 5 5M4 20h16"/>', download: '<path d="M12 4v12M7 11l5 5 5-5M4 20h16"/>',
  trash: '<path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/>', check: '<path d="m5 12.5 4.5 4.5L19 7"/>', x: '<path d="M6 6l12 12M18 6 6 18"/>',
  chev: '<path d="m6 9 6 6 6-6"/>', sparkles: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 8h.01M11 12h1v5h1"/>', globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3.2 3 14.8 0 18M12 3c-3 3.2-3 14.8 0 18"/>',
  link: '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/>',
  back: '<path d="M15 6l-6 6 6 6"/>', next: '<path d="M9 6l6 6-6 6"/>',
};
const icon = (n, cls = "") => `<svg class="i ${cls}" viewBox="0 0 24 24" aria-hidden="true">${ICONS[n] || ""}</svg>`;

/* ── safe markdown (everything is escaped first) ── */
function inline(s) {
  s = esc(s);
  s = s.replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>")
    .replace(/(https?:\/\/[^\s<)]+[^\s<).,;:])/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
  return s;
}
function md(text) {
  const lines = String(text ?? "").replace(/\r/g, "").split("\n");
  const out = [];
  let list = null, quote = [], code = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  const flushQuote = () => { if (quote.length) { out.push(`<blockquote>${quote.map(inline).join("<br>")}</blockquote>`); quote = []; } };
  for (const raw of lines) {
    if (code !== null) {
      if (/^\s*```/.test(raw)) { out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`); code = null; } else code.push(raw);
      continue;
    }
    if (/^\s*```/.test(raw)) { closeList(); flushQuote(); code = []; continue; }
    const line = raw.trimEnd();
    let m;
    if ((m = line.match(/^>\s?(.*)$/))) { closeList(); quote.push(m[1]); continue; }
    flushQuote();
    if ((m = line.match(/^(#{2,6})\s+(.*)$/))) { closeList(); const lvl = Math.min(5, m[1].length + 1); out.push(`<h${lvl}>${inline(m[2])}</h${lvl}>`); continue; }
    if ((m = line.match(/^\s*[-*]\s+(.*)$/))) { if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; } out.push(`<li>${inline(m[1])}</li>`); continue; }
    if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) { if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; } out.push(`<li>${inline(m[1])}</li>`); continue; }
    if (list && /^\s{2,}\S/.test(line) && out.length) { out[out.length - 1] = out[out.length - 1].replace(/<\/li>$/, `<br>${inline(line.trim())}</li>`); continue; }
    closeList();
    if (!line.trim()) { out.push('<div class="sp"></div>'); continue; }
    out.push(`<p>${inline(line)}</p>`);
  }
  if (code !== null) out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
  closeList(); flushQuote();
  return out.join("").replace(/(<div class="sp"><\/div>){2,}/g, '<div class="sp"></div>');
}

/* ── toasts ── */
function toast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = `${kind === "err" ? icon("x") : kind === "ok" ? icon("check") : icon("info")}<span>${esc(msg)}</span>`;
  $("#toasts").appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity .3s"; setTimeout(() => el.remove(), 320); }, kind === "err" ? 6500 : 3500);
}

/* ── confetti ── */
function confetti(power = 1) {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const c = $("#confetti"), ctx = c.getContext("2d");
  c.width = innerWidth; c.height = innerHeight;
  const colors = ["#7f9bff", "#3fe0c0", "#ffb36b", "#ff6b84", "#ffe08a", "#c3a6ff"];
  const parts = Array.from({ length: Math.round(150 * power) }, () => ({
    x: innerWidth / 2 + (Math.random() - .5) * 240, y: innerHeight * .42, vx: (Math.random() - .5) * 15, vy: -Math.random() * 15 - 5,
    r: 4 + Math.random() * 6, rot: Math.random() * 6, vr: (Math.random() - .5) * .4, col: colors[(Math.random() * colors.length) | 0], life: 0,
  }));
  let raf;
  const tick = () => {
    ctx.clearRect(0, 0, c.width, c.height);
    let alive = 0;
    for (const p of parts) {
      p.life++; p.vy += .42; p.vx *= .992; p.x += p.vx; p.y += p.vy; p.rot += p.vr;
      if (p.y < c.height + 20 && p.life < 240) { alive++; ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.rot); ctx.fillStyle = p.col; ctx.globalAlpha = Math.max(0, 1 - p.life / 240); ctx.fillRect(-p.r / 2, -p.r / 3, p.r, p.r * .6); ctx.restore(); }
    }
    if (alive) raf = requestAnimationFrame(tick); else ctx.clearRect(0, 0, c.width, c.height);
  };
  cancelAnimationFrame(raf); tick();
}

/* ── small helpers ── */
const pct = (x) => `${Math.round(x * 100)}%`;
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
function ringSVG(frac, color, size = 100, stroke = 9, markAt = null) {
  const r = (size - stroke) / 2, c = 2 * Math.PI * r, f = Math.max(0, Math.min(1, frac));
  let mark = "";
  if (markAt !== null) {
    const a = markAt * 2 * Math.PI, in_ = r - stroke / 2 - 1, out_ = r + stroke / 2 + 1, m = size / 2;
    mark = `<line x1="${m + in_ * Math.cos(a)}" y1="${m + in_ * Math.sin(a)}" x2="${m + out_ * Math.cos(a)}" y2="${m + out_ * Math.sin(a)}" stroke="var(--ink)" stroke-width="2" opacity=".75"/>`;
  }
  const stroke_ = color === "grad" ? "url(#rg)" : color;
  return `<svg viewBox="0 0 ${size} ${size}"><defs><linearGradient id="rg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#7f9bff"/><stop offset="1" stop-color="#3fe0c0"/></linearGradient></defs>`
    + `<circle class="track" cx="${size / 2}" cy="${size / 2}" r="${r}" stroke-width="${stroke}"/>`
    + `<circle class="val" cx="${size / 2}" cy="${size / 2}" r="${r}" stroke-width="${stroke}" stroke="${stroke_}" stroke-dasharray="${c}" stroke-dashoffset="${c * (1 - f)}"/>${mark}</svg>`;
}
