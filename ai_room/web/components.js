"use strict";
/* Reusable UI pieces: the subject explorer, syllabus drawer and the teacher builder. */

const MODE_LABEL = {
  exact: (s) => `${icon("sparkles")} Exact practice`,
  mixed: (s) => `${icon("sparkles")} ${s.exact_classes}/12 exact + 📚`,
  textbook: () => "📚 Textbooks",
};
const MODE_HELP = {
  exact: "Every class has problems generated and checked by code, so the answer key is always correct.",
  mixed: "Some classes have exact generated problems; the others use textbooks you fetch or import.",
  textbook: "The teacher teaches from textbooks (fetched from Wikipedia/Wikibooks or imported by you) and asks fill-in-the-blank and multiple-choice questions built from them.",
};

/* ── subject explorer ── */
function createSubjectPicker(root, selected, { onChange = () => {}, compact = false } = {}) {
  const st = { tab: "All", q: "" };
  const cats = META.categories;
  root.innerHTML = `
    <div class="row wrap gap-s" style="margin-bottom:10px">
      <div class="grow" style="position:relative;min-width:200px">
        <input type="search" data-q placeholder="Search ${META.subjects.length} subjects…" aria-label="Search subjects" style="padding-left:38px">
        <span style="position:absolute;left:12px;top:10px;color:var(--muted)">${icon("search")}</span>
      </div>
      <button class="btn sm" data-all type="button">Select shown</button><button class="btn sm ghost" data-none type="button">Clear</button>
      <span class="chip on" data-count></span>
    </div>
    <div class="tabs" data-tabs role="tablist"></div>
    <div class="grid g-auto ${compact ? "compact" : ""}" data-grid style="margin-top:12px;${compact ? "grid-template-columns:repeat(auto-fill,minmax(200px,1fr));max-height:340px;overflow-y:auto;padding:2px" : ""}"></div>`;
  const grid = $("[data-grid]", root), tabs = $("[data-tabs]", root);
  const visible = () => META.subjects.filter((s) =>
    (st.tab === "All" || (st.tab === "Selected" ? selected.has(s.key) : s.category === st.tab)) && (!st.q || (s.name + " " + s.category).toLowerCase().includes(st.q)));
  const paintTabs = () => {
    const list = ["All", "Selected", ...cats];
    tabs.innerHTML = list.map((t) => {
      const n = t === "All" ? META.subjects.length : t === "Selected" ? selected.size : META.subjects.filter((s) => s.category === t).length;
      const k = t === "All" || t === "Selected" ? n : META.subjects.filter((s) => s.category === t && selected.has(s.key)).length + "/" + n;
      return `<button type="button" role="tab" class="chip ${st.tab === t ? "on" : ""}" data-tab="${esc(t)}">${esc(t)} <span class="muted">${k}</span></button>`;
    }).join("");
    $("[data-count]", root).textContent = `${selected.size} selected`;
  };
  const paint = () => {
    paintTabs();
    const v = visible();
    grid.innerHTML = v.length ? v.map((s) => `
      <div class="scard ${selected.has(s.key) ? "on" : ""}" role="button" tabindex="0" aria-pressed="${selected.has(s.key)}" data-key="${s.key}">
        <div class="top"><div class="ic">${s.icon}</div><b>${esc(s.name)}</b><div class="tick">${icon("check")}</div></div>
        <div class="foot"><span class="mode ${s.mode}">${MODE_LABEL[s.mode](s)}</span><button type="button" class="btn sm icon ghost" data-info="${s.key}" aria-label="Show the 12-class syllabus for ${esc(s.name)}">${icon("info")}</button></div>
      </div>`).join("") : `<div class="empty" style="grid-column:1/-1"><div class="big">🔎</div>No subjects match “${esc(st.q)}”.</div>`;
  };
  const toggle = (key) => { selected.has(key) ? selected.delete(key) : selected.add(key); paint(); onChange(); };
  grid.addEventListener("click", (e) => {
    const info = e.target.closest("[data-info]");
    if (info) { e.stopPropagation(); openSyllabus(info.dataset.info); return; }
    const c = e.target.closest("[data-key]"); if (c) toggle(c.dataset.key);
  });
  grid.addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && e.target.matches("[data-key]")) { e.preventDefault(); toggle(e.target.dataset.key); } });
  tabs.addEventListener("click", (e) => { const t = e.target.closest("[data-tab]"); if (t) { st.tab = t.dataset.tab; paint(); } });
  $("[data-q]", root).addEventListener("input", (e) => { st.q = e.target.value.trim().toLowerCase(); paint(); });
  $("[data-all]", root).onclick = () => { visible().forEach((s) => selected.add(s.key)); paint(); onChange(); };
  $("[data-none]", root).onclick = () => { selected.clear(); paint(); onChange(); };
  paint();
  return { paint };
}

function openSyllabus(key) {
  const s = META.subjects.find((x) => x.key === key);
  const host = $("#drawerHost");
  host.innerHTML = `<div class="drawer-bg" data-close></div>
    <aside class="drawer" role="dialog" aria-label="${esc(s.name)} syllabus">
      <div class="row"><div class="av" style="width:52px;height:52px;font-size:28px">${s.icon}</div><div class="grow"><h3 style="font-size:20px">${esc(s.name)}</h3><div class="muted small">${esc(s.category)}</div></div>
        <button class="btn icon ghost" data-close aria-label="Close">${icon("x")}</button></div>
      <p class="small" style="margin-top:14px"><span class="mode ${s.mode}">${MODE_LABEL[s.mode](s)}</span> ${esc(MODE_HELP[s.mode])}</p>
      <div class="muted small">Exam question types: ${s.types.map((t) => `<span class="chip" style="padding:1px 8px">${t}</span>`).join(" ")}</div>
      <div class="tl">${s.syllabus.map((t, i) => `<div class="step" data-n="${i + 1}"><b>Class ${i + 1} · ${esc(META.difficulty[i])}</b><div>${esc(t)}</div></div>`).join("")}</div>
    </aside>`;
  const close = () => { host.innerHTML = ""; document.removeEventListener("keydown", onKey); };
  const onKey = (e) => e.key === "Escape" && close();
  document.addEventListener("keydown", onKey);
  $$("[data-close]", host).forEach((b) => (b.onclick = close));
  $(".drawer", host).querySelector("button")?.focus();
}

/* ── teacher builder ── */
const TONE_LABEL = { warm: "Warm", neutral: "Neutral", strict: "Strict", playful: "Playful", scholarly: "Scholarly" };
const APPROACH_LABEL = { explain: "Explain first", socratic: "Ask first (Socratic)", drill: "Short theory + drills" };
const AVATARS = ["🌱", "🏛️", "📐", "🎈", "🔬", "🏋️", "🦉", "🧙", "👩‍🏫", "👨‍🏫", "🤓", "🚀"];
const DIALS = [
  ["depth", "Depth", 1, 5, "textbook readings & detail"], ["examples", "Worked examples", 0, 5, "per idea"], ["practice", "Practice problems", 0, 10, "after each lesson"],
  ["praise", "Encouragement", 0, 3, "kind words"], ["humor", "Humour", 0, 3, "jokes per lesson"],
];

function createTeacherBuilder(root, init, { onChange = () => {}, showPresets = true } = {}) {
  const state = { persona: init.persona || "patient", style: { ...(META.personas.find((p) => p.key === (init.persona === "custom" ? "patient" : init.persona)) || META.personas[0]), ...(init.style || {}) }, dirty: init.persona === "custom" };
  delete state.style.key;
  const presetCards = () => META.personas.map((p) => `
    <button type="button" class="tcard ${!state.dirty && state.persona === p.key ? "on" : ""}" data-preset="${p.key}" aria-pressed="${!state.dirty && state.persona === p.key}">
      <div class="face">${p.avatar}</div><b>${esc(p.name)}</b><div class="q">${esc(p.tagline)}</div>
      <div class="row wrap gap-s"><span class="chip">${TONE_LABEL[p.tone]}</span><span class="chip">${p.approach === "explain" ? "Explains" : p.approach === "socratic" ? "Socratic" : "Drills"}</span></div>
    </button>`).join("");
  const dials = () => `
    <div class="stack">
      <div class="row wrap"><label class="field grow" style="min-width:160px"><span>Teacher name</span><input data-f="name" value="${esc(state.style.name)}" maxlength="40"></label></div>
      <div><div class="field"><span>Face</span></div><div class="emoji-row">${AVATARS.map((a) => `<button type="button" data-avatar="${a}" class="${state.style.avatar === a ? "on" : ""}" aria-label="Avatar ${a}">${a}</button>`).join("")}</div></div>
      <div><div class="field"><span>Tone — how they speak</span></div><div class="seg">${Object.keys(TONE_LABEL).map((t) => `<button type="button" data-tone="${t}" class="${state.style.tone === t ? "on" : ""}">${TONE_LABEL[t]}</button>`).join("")}</div></div>
      <div><div class="field"><span>Approach — how they teach</span></div><div class="seg">${Object.keys(APPROACH_LABEL).map((t) => `<button type="button" data-approach="${t}" class="${state.style.approach === t ? "on" : ""}">${APPROACH_LABEL[t]}</button>`).join("")}</div></div>
      ${DIALS.map(([k, label, lo, hi, hint]) => `<div class="dial"><label for="d-${k}">${label}<div class="tiny muted">${hint}</div></label><input id="d-${k}" type="range" min="${lo}" max="${hi}" value="${state.style[k]}" data-dial="${k}"><b>${state.style[k]}</b></div>`).join("")}
      <div class="row wrap gap-l"><label class="switch"><input type="checkbox" data-f="hints" ${state.style.hints ? "checked" : ""}><i></i>Point at common mistakes</label>
        <label class="switch"><input type="checkbox" data-f="sources" ${state.style.sources ? "checked" : ""}><i></i>Cite textbook sources</label></div>
    </div>`;
  const paint = () => {
    root.innerHTML = `${showPresets ? `<div class="grid g-auto" data-presets style="grid-template-columns:repeat(auto-fill,minmax(210px,1fr));margin-bottom:18px">${presetCards()}</div>` : ""}
      <div class="card flat"><h3>${icon("sparkles")} Fine-tune the teacher ${state.dirty ? '<span class="chip warn">custom</span>' : ""}</h3><div data-dials>${dials()}</div></div>`;
  };
  const change = () => { state.dirty = true; onChange(); };
  root.addEventListener("click", (e) => {
    const p = e.target.closest("[data-preset]");
    if (p) { const preset = META.personas.find((x) => x.key === p.dataset.preset); state.persona = preset.key; state.style = { ...preset }; delete state.style.key; state.dirty = false; paint(); onChange(); return; }
    const tone = e.target.closest("[data-tone]"); if (tone) { state.style.tone = tone.dataset.tone; change(); paintDials(); return; }
    const ap = e.target.closest("[data-approach]"); if (ap) { state.style.approach = ap.dataset.approach; change(); paintDials(); return; }
    const av = e.target.closest("[data-avatar]"); if (av) { state.style.avatar = av.dataset.avatar; change(); paintDials(); }
  });
  const paintDials = () => { const d = $("[data-dials]", root); if (d) d.innerHTML = dials(); const h = $(".card.flat h3", root); if (h && state.dirty && !h.querySelector(".chip")) h.insertAdjacentHTML("beforeend", '<span class="chip warn">custom</span>'); };
  root.addEventListener("input", (e) => {
    const d = e.target.dataset?.dial;
    if (d) { state.style[d] = Number(e.target.value); e.target.nextElementSibling.textContent = e.target.value; change(); return; }
    if (e.target.dataset?.f === "name") { state.style.name = e.target.value; change(); }
  });
  root.addEventListener("change", (e) => { const f = e.target.dataset?.f; if (f === "hints" || f === "sources") { state.style[f] = e.target.checked; change(); } });
  paint();
  return {
    state,
    value: () => (state.dirty ? { persona: "custom", style: { ...state.style } } : { persona: state.persona, style: {} }),
  };
}

function lessonPreviewHTML(res) {
  return `<div class="board"><div class="slate"><div class="row" style="padding:14px 18px"><div class="av" style="background:rgba(255,255,255,.1)">${esc(res.avatar)}</div>
    <div class="meta"><b>${esc(res.name)}</b><small style="color:var(--chalk-dim)">${res.available ? "live preview of a lesson" : "no material yet"}</small></div></div>
    <div class="body">${md(res.text)}</div></div></div>`;
}
