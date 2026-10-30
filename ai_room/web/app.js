"use strict";
/* AI Room — the application: state, views, live class feed and the setup wizard. */
let META = null;
const S = {
  rooms: [], id: null, room: null, view: store.get("view", "classroom"), feed: [], lastId: 0, es: null, renderer: null,
  notebook: null, lib: null, job: null, jobTimer: null, libSubjects: "room", dirty: null, chatTo: "student", nbFilter: { subject: "all", kind: "all", q: "" },
  teacherDraft: null, previewSubject: "physics", previewGrade: 5,
};
const VIEWS = [
  ["classroom", "Classroom", "classroom"], ["subjects", "Subjects", "subjects"], ["teacher", "Teacher", "teacher"],
  ["workshop", "Workshop", "tools"], ["library", "Library", "library"], ["notebook", "Notebook", "notebook"], ["report", "Report", "report"],
];
const SUBJ = (k) => META.subjects.find((s) => s.key === k);
const fmtStatus = (r) => (r.busy && r.status === "idle" ? "⏳ Starting…" : r.busy ? `${/exam/i.test(r.status) ? "📝" : /lesson/i.test(r.status) ? "📖" : /fetch|textbook/i.test(r.status) ? "📚" : "⏳"} ${r.status}` : "");

/* ───────────────────────── boot ───────────────────────── */
async function boot() {
  try { META = await api("/api/meta"); } catch (e) { $("#page").innerHTML = `<div class="empty"><div class="big">⚠️</div>Could not reach the AI Room server.<br><span class="small">${esc(e.message)}</span></div>`; return; }
  $("#nav").innerHTML = VIEWS.map(([k, label, ic]) => `<button data-view="${k}" aria-label="${label}">${icon(ic)}<span class="lbl">${label}</span></button>`).join("");
  $("#newBtn").innerHTML = `${icon("plus")}<span class="hide-sm">New classroom</span>`;
  paintTheme();
  document.addEventListener("click", onClick);
  document.addEventListener("submit", (e) => { e.preventDefault(); const f = e.target.closest("[data-form]"); if (f) FORMS[f.dataset.form]?.(f); });
  $("#roomSelect").onchange = (e) => openRoom(e.target.value);
  $("#themeBtn").onclick = toggleTheme;
  $("#modal").addEventListener("click", (e) => { if (e.target === e.currentTarget) e.currentTarget.close(); });
  await loadRooms();
  setInterval(poll, 2500);
}

function paintTheme() {
  const dark = (document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark")) === "dark";
  $("#themeBtn").innerHTML = `${icon(dark ? "sun" : "moon")}<span class="lbl">${dark ? "Light mode" : "Dark mode"}</span>`;
}
function toggleTheme() {
  const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next; store.set("theme", next); paintTheme();
}

async function loadRooms(selectId) {
  S.rooms = await api("/api/classrooms");
  const sel = $("#roomSelect");
  sel.innerHTML = S.rooms.map((r) => `<option value="${r.id}">${esc(r.name)} · class ${Math.min(r.grade, 12)}</option>`).join("");
  $("#roomPicker").hidden = !S.rooms.length;
  const want = selectId || (S.rooms.find((r) => r.id === store.get("room")) || S.rooms[0] || {}).id;
  if (want) { sel.value = want; await openRoom(want); } else { S.id = null; S.room = null; render(); }
}

async function openRoom(id) {
  S.id = id; store.set("room", id); S.feed = []; S.lastId = 0; S.notebook = null; S.renderer = null;
  if (S.es) S.es.close();
  try { S.room = await api(`/api/classrooms/${id}`); } catch (e) { toast(e.message, "err"); S.room = null; render(); return; }
  S.es = new EventSource(`/api/classrooms/${id}/events?after=0`);
  S.es.onmessage = (e) => ingest(JSON.parse(e.data));
  S.liveFrom = Infinity;
  fetch(`/api/classrooms/${id}/messages`).then((r) => r.json()).then((rows) => { S.liveFrom = rows.length ? rows[rows.length - 1].id : 0; });
  render();
}

function ingest(m) {
  if (m.id <= S.lastId) return;
  S.lastId = m.id; S.feed.push(m);
  if (S.view === "classroom" && S.renderer) { S.renderer.add(m, m.id > S.liveFrom); if (S.pinned) scrollFeed(); }
  const k = m.meta?.kind;
  if (["score", "promotion", "notes"].includes(k) || (m.role === "teacher" && ["lesson", "review"].includes(k))) { S.notebook = null; refreshRoom(); }
  if (k === "promotion" && m.id > S.liveFrom) { confetti(/GRADUATED/.test(m.content) ? 2 : 1); }
}
const isNearBottom = () => { const f = $("#feed"); return !f || f.scrollHeight - f.scrollTop - f.clientHeight < 140; };
const scrollFeed = () => { const f = $("#feed"); if (f) f.scrollTo({ top: f.scrollHeight, behavior: "instant" }); };

async function refreshRoom() {
  if (!S.id) return;
  try { S.room = await api(`/api/classrooms/${S.id}`); } catch (e) { return; }
  if (S.view === "classroom") paintClassroomParts();
  else if (S.view === "report") render();
  const opt = $(`#roomSelect option[value="${S.id}"]`); if (opt) opt.textContent = `${S.room.name} · class ${Math.min(S.room.grade, 12)}`;
}
let pollBusy = false;
async function poll() {
  if (pollBusy || document.hidden) return;
  pollBusy = true;
  try {
    if (S.id && (S.view === "classroom" || S.room?.busy)) await refreshRoom();
  } finally { pollBusy = false; }
}

/* ───────────────────────── routing / render ───────────────────────── */
function setView(v) { S.view = v; store.set("view", v); render(); window.scrollTo({ top: 0 }); }
function render() {
  $$("#nav button").forEach((b) => b.classList.toggle("on", b.dataset.view === S.view));
  const t = VIEWS.find((v) => v[0] === S.view); $("#pageTitle").textContent = S.view === "report" ? "Report card" : t ? t[1] : "AI Room";
  const page = $("#page");
  if (S.view === "workshop") { page.innerHTML = `<div class="view" id="ws"></div>`; renderWorkshop(); return; }
  if (S.view === "library") { page.innerHTML = `<div class="view" id="lib"></div>`; renderLibrary(); return; }
  if (!S.room) { page.innerHTML = `<div class="view">${welcomeHTML()}</div>`; return; }
  const fn = { classroom: renderClassroom, subjects: renderSubjects, teacher: renderTeacher, notebook: renderNotebook, report: renderReport }[S.view];
  page.innerHTML = `<div class="view" id="view"></div>`; fn($("#view"));
}

function welcomeHTML() {
  return `<div class="card" style="padding:clamp(22px,5vw,56px);text-align:center;overflow:hidden;position:relative">
    <div style="font-size:64px;animation:pop .8s">🎓</div>
    <h2 style="font-size:clamp(28px,5vw,46px);letter-spacing:-.04em;margin:8px 0 10px">A classroom where AI learns <span style="background:linear-gradient(90deg,var(--accent),var(--accent-2),var(--accent-3));-webkit-background-clip:text;background-clip:text;color:transparent">without training</span></h2>
    <p class="muted" style="max-width:62ch;margin:0 auto 22px;font-size:17px">A student model of up to 27B parameters attends 12 classes in up to 100 subjects. Its weights never change — it learns by taking notes, asking questions and studying from its own notebook.</p>
    <button class="btn primary big" data-act="wizard">${icon("plus")} Create your first classroom</button>
  </div>
  <div class="grid g-3" style="margin-top:16px">
    <div class="card"><h3>👩‍🏫 The teacher is not an AI</h3><p class="muted small">Lessons come from exact problem generators, open textbooks (Wikipedia, Wikibooks, or your own files) and question banks. Answer keys are computed or quoted — never guessed.</p></div>
    <div class="card"><h3>📓 The student keeps a notebook</h3><p class="muted small">After each lesson the student writes notes, asks the teacher questions, and re-reads the right pages before every exam question. Mistakes become lessons in its notebook.</p></div>
    <div class="card"><h3>🏆 Pass every subject to move up</h3><p class="muted small">Score at least 90% in <i>every</i> subject to reach the next class. Fail, and the teacher reviews the mistakes, re-teaches, and sets a fresh exam.</p></div>
  </div>`;
}

/* ───────────────────────── classroom view ───────────────────────── */
function renderClassroom(el) {
  el.innerHTML = `<div class="classroom">
    <div class="stack" style="gap:18px; min-width:0">
      <section class="card hero" id="hero"></section>
      <section class="card feed-wrap" aria-label="Class feed">
        <div class="feed-head"><b>${icon("sparkles")} Class feed</b><span class="muted small">live — lessons, questions and exams as they happen</span><div class="grow"></div>
          <button class="btn sm ghost" data-act="collapse">Collapse lessons</button></div>
        <div class="feed" id="feed" role="log" aria-live="polite"></div>
        <form class="dock" data-form="talk">
          <div class="seg" role="group" aria-label="Talk to"><button type="button" data-act="chat-to" data-arg="student" class="${S.chatTo === "student" ? "on" : ""}">🤖 Student</button><button type="button" data-act="chat-to" data-arg="teacher" class="${S.chatTo === "teacher" ? "on" : ""}">${esc(S.room.teacher.avatar)} Teacher</button></div>
          <input id="talkText" autocomplete="off" placeholder="${S.chatTo === "student" ? "Ask the student anything — it answers from its notebook…" : "Ask the teacher a question — it looks in the textbooks…"}" aria-label="Message">
          <button class="btn primary" aria-label="Send">${icon("send")}</button>
        </form>
      </section>
    </div>
    <aside class="side-stack" id="side"></aside></div>`;
  paintClassroomParts();
  S.renderer = new FeedRenderer($("#feed"));
  S.pinned = true;  // follow new messages until the user scrolls up
  $("#feed").addEventListener("scroll", () => { S.pinned = isNearBottom(); }, { passive: true });
  S.feed.forEach((m) => S.renderer.add(m, false));
  scrollFeed();
}

function paintClassroomParts() {
  const r = S.room; if (!r || !$("#hero")) return;
  const g = Math.min(r.grade, 12), card = r.report_card[g] || {}, subs = r.subjects;
  const passed = subs.filter((s) => card[s] && card[s].best >= r.pass_mark).length;
  const frac = r.graduated ? 1 : subs.length ? passed / subs.length : 0;
  const path = Array.from({ length: 12 }, (_, i) => {
    const n = i + 1, cls = r.graduated ? "won" : n < r.grade ? "done" : n === r.grade ? "now" : "";
    return `<div class="n ${cls}" title="Class ${n} — ${META.difficulty[i]}">${n < r.grade || r.graduated ? icon("check") : n}</div>${n < 12 ? `<div class="l ${n < r.grade || r.graduated ? "done" : ""}"></div>` : ""}`;
  }).join("");
  const t = r.teacher, running = r.busy;
  $("#hero").innerHTML = `
    <div class="ring">${ringSVG(frac, "grad", 128, 10)}<div class="mid"><small>${r.graduated ? "GRADUATED" : "CLASS"}</small><b>${r.graduated ? "🎓" : g}</b><small>${r.graduated ? "all 12 passed" : `${passed}/${subs.length} passed`}</small></div></div>
    <div style="min-width:0">
      <div class="muted small" style="font-weight:700;letter-spacing:.06em;text-transform:uppercase">${esc(r.name)}</div>
      <h2>${r.graduated ? "Graduated with honours" : `Class ${g} · ${META.difficulty[g - 1]}`}</h2>
      <div class="tags"><span class="chip">🤖 ${esc(r.config.student_name)} · ${esc(r.config.student.model)}</span><span class="chip">${esc(t.avatar)} ${esc(t.name)}</span><span class="chip">🎯 pass ≥ ${pct(r.pass_mark)} in every subject</span></div>
      <div class="cta">
        <button class="btn primary big" data-act="run" data-arg="class" ${running || r.graduated ? "disabled" : ""}>${icon("play", "fill")} ${r.graduated ? "Done" : running ? "Running…" : "Start class"}</button>
        <button class="btn" data-act="run" data-arg="all" ${running || r.graduated ? "disabled" : ""}>${icon("ff", "fill")} All 12</button>
        <button class="btn ghost" data-act="run" data-arg="lessons" title="Teach the lessons without sitting the exams" ${running || r.graduated ? "disabled" : ""}>Lessons</button>
        <button class="btn ghost" data-act="run" data-arg="exam" title="Sit the exams now" ${running || r.graduated ? "disabled" : ""}>Exam</button>
        <button class="btn danger" data-act="stop" ${running ? "" : "disabled"}>${icon("stop", "fill")} Stop</button>
      </div>
    </div>
    <div style="grid-column:1/-1;min-width:0"><div class="path" aria-label="Progress through 12 classes">${path}</div>
      <div class="ticker"><span class="pulse ${running ? "" : "idle"}"></span><span class="txt">${running ? esc(fmtStatus(r)) : r.graduated ? "All 12 classes completed 🎉" : "Ready. Press “Start class” — lessons first, then exams."}</span></div></div>`;
  const tilesHTML = `<div class="tiles">${subs.map((k) => {
    const s = SUBJ(k), c = card[k], best = c ? c.best : 0, ok = c && best >= r.pass_mark, live = running && r.status.includes(s.name);
    return `<div class="tile ${ok ? "pass" : c ? "fail" : ""} ${live ? "live" : ""}" title="${esc(s.name)}">
      <div class="mini">${ringSVG(best, ok ? "var(--good)" : c ? "var(--warn)" : "var(--accent)", 52, 6, r.pass_mark)}<div class="ic">${s.icon}</div></div>
      <div style="min-width:0"><b>${esc(s.name)}</b><small>${live ? "in progress…" : c ? `${pct(best)} · try ${c.attempts}${ok ? " ✓" : ""}` : "not started"}</small></div></div>`;
  }).join("")}</div>`;
  const pathEl = $(".path"), nowEl = $(".path .n.now");
  if (pathEl && nowEl && !pathEl.dataset.centered) { pathEl.scrollLeft = nowEl.offsetLeft - pathEl.clientWidth / 2 + 15; pathEl.dataset.centered = "1"; }
  const nb = r.notebook, sz = r.student_size_b;
  $("#side").innerHTML = `
    <div class="card"><h3>📋 This class <span class="muted small" style="margin-left:auto;font-weight:500">${passed}/${subs.length} passed</span></h3>${tilesHTML}</div>
    <div class="card"><h3>🤖 The student</h3>
      <div class="row"><div class="av student">🎓</div><div class="grow"><b>${esc(r.config.student_name)}</b><div class="muted small">${esc(r.config.student.model)}${r.config.student.backend === "demo" ? " · simulated" : ""}</div></div></div>
      <div class="meter-27"><div class="bar"><span style="width:${Math.min(100, sz / 27 * 100)}%"></span><i class="lim"></i></div>
        <div class="row small muted" style="justify-content:space-between;margin-top:6px"><span>${sz < 10 ? sz.toFixed(1) : Math.round(sz)}B parameters</span><span>limit 27B</span></div></div>
      ${r.config.student.backend === "demo" ? `<p class="small muted" style="margin:10px 0 0">⚠️ Simulated student: it is shown the answer key and does <b>not</b> learn. Use it only to explore the interface.</p>` : ""}
    </div>
    <div class="card"><h3>📓 Notebook <button class="btn sm ghost" style="margin-left:auto" data-act="view" data-arg="notebook">Open</button></h3>
      <div class="grid g-2" style="gap:8px"><div class="stat"><b>${nb.lesson}</b><span class="muted small">lesson notes</span></div><div class="stat"><b>${nb.qa}</b><span class="muted small">Q&amp;A</span></div>
        <div class="stat"><b>${nb.mistake}</b><span class="muted small">mistake logs</span></div><div class="stat"><b>${nb.summary}</b><span class="muted small">cheat-sheets</span></div></div></div>
    <div class="card"><h3>${esc(t.avatar)} ${esc(t.name)} <button class="btn sm ghost" style="margin-left:auto" data-act="view" data-arg="teacher">Change</button></h3>
      <div class="row wrap gap-s"><span class="chip">${TONE_LABEL[t.tone]}</span><span class="chip">${t.approach === "explain" ? "Explains first" : t.approach === "socratic" ? "Asks first" : "Drills"}</span><span class="chip">${t.practice} practice</span><span class="chip">${t.humor ? "😄 humour" : "serious"}</span></div>
      <p class="muted small" style="margin:10px 0 0">${esc(t.tagline)}</p></div>`;
}

/* ── feed renderer ── */
class FeedRenderer {
  constructor(el) { this.el = el; this.group = null; this.lastBoard = null; }
  node(html, cls = "") { const d = document.createElement("div"); if (cls) d.className = cls; d.innerHTML = html; this.el.appendChild(d); return d; }
  add(m, live) {
    const k = m.meta?.kind || "", meta = m.meta || {};
    if (m.role === "system") return this.system(m, k, meta, live);
    if (m.role === "student" && k === "exam_answer") return this.examItem(m, meta);
    if (m.role === "teacher" && k === "lesson") return this.lesson(m, meta);
    if (m.role === "teacher" && k === "review") return this.node(`<div class="who small" style="font-weight:700;margin-bottom:6px">${esc(meta.avatar || "📝")} ${esc(m.speaker)} reviews the mistakes</div><div class="md">${md(m.content)}</div>`, "review");
    const av = m.role === "teacher" ? esc(meta.avatar || "🎓") : m.role === "user" ? "🧑" : "🎓";
    this.node(`<div class="av ${m.role}">${av}</div><div class="bubble"><div class="who">${esc(m.speaker)}</div><div class="md">${md(m.content)}</div></div>`, `msg ${m.role}`);
  }
  system(m, k, meta, live) {
    if (k === "score") {
      if (this.group) { this.group.dataset.done = "1"; const ok = meta.passed; this.group.querySelector(".verdict").innerHTML = ok ? '<span class="chip good">passed</span>' : '<span class="chip bad">not yet</span>'; if (!live || ok) this.group.open = false; this.group = null; }
      const p = Math.round((meta.score || 0) * 100), name = SUBJ(meta.subject)?.name || "";
      return this.node(`<div class="top"><b>${SUBJ(meta.subject)?.icon || ""} ${esc(name)}</b><span class="chip ${meta.passed ? "good" : "warn"}">${p}% · ${meta.passed ? "PASSED" : "not yet"}</span></div>
        <div class="meter" style="--mark:${(S.room?.pass_mark || .9) * 100}%"><span style="width:${p}%"></span></div>`, `event score ${meta.passed ? "pass" : ""}`);
    }
    if (k === "promotion") return this.node(m.content.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])), "promo");
    if (k === "error") return this.node(md(m.content), "event error");
    if (k === "library") return this.node(`${icon("library")} ${esc(m.content)}`, "event library");
    if (k === "status") return this.node(esc(m.content), "event");
    this.node(esc(m.content), "event");
  }
  lesson(m, meta) {
    if (this.lastBoard) this.lastBoard.open = false;
    const sub = SUBJ(meta.subject), src = (meta.sources || []).slice(0, 8);
    const board = this.node(`<div class="slate"><details open><summary>
        <div class="av" style="background:rgba(255,255,255,.1);border-color:rgba(255,255,255,.2)">${esc(meta.avatar || "🎓")}</div>
        <div class="meta"><b>${esc(m.speaker)} teaches ${esc(sub?.name || "")}</b><small>${sub?.icon || ""} lesson on the board</small></div><span class="chev">${icon("chev")}</span></summary>
        <div class="body">${md(m.content)}</div>
        ${src.length ? `<div class="srcs">${src.filter((s) => s.title).map((s) => (s.url ? `<a class="chip" href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">📚 ${esc(s.title)}</a>` : `<span class="chip">📚 ${esc(s.title)}</span>`)).join("")}</div>` : ""}
      </details></div>`, "board");
    this.lastBoard = board.querySelector("details");
  }
  examItem(m, meta) {
    const sub = SUBJ(meta.subject);
    if (!this.group || this.group.dataset.subject !== meta.subject || this.group.dataset.done) {
      const d = document.createElement("details"); d.className = "exam-group"; d.open = true; d.dataset.subject = meta.subject;
      d.innerHTML = `<summary><span>📝</span><span>${esc(sub?.name || "Exam")} exam</span><span class="verdict"></span><span class="dots"></span></summary><div class="items"></div>`;
      this.el.appendChild(d); this.group = d;
    }
    const s = meta.score ?? 0, cls = s >= 1 ? "" : s > 0 ? "part" : "bad";
    const finals = [...(meta.reply || "").matchAll(/final answer\s*[:=-]\s*(.+)/gi)];
    const given = meta.given || (finals.length ? finals[finals.length - 1][1] : (meta.reply || "").slice(0, 160));
    const q = document.createElement("div"); q.className = `qa ${cls}`;
    q.innerHTML = `<div class="q md"><b>${meta.n || ""}.</b> ${md(meta.question || m.content)}</div>
      <div class="ans">🎓 <b>${esc(m.speaker)}:</b> <code>${esc(String(given).slice(0, 240))}</code> ${s >= 1 ? '<span class="chip good">✓ correct</span>' : s > 0 ? `<span class="chip warn">${pct(s)}</span>` : '<span class="chip bad">✗ wrong</span>'}</div>
      ${s < 1 ? `<div class="exp"><b>Correct answer:</b> ${esc(meta.expected || "")}${meta.feedback ? `<div class="small muted" style="margin-top:4px">${md(String(meta.feedback).slice(0, 600))}</div>` : ""}</div>` : ""}
      <details class="reply"><summary>Show the student's work</summary><pre>${esc(meta.reply || "")}</pre></details>`;
    this.group.querySelector(".items").appendChild(q);
    this.group.querySelector(".dots").insertAdjacentHTML("beforeend", `<i class="${s >= 1 ? "" : s > 0 ? "p" : "x"}" title="Question ${meta.n}"></i>`);
  }
}

/* ───────────────────────── subjects view ───────────────────────── */
function renderSubjects(el) {
  const r = S.room, original = new Set(r.config.subjects), selected = new Set(r.config.subjects);
  el.innerHTML = `<div class="page-head"><div><h2>Choose the subjects</h2><p>The student must score at least <b>${pct(r.pass_mark)}</b> in <i>every</i> selected subject to move to the next class. Open ⓘ on any subject to see its 12-class syllabus.</p></div></div>
    <div class="card"><div id="pick"></div></div><div id="savebar"></div>`;
  const bar = () => {
    const changed = selected.size !== original.size || [...selected].some((k) => !original.has(k));
    $("#savebar").innerHTML = changed ? `<div class="savebar"><b>${selected.size}</b> subject${selected.size === 1 ? "" : "s"} selected<span class="muted small">${S.room.busy ? "Stop the class before saving." : "Changes apply from the next lesson."}</span><div class="grow"></div>
      <button class="btn ghost" data-act="reset-subjects">Reset</button><button class="btn primary" data-act="save-subjects" ${selected.size && !S.room.busy ? "" : "disabled"}>Save subjects</button></div>` : "";
  };
  S.dirty = { selected, original, bar };
  createSubjectPicker($("#pick"), selected, { onChange: bar });
}

/* ───────────────────────── teacher view ───────────────────────── */
function renderTeacher(el) {
  const r = S.room;
  el.innerHTML = `<div class="page-head"><div><h2>Choose how your teacher behaves</h2><p>The teacher is not an AI model — it follows these dials. Tone and approach change how every lesson, answer and review is written.</p></div><div class="grow"></div>
      <button class="btn primary" data-act="save-teacher">${icon("check")} Use this teacher</button></div>
    <div class="tgrid">
      <div id="tb"></div>
      <div class="card preview-pane"><h3>${icon("search")} Live lesson preview</h3>
        <div class="row wrap" style="margin-bottom:12px"><label class="field grow"><span>Subject</span><select id="pvSubject">${r.config.subjects.concat(META.subjects.map((s) => s.key).filter((k) => !r.config.subjects.includes(k) && SUBJ(k).mode !== "textbook")).map((k) => `<option value="${k}" ${k === S.previewSubject ? "selected" : ""}>${esc(SUBJ(k).name)}</option>`).join("")}</select></label>
          <label class="field" style="width:120px"><span>Class</span><select id="pvGrade">${Array.from({ length: 12 }, (_, i) => `<option value="${i + 1}" ${i + 1 === S.previewGrade ? "selected" : ""}>${i + 1}</option>`).join("")}</select></label></div>
        <div id="pv"><div class="skeleton" style="height:260px"></div></div></div></div>`;
  if (!r.config.subjects.includes(S.previewSubject)) S.previewSubject = r.config.subjects[0];
  $("#pvSubject").value = S.previewSubject;
  const tb = createTeacherBuilder($("#tb"), { persona: r.config.persona, style: r.config.persona === "custom" ? r.config.style : {} }, { onChange: preview });
  S.teacherDraft = tb;
  const doPreview = async () => {
    const v = tb.value();
    try {
      const res = await api("/api/teacher/preview", { method: "POST", body: { ...v, subject: S.previewSubject, grade: S.previewGrade, student_name: r.config.student_name } });
      if ($("#pv")) $("#pv").innerHTML = lessonPreviewHTML(res);
    } catch (e) { if ($("#pv")) $("#pv").innerHTML = `<div class="empty">${esc(e.message)}</div>`; }
  };
  function preview() { debouncedPreview(); }
  const debouncedPreview = debounce(doPreview, 300);
  $("#pvSubject").onchange = (e) => { S.previewSubject = e.target.value; doPreview(); };
  $("#pvGrade").onchange = (e) => { S.previewGrade = Number(e.target.value); doPreview(); };
  doPreview();
}

/* ───────────────────────── library view ───────────────────────── */
async function renderLibrary() {
  const host = $("#lib"); if (!host) return;
  try { S.lib = await api("/api/library"); } catch (e) { host.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
  const L = S.lib, mine = S.room ? S.room.config.subjects : [];
  const rows = (S.libSubjects === "all" ? META.subjects.map((s) => s.key) : mine.length ? mine : META.subjects.slice(0, 12).map((s) => s.key));
  const running = L.jobs.find((j) => j.status === "running");
  if (running) { S.job = running.id; clearTimeout(S.jobTimer); S.jobTimer = setTimeout(() => S.view === "library" && renderLibrary(), 1200); }
  const needs = (k) => { const st = L.subjects[k], sub = SUBJ(k); return sub.mode !== "exact" && st.passages < 8 && st.questions < 8; };
  host.innerHTML = `
    <div class="page-head"><div><h2>The library</h2><p>The teacher's textbooks and question banks. Subjects with exact practice need none; the rest are taught from here. Text from Wikipedia and Wikibooks is CC BY-SA and always cited.</p></div></div>
    <div class="grid g-4" style="margin-bottom:16px">
      <div class="card stat-card"><b>${L.totals.docs}</b><span>books &amp; articles</span></div><div class="card stat-card"><b>${L.totals.passages.toLocaleString()}</b><span>passages the teacher can quote</span></div>
      <div class="card stat-card"><b>${L.totals.questions}</b><span>question-bank items</span></div><div class="card stat-card"><b>${Math.round(L.totals.chars / 1000).toLocaleString()}k</b><span>characters of text</span></div></div>
    ${running ? `<div class="job" style="margin-bottom:16px"><div class="row"><span class="pulse"></span><b>Fetching textbooks…</b><span class="muted small grow">${esc(running.message)}</span><button class="btn sm danger" data-act="job-stop" data-arg="${running.id}">Stop</button></div>
      <div class="bar"><span style="width:${Math.round((running.done / Math.max(1, running.total)) * 100)}%"></span></div><div class="small muted">${running.added} articles added so far · subject ${running.index + 1} of ${running.subjects.length}</div></div>` : ""}
    ${L.jobs.filter((j) => j.status !== "running").slice(-1).map((j) => `<div class="card flat" style="margin-bottom:16px"><div class="row wrap"><span class="chip ${j.errors.length && !j.added ? "bad" : "good"}">last fetch: ${j.errors.length && !j.added ? "failed" : esc(j.status)}</span><span class="small">${j.added} articles added</span>${j.errors.length ? `<span class="small" style="color:var(--bad)">${esc(j.errors[0])}</span>` : ""}</div></div>`).join("")}
    <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(320px,1fr));align-items:start">
      <div class="card"><h3>${icon("download")} Fetch free textbooks</h3><p class="muted small">Pick subjects; the teacher reads Simple English Wikipedia (early classes), Wikipedia and Wikibooks for every topic in the 12-class syllabus. Be patient — it is deliberately gentle with Wikimedia's servers.</p>
        <div class="row wrap">${(() => { const n = mine.filter((k) => SUBJ(k).mode !== "exact").length; return `<button class="btn primary" data-act="fetch-room" ${running || !n ? "disabled" : ""}>${icon("download")} ${n ? `Fetch for the ${n} subject${n === 1 ? "" : "s"} that need books` : "None of my subjects need books"}</button>`; })()}<button class="btn" data-act="fetch-pick" ${running ? "disabled" : ""}>Choose subjects…</button></div></div>
      <div class="card"><h3>${icon("upload")} Add your own books</h3>
        <label class="field"><span>Subject</span><select id="impSubject">${META.subjects.map((s) => `<option value="${s.key}" ${S.room && s.key === S.room.config.subjects[0] ? "selected" : ""}>${s.icon} ${esc(s.name)}</option>`).join("")}</select></label>
        <div class="drop" id="drop" tabindex="0" role="button" aria-label="Choose files to import" style="margin-top:10px"><div class="big">📚</div><b>Drop files here</b><div class="muted small">txt · md · html · pdf · epub · docx</div><input type="file" id="file" multiple hidden accept=".txt,.md,.markdown,.html,.htm,.pdf,.epub,.docx,.rst"></div>
        <div class="row" style="margin-top:10px"><input id="impUrl" type="url" placeholder="…or paste a page URL"><button class="btn" data-act="import-url">${icon("link")} Import</button></div><div id="impLog" class="small" style="margin-top:8px"></div></div>
      <div class="card"><h3>${icon("check")} Question banks</h3><p class="muted small">Import ready-made questions as JSON or JSON-lines — our format (<code>type, question, options, answer</code>) or MMLU, ARC, SciQ, GSM8K and MBPP exports. Each needs a <code>subject</code> key, or pick one here.</p>
        <label class="field"><span>Default subject (optional)</span><select id="bankSubject"><option value="">— from the file —</option>${META.subjects.map((s) => `<option value="${s.key}">${s.icon} ${esc(s.name)}</option>`).join("")}</select></label>
        <textarea id="bankText" placeholder='{"subject":"biology","type":"mcq","question":"…?","options":["a","b","c","d"],"answer":"B"}' style="margin-top:10px"></textarea>
        <div class="row" style="margin-top:10px"><label class="btn sm" for="bankFile">${icon("upload")} From file</label><input id="bankFile" type="file" accept=".json,.jsonl,.txt" hidden><div class="grow"></div><button class="btn primary" data-act="bank-import">Import questions</button></div></div>
      <div class="card"><h3>${icon("search")} Ask the library</h3>
        <select id="qSubject">${META.subjects.map((s) => `<option value="${s.key}">${s.icon} ${esc(s.name)}${L.subjects[s.key].passages ? "" : " (empty)"}</option>`).join("")}</select>
        <div class="row" style="margin-top:10px"><input id="qText" placeholder="e.g. what is a senate?"><button class="btn" data-act="lib-search" aria-label="Search">${icon("search")}</button></div><div id="qOut" class="stack" style="margin-top:12px"></div></div>
    </div>
    <div class="card" style="margin-top:16px"><div class="row wrap" style="margin-bottom:6px"><h3 style="margin:0">Coverage</h3><div class="seg"><button data-act="lib-rows" data-arg="room" class="${S.libSubjects === "room" ? "on" : ""}">My subjects</button><button data-act="lib-rows" data-arg="all" class="${S.libSubjects === "all" ? "on" : ""}">All 100</button></div></div>
      <div class="cov muted tiny" style="border-bottom:1px solid var(--line-2)"><span></span><span>Subject</span><span class="hid">Exact practice</span><span class="hid">Textbooks</span><span class="hid">Questions</span><span></span></div>
      ${rows.map((k) => { const s = SUBJ(k), st = L.subjects[k]; return `<div class="cov"><span style="font-size:20px">${s.icon}</span><span><b>${esc(s.name)}</b> ${needs(k) ? '<span class="chip warn tiny">needs textbooks</span>' : '<span class="chip good tiny">ready</span>'}</span>
        <span class="hid"><div class="bar"><span style="width:${s.exact_classes / 12 * 100}%"></span></div><span class="tiny muted">${s.exact_classes}/12 classes</span></span><span class="hid small">${st.docs} books · ${st.passages}</span><span class="hid small">${st.questions}</span>
        <span class="row gap-s">${s.mode !== "exact" ? `<button class="btn sm" data-act="fetch-one" data-arg="${k}" ${running ? "disabled" : ""}>Fetch</button>` : ""}${st.docs ? `<button class="btn sm ghost" data-act="docs" data-arg="${k}">Books</button>` : ""}</span></div>`; }).join("")}
      <div id="docsOut"></div></div>`;
  bindLibrary();
}

function bindLibrary() {
  const drop = $("#drop"), file = $("#file");
  if (!drop) return;
  drop.onclick = () => file.click(); drop.onkeydown = (e) => (e.key === "Enter" || e.key === " ") && file.click();
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => uploadFiles([...e.dataTransfer.files]));
  file.onchange = () => uploadFiles([...file.files]);
  $("#bankFile").onchange = async (e) => { const f = e.target.files[0]; if (f) $("#bankText").value = await f.text(); };
  $("#qText").onkeydown = (e) => e.key === "Enter" && ACTIONS["lib-search"]();
}
async function uploadFiles(files) {
  const subject = $("#impSubject").value, log = $("#impLog"); log.innerHTML = "";
  for (const f of files) {
    if (f.size > 25 * 1024 * 1024) { log.innerHTML += `<div style="color:var(--bad)">✗ ${esc(f.name)}: over 25 MB</div>`; continue; }
    const buf = new Uint8Array(await f.arrayBuffer()); let bin = ""; for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
    try { const r = await api("/api/library/import", { method: "POST", body: { subject, filename: f.name, data_b64: btoa(bin) } }); log.innerHTML += `<div style="color:var(--good)">✓ ${esc(r.title)} — ${r.chars.toLocaleString()} characters</div>`; }
    catch (e) { log.innerHTML += `<div style="color:var(--bad)">✗ ${esc(f.name)}: ${esc(e.message)}</div>`; }
  }
  renderLibraryKeepLog(log.innerHTML);
}
async function renderLibraryKeepLog(html) { await renderLibrary(); const l = $("#impLog"); if (l) l.innerHTML = html; }

/* ───────────────────────── notebook view ───────────────────────── */
async function renderNotebook(el) {
  const r = S.room;
  if (!S.notebook) { el.innerHTML = `<div class="skeleton" style="height:260px"></div>`; S.notebook = await api(`/api/classrooms/${S.id}/notebook`); }
  const f = S.nbFilter, kinds = ["lesson", "qa", "mistake", "summary"], names = { lesson: "Lesson notes", qa: "Q&A", mistake: "Mistake logs", summary: "Cheat-sheets" };
  const notes = S.notebook.filter((n) => (f.subject === "all" || n.subject === f.subject) && (f.kind === "all" || n.kind === f.kind) && (!f.q || n.content.toLowerCase().includes(f.q))).reverse();
  el.innerHTML = `<div class="page-head"><div><h2>${esc(r.config.student_name)}'s notebook</h2><p>This is how the student learns without training: before every exam question it re-reads the most relevant pages. Mistake logs are ranked highest.</p></div></div>
    <div class="row wrap" style="margin-bottom:14px"><div class="tabs grow"><button class="chip ${f.subject === "all" ? "on" : ""}" data-act="nb" data-k="subject" data-arg="all">All subjects</button>${r.config.subjects.map((k) => `<button class="chip ${f.subject === k ? "on" : ""}" data-act="nb" data-k="subject" data-arg="${k}">${SUBJ(k).icon} ${esc(SUBJ(k).name)}</button>`).join("")}</div></div>
    <div class="row wrap" style="margin-bottom:18px"><button class="chip ${f.kind === "all" ? "on" : ""}" data-act="nb" data-k="kind" data-arg="all">Everything</button>${kinds.map((k) => `<button class="chip ${f.kind === k ? "on" : ""}" data-act="nb" data-k="kind" data-arg="${k}">${names[k]} <span class="muted">${S.notebook.filter((n) => n.kind === k).length}</span></button>`).join("")}
      <div class="grow"></div><div style="position:relative;min-width:200px"><input type="search" id="nbq" value="${esc(f.q)}" placeholder="Search notes…" style="padding-left:38px"><span style="position:absolute;left:12px;top:10px;color:var(--muted)">${icon("search")}</span></div></div>
    ${notes.length ? `<div class="grid g-auto" style="grid-template-columns:repeat(auto-fill,minmax(340px,1fr));align-items:start;gap:22px 18px">${notes.slice(0, 120).map((n) => `<div class="note ${n.kind}"><span class="tag">${names[n.kind]} · ${esc(SUBJ(n.subject)?.name || "")} · class ${n.grade}</span><div class="txt md">${md(n.content.slice(0, 2400))}</div></div>`).join("")}</div>`
      : `<div class="empty"><div class="big">📓</div>Nothing here yet.<br><span class="small">Run a class — the student writes notes after every lesson.</span></div>`}`;
  const q = $("#nbq"); if (q) q.oninput = debounce((e) => { S.nbFilter.q = e.target.value.trim().toLowerCase(); renderNotebook(el).then(() => { const i = $("#nbq"); i.focus(); i.setSelectionRange(i.value.length, i.value.length); }); }, 250);
}

/* ───────────────────────── report card ───────────────────────── */
function renderReport(el) {
  const r = S.room, subs = r.subjects, card = r.report_card, pass = r.pass_mark;
  const cell = (g, s) => { const c = card[g]?.[s]; if (!c) return `<td class="${g === r.grade ? "now" : ""}">·</td>`; const cls = c.best >= pass ? "pass" : c.best >= .6 ? "mid" : "low"; return `<td class="${cls} ${g === r.grade ? "now" : ""}" title="${esc(SUBJ(s).name)} · class ${g}: best ${pct(c.best)} after ${plural(c.attempts, "attempt")}">${c.best >= pass ? Math.round(c.best * 100) : Math.round(c.best * 100)}</td>`; };
  const all = Object.values(card).flatMap((row) => Object.values(row)), avg = all.length ? all.reduce((a, c) => a + c.best, 0) / all.length : 0;
  const passedCells = all.filter((c) => c.best >= pass).length;
  el.innerHTML = `<div class="page-head"><div><h2>Report card</h2><p>Best score per subject in every class. Green means ≥ ${pct(pass)} (passed). The student moves up only when an entire column is green.</p></div></div>
    <div class="grid g-4" style="margin-bottom:16px"><div class="card stat-card"><b>${r.graduated ? "12/12" : `${Math.min(12, r.grade - 1)}/12`}</b><span>classes completed</span></div><div class="card stat-card"><b>${passedCells}</b><span>subject exams passed</span></div>
      <div class="card stat-card"><b>${all.length ? pct(avg) : "–"}</b><span>average best score</span></div><div class="card stat-card"><b>${Object.values(card).reduce((a, row) => a + Math.max(0, ...Object.values(row).map((c) => c.attempts)), 0)}</b><span>exam sittings</span></div></div>
    <div class="card"><div class="scroll-x"><table class="heat"><thead><tr><th>Subject</th>${Array.from({ length: 12 }, (_, i) => `<th title="${META.difficulty[i]}">${i + 1}</th>`).join("")}</tr></thead><tbody>
      ${subs.map((s) => `<tr><td>${SUBJ(s).icon} ${esc(SUBJ(s).name)}</td>${Array.from({ length: 12 }, (_, i) => cell(i + 1, s)).join("")}</tr>`).join("")}</tbody></table></div>
      <div class="row wrap gap-s small muted" style="margin-top:12px"><span class="chip good">≥ ${pct(pass)} passed</span><span class="chip warn">60–${Math.round(pass * 100) - 1}%</span><span class="chip bad">&lt; 60%</span><span class="chip">· not taken</span><span class="chip" style="box-shadow:inset 0 0 0 2px var(--accent)">current class</span></div></div>`;
}

/* ───────────────────────── actions ───────────────────────── */
const ACTIONS = {
  view: (a) => setView(a),
  wizard: () => openWizard(),
  async run(action) { try { await api(`/api/classrooms/${S.id}/run`, { method: "POST", body: { action } }); S.room.busy = true; paintClassroomParts(); } catch (e) { toast(e.message, "err"); } },
  async stop() { try { await api(`/api/classrooms/${S.id}/stop`, { method: "POST" }); toast("Stopping after the current step…"); } catch (e) { toast(e.message, "err"); } },
  "chat-to": (a) => { S.chatTo = a; $$("[data-act=chat-to]").forEach((b) => b.classList.toggle("on", b.dataset.arg === a)); $("#talkText").placeholder = a === "student" ? "Ask the student anything — it answers from its notebook…" : "Ask the teacher a question — it looks in the textbooks…"; $("#talkText").focus(); },
  collapse: () => { S.renderer?.el.querySelectorAll(".board details, .exam-group").forEach((d) => (d.open = false)); },
  "reset-subjects": () => { S.dirty.selected.clear(); S.dirty.original.forEach((k) => S.dirty.selected.add(k)); renderSubjects($("#view")); },
  async "save-subjects"() { try { S.room = await api(`/api/classrooms/${S.id}/config`, { method: "PATCH", body: { subjects: [...S.dirty.selected] } }); toast("Subjects saved", "ok"); renderSubjects($("#view")); } catch (e) { toast(e.message, "err"); } },
  async "save-teacher"() {
    const v = S.teacherDraft.value();
    try { S.room = await api(`/api/classrooms/${S.id}/config`, { method: "PATCH", body: v }); toast(`${S.room.teacher.name} is your teacher now`, "ok"); setView("classroom"); } catch (e) { toast(e.message, "err"); }
  },
  nb: (a, el) => { S.nbFilter[el.dataset.k] = a; renderNotebook($("#view")); },
  "lib-rows": (a) => { S.libSubjects = a; renderLibrary(); },
  async "fetch-room"() { startFetch(S.room.config.subjects.filter((k) => SUBJ(k).mode !== "exact")); },
  "fetch-one": (k) => startFetch([k]),
  "fetch-pick": () => openFetchPicker(),
  async "job-stop"(id) { await api(`/api/library/jobs/${id}/stop`, { method: "POST" }); toast("Stopping…"); },
  async "import-url"() {
    const url = $("#impUrl").value.trim(); if (!url) return;
    try { const r = await api("/api/library/import_url", { method: "POST", body: { subject: $("#impSubject").value, url } }); toast(`Imported “${r.title}”`, "ok"); renderLibrary(); } catch (e) { toast(e.message, "err"); }
  },
  async "bank-import"() {
    try { const r = await api("/api/library/bank", { method: "POST", body: { subject: $("#bankSubject").value || null, text: $("#bankText").value } }); toast(`Added ${r.added} questions`, "ok"); renderLibrary(); } catch (e) { toast(e.message, "err"); }
  },
  async "lib-search"() {
    const q = $("#qText").value.trim(); if (!q) return;
    try {
      const rows = await api(`/api/library/search?subject=${$("#qSubject").value}&q=${encodeURIComponent(q)}&k=4`);
      const words = q.toLowerCase().split(/\W+/).filter((w) => w.length > 2);
      $("#qOut").innerHTML = rows.length ? rows.map((p) => `<div class="result"><div class="small muted" style="margin-bottom:4px"><b>${esc(p.title)}</b>${p.heading ? ` · ${esc(p.heading)}` : ""} · ${esc(p.source)}${p.url ? ` · <a href="${esc(p.url)}" target="_blank" rel="noopener noreferrer">open</a>` : ""}</div>${highlight(p.text.slice(0, 600), words)}</div>`).join("") : `<div class="muted small">Nothing found. Add a book about this subject first.</div>`;
    } catch (e) { toast(e.message, "err"); }
  },
  async docs(k) {
    const docs = await api(`/api/library/documents?subject=${k}`);
    $("#docsOut").innerHTML = `<div class="stack" style="margin-top:14px"><b>${SUBJ(k).icon} Books for ${esc(SUBJ(k).name)}</b>${docs.map((d) => `<div class="row small"><span class="grow">${esc(d.title)} <span class="muted">· ${esc(d.source)} · ${d.passages} passages</span></span><button class="btn sm danger" data-act="doc-del" data-arg="${d.id}" aria-label="Delete">${icon("trash")}</button></div>`).join("") || "<span class='muted'>None</span>"}</div>`;
  },
  async "doc-del"(id) { await api(`/api/library/documents/${id}`, { method: "DELETE" }); renderLibrary(); },
};
const highlight = (text, words) => esc(text).replace(new RegExp(`(${words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") || "$^"})`, "gi"), "<mark>$1</mark>");

function onClick(e) {
  const el = e.target.closest("[data-act],[data-view]");
  if (!el) return;
  if (el.dataset.view) { setView(el.dataset.view); return; }
  const fn = ACTIONS[el.dataset.act]; if (fn) fn(el.dataset.arg, el);
}
const FORMS = {
  async talk() {
    const input = $("#talkText"), text = input.value.trim(); if (!text) return;
    input.value = ""; input.disabled = true;
    try { await api(`/api/classrooms/${S.id}/talk`, { method: "POST", body: { text, to: S.chatTo } }); } catch (e) { toast(e.message, "err"); } finally { input.disabled = false; input.focus(); }
  },
};
async function startFetch(subjects) {
  if (!subjects.length) { toast("Those subjects use exact practice problems — no textbooks needed.", ""); return; }
  try { await api("/api/library/fetch", { method: "POST", body: { subjects } }); toast("Fetching textbooks…", "ok"); renderLibrary(); } catch (e) { toast(e.message, "err"); }
}
function openFetchPicker() {
  const modal = $("#modal"), sel = new Set(S.room ? S.room.config.subjects.filter((k) => SUBJ(k).mode !== "exact") : []);
  modal.innerHTML = `<div class="mh"><h3 style="font-size:18px">Fetch textbooks for…</h3><button class="btn icon ghost" style="margin-left:auto" data-close aria-label="Close">${icon("x")}</button></div>
    <div class="mb"><div id="fp"></div></div><div class="mf"><span class="muted small grow">Fetching ~30 articles per subject, politely. You can keep using the app.</span><button class="btn primary" id="fgo">${icon("download")} Fetch</button></div>`;
  modal.showModal();
  createSubjectPicker($("#fp"), sel, { compact: true });
  $("[data-close]", modal).onclick = () => modal.close();
  $("#fgo").onclick = () => { modal.close(); startFetch([...sel]); };
}

/* ───────────────────────── setup wizard ───────────────────────── */
function openWizard() {
  const modal = $("#modal"), W = { step: 0, name: "My Classroom", student_name: "Ada", backend: "ollama", model: "qwen2.5:14b", base_url: "", params: "", questions: 10, pass: 90, attempts: 5, sq: 3, autofetch: true,
    subjects: new Set(["math", "physics", "logic", "code", "research"]), teacher: null, models: null, ollamaErr: "" };
  const STEPS = ["Student", "Teacher", "Subjects", "Rules"];
  const paint = () => {
    const body = [stepStudent, stepTeacher, stepSubjects, stepRules][W.step]();
    modal.innerHTML = `<div class="mh"><div><div class="muted tiny" style="font-weight:700;letter-spacing:.06em;text-transform:uppercase">New classroom · step ${W.step + 1} of 4</div><h3 style="font-size:20px">${["Meet the student", "Pick the teacher", "Choose the subjects", "Set the rules"][W.step]}</h3></div>
      <div class="steps">${STEPS.map((_, i) => `<i class="${i <= W.step ? "on" : ""}"></i>`).join("")}</div><button class="btn icon ghost" data-close aria-label="Close">${icon("x")}</button></div>
      <div class="mb" id="wbody"></div>
      <div class="mf"><span class="err" id="werr" role="alert"></span>${W.step ? `<button class="btn" id="wback">${icon("back")} Back</button>` : ""}<button class="btn primary" id="wnext">${W.step === 3 ? `${icon("check")} Create classroom` : `Next ${icon("next")}`}</button></div>`;
    $("#wbody").appendChild(body);
    $("[data-close]", modal).onclick = () => modal.close();
    if ($("#wback")) $("#wback").onclick = () => { collect(); W.step--; paint(); };
    $("#wnext").onclick = next;
  };
  const el = (html) => { const d = document.createElement("div"); d.innerHTML = html; return d; };
  function stepStudent() {
    const guess = /(\d+(?:\.\d+)?)\s*b\b/i.test(W.model);
    const d = el(`<div class="stack">
      <div class="seg" id="wb" role="group" aria-label="Student backend">
        <button type="button" data-b="ollama" class="${W.backend === "ollama" ? "on" : ""}">🦙 Ollama</button><button type="button" data-b="openai" class="${W.backend === "openai" ? "on" : ""}">🔌 OpenAI-compatible</button><button type="button" data-b="demo" class="${W.backend === "demo" ? "on" : ""}">🧪 Simulated (demo)</button></div>
      <div class="row wrap"><label class="field grow"><span>Student's name</span><input id="wsn" value="${esc(W.student_name)}" maxlength="30"></label><label class="field grow"><span>Classroom name</span><input id="wn" value="${esc(W.name)}" maxlength="40"></label></div>
      <div id="wmodel" ${W.backend === "demo" ? "hidden" : ""}>
        <div class="row wrap"><label class="field grow"><span>Model (must be ≤ 27B parameters)</span><input id="wm" list="models" value="${esc(W.model)}" placeholder="qwen2.5:14b" autocomplete="off"><datalist id="models">${(W.models || []).map((m) => `<option value="${esc(m)}">`).join("")}</datalist></label>
          ${W.backend === "ollama" ? `<button type="button" class="btn" id="wdetect" style="margin-top:20px">Detect installed models</button>` : ""}</div>
        <div class="small" id="wdet" style="margin-top:6px;color:${W.ollamaErr ? "var(--bad)" : "var(--muted)"}">${esc(W.ollamaErr || (W.models ? `${W.models.length} model(s) found.` : ""))}</div>
        <div class="row wrap" style="margin-top:10px"><label class="field grow"><span>Server address (optional)</span><input id="wu" value="${esc(W.base_url)}" placeholder="${W.backend === "ollama" ? "http://localhost:11434" : "http://localhost:8000/v1"}"></label>
          <label class="field" style="width:200px"><span>Size in billions ${guess ? "(detected from name)" : "(if not in the name)"}</span><input id="wp" type="number" step="0.1" min="0.1" max="27.9" value="${esc(W.params)}" placeholder="e.g. 14"></label></div></div>
      <div class="card flat" style="padding:14px"><b>${icon("info")} Why 27B?</b><p class="small muted" style="margin:6px 0 0">The classroom is an alternative to GPU training: nothing is trained, so a model of up to 27 billion parameters is plenty. Larger models are refused.${W.backend === "demo" ? " <b>The simulated student is shown the answer key and does not really learn</b> — use it just to explore the app." : ""}</p></div></div>`);
    d.querySelectorAll("[data-b]").forEach((b) => (b.onclick = () => { collect(); W.backend = b.dataset.b; if (W.backend === "demo") W.model = "simulated-student"; else if (W.model === "simulated-student") W.model = W.backend === "ollama" ? "qwen2.5:14b" : ""; paint(); }));
    const det = d.querySelector("#wdetect"); if (det) det.onclick = async () => { collect(); det.disabled = true; const r = await api(`/api/ollama/models${W.base_url ? `?base_url=${encodeURIComponent(W.base_url)}` : ""}`); W.models = r.models; W.ollamaErr = r.ok ? (r.models.length ? "" : "Ollama is running but has no models. Try: ollama pull qwen2.5:14b") : `${r.error}. Start it with “ollama serve”.`; paint(); };
    return d;
  }
  function stepTeacher() {
    const d = el(`<div id="wt"></div>`);
    queueMicrotask(() => { W.teacher = createTeacherBuilder($("#wt"), W.teacherInit || { persona: "patient", style: {} }, { onChange: () => (W.teacherInit = W.teacher.value()) }); });
    return d;
  }
  function stepSubjects() { const d = el(`<div id="ws"></div>`); queueMicrotask(() => createSubjectPicker($("#ws"), W.subjects, { compact: true })); return d; }
  function stepRules() {
    const needs = [...W.subjects].filter((k) => SUBJ(k).mode !== "exact");
    return el(`<div class="stack" style="gap:18px">
      <div class="dial" style="grid-template-columns:170px 1fr 44px"><label for="wq">Questions per exam<div class="tiny muted">per subject, each sitting</div></label><input id="wq" type="range" min="3" max="30" value="${W.questions}"><b id="wqv">${W.questions}</b></div>
      <div class="dial" style="grid-template-columns:170px 1fr 44px"><label for="wpass">Pass mark<div class="tiny muted">needed in every subject</div></label><input id="wpass" type="range" min="50" max="100" value="${W.pass}"><b id="wpv">${W.pass}%</b></div>
      <div class="dial" style="grid-template-columns:170px 1fr 44px"><label for="wa">Attempts per class<div class="tiny muted">before the class is held</div></label><input id="wa" type="range" min="1" max="10" value="${W.attempts}"><b id="wav">${W.attempts}</b></div>
      <div class="dial" style="grid-template-columns:170px 1fr 44px"><label for="wsq">Student questions<div class="tiny muted">asked after each lesson</div></label><input id="wsq" type="range" min="0" max="6" value="${W.sq}"><b id="wsv">${W.sq}</b></div>
      <label class="switch"><input type="checkbox" id="waf" ${W.autofetch ? "checked" : ""}><i></i>Fetch missing textbooks automatically ${needs.length ? `<span class="chip warn">${needs.length} subject${needs.length === 1 ? "" : "s"} need them</span>` : '<span class="chip good">none needed</span>'}</label>
      <div class="card flat small muted">${W.subjects.size} subject${W.subjects.size === 1 ? "" : "s"} × 12 classes. To pass class <i>n</i>, ${esc(W.student_name)} must score ≥ ${W.pass}% in all of them; failed subjects are reviewed, re-taught and re-examined with new questions.</div></div>`);
  }
  function collect() {
    const v = (id) => $(id, modal);
    if (W.step === 0 && v("#wsn")) { W.student_name = v("#wsn").value.trim() || "Ada"; W.name = v("#wn").value.trim() || "My Classroom"; if (v("#wm")) W.model = v("#wm").value.trim(); if (v("#wu")) W.base_url = v("#wu").value.trim(); if (v("#wp")) W.params = v("#wp").value; }
    if (W.step === 1 && W.teacher) W.teacherInit = W.teacher.value();
    if (W.step === 3 && v("#wq")) { W.questions = +v("#wq").value; W.pass = +v("#wpass").value; W.attempts = +v("#wa").value; W.sq = +v("#wsq").value; W.autofetch = v("#waf").checked; }
  }
  async function next() {
    collect(); const err = $("#werr");
    if (W.step === 0 && W.backend !== "demo" && !W.model) { err.textContent = "Enter the student model's name."; return; }
    if (W.step === 2 && !W.subjects.size) { err.textContent = "Pick at least one subject."; return; }
    if (W.step < 3) { W.step++; paint(); return; }
    const btn = $("#wnext"); btn.disabled = true; err.textContent = "Checking the student model…";
    const t = W.teacherInit || { persona: "patient", style: {} };
    try {
      const { id } = await api("/api/classrooms", { method: "POST", body: { name: W.name, config: {
        student: { backend: W.backend, model: W.model, base_url: W.base_url || null }, student_name: W.student_name, student_params_b: W.params ? Number(W.params) : null,
        persona: t.persona, style: t.style, subjects: [...W.subjects], questions_per_exam: W.questions, pass_mark: W.pass / 100, max_attempts: W.attempts, student_questions: W.sq, auto_fetch: W.autofetch } } });
      modal.close(); S.view = "classroom"; store.set("view", "classroom"); await loadRooms(id); toast("Classroom ready — press “Start class”", "ok"); confetti(.5);
    } catch (e) { err.textContent = e.message; btn.disabled = false; }
  }
  paint(); modal.showModal();
}
$("#newBtn")?.addEventListener("click", () => openWizard());

boot();
