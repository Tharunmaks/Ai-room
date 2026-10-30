"use strict";
const $ = (s) => document.querySelector(s);
const api = async (path, opts = {}) => {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
};

let META = null, ROOM = null, source = null, lastId = 0;

// --- tiny, safe markdown renderer -------------------------------------------
function esc(s) { return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function md(text) {
  const parts = String(text).split(/```[a-zA-Z]*\n?/);
  return parts.map((p, i) => i % 2 ? `<pre><code>${esc(p)}</code></pre>` :
    esc(p).replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
      .replace(/^#{1,4} (.*)$/gm, "<b>$1</b>")
      .replace(/\n/g, "<br>")).join("");
}

// --- setup ------------------------------------------------------------------
async function init() {
  META = await api("/api/meta");
  document.querySelectorAll(".maxb").forEach((e) => (e.textContent = META.max_params_b));
  const personaOpts = META.personas.map((p) => `<option value="${p.key}">${esc(p.name)} — ${esc(p.style.split(".")[0])}</option>`).join("")
    + `<option value="custom">Custom…</option>`;
  $("#newPersona").innerHTML = personaOpts;
  $("#cfgPersona").innerHTML = personaOpts;
  renderPicker("#newSubjects", "new", ["math", "physics", "logic", "code", "research"]);
  $("#newPersona").onchange = (e) => ($("#newForm").custom_style.hidden = e.target.value !== "custom");
  $("#cfgPersona").onchange = (e) => ($("#cfgCustom").hidden = e.target.value !== "custom");
  $("#newRoomBtn").onclick = $("#emptyNew").onclick = () => { $("#newError").textContent = ""; $("#newDialog").showModal(); };
  $("#cancelNew").onclick = () => $("#newDialog").close();
  $("#newForm").onsubmit = createRoom;
  $("#roomSelect").onchange = (e) => openRoom(e.target.value);
  document.querySelectorAll("[data-run]").forEach((b) => (b.onclick = () => run(b.dataset.run)));
  $("#stopBtn").onclick = () => ROOM && api(`/api/classrooms/${ROOM}/stop`, { method: "POST" });
  $("#talkForm").onsubmit = talk;
  $("#saveCfg").onclick = saveConfig;
  $("#nbSubject").onchange = loadNotebook;
  $("#deleteRoom").onclick = deleteRoom;
  await loadRooms();
  setInterval(() => ROOM && refresh(), 3000);
}

// Subject picker: 100 subjects grouped by category, with search and select-all per group.
function subjectChips(prefix, selected) {
  const groups = META.categories.map((cat) => {
    const subs = META.subjects.filter((s) => s.category === cat);
    const n = subs.filter((s) => selected.includes(s.key)).length;
    const chips = subs.map((s) => `<label data-name="${esc(s.name.toLowerCase())}" title="${esc(s.syllabus.join(" → "))}"><input type="checkbox" name="${prefix}_subj" value="${s.key}" ${selected.includes(s.key) ? "checked" : ""}> ${esc(s.name)}</label>`).join("");
    return `<details ${n ? "open" : ""}><summary><span>${esc(cat)} <span class="muted">(${subs.length})</span></span><a data-all="${esc(cat)}">select all</a></summary><div class="chips">${chips}</div></details>`;
  }).join("");
  return `<div class="bar"><input type="search" placeholder="Search ${META.subjects.length} subjects…" data-filter><span class="muted small" data-count></span></div><div class="groups">${groups}</div>`;
}
function wirePicker(root, prefix) {
  const count = () => (root.querySelector("[data-count]").textContent = `${checked(prefix).length} selected`);
  root.addEventListener("change", count);
  root.querySelector("[data-filter]").addEventListener("input", (e) => {
    const q = e.target.value.trim().toLowerCase();
    root.querySelectorAll("details").forEach((d) => {
      let any = false;
      d.querySelectorAll("label").forEach((l) => { const hit = !q || l.dataset.name.includes(q); l.hidden = !hit; any ||= hit; });
      d.hidden = !any; if (q) d.open = any;
    });
  });
  root.querySelectorAll("[data-all]").forEach((a) => a.addEventListener("click", (e) => {
    e.preventDefault();
    const boxes = [...a.closest("details").querySelectorAll("input")];
    const on = !boxes.every((b) => b.checked);
    boxes.forEach((b) => (b.checked = on));
    a.closest("details").open = true; count();
  }));
  count();
}
function renderPicker(id, prefix, selected) { const el = $(id); el.innerHTML = subjectChips(prefix, selected); wirePicker(el, prefix); }
const checked = (prefix) => [...document.querySelectorAll(`input[name=${prefix}_subj]:checked`)].map((i) => i.value);

async function loadRooms(selectId) {
  const rooms = await api("/api/classrooms");
  $("#roomSelect").innerHTML = rooms.map((r) => `<option value="${r.id}">${esc(r.name)} (class ${Math.min(r.grade, 12)})</option>`).join("");
  $("#empty").hidden = rooms.length > 0;
  $("#room").hidden = rooms.length === 0;
  $("#roomSelect").hidden = rooms.length === 0;
  const id = selectId || (rooms[0] && rooms[0].id);
  if (id) { $("#roomSelect").value = id; openRoom(id); }
}

async function createRoom(ev) {
  ev.preventDefault();
  const f = ev.target, btn = $("#createBtn");
  const subjects = checked("new");
  if (!subjects.length) { $("#newError").textContent = "Pick at least one subject."; return; }
  const config = {
    student_name: f.student_name.value,
    student: { backend: f.s_backend.value, model: f.s_model.value, base_url: f.s_base_url.value || null },
    teacher: { backend: f.t_backend.value, model: f.t_model.value, base_url: f.t_base_url.value || null, api_key: f.t_api_key.value || null },
    student_params_b: f.s_params.value ? Number(f.s_params.value) : null,
    persona: f.persona.value, custom_style: f.custom_style.value || null, subjects,
    questions_per_exam: Number(f.questions_per_exam.value), pass_mark: Number(f.pass_mark.value) / 100,
    max_attempts: Number(f.max_attempts.value),
  };
  btn.disabled = true; $("#newError").textContent = "Checking the student model…";
  try {
    const { id } = await api("/api/classrooms", { method: "POST", body: { name: f.name.value, config } });
    $("#newDialog").close();
    await loadRooms(id);
  } catch (e) { $("#newError").textContent = e.message; }
  finally { btn.disabled = false; }
}

// --- classroom view ---------------------------------------------------------
async function openRoom(id) {
  ROOM = id; lastId = 0;
  $("#transcript").innerHTML = "";
  if (source) source.close();
  try { await refresh(true); } catch (e) { $("#transcript").innerHTML = `<div class="msg system error">${esc(e.message)}</div>`; return; }
  source = new EventSource(`/api/classrooms/${id}/events?after=0`);
  source.onmessage = (e) => addMessage(JSON.parse(e.data));
}

async function refresh(full) {
  const r = await api(`/api/classrooms/${ROOM}`);
  $("#ladder").innerHTML = Array.from({ length: META.grades }, (_, i) => {
    const g = i + 1, cls = g < r.grade ? "done" : g === r.grade ? "now" : "";
    return `<div class="${cls}" title="${META.difficulty[i]}">${g}</div>`;
  }).join("");
  $("#status").textContent = r.graduated ? "🎓 Graduated" : r.busy ? `● ${r.status}` : "Idle";
  document.querySelectorAll("[data-run]").forEach((b) => (b.disabled = r.busy || r.graduated));
  $("#stopBtn").disabled = !r.busy;
  $("#studentInfo").innerHTML = `<b>${esc(r.config.student_name)}</b> · ${esc(r.config.student.model)} (~${r.student_size_b}B)<br>
    <span class="muted small">Teacher: ${esc(r.teacher.name)} · ${esc(r.config.teacher.model)}</span><br>
    <span class="muted small">Class ${Math.min(r.grade, 12)} of 12 — ${META.difficulty[Math.min(r.grade, 12) - 1]}</span>`;
  $("#passMark").textContent = `(pass ≥ ${Math.round(r.pass_mark * 100)}% in every subject)`;
  renderReport(r);
  const nb = r.notebook;
  $("#notebookStats").textContent = `${nb.lesson} lesson notes · ${nb.qa} Q&A · ${nb.mistake} mistake logs · ${nb.summary} cheat-sheets`;
  if (full) {
    $("#cfgPersona").value = r.config.persona;
    $("#cfgCustom").value = r.config.custom_style || "";
    $("#cfgCustom").hidden = r.config.persona !== "custom";
    renderPicker("#cfgSubjects", "cfg", r.config.subjects);
    $("#nbSubject").innerHTML = r.config.subjects.map((s) => `<option value="${s}">${esc(META.subjects.find((x) => x.key === s).name)}</option>`).join("");
    loadNotebook();
  }
}

// One row per subject: classes passed so far and best score in the current class.
function renderReport(r) {
  if (!Object.keys(r.report_card).length) { $("#reportCard").innerHTML = `<span class="muted small">No exams yet.</span>`; return; }
  const cur = String(Math.min(r.grade, 12));
  const name = (k) => META.subjects.find((s) => s.key === k).name;
  let h = `<table><tr><th>Subject</th><th>Classes passed</th><th>Class ${cur}</th></tr>`;
  for (const s of r.subjects) {
    let passed = 0;
    for (const g of Object.keys(r.report_card)) { const c = r.report_card[g][s]; if (c && c.best >= r.pass_mark) passed++; }
    const c = r.graduated ? null : (r.report_card[cur] || {})[s];
    const now = c ? `<td class="${c.best >= r.pass_mark ? "ok" : "bad"}" title="${c.attempts} attempt(s)">${Math.round(c.best * 100)}%</td>` : "<td>–</td>";
    h += `<tr><td>${esc(name(s))}</td><td><div class="bar" title="${passed}/12"><span style="width:${passed / 12 * 100}%"></span></div></td>${now}</tr>`;
  }
  $("#reportCard").innerHTML = h + "</table>";
}

function addMessage(m) {
  if (m.id <= lastId) return;
  lastId = m.id;
  const t = $("#transcript");
  const atBottom = t.scrollHeight - t.scrollTop - t.clientHeight < 80;
  const div = document.createElement("div");
  const kind = (m.meta && m.meta.kind) || "";
  div.className = `msg ${m.role} ${kind}`;
  if (m.role === "system") {
    div.innerHTML = esc(m.content);
  } else {
    let badge = "";
    if (kind === "exam_answer") {
      const ok = m.meta.score >= 1;
      badge = `<span class="badge ${ok ? "ok" : "bad"}">${ok ? "correct" : m.meta.score > 0 ? Math.round(m.meta.score * 100) + "%" : "wrong"}</span>`;
    }
    const long = kind === "lesson" || kind === "review";
    const body = long ? `<details open><summary>${kind === "lesson" ? "Lesson" : "Mistake review"} (${m.content.length} chars)</summary>${md(m.content)}</details>` : md(m.content);
    div.innerHTML = `<div class="who">${esc(m.speaker)}${badge}</div>${body}`;
  }
  t.appendChild(div);
  if (atBottom) t.scrollTop = t.scrollHeight;
}

async function run(action) {
  try { await api(`/api/classrooms/${ROOM}/run`, { method: "POST", body: { action } }); refresh(); }
  catch (e) { alert(e.message); }
}

async function talk(ev) {
  ev.preventDefault();
  const input = $("#talkText"), text = input.value.trim();
  if (!text) return;
  input.value = ""; input.disabled = true;
  try { await api(`/api/classrooms/${ROOM}/talk`, { method: "POST", body: { text, to: $("#talkTo").value } }); }
  catch (e) { alert(e.message); }
  finally { input.disabled = false; input.focus(); }
}

async function saveConfig() {
  const subjects = checked("cfg");
  if (!subjects.length) return alert("Pick at least one subject.");
  try {
    await api(`/api/classrooms/${ROOM}/config`, { method: "PATCH",
      body: { persona: $("#cfgPersona").value, custom_style: $("#cfgCustom").value || null, subjects } });
    refresh(true);
  } catch (e) { alert(e.message); }
}

async function loadNotebook() {
  const s = $("#nbSubject").value;
  if (!s) return;
  const notes = await api(`/api/classrooms/${ROOM}/notebook?subject=${s}`);
  $("#notebook").innerHTML = notes.length ? notes.slice().reverse().map((n) =>
    `<div class="note ${n.kind}"><span class="muted small">class ${n.grade} · ${n.kind}</span>\n${esc(n.content)}</div>`).join("")
    : `<span class="muted small">Empty — the student hasn't written anything here yet.</span>`;
}

async function deleteRoom() {
  if (!confirm("Delete this classroom, its notebook and all exam records?")) return;
  await api(`/api/classrooms/${ROOM}`, { method: "DELETE" });
  if (source) source.close();
  ROOM = null;
  loadRooms();
}

init();
