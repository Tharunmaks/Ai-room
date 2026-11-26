"use strict";
/* Lab: upload data, watch untrained models learn it from scratch, then use the winner. */
Object.assign(ICONS, {
  flask: '<path d="M9 3h6M10 3v6.5L4.6 18.2A2 2 0 0 0 6.3 21h11.4a2 2 0 0 0 1.7-2.8L14 9.5V3"/><path d="M7.5 14h9"/>',
  table: '<rect x="3" y="4" width="18" height="16" rx="2.5"/><path d="M3 10h18M3 15h18M9 4v16"/>',
  doc: '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4M9 12h6M9 16h6"/>',
  brain: '<path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 3 3h1V4z"/><path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-3 3h-1V4z"/>',
});
const LAB = {
  info: null, datasets: [], models: [], jobs: [], sel: null, ds: null, model: null, job: null, timer: null,
  speed: store.get("lab.speed", "fast"), ignore: new Set(), target: null, task: "auto", out: null,
};
try { LAB.sel = JSON.parse(store.get("lab.sel", "null")); } catch (e) { LAB.sel = null; }
const LEARNERS = {
  baseline: { label: "Baseline (no learning)", s: 0 }, oneshot: { label: "One-shot net", s: 1 }, linear: { label: "Linear model", s: 2 },
  boost: { label: "Boosted trees", s: 3 }, mlp: { label: "Neural network", s: 4 },
};
const KIND_LABEL = { number: "number", category: "category", text: "text", date: "date", id: "ignored · ID", constant: "ignored · constant", empty: "ignored · empty" };
const fmtBytes = (b) => (b >= 1e12 ? `${(b / 1e12).toFixed(2)} TB` : b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : b >= 1e3 ? `${Math.round(b / 1e3)} KB` : `${b} B`);
const approx = (n, exact) => `${exact === false ? "≈ " : ""}${(n || 0).toLocaleString()}`;
const budgetOf = (bytes, speed) => { const sp = LAB.info.speeds.find((x) => x.key === speed) || LAB.info.speeds[0]; const perMb = Math.max(1, (bytes / 1e6) * LAB.info.seconds_per_mb); return sp.key === "fast" ? perMb : Math.max(sp.seconds, perMb); };
const fmtSec = (s) => (s == null ? "—" : s < 0.01 ? "<0.01 s" : s < 10 ? `${s.toFixed(2)} s` : s < 120 ? `${s.toFixed(1)} s` : `${Math.round(s / 60)} min`);
const fmtNum = (x) => (x == null || !isFinite(x) ? "—" : Math.abs(x) >= 1000 ? Math.round(x).toLocaleString() : Math.abs(x) >= 10 ? x.toFixed(1) : x.toPrecision(3));
const scoreOf = (task, m) => (!m ? null : task === "classification" ? m.accuracy : m.r2);
const fmtScore = (task, v) => (v == null ? "—" : task === "classification" ? `${(v * 100).toFixed(1)}%` : v.toFixed(3));
const scoreName = (task) => (task === "classification" ? "accuracy" : "R²");
const seriesVar = (key) => (LEARNERS[key]?.s ? `var(--s${LEARNERS[key].s})` : "var(--muted)");
const selSave = () => store.set("lab.sel", JSON.stringify(LAB.sel));

/* ───────────────────────── page ───────────────────────── */
async function renderLab() {
  const el = $("#lab");
  el.innerHTML = `<div class="skeleton" style="height:260px"></div>`;
  try {
    [LAB.info, LAB.datasets, LAB.models, LAB.jobs] = await Promise.all([api("/api/lab/info"), api("/api/lab/datasets"), api("/api/lab/models"), api("/api/lab/jobs")]);
  } catch (e) { el.innerHTML = `<div class="empty"><div class="big">⚠️</div>${esc(e.message)}</div>`; return; }
  el.innerHTML = `
    <div class="page-head"><div><h2>Lab</h2><p>Upload data and brand-new, untrained models learn it from scratch on this computer — usually in seconds. Your data never leaves the machine.</p></div>
      <div class="grow"></div><div class="row wrap gap-s" id="labStats"></div></div>
    <div class="lab">
      <aside class="lab-side">
        <label class="lab-drop" id="labDrop" tabindex="0">
          <input type="file" id="labFile" hidden accept=".csv,.tsv,.tab,.json,.jsonl,.ndjson,.txt,.md,.markdown,.rst,.html,.htm,.pdf,.docx,.epub,.zip,.py,.js,.log">
          <span class="lab-drop-ic">${icon("upload")}</span><b>Drop a file here or click to upload</b>
          <span class="small muted">Tables: CSV, TSV, JSON · Text: TXT, MD, HTML, PDF, DOCX, EPUB · ZIP — up to ${esc(LAB.info.max_label)} per file (${esc(LAB.info.disk_free_label)} free on this disk)</span>
          <span class="lab-up" id="labUp" hidden><span class="bar"><span id="labUpBar" style="width:0%"></span></span><small id="labUpTxt"></small></span>
        </label>
        <div class="lab-samples"><span class="small muted">No data at hand? Try one:</span><div class="row wrap gap-s" id="labSamples"></div></div>
        <div id="labLists"></div>
      </aside>
      <section class="lab-main" id="labMain" aria-live="polite"></section>
    </div>`;
  paintStats();
  $("#labSamples").innerHTML = LAB.info.samples.map((s) => `<button class="chip" data-act="lab-sample" data-arg="${s.key}" title="${esc(s.about)}">${s.icon} ${esc(s.title)}</button>`).join("");
  bindDrop();
  paintLists();
  const running = LAB.jobs.find((j) => j.status === "running" || j.status === "queued");
  if (running) LAB.sel = { kind: "job", id: running.id };
  if (!LAB.sel || !valid(LAB.sel)) LAB.sel = LAB.models[0] ? { kind: "model", id: LAB.models[0].id } : LAB.datasets[0] ? { kind: "dataset", id: LAB.datasets[0].id } : null;
  openSel();
}
function paintStats() {
  const el = $("#labStats"); if (!el) return;
  el.innerHTML = `<span class="chip">${icon("brain")} ${plural(LAB.models.length, "model")}</span><span class="chip">${icon("table")} ${plural(LAB.datasets.length, "dataset")}</span><span class="chip" title="Training uses all CPU cores, no GPU needed">⚙️ ${LAB.info.cpus} CPU cores</span><span class="chip on" title="The fast setting gives the whole job 1 second per megabyte of data (at least 1 second)">⚡ 1 s per MB</span>`;
}
function valid(sel) {
  if (sel.kind === "model") return LAB.models.some((m) => m.id === sel.id);
  if (sel.kind === "dataset") return LAB.datasets.some((d) => d.id === sel.id);
  return LAB.jobs.some((j) => j.id === sel.id);
}

function bindDrop() {
  const drop = $("#labDrop"), file = $("#labFile");
  file.onchange = () => { if (file.files[0]) uploadData(file.files[0]); file.value = ""; };
  drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } };
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) uploadData(f); });
}

async function uploadData(f) {
  if (f.size > LAB.info.max_bytes) { toast(`That file is ${fmtBytes(f.size)} — more than the ${LAB.info.max_label} limit.`, "err"); return; }
  if (f.size > LAB.info.disk_free) { toast(`Not enough disk space: ${fmtBytes(f.size)} needed, ${LAB.info.disk_free_label} free.`, "err"); return; }
  const drop = $("#labDrop"), up = $("#labUp");
  drop.classList.add("busy"); $("b", drop).textContent = `Uploading ${f.name}…`; up.hidden = false;
  const t0 = performance.now();
  try {
    const data = await new Promise((resolve, reject) => {
      const x = new XMLHttpRequest();
      x.open("POST", `/api/lab/datasets?filename=${encodeURIComponent(f.name)}`);
      x.upload.onprogress = (e) => {
        if (!e.lengthComputable) return;
        const secs = (performance.now() - t0) / 1000, rate = e.loaded / Math.max(secs, 1e-3);
        $("#labUpBar").style.width = `${(100 * e.loaded / e.total).toFixed(1)}%`;
        $("#labUpTxt").textContent = `${fmtBytes(e.loaded)} of ${fmtBytes(e.total)} · ${fmtBytes(rate)}/s` + (e.loaded < e.total ? ` · ${fmtSec((e.total - e.loaded) / rate)} left` : " · reading it…");
      };
      x.onload = () => { let d = {}; try { d = JSON.parse(x.responseText); } catch (err) { /* not JSON */ } x.status < 300 ? resolve(d) : reject(new Error(typeof d.detail === "string" ? d.detail : `Upload failed (${x.status})`)); };
      x.onerror = () => reject(new Error("Upload failed: the connection was lost."));
      x.send(f);
    });
    LAB.datasets = await api("/api/lab/datasets");
    LAB.sel = { kind: "dataset", id: data.id }; LAB.ds = data;
    paintLists(); openSel();
    toast(`${f.name}: ${data.kind === "text" ? `${approx(data.chars, data.chars_exact)} characters` : `${approx(data.rows, data.rows_exact)} rows`}`, "ok");
  } catch (e) { toast(e.message, "err"); } finally {
    drop.classList.remove("busy"); $("b", drop).textContent = "Drop a file here or click to upload"; up.hidden = true;
  }
}

function paintLists() {
  paintStats();
  const box = $("#labLists"); if (!box) return;
  const on = (kind, id) => (LAB.sel && LAB.sel.kind === kind && LAB.sel.id === id ? " on" : "");
  const live = LAB.jobs.filter((j) => j.status === "running" || j.status === "queued");
  box.innerHTML = `
    ${live.length ? `<h4 class="lab-h">Learning now</h4>${live.map((j) => `<button class="lab-item${on("job", j.id)}" data-act="lab-open" data-arg="job:${j.id}"><span class="pulse"></span><span class="grow"><b>${esc(j.dataset_name)}</b><small>${j.status === "queued" ? "waiting for the trainer" : "training…"}</small></span></button>`).join("")}` : ""}
    <h4 class="lab-h">Trained models</h4>
    ${LAB.models.map((m) => `<button class="lab-item${on("model", m.id)}" data-act="lab-open" data-arg="model:${m.id}">
        <span class="lab-ic">${m.type === "text" ? "✍️" : m.task === "classification" ? "🏷️" : "📈"}</span>
        <span class="grow"><b>${esc(m.name)}</b><small>${m.type === "text" ? `${m.score.bits_per_char.toFixed(2)} bits/char` : `${scoreName(m.task)} ${fmtScore(m.task, scoreOf(m.task, m.score))}`} · ${fmtSec(m.seconds)}</small></span></button>`).join("") || `<p class="small muted">None yet. Pick a dataset and press <b>Train</b>.</p>`}
    <h4 class="lab-h">Datasets</h4>
    ${LAB.datasets.map((d) => `<button class="lab-item${on("dataset", d.id)}" data-act="lab-open" data-arg="dataset:${d.id}">
        <span class="lab-ic">${d.kind === "text" ? "📄" : "📊"}</span>
        <span class="grow"><b>${esc(d.name)}</b><small>${d.kind === "text" ? `${approx(d.chars, d.chars_exact)} characters` : `${approx(d.rows, d.rows_exact)} rows · ${(d.columns || []).length} columns`} · ${fmtBytes(d.bytes || 0)}</small></span></button>`).join("") || `<p class="small muted">Upload a file or try a sample.</p>`}`;
}

async function openSel() {
  const main = $("#labMain"); if (!main) return;
  clearTimeout(LAB.timer); LAB.timer = null;
  selSave(); paintLists();
  if (!LAB.sel) { main.innerHTML = labWelcome(); return; }
  main.innerHTML = `<div class="skeleton" style="height:320px"></div>`;
  try {
    if (LAB.sel.kind === "dataset") { LAB.ds = await api(`/api/lab/datasets/${LAB.sel.id}`); paintDataset(); }
    else if (LAB.sel.kind === "model") { LAB.model = await api(`/api/lab/models/${LAB.sel.id}`); LAB.out = null; paintModel(); }
    else { LAB.job = { ...(await api(`/api/lab/jobs/${LAB.sel.id}`)), all: [] }; LAB.job.all = LAB.job.events; paintJob(true); pollJob(); }
  } catch (e) { main.innerHTML = `<div class="card empty"><div class="big">⚠️</div>${esc(e.message)}</div>`; }
}

function labWelcome() {
  return `<div class="card lab-welcome">
    <div class="lab-hero-ic">🧪</div>
    <h2>Teach a brand-new AI with your own data</h2>
    <p class="muted">Upload a table and the Lab learns to predict any column from the others. Upload text and it learns to write like it — and to find the right passage when you ask a question.</p>
    <div class="grid g-3" style="margin-top:18px;text-align:left">
      <div class="card flat"><h3>⚡ Starts from zero</h3><p class="small muted">Every model starts with random weights. Nothing is pre-trained; it learns only from what you upload.</p></div>
      <div class="card flat"><h3>🏁 A race of four learners</h3><p class="small muted">A one-shot net, a linear model, boosted trees and a neural network learn side by side. The best one (or a blend) wins.</p></div>
      <div class="card flat"><h3>✅ An honest score</h3><p class="small muted">15% of rows are kept back and never seen while learning. The score you see is how it does on those.</p></div>
    </div></div>`;
}

/* ───────────────────────── dataset ───────────────────────── */
function paintDataset() {
  const d = LAB.ds, main = $("#labMain");
  if (d.kind === "text") {
    main.innerHTML = `<div class="card lab-panel">
      ${panelHead("📄", d.name, `${approx(d.chars, d.chars_exact)} ${d.chars_exact === false ? "bytes of" : "characters ·"} ${d.chars_exact === false ? "text" : plural(d.documents, "document")} · ${fmtBytes(d.bytes)}`, `dataset:${d.id}`)}
      <p class="muted small">A text model counts which character follows every short stretch of text (1 to 10 characters), in one pass. It then writes in the style of your text, and you can search it by asking questions.</p>
      <pre class="lab-pre">${esc(d.preview_text)}${d.chars > 1500 ? "\n…" : ""}</pre>
      <div class="row wrap" style="margin-top:16px"><button class="btn primary big" data-act="lab-train">${icon("play")} Learn this text</button><span class="small muted">Time allowed: <b>${fmtSec(budgetOf(d.bytes, "fast"))}</b> — 1 second per MB. Texts bigger than ${(32).toLocaleString()} million characters are learned from evenly spaced samples.</span></div>
    </div>`;
    return;
  }
  const usable = d.profile.filter((p) => ["number", "category", "text", "date"].includes(p.kind));
  if (!LAB.target || !d.columns.includes(LAB.target) || LAB.dsId !== d.id) { LAB.target = d.target; LAB.ignore = new Set(); LAB.task = "auto"; LAB.dsId = d.id; }
  const tp = d.profile.find((p) => p.name === LAB.target);
  main.innerHTML = `<div class="card lab-panel">
    ${panelHead("📊", d.name, `${approx(d.rows, d.rows_exact)} rows · ${d.columns.length} columns · ${fmtBytes(d.bytes)}`, `dataset:${d.id}`)}
    <div class="grid g-2 lab-setup">
      <label class="field"><span>What should it learn to predict?</span><select id="labTarget">${usable.filter((p) => p.kind !== "text").map((p) => `<option value="${esc(p.name)}"${p.name === LAB.target ? " selected" : ""}>${esc(p.name)}</option>`).join("")}</select></label>
      <div class="field"><span>The answer is…</span><div class="seg" id="labTask">${[["auto", "Auto"], ["classification", "A category"], ["regression", "A number"]].map(([k, l]) => `<button type="button" class="${LAB.task === k ? "on" : ""}" data-act="lab-task" data-arg="${k}">${l}</button>`).join("")}</div>
        <small class="muted" id="labTaskHelp">${taskHelp(tp)}</small></div>
    </div>
    <h4 class="lab-h" style="margin-top:18px">Columns it may learn from <span class="muted small" style="font-weight:500">— click to leave one out</span></h4>
    <div class="row wrap gap-s" id="labCols">${colChips(d)}</div>
    <div class="lab-table-wrap" tabindex="0" aria-label="First rows of the data"><table class="lab-table"><thead><tr>${d.columns.map((c) => `<th class="${c === LAB.target ? "tgt" : ""}">${esc(c)}</th>`).join("")}</tr></thead>
      <tbody>${d.preview.slice(0, 8).map((r) => `<tr>${r.map((v, i) => `<td class="${d.columns[i] === LAB.target ? "tgt" : ""}">${esc(v.length > 60 ? v.slice(0, 58) + "…" : v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
    <div class="lab-go">
      <div class="field"><span>How long may it learn? <b id="labBudget">${fmtSec(budgetOf(d.bytes, LAB.speed))}</b> for this file</span><div class="seg" id="labSpeed">${LAB.info.speeds.map((sp) => `<button type="button" class="${LAB.speed === sp.key ? "on" : ""}" data-act="lab-speed" data-arg="${sp.key}">${{ fast: "⚡ Fast", balanced: "⚖️ Balanced", thorough: "🔬 Thorough" }[sp.key]} <small>${esc(sp.label)}</small></button>`).join("")}</div></div>
      <button class="btn primary big" data-act="lab-train">${icon("play")} Train from scratch</button>
    </div>
    <p class="small muted" style="margin:12px 0 0">It splits the rows first: 70% to learn from, 15% to check progress, and 15% locked away for the final, honest test. Fast gives the whole job — reading, learning, testing, saving — 1 second per MB (at least 1 s).${d.rows < 200 ? " With so few rows, scores will jump around — more data helps more than anything." : ""}${d.rows_exact === false ? " If the rows do not fit in memory, it learns from a random sample of them." : ""}</p>
  </div>`;
  $("#labTarget").onchange = (e) => { LAB.target = e.target.value; LAB.ignore.delete(LAB.target); paintDataset(); };
}
function taskHelp(tp) {
  if (!tp) return "";
  const auto = tp.kind === "number" && !(tp.integers && tp.unique <= 20) ? "a number" : "a category";
  return LAB.task === "auto" ? `Auto: looks like ${auto} (${(tp.unique || 0).toLocaleString()} different values).` : "";
}
function colChips(d) {
  return d.profile.map((p) => {
    if (p.name === LAB.target) return `<span class="chip on" title="This is what it learns to predict">🎯 ${esc(p.name)}</span>`;
    const usable = ["number", "category", "text", "date"].includes(p.kind);
    if (!usable) return `<span class="chip lab-col off" title="${esc(p.why || "")}">${esc(p.name)} <small>${KIND_LABEL[p.kind] || p.kind}</small></span>`;
    const out = LAB.ignore.has(p.name);
    return `<button class="chip lab-col${out ? " off" : ""}" data-act="lab-col" data-arg="${esc(p.name)}" aria-pressed="${!out}" title="${out ? "Left out — click to use it" : "Click to leave it out"}">${esc(p.name)} <small>${p.kind}${p.missing > 0.005 ? ` · ${Math.round(p.missing * 100)}% blank` : ""}</small></button>`;
  }).join("");
}
function panelHead(ic, title, sub, del) {
  return `<div class="lab-head"><span class="lab-big-ic">${ic}</span><div class="grow"><h3>${esc(title)}</h3><div class="small muted">${sub}</div></div>
    ${del ? `<button class="btn sm ghost danger" data-act="lab-del" data-arg="${del}" aria-label="Delete">${icon("trash")}</button>` : ""}</div>`;
}

/* ───────────────────────── training (live) ───────────────────────── */
async function pollJob() {
  const j = LAB.job; if (!j || LAB.sel?.kind !== "job" || LAB.sel.id !== j.id) return;
  try {
    const r = await api(`/api/lab/jobs/${j.id}?after=${j.next}`);
    Object.assign(j, r, { all: j.all.concat(r.events) });
  } catch (e) { toast(e.message, "err"); return; }
  if (!$("#labMain")) return;
  paintJob(false);
  if (j.status === "queued" || j.status === "running") { LAB.timer = setTimeout(pollJob, 350); return; }
  LAB.jobs = await api("/api/lab/jobs"); LAB.models = await api("/api/lab/models");
  if (j.status === "done" && j.model) {
    const m = LAB.models.find((x) => x.id === j.model);
    toast(`Learned in ${fmtSec(m?.seconds)}${m && m.type === "table" ? ` — ${scoreName(m.task)} ${fmtScore(m.task, scoreOf(m.task, m.score))} on rows it never saw` : ""}`, "ok");
    confetti(0.5);
    LAB.sel = { kind: "model", id: j.model }; openSel();
  } else if (j.status === "error") { paintJob(false); paintLists(); }
  else { paintLists(); }
}

function raceState(events) {
  const L = {}, stages = [];
  let order = [];
  for (const e of events) {
    if (e.type === "stage") stages.push(e.text);
    else if (e.type === "plan") { for (const l of e.learners) { L[l.learner] = { key: l.learner, label: l.label, blurb: l.blurb, status: "waiting", pts: [], val: null, frac: 0, budget: 1 }; } order = e.learners.map((l) => l.learner); }
    else if (e.type === "learner_start") { L[e.learner] = { key: e.learner, label: e.label, blurb: e.blurb, budget: e.seconds, start: e.at, status: "learning", pts: [], val: null, frac: 0 }; if (!order.includes(e.learner)) order.push(e.learner); }
    else if (e.type === "progress" && L[e.learner]) { const l = L[e.learner]; if (e.val != null) { l.val = e.val; l.pts.push([e.at, e.val]); } l.frac = Math.max(l.frac, Math.min(1, (e.t || 0) / Math.max(l.budget, 0.1)), e.total ? e.step / e.total : 0); l.note = e.note; l.t = e.t; }
    else if (e.type === "learner_done" && L[e.learner]) { const l = L[e.learner]; l.status = "done"; l.frac = 1; l.valid = e.valid; l.test = e.test; l.seconds = e.seconds; l.val = scoreOf(e.valid.accuracy != null ? "classification" : "regression", e.valid); l.pts.push([e.at, l.val]); }
    else if (e.type === "learner_failed" && L[e.learner]) { L[e.learner].status = "failed"; L[e.learner].error = e.error; }
    else if (e.type === "learner_skipped" && L[e.learner]) { L[e.learner].status = "failed"; L[e.learner].error = "skipped: the time was used up"; }
  }
  return { L, order, stages };
}

function paintJob(full) {
  const j = LAB.job, main = $("#labMain");
  const { L, order, stages } = raceState(j.all);
  const live = j.status === "running" || j.status === "queued";
  const elapsed = j.all.length ? j.all[j.all.length - 1].at : 0;
  const task = j.all.find((e) => e.task)?.task || "classification";
  if (full || !$("#labJob")) {
    main.innerHTML = `<div class="card lab-panel" id="labJob">
      <div class="lab-head"><span class="lab-big-ic">${j.kind === "text" ? "✍️" : "🏁"}</span><div class="grow"><h3 id="labJobTitle"></h3><div class="small muted" id="labJobSub"></div></div>
        <button class="btn sm" data-act="lab-stop" id="labStop">${icon("stop")} Stop</button></div>
      <ol class="lab-stages" id="labStages"></ol>
      <div id="labRace"></div><div id="labChart"></div><div id="labJobErr"></div></div>`;
  }
  $("#labJobTitle").innerHTML = `${live ? '<span class="pulse"></span> ' : ""}${j.status === "queued" ? "Waiting for the trainer…" : live ? `Learning ${esc(j.dataset_name)}…` : j.status === "done" ? `Learned ${esc(j.dataset_name)}` : j.status === "stopped" ? "Stopped" : "Something went wrong"}`;
  $("#labJobSub").textContent = `${fmtSec(elapsed)}${j.budget ? ` of ${fmtSec(j.budget)} allowed (${j.speed === "fast" ? "1 s per MB" : j.speed})` : ""}`;
  $("#labStop").hidden = !live;
  $("#labStages").innerHTML = stages.slice(-4).map((s, i, a) => `<li class="${i === a.length - 1 && live ? "now" : ""}">${esc(s)}</li>`).join("");
  if (j.kind !== "text" && order.length) {
    const base = L.baseline;
    $("#labRace").innerHTML = `<div class="lab-race">${order.filter((k) => k !== "baseline").map((k) => { const l = L[k];
      return `<div class="lab-lane ${l.status}"><span class="lab-dot" style="background:${seriesVar(k)}"></span><span class="lab-lane-name"><b>${esc(l.label)}</b><small>${l.status === "failed" ? esc(l.error || "failed") : l.status === "done" ? `done in ${fmtSec(l.seconds)}` : l.status === "waiting" ? "waiting its turn" : esc(l.note || "starting…")}</small></span>
        <span class="bar grow"><span style="width:${Math.round(100 * l.frac)}%;background:${seriesVar(k)}"></span></span>
        <span class="lab-lane-val">${l.val == null ? "—" : fmtScore(task, l.val)}</span></div>`; }).join("")}
      </div>${base?.valid ? `<p class="small muted" style="margin:8px 0 0">Without learning anything (always guessing the ${base.valid.accuracy != null ? "most common answer" : "average"}): ${fmtScore(task, scoreOf(task, base.valid))}. Scores are on the rows it uses to check progress.</p>` : ""}`;
    const series = order.filter((k) => k !== "baseline" && L[k].pts.length).map((k) => ({ key: k, name: L[k].label, color: seriesVar(k), pts: L[k].pts }));
    if (series.length) lineChart($("#labChart"), series, { task, baseline: base?.valid ? scoreOf(task, base.valid) : null, title: `Validation ${scoreName(task)} while learning` });
  }
  $("#labJobErr").innerHTML = j.error ? `<div class="lab-err">${icon("x")} ${esc(j.error)}</div>` : "";
}

/* ───────────────────────── model ───────────────────────── */
function paintModel() {
  const m = LAB.model;
  if (m.type === "text") return paintTextModel();
  const r = m.report, main = $("#labMain"), task = r.task;
  const test = scoreOf(task, r.test), base = scoreOf(task, r.baseline_test);
  const classes = r.classes || [];
  main.innerHTML = `<div class="stack" style="gap:16px">
    <div class="card lab-panel">
      ${panelHead(task === "classification" ? "🏷️" : "📈", m.name, `Predicts <b>${esc(r.target)}</b> (${task === "classification" ? `one of ${classes.length} categories` : "a number"}) · winner: ${esc(r.winner_label)}`, `model:${m.id}`)}
      <div class="lab-tiles">
        ${tile(task === "classification" ? "Accuracy on unseen rows" : "R² on unseen rows", fmtScore(task, test), task === "classification" ? `${Math.round(test * r.split.test)} of ${r.split.test} rows right` : `typical error ±${fmtNum(r.test.mae)}`, true)}
        ${tile("Without learning", fmtScore(task, base), task === "classification" ? "always the most common answer" : "always the average")}
        ${tile("Learned in", fmtSec(m.seconds), m.budget ? `of ${fmtSec(m.budget)} allowed · ${r.split.train.toLocaleString()} rows` : `from ${r.split.train.toLocaleString()} rows`)}
        ${tile("First model ready", fmtSec(r.first_model_seconds), "the one-shot net needs no epochs")}
      </div>
      ${r.sampled ? `<p class="small muted" style="margin:10px 0 0">The file has ${approx(r.file_rows, r.rows_exact)} rows — more than fit in memory — so it learned from a random sample of ${r.rows.toLocaleString()}.</p>` : ""}
      ${r.ignored?.length ? `<p class="small muted" style="margin:10px 0 0">Not used: ${r.ignored.map((i) => `<b>${esc(i.name)}</b> (${esc(i.why)})`).join(", ")}.</p>` : ""}
    </div>
    <div class="grid g-2 lab-grid">
      <div class="card"><h3>${icon("sparkles")} Try it</h3><form data-form="lab-predict" id="labForm" class="lab-form">${m.inputs.map(inputField).join("")}
        <div class="row" style="grid-column:1/-1"><button class="btn primary" type="submit">${icon("play")} Predict ${esc(r.target)}</button></div></form>
        <div id="labOut" class="lab-out" aria-live="polite"></div></div>
      <div class="card"><h3>🏁 Which learner won</h3><div id="labBoard"></div>
        <p class="small muted" style="margin:10px 0 0">Every learner started untrained. The winner is picked on the progress-check rows; the bars show the locked-away test rows.</p></div>
    </div>
    ${m.curve && Object.keys(m.curve).length ? `<div class="card"><h3>📉 How fast each one learned</h3><div id="labCurve"></div></div>` : ""}
    <div class="grid g-2 lab-grid">
      <div class="card"><h3>🔎 What it pays attention to</h3><div id="labImp"></div><p class="small muted" style="margin:10px 0 0">How much worse it gets when a column's values are shuffled between rows.</p></div>
      ${task === "classification" && classes.length <= 12 && r.test.confusion ? `<div class="card"><h3>🧩 Where it gets confused</h3><div id="labConf"></div></div>` : `<div class="card"><h3>📏 Errors</h3>${errorsHTML(r)}</div>`}
    </div>
    <div class="grid g-2 lab-grid">
      <div class="card"><h3>${icon("table")} Predict a whole file</h3><p class="small muted">Upload a CSV with the same columns (the ${esc(r.target)} column may be missing). You get the same file back with a prediction on every row.</p>
        <label class="btn"><input type="file" id="labBatch" hidden accept=".csv,.tsv,.json,.jsonl,.txt">${icon("upload")} Choose a file…</label></div>
      <div class="card"><h3>${icon("link")} Use it from your own code</h3><pre class="lab-pre small">${esc(curlExample(m))}</pre></div>
    </div></div>`;
  barBoard($("#labBoard"), r.leaderboard.filter((b) => b.learner !== "baseline"), task, r.winner);
  LAB.redraw = () => $("#labCurve") && lineChart($("#labCurve"), Object.entries(m.curve).filter(([k]) => k !== "baseline").map(([k, pts]) => ({ key: k, name: LEARNERS[k]?.label || k, color: seriesVar(k), pts })),
    { task, baseline: scoreOf(task, r.leaderboard.find((b) => b.learner === "baseline")?.valid), title: `Validation ${scoreName(task)} over time` });
  LAB.redraw();
  importanceBars($("#labImp"), r.importance);
  if ($("#labConf")) confusion($("#labConf"), r.test.confusion, classes);
  $("#labBatch").onchange = (e) => batchPredict(e.target.files[0]);
}
function tile(label, value, sub, hero = false) {
  return `<div class="lab-tile${hero ? " hero" : ""}"><span class="lbl">${esc(label)}</span><b>${esc(value)}</b><small>${esc(sub)}</small></div>`;
}
function inputField(it) {
  const id = `in_${btoa(unescape(encodeURIComponent(it.name))).replace(/[^A-Za-z0-9]/g, "")}`;
  const lab = `<span>${esc(it.name)}</span>`;
  if (it.kind === "category") return `<label class="field" for="${id}">${lab}<select id="${id}" name="${esc(it.name)}">${it.choices.map((c) => `<option${c === it.default ? " selected" : ""}>${esc(c)}</option>`).join("")}</select></label>`;
  if (it.kind === "number") {
    const v = it.default == null ? "" : it.integers ? Math.round(it.default) : +it.default.toPrecision(4);
    return `<label class="field" for="${id}">${lab}<input type="number" step="any" id="${id}" name="${esc(it.name)}" value="${v}" placeholder="${it.min != null ? `${fmtNum(it.min)} – ${fmtNum(it.max)}` : ""}"></label>`;
  }
  if (it.kind === "text") return `<label class="field wide" for="${id}">${lab}<textarea id="${id}" name="${esc(it.name)}" rows="3">${esc(it.default || "")}</textarea></label>`;
  return `<label class="field" for="${id}">${lab}<input type="text" id="${id}" name="${esc(it.name)}" value="${esc(it.default || "")}"></label>`;
}
FORMS["lab-predict"] = async (form) => {
  const row = {};
  for (const el of form.elements) if (el.name) row[el.name] = el.value;
  const out = $("#labOut");
  try {
    const r = await api(`/api/lab/models/${LAB.model.id}/predict`, { method: "POST", body: { rows: [row] } });
    const p = r.predictions[0], task = LAB.model.report.task;
    if (task === "classification") {
      out.innerHTML = `<div class="lab-answer"><span class="muted small">Prediction</span><b>${esc(p.prediction)}</b><small>${Math.round(p.confidence * 100)}% sure · ${fmtSec(r.seconds)}</small></div>
        <div class="lab-probs">${p.top.map((t, i) => `<div class="lab-prob"><span class="nm">${esc(t.label)}</span><span class="track"><span style="width:${Math.max(1, t.p * 100)}%;${i ? "opacity:.55" : ""}"></span></span><span class="v">${(t.p * 100).toFixed(1)}%</span></div>`).join("")}</div>`;
    } else {
      out.innerHTML = `<div class="lab-answer"><span class="muted small">Prediction</span><b>${esc(fmtNum(p.prediction))}</b><small>usually within ±${esc(fmtNum(LAB.model.report.test.mae))} · ${fmtSec(r.seconds)}</small></div>`;
    }
  } catch (e) { out.innerHTML = `<div class="lab-err">${esc(e.message)}</div>`; }
};
async function batchPredict(f) {
  if (!f) return;
  try {
    const r = await fetch(`/api/lab/models/${LAB.model.id}/predict_file?filename=${encodeURIComponent(f.name)}`, { method: "POST", body: f });
    if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.detail || "Prediction failed"); }
    const blob = await r.blob(), a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = f.name.replace(/\.[^.]+$/, "") + "_predicted.csv"; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    toast("Predictions downloaded", "ok");
  } catch (e) { toast(e.message, "err"); }
}
function curlExample(m) {
  const row = {};
  for (const it of m.inputs.slice(0, 4)) row[it.name] = it.kind === "number" ? (it.integers ? Math.round(it.default) : +(+it.default).toPrecision(4)) : it.default;
  return `curl -s ${location.origin}/api/lab/models/${m.id}/predict \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify({ rows: [row] }).replace(/'/g, "'\\''")}'`;
}
function errorsHTML(r) {
  return `<div class="lab-tiles small-tiles">${tile("Typical error (MAE)", fmtNum(r.test.mae), "on unseen rows")}${tile("RMSE", fmtNum(r.test.rmse), "big misses count more")}${tile("R²", r.test.r2.toFixed(3), "1 = perfect, 0 = no better than the average")}</div>`;
}

/* ── text model ── */
function paintTextModel() {
  const m = LAB.model, r = m.report, main = $("#labMain");
  main.innerHTML = `<div class="stack" style="gap:16px">
    <div class="card lab-panel">
      ${panelHead("✍️", m.name, `Learned ${r.chars.toLocaleString()} characters from ${plural(r.documents, "document")} · ${r.vocab} different characters`, `model:${m.id}`)}
      <div class="lab-tiles">
        ${tile("Bits per character", r.valid_bits_per_char.toFixed(2), "on the last tenth, held back", true)}
        ${tile("Random guessing", r.random_bits_per_char.toFixed(2), "bits per character")}
        ${tile("Learned in", fmtSec(m.seconds || r.seconds), m.budget ? `of ${fmtSec(m.budget)} allowed · ${(r.learned_chars || r.chars).toLocaleString()} characters` : `${Math.round(r.chars_per_second).toLocaleString()} characters a second`)}
        ${tile("Looks back", `${r.order}`, "characters — the context that predicted best")}
      </div>
      <p class="small muted" style="margin:10px 0 0">Lower bits per character means better prediction: letter frequencies alone need ${r.letter_frequency_bits_per_char.toFixed(2)}. This model copies style, spelling and layout; it does not understand or reason, and knows nothing beyond your text.</p>
    </div>
    <div class="grid g-2 lab-grid">
      <div class="card"><h3>${icon("sparkles")} Write like your text</h3>
        <label class="field"><span>Start with (optional)</span><textarea id="labPrompt" rows="3" placeholder="The first words…"></textarea></label>
        <div class="grid g-2" style="margin-top:10px"><label class="field"><span>Length: <b id="labLenV">400</b> characters</span><input type="range" id="labLen" min="50" max="2000" step="50" value="400"></label>
          <label class="field"><span>Creativity: <b id="labTempV">0.8</b></span><input type="range" id="labTemp" min="0.1" max="1.5" step="0.05" value="0.8"></label></div>
        <div class="row" style="margin-top:12px"><button class="btn primary" data-act="lab-write">${icon("play")} Write</button><span class="small muted">Low creativity repeats the most likely text; high creativity takes risks.</span></div>
        <pre class="lab-pre lab-gen" id="labGen" hidden></pre></div>
      <div class="card"><h3>${icon("search")} Ask your text</h3>
        <form data-form="lab-ask" class="row"><label class="sr" for="labQ">Question</label><input type="search" id="labQ" placeholder="Ask a question or type key words…"><button class="btn" type="submit">${icon("search")}</button></form>
        <p class="small muted" style="margin:8px 0 0">Finds the passages of your text that best match (BM25 search) — the answer is quoted, not made up.</p>
        <div id="labHits" class="stack" style="margin-top:12px"></div></div>
    </div>
    <div class="card"><h3>📉 How much context helps</h3><div id="labOrder"></div></div></div>`;
  $("#labLen").oninput = (e) => ($("#labLenV").textContent = e.target.value);
  $("#labTemp").oninput = (e) => ($("#labTempV").textContent = (+e.target.value).toFixed(2));
  LAB.redraw = () => $("#labOrder") && orderChart($("#labOrder"), r.bits_by_order, r.order, r.random_bits_per_char);
  LAB.redraw();
}
FORMS["lab-ask"] = async () => {
  const q = $("#labQ").value.trim(); if (!q) return;
  const box = $("#labHits"); box.innerHTML = `<div class="skeleton" style="height:80px"></div>`;
  try {
    const r = await api(`/api/lab/models/${LAB.model.id}/predict`, { method: "POST", body: { query: q } });
    const words = q.split(/\W+/).filter((w) => w.length > 2);
    box.innerHTML = r.passages.map((p) => `<div class="lab-hit"><div class="small muted">${esc(p.title)} · match ${p.score.toFixed(1)}</div><div>${markWords(p.text.length > 900 ? p.text.slice(0, 900) + "…" : p.text, words)}</div></div>`).join("") || `<p class="muted small">Nothing in the text matches those words.</p>`;
  } catch (e) { box.innerHTML = `<div class="lab-err">${esc(e.message)}</div>`; }
};

const markWords = (text, words) => {   /* whole words only, so "are" does not light up inside "share" */
  if (!words.length) return esc(text);
  const re = new RegExp(`\\b(${words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\b`, "gi");
  return esc(text).replace(re, "<mark>$1</mark>");
};

/* ───────────────────────── charts (inline SVG) ───────────────────────── */
function chartTip(wrap) {
  let tip = $(".lab-tip", wrap);
  if (!tip) { tip = document.createElement("div"); tip.className = "lab-tip"; tip.hidden = true; wrap.appendChild(tip); }
  return tip;
}
function placeTip(wrap, tip, x, y) {
  const w = wrap.clientWidth;
  tip.hidden = false;
  const tw = tip.offsetWidth;
  tip.style.left = `${Math.min(Math.max(4, x + 14), w - tw - 4)}px`; tip.style.top = `${Math.max(0, y - 10)}px`;
}

/* Validation score over time, one line per learner, a dashed reference for "no learning". */
function lineChart(el, series, { task, baseline, title }) {
  const W = Math.max(280, el.clientWidth || 600), H = 230, m = { l: 46, r: 118, t: 14, b: 30 };
  if (W < 460) m.r = 12;
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const xs = series.flatMap((s) => s.pts.map((p) => p[0])), ys = series.flatMap((s) => s.pts.map((p) => p[1]));
  const xMax = Math.max(0.5, ...xs) * 1.02;
  let yMin = task === "classification" ? 0 : Math.min(0, ...ys, baseline ?? 0), yMax = 1;
  if (task === "classification") { const lo = Math.min(...ys, baseline ?? 1); yMin = Math.max(0, Math.floor((lo - 0.05) * 10) / 10); }
  yMin = Math.max(yMin, -1);
  const X = (v) => m.l + (v / xMax) * pw, Y = (v) => m.t + (1 - (Math.max(yMin, Math.min(yMax, v)) - yMin) / (yMax - yMin)) * ph;
  const ticks = []; const step = (yMax - yMin) > 0.5 ? 0.2 : 0.1;
  for (let v = Math.ceil(yMin / step) * step; v <= yMax + 1e-9; v += step) ticks.push(+v.toFixed(2));
  const fmtY = (v) => (task === "classification" ? `${Math.round(v * 100)}%` : v.toFixed(1));
  const fmtV = (v) => (task === "classification" ? `${(v * 100).toFixed(1)}%` : v.toFixed(3));
  const xt = niceTicks(xMax, 5);
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="${esc(title)}">`;
  svg += ticks.map((v) => `<line class="grid" x1="${m.l}" x2="${m.l + pw}" y1="${Y(v)}" y2="${Y(v)}"/><text class="ax" x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">${fmtY(v)}</text>`).join("");
  svg += xt.map((v) => `<text class="ax" x="${X(v)}" y="${H - 8}" text-anchor="middle">${v}s</text>`).join("");
  if (baseline != null && baseline > yMin) svg += `<line class="ref" x1="${m.l}" x2="${m.l + pw}" y1="${Y(baseline)}" y2="${Y(baseline)}"/><text class="ax" x="${m.l + 4}" y="${Y(baseline) - 5}">no learning ${fmtV(baseline)}</text>`;
  const ends = [];
  for (const s of series) {
    const pts = s.pts.slice().sort((a, b) => a[0] - b[0]);
    const d = pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    svg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    const last = pts[pts.length - 1];
    svg += `<circle cx="${X(last[0])}" cy="${Y(last[1])}" r="4" fill="${s.color}" class="ring"/>`;
    ends.push({ y: Y(last[1]), x: X(last[0]), name: s.name, v: last[1], color: s.color });
  }
  if (m.r > 20) {                      // direct labels at the right edge, nudged apart so they never overlap
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 26) ends[i].y = ends[i - 1].y + 26;
    for (const e of ends) svg += `<text class="lbl" x="${m.l + pw + 10}" y="${e.y}"><tspan>${esc(e.name)}</tspan><tspan x="${m.l + pw + 10}" dy="12" class="ax">${fmtV(e.v)}</tspan></text>`;
  }
  svg += `<line class="cross" id="xh" x1="0" x2="0" y1="${m.t}" y2="${m.t + ph}" visibility="hidden"/><rect class="hit" x="${m.l}" y="${m.t}" width="${pw}" height="${ph}"/></svg>`;
  el.innerHTML = `<div class="lab-chart"><div class="lab-legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("")}${baseline != null ? `<span><i class="dash"></i>no learning</span>` : ""}</div>${svg}</div>`;
  const wrap = $(".lab-chart", el), hit = $(".hit", el), xh = $("#xh", el), tip = chartTip(wrap);
  hit.addEventListener("pointermove", (ev) => {
    const box = hit.getBoundingClientRect(), t = ((ev.clientX - box.left) / box.width) * xMax;
    xh.setAttribute("x1", X(t)); xh.setAttribute("x2", X(t)); xh.setAttribute("visibility", "visible");
    const rows = series.map((s) => { const before = s.pts.filter((p) => p[0] <= t); return before.length ? { s, v: before[before.length - 1][1] } : null; }).filter(Boolean);
    tip.innerHTML = `<b>${t.toFixed(1)} s</b>${rows.map((r) => `<div><i style="background:${r.s.color}"></i>${esc(r.s.name)} <b>${fmtV(r.v)}</b></div>`).join("") || "<div>nothing learned yet</div>"}`;
    const wb = wrap.getBoundingClientRect(); placeTip(wrap, tip, ev.clientX - wb.left, ev.clientY - wb.top);
  });
  hit.addEventListener("pointerleave", () => { tip.hidden = true; xh.setAttribute("visibility", "hidden"); });
}
function niceTicks(max, n) {
  const raw = max / n, mag = 10 ** Math.floor(Math.log10(raw)), step = [1, 2, 5, 10].map((k) => k * mag).find((s) => s >= raw) || raw;
  const out = []; for (let v = 0; v <= max + 1e-9; v += step) out.push(+v.toFixed(3)); return out;
}

/* Test score per learner: one bar each, in the learner's own colour; the winner is named. */
function barBoard(el, rows, task, winners) {
  const vals = rows.map((r) => scoreOf(task, r.test));
  const lo = task === "classification" ? 0 : Math.min(0, ...vals);
  el.innerHTML = `<div class="lab-bars">${rows.map((r, i) => {
    const v = vals[i], w = Math.max(0.5, ((v - lo) / (1 - lo)) * 100), win = winners.includes(r.learner);
    return `<div class="lab-barrow" tabindex="0" data-tip="${esc(`${r.label}: test ${fmtScore(task, v)} · progress check ${fmtScore(task, scoreOf(task, r.valid))} · ${fmtSec(r.seconds)}`)}">
      <span class="nm">${esc(r.label)}${win ? ' <span class="chip good tiny-chip">winner</span>' : ""}<small>${fmtSec(r.seconds)}</small></span>
      <span class="track"><span style="width:${w}%;background:${seriesVar(r.learner)}"></span></span><span class="v">${fmtScore(task, v)}</span></div>`; }).join("")}</div>
    ${winners.length > 1 ? `<p class="small muted" style="margin:8px 0 0">The winner is a blend: it averages ${winners.map((k) => LEARNERS[k]?.label || k).join(", ")}.</p>` : ""}`;
  hoverTips(el);
}
function importanceBars(el, imp) {
  if (!imp?.length) { el.innerHTML = `<p class="muted small">Not enough rows to measure this.</p>`; return; }
  const top = imp.slice(0, 10), mx = Math.max(...top.map((i) => i.share), 1e-9);
  el.innerHTML = `<div class="lab-bars">${top.map((i) => `<div class="lab-barrow" tabindex="0" data-tip="${esc(`${i.column}: ${(i.share * 100).toFixed(1)}% of the total effect`)}"><span class="nm">${esc(i.column)}</span><span class="track"><span style="width:${Math.max(0.5, (i.share / mx) * 100)}%;background:var(--s1)"></span></span><span class="v">${Math.round(i.share * 100)}%</span></div>`).join("")}</div>`;
  hoverTips(el);
}
function hoverTips(el) {
  const wrap = el; wrap.style.position = "relative";
  const tip = chartTip(wrap);
  $$("[data-tip]", el).forEach((row) => {
    const show = (x, y) => { tip.textContent = row.dataset.tip; placeTip(wrap, tip, x, y); };
    row.addEventListener("pointermove", (ev) => { const b = wrap.getBoundingClientRect(); show(ev.clientX - b.left, ev.clientY - b.top); });
    row.addEventListener("pointerleave", () => (tip.hidden = true));
    row.addEventListener("focus", () => { const b = wrap.getBoundingClientRect(), r = row.getBoundingClientRect(); show(r.left - b.left + r.width / 2, r.top - b.top); });
    row.addEventListener("blur", () => (tip.hidden = true));
  });
}
function confusion(el, cm, classes) {
  const mx = Math.max(1, ...cm.flat());
  el.innerHTML = `<div class="lab-table-wrap"><table class="lab-conf"><thead><tr><th class="muted small">true ↓ · predicted →</th>${classes.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
    <tbody>${cm.map((row, i) => `<tr><th>${esc(classes[i])}</th>${row.map((v, j) => `<td class="${i === j ? "diag" : v ? "miss" : ""}" style="--a:${(v / mx).toFixed(3)}" title="${esc(`${v} rows that are ${classes[i]} were predicted as ${classes[j]}`)}">${v || ""}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
    <p class="small muted" style="margin:8px 0 0">Counts on the unseen test rows. The diagonal is right; anything else is a mistake.</p>`;
}
function orderChart(el, bits, chosen, random) {
  const W = Math.max(280, el.clientWidth || 600), H = 210, m = { l: 44, r: 16, t: 16, b: 34 }, pw = W - m.l - m.r, ph = H - m.t - m.b;
  const hi = Math.ceil(Math.max(...bits)), lo = Math.max(0, Math.floor(Math.min(...bits) - 0.5));
  const X = (k) => m.l + ((k - 1) / Math.max(1, bits.length - 1)) * pw, Y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * ph;
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Bits per character for each context length">`;
  for (let v = lo; v <= hi; v++) svg += `<line class="grid" x1="${m.l}" x2="${m.l + pw}" y1="${Y(v)}" y2="${Y(v)}"/><text class="ax" x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`;
  svg += bits.map((_, i) => `<text class="ax" x="${X(i + 1)}" y="${H - 14}" text-anchor="middle">${i + 1}</text>`).join("") + `<text class="ax" x="${m.l + pw / 2}" y="${H - 1}" text-anchor="middle">characters of context</text>`;
  svg += `<path d="${bits.map((b, i) => `${i ? "L" : "M"}${X(i + 1)},${Y(b)}`).join("")}" fill="none" stroke="var(--s1)" stroke-width="2"/>`;
  svg += bits.map((b, i) => `<circle cx="${X(i + 1)}" cy="${Y(b)}" r="${i + 1 === chosen ? 6 : 4}" fill="var(--s1)" class="ring" data-tip="${i + 1} characters of context: ${b.toFixed(3)} bits per character${i + 1 === chosen ? " (chosen)" : ""}" tabindex="0"/>`).join("");
  const cb = bits[chosen - 1];
  svg += `<text class="lbl" x="${Math.min(X(chosen), m.l + pw - 70)}" y="${Y(cb) - 12}" text-anchor="middle">chosen: ${cb.toFixed(2)}</text></svg>`;
  el.innerHTML = `<div class="lab-chart">${svg}</div><p class="small muted" style="margin:6px 0 0">Bits per character on the held-back tenth, if the model looked back that many characters (random guessing: ${random.toFixed(1)}). Too long a context starts to memorise instead of generalise.</p>`;
  hoverTips($(".lab-chart", el));
}

/* ───────────────────────── actions ───────────────────────── */
Object.assign(ACTIONS, {
  async "lab-sample"(key) {
    try {
      const d = await api(`/api/lab/samples/${key}`, { method: "POST" });
      LAB.datasets = await api("/api/lab/datasets"); LAB.sel = { kind: "dataset", id: d.id }; openSel();
    } catch (e) { toast(e.message, "err"); }
  },
  "lab-open"(arg) { const [kind, id] = arg.split(":"); LAB.sel = { kind, id }; openSel(); if (innerWidth < 1000) $("#labMain")?.scrollIntoView({ behavior: "smooth" }); },
  "lab-task"(k) { LAB.task = k; paintDataset(); },
  "lab-speed"(k) { LAB.speed = k; store.set("lab.speed", k); $$("#labSpeed button").forEach((b) => b.classList.toggle("on", b.dataset.arg === k)); if ($("#labBudget") && LAB.ds) $("#labBudget").textContent = fmtSec(budgetOf(LAB.ds.bytes, k)); },
  "lab-col"(name) { LAB.ignore.has(name) ? LAB.ignore.delete(name) : LAB.ignore.add(name); $("#labCols").innerHTML = colChips(LAB.ds); },
  async "lab-train"() {
    const d = LAB.ds;
    try {
      const j = await api("/api/lab/train", { method: "POST", body: { dataset: d.id, target: d.kind === "table" ? LAB.target : null, task: LAB.task, ignore: [...LAB.ignore], speed: LAB.speed } });
      LAB.jobs = await api("/api/lab/jobs"); LAB.sel = { kind: "job", id: j.id }; openSel();
    } catch (e) { toast(e.message, "err"); }
  },
  async "lab-stop"() { if (LAB.job) await api(`/api/lab/jobs/${LAB.job.id}/stop`, { method: "POST" }).catch(() => {}); toast("Stopping — keeping what it learned so far", ""); },
  async "lab-del"(arg) {
    const [kind, id] = arg.split(":");
    if (!confirm(kind === "model" ? "Delete this trained model?" : "Delete this dataset? Models trained on it are kept.")) return;
    try {
      await api(`/api/lab/${kind === "model" ? "models" : "datasets"}/${id}`, { method: "DELETE" });
      [LAB.datasets, LAB.models] = await Promise.all([api("/api/lab/datasets"), api("/api/lab/models")]);
      LAB.sel = LAB.models[0] ? { kind: "model", id: LAB.models[0].id } : LAB.datasets[0] ? { kind: "dataset", id: LAB.datasets[0].id } : null; openSel();
    } catch (e) { toast(e.message, "err"); }
  },
  async "lab-write"() {
    const out = $("#labGen"), prompt = $("#labPrompt").value;
    out.hidden = false; out.innerHTML = `<span class="muted">writing…</span>`;
    try {
      const r = await api(`/api/lab/models/${LAB.model.id}/predict`, { method: "POST", body: { prompt, length: +$("#labLen").value, temperature: +$("#labTemp").value } });
      out.innerHTML = `<span class="muted">${esc(prompt)}</span>${esc(r.text)}`;
    } catch (e) { out.innerHTML = `<span class="lab-err">${esc(e.message)}</span>`; }
  },
});
addEventListener("resize", debounce(() => { if (S.view === "lab" && LAB.sel?.kind === "model" && LAB.redraw) LAB.redraw(); }, 250));
