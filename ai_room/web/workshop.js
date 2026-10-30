"use strict";
/* Workshop: give the student a goal, watch it use the toolbox, and browse everything it made. */
Object.assign(ICONS, {
  tools: '<path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.6 2.6-2.4-.6-.6-2.4z"/>',
  film: '<rect x="3" y="4" width="18" height="16" rx="2.5"/><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4"/>',
  cube: '<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12 4 7.5M12 12l8-4.5M12 12v9"/>',
  key: '<circle cx="8" cy="15" r="4"/><path d="m11 12 9-9M16 7l3 3"/>', bolt: '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>', shield: '<path d="M12 3 4 6v6c0 5 3.4 8.2 8 9 4.6-.8 8-4 8-9V6z"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="2.5"/><circle cx="9" cy="10" r="1.8"/><path d="m21 16-5-5-8 8"/>', repeat: '<path d="M17 2l4 4-4 4"/><path d="M3 11V9a3 3 0 0 1 3-3h15M7 22l-4-4 4-4"/><path d="M21 13v2a3 3 0 0 1-3 3H3"/>',
});
const WS = { tab: store.get("ws.tab", "gallery"), summary: null, settings: null, recipes: null, files: null, run: null, timer: null, q: "", cat: "", ready: true, tools: null, showMore: 60, fileKind: "all", model: store.get("ws.model", "demo"), tag: store.get("ws.tag", "") };
const fileURL = (p, dl = false) => `/api/workspace/file?path=${encodeURIComponent(p)}${dl ? "&download=true" : ""}`;
function openPanel(title, body, width = "") {
  const m = $("#modal");
  m.innerHTML = `<div class="mh"><h3 style="font-size:17px;display:flex;align-items:center;gap:8px">${title}</h3><button class="btn icon ghost" style="margin-left:auto" data-close aria-label="Close">${icon("x")}</button></div><div class="mb">${body}</div>`;
  m.style.width = width; $("[data-close]", m).onclick = () => m.close(); m.showModal(); return m;
}
const RISK_ICON = { safe: "🧮", files: "📁", network: "🌐", exec: "⚙️", post: "📣" };
const EXAMPLES = [
  ["🏠", "Build a 3D house in Blender and show me a preview"], ["📊", "Make a bar chart of 12 18 9 22 15"], ["📸", "Make an Instagram post about learning without training"],
  ["▶️", "Create a YouTube thumbnail kit about black holes"], ["🎬", "Make a short explainer video about photosynthesis"], ["🧮", "Calculate 17*(23+4)^2"], ["📓", "Write a Colab and Kaggle notebook to train a classifier"],
];

async function renderWorkshop() {
  const el = $("#ws");
  el.innerHTML = `<div class="skeleton" style="height:260px"></div>`;
  try { [WS.summary, WS.settings] = await Promise.all([api("/api/tools/summary"), api("/api/workshop/settings")]); } catch (e) { el.innerHTML = `<div class="empty"><div class="big">⚠️</div>${esc(e.message)}</div>`; return; }
  const s = WS.summary;
  el.innerHTML = `
    <div class="page-head"><div><h2>Workshop</h2><p>Give the student a goal. It searches the toolbox, picks tools, checks what came back, and keeps going until it has made something.</p></div>
      <div class="grow"></div><div class="row wrap gap-s" id="wsStats"></div></div>
    <div class="ws-hero card">
      <div class="ws-prompt">
        <label class="sr" for="wsGoal">What should the student make or find out?</label>
        <textarea id="wsGoal" rows="3" placeholder="e.g. Build a 3D snowman in Blender, preview it, then make an Instagram post about it…"></textarea>
        <div class="row wrap gap-s" id="wsEx"></div>
      </div>
      <div class="ws-controls">
        <label class="field"><span>Who does the work?</span><select id="wsModel">
          <option value="demo">Demo planner — rule-based, not an AI</option><option value="ollama">Ollama model (≤ 27B)</option><option value="openai">OpenAI-compatible server (≤ 27B)</option></select></label>
        <label class="field" id="wsModelNameBox" hidden><span>Model name</span><input type="text" id="wsModelName" placeholder="qwen2.5:14b" value="${esc(store.get("ws.modelname", ""))}"></label>
        <label class="field" id="wsBaseBox" hidden><span>Server address (optional)</span><input type="text" id="wsBase" placeholder="http://localhost:11434"></label>
        <label class="field"><span>Max steps: <b id="wsStepsV">14</b></span><input type="range" id="wsSteps" min="3" max="30" value="14"></label>
        <div class="row"><button class="btn primary big grow" id="wsGo" data-act="ws-run">${icon("play")} Start</button><button class="btn big" id="wsStop" data-act="ws-stop" hidden>${icon("stop")} Stop</button></div>
      </div>
    </div>
    <div id="wsRun"></div>
    <div class="row wrap" style="margin:22px 0 14px"><div class="seg" id="wsTabs" role="tablist"></div></div>
    <div id="wsBody"></div>`;
  $("#wsStats").innerHTML = `<span class="chip on">${icon("tools")} ${s.total.toLocaleString()} tools</span><span class="chip good">${icon("check")} ${s.ready.toLocaleString()} ready</span>${s.needs_setup ? `<span class="chip warn">${s.needs_setup} need setup</span>` : ""}${(WS.recipes || []).length ? `<span class="chip">${WS.recipes.length} recipes</span>` : ""}`;
  $("#wsEx").innerHTML = EXAMPLES.map(([e, t]) => `<button class="chip" type="button" data-act="ws-example" data-arg="${esc(t)}">${e} ${esc(t.length > 38 ? t.slice(0, 36) + "…" : t)}</button>`).join("");
  const model = $("#wsModel"); model.value = WS.model;
  const syncModel = () => { WS.model = model.value; store.set("ws.model", WS.model); $("#wsModelNameBox").hidden = $("#wsBaseBox").hidden = WS.model === "demo"; };
  model.onchange = syncModel; syncModel();
  $("#wsSteps").oninput = (e) => ($("#wsStepsV").textContent = e.target.value);
  $("#wsGoal").onkeydown = (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") ACTIONS["ws-run"](); };
  if (WS.run) { WS.run.rendered = 0; WS.run.finalDone = false; paintRun(true); if (WS.run.status === "running" && !WS.timer) WS.timer = setTimeout(pollRun, 300); }
  paintTabs(); paintTab();
}

function paintTabs() {
  const tabs = [["gallery", "Made files", "image"], ["toolbox", "Toolbox", "tools"], ["recipes", "Recipes", "repeat"], ["connections", "Permissions & keys", "shield"]];
  $("#wsTabs").innerHTML = tabs.map(([k, l, ic]) => `<button role="tab" class="${WS.tab === k ? "on" : ""}" data-act="ws-tab" data-arg="${k}">${icon(ic)} ${l}</button>`).join("");
}
async function paintTab() {
  const body = $("#wsBody"); if (!body) return;
  body.innerHTML = `<div class="skeleton" style="height:180px"></div>`;
  try { await ({ gallery: tabGallery, toolbox: tabToolbox, recipes: tabRecipes, connections: tabConnections }[WS.tab])(body); } catch (e) { body.innerHTML = `<div class="empty"><div class="big">⚠️</div>${esc(e.message)}</div>`; }
}

/* ───────── running ───────── */
function paintRun(rebuild) {
  const r = WS.run; if (!r) return;
  const box = $("#wsRun"); if (!box) return;
  if (rebuild) box.innerHTML = `<div class="card ws-console"><div class="row" style="margin-bottom:10px"><span class="pulse" id="wsPulse"></span><b id="wsGoalTitle"></b><div class="grow"></div><span class="chip" id="wsStatus"></span></div><div class="ws-steps" id="wsSteps2"></div><div id="wsFinal"></div></div>`;
  $("#wsGoalTitle").textContent = r.goal;
  $("#wsGo").hidden = r.status === "running"; $("#wsStop").hidden = r.status !== "running";
  for (; r.rendered < r.events.length; r.rendered++) paintEvent(r.events[r.rendered]);
  const done = r.status !== "running";
  $("#wsPulse").style.display = done ? "none" : "";
  const label = { running: "working…", done: "finished", max_steps: "ran out of steps", gave_up: "got stuck", stopped: "stopped", error: "error" }[r.status] || r.status;
  const chip = $("#wsStatus"); chip.textContent = label; chip.className = "chip " + (r.status === "done" ? "good" : done ? "warn" : "on");
  if (done) paintFinal();
}
function paintEvent(e) {
  const st = $("#wsSteps2");
  const add = (html, cls = "") => { const d = document.createElement("div"); d.className = "ws-step " + cls; d.innerHTML = html; st.appendChild(d); d.scrollIntoView({ block: "nearest", behavior: "smooth" }); return d; };
  if (e.kind === "call") {
    const args = JSON.stringify(e.args ?? {});
    const d = add(`<div class="ws-ic">${icon("bolt")}</div><div class="grow"><div class="row wrap gap-s"><b class="ws-tool">${esc(e.tool)}</b><span class="muted tiny">step ${e.step}</span></div>${e.thought ? `<div class="ws-thought">${esc(e.thought)}</div>` : ""}
      <details><summary class="tiny muted">arguments</summary><pre class="ws-args">${esc(JSON.stringify(e.args ?? {}, null, 1))}</pre></details><div class="ws-out"><span class="muted small">running…</span></div></div>`, "pending");
    WS.run.last = d;
  } else if (e.kind === "observation") {
    const d = WS.run.last; if (!d) return;
    d.classList.remove("pending"); d.classList.add(e.ok ? "ok" : "bad");
    const files = (e.files || []).map((f) => /\.(png|jpe?g|gif|webp|svg)$/i.test(f) ? `<a href="${fileURL(f)}" target="_blank" rel="noopener"><img class="ws-thumb" loading="lazy" src="${fileURL(f)}" alt="${esc(f)}"></a>` : `<a class="chip" href="${fileURL(f)}" target="_blank" rel="noopener">${icon("link")} ${esc(f)}</a>`).join("");
    $(".ws-out", d).innerHTML = `<div class="ws-obs ${e.ok ? "" : "err-text"}">${esc((e.text || "").slice(0, 600))}</div>${files ? `<div class="ws-files">${files}</div>` : ""}${e.blocked ? `<div class="small" style="margin-top:6px"><button class="btn sm" data-act="ws-allow" data-arg="${esc(e.blocked)}">${icon("shield")} Allow “${esc(e.blocked)}” actions…</button></div>` : ""}`;
  } else if (e.kind === "search") {
    add(`<div class="ws-ic">${icon("search")}</div><div class="grow"><div><b>Looking for tools:</b> ${esc(e.query)}</div><div class="ws-files">${(e.tools || []).slice(0, 8).map((t) => `<span class="chip">${esc(t)}</span>`).join("")}</div></div>`);
  } else if (e.kind === "invalid") {
    add(`<div class="ws-ic">${icon("info")}</div><div class="grow"><b>The reply was not a usable action.</b><div class="muted small">${esc((e.raw || "").slice(0, 160))}</div></div>`, "bad");
  } else if (e.kind === "error") {
    add(`<div class="ws-ic">${icon("x")}</div><div class="grow"><b>Problem</b><div class="small">${esc(e.message)}</div></div>`, "bad");
  }
}
async function paintFinal() {
  const r = WS.run, box = $("#wsFinal"); if (!box || r.finalDone) return; r.finalDone = true;
  const files = r.files || [];
  box.innerHTML = `<div class="ws-final"><div class="row"><span style="font-size:22px">${r.status === "done" ? "✅" : "⚠️"}</span><b>${r.status === "done" ? "Done" : "Stopped"}</b></div><div class="md">${md(r.final || r.error || "")}</div>
    ${files.length ? `<div class="ws-files">${files.slice(0, 12).map((f) => /\.(png|jpe?g|gif|webp|svg)$/i.test(f) ? `<a href="${fileURL(f)}" target="_blank" rel="noopener"><img class="ws-thumb big" src="${fileURL(f)}" alt="${esc(f)}"></a>` : `<a class="chip" href="${fileURL(f)}" target="_blank" rel="noopener">${esc(f)}</a>`).join("")}</div>` : ""}
    <div class="row wrap gap-s" style="margin-top:12px"><button class="btn sm" data-act="ws-save-recipe">${icon("repeat")} Save these steps as a recipe</button><button class="btn sm" data-act="ws-tab" data-arg="gallery">${icon("image")} See all files</button></div></div>`;
  if (WS.tab === "gallery") paintTab();
}
async function pollRun() {
  const r = WS.run; if (!r) return;
  try {
    const d = await api(`/api/workshop/runs/${r.id}?after=${r.events.length}`);
    r.events.push(...d.events); Object.assign(r, { status: d.status, final: d.final, files: d.files, error: d.error });
    paintRun(false);
    if (d.status !== "running") { WS.timer = null; if (WS.tab === "gallery") paintTab(); return; }
  } catch (e) { toast(e.message, "err"); WS.timer = null; return; }
  WS.timer = setTimeout(pollRun, 600);
}

/* ───────── tabs ───────── */
async function tabGallery(body) {
  WS.files = await api("/api/workspace/files");
  const kinds = ["all", "image", "video", "audio", "model", "notebook", "code", "data", "text"];
  const have = new Set(WS.files.map((f) => f.kind));
  const shown = WS.files.filter((f) => WS.fileKind === "all" || f.kind === WS.fileKind);
  body.innerHTML = `<div class="row wrap gap-s" style="margin-bottom:12px">${kinds.filter((k) => k === "all" || have.has(k)).map((k) => `<button class="chip ${WS.fileKind === k ? "on" : ""}" data-act="ws-kind" data-arg="${k}">${k === "all" ? `All · ${WS.files.length}` : k}</button>`).join("")}<div class="grow"></div><span class="muted small" title="${esc(WS.settings.workspace)}">Workspace: ${esc(WS.settings.workspace.split("/").slice(-2).join("/"))}</span></div>
    ${shown.length ? `<div class="ws-gallery">${shown.slice(0, 120).map(fileCard).join("")}</div>` : `<div class="empty"><div class="big">🗂️</div>Nothing made yet. Start the student above, or try a recipe.</div>`}`;
}
function fileCard(f) {
  const name = f.path.split("/").pop(), url = fileURL(f.path);
  let media;
  if (f.kind === "image") media = `<img loading="lazy" src="${url}" alt="${esc(name)}">`;
  else if (f.kind === "video") media = `<video src="${url}" controls preload="metadata" muted playsinline></video>`;
  else if (f.kind === "audio") media = `<div class="ws-icon">🎵</div><audio src="${url}" controls preload="none"></audio>`;
  else media = `<div class="ws-icon">${{ model: "🧊", notebook: "📓", code: "🐍", data: "🗃️", text: "📝" }[f.kind] || "📄"}</div>`;
  const acts = [`<a class="btn sm ghost" href="${fileURL(f.path, true)}" aria-label="Download">${icon("download")}</a>`];
  if (f.kind === "model" && /\.(obj|stl)$/i.test(f.path)) acts.unshift(`<button class="btn sm" data-act="ws-preview3d" data-arg="${esc(f.path)}">${icon("cube")} View</button>`);
  if (["code", "data", "text", "notebook"].includes(f.kind) || /\.(obj|stl)$/i.test(f.path)) acts.unshift(`<button class="btn sm" data-act="ws-open" data-arg="${esc(f.path)}">Open</button>`);
  acts.push(`<button class="btn sm ghost danger" data-act="ws-del" data-arg="${esc(f.path)}" aria-label="Delete ${esc(name)}">${icon("trash")}</button>`);
  return `<figure class="ws-file"><div class="ws-media">${media}</div><figcaption><b title="${esc(f.path)}">${esc(name)}</b><span class="muted tiny">${esc(f.path.includes("/") ? f.path.split("/").slice(0, -1).join("/") : "·")} · ${(f.bytes / 1024).toFixed(f.bytes > 1e5 ? 0 : 1)} KB</span><div class="row gap-s wrap">${acts.join("")}</div></figcaption></figure>`;
}

async function tabToolbox(body) {
  const cats = Object.entries(WS.summary.categories);
  body.innerHTML = `<div class="row wrap gap-s" style="margin-bottom:12px"><div class="grow" style="position:relative;min-width:240px"><input type="search" id="tbQ" placeholder="Search ${WS.summary.total.toLocaleString()} tools — try “convert video to gif”, “blender material”, “youtube”…" value="${esc(WS.q)}" style="padding-left:38px"><span style="position:absolute;left:12px;top:10px;color:var(--muted)">${icon("search")}</span></div>
    <label class="switch"><input type="checkbox" id="tbReady" ${WS.ready ? "checked" : ""}><i></i>Ready only</label></div>
    <div class="tabs" id="tbCats"><button class="chip ${!WS.cat ? "on" : ""}" data-act="tb-cat" data-arg="">All</button>${cats.map(([c, n]) => `<button class="chip ${WS.cat === c ? "on" : ""}" data-act="tb-cat" data-arg="${esc(c)}">${esc(c.replace("_", " "))} <span class="muted">${n}</span></button>`).join("")}</div>
    <div id="tbList" style="margin-top:14px"></div>`;
  $("#tbQ").oninput = debounce((e) => { WS.q = e.target.value; WS.showMore = 60; loadTools(); }, 220);
  $("#tbReady").onchange = (e) => { WS.ready = e.target.checked; loadTools(); };
  await loadTools();
}
async function loadTools() {
  const box = $("#tbList"); if (!box) return;
  const p = new URLSearchParams({ q: WS.q, category: WS.cat, ready: WS.ready, limit: WS.showMore });
  const d = await api("/api/tools?" + p);
  WS.tools = d.tools;
  box.innerHTML = `<div class="muted small" style="margin-bottom:8px">${d.total.toLocaleString()} tools${WS.q ? ` matching “${esc(WS.q)}”` : ""}</div><div class="grid g-auto" style="grid-template-columns:repeat(auto-fill,minmax(300px,1fr))">${d.tools.map(toolCard).join("")}</div>
    ${d.total > d.tools.length ? `<div class="center" style="margin-top:16px"><button class="btn" data-act="tb-more">Show more</button></div>` : ""}`;
}
function toolCard(t) {
  const badge = !t.ready ? `<span class="chip warn" title="${esc(t.missing.join("; "))}">needs setup</span>` : !t.allowed ? `<span class="chip warn" title="Permission is switched off">locked</span>` : "";
  return `<div class="ws-tool-card ${t.ready && t.allowed ? "" : "dim"}"><div class="row gap-s"><b class="ws-tool">${esc(t.name)}</b><div class="grow"></div>${badge}<span class="chip" title="${esc(t.risk)}">${RISK_ICON[t.risk]} ${t.risk}</span></div>
    <div class="small muted">${esc(t.desc)}</div><div class="tiny mono muted">${esc(t.signature)}</div><div class="row gap-s wrap"><span class="chip">${esc(t.category.replace("_", " "))}</span><div class="grow"></div><button class="btn sm" data-act="tb-try" data-arg="${esc(t.name)}" ${t.ready ? "" : "disabled"}>Try it</button></div></div>`;
}
function paramInput(p) {
  const id = `p_${p.name}`, ph = p.default !== null && p.default !== undefined ? String(typeof p.default === "object" ? JSON.stringify(p.default) : p.default) : "";
  const lab = `${esc(p.name.replace(/_/g, " "))}${p.required ? " *" : ""}`;
  if (p.type === "bool") return `<label class="switch" style="margin:6px 0"><input type="checkbox" id="${id}" ${p.default ? "checked" : ""}><i></i>${lab}</label>`;
  if (p.type === "list" || p.type === "dict") return `<label class="field"><span>${lab} <span class="muted">(${p.type === "list" ? "[1, 2, 3] or a, b, c" : "JSON object"})</span></span><textarea id="${id}" rows="2" placeholder="${esc(ph)}"></textarea></label>`;
  if (p.type === "int" || p.type === "float") return `<label class="field"><span>${lab}</span><input type="number" step="${p.type === "int" ? 1 : "any"}" id="${id}" placeholder="${esc(ph)}"></label>`;
  return `<label class="field"><span>${lab}</span>${/code|content|text|query|slides|expression/.test(p.name) ? `<textarea id="${id}" rows="2" placeholder="${esc(ph)}"></textarea>` : `<input type="text" id="${id}" placeholder="${esc(ph)}">`}</label>`;
}
function readParams(params, root = document) {
  const args = {};
  for (const p of params) {
    const el = $("#p_" + CSS.escape(p.name), root); if (!el) continue;
    if (p.type === "bool") { args[p.name] = el.checked; continue; }
    const v = el.value.trim(); if (v === "") continue;
    if (p.type === "list" || p.type === "dict") { try { args[p.name] = JSON.parse(v); } catch (e) { args[p.name] = v; } }
    else if (p.type === "int" || p.type === "float") args[p.name] = Number(v);
    else args[p.name] = v;
  }
  return args;
}
function openToolDialog(t) {
  const m = openPanel(`${icon("tools")} <span class="mono">${esc(t.name)}</span>`, `<p class="muted small" style="margin-top:0">${esc(t.desc)}</p><div class="stack" style="gap:8px;margin:12px 0" id="tdParams">${t.params.map(paramInput).join("") || '<div class="muted small">No arguments.</div>'}</div>
    <div class="row"><button class="btn primary" type="button" id="tdRun">${icon("play")} Run</button><span class="muted small">${RISK_ICON[t.risk]} ${esc(t.risk)} action</span></div><div id="tdOut" style="margin-top:14px"></div>`, "min(640px,96vw)");
  $("#tdRun").onclick = async () => {
    const out = $("#tdOut"); out.innerHTML = `<div class="skeleton" style="height:60px"></div>`;
    try {
      const r = await api(`/api/tools/${t.name}/call`, { method: "POST", body: { args: readParams(t.params, m) } });
      const files = (r.artifacts || []).map((a) => /\.(png|jpe?g|gif|webp|svg)$/i.test(a.path) ? `<img class="ws-thumb big" src="${fileURL(a.path)}" alt="${esc(a.path)}">` : /\.(mp4|webm)$/i.test(a.path) ? `<video src="${fileURL(a.path)}" controls style="max-width:100%;border-radius:12px"></video>` : /\.(wav|mp3|ogg)$/i.test(a.path) ? `<audio controls src="${fileURL(a.path)}"></audio>` : `<a class="chip" href="${fileURL(a.path)}" target="_blank" rel="noopener">${esc(a.path)}</a>`).join("");
      out.innerHTML = `<div class="ws-obs ${r.ok ? "" : "err-text"}" style="max-height:240px;overflow:auto">${esc(r.ok ? r.result : r.error)}</div><div class="ws-files">${files}</div>${r.blocked ? `<div style="margin-top:8px"><button class="btn sm" type="button" data-act="ws-allow" data-arg="${esc(r.blocked)}">${icon("shield")} Allow “${esc(r.blocked)}” actions…</button></div>` : ""}`;
    } catch (e) { out.innerHTML = `<div class="err-text">${esc(e.message)}</div>`; }
  };
}

async function tabRecipes(body) {
  WS.recipes = await api("/api/recipes");
  body.innerHTML = `<p class="muted small" style="max-width:70ch;margin-bottom:14px">A recipe is a saved pipeline of tool calls — automation that needs no AI. Run it with new parameters, run it for a whole list, or let the student build one and save it from the run console.</p>
    <div class="grid" style="grid-template-columns:repeat(auto-fill,minmax(330px,1fr))">${WS.recipes.map((r) => `<div class="card ws-recipe" id="rc_${esc(r.name)}"><div class="row gap-s"><b>${esc(r.name.replace(/_/g, " "))}</b><div class="grow"></div>${r.builtin ? '<span class="chip">built-in</span>' : '<span class="chip on">saved</span>'}</div>
      <p class="small muted" style="margin:6px 0 10px">${esc(r.description)}</p><div class="row wrap gap-s" style="margin-bottom:10px">${r.steps.map((s, i) => `<span class="chip tiny">${i + 1}. ${esc(s.tool.replace("{{thing}}", "…"))}</span>`).join("")}</div>
      <div class="stack" style="gap:8px">${Object.entries(r.params || {}).map(([k, v]) => `<label class="field"><span>${esc(k.replace(/_/g, " "))}${v.desc ? ` <span class="muted">— ${esc(v.desc)}</span>` : ""}</span>${typeof v.default === "object" ? `<textarea rows="2" data-rp="${esc(k)}" data-json="1">${esc(JSON.stringify(v.default))}</textarea>` : `<input type="text" data-rp="${esc(k)}" value="${esc(v.default ?? "")}">`}</label>`).join("")}</div>
      <div class="row" style="margin-top:12px"><button class="btn primary sm" data-act="rc-run" data-arg="${esc(r.name)}">${icon("play")} Run</button><div class="grow"></div></div><div class="rc-out" style="margin-top:10px"></div></div>`).join("")}</div>`;
}
async function tabConnections(body) {
  const st = WS.settings = await api("/api/workshop/settings");
  const warn = { exec: "The student could run any Python or shell command on this computer. Only switch on if you trust the student model and run AI Room in a container or VM.", post: "The student could publish to your YouTube, Instagram or GitHub. Keep off unless you are watching each run." };
  body.innerHTML = `<div class="grid g-2" style="grid-template-columns:repeat(auto-fit,minmax(340px,1fr))">
    <div class="card"><h3>${icon("shield")} What the student may do</h3><div class="stack" style="gap:6px">${Object.entries(st.risks).map(([k, d]) => `<div class="ws-perm"><label class="switch"><input type="checkbox" data-risk="${k}" ${st.allowed.includes(k) ? "checked" : ""} ${k === "safe" ? "disabled" : ""}><i></i><span>${RISK_ICON[k]} <b>${k}</b></span></label><div class="muted small">${esc(d)}</div>${warn[k] ? `<div class="small" style="color:var(--warn)">⚠ ${esc(warn[k])}</div>` : ""}</div>`).join("")}</div>
      <p class="muted small" style="margin-top:12px">${WS.summary ? Object.entries(WS.summary.risks).map(([k, v]) => `${k}: ${v.count}`).join(" · ") : ""} tools per level.</p></div>
    <div class="card"><h3>${icon("key")} Connections</h3><p class="muted small" style="margin-top:-4px">Keys stay on this computer (a file with owner-only access) and are never sent back to the browser. Official APIs only — no passwords, no scraping.</p>
      <div class="stack" style="gap:10px">${st.connections.map((c) => `<label class="field"><span>${esc(c.name)} ${c.set ? `<span class="chip good tiny">${c.from_env ? "from environment" : "saved"}</span>` : ""}</span><input type="password" autocomplete="off" data-secret="${c.key}" placeholder="${c.set ? "•••••••• (leave empty to keep)" : "not set"}"><span class="tiny muted" style="margin-top:4px">${esc(c.help)}</span></label>`).join("")}</div>
      <div class="row" style="margin-top:12px"><button class="btn primary sm" data-act="ws-save-keys">Save keys</button><button class="btn sm ghost danger" data-act="ws-clear-keys">Remove all saved keys</button></div></div>
    <div class="card" style="grid-column:1/-1"><h3>${icon("cube")} Programs on this computer</h3><div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(240px,1fr))">${st.programs.map((p) => `<div class="ws-tool-card ${p.installed ? "" : "dim"}"><div class="row"><b>${esc(p.bin)}</b><div class="grow"></div><span class="chip ${p.installed ? "good" : "warn"}">${p.installed ? "installed" : "not found"}</span></div><div class="small muted">${esc(p.for)}</div>${p.installed ? "" : `<div class="tiny mono muted">${esc(p.how)}</div>`}</div>`).join("")}</div>
      <p class="muted small" style="margin-top:10px">Without Blender the Blender tools still build scenes, write scripts and draw previews. Without ffmpeg the video and audio tools are unavailable.</p></div></div>`;
  $$("[data-risk]", body).forEach((cb) => (cb.onchange = async () => {
    const allowed = $$("[data-risk]", body).filter((c) => c.checked).map((c) => c.dataset.risk);
    if ((cb.dataset.risk === "exec" || cb.dataset.risk === "post") && cb.checked && !confirm(`Allow “${cb.dataset.risk}” actions?\n\n${warn[cb.dataset.risk]}`)) { cb.checked = false; return; }
    try { WS.settings = await api("/api/workshop/settings", { method: "POST", body: { allowed } }); WS.summary = await api("/api/tools/summary"); toast("Permissions saved", "ok"); } catch (e) { toast(e.message, "err"); }
  }));
}

/* ───────── actions ───────── */
Object.assign(ACTIONS, {
  "ws-example": (t) => { $("#wsGoal").value = t; $("#wsGoal").focus(); },
  "ws-tab": (k) => { WS.tab = k; store.set("ws.tab", k); paintTabs(); paintTab(); if (k === "gallery" && $("#wsBody")) $("#wsBody").scrollIntoView({ behavior: "smooth", block: "start" }); },
  "ws-kind": (k) => { WS.fileKind = k; paintTab(); },
  async "ws-run"() {
    const goal = $("#wsGoal").value.trim(); if (!goal) { toast("Tell the student what to make or find out.", ""); $("#wsGoal").focus(); return; }
    const model = $("#wsModelName").value.trim(); store.set("ws.modelname", model);
    try {
      const { id } = await api("/api/workshop/run", { method: "POST", body: { goal, max_steps: +$("#wsSteps").value, backend: WS.model, model, base_url: $("#wsBase").value.trim() || null } });
      WS.run = { id, goal, status: "running", events: [], rendered: 0, last: null, finalDone: false, files: [] };
      paintRun(true); clearTimeout(WS.timer); WS.timer = setTimeout(pollRun, 300);
      $("#wsRun").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) { toast(e.message, "err"); }
  },
  async "ws-stop"() { if (WS.run) { await api(`/api/workshop/runs/${WS.run.id}/stop`, { method: "POST" }); toast("Stopping after this step…"); } },
  async "ws-save-recipe"() {
    const name = (prompt("Name for this recipe (letters, digits, - and _):", "my_recipe") || "").trim(); if (!name) return;
    try { const r = await api(`/api/workshop/runs/${WS.run.id}/save_recipe`, { method: "POST", body: { name } }); toast(`Saved recipe “${name}” (${r.steps} steps)`, "ok"); } catch (e) { toast(e.message, "err"); }
  },
  async "ws-allow"(risk) {
    if (!confirm(`Allow “${risk}” actions?`)) return;
    const allowed = [...new Set([...WS.settings.allowed, risk])];
    try { WS.settings = await api("/api/workshop/settings", { method: "POST", body: { allowed } }); toast(`“${risk}” actions allowed. Run the goal again.`, "ok"); $("#modal").close(); if (WS.tab === "connections") paintTab(); } catch (e) { toast(e.message, "err"); }
  },
  "tb-cat": (c) => { WS.cat = c; WS.showMore = 60; paintTab(); },
  "tb-more": () => { WS.showMore += 120; loadTools(); },
  "tb-try": (name) => { const t = (WS.tools || []).find((x) => x.name === name); if (t) openToolDialog(t); },
  async "rc-run"(name, btn) {
    const card = btn.closest(".ws-recipe"), out = $(".rc-out", card), params = {};
    $$("[data-rp]", card).forEach((i) => { let v = i.value; if (i.dataset.json) { try { v = JSON.parse(v); } catch (e) { /* keep text */ } } else if (v.trim() !== "" && !isNaN(+v)) v = +v; params[i.dataset.rp] = v; });
    btn.disabled = true; out.innerHTML = `<div class="skeleton" style="height:48px"></div>`;
    try {
      const r = await api(`/api/recipes/${name}/run`, { method: "POST", body: { params } });
      out.innerHTML = `<div class="stack" style="gap:6px">${r.steps.map((s) => `<div class="small row gap-s"><span class="chip ${s.ok ? "good" : "bad"} tiny">${s.ok ? "ok" : "failed"}</span><b>${esc(s.tool)}</b><span class="muted grow" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(s.output)}</span></div>`).join("")}</div>
        <div class="ws-files">${r.files.map((f) => /\.(png|jpe?g|gif|webp|svg)$/i.test(f) ? `<a href="${fileURL(f)}" target="_blank" rel="noopener"><img class="ws-thumb" src="${fileURL(f)}" alt="${esc(f)}"></a>` : `<a class="chip" href="${fileURL(f)}" target="_blank" rel="noopener">${esc(f)}</a>`).join("")}</div>`;
      toast(r.ok ? `Recipe finished — ${r.files.length} files` : "A step failed — see below", r.ok ? "ok" : "err");
    } catch (e) { out.innerHTML = `<div class="err-text small">${esc(e.message)}</div>`; } finally { btn.disabled = false; }
  },
  async "ws-save-keys"() {
    const secrets = {}; $$("[data-secret]").forEach((i) => { if (i.value.trim()) secrets[i.dataset.secret] = i.value.trim(); });
    if (!Object.keys(secrets).length) { toast("Nothing new to save.", ""); return; }
    try { WS.settings = await api("/api/workshop/settings", { method: "POST", body: { secrets } }); toast("Keys saved", "ok"); WS.summary = await api("/api/tools/summary"); paintTab(); } catch (e) { toast(e.message, "err"); }
  },
  async "ws-clear-keys"() {
    if (!confirm("Remove every saved key?")) return;
    const secrets = Object.fromEntries(WS.settings.connections.filter((c) => c.set && !c.from_env).map((c) => [c.key, ""]));
    try { WS.settings = await api("/api/workshop/settings", { method: "POST", body: { secrets } }); toast("Keys removed", "ok"); paintTab(); } catch (e) { toast(e.message, "err"); }
  },
  async "ws-open"(path) {
    const r = await fetch(fileURL(path)); const txt = (await r.text()).slice(0, 60000);
    openPanel(`<span class="mono">${esc(path)}</span>`, `<pre class="ws-args" style="max-height:68vh;overflow:auto;margin:0">${esc(txt)}</pre>`, "min(820px,96vw)");
  },
  async "ws-preview3d"(path) {
    toast("Rendering a preview…");
    try {
      const out = `renders/preview_${path.split("/").pop().replace(/\W+/g, "_")}.png`;
      const r = await api("/api/tools/mesh_render_png/call", { method: "POST", body: { args: { paths: [path], out } } });
      if (!r.ok) throw new Error(r.error);
      openPanel(`${icon("cube")} <span class="mono">${esc(path)}</span>`, `<img src="${fileURL(out)}" alt="3D preview" style="width:100%;border-radius:14px">`, "min(720px,96vw)");
      if (WS.tab === "gallery") paintTab();
    } catch (e) { toast(e.message, "err"); }
  },
  async "ws-del"(path) { if (!confirm(`Delete ${path}?`)) return; try { await api(`/api/workspace/file?path=${encodeURIComponent(path)}`, { method: "DELETE" }); paintTab(); } catch (e) { toast(e.message, "err"); } },
});
