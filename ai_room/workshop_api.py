"""HTTP API for the Workshop: browse and call tools, run the student on a goal, recipes, files, connections and permissions."""

from __future__ import annotations

import json
import mimetypes
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .llm import ModelTooLargeError, check_student_size, make_llm
from .tools import REGISTRY, call, categories, make_context, missing, search
from .tools import recipes as rec
from .tools.agent import RulePlanner, Workshop, run_to_recipe, save_run
from .tools.registry import DEFAULT_ALLOWED, RISKS, Context

# What each connection unlocks. Keys are read from this panel or the environment; they are never sent back to the browser.
CONNECTIONS = [
    {"key": "YOUTUBE_API_KEY", "name": "YouTube Data API key", "help": "Google Cloud Console → APIs → YouTube Data API v3 → Credentials → API key. Free quota; read-only.", "kind": "api key"},
    {"key": "YOUTUBE_ACCESS_TOKEN", "name": "YouTube OAuth token", "help": "Only needed to upload videos (OAuth, scope youtube.upload). Uploading is a 'post' action and is off by default.", "kind": "oauth token"},
    {"key": "INSTAGRAM_ACCESS_TOKEN", "name": "Instagram Graph API token", "help": "Needs an Instagram Business/Creator account linked to a Facebook Page and a Meta app. Official API only — no passwords, no scraping.", "kind": "oauth token"},
    {"key": "INSTAGRAM_USER_ID", "name": "Instagram account ID", "help": "The numeric Instagram Business account id from the Graph API.", "kind": "id"},
    {"key": "GITHUB_TOKEN", "name": "GitHub token (optional)", "help": "Raises rate limits and enables code search. A fine-grained read-only token is enough.", "kind": "token"},
    {"key": "HF_TOKEN", "name": "Hugging Face token", "help": "huggingface.co/settings/tokens — for the hosted Inference API.", "kind": "token"},
    {"key": "KAGGLE_USERNAME", "name": "Kaggle username", "help": "With KAGGLE_KEY, or put kaggle.json in ~/.kaggle. Needs the `kaggle` CLI installed (pip install kaggle).", "kind": "username"},
    {"key": "KAGGLE_KEY", "name": "Kaggle API key", "help": "kaggle.com → Settings → API → Create New Token.", "kind": "api key"},
]
PROGRAMS = [{"bin": "ffmpeg", "for": "video, audio, GIFs", "how": "apt install ffmpeg / brew install ffmpeg"}, {"bin": "blender", "for": "real 3D rendering and animation", "how": "blender.org/download (or apt install blender)"},
            {"bin": "kaggle", "for": "Kaggle datasets and competitions", "how": "pip install kaggle"}, {"bin": "yt-dlp", "for": "downloading videos you are allowed to download", "how": "pip install yt-dlp"}]


class RunBody(BaseModel):
    goal: str
    max_steps: int = 14
    backend: str = "demo"        # demo = rule-based planner (not an AI) | ollama | openai
    model: str = ""
    base_url: str | None = None
    student_params_b: float | None = None


class CallBody(BaseModel):
    args: dict = {}


class SettingsBody(BaseModel):
    allowed: list[str] | None = None
    secrets: dict[str, str] | None = None     # {KEY: value}; an empty value removes it


class RecipeRunBody(BaseModel):
    params: dict = {}


class SaveRecipeBody(BaseModel):
    name: str
    description: str = ""


def public_tool(t, ctx: Context) -> dict:
    miss = missing(t, ctx)
    return {**t.card(), "ready": not miss, "missing": miss, "allowed": t.risk in ctx.allowed}


def mount(app: FastAPI, data_dir: str | None, workspace: str | None = None, llm_factory=None) -> None:
    base = Path(data_dir) if data_dir else Path(tempfile.mkdtemp(prefix="ai_room_"))
    ws = Path(workspace or os.environ.get("AI_ROOM_WORKSPACE") or base / "workspace")
    ws.mkdir(parents=True, exist_ok=True)
    settings_path = base / "workshop_settings.json"
    secrets_path = base / "workshop_secrets.json"

    def read(p: Path, default):
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return default

    state = {"allowed": [r for r in read(settings_path, {}).get("allowed", list(DEFAULT_ALLOWED)) if r in RISKS], "secrets": {k: v for k, v in read(secrets_path, {}).items() if isinstance(v, str)}}
    runs: dict[str, dict] = {}
    lock = threading.Lock()

    def ctx() -> Context:
        return make_context(ws, state["secrets"], tuple(state["allowed"]))

    def persist():
        base.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps({"allowed": state["allowed"]}))
        secrets_path.write_text(json.dumps(state["secrets"]))
        try:
            os.chmod(secrets_path, 0o600)
        except OSError:
            pass

    # ------------------------------------------------------------------ tools
    @app.get("/api/tools/summary")
    def tools_summary():
        c = ctx()
        ready = sum(1 for t in REGISTRY.values() if not missing(t, c))
        return {"total": len(REGISTRY), "ready": ready, "needs_setup": len(REGISTRY) - ready, "categories": categories(),
                "risks": {k: {"desc": v, "count": sum(1 for t in REGISTRY.values() if t.risk == k), "allowed": k in c.allowed} for k, v in RISKS.items()}}

    @app.get("/api/tools")
    def tools_list(q: str = "", category: str = "", ready: bool = False, risk: str = "", limit: int = 60, offset: int = 0):
        c = ctx()
        limit = max(1, min(limit, 200))
        if q.strip():
            found = search(q, 400, c, category or None, ready)
        else:
            found = [t for t in sorted(REGISTRY.values(), key=lambda t: (t.category, t.name)) if (not category or t.category == category) and (not ready or not missing(t, c))]
        if risk:
            found = [t for t in found if t.risk == risk]
        return {"total": len(found), "tools": [public_tool(t, c) for t in found[offset:offset + limit]]}

    @app.get("/api/tools/{name}")
    def tool_detail(name: str):
        if name not in REGISTRY:
            raise HTTPException(404, "no such tool")
        return public_tool(REGISTRY[name], ctx())

    @app.post("/api/tools/{name}/call")
    def tool_call(name: str, body: CallBody):
        if name not in REGISTRY:
            raise HTTPException(404, "no such tool")
        return call(name, body.args, ctx())

    # ------------------------------------------------------------------ settings and connections
    @app.get("/api/workshop/settings")
    def get_settings():
        return {"allowed": state["allowed"], "risks": {k: v for k, v in RISKS.items()}, "defaults": list(DEFAULT_ALLOWED), "workspace": str(ws),
                "connections": [{**c, "set": bool(state["secrets"].get(c["key"]) or os.environ.get(c["key"])), "from_env": bool(os.environ.get(c["key"]) and not state["secrets"].get(c["key"]))} for c in CONNECTIONS],
                "programs": [{**p, "installed": __import__("shutil").which(p["bin"]) is not None} for p in PROGRAMS]}

    @app.post("/api/workshop/settings")
    def set_settings(body: SettingsBody):
        if body.allowed is not None:
            bad = [r for r in body.allowed if r not in RISKS]
            if bad:
                raise HTTPException(400, f"unknown permission levels: {bad}")
            state["allowed"] = sorted(set(body.allowed) | {"safe"})
        if body.secrets:
            known = {c["key"] for c in CONNECTIONS}
            for k, v in body.secrets.items():
                if k not in known:
                    raise HTTPException(400, f"unknown connection {k}")
                if v.strip():
                    state["secrets"][k] = v.strip()
                else:
                    state["secrets"].pop(k, None)
        persist()
        return get_settings()

    # ------------------------------------------------------------------ the student works with tools
    def make_student(body: RunBody):
        if llm_factory:
            return llm_factory(body)
        if body.backend == "demo":
            return RulePlanner()
        try:
            llm = make_llm(body.backend, body.model, body.base_url)
            check_student_size(llm, body.student_params_b)
            return llm
        except ModelTooLargeError as e:
            raise HTTPException(400, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))

    def worker(rid: str, llm, goal: str, max_steps: int):
        r = runs[rid]
        w = Workshop(llm, ctx(), max_steps, on_event=lambda e: r["events"].append(e), cancel=r["_cancel"])
        try:
            result = w.run(goal)
            r["result"] = result
            r["status"] = result["status"]
            try:
                r["saved"] = save_run(ctx(), result)
            except OSError:
                pass
        except Exception as e:  # never leave a run "running" forever
            r["status"] = "error"
            r["error"] = f"{e.__class__.__name__}: {e}"
            r["events"].append({"kind": "error", "message": r["error"]})
        r["finished"] = time.time()

    def public_run(r: dict, after: int = 0) -> dict:
        return {"id": r["id"], "goal": r["goal"], "status": r["status"], "error": r.get("error"), "events": r["events"][after:], "next": len(r["events"]), "started": r["started"], "finished": r.get("finished"),
                "model": r["model"], "final": (r.get("result") or {}).get("final"), "files": sorted({a["path"] for a in (r.get("result") or {}).get("artifacts", [])})}

    @app.post("/api/workshop/run")
    def workshop_run(body: RunBody):
        goal = body.goal.strip()
        if not goal:
            raise HTTPException(400, "Tell the student what to make or find out.")
        with lock:
            if any(r["status"] == "running" for r in runs.values()):
                raise HTTPException(409, "The student is already working on something. Wait for it or stop it first.")
            llm = make_student(body)
            rid = uuid.uuid4().hex[:10]
            runs[rid] = {"id": rid, "goal": goal, "status": "running", "events": [], "started": time.time(), "model": getattr(llm, "name", body.backend), "_cancel": threading.Event()}
        threading.Thread(target=worker, args=(rid, llm, goal, max(1, min(body.max_steps, 40))), daemon=True).start()
        return {"id": rid}

    @app.get("/api/workshop/runs")
    def workshop_runs():
        return sorted(({k: v for k, v in public_run(r).items() if k != "events"} for r in runs.values()), key=lambda r: -r["started"])[:20]

    @app.get("/api/workshop/runs/{rid}")
    def workshop_run_get(rid: str, after: int = 0):
        if rid not in runs:
            raise HTTPException(404, "no such run")
        return public_run(runs[rid], max(0, after))

    @app.post("/api/workshop/runs/{rid}/stop")
    def workshop_stop(rid: str):
        if rid not in runs:
            raise HTTPException(404, "no such run")
        runs[rid]["_cancel"].set()
        return {"ok": True}

    @app.post("/api/workshop/runs/{rid}/save_recipe")
    def workshop_save_recipe(rid: str, body: SaveRecipeBody):
        r = runs.get(rid)
        if not r or not r.get("result"):
            raise HTTPException(404, "that run has not finished")
        recipe = run_to_recipe(r["result"], body.name, body.description)
        if not recipe["steps"]:
            raise HTTPException(400, "the run made no successful tool calls to save")
        out = call("recipe_save", {"name": body.name, "description": recipe["description"], "steps": recipe["steps"]}, ctx())
        if not out["ok"]:
            raise HTTPException(400, out["error"])
        return {"ok": True, "steps": len(recipe["steps"])}

    # ------------------------------------------------------------------ recipes
    @app.get("/api/recipes")
    def recipes_list():
        c = ctx()
        out = []
        for name, desc in rec.list_recipes(c).items():
            r = rec.get_recipe(c, name)
            out.append({"name": name, "description": desc, "builtin": name in rec.BUILTIN, "params": r.get("params", {}), "steps": [{"tool": s["tool"], "args": s.get("args", {})} for s in r["steps"]]})
        return out

    @app.post("/api/recipes/{name}/run")
    def recipe_run(name: str, body: RecipeRunBody):
        out = call("recipe_run", {"name": name, "params": body.params}, ctx())
        if not out["ok"]:
            raise HTTPException(400, out["error"])
        return json.loads(out["result"])

    # ------------------------------------------------------------------ the workspace's files
    KIND = {".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image", ".webp": "image", ".svg": "image", ".mp4": "video", ".webm": "video", ".mov": "video", ".mkv": "video",
            ".wav": "audio", ".mp3": "audio", ".ogg": "audio", ".flac": "audio", ".m4a": "audio", ".obj": "model", ".stl": "model", ".gltf": "model", ".glb": "model", ".ply": "model",
            ".py": "code", ".ipynb": "notebook", ".json": "data", ".csv": "data", ".txt": "text", ".md": "text", ".blend": "model"}

    def safe_path(rel: str) -> Path:
        try:
            return ctx().path(rel, True)
        except Exception:
            raise HTTPException(404, "no such file")

    @app.get("/api/workspace/files")
    def workspace_files(kind: str = ""):
        root = ws.resolve()
        out = []
        for p in sorted(root.rglob("*"), key=lambda p: -p.stat().st_mtime if p.is_file() else 0):
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts):
                k = KIND.get(p.suffix.lower(), "file")
                if not kind or k == kind:
                    out.append({"path": str(p.relative_to(root)), "kind": k, "bytes": p.stat().st_size, "modified": p.stat().st_mtime})
            if len(out) >= 400:
                break
        return out

    @app.get("/api/workspace/file")
    def workspace_file(path: str, download: bool = False):
        p = safe_path(path)
        if not p.is_file():
            raise HTTPException(404, "no such file")
        mt = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if p.suffix.lower() in (".obj", ".stl", ".py", ".ipynb", ".csv", ".md", ".txt", ".json"):
            mt = "text/plain; charset=utf-8" if p.suffix.lower() != ".json" else "application/json"
        # Files are made by tools and models, so they are never trusted: no scripts may run from them.
        headers = {"Content-Security-Policy": "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:; media-src 'self'", "X-Content-Type-Options": "nosniff"}
        return FileResponse(p, media_type=mt, headers=headers, filename=p.name if download else None)

    @app.delete("/api/workspace/file")
    def workspace_delete(path: str):
        p = safe_path(path)
        if not p.is_file():
            raise HTTPException(404, "no such file")
        p.unlink()
        return {"ok": True}
