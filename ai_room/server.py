"""HTTP API + web UI for the AI Room."""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import os
import socket
import threading
import time
import traceback
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .classroom import ClassConfig, Classroom, Stopped
from .curriculum import CATEGORIES, DIFFICULTY, NUM_GRADES, SUBJECTS, topic_phrases
from .llm import MAX_STUDENT_PARAMS_B, ModelTooLargeError
from .memory import Store
from .teacher import PRESETS, Library, NoCourseMaterial, Teacher, style_from
from .teacher.fetch import USER_AGENT, fetch_subject
from .teacher.generators import families_for, has_generators
from .teacher.library import html_to_text, parse_bank_text
from .teacher.style import APPROACHES, TONES

WEB = Path(__file__).parent / "web"
MAX_UPLOAD = 25 * 1024 * 1024


class CreateBody(BaseModel):
    name: str = "My Classroom"
    config: dict


class RunBody(BaseModel):
    action: str  # lessons | exam | class | all


class TalkBody(BaseModel):
    text: str
    to: str = "student"


class FetchBody(BaseModel):
    subjects: list[str]
    per_site: int = 2
    sources: list[str] = ["simple", "wikipedia", "wikibooks"]


class ImportBody(BaseModel):
    subject: str
    filename: str = "upload.txt"
    data_b64: str | None = None
    text: str | None = None
    title: str | None = None


class UrlBody(BaseModel):
    subject: str
    url: str
    title: str | None = None


class BankBody(BaseModel):
    subject: str | None = None
    text: str


class PreviewBody(BaseModel):
    persona: str = "patient"
    style: dict = {}
    subject: str = "math"
    grade: int = 3
    student_name: str = "Ada"


def _safe_public_url(url: str) -> str:
    """Only http(s) URLs that resolve to public addresses (no SSRF into the local network)."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise HTTPException(400, "Only http(s) URLs are supported.")
    if os.environ.get("AI_ROOM_ALLOW_PRIVATE_URLS") == "1":
        return url
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))
    except socket.gaierror:
        raise HTTPException(400, f"Cannot resolve {p.hostname}.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise HTTPException(400, "That address is not public. Set AI_ROOM_ALLOW_PRIVATE_URLS=1 to allow it.")
    return url


def create_app(db_path: str | None = None, library_path: str | None = None, llm_factory=None, fetcher=None) -> FastAPI:
    """`llm_factory(cfg) -> student_llm` and `fetcher` let tests inject a model and a fake Wikimedia."""
    db = db_path or os.environ.get("AI_ROOM_DB", "data/ai_room.db")
    lib_path = library_path or os.environ.get("AI_ROOM_LIBRARY") or (":memory:" if db == ":memory:" else os.path.join(os.path.dirname(db) or ".", "library.db"))
    store = Store(db)
    library = Library(lib_path)
    fetch = fetcher or fetch_subject
    rooms: dict[str, Classroom] = {}
    workers: dict[str, threading.Thread] = {}
    jobs: dict[str, dict] = {}
    app = FastAPI(title="AI Room")

    # exact-practice coverage per subject, computed once
    coverage = {k: sum(1 for g in range(1, NUM_GRADES + 1) if families_for(k, g)) for k in SUBJECTS}

    def room(cid: str) -> Classroom:
        if cid not in rooms:
            data = store.get_classroom(cid)
            if data is None:
                raise HTTPException(404, "classroom not found")
            llm = llm_factory(ClassConfig.from_dict(data["config"])) if llm_factory else None
            try:
                rooms[cid] = Classroom(store, cid, library, llm, fetch)
            except (ModelTooLargeError, ValueError, RuntimeError) as e:
                raise HTTPException(400, str(e))
        return rooms[cid]

    def busy(cid: str) -> bool:
        t = workers.get(cid)
        return bool(t and t.is_alive())

    # ------------------------------------------------------------------ pages / meta
    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/meta")
    def meta():
        return {
            "grades": NUM_GRADES, "difficulty": DIFFICULTY, "max_params_b": MAX_STUDENT_PARAMS_B, "categories": CATEGORIES,
            "tones": TONES, "approaches": APPROACHES,
            "subjects": [{"key": s.key, "name": s.name, "category": s.category, "icon": s.icon, "types": s.question_types, "syllabus": s.grades,
                          "exact_classes": coverage[s.key], "mode": "exact" if coverage[s.key] == NUM_GRADES else "mixed" if coverage[s.key] else "textbook"}
                         for s in SUBJECTS.values()],
            "personas": [{"key": k, **v.to_dict()} for k, v in PRESETS.items()],
        }

    @app.get("/api/ollama/models")
    def ollama_models(base_url: str | None = None):
        base = (base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        base = base if base.startswith("http") else "http://" + base
        try:
            r = httpx.get(f"{base}/api/tags", timeout=3)
            r.raise_for_status()
            return {"ok": True, "models": [m["name"] for m in r.json().get("models", [])]}
        except (httpx.HTTPError, ValueError, KeyError) as e:
            return {"ok": False, "models": [], "error": f"Could not reach Ollama at {base}: {e.__class__.__name__}"}

    # ------------------------------------------------------------------ classrooms
    @app.get("/api/classrooms")
    def list_rooms():
        return [{**r, "config": ClassConfig.from_dict(r["config"]).to_dict()} for r in store.list_classrooms()]

    @app.post("/api/classrooms")
    def create(body: CreateBody):
        try:
            cfg = ClassConfig.from_dict(body.config)
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"Invalid config: {e}")
        cid = store.create_classroom(body.name, cfg.to_dict(redact=False))
        try:
            r = room(cid)  # validates the 27B limit and model reachability
        except HTTPException:
            store.delete_classroom(cid)
            raise
        store.add_message(cid, "Classroom", "system", f"Welcome to class 1 of {NUM_GRADES}! {r.teacher.name} will teach {r.student.name}. Press “Start class” to begin.", {"kind": "welcome"})
        return {"id": cid}

    def describe(cid: str) -> dict:
        r = room(cid)
        return {"id": cid, "name": store.get_classroom(cid)["name"], "busy": busy(cid), "config": r.cfg.to_dict(),
                "teacher": r.style.to_dict(), **r.report()}

    @app.get("/api/classrooms/{cid}")
    def get(cid: str):
        return describe(cid)

    @app.patch("/api/classrooms/{cid}/config")
    def update_config(cid: str, patch: dict):
        if busy(cid):
            raise HTTPException(409, "Stop the class before changing settings.")
        data = store.get_classroom(cid)
        if data is None:
            raise HTTPException(404, "classroom not found")
        allowed = {"persona", "style", "subjects", "questions_per_exam", "pass_mark", "max_attempts", "auto_fetch", "student_questions", "student_name"}
        merged = {**data["config"], **{k: v for k, v in patch.items() if k in allowed}}
        try:
            ClassConfig.from_dict(merged)
        except (TypeError, ValueError) as e:
            raise HTTPException(400, str(e))
        store.update_classroom(cid, config=merged)
        rooms.pop(cid, None)
        r = room(cid)
        r.say("system", f"Settings updated. Teacher: {r.teacher.name}. Subjects: {', '.join(SUBJECTS[s].name for s in r.cfg.subjects)}.", kind="settings")
        return describe(cid)

    @app.delete("/api/classrooms/{cid}")
    def delete(cid: str):
        if cid in rooms and busy(cid):
            rooms[cid].stop_event.set()
        rooms.pop(cid, None)
        store.delete_classroom(cid)
        return {"ok": True}

    @app.post("/api/classrooms/{cid}/run")
    def run(cid: str, body: RunBody):
        r = room(cid)
        if busy(cid):
            raise HTTPException(409, "The class is already running.")
        if r.graduated:
            raise HTTPException(400, "This student already graduated from all 12 classes.")
        actions = {"lessons": r.lessons, "exam": r.exam, "class": r.run_class, "all": r.run_all}
        if body.action not in actions:
            raise HTTPException(400, f"action must be one of {list(actions)}")

        def work():
            r.stop_event.clear()
            try:
                actions[body.action]()
            except Stopped:
                r.say("system", "Class paused.", kind="status")
            except NoCourseMaterial as e:
                r.say("system", f"📚 {e}", kind="error")
            except Exception as e:  # surface model/network errors in the transcript
                traceback.print_exc()
                r.say("system", f"⚠️ Error: {e}", kind="error")
            finally:
                r.status = "idle"

        t = threading.Thread(target=work, daemon=True)
        workers[cid] = t
        t.start()
        return {"started": body.action}

    @app.post("/api/classrooms/{cid}/stop")
    def stop(cid: str):
        room(cid).stop_event.set()
        return {"ok": True}

    @app.post("/api/classrooms/{cid}/talk")
    def talk(cid: str, body: TalkBody):
        if not body.text.strip():
            raise HTTPException(400, "empty message")
        if body.to not in ("student", "teacher"):
            raise HTTPException(400, "to must be 'student' or 'teacher'")
        try:
            return {"reply": room(cid).talk(body.text.strip(), body.to)}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"Model error: {e}")

    @app.get("/api/classrooms/{cid}/messages")
    def messages(cid: str, after: int = 0):
        return store.messages(cid, after)

    @app.get("/api/classrooms/{cid}/events")
    async def events(cid: str, after: int = 0):
        async def gen():
            last = after
            while True:
                for m in store.messages(cid, last):
                    last = m["id"]
                    yield f"data: {json.dumps(m)}\n\n"
                yield ": ping\n\n"
                await asyncio.sleep(0.7)
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/classrooms/{cid}/notebook")
    def notebook(cid: str, subject: str | None = None):
        return [n.__dict__ for n in store.notes(cid, subject)]

    @app.get("/api/classrooms/{cid}/exams")
    def exams(cid: str, grade: int | None = None):
        return store.exams(cid, grade)

    # ------------------------------------------------------------------ teacher preview
    @app.post("/api/teacher/preview")
    def preview(body: PreviewBody):
        if body.subject not in SUBJECTS:
            raise HTTPException(400, "unknown subject")
        grade = max(1, min(NUM_GRADES, body.grade))
        style = style_from(body.persona, body.style)
        t = Teacher(library, style, body.student_name or "Ada")
        try:
            text = t.lesson(body.subject, grade).text
            available = True
        except NoCourseMaterial as e:
            text, available = str(e), False
        return {"text": text[:6000], "available": available, "name": style.name, "avatar": style.avatar, "style": style.to_dict()}

    # ------------------------------------------------------------------ library
    def public(job: dict) -> dict:
        """A job record without its internal fields (the stop event is not JSON)."""
        return {k: v for k, v in job.items() if not k.startswith("_")}

    @app.get("/api/library")
    def library_status():
        stats = library.stats()
        subjects = {k: {**stats.get(k, {"docs": 0, "passages": 0, "chars": 0, "questions": 0}), "exact_classes": coverage[k]} for k in SUBJECTS}
        totals = {"docs": sum(s["docs"] for s in subjects.values()), "passages": sum(s["passages"] for s in subjects.values()),
                  "questions": sum(s["questions"] for s in subjects.values()), "chars": sum(s["chars"] for s in subjects.values())}
        return {"subjects": subjects, "totals": totals, "jobs": [public(j) for j in list(jobs.values())[-10:]]}

    @app.get("/api/library/documents")
    def library_documents(subject: str | None = None):
        return library.documents(subject)

    @app.delete("/api/library/documents/{doc_id}")
    def library_delete(doc_id: int):
        library.delete_document(doc_id)
        return {"ok": True}

    @app.get("/api/library/search")
    def library_search(subject: str, q: str, k: int = 5):
        if subject not in SUBJECTS:
            raise HTTPException(400, "unknown subject")
        return [{"title": p.title, "heading": p.heading, "text": p.text, "url": p.url, "source": p.source, "score": round(p.score, 2)}
                for p in library.search(subject, q, k=max(1, min(k, 20)))]

    def run_fetch_job(job_id: str, subjects: list[str], per_site: int, sources: tuple[str, ...]):
        job = jobs[job_id]
        stop_event: threading.Event = job["_stop"]
        try:
            for i, subject in enumerate(subjects):
                if stop_event.is_set():
                    break
                job.update(subject=subject, index=i)

                def progress(message, done, total, subject=subject):
                    job.update(message=f"{SUBJECTS[subject].name}: {message}", done=done, total=total)

                summary = fetch(library, subject, progress=progress, stop=stop_event, per_site=per_site, sources=sources)
                job["added"] += summary.get("added", 0)
                job["errors"] += summary.get("errors", [])
            job["status"] = "stopped" if stop_event.is_set() else "done"
        except Exception as e:  # keep the job record useful even when the network is down
            traceback.print_exc()
            job.update(status="error", errors=job["errors"] + [str(e)])
        job["finished"] = time.time()

    @app.post("/api/library/fetch")
    def library_fetch(body: FetchBody):
        bad = [s for s in body.subjects if s not in SUBJECTS]
        if bad or not body.subjects:
            raise HTTPException(400, f"unknown or empty subjects: {bad}")
        if any(j["status"] == "running" for j in jobs.values()):
            raise HTTPException(409, "A fetch is already running. Wait for it or stop it first.")
        job_id = uuid.uuid4().hex[:8]
        jobs[job_id] = {"id": job_id, "status": "running", "subjects": body.subjects, "subject": body.subjects[0], "index": 0, "message": "Starting…",
                        "done": 0, "total": 1, "added": 0, "errors": [], "started": time.time(), "_stop": threading.Event()}
        threading.Thread(target=run_fetch_job, args=(job_id, body.subjects, max(1, min(body.per_site, 4)), tuple(body.sources)), daemon=True).start()
        return {"id": job_id}

    @app.get("/api/library/jobs/{job_id}")
    def library_job(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "no such job")
        return public(jobs[job_id])

    @app.post("/api/library/jobs/{job_id}/stop")
    def library_job_stop(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "no such job")
        jobs[job_id]["_stop"].set()
        return {"ok": True}

    @app.post("/api/library/import")
    def library_import(body: ImportBody):
        if body.subject not in SUBJECTS:
            raise HTTPException(400, "unknown subject")
        try:
            if body.data_b64:
                data = base64.b64decode(body.data_b64, validate=False)
            elif body.text:
                data = body.text.encode()
            else:
                raise HTTPException(400, "Provide data_b64 or text.")
            if len(data) > MAX_UPLOAD:
                raise HTTPException(413, "File too large (25 MB limit).")
            return library.import_file(body.subject, body.filename, data, body.title)
        except HTTPException:
            raise
        except (ValueError, RuntimeError) as e:
            raise HTTPException(400, str(e))

    @app.post("/api/library/import_url")
    def library_import_url(body: UrlBody):
        if body.subject not in SUBJECTS:
            raise HTTPException(400, "unknown subject")
        _safe_public_url(body.url)
        try:
            r = httpx.get(body.url, timeout=20, headers={"User-Agent": USER_AGENT}, follow_redirects=False)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(502, f"Could not download that page: {e}")
        if len(r.content) > MAX_UPLOAD:
            raise HTTPException(413, "Page too large.")
        ctype = r.headers.get("content-type", "")
        if "html" in ctype or body.url.lower().endswith((".html", ".htm")):
            text, title = html_to_text(r.text)
        elif "text" in ctype:
            text, title = r.text, body.url
        else:
            raise HTTPException(400, f"Unsupported content type '{ctype}'. Download the file and upload it instead.")
        doc = library.add_document(body.subject, body.title or title or body.url, text, source="web", url=body.url)
        if doc is None:
            raise HTTPException(400, "No usable text on that page (or it is already in the library).")
        return {"doc_id": doc, "title": body.title or title, "chars": len(text)}

    @app.post("/api/library/bank")
    def library_bank(body: BankBody):
        if body.subject and body.subject not in SUBJECTS:
            raise HTTPException(400, "unknown subject")
        try:
            items = parse_bank_text(body.text, body.subject)
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(400, f"Could not read the question bank: {e}")
        by_subject: dict[str, list[dict]] = {}
        for subject, item in items:
            if subject in SUBJECTS:
                by_subject.setdefault(subject, []).append(item)
        added = sum(library.add_questions(s, qs) for s, qs in by_subject.items())
        if not added:
            raise HTTPException(400, "No usable questions found. Check the format in the README (question, answer, type, options…).")
        return {"added": added, "subjects": {s: len(q) for s, q in by_subject.items()}}

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


def main() -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="Run the AI Room web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--db", default=None)
    a = p.parse_args()
    uvicorn.run(create_app(a.db), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
