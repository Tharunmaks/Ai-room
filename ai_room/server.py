"""HTTP API + web UI for the AI Room."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .classroom import ClassConfig, Classroom, Stopped
from .curriculum import CATEGORIES, DIFFICULTY, NUM_GRADES, SUBJECTS
from .llm import MAX_STUDENT_PARAMS_B, ModelTooLargeError
from .personas import TEACHER_PERSONAS

WEB = Path(__file__).parent / "web"


class CreateBody(BaseModel):
    name: str = "My Classroom"
    config: dict


class RunBody(BaseModel):
    action: str  # lessons | exam | class | all


class TalkBody(BaseModel):
    text: str
    to: str = "student"


def create_app(db_path: str | None = None, llm_factory=None) -> FastAPI:
    """`llm_factory(cfg) -> (student_llm, teacher_llm)` lets tests inject models."""
    from .memory import Store

    store = Store(db_path or os.environ.get("AI_ROOM_DB", "data/ai_room.db"))
    rooms: dict[str, Classroom] = {}
    workers: dict[str, threading.Thread] = {}
    app = FastAPI(title="AI Room")

    def room(cid: str) -> Classroom:
        if cid not in rooms:
            data = store.get_classroom(cid)
            if data is None:
                raise HTTPException(404, "classroom not found")
            llms = llm_factory(ClassConfig.from_dict(data["config"])) if llm_factory else (None, None)
            try:
                rooms[cid] = Classroom(store, cid, *llms)
            except (ModelTooLargeError, ValueError, RuntimeError) as e:
                raise HTTPException(400, str(e))
        return rooms[cid]

    def busy(cid: str) -> bool:
        t = workers.get(cid)
        return bool(t and t.is_alive())

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/meta")
    def meta():
        return {
            "grades": NUM_GRADES, "difficulty": DIFFICULTY, "max_params_b": MAX_STUDENT_PARAMS_B,
            "categories": CATEGORIES,
            "subjects": [{"key": s.key, "name": s.name, "category": s.category, "types": s.question_types,
                          "syllabus": s.grades}
                         for s in SUBJECTS.values()],
            "personas": [{"key": k, **v} for k, v in TEACHER_PERSONAS.items()],
        }

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
            room(cid)  # validates the 27B limit and model reachability
        except HTTPException:
            store.delete_classroom(cid)
            raise
        store.add_message(cid, "Classroom", "system",
                          f"Welcome to class 1 of {NUM_GRADES}! Press “Run class” to start the first lessons.", {})
        return {"id": cid}

    @app.get("/api/classrooms/{cid}")
    def get(cid: str):
        r = room(cid)
        return {"id": cid, "name": store.get_classroom(cid)["name"], "busy": busy(cid),
                "config": r.cfg.to_dict(), "teacher": {"name": r.teacher.name, "style": r.teacher.style},
                **r.report()}

    @app.patch("/api/classrooms/{cid}/config")
    def update_config(cid: str, patch: dict):
        if busy(cid):
            raise HTTPException(409, "Stop the class before changing settings.")
        data = store.get_classroom(cid)
        if data is None:
            raise HTTPException(404, "classroom not found")
        merged = {**data["config"], **{k: v for k, v in patch.items() if k not in ("student",)}}
        try:
            ClassConfig.from_dict(merged)
        except (TypeError, ValueError) as e:
            raise HTTPException(400, str(e))
        store.update_classroom(cid, config=merged)
        rooms.pop(cid, None)
        r = room(cid)
        r.say("system", f"Settings updated. Teacher: {r.teacher.name}. Subjects: "
                        f"{', '.join(SUBJECTS[s].name for s in r.cfg.subjects)}.", kind="settings")
        return {"ok": True}

    @app.delete("/api/classrooms/{cid}")
    def delete(cid: str):
        if busy(cid):
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
