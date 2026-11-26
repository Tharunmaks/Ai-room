"""HTTP API for the Lab: upload data, train untrained models on it in the background, and use what they learned."""

from __future__ import annotations

import csv
import io
import json
import os
import queue
import re
import shutil
import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from . import samples
from .automl import SPEEDS, Trained, new_id, train_table
from .data import DataError, ext_of, pick_target, pick_task, profile, read_table, read_text, sniff_kind
from .textgen import TextModel, train_text

MAX_MB = float(os.environ.get("AI_ROOM_LAB_MAX_MB", "100"))
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class TrainBody(BaseModel):
    dataset: str
    target: str | None = None
    task: str = "auto"                 # auto | classification | regression
    ignore: list[str] = []
    speed: str = "balanced"            # quick | balanced | thorough
    name: str | None = None


class PredictBody(BaseModel):
    rows: list[dict] | None = None     # table models
    prompt: str | None = None          # text models: continue this text
    length: int = 400
    temperature: float = 0.8
    seed: int | None = None
    query: str | None = None           # text models: find passages


def _check_id(x: str) -> str:
    if not SAFE_ID.match(x or ""):
        raise HTTPException(400, "bad id")
    return x


def mount(app: FastAPI, data_dir: str | None) -> dict:
    import tempfile

    base = Path(data_dir) / "lab" if data_dir else Path(tempfile.mkdtemp(prefix="ai_room_lab_"))
    ds_dir, model_dir = base / "datasets", base / "models"
    ds_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, dict] = {}
    work: queue.Queue = queue.Queue()
    cache: OrderedDict[str, object] = OrderedDict()
    lock = threading.Lock()

    # ------------------------------------------------------------------ datasets
    def read_json(p: Path, default=None):
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return default

    def dataset_info(did: str) -> dict:
        info = read_json(ds_dir / _check_id(did) / "info.json")
        if info is None:
            raise HTTPException(404, "dataset not found")
        return info

    def raw_of(info: dict) -> bytes:
        return (ds_dir / info["id"] / info["file"]).read_bytes()

    def add_dataset(filename: str, raw: bytes, sample: str | None = None) -> dict:
        if not raw:
            raise HTTPException(400, "The file is empty.")
        filename = os.path.basename(filename or "upload.csv")[:120] or "upload.csv"
        try:
            kind = sniff_kind(filename, raw)
            info = {"name": re.sub(r"\.[A-Za-z0-9]+$", "", filename) or filename, "filename": filename, "kind": kind, "bytes": len(raw),
                    "created": time.time(), "sample": sample}
            if kind == "table":
                cols, rows = read_table(raw, filename)
                prof = profile(cols, rows)
                target = (samples.SAMPLES.get(sample) or {}).get("target") or pick_target(prof)
                tp = next((p for p in prof if p["name"] == target), None)
                info |= {"rows": len(rows), "columns": cols, "profile": prof, "preview": rows[:12], "target": target,
                         "task": pick_task(tp) if tp else "classification"}
            elif kind == "text":
                docs = read_text(raw, filename)
                text_len = sum(len(t) for _, t in docs)
                info |= {"documents": len(docs), "titles": [t for t, _ in docs][:50], "chars": text_len, "preview_text": docs[0][1][:1500]}
            else:
                raise DataError("Folders of images are not supported yet. Upload a table (CSV, TSV, JSON) or text (txt, md, html, pdf, docx, epub).")
        except DataError as e:
            raise HTTPException(400, str(e))
        did = new_id()
        info["id"] = did
        info["file"] = "data" + (ext_of(filename) or ".bin")
        folder = ds_dir / did
        folder.mkdir(parents=True)
        (folder / info["file"]).write_bytes(raw)
        (folder / "info.json").write_text(json.dumps(info))
        return info

    def brief(info: dict) -> dict:
        return {k: info.get(k) for k in ("id", "name", "filename", "kind", "bytes", "rows", "columns", "target", "task", "documents", "chars", "created", "sample")}

    @app.get("/api/lab/info")
    def lab_info():
        import numpy
        return {"speeds": SPEEDS, "max_mb": MAX_MB, "cpus": os.cpu_count(), "numpy": numpy.__version__,
                "samples": [{"key": k, **v} for k, v in samples.SAMPLES.items()], "busy": sum(j["status"] in ("queued", "running") for j in jobs.values())}

    @app.post("/api/lab/datasets")
    async def upload(request: Request, filename: str = "upload.csv"):
        size = int(request.headers.get("content-length") or 0)
        if size > MAX_MB * 1024 * 1024:
            raise HTTPException(413, f"That file is larger than {MAX_MB:g} MB. Set AI_ROOM_LAB_MAX_MB to allow bigger uploads.")
        raw = await request.body()
        if len(raw) > MAX_MB * 1024 * 1024:
            raise HTTPException(413, f"That file is larger than {MAX_MB:g} MB.")
        return add_dataset(filename, raw)

    @app.post("/api/lab/samples/{name}")
    def add_sample(name: str):
        if name not in samples.SAMPLES:
            raise HTTPException(404, "no such sample")
        fn, raw = samples.make(name)
        info = add_dataset(fn, raw, sample=name)
        return info

    @app.get("/api/lab/datasets")
    def list_datasets():
        out = [read_json(p / "info.json") for p in ds_dir.iterdir() if p.is_dir()]
        return sorted((brief(i) for i in out if i), key=lambda i: -i["created"])

    @app.get("/api/lab/datasets/{did}")
    def get_dataset(did: str):
        return dataset_info(did)

    @app.delete("/api/lab/datasets/{did}")
    def delete_dataset(did: str):
        dataset_info(did)
        shutil.rmtree(ds_dir / did, ignore_errors=True)
        return {"ok": True}

    # ------------------------------------------------------------------ training
    def run(job: dict):
        job["status"], job["started"] = "running", time.time()
        stop = job["stop"]

        def progress(e: dict):
            e["at"] = round(time.time() - job["started"], 3)
            job["events"].append(e)

        try:
            info = dataset_info(job["dataset"])
            raw = raw_of(info)
            mid = new_id()
            folder = model_dir / mid
            if info["kind"] == "text":
                model = train_text(read_text(raw, info["filename"]), progress=progress, stop=stop)
            else:
                cols, rows = read_table(raw, info["filename"])
                model = train_table(cols, rows, job["target"], task=job["task"], ignore=job["ignore"], budget=SPEEDS.get(job["speed"], 30.0),
                                    progress=progress, stop=stop)
            model.save(folder)
            write_card(folder, model, name=job["name"] or info["name"], dataset=info["id"], dataset_name=info["name"],
                       seconds=time.time() - job["started"], speed=job["speed"], events=job["events"])
            job["model"] = mid
            job["status"] = "stopped" if stop.is_set() else "done"
        except (DataError, ValueError) as e:
            job["status"], job["error"] = "error", str(e)
        except Exception as e:  # report anything else in plain words instead of losing the job silently
            traceback.print_exc()
            job["status"], job["error"] = "error", f"Training failed: {type(e).__name__}: {e}"
        finally:
            job["finished"] = time.time()

    def worker():
        while True:
            job = work.get()
            if job["stop"].is_set():
                job["status"] = "stopped"
                continue
            run(job)

    threading.Thread(target=worker, daemon=True, name="lab-trainer").start()

    @app.post("/api/lab/train")
    def train(body: TrainBody):
        info = dataset_info(body.dataset)
        if body.speed not in SPEEDS:
            raise HTTPException(400, f"speed must be one of {', '.join(SPEEDS)}")
        if body.task not in ("auto", "classification", "regression"):
            raise HTTPException(400, "task must be auto, classification or regression")
        if info["kind"] == "table" and body.target and body.target not in info["columns"]:
            raise HTTPException(400, f"There is no column called '{body.target}'.")
        jid = new_id()
        job = {"id": jid, "dataset": info["id"], "dataset_name": info["name"], "kind": info["kind"], "target": body.target or info.get("target"),
               "task": body.task, "ignore": body.ignore, "speed": body.speed, "name": body.name, "status": "queued", "events": [],
               "created": time.time(), "stop": threading.Event(), "model": None, "error": None}
        jobs[jid] = job
        work.put(job)
        return public_job(job)

    def public_job(job: dict, after: int = 0) -> dict:
        return {k: v for k, v in job.items() if k not in ("stop", "events")} | {"events": job["events"][after:], "next": len(job["events"])}

    @app.get("/api/lab/jobs")
    def list_jobs():
        return [public_job(j, len(j["events"])) for j in sorted(jobs.values(), key=lambda j: -j["created"])[:20]]

    @app.get("/api/lab/jobs/{jid}")
    def get_job(jid: str, after: int = 0):
        if jid not in jobs:
            raise HTTPException(404, "job not found")
        return public_job(jobs[jid], max(0, after))

    @app.post("/api/lab/jobs/{jid}/stop")
    def stop_job(jid: str):
        if jid not in jobs:
            raise HTTPException(404, "job not found")
        jobs[jid]["stop"].set()
        return {"ok": True}

    # ------------------------------------------------------------------ models
    def card_of(mid: str) -> dict:
        card = read_json(model_dir / _check_id(mid) / "card.json")
        if card is None:
            raise HTTPException(404, "model not found")
        return card

    def load(mid: str):
        with lock:
            if mid in cache:
                cache.move_to_end(mid)
                return cache[mid]
        card = card_of(mid)
        folder = model_dir / mid
        m = TextModel.load(folder) if card["type"] == "text" else Trained.load(folder)
        with lock:
            cache[mid] = m
            while len(cache) > 6:
                cache.popitem(last=False)
        return m

    @app.get("/api/lab/models")
    def list_models():
        cards = [read_json(p / "card.json") for p in model_dir.iterdir() if p.is_dir()]
        return sorted((c for c in cards if c), key=lambda c: -c["created"])

    @app.get("/api/lab/models/{mid}")
    def get_model(mid: str):
        card = card_of(mid)
        meta = read_json(model_dir / mid / "model.json", {})
        return card | {"report": meta.get("report", {})}

    @app.delete("/api/lab/models/{mid}")
    def delete_model(mid: str):
        card_of(mid)
        with lock:
            cache.pop(mid, None)
        shutil.rmtree(model_dir / mid, ignore_errors=True)
        return {"ok": True}

    @app.post("/api/lab/models/{mid}/predict")
    def predict(mid: str, body: PredictBody):
        m = load(mid)
        if isinstance(m, TextModel):
            out = {}
            if body.query is not None:
                out["passages"] = m.search(body.query, 5)
            if body.prompt is not None or body.query is None:
                t = time.perf_counter()
                out["text"] = m.model.generate(body.prompt or "", body.length, body.temperature, body.seed)
                out["seconds"] = round(time.perf_counter() - t, 3)
            return out
        if not body.rows:
            raise HTTPException(400, "Send rows to predict, e.g. {\"rows\": [{\"column\": \"value\"}]}")
        if len(body.rows) > 10000:
            raise HTTPException(400, "Send at most 10,000 rows at a time (or upload a file).")
        t = time.perf_counter()
        preds = m.predict(body.rows)
        return {"predictions": preds, "seconds": round(time.perf_counter() - t, 4)}

    @app.post("/api/lab/models/{mid}/predict_file")
    async def predict_file(mid: str, request: Request, filename: str = "rows.csv"):
        m = load(mid)
        if isinstance(m, TextModel):
            raise HTTPException(400, "Text models write text; upload a table to a table model.")
        raw = await request.body()
        if len(raw) > MAX_MB * 1024 * 1024:
            raise HTTPException(413, "file too large")
        try:
            cols, rows = read_table(raw, filename)
        except DataError as e:
            raise HTTPException(400, str(e))
        dicts = [dict(zip(cols, r)) for r in rows]
        preds = m.predict(dicts)
        buf = io.StringIO()
        w = csv.writer(buf)
        name = f"predicted {m.target}"
        w.writerow(cols + [name] + (["confidence"] if m.task.kind == "classification" else []))
        for r, p in zip(rows, preds):
            w.writerow(r + ([p["prediction"], round(p["confidence"], 4)] if m.task.kind == "classification" else [round(p["prediction"], 6)]))
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", re.sub(r"\.[A-Za-z0-9]+$", "", filename))[:60] or "rows"
        return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{stem}_predicted.csv"'})

    return {"jobs": jobs, "base": base}


def write_card(folder: Path, model, name: str, dataset: str | None = None, dataset_name: str = "", seconds: float = 0.0, speed: str = "",
               events=()) -> dict:
    """The summary the model list shows (card.json), next to the saved model."""
    report = model.report
    if isinstance(model, TextModel):
        card = {"type": "text", "summary": f"{report['chars']:,} characters, context {report['order']}",
                "score": {"bits_per_char": report["valid_bits_per_char"], "random_bits_per_char": report["random_bits_per_char"]}}
    else:
        card = {"type": "table", "task": report["task"], "target": report["target"], "winner": report["winner_label"],
                "score": report["test"], "baseline": report["baseline_test"], "inputs": model_inputs(model)}
    card |= {"id": Path(folder).name, "name": name, "dataset": dataset, "dataset_name": dataset_name or name, "created": time.time(),
             "seconds": round(seconds, 2), "speed": speed, "curve": learning_curve(list(events))}
    (Path(folder) / "card.json").write_text(json.dumps(card))
    return card


def model_inputs(model: Trained) -> list[dict]:
    """What the 'try it' form needs for each column the model uses."""
    prof = {p["name"]: p for p in model.report.get("profile", [])}
    out = []
    for s in model.encoder.spec:
        p = prof.get(s["col"], {})
        item = {"name": s["col"], "kind": s["kind"]}
        if s["kind"] == "number":
            item |= {"min": p.get("min"), "max": p.get("max"), "default": p.get("mean"), "integers": p.get("integers", False)}
        elif s["kind"] == "category":
            item |= {"choices": [c for c in s["cats"] if c], "default": next((c for c in s["cats"] if c), "")}
        elif s["kind"] == "date":
            item |= {"default": (p.get("examples") or [""])[0]}
        else:
            item |= {"default": (p.get("examples") or [""])[0]}
        out.append(item)
    return out


def learning_curve(events: list[dict], per_learner: int = 120) -> dict:
    """{learner: [[seconds since the job started, validation score], …]} from progress events, thinned out."""
    out: dict[str, list] = {}
    for e in events:
        if e.get("type") == "progress" and e.get("val") is not None:
            out.setdefault(e["learner"], []).append([e["at"], round(float(e["val"]), 5)])
        elif e.get("type") == "learner_done":
            v = e.get("valid", {})
            score = v.get("accuracy", v.get("r2"))
            if score is not None:
                out.setdefault(e["learner"], []).append([e["at"], round(float(score), 5)])
    for k, pts in out.items():
        if len(pts) > per_learner:
            step = len(pts) / per_learner
            out[k] = [pts[int(i * step)] for i in range(per_learner - 1)] + [pts[-1]]
    return out
