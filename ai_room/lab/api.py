"""HTTP API for the Lab: upload data, train untrained models on it in the background, and use what they learned."""

from __future__ import annotations

import csv
import json
import os
import queue
import re
import shutil
import threading
import time
import traceback
import zipfile
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from . import samples
from .automl import SECONDS_PER_MB, SPEED_LABELS, SPEEDS, Trained, budget_for, new_id, train_table
from .data import DataError, Table, ext_of, pick_target, pick_task, profile, read_table, read_text
from .ingest import DOC_EXT, head_text, human, inspect_table, load_table, load_text, max_cells, sniff_path
from .textgen import MAX_CHARS, TextModel, train_text

MAX_BYTES = int(float(os.environ.get("AI_ROOM_LAB_MAX_MB", "5000000")) * 1_000_000)   # 5 TB per upload (1 MB = 10^6 bytes)
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class TrainBody(BaseModel):
    dataset: str
    target: str | None = None
    task: str = "auto"                 # auto | classification | regression
    ignore: list[str] = []
    speed: str = "fast"                # fast = 1 s per MB | balanced | thorough
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

    def path_of(info: dict) -> Path:
        return ds_dir / info["id"] / info["file"]

    def analyse(folder: Path, filename: str, size: int, sample: str | None = None) -> dict:
        """Describe a stored upload by reading only as much of it as needed."""
        path = folder / ("data" + (ext_of(filename) or ".bin"))
        kind = sniff_path(path, filename)
        info = {"name": re.sub(r"\.[A-Za-z0-9]+$", "", filename) or filename, "filename": filename, "kind": kind, "bytes": size,
                "created": time.time(), "sample": sample, "file": path.name, "id": folder.name}
        if kind == "table":
            cols, rows, total, exact = inspect_table(path, filename)
            prof = profile(cols, rows)
            target = (samples.SAMPLES.get(sample) or {}).get("target") or pick_target(prof)
            tp = next((p for p in prof if p["name"] == target), None)
            info |= {"rows": total, "rows_exact": exact, "columns": cols, "profile": prof, "preview": rows[:12], "target": target,
                     "task": pick_task(tp) if tp else "classification"}
        elif kind == "text":
            if size <= 64 << 20 or ext_of(filename) in DOC_EXT | {".zip"}:
                docs = read_text(path.read_bytes(), filename)
                info |= {"documents": len(docs), "titles": [t for t, _ in docs][:50], "chars": sum(len(t) for _, t in docs),
                         "chars_exact": True, "preview_text": docs[0][1][:1500]}
            else:
                head, _ = head_text(path, 1 << 20)
                info |= {"documents": 1, "titles": [info["name"]], "chars": size, "chars_exact": False, "preview_text": head[:1500]}
        else:
            raise DataError("Folders of images are not supported yet. Upload a table (CSV, TSV, JSON) or text (txt, md, html, pdf, docx, epub).")
        info["budget_seconds"] = round(budget_for("fast", size), 2)
        (folder / "info.json").write_text(json.dumps(info))
        return info

    def new_folder(filename: str) -> tuple[Path, str]:
        filename = os.path.basename(filename or "upload.csv")[:120] or "upload.csv"
        folder = ds_dir / new_id()
        folder.mkdir(parents=True)
        return folder, filename

    def store(folder: Path, filename: str, size: int, sample: str | None = None) -> dict:
        if size == 0:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(400, "The file is empty.")
        try:
            return analyse(folder, filename, size, sample)
        except (DataError, zipfile.BadZipFile, UnicodeError) as e:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(400, str(e))

    def brief(info: dict) -> dict:
        return {k: info.get(k) for k in ("id", "name", "filename", "kind", "bytes", "rows", "rows_exact", "columns", "target", "task", "documents", "chars",
                                         "chars_exact", "created", "sample", "budget_seconds")}

    @app.get("/api/lab/info")
    def lab_info():
        import numpy
        free = shutil.disk_usage(ds_dir).free
        return {"speeds": [{"key": k, "label": SPEED_LABELS[k], "seconds": s} for k, s in SPEEDS.items()], "seconds_per_mb": SECONDS_PER_MB,
                "max_bytes": MAX_BYTES, "max_label": human(MAX_BYTES), "disk_free": free, "disk_free_label": human(free),
                "cpus": os.cpu_count(), "numpy": numpy.__version__, "memory_cells": max_cells(),
                "samples": [{"key": k, **v} for k, v in samples.SAMPLES.items()], "busy": sum(j["status"] in ("queued", "running") for j in jobs.values())}

    @app.post("/api/lab/datasets")
    async def upload(request: Request, filename: str = "upload.csv"):
        """The file is the request body. It is streamed to disk, so its size is limited by MAX_BYTES and free disk, not memory."""
        declared = int(request.headers.get("content-length") or 0) or None
        folder, filename = new_folder(filename)
        dest = folder / ("data" + (ext_of(filename) or ".bin"))
        size = 0
        try:
            free = shutil.disk_usage(ds_dir).free
            if declared and declared > MAX_BYTES:
                raise HTTPException(413, f"That file is {human(declared)}, more than the {human(MAX_BYTES)} limit per upload.")
            if declared and declared > free - (256 << 20):
                raise HTTPException(507, f"Not enough disk space: the file is {human(declared)} but only {human(free)} is free here.")
            with open(dest, "wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, f"That file is more than the {human(MAX_BYTES)} limit per upload.")
                    if size > free - (256 << 20):
                        raise HTTPException(507, f"Ran out of disk space after {human(size)}.")
                    f.write(chunk)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return await run_in_threadpool(store, folder, filename, size)

    @app.post("/api/lab/samples/{name}")
    def add_sample(name: str):
        if name not in samples.SAMPLES:
            raise HTTPException(404, "no such sample")
        fn, raw = samples.make(name)
        folder, fn = new_folder(fn)
        (folder / ("data" + ext_of(fn))).write_bytes(raw)
        return store(folder, fn, len(raw), sample=name)

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
        t0 = time.perf_counter()

        def progress(e: dict):
            e["at"] = round(time.perf_counter() - t0, 3)
            job["events"].append(e)

        def say(text: str):
            progress({"type": "stage", "text": text})

        try:
            info = dataset_info(job["dataset"])
            path = path_of(info)
            budget = budget_for(job["speed"], info["bytes"])
            job["budget"] = round(budget, 2)
            deadline = t0 + budget
            mid = new_id()
            folder = model_dir / mid
            say(f"Time allowed: {budget:.1f} s ({SECONDS_PER_MB:g} s per MB of the {human(info['bytes'])} file" + (", at least 1 s" if budget <= 1.0 else "") + ").")
            if info["kind"] == "text":
                docs, tinfo = load_text(path, info["filename"], MAX_CHARS, say=say)
                model = train_text(docs, progress=progress, stop=stop, deadline=deadline - 0.12 * budget)
            else:
                table = load_table(path, info["filename"], say=say)
                model = train_table(table, None, job["target"], task=job["task"], ignore=job["ignore"], progress=progress, stop=stop,
                                    deadline=deadline - 0.03 * budget)
            model.save(folder)
            secs = time.perf_counter() - t0
            write_card(folder, model, name=job["name"] or info["name"], dataset=info["id"], dataset_name=info["name"],
                       seconds=secs, speed=job["speed"], events=job["events"], budget=budget, size=info["bytes"])
            job["seconds"] = round(secs, 3)
            job["model"] = mid
            job["status"] = "stopped" if stop.is_set() else "done"
        except (DataError, ValueError) as e:
            job["status"], job["error"] = "error", str(e)
        except MemoryError:
            job["status"], job["error"] = "error", "Ran out of memory. Set AI_ROOM_LAB_MAX_CELLS lower so a smaller sample of the rows is used."
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
        if body.speed == "quick":            # the old name of the fastest setting
            body.speed = "fast"
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
        """Send a table (streamed to disk, any size up to the upload limit); get it back as CSV with a prediction on every row."""
        m = load(mid)
        if isinstance(m, TextModel):
            raise HTTPException(400, "Text models write text; upload a table to a table model.")
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="lab_predict_", dir=base))
        src = tmp / ("in" + (ext_of(filename) or ".csv"))
        size = 0
        try:
            with open(src, "wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, f"That file is more than the {human(MAX_BYTES)} limit.")
                    f.write(chunk)
            out = tmp / "out.csv"
            await run_in_threadpool(predict_table_file, m, src, filename, out)
        except DataError as e:
            shutil.rmtree(tmp, ignore_errors=True)
            raise HTTPException(400, str(e))
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", re.sub(r"\.[A-Za-z0-9]+$", "", filename))[:60] or "rows"
        return FileResponse(out, media_type="text/csv", filename=f"{stem}_predicted.csv", background=BackgroundTask(shutil.rmtree, tmp, True))

    return {"jobs": jobs, "base": base}


def write_card(folder: Path, model, name: str, dataset: str | None = None, dataset_name: str = "", seconds: float = 0.0, speed: str = "",
               events=(), budget: float | None = None, size: int | None = None) -> dict:
    """The summary the model list shows (card.json), next to the saved model."""
    report = model.report
    if isinstance(model, TextModel):
        card = {"type": "text", "summary": f"{report['chars']:,} characters, context {report['order']}",
                "score": {"bits_per_char": report["valid_bits_per_char"], "random_bits_per_char": report["random_bits_per_char"]}}
    else:
        card = {"type": "table", "task": report["task"], "target": report["target"], "winner": report["winner_label"],
                "score": report["test"], "baseline": report["baseline_test"], "inputs": model_inputs(model)}
    card |= {"id": Path(folder).name, "name": name, "dataset": dataset, "dataset_name": dataset_name or name, "created": time.time(),
             "seconds": round(seconds, 2), "speed": speed, "budget": None if budget is None else round(budget, 2), "bytes": size,
             "curve": learning_curve(list(events))}
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


def predict_table_file(model: Trained, src: Path, filename: str, out: Path, chunk: int = 50_000) -> int:
    """Predict every row of a table file, reading and writing it in pieces so any size fits in memory."""
    ext = ext_of(filename)
    cls = model.task.kind == "classification"
    rows_done = 0
    with open(out, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        if ext in (".json", ".zip") or src.stat().st_size <= 8 << 20:
            cols, rows = read_table(src.read_bytes(), filename)
            batches = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
        else:
            head, _ = head_text(src, 1 << 20)
            cols = read_table(head.encode(), "head.tsv" if ext in (".tsv", ".tab") else "head.csv")[0]
            delim = "\t" if ext in (".tsv", ".tab") else (csv.Sniffer().sniff(head[:50000], delimiters=",;\t|").delimiter if head else ",")
            batches = _csv_batches(src, delim, len(cols), chunk)
        w.writerow(cols + [f"predicted {model.target}"] + (["confidence"] if cls else []))
        for rows in batches:
            if not rows:
                continue
            preds = model.predict(Table.from_rows(cols, rows))
            for r, p in zip(rows, preds):
                w.writerow(list(r) + ([p["prediction"], round(p["confidence"], 4)] if cls else [round(p["prediction"], 6)]))
            rows_done += len(rows)
    return rows_done


def _csv_batches(src: Path, delim: str, width: int, chunk: int):
    with open(src, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=delim)
        next(reader, None)
        batch = []
        for r in reader:
            if not any(c.strip() for c in r):
                continue
            batch.append((r + [""] * width)[:width])
            if len(batch) >= chunk:
                yield batch
                batch = []
        if batch:
            yield batch
