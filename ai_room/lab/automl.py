"""Train several untrained models on a table within a time budget, keep the best, and grade it on rows it never saw.

The rows are split three ways before anything is learned:
  train      — the models learn from these;
  validation — used to stop training at the right time and to choose the winner;
  test       — touched once, at the end, to give an honest score for the winner.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import DataError, Encoder, Table, XData, cell_str, missing_mask, pick_target, pick_task, profile, to_float, to_floats
from .learners import Baseline, Boosted, Learner, Neural, OneShot, Task, WordCounts, linear, log_loss
from .lgbm import LightGBM, ready as lightgbm_ready
from .parallel import can_fork, finished_predictions, race

SECONDS_PER_GB = 300.0                       # 5 minutes per gigabyte
SECONDS_PER_MB = SECONDS_PER_GB / 1000.0     # = 0.3 s per MB (1 GB = 1,000 MB = 10^9 bytes)
MIN_SECONDS = 0.5                            # however small the file
RATE_LABEL = "5 min per GB"
# fast: the whole job (reading, learning, testing, saving) gets 5 minutes per GB of data, at least half a second.
# The four learners run at the same time (one process each), so each of them has all of that time.
# balanced / thorough: at least 30 s / 2 min, for more accuracy on small files (never less than the fast budget).
SPEEDS = {"fast": 1.0, "balanced": 30.0, "thorough": 120.0}
SPEED_LABELS = {"fast": RATE_LABEL, "balanced": "≈30 s", "thorough": "≈2 min"}


def budget_for(speed: str, size_bytes: float) -> float:
    """Seconds a training job may take: 5 minutes per GB (0.3 s per MB), at least MIN_SECONDS (fast) or SPEEDS[speed]."""
    speed = "fast" if speed == "quick" else speed
    per_mb = max(MIN_SECONDS, size_bytes / 1e6 * SECONDS_PER_MB)
    return per_mb if speed == "fast" else max(SPEEDS.get(speed, 30.0), per_mb)
SHARES = {"linear": 0.08, "oneshot": 0.12, "words": 0.05, "lgbm": 0.45, "boost": 0.45, "mlp": 0.35}
ORDER = ["baseline", "oneshot", "words", "linear", "lgbm", "boost", "mlp"]


def make_learner(key: str, seed: int = 0) -> Learner:
    return {"baseline": Baseline, "oneshot": lambda: OneShot(seed=seed), "linear": linear, "words": WordCounts,
            "lgbm": lambda: LightGBM(seed=seed), "boost": lambda: Boosted(seed=seed), "mlp": lambda: Neural(seed=seed)}[key]()


def default_learners() -> list[str]:
    """Every learner, with LightGBM (when installed and imported: see lgbm.warm) in place of the NumPy boosted trees,
    which do the same job slower."""
    return [k for k in ORDER if k != ("boost" if lightgbm_ready() else "lgbm")]


# ------------------------------------------------------------------ metrics
def metrics(task: Task, pred: np.ndarray, y: np.ndarray, y_mu: float = 0.0, y_sd: float = 1.0, labels: list[str] | None = None) -> dict:
    if task.kind == "classification":
        guess = pred.argmax(1)
        K = task.classes
        out = {"accuracy": float((guess == y).mean()), "log_loss": log_loss(pred, y)}
        f1s = []
        for k in range(K):
            tp = int(((guess == k) & (y == k)).sum())
            fp = int(((guess == k) & (y != k)).sum())
            fn = int(((guess != k) & (y == k)).sum())
            if tp + fn:
                f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
        out["macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
        if K <= 20:
            cm = np.zeros((K, K), np.int64)
            np.add.at(cm, (y, guess), 1)
            out["confusion"] = cm.tolist()
        return out
    p = pred * y_sd + y_mu
    t = y * y_sd + y_mu
    err = p - t
    var = float(((t - t.mean()) ** 2).mean()) or 1.0
    return {"r2": 1 - float((err ** 2).mean()) / var, "rmse": float(np.sqrt((err ** 2).mean())), "mae": float(np.abs(err).mean())}


def headline(task: Task, m: dict) -> float:
    return m["accuracy"] if task.kind == "classification" else m["r2"]


# ------------------------------------------------------------------ splitting
def split_rows(n: int, y: np.ndarray | None, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """70/15/15 (60/20/20 for small tables), stratified by class when there are classes."""
    rng = np.random.default_rng(seed)
    fv, ft = (0.2, 0.2) if n < 200 else (0.15, 0.15)
    groups = [np.arange(n)] if y is None else [np.flatnonzero(y == k) for k in np.unique(y)]
    tr, va, te = [], [], []
    for g in groups:
        g = rng.permutation(g)
        k = len(g)
        nt = int(round(k * ft)) if k >= 3 else 0
        nv = int(round(k * fv)) if k >= 3 else (1 if k == 2 else 0)
        te.append(g[:nt]); va.append(g[nt:nt + nv]); tr.append(g[nt + nv:])
    cat = lambda parts: rng.permutation(np.concatenate(parts)).astype(np.int64)
    return cat(tr), cat(va), cat(te)


# ------------------------------------------------------------------ the trained model
@dataclass
class Trained:
    task: Task
    target: str
    labels: list[str]
    y_mu: float
    y_sd: float
    encoder: Encoder
    learners: list[Learner]
    weights: list[float]
    report: dict = field(default_factory=dict)

    def _rows(self, rows: list) -> list[list[str]]:
        cols = self.encoder.columns
        out = []
        for r in rows:
            if isinstance(r, dict):
                out.append(["" if r.get(c) is None else str(r.get(c)) for c in cols])
            else:
                r = ["" if v is None else str(v) for v in r]
                out.append((r + [""] * len(cols))[: len(cols)])
        return out

    def predict_x(self, X: XData) -> np.ndarray:
        preds = [w * l.predict(X) for l, w in zip(self.learners, self.weights)]
        return sum(preds)

    def predict(self, rows) -> list[dict]:
        """Predictions for rows given as dicts {column: value}, lists in the training column order, or a Table."""
        if isinstance(rows, Table):
            t = rows
            rows = Table([c for c in self.encoder.columns], [t.col(c) if c in t.columns else np.full(t.n, "", dtype=object) for c in self.encoder.columns])
            if not rows.n:
                return []
            X = self.encoder.transform(rows)
        elif not rows:
            return []
        else:
            X = self.encoder.transform(self._rows(rows))
        p = self.predict_x(X)
        if self.task.kind == "classification":
            out = []
            for row in p:
                order = np.argsort(-row)[: min(5, len(row))]
                out.append({"prediction": self.labels[int(order[0])], "confidence": float(row[order[0]]),
                            "top": [{"label": self.labels[int(k)], "p": float(row[k])} for k in order]})
            return out
        return [{"prediction": float(v * self.y_sd + self.y_mu)} for v in p]

    # -- saving
    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        enc_meta, arrays = self.encoder.state()
        arrays = {f"enc.{k}": v for k, v in arrays.items()}
        lmeta = []
        for i, l in enumerate(self.learners):
            m, a = l.state()
            lmeta.append({"kind": l.key, "cls": type(l).__name__, "meta": m})
            arrays |= {f"l{i}.{k}": v for k, v in a.items()}
        np.savez(folder / "weights.npz", **arrays)
        meta = {"format": 1, "type": "table", "task": vars(self.task), "target": self.target, "labels": self.labels, "y_mu": self.y_mu, "y_sd": self.y_sd,
                "encoder": enc_meta, "learners": lmeta, "weights": self.weights, "report": self.report}
        (folder / "model.json").write_text(json.dumps(meta, indent=1, default=_json_default))
        return folder

    @classmethod
    def load(cls, folder: str | Path) -> "Trained":
        folder = Path(folder)
        meta = json.loads((folder / "model.json").read_text())
        with np.load(folder / "weights.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        enc = Encoder.from_state(meta["encoder"], {k[4:]: v for k, v in arrays.items() if k.startswith("enc.")})
        classes = {"Baseline": Baseline, "OneShot": OneShot, "Boosted": Boosted, "Neural": Neural, "WordCounts": WordCounts, "LightGBM": LightGBM}
        learners = []
        for i, lm in enumerate(meta["learners"]):
            pre = f"l{i}."
            learners.append(classes[lm["cls"]].from_state(lm["meta"], {k[len(pre):]: v for k, v in arrays.items() if k.startswith(pre)}))
        return cls(Task(**meta["task"]), meta["target"], meta["labels"], meta["y_mu"], meta["y_sd"], enc, learners, meta["weights"], meta["report"])


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ------------------------------------------------------------------ training
def prepare_target(table, rows=None, target: str | None = None, task: str = "auto"):
    """(profile, target, task kind, table of rows that have a target, y, class labels, mean, sd)."""
    if not isinstance(table, Table):
        table = Table.from_rows(table, rows or [])
    prof = profile(table)
    by = {p["name"]: p for p in prof}
    target = target or pick_target(prof)
    if target not in by:
        raise DataError(f"There is no column called '{target}'. Columns: {', '.join(table.columns)}")
    tp = by[target]
    if tp["kind"] in ("empty", "constant"):
        raise DataError(f"The column '{target}' {tp.get('why', 'cannot be predicted')}, so there is nothing to learn.")
    kind = pick_task(tp) if task == "auto" else task
    col = table.col(target)
    f = to_floats(col) if kind == "regression" else None
    keep = ~np.isnan(f) if f is not None else ~missing_mask(col)
    if keep.sum() < 10:
        raise DataError(f"Only {int(keep.sum())} rows have a value in '{target}'. At least 10 are needed to learn anything.")
    if not keep.all():
        idx = np.flatnonzero(keep)
        table, col = table.take(idx), col[idx]
        f = f[idx] if f is not None else None
    if kind == "classification":
        vals = col.tolist()
        label_of = {v: cell_str(v).strip() for v in set(vals)}
        uniq = sorted(set(label_of.values()), key=lambda v: (to_float(v) is None, to_float(v) or 0, v))
        if len(uniq) < 2:
            raise DataError(f"Every row has the same '{target}' ({uniq[0]}), so there is nothing to learn.")
        if len(uniq) > 200:
            raise DataError(f"'{target}' has {len(uniq)} different values — too many classes. If it is a number, train it as regression.")
        pos = {v: i for i, v in enumerate(uniq)}
        code = {v: pos[label_of[v]] for v in label_of}
        y = np.fromiter(map(code.__getitem__, vals), np.int64, count=len(vals))
        return prof, target, kind, table, y, uniq, 0.0, 1.0
    mu, sd = float(f.mean()), float(f.std() or 1.0)
    return prof, target, kind, table, ((f - mu) / sd).astype(np.float32), [], mu, sd


def train_table(columns, rows=None, target: str | None = None, task: str = "auto", ignore=(),
                budget: float = 30.0, learners: list[str] | None = None, seed: int = 0, progress=None, stop=None,
                deadline: float | None = None, image: bool | str = False, parallel: bool = True) -> Trained:
    """Learn to predict `target` from the other columns. `columns` may be a Table (then `rows` is None).

    Everything, from reading the columns to the final score, is fitted into `budget` seconds (or until the
    absolute `deadline`, a time.perf_counter() value). `progress(event: dict)` hears what happens as it happens."""
    t_start = time.perf_counter()
    deadline = deadline if deadline is not None else t_start + budget
    total_budget = max(0.05, deadline - t_start)
    reserve = min(0.12 * total_budget, 20.0)          # kept for choosing the winner, importance and saving
    emit = progress or (lambda e: None)
    say = lambda text, **kw: emit({"type": "stage", "text": text, "t": round(time.perf_counter() - t_start, 3), **kw})
    table = columns if isinstance(columns, Table) else Table.from_rows(columns, rows or [])
    columns = table.columns
    say("Reading the columns…")
    prof, target, kind, table, y, labels, y_mu, y_sd = prepare_target(table, None, target, task)
    T = Task(kind, len(labels) if kind == "classification" else 1)
    kinds = {p["name"]: p["kind"] for p in prof if p["name"] != target and p["name"] not in set(ignore)}
    n = table.n
    tr, va, te = split_rows(n, y if kind == "classification" else None, seed)
    sample_note = f" (a random sample of about {table.total_rows:,})" if table.sampled else ""
    say(f"Split {n:,} rows{sample_note}: {len(tr):,} to learn from, {len(va):,} to check progress, {len(te):,} kept back for the final test.",
        split={"train": len(tr), "valid": len(va), "test": len(te)}, task=kind, target=target, classes=len(labels))
    enc = Encoder(columns, kinds).fit(table.take(tr), full=table, subset=tr)      # words found once for every row
    Xtr, Xva, Xte = (enc.transform(table.take(part), subset=part) for part in (tr, va, te))
    enc.drop_cache()
    ytr, yva, yte = y[tr], y[va], y[te]
    used = sorted(set(enc.owner) | {s["col"] for s in enc.spec if s["kind"] == "text"}, key=columns.index)
    n_text = sum(s["kind"] == "text" for s in enc.spec)
    nd = Xtr.dense.shape[1]
    words = f"words from {n_text} text column{'s' * (n_text > 1)}" if n_text else ""
    say((f"Turned {len(used) - n_text} column{'s' * (len(used) - n_text != 1)} into {nd:,} number{'s' * (nd != 1)} per row" + (f", plus {words}" if words else "") if nd
         else f"Turned {words} into word features") + f" ({time.perf_counter() - t_start:.2f} s so far).",
        features=Xtr.dense.shape[1], text_columns=n_text, columns=used)

    wanted = [k for k in ORDER if k in (learners or default_learners()) or k == "baseline"]
    pool = []
    for k in wanted:
        l = make_learner(k, seed)
        if l.usable(Xtr) and not (k == "words" and kind != "classification"):
            pool.append(l)
    emit({"type": "plan", "learners": [{"learner": l.key, "label": l.label, "blurb": l.blurb} for l in pool]})
    board, fitted = [], []
    t_models = time.perf_counter()
    learn_until = deadline - reserve

    def finish(l, pv, pt, secs):
        mv, mt = metrics(T, pv, yva, y_mu, y_sd), metrics(T, pt, yte, y_mu, y_sd)
        row = {"learner": l.key, "label": l.label, "blurb": l.blurb, "seconds": round(secs, 3), "valid": mv, "test": mt,
               "rows_per_second": round(len(tr) / max(secs, 1e-9))}
        board.append(row)
        fitted.append((l, pv, pt, mv))
        emit({"type": "learner_done", **row})

    def fit_here(l, until):
        t0 = time.perf_counter()

        def report(step, total, val=None, train_loss=None, note=None, _k=l.key, _t0=t0):
            emit({"type": "progress", "learner": _k, "step": step, "total": total, "val": val, "train_loss": train_loss, "note": note,
                  "t": round(time.perf_counter() - _t0, 3)})
        try:
            l.extra_eval = Xte
            l.fit(Xtr, ytr, Xva, yva, T, until, report, stop)
        except Exception as e:  # one learner failing must not lose the others
            l.extra_eval = None
            emit({"type": "learner_failed", "learner": l.key, "error": str(e)})
            return
        secs = time.perf_counter() - t0
        finish(l, *finished_predictions(l, Xva, Xte), secs)

    real = [l for l in pool if l.key != "baseline"]
    if stop is not None and stop.is_set():
        raise DataError("Training was stopped before any model finished.")
    for l in pool:
        if l.key == "baseline":
            emit({"type": "learner_start", "learner": l.key, "label": l.label, "blurb": l.blurb, "seconds": 0.0})
            fit_here(l, learn_until)
    left = learn_until - time.perf_counter()
    if real and left < 0.05 * total_budget:
        # reading the data used up (nearly) all the time: rather than end with no model, the one-shot net (no epochs,
        # it sizes itself to the time) still learns, even if that takes a little longer than allowed
        keep = next((l for l in real if l.key == "oneshot"), real[0])
        for l in real:
            if l is not keep:
                emit({"type": "learner_skipped", "learner": l.key, "label": l.label, "why": "out of time"})
        say(f"Reading the data used almost all the time, so only the {keep.label.lower()} learns (it needs no epochs). "
            "Choose Balanced to give every learner time.")
        emit({"type": "learner_start", "learner": keep.key, "label": keep.label, "blurb": keep.blurb, "seconds": round(max(left, 0.02), 2)})
        fit_here(keep, time.perf_counter() + max(left, 0.02))
    elif parallel and len(real) > 1 and can_fork():
        # all learners at once, one process each, every one of them with all the time that is left
        give = max(0.02, learn_until - time.perf_counter())
        started = time.perf_counter()
        for l in real:
            emit({"type": "learner_start", "learner": l.key, "label": l.label, "blurb": l.blurb, "seconds": round(give, 2)})

        def on_progress(l, e):
            emit({"type": "progress", "learner": l.key, **e})

        def on_done(l, payload):
            model, pv, pt, secs = payload
            finish(model, pv, pt, secs)

        def on_failed(l, msg):
            emit({"type": "learner_failed", "learner": l.key, "error": msg})
        race(real, (Xtr, ytr, Xva, yva, Xte, T), learn_until, stop, on_progress, on_done, on_failed,
             hard_deadline=deadline - 0.4 * reserve)           # never past the job's own deadline
        board.sort(key=lambda r: ORDER.index(r["learner"]) if r["learner"] in ORDER else 99)
        emit({"type": "stage", "text": f"{len([f for f in fitted if f[0].key != 'baseline'])} learners trained side by side in "
                                       f"{time.perf_counter() - started:.1f} s.", "t": round(time.perf_counter() - t_start, 3)})
    else:
        shares = dict(SHARES, linear=0.5) if Xtr.has_text else SHARES     # only the linear model reads text, so give it time
        for i, l in enumerate(real):
            if stop is not None and stop.is_set():
                break
            have_model = any(f[0].key != "baseline" for f in fitted)
            if have_model and time.perf_counter() >= learn_until:
                emit({"type": "learner_skipped", "learner": l.key, "label": l.label, "why": "out of time"})
                continue
            left = max(0.0, learn_until - time.perf_counter())
            rest = sum(shares.get(q.key, 0) for q in real[i:]) or 1
            give = max(0.02, left * shares.get(l.key, 0) / rest)
            emit({"type": "learner_start", "learner": l.key, "label": l.label, "blurb": l.blurb, "seconds": round(give, 2)})
            fit_here(l, time.perf_counter() + give)

    learned = [f for f in fitted if f[0].key != "baseline"] or fitted
    if not learned:
        raise DataError("Training was stopped before any model finished.")
    # choose on the validation rows: the single model or average of the best few whose predictions fit them best.
    # Numbers: RMSE. Categories: log loss on small validation sets (steadier than accuracy there); accuracy first
    # from 1,000 validation rows on, because some learners (word counts) are right more often than their
    # probabilities suggest.
    if T.kind != "classification":
        fit_loss = lambda m: m["rmse"]
    elif len(yva) >= 1000:
        fit_loss = lambda m: (-m["accuracy"], m["log_loss"])
    else:
        fit_loss = lambda m: m["log_loss"]
    learned.sort(key=lambda f: fit_loss(f[3]))
    options = [[learned[0]]] + [learned[:k] for k in (2, 3, len(learned)) if 2 <= k <= len(learned)]
    scored = []
    for group in options:
        pv = sum(f[1] for f in group) / len(group)
        scored.append((fit_loss(metrics(T, pv, yva, y_mu, y_sd)), len(group), group))
    _, _, choice = min(scored, key=lambda t: (t[0], t[1]))
    choice_m = metrics(T, sum(f[1] for f in choice) / len(choice), yva, y_mu, y_sd)
    weights = [1.0 / len(choice)] * len(choice)
    test_pred = sum(w * f[2] for w, f in zip(weights, choice))
    test_m = metrics(T, test_pred, yte, y_mu, y_sd)
    base = next((r for r in board if r["learner"] == "baseline"), None)
    model = Trained(T, target, labels, y_mu, y_sd, enc, [f[0] for f in choice], weights)
    name = " + ".join(f[0].label for f in choice)
    say(f"Winner: {name}. Measuring which columns it relies on…")
    groups = None
    media = "image" if image is True else (image or None)      # "image", "audio" or "video": files from a zip of folders
    if media:                      # importance per kind of feature (shape, colours, edges / tone, timbre / motion…)
        from .media import groups_for
        groups = groups_for(media)
    importance = permutation_importance(model, Xva, yva, enc, used, until=deadline - 0.02 * total_budget, groups=groups)
    total = time.perf_counter() - t_start
    model.report = {
        "target": target, "task": T.kind, "classes": labels, "rows": n, "file_rows": table.total_rows, "rows_exact": table.exact, "sampled": table.sampled,
        "split": {"train": len(tr), "valid": len(va), "test": len(te)},
        "features": Xtr.dense.shape[1], "text_columns": n_text, "columns_used": used, "ignored": [{"name": p["name"], "why": p.get("why", "ignored by you")} for p in prof
                                                                  if p["name"] != target and (p["name"] not in kinds or kinds[p["name"]] not in ("number", "category", "text", "date"))],
        "profile": prof, "leaderboard": board, "winner": [f[0].key for f in choice], "winner_label": name,
        "valid": choice_m, "test": test_m, "baseline_test": base["test"] if base else None, "importance": importance,
        "seconds_total": round(total, 3), "seconds_learning": round(time.perf_counter() - t_models, 3), "budget": round(total_budget, 3),
        "first_model_seconds": min((r["seconds"] for r in board if r["learner"] != "baseline"), default=None),
    }
    if not importance and Xva.n >= 5:
        model.report["importance_note"] = "no time"        # the time ran out before it could be measured
    if media:
        model.report["media"] = media
        if media == "image":
            model.report["image"] = {"groups": list(groups), "pictures": n}
        else:
            model.report[media] = {"groups": list(groups), "items": n}
        model.report["profile"] = []                     # hundreds of number columns: not worth keeping
        model.report["ignored"] = []
        model.report["columns_used"] = list(groups)
    emit({"type": "done", "report": model.report})
    return model


def permutation_importance(model: Trained, X: XData, y: np.ndarray, enc: Encoder, columns: list[str], seed: int = 0,
                           until: float | None = None, groups: dict[str, list[str]] | None = None) -> list[dict]:
    """How much the validation score drops when one column's values are shuffled between rows (stops at `until`)."""
    if X.n < 5:
        return []
    rng = np.random.default_rng(seed)
    T = model.task
    if X.n > 2000:
        X = X.take(np.arange(2000))
        y = y[:2000]
    base = model.predict_x(X)
    loss = (lambda p: log_loss(p, y)) if T.kind == "classification" else (lambda p: float(((p - y) ** 2).mean()))
    b = loss(base)
    slices = enc.column_slices()
    if groups:                     # shuffle whole groups of columns together
        slices = {g: [i for c in cols for i in slices.get(c, [])] for g, cols in groups.items()}
        columns = [g for g in groups if slices[g]]
    out = []
    for c in columns[:80]:
        if until is not None and time.perf_counter() > until:
            break
        perm = rng.permutation(X.n)
        if c in slices:
            d = X.dense.copy()
            d[:, slices[c]] = d[perm][:, slices[c]]
            X2 = XData(d, X.sp_indptr, X.sp_indices, X.sp_data)
        elif X.has_text and len(enc.spec) and any(s["col"] == c for s in enc.spec if s["kind"] == "text"):
            if sum(s["kind"] == "text" for s in enc.spec) > 1:
                continue      # text columns share one hashed space; they cannot be shuffled one at a time
            sh = X.take(perm)
            X2 = XData(X.dense, sh.sp_indptr, sh.sp_indices, sh.sp_data)
        else:
            continue
        out.append({"column": c, "drop": max(0.0, loss(model.predict_x(X2)) - b)})
    total = sum(o["drop"] for o in out) or 1.0
    for o in out:
        o["share"] = o["drop"] / total
    return sorted(out, key=lambda o: -o["drop"])


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
