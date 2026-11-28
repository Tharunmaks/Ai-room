"""LightGBM as an optional learner for tables: Microsoft's gradient-boosted trees, compiled C++ (pip install lightgbm).

It starts untrained like every other learner and races them within the same time. When it is installed it takes the
place of the NumPy boosted trees, which do the same job more slowly.

Threads and forking: LightGBM uses OpenMP. A process that forks after using OpenMP threads leaves its children unable
to use them (they hang). The Lab forks one process per learner, so in the main process LightGBM is only ever used
with one OpenMP thread; inside its own learner process it uses several.
"""

from __future__ import annotations

import contextlib
import math
import os
import threading
import time

import numpy as np

from .data import DataError
from .learners import F32, Learner, Task

IN_WORKER = False          # True inside a forked learner process (set by parallel._child)
_READY = threading.Event()  # set once `import lightgbm` has finished in this process
_STARTED = False


def available() -> str | None:
    """LightGBM's version if it is installed (without importing it: that takes up to a second), else None."""
    if os.environ.get("AI_ROOM_LAB_LIGHTGBM", "1") == "0":
        return None
    try:
        import importlib.metadata
        import importlib.util
        if importlib.util.find_spec("lightgbm") is None:
            return None
        return importlib.metadata.version("lightgbm")
    except Exception:
        return None


def warm(wait: bool = False) -> None:
    """Import LightGBM ahead of time (in the background unless `wait`), so no training job pays for the import."""
    global _STARTED
    if _STARTED or not available():
        if wait:
            _READY.wait(30)
        return
    _STARTED = True

    def load():
        try:
            import lightgbm  # noqa: F401
            _READY.set()
        except Exception:  # installed but broken (a missing system library, say): the NumPy trees are used instead
            pass
    if wait:
        load()
    else:
        threading.Thread(target=load, daemon=True, name="lab-lightgbm-import").start()


def ready() -> bool:
    """Installed and already imported: it can join a race without spending the race's time on the import."""
    return _READY.is_set()


def threads(rows: int | None = None) -> int:
    """OpenMP threads LightGBM may use in this process (0 = all cores).

    In its own learner process it shares the computer with the other learners: one thread below 50,000 rows (more
    would mostly wait on each other), half the cores above. Measured on 4 cores, three threads made it slower and
    less predictable than two."""
    from .parallel import can_fork
    if IN_WORKER:
        env = int(os.environ.get("AI_ROOM_LAB_LIGHTGBM_THREADS", 0))
        return env or (1 if rows is not None and rows < 50_000 else max(1, (os.cpu_count() or 2) // 2))
    return 1 if can_fork() else 0


@contextlib.contextmanager
def _safe():
    """One OpenMP thread in a process that will fork learner processes later (see the module notes)."""
    from .parallel import can_fork
    if IN_WORKER or not can_fork():
        yield
        return
    try:
        from threadpoolctl import threadpool_limits
        limit = threadpool_limits(1, user_api="openmp")
    except Exception:  # threadpoolctl missing: LightGBM still gets num_threads=1 below
        limit = contextlib.nullcontext()
    with limit:
        yield


def _lgb():
    try:
        import lightgbm
        return lightgbm
    except Exception as e:
        raise DataError("This model was trained with LightGBM. Install it to use the model: pip install lightgbm") from e


class LightGBM(Learner):
    key, label = "lgbm", "LightGBM"
    blurb = ("Gradient-boosted decision trees in compiled C++ (Microsoft's LightGBM): hundreds of small trees, each one "
             "fixing the mistakes of the trees before it. Usually the strongest on tables.")

    def __init__(self, seed: int = 0, lr: float | None = None, max_rounds: int = 50000, patience: int | None = None):
        self.seed, self.lr, self.max_rounds, self.patience = seed, lr, max_rounds, patience
        self.model_str: str | None = None
        self._bst = None

    def usable(self, X):
        return (ready() or IN_WORKER) and X.dense.shape[1] > 0

    # the trees travel between processes as text, and are loaded (with one thread) only where they are used
    def _text(self) -> str:
        if self.model_str is None:
            self.model_str = self._bst.model_to_string(num_iteration=self.rounds)
        return self.model_str

    def __getstate__(self):
        self._text()
        d = dict(self.__dict__)
        d["_bst"] = None
        return d

    def _booster(self):
        if self._bst is None:
            lgb = _lgb()
            with _safe():
                self._bst = lgb.Booster(model_str=self.model_str)
        return self._bst

    def fit(self, Xtr, ytr, Xva, yva, task, deadline, report, stop):
        lgb = _lgb()
        self.task = task
        n, d = Xtr.dense.shape
        K = task.classes if task.kind == "classification" and task.classes > 2 else 1
        objective = "regression" if task.kind == "regression" else "binary" if task.classes == 2 else "multiclass"
        metric = {"regression": "l2", "binary": "binary_logloss", "multiclass": "multi_logloss"}[objective]
        th = threads(n)
        params = {
            "objective": objective, "metric": metric, "learning_rate": self.lr or 0.1,
            "num_leaves": 63 if n >= 20000 else 31 if n >= 2000 else 15,
            "min_data_in_leaf": int(np.clip(n / 300, 3, 20)), "lambda_l2": 1.0,
            "feature_fraction": 0.9 if d <= 284 else max(0.05, 256 / d),       # wide data: a few hundred columns per tree
            "bagging_fraction": 0.8 if n >= 20000 else 1.0, "bagging_freq": 1 if n >= 20000 else 0,
            "verbose": -1, "seed": self.seed, "num_threads": th,
        }
        if K > 1:
            params["num_class"] = K
        ds_params = {"max_bin": 255 if n >= 5000 else 63, "verbose": -1, "num_threads": th}
        yv = yva.astype(np.float64)
        var = float(yv.var()) or 1.0

        def score(preds, data):                 # accuracy or R², for the live learning curve
            st["preds"] = preds
            if task.kind == "regression":
                return "score", 1.0 - float(((preds - yv) ** 2).mean()) / var, True
            guess = preds.reshape(len(yv), -1).argmax(1) if K > 1 else (preds > 0.5).astype(np.int64)
            return "score", float((guess == yva).mean()), True

        patience = self.patience or (50 if n < 5000 else 30)
        st = {"best": (math.inf, 0, None, None), "anchor": math.inf, "last": 0, "t_report": 0.0, "pv": None, "t0": time.perf_counter()}
        Xe = getattr(self, "extra_eval", None)       # the test rows: scored once, with the kept trees, at the end
        rows_after = Xe.n if Xe is not None else Xva.n + 1

        def watch(env):
            it = env.iteration + 1
            res = {name: val for _, name, val, _ in env.evaluation_result_list}
            vl, sc = res[metric], res["score"]
            if vl < st["best"][0]:
                st["best"] = (vl, it, sc, list(env.evaluation_result_list))
                st["pv"] = np.array(st["preds"], copy=True)      # its validation answers: no need to work them out again
            if vl < st["anchor"] * (1 - 2e-3):          # only real progress (0.2% better) resets the patience clock
                st["anchor"], st["last"] = vl, it
            now = time.perf_counter()
            if now - st["t_report"] > 0.25:
                st["t_report"] = now
                report(step=it, total=self.max_rounds, val=st["best"][2], note=f"{it} trees")
            # leave time to score the test rows with every tree kept: about what scoring the validation rows took
            # while learning, i.e. a share of each round that grows with the rows to score
            if it == 1:
                st["t1"] = now                          # rounds are timed from here: the first one includes the setup
            per_round = (now - st["t1"]) / (it - 1) if it > 1 else 0.0
            finish = per_round * (1 + it * rows_after / max(n, 1) * 0.15) + 0.02 * (deadline - st["t0"])
            if it - st["last"] > patience or now + finish > deadline or (stop is not None and stop.is_set()):
                raise lgb.callback.EarlyStopException(st["best"][1] - 1, st["best"][3])

        with _safe():
            train = lgb.Dataset(Xtr.dense, label=ytr, params=ds_params, free_raw_data=True)
            valid = lgb.Dataset(Xva.dense, label=yva, reference=train, params=ds_params, free_raw_data=True)
            bst = lgb.train(params, train, num_boost_round=self.max_rounds, valid_sets=[valid], valid_names=["valid"],
                            feval=score, callbacks=[watch])
            keep = max(1, st["best"][1] or bst.current_iteration())
        self.rounds, self._bst, self.model_str = keep, bst, None
        if Xe is not None and st["pv"] is not None:
            pv = st["pv"]
            pv = pv.astype(F32) if task.kind == "regression" else (np.stack([1 - pv, pv], 1) if pv.ndim == 1 else pv.reshape(len(yv), -1)).astype(F32)
            self.done_preds = (pv, self.predict(Xe))
        report(step=keep, total=keep, val=st["best"][2], note=f"kept the best {keep} trees")

    def predict(self, X):
        bst = self._booster()
        with _safe():
            p = bst.predict(X.dense, num_iteration=self.rounds, num_threads=threads())
        if self.task.kind == "regression":
            return p.astype(F32)
        if p.ndim == 1:
            return np.stack([1 - p, p], 1).astype(F32)
        return p.astype(F32)

    def state(self):
        return ({"task": vars(self.task), "rounds": self.rounds, "lightgbm": available()},
                {"model": np.frombuffer(self._text().encode(), np.uint8)})

    @classmethod
    def from_state(cls, meta, arrays):
        m = cls()
        m.task, m.rounds = Task(**meta["task"]), meta["rounds"]
        m.model_str = bytes(arrays["model"]).decode()
        return m
