"""Run the learners at the same time, one process each.

Learners mostly run small NumPy steps in Python loops, so in threads they would queue behind Python's global lock.
Forked processes share the prepared data without copying it (copy-on-write), use one core each, and send back
their trained model and predictions. Each process is limited to one BLAS thread so four of them don't fight
over the cores. Where fork is not available (Windows, macOS) or there is only one learner, they run in turn.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import traceback
from multiprocessing.connection import wait

try:                               # imported here, not in a learner process: an import running in another thread when
    import threadpoolctl           # a process forks would leave that module locked in the new process
except ImportError:
    threadpoolctl = None


def can_fork() -> bool:
    return sys.platform.startswith("linux") and (os.cpu_count() or 1) > 1 and os.environ.get("AI_ROOM_LAB_PARALLEL", "1") != "0"


def _plain_signals() -> None:
    """A forked worker inherits the parent's signal handlers: under the web server, SIGTERM would only ask the (copied)
    server to shut down, and the worker would keep running. Workers get the default handlers back."""
    import signal
    for sig in (getattr(signal, n, None) for n in ("SIGTERM", "SIGINT", "SIGHUP")):
        if sig is None:
            continue
        try:
            signal.signal(sig, signal.SIG_DFL)
        except (OSError, ValueError):
            pass


def _end(p) -> None:
    """Stop a worker for sure."""
    p.terminate()
    p.join(0.5)
    if p.is_alive():
        p.kill()
        p.join(1)


def _limit_threads() -> None:
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:  # threadpoolctl missing or a BLAS it does not know: run with what there is
        pass


def _child(learner, data, deadline, stop, conn) -> None:
    _plain_signals()
    from . import learners as L, lgbm
    lgbm.IN_WORKER = True          # LightGBM may use several threads here (never in the parent, which forks)
    L._POOL = None                 # the parent's histogram threads do not exist in this process
    L.HIST_THREADS = 1             # one core per learner: no extra threads
    _limit_threads()
    Xtr, ytr, Xva, yva, Xte, task = data
    t0 = time.perf_counter()

    def report(step, total, val=None, train_loss=None, note=None):
        conn.send(("progress", {"step": step, "total": total, "val": val, "train_loss": train_loss, "note": note,
                                "t": round(time.perf_counter() - t0, 3)}))
    try:
        learner.extra_eval = Xte
        learner.fit(Xtr, ytr, Xva, yva, task, deadline, report, stop)
        secs = time.perf_counter() - t0
        pv, pt = finished_predictions(learner, Xva, Xte)
        conn.send(("done", (learner, pv, pt, secs)))
    except Exception as e:  # report in plain words; the parent carries on with the other learners
        conn.send(("failed", f"{type(e).__name__}: {e}" if str(e) else type(e).__name__))
        traceback.print_exc()
    finally:
        conn.close()


def finished_predictions(learner, Xva, Xte):
    """The learner's answers for the validation and test rows: kept from fitting when it scored them as it went,
    else worked out now. The test rows are let go of, so they are not sent back with the learner."""
    learner.extra_eval = None
    done = learner.__dict__.pop("done_preds", None)
    if done is not None:
        return done
    return learner.predict(Xva), learner.predict(Xte)


def _map_child(fn, args, conn) -> None:
    _plain_signals()
    _limit_threads()
    try:
        conn.send(("ok", fn(*args)))
    except Exception as e:  # sent back and raised in the parent
        conn.send(("err", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def fork_map(fn, jobs: list[tuple]) -> list:
    """[fn(*job) for job in jobs], one forked process per job (fn may be a closure: nothing is pickled but results)."""
    if not can_fork() or len(jobs) < 2:
        return [fn(*j) for j in jobs]
    ctx = mp.get_context("fork")
    import warnings
    conns = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for j in jobs:
            recv, send = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_map_child, args=(fn, j, send), daemon=True)
            p.start()
            send.close()
            conns.append((recv, p))
    out = []
    for recv, p in conns:
        try:
            kind, val = recv.recv()
        except EOFError:
            kind, val = "err", "a worker process ended unexpectedly"
        p.join()
        if kind == "err":
            raise RuntimeError(val)
        out.append(val)
    return out


def race(pool, data, deadline: float, stop, on_progress, on_done, on_failed, hard_deadline: float | None = None) -> None:
    """Fit every learner in `pool` at once, each until `deadline`. Callbacks run in the calling thread. A learner
    still busy at `hard_deadline` (default: 30 s after `deadline`) is stopped and counts as failed."""
    ctx = mp.get_context("fork")
    mp_stop = ctx.Event()
    running = {}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)     # "fork with threads": the children only run NumPy code
        for l in pool:
            recv, send = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_child, args=(l, data, deadline, mp_stop, send), daemon=True, name=f"lab-{l.key}")
            p.start()
            send.close()
            running[recv] = (l, p)
    hard_stop = deadline + 30.0 if hard_deadline is None else hard_deadline   # a learner that ignores its deadline is stopped
    while running:
        if stop is not None and stop.is_set():
            mp_stop.set()
        ready = wait(list(running), timeout=max(0.005, min(0.2, hard_stop - time.perf_counter())))
        for conn in ready:
            l, p = running[conn]
            try:
                kind, payload = conn.recv()
            except (EOFError, OSError):
                del running[conn]
                p.join(1)
                on_failed(l, "the learner's process ended unexpectedly (out of memory?)")
                continue
            if kind == "progress":
                on_progress(l, payload)
            else:
                del running[conn]
                p.join(5)
                (on_done if kind == "done" else on_failed)(l, payload)
        if time.perf_counter() > hard_stop:
            for conn, (l, p) in list(running.items()):
                _end(p)
                on_failed(l, "did not stop in time")
            running.clear()
