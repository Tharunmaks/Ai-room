"""Models that start untrained and learn from your data, in NumPy on the CPU.

* `Baseline`  — always guesses the most common answer (or the average). Shows what "learned nothing" scores.
* `OneShot`   — a random-feature network whose output layer is solved exactly in one step (ridge regression).
                No epochs at all, so it is usually ready in well under a second.
* `Boosted`   — gradient-boosted decision trees on binned features (the strongest family for tables).
* `Neural`    — a neural network trained with AdamW, mini-batches and early stopping. With no hidden
                layers it is a linear / logistic model.

Every learner starts from random or zero weights; nothing is pre-trained.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np

from .data import XData

F32 = np.float32


@dataclass
class Task:
    kind: str          # "classification" | "regression"
    classes: int = 1   # number of classes (classification)

    @property
    def outputs(self) -> int:
        return self.classes if self.kind == "classification" else 1


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return max(0.0, float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()))


def quick_score(task: Task, pred: np.ndarray, y: np.ndarray) -> float:
    """Accuracy (classification) or R² on the standardized target (regression). Higher is better."""
    if task.kind == "classification":
        return float((pred.argmax(1) == y).mean())
    var = float(((y - y.mean()) ** 2).mean()) or 1.0
    return 1.0 - float(((pred - y) ** 2).mean()) / var


def quick_loss(task: Task, pred: np.ndarray, y: np.ndarray) -> float:
    return log_loss(pred, y) if task.kind == "classification" else float(((pred - y) ** 2).mean())


class Learner:
    key = "base"
    label = "Model"
    blurb = ""

    def fit(self, Xtr: XData, ytr, Xva: XData, yva, task: Task, deadline: float, report, stop) -> None:
        raise NotImplementedError

    def predict(self, X: XData) -> np.ndarray:
        """(n, classes) probabilities, or (n,) standardized values."""
        raise NotImplementedError

    def state(self) -> tuple[dict, dict]:
        raise NotImplementedError

    @classmethod
    def from_state(cls, meta: dict, arrays: dict) -> "Learner":
        raise NotImplementedError

    def usable(self, X: XData) -> bool:
        return True


def _stopped(deadline, stop) -> bool:
    return time.perf_counter() > deadline or (stop is not None and stop.is_set())


# ------------------------------------------------------------------ baseline
class Baseline(Learner):
    key, label = "baseline", "Baseline (no learning)"
    blurb = "Always answers with the most common class or the average value. Anything useful must beat this."

    def fit(self, Xtr, ytr, Xva, yva, task, deadline, report, stop):
        self.task = task
        if task.kind == "classification":
            counts = np.bincount(ytr, minlength=task.classes).astype(np.float64) + 1e-3
            self.prior = (counts / counts.sum()).astype(F32)
        else:
            self.prior = np.array([float(np.mean(ytr))], F32)

    def predict(self, X):
        if self.task.kind == "classification":
            return np.tile(self.prior, (X.n, 1))
        return np.full(X.n, self.prior[0], F32)

    def state(self):
        return {"task": vars(self.task)}, {"prior": self.prior}

    @classmethod
    def from_state(cls, meta, arrays):
        m = cls()
        m.task, m.prior = Task(**meta["task"]), arrays["prior"]
        return m


# ------------------------------------------------------------------ one-shot random-feature network
class OneShot(Learner):
    key, label = "oneshot", "One-shot net"
    blurb = ("A wide random hidden layer whose output weights are solved exactly in one step (ridge regression), "
             "so it learns without any training epochs.")

    def __init__(self, seed: int = 0, max_features: int = 2048):
        self.seed, self.max_features = seed, max_features

    def usable(self, X):
        return 0 < X.dense.shape[1] <= 2500

    def _features(self, X: np.ndarray) -> np.ndarray:
        h = X @ self.W
        h += self.b
        np.cos(h, out=h)
        h *= math.sqrt(2.0 / self.W.shape[1])
        return np.hstack([X, h, np.ones((len(X), 1), F32)])

    def _solve(self, X, T, Xv, yva, task, M, scale, rng, lams=15):
        """Draw a random hidden layer, solve the output layer exactly for a range of ridge strengths, keep the best on validation."""
        d = X.shape[1]
        W = (rng.standard_normal((d, M)) / (scale * math.sqrt(max(d, 1)))).astype(F32)
        b = rng.uniform(0, 2 * math.pi, M).astype(F32)
        self.W, self.b = W, b
        p = d + M + 1
        G = np.zeros((p, p), F32)
        ZtT = np.zeros((p, T.shape[1]), F32)
        for s in range(0, len(X), 8192):
            Z = self._features(X[s:s + 8192])
            G += Z.T @ Z
            ZtT += Z.T @ T[s:s + 8192]
        evals, Q = np.linalg.eigh(G.astype(np.float64))
        evals = np.maximum(evals, 0)
        A = Q.T @ ZtT.astype(np.float64)
        ZvQ = self._features(Xv) @ Q.astype(F32)
        ref = float(evals.sum()) / p or 1.0
        best = None
        for lam in ref * np.logspace(-6, 1, lams):
            coef = A / (evals + lam)[:, None]
            out = ZvQ @ coef.astype(F32)
            if task.kind == "classification":
                tgt = np.full_like(out, -1.0 / task.classes)
                tgt[np.arange(len(yva)), yva] += 1
                sc = float((out.argmax(1) == yva).mean()) - 1e-3 * float(((out - tgt) ** 2).mean())
            else:
                sc = quick_score(task, out[:, 0], yva)
            if best is None or sc > best[0]:
                best = (sc, coef, lam / ref)
        return best[0], W, b, (Q @ best[1]).astype(F32), best[2]

    def fit(self, Xtr, ytr, Xva, yva, task, deadline, report, stop):
        self.task = task
        X, Xv = Xtr.dense, Xva.dense
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        # as many hidden units as fit in about a third of a second of arithmetic, between 256 and max_features
        M = int(np.clip(min(n, math.sqrt(2e10 / max(n, 1))), 256, self.max_features))
        M = max(64, min(M, 3000 - d))
        left = max(0.01, deadline - time.perf_counter())
        while M > 64 and (n * (d + M) ** 2 + 4 * (d + M) ** 3) / 1.2e10 > 0.6 * left:     # arithmetic that fits the time
            M = int(M * 0.7)
        if task.kind == "classification":
            T = np.full((n, task.classes), -1.0 / task.classes, F32)
            T[np.arange(n), ytr] += 1.0
        else:
            T = ytr.astype(F32)[:, None]
        t0 = time.perf_counter()
        scales = (0.5, 1.0, 2.0)
        # choose the kernel width with a small version of the network (on at most 3,000 rows), then solve the full one once
        small = left < 0.6
        idx = rng.choice(n, 1000 if small else 3000, replace=False) if n > (1000 if small else 3000) else np.arange(n)
        vi = np.arange(min(len(yva), 1000 if small else 3000))
        trial = []
        for sc in scales:
            trial.append((self._solve(X[idx], T[idx], Xv[vi], yva[vi], task, min(M, 128 if small else 384), sc, rng, 8)[0], sc))
            if time.perf_counter() > t0 + 0.3 * left:
                break
        scale = max(trial)[1]
        report(step=1, total=2, val=max(trial)[0], note=f"chose kernel width ×{scale}")
        score, self.W, self.b, self.coef, self.lam = self._solve(X, T, Xv, yva, task, M, scale, rng)
        report(step=2, total=2, val=score, note=f"{M} random features, solved exactly")
        self.scale = scale
        self.temp = 1.0
        if task.kind == "classification":
            raw = self._features(Xv) @ self.coef
            losses = [(log_loss(softmax(raw * t), yva), t) for t in np.logspace(-1, 2.3, 30)]
            self.temp = float(min(losses)[1])
        self.seconds = time.perf_counter() - t0

    def predict(self, X):
        out = self._features(X.dense) @ self.coef
        return softmax(out * self.temp) if self.task.kind == "classification" else out[:, 0]

    def state(self):
        return {"task": vars(self.task), "temp": self.temp, "scale": self.scale, "lam": self.lam}, {"W": self.W, "b": self.b, "coef": self.coef}

    @classmethod
    def from_state(cls, meta, arrays):
        m = cls()
        m.task, m.temp = Task(**meta["task"]), meta["temp"]
        m.scale, m.lam = meta.get("scale"), meta.get("lam")
        m.W, m.b, m.coef = arrays["W"], arrays["b"], arrays["coef"]
        return m


# ------------------------------------------------------------------ gradient-boosted trees
_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max_workers=max(1, min(8, os.cpu_count() or 1)), thread_name_prefix="lab-hist")
    return _POOL


def feature_edges(x: np.ndarray, bins: int) -> np.ndarray:
    """Cut points for one feature. Very common values get a bin of their own; the other values share the rest
    of the bins by how many rows they cover. (Plain quantiles waste bins on columns that are mostly zero.)"""
    u, c = np.unique(x, return_counts=True)
    if len(u) <= bins:
        return ((u[:-1] + u[1:]) / 2).astype(F32)
    n = c.sum()
    heavy = c >= n / bins
    cuts = set()
    for i in np.flatnonzero(heavy):
        if i > 0:
            cuts.add(int(i))
        if i < len(u) - 1:
            cuts.add(int(i) + 1)
    left = max(0, bins - 1 - len(cuts))
    light = c[~heavy].sum()
    if left and light:
        bounds = np.flatnonzero(np.diff(np.concatenate([[1], heavy.astype(np.int8), [1]])))   # starts/ends of runs of light values
        runs = [(a, b) for a, b in zip(bounds[::2], bounds[1::2]) if b > a]
        for a, b in runs:
            k = int(round(left * c[a:b].sum() / light))
            if k <= 0 or b - a < 2:
                continue
            cum = np.cumsum(c[a:b])
            pos = np.searchsorted(cum, cum[-1] * np.arange(1, k + 1) / (k + 1)) + a + 1
            cuts.update(int(q) for q in pos if a < q < b)
    idx = np.array(sorted(cuts), np.int64)
    if len(idx) > bins - 1:
        idx = idx[np.linspace(0, len(idx) - 1, bins - 1).round().astype(np.int64)]
    return ((u[idx - 1] + u[idx]) / 2).astype(F32)


def make_bins(X: np.ndarray, bins: int, rng) -> list[np.ndarray]:
    sample = X if len(X) <= 50000 else X[rng.choice(len(X), 50000, replace=False)]
    return [feature_edges(sample[:, j], bins) for j in range(X.shape[1])]


def find_bundles(X: np.ndarray, max_size: int = 62) -> tuple[list[int], list[tuple[np.ndarray, np.ndarray]]]:
    """Group two-valued columns that are never 'on' in the same row (like one-hot categories) into one feature each.

    Trees then look at one column per group instead of one per category, which is much faster and loses nothing:
    the group's value says which member is on. Returns (plain columns, [(member columns, their 'off' values)]).
    """
    n, d = X.shape
    sample = X if n <= 20000 else X[:20000]
    binary, plain = [], []
    for j in range(d):
        u, c = np.unique(sample[:, j], return_counts=True)
        if len(u) == 2:
            off = u[np.argmax(c)]
            binary.append((int((sample[:, j] != off).sum()), j, off))
        else:
            plain.append(j)
    bundles: list[list] = []          # [mask, [cols], [offs]]
    for _, j, off in sorted(binary, key=lambda t: -t[0]):
        on = sample[:, j] != off
        for b in bundles:
            if len(b[1]) < max_size and not (b[0] & on).any():
                b[0] |= on; b[1].append(j); b[2].append(off)
                break
        else:
            bundles.append([on.copy(), [j], [off]])
    out = []
    for _, cols, offs in bundles:
        if len(cols) == 1:
            plain.append(cols[0])
        else:
            out.append((np.array(cols, np.int64), np.array(offs, F32)))
    return sorted(plain), out


def bin_matrix(X: np.ndarray, plain: list[int], edges: list[np.ndarray], bundles) -> np.ndarray:
    """(features, rows) uint8 bin codes: plain columns first, then one row per bundle."""
    out = np.empty((len(plain) + len(bundles), len(X)), np.uint8)
    for i, (j, e) in enumerate(zip(plain, edges)):
        out[i] = np.searchsorted(e, X[:, j], side="right")
    for i, (cols, offs) in enumerate(bundles):
        on = X[:, cols] != offs
        code = np.where(on.any(1), on.argmax(1) + 1, 0)
        out[len(plain) + i] = code
    return out


def _hist(loc, Xa, gT, hT, L, B, h_const):
    """Gradient, hessian and count histograms, shape (features, nodes, bins[, outputs]). One task per feature, run in threads."""
    df, m = Xa.shape
    K = gT.shape[0]
    size = L * B

    def one(j):
        idx = loc * B + Xa[j]
        c = np.bincount(idx, minlength=size)
        g = np.stack([np.bincount(idx, weights=gT[k], minlength=size) for k in range(K)], -1)
        h = np.repeat((c * h_const)[:, None], K, 1) if hT is None else np.stack([np.bincount(idx, weights=hT[k], minlength=size) for k in range(K)], -1)
        return c, g, h

    if m * df < 400000:     # small: a few big bincounts beat many tiny ones
        flat = ((np.arange(df)[:, None] * L + loc) * B + Xa).ravel()
        tot = df * size
        C = np.bincount(flat, minlength=tot).reshape(df, L, B).astype(np.float64)
        G = np.stack([np.bincount(flat, weights=np.tile(gT[k], df), minlength=tot) for k in range(K)], -1).reshape(df, L, B, K)
        if hT is None:
            H = np.repeat((C * h_const)[..., None], K, -1)
        else:
            H = np.stack([np.bincount(flat, weights=np.tile(hT[k], df), minlength=tot) for k in range(K)], -1).reshape(df, L, B, K)
        return G, H, C
    res = list(_pool().map(one, range(df))) if df > 1 else [one(0)]
    C = np.stack([r[0] for r in res]).reshape(df, L, B).astype(np.float64)
    G = np.stack([r[1] for r in res]).reshape(df, L, B, K)
    H = np.stack([r[2] for r in res]).reshape(df, L, B, K)
    return G, H, C


def grow_tree(Xb, g, h, depth, B, l2, min_leaf, h_const=None):
    """Level-wise tree on binned features Xb (features, rows). `h_const` uses one hessian value when choosing splits
    (exact sums are still used for the leaf values). Returns heap-ordered (feature, threshold, value) arrays and gains."""
    df, m = Xb.shape
    K = g.shape[1]
    gT, hT = np.ascontiguousarray(g.T), (None if h_const is not None else np.ascontiguousarray(h.T))
    n_nodes = 2 ** (depth + 1) - 1
    feat = np.full(n_nodes, -1, np.int32)
    thr = np.zeros(n_nodes, np.uint8)
    gains = np.zeros(df)
    node = np.zeros(m, np.int64)
    alive = np.arange(m)
    exists = np.ones(1, bool)
    prev = None
    for lev in range(depth):
        L, base = 2 ** lev, 2 ** lev - 1
        if not len(alive):
            break
        loc = node[alive] - base
        Xa = Xb[:, alive]
        if prev is None:
            HG, HH, HC = _hist(loc, Xa, gT[:, alive], None if hT is None else hT[:, alive], L, B, h_const)
        else:                                   # histogram subtraction: right child = parent − left child
            left = (loc & 1) == 0
            la = alive[left]
            lG, lH, lC = _hist(loc[left] >> 1, Xa[:, left], gT[:, la], None if hT is None else hT[:, la], L // 2, B, h_const)
            pG, pH, pC = prev
            HG = np.empty((df, L) + lG.shape[2:]); HG[:, 0::2], HG[:, 1::2] = lG, pG - lG
            HH = np.empty((df, L) + lH.shape[2:]); HH[:, 0::2], HH[:, 1::2] = lH, pH - lH
            HC = np.empty((df, L, B)); HC[:, 0::2], HC[:, 1::2] = lC, pC - lC
        GT, HT, CT = HG[0].sum(1), HH[0].sum(1), HC[0].sum(1)              # (L, K), (L, K), (L,)
        GL, HL, CL = np.cumsum(HG, 2)[:, :, :-1], np.cumsum(HH, 2)[:, :, :-1], np.cumsum(HC, 2)[:, :, :-1]
        GR, HR, CR = GT[:, None] - GL, HT[:, None] - HL, CT[:, None] - CL
        parent = (GT ** 2 / (HT + l2)).sum(-1)
        gain = (GL ** 2 / (HL + l2) + GR ** 2 / (np.maximum(HR, 0) + l2)).sum(-1) - parent[:, None]
        gain[(CL < min_leaf) | (CR < min_leaf)] = -np.inf
        flat = gain.transpose(1, 0, 2).reshape(L, -1)
        best = flat.argmax(1)
        bgain = flat[np.arange(L), best]
        bf, bt = np.divmod(best, B - 1)
        split = exists & (bgain > 1e-9)
        feat[base:base + L][split] = bf[split]
        thr[base:base + L][split] = bt[split]
        np.add.at(gains, bf[split], bgain[split])
        s_split = split[loc]
        right = Xa[bf[loc], np.arange(len(alive))] > bt[loc]
        node[alive] = np.where(s_split, 2 * node[alive] + 1 + right, node[alive])
        alive = alive[s_split]
        exists = np.repeat(split, 2)
        prev = (HG, HH, HC)
    # exact Newton values for every leaf, from the samples that ended there
    value = np.zeros((n_nodes, K), np.float64)
    for k in range(K):
        Gs = np.bincount(node, weights=g[:, k], minlength=n_nodes)
        Hs = np.bincount(node, weights=h[:, k], minlength=n_nodes)
        value[:, k] = -Gs / (Hs + l2)
    return feat, thr, value, gains


def tree_leaves(feat, thr, Xb, depth):
    """Leaf (heap index) for every row of binned data Xb (features, rows)."""
    n = Xb.shape[1]
    idx = np.zeros(n, np.int64)
    rows = np.arange(n)
    for _ in range(depth):
        f = feat[idx]
        inner = f >= 0
        if not inner.any():
            break
        right = Xb[np.maximum(f, 0), rows] > thr[idx]
        idx = np.where(inner, 2 * idx + 1 + right, idx)
    return idx


class Boosted(Learner):
    key, label = "boost", "Boosted trees"
    blurb = "Hundreds of small decision trees, each one fixing the mistakes of the trees before it. Usually the strongest on tables."

    def __init__(self, lr: float | None = None, depth: int | None = None, bins: int | None = None, l2: float = 1.0,
                 subsample: float | None = None, colsample: float = 0.9, max_rounds: int = 3000, patience: int | None = None, seed: int = 0):
        self.lr, self.depth, self.bins, self.l2 = lr, depth, bins, l2
        self.subsample, self.colsample, self.max_rounds, self.patience, self.seed = subsample, colsample, max_rounds, patience, seed

    def usable(self, X):
        return X.dense.shape[1] > 0

    def _gh(self, F, y):
        t = self.task
        if t.kind == "regression":
            return F - y[:, None], np.ones_like(F)
        if t.classes == 2:
            p = 1 / (1 + np.exp(-F[:, 0]))
            return (p - y)[:, None], np.maximum(p * (1 - p), 1e-6)[:, None]
        P = softmax(F)
        h = np.maximum(P * (1 - P), 1e-6)
        P[np.arange(len(y)), y] -= 1
        return P, h

    def _out(self, F):
        t = self.task
        if t.kind == "regression":
            return F[:, 0]
        if t.classes == 2:
            p = 1 / (1 + np.exp(-F[:, 0]))
            return np.stack([1 - p, p], 1).astype(F32)
        return softmax(F).astype(F32)

    def _bin(self, X: np.ndarray) -> np.ndarray:
        return bin_matrix(X, self.plain, self.edges, self.bundles)

    def fit(self, Xtr, ytr, Xva, yva, task, deadline, report, stop):
        self.task = task
        rng = np.random.default_rng(self.seed)
        n = Xtr.n
        depth = self.depth or (4 if n < 2000 else 6)
        lr = self.lr or 0.1
        patience = self.patience or (50 if n < 5000 else 30)
        min_leaf = int(np.clip(n / 300, 3, 20))
        self.plain, self.bundles = find_bundles(Xtr.dense)
        bins = self.bins or (127 if n >= 5000 else 63)
        self.edges = make_bins(Xtr.dense[:, self.plain], bins, rng) if self.plain else []
        Xb, Xvb = self._bin(Xtr.dense), self._bin(Xva.dense)
        d = Xb.shape[0]
        B = max([len(e) + 1 for e in self.edges] + [len(c) + 1 for c, _ in self.bundles])
        if B < 2:
            raise ValueError("every feature has a single value in the training rows, so there is nothing to split on")
        K = 1 if task.kind == "regression" or task.classes == 2 else task.classes
        if task.kind == "regression":
            init = np.array([float(np.mean(ytr))])
        elif K == 1:
            p = float(np.clip(np.mean(ytr), 1e-4, 1 - 1e-4))
            init = np.array([math.log(p / (1 - p))])
        else:
            pri = np.bincount(ytr, minlength=K) + 1.0
            init = np.log(pri / pri.sum())
        self.init = init.astype(F32)
        F = np.tile(init, (n, 1))
        Fv = np.tile(init, (len(yva), 1))
        trees_f, trees_t, trees_v = [], [], []
        self.gain = np.zeros(d)
        best, last_gain, anchor = (math.inf, 0, -math.inf), 0, math.inf
        sub = self.subsample or (1.0 if n < 2000 else 0.85 if n < 20000 else 0.6 if n < 100000 else 0.4)
        m = min(n, max(2, int(n * sub)), 80000)
        dk = max(1, int(round(d * self.colsample))) if d > 4 else d
        t_report = 0.0
        for r in range(self.max_rounds):
            if r and _stopped(deadline, stop):
                break
            g, h = self._gh(F, ytr)
            rows = np.sort(rng.choice(n, m, replace=False)) if m < n else slice(None)
            cols = np.sort(rng.choice(d, dk, replace=False)) if dk < d else np.arange(d)
            part = Xb[cols][:, rows] if m < n else Xb[cols]
            gs, hs = g[rows], h[rows]
            h_const = 1.0 if task.kind == "regression" else (float(hs.mean()) if K > 1 else None)
            feat, thr, val, gains = grow_tree(part, gs, hs, depth, B, self.l2, min_leaf, h_const)
            feat = np.where(feat >= 0, cols[np.maximum(feat, 0)], -1).astype(np.int32)
            self.gain[cols] += gains
            val = (val * lr).astype(F32)
            F += val[tree_leaves(feat, thr, Xb, depth)]
            Fv += val[tree_leaves(feat, thr, Xvb, depth)]
            trees_f.append(feat); trees_t.append(thr); trees_v.append(val)
            out = self._out(Fv)
            vl = quick_loss(task, out, yva)
            if vl < best[0]:
                best = (vl, r + 1, quick_score(task, out, yva))
            if vl < anchor * (1 - 2e-3):             # only real progress (0.2% better) resets the patience clock
                anchor, last_gain = vl, r + 1
            if r + 1 - last_gain > patience:
                break
            now = time.perf_counter()
            if now - t_report > 0.25:
                t_report = now
                report(step=r + 1, total=self.max_rounds, val=best[2], train_loss=float(np.mean(g ** 2)), note=f"{r + 1} trees")
        keep = max(1, best[1])
        self.depth_ = depth
        self.feat = np.stack(trees_f[:keep])
        self.thr = np.stack(trees_t[:keep])
        self.val = np.stack(trees_v[:keep])
        report(step=keep, total=keep, val=best[2], note=f"kept the best {keep} trees")

    def raw(self, X: XData) -> np.ndarray:
        Xb = self._bin(X.dense)
        F = np.tile(self.init.astype(np.float64), (X.n, 1))
        for f, t, v in zip(self.feat, self.thr, self.val):
            F += v[tree_leaves(f, t, Xb, self.depth_)]
        return F

    def predict(self, X):
        return self._out(self.raw(X))

    def state(self):
        arrays = {"feat": self.feat, "thr": self.thr, "val": self.val, "init": self.init, "gain": self.gain, "plain": np.array(self.plain, np.int64),
                  "edges": np.concatenate(self.edges) if self.edges else np.zeros(0, F32), "edge_len": np.array([len(e) for e in self.edges], np.int64),
                  "b_cols": np.concatenate([c for c, _ in self.bundles]) if self.bundles else np.zeros(0, np.int64),
                  "b_offs": np.concatenate([o for _, o in self.bundles]) if self.bundles else np.zeros(0, F32),
                  "b_len": np.array([len(c) for c, _ in self.bundles], np.int64)}
        return {"task": vars(self.task), "depth": self.depth_, "trees": int(len(self.feat))}, arrays

    @classmethod
    def from_state(cls, meta, arrays):
        m = cls()
        m.task, m.depth_ = Task(**meta["task"]), meta["depth"]
        m.feat, m.thr, m.val, m.init, m.gain = arrays["feat"], arrays["thr"], arrays["val"], arrays["init"], arrays["gain"]
        m.plain = [int(j) for j in arrays["plain"]]
        el = arrays["edge_len"]
        m.edges = np.split(arrays["edges"], np.cumsum(el)[:-1]) if len(el) else []
        bl = arrays["b_len"]
        if len(bl):
            cuts = np.cumsum(bl)[:-1]
            m.bundles = list(zip(np.split(arrays["b_cols"], cuts), np.split(arrays["b_offs"], cuts)))
        else:
            m.bundles = []
        return m


# ------------------------------------------------------------------ neural network (and linear model)
def segment_sum(values: np.ndarray, indptr: np.ndarray, rows: int) -> np.ndarray:
    """Sum of `values` rows within each CSR row segment (empty segments give zeros).

    Done as small matrix products with a 0/1 indicator per block of rows, which is several times faster
    in NumPy than np.add.reduceat along axis 0."""
    out = np.zeros((rows, values.shape[1]), np.float32)
    seg = np.repeat(np.arange(rows), np.diff(indptr))
    for s in range(0, rows, 64):
        e = min(rows, s + 64)
        lo, hi = indptr[s], indptr[e]
        if hi > lo:
            S = np.zeros((e - s, hi - lo), np.float32)
            S[seg[lo:hi] - s, np.arange(hi - lo)] = 1
            out[s:e] = S @ values[lo:hi]
    return out


class Neural(Learner):
    """A fully connected network, or with no hidden layers a linear model.

    The linear model also reads hashed text: its sparse first layer touches only the weights of the words present in
    a batch (like fastText). Networks with hidden layers use the number, category and date columns; text would make
    every step hundreds of times more expensive on a CPU for little gain over the linear model."""
    key, label = "mlp", "Neural network"
    blurb = "Layers of neurons trained from random weights with AdamW, mini-batches and early stopping."

    def __init__(self, hidden: tuple[int, ...] = (256, 256), key: str | None = None, label: str | None = None, blurb: str | None = None,
                 lr: float | None = None, dropout: float = 0.1, wd: float = 1e-4, target_steps: int = 3000, max_epochs: int = 400, seed: int = 0):
        self.hidden, self.lr, self.dropout, self.wd = tuple(hidden), lr, dropout, wd
        self.target_steps, self.max_epochs, self.seed = target_steps, max_epochs, seed
        if key:
            self.key = key
        if label:
            self.label = label
        if blurb:
            self.blurb = blurb
        self.Ws = None
        self.text_bits = 0

    # -- the network
    def _first(self, X: XData, Wd, Ws, b0):
        z = X.dense @ Wd if X.dense.shape[1] else np.zeros((X.n, b0.shape[0]), F32)
        z += b0
        if Ws is not None and X.has_text and len(X.sp_indices):
            cols = X.sp_indices & ((1 << self.text_bits) - 1)
            z += segment_sum(Ws[cols] * X.sp_data[:, None], X.sp_indptr, X.n)
        return z

    def _forward(self, X: XData, P, rng=None):
        Wd, Ws, W, b = P
        layers = len(W) + 1
        h = self._first(X, Wd, Ws, b[0])
        acts = [None]
        for i in range(1, layers):
            np.maximum(h, 0, out=h)
            if rng is not None and self.dropout:      # random bytes are much cheaper than random floats
                keep = np.frombuffer(rng.bytes(h.size), np.uint8).reshape(h.shape) >= int(round(256 * self.dropout))
                h *= keep
                h *= F32(1 / (1 - self.dropout))
            acts.append(h)
            h = h @ W[i - 1]
            h += b[i]
        return acts, h

    def _predict_with(self, X: XData, P):
        outs = [self._forward(X.take(np.arange(s, min(X.n, s + 4096))), P)[1] for s in range(0, X.n, 4096)]
        out = np.vstack(outs) if outs else np.zeros((0, self.task.outputs), F32)
        return softmax(out) if self.task.kind == "classification" else out[:, 0]

    def usable(self, X):
        return X.dense.shape[1] > 0 or (X.has_text and not self.hidden)

    def fit(self, Xtr, ytr, Xva, yva, task, deadline, report, stop):
        self.task = task
        rng = np.random.default_rng(self.seed)
        n, d = Xtr.n, Xtr.dense.shape[1]
        hidden = self.hidden
        sizes = [*hidden, task.outputs]
        H1 = sizes[0]
        text = Xtr.has_text and not hidden
        if text:             # hashed word space: as big as memory allows (weights + two optimizer copies)
            self.text_bits = int(min(18, max(12, math.floor(math.log2((1 << 24) / max(H1, 1))))))
        last = not hidden
        Wd = (rng.standard_normal((d, H1)) * (0.0 if last else math.sqrt(2.0 / max(d, 1)))).astype(F32)
        Ws = np.zeros((1 << self.text_bits, H1), F32) if text else None
        W = []
        for i, (a, c) in enumerate(zip(sizes[:-1], sizes[1:])):
            W.append((rng.standard_normal((a, c)) * (0.01 if i == len(sizes) - 2 else math.sqrt(2.0 / a))).astype(F32))
        b = [np.zeros(c, F32) for c in sizes]
        dense_params = [Wd] + W + b
        m1 = [np.zeros_like(p) for p in dense_params]
        m2 = [np.zeros_like(p) for p in dense_params]
        if Ws is not None:
            s1, s2 = np.zeros_like(Ws), np.zeros_like(Ws)
        batch = n if n <= 64 else int(np.clip(n // 16, 64, 512))
        spe = math.ceil(n / batch)
        epochs = int(np.clip(math.ceil(self.target_steps / spe), 10, self.max_epochs))
        lr0 = self.lr or (2e-3 if hidden else 5e-2 if text else 1e-2)
        patience = max(10, epochs // 6)
        yt = ytr if task.kind == "classification" else ytr.astype(F32)[:, None]
        snapshot = lambda: ([p.copy() for p in dense_params], None if Ws is None else Ws.copy())
        best = (math.inf, 0, -math.inf, snapshot())
        step, t0 = 0, time.perf_counter()
        b1, b2, eps = 0.9, 0.999, 1e-8
        ep = 0
        mask = (1 << self.text_bits) - 1
        while ep < epochs:
            if ep and _stopped(deadline, stop):
                break
            perm = rng.permutation(n)
            tl, tn = 0.0, 0
            for s in range(0, n, batch):
                if step and step % 16 == 0 and time.perf_counter() > deadline:
                    break                                   # out of time: keep what this epoch learned so far
                idx = perm[s:s + batch]
                xb, yb = Xtr.take(idx), yt[idx]
                acts, out = self._forward(xb, (Wd, Ws, W, b), rng)
                B = len(idx)
                if task.kind == "classification":
                    Pr = softmax(out)
                    tl += float(-np.log(np.clip(Pr[np.arange(B), yb], 1e-12, 1)).sum())
                    Pr[np.arange(B), yb] -= 1
                    dz = Pr / B
                else:
                    diff = out - yb
                    tl += float((diff ** 2).sum())
                    dz = 2 * diff / B
                tn += B
                gW, gb = [None] * len(W), [None] * len(b)
                for i in range(len(W), 0, -1):          # back through the dense layers
                    gW[i - 1] = acts[i].T @ dz
                    gb[i] = dz.sum(0)
                    dz = dz @ W[i - 1].T
                    dz *= (acts[i] > 0) * F32(1 / (1 - self.dropout) if self.dropout else 1)
                gb[0] = dz.sum(0)
                gWd = xb.dense.T @ dz
                step += 1
                total_steps = epochs * spe
                warm = max(10, total_steps // 50)
                frac = min(1.0, step / total_steps)
                lr = lr0 * min(1.0, step / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
                c1, c2 = 1 - b1 ** step, 1 - b2 ** step
                for p, g, a, v in zip(dense_params, [gWd] + gW + gb, m1, m2):
                    a *= b1; a += (1 - b1) * g
                    v *= b2; v += (1 - b2) * (g * g)
                    if p.ndim == 2 and self.wd:
                        p *= F32(1 - lr * self.wd)
                    p -= F32(lr / c1) * a / (np.sqrt(v / c2) + eps)
                if Ws is not None and len(xb.sp_indices):   # lazy Adam on the word rows this batch touched
                    rows = np.repeat(np.arange(B), np.diff(xb.sp_indptr))
                    uniq, inv = np.unique(xb.sp_indices & mask, return_inverse=True)
                    contrib = dz[rows] * xb.sp_data[:, None]
                    g = np.stack([np.bincount(inv, weights=contrib[:, k], minlength=len(uniq)) for k in range(H1)], 1).astype(F32)
                    a = s1[uniq] * b1 + (1 - b1) * g
                    v = s2[uniq] * b2 + (1 - b2) * (g * g)
                    s1[uniq], s2[uniq] = a, v
                    Ws[uniq] -= F32(lr / c1) * a / (np.sqrt(v / c2) + eps)
            ep += 1
            if ep == 1:      # now we know how long an epoch takes: fit the plan into the time left
                per = time.perf_counter() - t0
                left = deadline - time.perf_counter()
                epochs = int(max(1, min(epochs, 1 + left / max(per, 1e-6))))
            pv = self._predict_with(Xva, (Wd, Ws, W, b))
            vl = quick_loss(task, pv, yva)
            if vl < best[0] - 1e-6:
                best = (vl, ep, quick_score(task, pv, yva), snapshot())
            report(step=ep, total=epochs, val=best[2], train_loss=tl / max(tn, 1), note=f"epoch {ep}")
            if ep - best[1] > patience:
                break
        params, self.Ws = best[3]
        k = len(W)
        self.Wd, self.W, self.b = params[0], params[1:1 + k], params[1 + k:]
        self.hidden_ = list(hidden)
        self.epochs = best[1]

    def predict(self, X):
        return self._predict_with(X, (self.Wd, self.Ws, self.W, self.b))

    def state(self):
        arrays = {"Wd": self.Wd} | {f"W{i}": w for i, w in enumerate(self.W)} | {f"b{i}": v for i, v in enumerate(self.b)}
        if self.Ws is not None:
            arrays["Ws"] = self.Ws
        return {"task": vars(self.task), "hidden": self.hidden_, "layers": len(self.W), "biases": len(self.b), "text_bits": self.text_bits,
                "key": self.key, "label": self.label}, arrays

    @classmethod
    def from_state(cls, meta, arrays):
        m = cls(hidden=tuple(meta["hidden"]), key=meta.get("key"), label=meta.get("label"))
        m.task, m.hidden_, m.text_bits = Task(**meta["task"]), meta["hidden"], meta["text_bits"]
        m.Wd, m.Ws = arrays["Wd"], arrays.get("Ws")
        m.W = [arrays[f"W{i}"] for i in range(meta["layers"])]
        m.b = [arrays[f"b{i}"] for i in range(meta["biases"])]
        return m


def linear() -> Neural:
    return Neural(hidden=(), key="linear", label="Linear model", dropout=0.0, wd=1e-5, target_steps=1500,
                  blurb="One weight per feature (logistic or linear regression). Fast and hard to fool, but it can only draw straight lines.")


KINDS = {"baseline": Baseline, "oneshot": OneShot, "boost": Boosted, "mlp": Neural, "linear": Neural}
