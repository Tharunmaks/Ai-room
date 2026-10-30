"""Train and use small machine-learning models locally in pure Python, plus classic algorithms and optimisers.

These train for real (the weights change) but only on small data — they are for learning, prototyping and automation.
For big GPU training use the Colab / Kaggle notebook generators. Models are saved as JSON in `models/` in the workspace.
"""

from __future__ import annotations

import csv
import io
import json
import math
import random
import re
from collections import Counter, defaultdict

from .registry import Context, ToolError, reg
from .t_math import evaluate

ML = "machine_learning"
FL = dict(ctx=True, risk="files")


# ------------------------------------------------------------------ data helpers
def table(ctx: Context, csv_path: str, target: str):
    p = ctx.path(csv_path, True)
    rows = list(csv.DictReader(io.StringIO(p.read_text(encoding="utf-8", errors="replace"))))
    if not rows or target not in rows[0]:
        raise ToolError(f"column '{target}' not found. Columns: {list(rows[0]) if rows else 'none (empty file)'}")
    feats = [c for c in rows[0] if c != target]
    X, y = [], []
    for r in rows:
        try:
            X.append([float(r[c]) for c in feats])
        except ValueError:
            raise ToolError("all feature columns must be numbers (encode text columns first, e.g. with ml_one_hot)")
        y.append(r[target])
    return X, y, feats


def labels_or_numbers(y):
    try:
        return [float(v) for v in y], True
    except (TypeError, ValueError):
        return list(y), False


def xy(ctx, X, y, csv_path, target):
    if csv_path:
        X, y, feats = table(ctx, csv_path, target or "target")
        return X, y, feats
    if X is None or y is None:
        raise ToolError("give X and y (lists), or csv and target (a CSV file and its label column)")
    if not X or len(X) != len(y):
        raise ToolError("X and y must be non-empty and the same length")
    X = [[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])] for r in X]
    return X, list(y), [f"x{i}" for i in range(len(X[0]))]


def save_model(ctx, name, model):
    p = ctx.path(f"models/{name}.json")
    p.write_text(json.dumps(model, ensure_ascii=False))
    ctx.produced(p, "model", model["type"])
    return f"models/{name}.json"


def load_model(ctx, name):
    p = ctx.path(name if name.endswith(".json") else f"models/{name}.json", True)
    return json.loads(p.read_text())


def solve(A, b):
    n = len(A)
    M = [row[:] + [bi] for row, bi in zip(A, b)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[p][i]) < 1e-12:
            raise ToolError("features are linearly dependent (or constant) — remove a column")
        M[i], M[p] = M[p], M[i]
        for r in range(n):
            if r != i:
                f = M[r][i] / M[i][i]
                M[r] = [a - f * c for a, c in zip(M[r], M[i])]
    return [M[i][n] / M[i][i] for i in range(n)]


def mean_std(X):
    cols = list(zip(*X))
    mu = [sum(c) / len(c) for c in cols]
    sd = [math.sqrt(sum((v - m) ** 2 for v in c) / len(c)) or 1.0 for c, m in zip(cols, mu)]
    return mu, sd


def scale(X, mu, sd):
    return [[(v - m) / s for v, m, s in zip(r, mu, sd)] for r in X]


def sigmoid(z):
    return 1 / (1 + math.exp(-max(-60, min(60, z))))


# ------------------------------------------------------------------ models
def fit_linear(X, y, ridge=0.0):
    Xb = [[1.0] + r for r in X]
    k = len(Xb[0])
    A = [[sum(r[i] * r[j] for r in Xb) + (ridge if i == j and i > 0 else 0) for j in range(k)] for i in range(k)]
    b = [sum(r[i] * t for r, t in zip(Xb, y)) for i in range(k)]
    return solve(A, b)


def fit_logistic(X, y, lr=0.3, epochs=800, l2=0.0):
    mu, sd = mean_std(X)
    Z = scale(X, mu, sd)
    classes = sorted(set(y))
    if len(classes) != 2:
        raise ToolError(f"logistic regression needs exactly 2 classes, found {len(classes)}: {classes[:6]}")
    t = [1.0 if v == classes[1] else 0.0 for v in y]
    w, b = [0.0] * len(Z[0]), 0.0
    for _ in range(epochs):
        gw, gb = [0.0] * len(w), 0.0
        for r, ti in zip(Z, t):
            e = sigmoid(sum(a * c for a, c in zip(w, r)) + b) - ti
            gw = [g + e * c for g, c in zip(gw, r)]
            gb += e
        w = [a - lr * (g / len(Z) + l2 * a) for a, g in zip(w, gw)]
        b -= lr * gb / len(Z)
    return {"w": w, "b": b, "mu": mu, "sd": sd, "classes": classes}


def fit_tree(X, y, depth, min_leaf=2):
    def gini(ys):
        c = Counter(ys)
        return 1 - sum((v / len(ys)) ** 2 for v in c.values())

    def build(idx, d):
        ys = [y[i] for i in idx]
        c = Counter(ys)
        if d == 0 or len(c) == 1 or len(idx) < 2 * min_leaf:
            return {"leaf": c.most_common(1)[0][0]}
        best = None
        for f in range(len(X[0])):
            vals = sorted({X[i][f] for i in idx})
            for a, b in zip(vals, vals[1:]):
                th = (a + b) / 2
                L = [i for i in idx if X[i][f] <= th]
                R = [i for i in idx if X[i][f] > th]
                if len(L) < min_leaf or len(R) < min_leaf:
                    continue
                g = (len(L) * gini([y[i] for i in L]) + len(R) * gini([y[i] for i in R])) / len(idx)
                if best is None or g < best[0]:
                    best = (g, f, th, L, R)
        if best is None:
            return {"leaf": c.most_common(1)[0][0]}
        _, f, th, L, R = best
        return {"f": f, "t": th, "l": build(L, d - 1), "r": build(R, d - 1)}
    return build(list(range(len(X))), depth)


def tree_predict(node, x):
    while "leaf" not in node:
        node = node["l"] if x[node["f"]] <= node["t"] else node["r"]
    return node["leaf"]


def fit_kmeans(X, k, iters=100, seed=0):
    r = random.Random(seed)
    cent = [X[i][:] for i in r.sample(range(len(X)), k)]
    for _ in range(iters):
        groups = [[] for _ in range(k)]
        for x in X:
            groups[min(range(k), key=lambda j: sum((a - b) ** 2 for a, b in zip(x, cent[j])))].append(x)
        new = [[sum(c) / len(g) for c in zip(*g)] if g else cent[j] for j, g in enumerate(groups)]
        if new == cent:
            break
        cent = new
    return cent


def fit_mlp(X, y, hidden=8, lr=0.1, epochs=2000, seed=0, classify=True):
    r = random.Random(seed)
    mu, sd = mean_std(X)
    Z = scale(X, mu, sd)
    classes = sorted(set(y)) if classify else None
    if classify and len(classes) != 2:
        raise ToolError("this small network handles 2 classes (use train_decision_tree or train_knn for more)")
    t = [1.0 if v == classes[1] else 0.0 for v in y] if classify else [float(v) for v in y]
    ym, ys = (0.0, 1.0) if classify else (sum(t) / len(t), (math.sqrt(sum((v - sum(t) / len(t)) ** 2 for v in t) / len(t)) or 1.0))
    t = [(v - ym) / ys for v in t]
    n = len(Z[0])
    W1 = [[r.gauss(0, .7) for _ in range(n)] for _ in range(hidden)]
    b1 = [0.0] * hidden
    W2 = [r.gauss(0, .7) for _ in range(hidden)]
    b2 = 0.0
    for _ in range(epochs):
        for xi, ti in zip(Z, t):
            h = [math.tanh(sum(a * c for a, c in zip(W1[j], xi)) + b1[j]) for j in range(hidden)]
            o = sum(a * c for a, c in zip(W2, h)) + b2
            p = sigmoid(o) if classify else o
            d = p - ti
            for j in range(hidden):
                dh = d * W2[j] * (1 - h[j] ** 2)
                W2[j] -= lr * d * h[j]
                W1[j] = [a - lr * dh * c for a, c in zip(W1[j], xi)]
                b1[j] -= lr * dh
            b2 -= lr * d
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "mu": mu, "sd": sd, "classes": classes, "ym": ym, "ys": ys}


def predict_one(m, x):
    t = m["type"]
    if t == "linear":
        return m["w"][0] + sum(a * b for a, b in zip(m["w"][1:], x))
    if t == "polynomial":
        v = x[0]
        return sum(c * v ** i for i, c in enumerate(m["w"]))
    if t == "logistic":
        z = [(a - mu) / s for a, mu, s in zip(x, m["mu"], m["sd"])]
        p = sigmoid(sum(a * b for a, b in zip(m["w"], z)) + m["b"])
        return m["classes"][1] if p >= .5 else m["classes"][0]
    if t == "knn":
        mu, sd = m["mu"], m["sd"]
        z = [(a - u) / s for a, u, s in zip(x, mu, sd)]
        near = sorted(zip(m["X"], m["y"]), key=lambda p: sum((a - (b - u) / s) ** 2 for a, b, u, s in zip(z, p[0], mu, sd)))[:m["k"]]
        if m["numeric"]:
            return sum(float(y) for _, y in near) / len(near)
        return Counter(y for _, y in near).most_common(1)[0][0]
    if t == "tree":
        return tree_predict(m["tree"], x)
    if t == "kmeans":
        return min(range(len(m["centroids"])), key=lambda j: sum((a - b) ** 2 for a, b in zip(x, m["centroids"][j])))
    if t == "perceptron":
        return m["classes"][1] if sum(a * b for a, b in zip(m["w"], x)) + m["b"] > 0 else m["classes"][0]
    if t == "mlp":
        z = [(a - u) / s for a, u, s in zip(x, m["mu"], m["sd"])]
        h = [math.tanh(sum(a * c for a, c in zip(m["W1"][j], z)) + m["b1"][j]) for j in range(len(m["W2"]))]
        o = sum(a * c for a, c in zip(m["W2"], h)) + m["b2"]
        if m["classes"]:
            return m["classes"][1] if sigmoid(o) >= .5 else m["classes"][0]
        return o * m["ys"] + m["ym"]
    raise ToolError(f"cannot predict with model type '{t}'")


def metrics(true, pred, regression=None):
    """Regression scores (R², RMSE…) when the model predicts numbers; accuracy and a confusion table when it predicts labels."""
    n = len(true)
    try:
        t = [float(v) for v in true]
        pr = [float(v) for v in pred]
        if regression or (regression is None and (len(set(t)) > 10 or any(v != int(v) for v in t))):
            mean = sum(t) / n
            ss = sum((a - mean) ** 2 for a in t) or 1
            return {"n": n, "mse": round(sum((a - b) ** 2 for a, b in zip(t, pr)) / n, 6), "rmse": round(math.sqrt(sum((a - b) ** 2 for a, b in zip(t, pr)) / n), 6), "mae": round(sum(abs(a - b) for a, b in zip(t, pr)) / n, 6), "r2": round(1 - sum((a - b) ** 2 for a, b in zip(t, pr)) / ss, 6)}
    except (TypeError, ValueError):
        pass
    ok = sum(str(a) == str(b) for a, b in zip(true, pred))
    return {"n": n, "accuracy": round(ok / n, 4), "confusion": {f"{a}→{b}": c for (a, b), c in Counter((str(a), str(b)) for a, b in zip(true, pred)).items()}}


def split(X, y, test, seed):
    idx = list(range(len(X)))
    random.Random(seed).shuffle(idx)
    k = max(1, int(len(X) * test))
    te, tr = idx[:k], idx[k:]
    return [X[i] for i in tr], [y[i] for i in tr], [X[i] for i in te], [y[i] for i in te]


def train(ctx, kind, name, X, y, csv_path, target, **kw):
    X, y, feats = xy(ctx, X, y, csv_path, target)
    Xtr, ytr, Xte, yte = split(X, y, kw.pop("test_fraction", 0.2), 7) if len(X) >= 10 else (X, y, X, y)
    numeric = kind in ("linear", "polynomial", "mlp_reg") or (kind == "knn" and labels_or_numbers(y)[1] and len(set(y)) > 10)
    if kind == "linear":
        yn = [float(v) for v in ytr]
        m = {"type": "linear", "w": fit_linear(Xtr, yn, kw.get("ridge", 0.0))}
    elif kind == "polynomial":
        deg = kw.get("degree", 2)
        if len(Xtr[0]) != 1:
            raise ToolError("polynomial regression uses a single feature column")
        m = {"type": "polynomial", "w": fit_linear([[r[0] ** i for i in range(1, deg + 1)] for r in Xtr], [float(v) for v in ytr])}
    elif kind == "logistic":
        m = {"type": "logistic", **fit_logistic(Xtr, ytr, epochs=kw.get("epochs", 800))}
    elif kind == "knn":
        mu, sd = mean_std(Xtr)
        m = {"type": "knn", "k": kw.get("k", 3), "X": Xtr, "y": ytr, "mu": mu, "sd": sd, "numeric": numeric}
    elif kind == "tree":
        m = {"type": "tree", "tree": fit_tree(Xtr, ytr, kw.get("depth", 4))}
    elif kind == "kmeans":
        m = {"type": "kmeans", "centroids": fit_kmeans(Xtr, kw.get("k", 3))}
    elif kind == "perceptron":
        cl = sorted(set(ytr))
        if len(cl) != 2:
            raise ToolError("perceptron needs exactly 2 classes")
        w, b = [0.0] * len(Xtr[0]), 0.0
        for _ in range(kw.get("epochs", 200)):
            for xi, yi in zip(Xtr, ytr):
                t = 1 if yi == cl[1] else -1
                if t * (sum(a * c for a, c in zip(w, xi)) + b) <= 0:
                    w = [a + t * c for a, c in zip(w, xi)]
                    b += t
        m = {"type": "perceptron", "w": w, "b": b, "classes": cl}
    elif kind in ("mlp", "mlp_reg"):
        m = {"type": "mlp", **fit_mlp(Xtr, ytr, kw.get("hidden", 8), epochs=kw.get("epochs", 1500), classify=kind == "mlp")}
    else:
        raise ToolError(f"unknown model kind {kind}")
    m["features"] = feats
    if kind != "kmeans":
        pred = [predict_one(m, r) for r in Xte]
        score = metrics(yte, pred, regression=numeric)
    else:
        score = {"inertia": round(sum(min(sum((a - b) ** 2 for a, b in zip(r, c)) for c in m["centroids"]) for r in X), 4)}
    path = save_model(ctx, name, m)
    return {"model": path, "trained_on": len(Xtr), "held_out_test": len(Xte) if len(X) >= 10 else "none (too little data; score is on training data)", "score": score}


_DATA = "X:list?, y:list?, csv:str?, target:str?"
for _k, _d, _extra in (
    ("linear", "Train a linear regression (least squares, optional ridge).", ", ridge:float=0"), ("polynomial", "Train a polynomial regression on one feature.", ", degree:int=2"),
    ("logistic", "Train a logistic-regression classifier (2 classes).", ", epochs:int=800"), ("knn", "Train a k-nearest-neighbours classifier or regressor.", ", k:int=3"),
    ("tree", "Train a decision-tree classifier.", ", depth:int=4"), ("perceptron", "Train a perceptron (2 classes).", ", epochs:int=200"),
    ("mlp", "Train a small neural network (1 hidden layer) for 2-class classification.", ", hidden:int=8, epochs:int=1500"), ("mlp_reg", "Train a small neural network for regression.", ", hidden:int=8, epochs:int=1500"),
):
    reg(f"train_{_k}", f"{_d} Give X and y lists, or a CSV file and its label column. Saves models/<name>.json and reports a held-out score.", ML, f"{_DATA}, name:str=model_{_k}{_extra}",
        (lambda k: lambda ctx, X, y, csv, target, name, **kw: train(ctx, k, name, X, y, csv, target, **kw))(_k), tags=("training", "fit", "learn"), produces="model", **FL)
reg("train_kmeans", "Cluster data into k groups (k-means). Give X (rows) or a CSV (all columns used; give target as '' ).", ML, "X:list?, csv:str?, k:int=3, name:str=model_kmeans",
    lambda ctx, X, csv, k, name: (lambda rows: train(ctx, "kmeans", name, rows[0], [0] * len(rows[0]), None, None, k=k))(_unlabelled(ctx, X, csv)), tags=("clustering", "training"), produces="model", **FL)


def _unlabelled(ctx, X, csv_path):
    if csv_path:
        rows = list(csv.DictReader(io.StringIO(ctx.path(csv_path, True).read_text())))
        return ([[float(v) for v in r.values()] for r in rows],)
    if not X:
        raise ToolError("give X or csv")
    return ([[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])] for r in X],)


reg("ml_predict", "Predict with a saved model: rows is a list of feature lists (or a single list).", ML, "model:str, rows:list", lambda ctx, model, rows: (lambda m, rs: [predict_one(m, [float(v) for v in r]) for r in rs])(load_model(ctx, model), rows if rows and isinstance(rows[0], (list, tuple)) else [rows]), tags=("inference", "use"), **FL)
reg("ml_model_info", "Describe a saved model (type, features, size).", ML, "model:str", lambda ctx, model: (lambda m: {"type": m["type"], "features": m.get("features"), **{k: (v if not isinstance(v, list) or len(v) < 12 else f"[{len(v)} values]") for k, v in m.items() if k not in ("features", "type", "X", "y", "tree")}})(load_model(ctx, model)), **FL)
reg("ml_list_models", "Models saved in the workspace.", ML, "", lambda ctx: sorted(p.stem for p in (ctx.workspace / "models").glob("*.json")) if (ctx.workspace / "models").exists() else [], **FL)
reg("ml_evaluate", "Score predictions against the truth (accuracy and confusion matrix, or R²/RMSE for numbers).", ML, "true:list, predicted:list", lambda true, predicted: metrics(true, predicted))
reg("ml_train_test_split", "Split a CSV into train.csv and test.csv in the workspace.", ML, "csv:str, test_fraction:float=0.2, seed:int=1", lambda ctx, csv, test_fraction, seed: _csv_split(ctx, csv, test_fraction, seed), **FL)
reg("ml_cross_validate", "k-fold cross-validation of a model kind (knn, tree, logistic, linear, perceptron) on a CSV.", ML, "csv:str, target:str, kind:str=tree, folds:int=5", lambda ctx, csv, target, kind, folds: _cv(ctx, csv, target, kind, folds), **FL)
reg("ml_standardize_csv", "Z-score every numeric column of a CSV and save the result.", ML, "csv:str, out:str=data/standardized.csv", lambda ctx, csv, out: _standardize(ctx, csv, out), **FL)
reg("ml_one_hot", "Replace a text column of a CSV with 0/1 columns.", ML, "csv:str, column:str, out:str=data/onehot.csv", lambda ctx, csv, column, out: _onehot(ctx, csv, column, out), **FL)
reg("ml_feature_importance", "Rank CSV columns by how strongly they correlate with the target.", ML, "csv:str, target:str", lambda ctx, csv, target: (lambda X, y, f: sorted(({"feature": n, "abs_correlation": round(abs(_corr([r[i] for r in X], [float(v) for v in y])), 4)} for i, n in enumerate(f)), key=lambda d: -d["abs_correlation"]))(*table(ctx, csv, target)), **FL)
reg("ml_pca", "Principal components of CSV numeric columns (power iteration): variance explained and first components.", ML, "csv:str, components:int=2", lambda ctx, csv, components: _pca(ctx, csv, components), **FL)
reg("ml_elbow_kmeans", "Inertia for k = 1…8 to choose the number of clusters.", ML, "csv:str", lambda ctx, csv: (lambda rows: {k: round(sum(min(sum((a - b) ** 2 for a, b in zip(r, c)) for c in fit_kmeans(rows, k)) for r in rows), 3) for k in range(1, min(9, len(rows)))})(_unlabelled(ctx, None, csv)[0]), **FL)
reg("ml_learning_curve_data", "Accuracy/R² as the training set grows (for plotting with line_chart).", ML, "csv:str, target:str, kind:str=tree", lambda ctx, csv, target, kind: _curve(ctx, csv, target, kind), **FL)


def _corr(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    d = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / d if d else 0.0


def _csv_split(ctx, path, frac, seed):
    rows = list(csv.reader(io.StringIO(ctx.path(path, True).read_text())))
    head, body = rows[0], rows[1:]
    random.Random(seed).shuffle(body)
    k = max(1, int(len(body) * frac))
    for nm, part in (("test", body[:k]), ("train", body[k:])):
        p = ctx.path(f"data/{nm}.csv")
        b = io.StringIO()
        csv.writer(b).writerows([head] + part)
        p.write_text(b.getvalue())
        ctx.produced(p, "table", f"{len(part)} rows")
    return {"train_rows": len(body) - k, "test_rows": k}


def _cv(ctx, path, target, kind, folds):
    X, y, _ = table(ctx, path, target)
    idx = list(range(len(X)))
    random.Random(1).shuffle(idx)
    scores = []
    for f in range(folds):
        te = set(idx[f::folds])
        Xtr, ytr = [X[i] for i in idx if i not in te], [y[i] for i in idx if i not in te]
        Xte, yte = [X[i] for i in idx if i in te], [y[i] for i in idx if i in te]
        mu, sd = mean_std(Xtr)
        m = {"type": kind, **({"tree": fit_tree(Xtr, ytr, 4)} if kind == "tree" else fit_logistic(Xtr, ytr) if kind == "logistic" else {"w": fit_linear(Xtr, [float(v) for v in ytr])} if kind == "linear" else {"k": 3, "X": Xtr, "y": ytr, "mu": mu, "sd": sd, "numeric": False} if kind == "knn" else {})}
        if kind == "linear":
            sc = metrics(yte, [predict_one(m, r) for r in Xte], regression=True).get("r2", 0)
        else:
            sc = metrics(yte, [predict_one(m, r) for r in Xte], regression=False)["accuracy"]
        scores.append(round(sc, 4))
    return {"folds": scores, "mean": round(sum(scores) / len(scores), 4)}


def _standardize(ctx, path, out):
    rows = list(csv.DictReader(io.StringIO(ctx.path(path, True).read_text())))
    for c in rows[0]:
        try:
            v = [float(r[c]) for r in rows]
        except ValueError:
            continue
        mu, sd = mean_std([[x] for x in v])
        for r, x in zip(rows, v):
            r[c] = round((x - mu[0]) / sd[0], 6)
    b = io.StringIO()
    w = csv.DictWriter(b, list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    p = ctx.path(out)
    p.write_text(b.getvalue())
    return {"saved": ctx.produced(p, "table")}


def _onehot(ctx, path, col, out):
    rows = list(csv.DictReader(io.StringIO(ctx.path(path, True).read_text())))
    if col not in rows[0]:
        raise ToolError(f"column '{col}' not found")
    cats = sorted({r[col] for r in rows})
    for r in rows:
        v = r.pop(col)
        for c in cats:
            r[f"{col}_{c}"] = int(v == c)
    b = io.StringIO()
    w = csv.DictWriter(b, list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    p = ctx.path(out)
    p.write_text(b.getvalue())
    return {"saved": ctx.produced(p, "table"), "new_columns": [f"{col}_{c}" for c in cats]}


def _pca(ctx, path, k):
    rows = _unlabelled(ctx, None, path)[0]
    mu = [sum(c) / len(c) for c in zip(*rows)]
    Z = [[a - m for a, m in zip(r, mu)] for r in rows]
    n, d = len(Z), len(Z[0])
    C = [[sum(r[i] * r[j] for r in Z) / (n - 1) for j in range(d)] for i in range(d)]
    total = sum(C[i][i] for i in range(d)) or 1
    comps, var = [], []
    for _ in range(min(k, d)):
        v = [1.0] * d
        for _ in range(200):
            w = [sum(C[i][j] * v[j] for j in range(d)) for i in range(d)]
            nrm = math.sqrt(sum(a * a for a in w)) or 1
            v = [a / nrm for a in w]
        lam = sum(v[i] * sum(C[i][j] * v[j] for j in range(d)) for i in range(d))
        comps.append([round(a, 4) for a in v])
        var.append(round(lam / total, 4))
        C = [[C[i][j] - lam * v[i] * v[j] for j in range(d)] for i in range(d)]
    return {"explained_variance_ratio": var, "components": comps}


def _curve(ctx, path, target, kind):
    X, y, _ = table(ctx, path, target)
    idx = list(range(len(X)))
    random.Random(2).shuffle(idx)
    te = idx[:max(1, len(X) // 5)]
    pool = idx[len(te):]
    out = []
    for frac in (.2, .4, .6, .8, 1.0):
        tr = pool[:max(3, int(len(pool) * frac))]
        Xtr, ytr = [X[i] for i in tr], [y[i] for i in tr]
        Xte, yte = [X[i] for i in te], [y[i] for i in te]
        m = {"type": "tree", "tree": fit_tree(Xtr, ytr, 4)} if kind == "tree" else {"type": "linear", "w": fit_linear(Xtr, [float(v) for v in ytr])}
        s = metrics(yte, [predict_one(m, r) for r in Xte], regression=kind != "tree")
        out.append({"train_size": len(tr), "score": s.get("accuracy", s.get("r2"))})
    return out


# ------------------------------------------------------------------ text models
TX = "text_ml"
_tok = lambda t: re.findall(r"[a-z0-9']+", t.lower())  # noqa: E731
reg("train_text_classifier", "Train a Naive-Bayes text classifier. examples: list of [text, label] pairs (or CSV with columns text,label).", TX, "examples:list?, csv:str?, name:str=text_classifier",
    lambda ctx, examples, csv, name: _nb_train(ctx, examples, csv, name), tags=("training", "nlp", "sentiment", "spam"), produces="model", **FL)
reg("classify_text", "Classify a text with a saved Naive-Bayes model.", TX, "model:str, text:str", lambda ctx, model, text: _nb_predict(load_model(ctx, model), text), **FL)
reg("train_markov_text", "Train a Markov-chain text generator from a workspace text file.", TX, "path:str, order:int=2, name:str=markov", lambda ctx, path, order, name: _mk_train(ctx, path, order, name), tags=("training", "generation"), produces="model", **FL)
reg("generate_markov_text", "Generate text from a trained Markov model.", TX, "model:str, words:int=60, seed:int=1", lambda ctx, model, words, seed: _mk_gen(load_model(ctx, model), words, seed), **FL)
reg("tfidf_keywords", "Top TF-IDF keywords of each document (workspace .txt files).", TX, "paths:list, top:int=6", lambda ctx, paths, top: _tfidf(ctx, paths, top), **FL)
reg("cosine_similarity_text", "Cosine similarity between two texts (bag of words).", TX, "a:str, b:str", lambda a, b: round((lambda ca, cb: sum(ca[w] * cb[w] for w in ca) / (math.sqrt(sum(v * v for v in ca.values())) * math.sqrt(sum(v * v for v in cb.values())) or 1))(Counter(_tok(a)), Counter(_tok(b))), 4))
reg("search_documents_bm25", "BM25 search over text/markdown files in a workspace folder.", TX, "query:str, folder:str=., top:int=5", lambda ctx, query, folder, top: _bm25(ctx, query, folder, top), tags=("search", "retrieval", "rag"), **FL)
reg("build_vocabulary", "Word counts of a text.", TX, "text:str, top:int=20", lambda text, top: Counter(_tok(text)).most_common(top))
reg("ngram_counts", "Most frequent n-grams of a text.", TX, "text:str, n:int=2, top:int=15", lambda text, n, top: [(" ".join(g), c) for g, c in Counter(zip(*[_tok(text)[i:] for i in range(n)])).most_common(top)])


def _nb_train(ctx, examples, csv_path, name):
    if csv_path:
        rows = list(csv.DictReader(io.StringIO(ctx.path(csv_path, True).read_text())))
        examples = [[r["text"], r["label"]] for r in rows]
    if not examples or len(examples) < 4:
        raise ToolError("give at least 4 [text, label] examples")
    counts, docs, vocab = defaultdict(Counter), Counter(), set()
    for t, l in examples:
        docs[str(l)] += 1
        for w in _tok(str(t)):
            counts[str(l)][w] += 1
            vocab.add(w)
    m = {"type": "naive_bayes", "counts": {k: dict(v) for k, v in counts.items()}, "docs": dict(docs), "vocab": len(vocab)}
    hits = sum(_nb_predict(m, t)["label"] == str(l) for t, l in examples)
    return {"model": save_model(ctx, name, m), "classes": list(docs), "training_accuracy": round(hits / len(examples), 4), "vocabulary": len(vocab)}


def _nb_predict(m, text):
    tot = sum(m["docs"].values())
    sc = {}
    for c, d in m["docs"].items():
        n = sum(m["counts"][c].values())
        sc[c] = math.log(d / tot) + sum(math.log((m["counts"][c].get(w, 0) + 1) / (n + m["vocab"] + 1)) for w in _tok(text))
    z = max(sc.values())
    p = {c: math.exp(v - z) for c, v in sc.items()}
    s = sum(p.values())
    best = max(p, key=p.get)
    return {"label": best, "probabilities": {c: round(v / s, 3) for c, v in p.items()}}


def _mk_train(ctx, path, order, name):
    w = ctx.path(path, True).read_text(errors="replace").split()
    if len(w) < order + 5:
        raise ToolError("the text is too short")
    chain = defaultdict(list)
    for i in range(len(w) - order):
        chain[" ".join(w[i:i + order])].append(w[i + order])
    return {"model": save_model(ctx, name, {"type": "markov", "order": order, "chain": chain, "words": len(w)}), "states": len(chain)}


def _mk_gen(m, n, seed):
    r = random.Random(seed)
    keys = list(m["chain"])
    cur = r.choice(keys).split()
    out = cur[:]
    for _ in range(min(n, 500)):
        nxt = m["chain"].get(" ".join(out[-m["order"]:]))
        if not nxt:
            out += r.choice(keys).split()[:1]
            continue
        out.append(r.choice(nxt))
    return " ".join(out)


def _tfidf(ctx, paths, top):
    docs = {p: Counter(_tok(ctx.path(p, True).read_text(errors="replace"))) for p in paths}
    df = Counter(w for c in docs.values() for w in c)
    n = len(docs)
    return {p: [w for w, _ in sorted(((w, c * math.log((1 + n) / (1 + df[w])) + 1e-9) for w, c in cnt.items() if len(w) > 2), key=lambda x: -x[1])[:top]] for p, cnt in docs.items()}


def _bm25(ctx, q, folder, top):
    files = [p for p in ctx.path(folder).resolve().rglob("*") if p.is_file() and p.suffix.lower() in (".txt", ".md", ".csv", ".json", ".py", ".html") and p.stat().st_size < 3_000_000][:500]
    docs = [(p, Counter(_tok(p.read_text(errors="ignore")))) for p in files]
    if not docs:
        raise ToolError("no text files found in that folder")
    avg = sum(sum(c.values()) for _, c in docs) / len(docs)
    df = Counter(w for _, c in docs for w in c)
    res = []
    for p, c in docs:
        ln = sum(c.values())
        s = sum(math.log(1 + (len(docs) - df[w] + .5) / (df[w] + .5)) * c[w] * 2.2 / (c[w] + 1.2 * (.25 + .75 * ln / avg)) for w in set(_tok(q)) if w in c)
        if s > 0:
            res.append({"file": str(p.relative_to(ctx.workspace.resolve())), "score": round(s, 3)})
    return sorted(res, key=lambda d: -d["score"])[:top]


# ------------------------------------------------------------------ optimisation and classic algorithms
OP = "algorithms"
reg("gradient_descent", "Minimise f(x) (use x as the variable) by gradient descent from a start point.", OP, "expression:str, start:float=1, rate:float=0.1, steps:int=200", lambda expression, start, rate, steps: _gd(expression, start, rate, steps))
reg("genetic_algorithm", "Maximise an expression of variables a,b,c… over given bounds with a genetic algorithm, e.g. '-(a-3)**2-(b+1)**2' with bounds [[-10,10],[-10,10]].", OP, "expression:str, bounds:list, generations:int=80, population:int=60, seed:int=1", lambda expression, bounds, generations, population, seed: _ga(expression, bounds, generations, population, seed))
reg("simulated_annealing", "Minimise an expression of variables a,b,c… with simulated annealing.", OP, "expression:str, bounds:list, steps:int=4000, seed:int=1", lambda expression, bounds, steps, seed: _sa(expression, bounds, steps, seed))
reg("hill_climb", "Maximise an expression of a,b,c… by random-restart hill climbing.", OP, "expression:str, bounds:list, restarts:int=10, seed:int=1", lambda expression, bounds, restarts, seed: _hc(expression, bounds, restarts, seed))
reg("monte_carlo_pi", "Estimate π by random sampling.", OP, "samples:int=200000, seed:int=1", lambda samples, seed: (lambda r: 4 * sum(r.random() ** 2 + r.random() ** 2 <= 1 for _ in range(min(samples, 2_000_000))) / min(samples, 2_000_000))(random.Random(seed)))
reg("monte_carlo_integral", "Estimate a definite integral of f(x) by random sampling.", OP, "expression:str, a:float, b:float, samples:int=50000, seed:int=1", lambda expression, a, b, samples, seed: (lambda r: (b - a) * sum(evaluate(expression, x=r.uniform(a, b)) for _ in range(min(samples, 200000))) / min(samples, 200000))(random.Random(seed)))
reg("knapsack", "0/1 knapsack: best value for items [[weight, value], …] and a capacity.", OP, "items:list, capacity:int", lambda items, capacity: _knap(items, capacity))
reg("tsp_nearest_neighbour", "A short tour through points [[x,y],…] (nearest neighbour + 2-opt).", OP, "points:list", lambda points: _tsp(points))
reg("shortest_path", "Dijkstra on edges [[a, b, weight], …] from a source to a target.", OP, "edges:list, source:str, target:str", lambda edges, source, target: _dijkstra(edges, source, target))
reg("minimum_spanning_tree", "Kruskal's MST of edges [[a, b, weight], …].", OP, "edges:list", lambda edges: _mst(edges))
reg("topological_sort", "Order tasks with dependencies [[before, after], …].", OP, "edges:list", lambda edges: _topo(edges))
reg("breadth_first_search", "BFS distance from a start over edges [[a, b], …] (undirected).", OP, "edges:list, start:str", lambda edges, start: _bfs(edges, start))
reg("pagerank", "PageRank of a directed graph given links [[from, to], …].", OP, "links:list, damping:float=0.85", lambda links, damping: _pr(links, damping))
reg("sudoku_solve", "Solve a Sudoku given 81 characters (0 or . for blanks).", OP, "puzzle:str", lambda puzzle: _sudoku(puzzle))
reg("n_queens", "Count and show a solution of the n-queens puzzle.", OP, "n:int=8", lambda n: _queens(n))
reg("game_of_life_step", "Advance a Game-of-Life grid (list of strings with # and .) by n generations.", OP, "grid:list, generations:int=1", lambda grid, generations: _life(grid, generations))
reg("sort_visualiser_steps", "Number of comparisons and swaps that bubble, insertion, selection, merge and quick sort need for your list.", OP, "numbers:list", lambda numbers: _sorts([float(x) for x in numbers]))
reg("lcs_diff", "Longest common subsequence of two strings.", OP, "a:str, b:str", lambda a, b: _lcs(a, b))
reg("levenshtein_alignment", "Edit operations that turn string a into b.", OP, "a:str, b:str", lambda a, b: _edit_ops(a, b))
reg("run_length_encode", "Run-length encode a text.", OP, "text:str", lambda text: "".join(f"{len(list(g))}{k}" for k, g in __import__("itertools").groupby(text)))
reg("huffman_codes", "Huffman code table for a text.", OP, "text:str", lambda text: _huff(text))
reg("linear_programming_2d", "Maximise c1·x + c2·y subject to constraints [[a, b, limit], …] meaning a·x + b·y ≤ limit, x,y ≥ 0.", OP, "c1:float, c2:float, constraints:list", lambda c1, c2, constraints: _lp(c1, c2, constraints))


def _vars(expr, bounds):
    names = "abcdefgh"[:len(bounds)]
    return names, lambda v: evaluate(expr, **dict(zip(names, v)))


def _gd(expr, x, lr, steps):
    h = 1e-6
    for _ in range(min(steps, 5000)):
        g = (evaluate(expr, x=x + h) - evaluate(expr, x=x - h)) / (2 * h)
        x -= lr * g
    return {"x": round(x, 8), "f": round(evaluate(expr, x=x), 8)}


def _ga(expr, bounds, gens, pop, seed):
    names, f = _vars(expr, bounds)
    r = random.Random(seed)
    P = [[r.uniform(lo, hi) for lo, hi in bounds] for _ in range(min(pop, 400))]
    for _ in range(min(gens, 400)):
        P.sort(key=f, reverse=True)
        elite = P[:len(P) // 4]
        P = elite[:]
        while len(P) < pop:
            a, b = r.sample(elite, 2)
            c = [(x + y) / 2 + r.gauss(0, (hi - lo) * .03) for x, y, (lo, hi) in zip(a, b, bounds)]
            P.append([min(hi, max(lo, v)) for v, (lo, hi) in zip(c, bounds)])
    best = max(P, key=f)
    return {"best": dict(zip(names, [round(v, 5) for v in best])), "value": round(f(best), 6)}


def _sa(expr, bounds, steps, seed):
    names, f = _vars(expr, bounds)
    r = random.Random(seed)
    cur = [r.uniform(lo, hi) for lo, hi in bounds]
    best, fb, fc = cur[:], f(cur), f(cur)
    for i in range(min(steps, 50000)):
        T = 1 - i / steps + 1e-3
        nxt = [min(hi, max(lo, v + r.gauss(0, (hi - lo) * .1 * T))) for v, (lo, hi) in zip(cur, bounds)]
        fn = f(nxt)
        if fn < fc or r.random() < math.exp(-(fn - fc) / (T * (abs(fc) + 1) * .1)):
            cur, fc = nxt, fn
            if fc < fb:
                best, fb = cur[:], fc
    return {"best": dict(zip(names, [round(v, 5) for v in best])), "value": round(fb, 6)}


def _hc(expr, bounds, restarts, seed):
    names, f = _vars(expr, bounds)
    r = random.Random(seed)
    best, fb = None, -1e300
    for _ in range(min(restarts, 100)):
        cur = [r.uniform(lo, hi) for lo, hi in bounds]
        fc = f(cur)
        step = [(hi - lo) / 10 for lo, hi in bounds]
        for _ in range(400):
            i = r.randrange(len(cur))
            for d in (-1, 1):
                n = cur[:]
                n[i] = min(bounds[i][1], max(bounds[i][0], n[i] + d * step[i]))
                if f(n) > fc:
                    cur, fc = n, f(n)
            step = [s * .99 for s in step]
        if fc > fb:
            best, fb = cur, fc
    return {"best": dict(zip(names, [round(v, 5) for v in best])), "value": round(fb, 6)}


def _knap(items, cap):
    n = len(items)
    dp = [[0] * (cap + 1) for _ in range(n + 1)]
    for i, (w, v) in enumerate(items, 1):
        for c in range(cap + 1):
            dp[i][c] = max(dp[i - 1][c], dp[i - 1][c - int(w)] + v if c >= w else 0)
    c, chosen = cap, []
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            chosen.append(i - 1)
            c -= int(items[i - 1][0])
    return {"best_value": dp[n][cap], "chosen_item_indexes": sorted(chosen)}


def _tsp(pts):
    n = len(pts)
    d = lambda a, b: math.dist(pts[a], pts[b])  # noqa: E731
    tour, left = [0], set(range(1, n))
    while left:
        j = min(left, key=lambda k: d(tour[-1], k))
        tour.append(j)
        left.remove(j)
    imp = True
    while imp and n > 3:
        imp = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if d(tour[i - 1], tour[j]) + d(tour[i], tour[(j + 1) % n]) < d(tour[i - 1], tour[i]) + d(tour[j], tour[(j + 1) % n]) - 1e-9:
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    imp = True
    return {"order": tour + [0], "length": round(sum(d(tour[i], tour[(i + 1) % n]) for i in range(n)), 4)}


def _dijkstra(edges, s, t):
    import heapq
    g = defaultdict(list)
    for a, b, w in edges:
        g[str(a)].append((str(b), float(w)))
        g[str(b)].append((str(a), float(w)))
    dist, prev, pq = {str(s): 0.0}, {}, [(0.0, str(s))]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, 1e300):
            continue
        for v, w in g[u]:
            if d + w < dist.get(v, 1e300):
                dist[v], prev[v] = d + w, u
                heapq.heappush(pq, (d + w, v))
    if str(t) not in dist:
        raise ToolError("no path between those nodes")
    path, c = [str(t)], str(t)
    while c in prev:
        c = prev[c]
        path.append(c)
    return {"distance": dist[str(t)], "path": path[::-1]}


def _mst(edges):
    par = {}
    find = lambda x: x if par.setdefault(x, x) == x else par.__setitem__(x, find(par[x])) or par[x]  # noqa: E731
    out, tot = [], 0
    for a, b, w in sorted(edges, key=lambda e: e[2]):
        ra, rb = find(str(a)), find(str(b))
        if ra != rb:
            par[ra] = rb
            out.append([a, b, w])
            tot += w
    return {"edges": out, "total_weight": tot}


def _topo(edges):
    g, ind = defaultdict(list), Counter()
    nodes = set()
    for a, b in edges:
        g[str(a)].append(str(b))
        ind[str(b)] += 1
        nodes |= {str(a), str(b)}
    q, out = sorted(n for n in nodes if not ind[n]), []
    while q:
        u = q.pop(0)
        out.append(u)
        for v in g[u]:
            ind[v] -= 1
            if not ind[v]:
                q.append(v)
    if len(out) != len(nodes):
        raise ToolError("there is a cycle — no valid order")
    return out


def _bfs(edges, s):
    g = defaultdict(list)
    for a, b in edges:
        g[str(a)].append(str(b))
        g[str(b)].append(str(a))
    d, q = {str(s): 0}, [str(s)]
    while q:
        u = q.pop(0)
        for v in g[u]:
            if v not in d:
                d[v] = d[u] + 1
                q.append(v)
    return d


def _pr(links, damp):
    nodes = sorted({str(x) for l in links for x in l})
    out = defaultdict(list)
    for a, b in links:
        out[str(a)].append(str(b))
    pr = {n: 1 / len(nodes) for n in nodes}
    for _ in range(100):
        new = {n: (1 - damp) / len(nodes) for n in nodes}
        for n in nodes:
            tgt = out[n] or nodes
            for t in tgt:
                new[t] += damp * pr[n] / len(tgt)
        pr = new
    return {k: round(v, 4) for k, v in sorted(pr.items(), key=lambda x: -x[1])}


def _sudoku(p):
    p = p.replace(".", "0").strip()
    if len(p) != 81 or not p.isdigit():
        raise ToolError("give exactly 81 characters (digits, 0 or . for blanks)")
    g = [int(c) for c in p]

    def ok(i, v):
        r, c = divmod(i, 9)
        return all(g[r * 9 + k] != v and g[k * 9 + c] != v and g[(r // 3 * 3 + k // 3) * 9 + c // 3 * 3 + k % 3] != v for k in range(9))

    def go():
        try:
            i = g.index(0)
        except ValueError:
            return True
        for v in range(1, 10):
            if ok(i, v):
                g[i] = v
                if go():
                    return True
                g[i] = 0
        return False
    if not go():
        raise ToolError("this puzzle has no solution")
    return "\n".join("".join(map(str, g[r * 9:r * 9 + 9])) for r in range(9))


def _queens(n):
    n = min(n, 12)
    sols, cols = [], []

    def go(r):
        if r == n:
            sols.append(cols[:])
            return
        for c in range(n):
            if all(c != cc and abs(c - cc) != r - rr for rr, cc in enumerate(cols)):
                cols.append(c)
                go(r + 1)
                cols.pop()
    go(0)
    return {"solutions": len(sols), "first": ["".join("Q" if c == i else "." for i in range(n)) for c in sols[0]] if sols else None}


def _life(grid, n):
    cells = {(x, y) for y, row in enumerate(grid) for x, c in enumerate(row) if c == "#"}
    for _ in range(min(n, 500)):
        cnt = Counter((x + dx, y + dy) for x, y in cells for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)
        cells = {c for c, k in cnt.items() if k == 3 or (k == 2 and c in cells)}
    if not cells:
        return ["(empty)"]
    xs, ys = [c[0] for c in cells], [c[1] for c in cells]
    return ["".join("#" if (x, y) in cells else "." for x in range(min(xs), max(xs) + 1)) for y in range(min(ys), max(ys) + 1)]


def _sorts(a):
    def bubble(x):
        x, c, s = x[:], 0, 0
        for i in range(len(x)):
            for j in range(len(x) - 1 - i):
                c += 1
                if x[j] > x[j + 1]:
                    x[j], x[j + 1] = x[j + 1], x[j]
                    s += 1
        return c, s

    def insertion(x):
        x, c, s = x[:], 0, 0
        for i in range(1, len(x)):
            j = i
            while j > 0:
                c += 1
                if x[j - 1] > x[j]:
                    x[j - 1], x[j] = x[j], x[j - 1]
                    s += 1
                    j -= 1
                else:
                    break
        return c, s

    def selection(x):
        x, c, s = x[:], 0, 0
        for i in range(len(x)):
            m = i
            for j in range(i + 1, len(x)):
                c += 1
                if x[j] < x[m]:
                    m = j
            if m != i:
                x[i], x[m] = x[m], x[i]
                s += 1
        return c, s

    def merge(x):
        c = 0

        def ms(l):
            nonlocal c
            if len(l) < 2:
                return l
            a, b = ms(l[:len(l) // 2]), ms(l[len(l) // 2:])
            o = []
            while a and b:
                c += 1
                o.append(a.pop(0) if a[0] <= b[0] else b.pop(0))
            return o + a + b
        ms(x[:])
        return c, 0

    def quick(x):
        c = 0

        def qs(l):
            nonlocal c
            if len(l) < 2:
                return l
            p = l[len(l) // 2]
            c += len(l)
            return qs([v for v in l if v < p]) + [v for v in l if v == p] + qs([v for v in l if v > p])
        qs(x[:])
        return c, 0
    if len(a) > 400:
        raise ToolError("use at most 400 numbers")
    return {n: dict(zip(("comparisons", "swaps"), f(a))) for n, f in (("bubble", bubble), ("insertion", insertion), ("selection", selection), ("merge", merge), ("quick", quick))}


def _lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    i, j, out = len(a), len(b), []
    while i and j:
        if a[i - 1] == b[j - 1]:
            out.append(a[i - 1])
            i, j = i - 1, j - 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return {"length": dp[-1][-1], "subsequence": "".join(reversed(out))}


def _edit_ops(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    i, j, ops = len(a), len(b), []
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] != b[j - 1]:
                ops.append(f"replace '{a[i - 1]}' with '{b[j - 1]}' at {i - 1}")
            i, j = i - 1, j - 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(f"delete '{a[i - 1]}' at {i - 1}")
            i -= 1
        else:
            ops.append(f"insert '{b[j - 1]}' at {i}")
            j -= 1
    return {"distance": dp[-1][-1], "operations": ops[::-1]}


def _huff(text):
    import heapq
    if not text:
        raise ToolError("text is empty")
    h = [[w, i, {c: ""}] for i, (c, w) in enumerate(Counter(text).items())]
    heapq.heapify(h)
    k = len(h)
    while len(h) > 1:
        a, b = heapq.heappop(h), heapq.heappop(h)
        heapq.heappush(h, [a[0] + b[0], k, {**{c: "0" + v for c, v in a[2].items()}, **{c: "1" + v for c, v in b[2].items()}}])
        k += 1
    codes = h[0][2] if len(h[0][2]) > 1 else {c: "0" for c in h[0][2]}
    bits = sum(len(codes[c]) for c in text)
    return {"codes": codes, "encoded_bits": bits, "original_bits": 8 * len(text), "ratio": round(bits / (8 * len(text)), 3)}


def _lp(c1, c2, cons):
    pts = [(0.0, 0.0)]
    lines = [(a, b, l) for a, b, l in cons] + [(1, 0, 0), (0, 1, 0)]
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a1, b1, l1 = lines[i]
            a2, b2, l2 = lines[j]
            d = a1 * b2 - a2 * b1
            if abs(d) > 1e-12:
                pts.append(((l1 * b2 - l2 * b1) / d, (a1 * l2 - a2 * l1) / d))
    ok = [p for p in pts if p[0] >= -1e-9 and p[1] >= -1e-9 and all(a * p[0] + b * p[1] <= l + 1e-9 for a, b, l in cons)]
    if not ok:
        raise ToolError("no feasible point")
    best = max(ok, key=lambda p: c1 * p[0] + c2 * p[1])
    return {"x": round(best[0], 6), "y": round(best[1], 6), "objective": round(c1 * best[0] + c2 * best[1], 6)}
