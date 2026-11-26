"""Learn a piece of writing in one pass: a character-level language model, plus search over its passages.

The model counts which character follows every context of 1 to `max_order` characters, and blends the long,
specific contexts with the short, general ones (interpolated modified Kneser–Ney smoothing).
Counting is one pass over the text, done in NumPy, so a few megabytes are learned in a few seconds. The context
length is chosen on the last tenth of the text, which the model has not seen while choosing.

What it is good at: continuing text in the style, spelling and layout of what it read (names, code, recipes,
song lyrics…). What it is not: it does not understand or reason. It has no knowledge beyond your text. For
answering questions, `Passages` finds the parts of your text that match the question.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SHIFT = 20                       # low bits of a key hold the next character
MASK = (1 << SHIFT) - 1
PRIME = np.uint64(0x9E3779B97F4A7C15)
MAX_CHARS = 4_000_000            # longer uploads are learned from their first 4 million characters


def _mix(h: np.ndarray) -> np.ndarray:
    return (h ^ (h >> np.uint64(29))) >> np.uint64(SHIFT)


def _discounts(c: np.ndarray) -> tuple[float, float, float]:
    """Modified Kneser–Ney discounts for counts of 1, 2 and 3+ (Chen & Goodman), estimated from counts-of-counts."""
    n = [int((c == k).sum()) for k in (1, 2, 3, 4)]
    if min(n) == 0:
        return (0.5, 1.0, 1.5) if n[0] == 0 or n[1] == 0 else (0.6, 1.1, 1.5)
    y = n[0] / (n[0] + 2 * n[1])
    d = (1 - 2 * y * n[1] / n[0], 2 - 3 * y * n[2] / n[1], 3 - 4 * y * n[3] / n[2])
    return tuple(float(np.clip(v, 0.05, k + 0.95)) for k, v in zip((0, 1, 2), d))


class Stats:
    """Counts of (context, next character) for one context length, sorted by key, with per-context sums."""

    def __init__(self, keys: np.ndarray, counts: np.ndarray, disc=None):
        self.keys, self.counts = keys, counts.astype(np.uint32)
        self.disc = tuple(disc) if disc else _discounts(self.counts)
        cx = keys >> np.uint64(SHIFT)
        self.ctx, start = np.unique(cx, return_index=True)
        c = self.counts.astype(np.float64)
        if len(c):
            self.tot = np.add.reduceat(c, start)
            d1, d2, d3 = self.disc
            dsum = np.where(c == 1, d1, np.where(c == 2, d2, d3))
            self.gamma = np.add.reduceat(dsum, start) / self.tot
        else:
            self.tot = self.gamma = np.zeros(0)

    def apply(self, cx: np.ndarray, nxt: np.ndarray, p_lower: np.ndarray) -> np.ndarray:
        """P(next | context) for many positions at once, given the lower-order probability of the same next character."""
        if not len(self.ctx):
            return p_lower
        i = np.clip(np.searchsorted(self.ctx, cx), 0, len(self.ctx) - 1)
        found = self.ctx[i] == cx
        key = (cx << np.uint64(SHIFT)) | nxt.astype(np.uint64)
        j = np.clip(np.searchsorted(self.keys, key), 0, len(self.keys) - 1)
        cnt = np.where(self.keys[j] == key, self.counts[j], 0).astype(np.float64)
        d1, d2, d3 = self.disc
        d = np.where(cnt == 1, d1, np.where(cnt == 2, d2, d3))
        tot = np.where(found, self.tot[i], 1.0)
        return np.where(found, np.maximum(cnt - d, 0) / tot + self.gamma[i] * p_lower, p_lower)

    def update(self, cx, p: np.ndarray) -> np.ndarray:
        """Full next-character distribution for one context, given the lower-order distribution."""
        i = int(np.searchsorted(self.ctx, cx))
        if i >= len(self.ctx) or self.ctx[i] != cx:
            return p
        lo = int(np.searchsorted(self.keys, cx << np.uint64(SHIFT)))
        hi = int(np.searchsorted(self.keys, (cx + np.uint64(1)) << np.uint64(SHIFT)))
        nexts = (self.keys[lo:hi] & np.uint64(MASK)).astype(np.int64)
        cnt = self.counts[lo:hi].astype(np.float64)
        d1, d2, d3 = self.disc
        d = np.where(cnt == 1, d1, np.where(cnt == 2, d2, d3))
        p = p * self.gamma[i]
        p[nexts] += np.maximum(cnt - d, 0) / self.tot[i]
        return p


def _positions(seq: np.ndarray, order: int):
    """Context hashes for k = 1..order (generated one k at a time) and the next characters."""
    P = np.concatenate([np.zeros(order, np.int64), seq])
    n = len(seq)
    nxt = P[order:]

    def gen():
        h = np.zeros(n, np.uint64)
        with np.errstate(over="ignore"):
            for k in range(1, order + 1):
                h = h * PRIME + (P[order - k: order - k + n].astype(np.uint64) + np.uint64(1))
                yield _mix(h)
    return gen(), nxt


class CharModel:
    """Character model with interpolated modified Kneser–Ney smoothing.

    The longest context uses real counts; shorter contexts use 'continuation' counts (in how many different
    contexts a character sequence appears), which is what makes Kneser–Ney better than plain back-off."""

    def __init__(self, max_order: int = 10):
        self.max_order = max_order
        self.order = max_order
        self.chars: list[str] = []
        self.levels: list[Stats] = []     # levels[k-1] = context length k (continuation counts, except the last)
        self.report: dict = {}

    def _encode(self, text: str, grow: bool = False) -> np.ndarray:
        if grow:
            self.chars = sorted(set(text))
            if len(self.chars) >= MASK:
                raise ValueError("too many different characters")
        pos = {c: i + 1 for i, c in enumerate(self.chars)}
        return np.array([pos.get(c, 0) for c in text], np.int64)

    def _count(self, seq: np.ndarray, order: int, raw_all: bool = False):
        """Statistics for context lengths 1..order: raw counts (all lengths if raw_all, else only the longest) and
        continuation counts for the shorter ones. The continuation count of (context, next) is the number of different
        one-longer contexts it appears in, read off the distinct keys of the next length."""
        hashes, nxt = _positions(seq, order)
        nxt64 = nxt.astype(np.uint64)
        raw, cont = [None] * order, [None] * order
        prev = None
        for k, h in enumerate(hashes):
            key = (h << np.uint64(SHIFT)) | nxt64
            if k:
                u, first, c = np.unique(key, return_index=True, return_counts=True)
                u2, cc = np.unique(prev[first], return_counts=True)
                cont[k - 1] = Stats(u2, cc)
            else:
                u, c = np.unique(key, return_counts=True)
            if raw_all or k == order - 1:
                raw[k] = Stats(u, c)
            if k == 0:
                first_level = u
            prev = key
        V = len(self.chars)
        # unigram continuation: in how many different one-character contexts each character appears
        cu = np.bincount((first_level & np.uint64(MASK)).astype(np.int64), minlength=V + 1).astype(np.float64)
        self.p0 = (cu + 0.5) / (cu.sum() + 0.5 * (V + 1))
        self.p0[0] = 0.0
        self.p0 /= self.p0.sum()
        return raw, cont

    def fit(self, text: str, progress=None) -> "CharModel":
        t0 = time.perf_counter()
        say = progress or (lambda **k: None)
        text = text[:MAX_CHARS]
        if len(text) < 200:
            raise ValueError("The text is too short to learn from. Upload at least a few paragraphs.")
        self._encode(text, grow=True)
        sample = text if len(text) <= 1_200_000 else text[:1_080_000] + text[-120_000:]
        cut = int(len(sample) * 0.9)
        tr, va = self._encode(sample[:cut]), self._encode(sample[cut:])
        say(stage="counting", text=f"Counting contexts of 1–{self.max_order} characters in {len(tr):,} characters…")
        raw, cont = self._count(tr, self.max_order, raw_all=True)
        say(stage="choosing", text="Measuring how well each context length predicts the held-back tenth…")
        curve = self._curve(tr, va, raw, cont)
        best = int(np.argmin(curve)) + 1
        uni = np.bincount(tr, minlength=len(self.chars) + 1).astype(np.float64)[1:] + 0.5
        vv = va[va > 0]
        uni_bits = float(-np.log2(uni[vv - 1] / uni.sum()).mean()) if len(vv) else 0.0
        say(stage="final", text=f"Best context: {best} characters. Learning all of the text with it…")
        self.order = best
        raw, cont = self._count(self._encode(text), best)
        self.levels = cont[:-1] + [raw[-1]]
        secs = time.perf_counter() - t0
        V = len(self.chars)
        self.report = {"chars": len(text), "vocab": V, "order": best, "seconds": round(secs, 3), "chars_per_second": round(len(text) / max(secs, 1e-9)),
                       "valid_bits_per_char": round(float(curve[best - 1]), 4), "bits_by_order": [round(float(b), 4) for b in curve],
                       "random_bits_per_char": round(math.log2(max(V, 2)), 4), "letter_frequency_bits_per_char": round(uni_bits, 4),
                       "contexts": int(sum(len(l.keys) for l in self.levels))}
        return self

    def _curve(self, tr, va, raw, cont) -> np.ndarray:
        """Bits per character on `va` (following `tr`) if the longest context were 1, 2, … characters."""
        order = len(raw)
        seq = np.concatenate([tr[-(order + 1):], va])
        hashes, nxt = _positions(seq, order)
        sel = slice(len(seq) - len(va), None)
        target = nxt[sel]
        ok = target > 0
        p = self.p0[target]
        out = []
        for k, h in enumerate(hashes):
            cx = h[sel]
            top = raw[k].apply(cx, target, p)
            out.append(float(-np.log2(np.clip(top[ok], 1e-12, 1)).mean()) if ok.any() else 0.0)
            if cont[k] is not None:
                p = cont[k].apply(cx, target, p)
        return np.array(out)

    def bits(self, text: str) -> float:
        """Bits per character the model needs for `text` (lower: the text looks more like what it learned)."""
        seq = self._encode(text)
        if not len(seq):
            return float("nan")
        hashes, nxt = _positions(seq, self.order)
        p = self.p0[nxt]
        for h, lv in zip(hashes, self.levels):
            p = lv.apply(h, nxt, p)
        ok = nxt > 0
        return float(-np.log2(np.clip(p[ok], 1e-12, 1)).mean()) if ok.any() else float("nan")

    # -- writing
    def next_distribution(self, history: list[int]) -> np.ndarray:
        p = self.p0.copy()
        hist = [0] * max(0, self.order - len(history)) + history[-self.order:]
        h = np.uint64(0)
        with np.errstate(over="ignore"):
            for k, lv in enumerate(self.levels, 1):
                h = h * PRIME + np.uint64(hist[-k] + 1)
                p = lv.update(_mix(h), p)
        return p

    def generate(self, prompt: str = "", length: int = 400, temperature: float = 0.8, seed: int | None = None) -> str:
        rng = np.random.default_rng(seed)
        hist = [int(x) for x in self._encode(prompt)] if prompt else []
        out = []
        temperature = float(np.clip(temperature, 0.05, 2.0))
        for _ in range(int(np.clip(length, 1, 5000))):
            p = self.next_distribution(hist)
            if temperature != 1.0:
                p = np.power(p, 1.0 / temperature)
            p[0] = 0.0
            p /= p.sum()
            c = int(rng.choice(len(p), p=p))
            hist.append(c)
            out.append(self.chars[c - 1])
        return "".join(out)

    # -- saving
    def state(self) -> tuple[dict, dict]:
        arrays = {"p0": self.p0} | {f"k{i}": l.keys for i, l in enumerate(self.levels)} | {f"c{i}": l.counts for i, l in enumerate(self.levels)}
        return {"chars": "".join(self.chars), "order": self.order, "max_order": self.max_order, "disc": [list(l.disc) for l in self.levels],
                "report": self.report}, arrays

    @classmethod
    def from_state(cls, meta: dict, arrays: dict) -> "CharModel":
        m = cls(meta["max_order"])
        m.chars, m.order, m.report = list(meta["chars"]), meta["order"], meta["report"]
        m.p0 = arrays["p0"]
        m.levels = [Stats(arrays[f"k{i}"], arrays[f"c{i}"], meta["disc"][i]) for i in range(m.order)]
        return m


# ------------------------------------------------------------------ search over passages
WORD = re.compile(r"\w+", re.UNICODE)
STOP = set("a an and are as at be been but by can could did do does for from had has have how i if in into is it its of on or "
           "should so than that the their them then there these they this to was we were what when where which who whom why will "
           "with would you your".split())


def passages_of(docs: list[tuple[str, str]], size: int = 700) -> list[dict]:
    """Split documents into passages of about `size` characters at paragraph and sentence breaks."""
    out = []
    for title, text in docs:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        buf = ""
        for para in paras:
            pieces = [para] if len(para) <= size * 1.5 else re.split(r"(?<=[.!?])\s+", para)
            for piece in pieces:
                if buf and len(buf) + len(piece) > size:
                    out.append({"title": title, "text": buf.strip()})
                    buf = ""
                buf += (" " if buf and piece is not para else "\n\n" if buf else "") + piece
        if buf.strip():
            out.append({"title": title, "text": buf.strip()})
    return out


class Passages:
    """BM25 search over passages of the uploaded text."""

    def __init__(self, passages: list[dict]):
        self.items = passages
        self.tf = [Counter(w.lower() for w in WORD.findall(p["text"])) for p in passages]
        self.len = np.array([sum(t.values()) for t in self.tf], np.float64)
        self.avg = float(self.len.mean()) if len(self.len) else 1.0
        self.post: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(self.tf):
            for w in t:
                self.post[w].append(i)

    def search(self, query: str, k: int = 5) -> list[dict]:
        words = [w.lower() for w in WORD.findall(query)]
        words = [w for w in words if w not in STOP] or words      # question words carry no topic
        if not words or not self.items:
            return []
        N = len(self.items)
        score = np.zeros(N)
        for w in set(words):
            docs = self.post.get(w)
            if not docs:
                continue
            idf = math.log(1 + (N - len(docs) + 0.5) / (len(docs) + 0.5))
            tf = np.array([self.tf[i][w] for i in docs], np.float64)
            score[docs] += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * self.len[docs] / self.avg))
        top = [int(i) for i in np.argsort(-score)[:k] if score[i] > 0]
        return [{**self.items[i], "score": round(float(score[i]), 3)} for i in top]


# ------------------------------------------------------------------ the saved text model
class TextModel:
    def __init__(self, model: CharModel, passages: list[dict], titles: list[str]):
        self.model, self.passages, self.titles = model, passages, titles
        self._search = None

    @property
    def report(self) -> dict:
        return {**self.model.report, "documents": len(self.titles), "titles": self.titles[:50], "passages": len(self.passages)}

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self._search is None:
            self._search = Passages(self.passages)
        return self._search.search(query, k)

    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        meta, arrays = self.model.state()
        np.savez_compressed(folder / "weights.npz", **arrays)
        (folder / "model.json").write_text(json.dumps({"format": 1, "type": "text", "model": meta, "titles": self.titles, "report": self.report}))
        (folder / "passages.json").write_text(json.dumps(self.passages, ensure_ascii=False))
        return folder

    @classmethod
    def load(cls, folder: str | Path) -> "TextModel":
        folder = Path(folder)
        meta = json.loads((folder / "model.json").read_text())
        with np.load(folder / "weights.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        return cls(CharModel.from_state(meta["model"], arrays), json.loads((folder / "passages.json").read_text()), meta["titles"])


def train_text(docs: list[tuple[str, str]], max_order: int = 10, progress=None, stop=None) -> TextModel:
    emit = progress or (lambda e: None)
    t0 = time.perf_counter()
    text = "\n\n".join(t.strip() for _, t in docs if t.strip())

    def say(stage, text):
        emit({"type": "stage", "stage": stage, "text": text, "t": round(time.perf_counter() - t0, 3)})

    say("reading", f"Read {len(docs)} document{'s' * (len(docs) != 1)}, {len(text):,} characters.")
    m = CharModel(max_order).fit(text, progress=lambda **k: say(k["stage"], k["text"]))
    passages = passages_of(docs)
    tm = TextModel(m, passages, [t for t, _ in docs])
    emit({"type": "done", "report": tm.report})
    return tm
