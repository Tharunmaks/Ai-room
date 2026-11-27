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
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CBITS = 14                       # a character id takes 14 bits (up to 16,383 different characters)
CMASK = (1 << CBITS) - 1
CTX = 36                         # a context is remembered by a 36-bit hash; key = context << 14 | next character
PRIME = np.uint64(0x9E3779B97F4A7C15)
MAX_CHARS = int(float(__import__("os").environ.get("AI_ROOM_LAB_TEXT_CHARS", 32_000_000)))   # learned at most; bigger texts are sampled
PARALLEL_CHARS = 1_000_000       # from this size on, several context lengths are counted at once (threads)
U = np.uint64


def _mix(h: np.ndarray) -> np.ndarray:
    return (h ^ (h >> U(29))) >> U(64 - CTX)


def _discounts(c: np.ndarray) -> tuple[float, float, float]:
    """Modified Kneser–Ney discounts for counts of 1, 2 and 3+ (Chen & Goodman), estimated from counts-of-counts."""
    n = np.bincount(np.minimum(c, 5).astype(np.int64), minlength=6)[1:5].tolist()
    if min(n) == 0:
        return (0.5, 1.0, 1.5) if n[0] == 0 or n[1] == 0 else (0.6, 1.1, 1.5)
    y = n[0] / (n[0] + 2 * n[1])
    d = (1 - 2 * y * n[1] / n[0], 2 - 3 * y * n[2] / n[1], 3 - 4 * y * n[3] / n[2])
    return tuple(float(np.clip(v, 0.05, k + 0.95)) for k, v in zip((0, 1, 2), d))


def _runs(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct values and how often each occurs, for an already sorted array."""
    if not len(s):
        return s, np.zeros(0, np.int64)
    starts = np.concatenate([[0], np.flatnonzero(s[1:] != s[:-1]) + 1])
    return s[starts], np.diff(np.append(starts, len(s)))


class Stats:
    """Counts of (context, next character) for one context length, sorted by key, with per-context sums."""

    def __init__(self, keys: np.ndarray, counts: np.ndarray, disc=None):
        self.keys, self.counts = keys, counts.astype(np.uint32)
        self.disc = tuple(disc) if disc else _discounts(self.counts)
        cx = keys >> U(CBITS)
        if len(cx):
            start = np.concatenate([[0], np.flatnonzero(cx[1:] != cx[:-1]) + 1])
            self.ctx = cx[start]
            self.tot = np.add.reduceat(self.counts, start, dtype=np.float64)
            dlut = np.array([0.0, *self.disc], np.float64)
            self.gamma = np.add.reduceat(dlut[np.minimum(self.counts, 3)], start) / self.tot
        else:
            self.ctx = cx
            self.tot = self.gamma = np.zeros(0)

    def apply(self, cx: np.ndarray, nxt: np.ndarray, p_lower: np.ndarray) -> np.ndarray:
        """P(next | context) for many positions at once, given the lower-order probability of the same next character."""
        if not len(self.ctx):
            return p_lower
        i = np.clip(np.searchsorted(self.ctx, cx), 0, len(self.ctx) - 1)
        found = self.ctx[i] == cx
        key = (cx << U(CBITS)) | nxt.astype(np.uint64)
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
        lo = int(np.searchsorted(self.keys, cx << U(CBITS)))
        hi = int(np.searchsorted(self.keys, (cx + U(1)) << U(CBITS)))
        nexts = (self.keys[lo:hi] & U(CMASK)).astype(np.int64)
        cnt = self.counts[lo:hi].astype(np.float64)
        d1, d2, d3 = self.disc
        d = np.where(cnt == 1, d1, np.where(cnt == 2, d2, d3))
        p = p * self.gamma[i]
        p[nexts] += np.maximum(cnt - d, 0) / self.tot[i]
        return p


def _hashes(P1: np.ndarray, order: int, pad: int):
    """For k = 1..order: the hash of the k characters before each position. P1 = character ids + 1 as uint64,
    starting with `pad` padding entries."""
    n = len(P1) - pad
    h = np.zeros(n, np.uint64)
    with np.errstate(over="ignore"):
        for k in range(1, order + 1):
            h *= PRIME
            h += P1[pad - k: pad - k + n]
            yield k, _mix(h)


def _padded(ids: np.ndarray, pad: int) -> np.ndarray:
    return np.concatenate([np.ones(pad, np.uint64), ids.astype(np.uint64) + U(1)])


def _count_one(P1, pad, n, h0, k0, k, order, raw_all):
    """The counts for context length k, continuing the unmixed running hash h0 of length k0 (< k)."""
    h = h0.copy()
    with np.errstate(over="ignore"):
        for j in range(k0 + 1, k + 1):
            h *= PRIME
            h += P1[pad - j: pad - j + n]
    nxt = P1[pad:] - U(1)
    key = _mix(h) << U(CBITS)
    key |= nxt
    if k < order:
        triples = key << U(CBITS)
        triples |= P1[pad - k - 1: pad - k - 1 + n]
        triples.sort()
        r = _runs(triples >> U(CBITS)) if raw_all else None
        return k, r, _runs(_runs(triples)[0] >> U(CBITS))
    return k, _runs(np.sort(key)), None


def _count_iter(P1, pad, n, ks, order, raw_all):
    """(k, raw (keys, counts) or None, continuation (keys, counts) or None) for each context length k in ks.

    One sort of packed (context, next, previous character) numbers gives the raw counts (keys come out sorted)
    and the distinct triples, i.e. the continuation counts."""
    nxt = P1[pad:] - U(1)
    want = set(ks)
    for k, h in _hashes(P1, max(ks), pad):
        if k not in want:
            continue
        key = h << U(CBITS)
        key |= nxt
        if k < order:
            triples = key << U(CBITS)
            triples |= P1[pad - k - 1: pad - k - 1 + n]          # previous character + 1 (padding = 1)
            triples.sort()
            r = _runs(triples >> U(CBITS)) if raw_all else None
            c = _runs(_runs(triples)[0] >> U(CBITS))
            del triples
        else:
            r, c = _runs(np.sort(key)), None
        yield k, r, c


class CharModel:
    """Character model with interpolated modified Kneser–Ney smoothing.

    The longest context uses real counts; shorter contexts use 'continuation' counts (in how many different
    contexts a character sequence appears), which is what makes Kneser–Ney better than plain back-off.
    Counting uses only plain sorts of 64-bit numbers: a (context, next, previous character) triple is packed
    into one number, so distinct triples, and hence continuation counts, come out of a sort."""

    def __init__(self, max_order: int = 10):
        self.max_order = max_order
        self.order = max_order
        self.chars: list[str] = []
        self.levels: list[Stats] = []     # levels[k-1] = context length k (continuation counts, except the last)
        self.report: dict = {}
        self._lut: dict | None = None

    # -- characters <-> ids (0 is padding; the rarest characters share the last id if there are too many)
    def _vocab(self, text: str) -> np.ndarray:
        cp = np.frombuffer(text.encode("utf-32-le", "surrogatepass"), dtype=np.uint32)
        freq = np.bincount(cp)
        present = np.flatnonzero(freq)
        if len(present) > CMASK - 1:
            present = np.sort(present[np.argsort(-freq[present], kind="stable")[: CMASK - 2]])
            self.rare = True
        else:
            self.rare = False
        self.chars = [chr(c) for c in present] + (["�"] if self.rare else [])
        return self._ids(cp)

    def _ids(self, cp: np.ndarray) -> np.ndarray:
        codes = np.array([ord(c) for c in self.chars[: len(self.chars) - self.rare]], np.uint32)
        lut = np.full(max(int(codes.max()) + 1 if len(codes) else 1, int(cp.max()) + 1 if len(cp) else 1), len(self.chars) if self.rare else 0, np.int64)
        lut[codes] = np.arange(1, len(codes) + 1)
        return lut[cp]

    def _encode(self, text: str) -> np.ndarray:
        return self._ids(np.frombuffer(text.encode("utf-32-le", "surrogatepass"), dtype=np.uint32))

    def _count(self, ids: np.ndarray, order: int, raw_all: bool = False, stop_at: float | None = None, keep_last_raw: bool = False):
        """Raw counts (all lengths if raw_all, else only the longest) and continuation counts for 1..order.

        Each context length is counted independently, so on big texts several lengths are counted at once in
        threads. The work is NumPy sorting and array arithmetic, which runs outside Python's global lock, and
        threads share the arrays instead of copying them between processes."""
        from concurrent.futures import ThreadPoolExecutor
        pad = order + 1
        P1 = _padded(ids, pad)
        n = len(ids)
        raw, cont = [None] * order, [None] * order
        got = order
        workers = max(1, min(order, os.cpu_count() or 1))
        if n >= PARALLEL_CHARS and workers > 1:
            # count `workers` lengths at a time, one thread each, every one continuing from the running hash;
            # between batches, check that the next batch still fits the time
            h, hk = np.zeros(n, np.uint64), 0
            while hk < order:
                t_batch = time.perf_counter()
                ks = list(range(hk + 1, min(order, hk + workers) + 1))
                with ThreadPoolExecutor(len(ks)) as ex:
                    done = list(ex.map(lambda k, h=h, hk=hk: _count_one(P1, pad, n, h, hk, k, order, raw_all or keep_last_raw), ks))
                for k, r, c in done:
                    raw[k - 1] = None if r is None else Stats(*r)
                    cont[k - 1] = None if c is None else Stats(*c)
                if keep_last_raw and not raw_all:          # only the longest length so far may become the model's top
                    for j in range(1, ks[-1]):
                        raw[j - 1] = None
                with np.errstate(over="ignore"):
                    for k in ks:
                        h = h * PRIME + P1[pad - k: pad - k + n]
                hk = ks[-1]
                if stop_at is not None and hk < order and time.perf_counter() + (time.perf_counter() - t_batch) > stop_at:
                    got = hk
                    break
            first = (raw[0] or cont[0]).keys
        else:
            for k, r, c in _count_iter(P1, pad, n, list(range(1, order + 1)), order, raw_all):
                raw[k - 1] = None if r is None else Stats(*r)
                cont[k - 1] = None if c is None else Stats(*c)
                if k == 1:
                    first = (raw[0] or cont[0]).keys
                if stop_at is not None and time.perf_counter() > stop_at and k < order and (raw_all or k == order):
                    got = k
                    break
        raw, cont = raw[:got], cont[:got]
        if got < order:
            cont[-1] = None
        V = len(self.chars)
        # unigram continuation: in how many different one-character contexts each character appears
        cu = np.bincount((first & U(CMASK)).astype(np.int64), minlength=V + 1).astype(np.float64)
        self.p0 = (cu + 0.5) / (cu.sum() + 0.5 * (V + 1))
        self.p0[0] = 0.0
        self.p0 /= self.p0.sum()
        return raw, cont, got

    def fit(self, text: str, progress=None, deadline: float | None = None) -> "CharModel":
        """Choose the context length on held-back text, then learn as much of the text as the time allows.

        The choice is made on up to 2 million characters from 20 evenly spaced places (90% learned, 10% held back).
        Then the whole text is learned once at the chosen length (all of it, unless the deadline says otherwise)."""
        t0 = time.perf_counter()
        say = progress or (lambda **k: None)
        text = text[:MAX_CHARS]
        if len(text) < 200:
            raise ValueError("The text is too short to learn from. Upload at least a few paragraphs.")
        ids = self._vocab(text)
        # the choice is made on (at most) 2 million characters taken from 20 evenly spaced places, each split 90/10,
        # so both the learned part and the held-back part represent the whole text, not just its beginning
        S = min(len(ids), 2_000_000)
        if deadline is not None:                   # a short budget gets a smaller sample, so every length can be tried
            S = min(S, max(100_000, int((deadline - t0) * 400_000)))
        pieces = 20 if len(ids) > 200_000 else 1
        size = S // pieces
        starts = np.linspace(0, len(ids) - size, pieces).astype(np.int64)
        tr = np.concatenate([ids[a: a + int(size * 0.9)] for a in starts])
        va = np.concatenate([ids[a + int(size * 0.9): a + size] for a in starts])
        budget = None if deadline is None else max(0.05, deadline - t0)
        say(stage="counting", text=f"Counting contexts of 1–{self.max_order} characters in {len(tr):,} characters…")
        t1 = time.perf_counter()
        raw, cont, got = self._count(tr, self.max_order, raw_all=True, stop_at=None if budget is None else t0 + 0.45 * budget)
        rate = len(tr) * got / max(time.perf_counter() - t1, 1e-6)      # characters × lengths per second (on the cautious side)
        if got < self.max_order:
            say(stage="counting", text=f"Out of time after contexts of {got} characters; choosing among those.")
        curve = self._curve(tr, va, raw, cont)
        best = int(np.argmin(curve)) + 1
        uni = np.bincount(tr, minlength=len(self.chars) + 1).astype(np.float64)[1:] + 0.5
        vv = va[va > 0]
        uni_bits = float(-np.log2(uni[vv - 1] / uni.sum()).mean()) if len(vv) else 0.0
        self.order = best
        self.levels = cont[: best - 1] + [raw[best - 1]]
        learned = len(tr)
        # then learn as much of the text as the measured speed says fits, at the chosen length
        left = None if deadline is None else deadline - time.perf_counter() - 0.03 * budget
        can = len(ids) if left is None else int(rate * max(left, 0) / best)
        if can >= min(len(ids), max(len(tr) * 1.05, 50_000)) or (pieces > 1 and can >= len(tr)):
            take = min(len(ids), can)
            say(stage="final", text=f"Best context: {best} characters. Learning {'all of the text' if take == len(ids) else f'the first {take:,} characters'} with it…")
            del raw, cont
            raw2, cont2, got2 = self._count(ids[:take], best, stop_at=None if deadline is None else deadline - 0.05 * budget, keep_last_raw=True)
            self.order = got2                       # lower than `best` only if a batch of lengths would not have fitted
            self.levels = cont2[: got2 - 1] + [raw2[got2 - 1]]
            learned = take
        else:
            say(stage="final", text=f"Best context: {best} characters. Keeping the model learned from the first {learned:,} characters (no time for more).")
        secs = time.perf_counter() - t0
        V = len(self.chars)
        self.report = {"chars": len(text), "learned_chars": learned, "vocab": V, "order": self.order, "seconds": round(secs, 3),
                       "chars_per_second": round(learned / max(secs, 1e-9)), "valid_bits_per_char": round(float(curve[self.order - 1]), 4),
                       "valid_chars": int(len(va)), "bits_by_order": [round(float(b), 4) for b in curve], "random_bits_per_char": round(math.log2(max(V, 2)), 4),
                       "letter_frequency_bits_per_char": round(uni_bits, 4), "contexts": int(sum(len(l.keys) for l in self.levels))}
        return self

    def _curve(self, tr, va, raw, cont) -> np.ndarray:
        """Bits per character on `va` (following `tr`) if the longest context were 1, 2, … characters."""
        order = len(raw)
        ctx = tr[-(order + 1):]
        P1 = _padded(np.concatenate([ctx, va]), order + 1 - len(ctx) + len(ctx))
        P1[: order + 1 - len(ctx)] = 1
        pad = len(P1) - len(va)
        target = va
        ok = target > 0
        p = self.p0[target]
        out = []
        for k, h in _hashes(P1, order, pad):
            top = raw[k - 1].apply(h, target, p)
            out.append(float(-np.log2(np.clip(top[ok], 1e-12, 1)).mean()) if ok.any() else 0.0)
            if cont[k - 1] is not None:
                p = cont[k - 1].apply(h, target, p)
        return np.array(out)

    def bits(self, text: str) -> float:
        """Bits per character the model needs for `text` (lower: the text looks more like what it learned)."""
        seq = self._encode(text)
        if not len(seq):
            return float("nan")
        pad = self.order + 1
        P1 = _padded(seq, pad)
        p = self.p0[seq]
        for (k, h), lv in zip(_hashes(P1, self.order, pad), self.levels):
            p = lv.apply(h, seq, p)
        ok = seq > 0
        return float(-np.log2(np.clip(p[ok], 1e-12, 1)).mean()) if ok.any() else float("nan")

    # -- writing
    def next_distribution(self, history: list[int]) -> np.ndarray:
        p = self.p0.copy()
        hist = [0] * max(0, self.order - len(history)) + history[-self.order:]
        h = U(0)
        with np.errstate(over="ignore"):
            for k, lv in enumerate(self.levels, 1):
                h = h * PRIME + U(hist[-k] + 1)
                p = lv.update(_mix(np.array([h]))[0], p)
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
        return {"format": 2, "chars": "".join(self.chars), "rare": self.rare, "order": self.order, "max_order": self.max_order,
                "disc": [list(l.disc) for l in self.levels], "report": self.report}, arrays

    @classmethod
    def from_state(cls, meta: dict, arrays: dict) -> "CharModel":
        if meta.get("format") != 2:
            raise ValueError("This text model was saved by an older version of the Lab. Train it again.")
        m = cls(meta["max_order"])
        m.chars, m.rare, m.order, m.report = list(meta["chars"]), meta["rare"], meta["order"], meta["report"]
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
        np.savez(folder / "weights.npz", **arrays)          # uncompressed: saving must not eat the time budget
        (folder / "model.json").write_text(json.dumps({"format": 2, "type": "text", "model": meta, "titles": self.titles, "report": self.report}))
        (folder / "passages.json").write_text(json.dumps(self.passages, ensure_ascii=False))
        return folder

    @classmethod
    def load(cls, folder: str | Path) -> "TextModel":
        folder = Path(folder)
        meta = json.loads((folder / "model.json").read_text())
        with np.load(folder / "weights.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        return cls(CharModel.from_state(meta["model"], arrays), json.loads((folder / "passages.json").read_text()), meta["titles"])


def train_text(docs: list[tuple[str, str]], max_order: int = 10, progress=None, stop=None, deadline: float | None = None) -> TextModel:
    """Learn the documents. With a `deadline` (a time.perf_counter() value) it learns as much as fits before then."""
    emit = progress or (lambda e: None)
    t0 = time.perf_counter()
    text = "\n\n".join(t.strip() for _, t in docs if t.strip())

    def say(stage, text):
        emit({"type": "stage", "stage": stage, "text": text, "t": round(time.perf_counter() - t0, 3)})

    say("reading", f"Read {len(docs)} document{'s' * (len(docs) != 1)}, {len(text):,} characters.")
    m = CharModel(max_order).fit(text, progress=lambda **k: say(k["stage"], k["text"]), deadline=deadline)
    passages = passages_of(docs)
    tm = TextModel(m, passages, [t for t, _ in docs])
    emit({"type": "done", "report": tm.report})
    return tm
