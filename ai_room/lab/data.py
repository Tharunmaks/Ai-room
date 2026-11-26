"""Read what people upload, work out what each column holds, and turn rows into numbers a model can learn from.

Everything here is fitted on the training rows only, so nothing about the validation or test rows leaks into training.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import zipfile
import zlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import numpy as np

MISSING = {"", "na", "n/a", "nan", "null", "none", "?", "-", "--", "missing", "#n/a"}
TABLE_EXT = {".csv", ".tsv", ".tab", ".json", ".jsonl", ".ndjson"}
TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".pdf", ".epub", ".docx", ".py", ".js", ".log"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
TEXT_DIM = 1 << 18          # hashed word features (shared by all text columns); models may fold them smaller
MAX_CATEGORIES = 40         # one-hot the most common categories; the rest share "other"
TOKEN = re.compile(r"\w+", re.UNICODE)


class DataError(ValueError):
    """The upload cannot be used as it is. The message says why, in plain words."""


def ext_of(filename: str) -> str:
    m = re.search(r"\.[A-Za-z0-9]+$", filename or "")
    return m.group(0).lower() if m else ""


def is_missing(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in MISSING)


def to_float(v) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v) if math.isfinite(v) else None
    s = str(v).strip().replace(",", "") if v is not None else ""
    if s.endswith("%"):
        s = s[:-1]
    if s.startswith("$") or s.startswith("€") or s.startswith("£"):
        s = s[1:]
    try:
        f = float(s)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


_DATE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?)?(Z|[+-]\d{2}:?\d{2})?$")


def to_date(v) -> datetime | None:
    s = str(v).strip()
    if not _DATE.match(s):
        return None
    try:
        return datetime.fromisoformat(s.replace("/", "-").replace("Z", "+00:00"))
    except ValueError:
        return None


# ------------------------------------------------------------------ reading uploads
def decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig",):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1")


def sniff_kind(filename: str, raw: bytes) -> str:
    """'table', 'text' or 'images' for an upload."""
    ext = ext_of(filename)
    if ext == ".zip":
        try:
            names = [n for n in zipfile.ZipFile(io.BytesIO(raw)).namelist() if not n.endswith("/")]
        except zipfile.BadZipFile:
            raise DataError("That .zip file is damaged or not really a zip.")
        if sum(ext_of(n) in IMAGE_EXT for n in names) >= max(2, len(names) // 2):
            return "images"
        if any(ext_of(n) in TABLE_EXT for n in names) and not any(ext_of(n) in TEXT_EXT for n in names):
            return "table"
        return "text"
    if ext in TABLE_EXT:
        return "table"
    if ext in (".txt", ""):
        head = decode(raw[:20000])
        lines = [l for l in head.splitlines()[:30] if l.strip()]
        for d in (",", "\t", ";", "|"):
            counts = [l.count(d) for l in lines]
            if len(lines) >= 3 and counts[0] >= 1 and len(set(counts)) == 1:
                return "table"
    return "text"


def read_table(raw: bytes, filename: str) -> tuple[list[str], list[list[str]]]:
    """Columns and rows (as strings) from CSV/TSV/JSON/JSON-lines, or the first table inside a zip."""
    ext = ext_of(filename)
    if ext == ".zip":
        z = zipfile.ZipFile(io.BytesIO(raw))
        inner = [n for n in z.namelist() if ext_of(n) in TABLE_EXT]
        if not inner:
            raise DataError("The zip has no .csv, .tsv or .json file in it.")
        return read_table(z.read(inner[0]), inner[0])
    text = decode(raw)
    if ext in (".json", ".jsonl", ".ndjson") or text.lstrip()[:1] in ("[", "{"):
        try:
            return _read_json(text)
        except (ValueError, DataError):
            if ext in (".json", ".jsonl", ".ndjson"):
                raise DataError("This JSON is not a list of records. Use a list of objects, or one object per line.")
    sample = text[:50000]
    delim = "\t" if ext in (".tsv", ".tab") else None
    if delim is None:
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delim = ","
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise DataError("The file needs a header row and at least one row of data.")
    header = [h.strip() for h in rows[0]]
    # a header row is mostly not numbers; if it is, the file has no header
    if sum(to_float(h) is not None for h in header) > len(header) / 2:
        header = [f"column {i + 1}" for i in range(len(header))]
    else:
        rows = rows[1:]
    header = _unique_names(header)
    w = len(header)
    rows = [(r + [""] * (w - len(r)))[:w] for r in rows]
    return header, rows


def _read_json(text: str) -> tuple[list[str], list[list[str]]]:
    t = text.strip()
    if t.startswith("["):
        items = json.loads(t)
    else:
        items = [json.loads(l) for l in t.splitlines() if l.strip()]
    if isinstance(items, dict):
        items = next((v for v in items.values() if isinstance(v, list)), [items])
    items = [it for it in items if isinstance(it, dict)]
    if not items:
        raise DataError("no records")
    cols: list[str] = []
    seen = set()
    for it in items[:2000]:
        for k in it:
            if k not in seen:
                seen.add(k)
                cols.append(str(k))
    def cell(v):
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return str(v)
    return cols, [[cell(it.get(c)) for c in cols] for it in items]


def _unique_names(names: list[str]) -> list[str]:
    out, seen = [], Counter()
    for i, n in enumerate(names):
        n = n or f"column {i + 1}"
        seen[n] += 1
        out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return out


PLAIN_EXT = {".txt", ".md", ".markdown", ".rst", ".py", ".js", ".log", ".text", ""}


def _one_text(filename: str, raw: bytes) -> tuple[str, str]:
    """(title, text). Plain text keeps its exact spacing and line breaks (they are part of its style)."""
    title = re.sub(r"\.[A-Za-z0-9]+$", "", filename.rsplit("/", 1)[-1]).replace("_", " ").strip() or filename
    if ext_of(filename) in PLAIN_EXT:
        return title, decode(raw).replace("\r\n", "\n").replace("\r", "\n")
    from ..teacher.library import extract_text
    text, t = extract_text(filename, raw)
    return t or title, text


def read_text(raw: bytes, filename: str) -> list[tuple[str, str]]:
    """(title, text) documents from a text upload; a zip may hold many."""
    if ext_of(filename) == ".zip":
        docs = []
        z = zipfile.ZipFile(io.BytesIO(raw))
        for n in sorted(z.namelist()):
            if n.endswith("/") or ext_of(n) not in TEXT_EXT | {""} or "/." in "/" + n:
                continue
            try:
                title, text = _one_text(n, z.read(n))
            except Exception:  # an unreadable file inside a zip should not sink the others
                continue
            if text.strip():
                docs.append((title, text))
        if not docs:
            raise DataError("The zip has no readable text files (txt, md, html, pdf, epub, docx).")
        return docs
    try:
        title, text = _one_text(filename, raw)
    except Exception as e:
        raise DataError(f"Could not read text from {filename}: {e}")
    if not text.strip():
        raise DataError("No text found in that file.")
    return [(title, text)]


# ------------------------------------------------------------------ what is in each column
ID_NAME = re.compile(r"(^|[\s_\-.])(id|uuid|guid|index|key)$|^unnamed: ?0$|^(row ?)?(no|num|number|#)$", re.I)


def profile(columns: list[str], rows: list[list[str]], sample: int = 4000) -> list[dict]:
    """For each column: its kind (number, category, text, date) or why it is ignored, and a few facts."""
    out = []
    n = len(rows)
    step = max(1, n // sample)
    for j, name in enumerate(columns):
        vals = [r[j] for r in rows[::step]]
        present = [v for v in vals if not is_missing(v)]
        info = {"name": name, "missing": round(1 - len(present) / max(1, len(vals)), 4)}
        if not present:
            out.append({**info, "kind": "empty", "why": "every value is missing"})
            continue
        uniq = Counter(v.strip() for v in present)
        info["unique"] = len(uniq) if step == 1 else len(Counter(r[j].strip() for r in rows if not is_missing(r[j])))
        info["examples"] = [v for v, _ in uniq.most_common(5)]
        if len(uniq) == 1 and info["unique"] == 1:
            out.append({**info, "kind": "constant", "why": "has the same value in every row"})
            continue
        nums = [to_float(v) for v in present]
        good = [f for f in nums if f is not None]
        if len(good) >= 0.97 * len(present):
            ints = all(f == int(f) for f in good)
            kind = "number"
            if ID_NAME.search(name) and ints and info["unique"] >= 0.95 * (n - info["missing"] * n):
                out.append({**info, "kind": "id", "why": "looks like a row number or ID"})
                continue
            out.append({**info, "kind": kind, "min": min(good), "max": max(good), "mean": sum(good) / len(good), "integers": ints})
            continue
        if sum(to_date(v) is not None for v in present[:500]) >= 0.9 * min(500, len(present)):
            out.append({**info, "kind": "date"})
            continue
        avg_len = sum(len(v) for v in present) / len(present)
        avg_words = sum(len(TOKEN.findall(v)) for v in present) / len(present)
        ratio = info["unique"] / max(1, n * (1 - info["missing"]))
        if (avg_words >= 4 or avg_len >= 30) and ratio > 0.2:
            out.append({**info, "kind": "text", "avg_words": round(avg_words, 1)})
        elif ratio >= 0.95 and info["unique"] > 30:
            out.append({**info, "kind": "id", "why": "almost every value is different, like a name or ID"})
        else:
            out.append({**info, "kind": "category"})
    return out


def pick_target(prof: list[dict]) -> str | None:
    usable = [p for p in prof if p["kind"] in ("number", "category")]
    names = {p["name"].strip().lower() for p in prof}
    wanted = ["target", "label", "labels", "class", "outcome", "result", "category", "species", "price", "sentiment"]
    if "x" not in names:                  # next to an 'x' column, 'y' is a coordinate rather than the answer
        wanted.append("y")
    for p in usable:
        if p["name"].strip().lower() in wanted:
            return p["name"]
    return usable[-1]["name"] if usable else (prof[-1]["name"] if prof else None)


def pick_task(target_profile: dict) -> str:
    p = target_profile
    if p["kind"] == "number" and not (p.get("integers") and p.get("unique", 99) <= 20):
        return "regression"
    return "classification"


# ------------------------------------------------------------------ rows -> numbers
@dataclass
class XData:
    """Dense features (numbers, categories, dates) plus optional hashed text features, stored sparsely."""
    dense: np.ndarray                      # (n, d) float32
    sp_indptr: np.ndarray | None = None    # CSR rows of the hashed text features
    sp_indices: np.ndarray | None = None
    sp_data: np.ndarray | None = None

    @property
    def n(self) -> int:
        return self.dense.shape[0]

    @property
    def has_text(self) -> bool:
        return self.sp_indptr is not None

    @property
    def width(self) -> int:
        return self.dense.shape[1] + (TEXT_DIM if self.has_text else 0)

    def take(self, idx: np.ndarray) -> "XData":
        if not self.has_text:
            return XData(self.dense[idx])
        starts, ends = self.sp_indptr[idx], self.sp_indptr[idx + 1]
        lens = ends - starts
        pos = _ranges(starts, lens)
        return XData(self.dense[idx], np.concatenate([[0], np.cumsum(lens)]).astype(np.int64), self.sp_indices[pos], self.sp_data[pos])


def _ranges(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    offs = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
    return offs + np.arange(total)


def _date_parts(d: datetime | None) -> list[float]:
    if d is None:
        return [math.nan] * 5
    return [d.timestamp() / 86400.0 if d.tzinfo else (d - datetime(1970, 1, 1)).total_seconds() / 86400.0, d.year, d.month, d.weekday(), d.hour + d.minute / 60]


DATE_PARTS = ["days", "year", "month", "weekday", "hour"]


def text_tokens(s: str) -> list[str]:
    words = TOKEN.findall(s.lower())
    return words + [a + " " + b for a, b in zip(words, words[1:])]


class Encoder:
    """Turns rows (lists of strings) into XData. Fitted on training rows only."""

    def __init__(self, columns: list[str], kinds: dict[str, str]):
        self.columns = columns
        self.kinds = {c: k for c, k in kinds.items() if k in ("number", "category", "text", "date")}
        self.spec: list[dict] = []
        self.idf: np.ndarray | None = None
        self.names: list[str] = []      # name of each dense feature
        self.owner: list[str] = []      # which input column each dense feature came from

    # -- fitting
    def fit(self, rows: list[list[str]]) -> "Encoder":
        self.spec, self.names, self.owner = [], [], []
        idx = {c: i for i, c in enumerate(self.columns)}
        n = len(rows)
        for c, kind in self.kinds.items():
            j = idx[c]
            vals = [r[j] for r in rows]
            if kind == "number":
                f = np.array([np.nan if is_missing(v) else (to_float(v) if to_float(v) is not None else np.nan) for v in vals], np.float64)
                ok = f[~np.isnan(f)]
                if not len(ok):
                    continue
                log = bool(ok.min() >= 0 and ok.max() > 50 and _skew(ok) > 2)
                t = np.log1p(ok) if log else ok
                mu, sd = float(t.mean()), float(t.std() or 1.0)
                miss = bool(np.isnan(f).any())
                self.spec.append({"col": c, "j": j, "kind": "number", "log": log, "mu": mu, "sd": sd, "missing": miss})
                self._name(c, [c] + ([f"{c} is missing"] if miss else []))
            elif kind == "date":
                parts = np.array([_date_parts(to_date(v)) for v in vals], np.float64)
                mus, sds = [], []
                for k in range(parts.shape[1]):
                    col = parts[:, k][~np.isnan(parts[:, k])]
                    mus.append(float(col.mean()) if len(col) else 0.0)
                    sds.append(float(col.std() or 1.0) if len(col) else 1.0)
                keep = [k for k in range(parts.shape[1]) if sds[k] > 1e-9 and len(set(parts[:, k][~np.isnan(parts[:, k])].round(6))) > 1]
                if not keep:
                    continue
                self.spec.append({"col": c, "j": j, "kind": "date", "parts": keep, "mu": [mus[k] for k in keep], "sd": [sds[k] for k in keep]})
                self._name(c, [f"{c} ({DATE_PARTS[k]})" for k in keep])
            elif kind == "category":
                counts = Counter("" if is_missing(v) else v.strip() for v in vals)
                min_count = 2 if n >= 50 else 1
                cats = [v for v, k in counts.most_common(MAX_CATEGORIES) if k >= min_count]
                if not cats:
                    continue
                other = len(cats) < len(counts)
                self.spec.append({"col": c, "j": j, "kind": "category", "cats": cats, "other": other})
                self._name(c, [f"{c} = {v or '(missing)'}" for v in cats] + ([f"{c} = (other)"] if other else []))
            elif kind == "text":
                self.spec.append({"col": c, "j": j, "kind": "text", "salt": zlib.crc32(c.encode()) & 0xFFFF})
        if any(s["kind"] == "text" for s in self.spec):
            df = np.zeros(TEXT_DIM, np.float64)
            for cols in self._hash_rows(rows):
                df[np.unique(cols)] += 1
            self.idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
        if not self.spec:
            raise DataError("None of the columns can be used to learn from (they are all empty, constant, or IDs).")
        return self

    def _name(self, col: str, names: list[str]) -> None:
        self.names += names
        self.owner += [col] * len(names)

    def _hash_rows(self, rows):
        texts = [s for s in self.spec if s["kind"] == "text"]
        for r in rows:
            cols = []
            for s in texts:
                v = r[s["j"]]
                if is_missing(v):
                    continue
                salt = s["salt"].to_bytes(2, "little")
                cols += [zlib.crc32(salt + t.encode()) & (TEXT_DIM - 1) for t in text_tokens(v)]
            yield np.array(cols, np.int64)

    # -- transforming
    def transform(self, rows: list[list[str]]) -> XData:
        n = len(rows)
        blocks = []
        for s in self.spec:
            j = s["j"]
            if s["kind"] == "number":
                f = np.array([np.nan if is_missing(r[j]) else (to_float(r[j]) if to_float(r[j]) is not None else np.nan) for r in rows], np.float64)
                miss = np.isnan(f)
                if s["log"]:
                    f = np.log1p(np.maximum(f, 0))
                z = np.clip((f - s["mu"]) / s["sd"], -6, 6)
                z[miss] = 0.0
                blocks.append(z[:, None])
                if s["missing"]:
                    blocks.append(miss[:, None].astype(np.float64))
            elif s["kind"] == "date":
                parts = np.array([_date_parts(to_date(r[j])) for r in rows], np.float64).reshape(n, -1)[:, s["parts"]]
                z = np.clip((parts - np.array(s["mu"])) / np.array(s["sd"]), -6, 6)
                blocks.append(np.nan_to_num(z, nan=0.0))
            elif s["kind"] == "category":
                pos = {v: i for i, v in enumerate(s["cats"])}
                width = len(s["cats"]) + (1 if s["other"] else 0)
                m = np.zeros((n, width))
                for i, r in enumerate(rows):
                    v = "" if is_missing(r[j]) else r[j].strip()
                    k = pos.get(v, len(s["cats"]) if s["other"] else None)
                    if k is not None:
                        m[i, k] = 1.0
                blocks.append(m)
        dense = (np.hstack(blocks) if blocks else np.zeros((n, 0))).astype(np.float32)
        if self.idf is None:
            return XData(dense)
        indptr, indices, data = [0], [], []
        for cols in self._hash_rows(rows):
            if len(cols):
                u, cnt = np.unique(cols, return_counts=True)
                w = (1 + np.log(cnt)) * self.idf[u]
                w = w / (np.sqrt((w * w).sum()) or 1.0)
                indices.append(u)
                data.append(w.astype(np.float32))
                indptr.append(indptr[-1] + len(u))
            else:
                indptr.append(indptr[-1])
        return XData(dense, np.array(indptr, np.int64), np.concatenate(indices) if indices else np.zeros(0, np.int64),
                     np.concatenate(data) if data else np.zeros(0, np.float32))

    # -- saving
    def state(self) -> tuple[dict, dict]:
        return {"columns": self.columns, "kinds": self.kinds, "spec": self.spec, "names": self.names, "owner": self.owner}, \
            ({"idf": self.idf} if self.idf is not None else {})

    @classmethod
    def from_state(cls, meta: dict, arrays: dict) -> "Encoder":
        e = cls(meta["columns"], meta["kinds"])
        e.spec, e.names, e.owner = meta["spec"], meta["names"], meta["owner"]
        e.idf = arrays.get("idf")
        return e

    def column_slices(self) -> dict[str, list[int]]:
        """Dense feature positions per input column (text columns have none: they live in the sparse part)."""
        out: dict[str, list[int]] = {}
        for i, c in enumerate(self.owner):
            out.setdefault(c, []).append(i)
        return out


def _skew(a: np.ndarray) -> float:
    sd = a.std()
    return float(((a - a.mean()) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
