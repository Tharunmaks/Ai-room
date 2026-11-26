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
from itertools import chain
from dataclasses import dataclass
from datetime import datetime

import numpy as np

MISSING = {"", "na", "n/a", "nan", "null", "none", "?", "-", "--", "missing", "#n/a"}
TABLE_EXT = {".csv", ".tsv", ".tab", ".json", ".jsonl", ".ndjson", ".xlsx", ".xlsm"}
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
    """Columns and rows (as strings) from CSV/TSV/JSON/JSON-lines/Excel, or the first table inside a zip."""
    ext = ext_of(filename)
    if ext in (".xlsx", ".xlsm"):
        from .xlsx import read_xlsx
        return read_xlsx(io.BytesIO(raw))
    if ext == ".xls":
        raise DataError("Old .xls workbooks are not supported. In Excel, use File → Save As → Excel Workbook (.xlsx) or CSV.")
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


# ------------------------------------------------------------------ tables as columns
class Table:
    """Columns of a table: each a NumPy array of strings (dtype object) or, for clean numbers, float64."""

    def __init__(self, columns: list[str], data: list[np.ndarray], total_rows: int | None = None, exact: bool = True, sampled: bool = False):
        self.columns, self.data = list(columns), data
        self.total_rows, self.exact, self.sampled = (len(data[0]) if data else 0) if total_rows is None else total_rows, exact, sampled

    @property
    def n(self) -> int:
        return len(self.data[0]) if self.data else 0

    @classmethod
    def from_rows(cls, columns: list[str], rows: list) -> "Table":
        w = len(columns)
        clean = all(isinstance(r, list) and len(r) == w for r in rows[:1000]) and all(len(r) == w for r in rows)
        if not clean:
            rows = [(["" if v is None else str(v) for v in r] + [""] * w)[:w] for r in rows]
        data = []
        for c in (zip(*rows) if rows else [()] * w):
            a = np.empty(len(c), dtype=object)
            a[:] = c
            data.append(a)
        return cls(columns, data, len(rows))

    def col(self, name: str) -> np.ndarray:
        return self.data[self.columns.index(name)]

    def take(self, idx) -> "Table":
        return Table(self.columns, [c[idx] for c in self.data], self.total_rows, self.exact, self.sampled)

    def rows(self, limit: int | None = None) -> list[list[str]]:
        k = self.n if limit is None else min(limit, self.n)
        return [[cell_str(c[i]) for c in self.data] for i in range(k)]


def cell_str(v) -> str:
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        return str(int(v)) if float(v).is_integer() and abs(v) < 1e15 else repr(float(v))
    return "" if v is None else str(v)


def _mapped(col: np.ndarray, fn, dtype) -> np.ndarray:
    """fn applied to every value, computed once per distinct value (columns repeat values a lot)."""
    table = {v: fn(v) for v in set(col.tolist())}
    return np.fromiter(map(table.__getitem__, col.tolist()), dtype=dtype, count=len(col))


def to_floats(col: np.ndarray) -> np.ndarray:
    """Numbers from a column; missing or unreadable values become NaN."""
    if col.dtype != object:
        out = col.astype(np.float64)
    else:
        try:
            out = col.astype(np.float64)                       # clean numbers: parsed in C
        except (ValueError, TypeError):
            out = _mapped(col, lambda v: np.nan if is_missing(v) else (to_float(v) if to_float(v) is not None else np.nan), np.float64)
    out[~np.isfinite(out)] = np.nan
    return out


def missing_mask(col: np.ndarray) -> np.ndarray:
    if col.dtype != object:
        return np.isnan(col)
    return _mapped(col, is_missing, bool)


def _runs(s: np.ndarray):
    if not len(s):
        return s, np.zeros(0, np.int64)
    starts = np.concatenate([[0], np.flatnonzero(s[1:] != s[:-1]) + 1])
    return s[starts], np.diff(np.append(starts, len(s)))


# ------------------------------------------------------------------ what is in each column
ID_NAME = re.compile(r"(^|[\s_\-.])(id|uuid|guid|index|key)$|^unnamed: ?0$|^(row ?)?(no|num|number|#)$", re.I)


def profile(table, rows: list[list[str]] | None = None, sample: int = 4000) -> list[dict]:
    """For each column: its kind (number, category, text, date) or why it is ignored, and a few facts.

    Takes a Table, or (columns, rows). Kinds are judged on up to `sample` values per column; counts of distinct
    values use the whole column."""
    if rows is not None or not isinstance(table, Table):
        table = Table.from_rows(table, rows or [])
    out = []
    n = table.n
    step = max(1, n // sample)
    for j, name in enumerate(table.columns):
        col = table.data[j]
        if col.dtype != object:
            out.append(_profile_numeric(name, col))
            continue
        vals = col[::step].tolist()
        present = [v for v in vals if not is_missing(v)]
        info = {"name": name, "missing": round(1 - len(present) / max(1, len(vals)), 4)}
        if not present:
            out.append({**info, "kind": "empty", "why": "every value is missing"})
            continue
        uniq = Counter(v.strip() for v in present)
        if step == 1:
            info["unique"] = len(uniq)
        else:
            whole = set(col.tolist())
            info["unique"] = len(whole) - ("" in whole)
        info["examples"] = [v for v, _ in uniq.most_common(5)]
        if len(uniq) == 1 and info["unique"] <= 1:
            out.append({**info, "kind": "constant", "why": "has the same value in every row"})
            continue
        nums = [to_float(v) for v in present]
        good = [f for f in nums if f is not None]
        if len(good) >= 0.97 * len(present):
            ints = all(f == int(f) for f in good)
            if ID_NAME.search(name) and ints and info["unique"] >= 0.95 * (n - info["missing"] * n):
                out.append({**info, "kind": "id", "why": "looks like a row number or ID"})
                continue
            out.append({**info, "kind": "number", "min": min(good), "max": max(good), "mean": sum(good) / len(good), "integers": ints})
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


def _profile_numeric(name: str, col: np.ndarray) -> dict:
    ok = col[~np.isnan(col)]
    info = {"name": name, "missing": round(1 - len(ok) / max(1, len(col)), 4)}
    if not len(ok):
        return {**info, "kind": "empty", "why": "every value is missing"}
    u, _ = _runs(np.sort(ok))
    info["unique"] = int(len(u))
    info["examples"] = [cell_str(v) for v in u[:5]]
    if len(u) == 1:
        return {**info, "kind": "constant", "why": "has the same value in every row"}
    ints = bool(np.all(ok == np.round(ok)))
    if ID_NAME.search(name) and ints and len(u) >= 0.95 * len(ok):
        return {**info, "kind": "id", "why": "looks like a row number or ID"}
    return {**info, "kind": "number", "min": float(ok.min()), "max": float(ok.max()), "mean": float(ok.mean()), "integers": ints}


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
    """Turns table columns into XData. Fitted on training rows only. Works column by column; every distinct
    value is converted once, which is what makes millions of rows fast."""

    def __init__(self, columns: list[str], kinds: dict[str, str]):
        self.columns = columns
        self.kinds = {c: k for c, k in kinds.items() if k in ("number", "category", "text", "date")}
        self.spec: list[dict] = []
        self.idf: np.ndarray | None = None
        self.names: list[str] = []      # name of each dense feature
        self.owner: list[str] = []      # which input column each dense feature came from
        self._memo: dict[str, int] = {}

    # -- fitting
    def fit(self, rows) -> "Encoder":
        t = rows if isinstance(rows, Table) else Table.from_rows(self.columns, rows)
        self.spec, self.names, self.owner = [], [], []
        idx = {c: i for i, c in enumerate(self.columns)}
        n = t.n
        for c, kind in self.kinds.items():
            j = idx[c]
            col = t.data[j]
            if kind == "number":
                f = to_floats(col)
                ok = f[~np.isnan(f)]
                if not len(ok):
                    continue
                log = bool(ok.min() >= 0 and ok.max() > 50 and _skew(ok) > 2)
                tr = np.log1p(ok) if log else ok
                mu, sd = float(tr.mean()), float(tr.std() or 1.0)
                miss = bool(len(ok) < len(f))
                self.spec.append({"col": c, "j": j, "kind": "number", "log": log, "mu": mu, "sd": sd, "missing": miss})
                self._name(c, [c] + ([f"{c} is missing"] if miss else []))
            elif kind == "date":
                parts = self._dates(col)
                mus, sds, keep = [], [], []
                for k in range(parts.shape[1]):
                    v = parts[:, k][~np.isnan(parts[:, k])]
                    mus.append(float(v.mean()) if len(v) else 0.0)
                    sds.append(float(v.std() or 1.0) if len(v) else 1.0)
                    if len(v) and float(v.std()) > 1e-9:
                        keep.append(k)
                if not keep:
                    continue
                self.spec.append({"col": c, "j": j, "kind": "date", "parts": keep, "mu": [mus[k] for k in keep], "sd": [sds[k] for k in keep]})
                self._name(c, [f"{c} ({DATE_PARTS[k]})" for k in keep])
            elif kind == "category":
                raw = Counter(col.tolist() if col.dtype == object else [cell_str(v) for v in col])
                counts = Counter()
                for v, k in raw.items():
                    counts["" if is_missing(v) else str(v).strip()] += k
                min_count = 2 if n >= 50 else 1
                cats = [v for v, k in counts.most_common(MAX_CATEGORIES) if k >= min_count]
                if not cats:
                    continue
                other = len(cats) < len(counts)
                self.spec.append({"col": c, "j": j, "kind": "category", "cats": cats, "other": other})
                self._name(c, [f"{c} = {v or '(missing)'}" for v in cats] + ([f"{c} = (other)"] if other else []))
            elif kind == "text":
                self.spec.append({"col": c, "j": j, "kind": "text", "salt": zlib.crc32(c.encode()) & 0xFFFF, "v": 2})
        if any(s["kind"] == "text" for s in self.spec):
            u = _runs(np.sort(self._text_keys(t)))[0]                 # distinct (row, word) pairs
            df = np.bincount((u % TEXT_DIM).astype(np.int64), minlength=TEXT_DIM).astype(np.float64)
            self.idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
        if not self.spec:
            raise DataError("None of the columns can be used to learn from (they are all empty, constant, or IDs).")
        return self

    def _name(self, col: str, names: list[str]) -> None:
        self.names += names
        self.owner += [col] * len(names)

    @staticmethod
    def _dates(col: np.ndarray) -> np.ndarray:
        if col.dtype != object:
            return np.full((len(col), len(DATE_PARTS)), np.nan)
        table = {v: _date_parts(to_date(v)) for v in set(col.tolist())}
        return np.array([table[v] for v in col.tolist()], np.float64).reshape(len(col), len(DATE_PARTS))

    def _text_keys(self, t: Table) -> np.ndarray:
        """row × TEXT_DIM + bucket for every word and word pair of every text column, computed a column at a time:
        each distinct word is hashed once (CRC-32, remembered), word pairs are mixed from their two words' hashes."""
        n = t.n
        keys = []
        memos = getattr(self, "_memos", None)
        if memos is None:
            memos = self._memos = {}
        for s in self.spec:
            if s["kind"] != "text":
                continue
            salt = s["salt"].to_bytes(2, "little")
            memo = memos.get(s["col"])
            if memo is None:
                memo = memos[s["col"]] = _Memo(salt)
            toks = [TOKEN.findall(v.lower()) if isinstance(v, str) and not is_missing(v) else [] for v in t.data[s["j"]].tolist()]
            lens = np.fromiter(map(len, toks), np.int64, count=n)
            total = int(lens.sum())
            if not total:
                continue
            h = np.fromiter(map(memo.__getitem__, chain.from_iterable(toks)), np.uint64, count=total)
            rows = np.repeat(np.arange(n, dtype=np.uint64), lens)
            keys.append(rows * U64(TEXT_DIM) + (h & U64(TEXT_DIM - 1)))
            has_next = np.ones(total, bool)
            has_next[np.cumsum(lens)[lens > 0] - 1] = False          # the last word of a row starts no pair
            i = np.flatnonzero(has_next)
            with np.errstate(over="ignore"):
                pair = (h[i] * U64(0x9E3779B97F4A7C15)) ^ (h[i + 1] * U64(0xC2B2AE3D27D4EB4F)) ^ U64(s["salt"] * 0x165667B1)
            keys.append(rows[i] * U64(TEXT_DIM) + ((pair >> U64(23)) & U64(TEXT_DIM - 1)))
        return np.concatenate(keys) if keys else np.zeros(0, np.uint64)

    def _bucket(self, salt: bytes, tok: str) -> int:
        key = salt + tok.encode()
        b = self._memo.get(key)
        if b is None:
            b = self._memo[key] = zlib.crc32(key) & (TEXT_DIM - 1)
        return b

    def _hash_rows(self, t: Table):
        texts = [(t.data[s["j"]], s["salt"].to_bytes(2, "little")) for s in self.spec if s["kind"] == "text"]
        memo = getattr(self, "_memo", None)
        if memo is None:
            self._memo = memo = {}
        for i in range(t.n):
            cols = []
            for col, salt in texts:
                v = col[i]
                if isinstance(v, str) and not is_missing(v):
                    cols += [self._bucket(salt, tok) for tok in text_tokens(v)]
            yield np.array(cols, np.int64)

    # -- transforming
    def transform(self, rows) -> XData:
        t = rows if isinstance(rows, Table) else Table.from_rows(self.columns, rows)
        n = t.n
        blocks = []
        for s in self.spec:
            col = t.data[s["j"]]
            if s["kind"] == "number":
                f = to_floats(col)
                miss = np.isnan(f)
                if s["log"]:
                    f = np.log1p(np.maximum(f, 0))
                z = np.clip((f - s["mu"]) / s["sd"], -6, 6)
                z[miss] = 0.0
                blocks.append(z[:, None])
                if s["missing"]:
                    blocks.append(miss[:, None].astype(np.float64))
            elif s["kind"] == "date":
                parts = self._dates(col)[:, s["parts"]]
                z = np.clip((parts - np.array(s["mu"])) / np.array(s["sd"]), -6, 6)
                blocks.append(np.nan_to_num(z, nan=0.0))
            elif s["kind"] == "category":
                pos = {v: i for i, v in enumerate(s["cats"])}
                none = len(s["cats"]) if s["other"] else -1
                vals = col.tolist() if col.dtype == object else [cell_str(v) for v in col]
                code_of = {v: pos.get("" if is_missing(v) else str(v).strip(), none) for v in set(vals)}
                codes = np.fromiter(map(code_of.__getitem__, vals), np.int64, count=n)
                m = np.zeros((n, len(s["cats"]) + (1 if s["other"] else 0)), np.float32)
                ok = codes >= 0
                m[np.flatnonzero(ok), codes[ok]] = 1.0
                blocks.append(m)
        dense = (np.hstack([b.astype(np.float32, copy=False) for b in blocks]) if blocks else np.zeros((n, 0))).astype(np.float32)
        if self.idf is None:
            return XData(dense)
        if all(s.get("v") == 2 for s in self.spec if s["kind"] == "text"):
            u, cnt = _runs(np.sort(self._text_keys(t)))
            rows, cols = (u // TEXT_DIM).astype(np.int64), (u % TEXT_DIM).astype(np.int64)
            w = (1 + np.log(cnt)) * self.idf[cols]
            norm = np.sqrt(np.bincount(rows, weights=w * w, minlength=n))
            w = (w / np.where(norm[rows] > 0, norm[rows], 1.0)).astype(np.float32)
            indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n))]).astype(np.int64)
            return XData(dense, indptr, cols, w)
        indptr, indices, data = [0], [], []                    # models saved before version 2 hash row by row
        for cols in self._hash_rows(t):
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


U64 = np.uint64


class _Memo(dict):
    """word → its CRC-32 (salted per column), computed the first time a word is seen."""

    def __init__(self, salt: bytes):
        super().__init__()
        self.salt = salt

    def __missing__(self, tok: str) -> int:
        h = self[tok] = zlib.crc32(self.salt + tok.encode())
        return h


def _skew(a: np.ndarray) -> float:
    sd = a.std()
    return float(((a - a.mean()) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
