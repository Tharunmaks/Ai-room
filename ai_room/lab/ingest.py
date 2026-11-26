"""Reading uploads of any size (up to terabytes) without running out of memory.

* Uploads are streamed to disk in chunks; nothing holds the whole file in memory.
* A first look (columns, kinds, preview, row count) reads only the head of the file.
* Training loads the table column by column. All-numeric files take NumPy's C parser (tens of MB/s);
  other files go through the csv module. A file with more cells than fit in memory is *sampled*: rows are read
  from random places in the file, so a terabyte-sized file is learned from a fair sample in minutes, not days.
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import zipfile
from pathlib import Path

import numpy as np

from .data import DataError, Table, decode, ext_of, read_table, to_float

csv.field_size_limit(min(2**31 - 1, 1 << 40))     # long text cells (whole articles) are fine
HEAD_BYTES = 4 << 20
CHUNK = 8 << 20


def max_cells() -> int:
    """How many table cells to hold in memory at once (about 70 bytes each as text), from the RAM there is."""
    env = os.environ.get("AI_ROOM_LAB_MAX_CELLS")
    if env:
        return int(float(env))
    try:
        avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        avail = 4 << 30
    return int(np.clip(avail * 0.25 / 70, 2_000_000, 120_000_000))


# ------------------------------------------------------------------ saving uploads
def save_stream(chunks, dest: Path, limit_bytes: int, declared: int | None = None) -> int:
    """Write an iterable of byte chunks to `dest`, refusing more than `limit_bytes` or more than the disk can hold."""
    import shutil
    dest.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest.parent).free
    if declared is not None and declared > limit_bytes:
        raise DataError(f"That file is {human(declared)}, more than the {human(limit_bytes)} limit per upload.")
    if declared is not None and declared > free - (256 << 20):
        raise DataError(f"Not enough disk space: the file is {human(declared)} but only {human(free)} is free here.")
    size = 0
    try:
        with open(dest, "wb") as f:
            for chunk in chunks:
                size += len(chunk)
                if size > limit_bytes:
                    raise DataError(f"That file is more than the {human(limit_bytes)} limit per upload.")
                if size > free - (256 << 20):
                    raise DataError(f"Ran out of disk space after {human(size)} ({human(free)} was free).")
                f.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return size


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


# ------------------------------------------------------------------ first look
def sniff_path(path: Path, filename: str) -> str:
    """'table', 'text' or 'images', looking only at the start of the file (or a zip's list of names)."""
    from .data import IMAGE_EXT, TEXT_EXT, TABLE_EXT, sniff_kind
    if ext_of(filename) == ".zip":
        try:
            names = [n for n in zipfile.ZipFile(path).namelist() if not n.endswith("/")]
        except zipfile.BadZipFile:
            raise DataError("That .zip file is damaged or not really a zip.")
        if sum(ext_of(n) in IMAGE_EXT for n in names) >= max(2, len(names) // 2):
            return "images"
        if any(ext_of(n) in TABLE_EXT for n in names) and not any(ext_of(n) in TEXT_EXT for n in names):
            return "table"
        return "text"
    with open(path, "rb") as f:
        return sniff_kind(filename, f.read(64 << 10))


def head_text(path: Path, nbytes: int = HEAD_BYTES) -> tuple[str, bool]:
    """Decoded start of a file, cut at the last full line; and whether that is the whole file."""
    with open(path, "rb") as f:
        raw = f.read(nbytes + 1)
    whole = len(raw) <= nbytes
    if not whole:
        raw = raw[:nbytes]
        cut = raw.rfind(b"\n")
        if cut > 0:
            raw = raw[: cut + 1]
    return decode(raw), whole


def _dialect(text: str, filename: str) -> str:
    if ext_of(filename) in (".tsv", ".tab"):
        return "\t"
    try:
        return csv.Sniffer().sniff(text[:50000], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def inspect_table(path: Path, filename: str) -> tuple[list[str], list[list[str]], int, bool]:
    """(columns, rows from the head, total rows, exact?) without reading a big file entirely."""
    size = path.stat().st_size
    ext = ext_of(filename)
    if ext in (".xlsx", ".xlsm") and size > 64 << 20:          # big workbook: read the first rows, ask the sheet how many it has
        from .xlsx import Workbook, read_xlsx
        cols, rows = read_xlsx(path, limit=2000)
        total = Workbook(path).dimension_rows()
        return cols, rows, (total - 1 if total else len(rows)), False
    if ext == ".zip" or ext == ".json" or size <= 64 << 20:
        if ext == ".json" and size > 2 << 30:
            raise DataError("JSON arrays bigger than 2 GB cannot be read in pieces. Save the data as JSON-lines (one record per line) or CSV.")
        if ext == ".zip" and size > 4 << 30:
            raise DataError("Zipped tables bigger than 4 GB are not supported. Upload the CSV itself.")
        cols, rows = read_table(path.read_bytes(), filename)
        return cols, rows, len(rows), True
    text, _ = head_text(path)
    cols, rows = read_table(text.encode(), filename if ext else "head.csv")
    est = int(size / max(1, len(text.encode())) * len(rows))
    return cols, rows, est, False


# ------------------------------------------------------------------ loading a table for training
def load_table(path: Path, filename: str, limit_cells: int | None = None, seed: int = 0, say=None) -> Table:
    """The table, column by column; sampled by random seeks if it has more cells than fit in memory."""
    say = say or (lambda text: None)
    limit_cells = limit_cells or max_cells()
    path = Path(path)
    size = path.stat().st_size
    ext = ext_of(filename)
    if ext in (".xlsx", ".xlsm"):
        from .xlsx import read_xlsx
        cols, rows = read_xlsx(path)
        return _cap(Table.from_rows(cols, rows), limit_cells, seed, say)
    if ext in (".json", ".zip") or size <= 8 << 20:
        if ext == ".json" and size > 2 << 30:
            raise DataError("JSON arrays bigger than 2 GB cannot be read in pieces. Save the data as JSON-lines (one record per line) or CSV.")
        cols, rows = read_table(path.read_bytes(), filename)
        t = Table.from_rows(cols, rows)
        return _cap(t, limit_cells, seed, say)
    head, _ = head_text(path)
    jsonl = ext in (".jsonl", ".ndjson") or head.lstrip()[:1] == "{"
    if jsonl:
        cols, sample_rows = read_table(head.encode(), "head.jsonl")
    else:
        delim = _dialect(head, filename)
        cols, sample_rows = read_table(head.encode(), "head.tsv" if delim == "\t" else "head.csv")
    has_header = not cols[0].startswith("column 1") if cols else True
    head_rows = max(1, len(sample_rows))
    est_rows = int(size / max(1, len(head.encode())) * head_rows)
    width = max(1, len(cols))
    if est_rows * width > limit_cells:
        want = max(1000, limit_cells // width)
        multiline = not jsonl and _multiline(head, delim)
        say(f"The file has about {est_rows:,} rows — more than fit in memory. Learning from a random sample of {want:,} rows.")
        if multiline:
            t = _reservoir(path, cols, delim, want, seed, has_header)
        elif want > 0.05 * est_rows:            # a big share of the rows: one sequential pass is faster than seeking
            t = _bernoulli(path, cols, "json" if jsonl else delim, want / est_rows, seed, has_header and not jsonl, want)
        else:
            t = _seek_sample(path, cols, "json" if jsonl else delim, want, seed, len(head.encode()) if size > HEAD_BYTES else 0, sample_rows)
        t.total_rows, t.exact, t.sampled = est_rows, False, True
        return t
    if not jsonl and _all_numeric(sample_rows):
        try:
            arr = np.loadtxt(path, delimiter=delim, skiprows=1 if has_header else 0, dtype=np.float64, ndmin=2, quotechar='"', encoding="utf-8")
            if arr.shape[1] == width:
                return Table(cols, [arr[:, j].copy() for j in range(width)], len(arr))
        except (ValueError, UnicodeDecodeError):
            pass          # blanks or text somewhere further down: take the general path
    cols, rows = read_table(path.read_bytes(), filename)
    return _cap(Table.from_rows(cols, rows), limit_cells, seed, say)


def _cap(t: Table, limit_cells: int, seed: int, say) -> Table:
    width = max(1, len(t.columns))
    if t.n * width <= limit_cells:
        t.total_rows = t.n
        return t
    want = max(1000, limit_cells // width)
    say(f"The table has {t.n:,} rows — more than fit in memory. Learning from a random sample of {want:,} rows.")
    idx = np.sort(np.random.default_rng(seed).choice(t.n, want, replace=False))
    s = t.take(idx)
    s.total_rows, s.exact, s.sampled = t.n, True, True
    return s


def _all_numeric(rows: list[list[str]]) -> bool:
    if not rows:
        return False
    for r in rows[:2000]:
        for v in r:
            if to_float(v) is None:
                return False
    return True


def _multiline(head: str, delim: str) -> bool:
    """True if some cells contain line breaks (then lines are not rows, and random seeks would land mid-cell)."""
    lines = head.count("\n")
    records = sum(1 for _ in csv.reader(io.StringIO(head), delimiter=delim))
    return records < lines * 0.98


def _seek_sample(path: Path, cols: list[str], delim: str, want: int, seed: int, start: int, head_rows: list) -> Table:
    """Rows read from `want` random places in the file (each: skip to the next line break, read one line)."""
    size = path.stat().st_size
    rng = random.Random(seed)
    offsets = sorted(rng.randrange(start, size) for _ in range(want)) if size > start else []
    width = len(cols)
    out: list[list[str]] = []
    last = -1
    with open(path, "rb") as f:
        for off in offsets:
            f.seek(off)
            f.readline()                         # the rest of the line we landed in
            start = f.tell()
            if start == last:                    # two seeks landed in the same line: take each row once
                continue
            last = start
            line = f.readline()
            if not line:
                continue
            text = decode(line).rstrip("\r\n")
            if not text.strip():
                continue
            if delim == "json":
                try:
                    obj = json.loads(text)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    out.append(["" if obj.get(c) is None else (json.dumps(obj[c]) if isinstance(obj[c], (dict, list)) else str(obj[c])) for c in cols])
                continue
            row = next(csv.reader([text], delimiter=delim), [])
            if len(row) == width:
                out.append(row)
    if len(out) < 10:
        out = head_rows + out
    return Table.from_rows(cols, out)


def _row_from(text: str, cols: list[str], delim: str) -> list[str] | None:
    if delim == "json":
        try:
            obj = json.loads(text)
        except ValueError:
            return None
        if not isinstance(obj, dict):
            return None
        return ["" if obj.get(c) is None else (json.dumps(obj[c]) if isinstance(obj[c], (dict, list)) else str(obj[c])) for c in cols]
    row = next(csv.reader([text], delimiter=delim), [])
    return row if len(row) == len(cols) else None


def _bernoulli(path: Path, cols: list[str], delim: str, p: float, seed: int, skip_header: bool, cap: int) -> Table:
    """One pass through the file keeping each row with probability p (every row at most once)."""
    rng = np.random.default_rng(seed)
    out: list[list[str]] = []
    with open(path, "rb") as f:
        if skip_header:
            f.readline()
        while len(out) < cap:
            block = f.readlines(32 << 20)
            if not block:
                break
            pick = np.flatnonzero(rng.random(len(block)) < p)
            for i in pick:
                text = decode(block[i]).rstrip("\r\n")
                if text.strip():
                    row = _row_from(text, cols, delim)
                    if row is not None:
                        out.append(row)
    return Table.from_rows(cols, out[:cap])


def _reservoir(path: Path, cols: list[str], delim: str, want: int, seed: int, has_header: bool) -> Table:
    """A uniform sample read in one pass (for files whose cells contain line breaks, where seeking is unsafe)."""
    rng = random.Random(seed)
    keep: list[list[str]] = []
    width = len(cols)
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=delim)
        if has_header:
            next(reader, None)
        for i, row in enumerate(reader):
            if len(row) != width:
                continue
            if len(keep) < want:
                keep.append(row)
            else:
                j = rng.randrange(i + 1)
                if j < want:
                    keep[j] = row
    return Table.from_rows(cols, keep)


# ------------------------------------------------------------------ text
DOC_EXT = {".pdf", ".epub", ".docx", ".html", ".htm"}


def load_text(path: Path, filename: str, max_chars: int, say=None) -> tuple[list[tuple[str, str]], dict]:
    """(documents, info). Plain text bigger than `max_chars` is sampled as evenly spaced blocks of 1 MB."""
    from .data import read_text
    say = say or (lambda text: None)
    path = Path(path)
    size = path.stat().st_size
    ext = ext_of(filename)
    if ext in DOC_EXT or ext == ".zip":
        if size > 4 << 30:
            raise DataError(f"{ext} files bigger than 4 GB cannot be read. Convert them to plain .txt first.")
        return read_text(path.read_bytes(), filename), {"sampled": False, "bytes": size}
    budget = max_chars * 2                       # bytes; UTF-8 text is 1–4 bytes a character
    if size <= budget:
        return read_text(path.read_bytes(), filename), {"sampled": False, "bytes": size}
    block = 1 << 20
    n_blocks = max(1, budget // block)
    step = size / n_blocks
    parts = []
    with open(path, "rb") as f:
        for i in range(n_blocks):
            f.seek(int(i * step))
            raw = f.read(block)
            parts.append(decode(raw).split("\n", 1)[-1] if i else decode(raw))
    say(f"The text is {human(size)} — more than fits in memory. Learning from {n_blocks:,} evenly spaced pieces of 1 MB.")
    title = Path(filename).stem.replace("_", " ")
    return [(title, "\n\n".join(parts))], {"sampled": True, "bytes": size, "pieces": n_blocks}
