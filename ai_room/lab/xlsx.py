"""Read Excel workbooks (.xlsx / .xlsm) with nothing but the standard library.

An .xlsx file is a zip of XML parts. This reads the first worksheet as a stream (so big sheets do not need to be
held as XML in memory), turns shared strings back into text, and shows cells that Excel formats as dates as
ISO dates (2024-03-01) so the Lab sees them as dates. Formulas give their last computed value, as Excel saved it.
"""

from __future__ import annotations

import re
import zipfile
from datetime import datetime, timedelta
from xml.etree.ElementTree import iterparse

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
DATE_IDS = set(range(14, 23)) | {45, 46, 47}
_COL = re.compile(r"^([A-Z]+)")


class XlsxError(ValueError):
    pass


def _col_index(ref: str) -> int:
    m = _COL.match(ref or "")
    if not m:
        return -1
    n = 0
    for ch in m.group(1):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _text(el) -> str:
    """All text inside an element (shared strings may be split into formatted runs)."""
    return "".join(t.text or "" for t in el.iter(NS + "t"))


def _is_date_format(code: str) -> bool:
    code = re.sub(r'"[^"]*"|\[[^\]]*\]|\\.', "", code.lower())
    return bool(re.search(r"[dmyhs]", code)) and not re.fullmatch(r"[0#.,%e+\- ]*", code)


class Workbook:
    def __init__(self, source):
        try:
            self.z = zipfile.ZipFile(source)
        except zipfile.BadZipFile:
            raise XlsxError("This does not look like an .xlsx workbook. Old .xls files are not supported — save it as .xlsx or CSV.")
        names = set(self.z.namelist())
        if "xl/workbook.xml" not in names:
            raise XlsxError("This zip is not an Excel workbook (no xl/workbook.xml).")
        self.names = names
        self.strings = self._shared_strings()
        self.date_styles, self.epoch = self._styles(), self._epoch()
        self.sheets = self._sheets()

    def _shared_strings(self) -> list[str]:
        if "xl/sharedStrings.xml" not in self.names:
            return []
        out = []
        with self.z.open("xl/sharedStrings.xml") as f:
            for _, el in iterparse(f):
                if el.tag == NS + "si":
                    out.append(_text(el))
                    el.clear()
        return out

    def _styles(self) -> set[int]:
        """Indexes of cell styles that show a date."""
        if "xl/styles.xml" not in self.names:
            return set()
        from xml.etree.ElementTree import fromstring
        root = fromstring(self.z.read("xl/styles.xml"))
        custom = {int(n.get("numFmtId")): n.get("formatCode", "") for n in root.iter(NS + "numFmt")}
        xfs = root.find(NS + "cellXfs")
        dates = set()
        if xfs is not None:
            for i, xf in enumerate(xfs.findall(NS + "xf")):
                fid = int(xf.get("numFmtId", "0"))
                if fid in DATE_IDS or (fid in custom and _is_date_format(custom[fid])):
                    dates.add(i)
        return dates

    def _epoch(self) -> datetime:
        head = self.z.read("xl/workbook.xml")[:4000].decode("utf-8", "replace")
        return datetime(1904, 1, 1) if re.search(r'date1904="(1|true)"', head) else datetime(1899, 12, 30)

    def _sheets(self) -> list[tuple[str, str]]:
        from xml.etree.ElementTree import fromstring
        wb = fromstring(self.z.read("xl/workbook.xml"))
        rels = {}
        if "xl/_rels/workbook.xml.rels" in self.names:
            for r in fromstring(self.z.read("xl/_rels/workbook.xml.rels")):
                rels[r.get("Id")] = r.get("Target", "")
        out = []
        for sh in wb.iter(NS + "sheet"):
            target = rels.get(sh.get(REL + "id"), "")
            target = target.lstrip("/")
            path = target if target.startswith("xl/") else "xl/" + target
            if path in self.names:
                out.append((sh.get("name", "Sheet"), path))
        if not out:
            out = [(n.rsplit("/", 1)[-1], n) for n in sorted(self.names) if n.startswith("xl/worksheets/sheet")]
        if not out:
            raise XlsxError("The workbook has no worksheets.")
        return out

    def dimension_rows(self, sheet: int = 0) -> int | None:
        """Rows the sheet says it has (from <dimension ref="A1:K5000">), without reading it."""
        with self.z.open(self.sheets[sheet][1]) as f:
            head = f.read(4096).decode("utf-8", "replace")
        m = re.search(r'<(?:\w+:)?dimension ref="[A-Z]+(\d+)(?::[A-Z]+(\d+))?"', head)
        if not m:
            return None
        return int(m.group(2) or m.group(1)) - int(m.group(1)) + 1

    def _value(self, c) -> str:
        t = c.get("t", "n")
        if t == "inlineStr":
            is_ = c.find(NS + "is")
            return _text(is_) if is_ is not None else ""
        v = c.find(NS + "v")
        if v is None or v.text is None:
            return ""
        x = v.text
        if t == "s":
            i = int(x)
            return self.strings[i] if 0 <= i < len(self.strings) else ""
        if t == "b":
            return "TRUE" if x == "1" else "FALSE"
        if t in ("str", "e"):
            return x
        s = c.get("s")
        if s is not None and int(s) in self.date_styles:
            try:
                d = self.epoch + timedelta(days=float(x))
                return d.strftime("%Y-%m-%d") if d.hour == d.minute == d.second == 0 else d.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, OverflowError):
                return x
        if x.endswith(".0"):
            x = x[:-2]
        return x

    def rows(self, sheet: int = 0, limit: int | None = None):
        """Each row as a list of strings (missing cells are empty)."""
        n = 0
        with self.z.open(self.sheets[sheet][1]) as f:
            for _, el in iterparse(f):
                if el.tag != NS + "row":
                    continue
                cells: dict[int, str] = {}
                for k, c in enumerate(el.findall(NS + "c")):
                    j = _col_index(c.get("r", ""))
                    cells[j if j >= 0 else k] = self._value(c)
                el.clear()
                if not any(v.strip() for v in cells.values()):
                    continue
                width = max(cells) + 1
                yield [cells.get(j, "") for j in range(width)]
                n += 1
                if limit is not None and n >= limit:
                    return


def read_xlsx(source, limit: int | None = None) -> tuple[list[str], list[list[str]]]:
    """(columns, rows) from the first worksheet. The first row is the header unless it is mostly numbers."""
    from .data import DataError, _unique_names, to_float
    try:
        wb = Workbook(source)
        it = wb.rows(0, None if limit is None else limit + 1)
        first = next(it, None)
        if first is None:
            raise DataError("The first sheet of the workbook is empty.")
        rest = list(it)
    except XlsxError as e:
        raise DataError(str(e))
    width = max([len(first)] + [len(r) for r in rest[:5000]])
    if sum(to_float(h) is not None for h in first if h.strip()) > len([h for h in first if h.strip()]) / 2:
        header, rows = [f"column {i + 1}" for i in range(width)], [first] + rest
    else:
        header, rows = [h.strip() or f"column {i + 1}" for i, h in enumerate(first + [""] * (width - len(first)))], rest
    header = _unique_names(header)
    rows = [(r + [""] * width)[:width] for r in rows]
    if not rows:
        raise DataError("The first sheet has a header row but no data under it.")
    return header, rows
