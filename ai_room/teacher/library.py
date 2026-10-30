"""The textbook library: documents split into passages, searchable with BM25.

Content comes from importers (txt, md, html, pdf, epub, docx, URLs) and from the
Wikimedia fetcher (fetch.py). The teacher quotes passages in lessons, answers the
student's questions from them, and builds fill-in-the-blank questions from them.
No model is involved at any point.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO

from ..memory import tokenize

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT, title TEXT, source TEXT, url TEXT, level TEXT,
    topic TEXT, license TEXT, chars INTEGER, added REAL, UNIQUE(subject, title, source));
CREATE TABLE IF NOT EXISTS passages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id INTEGER, subject TEXT, topic TEXT, level TEXT,
    heading TEXT, idx INTEGER, text TEXT);
CREATE TABLE IF NOT EXISTS bank (
    id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT, grade INTEGER, type TEXT, question TEXT,
    options TEXT, answer TEXT, explanation TEXT, extra TEXT, source TEXT, UNIQUE(subject, question));
CREATE INDEX IF NOT EXISTS passages_subject ON passages(subject);
CREATE INDEX IF NOT EXISTS bank_subject ON bank(subject, grade);
"""

STOP_SECTIONS = {"see also", "references", "external links", "notes", "further reading", "sources", "bibliography", "citations",
                 "footnotes", "gallery", "notes and references", "references and notes", "works cited", "contents"}
MIN_PASSAGE = 120
TARGET_PASSAGE = 900
MAX_PASSAGE = 1600


@dataclass
class Passage:
    id: int
    doc_id: int
    subject: str
    topic: str
    level: str
    heading: str
    text: str
    title: str = ""
    url: str = ""
    source: str = ""
    score: float = 0.0


# ---------------------------------------------------------------- text extraction

class _TextExtractor(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "blockquote", "table", "ul", "ol"}
    SKIP = {"script", "style", "nav", "footer", "header", "noscript", "svg", "form", "aside"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0
        self.title = ""
        self._in_title = False
        self._heading = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in ("h1", "h2", "h3", "h4"):
            self.parts.append("\n\n== ")
            self._heading = True
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in ("h1", "h2", "h3", "h4"):
            self.parts.append(" ==\n\n")
            self._heading = False
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    p = _TextExtractor()
    p.feed(html)
    return clean_text("".join(p.parts)), p.title.strip()


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(filename: str, data: bytes) -> tuple[str, str]:
    """(text, suggested title) from the bytes of a txt/md/html/pdf/epub/docx file."""
    name = filename.lower()
    title = os.path.splitext(os.path.basename(filename))[0].replace("_", " ").replace("-", " ").strip()
    if name.endswith((".txt", ".md", ".markdown", ".rst", ".text")):
        return clean_text(data.decode("utf-8", errors="replace")), title
    if name.endswith((".html", ".htm", ".xhtml")):
        text, t = html_to_text(data.decode("utf-8", errors="replace"))
        return text, t or title
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError("Reading PDFs needs pypdf: pip install pypdf") from e
        reader = PdfReader(BytesIO(data))
        return clean_text("\n\n".join((page.extract_text() or "") for page in reader.pages)), title
    if name.endswith(".epub"):
        with zipfile.ZipFile(BytesIO(data)) as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith((".xhtml", ".html", ".htm")))
            texts = [html_to_text(z.read(n).decode("utf-8", errors="replace"))[0] for n in names]
        return clean_text("\n\n".join(texts)), title
    if name.endswith(".docx"):
        with zipfile.ZipFile(BytesIO(data)) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="replace")
        xml = re.sub(r"</w:p>", "\n\n", xml)
        return clean_text(re.sub(r"<[^>]+>", "", xml)), title
    raise ValueError(f"Unsupported file type: {filename} (use txt, md, html, pdf, epub or docx)")


# ---------------------------------------------------------------- chunking

_HEADING = re.compile(r"^\s*(?:={2,6}\s*(.+?)\s*={2,6}|#{1,6}\s+(.+?))\s*$")


def split_sections(text: str) -> list[tuple[str, str]]:
    """[(heading, body)] using wiki/markdown headings; sections like References are dropped."""
    sections, heading, buf = [], "", []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((heading, body))

    skipping = False
    for line in text.split("\n"):
        m = _HEADING.match(line)
        if m:
            flush()
            buf.clear()
            heading = (m.group(1) or m.group(2)).strip()
            skipping = heading.lower() in STOP_SECTIONS
        elif not skipping:
            buf.append(line)
    flush()
    return sections


def chunk(text: str) -> list[tuple[str, str]]:
    """[(heading, passage)] of roughly TARGET_PASSAGE characters, cut at sentence boundaries."""
    out = []
    for heading, body in split_sections(text):
        paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        paras = [p for p in paras if len(p) >= 40 or p.endswith((".", "!", "?"))]
        cur = ""
        for p in paras:
            pieces = [p]
            if len(p) > MAX_PASSAGE:
                pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z])", p)
            for piece in pieces:
                if cur and len(cur) + len(piece) > TARGET_PASSAGE:
                    out.append((heading, cur.strip()))
                    cur = ""
                cur += (" " if cur else "") + piece
        if cur.strip():
            out.append((heading, cur.strip()))
    # merge tiny passages into the previous one
    merged: list[tuple[str, str]] = []
    for h, p in out:
        if merged and len(p) < MIN_PASSAGE and merged[-1][0] == h:
            merged[-1] = (h, merged[-1][1] + " " + p)
        elif len(p) >= 40:
            merged.append((h, p))
    return merged


# ---------------------------------------------------------------- library


class Library:
    def __init__(self, path: str = "data/library.db"):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.executescript(_SCHEMA)
        self._index: dict[str, tuple] = {}
        self.version = 0

    def _exec(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur

    def _all(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    # -- documents ----------------------------------------------------------
    def add_document(self, subject: str, title: str, text: str, *, source: str = "local", url: str | None = None,
                     level: str = "standard", topic: str | None = None, license: str | None = None) -> int | None:
        """Chunk and store a document. Returns the new document id, or None if nothing usable / already present."""
        parts = chunk(text)
        if not parts:
            return None
        with self.lock:
            try:
                cur = self.db.execute(
                    "INSERT INTO docs (subject, title, source, url, level, topic, license, chars, added) VALUES (?,?,?,?,?,?,?,?,?)",
                    (subject, title, source, url, level, topic, license, len(text), time.time()))
            except sqlite3.IntegrityError:
                return None
            doc_id = cur.lastrowid
            self.db.executemany(
                "INSERT INTO passages (doc_id, subject, topic, level, heading, idx, text) VALUES (?,?,?,?,?,?,?)",
                [(doc_id, subject, topic or "", level, h, i, p) for i, (h, p) in enumerate(parts)])
            self.db.commit()
        self.version += 1
        return doc_id

    def import_file(self, subject: str, filename: str, data: bytes, title: str | None = None) -> dict:
        text, suggested = extract_text(filename, data)
        doc = self.add_document(subject, title or suggested, text, source="import:" + os.path.basename(filename))
        if doc is None:
            raise ValueError("No usable text found (or this file is already in the library).")
        return {"doc_id": doc, "title": title or suggested, "chars": len(text)}

    def import_path(self, subject: str, path: str) -> list[dict]:
        """Import one file or every supported file in a folder."""
        paths = [path] if os.path.isfile(path) else [os.path.join(r, f) for r, _, fs in os.walk(path) for f in sorted(fs)]
        out = []
        for p in paths:
            if not p.lower().endswith((".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".pdf", ".epub", ".docx")):
                continue
            with open(p, "rb") as f:
                try:
                    out.append(self.import_file(subject, os.path.basename(p), f.read()))
                except (ValueError, RuntimeError) as e:
                    out.append({"title": os.path.basename(p), "error": str(e)})
        return out

    def has_document(self, subject: str, title: str, source: str) -> bool:
        return bool(self._all("SELECT 1 FROM docs WHERE subject=? AND title=? AND source=?", (subject, title, source)))

    def documents(self, subject: str | None = None) -> list[dict]:
        sql = "SELECT d.*, (SELECT COUNT(*) FROM passages p WHERE p.doc_id = d.id) AS passages FROM docs d"
        rows = self._all(sql + (" WHERE subject=?" if subject else "") + " ORDER BY id DESC", (subject,) if subject else ())
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: int) -> None:
        self._exec("DELETE FROM passages WHERE doc_id=?", (doc_id,))
        self._exec("DELETE FROM docs WHERE id=?", (doc_id,))
        self.version += 1

    def clear_subject(self, subject: str) -> None:
        for t in ("passages", "docs", "bank"):
            self._exec(f"DELETE FROM {t} WHERE subject=?", (subject,))
        self.version += 1

    def stats(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in self._all("SELECT subject, COUNT(*) AS docs, SUM(chars) AS chars FROM docs GROUP BY subject"):
            out[r["subject"]] = {"docs": r["docs"], "chars": r["chars"] or 0, "passages": 0, "questions": 0}
        for r in self._all("SELECT subject, COUNT(*) AS n FROM passages GROUP BY subject"):
            out.setdefault(r["subject"], {"docs": 0, "chars": 0, "passages": 0, "questions": 0})["passages"] = r["n"]
        for r in self._all("SELECT subject, COUNT(*) AS n FROM bank GROUP BY subject"):
            out.setdefault(r["subject"], {"docs": 0, "chars": 0, "passages": 0, "questions": 0})["questions"] = r["n"]
        return out

    def has_text(self, subject: str, min_passages: int = 8) -> bool:
        r = self._all("SELECT COUNT(*) AS n FROM passages WHERE subject=?", (subject,))
        return r[0]["n"] >= min_passages

    # -- search -------------------------------------------------------------
    def _load_index(self, subject: str):
        cached = self._index.get(subject)
        if cached and cached[0] == self.version:
            return cached
        rows = self._all("SELECT p.*, d.title AS title, d.url AS url, d.source AS source FROM passages p JOIN docs d ON d.id = p.doc_id WHERE p.subject=? ORDER BY p.id", (subject,))
        passages = [Passage(r["id"], r["doc_id"], r["subject"], r["topic"], r["level"], r["heading"], r["text"], r["title"], r["url"] or "", r["source"]) for r in rows]
        docs = [tokenize(p.heading + " " + p.text) for p in passages]
        df = Counter(t for d in docs for t in set(d))
        avg = (sum(len(d) for d in docs) / len(docs)) if docs else 1.0
        tfs = [Counter(d) for d in docs]
        self._index[subject] = (self.version, passages, docs, tfs, df, avg)
        return self._index[subject]

    def passages(self, subject: str) -> list[Passage]:
        return self._load_index(subject)[1]

    def search(self, subject: str, query: str, k: int = 6, topic: str | None = None, grade: int | None = None,
               topics: set[str] | None = None) -> list[Passage]:
        _, passages, docs, tfs, df, avg = self._load_index(subject)
        if not passages:
            return []
        q = set(tokenize(query))
        wanted = set(topics or ()) | ({topic} if topic else set())
        n = len(passages)
        k1, b = 1.4, 0.75
        scored = []
        for p, d, tf in zip(passages, docs, tfs):
            s = 0.0
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avg))
            on_topic = p.topic in wanted if wanted else False
            if s <= 0 and not on_topic:
                continue
            if on_topic:
                s = s * 1.5 + 1.0
            if grade is not None:
                if grade <= 4:
                    s *= 1.3 if p.level == "simple" else 0.85
                elif grade >= 9:
                    s *= 0.5 if p.level == "simple" else 1.15
            scored.append((s, p))
        scored.sort(key=lambda x: -x[0])
        out = []
        for s, p in scored[:k]:
            p.score = s
            out.append(p)
        return out

    # -- question bank ------------------------------------------------------
    def add_questions(self, subject: str, items: list[dict], source: str = "bank") -> int:
        added = 0
        with self.lock:
            for it in items:
                try:
                    self.db.execute(
                        "INSERT INTO bank (subject, grade, type, question, options, answer, explanation, extra, source) VALUES (?,?,?,?,?,?,?,?,?)",
                        (subject, it.get("grade"), it["type"], it["question"], json.dumps(it.get("options")) if it.get("options") else None,
                         str(it["answer"]), it.get("explanation", ""), json.dumps({k: v for k, v in it.items() if k in ("tests", "function_name", "tol")}), source))
                    added += 1
                except sqlite3.IntegrityError:
                    continue
            self.db.commit()
        return added

    def questions(self, subject: str, grade: int | None = None) -> list[dict]:
        if grade is None:
            rows = self._all("SELECT * FROM bank WHERE subject=? ORDER BY id", (subject,))
        else:
            rows = self._all("SELECT * FROM bank WHERE subject=? AND (grade IS NULL OR grade=?) ORDER BY id", (subject, grade))
        out = []
        for r in rows:
            q = {"type": r["type"], "question": r["question"], "answer": r["answer"], "explanation": r["explanation"] or "", "concept": f"bank:{r['id']}",
                 "source": {"title": r["source"]}}
            if r["options"]:
                q["options"] = json.loads(r["options"])
            q.update(json.loads(r["extra"] or "{}"))
            out.append(q)
        return out


# ---------------------------------------------------------------- question-bank formats

def normalize_bank_item(raw: dict, default_subject: str | None = None) -> tuple[str | None, dict] | None:
    """Accepts our own schema and the common public dataset layouts (MMLU, ARC, SciQ, GSM8K, MBPP)."""
    letters = "ABCDEFGH"
    subject = raw.get("subject") or default_subject
    grade = raw.get("grade")
    explanation = raw.get("explanation") or raw.get("support") or ""

    def mc(question, options, correct_idx):
        return {"type": "mcq", "question": str(question).strip(), "options": [f"{letters[i]}) {o}" for i, o in enumerate(options)],
                "answer": letters[correct_idx], "explanation": explanation, "grade": grade}

    if "type" in raw and "question" in raw and "answer" in raw:  # native
        item = {k: raw[k] for k in ("type", "question", "answer", "explanation", "options", "tests", "function_name", "grade") if k in raw}
        if item["type"] == "mcq" and item.get("options") and not str(item["options"][0]).startswith(("A)", "A.")):
            item["options"] = [f"{letters[i]}) {o}" for i, o in enumerate(item["options"])]
        return subject, item
    if isinstance(raw.get("choices"), list) and isinstance(raw.get("answer"), int):  # MMLU-style
        return subject, mc(raw["question"], raw["choices"], raw["answer"])
    if isinstance(raw.get("question"), dict) and "choices" in raw["question"]:  # ARC
        ch = raw["question"]["choices"]
        labels = [c["label"] for c in ch]
        if raw.get("answerKey") in labels:
            return subject, mc(raw["question"]["stem"], [c["text"] for c in ch], labels.index(raw["answerKey"]))
    if "correct_answer" in raw and "distractor1" in raw:  # SciQ
        opts = [raw["correct_answer"], raw["distractor1"], raw.get("distractor2", ""), raw.get("distractor3", "")]
        opts = [o for o in opts if o]
        order = sorted(range(len(opts)), key=lambda i: hashlib.sha1(f"{raw['question']}|{i}".encode()).hexdigest())
        return subject, mc(raw["question"], [opts[i] for i in order], order.index(0))
    if isinstance(raw.get("answer"), str) and "####" in raw["answer"] and "question" in raw:  # GSM8K
        final = raw["answer"].split("####")[-1].strip().replace(",", "")
        return subject, {"type": "numeric", "question": raw["question"], "answer": final, "explanation": raw["answer"].split("####")[0].strip(), "grade": grade}
    if "test_list" in raw and "code" in raw and "text" in raw:  # MBPP
        m = re.search(r"def\s+(\w+)", raw["code"])
        if m:
            return subject, {"type": "code", "question": f"{raw['text']} Write a Python function `{m.group(1)}`.", "answer": raw["code"],
                             "tests": list(raw["test_list"]), "function_name": m.group(1), "explanation": "", "grade": grade}
    return None


def parse_bank_text(text: str, default_subject: str | None = None) -> list[tuple[str | None, dict]]:
    """JSON array or JSON-lines."""
    text = text.strip()
    rows = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    out = []
    for raw in rows:
        n = normalize_bank_item(raw, default_subject)
        if n and n[0]:
            out.append(n)
    return out
