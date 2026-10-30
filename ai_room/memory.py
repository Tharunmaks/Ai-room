"""Persistent storage: the student's notebook, exam records and class transcripts.

The notebook is how the student "learns without training": every lesson,
answered question and corrected mistake is written down here, and the most
relevant notes are retrieved (BM25) into the student's prompt whenever it has
to answer something.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass

# Mistake notes and grade summaries are the most valuable things to re-read.
KIND_BOOST = {"mistake": 1.6, "summary": 1.4, "qa": 1.15, "lesson": 1.0, "practice": 1.0}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS classrooms (
    id TEXT PRIMARY KEY, name TEXT, config TEXT, grade INTEGER, created_at REAL);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, classroom_id TEXT, subject TEXT, grade INTEGER,
    kind TEXT, content TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS exams (
    id INTEGER PRIMARY KEY AUTOINCREMENT, classroom_id TEXT, grade INTEGER, attempt INTEGER,
    subject TEXT, score REAL, details TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS transcript (
    id INTEGER PRIMARY KEY AUTOINCREMENT, classroom_id TEXT, speaker TEXT, role TEXT,
    content TEXT, meta TEXT, created_at REAL);
CREATE INDEX IF NOT EXISTS notes_by_room ON notes(classroom_id, subject);
CREATE INDEX IF NOT EXISTS transcript_by_room ON transcript(classroom_id, id);
"""

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("the a an and or of to in is are be for on with as by it this that at from was were "
            "which what how why when your you we i".split())


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


@dataclass
class Note:
    id: int
    subject: str
    grade: int
    kind: str
    content: str


class Store:
    def __init__(self, path: str = "data/ai_room.db"):
        if path != ":memory:":
            import os
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(_SCHEMA)

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur

    def _all(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    # -- classrooms ---------------------------------------------------------
    def create_classroom(self, name: str, config: dict) -> str:
        cid = uuid.uuid4().hex[:10]
        self._exec("INSERT INTO classrooms VALUES (?,?,?,?,?)", (cid, name, json.dumps(config), 1, time.time()))
        return cid

    def get_classroom(self, cid: str) -> dict | None:
        rows = self._all("SELECT * FROM classrooms WHERE id=?", (cid,))
        if not rows:
            return None
        r = rows[0]
        return {"id": r["id"], "name": r["name"], "config": json.loads(r["config"]), "grade": r["grade"]}

    def list_classrooms(self) -> list[dict]:
        rows = self._all("SELECT id, name, grade, config FROM classrooms ORDER BY created_at DESC")
        return [{"id": r["id"], "name": r["name"], "grade": r["grade"], "config": json.loads(r["config"])} for r in rows]

    def update_classroom(self, cid: str, *, grade: int | None = None, config: dict | None = None) -> None:
        if grade is not None:
            self._exec("UPDATE classrooms SET grade=? WHERE id=?", (grade, cid))
        if config is not None:
            self._exec("UPDATE classrooms SET config=? WHERE id=?", (json.dumps(config), cid))

    def delete_classroom(self, cid: str) -> None:
        for table in ("notes", "exams", "transcript"):
            self._exec(f"DELETE FROM {table} WHERE classroom_id=?", (cid,))
        self._exec("DELETE FROM classrooms WHERE id=?", (cid,))

    # -- notebook -----------------------------------------------------------
    def add_note(self, cid: str, subject: str, grade: int, kind: str, content: str) -> int:
        cur = self._exec("INSERT INTO notes (classroom_id, subject, grade, kind, content, created_at) "
                         "VALUES (?,?,?,?,?,?)", (cid, subject, grade, kind, content.strip(), time.time()))
        return cur.lastrowid

    def notes(self, cid: str, subject: str | None = None, grade: int | None = None,
              kind: str | None = None) -> list[Note]:
        sql, args = "SELECT * FROM notes WHERE classroom_id=?", [cid]
        for col, val in (("subject", subject), ("grade", grade), ("kind", kind)):
            if val is not None:
                sql += f" AND {col}=?"
                args.append(val)
        rows = self._all(sql + " ORDER BY id", tuple(args))
        return [Note(r["id"], r["subject"], r["grade"], r["kind"], r["content"]) for r in rows]

    def search_notes(self, cid: str, subject: str, query: str, k: int = 6,
                     max_chars: int = 6000) -> list[Note]:
        """BM25 search over the student's notes for one subject."""
        notes = self.notes(cid, subject)
        if not notes:
            return []
        docs = [tokenize(n.content) for n in notes]
        q = set(tokenize(query))
        n_docs = len(docs)
        avg_len = sum(len(d) for d in docs) / n_docs or 1.0
        df = Counter(t for d in docs for t in set(d))
        k1, b = 1.5, 0.75
        scored = []
        for note, doc in zip(notes, docs):
            tf = Counter(doc)
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                idf = math.log(1 + (n_docs - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(doc) / avg_len))
            s *= KIND_BOOST.get(note.kind, 1.0)
            if s > 0:
                scored.append((s, note))
        scored.sort(key=lambda x: -x[0])
        picked, used = [], 0
        for _, note in scored[:k]:
            if used + len(note.content) > max_chars and picked:
                break
            picked.append(note)
            used += len(note.content)
        return picked

    # -- exams --------------------------------------------------------------
    def add_exam(self, cid: str, grade: int, attempt: int, subject: str, score: float, details: dict) -> None:
        self._exec("INSERT INTO exams (classroom_id, grade, attempt, subject, score, details, created_at) "
                   "VALUES (?,?,?,?,?,?,?)", (cid, grade, attempt, subject, score, json.dumps(details), time.time()))

    def exams(self, cid: str, grade: int | None = None) -> list[dict]:
        sql, args = "SELECT * FROM exams WHERE classroom_id=?", [cid]
        if grade is not None:
            sql += " AND grade=?"
            args.append(grade)
        rows = self._all(sql + " ORDER BY id", tuple(args))
        return [{"grade": r["grade"], "attempt": r["attempt"], "subject": r["subject"], "score": r["score"],
                 "details": json.loads(r["details"])} for r in rows]

    def last_attempt(self, cid: str, grade: int) -> int:
        rows = self._all("SELECT MAX(attempt) AS a FROM exams WHERE classroom_id=? AND grade=?", (cid, grade))
        return rows[0]["a"] or 0

    # -- transcript ---------------------------------------------------------
    def add_message(self, cid: str, speaker: str, role: str, content: str, meta: dict | None = None) -> dict:
        now = time.time()
        cur = self._exec("INSERT INTO transcript (classroom_id, speaker, role, content, meta, created_at) "
                         "VALUES (?,?,?,?,?,?)", (cid, speaker, role, content, json.dumps(meta or {}), now))
        return {"id": cur.lastrowid, "speaker": speaker, "role": role, "content": content,
                "meta": meta or {}, "time": now}

    def messages(self, cid: str, after_id: int = 0, limit: int = 500) -> list[dict]:
        rows = self._all("SELECT * FROM transcript WHERE classroom_id=? AND id>? ORDER BY id LIMIT ?",
                         (cid, after_id, limit))
        return [{"id": r["id"], "speaker": r["speaker"], "role": r["role"], "content": r["content"],
                 "meta": json.loads(r["meta"]), "time": r["created_at"]} for r in rows]
