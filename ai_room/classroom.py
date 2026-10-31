"""Classroom orchestration: lessons, Q&A, exams, reflection and promotion.

The teacher is not a model (see ai_room.teacher); the student is the only AI.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Callable

from .agents import Student
from .curriculum import DEFAULT_SUBJECTS, NUM_GRADES, PASS_MARK, SUBJECTS, difficulty
from .grading import extract_code, grade_question, run_code_tests
from .llm import LLM, check_student_size, make_llm
from .memory import Store
from .teacher import PRESETS, Library, NoCourseMaterial, Teacher, style_from
from .teacher.fetch import fetch_subject


@dataclass
class ModelConfig:
    backend: str = "ollama"
    model: str = "qwen2.5:14b"
    base_url: str | None = None
    api_key: str | None = None


@dataclass
class ClassConfig:
    student: ModelConfig = field(default_factory=ModelConfig)
    student_name: str = "Ada"
    student_params_b: float | None = None  # declare size when it can't be detected
    persona: str = "patient"
    style: dict = field(default_factory=dict)  # dial overrides on top of the persona preset
    subjects: list[str] = field(default_factory=lambda: list(DEFAULT_SUBJECTS))
    questions_per_exam: int = 10
    student_questions: int = 3
    max_attempts: int = 5
    pass_mark: float = PASS_MARK
    auto_fetch: bool = True  # fetch textbooks from Wikimedia when a subject has none

    @classmethod
    def from_dict(cls, d: dict) -> "ClassConfig":
        d = dict(d)
        d["student"] = ModelConfig(**{k: v for k, v in (d.get("student") or {}).items() if k in ModelConfig.__dataclass_fields__})
        known = cls.__dataclass_fields__  # tolerant of settings saved by older versions (teacher model, custom_style, ...)
        cfg = cls(**{k: v for k, v in d.items() if k in known})
        bad = [s for s in cfg.subjects if s not in SUBJECTS]
        if bad or not cfg.subjects:
            raise ValueError(f"Unknown or empty subjects: {bad}")
        if not 0 < cfg.pass_mark <= 1:
            raise ValueError("pass_mark must be in (0, 1]")
        if not 3 <= int(cfg.questions_per_exam) <= 40:
            raise ValueError("questions_per_exam must be between 3 and 40")
        if cfg.persona != "custom" and cfg.persona not in PRESETS:
            raise ValueError(f"Unknown teacher persona '{cfg.persona}'")
        return cfg

    def to_dict(self, redact: bool = True) -> dict:
        d = asdict(self)
        if redact and d["student"].get("api_key"):
            d["student"]["api_key"] = "***"
        return d


class Stopped(Exception):
    pass


Fetcher = Callable[..., dict]


class Classroom:
    def __init__(self, store: Store, classroom_id: str, library: Library, student_llm: LLM | None = None, fetcher: Fetcher | None = None):
        room = store.get_classroom(classroom_id)
        if room is None:
            raise KeyError(classroom_id)
        self.store = store
        self.library = library
        self.id = classroom_id
        self.cfg = ClassConfig.from_dict(room["config"])
        self.fetcher = fetcher or fetch_subject
        s_llm = student_llm or make_llm(**asdict(self.cfg.student))
        self.student_size_b = check_student_size(s_llm, self.cfg.student_params_b)
        self.style = style_from(self.cfg.persona, self.cfg.style)
        self.teacher = Teacher(library, self.style, self.cfg.student_name)
        self.student = Student(s_llm, store, classroom_id, name=self.cfg.student_name)
        self.status = "idle"
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    # -- helpers ------------------------------------------------------------
    @property
    def grade(self) -> int:
        return self.store.get_classroom(self.id)["grade"]

    @property
    def graduated(self) -> bool:
        return self.grade > NUM_GRADES

    def say(self, who: str, content: str, **meta) -> None:
        if who == "teacher":
            speaker = self.teacher.name
            meta.setdefault("avatar", self.style.avatar)
        elif who == "student":
            speaker = self.student.name
        elif who == "user":
            speaker = "You"
        else:
            speaker = "Classroom"
        self.store.add_message(self.id, speaker, who, content, meta)

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise Stopped()

    def _set_status(self, status: str, kind: str = "status", **meta) -> None:
        self._check_stop()
        self.status = status
        self.say("system", status, kind=kind, **meta)

    def _passed_subjects(self, grade: int) -> set[str]:
        return {e["subject"] for e in self.store.exams(self.id, grade) if e["score"] >= self.cfg.pass_mark}

    def _ensure_material(self, subject: str, grade: int) -> None:
        """Make sure the teacher has something to teach: generators, textbooks or a question bank."""
        if self.teacher.has_material(subject, grade):
            return
        name = SUBJECTS[subject].name
        if not self.cfg.auto_fetch:
            raise NoCourseMaterial(self.teacher.missing_message(subject))
        self._set_status(f"📚 No textbooks for {name} yet — fetching from Wikipedia and Wikibooks…", "library")

        def progress(message: str, done: int, total: int) -> None:
            self._check_stop()
            self.status = f"📚 {name}: {message} ({done}/{total})"

        summary = self.fetcher(self.library, subject, progress=progress, stop=self.stop_event)
        self._check_stop()
        if summary.get("added"):
            self.say("system", f"📚 Added {summary['added']} textbook articles for {name}.", kind="library")
        if not self.teacher.has_material(subject, grade):
            hint = f" ({summary['errors'][0]})" if summary.get("errors") else ""
            raise NoCourseMaterial(self.teacher.missing_message(subject) + hint)

    # -- lessons ------------------------------------------------------------
    def lesson(self, subject: str, focus: list[str] | None = None) -> None:
        g = self.grade
        name = SUBJECTS[subject].name
        self._ensure_material(subject, g)
        self._set_status(f"Class {g} ({difficulty(g)}) — {name} lesson")
        lesson = self.teacher.lesson(subject, g, focus)
        self.say("teacher", lesson.text, subject=subject, kind="lesson", sources=lesson.sources)
        self.store.add_note(self.id, subject, g, "lesson", self.student.take_notes(subject, g, lesson.text))
        for q in self.student.ask_questions(subject, g, lesson.text, self.cfg.student_questions):
            self._check_stop()
            self.say("student", q, subject=subject, kind="question")
            ans = self.teacher.answer([subject], g, q)
            self.say("teacher", ans, subject=subject, kind="answer")
            self.store.add_note(self.id, subject, g, "qa", f"Q: {q}\nA: {ans}")
        self.say("system", f"{self.student.name} saved notes for {name}.", kind="notes")

    def lessons(self, subjects: list[str] | None = None) -> None:
        for s in subjects or self.cfg.subjects:
            self.lesson(s)

    # -- exams --------------------------------------------------------------
    @staticmethod
    def _expected(q: dict) -> str:
        if q["type"] == "code":
            return "; ".join(q.get("tests", []))[:400]
        if q["type"] == "mcq":
            opt = next((o for o in q.get("options", []) if o.startswith(q["answer"])), q["answer"])
            return opt
        return str(q["answer"])

    def _grade_one(self, q: dict, reply: str) -> tuple[float, str, str]:
        """Returns (score 0..1, what the student gave, feedback). All grading is deterministic."""
        feedback = q.get("explanation", "")
        if q["type"] == "code":
            passed, total, log = run_code_tests(extract_code(reply), list(q.get("tests", [])))
            return (passed / total if total else 0.0), f"{passed}/{total} tests passed", (log if passed < total else "") or feedback
        score, given = grade_question(q, reply)
        return score, given, feedback

    def exam_subject(self, subject: str, attempt: int) -> dict:
        g = self.grade
        name = SUBJECTS[subject].name
        self._ensure_material(subject, g)
        self._set_status(f"Class {g} — {name} exam (attempt {attempt})")
        previous = {d["question"] for e in self.store.exams(self.id, g) if e["subject"] == subject for d in e["details"]["items"]}
        questions = self.teacher.exam(subject, g, self.cfg.questions_per_exam, attempt, avoid=previous)
        items, total = [], 0.0
        for i, q in enumerate(questions, 1):
            self._check_stop()
            reply = self.student.answer(subject, g, q)
            score, given, feedback = self._grade_one(q, reply)
            total += score
            items.append({"question": q["question"], "type": q["type"], "options": q.get("options"), "reply": reply, "given": given,
                          "expected": self._expected(q), "score": score, "feedback": feedback, "concept": q.get("concept", ""),
                          "source": q.get("source")})
            options = ("\n" + "\n".join(q["options"])) if q["type"] == "mcq" else ""
            self.say("student", f"**Q{i}.** {q['question']}{options}\n\n{reply}", subject=subject, kind="exam_answer", score=score,
                     expected=self._expected(q), feedback=feedback, question=q["question"] + options, reply=reply, n=i, total=len(questions), given=given)
        pct = total / len(questions)
        self.store.add_exam(self.id, g, attempt, subject, pct, {"items": items})
        passed = pct >= self.cfg.pass_mark
        self.say("system", f"{name}: {pct:.0%} — {'PASSED' if passed else 'not yet'} (needs {self.cfg.pass_mark:.0%})",
                 kind="score", subject=subject, score=pct, passed=passed)
        if not passed:
            mistakes = [it for it in items if it["score"] < 1.0]
            review = self.teacher.review(subject, g, mistakes)
            self.say("teacher", review, subject=subject, kind="review")
            self.store.add_note(self.id, subject, g, "mistake", self.student.reflect(subject, g, review, mistakes))
            self.say("system", f"{self.student.name} wrote lessons from its {name} mistakes into the notebook.", kind="notes")
        return {"subject": subject, "score": pct, "passed": passed}

    def exam(self) -> dict:
        """Sit (or re-sit) the exams for the current class. Subjects already passed this class are not re-examined."""
        g = self.grade
        attempt = self.store.last_attempt(self.id, g) + 1
        done = self._passed_subjects(g)
        results = [self.exam_subject(s, attempt) for s in self.cfg.subjects if s not in done]
        remaining = [s for s in self.cfg.subjects if s not in self._passed_subjects(g)]
        promoted = not remaining
        if promoted:
            for s in self.cfg.subjects:
                self.store.add_note(self.id, s, g, "summary", self.student.summarize(s, g))
            self.store.update_classroom(self.id, grade=g + 1)
            msg = (f"🎓 {self.student.name} GRADUATED from all 12 classes!" if g == NUM_GRADES
                   else f"🎉 {self.student.name} passed every subject in class {g} and moves up to class {g + 1}.")
            self.say("system", msg, kind="promotion", grade=g + 1)
        else:
            self.say("system", f"Still to pass in class {g}: {', '.join(SUBJECTS[s].name for s in remaining)}.", kind="result")
        return {"grade": g, "attempt": attempt, "results": results, "promoted": promoted, "remaining": remaining}

    # -- full loop ----------------------------------------------------------
    def _weak_concepts(self, g: int, subject: str) -> list[str]:
        last = [e for e in self.store.exams(self.id, g) if e["subject"] == subject][-1]
        return [it["concept"] for it in last["details"]["items"] if it["score"] < 1 and it.get("concept")]

    def run_class(self) -> dict:
        """Lessons, then exams; failed subjects get a focused re-lesson and a fresh exam, up to max_attempts."""
        g = self.grade
        if self.store.last_attempt(self.id, g) == 0:
            self.lessons()
        result = {"promoted": False, "remaining": list(self.cfg.subjects)}
        while self.store.last_attempt(self.id, g) < self.cfg.max_attempts:
            result = self.exam()
            if result["promoted"]:
                return result
            if self.store.last_attempt(self.id, g) >= self.cfg.max_attempts:
                break
            for s in result["remaining"]:
                self.lesson(s, focus=self._weak_concepts(g, s))
        self.say("system", f"{self.student.name} used all {self.cfg.max_attempts} attempts for class {g}. "
                           "Try a different teacher style, add more textbooks, or press Run again.", kind="held")
        return result

    def run_all(self) -> None:
        while not self.graduated:
            if not self.run_class()["promoted"]:
                return

    # -- talking ------------------------------------------------------------
    def talk(self, text: str, to: str = "student") -> str:
        history = [m for m in self.store.messages(self.id)[-40:] if m["role"] in ("student", "teacher", "user")]
        self.say("user", text, to=to)
        g = min(self.grade, NUM_GRADES)
        if to == "teacher":
            reply = self.teacher.answer(self.cfg.subjects, g, text)
            self.say("teacher", reply, kind="chat")
        else:
            reply = self.student.chat(text, history, g, self.cfg.subjects)
            self.say("student", reply, kind="chat")
        return reply

    def report(self) -> dict:
        exams = self.store.exams(self.id)
        card: dict[int, dict[str, dict]] = {}
        for e in exams:
            row = card.setdefault(e["grade"], {})
            best = row.get(e["subject"])
            if best is None or e["score"] >= best["best"]:
                row[e["subject"]] = {"best": e["score"], "attempts": e["attempt"]}
            else:
                best["attempts"] = e["attempt"]
        notes = self.store.notes(self.id)
        return {
            "grade": self.grade, "graduated": self.graduated, "status": self.status,
            "student_size_b": self.student_size_b, "pass_mark": self.cfg.pass_mark,
            "subjects": self.cfg.subjects,
            "report_card": {str(g): v for g, v in sorted(card.items())},
            "notebook": {k: sum(1 for n in notes if n.kind == k) for k in ("lesson", "qa", "mistake", "summary")},
        }
