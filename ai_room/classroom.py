"""Classroom orchestration: lessons, Q&A, exams, reflection and promotion."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field

from .agents import Student, Teacher
from .curriculum import DEFAULT_SUBJECTS, NUM_GRADES, PASS_MARK, SUBJECTS, difficulty
from .grading import extract_code, extract_final, mcq_match, numeric_match, run_code_tests
from .llm import LLM, check_student_size, make_llm
from .memory import Store
from .personas import DEFAULT_PERSONA, persona_text


@dataclass
class ModelConfig:
    backend: str = "ollama"
    model: str = "qwen2.5:14b"
    base_url: str | None = None
    api_key: str | None = None


@dataclass
class ClassConfig:
    student: ModelConfig = field(default_factory=ModelConfig)
    teacher: ModelConfig = field(default_factory=lambda: ModelConfig(model="gemma3:27b"))
    student_name: str = "Ada"
    student_params_b: float | None = None  # declare size when it can't be detected
    persona: str = DEFAULT_PERSONA
    custom_style: str | None = None
    subjects: list[str] = field(default_factory=lambda: list(DEFAULT_SUBJECTS))
    questions_per_exam: int = 10
    student_questions: int = 3
    max_attempts: int = 5
    pass_mark: float = PASS_MARK
    verify_exams: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "ClassConfig":
        d = dict(d)
        d["student"] = ModelConfig(**d.get("student", {}))
        d["teacher"] = ModelConfig(**d.get("teacher", {}))
        known = cls.__dataclass_fields__
        cfg = cls(**{k: v for k, v in d.items() if k in known})
        bad = [s for s in cfg.subjects if s not in SUBJECTS]
        if bad or not cfg.subjects:
            raise ValueError(f"Unknown or empty subjects: {bad}")
        if not 0 < cfg.pass_mark <= 1:
            raise ValueError("pass_mark must be in (0, 1]")
        return cfg

    def to_dict(self, redact: bool = True) -> dict:
        d = asdict(self)
        if redact:
            for side in ("student", "teacher"):
                if d[side].get("api_key"):
                    d[side]["api_key"] = "***"
        return d


class Stopped(Exception):
    pass


class Classroom:
    def __init__(self, store: Store, classroom_id: str,
                 student_llm: LLM | None = None, teacher_llm: LLM | None = None):
        room = store.get_classroom(classroom_id)
        if room is None:
            raise KeyError(classroom_id)
        self.store = store
        self.id = classroom_id
        self.cfg = ClassConfig.from_dict(room["config"])
        s_llm = student_llm or make_llm(**asdict(self.cfg.student))
        t_llm = teacher_llm or make_llm(**asdict(self.cfg.teacher))
        if self.cfg.student.backend in ("anthropic", "claude"):
            raise ValueError("The student must be a local/open model of at most 27B parameters.")
        self.student_size_b = check_student_size(s_llm, self.cfg.student_params_b)
        tname, tstyle = persona_text(self.cfg.persona, self.cfg.custom_style)
        self.teacher = Teacher(t_llm, tname, tstyle, verify_exams=self.cfg.verify_exams)
        self.student = Student(s_llm, store, classroom_id, name=self.cfg.student_name)
        self.status = "idle"
        self.stop_event = threading.Event()
        self.lock = threading.Lock()  # one activity at a time

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
        elif who == "student":
            speaker = self.student.name
        elif who == "user":
            speaker = "You"
        else:
            speaker = "Classroom"
        self.store.add_message(self.id, speaker, who, content, meta)

    def _set_status(self, status: str) -> None:
        if self.stop_event.is_set():
            raise Stopped()
        self.status = status
        self.say("system", status, kind="status")

    def _passed_subjects(self, grade: int) -> set[str]:
        return {e["subject"] for e in self.store.exams(self.id, grade) if e["score"] >= self.cfg.pass_mark}

    # -- lessons ------------------------------------------------------------
    def lesson(self, subject: str, focus: str | None = None) -> None:
        g = self.grade
        name = SUBJECTS[subject].name
        self._set_status(f"Class {g} ({difficulty(g)}) — {name} lesson")
        lesson = self.teacher.teach(subject, g, focus)
        self.say("teacher", lesson, subject=subject, kind="lesson")
        self.store.add_note(self.id, subject, g, "lesson", self.student.take_notes(subject, g, lesson))
        for q in self.student.ask_questions(subject, g, lesson, self.cfg.student_questions):
            if self.stop_event.is_set():
                raise Stopped()
            self.say("student", q, subject=subject, kind="question")
            ans = self.teacher.answer_question(subject, g, lesson, q)
            self.say("teacher", ans, subject=subject, kind="answer")
            self.store.add_note(self.id, subject, g, "qa", f"Q: {q}\nA: {ans}")
        self.say("system", f"{self.student.name} saved notes for {name}.", kind="notes")

    def lessons(self, subjects: list[str] | None = None) -> None:
        for s in subjects or self.cfg.subjects:
            self.lesson(s)

    # -- exams --------------------------------------------------------------
    def _grade_one(self, subject: str, g: int, q: dict, reply: str) -> tuple[float, str, str]:
        """Returns (score 0..1, what the student gave, feedback)."""
        t = q["type"]
        if t == "numeric":
            given = extract_final(reply)
            return float(numeric_match(given, str(q.get("answer")))), given, q.get("explanation", "")
        if t == "mcq":
            given = extract_final(reply)
            return float(mcq_match(given, str(q.get("answer")))), given, q.get("explanation", "")
        if t == "code":
            passed, total, log = run_code_tests(extract_code(reply), list(q.get("tests", [])))
            return (passed / total if total else 0.0), f"{passed}/{total} tests passed", log
        score, feedback = self.teacher.grade_short(subject, g, q, reply)
        return score, reply[:400], feedback

    def exam_subject(self, subject: str, attempt: int) -> dict:
        g = self.grade
        name = SUBJECTS[subject].name
        self._set_status(f"Class {g} — {name} exam (attempt {attempt})")
        previous = [d["question"] for e in self.store.exams(self.id, g) if e["subject"] == subject
                    for d in e["details"]["items"]]
        questions = self.teacher.make_exam(subject, g, self.cfg.questions_per_exam, avoid=previous)
        if not questions:
            raise RuntimeError(f"The teacher could not produce a valid {name} exam.")
        items, total = [], 0.0
        for i, q in enumerate(questions, 1):
            if self.stop_event.is_set():
                raise Stopped()
            reply = self.student.answer(subject, g, q)
            score, given, feedback = self._grade_one(subject, g, q, reply)
            total += score
            expected = q.get("answer", "") if q["type"] != "code" else "; ".join(q.get("tests", []))
            items.append({"question": q["question"], "type": q["type"], "options": q.get("options"),
                          "reply": reply, "given": given, "expected": str(expected),
                          "score": score, "feedback": feedback})
            options = "\n" + "\n".join(q.get("options") or []) if q["type"] == "mcq" else ""
            self.say("student", f"**Q{i}.** {q['question']}{options}\n\n{reply}", subject=subject, kind="exam_answer",
                     score=score)
        pct = total / len(questions)
        self.store.add_exam(self.id, g, attempt, subject, pct, {"items": items})
        passed = pct >= self.cfg.pass_mark
        self.say("system", f"{name}: {pct:.0%} — {'PASSED' if passed else 'not yet'} "
                           f"(needs {self.cfg.pass_mark:.0%})", kind="score", subject=subject, score=pct)
        if not passed:
            mistakes = [it for it in items if it["score"] < 1.0]
            review = self.teacher.review_mistakes(subject, g, mistakes)
            self.say("teacher", review, subject=subject, kind="review")
            self.store.add_note(self.id, subject, g, "mistake", self.student.reflect(subject, g, review, mistakes))
            self.say("system", f"{self.student.name} wrote lessons from its {name} mistakes into the notebook.",
                     kind="notes")
        return {"subject": subject, "score": pct, "passed": passed}

    def exam(self) -> dict:
        """Sit (or re-sit) the exams for the current class. Subjects already
        passed this class are not re-examined."""
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
            self.say("system", msg, kind="promotion")
        else:
            self.say("system", "Still to pass in class "
                     f"{g}: {', '.join(SUBJECTS[s].name for s in remaining)}.", kind="result")
        return {"grade": g, "attempt": attempt, "results": results, "promoted": promoted, "remaining": remaining}

    # -- full loop ----------------------------------------------------------
    def run_class(self) -> dict:
        """Lessons, then exams; failed subjects get a focused re-lesson and a
        fresh exam, up to max_attempts."""
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
                last = [e for e in self.store.exams(self.id, g) if e["subject"] == s][-1]
                weak = "\n".join(f"- {it['question']}" for it in last["details"]["items"] if it["score"] < 1)
                self.lesson(s, focus=weak)
        self.say("system", f"{self.student.name} used all {self.cfg.max_attempts} attempts for class {g}. "
                           "Try another teacher style, a stronger teacher model, or press Run again.", kind="held")
        return result

    def run_all(self) -> None:
        while not self.graduated:
            if not self.run_class()["promoted"]:
                return

    # -- talking ------------------------------------------------------------
    def talk(self, text: str, to: str = "student") -> str:
        history = [m for m in self.store.messages(self.id)[-40:] if m["role"] in ("student", "teacher", "user")]
        self.say("user", text, to=to)
        if to == "teacher":
            g = min(self.grade, NUM_GRADES)
            reply = self.teacher._ask(
                f"You teach class {g} of 12 ({difficulty(g)}). The student's guardian/user says to you:\n{text}\n\n"
                "Reply in character.", max_tokens=1500)
            self.say("teacher", reply, kind="chat")
        else:
            reply = self.student.chat(text, history, min(self.grade, NUM_GRADES), self.cfg.subjects)
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
