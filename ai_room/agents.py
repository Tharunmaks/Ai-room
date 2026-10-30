"""The Teacher and Student agents and their prompts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .curriculum import SUBJECTS, difficulty, topics_for
from .grading import extract_final, mcq_match, numeric_match
from .llm import LLM, Message
from .memory import Note, Store


def parse_json(text: str):
    """Pull the first JSON object/array out of a model reply."""
    text = re.sub(r"```(?:json)?", "", text or "")
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No JSON found in reply: {text[:200]!r}")


def _grade_line(subject: str, grade: int) -> str:
    return (f"Subject: {SUBJECTS[subject].name}. Class {grade} of 12 ({difficulty(grade)} level).\n"
            f"Syllabus for this class: {topics_for(subject, grade)}.")


# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------

@dataclass
class Teacher:
    llm: LLM
    name: str
    style: str
    verify_exams: bool = True

    def _system(self) -> str:
        return (f"You are {self.name}, a teacher in the AI Room: a classroom whose student is an AI "
                f"language model that cannot be retrained, so it learns only from what you teach and "
                f"what it writes in its notebook. Your teaching style: {self.style}\n"
                "Always be factually correct. Teach so that the student can solve exam problems on its own.")

    def _ask(self, prompt: str, temperature: float = 0.7, max_tokens: int = 3000) -> str:
        return self.llm.chat([Message("system", self._system()), Message("user", prompt)],
                             temperature=temperature, max_tokens=max_tokens).strip()

    def teach(self, subject: str, grade: int, focus: str | None = None) -> str:
        extra = f"\nFocus especially on these weak points from the last exam:\n{focus}" if focus else ""
        return self._ask(
            f"[[LESSON]]\n{_grade_line(subject, grade)}{extra}\n\n"
            "Give a complete lesson covering the syllabus: core concepts and definitions, key formulas or "
            "rules, step-by-step methods, at least two fully worked examples, and common mistakes to avoid. "
            "Stay in character.", max_tokens=4000)

    def answer_question(self, subject: str, grade: int, lesson: str, question: str) -> str:
        return self._ask(
            f"{_grade_line(subject, grade)}\n\nYou just gave this lesson:\n{lesson[:6000]}\n\n"
            f"The student asks: {question}\n\nAnswer the question clearly and correctly, in character.",
            max_tokens=1500)

    def make_exam(self, subject: str, grade: int, count: int, avoid: list[str] | None = None) -> list[dict]:
        types = SUBJECTS[subject].question_types
        avoid_txt = ""
        if avoid:
            avoid_txt = "\nDo NOT reuse any of these earlier questions:\n" + "\n".join(f"- {q}" for q in avoid[-30:])
        prompt = (
            f"[[EXAM]] COUNT={count}\n{_grade_line(subject, grade)}\n\n"
            f"Write a new exam of exactly {count} questions covering the syllabus at this level. "
            f"Allowed question types: {', '.join(types)}.{avoid_txt}\n\n"
            "Return ONLY JSON: {\"questions\": [...]} where each question is one of:\n"
            '  {"type":"numeric","question":"...","answer":"<single number>","explanation":"..."}\n'
            '  {"type":"mcq","question":"...","options":["A) ...","B) ...","C) ...","D) ..."],"answer":"B","explanation":"..."}\n'
            '  {"type":"short","question":"...","answer":"<model answer>","rubric":"<what a full-credit answer must contain>"}\n'
            '  {"type":"code","question":"Write a Python function `name(args)` that ...","function_name":"name",'
            '"tests":["assert name(1) == 2", "..."],"answer":"<reference solution>"}\n'
            "Numeric answers must be a single number (state units in the question). Code questions need 3-6 "
            "assert tests that use only the standard library. Make questions unambiguous and the answer key correct.")
        data = parse_json(self._ask(prompt, temperature=0.8, max_tokens=6000))
        questions = data["questions"] if isinstance(data, dict) else data
        clean = [q for q in questions if isinstance(q, dict) and q.get("question") and q.get("type") in types]
        if self.verify_exams:
            clean = [q for q in clean if self._key_is_consistent(subject, grade, q)]
        return clean[:count]

    def _key_is_consistent(self, subject: str, grade: int, q: dict) -> bool:
        """Re-solve numeric/MCQ questions independently; drop ones whose answer key disagrees."""
        if q["type"] not in ("numeric", "mcq"):
            return True
        body = q["question"] + ("\n" + "\n".join(q.get("options", [])) if q["type"] == "mcq" else "")
        reply = self._ask(f"[[SOLVE]]\n{_grade_line(subject, grade)}\n\nSolve carefully:\n{body}\n\n"
                          "End with a line 'FINAL ANSWER: <answer>' (a number, or the option letter).",
                          temperature=0.0, max_tokens=2000)
        given = extract_final(reply)
        key = str(q.get("answer", ""))
        return numeric_match(given, key) if q["type"] == "numeric" else mcq_match(given, key)

    def grade_short(self, subject: str, grade: int, q: dict, student_answer: str) -> tuple[float, str]:
        reply = self._ask(
            f"[[GRADE]]\n{_grade_line(subject, grade)}\n\nQuestion: {q['question']}\n"
            f"Reference answer: {q.get('answer', '')}\nRubric: {q.get('rubric', 'Matches the reference answer.')}\n\n"
            f"Student answer:\n{student_answer}\n\n"
            "Grade strictly against the rubric. Return ONLY JSON: {\"score\": <0.0-1.0>, \"feedback\": \"...\"}",
            temperature=0.0, max_tokens=800)
        try:
            data = parse_json(reply)
            return max(0.0, min(1.0, float(data.get("score", 0)))), str(data.get("feedback", ""))
        except (ValueError, TypeError):
            return 0.0, "Could not grade this answer."

    def review_mistakes(self, subject: str, grade: int, mistakes: list[dict]) -> str:
        items = "\n\n".join(
            f"Q: {m['question']}\nStudent answered: {m['given']}\nCorrect: {m['expected']}" for m in mistakes)
        return self._ask(
            f"{_grade_line(subject, grade)}\n\nThe student got these exam questions wrong:\n\n{items}\n\n"
            "For each one explain the correct reasoning, name the misconception behind the error, and give "
            "a general rule the student can apply next time. Stay in character.", max_tokens=4000)


# ---------------------------------------------------------------------------
# Student
# ---------------------------------------------------------------------------

@dataclass
class Student:
    llm: LLM
    store: Store
    classroom_id: str
    name: str = "Student"
    notes_k: int = 6

    def _system(self) -> str:
        return (f"You are {self.name}, an AI student in the AI Room. Your weights never change, so the "
                "only way you learn is by writing good notes in your notebook and re-reading them. "
                "Be curious, honest about what you don't understand, and careful in your reasoning.")

    def _ask(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2000) -> str:
        return self.llm.chat([Message("system", self._system()), Message("user", prompt)],
                             temperature=temperature, max_tokens=max_tokens).strip()

    def recall(self, subject: str, query: str) -> list[Note]:
        return self.store.search_notes(self.classroom_id, subject, query, k=self.notes_k)

    @staticmethod
    def _notebook(notes: list[Note]) -> str:
        if not notes:
            return "(your notebook has nothing relevant yet)"
        return "\n\n".join(f"[{n.kind}, class {n.grade}] {n.content}" for n in notes)

    def take_notes(self, subject: str, grade: int, lesson: str) -> str:
        return self._ask(
            f"[[NOTES]]\n{_grade_line(subject, grade)}\n\nThe teacher's lesson:\n{lesson}\n\n"
            "Write concise study notes for your notebook: definitions, formulas, step-by-step methods, a short "
            "worked example and pitfalls. Use bullet points. These notes are all you will have in the exam.",
            max_tokens=2000)

    def ask_questions(self, subject: str, grade: int, lesson: str, max_q: int) -> list[str]:
        if max_q <= 0:
            return []
        reply = self._ask(
            f"[[QUESTIONS]]\n{_grade_line(subject, grade)}\n\nThe teacher's lesson:\n{lesson}\n\n"
            f"What is unclear or worth digging deeper into? Ask up to {max_q} short questions, one per line, "
            "numbered. If everything is clear, reply exactly NONE.", temperature=0.6, max_tokens=500)
        if reply.strip().upper().startswith("NONE"):
            return []
        qs = [re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", ln).strip() for ln in reply.splitlines()]
        return [q for q in qs if len(q) > 5][:max_q]

    def answer(self, subject: str, grade: int, q: dict) -> str:
        notes = self.recall(subject, q["question"] + " " + " ".join(q.get("options", [])))
        body = q["question"]
        if q["type"] == "mcq":
            body += "\n" + "\n".join(q.get("options", []))
            fmt = "Reason step by step, then end with a line 'FINAL ANSWER: <option letter>'."
        elif q["type"] == "numeric":
            fmt = "Reason step by step, then end with a line 'FINAL ANSWER: <number only>'."
        elif q["type"] == "code":
            fmt = "Reply with the complete Python solution in a single ```python code block."
        else:
            fmt = "Answer clearly and completely in a few sentences."
        return self._ask(
            f"[[ANSWER]]\n{_grade_line(subject, grade)}\n\nRelevant pages from your notebook:\n"
            f"{self._notebook(notes)}\n\nExam question:\n{body}\n\n{fmt}", temperature=0.2, max_tokens=2500)

    def reflect(self, subject: str, grade: int, review: str, mistakes: list[dict]) -> str:
        items = "\n".join(f"- {m['question']} (you said {m['given']}, correct: {m['expected']})" for m in mistakes)
        return self._ask(
            f"[[REFLECT]]\n{_grade_line(subject, grade)}\n\nYou got these wrong:\n{items}\n\n"
            f"The teacher's review:\n{review}\n\n"
            "Write mistake-log notes for your notebook: for each error, the general rule or method that "
            "prevents it in the future. Do not just copy answers — write lessons that transfer to new questions.",
            max_tokens=1500)

    def summarize(self, subject: str, grade: int) -> str:
        notes = self.store.notes(self.classroom_id, subject, grade)
        joined = "\n\n".join(n.content for n in notes)[-12000:]
        return self._ask(
            f"[[SUMMARY]]\n{_grade_line(subject, grade)}\n\nAll your notes from this class:\n{joined}\n\n"
            "You passed this class. Write a compact cheat-sheet of the most important facts, methods and "
            "lessons-from-mistakes to carry into the next classes.", max_tokens=1500)

    def chat(self, message: str, history: list[dict], grade: int, subjects: list[str]) -> str:
        notes: list[Note] = []
        for s in subjects:
            notes += self.recall(s, message)[:2]
        msgs = [Message("system", self._system() + f"\nYou are currently in class {grade} of 12, studying "
                        f"{', '.join(SUBJECTS[s].name for s in subjects)}.\nRelevant notebook pages:\n"
                        + self._notebook(notes))]
        for h in history[-12:]:
            msgs.append(Message("assistant" if h["role"] == "student" else "user", f"{h['speaker']}: {h['content']}"))
        msgs.append(Message("user", message))
        return self.llm.chat(msgs, temperature=0.7, max_tokens=1200).strip()
