"""The student: the only AI in the classroom.

Its weights never change. It learns by writing notes after each lesson, asking
the teacher questions, re-reading the relevant notebook pages before every exam
question, and writing mistake logs after a failed exam.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .curriculum import SUBJECTS, difficulty, topics_for
from .llm import LLM, Message
from .memory import Note, Store


def _grade_line(subject: str, grade: int) -> str:
    return (f"Subject: {SUBJECTS[subject].name}. Class {grade} of 12 ({difficulty(grade)} level).\n"
            f"Syllabus for this class: {topics_for(subject, grade)}.")


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
            f"[[NOTES]]\n{_grade_line(subject, grade)}\n\nThe teacher's lesson:\n{lesson[:14000]}\n\n"
            "Write concise study notes for your notebook: definitions (keep the exact wording of key terms), formulas, "
            "step-by-step methods, a short worked example and pitfalls. Use bullet points. These notes are all you will have in the exam.",
            max_tokens=2500)

    def ask_questions(self, subject: str, grade: int, lesson: str, max_q: int) -> list[str]:
        if max_q <= 0:
            return []
        reply = self._ask(
            f"[[QUESTIONS]]\n{_grade_line(subject, grade)}\n\nThe teacher's lesson:\n{lesson[:14000]}\n\n"
            f"What is unclear or worth digging deeper into? Ask up to {max_q} short questions, one per line, "
            "numbered. If everything is clear, reply exactly NONE.", temperature=0.6, max_tokens=500)
        if reply.strip().upper().startswith("NONE"):
            return []
        qs = [re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", ln).strip() for ln in reply.splitlines()]
        return [q for q in qs if len(q) > 5][:max_q]

    def answer(self, subject: str, grade: int, q: dict) -> str:
        # the family/source title matches the headings of the lesson the student took notes on
        hint = q.get("family") or (q.get("source") or {}).get("title") or ""
        notes = self.recall(subject, f"{hint} {q['question']} {' '.join(q.get('options', []))}")
        body = q["question"]
        if q["type"] == "mcq":
            body += "\n" + "\n".join(q.get("options", []))
            fmt = "Reason step by step, then end with a line 'FINAL ANSWER: <option letter>'."
        elif q["type"] == "numeric":
            fmt = "Reason step by step, then end with a line 'FINAL ANSWER: <number only>'."
        elif q["type"] == "code":
            fmt = "Reply with the complete Python solution in a single ```python block."
        else:
            fmt = "End with a line 'FINAL ANSWER: <the missing term or the exact answer>' — use the wording from your notes when you remember it."
        prompt = (f"[[ANSWER]]\n{_grade_line(subject, grade)}\n\nRelevant pages from your notebook:\n"
                  f"{self._notebook(notes)}\n\nExam question:\n{body}\n\n{fmt}")
        if getattr(self.llm, "demo_oracle", False):
            studied = bool(self.store.notes(self.classroom_id, subject))
            prompt += f"\n[[DEMO-STUDIED]] {int(studied)}\n[[DEMO-KEY]] {q['answer']}"  # only the simulated demo student is ever given the key
        return self._ask(prompt, temperature=0.2, max_tokens=2500)

    def reflect(self, subject: str, grade: int, review: str, mistakes: list[dict]) -> str:
        items = "\n".join(f"- {m['question'].splitlines()[0][:200]} (you said {m['given']}, correct: {m['expected']})" for m in mistakes)
        return self._ask(
            f"[[REFLECT]]\n{_grade_line(subject, grade)}\n\nYou got these wrong:\n{items}\n\n"
            f"The teacher's review:\n{review[:6000]}\n\n"
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
