import json

import pytest

from ai_room.classroom import ClassConfig, Classroom
from ai_room.llm import MockLLM
from ai_room.memory import Store
from ai_room.teacher import Library


@pytest.fixture
def lib():
    return Library(":memory:")


def scripted_student(pattern):
    """A student that answers using the answer key it is handed (demo oracle): right or wrong as `pattern` says.

    `pattern(n)` receives how many exam answers have been given so far and returns True for a correct one.
    """
    state = {"n": 0}

    def respond(msgs):
        import re
        p = msgs[-1].content
        if "[[ANSWER]]" in p:
            key = re.search(r"\[\[DEMO-KEY\]\]\s*(.*)\Z", p, re.S).group(1).strip()
            right = pattern(state["n"])
            state["n"] += 1
            if "complete Python solution" in p:
                return f"```python\n{key if right else 'def x(*a):\n    return None'}\n```"
            if right:
                return f"FINAL ANSWER: {key}"
            return "FINAL ANSWER: " + ("Z" if "option letter" in p else "-12345")
        if "[[QUESTIONS]]" in p:
            return "1. Why does this method work?"
        return "- a note: " + p[-60:].replace("\n", " ")

    return MockLLM(responder=respond, demo_oracle=True, size_b=14)


def make_room(subjects=("math",), questions=4, attempts=3, pattern=lambda n: True, library=None, **cfg_extra):
    store = Store(":memory:")
    cfg = ClassConfig.from_dict({"subjects": list(subjects), "questions_per_exam": questions, "max_attempts": attempts, "student_questions": 1,
                                 "auto_fetch": False, "student": {"backend": "mock", "model": "m"}, **cfg_extra})
    cid = store.create_classroom("t", cfg.to_dict(redact=False))
    library = library or Library(":memory:")
    room = Classroom(store, cid, library, scripted_student(pattern))
    return room, store, cid
