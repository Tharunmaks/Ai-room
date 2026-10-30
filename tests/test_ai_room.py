import json

import pytest
from fastapi.testclient import TestClient

from ai_room.classroom import ClassConfig, Classroom
from ai_room.grading import extract_final, mcq_match, numeric_match, parse_number, run_code_tests
from ai_room.llm import MockLLM, ModelTooLargeError, check_student_size, parse_param_size
from ai_room.memory import Store
from ai_room.server import create_app


# -- 27B guard ---------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("gemma3:27b", 27), ("27.4B", 27.4), ("qwen2.5-14b-instruct", 14), ("llama3.1:70b", 70),
    ("mixtral-8x7b", 56), ("qwen3-30b-a3b", 30), ("smollm-360m", 0.36), ("mistral", None),
])
def test_parse_param_size(text, expected):
    assert parse_param_size(text) == expected


def test_size_guard():
    assert check_student_size(MockLLM(size_b=27.4)) == 27.4  # Gemma 3 "27B" reports 27.4B
    with pytest.raises(ModelTooLargeError):
        check_student_size(MockLLM(size_b=32))
    with pytest.raises(ModelTooLargeError):
        check_student_size(MockLLM(size_b=None))
    assert check_student_size(MockLLM(size_b=None), declared_b=8) == 8


# -- grading -----------------------------------------------------------------

def test_answer_extraction_and_matching():
    assert extract_final("work...\nFINAL ANSWER: **42**") == "42"
    assert parse_number("about 3/4 of it") == 0.75
    assert parse_number("1,200 m") == 1200
    assert parse_number("3.0 x 10^8 m/s") == 3e8
    assert parse_number("25%") == 0.25
    assert numeric_match("9.79", "9.8")
    assert not numeric_match("10.5", "9.8")
    assert mcq_match("(B) because...", "B")
    assert not mcq_match("C", "B")


def test_code_tests():
    code = "def add(a, b):\n    return a + b"
    assert run_code_tests(code, ["assert add(1, 2) == 3", "assert add(0, 0) == 1"])[:2] == (1, 2)
    assert run_code_tests("while True: pass", ["assert True"], timeout=1)[0] == 0


# -- notebook retrieval ------------------------------------------------------

def test_notebook_search_prefers_relevant_and_mistakes():
    s = Store(":memory:")
    s.add_note("c", "physics", 1, "lesson", "Newton's second law: force equals mass times acceleration.")
    s.add_note("c", "physics", 1, "lesson", "Sound travels as a longitudinal wave.")
    s.add_note("c", "physics", 1, "mistake", "Remember: force = mass x acceleration, convert grams to kg first.")
    hits = s.search_notes("c", "physics", "What force accelerates a mass of 2 kg?")
    assert hits[0].kind == "mistake"
    assert all("wave" not in h.content for h in hits)


# -- full classroom loop with scripted models --------------------------------

def make_room(student_correct: list[bool], subjects=("math",), questions=4, attempts=3):
    """Teacher writes numeric exams; the student is right/wrong per call."""
    store = Store(":memory:")
    cfg = ClassConfig.from_dict({"subjects": list(subjects), "questions_per_exam": questions,
                                 "max_attempts": attempts, "verify_exams": False, "student_questions": 1})
    cid = store.create_classroom("t", cfg.to_dict(redact=False))
    answers = iter(student_correct)

    def student(msgs):
        p = msgs[-1].content
        if "[[ANSWER]]" in p:
            return "FINAL ANSWER: 7" if next(answers, True) else "FINAL ANSWER: 8"
        if "[[QUESTIONS]]" in p:
            return "1. Why is that?"
        return "- a note"

    def teacher(msgs):
        p = msgs[-1].content
        if "[[EXAM]]" in p:
            return json.dumps({"questions": [{"type": "numeric", "question": f"3+4 ({i})", "answer": "7"}
                                             for i in range(questions)]})
        return "lesson text"

    room = Classroom(store, cid, MockLLM(responder=student, size_b=14), MockLLM(responder=teacher))
    return room, store, cid


def test_promotion_requires_pass_mark_in_every_subject():
    # math: 4/4 correct, physics: 3/4 = 75% -> stays in class 1
    room, store, cid = make_room([True] * 4 + [True, True, True, False], subjects=("math", "physics"))
    result = room.exam()
    assert not result["promoted"] and result["remaining"] == ["physics"]
    assert room.grade == 1
    assert store.notes(cid, "physics", kind="mistake")  # reflection was written down
    # retake: only physics is re-examined, and it passes
    result = room.exam()
    assert [r["subject"] for r in result["results"]] == ["physics"]
    assert result["promoted"] and room.grade == 2
    assert store.notes(cid, "math", 1, "summary")


def test_run_class_stops_after_max_attempts():
    room, store, _ = make_room([False] * 100, attempts=2)
    assert not room.run_class()["promoted"]
    assert store.last_attempt(room.id, 1) == 2 and room.grade == 1


def test_lesson_stores_notes_and_qa():
    room, store, cid = make_room([])
    room.lesson("math")
    kinds = {n.kind for n in store.notes(cid, "math")}
    assert kinds == {"lesson", "qa"}


# -- API ---------------------------------------------------------------------

def test_api_rejects_large_student_and_runs_mock(tmp_path):
    client = TestClient(create_app(str(tmp_path / "db.sqlite")))
    bad = client.post("/api/classrooms", json={"config": {"student": {"backend": "openai", "model": "llama3.1-70b"}}})
    assert bad.status_code == 400 and "27" in bad.json()["detail"]

    cfg = {"student": {"backend": "mock", "model": "mock-1b"}, "teacher": {"backend": "mock", "model": "mock"},
           "subjects": ["math"], "questions_per_exam": 3}
    cid = client.post("/api/classrooms", json={"config": cfg}).json()["id"]
    assert client.post(f"/api/classrooms/{cid}/talk", json={"text": "hi"}).status_code == 200
    assert client.patch(f"/api/classrooms/{cid}/config", json={"persona": "socratic"}).status_code == 200
    room = client.get(f"/api/classrooms/{cid}").json()
    assert room["teacher"]["name"] == "Professor Socrates" and room["grade"] == 1
    assert client.get("/api/meta").json()["grades"] == 12


# -- subject catalogue -------------------------------------------------------

def test_catalog_has_100_subjects_with_12_classes():
    from ai_room.curriculum import CATEGORIES, SUBJECTS
    assert len(SUBJECTS) == 100
    assert len(CATEGORIES) == 9
    for s in SUBJECTS.values():
        assert len(s.grades) == 12 and s.question_types
        assert set(s.question_types) <= {"numeric", "mcq", "short", "code"}


def test_mock_classroom_passes_every_question_type():
    store = Store(":memory:")
    cfg = ClassConfig.from_dict({"subjects": ["code", "spanish", "statistics", "history"], "questions_per_exam": 2,
                                 "student": {"backend": "mock", "model": "m"}, "teacher": {"backend": "mock", "model": "m"}})
    cid = store.create_classroom("t", cfg.to_dict(redact=False))
    assert Classroom(store, cid).exam()["promoted"]
