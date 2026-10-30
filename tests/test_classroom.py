import pytest

from conftest import make_room, scripted_student
from ai_room.classroom import ClassConfig, Classroom
from ai_room.curriculum import NUM_GRADES, SUBJECTS
from ai_room.llm import MockLLM, ModelTooLargeError, check_student_size, make_llm, parse_param_size
from ai_room.memory import Store
from ai_room.teacher import Library, NoCourseMaterial


@pytest.mark.parametrize("text,expected", [("gemma3:27b", 27), ("27.4B", 27.4), ("qwen2.5-14b-instruct", 14), ("llama3.1:70b", 70), ("mixtral-8x7b", 56),
                                           ("qwen3-30b-a3b", 30), ("smollm-360m", 0.36), ("mistral", None)])
def test_parse_param_size(text, expected):
    assert parse_param_size(text) == expected


def test_student_size_guard():
    assert check_student_size(MockLLM(size_b=27.4)) == 27.4  # Gemma 3 "27B" reports 27.4B
    with pytest.raises(ModelTooLargeError):
        check_student_size(MockLLM(size_b=32))
    with pytest.raises(ModelTooLargeError):
        check_student_size(MockLLM(size_b=None))
    assert check_student_size(MockLLM(size_b=None), declared_b=8) == 8


def test_the_teacher_cannot_be_a_model():
    for backend in ("anthropic", "claude", "gpt"):
        with pytest.raises(ValueError, match="not an AI"):
            make_llm(backend, "x")
    cfg = ClassConfig.from_dict({"teacher": {"backend": "ollama", "model": "gemma3:27b"}, "custom_style": "x", "subjects": ["math"]})  # old settings are ignored
    assert not hasattr(cfg, "teacher")


def test_config_validation():
    with pytest.raises(ValueError):
        ClassConfig.from_dict({"subjects": ["nope"]})
    with pytest.raises(ValueError):
        ClassConfig.from_dict({"subjects": ["math"], "pass_mark": 1.5})
    with pytest.raises(ValueError):
        ClassConfig.from_dict({"subjects": ["math"], "persona": "robot"})


def test_promotion_needs_the_pass_mark_in_every_subject():
    # math is perfect, physics gets 3 of 4 (75%) → nobody moves up
    room, store, cid = make_room(("math", "physics"), pattern=lambda n: n != 7)
    result = room.exam()
    assert not result["promoted"] and result["remaining"] == ["physics"] and room.grade == 1
    assert store.notes(cid, "physics", kind="mistake")  # the student wrote a mistake log
    assert any(m["role"] == "teacher" and m["meta"].get("kind") == "review" for m in store.messages(cid))
    # the re-sit covers only the failed subject, with brand-new questions, and then promotes
    first = {i["question"] for e in store.exams(cid) for i in e["details"]["items"] if e["subject"] == "physics"}
    result = room.exam()
    assert [r["subject"] for r in result["results"]] == ["physics"] and result["promoted"] and room.grade == 2
    second = {i["question"] for e in store.exams(cid) if e["attempt"] == 2 for i in e["details"]["items"]}
    assert not (first & second)
    assert store.notes(cid, "math", 1, "summary")


def test_exactly_the_pass_mark_passes():
    room, _, _ = make_room(("math",), questions=10, pattern=lambda n: n != 4)  # 9/10 = 90%
    assert room.exam()["promoted"]


def test_failed_subjects_are_retaught_with_a_focus_then_the_class_is_held_after_max_attempts():
    room, store, cid = make_room(("math",), attempts=2, pattern=lambda n: False)
    result = room.run_class()
    assert not result["promoted"] and store.last_attempt(cid, 1) == 2 and room.grade == 1
    lessons = [m for m in store.messages(cid) if m["meta"].get("kind") == "lesson"]
    assert len(lessons) == 2 and "revisit" in lessons[1]["content"]  # the second lesson targets the gaps
    assert any(m["meta"].get("kind") == "held" for m in store.messages(cid))


def test_full_run_graduates():
    room, store, cid = make_room(("math", "logic"), questions=3)
    room.run_all()
    assert room.graduated and room.grade == NUM_GRADES + 1
    assert any("GRADUATED" in m["content"] for m in store.messages(cid))
    assert len(store.notes(cid, "math", kind="summary")) == NUM_GRADES


def test_lesson_stores_notes_and_teacher_answers_the_students_questions():
    room, store, cid = make_room(("math",))
    room.lesson("math")
    assert {n.kind for n in store.notes(cid, "math")} == {"lesson", "qa"}
    kinds = [m["meta"].get("kind") for m in store.messages(cid)]
    assert "question" in kinds and "answer" in kinds


def test_all_four_answer_types_are_graded_deterministically():
    room, _, _ = make_room(("math",))
    assert room._grade_one({"type": "numeric", "answer": "200928", "tol": 0.01, "question": "q"}, "FINAL ANSWER: 201000")[0] == 1
    assert room._grade_one({"type": "numeric", "answer": "7", "question": "q"}, "FINAL ANSWER: 7.5")[0] == 0
    assert room._grade_one({"type": "mcq", "answer": "B", "question": "q"}, "so... FINAL ANSWER: (B)")[0] == 1
    assert room._grade_one({"type": "cloze", "answer": "mitochondrion", "question": "q"}, "FINAL ANSWER: The Mitochondrion.")[0] == 1
    assert room._grade_one({"type": "cloze", "answer": "3 1 * 9 +", "exact": True, "question": "q"}, "FINAL ANSWER: 3 1 9")[0] == 0
    code = {"type": "code", "tests": ["assert f(2) == 4", "assert f(3) == 9"], "question": "q", "answer": ""}
    assert room._grade_one(code, "```python\ndef f(x):\n    return x * x\n```")[0] == 1
    assert room._grade_one(code, "```python\ndef f(x):\n    return 4\n```")[0] == 0.5


def test_textbook_subject_is_learned_from_notes_and_missing_books_are_fetched():
    text = ("Stoicism is a school of philosophy founded in Athens by Zeno of Citium. Virtue is the highest good according to the Stoics and their teachers. "
            "Apathy is the freedom from disturbing passions that the Stoics sought to achieve. Epictetus was a former slave who became an influential Stoic teacher. "
            "Seneca was a Roman statesman and Stoic writer who advised the emperor. Logic is the study of correct reasoning and valid arguments. ") * 2
    calls = []

    def fake_fetch(lib, subject, progress=None, stop=None, **kw):
        calls.append(subject)
        lib.add_document(subject, "Stoicism", text, source="Fake", topic="asking philosophical questions")
        return {"added": 1, "skipped": 0, "errors": []}

    store = Store(":memory:")
    cfg = ClassConfig.from_dict({"subjects": ["philosophy"], "questions_per_exam": 5, "student": {"backend": "mock", "model": "m"}})
    cid = store.create_classroom("t", cfg.to_dict(redact=False))
    # the honest stand-in student can only answer from its own notebook
    room = Classroom(store, cid, Library(":memory:"), make_llm("mock", "mock-1b"), fake_fetch)
    result = room.run_class()
    assert calls == ["philosophy"] and result["promoted"]
    items = store.exams(cid)[-1]["details"]["items"]
    assert items and all(i["concept"].startswith("kb:") and i["score"] == 1 for i in items)


def test_missing_textbooks_stop_the_class_with_instructions_when_fetching_is_off():
    room, _, _ = make_room(("history",))
    with pytest.raises(NoCourseMaterial, match="Library"):
        room.lesson("history")


def test_demo_student_is_the_only_one_ever_given_the_key():
    honest = []
    llm = MockLLM(responder=lambda m: honest.append(m[-1].content) or "FINAL ANSWER: 1", size_b=1)
    room, _, _ = make_room(("math",))
    room.student.llm = llm
    room.student.answer("math", 1, {"type": "numeric", "question": "What is 1 + 1?", "answer": "2"})
    assert "DEMO-KEY" not in honest[0] and "2" not in honest[0].split("Exam question:")[1].split("FINAL")[0].replace("What is 1 + 1?", "")


def test_every_subject_has_a_syllabus_for_all_12_classes():
    assert len(SUBJECTS) == 100 and all(len(s.grades) == 12 for s in SUBJECTS.values())
