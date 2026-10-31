"""The CPU trainer's data and bookkeeping (the training run itself needs torch and a model download, so it is not run here)."""

from ai_room.grading import grade_question
from ai_room.train_cpu import make_problems, parse_grades, prompt_of, target_of


def test_training_data_has_code_computed_answers_that_the_grader_accepts():
    probs = make_problems(["math", "physics"], [1, 2, 3, 4], 200, seed=1)
    assert len(probs) == 200 and len({p["question"] for p in probs}) == 200
    for p in probs:
        assert p["type"] != "code"
        score, _ = grade_question(p, target_of(p))          # the training target is graded correct by the exam grader
        assert score == 1.0, (p["question"], target_of(p))
        assert "FINAL ANSWER" in prompt_of(p)


def test_held_out_exam_never_overlaps_training_data():
    exam = make_problems(["math"], [1, 2, 3], 100, seed=1007)
    train = make_problems(["math"], [1, 2, 3], 1000, seed=7, avoid={q["question"] for q in exam})
    assert not {q["question"] for q in exam} & {q["question"] for q in train}


def test_mcq_prompts_show_the_options():
    p = next(q for q in make_problems(["math", "chemistry", "physics", "logic"], list(range(1, 13)), 400, seed=3) if q["type"] == "mcq")
    assert "A)" in prompt_of(p) and target_of(p).endswith(f"FINAL ANSWER: {p['answer']}")


def test_parse_grades():
    assert parse_grades("1-3") == [1, 2, 3]
    assert parse_grades("2,5-6,40") == [2, 5, 6]
