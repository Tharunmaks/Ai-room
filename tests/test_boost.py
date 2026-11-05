"""`ai-room boost`: better answers without training, measured on the same held-out exam as `ai-room train`."""

import re

import pytest

from ai_room.boost import BoostConfig, Notebook, apply_calculator, build_prompt, mistake_page, run, study_page
from ai_room.llm import LLM
from ai_room.train_cpu import make_problems


class Scripted(LLM):
    """Answers from a rule; counts how it was called."""
    name = "scripted-1.5b"

    def __init__(self, rule):
        self.rule, self.calls = rule, 0

    def chat_with_usage(self, messages, temperature=0.7, max_tokens=2048):
        self.calls += 1
        return self.rule(messages[-1].content, temperature, self.calls), {"prompt_tokens": len(messages[-1].content.split()), "completion_tokens": 5, "completion_seconds": 0.01}

    def chat(self, messages, temperature=0.7, max_tokens=2048):
        return self.chat_with_usage(messages, temperature, max_tokens)[0]

    def parameter_count_b(self):
        return 1.5


def test_exam_is_the_same_one_ai_room_train_uses():
    a = make_problems(["math"], [1, 2, 3], 100, 7 + 1000)
    b = make_problems(["math"], [1, 2, 3], 100, 1007)
    assert [q["question"] for q in a] == [q["question"] for q in b]


def test_notebook_finds_pages_on_the_same_topic():
    nb = Notebook()
    for q in make_problems(["math"], [1, 2, 3, 4, 5], 200, seed=1):
        nb.add(study_page(q), q["family"], "study")
    q = next(q for q in make_problems(["math"], [1, 2, 3, 4, 5], 50, seed=99) if q["family"] == "Fractions")
    found = nb.search(q["question"] + " " + q["family"], 3)
    assert len(found) == 3 and sum(p["topic"] == "Fractions" for p in found) >= 2
    assert "Pages from your notebook" in build_prompt(q, nb, 3, False) and q["question"] in build_prompt(q, nb, 3, False)
    assert build_prompt(q, None, 3, False).startswith(q["question"].split()[0])


def test_mistake_pages_record_the_wrong_answer_and_the_right_method():
    q = make_problems(["math"], [2], 1, seed=3)[0]
    page = mistake_page(q, "999")
    assert "999" in page and f"FINAL ANSWER: {q['answer']}" in page


@pytest.mark.parametrize("reply,expected,changed", [
    ("so FINAL ANSWER: (120 - 45) * 3", "225", True),
    ("FINAL ANSWER: 120/2", None, False),     # a/b is left for the grader, which already reads it as a fraction
    ("FINAL ANSWER: 2^10", "1024", True),
    ("FINAL ANSWER: 3/4", None, False),        # a plain fraction is an answer, not a sum to work out
    ("FINAL ANSWER: -12", None, False),
    ("FINAL ANSWER: 15%", None, False),
    ("FINAL ANSWER: x = 4", None, False),
    ("FINAL ANSWER: __import__('os')", None, False),
])
def test_calculator_only_evaluates_arithmetic(reply, expected, changed):
    q = {"type": "numeric", "question": "?", "answer": "0"}
    out, did = apply_calculator(q, reply)
    assert did is changed
    if changed:
        assert out.rstrip().endswith(f"FINAL ANSWER: {expected}")


def test_grader_already_reads_simple_fractions():
    from ai_room.grading import grade_question
    assert grade_question({"type": "numeric", "answer": "60"}, "FINAL ANSWER: 120/2")[0] == 1.0


def test_calculator_leaves_multiple_choice_alone():
    assert apply_calculator({"type": "mcq", "answer": "A"}, "FINAL ANSWER: 2+2") == ("FINAL ANSWER: 2+2", False)


def keyed_rule(exam_answers):
    """A student that is right only when its notebook is in the prompt, so the test controls exactly what changes the score."""
    def rule(prompt, temperature, n):
        q = re.findall(r"Topic: [^\n]*\n(.*?)\n", prompt) or [prompt.split("\n")[0]]
        key = exam_answers.get(q[-1].strip())
        if "Pages from your notebook" in prompt and key is not None:
            return f"FINAL ANSWER: {key}"
        return "FINAL ANSWER: -424242"
    return rule


def test_run_reports_each_method_on_the_same_exam(tmp_path):
    cfg = BoostConfig(eval_examples=20, study_examples=80, practice=10, methods=["plain", "notes", "mistakes"], out=str(tmp_path / "r.json"))
    exam = make_problems(cfg.subjects, cfg.grades, cfg.eval_examples, cfg.seed + 1000)
    key = {q["question"]: q["answer"] for q in exam}
    llm = Scripted(keyed_rule(key))
    rep = run(llm, cfg, log=lambda m: None)
    names = [r["name"] for r in rep["results"]]
    assert names == ["plain", "plain + notes", "plain + notes + mistakes"]
    assert rep["results"][0]["accuracy"] == 0.0
    assert rep["results"][1]["accuracy"] > 0.5           # the notebook is what changed the result, not the weights
    assert [r["question"] for r in rep["results"][0]["rows"]] == [r["question"] for r in rep["results"][2]["rows"]]
    assert rep["practice"]["mistakes_written"] >= 1
    assert (tmp_path / "r.json").exists()
    assert rep["results"][0]["speed"]["completion_tokens"] == 20 * 5


def test_vote_takes_the_most_common_answer(tmp_path):
    exam = make_problems(["math"], [1], 5, 7 + 1000)
    key = {q["question"]: q["answer"] for q in exam}

    def rule(prompt, temperature, n):
        q = prompt.split("\n")[0]
        return f"FINAL ANSWER: {key.get(q, 0)}" if n % 3 else "FINAL ANSWER: 1234567"   # wrong one time in three
    rep = run(Scripted(rule), BoostConfig(grades=[1], eval_examples=5, methods=["plain", "vote"], votes=3, out=str(tmp_path / "v.json")), log=lambda m: None)
    assert rep["results"][-1]["name"] == "plain + vote" and rep["results"][-1]["accuracy"] == 1.0


def test_a_failing_model_counts_as_wrong_answers_not_a_crash(tmp_path):
    class Down(Scripted):
        def chat_with_usage(self, *a, **k):
            raise ConnectionError("server down")
    rep = run(Down(lambda *a: ""), BoostConfig(eval_examples=4, methods=["plain"], out=str(tmp_path / "d.json")), log=lambda m: None)
    assert rep["results"][0]["accuracy"] == 0.0 and "did not answer" in rep["results"][0]["rows"][0]["reply"]


def test_unknown_methods_are_refused(tmp_path):
    with pytest.raises(ValueError):
        run(Scripted(lambda *a: ""), BoostConfig(methods=["telepathy"], out=str(tmp_path / "x.json")), log=lambda m: None)
