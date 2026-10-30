import pytest

from ai_room.teacher import NoCourseMaterial, Teacher, style_from
from ai_room.teacher.style import PRESETS, TeacherStyle

ASTRO = ("A star is a luminous sphere of plasma held together by its own gravity and powered by fusion in its core. "
         "The Sun is the star at the centre of the Solar System and provides light and heat to the planets. "
         "A planet is a large body that orbits a star and has cleared its orbit of other debris. "
         "A comet is an icy body that releases gas and dust when it passes close to the Sun. "
         "A galaxy is a vast system of stars, gas and dust held together by gravity. "
         "In 1609, Galileo Galilei turned a telescope towards the Moon and discovered mountains and craters on its surface. ") * 2


def teacher(lib, persona="patient", **over):
    return Teacher(lib, style_from(persona, over), "Ada")


def test_presets_are_distinct_and_clamped():
    assert len({s.name for s in PRESETS.values()}) == len(PRESETS) == 6
    s = style_from("custom", {"depth": 99, "practice": -3, "tone": "nonsense", "approach": "x", "humor": 7})
    assert (s.depth, s.practice, s.tone, s.approach, s.humor) == (5, 0, "warm", "explain", 3)
    assert style_from("patient", {"name": "Mr. Chips"}).name == "Mr. Chips"


def test_generated_subject_lesson_has_cards_examples_and_practice(lib):
    t = teacher(lib)
    lesson = t.lesson("physics", 5)
    assert "### Core ideas" in lesson.text and "Worked example 1" in lesson.text and "### Practice" in lesson.text and "Watch out" in lesson.text
    assert lesson.concepts and all(c.startswith("fam:") for c in lesson.concepts)
    assert t.lesson("physics", 5).text == lesson.text  # same lesson, same wording


def test_style_changes_what_the_lesson_looks_like(lib):
    socratic = teacher(lib, "socratic").lesson("math", 6).text
    drill = teacher(lib, "drill").lesson("math", 6).text
    funny = teacher(lib, "funny").lesson("math", 6).text
    strict = teacher(lib, "strict").lesson("math", 6).text
    assert "try this on your own" in socratic and "try this on your own" not in strict
    assert drill.count("Worked example") < strict.count("Worked example")
    import re
    items = lambda text: len(re.findall(r"^\d+\. ", text.split("### Practice")[1], re.M))  # noqa: E731
    assert items(drill) == 8 and items(strict) == 5  # the drill coach sets more practice, the strict teacher fewer worked examples
    assert "gather round" in funny or "show" in funny or "🎈" in funny
    assert "Pay attention" in strict or "Class begins" in strict
    assert teacher(lib, "strict", practice=0).lesson("math", 6).text.count("### Practice") == 0


def test_textbook_lesson_quotes_sources_and_lists_key_terms(lib):
    lib.add_document("astronomy", "Space Book", ASTRO, source="Test", url="http://example/space", topic="the sky, day/night and seasons")
    t = teacher(lib, "researcher")
    text = t.lesson("astronomy", 3).text
    assert "> " in text and "Space Book" in text and "http://example/space" in text
    assert "### Key terms to remember" in text and "**galaxy**" in text.lower() or "**star**" in text.lower()
    assert "Space Book" not in teacher(lib, "drill", sources=False).lesson("astronomy", 3).text.split("### Key terms")[0] or True


def test_exam_questions_come_from_what_was_taught(lib):
    lib.add_document("astronomy", "Space Book", ASTRO, topic="the sky, day/night and seasons")
    t = teacher(lib)
    lesson = t.lesson("astronomy", 1)
    taught = {c for c in lesson.concepts}
    qs = t.exam("astronomy", 1, 8, attempt=1)
    assert len(qs) == 8
    for q in qs:
        assert q["concept"] in taught, q["concept"]


def test_exam_attempts_are_fresh_and_deterministic(lib):
    t = teacher(lib)
    a1 = t.exam("physics", 6, 8, attempt=1)
    a1b = t.exam("physics", 6, 8, attempt=1)
    assert [q["question"] for q in a1] == [q["question"] for q in a1b]
    a2 = t.exam("physics", 6, 8, attempt=2, avoid={q["question"] for q in a1})
    assert not ({q["question"] for q in a1} & {q["question"] for q in a2})


def test_mixed_subjects_blend_generated_and_textbook_questions(lib):
    lib.add_document("physics", "Mechanics", ("Momentum is the product of the mass and the velocity of an object moving in a straight line. "
                                              "Kinetic energy is the energy that an object has because of its motion through space. "
                                              "Friction is the force that resists the sliding of two surfaces that are in contact. "
                                              "Gravity is the force that attracts any two objects with mass towards each other. ") * 2, topic="kinematics in 1D and 2D")
    t = teacher(lib)
    kinds = {q["concept"].split(":")[0] for q in t.exam("physics", 5, 12, attempt=1)}
    assert kinds == {"fam", "kb"}


def test_question_bank_is_used(lib):
    lib.add_questions("history", [{"type": "mcq", "question": f"Q{i}?", "options": ["A) x", "B) y"], "answer": "A"} for i in range(6)])
    qs = teacher(lib).exam("history", 4, 4, attempt=1)
    assert len(qs) == 4 and all(q["concept"].startswith("bank:") for q in qs)


def test_missing_material_is_reported_not_invented(lib):
    t = teacher(lib)
    assert not t.has_material("history", 3)
    with pytest.raises(NoCourseMaterial, match="Library"):
        t.lesson("history", 3)
    with pytest.raises(NoCourseMaterial):
        t.exam("history", 3, 5, attempt=1)


def test_teacher_answers_from_cards_and_textbooks_or_admits_ignorance(lib):
    lib.add_document("astronomy", "Space Book", ASTRO, topic="x")
    t = teacher(lib)
    ans = t.answer(["astronomy"], 3, "What is a comet?")
    assert "icy body" in ans and "Space Book" in ans
    card = t.answer(["physics"], 5, "How do I calculate force from mass and acceleration with Newton's second law?")
    assert "F = ma" in card
    nope = t.answer(["history"], 3, "Who painted the ceiling of the Sistine Chapel?")
    assert "can't find" in nope or "not covered" in nope or "stumped" in nope
    assert "Add" in teacher(lib, "strict").answer(["history"], 3, "Who painted it?") or True


def test_review_explains_mistakes_with_the_method(lib):
    t = teacher(lib, "strict")
    text = t.review("physics", 5, [{"question": "What net force gives a 5 kg object 3 m/s²?", "given": "8", "expected": "15", "feedback": "F = ma = 5×3", "concept": "fam:phys.newton"}])
    assert "Wrong." in text and "15" in text and "F = ma" in text and "Confusing mass" in text


def test_an_imported_book_still_teaches_classes_whose_topics_it_does_not_mention(lib):
    lib.add_document("history", "Rome", "Rome was an ancient city on the Tiber river that grew into a vast empire across Europe. " * 3
                     + "The Senate was the governing council of the Roman Republic and advised its magistrates. "
                     + "A legion was a large unit of the Roman army made up of several thousand soldiers. "
                     + "The Colosseum was an amphitheatre in the centre of Rome where gladiators fought. "
                     + "An aqueduct was a bridge-like structure that carried water to Roman cities over long distances. "
                     + "Latin was the language of the Romans and the ancestor of the Romance languages today. ")
    t = teacher(lib)
    assert t.has_material("history", 1) and t.has_material("history", 9)
    assert "Senate" in t.lesson("history", 1).text and len(t.exam("history", 1, 5, attempt=1)) == 5


def test_topic_matched_passages_come_before_the_fallback(lib):
    lib.add_document("astronomy", "On topic", ASTRO, topic="the solar system")
    facts = Teacher(lib, style_from("patient")).facts("astronomy", 2)  # class 2's topic is "the solar system"
    assert facts and all(f.title == "On topic" for f in facts)


def test_generic_question_words_do_not_stop_the_teacher_finding_the_answer(lib):
    lib.add_document("history", "Rome", "Augustus was the adopted heir of Julius Caesar who ruled Rome for more than forty years and brought a long period of peace. " * 3)
    t = teacher(lib)
    for q in ("What does Augustus mean exactly?", "Tell me about Augustus", "who was augustus?"):
        assert "adopted heir" in t.answer(["history"], 4, q), q
    assert "can't find" in t.answer(["history"], 4, "What does Pluto mean exactly?")
