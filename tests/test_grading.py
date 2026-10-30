from ai_room.grading import cloze_match, exact_match, extract_code, extract_final, mcq_match, numeric_match, parse_number, run_code_tests


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


def test_integer_keys_are_exact_but_rounded_keys_accept_rounding():
    assert numeric_match("7.0", "7") and not numeric_match("7.05", "7")
    assert numeric_match("200928", "200928") and not numeric_match("201000", "200928")
    assert numeric_match("201000", "200928", exact=False) and numeric_match("2.01e5", "200928", exact=False)
    assert numeric_match("9.00×10^10", "90000000000", exact=False)


def test_cloze_matching_is_lenient_about_form_but_not_about_content():
    assert cloze_match("The Photosynthesis.", "photosynthesis")
    assert cloze_match("cell membranes", "cell membrane")
    assert not cloze_match("mitosis", "meiosis")
    assert cloze_match("dreissig", "dreißig") and cloze_match("soeur", "sœur") and cloze_match("sabado", "sábado")
    assert cloze_match("水", "水") and not cloze_match("火", "水")
    assert cloze_match("पानी", "पानी") and not cloze_match("पाना", "पानी")
    assert cloze_match("1969", "1969") and not cloze_match("1970", "1969")


def test_exact_match_keeps_symbols():
    assert exact_match("3 1 * 9 + 6 +", "3 1 * 9 + 6 +")
    assert not exact_match("3 1 9 6", "3 1 * 9 + 6 +")
    assert exact_match("192.168.5.0", "192.168.5.0") and not exact_match("192.168.5.1", "192.168.5.0")


def test_code_tests():
    code = "def add(a, b):\n    return a + b"
    assert run_code_tests(code, ["assert add(1, 2) == 3", "assert add(0, 0) == 1"])[:2] == (1, 2)
    assert run_code_tests("while True: pass", ["assert True"], timeout=1)[0] == 0
    assert extract_code("Here:\n```python\nx = 1\n```") == "x = 1\n"
