import io
import json
import zipfile

import httpx
import pytest

from ai_room.teacher.fetch import FetchError, Wikimedia, fetch_subject, page_url
from ai_room.teacher.knowledge import facts_from_passage, facts_from_passages, make_question, split_sentences
from ai_room.teacher.library import Library, Passage, chunk, extract_text, parse_bank_text

BIO = """Photosynthesis is a biological process used by plants to convert light energy into chemical energy. The process takes place in the chloroplasts and uses chlorophyll to absorb sunlight.

== Light reactions ==
The light reactions occur in the thylakoid membranes. Water is split and oxygen is released as a by-product. ATP and NADPH are produced for the next stage of the process.

== Calvin cycle ==
The Calvin cycle uses ATP and NADPH to fix carbon dioxide into sugars. It takes place in the stroma of the chloroplast and does not require light directly.

== References ==
Smith 2001. Some book that must be dropped."""


def test_chunking_keeps_headings_and_drops_reference_sections():
    parts = chunk(BIO)
    assert [h for h, _ in parts] == ["", "Light reactions", "Calvin cycle"]
    assert all("Smith 2001" not in p for _, p in parts)


def test_documents_are_deduplicated_and_searchable(lib):
    assert lib.add_document("biology", "Photosynthesis", BIO, topic="photosynthesis")
    assert lib.add_document("biology", "Photosynthesis", BIO, topic="photosynthesis") is None
    hits = lib.search("biology", "where does the Calvin cycle take place", k=2)
    assert hits[0].heading == "Calvin cycle" and "stroma" in hits[0].text
    assert lib.search("biology", "volcano", k=2) == []
    assert lib.stats()["biology"]["passages"] == 3 and lib.has_text("biology", 3)


def test_topic_and_level_steer_ranking(lib):
    lib.add_document("astronomy", "Kids", "Stars are huge balls of hot gas that shine in the sky at night because of nuclear reactions in their cores. " * 4, level="simple", topic="stars")
    lib.add_document("astronomy", "Uni", "Stars are luminous spheres of plasma held together by their own gravity and powered by nuclear fusion in their cores. " * 4, level="standard", topic="stars")
    assert lib.search("astronomy", "stars", k=1, grade=2)[0].title == "Kids"
    assert lib.search("astronomy", "stars", k=1, grade=11)[0].title == "Uni"


def test_import_formats():
    html = b"<html><head><title>Cells</title></head><body><nav>menu</nav><h1>Cells</h1><p>A cell is the basic unit of life and all organisms are made of cells. Cells divide to make new cells.</p><script>x=1</script></body></html>"
    text, title = extract_text("cells.html", html)
    assert title == "Cells" and "basic unit of life" in text and "menu" not in text and "x=1" not in text
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("OEBPS/ch1.xhtml", "<html><body><h1>One</h1><p>An epub chapter that explains how gravity pulls objects towards the centre of the Earth every day.</p></body></html>")
    assert "epub chapter" in extract_text("book.epub", buf.getvalue())[0]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", "<w:document><w:p><w:t>Docx paragraph about electrons orbiting a nucleus in an atom.</w:t></w:p></w:document>")
    assert "Docx paragraph" in extract_text("a.docx", buf.getvalue())[0]
    with pytest.raises(ValueError):
        extract_text("x.exe", b"")


def test_import_path_reads_folders(tmp_path, lib):
    (tmp_path / "a.txt").write_text("Gravity is the force by which a planet draws objects toward its centre. " * 6)
    (tmp_path / "skip.bin").write_bytes(b"\x00")
    out = lib.import_path("physics", str(tmp_path))
    assert len(out) == 1 and out[0]["chars"] > 100 and lib.has_text("physics", 1)


def test_question_bank_formats(lib):
    rows = [
        {"subject": "biology", "type": "numeric", "question": "2+2?", "answer": "4"},
        {"subject": "biology", "question": "Capital?", "choices": ["Rome", "Paris"], "answer": 1},                                  # MMLU
        {"subject": "biology", "question": {"stem": "Plants make food by?", "choices": [{"label": "A", "text": "eating"}, {"label": "B", "text": "photosynthesis"}]}, "answerKey": "B"},  # ARC
        {"subject": "biology", "question": "Cells?", "correct_answer": "unit of life", "distractor1": "a rock", "distractor2": "a star", "distractor3": "a song"},  # SciQ
        {"subject": "math", "question": "Ann has 3, gets 4. How many?", "answer": "3+4=7\n#### 7"},                                  # GSM8K
        {"subject": "code", "text": "Add numbers", "code": "def add(a,b):\n  return a+b", "test_list": ["assert add(1,2)==3"]},   # MBPP
    ]
    items = parse_bank_text("\n".join(json.dumps(r) for r in rows))
    assert [s for s, _ in items] == ["biology"] * 4 + ["math", "code"]
    kinds = {i["type"] for _, i in items}
    assert kinds == {"numeric", "mcq", "code"}
    for s, it in items:
        lib.add_questions(s, [it])
    sci = next(i for s, i in items if i["question"] == "Cells?")
    assert sci["options"][ord(sci["answer"]) - 65].endswith("unit of life")
    assert next(i for s, i in items if s == "math")["answer"] == "7"
    assert lib.questions("math")[0]["concept"].startswith("bank:")


# ---- Wikimedia fetcher against mocked responses -------------------------------------------------

def wiki_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": "test"})


ARTICLE = "Astronomy is the scientific study of celestial objects, space and the universe as a whole. " * 10


def ok_handler(calls):
    def h(request: httpx.Request):
        calls.append(str(request.url))
        q = dict(request.url.params)
        if q.get("list") == "search":
            return httpx.Response(200, json={"query": {"search": [{"title": "Star"}, {"title": "Planet"}, {"title": "Moon"}]}})
        if q.get("prop") == "extracts":
            body = "Intro. " + ARTICLE + f"\n\n== About {q['titles']} ==\n" + ARTICLE + "\n\n== References ==\nbad"
            if q["titles"] == "Moon":
                return httpx.Response(200, json={"query": {"pages": [{"title": "Moon", "missing": True}]}})
            return httpx.Response(200, json={"query": {"pages": [{"title": q["titles"], "extract": body}]}})
        return httpx.Response(400)
    return h


def test_fetch_subject_adds_documents_with_levels_sources_and_urls(lib):
    calls = []
    wiki = Wikimedia(client=wiki_client(ok_handler(calls)), min_interval=0, sleep=lambda s: None)
    progress = []
    summary = fetch_subject(lib, "astronomy", wiki=wiki, per_site=1, sources=("simple", "wikipedia"), progress=lambda m, i, t: progress.append((i, t)))
    assert summary["added"] > 0 and not summary["errors"]
    docs = lib.documents("astronomy")
    assert {d["level"] for d in docs} == {"simple", "standard"} and all(d["url"].startswith("https://") and d["license"] == "CC BY-SA 4.0" for d in docs)
    assert any("simple.wikipedia.org" in c for c in calls) and progress[-1][0] == progress[-1][1]
    again = fetch_subject(lib, "astronomy", wiki=wiki, per_site=1, sources=("simple",))
    assert again["added"] == 0 and again["skipped"] > 0  # nothing is downloaded twice


def test_fetcher_tolerates_both_mediawiki_response_shapes_and_skips_stubs():
    def h(request):
        q = dict(request.url.params)
        if q["titles"] == "Old":  # formatversion 1 style: pages is a dict keyed by id
            return httpx.Response(200, json={"query": {"pages": {"123": {"title": "Old", "extract": ARTICLE}}}})
        if q["titles"] == "Stub":
            return httpx.Response(200, json={"query": {"pages": [{"title": "Stub", "extract": "Too short."}]}})
        return httpx.Response(200, json={"query": {"pages": [{"title": "Dis", "extract": "Mercury may refer to: " + "x" * 700}]}})
    w = Wikimedia(client=wiki_client(h), min_interval=0, sleep=lambda s: None)
    assert w.extract("en.wikipedia.org", "Old")
    assert w.extract("en.wikipedia.org", "Stub") is None and w.extract("en.wikipedia.org", "Dis") is None


def test_rate_limits_are_retried_and_blocks_are_reported():
    waits = []
    state = {"n": 0}

    def limited(request):
        state["n"] += 1
        if state["n"] < 3:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, json={"query": {"search": [{"title": "A"}]}})

    w = Wikimedia(client=wiki_client(limited), min_interval=0, sleep=waits.append)
    assert w.search("en.wikipedia.org", "x") == ["A"] and 7.0 in waits

    blocked = Wikimedia(client=wiki_client(lambda r: httpx.Response(403, text="robot policy")), min_interval=0, sleep=lambda s: None)
    with pytest.raises(FetchError, match="refused"):
        blocked.search("en.wikipedia.org", "x")


def test_fetch_survives_errors_and_stops_when_asked(lib):
    w = Wikimedia(client=wiki_client(lambda r: httpx.Response(403)), min_interval=0, sleep=lambda s: None)
    s = fetch_subject(lib, "astronomy", wiki=w, sources=("wikipedia",))
    assert s["added"] == 0 and 1 <= len(s["errors"]) <= 6
    import threading
    stop = threading.Event()
    stop.set()
    assert fetch_subject(lib, "astronomy", wiki=w, stop=stop).get("stopped")


def test_page_url():
    assert page_url("en.wikipedia.org", "Black hole") == "https://en.wikipedia.org/wiki/Black_hole"


# ---- fact extraction ----------------------------------------------------------------------------

def P(text, id=1):
    return Passage(id, 1, "biology", "", "standard", "Heading", text, "Title", "http://u", "src")


def test_definitions_and_years_become_questions():
    p = P("A cell is the basic structural and functional unit of all living organisms. Mitochondria are organelles that release energy from food for the cell to use. "
          "In 1665, Robert Hooke looked at cork through a microscope and named the cells he saw. It was a long time ago. This is not a definition of anything useful. "
          "Chlorophyll is the green pigment that absorbs sunlight in plants and algae. The nucleus is the part of a cell that contains its genetic material and controls growth.")
    facts = facts_from_passage(p)
    answers = {f.answer for f in facts}
    assert {"cell", "Mitochondria", "1665", "Chlorophyll", "nucleus"} <= answers
    assert "It" not in answers and "This" not in answers
    import random
    rng = random.Random(1)
    cloze = make_question(next(f for f in facts if f.answer == "cell"), facts, rng, prefer_mcq=False)
    assert cloze["type"] == "cloze" and "_____" in cloze["question"] and "cell" not in cloze["question"].split("\n")[1].lower().split("_____")[0] and cloze["answer"] == "cell"
    mcq = make_question(next(f for f in facts if f.answer == "nucleus"), facts, rng, prefer_mcq=True)
    assert mcq["type"] == "mcq" and len(mcq["options"]) == 4
    assert mcq["options"][ord(mcq["answer"]) - 65][3:].lower() == "nucleus"
    year = make_question(next(f for f in facts if f.kind == "year"), facts, rng, prefer_mcq=True)
    assert year["type"] == "mcq" and year["options"][ord(year["answer"]) - 65][3:] == "1665"


def test_sentence_splitting_respects_abbreviations():
    assert len(split_sentences("Dr. Smith met Mr. Jones in the U.S. last year. They talked about e.g. gravity. Then they left.")) == 3


def test_facts_are_deduplicated_across_passages():
    ps = [P("A cell is the basic structural unit of all living organisms on Earth.", 1), P("A cell is the basic structural unit of all living organisms on Earth.", 2)]
    assert len(facts_from_passages(ps)) == 1
