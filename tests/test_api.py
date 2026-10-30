import base64
import json
import time

import pytest
from fastapi.testclient import TestClient

from ai_room.server import create_app

TEXT = ("Stoicism is a school of philosophy founded in Athens by Zeno of Citium. Virtue is the highest good according to the Stoics and their teachers. "
        "Epictetus was a former slave who became an influential Stoic teacher in Rome. ") * 3


def fake_fetch(lib, subject, progress=None, stop=None, **kw):
    if progress:
        progress("fake", 1, 1)
    lib.add_document(subject, "Stoicism", TEXT, source="Fake", topic="x")
    return {"added": 1, "skipped": 0, "errors": []}


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(str(tmp_path / "db.sqlite"), fetcher=fake_fetch))


def wait_idle(c, cid, timeout=40):
    for _ in range(int(timeout / .2)):
        time.sleep(.2)
        if not c.get(f"/api/classrooms/{cid}").json()["busy"]:
            return
    raise AssertionError("class still running")


def new_room(c, **cfg):
    body = {"student": {"backend": "demo", "model": "x"}, "subjects": ["math"], "questions_per_exam": 4, **cfg}
    return c.post("/api/classrooms", json={"name": "t", "config": body}).json()["id"]


def test_meta_describes_100_subjects_and_6_teachers(client):
    m = client.get("/api/meta").json()
    assert len(m["subjects"]) == 100 and {s["mode"] for s in m["subjects"]} == {"exact", "mixed", "textbook"}
    assert [p["key"] for p in m["personas"]] == ["patient", "socratic", "strict", "funny", "researcher", "drill"]
    assert all(p["avatar"] and p["tone"] and p["approach"] for p in m["personas"]) and m["max_params_b"] == 27


def test_students_over_27b_and_ai_teachers_are_rejected(client):
    r = client.post("/api/classrooms", json={"config": {"student": {"backend": "openai", "model": "llama3.1-70b"}}})
    assert r.status_code == 400 and "27" in r.json()["detail"]
    r = client.post("/api/classrooms", json={"config": {"student": {"backend": "anthropic", "model": "claude"}}})
    assert r.status_code == 400 and "not an AI" in r.json()["detail"]


def test_run_a_class_over_http_and_change_the_teacher(client):
    cid = new_room(client, persona="funny")
    info = client.get(f"/api/classrooms/{cid}").json()
    assert info["teacher"]["name"] == "Mr. Giggles" and info["student_size_b"] == 1.0
    client.post(f"/api/classrooms/{cid}/run", json={"action": "class"})
    wait_idle(client, cid)
    assert client.get(f"/api/classrooms/{cid}").json()["grade"] == 2
    msgs = client.get(f"/api/classrooms/{cid}/messages").json()
    assert any(m["meta"].get("kind") == "lesson" and m["meta"].get("avatar") == "🎈" for m in msgs)
    r = client.patch(f"/api/classrooms/{cid}/config", json={"persona": "custom", "style": {"name": "Ms. Q", "tone": "strict", "practice": 6}})
    assert r.status_code == 200 and r.json()["teacher"]["name"] == "Ms. Q" and r.json()["teacher"]["practice"] == 6
    assert client.patch(f"/api/classrooms/{cid}/config", json={"subjects": ["nope"]}).status_code == 400
    assert client.post(f"/api/classrooms/{cid}/talk", json={"text": "What is Newton's second law?", "to": "teacher"}).status_code == 200


def test_library_import_search_bank_and_delete(client):
    r = client.post("/api/library/import", json={"subject": "history", "filename": "rome.md", "data_b64": base64.b64encode(("# Rome\n\n" + TEXT).encode()).decode()})
    assert r.status_code == 200
    assert client.post("/api/library/import", json={"subject": "history", "filename": "bad.exe", "text": "x"}).status_code == 400
    hits = client.get("/api/library/search", params={"subject": "history", "q": "who was Epictetus"}).json()
    assert hits and "Epictetus" in hits[0]["text"]
    bank = [{"type": "mcq", "question": "Q?", "options": ["x", "y"], "answer": "A"}]
    assert client.post("/api/library/bank", json={"subject": "history", "text": json.dumps(bank)}).json()["added"] == 1
    assert client.post("/api/library/bank", json={"subject": "history", "text": "[]"}).status_code == 400
    lib = client.get("/api/library").json()
    assert lib["subjects"]["history"]["docs"] == 1 and lib["subjects"]["history"]["questions"] == 1 and lib["totals"]["docs"] == 1
    doc = client.get("/api/library/documents", params={"subject": "history"}).json()[0]
    assert client.delete(f"/api/library/documents/{doc['id']}").status_code == 200
    assert client.get("/api/library").json()["subjects"]["history"]["docs"] == 0


def test_fetch_job_reports_progress_and_only_one_runs_at_a_time(client):
    job = client.post("/api/library/fetch", json={"subjects": ["ethics"]}).json()["id"]
    for _ in range(50):
        status = client.get(f"/api/library/jobs/{job}").json()
        if status["status"] != "running":
            break
        time.sleep(.1)
    assert status["status"] == "done" and status["added"] == 1
    listed = client.get("/api/library")  # the page the UI loads right after starting a fetch
    assert listed.status_code == 200 and listed.json()["jobs"][-1]["id"] == job and "_stop" not in listed.json()["jobs"][-1]
    assert client.post("/api/library/fetch", json={"subjects": ["nope"]}).status_code == 400


def test_class_without_textbooks_fetches_them_automatically(client):
    cid = new_room(client, subjects=["philosophy"], questions_per_exam=3)
    client.post(f"/api/classrooms/{cid}/run", json={"action": "lessons"})
    wait_idle(client, cid)
    kinds = [m["meta"].get("kind") for m in client.get(f"/api/classrooms/{cid}/messages").json()]
    assert "library" in kinds and "lesson" in kinds


def test_teacher_preview_and_safety(client):
    r = client.post("/api/teacher/preview", json={"persona": "socratic", "subject": "physics", "grade": 4}).json()
    assert r["available"] and "try this on your own" in r["text"] and r["name"] == "Professor Socrates"
    assert not client.post("/api/teacher/preview", json={"subject": "history", "grade": 2}).json()["available"]
    assert client.post("/api/library/import_url", json={"subject": "history", "url": "http://127.0.0.1:9/x"}).status_code == 400
    assert client.post("/api/library/import_url", json={"subject": "history", "url": "file:///etc/passwd"}).status_code == 400
    assert client.get("/api/ollama/models", params={"base_url": "http://127.0.0.1:9"}).json()["ok"] is False


def test_the_web_app_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "AI Room" in page.text
    for asset in ("core.js", "components.js", "app.js", "style.css"):
        assert f"/static/{asset}" in page.text and client.get(f"/static/{asset}").status_code == 200
