"""The toolbox: registry rules, exact answers, safety limits, 3D / Blender output, recipes, the Workshop loop and its API."""

import json
import math
import shutil
import sys
import time
import xml.dom.minidom
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from ai_room.llm import LLM, MockLLM
from ai_room.server import create_app
from ai_room.tools import REGISTRY, call, categories, make_context, missing, search
from ai_room.tools import recipes as rec
from ai_room.tools import t_blender as B
from ai_room.tools import t_mesh as M
from ai_room.tools.agent import RulePlanner, Workshop, parse_action, run_to_recipe
from ai_room.tools.registry import DEFAULT_ALLOWED

HAS_PIL = shutil.which("python") and __import__("importlib.util", fromlist=["x"]).find_spec("PIL") is not None
HAS_FFMPEG = shutil.which("ffmpeg") is not None
needs_pil = pytest.mark.skipif(not HAS_PIL, reason="Pillow not installed")
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")


@pytest.fixture
def ctx(tmp_path):
    return make_context(tmp_path / "ws")


def ok(r):
    assert r["ok"], r.get("error")
    return r["result"]


def val(r):
    return json.loads(ok(r))


# ---------------------------------------------------------------- the registry itself
def test_toolbox_is_large_and_honest_about_what_needs_setup(ctx):
    assert len(REGISTRY) >= 1000
    ready = [n for n, t in REGISTRY.items() if not missing(t, ctx)]
    assert len(ready) >= 1000                        # in this environment, with Pillow/ffmpeg present
    for t in REGISTRY.values():                      # every tool explains itself
        assert t.desc and t.category and t.name == t.name.lower()
    must_exist = ["calculator", "convert_units", "wikipedia_search", "youtube_search_videos", "instagram_post_image", "blender_build_house", "blender_render", "colab_notebook_classifier",
                  "kaggle_notebook_classifier", "run_python", "video_to_gif", "image_text_card_instagram_post", "train_logistic", "recipe_run"]
    assert [n for n in must_exist if n not in REGISTRY] == []
    assert categories()["blender"] >= 150


def test_search_finds_the_right_tool_for_plain_requests(ctx):
    assert search("convert kilometres to miles", 3, ctx)[0].name.startswith("convert_")
    assert any(t.name == "loan_payment" for t in search("monthly mortgage payment", 5, ctx))
    assert any(t.name.startswith("video_to_gif") for t in search("turn a video into a gif", 8, ctx))
    assert any("blender" in t.name for t in search("3d model material gold", 8, ctx))
    assert search("zzzzqqq nonsense", 5, ctx) == []


def test_call_never_raises_and_explains_mistakes(ctx):
    assert call("no_such_tool", {}, ctx)["ok"] is False
    r = call("calculator", {}, ctx)
    assert not r["ok"] and "expression" in r["error"]
    r = call("calculator", {"expression": "1+1", "bogus": 1}, ctx)
    assert not r["ok"] and "bogus" in r["error"]
    r = call("factorial", {"n": "banana"}, ctx)
    assert not r["ok"]
    assert call("calculator", {"expression": "1/0"}, ctx)["ok"] is False        # a math error is an answer, not a crash


def test_argument_coercion_is_forgiving(ctx):
    assert ok(call("power", {"base": "2", "exponent": "10"}, ctx)) == "1024"
    assert val(call("stats_summary", {"numbers": "1, 2, 3, 4"}, ctx))["mean"] == 2.5
    assert val(call("stats_summary", {"numbers": [1, 2, 3, 4]}, ctx))["median"] == 2.5


def test_permissions_gate_dangerous_tools(ctx):
    assert DEFAULT_ALLOWED == ("safe", "files", "network")
    r = call("run_python", {"code": "print(1)"}, ctx)
    assert not r["ok"] and r["blocked"] == "exec"
    assert call("instagram_post_image", {"image_url": "https://x/y.jpg", "caption": "hi"}, ctx)["blocked"] == "post"
    assert call("youtube_upload_video", {"path": "a.mp4", "title": "t"}, ctx).get("blocked") == "post"
    allowed = make_context(ctx.workspace, allowed=DEFAULT_ALLOWED + ("exec",))
    assert json.loads(ok(call("run_python", {"code": "print(6*7)"}, allowed)))["stdout"].strip() == "42"


def test_tools_that_need_setup_say_exactly_what_is_missing(ctx):
    r = call("instagram_account_info", {}, ctx)
    assert not r["ok"] and "INSTAGRAM_ACCESS_TOKEN" in r["error"]
    r = call("blender_version", {}, make_context(ctx.workspace, allowed=DEFAULT_ALLOWED + ("exec",)))
    assert shutil.which("blender") or "blender" in r["error"]


def test_the_workspace_cannot_be_escaped(ctx):
    for bad in ("../x.txt", "/etc/passwd", "a/../../x", ""):
        r = call("write_file", {"path": bad, "content": "no"}, ctx)
        assert not r["ok"], bad
    assert call("read_file", {"path": "../../etc/passwd"}, ctx)["ok"] is False
    assert ok(call("write_file", {"path": "ok/inside.txt", "content": "hi"}, ctx))
    assert ok(call("read_file", {"path": "ok/inside.txt"}, ctx)) == "hi"


def test_zip_slip_is_refused(ctx):
    import zipfile
    z = ctx.workspace / "evil.zip"
    z.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("../escape.txt", "x")
    assert not call("unzip_file", {"archive": "evil.zip"}, ctx)["ok"]
    assert not (ctx.workspace.parent / "escape.txt").exists()


def test_slow_tools_are_stopped(tmp_path):
    c = make_context(tmp_path, allowed=DEFAULT_ALLOWED + ("exec",), timeout=1.0)
    t0 = time.time()
    r = call("run_python", {"code": "import time; time.sleep(30)"}, c)
    assert not r["ok"] and time.time() - t0 < 6


# ---------------------------------------------------------------- exact answers
def test_calculator_is_exact_and_safe(ctx):
    assert ok(call("calculator", {"expression": "(3+4)*2^3/7"}, ctx)) == "8"
    assert ok(call("calculator", {"expression": "1,000 + max(2,3)"}, ctx)) == "1003"
    assert float(ok(call("calculator", {"expression": "sqrt(2)*sin(pi/4)"}, ctx))) == pytest.approx(1.0)
    for evil in ("__import__('os').system('id')", "open('x')", "().__class__", "9**9**9", "[x for x in range(9)]"):
        assert not call("calculator", {"expression": evil}, ctx)["ok"], evil


def test_math_and_science_values(ctx):
    assert val(call("solve_quadratic", {"a": 1, "b": -3, "c": 2}, ctx)) == [2, 1]
    assert float(ok(call("convert_units", {"value": 5, "from_unit": "km", "to_unit": "mi"}, ctx))) == pytest.approx(3.10685596, rel=1e-6)
    assert float(ok(call("convert_units", {"value": 100, "from_unit": "C", "to_unit": "F"}, ctx))) == 212
    assert float(ok(call("molar_mass", {"formula": "Ca(OH)2"}, ctx))) == pytest.approx(74.092, abs=0.01)
    assert float(ok(call("loan_payment", {"principal": 200000, "annual_rate_percent": 6, "years": 30}, ctx))) == pytest.approx(1199.10, abs=0.01)
    assert val(call("polynomial_roots", {"coefficients": [1, -6, 11, -6]}, ctx)) == [1, 2, 3]
    assert val(call("matrix_inverse", {"A": [[1, 2], [3, 4]]}, ctx)) == [[-2, 1], [1.5, -0.5]]
    assert ok(call("to_roman", {"n": 1994}, ctx)) == "MCMXCIV"
    assert ok(call("is_prime", {"n": 1_000_000_007}, ctx)) == "true" or ok(call("is_prime", {"n": 1_000_000_007}, ctx)) is True
    assert ok(call("easter_date", {"year": 2025}, ctx)) == "2025-04-20"
    assert float(ok(call("haversine_distance", {"lat1": 51.5, "lon1": -0.12, "lat2": 48.85, "lon2": 2.35}, ctx))) == pytest.approx(343.5, abs=2)


def test_text_and_crypto_tools(ctx):
    assert ok(call("hash_sha256", {"text": "abc"}, ctx)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert ok(call("from_morse", {"morse": ok(call("to_morse", {"text": "sos"}, ctx))}, ctx)) == "SOS"
    assert ok(call("base64_decode", {"data": ok(call("base64_encode", {"text": "héllo"}, ctx))}, ctx)) == "héllo"
    assert ok(call("edit_distance", {"a": "kitten", "b": "sitting"}, ctx)) == "3"
    assert ok(call("slugify", {"text": "Héllo World!"}, ctx)) == "hello-world"


def test_csv_tools(ctx):
    data = "name,score\nann,3\nbob,5\nann,4\n"
    assert val(call("csv_group_sum", {"text": data, "group": "name", "value": "score"}, ctx)) == {"ann": 7.0, "bob": 5.0}
    assert val(call("csv_summary", {"text": data}, ctx))["columns"]["score"]["mean"] == 4.0


# ---------------------------------------------------------------- 3D and Blender
def test_mesh_geometry_is_correct():
    cube = M.cube(2)
    s = M.stats(cube)
    assert s["volume"] == 8 and s["surface_area"] == 24 and s["watertight"]
    sph = M.stats(M.uv_sphere(1, 48, 24))
    assert sph["volume"] == pytest.approx(4 / 3 * math.pi, rel=0.01) and sph["watertight"]
    assert M.stats(M.torus(1, .3, 48, 24))["volume"] == pytest.approx(2 * math.pi ** 2 * 1 * .3 ** 2, rel=0.02)
    assert M.stats(M.dodecahedron(1))["watertight"] and len(M.dodecahedron(1).f) == 12


def test_every_mesh_primitive_exports_and_round_trips(ctx):
    for name in M.PRIMS:
        r = call(f"mesh_{name}", {"format": "obj"}, ctx)
        assert r["ok"], (name, r.get("error"))
    s0 = val(call("mesh_stats", {"path": "models/gear.obj"}, ctx))
    ok(call("mesh_convert", {"path": "models/gear.obj", "format": "stl"}, ctx))
    s1 = val(call("mesh_stats", {"path": "models/gear.stl"}, ctx))
    assert s1["triangles"] == s0["triangles"] and s1["volume"] == pytest.approx(s0["volume"], rel=1e-3)
    ok(call("mesh_convert", {"path": "models/torus.obj", "format": "gltf"}, ctx))
    g = json.loads((ctx.workspace / "models/torus.gltf").read_text())
    assert g["asset"]["version"] == "2.0" and g["accessors"][0]["type"] == "VEC3"


def test_svg_renderer_output_is_well_formed(ctx):
    ok(call("mesh_dodecahedron", {}, ctx))
    ok(call("mesh_render_svg", {"paths": ["models/dodecahedron.obj"]}, ctx))
    xml.dom.minidom.parse(str(ctx.workspace / "renders/view.svg"))


def test_every_scene_generator_and_add_tool_works(ctx):
    for g in B.GENERATORS:
        ok(call("blender_new_scene", {"scene": g}, ctx))
        r = call(f"blender_build_{g}", {"scene": g}, ctx)
        assert r["ok"], (g, r.get("error"))
    ok(call("blender_new_scene", {"scene": "all"}, ctx))
    for k in list(B.BPY_PRIMS) + list(B.CUSTOM):
        assert call(f"blender_add_{k}", {"scene": "all", "name": k}, ctx)["ok"], k
    assert len(json.loads((ctx.workspace / "scenes/all.json").read_text())["objects"]) >= 30


def test_blender_script_compiles_and_runs_against_a_fake_bpy(ctx, monkeypatch):
    ok(call("blender_new_scene", {"scene": "s"}, ctx))
    ok(call("blender_build_robot", {"scene": "s"}, ctx))
    ok(call("blender_material_gold", {"scene": "s", "object": "Body"}, ctx))
    ok(call("blender_modifier_subsurf", {"scene": "s", "object": "Body", "levels": 3}, ctx))
    ok(call("blender_modifier_array", {"scene": "s", "object": "Leg", "count": 3}, ctx))
    ok(call("blender_anim_spin_object", {"scene": "s", "object": "Head"}, ctx))
    ok(call("blender_anim_orbit_camera", {"scene": "s"}, ctx))
    ok(call("blender_render_as_video", {"scene": "s"}, ctx))
    ok(call("blender_export_glb", {"scene": "s"}, ctx))
    ok(call("blender_build_script", {"scene": "s"}, ctx))
    code = (ctx.workspace / "scenes/s.py").read_text()
    assert "modifiers.new" in code and "FFMPEG" in code and "export_scene.gltf" in code and "keyframe_insert" in code
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "bpy", fake)
    monkeypatch.setitem(sys.modules, "mathutils", MagicMock())
    exec(compile(code, "s.py", "exec"), {"__name__": "scene"})
    assert fake.ops.render.render.called                      # reached the render step without a Python error


def test_blender_scene_edits_are_validated(ctx):
    ok(call("blender_new_scene", {"scene": "v"}, ctx))
    assert not call("blender_add_cube", {"scene": "missing"}, ctx)["ok"]
    assert not call("blender_add_cube", {"scene": "v", "material": "no_such_material"}, ctx)["ok"]
    assert not call("blender_material_gold", {"scene": "v", "object": "ghost"}, ctx)["ok"]
    ok(call("blender_add_cube", {"scene": "v", "name": "box", "x": 3, "material": "gold"}, ctx))
    ok(call("blender_set_object", {"scene": "v", "object": "box", "z": 2, "rz": 90}, ctx))
    sc = json.loads((ctx.workspace / "scenes/v.json").read_text())
    box = next(o for o in sc["objects"] if o["name"] == "box")
    assert box["loc"] == [3, 0, 2] and box["rot"][2] == pytest.approx(math.pi / 2)


@needs_pil
def test_scene_preview_shows_the_objects_not_the_ground(ctx):
    ok(call("blender_new_scene", {"scene": "p"}, ctx))
    ok(call("blender_build_house", {"scene": "p"}, ctx))
    ok(call("blender_preview_png", {"scene": "p", "out": "p.png", "width": 400, "height": 300}, ctx))
    from PIL import Image
    im = Image.open(ctx.workspace / "p.png").convert("RGB")
    import numpy as np
    a = np.asarray(im).astype(int)
    brick = int(((a[..., 0] > 50) & (a[..., 0] > a[..., 1] * 1.5) & (a[..., 0] > a[..., 2] * 1.7)).sum())   # the house's red-brown walls are drawn
    assert brick > 2500


# ---------------------------------------------------------------- images, video, charts, notebooks, ML
def test_charts_are_valid_svg(ctx):
    for n, a in (("bar_chart", {"values": [3, 5, 2], "labels": "a,b,c"}), ("pie_chart", {"values": [1, 2, 3]}), ("line_chart", {"values": {"x": [1, 2], "y": [2, 1]}}), ("plot_function", {"expressions": "sin(x); x^2/10"}),
                 ("heatmap", {"matrix": [[1, 2], [3, 4]]}), ("radar_chart", {"values": [3, 4, 5, 2]}), ("timeline", {"events": [[1900, "A"], [2000, "B"]]})):
        r = call(n, {"path": f"c/{n}", **a}, ctx)
        assert r["ok"], (n, r.get("error"))
        xml.dom.minidom.parse(str(ctx.workspace / f"c/{n}.svg"))


def test_notebooks_are_valid_nbformat(ctx):
    for plat in ("colab", "kaggle"):
        r = call(f"{plat}_notebook_classifier", {}, ctx)
        assert r["ok"]
        nb = json.loads(next((ctx.workspace / "notebooks").glob(f"{plat}_*")).read_text())
        assert nb["nbformat"] == 4 and all("source" in c and c["cell_type"] in ("code", "markdown") for c in nb["cells"])
    assert len([n for n in REGISTRY if n.startswith("colab_notebook_")]) >= 20


@needs_pil
def test_image_tools_make_real_pictures(ctx):
    from PIL import Image
    ok(call("image_text_card_instagram_post", {"text": "Hello", "sub": "world", "out": "i.png"}, ctx))
    assert Image.open(ctx.workspace / "i.png").size == (1080, 1080)
    ok(call("image_fit_youtube_thumbnail", {"path": "i.png", "out": "t.png"}, ctx))
    assert Image.open(ctx.workspace / "t.png").size == (1280, 720)
    ok(call("image_grayscale", {"path": "i.png", "out": "g.png"}, ctx))
    assert Image.open(ctx.workspace / "g.png").mode in ("L", "RGB")
    assert not call("image_resize", {"path": "nope.png", "width": 10}, ctx)["ok"]


@needs_ffmpeg
def test_video_tools_produce_playable_files(ctx):
    ok(call("video_generate_test_pattern", {"seconds": 1, "size": "160x90", "out": "v.mp4"}, ctx))
    info = val(call("video_info", {"path": "v.mp4"}, ctx))
    assert info["streams"][0]["width"] == 160 and 0.8 < info["duration_s"] < 1.4
    ok(call("video_fit_instagram_reel", {"path": "v.mp4", "out": "r.mp4"}, ctx))
    assert val(call("video_info", {"path": "r.mp4"}, ctx))["streams"][0]["height"] == 1920
    ok(call("audio_tone", {"frequency": 440, "seconds": 1, "out": "a.wav"}, ctx))
    ok(call("video_add_audio", {"video": "v.mp4", "audio": "a.wav", "out": "va.mp4"}, ctx))
    assert any(s["type"] == "audio" for s in val(call("video_info", {"path": "va.mp4"}, ctx))["streams"])
    assert not call("video_trim", {"path": "ghost.mp4"}, ctx)["ok"]


def test_training_tools_really_learn(ctx):
    import random
    r = random.Random(3)
    rows = ["a,b,label"] + [f"{r.gauss(c * 4, 1):.3f},{r.gauss(c * 4, 1):.3f},{c}" for c in (0, 1) for _ in range(60)]
    (ctx.workspace).mkdir(parents=True, exist_ok=True)
    (ctx.workspace / "d.csv").write_text("\n".join(rows))
    for kind in ("logistic", "tree", "knn", "perceptron", "mlp"):
        res = val(call(f"train_{kind}", {"csv": "d.csv", "target": "label", "name": kind}, ctx))
        assert res["score"]["accuracy"] >= 0.9, (kind, res)
    assert val(call("ml_predict", {"model": "logistic", "rows": [[4, 4], [0, 0]]}, ctx)) == ["1", "0"]
    lin = ["x,y"] + [f"{x},{3 * x + 2}" for x in range(20)]
    (ctx.workspace / "l.csv").write_text("\n".join(lin))
    assert val(call("train_linear", {"csv": "l.csv", "target": "y"}, ctx))["score"]["r2"] > 0.9999
    assert val(call("train_text_classifier", {"examples": [["great fun love", "pos"], ["awful bad hate", "neg"], ["love it", "pos"], ["hate it", "neg"]]}, ctx))["training_accuracy"] == 1.0
    assert val(call("classify_text", {"model": "text_classifier", "text": "love"}, ctx))["label"] == "pos"


def test_algorithms_give_known_answers(ctx):
    assert ok(call("sudoku_solve", {"puzzle": "530070000600195000098000060800060003400803001700020006060000280000419005000080079"}, ctx)).splitlines()[0] == "534678912"
    assert val(call("n_queens", {"n": 8}, ctx))["solutions"] == 92
    assert val(call("shortest_path", {"edges": [["a", "b", 1], ["b", "c", 2], ["a", "c", 5]], "source": "a", "target": "c"}, ctx))["distance"] == 3
    assert val(call("knapsack", {"items": [[2, 3], [3, 4], [4, 5], [5, 6]], "capacity": 5}, ctx))["best_value"] == 7
    assert val(call("linear_programming_2d", {"c1": 3, "c2": 2, "constraints": [[1, 1, 4], [1, 3, 6]]}, ctx))["objective"] == 12
    assert not call("topological_sort", {"edges": [["a", "b"], ["b", "a"]]}, ctx)["ok"]


# ---------------------------------------------------------------- the internet tools (offline: a fake server)
def fake_web(handler):
    c = make_context(".", secrets={"YOUTUBE_API_KEY": "KEY"})
    c.http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_wikimedia_tools_parse_the_api_and_cite(tmp_path):
    def h(req):
        u = str(req.url)
        assert "AI-Room" in req.headers["user-agent"]
        if "list=search" in u:
            return httpx.Response(200, json={"query": {"search": [{"title": "Alan Turing", "snippet": "A <b>mathematician</b>"}]}})
        return httpx.Response(200, json={"query": {"pages": [{"title": "Alan Turing", "extract": "Alan Turing was an English mathematician."}]}})
    c = fake_web(h)
    c.workspace = tmp_path
    r = val(call("wikipedia_search", {"query": "turing"}, c))
    assert r[0]["title"] == "Alan Turing" and "<b>" not in r[0]["snippet"] and r[0]["url"].endswith("Alan_Turing")
    page = val(call("wikipedia_page", {"title": "Alan Turing"}, c))
    assert "CC BY-SA" in page["license"] and page["url"].startswith("https://en.wikipedia.org/")


def test_web_errors_are_explained_not_raised(tmp_path):
    def h(req):
        return httpx.Response({"/limit": 429, "/block": 403}.get(req.url.path, 500), json={"error": "nope"})
    c = fake_web(h)
    assert "too many requests" in call("fetch_json", {"url": "https://example.com/limit"}, c)["error"]
    assert "refused" in call("fetch_json", {"url": "https://example.com/block"}, c)["error"]
    assert "https" in call("fetch_json", {"url": "http://example.com/x"}, c)["error"]


@pytest.mark.parametrize("url", ["https://localhost/x", "https://127.0.0.1/a", "https://192.168.1.5/", "https://10.0.0.1/", "https://printer.local/", "https://169.254.169.254/latest"])
def test_page_fetch_refuses_local_addresses(url, tmp_path):
    c = fake_web(lambda r: httpx.Response(200, text="secret"))
    assert not call("fetch_web_page_text", {"url": url}, c)["ok"]


def test_youtube_uses_the_official_api_with_the_key():
    seen = {}

    def h(req):
        seen["url"] = str(req.url)
        return httpx.Response(200, json={"items": [{"id": {"videoId": "abc12345678"}, "snippet": {"title": "T", "channelTitle": "C", "publishedAt": "2020-01-01T00:00:00Z"}}]})
    c = fake_web(h)
    r = val(call("youtube_search_videos", {"query": "python"}, c))
    assert r[0]["url"] == "https://youtu.be/abc12345678" and "key=KEY" in seen["url"] and "googleapis.com/youtube/v3/search" in seen["url"]


def test_instagram_publishing_needs_explicit_permission_and_keys(tmp_path):
    posted = []

    def h(req):
        posted.append((req.method, req.url.path))
        if req.url.path.endswith("/media"):
            return httpx.Response(200, json={"id": "c1"})
        if req.url.path.endswith("/media_publish"):
            return httpx.Response(200, json={"id": "p1"})
        return httpx.Response(200, json={"status_code": "FINISHED"})
    c = fake_web(h)
    c.secrets |= {"INSTAGRAM_ACCESS_TOKEN": "t", "INSTAGRAM_USER_ID": "9"}
    r = call("instagram_post_image", {"image_url": "https://x/y.jpg", "caption": "hi"}, c)
    assert r["blocked"] == "post" and posted == []                        # nothing was sent
    c.allowed = c.allowed + ("post",)
    assert val(call("instagram_post_image", {"image_url": "https://x/y.jpg", "caption": "hi"}, c))["published_id"] == "p1"


def test_caption_and_hashtag_helpers_are_rule_based(ctx):
    cap = ok(call("instagram_caption_template", {"topic": "AI classroom", "tone": "funny"}, ctx))
    assert "AI classroom" in cap and "#" in cap
    tags = val(call("instagram_hashtag_ideas", {"topic": "learning ai", "count": 5}, ctx))
    assert len(tags) == 5 and all(t.startswith("#") for t in tags)


# ---------------------------------------------------------------- recipes
def test_recipes_substitute_parameters_and_chain_steps(ctx):
    steps = [{"tool": "calculator", "args": {"expression": "{{a}}*2"}}, {"tool": "write_file", "args": {"path": "out/{{name}}.txt", "content": "result={{step1}}"}}]
    ok(call("recipe_save", {"name": "double", "description": "doubles", "steps": steps, "params": {"a": {"default": 4}, "name": {"default": "x"}}}, ctx))
    out = val(call("recipe_run", {"name": "double", "params": {"a": 21, "name": "answer"}}, ctx))
    assert out["ok"] and (ctx.workspace / "out/answer.txt").read_text() == "result=42"
    assert val(call("recipe_run_batch", {"name": "double", "param_sets": [{"a": 1, "name": "p"}, {"a": 2, "name": "q"}]}, ctx))[1]["ok"]
    assert (ctx.workspace / "out/q.txt").read_text() == "result=4"


def test_recipe_errors_stop_the_run_and_unknown_tools_are_rejected(ctx):
    assert not call("recipe_save", {"name": "bad", "description": "", "steps": [{"tool": "nope"}]}, ctx)["ok"]
    assert not call("recipe_save", {"name": "../bad", "description": "", "steps": [{"tool": "calculator"}]}, ctx)["ok"]
    ok(call("recipe_save", {"name": "fails", "description": "", "steps": [{"tool": "calculator", "args": {"expression": "1/0"}}, {"tool": "write_file", "args": {"path": "never.txt", "content": "x"}}]}, ctx))
    out = val(call("recipe_run", {"name": "fails"}, ctx))
    assert not out["ok"] and len(out["steps"]) == 1 and not (ctx.workspace / "never.txt").exists()
    assert not call("recipe_run", {"name": "ghost"}, ctx)["ok"]


def test_built_in_recipes_are_valid(ctx):
    for name in rec.BUILTIN:
        assert val(call("recipe_validate", {"name": name}, ctx))["issues"] == [], name


@needs_pil
def test_instagram_recipe_makes_post_story_and_caption(ctx):
    out = val(call("recipe_run", {"name": "instagram_post_pack", "params": {"topic": "Robots"}}, ctx))
    assert out["ok"] and {"instagram/post.png", "instagram/story.png", "instagram/caption.txt"} <= set(out["files"])
    assert "Robots" in (ctx.workspace / "instagram/caption.txt").read_text()


@needs_pil
def test_blender_recipe_builds_previews_and_script(ctx):
    out = val(call("recipe_run", {"name": "blender_showcase", "params": {"thing": "snowman"}}, ctx))
    assert out["ok"] and "scenes/showcase.py" in out["files"] and "renders/showcase_preview.png" in out["files"]


# ---------------------------------------------------------------- the Workshop loop
class Scripted(LLM):
    """A student that replies from a list; records what it was shown."""
    name = "scripted-7b"

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages, temperature=0.7, max_tokens=2048):
        self.seen.append(messages)
        return self.replies.pop(0) if self.replies else '{"final": "out of script"}'

    def parameter_count_b(self):
        return 7.0


def test_parse_action_accepts_what_small_models_actually_write():
    assert parse_action('{"tool": "calculator", "args": {"expression": "1+1"}}')["tool"] == "calculator"
    assert parse_action('Sure! Here:\n```json\n{"tool_name": "calculator", "arguments": {"expression": "1+1"}}\n```')["args"] == {"expression": "1+1"}
    assert parse_action("{'tool': 'x', 'args': {'a': 1,}}")["tool"] == "x"
    assert parse_action('{"answer": "all done"}')["final"] == "all done"
    assert parse_action('{"tool": "t", "args": "{\\"a\\": 2}"}')["args"] == {"a": 2}
    for bad in ("no json here", "[1, 2]", '{"thought": "hmm"}'):
        with pytest.raises(ValueError):
            parse_action(bad)


def test_workshop_calls_tools_and_feeds_back_observations(ctx):
    llm = Scripted(['{"thought": "add", "tool": "calculator", "args": {"expression": "6*7"}}', '{"final": "It is 42."}'])
    events = []
    run = Workshop(llm, ctx, on_event=events.append).run("what is 6*7?")
    assert run["status"] == "done" and run["final"] == "It is 42." and run["tool_calls"] == 1
    assert [e["kind"] for e in events] == ["start", "call", "observation", "final", "end"]
    second_prompt = " ".join(m.content for m in llm.seen[1])
    assert "OBSERVATION: 42" in second_prompt and "calculator" in second_prompt
    assert "Shortlist of tools" in llm.seen[0][-1].content          # the student sees retrieved tools, not all 1,200


def test_workshop_recovers_from_bad_replies_wrong_names_and_errors(ctx):
    llm = Scripted(["I would use a calculator", '{"tool": "calculater", "args": {}}', '{"tool": "calculator", "args": {"expression": "1/0"}}', '{"search": "convert units"}', '{"final": "gave up politely"}'])
    run = Workshop(llm, ctx).run("x")
    kinds = [s["type"] for s in run["steps"]]
    assert run["status"] == "done" and "search" in kinds
    obs = [s["observation"] for s in run["steps"] if s["type"] == "tool"]
    assert "no tool called 'calculater'" in obs[0] and "Similar:" in obs[0] and obs[1].startswith("ERROR")


def test_workshop_gives_up_after_repeated_nonsense_and_respects_step_limit(ctx):
    assert Workshop(Scripted(["nope"] * 5), ctx).run("x")["status"] == "gave_up"
    loop = Scripted(['{"tool": "calculator", "args": {"expression": "1+1"}}'] * 30)
    run = Workshop(loop, ctx, max_steps=4).run("x")
    assert run["status"] == "max_steps" and run["tool_calls"] == 4
    assert "exact call three times" in run["steps"][-1]["observation"]       # identical repeats are interrupted


def test_workshop_can_be_stopped_and_survives_a_dead_model(ctx):
    import threading
    stop = threading.Event()
    stop.set()
    assert Workshop(Scripted([]), ctx, cancel=stop).run("x")["status"] == "stopped"

    class Dead(LLM):
        name = "dead-7b"

        def chat(self, *a, **k):
            raise ConnectionError("refused")
    r = Workshop(Dead(), ctx).run("x")
    assert r["status"] == "error" and "could not be reached" in r["error"]


def test_blocked_tool_tells_the_student_why(ctx):
    llm = Scripted(['{"tool": "run_python", "args": {"code": "print(1)"}}', '{"final": "asked the user"}'])
    run = Workshop(llm, ctx).run("run code")
    assert "switched off" in run["steps"][0]["observation"]


def test_a_run_can_become_a_recipe(ctx):
    llm = Scripted(['{"tool": "calculator", "args": {"expression": "2+2"}}', '{"tool": "write_file", "args": {"path": "n.txt", "content": "4"}}', '{"final": "ok"}'])
    run = Workshop(llm, ctx).run("save four")
    recipe = run_to_recipe(run, "four")
    assert [s["tool"] for s in recipe["steps"]] == ["calculator", "write_file"]
    ok(call("recipe_save", {"name": "four", "description": recipe["description"], "steps": recipe["steps"]}, ctx))
    (ctx.workspace / "n.txt").unlink()
    assert val(call("recipe_run", {"name": "four"}, ctx))["ok"] and (ctx.workspace / "n.txt").exists()


def test_demo_planner_is_a_tool_user_not_an_ai(ctx):
    run = Workshop(RulePlanner(), ctx).run("calculate 12*(3+4)")
    assert run["status"] == "done" and run["steps"][0]["tool"] == "calculator" and "84" in run["steps"][0]["observation"]
    assert RulePlanner().parameter_count_b() == 0.0


# ---------------------------------------------------------------- HTTP API
@pytest.fixture
def api(tmp_path):
    return TestClient(create_app(str(tmp_path / "r.db"), workspace=str(tmp_path / "ws"),
                                 workshop_llm_factory=lambda body: Scripted(['{"tool": "bar_chart", "args": {"path": "charts/c", "values": [3, 5, 2]}}', '{"final": "Made charts/c.svg"}'])))


def wait(api, rid):
    for _ in range(100):
        r = api.get(f"/api/workshop/runs/{rid}").json()
        if r["status"] != "running":
            return r
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_api_lists_searches_and_calls_tools(api):
    s = api.get("/api/tools/summary").json()
    assert s["total"] >= 1000 and s["ready"] >= 1000 and s["risks"]["exec"]["allowed"] is False
    found = api.get("/api/tools", params={"q": "mortgage payment", "limit": 3}).json()
    assert found["tools"][0]["name"] == "loan_payment" and found["tools"][0]["ready"]
    assert api.get("/api/tools", params={"category": "blender", "limit": 5}).json()["total"] >= 150
    assert api.post("/api/tools/calculator/call", json={"args": {"expression": "2^10"}}).json()["result"] == "1024"
    assert api.get("/api/tools/ghost").status_code == 404


def test_api_run_streams_events_and_serves_files_safely(api):
    rid = api.post("/api/workshop/run", json={"goal": "chart it"}).json()["id"]
    r = wait(api, rid)
    assert r["status"] == "done" and r["files"] == ["charts/c.svg"] and [e["kind"] for e in r["events"]][:3] == ["start", "call", "observation"]
    assert api.get(f"/api/workshop/runs/{rid}", params={"after": len(r["events"])}).json()["events"] == []
    f = api.get("/api/workspace/file", params={"path": "charts/c.svg"})
    assert f.status_code == 200 and f.headers["content-type"].startswith("image/svg")
    assert "sandbox" in f.headers["content-security-policy"] and f.headers["x-content-type-options"] == "nosniff"
    for evil in ("../../etc/passwd", "/etc/passwd", "..%2f..%2fetc%2fpasswd"):
        assert api.get("/api/workspace/file", params={"path": evil}).status_code == 404
    assert api.delete("/api/workspace/file", params={"path": "../x"}).status_code == 404
    assert api.post(f"/api/workshop/runs/{rid}/save_recipe", json={"name": "chart_recipe"}).json()["ok"]
    assert any(x["name"] == "chart_recipe" and not x["builtin"] for x in api.get("/api/recipes").json())


def test_api_one_run_at_a_time(tmp_path):
    gate = __import__("threading").Event()

    class Slow(Scripted):
        def chat(self, *a, **k):
            gate.wait(5)
            return '{"final": "done"}'
    app = TestClient(create_app(str(tmp_path / "r.db"), workspace=str(tmp_path / "ws"), workshop_llm_factory=lambda b: Slow([])))
    first = app.post("/api/workshop/run", json={"goal": "a"}).json()["id"]
    assert app.post("/api/workshop/run", json={"goal": "b"}).status_code == 409
    app.post(f"/api/workshop/runs/{first}/stop")
    gate.set()
    assert wait(app, first)["status"] in ("done", "stopped")
    assert app.post("/api/workshop/run", json={"goal": ""}).status_code == 400


def test_api_student_for_the_workshop_obeys_the_27b_limit(tmp_path):
    app = TestClient(create_app(str(tmp_path / "r.db"), workspace=str(tmp_path / "ws")))
    r = app.post("/api/workshop/run", json={"goal": "x", "backend": "ollama", "model": "llama3:70b"})
    assert r.status_code == 400 and "27" in r.json()["detail"]
    assert app.post("/api/workshop/run", json={"goal": "calculate 2+2*5", "backend": "demo"}).status_code == 200


def test_api_permissions_and_secrets(api, tmp_path):
    assert api.post("/api/tools/run_python/call", json={"args": {"code": "print(1)"}}).json()["blocked"] == "exec"
    st = api.post("/api/workshop/settings", json={"allowed": ["safe", "files", "network", "exec"]}).json()
    assert "exec" in st["allowed"]
    assert json.loads(api.post("/api/tools/run_python/call", json={"args": {"code": "print(1)"}}).json()["result"])["stdout"] == "1\n"
    assert api.post("/api/workshop/settings", json={"allowed": ["root"]}).status_code == 400
    r = api.post("/api/workshop/settings", json={"secrets": {"YOUTUBE_API_KEY": "sekret-123"}})
    assert [c for c in r.json()["connections"] if c["key"] == "YOUTUBE_API_KEY"][0]["set"] is True
    assert "sekret-123" not in r.text and "sekret-123" not in api.get("/api/workshop/settings").text
    assert api.post("/api/workshop/settings", json={"secrets": {"PATH": "x"}}).status_code == 400
    assert api.get("/api/tools/youtube_search_videos").json()["ready"] is True
    secrets_file = tmp_path / "workshop_secrets.json"
    assert secrets_file.exists() and (secrets_file.stat().st_mode & 0o077) == 0


def test_api_recipes_run(api):
    names = [r["name"] for r in api.get("/api/recipes").json()]
    assert {"instagram_post_pack", "blender_showcase", "youtube_launch_kit"} <= set(names)
    r = api.post("/api/recipes/chart_report/run", json={"params": {"values": [1, 2, 3]}})
    assert r.status_code == 200 and r.json()["ok"] and "charts/bar.svg" in r.json()["files"]
    assert api.post("/api/recipes/ghost/run", json={}).status_code == 400


def test_the_web_app_serves_the_workshop(api):
    html = api.get("/").text
    assert "workshop.js" in html and "workshop.css" in html
    assert api.get("/static/workshop.js").status_code == 200
