"""Recipes: saved tool pipelines (automation). A recipe is a list of tool calls with {{placeholders}}; running it needs no AI.

    {"description": "...", "params": {"topic": {"default": "robots"}}, "steps": [{"tool": "image_text_card_instagram_post", "args": {"text": "{{topic}}"}}]}

Placeholders: {{name}} is a parameter; {{stepN}} is the text result of step N (1-based). A step may set "optional": true to continue after an error.
"""

from __future__ import annotations

import json
import re

from .registry import Context, ToolError, call, reg

BUILTIN: dict[str, dict] = {
    "instagram_post_pack": {
        "description": "A designed Instagram post image, story version, caption and hashtags for a topic.",
        "params": {"topic": {"default": "Learning without training", "desc": "what the post is about"}, "tagline": {"default": "Swipe for more", "desc": "small line under the headline"}, "tone": {"default": "friendly"}},
        "steps": [
            {"tool": "image_text_card_instagram_post", "args": {"text": "{{topic}}", "sub": "{{tagline}}", "out": "instagram/post.png"}},
            {"tool": "image_text_card_instagram_story", "args": {"text": "{{topic}}", "sub": "{{tagline}}", "out": "instagram/story.png"}},
            {"tool": "instagram_caption_template", "args": {"topic": "{{topic}}", "tone": "{{tone}}"}},
            {"tool": "write_file", "args": {"path": "instagram/caption.txt", "content": "{{step3}}"}},
        ]},
    "youtube_launch_kit": {
        "description": "Thumbnail, title ideas and a description template for a new YouTube video.",
        "params": {"topic": {"default": "AI that learns without training"}, "subtitle": {"default": "Explained simply"}},
        "steps": [
            {"tool": "image_text_card_youtube_thumbnail", "args": {"text": "{{topic}}", "sub": "{{subtitle}}", "badge": "NEW", "out": "youtube/thumbnail.png"}},
            {"tool": "youtube_title_ideas", "args": {"topic": "{{topic}}"}},
            {"tool": "youtube_description_template", "args": {"title": "{{topic}}", "chapters": [["0:00", "Intro"], ["1:00", "The idea"], ["3:00", "Demo"], ["5:00", "Wrap-up"]], "hashtags": "#learning #ai"}},
            {"tool": "write_file", "args": {"path": "youtube/titles.txt", "content": "{{step2}}"}},
            {"tool": "write_file", "args": {"path": "youtube/description.txt", "content": "{{step3}}"}},
        ]},
    "blender_showcase": {
        "description": "Build a ready-made 3D object in a Blender scene, set camera, preview PNG + turntable GIF, and write the bpy script (render it with real Blender if installed).",
        "params": {"thing": {"default": "house", "desc": "see blender_list_generators (house, robot, car, snowman, castle, rocket, windmill, …)"}, "scene": {"default": "showcase"}},
        "steps": [
            {"tool": "blender_new_scene", "args": {"scene": "{{scene}}"}},
            {"tool": "blender_build_{{thing}}", "args": {"scene": "{{scene}}"}},
            {"tool": "blender_camera_frame_all", "args": {"scene": "{{scene}}"}},
            {"tool": "blender_preview_png", "args": {"scene": "{{scene}}"}},
            {"tool": "blender_preview_turntable_gif", "args": {"scene": "{{scene}}", "frames": 24}},
            {"tool": "blender_build_script", "args": {"scene": "{{scene}}"}},
        ]},
    "blender_animated_turntable": {
        "description": "A scene with an orbiting camera, written as a script that renders an MP4 in real Blender.",
        "params": {"thing": {"default": "robot"}, "scene": {"default": "turntable"}},
        "steps": [
            {"tool": "blender_new_scene", "args": {"scene": "{{scene}}"}},
            {"tool": "blender_build_{{thing}}", "args": {"scene": "{{scene}}"}},
            {"tool": "blender_camera_frame_all", "args": {"scene": "{{scene}}"}},
            {"tool": "blender_anim_orbit_camera", "args": {"scene": "{{scene}}", "frames": 120}},
            {"tool": "blender_render_preset_hd_1080p", "args": {"scene": "{{scene}}"}},
            {"tool": "blender_render_as_video", "args": {"scene": "{{scene}}"}},
            {"tool": "blender_build_script", "args": {"scene": "{{scene}}"}},
            {"tool": "blender_preview_turntable_gif", "args": {"scene": "{{scene}}", "frames": 24}},
        ]},
    "explainer_video": {
        "description": "A slide video from a title and key points, with subtitles and a vertical (reel) version.",
        "params": {"title": {"default": "How AI Room works"}, "points": {"default": ["A teacher made of textbooks|No AI teacher to hallucinate", "The student takes notes|Weights never change", "Exams in 12 classes|Pass 90% in every subject"], "desc": "list of 'Headline | detail'"}},
        "steps": [
            {"tool": "video_from_text_slides", "args": {"slides": ["{{title}}"], "seconds": 3, "out": "videos/explainer_title.mp4", "crossfade": False}},
            {"tool": "video_from_text_slides", "args": {"slides": "{{points}}", "seconds": 4, "out": "videos/explainer_body.mp4"}},
            {"tool": "video_concat", "args": {"paths": ["videos/explainer_title.mp4", "videos/explainer_body.mp4"], "out": "videos/explainer.mp4"}},
            {"tool": "video_fit_instagram_reel", "args": {"path": "videos/explainer.mp4", "out": "videos/explainer_reel.mp4"}},
        ]},
    "study_flashcard_video": {
        "description": "Turn question|answer lines into a flashcard slide video.",
        "params": {"cards": {"default": ["What is 7 × 8?|56", "Capital of Japan?|Tokyo", "H2O is…|Water"]}},
        "steps": [{"tool": "video_from_text_slides", "args": {"slides": "{{cards}}", "seconds": 3, "out": "videos/flashcards.mp4"}}]},
    "brand_pack": {
        "description": "A badge, round avatar and favicon for a name.",
        "params": {"name": {"default": "AI Room"}, "color": {"default": "#4f7cff"}},
        "steps": [
            {"tool": "image_badge", "args": {"text": "{{name}}", "color": "{{color}}", "out": "brand/badge.png"}},
            {"tool": "image_avatar_initials", "args": {"name": "{{name}}", "out": "brand/avatar.png"}},
            {"tool": "image_fit_favicon", "args": {"path": "brand/badge.png", "out": "brand/favicon.png"}},
            {"tool": "image_fit_app_icon", "args": {"path": "brand/badge.png", "out": "brand/app_icon.png"}},
        ]},
    "chart_report": {
        "description": "Bar, line and pie charts plus summary statistics for a list of numbers.",
        "params": {"values": {"default": [12, 18, 9, 22, 15]}, "labels": {"default": "Mon,Tue,Wed,Thu,Fri"}, "title": {"default": "Weekly results"}},
        "steps": [
            {"tool": "stats_summary", "args": {"numbers": "{{values}}"}},
            {"tool": "bar_chart", "args": {"path": "charts/bar", "values": "{{values}}", "labels": "{{labels}}", "title": "{{title}}"}},
            {"tool": "line_chart", "args": {"path": "charts/line", "values": "{{values}}", "labels": "{{labels}}", "title": "{{title}}"}},
            {"tool": "pie_chart", "args": {"path": "charts/pie", "values": "{{values}}", "labels": "{{labels}}", "title": "{{title}}"}},
        ]},
    "ml_quickstart": {
        "description": "Train a decision tree and a logistic regression on a CSV, rank features, and write a Colab notebook for scaling up.",
        "params": {"csv": {"default": "data/train.csv"}, "target": {"default": "label"}},
        "steps": [
            {"tool": "ml_feature_importance", "args": {"csv": "{{csv}}", "target": "{{target}}"}},
            {"tool": "train_tree", "args": {"csv": "{{csv}}", "target": "{{target}}", "name": "quick_tree"}},
            {"tool": "train_logistic", "args": {"csv": "{{csv}}", "target": "{{target}}", "name": "quick_logistic"}, "optional": True},
            {"tool": "colab_notebook_classifier", "args": {}},
        ]},
    "model_3d_check": {
        "description": "Make a 3D shape, measure it (volume, watertight?) and render a picture.",
        "params": {"shape": {"default": "gear", "desc": "see mesh_primitives_list"}},
        "steps": [
            {"tool": "mesh_{{shape}}", "args": {"path": "models/{{shape}}", "format": "stl"}},
            {"tool": "mesh_stats", "args": {"path": "models/{{shape}}.stl"}},
            {"tool": "mesh_render_png", "args": {"paths": ["models/{{shape}}.stl"], "out": "renders/{{shape}}.png"}},
        ]},
    "colab_kaggle_kit": {
        "description": "The same starter notebook for Colab and for Kaggle.",
        "params": {"kind": {"default": "classifier", "desc": "e.g. classifier, regression, pytorch_mnist, lora_finetune, image_generation"}},
        "steps": [{"tool": "colab_notebook_{{kind}}", "args": {}}, {"tool": "kaggle_notebook_{{kind}}", "args": {}}]},
    "wiki_research_note": {
        "description": "Pull a Wikipedia summary and save it as a cited note (needs internet).",
        "params": {"topic": {"default": "Alan Turing"}},
        "steps": [
            {"tool": "wikipedia_summary", "args": {"title": "{{topic}}"}},
            {"tool": "write_file", "args": {"path": "notes/{{topic}}.md", "content": "# {{topic}}\n\n{{step1}}\n\nSource: Wikipedia (CC BY-SA 4.0)\n"}},
        ]},
    "gif_from_text": {
        "description": "A short animated GIF of text slides.",
        "params": {"slides": {"default": ["3", "2", "1", "Go!"]}},
        "steps": [
            {"tool": "video_from_text_slides", "args": {"slides": "{{slides}}", "seconds": 1, "size": "640x360", "crossfade": False, "out": "videos/slides.mp4"}},
            {"tool": "video_to_gif", "args": {"path": "videos/slides.mp4", "seconds": 6, "width": 480, "out": "videos/slides.gif"}},
        ]},
}

_PH = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


def subst(v, params: dict, steps: list):
    """Replace placeholders; a string that is exactly one placeholder keeps its original type (list, number…)."""
    if isinstance(v, str):
        m = _PH.fullmatch(v.strip())
        if m:
            k = m.group(1)
            if k in params:
                return params[k]
            if k.startswith("step") and k[4:].isdigit():
                n = int(k[4:])
                if 1 <= n <= len(steps):
                    return steps[n - 1]
                raise ToolError(f"{{{{{k}}}}} refers to a step that has not run yet")
            raise ToolError(f"recipe parameter '{k}' was not given")

        def rep(mm):
            k = mm.group(1)
            if k in params:
                val = params[k]
                return val if isinstance(val, str) else json.dumps(val, ensure_ascii=False) if isinstance(val, (list, dict)) else str(val)
            if k.startswith("step") and k[4:].isdigit() and 1 <= int(k[4:]) <= len(steps):
                return steps[int(k[4:]) - 1]
            raise ToolError(f"recipe parameter '{k}' was not given")
        return _PH.sub(rep, v)
    if isinstance(v, list):
        return [subst(x, params, steps) for x in v]
    if isinstance(v, dict):
        return {k: subst(x, params, steps) for k, x in v.items()}
    return v


def _folder(ctx: Context):
    return ctx.workspace / "recipes"


def get_recipe(ctx: Context, name: str) -> dict:
    p = _folder(ctx) / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    if name in BUILTIN:
        return BUILTIN[name]
    raise ToolError(f"no recipe named '{name}'. Available: {', '.join(sorted(list_recipes(ctx)))}")


def list_recipes(ctx: Context) -> dict[str, str]:
    out = {n: r["description"] for n, r in BUILTIN.items()}
    if _folder(ctx).exists():
        for p in sorted(_folder(ctx).glob("*.json")):
            try:
                out[p.stem] = json.loads(p.read_text()).get("description", "(saved recipe)")
            except ValueError:
                pass
    return out


def run_recipe(ctx: Context, name: str, params: dict | None = None, stop_on_error: bool = True) -> dict:
    r = get_recipe(ctx, name)
    given = dict(params or {})
    p = {k: given.get(k, spec.get("default")) for k, spec in (r.get("params") or {}).items()} | {k: v for k, v in given.items() if k not in (r.get("params") or {})}
    for k, spec in (r.get("params") or {}).items():
        if p.get(k) is None:
            raise ToolError(f"recipe '{name}' needs the parameter '{k}'" + (f" ({spec['desc']})" if spec.get("desc") else ""))
    results, log, ok_all = [], [], True
    for i, step in enumerate(r["steps"], 1):
        tool = subst(step["tool"], p, results)
        args = subst(step.get("args", {}), p, results)
        res = call(tool, args, ctx)
        text = res.get("result", res.get("error", ""))
        results.append(text)
        log.append({"step": i, "tool": tool, "ok": res["ok"], "output": text[:300], "files": [a["path"] for a in res.get("artifacts", [])]})
        if not res["ok"]:
            ok_all = False
            if stop_on_error and not step.get("optional"):
                break
    return {"recipe": name, "ok": ok_all, "steps": log, "files": sorted({f for s in log for f in s["files"]})}


# ------------------------------------------------------------------ tools
A = "automation"
reg("recipe_list", "List automation recipes (built-in and saved).", A, "", lambda ctx: list_recipes(ctx), ctx=True, risk="files", tags=("workflow", "pipeline"))
reg("recipe_show", "Show a recipe's parameters and steps.", A, "name:str", lambda ctx, name: get_recipe(ctx, name), ctx=True, risk="files")
reg("recipe_run", "Run a recipe with parameters (a dict). Runs every step in order and reports each result.", A, "name:str, params:dict?, stop_on_error:bool=true", lambda ctx, name, params, stop_on_error: run_recipe(ctx, name, params, stop_on_error), ctx=True, risk="files", timeout=900.0, tags=("workflow", "pipeline", "batch"))
reg("recipe_run_batch", "Run a recipe once for each parameter set in a list — e.g. a post pack for ten topics.", A, "name:str, param_sets:list", lambda ctx, name, param_sets: [{"params": ps, **{k: v for k, v in run_recipe(ctx, name, ps).items() if k in ("ok", "files")}} for ps in param_sets[:30]], ctx=True, risk="files", timeout=1800.0, tags=("batch", "bulk"))
reg("recipe_save", "Save a new recipe: description, params {name: {default, desc}} and steps [{tool, args}].", A, "name:str, description:str, steps:list, params:dict?", lambda ctx, name, description, steps, params: _save(ctx, name, description, steps, params), ctx=True, risk="files")
reg("recipe_delete", "Delete a saved recipe (built-ins cannot be deleted).", A, "name:str", lambda ctx, name: ((_folder(ctx) / f"{name}.json").unlink(), "deleted")[1] if (_folder(ctx) / f"{name}.json").exists() else (_ for _ in ()).throw(ToolError("only saved recipes can be deleted")), ctx=True, risk="files")
reg("recipe_validate", "Check that every tool a recipe uses exists and that its arguments match.", A, "name:str", lambda ctx, name: _validate(ctx, name), ctx=True, risk="files")


def _save(ctx, name, description, steps, params):
    if not re.fullmatch(r"[\w-]{1,48}", name):
        raise ToolError("recipe names use letters, digits, - and _")
    from .registry import REGISTRY
    for s in steps:
        if "tool" not in s:
            raise ToolError("every step needs a 'tool'")
        if "{{" not in s["tool"] and s["tool"] not in REGISTRY:
            raise ToolError(f"unknown tool '{s['tool']}' — search for it first")
    _folder(ctx).mkdir(exist_ok=True)
    p = _folder(ctx) / f"{name}.json"
    p.write_text(json.dumps({"description": description, "params": params or {}, "steps": steps}, indent=1, ensure_ascii=False))
    return {"saved": ctx.produced(p, "recipe", description)}


def _validate(ctx, name):
    from .registry import REGISTRY, missing
    r = get_recipe(ctx, name)
    issues = []
    for i, s in enumerate(r["steps"], 1):
        t = s["tool"]
        if "{{" in t:
            continue
        tool = REGISTRY.get(t)
        if not tool:
            issues.append(f"step {i}: unknown tool {t}")
            continue
        for p in tool.params:
            if p.required and p.name not in s.get("args", {}):
                issues.append(f"step {i}: {t} needs '{p.name}'")
        for a in s.get("args", {}):
            if a not in {p.name for p in tool.params}:
                issues.append(f"step {i}: {t} has no argument '{a}'")
        for m in missing(tool, ctx):
            issues.append(f"step {i}: {t} needs {m}")
    return {"ok": not issues, "issues": issues}
