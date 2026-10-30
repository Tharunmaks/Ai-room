"""The Workshop: the student model uses the toolbox to reach a goal.

Small models rarely have reliable native function-calling, so the protocol is plain JSON in text, one action per turn:

    {"thought": "…", "tool": "bar_chart", "args": {"path": "charts/sales", "values": [3, 5, 2]}}
    {"search": "convert a video to a gif"}
    {"final": "what was done and where the files are"}

Each turn the model sees the goal, its recent steps and a shortlist of tools retrieved by BM25 from the whole toolbox
(so it never has to read all 1,200 descriptions), and it can ask for a different shortlist with `search`.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from typing import Callable

from ..llm import LLM, Message
from .registry import REGISTRY, Context, call, missing, search

SYSTEM = """You are a student in AI Room, working in a workshop with a large toolbox. Reach the goal by calling tools, one per turn.

Reply with exactly ONE JSON object and nothing else:
  To call a tool:   {"thought": "why", "tool": "tool_name", "args": {"argument": value}}
  To find tools:    {"search": "what you need, in a few words"}
  When finished:    {"final": "what you did, what you found, and the files you made"}

Rules:
- Use only tool names and argument names from the shortlist (or from a search). Never invent results.
- File paths are relative to the workspace (for example "charts/sales"). Tools tell you where they saved things.
- Read each observation before choosing the next step. If a tool returns an error, read it, fix the call or pick another tool.
- Prefer a few precise steps. When the goal is done, give the final answer."""


def _balanced(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def parse_action(text: str) -> dict:
    """Extract the model's JSON action; tolerate code fences, prose around it, and common key aliases."""
    t = re.sub(r"```(?:json)?", "", text)
    blob = _balanced(t)
    if not blob:
        raise ValueError("no JSON object found")
    try:
        d = json.loads(blob)
    except json.JSONDecodeError:
        d = json.loads(re.sub(r",\s*([}\]])", r"\1", blob.replace("'", '"')))
    if not isinstance(d, dict):
        raise ValueError("the action must be a JSON object")
    for alias in ("tool_name", "action", "function", "name"):
        if alias in d and "tool" not in d and isinstance(d[alias], str):
            d["tool"] = d.pop(alias)
    for alias in ("arguments", "parameters", "input", "params"):
        if alias in d and "args" not in d:
            d["args"] = d.pop(alias)
    if isinstance(d.get("args"), str):
        try:
            d["args"] = json.loads(d["args"])
        except json.JSONDecodeError:
            raise ValueError("'args' must be a JSON object")
    for alias in ("answer", "final_answer", "result", "done"):
        if alias in d and "final" not in d and "tool" not in d and "search" not in d and isinstance(d[alias], str):
            d["final"] = d.pop(alias)
    if not any(k in d for k in ("tool", "search", "final")):
        raise ValueError("the object needs one of: tool, search, final")
    return d


def shortlist(query: str, ctx: Context, k: int = 12, show_blocked: bool = False) -> str:
    seen, lines = set(), []
    for t in search(query, k * 2, ctx, ready_only=True):
        if t.name in seen:
            continue
        seen.add(t.name)
        blocked = t.risk not in ctx.allowed
        if blocked and not show_blocked:
            continue
        lines.append(f"- {t.signature()} — {t.desc[:150]}" + (" [needs permission]" if blocked else ""))
        if len(lines) >= k:
            break
    return "\n".join(lines) or "(no matching tools — try different words)"


class Workshop:
    def __init__(self, llm: LLM, ctx: Context, max_steps: int = 14, on_event: Callable[[dict], None] | None = None, cancel: threading.Event | None = None, history_steps: int = 8):
        self.llm, self.ctx, self.max_steps, self.cancel = llm, ctx, max_steps, cancel or threading.Event()
        self.on_event = on_event or (lambda e: None)
        self.history_steps = history_steps

    def emit(self, kind: str, **kw):
        self.on_event({"kind": kind, "t": round(time.time(), 2), **kw})

    def run(self, goal: str) -> dict:
        run = {"id": uuid.uuid4().hex[:10], "goal": goal, "steps": [], "final": None, "status": "running", "artifacts": [], "started": time.time()}
        self.emit("start", goal=goal, run_id=run["id"])
        hist: list[tuple[str, str]] = []  # (assistant json, observation)
        query, bad, last_sig, repeats = goal, 0, None, 0
        for step in range(1, self.max_steps + 1):
            if self.cancel.is_set():
                run["status"] = "stopped"
                break
            msgs = [Message("system", SYSTEM), Message("user", f"GOAL: {goal}")]
            for a, o in hist[-self.history_steps:]:
                msgs += [Message("assistant", a), Message("user", "OBSERVATION: " + o)]
            msgs.append(Message("user", ("Shortlist of tools that look relevant (search again for others):\n" + shortlist(query, self.ctx)) + "\n\nYour next action as one JSON object:"))
            try:
                raw = self.llm.chat(msgs, temperature=0.2, max_tokens=900)
            except Exception as e:  # model server down, etc.
                run["status"] = "error"
                run["error"] = f"the student model could not be reached: {e}"
                self.emit("error", message=run["error"])
                break
            try:
                act = parse_action(raw)
                bad = 0
            except (ValueError, json.JSONDecodeError) as e:
                bad += 1
                obs = f"That was not a valid action ({e}). Reply with one JSON object such as {{\"tool\": \"calculator\", \"args\": {{\"expression\": \"2+2\"}}}}."
                hist.append((raw[:400], obs))
                self.emit("invalid", step=step, raw=raw[:300], message=obs)
                if bad >= 3:
                    run["status"] = "gave_up"
                    run["error"] = "the model kept replying without a valid JSON action"
                    break
                continue
            if "final" in act:
                run["final"] = str(act["final"])
                run["status"] = "done"
                self.emit("final", step=step, text=run["final"], thought=act.get("thought"))
                break
            if "search" in act:
                query = str(act["search"])
                obs = shortlist(query, self.ctx, 14, show_blocked=True)
                rec = {"step": step, "type": "search", "query": query, "observation": obs}
                run["steps"].append(rec)
                hist.append((json.dumps(act, ensure_ascii=False)[:500], obs))
                self.emit("search", step=step, query=query, tools=[l.split("(")[0][2:] for l in obs.splitlines() if l.startswith("- ")])
                continue
            tool, args = str(act["tool"]), act.get("args") or {}
            sig = json.dumps([tool, args], sort_keys=True, default=str)
            repeats = repeats + 1 if sig == last_sig else 0
            last_sig = sig
            self.emit("call", step=step, tool=tool, args=args, thought=act.get("thought"))
            if tool not in REGISTRY:
                near = [t.name for t in search(tool.replace("_", " "), 5, self.ctx)]
                res = {"ok": False, "error": f"there is no tool called '{tool}'. Similar: {', '.join(near) or 'none'}. Use a search action to find the right one.", "artifacts": []}
            elif repeats >= 2:
                res = {"ok": False, "error": "you have made this exact call three times. Change the arguments or choose a different step, or give your final answer.", "artifacts": []}
            else:
                res = call(tool, args, self.ctx)
            obs = res.get("result") if res["ok"] else "ERROR: " + res["error"]
            obs = (obs or "")[:1500]
            rec = {"step": step, "type": "tool", "tool": tool, "args": args, "ok": res["ok"], "observation": obs, "files": [a["path"] for a in res.get("artifacts", [])], "thought": act.get("thought")}
            run["steps"].append(rec)
            run["artifacts"] += res.get("artifacts", [])
            hist.append((json.dumps({k: act[k] for k in ("thought", "tool", "args") if k in act}, ensure_ascii=False, default=str)[:700], obs + (f"\n(files created: {', '.join(rec['files'])})" if rec["files"] else "")))
            query = goal + " " + tool + " " + (obs[:160] if not res["ok"] else "")
            self.emit("observation", step=step, tool=tool, ok=res["ok"], text=obs, files=rec["files"], seconds=res.get("seconds"), blocked=res.get("blocked"))
        else:
            run["status"] = "max_steps"
            run["final"] = f"I used all {self.max_steps} steps without finishing. Here is what exists so far: " + (", ".join(a["path"] for a in run["artifacts"][-6:]) or "no files yet.")
            self.emit("final", step=self.max_steps, text=run["final"])
        run["seconds"] = round(time.time() - run["started"], 1)
        run["tool_calls"] = sum(1 for s in run["steps"] if s["type"] == "tool")
        self.emit("end", status=run["status"], seconds=run["seconds"], files=len({a["path"] for a in run["artifacts"]}))
        return run


def save_run(ctx: Context, run: dict) -> str:
    p = ctx.path(f"runs/{run['id']}.json")
    p.write_text(json.dumps({k: v for k, v in run.items()}, indent=1, default=str, ensure_ascii=False))
    return f"runs/{run['id']}.json"


def run_to_recipe(run: dict, name: str, description: str = "") -> dict:
    """Turn the successful tool calls of a run into a recipe, so the same job can be repeated without a model."""
    steps = [{"tool": s["tool"], "args": s["args"]} for s in run["steps"] if s["type"] == "tool" and s["ok"]]
    return {"description": description or f"Saved from the goal: {run['goal'][:100]}", "params": {}, "steps": steps}


# ------------------------------------------------------------------ rule-based planner (a stand-in "student" for demos and tests)
class RulePlanner(LLM):
    """Not an AI: picks tools by keywords so the Workshop can be explored without a model. It follows the same JSON protocol."""
    name = "demo-planner"
    demo_oracle = False

    def parameter_count_b(self):
        return 0.0

    def chat(self, messages, temperature=0.0, max_tokens=900):
        goal = next((m.content[6:] for m in messages if m.role == "user" and m.content.startswith("GOAL: ")), "")
        done = [m for m in messages if m.role == "assistant"]
        g = goal.lower()
        plan = self.plan(goal, g)
        if len(done) < len(plan):
            return json.dumps(plan[len(done)])
        last_obs = [m.content for m in messages if m.role == "user" and m.content.startswith("OBSERVATION")]
        return json.dumps({"final": "Done. " + (last_obs[-1][13:240] if last_obs else "")})

    @staticmethod
    def plan(goal: str, g: str) -> list[dict]:
        nums = re.findall(r"-?\d+(?:\.\d+)?", goal)
        expr = re.search(r"([\d\s.+\-*/^()x×÷]{3,})", goal)
        if re.search(r"\b(calculate|compute|what is|evaluate|solve)\b", g) and expr and re.search(r"[+\-*/^×÷]", expr.group(1)):
            return [{"thought": "This is arithmetic.", "tool": "calculator", "args": {"expression": expr.group(1).strip().replace("x", "*")}}]
        if re.search(r"\b(chart|plot|graph)\b", g) and len(nums) >= 2:
            return [{"thought": "Draw the numbers as a bar chart.", "tool": "bar_chart", "args": {"path": "charts/chart", "values": [float(n) for n in nums], "title": "Chart"}}]
        for t in ("house", "robot", "car", "snowman", "castle", "tower", "rocket", "windmill", "tree", "bridge", "solar system", "chessboard", "dna"):
            if t in g and re.search(r"\b(3d|blender|model|scene|build|make|create)\b", g):
                key = {"solar system": "solar_system", "dna": "dna_helix", "tree": "tree"}.get(t, t)
                return [{"thought": f"Use the Blender showcase recipe for a {t}.", "tool": "recipe_run", "args": {"name": "blender_showcase", "params": {"thing": key if key != "dna_helix" else "dna_helix"}}}]
        if re.search(r"\b(instagram|insta|post)\b", g):
            topic = re.sub(r"(?i)\b(make|create|an?|the|instagram|insta|post|about|for|pack)\b", " ", goal).strip(" .") or "My topic"
            return [{"thought": "Use the Instagram post pack recipe.", "tool": "recipe_run", "args": {"name": "instagram_post_pack", "params": {"topic": topic.strip().capitalize()}}}]
        if re.search(r"\b(youtube|thumbnail)\b", g):
            topic = re.sub(r"(?i)\b(make|create|an?|the|youtube|thumbnail|video|about|for|kit|launch)\b", " ", goal).strip(" .") or "My video"
            return [{"thought": "Use the YouTube launch kit.", "tool": "recipe_run", "args": {"name": "youtube_launch_kit", "params": {"topic": topic.strip().capitalize()}}}]
        if re.search(r"\b(video|reel|slides?)\b", g):
            return [{"thought": "Make a slide video.", "tool": "recipe_run", "args": {"name": "explainer_video", "params": {"title": goal[:60]}}}]
        if re.search(r"\b(colab|kaggle|notebook|train)\b", g):
            return [{"thought": "Write starter notebooks.", "tool": "recipe_run", "args": {"name": "colab_kaggle_kit", "params": {"kind": "classifier"}}}]
        m = re.search(r"\b(?:who|what) (?:is|was|are) (.+?)\??$", goal, re.I)
        if m:
            return [{"thought": "Look it up on Wikipedia.", "tool": "wikipedia_summary", "args": {"title": m.group(1).strip().title()}}]
        return [{"thought": "Find tools that fit the goal.", "search": goal}]
