"""The tool registry: every tool the student can use, with a schema, a risk level and what it needs.

A *tool* is a named function with typed parameters. Tools are found by search (BM25
over their descriptions) rather than listed in a prompt, so the catalogue can be
large. Risky tools are off unless the user allows their risk level.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..memory import tokenize

# What a tool may touch. The user decides which levels are allowed.
RISKS = {
    "safe": "pure computation — no files, no network",
    "files": "reads and writes files inside the workspace folder",
    "network": "reads public web pages and APIs",
    "exec": "runs programs on this computer (Python, ffmpeg, Blender…)",
    "post": "publishes or changes something on an outside account (YouTube, Instagram, GitHub…)",
}
DEFAULT_ALLOWED = ("safe", "files", "network", "exec")  # posting is always opt-in


class ToolError(Exception):
    """A problem the student can read and react to (bad argument, missing software, …)."""


@dataclass
class Param:
    name: str
    type: str = "str"           # str | int | float | bool | list | dict | path
    desc: str = ""
    default: Any = None
    required: bool = True
    choices: tuple | None = None


@dataclass
class Tool:
    name: str
    title: str
    desc: str
    category: str
    params: list[Param]
    fn: Callable[..., Any]
    risk: str = "safe"
    needs: tuple[str, ...] = ()   # "py:PIL", "bin:ffmpeg", "env:YOUTUBE_API_KEY", "net"
    tags: tuple[str, ...] = ()
    produces: str = ""            # "image" | "video" | "model3d" | "audio" | "text" | "file"
    timeout: float = 0.0          # longer limit for slow tools (rendering, encoding); 0 = the context's limit

    def signature(self) -> str:
        args = ", ".join(f"{p.name}{'' if p.required else '?'}:{p.type}" for p in self.params)
        return f"{self.name}({args})"

    def card(self) -> dict:
        return {"name": self.name, "title": self.title, "desc": self.desc, "category": self.category, "risk": self.risk, "needs": list(self.needs),
                "produces": self.produces, "signature": self.signature(), "params": [p.__dict__ | {"choices": list(p.choices) if p.choices else None} for p in self.params]}


@dataclass
class Context:
    """What a tool call may use: the workspace folder, secrets, and a place to report produced files."""
    workspace: Path
    secrets: dict[str, str] = field(default_factory=dict)
    allowed: tuple[str, ...] = DEFAULT_ALLOWED
    timeout: float = 60.0
    artifacts: list[dict] = field(default_factory=list)
    http: Any = None  # injected httpx-like client (tests)

    def path(self, rel: str, must_exist: bool = False) -> Path:
        """Resolve a user-supplied path inside the workspace; anything that escapes it is refused."""
        if not rel or "\x00" in rel:
            raise ToolError("a file path is required")
        p = (self.workspace / rel).resolve()
        root = self.workspace.resolve()
        if p != root and root not in p.parents:
            raise ToolError(f"'{rel}' is outside the workspace; use a relative path like 'work/{Path(rel).name}'")
        if must_exist and not p.exists():
            raise ToolError(f"file not found in the workspace: {rel}")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def secret(self, key: str) -> str:
        v = self.secrets.get(key) or os.environ.get(key, "")
        if not v:
            raise ToolError(f"{key} is not set. Add it in the Workshop → Connections panel (or as an environment variable).")
        return v

    def produced(self, path: Path, kind: str = "file", note: str = "") -> str:
        rel = str(path.resolve().relative_to(self.workspace.resolve()))
        self.artifacts.append({"path": rel, "kind": kind, "bytes": path.stat().st_size if path.exists() else 0, "note": note})
        return rel


REGISTRY: dict[str, Tool] = {}


def _coerce(p: Param, v: Any) -> Any:
    if v is None:
        return p.default
    t = p.type
    try:
        if t == "int":
            return int(float(v))
        if t == "float":
            return float(v)
        if t == "bool":
            return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "y", "on")
        if t == "list":
            if isinstance(v, str):
                s = v.strip()
                if s.startswith("{"):
                    return json.loads(s)
                return json.loads(s) if s.startswith("[") else [x.strip() for x in s.split(",") if x.strip()]
            return v if isinstance(v, dict) else list(v)
        if t == "dict":
            return json.loads(v) if isinstance(v, str) else dict(v)
        return str(v)
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        raise ToolError(f"argument '{p.name}' should be {t}, got {v!r}") from e


def tool(name: str, desc: str, category: str, params: list[Param] | None = None, *, title: str = "", risk: str = "safe", needs: tuple = (), tags: tuple = (), produces: str = ""):
    """Register a function as a tool. The function receives (ctx, **args)."""
    def deco(fn):
        add(Tool(name, title or name.replace("_", " ").capitalize(), desc, category, params or [], fn, risk, tuple(needs), tuple(tags), produces))
        return fn
    return deco


def add(t: Tool) -> Tool:
    if t.name in REGISTRY:
        raise ValueError(f"duplicate tool {t.name}")
    if t.risk not in RISKS:
        raise ValueError(f"unknown risk {t.risk} for {t.name}")
    REGISTRY[t.name] = t
    return t


def P(name, type="str", desc="", default=None, required=None, choices=None) -> Param:
    return Param(name, type, desc, default, (default is None) if required is None else required, tuple(choices) if choices else None)


# ---------------------------------------------------------------- availability

def missing(t: Tool, ctx: Context | None = None) -> list[str]:
    out = []
    for n in t.needs:
        kind, _, what = n.partition(":")
        if kind == "py" and importlib.util.find_spec(what) is None:
            out.append(f"Python package '{what}' (pip install {what})")
        elif kind == "bin" and shutil.which(what) is None:
            out.append(f"program '{what}' installed and on PATH")
        elif kind == "env" and not ((ctx.secrets.get(what) if ctx else "") or os.environ.get(what)):
            out.append(f"{what} (set it in Connections)")
    return out


def status(t: Tool, ctx: Context | None = None) -> str:
    if missing(t, ctx):
        return "needs-setup"
    return "ready"


# ---------------------------------------------------------------- search

_index_cache: dict[str, Any] = {"n": -1}


def _index():
    if _index_cache["n"] != len(REGISTRY):
        names = list(REGISTRY)
        docs = [tokenize(" ".join([REGISTRY[n].name.replace("_", " "), REGISTRY[n].title, REGISTRY[n].desc, REGISTRY[n].category, " ".join(REGISTRY[n].tags)])) for n in names]
        df = Counter(t for d in docs for t in set(d))
        avg = sum(len(d) for d in docs) / max(1, len(docs))
        _index_cache.update(n=len(REGISTRY), names=names, docs=docs, tfs=[Counter(d) for d in docs], df=df, avg=avg)
    return _index_cache


SYNONYMS = {
    "kilometres": ("km", "length"), "kilometers": ("km", "length"), "miles": ("mi", "length"), "metres": ("m", "length"), "meters": ("m", "length"),
    "pounds": ("lb", "mass"), "kilograms": ("kg", "mass"), "ounces": ("oz",), "gallons": ("gal", "volume"), "litres": ("l", "volume"), "liters": ("l", "volume"),
    "celsius": ("temperature",), "fahrenheit": ("temperature",), "kelvin": ("temperature",), "plus": ("add",), "minus": ("subtract",), "times": ("multiply",),
    "divide": ("division",), "average": ("mean",), "wiki": ("wikipedia", "wikimedia"), "photo": ("image",), "picture": ("image",), "movie": ("video",),
    "clip": ("video",), "3d": ("blender", "mesh"), "model": ("mesh",), "csv": ("table", "data"), "spreadsheet": ("table", "data"), "plot": ("chart",), "graph": ("chart",),
}


def search(query: str, k: int = 10, ctx: Context | None = None, category: str | None = None, ready_only: bool = False) -> list[Tool]:
    ix = _index()
    q = set(tokenize(query.replace("_", " ")))
    for w in list(q):
        q.update(SYNONYMS.get(w, ()))
    n = len(ix["names"])
    scored = []
    for name, d, tf in zip(ix["names"], ix["docs"], ix["tfs"]):
        t = REGISTRY[name]
        if category and t.category != category:
            continue
        if ready_only and missing(t, ctx):
            continue
        s = 0.0
        for term in q:
            f = tf.get(term)
            if f:
                idf = math.log(1 + (n - ix["df"][term] + 0.5) / (ix["df"][term] + 0.5))
                s += idf * f * 2.4 / (f + 1.4 * (0.25 + 0.75 * len(d) / ix["avg"]))
        if q & set(tokenize(t.name.replace("_", " "))):
            s *= 1.5
        if s > 0:
            scored.append((s, t))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:k]]


def categories() -> dict[str, int]:
    return dict(sorted(Counter(t.category for t in REGISTRY.values()).items()))


# ---------------------------------------------------------------- calling

MAX_RESULT = 6000


def call(name: str, args: dict | None, ctx: Context) -> dict:
    """Run a tool. Always returns {"ok": bool, "result"|"error": ..., "artifacts": [...]} — never raises."""
    t0 = time.time()
    t = REGISTRY.get(name)
    if t is None:
        near = [x.name for x in search(name, 3)]
        return {"ok": False, "error": f"no tool called '{name}'." + (f" Did you mean: {', '.join(near)}?" if near else " Use find_tools to search."), "artifacts": []}
    if t.risk not in ctx.allowed:
        return {"ok": False, "error": f"'{name}' needs permission '{t.risk}' ({RISKS[t.risk]}), which is switched off.", "artifacts": [], "blocked": t.risk}
    gaps = missing(t, ctx)
    if gaps:
        return {"ok": False, "error": f"'{name}' is not set up: needs " + "; ".join(gaps) + ".", "artifacts": [], "needs": gaps}
    args = dict(args or {})
    unknown = set(args) - {p.name for p in t.params}
    if unknown:
        return {"ok": False, "error": f"unknown argument(s) {sorted(unknown)}. Signature: {t.signature()}", "artifacts": []}
    try:
        kwargs = {}
        for p in t.params:
            v = args.get(p.name)
            if v is None and p.required:
                raise ToolError(f"missing required argument '{p.name}'. Signature: {t.signature()}")
            v = _coerce(p, v)
            if p.choices and v not in p.choices:
                raise ToolError(f"argument '{p.name}' must be one of {list(p.choices)}")
            kwargs[p.name] = v
        before = len(ctx.artifacts)
        limit = max(ctx.timeout, t.timeout)
        out = _run_with_timeout(lambda: t.fn(ctx, **kwargs), limit)
        text = out if isinstance(out, str) else json.dumps(out, default=str, ensure_ascii=False)
        if len(text) > MAX_RESULT:
            text = text[:MAX_RESULT] + f"… [truncated, {len(text)} characters]"
        return {"ok": True, "result": text, "artifacts": ctx.artifacts[before:], "seconds": round(time.time() - t0, 3)}
    except ToolError as e:
        return {"ok": False, "error": str(e), "artifacts": []}
    except TimeoutError:
        return {"ok": False, "error": f"'{name}' took longer than {limit:g}s and was stopped.", "artifacts": []}
    except Exception as e:  # a tool bug or bad input must never crash the student's session
        return {"ok": False, "error": f"{e.__class__.__name__}: {e}", "artifacts": []}


def _run_with_timeout(fn, seconds: float):
    box: dict[str, Any] = {}

    def run():
        try:
            box["v"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["e"] = e

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(seconds)
    if th.is_alive():
        raise TimeoutError()
    if "e" in box:
        raise box["e"]
    return box.get("v")


def make_context(workspace: str | Path, secrets: dict | None = None, allowed=DEFAULT_ALLOWED, timeout: float = 60.0) -> Context:
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    return Context(ws, dict(secrets or {}), tuple(allowed), timeout)


# ---------------------------------------------------------------- terse registration

def parse_sig(sig: str) -> list[Param]:
    """'x:float, n:int=2, label:str?' → parameters. '?' marks optional; '=v' gives a default."""
    out = []
    for part in [s.strip() for s in re.split(r",(?![^\[]*\])", sig) if s.strip()]:
        default, required = None, True
        if "=" in part:
            part, d = part.split("=", 1)
            default, required = d.strip(), False
        if part.endswith("?"):
            part, required = part[:-1], False
        name, _, typ = part.partition(":")
        typ = (typ or "str").strip()
        if default is not None:
            default = {"int": int, "float": float, "bool": lambda s: s.lower() in ("1", "true", "yes")}.get(typ, str)(default)
        out.append(Param(name.strip(), typ, name.strip().replace("_", " "), default, required))
    return out


def reg(name: str, desc: str, category: str, sig: str, fn: Callable, *, ctx: bool = False, **meta) -> Tool:
    """Register a plain function. With ctx=True it receives the Context first."""
    wrapped = fn if ctx else (lambda _ctx, /, **kw: fn(**kw))
    return add(Tool(name, meta.pop("title", "") or name.replace("_", " ").capitalize(), desc, category, parse_sig(sig), wrapped,
                    meta.pop("risk", "safe"), tuple(meta.pop("needs", ())), tuple(meta.pop("tags", ())), meta.pop("produces", ""), meta.pop("timeout", 0.0)))
