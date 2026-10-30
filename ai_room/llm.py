"""Model backends and the 27B parameter guard.

Supported backends:
  * ollama     – local models served by Ollama (http://localhost:11434)
  * openai     – any OpenAI-compatible server (llama.cpp, vLLM, LM Studio, ...)
  * anthropic  – Claude, usable as a *teacher* only (needs `pip install anthropic`)
  * mock       – deterministic offline backend for demos and tests
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable

import httpx

MAX_STUDENT_PARAMS_B = 27.0
# Vendors report slightly-off sizes for nominal 27B models (e.g. Gemma 3 27B
# reports "27.4B"), so anything that rounds down to 27B is accepted.
_PARAM_TOLERANCE_B = 0.99


class ModelTooLargeError(ValueError):
    pass


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLM:
    """Minimal chat interface shared by all backends."""

    name: str = "llm"

    def chat(self, messages: list[Message], temperature: float = 0.7, max_tokens: int = 2048) -> str:
        raise NotImplementedError

    def parameter_count_b(self) -> float | None:
        """Model size in billions of parameters, or None if unknown."""
        return None


# ---------------------------------------------------------------------------
# Parameter-size detection
# ---------------------------------------------------------------------------

def parse_param_size(text: str) -> float | None:
    """Extract a parameter count (in billions) from a size string or model name.

    Handles "27B", "27.4B", "gemma3:27b", "qwen2.5-14b-instruct", "8x7b" (MoE,
    counts total parameters), "30b-a3b" (MoE total/active, uses the total) and
    "350M".
    """
    if not text:
        return None
    s = text.lower()
    moe = re.search(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*b(?![a-z])", s)
    if moe:
        return int(moe.group(1)) * float(moe.group(2))
    m = re.search(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*b(?![a-z])", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*m(?![a-z])", s)
    if m:
        return float(m.group(1)) / 1000
    return None


def check_student_size(llm: LLM, declared_b: float | None = None) -> float:
    """Raise unless the student model is at most 27B parameters.

    Returns the size (in billions) that was verified.
    """
    size = llm.parameter_count_b()
    if size is None:
        size = declared_b
    if size is None:
        raise ModelTooLargeError(
            f"Could not determine the parameter count of '{llm.name}'. "
            "Use a model name that includes its size (e.g. 'qwen2.5:14b') "
            "or declare it with student_params_b."
        )
    if size > MAX_STUDENT_PARAMS_B + _PARAM_TOLERANCE_B:
        raise ModelTooLargeError(
            f"Student model '{llm.name}' has ~{size:g}B parameters; "
            f"the classroom only admits models up to {MAX_STUDENT_PARAMS_B:g}B."
        )
    return size


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class OllamaLLM(LLM):
    def __init__(self, model: str, base_url: str | None = None, timeout: float = 600):
        self.name = model
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def chat(self, messages, temperature=0.7, max_tokens=2048):
        r = self.client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [m.__dict__ for m in messages],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def parameter_count_b(self):
        try:
            r = self.client.post(f"{self.base_url}/api/show", json={"model": self.model})
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, ValueError):
            return parse_param_size(self.model)
        info = data.get("model_info") or {}
        exact = info.get("general.parameter_count")
        if isinstance(exact, (int, float)) and exact > 0:
            return exact / 1e9
        size = parse_param_size((data.get("details") or {}).get("parameter_size", ""))
        return size if size is not None else parse_param_size(self.model)


class OpenAICompatLLM(LLM):
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, timeout: float = 600):
        self.name = model
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1").rstrip("/")
        key = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self.client = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {key}"})

    def chat(self, messages, temperature=0.7, max_tokens=2048):
        r = self.client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [m.__dict__ for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

    def parameter_count_b(self):
        return parse_param_size(self.model)


class AnthropicLLM(LLM):
    """Claude as a teacher. Never allowed as the student (its size is not ≤27B)."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium"):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the optional dependency: pip install anthropic") from e
        self.name = model
        self.model = model
        self.effort = effort
        self.client = anthropic.Anthropic()

    def chat(self, messages, temperature=0.7, max_tokens=16000):
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        convo = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        kwargs = dict(
            model=self.model,
            max_tokens=max(max_tokens, 4096),
            system=system,
            messages=convo,
            output_config={"effort": self.effort},
        )
        try:
            # Server-side fallback routes refused requests to another model.
            resp = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        except TypeError:  # older SDK without the `fallbacks` keyword
            resp = self.client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            return "(The teacher declined to answer this request.)"
        return "".join(b.text for b in resp.content if b.type == "text")


@dataclass
class MockLLM(LLM):
    """Offline backend. Pass `responder` for scripted tests; otherwise a small
    built-in heuristic plays both teacher and student so the UI can be demoed
    without any model installed."""

    name: str = "mock-1b"
    responder: Callable[[list[Message]], str] | None = None
    size_b: float = 1.0
    calls: list[list[Message]] = field(default_factory=list)

    def chat(self, messages, temperature=0.7, max_tokens=2048):
        self.calls.append(messages)
        if self.responder:
            return self.responder(messages)
        return _mock_reply(messages)

    def parameter_count_b(self):
        return self.size_b


def _mock_reply(messages: list[Message]) -> str:
    prompt = messages[-1].content
    if "[[EXAM]]" in prompt:
        qs = []
        nums = re.findall(r"COUNT=(\d+)", prompt)
        n = int(nums[0]) if nums else 5
        allowed = re.search(r"Allowed question types: ([a-z, ]+)", prompt)
        kind = allowed.group(1).split(",")[0].strip() if allowed else "numeric"
        for i in range(n):
            a, b = 3 + i * 7, 4 + i * 5
            if kind == "code":
                qs.append({"type": "code", "question": "Write a Python function `add(a, b)` that returns a + b.",
                           "function_name": "add", "tests": [f"assert add({a}, {b}) == {a + b}"]})
            elif kind == "mcq":
                qs.append({"type": "mcq", "question": f"Compute {a} + {b}.",
                           "options": [f"A) {a + b - 1}", f"B) {a + b}", f"C) {a + b + 1}", f"D) {a * b}"],
                           "answer": "B"})
            elif kind == "short":
                qs.append({"type": "short", "question": f"Explain how to compute {a} + {b}.",
                           "answer": f"{a + b}", "rubric": "Correct total."})
            else:
                qs.append({"type": "numeric", "question": f"Compute {a} + {b}.", "answer": str(a + b),
                           "explanation": f"{a} + {b} = {a + b}"})
        return json.dumps({"questions": qs})
    if "[[GRADE]]" in prompt:
        return json.dumps({"score": 1.0, "feedback": "Good answer."})
    if "[[NOTES]]" in prompt:
        return "- To add two numbers, combine their values: a + b.\n- Always double-check arithmetic."
    if "[[QUESTIONS]]" in prompt:
        return "1. Why does the order of addition not matter?"
    if "[[REFLECT]]" in prompt:
        return "- Lesson: re-check each arithmetic step before answering."
    if "`add(a, b)`" in prompt:
        return "```python\ndef add(a, b):\n    return a + b\n```"
    m = re.search(r"Compute (\d+) \+ (\d+)", prompt)
    if m and "B) " in prompt:
        return "FINAL ANSWER: B"
    if m:
        total = int(m.group(1)) + int(m.group(2))
        return f"Adding the numbers gives {total}.\nFINAL ANSWER: {total}"
    if "[[LESSON]]" in prompt:
        return "Today we study addition. Addition combines quantities: 2 + 3 = 5. It is commutative: a + b = b + a."
    return "Because addition is commutative: a + b = b + a for all numbers."


def make_llm(backend: str, model: str, base_url: str | None = None, api_key: str | None = None) -> LLM:
    backend = backend.lower()
    if backend == "ollama":
        return OllamaLLM(model, base_url)
    if backend in ("openai", "openai-compatible", "vllm", "llamacpp", "lmstudio"):
        return OpenAICompatLLM(model, base_url, api_key)
    if backend in ("anthropic", "claude"):
        return AnthropicLLM(model or "claude-opus-5-5")
    if backend == "mock":
        return MockLLM(name=model or "mock-1b", size_b=parse_param_size(model or "") or 1.0)
    raise ValueError(f"Unknown backend '{backend}'")
