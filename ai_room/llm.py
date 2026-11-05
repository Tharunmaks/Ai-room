"""Student-model backends and the 27B parameter guard.

Only the *student* is a language model. Supported backends:
  * ollama   – local models served by Ollama (http://localhost:11434)
  * openai   – any OpenAI-compatible server (llama.cpp, vLLM, LM Studio, ...)
  * demo     – a simulated student for trying the interface without any model
  * mock     – a tiny deterministic backend used by the tests
"""

from __future__ import annotations

import hashlib
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
    demo_oracle: bool = False

    def chat(self, messages: list[Message], temperature: float = 0.7, max_tokens: int = 2048) -> str:
        raise NotImplementedError

    def chat_with_usage(self, messages: list[Message], temperature: float = 0.7, max_tokens: int = 2048) -> tuple[str, dict]:
        """The reply plus whatever token counts / timings the backend reports (empty if it reports none)."""
        return self.chat(messages, temperature, max_tokens), {}

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
# Real backends
# ---------------------------------------------------------------------------

class OllamaLLM(LLM):
    def __init__(self, model: str, base_url: str | None = None, timeout: float = 600):
        self.name = model
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.base_url.startswith("http"):
            self.base_url = "http://" + self.base_url
        self.client = httpx.Client(timeout=timeout)

    def chat(self, messages, temperature=0.7, max_tokens=2048):
        return self.chat_with_usage(messages, temperature, max_tokens)[0]

    def chat_with_usage(self, messages, temperature=0.7, max_tokens=2048):
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
        j = r.json()
        usage = {"prompt_tokens": j.get("prompt_eval_count"), "completion_tokens": j.get("eval_count"),
                 "prompt_seconds": (j.get("prompt_eval_duration") or 0) / 1e9 or None, "completion_seconds": (j.get("eval_duration") or 0) / 1e9 or None}
        return j["message"]["content"], {k: v for k, v in usage.items() if v}

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
        return self.chat_with_usage(messages, temperature, max_tokens)[0]

    def chat_with_usage(self, messages, temperature=0.7, max_tokens=2048):
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
        j = r.json()
        u = j.get("usage") or {}
        t = j.get("timings") or {}            # llama.cpp's server adds exact timings
        usage = {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                 "prompt_seconds": (t.get("prompt_ms") or 0) / 1000 or None, "completion_seconds": (t.get("predicted_ms") or 0) / 1000 or None}
        return j["choices"][0]["message"]["content"] or "", {k: v for k, v in usage.items() if v}

    def parameter_count_b(self):
        return parse_param_size(self.model)


# ---------------------------------------------------------------------------
# Offline stand-ins
# ---------------------------------------------------------------------------

@dataclass
class MockLLM(LLM):
    """Deterministic stand-in student.

    With `responder` it plays a script (tests). Otherwise it behaves like a very
    simple student: it copies key lines into its notes and answers fill-in-the-blank
    and multiple-choice questions by looking them up in its notebook pages.
    With `demo_oracle=True` it is the *simulated student* for trying the
    interface: it is handed the answer key and gets most answers right, so it
    does not actually learn anything.
    """

    name: str = "mock-1b"
    responder: Callable[[list[Message]], str] | None = None
    size_b: float | None = 1.0
    demo_oracle: bool = False
    calls: list[list[Message]] = field(default_factory=list)

    def chat(self, messages, temperature=0.7, max_tokens=2048):
        self.calls.append(messages)
        if self.responder:
            return self.responder(messages)
        return _mock_reply(messages, self.demo_oracle)

    def parameter_count_b(self):
        return self.size_b


def _roll(prompt: str) -> float:
    return int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _notes_from_lesson(lesson: str) -> str:
    """What a diligent student copies down: key-term bullets, answers and formulas."""
    keep = []
    for ln in lesson.splitlines():
        s = ln.strip()
        if not s or s.startswith(("#", ">", "*Think", "*Before", "*…")):
            continue
        if s.startswith("- **") or "**Answer" in s or s.startswith(("Newton", "Work ", "Heat ", "Kinetic", "Weight")) or "=" in s[:80] or s.startswith("⚠️"):
            keep.append("- " + s.lstrip("- "))
    return "\n".join(keep[:60])[:3500] or "- (no key points found)"


def _mock_reply(messages: list[Message], oracle: bool) -> str:
    prompt = messages[-1].content
    if "[[NOTES]]" in prompt:
        return _notes_from_lesson(prompt)
    if "[[QUESTIONS]]" in prompt:
        terms = []
        if "Key terms to remember" in prompt:
            terms = re.findall(r"\*\*([A-Za-z][^*\n]{2,40})\*\*", prompt.split("Key terms to remember")[-1])
            terms = [t for t in terms if not t.startswith(("Answer", "Worked", "Watch"))]
        if terms:
            return f"1. What does {terms[0]} mean exactly?"
        heads = re.findall(r"^#### (.+)$", prompt, re.M)
        return f"1. How do I use {heads[0].lower()}?" if heads else "NONE"
    if "[[REFLECT]]" in prompt:
        return "- Lesson: re-read the relevant notebook page and check every step before answering."
    if "[[SUMMARY]]" in prompt:
        return _notes_from_lesson(prompt.split("All your notes from this class:")[-1])[:1500]
    if "[[ANSWER]]" in prompt:
        return _mock_answer(prompt, oracle)
    return "I'm a tiny stand-in model, so I can't say much — but I'm taking notes!"


def _mock_answer(prompt: str, oracle: bool) -> str:
    notebook = prompt.split("Relevant pages from your notebook:")[-1].split("Exam question:")[0]
    studied = re.search(r"\[\[DEMO-STUDIED\]\]\s*(\d)", prompt)
    has_notes = bool(int(studied.group(1))) if studied else "nothing relevant yet" not in notebook
    wants_code = "complete Python solution" in prompt
    wants_letter = "option letter" in prompt
    m = re.search(r"\[\[DEMO-KEY\]\]\s*(.*)\Z", prompt, re.S)
    prompt_for_roll = prompt
    if oracle and m:
        key = m.group(1).strip()
        right = _roll(prompt) < (0.97 if has_notes else 0.45)
        if wants_code:
            return f"```python\n{key if right else 'def solution(*args):\n    return None'}\n```"
        if right:
            return f"I checked my notes.\nFINAL ANSWER: {key}"
        return "FINAL ANSWER: " + ("A" if wants_letter and key != "A" else "0")
    question = prompt.split("Exam question:")[-1]
    if "_____" in question:  # fill in the blank: look the sentence up in the notebook
        parts = question.split("_____", 1)
        tail = re.sub(r"\s+", " ", parts[1].split("\n")[0]).strip().lower()[:40]
        for line in notebook.splitlines():
            flat = re.sub(r"\*\*|\s+", " ", line).lower()
            if tail and tail in flat:
                hit = re.search(r"\*\*([^*]+)\*\*", line)
                if hit:
                    answer = hit.group(1)
                    if wants_letter:
                        for opt in re.findall(r"^([A-D])\) (.+)$", question, re.M):
                            if opt[1].strip().lower() == answer.lower():
                                return f"FINAL ANSWER: {opt[0]}"
                    else:
                        return f"FINAL ANSWER: {answer}"
    return "FINAL ANSWER: A" if wants_letter else "FINAL ANSWER: 0"


def make_llm(backend: str, model: str, base_url: str | None = None, api_key: str | None = None) -> LLM:
    backend = (backend or "").lower()
    if backend == "ollama":
        return OllamaLLM(model, base_url)
    if backend in ("openai", "openai-compatible", "vllm", "llamacpp", "lmstudio"):
        return OpenAICompatLLM(model, base_url, api_key)
    if backend == "demo":
        return MockLLM(name="simulated-student", size_b=1.0, demo_oracle=True)
    if backend == "mock":
        return MockLLM(name=model or "mock-1b", size_b=parse_param_size(model or "") or 1.0)
    raise ValueError(f"Unknown student backend '{backend}'. Use ollama, openai or demo — the teacher is not an AI model.")
