"""Real fine-tuning on a CPU: the model's weights change, and the result is measured.

This is separate from the classroom, where the student never changes. Here a small open model (Hugging Face format) is
trained with LoRA on problems from the teacher's exact generators — every answer in the training data is computed by code,
so the model is never trained on a wrong answer. Before and after training it sits the same held-out exam, graded with the
classroom's deterministic grader, so you see exactly what training changed.

Why this is practical on a CPU, and its limits:
- LoRA trains ~1% of the weights; the rest stay frozen, so each step is cheap and memory stays low.
- Short sequences (problem + worked answer, usually under 160 tokens) and only the answer tokens count toward the loss.
- float32 by default; `--dtype bf16` is available but gave no speed-up on the CPU it was tested on.
- Realistic sizes: 135M–1.5B parameters in minutes to a few hours. A 7B model needs ~30 GB of RAM and days; 27B is not
  feasible on a CPU. Use the Colab/Kaggle notebooks for those.

Needs: pip install "ai-room[train]"   (torch CPU build, transformers, peft)
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .grading import grade_question
from .teacher import generators as G
from .teacher import teacher as _register_families  # noqa: F401  (importing registers every problem family)

PROMPT = "{question}{options}\nWork it out, then end with a line 'FINAL ANSWER: <answer>'."


# ------------------------------------------------------------------ data with answers computed by code
def make_problems(subjects: list[str], grades: list[int], n: int, seed: int, avoid: set[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    pools = [(s, g, f) for s in subjects for g in grades for f in G.families_for(s, g)]
    if not pools:
        raise ValueError(f"no exact problem families for subjects {subjects} in classes {grades}")
    out, seen, tries = [], set(avoid or ()), 0
    while len(out) < n and tries < n * 30:
        tries += 1
        s, g, f = rng.choice(pools)
        q = G.generate(f, s, g, rng)
        if q["type"] == "code" or q["question"] in seen:
            continue
        seen.add(q["question"])
        out.append({**q, "subject": s, "grade": g})
    return out


def prompt_of(q: dict) -> str:
    opts = ("\n" + "\n".join(q["options"])) if q.get("options") else ""
    return PROMPT.format(question=q["question"], options=opts)


def target_of(q: dict) -> str:
    why = (q.get("explanation") or "").strip()
    return (why + "\n" if why else "") + f"FINAL ANSWER: {q['answer']}"


# ------------------------------------------------------------------ the run
@dataclass
class TrainConfig:
    model: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    subjects: list[str] = field(default_factory=lambda: ["math"])
    grades: list[int] = field(default_factory=lambda: [1, 2, 3])
    examples: int = 800
    eval_examples: int = 100
    epochs: int = 2
    batch_size: int = 8
    lr: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    max_len: int = 192
    max_minutes: float = 0.0            # 0 = no time limit
    threads: int = 0                    # 0 = all cores
    dtype: str = "fp32"                 # fp32 | bf16 (try bf16 on CPUs with AMX; measure before relying on it)
    seed: int = 7
    out: str = "data/trained/model"
    skip_eval: bool = False


def cpu_has_bf16() -> bool:
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return bool(re.search(r"\b(avx512_bf16|amx_bf16)\b", flags))


def ram_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return None


def _chat(tok, prompt: str, answer: str | None = None) -> str:
    msgs = [{"role": "user", "content": prompt}]
    if getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        text = f"Question: {prompt}\nAnswer: "
    return text + (answer + (tok.eos_token or "") if answer is not None else "")


def evaluate(model, tok, problems: list[dict], log: Callable[[str], None], batch: int = 16, max_new: int = 96) -> dict:
    import torch
    model.eval()
    tok.padding_side = "left"
    right, rows = 0.0, []
    for i in range(0, len(problems), batch):
        chunk = problems[i:i + batch]
        enc = tok([_chat(tok, prompt_of(q)) for q in chunk], return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.no_grad(), _autocast(model):
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for q, g in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
            reply = tok.decode(g, skip_special_tokens=True)
            score, given = grade_question(q, reply)
            right += score
            rows.append({"question": q["question"], "expected": q["answer"], "given": given, "correct": bool(score)})
        log(f"  graded {min(i + batch, len(problems))}/{len(problems)}")
    by_family: dict[str, list[int]] = {}
    for q, r in zip(problems, rows):
        by_family.setdefault(q["family"], []).append(int(r["correct"]))
    return {"accuracy": right / max(1, len(problems)), "n": len(problems), "by_family": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_family.items())}, "rows": rows}


_AUTOCAST = {"on": False}


def _autocast(model):
    import contextlib

    import torch
    return torch.autocast("cpu", dtype=torch.bfloat16) if _AUTOCAST["on"] else contextlib.nullcontext()


def train(cfg: TrainConfig, log: Callable[[str], None] = print) -> dict:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(f"Training needs PyTorch, transformers and peft ({e}). Install them with:\n"
                         "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n  pip install transformers peft accelerate")
    t_start = time.time()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    _AUTOCAST["on"] = cfg.dtype == "bf16"  # measured: no speed-up over fp32 on a Xeon with AMX for these sizes, so fp32 is the default
    log(f"CPU threads: {torch.get_num_threads()} · math: {'bf16 autocast (native CPU support)' if _AUTOCAST['on'] else 'float32'}")

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.float32)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Loaded {cfg.model}: {n_params / 1e6:.0f}M parameters")
    free = ram_gb()
    if free and n_params * 4 * 3 / 1e9 > free:
        raise SystemExit(f"This model needs about {n_params * 12 / 1e9:.0f} GB of RAM to train on a CPU and only {free:.0f} GB is free. Pick a smaller model.")
    if n_params > 27.5e9:
        raise SystemExit("AI Room's limit is 27B parameters.")

    # data: training problems and a separate held-out exam that shares no question with them
    exam = [] if cfg.skip_eval else make_problems(cfg.subjects, cfg.grades, cfg.eval_examples, cfg.seed + 1000)
    data = make_problems(cfg.subjects, cfg.grades, cfg.examples, cfg.seed, avoid={q["question"] for q in exam})
    log(f"Training problems: {len(data)} · held-out exam: {len(exam)} (no overlap) · answers computed by code")

    report: dict = {"config": asdict(cfg), "parameters": n_params}
    if exam:
        log("Exam BEFORE training…")
        report["before"] = evaluate(model, tok, exam, log)
        log(f"  before: {report['before']['accuracy']:.1%}")

    model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"LoRA: training {trainable / 1e6:.2f}M weights ({trainable / n_params:.2%} of the model)")

    # tokenise once; loss only on the answer part
    tok.padding_side = "right"
    feats = []
    for q in data:
        p_ids = tok(_chat(tok, prompt_of(q)), add_special_tokens=False)["input_ids"]
        full = tok(_chat(tok, prompt_of(q), target_of(q)), add_special_tokens=False)["input_ids"][:cfg.max_len]
        labels = [-100] * min(len(p_ids), len(full)) + full[len(p_ids):]
        if any(l != -100 for l in labels):
            feats.append((full, labels))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=0.0)
    steps_total = cfg.epochs * math.ceil(len(feats) / cfg.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps_total // 20)) * max(0.0, 1 - s / max(1, steps_total)))
    model.train()
    step, losses, stopped_early = 0, [], False
    deadline = t_start + cfg.max_minutes * 60 if cfg.max_minutes else None
    log(f"Training: {cfg.epochs} epochs × {math.ceil(len(feats) / cfg.batch_size)} steps (batch {cfg.batch_size})")
    t_train = time.time()
    for ep in range(cfg.epochs):
        random.shuffle(feats)
        for i in range(0, len(feats), cfg.batch_size):
            chunk = feats[i:i + cfg.batch_size]
            L = max(len(f) for f, _ in chunk)
            ids = torch.tensor([f + [tok.pad_token_id] * (L - len(f)) for f, _ in chunk])
            lab = torch.tensor([l + [-100] * (L - len(l)) for _, l in chunk])
            att = torch.tensor([[1] * len(f) + [0] * (L - len(f)) for f, _ in chunk])
            with _autocast(model):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(loss.item())
            if step % 10 == 0 or step == 1 or step == steps_total:
                el = time.time() - t_train
                log(f"  step {step}/{steps_total} · loss {sum(losses[-10:]) / len(losses[-10:]):.4f} · {el / step:.2f}s/step · ~{el / step * (steps_total - step) / 60:.1f} min left")
            if deadline and time.time() > deadline:
                stopped_early = True
                break
        if stopped_early:
            log("Time limit reached — stopping training early.")
            break
    w = max(1, len(losses) // 10)
    report["train"] = {"steps": step, "first_loss": round(sum(losses[:w]) / w, 4), "last_loss": round(sum(losses[-w:]) / w, 4),
                       "seconds": round(time.time() - t_train, 1), "stopped_early": stopped_early, "trainable_weights": trainable}

    if exam:
        log("Exam AFTER training (same questions)…")
        report["after"] = evaluate(model, tok, exam, log)
        log(f"  after: {report['after']['accuracy']:.1%}")

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "lora_adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tok.save_pretrained(out / "merged")
    report["saved"] = {"adapter": str(out / "lora_adapter"), "merged_model": str(out / "merged")}
    report["seconds_total"] = round(time.time() - t_start, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    log(f"Saved the trained model to {out / 'merged'} (and the small LoRA adapter to {out / 'lora_adapter'}).")
    return report


def ask(model_dir: str, question: str, max_new: int = 128) -> str:
    """Ask a saved (merged) model a question."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    enc = tok(_chat(tok, PROMPT.format(question=question, options="")), return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_grades(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return sorted(set(g for g in out if 1 <= g <= 12))
