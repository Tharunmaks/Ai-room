"""Will a big model run on this device, and how fast can it possibly go? Measure the device, then do the arithmetic.

Generating one token means reading every weight that token uses, once. So the speed limit is
    tokens/second <= (bytes per second the device can deliver) / (bytes one token needs)
from RAM when the model fits, from storage when the weights have to be streamed. No engine can beat that limit; a good one
(llama.cpp) gets close to it. Three cases:

- Dense model: every token reads all the weights.
- Mixture-of-experts (MoE): every token reads the shared weights plus a few experts, e.g. about 3.3B of Qwen3-30B-A3B's
  30.5B. That is why MoE models run several times faster than dense models of the same total size.
- Expert streaming: if the model does not fit in RAM, the shared weights stay in RAM and the chosen experts are read from
  storage for each token, minus whatever happens to be cached. It lets the model run at all; it does not make it fast.

`plan` needs nothing but Python, so it runs on a phone. `bench` needs llama-cpp-python and a GGUF file.
"""

from __future__ import annotations

import os
import resource
import tempfile
import time
from dataclasses import dataclass

GB = 1e9
Q4_BITS = 4.9           # average bits per weight of a Q4_K_M GGUF (measured from real files: 16.7 GB for 27B, 18.6 GB for 30.5B)
SHARED_SHARE = 0.4      # assumed share of an MoE token's weights that are not experts (attention, embeddings, shared experts)


# ------------------------------------------------------------------ measuring the device
def ram_gb() -> dict:
    """Total and available RAM from /proc/meminfo (Linux and Android)."""
    out = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                out[k] = int(v.split()[0]) * 1024 / GB
    except OSError:
        pass
    return {"total": out.get("MemTotal"), "available": out.get("MemAvailable")}


def memory_bandwidth_gbps(mb: int = 256, rounds: int = 5) -> float:
    """Copy a buffer in RAM and time it (one thread, so engines using several cores may reach somewhat more)."""
    n = mb * 1024 * 1024
    src, dst = bytearray(n), bytearray(n)
    view = memoryview(dst)
    best = 0.0
    for _ in range(rounds):
        t = time.perf_counter()
        view[:] = src                      # a plain memcpy: read n bytes, write n bytes
        dt = time.perf_counter() - t
        best = max(best, 2 * n / dt / GB)
    return round(best, 1)


def storage_read_gbps(folder: str | None = None, mb: int = 1024, chunk_mb: int = 4) -> dict:
    """Write a test file, ask the system to drop it from its cache, then read it back in expert-sized chunks.

    If the cache cannot be dropped (no posix_fadvise), the result may really be RAM speed; that is reported. Drives and
    virtual disks also have caches of their own, so even then a test can read faster than a model that is many GB large.
    """
    folder = folder or tempfile.gettempdir()
    path = os.path.join(folder, f".ai_room_read_test_{os.getpid()}")
    block = os.urandom(chunk_mb * 1024 * 1024)
    try:
        with open(path, "wb") as f:
            for _ in range(max(1, mb // chunk_mb)):
                f.write(block)
            f.flush()
            os.fsync(f.fileno())
        dropped = False
        fd = os.open(path, os.O_RDONLY)
        try:
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                dropped = True
            t = time.perf_counter()
            total = 0
            while True:
                b = os.read(fd, chunk_mb * 1024 * 1024)
                if not b:
                    break
                total += len(b)
            dt = time.perf_counter() - t
        finally:
            os.close(fd)
        return {"gbps": round(total / dt / GB, 2), "cache_dropped": dropped, "folder": folder}
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def measure_device(storage_folder: str | None = None) -> dict:
    return {"ram_gb": ram_gb(), "cores": os.cpu_count(), "memory_gbps": memory_bandwidth_gbps(), "storage": storage_read_gbps(storage_folder)}


# ------------------------------------------------------------------ the arithmetic
@dataclass
class ModelShape:
    name: str
    total_b: float                  # all parameters, in billions
    active_b: float | None = None   # parameters one token uses (MoE); None means dense
    bits: float = Q4_BITS

    @property
    def moe(self) -> bool:
        return self.active_b is not None and self.active_b < self.total_b

    @property
    def file_gb(self) -> float:
        return self.total_b * self.bits / 8

    @property
    def token_gb(self) -> float:
        return (self.active_b if self.moe else self.total_b) * self.bits / 8


DEFAULT_SHAPES = [
    ModelShape("dense 1.5B", 1.5), ModelShape("dense 3B", 3), ModelShape("dense 7B", 7), ModelShape("dense 14B", 14),
    ModelShape("dense 27B", 27), ModelShape("dense 70B", 70),
    ModelShape("MoE 30B (3.3B active)", 30.5, 3.3), ModelShape("MoE 120B (5B active)", 117, 5.1), ModelShape("MoE 235B (22B active)", 235, 22),
]


def parse_shape(spec: str, bits: float = Q4_BITS) -> ModelShape:
    """'27' = dense 27B; '30.5/3.3' = MoE with 30.5B total and 3.3B active."""
    total, _, active = spec.partition("/")
    t = float(total.lower().rstrip("b"))
    a = float(active.lower().rstrip("b")) if active else None
    return ModelShape(f"MoE {t:g}B ({a:g}B active)" if a else f"dense {t:g}B", t, a, bits)


def plan_one(m: ModelShape, ram_free_gb: float, memory_gbps: float, storage_gbps: float, reserve_gb: float = 1.0) -> dict:
    """Fits in RAM? And the fastest it could go, from RAM or streamed from storage."""
    usable = max(0.0, ram_free_gb - reserve_gb)          # room for the OS, the context cache and the app itself
    row = {"model": m.name, "file_gb": round(m.file_gb, 1), "token_gb": round(m.token_gb, 2), "fits_in_ram": m.file_gb <= usable}
    if row["fits_in_ram"]:
        row["max_tok_s"] = round(memory_gbps / m.token_gb, 2)
        row["how"] = "all in RAM"
        return row
    if m.moe:
        shared = m.active_b * SHARED_SHARE * m.bits / 8               # stays in RAM
        expert_per_token = m.token_gb - shared                        # read from storage unless cached
        experts_total = m.file_gb - shared
        cached = max(0.0, min(1.0, (usable - shared) / experts_total)) if experts_total > 0 else 0.0
        row["how"] = "shared weights in RAM, experts streamed from storage"
        if shared > usable:
            row["how"] = "does not fit: even the shared weights are larger than free RAM"
            row["max_tok_s"] = 0.0
            return row
    else:
        shared, expert_per_token = 0.0, m.token_gb
        cached = max(0.0, min(1.0, usable / m.file_gb))
        row["how"] = "dense weights streamed from storage on every token"
    for label, hit in (("max_tok_s_nothing_cached", 0.0), ("max_tok_s_with_cache", cached)):
        t = shared / memory_gbps + expert_per_token * hit / memory_gbps + expert_per_token * (1 - hit) / storage_gbps
        row[label] = round(1 / t, 3) if t > 0 else None
    row["cached_share"] = round(cached, 2)
    row["max_tok_s"] = row["max_tok_s_with_cache"]
    return row


def plan(shapes: list[ModelShape], device: dict) -> list[dict]:
    free = device["ram_gb"]["available"] or device["ram_gb"]["total"] or 0.0
    return [plan_one(m, free, device["memory_gbps"], device["storage"]["gbps"]) for m in shapes]


def plan_table(rows: list[dict], target: float | None = None) -> str:
    lines = [f"{'model':<26}{'file GB':>8}{'GB/token':>9}  {'fits':<5}{'max tok/s':>10}  how"]
    for r in rows:
        tok = r.get("max_tok_s")
        extra = ""
        if not r["fits_in_ram"] and r.get("max_tok_s_nothing_cached") is not None:
            extra = f" (nothing cached: {r['max_tok_s_nothing_cached']:g}, cache holds {r['cached_share']:.0%})"
        lines.append(f"{r['model']:<26}{r['file_gb']:>8.1f}{r['token_gb']:>9.2f}  {'yes' if r['fits_in_ram'] else 'no':<5}{(f'{tok:g}' if tok is not None else 'n/a'):>10}  {r['how']}{extra}")
        if target and tok:
            lines[-1] += f"  [{target / tok:.0f}× short of {target:g} tok/s]" if tok < target else f"  [reaches {target:g} tok/s]"
    return "\n".join(lines)


# ------------------------------------------------------------------ measuring a real model (needs llama-cpp-python)
def _storage_bytes_read() -> int | None:
    """Bytes this process made the system fetch from storage, including memory-mapped weights (Linux / Android)."""
    try:
        for line in open("/proc/self/io"):
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def bench(model_path: str, threads: int = 0, gen_tokens: int = 64, n_ctx: int = 2048) -> dict:
    """Load a GGUF with memory-mapping (the weights are paged in from storage as they are used) and time it.

    Major page faults count reads from storage, so they show how much of the model was streamed rather than held in RAM.
    """
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise SystemExit(f"bench needs llama-cpp-python ({e}). Plan works without it.")
    size = os.path.getsize(model_path) / GB
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    io0 = _storage_bytes_read()
    t = time.perf_counter()
    llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=threads or os.cpu_count(), use_mmap=True, use_mlock=False, verbose=False)
    load = time.perf_counter() - t
    prompt = "Explain, step by step, why the sky looks blue during the day and red at sunset."
    t = time.perf_counter()
    out = llm.create_completion(prompt, max_tokens=gen_tokens, temperature=0.0)
    gen = time.perf_counter() - t
    t = time.perf_counter()
    out2 = llm.create_completion(prompt + " Keep it short.", max_tokens=gen_tokens, temperature=0.0)   # second pass: what stayed cached
    gen2 = time.perf_counter() - t
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    io1 = _storage_bytes_read()
    n1, n2 = out["usage"]["completion_tokens"], out2["usage"]["completion_tokens"]
    page = resource.getpagesize()
    read = (io1 - io0) if io0 is not None and io1 is not None else (r1.ru_majflt - r0.ru_majflt) * page
    return {"model": os.path.basename(model_path), "file_gb": round(size, 2), "ram_gb": ram_gb(), "load_seconds": round(load, 1),
            "first_run_tok_s": round(n1 / gen, 2), "second_run_tok_s": round(n2 / gen2, 2), "tokens": [n1, n2],
            "read_from_storage_gb": round(read / GB, 2),
            "peak_resident_gb": round(r1.ru_maxrss * 1024 / GB, 2), "sample": out2["choices"][0]["text"][:200]}
