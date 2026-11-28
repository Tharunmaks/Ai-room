"""Learn sounds and videos from a zip of folders: one folder per category (barks/…, meows/…).

Sounds (WAV with the standard library; MP3, OGG, FLAC, M4A, AAC, OPUS, WMA… through ffmpeg when it is installed)
become 212 numbers each: the loudness at 40 pitch bands (a mel spectrogram: mean, spread, change, peak), 13 timbre
coefficients (MFCC: mean and spread), brightness, noisiness, zero crossings and loudness levels, and how the
loudness rises and falls over the clip.

Videos (animated GIF / WebP / PNG with Pillow; MP4, MOV, WebM, MKV, AVI, M4V, MPEG through ffmpeg) become 1,172
numbers each: what up to 8 evenly spaced frames look like (the picture features, averaged), how and where things
move between frames, and the sound track (or zeros when there is none).

Like pictures, this is fast and works well when sound or look and motion tell the categories apart; it is not a
deep network.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import time
import wave
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from .data import IMAGE_EXT, DataError, Table, ext_of
from . import images

AUDIO_EXT = {".wav", ".wave", ".mp3", ".ogg", ".oga", ".flac", ".m4a", ".aac", ".opus", ".wma", ".aif", ".aiff", ".weba"}
VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".3gp", ".gif"}
NATIVE_AUDIO = {".wav", ".wave"}
NATIVE_VIDEO = {".gif"}
SR = 16000
MAX_SECONDS = 30.0             # of each sound (and of a video's sound track)
VIDEO_SECONDS = 120.0          # of each video, for the frames
VIDEO_FPS = 4                  # frames looked at per second, before 8 evenly spaced ones are kept
N_MELS, N_MFCC, N_ENV = 40, 13, 10
AUDIO_GROUPS = {"tone (pitch bands)": N_MELS * 4, "timbre": N_MFCC * 2, "loudness & noisiness": 16, "changes over time": N_ENV}
AUDIO_FEATURES = sum(AUDIO_GROUPS.values())                                    # 212
FRAMES = 8
MOTION = 4 * 4 + 3 + 4
VIDEO_GROUPS = {"look (frames)": images.FEATURES, "motion": MOTION, "sound": AUDIO_FEATURES + 1}
VIDEO_FEATURES = sum(VIDEO_GROUPS.values())                                    # 1,172


def ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _need_ffmpeg(name: str):
    raise DataError(f"Reading {ext_of(name) or 'this file'} needs ffmpeg, which is not installed. Install it (apt install ffmpeg, brew install ffmpeg, "
                    "or ffmpeg.org) — or use WAV for sounds and animated GIF for videos, which work without it.")


# ------------------------------------------------------------------ decoding
def decode_audio(raw: bytes, name: str, seconds: float = MAX_SECONDS) -> np.ndarray:
    """Mono float32 samples at 16 kHz (the first `seconds`)."""
    if ext_of(name) in NATIVE_AUDIO:
        try:
            with wave.open(io.BytesIO(raw)) as w:
                ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
                frames = w.readframes(min(w.getnframes(), int(rate * seconds)))
        except (wave.Error, EOFError) as e:
            raise DataError(f"Could not read that WAV file: {e}")
        if width == 1:
            x = (np.frombuffer(frames, np.uint8).astype(np.float32) - 128) / 128
        elif width == 2:
            x = np.frombuffer(frames, "<i2").astype(np.float32) / 32768
        elif width == 3:
            b = np.frombuffer(frames, np.uint8).reshape(-1, 3)
            x = ((b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)) << 8 >> 8).astype(np.float32) / 8388608
        else:
            x = np.frombuffer(frames, "<i4").astype(np.float32) / 2147483648
        x = x.reshape(-1, ch).mean(1) if ch > 1 else x
        return _resample(x, rate)
    if not ffmpeg():
        _need_ffmpeg(name)
    return _ffmpeg_audio(raw, name, seconds)


def _resample(x: np.ndarray, rate: int) -> np.ndarray:
    if rate == SR or not len(x):
        return x.astype(np.float32)
    t = np.arange(int(len(x) * SR / rate)) * (rate / SR)
    return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def _with_file(raw: bytes, name: str, fn):
    """ffmpeg reads some formats only from a seekable file (MP4/M4A keep their index at the end)."""
    import os
    fd, path = tempfile.mkstemp(suffix=ext_of(name) or ".bin")     # closed before ffmpeg opens it (needed on Windows)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        return fn(path)
    finally:
        Path(path).unlink(missing_ok=True)


def _ffmpeg_audio(raw: bytes, name: str, seconds: float, start: float | None = None) -> np.ndarray:
    def run(path):
        cmd = [ffmpeg(), "-nostdin", "-loglevel", "error"] + (["-ss", f"{start:.3f}"] if start else []) + \
              ["-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-t", f"{seconds:.3f}", "-f", "f32le", "pipe:1"]
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode != 0 and not r.stdout:
            msg = r.stderr.decode("utf-8", "replace").strip().splitlines()
            raise DataError(f"ffmpeg could not read {name}: {msg[-1] if msg else 'unknown error'}")
        return np.frombuffer(r.stdout, np.float32).copy()
    return _with_file(raw, name, run)


def decode_video(raw: bytes, name: str, frames: int = FRAMES, size: int = 64) -> tuple[list, np.ndarray | None]:
    """(up to `frames` evenly spaced PIL frames, sound samples or None)."""
    from PIL import Image, ImageSequence
    if ext_of(name) in NATIVE_VIDEO:
        try:
            img = Image.open(io.BytesIO(raw))
            n = getattr(img, "n_frames", 1)
            pick = set(np.linspace(0, n - 1, min(frames, n)).round().astype(int).tolist())
            out = []
            for i, fr in enumerate(ImageSequence.Iterator(img)):
                if i in pick:
                    out.append(fr.convert("RGB").resize((size, size)))
                if i >= max(pick):
                    break
            return out, None
        except Exception as e:
            raise DataError(f"Could not read {name} as an animation: {e}")
    if not ffmpeg():
        _need_ffmpeg(name)

    def run(path):
        # one ffmpeg run: small frames at a few per second (to the first VIDEO_SECONDS) and the sound track, if any,
        # each to its own file; then 8 evenly spaced frames are kept
        vid, aud = path + ".rgb", path + ".pcm"
        # small frames do not need the decoder's finishing touches: skip the deblocking filter and frames that no
        # other frame depends on (about a third faster, and invisible at 64×64)
        cmd = [ffmpeg(), "-nostdin", "-loglevel", "error", "-y", "-skip_loop_filter", "all", "-skip_frame", "noref",
               "-t", f"{VIDEO_SECONDS:g}", "-i", path,
               "-map", "0:v:0", "-vf", f"fps={VIDEO_FPS:g},scale={size}:{size}", "-f", "rawvideo", "-pix_fmt", "rgb24", vid]
        sound_out = ["-map", "0:a:0?", "-ac", "1", "-ar", str(SR), "-t", f"{MAX_SECONDS:g}", "-f", "f32le", aud]
        try:
            r = subprocess.run(cmd + sound_out, capture_output=True, timeout=300)
            if r.returncode and b"does not contain any stream" in r.stderr:          # no sound track: frames only
                r = subprocess.run(cmd, capture_output=True, timeout=300)
            data = np.fromfile(vid, np.uint8) if Path(vid).exists() else np.zeros(0, np.uint8)
            sound = np.fromfile(aud, np.float32) if Path(aud).exists() else np.zeros(0, np.float32)
        finally:
            for f in (vid, aud):
                Path(f).unlink(missing_ok=True)
        k = len(data) // (size * size * 3)
        if not k:
            msg = r.stderr.decode("utf-8", "replace").strip().splitlines()
            raise DataError(f"ffmpeg could not read {name}: {msg[-1] if msg else 'no video frames'}")
        allf = data[: k * size * size * 3].reshape(k, size, size, 3)
        pick = np.unique(np.linspace(0, k - 1, min(frames, k)).round().astype(int))
        frs = [Image.fromarray(allf[i]) for i in pick]
        sound = sound if len(sound) else None
        return frs, sound
    return _with_file(raw, name, run)


# ------------------------------------------------------------------ features
_MEL = None


def _mel_bank(n_fft: int = 512) -> np.ndarray:
    global _MEL
    if _MEL is None:
        mel = lambda f: 2595 * np.log10(1 + f / 700)
        inv = lambda m: 700 * (10 ** (m / 2595) - 1)
        pts = inv(np.linspace(mel(40), mel(SR / 2), N_MELS + 2))
        bins = np.floor((n_fft + 1) * pts / SR).astype(int)
        fb = np.zeros((N_MELS, n_fft // 2 + 1), np.float32)
        for i in range(N_MELS):
            a, b, c = bins[i], max(bins[i + 1], bins[i] + 1), max(bins[i + 2], bins[i + 1] + 2)
            fb[i, a:b] = (np.arange(a, b) - a) / max(b - a, 1)
            fb[i, b:c] = (c - np.arange(b, c)) / max(c - b, 1)
        _MEL = fb
    return _MEL


def audio_features(x: np.ndarray) -> np.ndarray:
    """212 numbers describing a sound (mono, 16 kHz)."""
    if len(x) < 400:
        x = np.pad(x, (0, 400 - len(x)))
    x = x - x.mean()
    n_fft, hop = 512, 160
    nfr = 1 + (len(x) - 400) // hop
    idx = np.arange(400)[None, :] + hop * np.arange(nfr)[:, None]
    frames = x[idx] * np.hanning(400).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames, n_fft)) ** 2                              # (frames, 257)
    mel = np.log(spec @ _mel_bank().T + 1e-8)                                   # (frames, 40)
    d = np.diff(mel, axis=0) if nfr > 1 else np.zeros_like(mel)
    tone = np.concatenate([mel.mean(0), mel.std(0), np.abs(d).mean(0), mel.max(0)])
    k = np.arange(N_MELS)
    dct = np.cos(np.pi / N_MELS * (k[None, :] + 0.5) * np.arange(N_MFCC)[:, None])
    mfcc = mel @ dct.T
    timbre = np.concatenate([mfcc.mean(0), mfcc.std(0)])
    freqs = np.linspace(0, SR / 2, spec.shape[1])
    p = spec / (spec.sum(1, keepdims=True) + 1e-12)
    centroid = (p * freqs).sum(1)
    bandwidth = np.sqrt((p * (freqs - centroid[:, None]) ** 2).sum(1))
    rolloff = freqs[np.minimum((np.cumsum(p, 1) < 0.85).sum(1), len(freqs) - 1)]
    flat = np.exp(np.log(spec + 1e-12).mean(1)) / (spec.mean(1) + 1e-12)
    zcr = (np.abs(np.diff(np.sign(frames), axis=1)) > 0).mean(1)
    rms = np.sqrt((frames ** 2).mean(1)) + 1e-9
    db = 20 * np.log10(rms)
    loud = np.array([centroid.mean() / 1000, centroid.std() / 1000, bandwidth.mean() / 1000, bandwidth.std() / 1000,
                     rolloff.mean() / 1000, rolloff.std() / 1000, flat.mean(), flat.std(), zcr.mean(), zcr.std(),
                     db.mean() / 20, db.std() / 20, *np.percentile(db, [10, 50, 90]) / 20, np.log(len(x) / SR + 1e-3)])
    env = np.array([r.mean() for r in np.array_split(rms, N_ENV)]) if nfr >= N_ENV else np.resize(rms, N_ENV)
    env = env / (env.max() + 1e-9)
    out = np.concatenate([tone, timbre, loud, env]).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def video_features(frames: list, sound: np.ndarray | None) -> np.ndarray:
    """1,172 numbers describing a clip: look (picture features averaged over frames), motion, sound."""
    look = np.mean([images.features(f) for f in frames], 0)
    g = np.stack([np.asarray(f.convert("L").resize((24, 24)), np.float32) / 255 for f in frames])
    if len(g) > 1:
        diff = np.abs(np.diff(g, axis=0))                                      # (t-1, 24, 24)
        regions = diff.reshape(len(diff), 4, 6, 4, 6).mean((2, 4)).mean(0).ravel()   # where things move
        per = diff.mean((1, 2))
        ys, xs = np.mgrid[0:24, 0:24] / 23.0
        w = diff.sum((1, 2)) + 1e-9
        cy, cx = (diff * ys).sum((1, 2)) / w, (diff * xs).sum((1, 2)) / w      # where the change is, frame to frame
        vx, vy = np.diff(cx) if len(cx) > 1 else np.zeros(1), np.diff(cy) if len(cy) > 1 else np.zeros(1)
        motion = np.concatenate([regions * 10, [per.mean() * 10, per.std() * 10, per.max() * 10], [vx.mean() * 10, vx.std() * 10, vy.mean() * 10, vy.std() * 10]])
    else:
        motion = np.zeros(MOTION)
    snd = np.concatenate([audio_features(sound), [1.0]]) if sound is not None and len(sound) > 800 else np.zeros(AUDIO_FEATURES + 1)
    return np.concatenate([look, motion, snd]).astype(np.float32)


def features_of(kind: str, raw: bytes, name: str) -> np.ndarray:
    if kind == "audio":
        return audio_features(decode_audio(raw, name))
    if kind == "video":
        return video_features(*decode_video(raw, name))
    return images.features(images.open_image(raw))


def names_for(kind: str) -> list[str]:
    groups = {"audio": AUDIO_GROUPS, "video": VIDEO_GROUPS, "image": images.GROUPS}[kind]
    return [f"{g} {i + 1}" for g, k in groups.items() for i in range(k)]


def groups_for(kind: str) -> dict[str, list[str]]:
    groups = {"audio": AUDIO_GROUPS, "video": VIDEO_GROUPS, "image": images.GROUPS}[kind]
    names, out, i = names_for(kind), {}, 0
    for g, k in groups.items():
        out[g] = names[i:i + k]
        i += k
    return out


def exts_for(kind: str) -> set[str]:
    return {"audio": AUDIO_EXT, "video": VIDEO_EXT, "image": IMAGE_EXT}[kind]


# ------------------------------------------------------------------ zips of folders
KIND_OF = {"image": "images", "audio": "sounds", "video": "videos"}       # dataset kind for each media kind
MEDIA_OF = {v: k for k, v in KIND_OF.items()}


def zip_kind(names: list[str], read=None) -> str | None:
    """'image', 'audio' or 'video' if most files in the zip are of that kind. A zip of only GIFs is video when
    they move (`read(name)` gives a member's bytes, to look)."""
    files = [n for n in names if not n.endswith("/") and "__MACOSX" not in n and not PurePosixPath(n).name.startswith(".")]
    if not files:
        return None
    a = sum(ext_of(n) in AUDIO_EXT for n in files)
    v = sum(ext_of(n) in VIDEO_EXT - {".gif"} for n in files)
    gifs = [n for n in files if ext_of(n) == ".gif"]
    img = sum(ext_of(n) in IMAGE_EXT - {".gif"} for n in files)
    half = max(2, len(files) // 2)
    if a >= half:
        return "audio"
    if v and v + len(gifs) >= half:
        return "video"
    if gifs and not img and len(gifs) >= half and read and sum(_animated(read(n)) for n in gifs[:5]) >= min(3, len(gifs)):
        return "video"
    if img + len(gifs) >= half:
        return "image"
    return None


def _animated(raw: bytes) -> bool:
    from PIL import Image
    try:
        return getattr(Image.open(io.BytesIO(raw)), "n_frames", 1) > 1
    except Exception:
        return False


def sniff_zip(z: zipfile.ZipFile) -> str | None:
    """'images', 'sounds', 'videos' or None for an open zip."""
    k = zip_kind(z.namelist(), z.read)
    return KIND_OF.get(k) if k else None


def members(z: zipfile.ZipFile, kind: str) -> list[tuple[str, str]]:
    """(category, member) for every file of this kind; the category is its folder (a shared top folder is skipped)."""
    exts = exts_for(kind)
    files = [n for n in z.namelist() if not n.endswith("/") and ext_of(n) in exts and "__MACOSX" not in n
             and not PurePosixPath(n).name.startswith(".")]
    parts = [PurePosixPath(n).parts for n in files]
    if not parts:
        return []
    depth = 0
    while all(len(p) > depth + 2 for p in parts) and len({p[depth] for p in parts}) == 1:
        depth += 1
    return [(p[depth] if len(p) > depth + 1 else "(no folder)", n) for n, p in zip(files, parts)]


NOUN = {"image": "pictures", "audio": "sounds", "video": "videos"}


def describe(path: Path, kind: str) -> dict:
    z = zipfile.ZipFile(path)
    counts: dict[str, int] = {}
    for label, _ in members(z, kind):
        counts[label] = counts.get(label, 0) + 1
    if len(counts) < 2:
        raise DataError(f"Put the {NOUN[kind]} in one folder per category (for example barks/ and meows/) — this zip has "
                        + ("none." if not counts else f"only one: {next(iter(counts))}."))
    needs = sorted({ext_of(n) for _, n in members(z, kind)} - (NATIVE_AUDIO if kind == "audio" else NATIVE_VIDEO if kind == "video" else IMAGE_EXT))
    if kind in ("audio", "video") and needs and not ffmpeg():
        _need_ffmpeg(needs[0])
    return {"classes": dict(sorted(counts.items())), "images" if kind == "image" else "items": sum(counts.values()), "media": kind}


def load_zip(path: Path, kind: str, deadline: float | None = None, say=None, limit: int | None = None) -> Table:
    """A table with one row per file: its features and 'category'. Stops reading if time runs short, keeping a
    fair random mix; long jobs are shared out over the cores."""
    from .parallel import can_fork, fork_map
    say = say or (lambda t: None)
    z = zipfile.ZipFile(path)
    items = members(z, kind)
    rng = np.random.default_rng(1)
    if limit and len(items) > limit:
        items = [items[i] for i in sorted(rng.choice(len(items), limit, replace=False))]
    order = [items[i] for i in rng.permutation(len(items))]       # if time runs out, what was read is a fair mix
    t0 = time.perf_counter()
    width = {"audio": AUDIO_FEATURES, "video": VIDEO_FEATURES, "image": images.FEATURES}[kind]

    def work(lo, hi):
        zz = zipfile.ZipFile(path)
        feats, labels, bad = [], [], 0
        for k, (label, name) in enumerate(order[lo:hi]):
            try:
                feats.append(features_of(kind, zz.read(name), name))
                labels.append(label)
            except DataError:
                bad += 1
            if deadline is not None and k % 16 == 0 and time.perf_counter() > deadline:
                break
        return (np.stack(feats) if feats else np.zeros((0, width), np.float32)), labels, bad

    parts = max(1, min(8, (__import__("os").cpu_count() or 1))) if len(order) >= 64 else 1
    bounds = list(zip(*(lambda b: (b[:-1], b[1:]))(np.linspace(0, len(order), parts + 1).astype(int))))
    if parts == 1:
        results = [work(0, len(order))]
    elif can_fork():
        results = fork_map(work, bounds)
    else:                          # Windows and macOS: threads (decoding and the FFTs let other threads run meanwhile)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(parts) as ex:
            results = list(ex.map(lambda b: work(*b), bounds))
    X = np.concatenate([r[0] for r in results]).astype(np.float64)
    labels = [l for r in results for l in r[1]]
    bad = sum(r[2] for r in results)
    if len(labels) < len(order) - bad:
        say(f"Out of time for reading {NOUN[kind]}: using the {len(labels):,} read so far (of {len(order):,}).")
    if bad:
        say(f"Skipped {bad} file{'s' * (bad > 1)} that could not be read.")
    if len(labels) < 10:
        raise DataError(f"Fewer than 10 {NOUN[kind]} could be read; at least 10 are needed (more is much better).")
    say(f"Read {len(labels):,} {NOUN[kind]} in {time.perf_counter() - t0:.1f} s and turned each into {width:,} numbers "
        f"({', '.join({'audio': AUDIO_GROUPS, 'video': VIDEO_GROUPS, 'image': images.GROUPS}[kind])}).")
    lab = np.empty(len(labels), dtype=object)
    lab[:] = labels
    return Table(names_for(kind) + ["category"], [X[:, j] for j in range(width)] + [lab], len(labels))


def media_table(kind: str, files: list[tuple[str, bytes]]) -> Table:
    """Feature columns for uploaded files (name, bytes), ready for Trained.predict."""
    X = np.stack([features_of(kind, raw, name) for name, raw in files]).astype(np.float64)
    return Table(names_for(kind), [X[:, j] for j in range(X.shape[1])], len(files))


# ------------------------------------------------------------------ previews
def preview_sheet(path: Path, kind: str, per_class: int = 6, max_classes: int = 8, cell: int = 72) -> bytes:
    """PNG: a few examples per category — frames for videos, spectrograms for sounds, the pictures themselves."""
    if kind == "image":
        return images.contact_sheet(path, per_class, max_classes, cell)
    from PIL import Image, ImageDraw
    z = zipfile.ZipFile(path)
    by: dict[str, list[str]] = {}
    for label, name in members(z, kind):
        by.setdefault(label, []).append(name)
    classes = sorted(by)[:max_classes]
    w = cell * 2 if kind == "audio" else cell
    cols = per_class if kind == "video" else max(1, per_class // 2)
    sheet = Image.new("RGB", (w * cols, cell * len(classes)), (24, 28, 48))
    draw = ImageDraw.Draw(sheet)
    for r, c in enumerate(classes):
        names = by[c]
        step = max(1, len(names) // cols)
        for k, name in enumerate(names[::step][:cols]):
            try:
                tile = _thumb(kind, z.read(name), name, w - 4, cell - 4)
            except (DataError, subprocess.TimeoutExpired):
                continue
            sheet.paste(tile, (k * w + 2, r * cell + 2))
        draw.rectangle([0, r * cell + cell - 16, min(w * cols, 9 * len(c) + 10), r * cell + cell], fill=(0, 0, 0))
        draw.text((4, r * cell + cell - 15), c[:40], fill=(255, 255, 255))
    buf = io.BytesIO()
    sheet.save(buf, "PNG")
    return buf.getvalue()


def _thumb(kind: str, raw: bytes, name: str, w: int, h: int):
    from PIL import Image
    if kind == "video":
        frames, _ = decode_video(raw, name, frames=1, size=max(w, h))
        return frames[0].resize((w, h))
    x = decode_audio(raw, name, seconds=5)
    nfr = max(1, 1 + (len(x) - 400) // 160)
    if len(x) < 400:
        x = np.pad(x, (0, 400 - len(x)))
    idx = np.arange(400)[None, :] + 160 * np.arange(nfr)[:, None]
    mel = np.log(np.abs(np.fft.rfft(x[idx] * np.hanning(400), 512)) ** 2 @ _mel_bank().T + 1e-8)
    m = (mel - mel.min()) / (np.ptp(mel) + 1e-9)
    rgb = np.stack([np.clip(m * 1.6 - 0.4, 0, 1), np.clip(m * 1.2 - 0.1, 0, 1), np.clip(0.35 + m * 0.65, 0, 1)], -1)
    return Image.fromarray((rgb.transpose(1, 0, 2)[::-1] * 255).astype(np.uint8)).resize((w, h))
