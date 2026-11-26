"""Learn pictures from a zip of folders: one folder per category (cats/…, dogs/…).

Every picture is turned into 936 numbers that a model can learn from, in four groups:
  shape        — a 24×24 greyscale thumbnail (576 numbers),
  colour layout — an 8×8 colour thumbnail (192),
  edges        — which way edges point in each of 4×4 regions (9 directions each: 144),
  colour mix   — how much of each shade of red, green and blue the picture has (24).
The same learners as for tables then learn from those numbers, starting untrained. This is quick and works well
for pictures where shape and colour tell the categories apart (products, plants, symbols, simple scenes, scans);
it is not a deep network and will not match one on hard photographs.
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from .data import IMAGE_EXT, DataError, Table, ext_of

SIZE = 24
GROUPS = {"shape": SIZE * SIZE, "colour layout": 8 * 8 * 3, "edges": 4 * 4 * 9, "colour mix": 3 * 8}
FEATURES = sum(GROUPS.values())


def feature_names() -> list[str]:
    return [f"{g} {i + 1}" for g, k in GROUPS.items() for i in range(k)]


def groups() -> dict[str, list[str]]:
    names, out, i = feature_names(), {}, 0
    for g, k in GROUPS.items():
        out[g] = names[i:i + k]
        i += k
    return out


def _pil():
    try:
        from PIL import Image
    except ImportError as e:
        raise DataError("Learning pictures needs Pillow: pip install pillow") from e
    return Image


def open_image(raw: bytes):
    Image = _pil()
    try:
        img = Image.open(io.BytesIO(raw))
        img.draft("RGB", (SIZE * 4, SIZE * 4))         # JPEGs decode straight at a small size: much faster
        return img.convert("RGB")
    except Exception as e:  # Pillow raises many kinds of errors for broken files
        raise DataError(f"Could not read that picture: {e}")


def features(img) -> np.ndarray:
    """The 936 numbers for one picture (PIL image, any size)."""
    Image = _pil()
    small = np.asarray(img.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0     # (24, 24, 3)
    grey = small @ np.array([0.299, 0.587, 0.114], np.float32)
    layout = np.asarray(img.resize((8, 8), Image.BOX), dtype=np.float32).reshape(-1) / 255.0
    gx = np.zeros_like(grey); gy = np.zeros_like(grey)
    gx[:, 1:-1] = grey[:, 2:] - grey[:, :-2]
    gy[1:-1, :] = grey[2:, :] - grey[:-2, :]
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * 9
    b = np.minimum(ang.astype(np.int64), 8)
    cell = (np.arange(SIZE) // (SIZE // 4))
    idx = (cell[:, None] * 4 + cell[None, :]) * 9 + b
    edges = np.bincount(idx.ravel(), weights=mag.ravel(), minlength=144).astype(np.float32)
    edges /= np.linalg.norm(edges) + 1e-6
    q = np.minimum((small * 8).astype(np.int64), 7)
    mix = np.concatenate([np.bincount(q[..., c].ravel(), minlength=8) for c in range(3)]).astype(np.float32) / (SIZE * SIZE)
    return np.concatenate([grey.reshape(-1), layout, edges, mix])


def _members(z: zipfile.ZipFile) -> list[tuple[str, str]]:
    """(label, member name) for every picture; the label is the folder it is in (a shared top folder is skipped)."""
    files = [n for n in z.namelist() if not n.endswith("/") and ext_of(n) in IMAGE_EXT and "__MACOSX" not in n
             and not PurePosixPath(n).name.startswith(".")]
    parts = [PurePosixPath(n).parts for n in files]
    if not parts:
        return []
    depth = 0                                  # skip folders every file shares (e.g. 'photos/cats/1.jpg' → 'cats')
    while all(len(p) > depth + 2 for p in parts) and len({p[depth] for p in parts}) == 1:
        depth += 1
    out = []
    for n, p in zip(files, parts):
        out.append((p[depth] if len(p) > depth + 1 else "(no folder)", n))
    return out


def describe_zip(path: Path) -> dict:
    """Categories and counts, without opening the pictures."""
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise DataError("That .zip file is damaged or not really a zip.")
    members = _members(z)
    counts: dict[str, int] = {}
    for label, _ in members:
        counts[label] = counts.get(label, 0) + 1
    if len(counts) < 2:
        raise DataError("Put the pictures in one folder per category (for example cats/ and dogs/) — this zip has "
                        + ("no pictures." if not counts else f"only one: {next(iter(counts))}."))
    return {"classes": dict(sorted(counts.items())), "images": len(members)}


def load_image_zip(path: Path, deadline: float | None = None, say=None, limit: int | None = None) -> Table:
    """A table: 936 number columns (the picture's features) and 'category'. Stops reading if time runs short."""
    say = say or (lambda t: None)
    z = zipfile.ZipFile(path)
    members = _members(z)
    if limit and len(members) > limit:
        rng = np.random.default_rng(0)
        members = [members[i] for i in sorted(rng.choice(len(members), limit, replace=False))]
    order = np.random.default_rng(1).permutation(len(members))   # if time runs out, what was read is still a fair mix
    feats, labels, bad = [], [], 0
    t0 = time.perf_counter()
    for k, i in enumerate(order):
        label, name = members[i]
        try:
            feats.append(features(open_image(z.read(name))))
            labels.append(label)
        except DataError:
            bad += 1
        if deadline is not None and k % 64 == 0 and time.perf_counter() > deadline:
            say(f"Out of time for reading pictures: using the {len(labels):,} read so far (of {len(members):,}).")
            break
    if bad:
        say(f"Skipped {bad} file{'s' * (bad > 1)} that could not be read as pictures.")
    if len(labels) < 10:
        raise DataError("Fewer than 10 pictures could be read; at least 10 are needed (more is much better).")
    X = np.stack(feats).astype(np.float64)
    say(f"Read {len(labels):,} pictures in {time.perf_counter() - t0:.1f} s and turned each into {FEATURES} numbers (shape, colour layout, edges, colour mix).")
    lab = np.empty(len(labels), dtype=object)
    lab[:] = labels
    return Table(feature_names() + ["category"], [X[:, j] for j in range(FEATURES)] + [lab], len(labels))


def contact_sheet(path: Path, per_class: int = 6, max_classes: int = 8, cell: int = 72) -> bytes:
    """A PNG grid with a few example pictures of each category (for the dataset preview)."""
    Image = _pil()
    from PIL import ImageDraw
    z = zipfile.ZipFile(path)
    by: dict[str, list[str]] = {}
    for label, name in _members(z):
        by.setdefault(label, []).append(name)
    classes = sorted(by)[:max_classes]
    sheet = Image.new("RGB", (cell * per_class, cell * len(classes)), (24, 28, 48))
    draw = ImageDraw.Draw(sheet)
    for r, c in enumerate(classes):
        names = by[c]
        step = max(1, len(names) // per_class)
        for k, name in enumerate(names[::step][:per_class]):
            try:
                img = open_image(z.read(name))
            except DataError:
                continue
            img.thumbnail((cell - 4, cell - 4))
            sheet.paste(img, (k * cell + (cell - img.width) // 2, r * cell + (cell - img.height) // 2))
        draw.rectangle([0, r * cell + cell - 16, min(cell * per_class, 9 * len(c) + 10), r * cell + cell], fill=(0, 0, 0))
        draw.text((4, r * cell + cell - 15), c[:40], fill=(255, 255, 255))
    buf = io.BytesIO()
    sheet.save(buf, "PNG")
    return buf.getvalue()


def image_table(raws: list[bytes]) -> Table:
    """Feature columns for uploaded pictures, ready for Trained.predict."""
    X = np.stack([features(open_image(r)) for r in raws]).astype(np.float64)
    return Table(feature_names(), [X[:, j] for j in range(FEATURES)], len(raws))
