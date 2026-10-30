"""Pure-Python 3D: mesh primitives, transforms, OBJ/STL/PLY/glTF export, and a small software renderer (SVG / PNG / GIF)."""

from __future__ import annotations

import base64
import json
import math
import random
import struct

from .registry import Context, ToolError, reg

Vec = tuple[float, float, float]


class Mesh:
    def __init__(self, v: list[Vec] | None = None, f: list[tuple[int, ...]] | None = None, color: str = "#8aa4ff"):
        self.v, self.f, self.color = v or [], f or [], color

    def copy(self) -> "Mesh":
        return Mesh(list(self.v), list(self.f), self.color)

    def add(self, other: "Mesh") -> "Mesh":
        o = len(self.v)
        self.v += other.v
        self.f += [tuple(i + o for i in face) for face in other.f]
        return self

    def tris(self):
        for face in self.f:
            for i in range(1, len(face) - 1):
                yield face[0], face[i], face[i + 1]

    def transform(self, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)) -> "Mesh":
        sx, sy, sz = (scale if hasattr(scale, "__len__") else (scale,) * 3)
        cx, sx_ = math.cos(rot[0]), math.sin(rot[0])
        cy, sy_ = math.cos(rot[1]), math.sin(rot[1])
        cz, sz_ = math.cos(rot[2]), math.sin(rot[2])
        out = []
        for x, y, z in self.v:
            x, y, z = x * sx, y * sy, z * sz
            y, z = y * cx - z * sx_, y * sx_ + z * cx          # X
            x, z = x * cy + z * sy_, -x * sy_ + z * cy         # Y
            x, y = x * cz - y * sz_, x * sz_ + y * cz          # Z
            out.append((x + loc[0], y + loc[1], z + loc[2]))
        return Mesh(out, list(self.f), self.color)


def _sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def _cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def _dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def _norm(a):
    l = math.sqrt(_dot(a, a)) or 1
    return (a[0] / l, a[1] / l, a[2] / l)


# ------------------------------------------------------------------ primitives
def cube(s=2.0): h = s / 2; return Mesh([(x, y, z) for x in (-h, h) for y in (-h, h) for z in (-h, h)], [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
def box(w=2.0, d=2.0, h=2.0): return cube(1).transform(scale=(w, d, h))


def plane(w=2.0, d=2.0, sx=1, sy=1):
    v = [(-w / 2 + w * i / sx, -d / 2 + d * j / sy, 0.0) for j in range(sy + 1) for i in range(sx + 1)]
    f = [(j * (sx + 1) + i, j * (sx + 1) + i + 1, (j + 1) * (sx + 1) + i + 1, (j + 1) * (sx + 1) + i) for j in range(sy) for i in range(sx)]
    return Mesh(v, f)


def uv_sphere(r=1.0, seg=24, rings=12):
    v = [(0, 0, r)] + [(r * math.sin(math.pi * i / rings) * math.cos(2 * math.pi * j / seg), r * math.sin(math.pi * i / rings) * math.sin(2 * math.pi * j / seg), r * math.cos(math.pi * i / rings)) for i in range(1, rings) for j in range(seg)] + [(0, 0, -r)]
    f = [(0, 1 + (j + 1) % seg, 1 + j) for j in range(seg)]
    for i in range(rings - 2):
        for j in range(seg):
            a, b = 1 + i * seg + j, 1 + i * seg + (j + 1) % seg
            f.append((a, b, b + seg, a + seg))
    last = len(v) - 1
    f += [(last, 1 + (rings - 2) * seg + j, 1 + (rings - 2) * seg + (j + 1) % seg) for j in range(seg)]
    return Mesh(v, f)


def cylinder(r=1.0, h=2.0, seg=32, r2=None, caps=True):
    r2 = r if r2 is None else r2
    v = [(r * math.cos(2 * math.pi * j / seg), r * math.sin(2 * math.pi * j / seg), -h / 2) for j in range(seg)] + [(r2 * math.cos(2 * math.pi * j / seg), r2 * math.sin(2 * math.pi * j / seg), h / 2) for j in range(seg)]
    f = [(j, (j + 1) % seg, seg + (j + 1) % seg, seg + j) for j in range(seg)]
    if caps:
        if r > 0:
            f.append(tuple(reversed(range(seg))))
        if r2 > 0:
            f.append(tuple(range(seg, 2 * seg)))
    return Mesh(v, f)


def cone(r=1.0, h=2.0, seg=32):
    v = [(r * math.cos(2 * math.pi * j / seg), r * math.sin(2 * math.pi * j / seg), -h / 2) for j in range(seg)] + [(0, 0, h / 2)]
    return Mesh(v, [(j, (j + 1) % seg, seg) for j in range(seg)] + [tuple(reversed(range(seg)))])


def torus(R=1.0, r=0.3, seg=32, sides=16):
    v = [((R + r * math.cos(2 * math.pi * j / sides)) * math.cos(2 * math.pi * i / seg), (R + r * math.cos(2 * math.pi * j / sides)) * math.sin(2 * math.pi * i / seg), r * math.sin(2 * math.pi * j / sides)) for i in range(seg) for j in range(sides)]
    f = [(i * sides + j, i * sides + (j + 1) % sides, ((i + 1) % seg) * sides + (j + 1) % sides, ((i + 1) % seg) * sides + j) for i in range(seg) for j in range(sides)]
    return Mesh(v, f)


def icosphere(r=1.0, subdiv=2):
    t = (1 + 5 ** .5) / 2
    v = [_norm(p) for p in [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(min(subdiv, 4)):
        cache, nf = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                v.append(_norm(tuple((v[a][i] + v[b][i]) / 2 for i in range(3))))
                cache[k] = len(v) - 1
            return cache[k]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return Mesh([(x * r, y * r, z * r) for x, y, z in v], f)


def prism(n=6, r=1.0, h=2.0): return cylinder(r, h, n)
def pyramid(n=4, r=1.0, h=2.0): return cone(r, h, n)
def tetrahedron(r=1.0): return Mesh([(r, r, r), (r, -r, -r), (-r, r, -r), (-r, -r, r)], [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]).transform(scale=1 / math.sqrt(3))
def octahedron(r=1.0): return Mesh([(r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0), (0, 0, r), (0, 0, -r)], [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)])
def icosahedron(r=1.0): return icosphere(r, 0)


def dodecahedron(r=1.0):
    ico = icosphere(1, 0)
    cents = []
    for a, b, c in ico.f:
        cents.append(_norm(tuple(sum(ico.v[i][k] for i in (a, b, c)) / 3 for k in range(3))))
    faces = []
    for vi in range(len(ico.v)):
        adj = [i for i, f in enumerate(ico.f) if vi in f]
        n = ico.v[vi]
        ref = _norm(_sub(cents[adj[0]], tuple(n[k] * _dot(cents[adj[0]], n) for k in range(3))))
        up = _cross(n, ref)
        adj.sort(key=lambda i: math.atan2(_dot(cents[i], up), _dot(cents[i], ref)))
        faces.append(tuple(adj))
    return Mesh([(x * r, y * r, z * r) for x, y, z in cents], faces)


def capsule(r=0.5, h=2.0, seg=24):
    s = uv_sphere(r, seg, 12)
    top = s.transform((0, 0, h / 2))
    bot = s.transform((0, 0, -h / 2))
    return Mesh().add(top).add(bot).add(cylinder(r, h, seg, caps=False))


def spring(R=1.0, r=0.12, turns=5, height=3.0, seg=120, sides=8):
    v, f = [], []
    for i in range(seg + 1):
        t = i / seg
        a = 2 * math.pi * turns * t
        c = (R * math.cos(a), R * math.sin(a), height * (t - .5))
        n = (math.cos(a), math.sin(a), 0)
        for j in range(sides):
            b = 2 * math.pi * j / sides
            v.append((c[0] + r * (math.cos(b) * n[0]), c[1] + r * (math.cos(b) * n[1]), c[2] + r * math.sin(b)))
    for i in range(seg):
        for j in range(sides):
            f.append((i * sides + j, i * sides + (j + 1) % sides, (i + 1) * sides + (j + 1) % sides, (i + 1) * sides + j))
    return Mesh(v, f)


def gear(teeth=12, r=1.0, tooth=0.2, thick=0.4, hole=0.25):
    n = teeth * 4
    ring = [((r + (tooth if (i // 2) % 2 == 0 else 0)) * math.cos(2 * math.pi * i / n), (r + (tooth if (i // 2) % 2 == 0 else 0)) * math.sin(2 * math.pi * i / n)) for i in range(n)]
    inner = [(hole * math.cos(2 * math.pi * i / n), hole * math.sin(2 * math.pi * i / n)) for i in range(n)]
    v = [(x, y, -thick / 2) for x, y in ring] + [(x, y, thick / 2) for x, y in ring] + [(x, y, -thick / 2) for x, y in inner] + [(x, y, thick / 2) for x, y in inner]
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, n + j, n + i), (2 * n + j, 2 * n + i, 3 * n + i, 3 * n + j), (n + i, n + j, 3 * n + j, 3 * n + i), (j, i, 2 * n + i, 2 * n + j)]
    return Mesh(v, f)


def stairs(steps=8, w=2.0, rise=0.2, run=0.3):
    m = Mesh()
    for i in range(steps):
        m.add(box(w, run, rise).transform((0, i * run, rise / 2 + i * rise)))
    return m


def heightmap(w=8.0, d=8.0, n=32, amp=1.0, seed=1, octaves=3):
    rnd = random.Random(seed)
    waves = [(rnd.uniform(.3, 1.2) * (2 ** o), rnd.uniform(.3, 1.2) * (2 ** o), rnd.uniform(0, 6.28), rnd.uniform(0, 6.28), 1 / (2 ** o)) for o in range(octaves) for _ in range(3)]
    m = plane(w, d, n, n)
    m.v = [(x, y, amp * sum(a * math.sin(fx * x + p) * math.cos(fy * y + q) for fx, fy, p, q, a in waves) / 3) for x, y, _ in m.v]
    return m


def lathe(profile: list[tuple[float, float]], seg=32):
    v = [(r * math.cos(2 * math.pi * j / seg), r * math.sin(2 * math.pi * j / seg), z) for r, z in profile for j in range(seg)]
    f = [(i * seg + j, i * seg + (j + 1) % seg, (i + 1) * seg + (j + 1) % seg, (i + 1) * seg + j) for i in range(len(profile) - 1) for j in range(seg)]
    return Mesh(v, f)


def extrude(poly: list[tuple[float, float]], h=1.0):
    n = len(poly)
    v = [(x, y, 0.0) for x, y in poly] + [(x, y, h) for x, y in poly]
    f = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)] + [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
    return Mesh(v, f)


def vase(h=3.0, seg=32): return lathe([(0.01, 0), (0.8, 0), (1.0, h * .2), (0.7, h * .5), (0.45, h * .8), (0.6, h), (0.5, h)], seg)
def wineglass(seg=32): return lathe([(0.01, 0), (0.7, 0), (0.1, 0.1), (0.08, 1.2), (0.9, 1.9), (1.0, 3.0), (0.95, 3.0)], seg)
def chess_pawn(seg=32): return lathe([(0.01, 0), (0.8, 0), (0.8, 0.15), (0.45, 0.4), (0.25, 1.2), (0.5, 1.3), (0.3, 1.45), (0.4, 1.7), (0.3, 2.0), (0.01, 2.1)], seg)


PRIMS = {
    "cube": (cube, "s:float=2"), "box": (box, "w:float=2, d:float=2, h:float=2"), "plane": (plane, "w:float=2, d:float=2, sx:int=1, sy:int=1"),
    "uv_sphere": (uv_sphere, "r:float=1, seg:int=24, rings:int=12"), "icosphere": (icosphere, "r:float=1, subdiv:int=2"), "cylinder": (cylinder, "r:float=1, h:float=2, seg:int=32"),
    "cone": (cone, "r:float=1, h:float=2, seg:int=32"), "torus": (torus, "R:float=1, r:float=0.3, seg:int=32, sides:int=16"), "prism": (prism, "n:int=6, r:float=1, h:float=2"),
    "pyramid": (pyramid, "n:int=4, r:float=1, h:float=2"), "tetrahedron": (tetrahedron, "r:float=1"), "octahedron": (octahedron, "r:float=1"), "icosahedron": (icosahedron, "r:float=1"),
    "dodecahedron": (dodecahedron, "r:float=1"), "capsule": (capsule, "r:float=0.5, h:float=2, seg:int=24"), "spring": (spring, "R:float=1, r:float=0.12, turns:float=5, height:float=3"),
    "gear": (gear, "teeth:int=12, r:float=1, tooth:float=0.2, thick:float=0.4, hole:float=0.25"), "stairs": (stairs, "steps:int=8, w:float=2, rise:float=0.2, run:float=0.3"),
    "terrain": (heightmap, "w:float=8, d:float=8, n:int=32, amp:float=1, seed:int=1"), "vase": (vase, "h:float=3, seg:int=32"), "wineglass": (wineglass, "seg:int=32"), "chess_pawn": (chess_pawn, "seg:int=32"),
}


# ------------------------------------------------------------------ file formats
def to_obj(m: Mesh, name="mesh") -> str:
    return f"# AI Room\no {name}\n" + "".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in m.v) + "".join("f " + " ".join(str(i + 1) for i in f) + "\n" for f in m.f)


def to_stl(m: Mesh, name="mesh") -> str:
    out = [f"solid {name}"]
    for a, b, c in m.tris():
        n = _norm(_cross(_sub(m.v[b], m.v[a]), _sub(m.v[c], m.v[a])))
        out.append(f"facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n outer loop\n" + "".join(f"  vertex {m.v[i][0]:.6f} {m.v[i][1]:.6f} {m.v[i][2]:.6f}\n" for i in (a, b, c)) + " endloop\nendfacet")
    return "\n".join(out) + f"\nendsolid {name}\n"


def to_ply(m: Mesh) -> str:
    return f"ply\nformat ascii 1.0\nelement vertex {len(m.v)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(m.f)}\nproperty list uchar int vertex_indices\nend_header\n" + "".join(f"{x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in m.v) + "".join(f"{len(f)} " + " ".join(map(str, f)) + "\n" for f in m.f)


def to_gltf(m: Mesh, name="mesh") -> str:
    tris = list(m.tris())
    pos = b"".join(struct.pack("<3f", *p) for p in m.v)
    idx = b"".join(struct.pack("<3I", *t) for t in tris)
    pad = b"\0" * (-len(pos) % 4)
    buf = pos + pad + idx
    xs, ys, zs = zip(*m.v)
    c = [int(m.color.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return json.dumps({"asset": {"version": "2.0", "generator": "AI Room"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
                       "materials": [{"pbrMetallicRoughness": {"baseColorFactor": c + [1], "metallicFactor": 0.1, "roughnessFactor": 0.6}, "doubleSided": True}],
                       "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
                       "buffers": [{"byteLength": len(buf), "uri": "data:application/octet-stream;base64," + base64.b64encode(buf).decode()}],
                       "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos), "target": 34962}, {"buffer": 0, "byteOffset": len(pos) + len(pad), "byteLength": len(idx), "target": 34963}],
                       "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(m.v), "type": "VEC3", "min": [min(xs), min(ys), min(zs)], "max": [max(xs), max(ys), max(zs)]},
                                     {"bufferView": 1, "componentType": 5125, "count": len(tris) * 3, "type": "SCALAR"}]})


WRITERS = {"obj": to_obj, "stl": to_stl, "ply": lambda m, n="": to_ply(m), "gltf": to_gltf}


def parse_obj(text: str) -> Mesh:
    v, f = [], []
    for line in text.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "v" and len(p) >= 4:
            v.append((float(p[1]), float(p[2]), float(p[3])))
        elif p[0] == "f":
            idx = [int(x.split("/")[0]) for x in p[1:]]
            f.append(tuple(i - 1 if i > 0 else len(v) + i for i in idx))
    if not v:
        raise ToolError("no vertices found — is this an OBJ file?")
    return Mesh(v, f)


def parse_stl(text: str) -> Mesh:
    v, f, cur = [], [], []
    for line in text.splitlines():
        p = line.split()
        if p[:1] == ["vertex"]:
            v.append(tuple(map(float, p[1:4])))
            cur.append(len(v) - 1)
            if len(cur) == 3:
                f.append(tuple(cur))
                cur = []
    if not v:
        raise ToolError("no triangles found — only ASCII STL is supported")
    return Mesh(v, f)


def load(ctx: Context, path: str) -> Mesh:
    p = ctx.path(path, True)
    return parse_stl(p.read_text(errors="replace")) if p.suffix.lower() == ".stl" else parse_obj(p.read_text(errors="replace"))


def stats(m: Mesh) -> dict:
    xs, ys, zs = zip(*m.v)
    area = vol = 0.0
    for a, b, c in m.tris():
        pa, pb, pc = m.v[a], m.v[b], m.v[c]
        area += math.sqrt(_dot(*(2 * [_cross(_sub(pb, pa), _sub(pc, pa))]))) / 2
        vol += _dot(pa, _cross(pb, pc)) / 6
    edges: dict = {}
    for f in m.f:
        for i in range(len(f)):
            e = tuple(sorted((f[i], f[(i + 1) % len(f)])))
            edges[e] = edges.get(e, 0) + 1
    return {"vertices": len(m.v), "faces": len(m.f), "triangles": sum(1 for _ in m.tris()), "bbox_min": [round(min(xs), 4), round(min(ys), 4), round(min(zs), 4)], "bbox_max": [round(max(xs), 4), round(max(ys), 4), round(max(zs), 4)],
            "surface_area": round(area, 4), "volume": round(abs(vol), 4), "watertight": all(c == 2 for c in edges.values())}


def save_mesh(ctx: Context, m: Mesh, path: str, fmt: str | None = None, note: str = "") -> dict:
    fmt = (fmt or (path.rsplit(".", 1)[-1] if "." in path else "obj")).lower()
    if fmt not in WRITERS:
        raise ToolError(f"format must be one of {', '.join(WRITERS)}")
    if not path.lower().endswith("." + fmt):
        path = path.rsplit(".", 1)[0] + "." + fmt if "." in path.split("/")[-1] else path + "." + fmt
    p = ctx.path(path)
    p.write_text(WRITERS[fmt](m, p.stem), encoding="utf-8")
    return {"saved": ctx.produced(p, "model", note or f"{len(m.v)} vertices"), "vertices": len(m.v), "faces": len(m.f)}


# ------------------------------------------------------------------ software renderer
def project(m: Mesh, yaw=35.0, pitch=28.0, size=(640, 480), zoom=1.0, extra: list[Mesh] | None = None):
    meshes = [m] + (extra or [])
    pts = [p for mm in meshes for p in mm.v]
    cx, cy, cz = (sum(p[i] for p in pts) / len(pts) for i in range(3))
    rad = max(math.dist(p, (cx, cy, cz)) for p in pts) or 1
    ya, pa = math.radians(yaw), math.radians(pitch)
    light = _norm((0.4, -0.5, 0.8))
    polys = []
    for mm in meshes:
        rgb = [int(mm.color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
        for a, b, c in mm.tris():
            q = []
            for i in (a, b, c):
                x, y, z = mm.v[i][0] - cx, mm.v[i][1] - cy, mm.v[i][2] - cz
                x, y = x * math.cos(ya) - y * math.sin(ya), x * math.sin(ya) + y * math.cos(ya)
                y, z = y * math.cos(pa) - z * math.sin(pa), y * math.sin(pa) + z * math.cos(pa)
                q.append((x, y, z))
            n = _norm(_cross(_sub(q[1], q[0]), _sub(q[2], q[0])))
            lam = abs(_dot(n, light))
            depth = sum(p[1] for p in q) / 3
            scale = min(size) * 0.42 / rad * zoom
            xy = [(size[0] / 2 + p[0] * scale, size[1] / 2 - p[2] * scale) for p in q]
            k = 0.35 + 0.65 * lam
            polys.append((depth, xy, tuple(min(255, int(c * k)) for c in rgb)))
    polys.sort(key=lambda t: -t[0])
    return polys


def render_svg(meshes: list[Mesh], yaw=35.0, pitch=28.0, size=(640, 480), bg="#f4f6fb") -> str:
    polys = project(meshes[0], yaw, pitch, size, extra=meshes[1:])
    body = "".join('<polygon points="%s" fill="rgb(%d,%d,%d)" stroke="rgb(%d,%d,%d)" stroke-width="0.4" stroke-linejoin="round"/>' % (" ".join("%.1f,%.1f" % p for p in xy), *c, *c) for _, xy, c in polys)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size[0]} {size[1]}" width="{size[0]}" height="{size[1]}"><rect width="100%" height="100%" fill="{bg}"/>{body}</svg>'


def render_pil(meshes: list[Mesh], yaw=35.0, pitch=28.0, size=(640, 480), bg="#f4f6fb"):
    from PIL import Image, ImageDraw
    S = 2
    im = Image.new("RGB", (size[0] * S, size[1] * S), bg)
    d = ImageDraw.Draw(im)
    for _, xy, c in project(meshes[0], yaw, pitch, (size[0] * S, size[1] * S), extra=meshes[1:]):
        d.polygon(xy, fill=c, outline=c)
    return im.resize(size, Image.LANCZOS)


# ------------------------------------------------------------------ tools
M3 = "3d"


def _prim_tool(name, fn, sig):
    def run(ctx, path=None, format="obj", color="#8aa4ff", **kw):
        m = fn(**kw)
        m.color = color
        return save_mesh(ctx, m, path or f"models/{name}", format, name.replace("_", " "))
    reg(f"mesh_{name}", f"Create a {name.replace('_', ' ')} 3D mesh and save it (obj, stl, ply or gltf).", M3, "path:str?, format:str=obj, color:str=#8aa4ff, " + sig, run, ctx=True, risk="files", produces="model", tags=("blender", "3d", "model", "shape"))


for _n, (_f, _s) in PRIMS.items():
    _prim_tool(_n, _f, _s)


def _tf(ctx, path, out, tx, ty, tz, rx, ry, rz, sx, sy, sz):
    m = load(ctx, path).transform((tx, ty, tz), tuple(math.radians(a) for a in (rx, ry, rz)), (sx, sy, sz))
    return save_mesh(ctx, m, out or path.rsplit(".", 1)[0] + "_t.obj")


reg("mesh_transform", "Move (tx,ty,tz), rotate (degrees) and scale a mesh file; saves a new file.", M3, "path:str, out:str?, tx:float=0, ty:float=0, tz:float=0, rx:float=0, ry:float=0, rz:float=0, sx:float=1, sy:float=1, sz:float=1", _tf, ctx=True, risk="files", produces="model")
reg("mesh_merge", "Merge several mesh files into one.", M3, "paths:list, out:str=models/merged.obj", lambda ctx, paths, out: save_mesh(ctx, (lambda ms: ms[0].copy() if len(ms) == 1 else [ms[0].add(x) for x in ms[1:]][-1])([load(ctx, p) for p in paths]), out), ctx=True, risk="files", produces="model")
reg("mesh_convert", "Convert a mesh between obj, stl (ascii), ply and gltf.", M3, "path:str, format:str=stl, out:str?", lambda ctx, path, format, out: save_mesh(ctx, load(ctx, path), out or path.rsplit(".", 1)[0], format), ctx=True, risk="files", produces="model")
reg("mesh_stats", "Vertices, faces, bounding box, surface area, volume and watertightness of a mesh file.", M3, "path:str", lambda ctx, path: stats(load(ctx, path)), ctx=True, risk="files")
reg("mesh_center", "Move a mesh so its centre is at the origin.", M3, "path:str, out:str?", lambda ctx, path, out: (lambda m: save_mesh(ctx, m.transform(tuple(-sum(p[i] for p in m.v) / len(m.v) for i in range(3))), out or path.rsplit(".", 1)[0] + "_c.obj"))(load(ctx, path)), ctx=True, risk="files", produces="model")
reg("mesh_scale_to_size", "Scale a mesh so its longest side equals a size.", M3, "path:str, size:float=2, out:str?", lambda ctx, path, size, out: (lambda m: save_mesh(ctx, m.transform(scale=size / (max(max(p[i] for p in m.v) - min(p[i] for p in m.v) for i in range(3)) or 1)), out or path.rsplit(".", 1)[0] + "_s.obj"))(load(ctx, path)), ctx=True, risk="files", produces="model")
reg("mesh_flip_normals", "Reverse face winding.", M3, "path:str, out:str?", lambda ctx, path, out: (lambda m: save_mesh(ctx, Mesh(m.v, [tuple(reversed(f)) for f in m.f]), out or path.rsplit(".", 1)[0] + "_f.obj"))(load(ctx, path)), ctx=True, risk="files", produces="model")
reg("mesh_triangulate", "Split all faces into triangles.", M3, "path:str, out:str?", lambda ctx, path, out: (lambda m: save_mesh(ctx, Mesh(m.v, list(m.tris())), out or path.rsplit(".", 1)[0] + "_tri.obj"))(load(ctx, path)), ctx=True, risk="files", produces="model")
reg("mesh_extrude_polygon", "Extrude a 2-D polygon [[x,y],…] into a solid.", M3, "points:list, height:float=1, out:str=models/extrusion.obj", lambda ctx, points, height, out: save_mesh(ctx, extrude([tuple(p) for p in points], height), out), ctx=True, risk="files", produces="model")
reg("mesh_lathe", "Spin a profile [[radius,height],…] around the vertical axis (vases, bottles, chess pieces).", M3, "profile:list, segments:int=32, out:str=models/lathe.obj", lambda ctx, profile, segments, out: save_mesh(ctx, lathe([tuple(p) for p in profile], segments), out), ctx=True, risk="files", produces="model")
reg("mesh_heightmap_from_data", "Build a 3-D surface from a grid of heights (list of rows).", M3, "heights:list, cell:float=1, out:str=models/surface.obj", lambda ctx, heights, cell, out: save_mesh(ctx, Mesh([(j * cell, i * cell, float(z)) for i, r in enumerate(heights) for j, z in enumerate(r)], [(i * len(heights[0]) + j, i * len(heights[0]) + j + 1, (i + 1) * len(heights[0]) + j + 1, (i + 1) * len(heights[0]) + j) for i in range(len(heights) - 1) for j in range(len(heights[0]) - 1)]), out), ctx=True, risk="files", produces="model")
reg("mesh_function_surface", "Surface z = f(x, y) over a range, e.g. 'sin(x)*cos(y)'.", M3, "expression:str, xmin:float=-3, xmax:float=3, steps:int=40, out:str=models/surface.obj", lambda ctx, expression, xmin, xmax, steps, out: _surf(ctx, expression, xmin, xmax, steps, out), ctx=True, risk="files", produces="model")


def _surf(ctx, expr, lo, hi, n, out):
    from .t_math import evaluate
    n = min(max(n, 2), 120)
    g = [[evaluate(expr, x=lo + (hi - lo) * j / n, y=lo + (hi - lo) * i / n) for j in range(n + 1)] for i in range(n + 1)]
    g = [[(z.real if isinstance(z, complex) else z) for z in r] for r in g]
    return save_mesh(ctx, Mesh([(lo + (hi - lo) * j / n, lo + (hi - lo) * i / n, g[i][j]) for i in range(n + 1) for j in range(n + 1)], [(i * (n + 1) + j, i * (n + 1) + j + 1, (i + 1) * (n + 1) + j + 1, (i + 1) * (n + 1) + j) for i in range(n) for j in range(n)]), out)


def _render(ctx, paths, out, yaw, pitch, w, h, png):
    ms = []
    for i, p in enumerate(paths if isinstance(paths, list) else [paths]):
        m = load(ctx, p)
        m.color = ["#8aa4ff", "#ff9a76", "#5fd0a0", "#f2c14e", "#b78cf2"][i % 5]
        ms.append(m)
    o = ctx.path(out if "." in out.split("/")[-1] else out + (".png" if png else ".svg"))
    if png:
        render_pil(ms, yaw, pitch, (w, h)).save(o)
    else:
        o.write_text(render_svg(ms, yaw, pitch, (w, h)), encoding="utf-8")
    return {"saved": ctx.produced(o, "image", "3D render")}


reg("mesh_render_svg", "Render mesh file(s) from a viewpoint to an SVG picture (software renderer, no Blender needed).", M3, "paths:list, out:str=renders/view.svg, yaw:float=35, pitch:float=28, width:int=640, height:int=480",
    lambda ctx, paths, out, yaw, pitch, width, height: _render(ctx, paths, out, yaw, pitch, width, height, False), ctx=True, risk="files", produces="image", tags=("preview", "render"))
reg("mesh_render_png", "Render mesh file(s) from a viewpoint to a PNG picture (software renderer, no Blender needed).", M3, "paths:list, out:str=renders/view.png, yaw:float=35, pitch:float=28, width:int=640, height:int=480",
    lambda ctx, paths, out, yaw, pitch, width, height: _render(ctx, paths, out, yaw, pitch, width, height, True), ctx=True, risk="files", produces="image", needs=("py:PIL",), tags=("preview", "render"))


def _turntable(ctx, path, out, frames, width, height, ms):
    ims = [render_pil([load(ctx, path)], 360 * i / frames, 25, (width, height)) for i in range(min(frames, 72))]
    o = ctx.path(out)
    ims[0].save(o, save_all=True, append_images=ims[1:], duration=ms, loop=0, optimize=True)
    return {"saved": ctx.produced(o, "image", f"{len(ims)}-frame turntable")}


reg("mesh_turntable_gif", "Animated GIF of a mesh spinning around (software renderer).", M3, "path:str, out:str=renders/turntable.gif, frames:int=36, width:int=400, height:int=300, ms:int=60", _turntable, ctx=True, risk="files", produces="image", needs=("py:PIL",), tags=("animation", "video", "preview"))
reg("mesh_primitives_list", "All 3-D shapes the mesh tools can create.", M3, "", lambda: sorted(PRIMS))
