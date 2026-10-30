"""Blender: build a scene description with tools, turn it into a bpy script, preview it without Blender, and render it when Blender is installed.

A scene lives in `scenes/<name>.json` in the workspace. Every `blender_*` edit tool changes that file (pure Python, always works).
`blender_build_script` writes `scenes/<name>.py` — a script for `blender -b -P`. `blender_preview_*` draws the scene with the built-in
software renderer. `blender_render*` / `blender_run_script` call the real Blender and show as "needs setup" until it is installed.
"""

from __future__ import annotations

import json
import math
import re
import subprocess

from . import t_mesh as mesh
from .registry import Context, ToolError, reg

B = "blender"


# ------------------------------------------------------------------ scene file
def new_scene() -> dict:
    return {"objects": [], "materials": {}, "lights": [], "camera": {"loc": [7, -7, 5], "target": [0, 0, 0.8], "lens": 50}, "world": {"color": [0.05, 0.06, 0.09], "strength": 1.0},
            "render": {"engine": "EEVEE", "width": 1280, "height": 720, "samples": 64, "format": "PNG", "output": "renders/render", "transparent": False, "animation": False},
            "frames": {"start": 1, "end": 60, "fps": 24}, "export": [], "notes": []}


def load_scene(ctx: Context, name: str) -> dict:
    p = ctx.path(f"scenes/{name.removesuffix('.json')}.json")
    if not p.exists():
        raise ToolError(f"scene '{name}' does not exist yet — call blender_new_scene first")
    return json.loads(p.read_text())


def save_scene(ctx: Context, name: str, sc: dict, note: str = "") -> str:
    p = ctx.path(f"scenes/{name.removesuffix('.json')}.json")
    p.write_text(json.dumps(sc, indent=1))
    return ctx.produced(p, "scene", note)


def _edit(fn):
    """Wrap an editing function so it loads, edits and saves the scene named by the `scene` argument."""
    def run(ctx, scene="scene", **kw):
        sc = load_scene(ctx, scene)
        out = fn(sc, **kw)
        save_scene(ctx, scene, sc)
        return out if isinstance(out, (dict, list, str, int, float)) else "ok"
    return run


def _obj(sc, name):
    for o in sc["objects"]:
        if o["name"] == name:
            return o
    raise ToolError(f"no object named '{name}' in the scene. Objects: {[o['name'] for o in sc['objects']][:20]}")


def _uniq(sc, base):
    names = {o["name"] for o in sc["objects"]}
    if base not in names:
        return base
    i = 2
    while f"{base}.{i:03d}" in names:
        i += 1
    return f"{base}.{i:03d}"


def _rgb(c) -> list[float]:
    if isinstance(c, (list, tuple)):
        return [float(x) for x in c[:3]]
    c = str(c).strip()
    if re.fullmatch(r"#?[0-9a-fA-F]{6}", c):
        c = c.lstrip("#")
        lin = lambda v: (v / 255) ** 2.2  # noqa: E731  (approximate sRGB → linear, as Blender expects)
        return [round(lin(int(c[i:i + 2], 16)), 4) for i in (0, 2, 4)]
    named = {"red": "#e53935", "green": "#43a047", "blue": "#1e88e5", "yellow": "#fdd835", "white": "#f5f5f5", "black": "#1b1b1b", "orange": "#fb8c00", "purple": "#8e24aa", "pink": "#ec407a", "brown": "#6d4c41", "grey": "#9e9e9e", "gray": "#9e9e9e", "cyan": "#00acc1", "gold": "#ffc107"}
    if c.lower() in named:
        return _rgb(named[c.lower()])
    raise ToolError("colour must be like #ff8800 or a name such as red")


# ------------------------------------------------------------------ tables
BPY_PRIMS = {
    "cube": "bpy.ops.mesh.primitive_cube_add(size={s})", "uv_sphere": "bpy.ops.mesh.primitive_uv_sphere_add(radius={s}, segments=48, ring_count=24)",
    "ico_sphere": "bpy.ops.mesh.primitive_ico_sphere_add(radius={s}, subdivisions=3)", "cylinder": "bpy.ops.mesh.primitive_cylinder_add(radius={s}, depth={s}*2)",
    "cone": "bpy.ops.mesh.primitive_cone_add(radius1={s}, depth={s}*2)", "torus": "bpy.ops.mesh.primitive_torus_add(major_radius={s}, minor_radius={s}*0.25)",
    "plane": "bpy.ops.mesh.primitive_plane_add(size={s})", "circle": "bpy.ops.mesh.primitive_circle_add(radius={s}, fill_type='NGON')",
    "grid": "bpy.ops.mesh.primitive_grid_add(x_subdivisions=24, y_subdivisions=24, size={s})", "monkey": "bpy.ops.mesh.primitive_monkey_add(size={s})",
    "text": "bpy.ops.object.text_add()", "empty": "bpy.ops.object.empty_add(radius={s})", "bezier_curve": "bpy.ops.curve.primitive_bezier_curve_add(radius={s})",
    "bezier_circle": "bpy.ops.curve.primitive_bezier_circle_add(radius={s})", "metaball": "bpy.ops.object.metaball_add(type='BALL', radius={s})", "armature": "bpy.ops.object.armature_add(radius={s})",
}
PREVIEW = {"cube": lambda s: mesh.cube(s), "uv_sphere": lambda s: mesh.uv_sphere(s, 20, 10), "ico_sphere": lambda s: mesh.icosphere(s, 2), "cylinder": lambda s: mesh.cylinder(s, 2 * s, 24),
           "cone": lambda s: mesh.cone(s, 2 * s, 24), "torus": lambda s: mesh.torus(s, .25 * s, 28, 12), "plane": lambda s: mesh.plane(s, s), "circle": lambda s: mesh.cylinder(s, .001, 24),
           "grid": lambda s: mesh.plane(s, s, 8, 8), "monkey": lambda s: mesh.uv_sphere(s, 16, 8), "text": lambda s: mesh.box(s * 3, s * .3, s), "metaball": lambda s: mesh.uv_sphere(s, 16, 8)}
CUSTOM = {k: v[0] for k, v in mesh.PRIMS.items() if k not in ("cube", "plane", "uv_sphere", "icosphere", "cylinder", "cone", "torus")}  # meshes sent to Blender as raw vertices

MATERIALS = {  # colour, metallic, roughness, extras
    "gold": ("#ffc94a", 1, .25, {}), "silver": ("#d9dde3", 1, .2, {}), "copper": ("#d97b4a", 1, .3, {}), "chrome": ("#f2f4f7", 1, .02, {}), "iron": ("#6f737a", 1, .55, {}), "brass": ("#c9a14a", 1, .35, {}),
    "plastic_red": ("#d62828", 0, .35, {}), "plastic_blue": ("#1d4ed8", 0, .35, {}), "plastic_green": ("#2e9e44", 0, .35, {}), "plastic_white": ("#f1f1f1", 0, .4, {}), "plastic_black": ("#1b1b1f", 0, .35, {}),
    "rubber": ("#202020", 0, .9, {}), "wood": ("#8b5a2b", 0, .6, {}), "dark_wood": ("#4a2c17", 0, .55, {}), "marble": ("#e9e6df", 0, .15, {}), "concrete": ("#8c8c8c", 0, .95, {}), "brick": ("#a4452c", 0, .85, {}),
    "clay": ("#b5651d", 0, .8, {}), "grass": ("#3f8f3a", 0, .9, {}), "sand": ("#d8c38a", 0, .95, {}), "snow": ("#fafcff", 0, .6, {}), "skin": ("#e0a98a", 0, .5, {}),
    "glass": ("#ffffff", 0, .02, {"transmission": 1.0, "ior": 1.45}), "water": ("#9fd8ff", 0, .02, {"transmission": 1.0, "ior": 1.33}), "ice": ("#cfeaff", 0, .08, {"transmission": .9, "ior": 1.31}),
    "emission_white": ("#ffffff", 0, .5, {"emission": "#ffffff", "emission_strength": 8.0}), "emission_orange": ("#ff8a2a", 0, .5, {"emission": "#ff8a2a", "emission_strength": 12.0}),
    "emission_blue": ("#3a7bff", 0, .5, {"emission": "#3a7bff", "emission_strength": 10.0}), "neon_pink": ("#ff3ea5", 0, .4, {"emission": "#ff3ea5", "emission_strength": 14.0}),
    "lava": ("#ff4d00", 0, .6, {"emission": "#ff4d00", "emission_strength": 6.0}), "ceramic": ("#f4f1ea", 0, .12, {}), "leather": ("#5a3a26", 0, .5, {}), "fabric_red": ("#9b1c31", 0, .95, {}),
}

MODIFIERS = {  # blender type → (default props, description)
    "SUBSURF": ({"levels": 2, "render_levels": 3}, "smooth the surface (subdivision)"), "MIRROR": ({"use_axis": [True, False, False]}, "mirror along an axis"),
    "ARRAY": ({"count": 5, "relative_offset_displace": [1.1, 0, 0]}, "repeat the object in a row"), "BEVEL": ({"width": 0.05, "segments": 3}, "round off hard edges"),
    "SOLIDIFY": ({"thickness": 0.1}, "give a surface thickness"), "BOOLEAN": ({"operation": "DIFFERENCE", "operand_object": ""}, "cut or merge with another object"),
    "DECIMATE": ({"ratio": 0.5}, "reduce the polygon count"), "REMESH": ({"mode": "VOXEL", "voxel_size": 0.05}, "rebuild the mesh evenly"), "SMOOTH": ({"factor": 1.0, "iterations": 5}, "smooth vertices"),
    "WIREFRAME": ({"thickness": 0.03}, "turn edges into wires"), "SKIN": ({}, "make a mesh skeleton solid"), "SCREW": ({"angle": 6.283, "steps": 32, "screw_offset": 0}, "spin a profile into a solid"),
    "DISPLACE": ({"strength": 0.5}, "push the surface by a texture"), "WAVE": ({"height": 0.3, "width": 1.0, "speed": 0.5}, "ripple the surface"), "SIMPLE_DEFORM": ({"deform_method": "TWIST", "angle": 1.0}, "twist, bend, taper or stretch"),
    "CAST": ({"factor": 0.5}, "pull toward a sphere or cylinder"), "EDGE_SPLIT": ({}, "split sharp edges"), "TRIANGULATE": ({}, "convert faces to triangles"), "WEIGHTED_NORMAL": ({}, "better shading on bevels"),
    "SHRINKWRAP": ({"wrap_method": "NEAREST_SURFACEPOINT", "target": ""}, "stick to another surface"), "BUILD": ({"frame_duration": 60}, "build up face by face over time"), "EXPLODE": ({}, "blow faces apart (with particles)"),
    "OCEAN": ({"resolution": 16, "wave_scale": 1.0}, "animated ocean surface"), "CLOTH": ({}, "cloth simulation"), "SOFT_BODY": ({}, "soft-body simulation"), "COLLISION": ({}, "make objects collide with simulations"),
    "MIRROR_Y": None, "NORMAL_EDIT": ({}, "edit normals"), "LAPLACIANSMOOTH": ({"lambda_factor": 0.5, "iterations": 4}, "smooth while keeping volume"), "CORRECTIVE_SMOOTH": ({}, "smooth deformations"),
    "MULTIRES": ({}, "sculpt-friendly subdivisions"), "WELD": ({"merge_threshold": 0.001}, "merge nearby vertices"), "LATTICE": ({}, "deform with a lattice"), "CURVE": ({}, "deform along a curve"),
    "PARTICLE_SYSTEM": ({}, "emit particles"), "VOLUME_DISPLACE": ({}, "displace a volume"), "MASK": ({}, "hide parts of a mesh"), "DATA_TRANSFER": ({}, "copy data between meshes"),
}
MODIFIERS = {k: v for k, v in MODIFIERS.items() if v}

LIGHTS = {"point": ("POINT", 1000, 4), "sun": ("SUN", 4, 0), "spot": ("SPOT", 1500, 4), "area_soft": ("AREA", 800, 3), "area_big": ("AREA", 2000, 6)}
WORLDS = {"studio_grey": ([.3, .3, .32], 1.0), "night": ([.01, .015, .04], .6), "sunset": ([.9, .45, .2], 1.2), "daylight_sky": ([.45, .65, .95], 1.3), "white": ([1, 1, 1], 1.0), "black": ([0, 0, 0], 0.0), "overcast": ([.6, .62, .65], 1.0), "space": ([0, 0, .01], .2)}
CAMERAS = {"front": ([0, -9, 1.5], [0, 0, 1], 50), "back": ([0, 9, 1.5], [0, 0, 1], 50), "side": ([9, 0, 1.5], [0, 0, 1], 50), "top": ([0.001, 0, 12], [0, 0, 0], 50), "isometric": ([8, -8, 7], [0, 0, 0.5], 50), "three_quarter": ([6, -6, 3.5], [0, 0, 0.8], 50),
           "close_up": ([2.5, -2.5, 1.5], [0, 0, 0.8], 85), "wide": ([12, -12, 6], [0, 0, 1], 24), "low_angle": ([5, -5, 0.4], [0, 0, 1.2], 35), "bird_eye": ([6, -6, 14], [0, 0, 0], 40)}
RENDERS = {"draft_720p": ("EEVEE", 1280, 720, 16), "hd_1080p": ("EEVEE", 1920, 1080, 64), "uhd_4k": ("CYCLES", 3840, 2160, 128), "cycles_1080p": ("CYCLES", 1920, 1080, 128), "cycles_preview": ("CYCLES", 960, 540, 32),
           "square_instagram": ("EEVEE", 1080, 1080, 64), "story_1080x1920": ("EEVEE", 1080, 1920, 64), "youtube_1080p": ("EEVEE", 1920, 1080, 64), "thumbnail_1280x720": ("EEVEE", 1280, 720, 48), "workbench_fast": ("WORKBENCH", 1280, 720, 1)}


# ------------------------------------------------------------------ scene building tools
def _add(sc, kind, name=None, x=0, y=0, z=0, rx=0, ry=0, rz=0, sx=1, sy=1, sz=1, material=None, size=1.0, text="Text", smooth=True, parent=None):
    nm = _uniq(sc, name or kind)
    o = {"name": nm, "kind": kind, "size": size, "loc": [x, y, z], "rot": [rx, ry, rz], "scale": [sx, sy, sz], "material": material, "modifiers": [], "smooth": smooth, "text": text, "parent": parent, "keys": []}
    if material and material not in sc["materials"]:
        if material in MATERIALS:
            _set_mat(sc, material, material)
        else:
            raise ToolError(f"unknown material '{material}'. Presets: {', '.join(MATERIALS)} (or create one with blender_material_custom)")
    sc["objects"].append(o)
    return {"added": nm, "objects_in_scene": len(sc["objects"])}


def _set_mat(sc, name, preset):
    c, m, r, ex = MATERIALS[preset]
    sc["materials"][name] = {"color": _rgb(c), "metallic": m, "roughness": r, **{k: (_rgb(v) if k == "emission" else v) for k, v in ex.items()}}


_PLACE = "scene:str=scene, name:str?, x:float=0, y:float=0, z:float=0, rx:float=0, ry:float=0, rz:float=0, sx:float=1, sy:float=1, sz:float=1, material:str?, size:float=1, smooth:bool=true"
for _k in list(BPY_PRIMS) + list(CUSTOM):
    reg(f"blender_add_{_k}", f"Add a {_k.replace('_', ' ')} to a Blender scene (position x,y,z; rotation in degrees; size is the radius or edge). Optional material preset.", B,
        _PLACE + (", text:str=Text" if _k == "text" else ""), (lambda k: _edit(lambda sc, **kw: _add(sc, k, **{**kw, **{a: math.radians(kw[a]) for a in ("rx", "ry", "rz")}})))(_k), ctx=True, risk="files", produces="scene", tags=("3d", "create", "model", "mesh"))

reg("blender_new_scene", "Start a new Blender scene file (optionally with a ground plane and a camera/lights/world preset).", B, "scene:str=scene, ground:bool=true, lighting:str=three_point, world:str=studio_grey, camera:str=three_quarter",
    lambda ctx, scene, ground, lighting, world, camera: _new(ctx, scene, ground, lighting, world, camera), ctx=True, risk="files", produces="scene", tags=("3d", "create"))


def _new(ctx, scene, ground, lighting, world, camera):
    sc = new_scene()
    if ground:
        _add(sc, "plane", "Ground", size=30, material="concrete")
    if lighting != "none":
        _lights(sc, lighting)
    _world(sc, world)
    _camera(sc, camera)
    return {"scene": save_scene(ctx, scene, sc, "new Blender scene")}


def _lights(sc, kind):
    sc["lights"] = []
    if kind == "three_point":
        sc["lights"] = [{"name": "Key", "type": "AREA", "energy": 1200, "loc": [4, -4, 6], "color": [1, .95, .9], "size": 3}, {"name": "Fill", "type": "AREA", "energy": 450, "loc": [-5, -3, 3], "color": [.8, .9, 1], "size": 4}, {"name": "Rim", "type": "AREA", "energy": 700, "loc": [0, 6, 5], "color": [1, 1, 1], "size": 2}]
    elif kind == "studio":
        sc["lights"] = [{"name": "Top", "type": "AREA", "energy": 1800, "loc": [0, 0, 8], "color": [1, 1, 1], "size": 6}, {"name": "Left", "type": "AREA", "energy": 600, "loc": [-6, -2, 3], "color": [1, 1, 1], "size": 3}, {"name": "Right", "type": "AREA", "energy": 600, "loc": [6, -2, 3], "color": [1, 1, 1], "size": 3}]
    elif kind == "sunlight":
        sc["lights"] = [{"name": "Sun", "type": "SUN", "energy": 4, "loc": [5, -5, 10], "color": [1, .96, .88], "size": 1, "rot": [math.radians(50), 0, math.radians(35)]}]
    elif kind == "dramatic":
        sc["lights"] = [{"name": "Spot", "type": "SPOT", "energy": 3000, "loc": [3, -4, 7], "color": [1, .85, .7], "size": 1, "target": [0, 0, 0]}]
    else:
        raise ToolError("lighting: three_point, studio, sunlight, dramatic, none")


def _world(sc, name):
    if name not in WORLDS:
        raise ToolError(f"world presets: {', '.join(WORLDS)}")
    sc["world"] = {"color": WORLDS[name][0], "strength": WORLDS[name][1]}


def _camera(sc, name):
    if name not in CAMERAS:
        raise ToolError(f"camera presets: {', '.join(CAMERAS)}")
    l, t, f = CAMERAS[name]
    sc["camera"] = {"loc": list(l), "target": list(t), "lens": f}


reg("blender_light_rig", "Set the scene lighting: three_point, studio, sunlight or dramatic.", B, "scene:str=scene, rig:str=three_point", _edit(lambda sc, rig: _lights(sc, rig)), ctx=True, risk="files", produces="scene")
for _lk, (_lt, _le, _ls) in LIGHTS.items():
    reg(f"blender_add_light_{_lk}", f"Add a {_lk.replace('_', ' ')} light.", B, "scene:str=scene, name:str?, x:float=3, y:float=-3, z:float=5, energy:float=0, color:str=#ffffff, size:float=0",
        (lambda lt, le, ls: _edit(lambda sc, name, x, y, z, energy, color, size: sc["lights"].append({"name": name or lt.title(), "type": lt, "energy": energy or le, "loc": [x, y, z], "color": _rgb(color), "size": size or ls or 1})))(_lt, _le, _ls), ctx=True, risk="files", produces="scene")
for _w in WORLDS:
    reg(f"blender_world_{_w}", f"Set the background/world to '{_w.replace('_', ' ')}'.", B, "scene:str=scene", (lambda w: _edit(lambda sc: _world(sc, w)))(_w), ctx=True, risk="files", produces="scene")
reg("blender_world_custom", "Set a custom background colour and strength.", B, "scene:str=scene, color:str=#0d0f18, strength:float=1", _edit(lambda sc, color, strength: sc.__setitem__("world", {"color": _rgb(color), "strength": strength})), ctx=True, risk="files", produces="scene")
for _c in CAMERAS:
    reg(f"blender_camera_{_c}", f"Place the camera at the '{_c.replace('_', ' ')}' viewpoint.", B, "scene:str=scene", (lambda c: _edit(lambda sc: _camera(sc, c)))(_c), ctx=True, risk="files", produces="scene")
reg("blender_camera_custom", "Place the camera at x,y,z looking at a target, with a lens in mm.", B, "scene:str=scene, x:float=7, y:float=-7, z:float=5, tx:float=0, ty:float=0, tz:float=1, lens:float=50", _edit(lambda sc, x, y, z, tx, ty, tz, lens: sc.__setitem__("camera", {"loc": [x, y, z], "target": [tx, ty, tz], "lens": lens})), ctx=True, risk="files", produces="scene")
reg("blender_camera_frame_all", "Move the camera back so every object is in view.", B, "scene:str=scene", _edit(lambda sc: _frame(sc)), ctx=True, risk="files", produces="scene")


def _frame(sc):
    pts = [o["loc"] for o in sc["objects"] if o["name"] != "Ground"] or [[0, 0, 0]]
    c = [sum(p[i] for p in pts) / len(pts) for i in range(3)]
    r = max(max(math.dist(p, c) for p in pts) + 2, 3)
    sc["camera"] = {"loc": [c[0] + r * 1.4, c[1] - r * 1.4, c[2] + r], "target": c, "lens": 50}
    return {"camera": sc["camera"]}


for _m in MATERIALS:
    reg(f"blender_material_{_m}", f"Give an object the '{_m.replace('_', ' ')}' material.", B, "scene:str=scene, object:str", (lambda m: _edit(lambda sc, object: (_ensure_mat(sc, m), _obj(sc, object).__setitem__("material", m))))(_m), ctx=True, risk="files", produces="scene", tags=("shader", "texture"))


def _ensure_mat(sc, m):
    if m not in sc["materials"]:
        _set_mat(sc, m, m)


reg("blender_material_custom", "Create a material (colour, metallic 0–1, roughness 0–1, optional emission strength) and assign it to objects (comma-separated names).", B, "scene:str=scene, name:str, color:str=#cccccc, metallic:float=0, roughness:float=0.5, emission_strength:float=0, objects:str?",
    _edit(lambda sc, name, color, metallic, roughness, emission_strength, objects: (sc["materials"].__setitem__(name, {"color": _rgb(color), "metallic": metallic, "roughness": roughness, **({"emission": _rgb(color), "emission_strength": emission_strength} if emission_strength else {})}),
                                                                                    [_obj(sc, n.strip()).__setitem__("material", name) for n in (objects or "").split(",") if n.strip()])), ctx=True, risk="files", produces="scene")
reg("blender_recolor_object", "Quickly colour an object (creates a plain material).", B, "scene:str=scene, object:str, color:str", _edit(lambda sc, object, color: (sc["materials"].__setitem__("color_" + object, {"color": _rgb(color), "metallic": 0, "roughness": .5}), _obj(sc, object).__setitem__("material", "color_" + object))), ctx=True, risk="files", produces="scene")


def _mod(sc, mtype, object, name=None, **props):
    o = _obj(sc, object)
    d = {**MODIFIERS[mtype][0], **{k: v for k, v in props.items() if v not in (None, "")}}
    o["modifiers"].append({"type": mtype, "name": name or mtype.title(), "props": d})
    return {"object": object, "modifiers": [m["type"] for m in o["modifiers"]]}


for _t, (_d, _desc) in MODIFIERS.items():
    _sig = "scene:str=scene, object:str" + "".join(f", {k}:{'float' if isinstance(v, float) else 'int' if isinstance(v, int) and not isinstance(v, bool) else 'str'}?" for k, v in _d.items() if isinstance(v, (int, float, str)) and not isinstance(v, bool))
    reg(f"blender_modifier_{_t.lower()}", f"Add a {_t.replace('_', ' ').title()} modifier: {_desc}.", B, _sig, (lambda t: _edit(lambda sc, object, **kw: _mod(sc, t, object, **kw)))(_t), ctx=True, risk="files", produces="scene", tags=("modifier", "mesh"))

reg("blender_set_object", "Change an object's position, rotation (degrees), scale or size.", B, "scene:str=scene, object:str, x:float?, y:float?, z:float?, rx:float?, ry:float?, rz:float?, sx:float?, sy:float?, sz:float?",
    _edit(lambda sc, object, x, y, z, rx, ry, rz, sx, sy, sz: _setobj(sc, object, x, y, z, rx, ry, rz, sx, sy, sz)), ctx=True, risk="files", produces="scene")


def _setobj(sc, name, x, y, z, rx, ry, rz, sx, sy, sz):
    o = _obj(sc, name)
    for i, v in enumerate((x, y, z)):
        if v is not None:
            o["loc"][i] = v
    for i, v in enumerate((rx, ry, rz)):
        if v is not None:
            o["rot"][i] = math.radians(v)
    for i, v in enumerate((sx, sy, sz)):
        if v is not None:
            o["scale"][i] = v
    return {name: o["loc"]}


reg("blender_duplicate_object", "Copy an object, offset by dx,dy,dz.", B, "scene:str=scene, object:str, dx:float=1, dy:float=0, dz:float=0", _edit(lambda sc, object, dx, dy, dz: (lambda o: (lambda c: (c.update(name=_uniq(sc, o["name"]), loc=[o["loc"][0] + dx, o["loc"][1] + dy, o["loc"][2] + dz], modifiers=json.loads(json.dumps(o["modifiers"])), keys=[]), sc["objects"].append(c), c["name"])[-1])(dict(o)))(_obj(sc, object))), ctx=True, risk="files", produces="scene")
reg("blender_delete_object", "Remove an object from the scene.", B, "scene:str=scene, object:str", _edit(lambda sc, object: sc.__setitem__("objects", [o for o in sc["objects"] if o["name"] != object])), ctx=True, risk="files", produces="scene")
reg("blender_parent_object", "Parent one object to another.", B, "scene:str=scene, child:str, parent:str", _edit(lambda sc, child, parent: (_obj(sc, parent), _obj(sc, child).__setitem__("parent", parent))), ctx=True, risk="files", produces="scene")
reg("blender_array_grid", "Fill a grid of copies of an object (nx × ny, spacing).", B, "scene:str=scene, object:str, nx:int=3, ny:int=3, spacing:float=2.5", _edit(lambda sc, object, nx, ny, spacing: _grid(sc, object, nx, ny, spacing)), ctx=True, risk="files", produces="scene")
reg("blender_circle_of_objects", "Place copies of an object around a circle.", B, "scene:str=scene, object:str, count:int=8, radius:float=4", _edit(lambda sc, object, count, radius: _ring(sc, object, count, radius)), ctx=True, risk="files", produces="scene")


def _grid(sc, name, nx, ny, sp):
    o, n = _obj(sc, name), 0
    for i in range(min(nx, 30)):
        for j in range(min(ny, 30)):
            if i or j:
                c = json.loads(json.dumps(o))
                c.update(name=_uniq(sc, name), loc=[o["loc"][0] + i * sp, o["loc"][1] + j * sp, o["loc"][2]], keys=[])
                sc["objects"].append(c)
                n += 1
    return {"copies": n}


def _ring(sc, name, count, r):
    o = _obj(sc, name)
    for i in range(min(count, 200)):
        a = 2 * math.pi * i / count
        c = json.loads(json.dumps(o))
        c.update(name=_uniq(sc, name), loc=[o["loc"][0] + r * math.cos(a), o["loc"][1] + r * math.sin(a), o["loc"][2]], rot=[o["rot"][0], o["rot"][1], a + math.pi / 2], keys=[])
        sc["objects"].append(c)
    return {"copies": min(count, 200)}


# ---- animation
def _key(sc, object, frame, x=None, y=None, z=None, rx=None, ry=None, rz=None, sx=None, sy=None, sz=None):
    o = _obj(sc, object)
    k = {"frame": frame, "loc": [x if x is not None else o["loc"][0], y if y is not None else o["loc"][1], z if z is not None else o["loc"][2]],
         "rot": [math.radians(v) if v is not None else o["rot"][i] for i, v in enumerate((rx, ry, rz))], "scale": [v if v is not None else o["scale"][i] for i, v in enumerate((sx, sy, sz))]}
    o["keys"] = sorted([q for q in o["keys"] if q["frame"] != frame] + [k], key=lambda q: q["frame"])
    sc["frames"]["end"] = max(sc["frames"]["end"], frame)
    return {"keyframes": len(o["keys"])}


reg("blender_keyframe", "Set a keyframe: the object's pose at a frame (leave a value empty to keep it). Position/scale in scene units, rotation in degrees.", B, "scene:str=scene, object:str, frame:int, x:float?, y:float?, z:float?, rx:float?, ry:float?, rz:float?, sx:float?, sy:float?, sz:float?", _edit(_key), ctx=True, risk="files", produces="scene", tags=("animation",))
reg("blender_set_frames", "Set the animation start, end frame and frames per second.", B, "scene:str=scene, start:int=1, end:int=120, fps:int=24", _edit(lambda sc, start, end, fps: sc.__setitem__("frames", {"start": start, "end": end, "fps": fps})), ctx=True, risk="files", produces="scene", tags=("animation",))


def _spin(sc, object, frames=120, turns=1.0, axis="z"):
    o = _obj(sc, object)
    i = "xyz".index(axis)
    for f, a in ((1, 0), (frames, 360 * turns)):
        r = [None] * 3
        r[i] = a
        _key(sc, object, f, rx=r[0], ry=r[1], rz=r[2])
    sc["frames"]["end"] = max(frames, 2)
    return {"frames": frames}


def _bounce(sc, object, height=2.0, frames=48, bounces=3):
    o = _obj(sc, object)
    z0 = o["loc"][2]
    for b in range(bounces):
        for k, (t, z) in enumerate(((0, z0), (0.5, z0 + height * (0.6 ** b)), (1, z0))):
            _key(sc, object, 1 + round((b + t) * frames / bounces), z=z)
    return {"frames": frames}


def _pulse(sc, object, frames=48, amount=0.25):
    o = _obj(sc, object)
    s = o["scale"][0]
    for f, k in ((1, 1), (frames // 2, 1 + amount), (frames, 1)):
        _key(sc, object, f, sx=s * k, sy=s * k, sz=s * k)


def _slide(sc, object, dx=5.0, dy=0.0, dz=0.0, frames=60):
    o = _obj(sc, object)
    _key(sc, object, 1)
    _key(sc, object, frames, x=o["loc"][0] + dx, y=o["loc"][1] + dy, z=o["loc"][2] + dz)


def _grow(sc, object, frames=30):
    o = _obj(sc, object)
    s = o["scale"]
    _key(sc, object, 1, sx=0.001, sy=0.001, sz=0.001)
    _key(sc, object, frames, sx=s[0], sy=s[1], sz=s[2])


def _orbit(sc, frames=120, radius=0.0, height=0.0):
    c, t = sc["camera"], sc["camera"]["target"]
    r = radius or math.hypot(c["loc"][0] - t[0], c["loc"][1] - t[1])
    z = height or c["loc"][2]
    sc["camera"]["orbit"] = {"frames": frames, "radius": r, "height": z}
    sc["frames"]["end"] = max(frames, 2)
    sc["render"]["animation"] = True
    return {"frames": frames}


def _wobble(sc, object, frames=48, degrees=15):
    for f, a in ((1, 0), (frames // 4, degrees), (frames * 3 // 4, -degrees), (frames, 0)):
        _key(sc, object, f, rx=a)


def _flythrough(sc, points, frames=120):
    sc["camera"]["path"] = [[float(c) for c in p] for p in points]
    sc["frames"]["end"] = frames
    sc["render"]["animation"] = True


for _n, _f, _d, _s in (
    ("spin_object", _spin, "Make an object rotate (turns × 360°) around an axis x, y or z.", "scene:str=scene, object:str, frames:int=120, turns:float=1, axis:str=z"),
    ("bounce_object", _bounce, "Make an object bounce up and down.", "scene:str=scene, object:str, height:float=2, frames:int=48, bounces:int=3"),
    ("pulse_object", _pulse, "Make an object grow and shrink rhythmically.", "scene:str=scene, object:str, frames:int=48, amount:float=0.25"),
    ("slide_object", _slide, "Slide an object by dx,dy,dz over some frames.", "scene:str=scene, object:str, dx:float=5, dy:float=0, dz:float=0, frames:int=60"),
    ("grow_object", _grow, "Make an object grow from nothing to full size.", "scene:str=scene, object:str, frames:int=30"),
    ("wobble_object", _wobble, "Make an object rock back and forth.", "scene:str=scene, object:str, frames:int=48, degrees:float=15"),
    ("orbit_camera", _orbit, "Make the camera circle the scene (a turntable shot) and switch to animation rendering.", "scene:str=scene, frames:int=120, radius:float=0, height:float=0"),
    ("camera_flythrough", _flythrough, "Fly the camera through a list of [x,y,z] points.", "scene:str=scene, points:list, frames:int=120"),
):
    reg(f"blender_anim_{_n}", _d, B, _s, (lambda f: _edit(f))(_f), ctx=True, risk="files", produces="scene", tags=("animation", "video"))

# ---- render + export settings
for _r, (_e, _w_, _h, _sm) in RENDERS.items():
    reg(f"blender_render_preset_{_r}", f"Render settings: {_e.title()} at {_w_}×{_h}.", B, "scene:str=scene", (lambda e, w, h, s: _edit(lambda sc: sc["render"].update(engine=e, width=w, height=h, samples=s)))(_e, _w_, _h, _sm), ctx=True, risk="files", produces="scene")
reg("blender_render_settings", "Custom render settings: engine EEVEE/CYCLES/WORKBENCH, size, samples, format (PNG, JPEG, MP4), transparent background.", B, "scene:str=scene, engine:str?, width:int?, height:int?, samples:int?, format:str?, transparent:bool?, output:str?",
    _edit(lambda sc, engine, width, height, samples, format, transparent, output: sc["render"].update({k: v for k, v in dict(engine=(engine or "").upper() or None, width=width, height=height, samples=samples, format=(format or "").upper() or None, transparent=transparent, output=output).items() if v is not None})), ctx=True, risk="files", produces="scene")
reg("blender_render_as_video", "Render the animation as an MP4 video (H.264).", B, "scene:str=scene", _edit(lambda sc: sc["render"].update(format="MP4", animation=True, output="renders/animation")), ctx=True, risk="files", produces="scene", tags=("video",))
for _x in ("glb", "obj", "stl", "fbx", "usd", "blend"):
    reg(f"blender_export_{_x}", f"After running the script, also export the scene as .{_x}.", B, "scene:str=scene, path:str?", (lambda x: _edit(lambda sc, path: sc["export"].append({"format": x, "path": path or f"exports/scene.{x}"})))(_x), ctx=True, risk="files", produces="scene")

# ------------------------------------------------------------------ generators (recipes that build whole scenes from primitives)
def _g(sc, kind, name, loc, size=1.0, mat=None, scale=(1, 1, 1), rot=(0, 0, 0)):
    return _add(sc, kind, name, loc[0], loc[1], loc[2], rot[0], rot[1], rot[2], scale[0], scale[1], scale[2], mat, size)


def gen_table(sc, x=0, y=0, w=3.0, d=1.6, h=1.5):
    _g(sc, "cube", "TableTop", (x, y, h), 1, "wood", (w, d, .12))
    for sx in (-1, 1):
        for sy in (-1, 1):
            _g(sc, "cube", "TableLeg", (x + sx * (w / 2 - .15), y + sy * (d / 2 - .15), h / 2), 1, "dark_wood", (.14, .14, h))


def gen_chair(sc, x=0, y=0, rz=0):
    _g(sc, "cube", "Seat", (x, y, .9), 1, "wood", (1, 1, .1), (0, 0, math.radians(rz)))
    _g(sc, "cube", "ChairBack", (x, y + .45, 1.6), 1, "wood", (1, .1, 1.3))
    for sx in (-1, 1):
        for sy in (-1, 1):
            _g(sc, "cube", "ChairLeg", (x + sx * .42, y + sy * .42, .43), 1, "dark_wood", (.1, .1, .86))


def gen_house(sc, x=0, y=0):
    _g(sc, "cube", "HouseWalls", (x, y, 1.5), 1, "brick", (4, 3.2, 3))
    _g(sc, "cone", "Roof", (x, y, 3.8), 1, "dark_wood", (2.7, 2.7, .8))
    _g(sc, "cube", "Door", (x, y - 1.61, .9), 1, "dark_wood", (.8, .1, 1.8))
    for sx in (-1.2, 1.2):
        _g(sc, "cube", "Window", (x + sx, y - 1.61, 1.8), 1, "glass", (.7, .06, .7))
    _g(sc, "cube", "Chimney", (x + 1.2, y + .6, 4.2), 1, "brick", (.5, .5, 1.2))


def gen_tree(sc, x=0, y=0, s=1.0):
    _g(sc, "cylinder", "Trunk", (x, y, s), .25 * s, "dark_wood", (1, 1, 4))
    for i, (r, z) in enumerate(((1.4, 2.6), (1.1, 3.4), (.8, 4.1))):
        _g(sc, "ico_sphere", "Leaves", (x, y, z * s), r * s, "grass")


def gen_pine(sc, x=0, y=0):
    _g(sc, "cylinder", "Trunk", (x, y, .5), .2, "dark_wood", (1, 1, 2.5))
    for i, (r, z) in enumerate(((1.2, 1.4), (.95, 2.2), (.7, 2.9), (.45, 3.5))):
        _g(sc, "cone", "Needles", (x, y, z), r, "grass", (1, 1, .55))


def gen_snowman(sc, x=0, y=0):
    for r, z in ((.9, .9), (.65, 2.2), (.45, 3.2)):
        _g(sc, "uv_sphere", "SnowBall", (x, y, z), r, "snow")
    _g(sc, "cone", "Nose", (x, y - .45, 3.2), .08, "emission_orange", (1, 1, 2.5), (math.pi / 2, 0, 0))
    for sx in (-.15, .15):
        _g(sc, "uv_sphere", "Eye", (x + sx, y - .38, 3.35), .05, "plastic_black")


def gen_robot(sc, x=0, y=0):
    _g(sc, "cube", "Body", (x, y, 1.6), 1, "chrome", (1.2, .8, 1.4))
    _g(sc, "cube", "Head", (x, y, 2.7), 1, "silver", (.9, .8, .7))
    for sx in (-.25, .25):
        _g(sc, "uv_sphere", "Eye", (x + sx, y - .42, 2.78), .12, "emission_blue")
        _g(sc, "cylinder", "Leg", (x + sx * 1.8, y, .55), .17, "iron", (1, 1, 1.5))
        _g(sc, "cylinder", "Arm", (x + sx * 3.1, y, 1.7), .13, "iron", (1, 1, 1.5))
    _g(sc, "cylinder", "Antenna", (x, y, 3.3), .03, "iron", (1, 1, 2))
    _g(sc, "uv_sphere", "AntennaTip", (x, y, 3.55), .08, "emission_orange")


def gen_car(sc, x=0, y=0):
    _g(sc, "cube", "CarBody", (x, y, .75), 1, "plastic_red", (4, 1.8, .7))
    _g(sc, "cube", "Cabin", (x - .2, y, 1.35), 1, "plastic_red", (2, 1.6, .6))
    _g(sc, "cube", "Windows", (x - .2, y, 1.4), 1, "glass", (2.02, 1.62, .4))
    for sx in (-1.3, 1.3):
        for sy in (-.95, .95):
            _g(sc, "cylinder", "Wheel", (x + sx, y + sy, .45), .45, "rubber", (1, 1, .25), (math.pi / 2, 0, 0))


def gen_tower(sc, x=0, y=0, floors=6):
    for i in range(floors):
        _g(sc, "cube", "TowerFloor", (x, y, .6 + i * 1.2), 1, "concrete" if i % 2 else "brick", (2 - i * .1, 2 - i * .1, 1.2))
    _g(sc, "cone", "Spire", (x, y, floors * 1.2 + .8), 1, "gold", (1.2, 1.2, 1.6))


def gen_castle(sc, x=0, y=0):
    _g(sc, "cube", "Keep", (x, y, 2), 1, "concrete", (4, 4, 4))
    for sx in (-2.5, 2.5):
        for sy in (-2.5, 2.5):
            _g(sc, "cylinder", "CastleTower", (x + sx, y + sy, 2.5), .8, "concrete", (1, 1, 2.5))
            _g(sc, "cone", "TowerRoof", (x + sx, y + sy, 5.6), .95, "plastic_red", (1, 1, 1.1))
    _g(sc, "cube", "Gate", (x, y - 2.02, 1), 1, "dark_wood", (1.2, .1, 2))


def gen_bridge(sc, x=0, y=0, span=8):
    _g(sc, "cube", "Deck", (x, y, 2), 1, "concrete", (span, 1.6, .25))
    for i in range(-2, 3):
        _g(sc, "cylinder", "Pillar", (x + i * span / 5, y, 1), .22, "concrete", (1, 1, 1))
    for sy in (-.75, .75):
        _g(sc, "cube", "Rail", (x, y + sy, 2.5), 1, "iron", (span, .08, .08))


def gen_windmill(sc, x=0, y=0):
    _g(sc, "cone", "MillBody", (x, y, 2), 1.2, "brick", (1, 1, 1.5))
    _g(sc, "cone", "MillRoof", (x, y, 4.3), .9, "dark_wood", (1, 1, .5))
    for i in range(4):
        a = i * math.pi / 2
        _g(sc, "cube", "Blade", (x + math.sin(a) * 1.3, y - 1.25, 3.4 + math.cos(a) * 1.3), 1, "wood", (.25, .05, 2.2), (0, a, 0))


def gen_lamp(sc, x=0, y=0):
    _g(sc, "cylinder", "LampBase", (x, y, .1), .5, "iron", (1, 1, .1))
    _g(sc, "cylinder", "LampPole", (x, y, 1.8), .06, "iron", (1, 1, 1.6))
    _g(sc, "uv_sphere", "LampBulb", (x, y, 3.7), .28, "emission_white")
    sc["lights"].append({"name": "LampLight", "type": "POINT", "energy": 600, "loc": [x, y, 3.7], "color": [1, .9, .7], "size": 1})


def gen_rocket(sc, x=0, y=0):
    _g(sc, "cylinder", "RocketBody", (x, y, 3), .6, "plastic_white", (1, 1, 2))
    _g(sc, "cone", "Nosecone", (x, y, 5.6), .6, "plastic_red", (1, 1, 1.3))
    for a in range(3):
        t = a * 2 * math.pi / 3
        _g(sc, "cube", "Fin", (x + math.cos(t) * .8, y + math.sin(t) * .8, 1.2), 1, "plastic_red", (.08, .6, 1.1), (0, 0, t))
    _g(sc, "cone", "Flame", (x, y, .25), .4, "emission_orange", (1, 1, 1.2), (math.pi, 0, 0))


def gen_solar_system(sc):
    _g(sc, "uv_sphere", "Sun", (0, 0, 0), 2, "emission_orange")
    for i, (r, d, m) in enumerate(((.25, 3.5, "iron"), (.4, 5, "sand"), (.45, 7, "plastic_blue"), (.35, 9, "lava"), (1.0, 12.5, "clay"), (.85, 16, "sand"), (.55, 19, "ice"), (.5, 22, "plastic_blue"))):
        _g(sc, "uv_sphere", ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"][i], (d, 0, 0), r, m)
    sc["world"] = {"color": [0, 0, .005], "strength": .2}
    sc["lights"] = [{"name": "SunLight", "type": "POINT", "energy": 200000, "loc": [0, 0, 0], "color": [1, .9, .7], "size": 2}]
    _camera(sc, "wide")
    sc["camera"]["loc"] = [0, -30, 18]
    sc["camera"]["target"] = [10, 0, 0]


def gen_molecule(sc, formula="H2O"):
    sets = {"H2O": [("O", (0, 0, 0), .6, "plastic_red"), ("H", (.95, 0, .6), .35, "plastic_white"), ("H", (-.95, 0, .6), .35, "plastic_white")],
            "CO2": [("C", (0, 0, 0), .55, "plastic_black"), ("O", (1.6, 0, 0), .6, "plastic_red"), ("O", (-1.6, 0, 0), .6, "plastic_red")],
            "CH4": [("C", (0, 0, 0), .6, "plastic_black")] + [("H", p, .35, "plastic_white") for p in ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))],
            "NH3": [("N", (0, 0, 0), .6, "plastic_blue")] + [("H", p, .35, "plastic_white") for p in ((.9, 0, -.5), (-.45, .8, -.5), (-.45, -.8, -.5))]}
    if formula not in sets:
        raise ToolError(f"molecules available: {', '.join(sets)}")
    for el, p, r, m in sets[formula]:
        _g(sc, "uv_sphere", f"Atom_{el}", (p[0], p[1], p[2] + 1.5), r, m)
    c = sets[formula][0][1]
    for el, p, r, m in sets[formula][1:]:
        mid = [(p[i] + c[i]) / 2 for i in range(3)]
        d = math.dist(p, c)
        _g(sc, "cylinder", "Bond", (mid[0], mid[1], mid[2] + 1.5), .08, "chrome", (1, 1, d / .16), (math.acos((p[2] - c[2]) / d), 0, math.atan2(p[1] - c[1], p[0] - c[0]) - math.pi / 2) if d else (0, 0, 0))


def gen_dna(sc, turns=3, per_turn=10):
    for i in range(turns * per_turn):
        a = 2 * math.pi * i / per_turn
        z = i * .3
        _g(sc, "uv_sphere", "StrandA", (math.cos(a), math.sin(a), z), .17, "plastic_blue")
        _g(sc, "uv_sphere", "StrandB", (-math.cos(a), -math.sin(a), z), .17, "plastic_red")
        _g(sc, "cylinder", "Rung", (0, 0, z), .04, "plastic_white", (1, 1, 1), (math.pi / 2, 0, a - math.pi / 2))


def gen_chessboard(sc):
    for i in range(8):
        for j in range(8):
            _g(sc, "cube", "Square", (i - 3.5, j - 3.5, 0), 1, "plastic_white" if (i + j) % 2 else "plastic_black", (1, 1, .12))
    for i in range(8):
        _g(sc, "uv_sphere", "Pawn", (i - 3.5, -2.5, .45), .3, "plastic_white")
        _g(sc, "uv_sphere", "Pawn", (i - 3.5, 2.5, .45), .3, "plastic_black")


def gen_staircase(sc, steps=10):
    for i in range(steps):
        _g(sc, "cube", "Step", (0, i * .5, .15 + i * .3), 1, "concrete", (2, .5, .3 + i * .3 * 0))


def gen_spiral_stairs(sc, steps=16):
    _g(sc, "cylinder", "Pole", (0, 0, 2.4), .15, "iron", (1, 1, 2.4))
    for i in range(steps):
        a = i * math.pi / 8
        _g(sc, "cube", "SpiralStep", (math.cos(a) * 1.0, math.sin(a) * 1.0, .2 + i * .3), 1, "wood", (1.6, .6, .1), (0, 0, a))


def gen_pyramid(sc):
    for i in range(8):
        _g(sc, "cube", "PyramidLayer", (0, 0, .5 + i), 1, "sand", (8 - i, 8 - i, 1))


def gen_city(sc, n=5, seed=3):
    import random
    r = random.Random(seed)
    for i in range(n):
        for j in range(n):
            h = r.choice((1.5, 2.5, 4, 6, 9))
            _g(sc, "cube", "Building", ((i - n / 2) * 3, (j - n / 2) * 3, h / 2), 1, r.choice(("concrete", "brick", "glass", "iron")), (2, 2, h))


def gen_terrain_scene(sc, seed=2):
    _add(sc, "terrain", "Terrain", 0, 0, 0, material="grass", size=1.5)
    sc["objects"][-1]["args"] = {"seed": seed, "amp": 1.4}
    _add(sc, "plane", "Water", 0, 0, -.1, size=14, material="water")


def gen_flower(sc, x=0, y=0):
    _g(sc, "cylinder", "Stem", (x, y, 1), .04, "grass", (1, 1, 1.3))
    _g(sc, "uv_sphere", "Centre", (x, y, 2.05), .2, "gold")
    for i in range(8):
        a = i * math.pi / 4
        _g(sc, "uv_sphere", "Petal", (x + math.cos(a) * .4, y + math.sin(a) * .4, 2.05), .22, "neon_pink", (1, .5, .3), (0, 0, a))


def gen_mushroom(sc, x=0, y=0):
    _g(sc, "cylinder", "MushStem", (x, y, .5), .22, "plastic_white", (1, 1, .5))
    _g(sc, "uv_sphere", "MushCap", (x, y, 1.1), .7, "plastic_red", (1, 1, .6))
    for a in range(5):
        _g(sc, "uv_sphere", "Dot", (x + math.cos(a * 1.3) * .35, y + math.sin(a * 1.3) * .35, 1.45), .09, "plastic_white")


def gen_cactus(sc, x=0, y=0):
    _g(sc, "cylinder", "CactusBody", (x, y, 1), .35, "grass", (1, 1, 1.4))
    for sx in (-1, 1):
        _g(sc, "cylinder", "CactusArm", (x + sx * .55, y, 1.5), .18, "grass", (1, 1, .6))
        _g(sc, "cylinder", "CactusArmUp", (x + sx * .55, y, 1.9), .16, "grass", (1, 1, .6))


def gen_fence(sc, x=0, y=0, n=8):
    for i in range(n):
        _g(sc, "cube", "Post", (x + i * .6, y, .6), 1, "wood", (.12, .12, 1.2))
    for z in (.4, .9):
        _g(sc, "cube", "Rail", (x + (n - 1) * .3, y, z), 1, "wood", (n * .6, .06, .1))


def gen_ferris_wheel(sc):
    _g(sc, "torus", "Wheel", (0, 0, 4.5), 3.5, "chrome", (1, 1, 1), (math.pi / 2, 0, 0))
    for i in range(10):
        a = i * math.pi / 5
        _g(sc, "cube", "Cabin", (math.cos(a) * 3.5, 0, 4.5 + math.sin(a) * 3.5), 1, "plastic_red" if i % 2 else "plastic_blue", (.6, .5, .5))
    for sx in (-.7, .7):
        _g(sc, "cube", "Leg", (sx, 0, 2.2), 1, "iron", (.15, .15, 4.6), (0, .18 * (-1 if sx < 0 else 1), 0))


def gen_atom(sc, electrons=6):
    _g(sc, "uv_sphere", "Nucleus", (0, 0, 1.5), .7, "lava")
    for k in range(3):
        _g(sc, "torus", "Orbit", (0, 0, 1.5), 2.2, "ice", (1, 1, 1), (k * math.pi / 3, math.pi / 2, 0))
    for i in range(electrons):
        a = i * 2 * math.pi / electrons
        _g(sc, "uv_sphere", "Electron", (math.cos(a) * 2.2, math.sin(a) * 2.2, 1.5), .15, "emission_blue")


def gen_ring_planet(sc):
    _g(sc, "uv_sphere", "Planet", (0, 0, 2), 1.6, "sand")
    _g(sc, "torus", "Ring", (0, 0, 2), 2.8, "ice", (1, 1, .06), (.4, 0, 0))
    sc["world"] = {"color": [0, 0, .01], "strength": .3}


def gen_podium(sc):
    for i, h in enumerate((1.4, 1.8, 1.1)):
        _g(sc, "cube", "Podium", ((i - 1) * 1.6, 0, h / 2), 1, ("gold", "silver", "copper")[i] if i != 1 else "gold", (1.5, 1.5, h))


def gen_bookshelf(sc, x=0, y=0):
    for z in (0.1, 1, 1.9, 2.8):
        _g(sc, "cube", "Shelf", (x, y, z), 1, "dark_wood", (2.4, .6, .08))
    for sx in (-1.2, 1.2):
        _g(sc, "cube", "ShelfSide", (x + sx, y, 1.45), 1, "dark_wood", (.08, .6, 2.9))
    for i, z in enumerate((.58, 1.48, 2.38)):
        for j in range(7):
            _g(sc, "cube", "Book", (x - 1 + j * .32, y, z), 1, ("plastic_red", "plastic_blue", "plastic_green", "gold")[(i + j) % 4], (.22, .4, .75))


GENERATORS = {
    "table": (gen_table, "x:float=0, y:float=0, w:float=3, d:float=1.6, h:float=1.5"), "chair": (gen_chair, "x:float=0, y:float=0, rz:float=0"), "house": (gen_house, "x:float=0, y:float=0"),
    "tree": (gen_tree, "x:float=0, y:float=0, s:float=1"), "pine_tree": (gen_pine, "x:float=0, y:float=0"), "snowman": (gen_snowman, "x:float=0, y:float=0"), "robot": (gen_robot, "x:float=0, y:float=0"),
    "car": (gen_car, "x:float=0, y:float=0"), "tower": (gen_tower, "x:float=0, y:float=0, floors:int=6"), "castle": (gen_castle, "x:float=0, y:float=0"), "bridge": (gen_bridge, "x:float=0, y:float=0, span:float=8"),
    "windmill": (gen_windmill, "x:float=0, y:float=0"), "street_lamp": (gen_lamp, "x:float=0, y:float=0"), "rocket": (gen_rocket, "x:float=0, y:float=0"), "solar_system": (gen_solar_system, ""),
    "molecule": (gen_molecule, "formula:str=H2O"), "dna_helix": (gen_dna, "turns:int=3, per_turn:int=10"), "chessboard": (gen_chessboard, ""), "staircase": (gen_staircase, "steps:int=10"),
    "spiral_staircase": (gen_spiral_stairs, "steps:int=16"), "pyramid": (gen_pyramid, ""), "city_blocks": (gen_city, "n:int=5, seed:int=3"), "terrain_with_water": (gen_terrain_scene, "seed:int=2"), "flower": (gen_flower, "x:float=0, y:float=0"),
    "mushroom": (gen_mushroom, "x:float=0, y:float=0"), "cactus": (gen_cactus, "x:float=0, y:float=0"), "fence": (gen_fence, "x:float=0, y:float=0, n:int=8"), "ferris_wheel": (gen_ferris_wheel, ""),
    "atom_model": (gen_atom, "electrons:int=6"), "ringed_planet": (gen_ring_planet, ""), "winners_podium": (gen_podium, ""), "bookshelf": (gen_bookshelf, "x:float=0, y:float=0"),
}
for _n, (_f, _s) in GENERATORS.items():
    reg(f"blender_build_{_n}", f"Build a complete {_n.replace('_', ' ')} out of primitives and materials in the scene.", B, "scene:str=scene" + (", " + _s if _s else ""), (lambda f: _edit(lambda sc, **kw: (f(sc, **kw), {"objects_in_scene": len(sc["objects"])})[1]))(_f), ctx=True, risk="files", produces="scene", tags=("3d", "create", "automation", "model"))


# ------------------------------------------------------------------ scene → Blender script
def _lit(v):
    return repr(v)


def build_script(sc: dict, base: str) -> str:
    L = ["import bpy, math", "from mathutils import Vector", "", "bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()", "SC = bpy.context.scene", f"BASE = {base!r}", "OBJ = {}", "MAT = {}", "",
         "def make_mat(name, color, metallic, roughness, extra):",
         "    m = bpy.data.materials.new(name); m.use_nodes = True",
         "    b = m.node_tree.nodes.get('Principled BSDF')",
         "    def put(names, value):",
         "        for n in names:",
         "            if n in b.inputs: b.inputs[n].default_value = value; return",
         "    put(['Base Color'], (*color, 1)); put(['Metallic'], metallic); put(['Roughness'], roughness)",
         "    if 'transmission' in extra: put(['Transmission Weight', 'Transmission'], extra['transmission'])",
         "    if 'ior' in extra: put(['IOR'], extra['ior'])",
         "    if 'emission' in extra: put(['Emission Color', 'Emission'], (*extra['emission'], 1)); put(['Emission Strength'], extra.get('emission_strength', 1.0))",
         "    return m", ""]
    for n, m in sc["materials"].items():
        extra = {k: v for k, v in m.items() if k in ("transmission", "ior", "emission", "emission_strength")}
        L.append(f"MAT[{n!r}] = make_mat({n!r}, {m['color']}, {m['metallic']}, {m['roughness']}, {extra!r})")
    L += ["", "def finish(o, name, loc, rot, scale, material, smooth):",
          "    o.name = name; o.location = loc; o.rotation_euler = rot; o.scale = scale",
          "    if material and hasattr(o.data, 'materials'): o.data.materials.append(MAT[material])",
          "    if smooth and o.type == 'MESH':",
          "        for p in o.data.polygons: p.use_smooth = True",
          "    OBJ[name] = o", "",
          "def raw_mesh(name, verts, faces):",
          "    me = bpy.data.meshes.new(name); me.from_pydata(verts, [], faces); me.update()",
          "    o = bpy.data.objects.new(name, me); bpy.context.collection.objects.link(o); return o", ""]
    for o in sc["objects"]:
        k, s = o["kind"], o["size"]
        if k in CUSTOM:
            kw = o.get("args", {})
            m = CUSTOM[k](**kw) if kw else CUSTOM[k]()
            m = m.transform(scale=s)
            L.append(f"_o = raw_mesh({o['name']!r}, {[tuple(round(c, 5) for c in v) for v in m.v]!r}, {m.f!r})")
        else:
            L.append(BPY_PRIMS[k].format(s=s))
            L.append("_o = bpy.context.active_object")
            if k == "text":
                L.append(f"_o.data.body = {o['text']!r}; _o.data.size = {s}")
        L.append(f"finish(_o, {o['name']!r}, {o['loc']}, {o['rot']}, {o['scale']}, {o['material']!r}, {o['smooth']})")
    for o in sc["objects"]:
        if o.get("parent"):
            L.append(f"OBJ[{o['name']!r}].parent = OBJ[{o['parent']!r}]")
    for o in sc["objects"]:
        for m in o["modifiers"]:
            L.append(f"_m = OBJ[{o['name']!r}].modifiers.new({m['name']!r}, {m['type']!r})")
            for pk, pv in m["props"].items():
                if pk in ("operand_object", "target"):
                    if pv:
                        L.append(f"_m.{pk} = OBJ[{pv!r}]")
                else:
                    L.append(f"try: _m.{pk} = {pv!r}\nexcept Exception as e: print('modifier property', {pk!r}, 'skipped:', e)")
    L.append("")
    for li in sc["lights"]:
        L.append(f"_d = bpy.data.lights.new({li['name']!r}, {li['type']!r}); _d.energy = {li['energy']}; _d.color = {tuple(li['color'])}")
        if li["type"] == "AREA":
            L.append(f"_d.size = {li.get('size', 2)}")
        L.append(f"_l = bpy.data.objects.new({li['name']!r}, _d); bpy.context.collection.objects.link(_l); _l.location = {li['loc']}")
        if "rot" in li:
            L.append(f"_l.rotation_euler = {li['rot']}")
        else:
            L.append(f"_l.rotation_euler = (Vector({li.get('target', [0, 0, 0.5])}) - _l.location).to_track_quat('-Z', 'Y').to_euler()")
    c = sc["camera"]
    L += [f"_cd = bpy.data.cameras.new('Camera'); _cd.lens = {c['lens']}", "CAM = bpy.data.objects.new('Camera', _cd); bpy.context.collection.objects.link(CAM)", f"CAM.location = {c['loc']}",
          f"CAM.rotation_euler = (Vector({c['target']}) - CAM.location).to_track_quat('-Z', 'Y').to_euler()", "SC.camera = CAM"]
    w = sc["world"]
    L += ["if SC.world is None: SC.world = bpy.data.worlds.new('World')", "SC.world.use_nodes = True", "_bg = SC.world.node_tree.nodes.get('Background')",
          f"_bg.inputs[0].default_value = ({w['color'][0]}, {w['color'][1]}, {w['color'][2]}, 1); _bg.inputs[1].default_value = {w['strength']}"]
    r, fr = sc["render"], sc["frames"]
    eng = {"EEVEE": "BLENDER_EEVEE_NEXT", "CYCLES": "CYCLES", "WORKBENCH": "BLENDER_WORKBENCH"}.get(r["engine"], "BLENDER_EEVEE_NEXT")
    L += [f"try: SC.render.engine = {eng!r}", "except TypeError: SC.render.engine = 'BLENDER_EEVEE'",
          f"SC.render.resolution_x, SC.render.resolution_y, SC.render.resolution_percentage = {r['width']}, {r['height']}, 100",
          f"SC.frame_start, SC.frame_end, SC.render.fps = {fr['start']}, {fr['end']}, {fr['fps']}",
          f"if SC.render.engine == 'CYCLES': SC.cycles.samples = {r['samples']}",
          f"SC.render.film_transparent = {bool(r['transparent'])}"]
    if r["format"] == "MP4":
        L += ["SC.render.image_settings.file_format = 'FFMPEG'; SC.render.ffmpeg.format = 'MPEG4'; SC.render.ffmpeg.codec = 'H264'"]
    else:
        L.append(f"SC.render.image_settings.file_format = {r['format']!r}")
    L.append(f"SC.render.filepath = BASE + '/' + {r['output']!r}")
    # animation
    for o in sc["objects"]:
        for k in o["keys"]:
            L.append(f"_o = OBJ[{o['name']!r}]; _o.location = {k['loc']}; _o.rotation_euler = {k['rot']}; _o.scale = {k['scale']}; "
                     f"[_o.keyframe_insert(p, frame={k['frame']}) for p in ('location', 'rotation_euler', 'scale')]")
        if o["keys"]:
            L.append("try:\n    for fc in OBJ[%r].animation_data.action.fcurves:\n        for kp in fc.keyframe_points: kp.interpolation = 'LINEAR'\nexcept Exception: pass" % o["name"])
    orb = c.get("orbit")
    if orb:
        L += ["_pivot = bpy.data.objects.new('Pivot', None); bpy.context.collection.objects.link(_pivot)", f"_pivot.location = {c['target']}", "CAM.parent = _pivot; CAM.matrix_parent_inverse = _pivot.matrix_world.inverted()",
              f"_pivot.rotation_euler = (0, 0, 0); _pivot.keyframe_insert('rotation_euler', frame=1)", f"_pivot.rotation_euler = (0, 0, 2 * math.pi); _pivot.keyframe_insert('rotation_euler', frame={orb['frames']})",
              "try:\n    for fc in _pivot.animation_data.action.fcurves:\n        for kp in fc.keyframe_points: kp.interpolation = 'LINEAR'\nexcept Exception: pass"]
    for i, p in enumerate(c.get("path", [])):
        L.append(f"CAM.location = {p}; CAM.keyframe_insert('location', frame={1 + round(i * (fr['end'] - 1) / max(1, len(c['path']) - 1))})")
    L.append("")
    L.append("if " + ("True" if r["animation"] else "False") + ": bpy.ops.render.render(animation=True)")
    L.append("elif '--no-render' not in __import__('sys').argv: bpy.ops.render.render(write_still=True)")
    for e in sc["export"]:
        f, pth = e["format"], f"BASE + '/' + {e['path']!r}"
        call = {"glb": f"bpy.ops.export_scene.gltf(filepath={pth}, export_format='GLB')", "obj": f"bpy.ops.wm.obj_export(filepath={pth})", "stl": f"bpy.ops.wm.stl_export(filepath={pth})", "fbx": f"bpy.ops.export_scene.fbx(filepath={pth})",
                "usd": f"bpy.ops.wm.usd_export(filepath={pth})", "blend": f"bpy.ops.wm.save_as_mainfile(filepath={pth})"}[f]
        L.append(f"try: {call}\nexcept Exception as e: print('export {f} failed:', e)")
    return "\n".join(L) + "\n"


def scene_meshes(sc: dict) -> list[mesh.Mesh]:
    """The scene as coloured meshes for the software preview (arrays and mirrors applied; other modifiers are shown only in Blender)."""
    out = []
    for o in sc["objects"]:
        k, s = o["kind"], o["size"]
        if k in CUSTOM:
            m = CUSTOM[k](**o.get("args", {})) if o.get("args") else CUSTOM[k]()
            m = m.transform(scale=s)
        elif k in PREVIEW:
            m = PREVIEW[k](s)
        else:
            continue
        mat = sc["materials"].get(o["material"] or "", None)
        if mat:
            m.color = "#%02x%02x%02x" % tuple(round(min(1, c) ** (1 / 2.2) * 255) for c in mat["color"])
            if mat.get("emission_strength"):
                m.color = "#%02x%02x%02x" % tuple(min(255, int(c * 255 + 40)) for c in mat["emission"])
        pose = o["keys"][0] if o["keys"] else None
        loc, rot, scl = (pose["loc"], pose["rot"], pose["scale"]) if pose else (o["loc"], o["rot"], o["scale"])
        m = m.transform(loc, rot, scl)
        for md in o["modifiers"]:
            if md["type"] == "ARRAY":
                n, d = int(md["props"].get("count", 2)), md["props"].get("relative_offset_displace", [1, 0, 0])
                ext = [max(p[i] for p in m.v) - min(p[i] for p in m.v) for i in range(3)]
                base = m.copy()
                for i in range(1, min(n, 40)):
                    m.add(base.transform(tuple(d[j] * ext[j] * i for j in range(3))))
            elif md["type"] == "MIRROR":
                ax = md["props"].get("use_axis", [True, False, False])
                base = m.copy()
                m.add(Mesh_flip(base, ax))
        out.append(m)
    return out


def Mesh_flip(m, ax):
    return mesh.Mesh([tuple(-c if ax[i] else c for i, c in enumerate(v)) for v in m.v], [tuple(reversed(f)) for f in m.f], m.color)


def _build(ctx, scene, out=None):
    sc = load_scene(ctx, scene)
    p = ctx.path(out or f"scenes/{scene.removesuffix('.json')}.py")
    p.write_text(build_script(sc, str(ctx.workspace.resolve())), encoding="utf-8")
    try:
        compile(p.read_text(), str(p), "exec")
    except SyntaxError as e:
        raise ToolError(f"generated script has a syntax error: {e}")
    return {"saved": ctx.produced(p, "script", "bpy script"), "objects": len(sc["objects"]), "run_with": f"blender -b -P {p.name}"}


reg("blender_build_script", "Write the Blender Python (bpy) script for a scene. Run it with `blender -b -P scene.py`, or paste it into Blender's Scripting tab.", B, "scene:str=scene, out:str?", _build, ctx=True, risk="files", produces="script", tags=("python", "bpy", "automation"))


def _framed(sc: dict) -> list:
    """Meshes to show in a preview: everything except the big ground plane, so the objects fill the picture."""
    pairs = list(zip(sc["objects"], scene_meshes(sc)))
    keep = [m for o, m in pairs if not (o["name"] == "Ground" and len(pairs) > 1)]
    if not keep:
        raise ToolError("the scene has no previewable objects yet")
    return keep


def _preview(ctx, scene, out, yaw, pitch, w, h, png):
    sc = load_scene(ctx, scene)
    ms = _framed(sc)
    o = ctx.path(out or f"renders/{scene}_preview.{'png' if png else 'svg'}")
    bg = "#%02x%02x%02x" % tuple(round(min(1, c) ** (1 / 2.2) * 255 * .6 + 90) for c in sc["world"]["color"])
    if png:
        mesh.render_pil(ms, yaw, pitch, (w, h), bg).save(o)
    else:
        o.write_text(mesh.render_svg(ms, yaw, pitch, (w, h), bg), encoding="utf-8")
    return {"saved": ctx.produced(o, "image", "scene preview (software renderer, not Blender)"), "objects": len(sc["objects"]), "note": "Preview shows shapes, colours, arrays and mirrors; lights, glass, subdivision and other modifiers need real Blender."}


reg("blender_preview_png", "Quick PNG picture of a scene using the built-in software renderer (no Blender needed).", B, "scene:str=scene, out:str?, yaw:float=35, pitch:float=28, width:int=800, height:int=520", lambda ctx, scene, out, yaw, pitch, width, height: _preview(ctx, scene, out, yaw, pitch, width, height, True), ctx=True, risk="files", produces="image", needs=("py:PIL",), tags=("render", "image"))
reg("blender_preview_svg", "Quick SVG picture of a scene using the built-in software renderer (no Blender needed).", B, "scene:str=scene, out:str?, yaw:float=35, pitch:float=28, width:int=800, height:int=520", lambda ctx, scene, out, yaw, pitch, width, height: _preview(ctx, scene, out, yaw, pitch, width, height, False), ctx=True, risk="files", produces="image", tags=("render", "image"))


def _gif(ctx, scene, out, frames, width, height):
    ms = _framed(load_scene(ctx, scene))
    ims = [mesh.render_pil(ms, 360 * i / frames, 25, (width, height)) for i in range(min(frames, 60))]
    o = ctx.path(out or f"renders/{scene}_turntable.gif")
    ims[0].save(o, save_all=True, append_images=ims[1:], duration=70, loop=0, optimize=True)
    return {"saved": ctx.produced(o, "image", f"{len(ims)}-frame turntable")}


reg("blender_preview_turntable_gif", "Animated turntable GIF of a scene with the built-in software renderer.", B, "scene:str=scene, out:str?, frames:int=36, width:int=480, height:int=320", _gif, ctx=True, risk="files", produces="image", needs=("py:PIL",), tags=("animation", "video", "render"))
reg("blender_export_mesh_preview", "Export the scene's shapes to a single OBJ using the pure-Python mesher (no Blender needed; opens in Blender, MeshLab…).", B, "scene:str=scene, out:str?",
    lambda ctx, scene, out: mesh.save_mesh(ctx, (lambda ms: [ms[0].add(m) for m in ms[1:]][-1] if len(ms) > 1 else ms[0])([m.copy() for m in scene_meshes(load_scene(ctx, scene))]), out or f"exports/{scene}.obj"), ctx=True, risk="files", produces="model")
reg("blender_scene_summary", "List the objects, materials, lights, camera and render settings of a scene.", B, "scene:str=scene", lambda ctx, scene: (lambda sc: {"objects": [f"{o['name']} ({o['kind']}) at {[round(v, 2) for v in o['loc']]}" + (f" material={o['material']}" if o["material"] else "") + (f" modifiers={[m['type'] for m in o['modifiers']]}" if o["modifiers"] else "") + (f" keys={len(o['keys'])}" if o["keys"] else "") for o in sc["objects"]][:80],
                                                                                                          "count": len(sc["objects"]), "lights": [l["name"] for l in sc["lights"]], "camera": sc["camera"], "render": sc["render"], "frames": sc["frames"]})(load_scene(ctx, scene)), ctx=True, risk="files")
reg("blender_list_scenes", "Scenes saved in the workspace.", B, "", lambda ctx: sorted(p.stem for p in (ctx.workspace / "scenes").glob("*.json")) if (ctx.workspace / "scenes").exists() else [], ctx=True, risk="files")
reg("blender_list_materials", "Material presets available.", B, "", lambda: sorted(MATERIALS))
reg("blender_list_generators", "Ready-made things blender_build_* can make.", B, "", lambda: sorted(GENERATORS))
reg("blender_list_modifiers", "Modifiers available (with what each does).", B, "", lambda: {k.lower(): v[1] for k, v in MODIFIERS.items()})
reg("blender_list_primitives", "Shapes blender_add_* can add.", B, "", lambda: sorted(list(BPY_PRIMS) + list(CUSTOM)))
reg("blender_delete_scene", "Delete a scene file.", B, "scene:str", lambda ctx, scene: (ctx.path(f"scenes/{scene}.json", True).unlink(), "deleted")[1], ctx=True, risk="files")

# ---- tools that need real Blender
reg("blender_version", "Version of the installed Blender.", B, "", lambda ctx: _blender(ctx, ["--version"]).splitlines()[0], ctx=True, risk="exec", needs=("bin:blender",))


def _blender(ctx, args, timeout=None):
    try:
        r = subprocess.run(["blender", *args], cwd=ctx.workspace, capture_output=True, text=True, timeout=timeout or max(30, ctx.timeout * 5), stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise ToolError("Blender timed out — lower the resolution/samples or use EEVEE")
    out = (r.stdout + r.stderr)
    if r.returncode:
        raise ToolError("Blender failed:\n" + "\n".join(out.splitlines()[-12:]))
    return out


def _render_real(ctx, scene, no_render=False):
    sc = load_scene(ctx, scene)
    script = ctx.path(f"scenes/{scene.removesuffix('.json')}.py")
    script.write_text(build_script(sc, str(ctx.workspace.resolve())), encoding="utf-8")
    out = _blender(ctx, ["-b", "-noaudio", "-P", str(script)] + (["--", "--no-render"] if no_render else []))
    ctx.path("renders/x").parent  # ensure the folder exists
    found = []
    for p in sorted((ctx.workspace / "renders").glob(f"{_stem(sc['render']['output'])}*")) + [ctx.workspace / e["path"] for e in sc["export"]]:
        if p.exists():
            found.append(ctx.produced(p, "video" if p.suffix in (".mp4", ".mkv") else "model" if p.suffix in (".glb", ".obj", ".stl", ".fbx", ".usd", ".blend") else "image", "Blender render"))
    return {"files": found, "log_tail": out.strip().splitlines()[-3:]}


def _stem(s):
    return s.split("/")[-1]


reg("blender_render", "Render a scene with real Blender (still image, or video if the scene is animated). Slow for Cycles.", B, "scene:str=scene", lambda ctx, scene: _render_real(ctx, scene), ctx=True, risk="exec", needs=("bin:blender",), produces="image", tags=("render", "video", "image"))
reg("blender_run_script", "Run any Blender Python script from the workspace headlessly.", B, "path:str", lambda ctx, path: _blender(ctx, ["-b", "-noaudio", "-P", str(ctx.path(path, True))])[-3000:], ctx=True, risk="exec", needs=("bin:blender",), tags=("python", "bpy"))
reg("blender_run_code", "Run a snippet of bpy code in a headless Blender (the default scene is cleared first).", B, "code:str", lambda ctx, code: (lambda p: (p.write_text("import bpy\n" + code), _blender(ctx, ["-b", "-noaudio", "-P", str(p)])[-3000:])[1])(ctx.path("scenes/_snippet.py")), ctx=True, risk="exec", needs=("bin:blender",), tags=("python", "bpy"))
reg("blender_convert_model", "Convert a model file (obj, fbx, stl, glb) to glb/obj/stl/fbx using Blender.", B, "path:str, out:str", lambda ctx, path, out: _convert(ctx, path, out), ctx=True, risk="exec", needs=("bin:blender",), produces="model")


def _convert(ctx, path, out):
    src, dst = ctx.path(path, True), ctx.path(out)
    imp = {".obj": "bpy.ops.wm.obj_import(filepath=S)", ".stl": "bpy.ops.wm.stl_import(filepath=S)", ".fbx": "bpy.ops.import_scene.fbx(filepath=S)", ".glb": "bpy.ops.import_scene.gltf(filepath=S)", ".gltf": "bpy.ops.import_scene.gltf(filepath=S)"}
    exp = {".obj": "bpy.ops.wm.obj_export(filepath=D)", ".stl": "bpy.ops.wm.stl_export(filepath=D)", ".fbx": "bpy.ops.export_scene.fbx(filepath=D)", ".glb": "bpy.ops.export_scene.gltf(filepath=D, export_format='GLB')"}
    if src.suffix.lower() not in imp or dst.suffix.lower() not in exp:
        raise ToolError("supported: import obj/stl/fbx/glb/gltf, export obj/stl/fbx/glb")
    s = ctx.path("scenes/_convert.py")
    s.write_text(f"import bpy\nS={str(src)!r}; D={str(dst)!r}\nbpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()\n{imp[src.suffix.lower()]}\n{exp[dst.suffix.lower()]}\n")
    _blender(ctx, ["-b", "-noaudio", "-P", str(s)])
    return {"saved": ctx.produced(dst, "model")}
