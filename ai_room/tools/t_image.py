"""Images with Pillow: create, edit, filter, resize for social media, generate procedural art, compose posters, collages and GIFs."""

from __future__ import annotations

import colorsys
import math
import random

from .registry import Context, ToolError, reg

I = "image"
NEED = ("py:PIL",)


def _pil():
    from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
    return Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


def _open(ctx: Context, path: str):
    Image = _pil()[0]
    try:
        im = Image.open(ctx.path(path, True))
        im.load()
    except OSError as e:
        raise ToolError(f"cannot open '{path}' as an image: {e}")
    return im


def _out(ctx: Context, im, out: str | None, default: str, note: str = "") -> dict:
    p = ctx.path(out or default)
    if p.suffix.lower() in (".jpg", ".jpeg") and im.mode in ("RGBA", "P", "LA"):
        im = im.convert("RGB")
    if not p.suffix:
        p = p.with_suffix(".png")
    im.save(p)
    return {"saved": ctx.produced(p, "image", note), "size": list(im.size)}


def _stem(path: str) -> str:
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def _rgb(c, default=(0, 0, 0)):
    if c in (None, ""):
        return default
    if isinstance(c, (list, tuple)):
        return tuple(int(x) for x in c[:3])
    c = str(c).strip().lstrip("#")
    names = {"white": "ffffff", "black": "000000", "red": "e53935", "green": "43a047", "blue": "1e88e5", "yellow": "fdd835", "orange": "fb8c00", "purple": "8e24aa", "pink": "ec407a", "gray": "9e9e9e", "grey": "9e9e9e", "cyan": "00acc1", "gold": "ffc107", "navy": "0d1b3e", "teal": "00897b"}
    c = names.get(c.lower(), c)
    if len(c) == 3:
        c = "".join(x * 2 for x in c)
    try:
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        raise ToolError("colour must look like #1a2b3c or be a name such as red")


def font(size: int, bold: bool = True):
    ImageFont = _pil()[5]
    for name in (("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf") if bold else ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def wrap(draw, text: str, fnt, width: int) -> list[str]:
    lines = []
    for para in str(text).split("\n"):
        cur = ""
        for w in para.split():
            t = (cur + " " + w).strip()
            if draw.textlength(t, font=fnt) <= width or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


# ------------------------------------------------------------------ filters, adjustments, geometry (table-driven)
def _filters():
    F = _pil()[4]
    return {"blur": F.BLUR, "gaussian_blur": F.GaussianBlur(4), "box_blur": F.BoxBlur(4), "sharpen": F.SHARPEN, "smooth": F.SMOOTH, "smooth_more": F.SMOOTH_MORE, "detail": F.DETAIL, "edge_enhance": F.EDGE_ENHANCE,
            "edge_enhance_more": F.EDGE_ENHANCE_MORE, "contour": F.CONTOUR, "emboss": F.EMBOSS, "find_edges": F.FIND_EDGES, "min_filter": F.MinFilter(3), "max_filter": F.MaxFilter(3), "median_filter": F.MedianFilter(5),
            "mode_filter": F.ModeFilter(5), "unsharp_mask": F.UnsharpMask(radius=2, percent=150)}


FILTER_NAMES = ["blur", "gaussian_blur", "box_blur", "sharpen", "smooth", "smooth_more", "detail", "edge_enhance", "edge_enhance_more", "contour", "emboss", "find_edges", "min_filter", "max_filter", "median_filter", "mode_filter", "unsharp_mask"]


def _apply_filter(ctx, path, out, name, radius):
    F = _pil()[4]
    im = _open(ctx, path).convert("RGB")
    f = _filters()[name]
    if radius and name in ("gaussian_blur", "box_blur"):
        f = F.GaussianBlur(radius) if name == "gaussian_blur" else F.BoxBlur(radius)
    return _out(ctx, im.filter(f), out, f"images/{_stem(path)}_{name}.png", name)


for _f in FILTER_NAMES:
    reg(f"image_filter_{_f}", f"Apply the {_f.replace('_', ' ')} filter to an image.", I, "path:str, out:str?, radius:float?", (lambda f: lambda ctx, path, out, radius: _apply_filter(ctx, path, out, f, radius))(_f), ctx=True, risk="files", needs=NEED, produces="image", tags=("photo", "effect"))


def _ops(name):
    Image, Chops, Draw, Enh, Fl, Fo, Ops = _pil()
    return {
        "grayscale": lambda im, v: Ops.grayscale(im.convert("RGB")), "invert": lambda im, v: Ops.invert(im.convert("RGB")), "autocontrast": lambda im, v: Ops.autocontrast(im.convert("RGB"), cutoff=v or 1),
        "equalize": lambda im, v: Ops.equalize(im.convert("RGB")), "posterize": lambda im, v: Ops.posterize(im.convert("RGB"), int(v or 3)), "solarize": lambda im, v: Ops.solarize(im.convert("RGB"), int(v or 128)),
        "mirror": lambda im, v: Ops.mirror(im), "flip": lambda im, v: Ops.flip(im), "sepia": lambda im, v: Ops.colorize(Ops.grayscale(im.convert("RGB")), "#2b1a0a", "#f5deb3", mid="#a67c52"),
        "threshold": lambda im, v: Ops.grayscale(im.convert("RGB")).point(lambda p: 255 if p > (v or 128) else 0), "duotone": lambda im, v: Ops.colorize(Ops.grayscale(im.convert("RGB")), "#1b2a6b", "#ffb86b"),
        "pixelate": lambda im, v: im.resize((max(1, im.width // int(v or 12)), max(1, im.height // int(v or 12))), Image.BILINEAR).resize(im.size, Image.NEAREST),
        "vignette": lambda im, v: _vignette(im.convert("RGB")), "halftone": lambda im, v: _halftone(im, int(v or 8)), "glitch": lambda im, v: _glitch(im.convert("RGB")),
        "tilt_shift": lambda im, v: _tilt(im.convert("RGB")), "warm": lambda im, v: _tint(im.convert("RGB"), (1.1, 1.0, 0.88)), "cool": lambda im, v: _tint(im.convert("RGB"), (0.9, 1.0, 1.12)),
        "vintage": lambda im, v: _vignette(_tint(Enh.Color(im.convert("RGB")).enhance(.7), (1.08, 1.0, .85))), "high_contrast_bw": lambda im, v: Ops.autocontrast(Ops.grayscale(im.convert("RGB")), cutoff=5),
        "neon_edges": lambda im, v: Ops.colorize(Ops.grayscale(im.convert("RGB").filter(Fl.FIND_EDGES)), "#000000", "#39ff14"), "oil_paint": lambda im, v: im.convert("RGB").filter(Fl.ModeFilter(7)).filter(Fl.SMOOTH),
        "mosaic": lambda im, v: im.resize((max(1, im.width // 20), max(1, im.height // 20)), Image.BOX).resize(im.size, Image.NEAREST), "swap_rb": lambda im, v: Image.merge("RGB", im.convert("RGB").split()[::-1]),
    }[name]


def _vignette(im):
    Image, _, Draw, *_ = _pil()
    w, h = im.size
    m = Image.new("L", (w, h), 0)
    Draw.Draw(m).ellipse((-w * .15, -h * .15, w * 1.15, h * 1.15), fill=255)
    m = m.filter(_pil()[4].GaussianBlur(min(w, h) / 6))
    return Image.composite(im, Image.new("RGB", (w, h), (0, 0, 0)), m)


def _tint(im, k):
    r, g, b = im.split()
    return _pil()[0].merge("RGB", (r.point(lambda p: min(255, int(p * k[0]))), g.point(lambda p: min(255, int(p * k[1]))), b.point(lambda p: min(255, int(p * k[2])))))


def _halftone(im, step):
    Image, _, Draw, *_ = _pil()
    g = im.convert("L")
    out = Image.new("L", g.size, 255)
    d = Draw.Draw(out)
    for y in range(0, g.height, step):
        for x in range(0, g.width, step):
            v = 1 - g.getpixel((min(x + step // 2, g.width - 1), min(y + step // 2, g.height - 1))) / 255
            r = v * step * .72
            d.ellipse((x + step / 2 - r, y + step / 2 - r, x + step / 2 + r, y + step / 2 + r), fill=0)
    return out


def _glitch(im, seed=3):
    r = random.Random(seed)
    out = im.copy()
    for _ in range(14):
        y, h = r.randrange(im.height), r.randrange(4, max(5, im.height // 12))
        dx = r.randrange(-im.width // 10, im.width // 10 + 1)
        out.paste(im.crop((0, y, im.width, min(im.height, y + h))), (dx, y))
    R, G, B = out.split()
    return _pil()[0].merge("RGB", (R.transform(out.size, _pil()[0].AFFINE, (1, 0, 6, 0, 1, 0)), G, B.transform(out.size, _pil()[0].AFFINE, (1, 0, -6, 0, 1, 0))))


def _tilt(im):
    Image = _pil()[0]
    w, h = im.size
    blur = im.filter(_pil()[4].GaussianBlur(6))
    m = Image.linear_gradient("L").resize((w, h))
    m = m.point(lambda p: int(255 * min(1, abs(p - 128) / 90)))
    return Image.composite(blur, im, m)


OP_NAMES = ["grayscale", "invert", "autocontrast", "equalize", "posterize", "solarize", "mirror", "flip", "sepia", "threshold", "duotone", "pixelate", "vignette", "halftone", "glitch", "tilt_shift", "warm", "cool", "vintage", "high_contrast_bw", "neon_edges", "oil_paint", "mosaic", "swap_rb"]
for _o in OP_NAMES:
    reg(f"image_{_o}", f"{_o.replace('_', ' ').capitalize()} effect on an image (optional strength).", I, "path:str, out:str?, strength:float?", (lambda o: lambda ctx, path, out, strength: _out(ctx, _ops(o)(_open(ctx, path), strength), out, f"images/{_stem(path)}_{o}.png", o))(_o), ctx=True, risk="files", needs=NEED, produces="image", tags=("photo", "effect", "filter", "instagram"))

for _n, _cls in (("brightness", "Brightness"), ("contrast", "Contrast"), ("saturation", "Color"), ("sharpness", "Sharpness")):
    reg(f"image_adjust_{_n}", f"Change the {_n} of an image (1 = unchanged, 1.5 = +50%, 0.5 = −50%).", I, "path:str, factor:float=1.3, out:str?", (lambda c, n: lambda ctx, path, factor, out: _out(ctx, getattr(_pil()[3], c)(_open(ctx, path).convert("RGB")).enhance(factor), out, f"images/{_stem(path)}_{n}.png", n))(_cls, _n), ctx=True, risk="files", needs=NEED, produces="image", tags=("photo", "edit"))


def _resize(ctx, path, width, height, out, keep_ratio):
    im = _open(ctx, path)
    w, h = width or im.width, height or im.height
    if keep_ratio:
        k = min(w / im.width, h / im.height) if width and height else (w / im.width if width else h / im.height)
        w, h = max(1, round(im.width * k)), max(1, round(im.height * k))
    return _out(ctx, im.resize((w, h), _pil()[0].LANCZOS), out, f"images/{_stem(path)}_{w}x{h}.png", "resized")


def _fill(im, w, h):
    Image = _pil()[0]
    k = max(w / im.width, h / im.height)
    r = im.resize((max(w, round(im.width * k)), max(h, round(im.height * k))), Image.LANCZOS)
    x, y = (r.width - w) // 2, (r.height - h) // 2
    return r.crop((x, y, x + w, y + h))


reg("image_info", "Size, format, mode and EXIF-free summary of an image.", I, "path:str", lambda ctx, path: (lambda im: {"width": im.width, "height": im.height, "mode": im.mode, "format": im.format, "megapixels": round(im.width * im.height / 1e6, 2), "aspect": round(im.width / im.height, 3), "frames": getattr(im, "n_frames", 1)})(_open(ctx, path)), ctx=True, risk="files", needs=NEED)
reg("image_resize", "Resize an image (give width and/or height; keep_ratio fits inside the box).", I, "path:str, width:int?, height:int?, out:str?, keep_ratio:bool=true", _resize, ctx=True, risk="files", needs=NEED, produces="image")
reg("image_crop", "Crop a box (left, top, right, bottom in pixels).", I, "path:str, left:int, top:int, right:int, bottom:int, out:str?", lambda ctx, path, left, top, right, bottom, out: _out(ctx, _open(ctx, path).crop((left, top, right, bottom)), out, f"images/{_stem(path)}_crop.png", "cropped"), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_crop_square", "Crop the centre square.", I, "path:str, out:str?", lambda ctx, path, out: (lambda im: _out(ctx, _fill(im, min(im.size), min(im.size)), out, f"images/{_stem(path)}_square.png"))(_open(ctx, path)), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_rotate", "Rotate by degrees counter-clockwise.", I, "path:str, degrees:float=90, out:str?, expand:bool=true", lambda ctx, path, degrees, out, expand: _out(ctx, _open(ctx, path).convert("RGBA").rotate(degrees, expand=expand, resample=_pil()[0].BICUBIC), out, f"images/{_stem(path)}_rot.png"), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_pad", "Add a coloured border or padding.", I, "path:str, pixels:int=20, color:str=#ffffff, out:str?", lambda ctx, path, pixels, color, out: _out(ctx, _pil()[6].expand(_open(ctx, path).convert("RGB"), pixels, _rgb(color)), out, f"images/{_stem(path)}_pad.png"), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_round_corners", "Round the corners (transparent PNG).", I, "path:str, radius:int=40, out:str?", lambda ctx, path, radius, out: _round(ctx, path, radius, out), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_circle_crop", "Crop to a circle (avatar).", I, "path:str, out:str?", lambda ctx, path, out: _round(ctx, path, 10 ** 6, out, True), ctx=True, risk="files", needs=NEED, produces="image", tags=("instagram", "avatar"))
reg("image_drop_shadow", "Add a soft drop shadow behind an image.", I, "path:str, offset:int=12, blur:int=14, out:str?", lambda ctx, path, offset, blur, out: _shadow(ctx, path, offset, blur, out), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_convert_format", "Convert to png, jpg, webp, gif, bmp or tiff.", I, "path:str, format:str=png", lambda ctx, path, format: _out(ctx, _open(ctx, path), None, f"images/{_stem(path)}.{format.lower().lstrip('.')}", f"converted to {format}"), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_to_ascii", "Turn an image into ASCII art text.", I, "path:str, columns:int=80", lambda ctx, path, columns: _ascii(_open(ctx, path), columns), ctx=True, risk="files", needs=NEED)
reg("image_palette", "The dominant colours of an image as hex codes.", I, "path:str, count:int=6", lambda ctx, path, count: _palette(_open(ctx, path), count), ctx=True, risk="files", needs=NEED)
reg("image_average_color", "Average colour of an image.", I, "path:str", lambda ctx, path: "#%02x%02x%02x" % _open(ctx, path).convert("RGB").resize((1, 1), _pil()[0].BOX).getpixel((0, 0)), ctx=True, risk="files", needs=NEED)
reg("image_histogram_chart", "Colour histogram of an image, as a PNG chart.", I, "path:str, out:str?", lambda ctx, path, out: _hist(ctx, path, out), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_compare", "How different are two images (0 = identical)?", I, "a:str, b:str", lambda ctx, a, b: (lambda d: {"mean_difference": round(sum(i * c for i, c in enumerate(d.histogram()[:256])) / max(1, d.width * d.height), 3)})(_pil()[1].difference(_open(ctx, a).convert("L"), _open(ctx, b).convert("L").resize(_open(ctx, a).size))), ctx=True, risk="files", needs=NEED)
reg("image_remove_color", "Make a colour (e.g. a green screen or white background) transparent.", I, "path:str, color:str=#ffffff, tolerance:int=40, out:str?", lambda ctx, path, color, tolerance, out: _chroma(ctx, path, color, tolerance, out), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_overlay_text", "Write text on an image (position: top, center or bottom).", I, "path:str, text:str, position:str=bottom, size:int=0, color:str=#ffffff, out:str?", lambda ctx, path, text, position, size, color, out: _caption(ctx, path, text, position, size, color, out), ctx=True, risk="files", needs=NEED, produces="image", tags=("meme", "caption"))
reg("image_watermark", "Stamp a semi-transparent text watermark in the corner.", I, "path:str, text:str, out:str?", lambda ctx, path, text, out: _watermark(ctx, path, text, out), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_overlay_image", "Paste one image on another (position x,y; scale as a fraction of the base width).", I, "base:str, overlay:str, x:int=10, y:int=10, scale:float=0.25, opacity:float=1.0, out:str?", lambda ctx, base, overlay, x, y, scale, opacity, out: _overlay(ctx, base, overlay, x, y, scale, opacity, out), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_blend", "Cross-fade two images (alpha 0–1).", I, "a:str, b:str, alpha:float=0.5, out:str?", lambda ctx, a, b, alpha, out: (lambda ia: _out(ctx, _pil()[0].blend(ia, _open(ctx, b).convert("RGB").resize(ia.size), alpha), out, "images/blend.png"))(_open(ctx, a).convert("RGB")), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_collage", "Grid collage of several images.", I, "paths:list, columns:int=3, cell:int=400, gap:int=8, background:str=#ffffff, out:str=images/collage.png", lambda ctx, paths, columns, cell, gap, background, out: _collage(ctx, paths, columns, cell, gap, background, out), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_side_by_side", "Put two images next to each other (before / after).", I, "a:str, b:str, out:str=images/side_by_side.png", lambda ctx, a, b, out: _collage(ctx, [a, b], 2, 600, 6, "#ffffff", out), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_gif_from_images", "Animated GIF from images in the workspace.", I, "paths:list, out:str=images/animation.gif, ms:int=400, width:int=480", lambda ctx, paths, out, ms, width: _gif(ctx, paths, out, ms, width), ctx=True, risk="files", needs=NEED, produces="image", tags=("animation", "video"))
reg("image_gif_to_frames", "Split a GIF into PNG frames.", I, "path:str, folder:str=frames", lambda ctx, path, folder: _frames(ctx, path, folder), ctx=True, risk="files", needs=NEED)
reg("image_sprite_sheet", "Combine images into a horizontal sprite sheet.", I, "paths:list, out:str=images/sprites.png", lambda ctx, paths, out: _collage(ctx, paths, len(paths), 256, 0, "#00000000", out), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_contact_sheet", "Contact sheet (thumbnails with file names) of every image in a folder.", I, "folder:str=images, out:str=images/contact_sheet.png", lambda ctx, folder, out: _contact(ctx, folder, out), ctx=True, risk="files", needs=NEED, produces="image")

# ---- fit to a platform size (crop to fill)
SIZES = {"instagram_post": (1080, 1080), "instagram_portrait": (1080, 1350), "instagram_story": (1080, 1920), "instagram_reel_cover": (1080, 1920), "youtube_thumbnail": (1280, 720), "youtube_banner": (2560, 1440),
         "twitter_post": (1600, 900), "twitter_header": (1500, 500), "facebook_cover": (820, 312), "linkedin_banner": (1584, 396), "pinterest_pin": (1000, 1500), "og_image": (1200, 630), "favicon": (64, 64),
         "app_icon": (1024, 1024), "hd_wallpaper": (1920, 1080), "uhd_wallpaper": (3840, 2160), "phone_wallpaper": (1170, 2532), "a4_print_300dpi": (2480, 3508), "passport_photo": (600, 600), "tiktok_video_cover": (1080, 1920)}
for _s, (_w, _h) in SIZES.items():
    reg(f"image_fit_{_s}", f"Crop and resize an image to {_s.replace('_', ' ')} size ({_w}×{_h}).", I, "path:str, out:str?", (lambda s, w, h: lambda ctx, path, out: _out(ctx, _fill(_open(ctx, path).convert("RGB"), w, h), out, f"images/{_stem(path)}_{s}.png", s))(_s, _w, _h), ctx=True, risk="files", needs=NEED, produces="image", tags=("instagram", "youtube", "social"))
    reg(f"image_blank_{_s}", f"Create a blank {_s.replace('_', ' ')} canvas ({_w}×{_h}) with a colour or gradient.", I, "color:str=#1b2a6b, color2:str?, out:str?", (lambda s, w, h: lambda ctx, color, color2, out: _out(ctx, _gradient(w, h, _rgb(color), _rgb(color2) if color2 else None, "vertical"), out, f"images/{s}.png", s))(_s, _w, _h), ctx=True, risk="files", needs=NEED, produces="image")


# ------------------------------------------------------------------ creation
def _gradient(w, h, c1, c2=None, mode="vertical"):
    Image = _pil()[0]
    if c2 is None:
        return Image.new("RGB", (w, h), c1)
    if mode == "radial":
        g = Image.radial_gradient("L").resize((w, h))
        return Image.composite(Image.new("RGB", (w, h), c1), Image.new("RGB", (w, h), c2), g.point(lambda p: 255 - p))
    g = Image.linear_gradient("L")
    if mode == "horizontal":
        g = g.rotate(90, expand=True)
    if mode == "diagonal":
        gv, gh = g.resize((w, h)), g.rotate(90, expand=True).resize((w, h))
        g = _pil()[1].add(gv, gh, scale=2)
    g = g.resize((w, h))
    return Image.composite(Image.new("RGB", (w, h), c2), Image.new("RGB", (w, h), c1), g)


reg("image_new", "Create a solid-colour image.", I, "width:int=1024, height:int=768, color:str=#ffffff, out:str=images/new.png", lambda ctx, width, height, color, out: _out(ctx, _gradient(width, height, _rgb(color)), out, "", "new image"), ctx=True, risk="files", needs=NEED, produces="image")
for _m in ("vertical", "horizontal", "diagonal", "radial"):
    reg(f"image_gradient_{_m}", f"{_m.capitalize()} gradient between two colours.", I, "width:int=1280, height:int=720, color1:str=#4f7cff, color2:str=#9b5de5, out:str?", (lambda m: lambda ctx, width, height, color1, color2, out: _out(ctx, _gradient(width, height, _rgb(color1), _rgb(color2), m), out, f"images/gradient_{m}.png", "gradient"))(_m), ctx=True, risk="files", needs=NEED, produces="image", tags=("background", "wallpaper"))


def _card(ctx, text, sub, w, h, c1, c2, fg, out, align="center", badge=None):
    Image, _, Draw, *_ = _pil()
    im = _gradient(w, h, _rgb(c1), _rgb(c2), "diagonal")
    d = Draw.Draw(im)
    size = max(24, min(w, h) // 8)
    f = font(size)
    lines = wrap(d, text, f, int(w * .8))
    while len(lines) * size * 1.2 > h * .6 and size > 18:
        size = int(size * .88)
        f = font(size)
        lines = wrap(d, text, f, int(w * .8))
    y = (h - len(lines) * size * 1.2) / 2 - (size * .5 if sub else 0)
    for l in lines:
        tw = d.textlength(l, font=f)
        d.text(((w - tw) / 2 if align == "center" else w * .1, y), l, font=f, fill=_rgb(fg, (255, 255, 255)))
        y += size * 1.2
    if sub:
        fs = font(max(14, size // 3), False)
        for l in wrap(d, sub, fs, int(w * .8)):
            d.text(((w - d.textlength(l, font=fs)) / 2 if align == "center" else w * .1, y + size * .2), l, font=fs, fill=_rgb(fg, (255, 255, 255)))
            y += size * .4
    if badge:
        fb = font(max(12, size // 4))
        d.rounded_rectangle((w * .06, h * .06, w * .06 + d.textlength(badge, font=fb) + 28, h * .06 + size * .5 + 14), 12, fill=(255, 255, 255))
        d.text((w * .06 + 14, h * .06 + 7), badge, font=fb, fill=_rgb(c1))
    return _out(ctx, im, out, "images/card.png", "text card")


for _nm, (_w, _h) in (("instagram_post", (1080, 1080)), ("instagram_story", (1080, 1920)), ("youtube_thumbnail", (1280, 720)), ("twitter_post", (1600, 900)), ("og_image", (1200, 630)), ("linkedin_banner", (1584, 396)), ("slide_16x9", (1920, 1080))):
    reg(f"image_text_card_{_nm}", f"Designed text card for {_nm.replace('_', ' ')} ({_w}×{_h}): big headline, optional sub-line and badge on a gradient.", I, "text:str, sub:str?, color1:str=#4f7cff, color2:str=#9b5de5, text_color:str=#ffffff, badge:str?, out:str?",
        (lambda nm, w, h: lambda ctx, text, sub, color1, color2, text_color, badge, out: _card(ctx, text, sub, w, h, color1, color2, text_color, out or f"images/{nm}_card.png", badge=badge))(_nm, _w, _h), ctx=True, risk="files", needs=NEED, produces="image", tags=("instagram", "youtube", "poster", "design"))
reg("image_quote_card", "Quote card with the author's name.", I, "quote:str, author:str?, color1:str=#0d1b3e, color2:str=#1e88e5, out:str?", lambda ctx, quote, author, color1, color2, out: _card(ctx, "“" + quote + "”", "— " + author if author else None, 1080, 1080, color1, color2, "#ffffff", out or "images/quote.png"), ctx=True, risk="files", needs=NEED, produces="image", tags=("instagram",))
reg("image_certificate", "A certificate of achievement (landscape A4).", I, "name:str, achievement:str, issuer:str=AI Room, out:str=images/certificate.png", lambda ctx, name, achievement, issuer, out: _certificate(ctx, name, achievement, issuer, out), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_meme", "Classic meme: top and bottom captions on an image.", I, "path:str, top:str?, bottom:str?, out:str?", lambda ctx, path, top, bottom, out: _meme(ctx, path, top, bottom, out), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_avatar_initials", "Round avatar with initials.", I, "name:str, size:int=512, out:str=images/avatar.png", lambda ctx, name, size, out: _avatar(ctx, name, size, out), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_badge", "A round badge / sticker with text.", I, "text:str, color:str=#e5486b, size:int=512, out:str=images/badge.png", lambda ctx, text, color, size, out: _badge(ctx, text, color, size, out), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_calendar_page", "A month calendar page.", I, "year:int, month:int, out:str=images/calendar.png", lambda ctx, year, month, out: _calendar(ctx, year, month, out), ctx=True, risk="files", needs=NEED, produces="image")
reg("image_qr_like_grid", "A deterministic pixel identicon from a text (not a scannable QR code).", I, "text:str, size:int=512, out:str=images/identicon.png", lambda ctx, text, size, out: _identicon(ctx, text, size, out), ctx=True, risk="files", needs=NEED, produces="image")


# ---- procedural art
def _mandel(w, h, it, julia=None, cx=-0.5, cy=0, zoom=1.0, hue=0.6):
    Image = _pil()[0]
    px = Image.new("RGB", (w, h)).load() if False else None
    im = Image.new("RGB", (w, h))
    pix = im.load()
    sc = 3.0 / zoom / min(w, h)
    for y in range(h):
        for x in range(w):
            c = complex(cx + (x - w / 2) * sc, cy + (y - h / 2) * sc)
            z = c if julia is None else c
            k = julia if julia is not None else c
            n = 0
            while abs(z) <= 2 and n < it:
                z = z * z + k
                n += 1
            if n == it:
                pix[x, y] = (0, 0, 0)
            else:
                v = n - math.log2(max(1e-9, math.log2(abs(z) + 1e-9) + 1e-9)) if abs(z) > 1 else n
                r, g, b = colorsys.hsv_to_rgb((hue + v / 60) % 1, .75, 1)
                pix[x, y] = (int(r * 255), int(g * 255), int(b * 255))
    return im


def _sierpinski(n, size=900):
    Image, _, Draw, *_ = _pil()
    im = Image.new("RGB", (size, size), "#0d1b3e")
    d = Draw.Draw(im)

    def rec(a, b, c, k):
        if k == 0:
            d.polygon([a, b, c], fill=(255, 184, 107))
            return
        ab, bc, ca = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), ((b[0] + c[0]) / 2, (b[1] + c[1]) / 2), ((c[0] + a[0]) / 2, (c[1] + a[1]) / 2)
        rec(a, ab, ca, k - 1), rec(ab, b, bc, k - 1), rec(ca, bc, c, k - 1)
    rec((size / 2, 30), (30, size - 30), (size - 30, size - 30), min(n, 8))
    return im


def _koch(n, size=900):
    Image, _, Draw, *_ = _pil()
    im = Image.new("RGB", (size, size), "#0d1b3e")
    d = Draw.Draw(im)

    def seg(p, q, k):
        if k == 0:
            return [p, q]
        dx, dy = (q[0] - p[0]) / 3, (q[1] - p[1]) / 3
        a, b = (p[0] + dx, p[1] + dy), (p[0] + 2 * dx, p[1] + 2 * dy)
        pk = (a[0] + (b[0] - a[0]) * .5 - (b[1] - a[1]) * math.sqrt(3) / 2, a[1] + (b[1] - a[1]) * .5 + (b[0] - a[0]) * math.sqrt(3) / 2)
        return seg(p, a, k - 1)[:-1] + seg(a, pk, k - 1)[:-1] + seg(pk, b, k - 1)[:-1] + seg(b, q, k - 1)
    pts = [(size * .12, size * .7), (size * .5, size * .08), (size * .88, size * .7), (size * .12, size * .7)]
    path = []
    for i in range(3):
        path += seg(pts[i], pts[i + 1], min(n, 5))[:-1]
    d.polygon(path, fill=(120, 190, 255))
    return im


def _plasma(w, h, seed):
    Image = _pil()[0]
    r = random.Random(seed)
    ph = [r.uniform(0, 6.28) for _ in range(6)]
    im = Image.new("RGB", (w, h))
    pix = im.load()
    for y in range(h):
        for x in range(w):
            v = (math.sin(x / 32 + ph[0]) + math.sin(y / 24 + ph[1]) + math.sin((x + y) / 40 + ph[2]) + math.sin(math.hypot(x - w / 2, y - h / 2) / 28 + ph[3])) / 4
            rr, gg, bb = colorsys.hsv_to_rgb((v * .5 + .6 + ph[4] / 20) % 1, .8, .95)
            pix[x, y] = (int(rr * 255), int(gg * 255), int(bb * 255))
    return im


def _noise(w, h, seed, cell=24):
    Image = _pil()[0]
    r = random.Random(seed)
    gw, gh = w // cell + 2, h // cell + 2
    grid = [[r.random() for _ in range(gw)] for _ in range(gh)]
    im = Image.new("RGB", (w, h))
    pix = im.load()
    for y in range(h):
        for x in range(w):
            gx, gy = x / cell, y / cell
            x0, y0 = int(gx), int(gy)
            fx, fy = gx - x0, gy - y0
            fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
            v = (grid[y0][x0] * (1 - fx) + grid[y0][x0 + 1] * fx) * (1 - fy) + (grid[y0 + 1][x0] * (1 - fx) + grid[y0 + 1][x0 + 1] * fx) * fy
            rr, gg, bb = colorsys.hsv_to_rgb(.55 + v * .15, .5, .3 + v * .7)
            pix[x, y] = (int(rr * 255), int(gg * 255), int(bb * 255))
    return im


def _voronoi(w, h, n, seed):
    Image = _pil()[0]
    r = random.Random(seed)
    pts = [(r.randrange(w), r.randrange(h), tuple(int(c * 255) for c in colorsys.hsv_to_rgb(r.random(), .55, .95))) for _ in range(n)]
    im = Image.new("RGB", (w, h))
    pix = im.load()
    for y in range(h):
        for x in range(w):
            pix[x, y] = min(pts, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)[2]
    return im


def _spiro(size, R=5, r=3, d=5):
    Image, _, Draw, *_ = _pil()
    im = Image.new("RGB", (size, size), "#0d1b3e")
    dr = Draw.Draw(im)
    pts = []
    for i in range(6000):
        t = i / 6000 * 2 * math.pi * (r / math.gcd(int(R), int(r)))
        pts.append((size / 2 + (size * .4 / (R + d)) * ((R - r) * math.cos(t) + d * math.cos((R - r) / r * t)), size / 2 + (size * .4 / (R + d)) * ((R - r) * math.sin(t) - d * math.sin((R - r) / r * t))))
    dr.line(pts, fill=(255, 184, 107), width=2)
    return im


def _mandala(size, petals, seed):
    Image, _, Draw, *_ = _pil()
    r = random.Random(seed)
    im = Image.new("RGB", (size, size), "#0d1b3e")
    d = Draw.Draw(im)
    for ring in range(6, 0, -1):
        col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb((ring * .13 + seed * .07) % 1, .6, 1))
        rr = size * .46 * ring / 6
        for i in range(petals):
            a = 2 * math.pi * i / petals
            cx, cy = size / 2 + rr * .62 * math.cos(a), size / 2 + rr * .62 * math.sin(a)
            pr = rr * .38
            d.ellipse((cx - pr, cy - pr, cx + pr, cy + pr), outline=col, width=2)
    return im


def _life(w, h, steps, seed):
    Image = _pil()[0]
    r = random.Random(seed)
    g = [[r.random() < .3 for _ in range(w)] for _ in range(h)]
    for _ in range(steps):
        g = [[(sum(g[(y + dy) % h][(x + dx) % w] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy) in ((2, 3) if g[y][x] else (3,))) for x in range(w)] for y in range(h)]
    im = Image.new("L", (w, h))
    im.putdata([255 if c else 20 for row in g for c in row])
    return im.resize((w * 8, h * 8), Image.NEAREST)


def _automaton(rule, w, h):
    Image = _pil()[0]
    row = [0] * w
    row[w // 2] = 1
    im = Image.new("L", (w, h), 255)
    pix = im.load()
    for y in range(h):
        for x in range(w):
            if row[x]:
                pix[x, y] = 0
        row = [(rule >> (row[(x - 1) % w] * 4 + row[x] * 2 + row[(x + 1) % w])) & 1 for x in range(w)]
    return im.resize((w * 4, h * 4), Image.NEAREST)


def _maze(w, h, seed):
    Image = _pil()[0]
    r = random.Random(seed)
    cw, ch = w | 1, h | 1
    g = [[1] * cw for _ in range(ch)]
    st = [(1, 1)]
    g[1][1] = 0
    while st:
        x, y = st[-1]
        nb = [(x + dx, y + dy, dx, dy) for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)) if 0 < x + dx < cw - 1 and 0 < y + dy < ch - 1 and g[y + dy][x + dx]]
        if not nb:
            st.pop()
            continue
        nx, ny, dx, dy = r.choice(nb)
        g[y + dy // 2][x + dx // 2] = g[ny][nx] = 0
        st.append((nx, ny))
    im = Image.new("L", (cw, ch))
    im.putdata([0 if c else 255 for row in g for c in row])
    return im.resize((cw * 10, ch * 10), Image.NEAREST)


def _fractal_tree(size, depth):
    Image, _, Draw, *_ = _pil()
    im = Image.new("RGB", (size, size), "#0d1b3e")
    d = Draw.Draw(im)

    def rec(x, y, a, l, k):
        if k == 0:
            return
        x2, y2 = x + l * math.cos(a), y - l * math.sin(a)
        d.line((x, y, x2, y2), fill=(int(120 + 20 * (depth - k)), int(200 - 12 * (depth - k)), 120), width=max(1, k))
        rec(x2, y2, a + .45, l * .75, k - 1), rec(x2, y2, a - .45, l * .75, k - 1)
    rec(size / 2, size - 20, math.pi / 2, size * .22, min(depth, 12))
    return im


def _lissajous(size, a, b):
    Image, _, Draw, *_ = _pil()
    im = Image.new("RGB", (size, size), "#0d1b3e")
    Draw.Draw(im).line([(size / 2 + size * .42 * math.sin(a * t / 500 * 2 * math.pi + math.pi / 2), size / 2 + size * .42 * math.sin(b * t / 500 * 2 * math.pi)) for t in range(501)], fill=(110, 231, 183), width=3)
    return im


def _circles(w, h, n, seed):
    Image, _, Draw, *_ = _pil()
    r = random.Random(seed)
    im = Image.new("RGB", (w, h), "#101828")
    d = Draw.Draw(im, "RGBA")
    for _ in range(n):
        rad = r.randrange(10, min(w, h) // 5)
        x, y = r.randrange(w), r.randrange(h)
        d.ellipse((x - rad, y - rad, x + rad, y + rad), fill=tuple(int(c * 255) for c in colorsys.hsv_to_rgb(r.random(), .6, 1)) + (110,))
    return im


def _hilbert(order, size=800):
    Image, _, Draw, *_ = _pil()
    n = 2 ** min(order, 7)
    pts = []
    for i in range(n * n):
        x = y = 0
        t, s = i, 1
        while s < n:
            rx, ry = 1 & (t // 2), 1 & (t ^ (1 if (1 & (t // 2)) == 0 else 0))
            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        pts.append(((x + .5) * size / n, (y + .5) * size / n))
    im = Image.new("RGB", (size, size), "#0d1b3e")
    Draw.Draw(im).line(pts, fill=(255, 184, 107), width=2)
    return im


ART = {
    "mandelbrot": (lambda ctx, size, seed: _mandel(size, size * 2 // 3, 60), "Mandelbrot set"), "julia_set": (lambda ctx, size, seed: _mandel(size, size * 2 // 3, 80, julia=complex(-.8, .156), cx=0, hue=.1), "Julia set"),
    "sierpinski_triangle": (lambda ctx, size, seed: _sierpinski(6, size), "Sierpinski triangle"), "koch_snowflake": (lambda ctx, size, seed: _koch(4, size), "Koch snowflake"), "plasma": (lambda ctx, size, seed: _plasma(size, size * 2 // 3, seed), "plasma colours"),
    "value_noise": (lambda ctx, size, seed: _noise(size, size * 2 // 3, seed), "smooth noise texture"), "voronoi": (lambda ctx, size, seed: _voronoi(size // 2, size // 3, 40, seed), "Voronoi cells"), "spirograph": (lambda ctx, size, seed: _spiro(size, 5 + seed % 4, 3, 5), "spirograph"),
    "mandala": (lambda ctx, size, seed: _mandala(size, 8 + seed % 8, seed), "mandala of circles"), "game_of_life": (lambda ctx, size, seed: _life(min(size // 8, 90), min(size // 8, 90), 40, seed), "Conway's Game of Life after 40 steps"),
    "rule_30": (lambda ctx, size, seed: _automaton(30, 120, 120), "cellular automaton rule 30"), "rule_110": (lambda ctx, size, seed: _automaton(110, 120, 120), "cellular automaton rule 110"), "rule_90": (lambda ctx, size, seed: _automaton(90, 120, 120), "cellular automaton rule 90"),
    "maze": (lambda ctx, size, seed: _maze(41, 41, seed), "random maze"), "fractal_tree": (lambda ctx, size, seed: _fractal_tree(size, 10), "fractal tree"), "lissajous": (lambda ctx, size, seed: _lissajous(size, 3 + seed % 3, 2 + seed % 2), "Lissajous curve"),
    "bubbles": (lambda ctx, size, seed: _circles(size, size * 2 // 3, 80, seed), "translucent bubbles"), "hilbert_curve": (lambda ctx, size, seed: _hilbert(5, size), "Hilbert curve"),
}
for _a, (_fn, _d) in ART.items():
    reg(f"image_art_{_a}", f"Generate art: {_d}. Different seed → different picture.", I, "size:int=800, seed:int=1, out:str?", (lambda a, fn: lambda ctx, size, seed, out: _out(ctx, fn(ctx, min(max(size, 200), 1600), seed), out, f"images/art_{a}_{seed}.png", a))(_a, _fn), ctx=True, risk="files", needs=NEED, produces="image", tags=("generative", "fractal", "creative"))


# ------------------------------------------------------------------ helpers
def _round(ctx, path, radius, out, circle=False):
    Image, _, Draw, *_ = _pil()
    im = _open(ctx, path).convert("RGBA")
    if circle:
        s = min(im.size)
        im = _fill(im.convert("RGB"), s, s).convert("RGBA")
    m = Image.new("L", im.size, 0)
    Draw.Draw(m).rounded_rectangle((0, 0, *im.size), min(radius, min(im.size) // 2), fill=255)
    im.putalpha(m)
    return _out(ctx, im, out, f"images/{_stem(path)}_round.png")


def _shadow(ctx, path, off, blur, out):
    Image, _, Draw, _, Fl, *_ = _pil()
    im = _open(ctx, path).convert("RGBA")
    pad = blur * 3 + off
    base = Image.new("RGBA", (im.width + pad * 2, im.height + pad * 2), (255, 255, 255, 0))
    sh = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sh.paste((0, 0, 0, 150), (pad + off, pad + off, pad + off + im.width, pad + off + im.height))
    base = Image.alpha_composite(base, sh.filter(Fl.GaussianBlur(blur)))
    base.paste(im, (pad, pad), im)
    return _out(ctx, base, out, f"images/{_stem(path)}_shadow.png")


def _ascii(im, cols):
    chars = " .:-=+*#%@"
    g = im.convert("L")
    h = max(1, int(g.height / g.width * cols * .5))
    g = g.resize((cols, h))
    return "\n".join("".join(chars[g.getpixel((x, y)) * (len(chars) - 1) // 255] for x in range(cols)) for y in range(h))


def _palette(im, n):
    q = im.convert("RGB").resize((120, 120)).quantize(colors=max(2, min(n, 16)), method=_pil()[0].MEDIANCUT)
    pal = q.getpalette()
    cnt = sorted(q.getcolors(), reverse=True)
    return [{"hex": "#%02x%02x%02x" % tuple(pal[i * 3:i * 3 + 3]), "share": round(c / (120 * 120), 3)} for c, i in cnt[:n]]


def _hist(ctx, path, out):
    Image, _, Draw, *_ = _pil()
    im = _open(ctx, path).convert("RGB")
    h = im.histogram()
    W, H = 768, 300
    c = Image.new("RGB", (W, H), "white")
    d = Draw.Draw(c, "RGBA")
    mx = max(h) or 1
    for ch, col in enumerate(((229, 57, 53), (67, 160, 71), (30, 136, 229))):
        pts = [(i * 3, H - 10 - h[ch * 256 + i] / mx * (H - 30)) for i in range(256)]
        d.polygon([(0, H - 10)] + pts + [(765, H - 10)], fill=col + (70,))
        d.line(pts, fill=col + (255,), width=2)
    return _out(ctx, c, out, f"images/{_stem(path)}_histogram.png")


def _chroma(ctx, path, color, tol, out):
    im = _open(ctx, path).convert("RGBA")
    t = _rgb(color)
    im.putdata([(r, g, b, 0) if abs(r - t[0]) + abs(g - t[1]) + abs(b - t[2]) <= tol * 3 else (r, g, b, a) for r, g, b, a in im.getdata()])
    return _out(ctx, im, out, f"images/{_stem(path)}_cutout.png")


def _caption(ctx, path, text, pos, size, color, out):
    _, _, Draw, *_ = _pil()
    im = _open(ctx, path).convert("RGB")
    d = Draw.Draw(im)
    size = size or max(16, im.width // 14)
    f = font(size)
    lines = wrap(d, text, f, int(im.width * .9))
    h = len(lines) * size * 1.15
    y = {"top": im.height * .04, "center": (im.height - h) / 2}.get(pos, im.height * .96 - h)
    for l in lines:
        x = (im.width - d.textlength(l, font=f)) / 2
        d.text((x, y), l, font=f, fill=_rgb(color, (255, 255, 255)), stroke_width=max(2, size // 12), stroke_fill=(0, 0, 0))
        y += size * 1.15
    return _out(ctx, im, out, f"images/{_stem(path)}_caption.png")


def _meme(ctx, path, top, bottom, out):
    _, _, Draw, *_ = _pil()
    im = _open(ctx, path).convert("RGB")
    d = Draw.Draw(im)
    size = max(20, im.width // 10)
    f = font(size)
    for text, atop in ((top, True), (bottom, False)):
        if not text:
            continue
        lines = wrap(d, text.upper(), f, int(im.width * .94))
        y = im.height * .02 if atop else im.height * .98 - len(lines) * size * 1.1
        for l in lines:
            d.text(((im.width - d.textlength(l, font=f)) / 2, y), l, font=f, fill="white", stroke_width=max(3, size // 9), stroke_fill="black")
            y += size * 1.1
    return _out(ctx, im, out, f"images/{_stem(path)}_meme.png")


def _watermark(ctx, path, text, out):
    Image, _, Draw, *_ = _pil()
    im = _open(ctx, path).convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = Draw.Draw(layer)
    f = font(max(14, im.width // 28))
    d.text((im.width - d.textlength(text, font=f) - 16, im.height - f.size * 1.5), text, font=f, fill=(255, 255, 255, 170))
    return _out(ctx, Image.alpha_composite(im, layer), out, f"images/{_stem(path)}_wm.png")


def _overlay(ctx, base, overlay, x, y, scale, opacity, out):
    Image = _pil()[0]
    b, o = _open(ctx, base).convert("RGBA"), _open(ctx, overlay).convert("RGBA")
    w = max(1, int(b.width * scale))
    o = o.resize((w, max(1, int(o.height * w / o.width))), Image.LANCZOS)
    if opacity < 1:
        o.putalpha(o.getchannel("A").point(lambda p: int(p * opacity)))
    b.paste(o, (x, y), o)
    return _out(ctx, b, out, f"images/{_stem(base)}_overlay.png")


def _collage(ctx, paths, cols, cell, gap, bg, out):
    Image = _pil()[0]
    ims = [_fill(_open(ctx, p).convert("RGB"), cell, cell) if len(paths) > 2 or cols != 2 else _open(ctx, p).convert("RGB") for p in paths]
    if cols == 2 and len(paths) == 2 and cell == 600:
        h = 600
        ims = [i.resize((round(i.width * h / i.height), h)) for i in ims]
        cw = [i.width for i in ims]
        c = Image.new("RGB", (sum(cw) + gap, h), _rgb(bg, (255, 255, 255)))
        c.paste(ims[0], (0, 0))
        c.paste(ims[1], (cw[0] + gap, 0))
        return _out(ctx, c, out, "images/collage.png")
    rows = math.ceil(len(ims) / cols)
    transparent = bg.endswith("00") and len(bg) == 9
    c = Image.new("RGBA" if transparent else "RGB", (cols * cell + (cols - 1) * gap, rows * cell + (rows - 1) * gap), (0, 0, 0, 0) if transparent else _rgb(bg, (255, 255, 255)))
    for i, im in enumerate(ims):
        c.paste(im, ((i % cols) * (cell + gap), (i // cols) * (cell + gap)))
    return _out(ctx, c, out, "images/collage.png")


def _gif(ctx, paths, out, ms, width):
    Image = _pil()[0]
    ims = [_open(ctx, p).convert("RGB") for p in paths]
    ims = [i.resize((width, round(i.height * width / i.width))) for i in ims]
    ims[0].save(ctx.path(out), save_all=True, append_images=ims[1:], duration=ms, loop=0)
    return {"saved": ctx.produced(ctx.path(out), "image", f"{len(ims)}-frame GIF"), "frames": len(ims)}


def _frames(ctx, path, folder):
    im = _open(ctx, path)
    n = getattr(im, "n_frames", 1)
    for i in range(min(n, 200)):
        im.seek(i)
        p = ctx.path(f"{folder}/frame_{i:03d}.png")
        im.convert("RGB").save(p)
        ctx.produced(p, "image")
    return {"frames": min(n, 200), "folder": folder}


def _contact(ctx, folder, out):
    Image, _, Draw, *_ = _pil()
    files = sorted(p for p in ctx.path(folder + "/.keep").parent.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"))[:60]
    if not files:
        raise ToolError("no images in that folder")
    cell, cols = 220, 5
    rows = math.ceil(len(files) / cols)
    c = Image.new("RGB", (cols * (cell + 10) + 10, rows * (cell + 34) + 10), "white")
    d = Draw.Draw(c)
    f = font(12, False)
    for i, p in enumerate(files):
        im = Image.open(p).convert("RGB")
        im.thumbnail((cell, cell))
        x, y = 10 + (i % cols) * (cell + 10), 10 + (i // cols) * (cell + 34)
        c.paste(im, (x + (cell - im.width) // 2, y))
        d.text((x, y + cell + 6), p.name[:30], font=f, fill=(60, 60, 60))
    return _out(ctx, c, out, "")


def _certificate(ctx, name, ach, issuer, out):
    Image, _, Draw, *_ = _pil()
    w, h = 1754, 1240
    im = Image.new("RGB", (w, h), "#fdfaf1")
    d = Draw.Draw(im)
    d.rectangle((40, 40, w - 40, h - 40), outline="#c9a14a", width=8)
    d.rectangle((70, 70, w - 70, h - 70), outline="#c9a14a", width=2)
    for t, y, s, b in (("CERTIFICATE", 190, 110, True), ("of achievement", 330, 46, False), ("This certifies that", 470, 40, False), (name, 560, 100, True), (ach, 760, 46, False)):
        f = font(s, b)
        for i, l in enumerate(wrap(d, t, f, w - 300)):
            d.text(((w - d.textlength(l, font=f)) / 2, y + i * s * 1.2), l, font=f, fill="#2b2b2b" if t != name else "#8a6a1f")
    f = font(34, False)
    d.text(((w - d.textlength(issuer, font=f)) / 2, h - 210), issuer, font=f, fill="#555")
    d.line((w / 2 - 200, h - 220, w / 2 + 200, h - 220), fill="#999", width=2)
    return _out(ctx, im, out, "")


def _avatar(ctx, name, size, out):
    Image, _, Draw, *_ = _pil()
    h = sum(map(ord, name)) % 360 / 360
    bg = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, .55, .85))
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = Draw.Draw(im)
    d.ellipse((0, 0, size - 1, size - 1), fill=bg)
    ini = "".join(w[0] for w in name.split()[:2]).upper() or "?"
    f = font(size // 2)
    bb = d.textbbox((0, 0), ini, font=f)
    d.text(((size - (bb[2] - bb[0])) / 2 - bb[0], (size - (bb[3] - bb[1])) / 2 - bb[1]), ini, font=f, fill="white")
    return _out(ctx, im, out, "")


def _badge(ctx, text, color, size, out):
    Image, _, Draw, *_ = _pil()
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = Draw.Draw(im)
    c = _rgb(color)
    d.ellipse((4, 4, size - 4, size - 4), fill=c)
    d.ellipse((size * .08, size * .08, size * .92, size * .92), outline=(255, 255, 255), width=max(3, size // 60))
    f = font(size // 6)
    lines = wrap(d, text, f, int(size * .7))
    y = (size - len(lines) * size / 6 * 1.2) / 2
    for l in lines:
        d.text(((size - d.textlength(l, font=f)) / 2, y), l, font=f, fill="white")
        y += size / 6 * 1.2
    return _out(ctx, im, out, "")


def _calendar(ctx, year, month, out):
    import calendar
    Image, _, Draw, *_ = _pil()
    w, h = 1400, 1000
    im = Image.new("RGB", (w, h), "white")
    d = Draw.Draw(im)
    d.rectangle((0, 0, w, 150), fill="#4f7cff")
    t = f"{calendar.month_name[month]} {year}"
    f = font(80)
    d.text(((w - d.textlength(t, font=f)) / 2, 30), t, font=f, fill="white")
    cw, ch = w // 7, (h - 220) // 6
    fs, fd = font(32), font(40, False)
    for i, n in enumerate(calendar.day_abbr):
        d.text((i * cw + 20, 170), n, font=fs, fill="#6b7689")
    for r, wk in enumerate(calendar.monthcalendar(year, month)):
        for c, dd in enumerate(wk):
            d.rectangle((c * cw, 220 + r * ch, (c + 1) * cw, 220 + (r + 1) * ch), outline="#e3e8f0")
            if dd:
                d.text((c * cw + 16, 230 + r * ch), str(dd), font=fd, fill="#e5486b" if c >= 5 else "#1c2433")
    return _out(ctx, im, out, "")


def _identicon(ctx, text, size, out):
    import hashlib
    Image, _, Draw, *_ = _pil()
    h = hashlib.sha256(text.encode()).digest()
    col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h[0] / 255, .6, .85))
    n = 7
    im = Image.new("RGB", (size, size), "#f4f6fb")
    d = Draw.Draw(im)
    cell = size / (n + 2)
    for y in range(n):
        for x in range((n + 1) // 2):
            if h[1 + y * 4 + x] % 2:
                for xx in (x, n - 1 - x):
                    d.rectangle(((xx + 1) * cell, (y + 1) * cell, (xx + 2) * cell, (y + 2) * cell), fill=col)
    return _out(ctx, im, out, "")
