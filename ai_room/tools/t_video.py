"""Video and audio with ffmpeg: convert, edit, filter, generate, subtitle, and fit to social-media formats."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .registry import Context, ToolError, reg

V, A = "video", "audio"
NEED = ("bin:ffmpeg",)
SLOW = 600.0


def ff(ctx: Context, args: list[str], out: Path | None = None, note: str = "", kind: str = "video") -> dict:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args] + ([str(out)] if out else [])
    try:
        r = subprocess.run(cmd, cwd=ctx.workspace, capture_output=True, text=True, timeout=SLOW, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise ToolError("ffmpeg timed out — use a shorter clip or a smaller size")
    if r.returncode:
        raise ToolError("ffmpeg failed: " + (r.stderr.strip().splitlines() or ["unknown error"])[-1][:300])
    if out is None:
        return {"ok": True}
    return {"saved": ctx.produced(out, kind, note), "bytes": out.stat().st_size}


def probe(ctx: Context, path: str) -> dict:
    p = ctx.path(path, True)
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(p)], capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        raise ToolError("ffprobe is not installed")
    if r.returncode:
        raise ToolError("cannot read this media file: " + r.stderr.strip()[:200])
    j = json.loads(r.stdout)
    out = {"duration_s": round(float(j["format"].get("duration", 0)), 3), "size_bytes": int(j["format"].get("size", 0)), "format": j["format"].get("format_name"), "streams": []}
    for s in j["streams"]:
        d = {"type": s["codec_type"], "codec": s.get("codec_name")}
        if s["codec_type"] == "video":
            fr = s.get("r_frame_rate", "0/1").split("/")
            d |= {"width": s.get("width"), "height": s.get("height"), "fps": round(int(fr[0]) / max(1, int(fr[1])), 2)}
        elif s["codec_type"] == "audio":
            d |= {"sample_rate": s.get("sample_rate"), "channels": s.get("channels")}
        out["streams"].append(d)
    return out


def dur(ctx, path) -> float:
    return probe(ctx, path)["duration_s"]


def _o(ctx, out, default):
    return ctx.path(out or default)


def _stem(p):
    return p.rsplit("/", 1)[-1].rsplit(".", 1)[0]


R = dict(ctx=True, risk="files", needs=NEED, timeout=SLOW)
reg("video_info", "Duration, size, codecs, resolution and frame rate of a video or audio file.", V, "path:str", probe, ctx=True, risk="files", needs=("bin:ffprobe",))

# ------------------------------------------------------------------ format conversion
for _f, _args in (("mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart"]), ("webm", ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "34", "-c:a", "libopus"]), ("mkv", ["-c:v", "libx264", "-c:a", "aac"]),
                  ("mov", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"]), ("avi", ["-c:v", "mpeg4", "-c:a", "libmp3lame"])):
    reg(f"video_convert_to_{_f}", f"Convert a video to {_f.upper()}.", V, "path:str, out:str?", (lambda f, a: lambda ctx, path, out: ff(ctx, ["-i", str(ctx.path(path, True)), *a], _o(ctx, out, f"videos/{_stem(path)}.{f}"), f"converted to {f}"))(_f, _args), produces="video", **R)
reg("video_to_gif", "Turn a video (or a part of it) into a small animated GIF.", V, "path:str, start:float=0, seconds:float=5, width:int=480, fps:int=12, out:str?",
    lambda ctx, path, start, seconds, width, fps, out: ff(ctx, ["-ss", str(start), "-t", str(seconds), "-i", str(ctx.path(path, True)), "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse", "-loop", "0"], _o(ctx, out, f"videos/{_stem(path)}.gif"), "GIF", "image"), produces="image", **R)
for _f, _c in (("mp3", ["-c:a", "libmp3lame", "-q:a", "2"]), ("wav", ["-c:a", "pcm_s16le"]), ("ogg", ["-c:a", "libvorbis", "-q:a", "5"]), ("opus", ["-c:a", "libopus"]), ("aac", ["-c:a", "aac"]), ("m4a", ["-c:a", "aac"]), ("flac", ["-c:a", "flac"])):
    reg(f"audio_convert_to_{_f}", f"Convert audio (or extract the sound of a video) to {_f.upper()}.", A, "path:str, out:str?", (lambda f, c: lambda ctx, path, out: ff(ctx, ["-i", str(ctx.path(path, True)), "-vn", *c], _o(ctx, out, f"audio/{_stem(path)}.{f}"), f"audio {f}", "audio"))(_f, _c), produces="audio", **R)

# ------------------------------------------------------------------ editing
reg("video_trim", "Cut a section out of a video (start and length in seconds).", V, "path:str, start:float=0, seconds:float=10, out:str?", lambda ctx, path, start, seconds, out: ff(ctx, ["-ss", str(start), "-i", str(ctx.path(path, True)), "-t", str(seconds), "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p"], _o(ctx, out, f"videos/{_stem(path)}_trim.mp4"), "trimmed"), produces="video", **R)
reg("video_concat", "Join several videos one after another (they are re-encoded to match).", V, "paths:list, width:int=1280, height:int=720, out:str=videos/joined.mp4", lambda ctx, paths, width, height, out: _concat(ctx, paths, width, height, out), produces="video", **R)
reg("video_speed", "Change speed (2 = twice as fast, 0.5 = half speed), audio follows.", V, "path:str, factor:float=2, out:str?", lambda ctx, path, factor, out: ff(ctx, ["-i", str(ctx.path(path, True)), "-filter_complex", f"[0:v]setpts=PTS/{factor}[v];[0:a]{_atempo(factor)}[a]", "-map", "[v]", "-map", "[a]", "-pix_fmt", "yuv420p"], _o(ctx, out, f"videos/{_stem(path)}_x{factor:g}.mp4")) if _has_audio(ctx, path) else ff(ctx, ["-i", str(ctx.path(path, True)), "-vf", f"setpts=PTS/{factor}", "-an", "-pix_fmt", "yuv420p"], _o(ctx, out, f"videos/{_stem(path)}_x{factor:g}.mp4")), produces="video", **R)
reg("video_reverse", "Play a (short) video backwards.", V, "path:str, out:str?", lambda ctx, path, out: ff(ctx, ["-i", str(ctx.path(path, True)), "-vf", "reverse", "-af", "areverse", "-pix_fmt", "yuv420p"] if _has_audio(ctx, path) else ["-i", str(ctx.path(path, True)), "-vf", "reverse", "-an", "-pix_fmt", "yuv420p"], _o(ctx, out, f"videos/{_stem(path)}_rev.mp4")), produces="video", **R)
reg("video_loop", "Repeat a video N times.", V, "path:str, times:int=3, out:str?", lambda ctx, path, times, out: ff(ctx, ["-stream_loop", str(max(0, times - 1)), "-i", str(ctx.path(path, True)), "-c", "copy"], _o(ctx, out, f"videos/{_stem(path)}_loop.mp4")), produces="video", **R)
reg("video_fade", "Fade in at the start and out at the end.", V, "path:str, seconds:float=1, out:str?", lambda ctx, path, seconds, out: (lambda d: ff(ctx, ["-i", str(ctx.path(path, True)), "-vf", f"fade=t=in:st=0:d={seconds},fade=t=out:st={max(0, d - seconds)}:d={seconds}", "-pix_fmt", "yuv420p"] + (["-af", f"afade=t=in:d={seconds},afade=t=out:st={max(0, d - seconds)}:d={seconds}"] if _has_audio(ctx, path) else []), _o(ctx, out, f"videos/{_stem(path)}_fade.mp4")))(dur(ctx, path)), produces="video", **R)
reg("video_crop", "Crop to width × height starting at x,y.", V, "path:str, width:int, height:int, x:int=0, y:int=0, out:str?", lambda ctx, path, width, height, x, y, out: ff(ctx, ["-i", str(ctx.path(path, True)), "-vf", f"crop={width}:{height}:{x}:{y}", "-pix_fmt", "yuv420p"], _o(ctx, out, f"videos/{_stem(path)}_crop.mp4")), produces="video", **R)
reg("video_resize", "Scale to a width (height follows).", V, "path:str, width:int=1280, out:str?", lambda ctx, path, width, out: ff(ctx, ["-i", str(ctx.path(path, True)), "-vf", f"scale={width}:-2", "-pix_fmt", "yuv420p"], _o(ctx, out, f"videos/{_stem(path)}_{width}w.mp4")), produces="video", **R)
reg("video_set_fps", "Change the frame rate.", V, "path:str, fps:int=30, out:str?", lambda ctx, path, fps, out: ff(ctx, ["-i", str(ctx.path(path, True)), "-r", str(fps), "-pix_fmt", "yuv420p"], _o(ctx, out, f"videos/{_stem(path)}_{fps}fps.mp4")), produces="video", **R)
reg("video_smooth_slowmo", "Interpolate frames for smoother slow motion (slow-down factor × frames).", V, "path:str, factor:int=2, out:str?", lambda ctx, path, factor, out: ff(ctx, ["-i", str(ctx.path(path, True)), "-vf", f"setpts={factor}*PTS,minterpolate=fps=30", "-an", "-pix_fmt", "yuv420p"], _o(ctx, out, f"videos/{_stem(path)}_slowmo.mp4")), produces="video", **R)
reg("video_remove_audio", "Strip the sound track.", V, "path:str, out:str?", lambda ctx, path, out: ff(ctx, ["-i", str(ctx.path(path, True)), "-c:v", "copy", "-an"], _o(ctx, out, f"videos/{_stem(path)}_silent.mp4")), produces="video", **R)
reg("video_add_audio", "Put an audio file under a video (shortest wins).", V, "video:str, audio:str, out:str?", lambda ctx, video, audio, out: ff(ctx, ["-i", str(ctx.path(video, True)), "-i", str(ctx.path(audio, True)), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest"], _o(ctx, out, f"videos/{_stem(video)}_audio.mp4")), produces="video", **R)
reg("video_overlay_image", "Put a logo/image on a video (position top-left, top-right, bottom-left, bottom-right, center).", V, "video:str, image:str, position:str=bottom-right, scale:int=160, out:str?",
    lambda ctx, video, image, position, scale, out: ff(ctx, ["-i", str(ctx.path(video, True)), "-i", str(ctx.path(image, True)), "-filter_complex", f"[1:v]scale={scale}:-1[l];[0:v][l]overlay={ {'top-left': '10:10', 'top-right': 'W-w-10:10', 'bottom-left': '10:H-h-10', 'bottom-right': 'W-w-10:H-h-10', 'center': '(W-w)/2:(H-h)/2'}.get(position, 'W-w-10:H-h-10') }", "-pix_fmt", "yuv420p"], _o(ctx, out, f"videos/{_stem(video)}_logo.mp4")), produces="video", **R)
reg("video_add_text", "Burn text onto a video (position top, center or bottom).", V, "path:str, text:str, position:str=bottom, size:int=48, color:str=white, out:str?", lambda ctx, path, text, position, size, color, out: ff(ctx, ["-i", str(ctx.path(path, True)), "-vf", f"drawtext=text='{_dt(text)}':fontcolor={color}:fontsize={size}:box=1:boxcolor=black@0.5:boxborderw=12:x=(w-text_w)/2:y={ {'top': '40', 'center': '(h-text_h)/2'}.get(position, 'h-text_h-50') }", "-pix_fmt", "yuv420p", "-c:a", "copy"], _o(ctx, out, f"videos/{_stem(path)}_text.mp4")), produces="video", **R)
reg("video_burn_subtitles", "Burn an .srt subtitle file into a video.", V, "video:str, subtitles:str, out:str?", lambda ctx, video, subtitles, out: ff(ctx, ["-i", str(ctx.path(video, True)), "-vf", f"subtitles={ctx.path(subtitles, True).name}" if ctx.path(subtitles, True).parent == ctx.workspace else f"subtitles='{ctx.path(subtitles, True)}'", "-pix_fmt", "yuv420p"], _o(ctx, out, f"videos/{_stem(video)}_subs.mp4")), produces="video", **R)
reg("video_stabilize", "Reduce camera shake (two-pass vidstab).", V, "path:str, out:str?", lambda ctx, path, out: _stab(ctx, path, out), produces="video", **R)
reg("video_extract_frame", "Save one frame as a PNG (at a time in seconds).", V, "path:str, at:float=0, out:str?", lambda ctx, path, at, out: ff(ctx, ["-ss", str(at), "-i", str(ctx.path(path, True)), "-frames:v", "1"], _o(ctx, out, f"images/{_stem(path)}_{at:g}s.png"), "frame", "image"), produces="image", **R)
reg("video_extract_frames", "Save frames as PNGs (one every N seconds).", V, "path:str, every:float=1, folder:str=frames", lambda ctx, path, every, folder: (ff(ctx, ["-i", str(ctx.path(path, True)), "-vf", f"fps=1/{every}", str(ctx.path(folder + "/f_%04d.png"))]), {"folder": folder, "frames": len(list((ctx.workspace / folder).glob("f_*.png")))})[1], **R)
reg("video_thumbnail", "Pick a representative frame as a thumbnail image.", V, "path:str, out:str?", lambda ctx, path, out: ff(ctx, ["-i", str(ctx.path(path, True)), "-vf", "thumbnail=100", "-frames:v", "1"], _o(ctx, out, f"images/{_stem(path)}_thumb.png"), "thumbnail", "image"), produces="image", **R)
reg("video_contact_sheet", "A tiled overview picture of a video (columns × rows of frames).", V, "path:str, columns:int=4, rows:int=3, out:str?", lambda ctx, path, columns, rows, out: ff(ctx, ["-i", str(ctx.path(path, True)), "-vf", f"fps={columns * rows}/{max(1, dur(ctx, path)):.3f},scale=320:-1,tile={columns}x{rows}", "-frames:v", "1"], _o(ctx, out, f"images/{_stem(path)}_sheet.png"), "contact sheet", "image"), produces="image", **R)
reg("video_compare_side_by_side", "Two videos side by side.", V, "a:str, b:str, out:str=videos/compare.mp4", lambda ctx, a, b, out: ff(ctx, ["-i", str(ctx.path(a, True)), "-i", str(ctx.path(b, True)), "-filter_complex", "[0:v]scale=-2:480[l];[1:v]scale=-2:480[r];[l][r]hstack=shortest=1", "-pix_fmt", "yuv420p"], ctx.path(out)), produces="video", **R)
reg("video_xfade", "Cross-fade from one video into another (transition: fade, wipeleft, slideup, circleopen, dissolve, pixelize…).", V, "a:str, b:str, transition:str=fade, seconds:float=1, out:str=videos/xfade.mp4", lambda ctx, a, b, transition, seconds, out: ff(ctx, ["-i", str(ctx.path(a, True)), "-i", str(ctx.path(b, True)), "-filter_complex", f"[0:v][1:v]xfade=transition={transition}:duration={seconds}:offset={max(0, dur(ctx, a) - seconds)},format=yuv420p", "-an"], ctx.path(out)), produces="video", **R)

# ---- table-driven visual filters
VF = {
    "grayscale": ("Black and white.", "hue=s=0", ""), "sepia": ("Sepia tone.", "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131", ""), "negative": ("Inverted colours.", "negate", ""),
    "blur": ("Soft blur.", "gblur=sigma={amount}", "amount:float=6"), "sharpen": ("Sharpen details.", "unsharp=5:5:{amount}", "amount:float=1.5"), "brightness": ("Brighten (−1…1).", "eq=brightness={amount}", "amount:float=0.15"),
    "contrast": ("Contrast (1 = normal).", "eq=contrast={amount}", "amount:float=1.3"), "saturation": ("Saturation (1 = normal, 0 = grey).", "eq=saturation={amount}", "amount:float=1.5"), "gamma": ("Gamma (1 = normal).", "eq=gamma={amount}", "amount:float=1.2"),
    "hue_shift": ("Rotate the colours by degrees.", "hue=h={amount}", "amount:float=90"), "vignette": ("Dark edges.", "vignette", ""), "edges": ("Edge-detection look.", "edgedetect", ""), "pixelate": ("Pixelated look.", "pixelize=w={amount}:h={amount}", "amount:int=16"),
    "flip_horizontal": ("Mirror left–right.", "hflip", ""), "flip_vertical": ("Flip upside down.", "vflip", ""), "rotate_90": ("Rotate 90° clockwise.", "transpose=1", ""), "rotate_180": ("Rotate 180°.", "hflip,vflip", ""),
    "warm": ("Warm colour grade.", "colorbalance=rs=.12:bs=-.12", ""), "cool": ("Cool colour grade.", "colorbalance=rs=-.12:bs=.12", ""), "vintage": ("Faded vintage look.", "curves=vintage,vignette", ""), "high_contrast": ("Punchy contrast + saturation.", "eq=contrast=1.4:saturation=1.3", ""),
    "echo_trail": ("Ghostly motion trails.", "tblend=all_mode=average,framestep=1", ""), "zoom_in": ("Slow zoom-in (Ken Burns).", "zoompan=z='min(zoom+0.0015,1.4)':d=1:s=1280x720", ""), "denoise": ("Reduce noise.", "hqdn3d", ""),
    "letterbox": ("Cinematic black bars (2.35:1).", "crop=iw:iw/2.35,pad=iw:ih*2.35/1:0:(oh-ih)/2", ""), "deshake": ("Quick de-shake.", "deshake", ""), "film_grain": ("Film grain.", "noise=alls=18:allf=t", ""),
}
for _n, (_d, _t, _sig) in VF.items():
    reg(f"video_{_n}", f"{_d} (video effect).", V, "path:str, out:str?" + (", " + _sig if _sig else ""), (lambda n, t: lambda ctx, path, out, **kw: ff(ctx, ["-i", str(ctx.path(path, True)), "-vf", t.format(**kw), "-pix_fmt", "yuv420p"] + (["-c:a", "copy"] if _has_audio(ctx, path) else []), _o(ctx, out, f"videos/{_stem(path)}_{n}.mp4"), n))(_n, _t), produces="video", tags=("filter", "effect", "reel"), **R)

# ---- fit to a platform
FITS = {"instagram_reel": (1080, 1920, "9:16 reel"), "instagram_story": (1080, 1920, "9:16 story"), "tiktok": (1080, 1920, "9:16 TikTok"), "youtube_short": (1080, 1920, "9:16 Short"), "instagram_square": (1080, 1080, "1:1 post"),
        "instagram_portrait": (1080, 1350, "4:5 post"), "youtube_1080p": (1920, 1080, "16:9 HD"), "youtube_720p": (1280, 720, "16:9 720p"), "youtube_4k": (3840, 2160, "16:9 4K"), "twitter": (1280, 720, "16:9"), "whatsapp_status": (720, 1280, "9:16 small")}
for _n, (_w, _h, _d) in FITS.items():
    reg(f"video_fit_{_n}", f"Resize + crop a video to fill {_n.replace('_', ' ')} ({_w}×{_h}, {_d}), H.264, ready to upload.", V, "path:str, out:str?",
        (lambda n, w, h: lambda ctx, path, out: ff(ctx, ["-i", str(ctx.path(path, True)), "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1", "-r", "30", "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart"], _o(ctx, out, f"videos/{_stem(path)}_{n}.mp4"), n))(_n, _w, _h), produces="video", tags=("instagram", "youtube", "social", "upload"), **R)
for _n, _crf in (("small", 32), ("medium", 26), ("high_quality", 20)):
    reg(f"video_compress_{_n}", f"Compress a video ({_n.replace('_', ' ')}: CRF {_crf}).", V, "path:str, out:str?", (lambda n, c: lambda ctx, path, out: ff(ctx, ["-i", str(ctx.path(path, True)), "-c:v", "libx264", "-crf", str(c), "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart"], _o(ctx, out, f"videos/{_stem(path)}_{n}.mp4"), n))(_n, _crf), produces="video", **R)
reg("video_compress_to_size", "Compress a video to about N megabytes.", V, "path:str, megabytes:float=16, out:str?", lambda ctx, path, megabytes, out: (lambda d: ff(ctx, ["-i", str(ctx.path(path, True)), "-c:v", "libx264", "-b:v", f"{max(100, int(megabytes * 8192 / max(d, 1) - 96))}k", "-maxrate", f"{max(100, int(megabytes * 8192 / max(d, 1) - 96)) * 2}k", "-bufsize", "2M", "-c:a", "aac", "-b:a", "96k", "-pix_fmt", "yuv420p"], _o(ctx, out, f"videos/{_stem(path)}_{megabytes:g}MB.mp4")))(dur(ctx, path)), produces="video", **R)

# ------------------------------------------------------------------ generating video from nothing
GEN = {
    "test_pattern": ("A colourful test pattern with a running clock.", "testsrc2=size={size}:rate=30"), "color_bars": ("SMPTE colour bars.", "smptebars=size={size}:rate=30"), "mandelbrot_zoom": ("An endless zoom into the Mandelbrot set.", "mandelbrot=size={size}:rate=30"),
    "game_of_life": ("Conway's Game of Life evolving.", "life=size={size}:rate=15:ratio=0.1:death_color=#0d1b3e:life_color=#ffb86b:mold=10:mold_color=#4f7cff,scale={size}:flags=neighbor"), "cellular_automaton": ("A cellular automaton drawing itself.", "cellauto=size={size}:rate=30:rule=110:scroll=1"),
    "sierpinski": ("A moving Sierpinski carpet.", "sierpinski=size={size}:rate=30:type=carpet"), "gradient_flow": ("Slowly shifting colour gradients.", "gradients=size={size}:rate=30:speed=0.015:n=4"), "rgb_test": ("RGB test pattern.", "rgbtestsrc=size={size}:rate=30"),
    "solid_colour": ("A plain coloured video.", "color=c={color}:size={size}:rate=30"),
}
for _n, (_d, _src) in GEN.items():
    reg(f"video_generate_{_n}", f"Generate a video: {_d}", V, "seconds:float=5, size:str=1280x720, out:str?" + (", color:str=#4f7cff" if "{color}" in _src else ""),
        (lambda n, src: lambda ctx, seconds, size, out, color="#4f7cff": ff(ctx, ["-f", "lavfi", "-i", src.format(size=size, color=color.replace("#", "0x")), "-t", str(seconds), "-pix_fmt", "yuv420p", "-c:v", "libx264"], _o(ctx, out, f"videos/{n}.mp4"), n))(_n, _src), produces="video", tags=("create", "generative"), **R)


def _slides(ctx, texts, seconds, size, c1, c2, out, fade):
    from .t_image import _card
    w, h = (int(x) for x in size.split("x"))
    folder = Path(tempfile.mkdtemp(dir=ctx.workspace))
    try:
        parts = []
        for i, t in enumerate(texts):
            head, _, sub = str(t).partition("|")
            rel = f"{folder.name}/s{i:03d}.png"
            _card(ctx, head.strip(), sub.strip() or None, w, h, c1, c2, "#ffffff", rel)
            ctx.artifacts.pop()
            parts.append(rel)
        args = []
        for p in parts:
            args += ["-loop", "1", "-t", str(seconds), "-i", p]
        n = len(parts)
        if n == 1:
            chain = "[0:v]format=yuv420p[v]"
        elif fade:
            chain, last = "", "[0:v]"
            for i in range(1, n):
                chain += f"{last}[{i}:v]xfade=transition=fade:duration=0.6:offset={i * seconds - 0.6 * i:.2f}[x{i}];"
                last = f"[x{i}]"
            chain += f"{last}format=yuv420p[v]"
        else:
            chain = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0,format=yuv420p[v]"
        return ff(ctx, args + ["-filter_complex", chain, "-map", "[v]", "-r", "30", "-c:v", "libx264"], ctx.path(out), f"{n}-slide video")
    finally:
        shutil.rmtree(folder, ignore_errors=True)


reg("video_from_text_slides", "Make a slide video from texts (use 'Headline | smaller line'). Each slide is a designed gradient card.", V, "slides:list, seconds:float=3, size:str=1280x720, color1:str=#4f7cff, color2:str=#9b5de5, crossfade:bool=true, out:str=videos/slides.mp4",
    lambda ctx, slides, seconds, size, color1, color2, crossfade, out: _slides(ctx, slides, seconds, size, color1, color2, out, crossfade), produces="video", needs=NEED + ("py:PIL",), ctx=True, risk="files", timeout=SLOW, tags=("reel", "create", "explainer"))
reg("video_title_card", "A short title-card video.", V, "title:str, subtitle:str?, seconds:float=4, size:str=1280x720, out:str=videos/title.mp4", lambda ctx, title, subtitle, seconds, size, out: _slides(ctx, [title + ("|" + subtitle if subtitle else "")], seconds, size, "#0d1b3e", "#4f7cff", out, False), produces="video", needs=NEED + ("py:PIL",), ctx=True, risk="files", timeout=SLOW)
reg("video_countdown", "A countdown video (N…1).", V, "from_number:int=5, size:str=1280x720, out:str=videos/countdown.mp4", lambda ctx, from_number, size, out: _slides(ctx, [str(i) for i in range(from_number, 0, -1)], 1, size, "#0d1b3e", "#e5486b", out, False), produces="video", needs=NEED + ("py:PIL",), ctx=True, risk="files", timeout=SLOW)
reg("video_slideshow", "Slideshow of images from the workspace with optional music.", V, "images:list, seconds:float=3, size:str=1280x720, audio:str?, out:str=videos/slideshow.mp4", lambda ctx, images, seconds, size, audio, out: _show(ctx, images, seconds, size, audio, out), produces="video", **R)
reg("video_ken_burns", "Turn one still image into a slowly zooming video.", V, "image:str, seconds:float=6, size:str=1280x720, out:str=videos/ken_burns.mp4", lambda ctx, image, seconds, size, out: ff(ctx, ["-loop", "1", "-i", str(ctx.path(image, True)), "-t", str(seconds), "-vf", f"scale=3000:-1,zoompan=z='min(zoom+0.0012,1.35)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={int(seconds * 30)}:s={size}:fps=30,format=yuv420p", "-c:v", "libx264"], ctx.path(out), "Ken Burns"), produces="video", **R)
reg("video_from_frames", "Assemble numbered PNG/JPG frames (a pattern like frames/f_%04d.png) into a video.", V, "pattern:str=frames/f_%04d.png, fps:int=24, out:str=videos/frames.mp4", lambda ctx, pattern, fps, out: ff(ctx, ["-framerate", str(fps), "-i", str(ctx.path(pattern)), "-c:v", "libx264", "-pix_fmt", "yuv420p"], ctx.path(out), "assembled"), produces="video", **R)
reg("video_from_gif", "Convert an animated GIF to MP4.", V, "path:str, out:str?", lambda ctx, path, out: ff(ctx, ["-i", str(ctx.path(path, True)), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-movflags", "+faststart"], _o(ctx, out, f"videos/{_stem(path)}.mp4")), produces="video", **R)
reg("video_audio_visualizer", "Animated waveform video from an audio file.", V, "audio:str, size:str=1280x720, mode:str=cline, out:str=videos/visualizer.mp4", lambda ctx, audio, size, mode, out: ff(ctx, ["-i", str(ctx.path(audio, True)), "-filter_complex", f"[0:a]showwaves=s={size}:mode={mode}:colors=0x6ee7b7|0x4f7cff,format=yuv420p[v]", "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-c:a", "aac"], ctx.path(out), "waveform video"), produces="video", **R)
reg("video_spectrum_visualizer", "Animated frequency-spectrum video from an audio file.", V, "audio:str, size:str=1280x720, out:str=videos/spectrum.mp4", lambda ctx, audio, size, out: ff(ctx, ["-i", str(ctx.path(audio, True)), "-filter_complex", f"[0:a]showspectrum=s={size}:slide=scroll:color=rainbow,format=yuv420p[v]", "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-c:a", "aac"], ctx.path(out), "spectrum video"), produces="video", **R)
reg("blender_frames_to_video", "Join Blender-rendered frames (renders/frame_0001.png …) into an MP4.", V, "pattern:str=renders/frame_%04d.png, fps:int=24, out:str=videos/render.mp4", lambda ctx, pattern, fps, out: ff(ctx, ["-framerate", str(fps), "-i", str(ctx.path(pattern)), "-c:v", "libx264", "-pix_fmt", "yuv420p"], ctx.path(out), "render"), produces="video", tags=("blender", "3d", "animation"), **R)

# ------------------------------------------------------------------ audio
AF = {
    "volume": ("Change volume (1 = same, 2 = double).", "volume={amount}", "amount:float=1.5"), "normalize": ("Even out the loudness (EBU R128).", "loudnorm", ""), "fade_in_out": ("Fade in and out.", "afade=t=in:d={amount},areverse,afade=t=in:d={amount},areverse", "amount:float=2"),
    "reverse": ("Play backwards.", "areverse", ""), "speed": ("Change speed and pitch together (2 = double).", "asetrate=44100*{amount},aresample=44100", "amount:float=1.25"), "tempo": ("Change tempo, keep pitch (0.5–2).", "atempo={amount}", "amount:float=1.25"),
    "echo": ("Add an echo.", "aecho=0.8:0.88:60:0.4", ""), "chorus": ("Chorus effect.", "chorus=0.5:0.9:50|60:0.4|0.32:0.25|0.4:2|1.3", ""), "flanger": ("Flanger effect.", "flanger", ""), "tremolo": ("Tremolo (wobbling volume).", "tremolo=f=6:d=0.7", ""),
    "vibrato": ("Vibrato (wobbling pitch).", "vibrato=f=6:d=0.5", ""), "bitcrush": ("Lo-fi bit-crusher.", "acrusher=bits=8:mix=0.6", ""), "bass_boost": ("Boost the bass.", "bass=g=10", ""), "treble_boost": ("Boost the treble.", "treble=g=8", ""),
    "high_pass": ("Remove low rumble below N Hz.", "highpass=f={amount}", "amount:float=200"), "low_pass": ("Remove highs above N Hz (muffled).", "lowpass=f={amount}", "amount:float=3000"), "telephone": ("Old-telephone sound.", "highpass=f=300,lowpass=f=3400", ""),
    "mono": ("Mix to mono.", "pan=mono|c0=0.5*c0+0.5*c1", ""), "pitch_up": ("Raise pitch by a semitone ratio without changing speed.", "rubberband=pitch={amount}", "amount:float=1.2"), "pitch_down": ("Lower pitch without changing speed.", "rubberband=pitch={amount}", "amount:float=0.85"),
    "trim_silence": ("Remove silence at the start and end.", "silenceremove=start_periods=1:start_threshold=-50dB:stop_periods=1:stop_threshold=-50dB", ""), "robot_voice": ("Robot voice.", "afftfilt=real='hypot(re,im)*sin(0)':imag='hypot(re,im)*cos(0)':win_size=512:overlap=0.75", ""),
}
for _n, (_d, _f, _sig) in AF.items():
    reg(f"audio_{_n}", f"{_d} (audio effect).", A, "path:str, out:str?" + (", " + _sig if _sig else ""), (lambda n, f: lambda ctx, path, out, **kw: ff(ctx, ["-i", str(ctx.path(path, True)), "-vn", "-af", f.format(**kw)], _o(ctx, out, f"audio/{_stem(path)}_{n}.wav"), n, "audio"))(_n, _f), produces="audio", tags=("sound", "effect"), **R)
reg("audio_trim", "Cut part of an audio file.", A, "path:str, start:float=0, seconds:float=10, out:str?", lambda ctx, path, start, seconds, out: ff(ctx, ["-ss", str(start), "-i", str(ctx.path(path, True)), "-t", str(seconds)], _o(ctx, out, f"audio/{_stem(path)}_trim.wav"), "trimmed", "audio"), produces="audio", **R)
reg("audio_concat", "Join audio files end to end.", A, "paths:list, out:str=audio/joined.wav", lambda ctx, paths, out: ff(ctx, sum([["-i", str(ctx.path(p, True))] for p in paths], []) + ["-filter_complex", "".join(f"[{i}:a]" for i in range(len(paths))) + f"concat=n={len(paths)}:v=0:a=1[a]", "-map", "[a]"], ctx.path(out), "joined", "audio"), produces="audio", **R)
reg("audio_mix", "Mix several audio files on top of each other.", A, "paths:list, out:str=audio/mix.wav", lambda ctx, paths, out: ff(ctx, sum([["-i", str(ctx.path(p, True))] for p in paths], []) + ["-filter_complex", f"amix=inputs={len(paths)}:duration=longest"], ctx.path(out), "mixed", "audio"), produces="audio", **R)
reg("audio_tone", "Generate a sine tone.", A, "frequency:float=440, seconds:float=2, out:str=audio/tone.wav", lambda ctx, frequency, seconds, out: ff(ctx, ["-f", "lavfi", "-i", f"sine=frequency={frequency}:duration={seconds}"], ctx.path(out), "tone", "audio"), produces="audio", **R)
reg("audio_silence", "Generate silence.", A, "seconds:float=2, out:str=audio/silence.wav", lambda ctx, seconds, out: ff(ctx, ["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo", "-t", str(seconds)], ctx.path(out), "silence", "audio"), produces="audio", **R)
for _c in ("white", "pink", "brown"):
    reg(f"audio_noise_{_c}", f"Generate {_c} noise.", A, "seconds:float=5, out:str?", (lambda c: lambda ctx, seconds, out: ff(ctx, ["-f", "lavfi", "-i", f"anoisesrc=color={c}:duration={seconds}:amplitude=0.3"], _o(ctx, out, f"audio/{c}_noise.wav"), f"{c} noise", "audio"))(_c), produces="audio", **R)
reg("audio_melody", "Play a melody from note names like 'C4 E4 G4 C5' (quarter-second notes; '-' is a rest).", A, "notes:str, note_seconds:float=0.3, out:str=audio/melody.wav", lambda ctx, notes, note_seconds, out: _melody(ctx, notes, note_seconds, out), produces="audio", **R)
reg("audio_chord", "Play a chord from frequencies or note names, e.g. 'C4 E4 G4'.", A, "notes:str, seconds:float=2, out:str=audio/chord.wav", lambda ctx, notes, seconds, out: ff(ctx, sum([["-f", "lavfi", "-i", f"sine=frequency={_hz(n):.2f}:duration={seconds}"] for n in notes.split()], []) + ["-filter_complex", f"amix=inputs={len(notes.split())}:normalize=0,volume=0.6"], ctx.path(out), "chord", "audio"), produces="audio", **R)
reg("audio_waveform_image", "Draw an audio file's waveform as a PNG.", A, "path:str, size:str=1200x300, color:str=4f7cff, out:str?", lambda ctx, path, size, color, out: ff(ctx, ["-i", str(ctx.path(path, True)), "-filter_complex", f"showwavespic=s={size}:colors=0x{color.lstrip('#')}", "-frames:v", "1"], _o(ctx, out, f"images/{_stem(path)}_wave.png"), "waveform", "image"), produces="image", **R)
reg("audio_spectrogram_image", "Draw an audio file's spectrogram as a PNG.", A, "path:str, size:str=1200x500, out:str?", lambda ctx, path, size, out: ff(ctx, ["-i", str(ctx.path(path, True)), "-lavfi", f"showspectrumpic=s={size}:legend=0", "-frames:v", "1"], _o(ctx, out, f"images/{_stem(path)}_spectrogram.png"), "spectrogram", "image"), produces="image", **R)
reg("audio_info", "Duration and format of an audio file.", A, "path:str", probe, ctx=True, risk="files", needs=("bin:ffprobe",))
reg("audio_extract_from_video", "Save a video's sound track as MP3.", A, "path:str, out:str?", lambda ctx, path, out: ff(ctx, ["-i", str(ctx.path(path, True)), "-vn", "-c:a", "libmp3lame", "-q:a", "2"], _o(ctx, out, f"audio/{_stem(path)}.mp3"), "audio", "audio"), produces="audio", **R)
reg("subtitles_make_srt", "Write an .srt subtitle file from lines of text spread evenly over a duration.", V, "lines:list, total_seconds:float=30, out:str=subtitles/subs.srt", lambda ctx, lines, total_seconds, out: _srt(ctx, lines, total_seconds, out), ctx=True, risk="files", produces="file")


# ------------------------------------------------------------------ helpers
def _has_audio(ctx, path):
    return any(s["type"] == "audio" for s in probe(ctx, path)["streams"])


def _atempo(f):
    parts = []
    while f > 2:
        parts.append("atempo=2.0")
        f /= 2
    while f < 0.5:
        parts.append("atempo=0.5")
        f /= 0.5
    return ",".join(parts + [f"atempo={f:.4f}"])


def _dt(t):
    return t.replace("\\", "\\\\").replace("'", "’").replace(":", "\\:").replace("%", "\\%")


def _concat(ctx, paths, w, h, out):
    args, chain = [], ""
    for i, p in enumerate(paths):
        args += ["-i", str(ctx.path(p, True))]
        chain += f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[v{i}];"
    chain += "".join(f"[v{i}]" for i in range(len(paths))) + f"concat=n={len(paths)}:v=1:a=0,format=yuv420p[v]"
    return ff(ctx, args + ["-filter_complex", chain, "-map", "[v]", "-c:v", "libx264"], ctx.path(out), "joined (video only)")


def _show(ctx, images, seconds, size, audio, out):
    w, h = (int(x) for x in size.split("x"))
    args, chain = [], ""
    for i, p in enumerate(images):
        args += ["-loop", "1", "-t", str(seconds), "-i", str(ctx.path(p, True))]
        chain += f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[v{i}];"
    chain += "".join(f"[v{i}]" for i in range(len(images))) + f"concat=n={len(images)}:v=1:a=0,format=yuv420p[v]"
    if audio:
        args += ["-i", str(ctx.path(audio, True))]
    return ff(ctx, args + ["-filter_complex", chain, "-map", "[v]"] + (["-map", f"{len(images)}:a", "-c:a", "aac", "-shortest"] if audio else []) + ["-c:v", "libx264"], ctx.path(out), "slideshow")


def _stab(ctx, path, out):
    src = str(ctx.path(path, True))
    trf = str(ctx.path("videos/.transforms.trf"))
    ff(ctx, ["-i", src, "-vf", f"vidstabdetect=result={trf}", "-f", "null", "-"])
    return ff(ctx, ["-i", src, "-vf", f"vidstabtransform=input={trf}:zoom=1:smoothing=10,unsharp=5:5:0.8", "-pix_fmt", "yuv420p"], _o(ctx, out, f"videos/{_stem(path)}_stable.mp4"), "stabilised")


_N = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _hz(n: str) -> float:
    try:
        return float(n)
    except ValueError:
        pass
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(\d)", n)
    if not m:
        raise ToolError(f"cannot read note '{n}' (use names like C4, F#3, Bb5 or a frequency)")
    k = _N[m.group(1).upper()] + (1 if m.group(2) == "#" else -1 if m.group(2) == "b" else 0) + 12 * (int(m.group(3)) + 1)
    return 440 * 2 ** ((k - 69) / 12)


def _melody(ctx, notes, dur_, out):
    toks = notes.split()
    if not toks or len(toks) > 128:
        raise ToolError("give 1–128 notes")
    args, chain = [], ""
    for i, n in enumerate(toks):
        src = f"anullsrc=r=44100:cl=mono,atrim=0:{dur_}" if n == "-" else f"sine=frequency={_hz(n):.2f}:duration={dur_}"
        args += ["-f", "lavfi", "-i", src]
        chain += f"[{i}:a]afade=t=out:st={dur_ * .8:.3f}:d={dur_ * .2:.3f}[a{i}];"
    chain += "".join(f"[a{i}]" for i in range(len(toks))) + f"concat=n={len(toks)}:v=0:a=1[a]"
    return ff(ctx, args + ["-filter_complex", chain, "-map", "[a]"], ctx.path(out), "melody", "audio")


def _srt(ctx, lines, total, out):
    def t(s):
        return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d},{int(s % 1 * 1000):03d}"
    step = total / max(1, len(lines))
    p = ctx.path(out)
    p.write_text("\n".join(f"{i + 1}\n{t(i * step)} --> {t((i + 1) * step - .05)}\n{l}\n" for i, l in enumerate(lines)), encoding="utf-8")
    return {"saved": ctx.produced(p, "file", "subtitles")}
