"""Charts, diagrams and geometry figures drawn as SVG — no plotting library required."""

from __future__ import annotations

import html
import math

from .registry import Context, ToolError, reg
from .t_math import evaluate, num_list

PAL = ["#4f7cff", "#ff7a59", "#2bb673", "#f2b134", "#9b5de5", "#00b8d9", "#e5486b", "#7a8599"]
INK, MUTED, GRID, BG = "#1c2433", "#6b7689", "#e3e8f0", "#ffffff"
W, H, M = 640, 400, 52


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def svg(body: str, w: int = W, h: int = H, bg: str = BG) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="system-ui,Segoe UI,Arial,sans-serif" font-size="12">'
            f'<rect width="{w}" height="{h}" fill="{bg}"/>{body}</svg>')


def title(t: str, w: int = W) -> str:
    return f'<text x="{w / 2}" y="24" text-anchor="middle" font-size="16" font-weight="600" fill="{INK}">{esc(t)}</text>' if t else ""


def save(ctx: Context, path: str, body: str, note: str, w: int = W, h: int = H) -> dict:
    p = ctx.path(path if path.endswith(".svg") else path + ".svg")
    p.write_text(svg(body, w, h), encoding="utf-8")
    return {"saved": ctx.produced(p, "image", note)}


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi == lo:
        hi = lo + 1
    step = 10 ** math.floor(math.log10((hi - lo) / n))
    for m in (1, 2, 2.5, 5, 10):
        if (hi - lo) / (step * m) <= n:
            step *= m
            break
    t, out = math.floor(lo / step) * step, []
    while t <= hi + step * 0.5:
        out.append(round(t, 10))
        t += step
    return out


def axes(xlo, xhi, ylo, yhi, xlabel="", ylabel="", xticks=True):
    sx = lambda x: M + (x - xlo) / ((xhi - xlo) or 1) * (W - 2 * M)  # noqa: E731
    sy = lambda y: H - M - (y - ylo) / ((yhi - ylo) or 1) * (H - 2 * M)  # noqa: E731
    g = ""
    for t in nice_ticks(ylo, yhi):
        if ylo <= t <= yhi:
            g += f'<line x1="{M}" x2="{W - M}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="{GRID}"/><text x="{M - 8}" y="{sy(t) + 4:.1f}" text-anchor="end" fill="{MUTED}">{t:g}</text>'
    if xticks:
        for t in nice_ticks(xlo, xhi):
            if xlo <= t <= xhi:
                g += f'<text x="{sx(t):.1f}" y="{H - M + 18}" text-anchor="middle" fill="{MUTED}">{t:g}</text>'
    g += f'<line x1="{M}" x2="{W - M}" y1="{H - M}" y2="{H - M}" stroke="{MUTED}"/><line x1="{M}" x2="{M}" y1="{M}" y2="{H - M}" stroke="{MUTED}"/>'
    if xlabel:
        g += f'<text x="{W / 2}" y="{H - 10}" text-anchor="middle" fill="{MUTED}">{esc(xlabel)}</text>'
    if ylabel:
        g += f'<text transform="translate(14 {H / 2}) rotate(-90)" text-anchor="middle" fill="{MUTED}">{esc(ylabel)}</text>'
    return sx, sy, g


def labels_list(v, n):
    if v in (None, "", []):
        return [str(i + 1) for i in range(n)]
    return [s.strip() for s in (v.split(",") if isinstance(v, str) else map(str, v))][:n]


def series_arg(values):
    """A flat list → one series; a list of lists or dict → several."""
    if isinstance(values, dict):
        return {k: num_list(v) for k, v in values.items()}
    if isinstance(values, list) and values and isinstance(values[0], (list, tuple)):
        return {f"Series {i + 1}": num_list(v) for i, v in enumerate(values)}
    return {"": num_list(values)}


def legend(names):
    names = [n for n in names if n]
    return "".join(f'<rect x="{W - M - 110}" y="{34 + i * 18}" width="10" height="10" rx="2" fill="{PAL[i % len(PAL)]}"/><text x="{W - M - 94}" y="{43 + i * 18}" fill="{INK}">{esc(n)}</text>' for i, n in enumerate(names))


C = "charts"


def bar_chart(ctx, path, values, labels=None, title_="", xlabel="", ylabel="", horizontal=False):
    s = series_arg(values)
    n = max(len(v) for v in s.values())
    lab = labels_list(labels, n)
    hi = max(max(v) for v in s.values())
    lo = min(0, min(min(v) for v in s.values()))
    if horizontal:
        sx, sy, g = axes(lo, hi, 0, n, xlabel, ylabel, True)
        g = "".join(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{M}" y2="{H - M}" stroke="{GRID}"/><text x="{sx(t):.1f}" y="{H - M + 18}" text-anchor="middle" fill="{MUTED}">{t:g}</text>' for t in nice_ticks(lo, hi))
        g += f'<line x1="{M}" x2="{M}" y1="{M}" y2="{H - M}" stroke="{MUTED}"/>'
        bh = (H - 2 * M) / n
        bars = ""
        for i in range(n):
            bars += f'<text x="{M - 8}" y="{M + i * bh + bh / 2 + 4:.1f}" text-anchor="end" fill="{INK}">{esc(lab[i])}</text>'
            for j, (name, v) in enumerate(s.items()):
                if i < len(v):
                    h_ = bh * 0.7 / len(s)
                    bars += f'<rect x="{sx(0):.1f}" y="{M + i * bh + bh * .15 + j * h_:.1f}" width="{abs(sx(v[i]) - sx(0)):.1f}" height="{h_:.1f}" rx="3" fill="{PAL[j % len(PAL)]}"/>'
        return save(ctx, path, title(title_) + g + bars + legend(s), "bar chart")
    sx, sy, g = axes(0, n, lo, hi, xlabel, ylabel, False)
    bw = (W - 2 * M) / n
    bars = ""
    for i in range(n):
        bars += f'<text x="{M + i * bw + bw / 2:.1f}" y="{H - M + 18}" text-anchor="middle" fill="{MUTED}">{esc(lab[i])}</text>'
        for j, (name, v) in enumerate(s.items()):
            if i < len(v):
                w_ = bw * 0.72 / len(s)
                x = M + i * bw + bw * .14 + j * w_
                y0, y1 = sy(0), sy(v[i])
                bars += f'<rect x="{x:.1f}" y="{min(y0, y1):.1f}" width="{w_:.1f}" height="{abs(y0 - y1):.1f}" rx="3" fill="{PAL[j % len(PAL)]}"><title>{esc(name)} {esc(lab[i])}: {v[i]:g}</title></rect>'
    return save(ctx, path, title(title_) + g + bars + legend(s), "bar chart")


def line_chart(ctx, path, values, labels=None, title_="", xlabel="", ylabel="", area=False):
    s = series_arg(values)
    n = max(len(v) for v in s.values())
    lab = labels_list(labels, n)
    hi, lo = max(max(v) for v in s.values()), min(min(v) for v in s.values())
    lo = min(lo, 0) if lo > 0 and lo < hi * .5 else lo
    sx, sy, g = axes(0, max(n - 1, 1), lo, hi, xlabel, ylabel, False)
    out = ""
    for i in range(0, n, max(1, n // 10)):
        out += f'<text x="{sx(i):.1f}" y="{H - M + 18}" text-anchor="middle" fill="{MUTED}">{esc(lab[i])}</text>'
    for j, (name, v) in enumerate(s.items()):
        pts = " ".join(f"{sx(i):.1f},{sy(y):.1f}" for i, y in enumerate(v))
        c = PAL[j % len(PAL)]
        if area:
            out += f'<polygon points="{sx(0):.1f},{sy(max(lo, 0)):.1f} {pts} {sx(len(v) - 1):.1f},{sy(max(lo, 0)):.1f}" fill="{c}" opacity=".18"/>'
        out += f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.4" stroke-linejoin="round"/>' + "".join(f'<circle cx="{sx(i):.1f}" cy="{sy(y):.1f}" r="3" fill="{c}"><title>{esc(lab[i])}: {y:g}</title></circle>' for i, y in enumerate(v) if n <= 40)
    return save(ctx, path, title(title_) + g + out + legend(s), "line chart")


def pie_chart(ctx, path, values, labels=None, title_="", donut=False):
    v = num_list(values)
    tot = sum(v) or 1
    lab = labels_list(labels, len(v))
    cx, cy, r = W / 2 - 60, H / 2 + 10, 130
    a, out = -math.pi / 2, ""
    for i, x in enumerate(v):
        b = a + x / tot * 2 * math.pi
        large = 1 if b - a > math.pi else 0
        x1, y1, x2, y2 = cx + r * math.cos(a), cy + r * math.sin(a), cx + r * math.cos(b), cy + r * math.sin(b)
        out += f'<path d="M{cx},{cy} L{x1:.1f},{y1:.1f} A{r},{r} 0 {large} 1 {x2:.1f},{y2:.1f} Z" fill="{PAL[i % len(PAL)]}" stroke="#fff" stroke-width="2"><title>{esc(lab[i])}: {x:g} ({x / tot * 100:.1f}%)</title></path>'
        out += f'<rect x="{W - 190}" y="{60 + i * 22}" width="12" height="12" rx="3" fill="{PAL[i % len(PAL)]}"/><text x="{W - 172}" y="{71 + i * 22}" fill="{INK}">{esc(lab[i])} · {x / tot * 100:.0f}%</text>'
        a = b
    if donut:
        out += f'<circle cx="{cx}" cy="{cy}" r="{r * .55}" fill="{BG}"/><text x="{cx}" y="{cy + 6}" text-anchor="middle" font-size="18" font-weight="600" fill="{INK}">{tot:g}</text>'
    return save(ctx, path, title(title_) + out, "pie chart")


def scatter_chart(ctx, path, x, y, title_="", xlabel="", ylabel="", fit=False):
    xs, ys = num_list(x), num_list(y)
    if len(xs) != len(ys):
        raise ToolError("x and y must have the same length")
    sx, sy, g = axes(min(xs), max(xs), min(ys), max(ys), xlabel, ylabel)
    out = "".join(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="4" fill="{PAL[0]}" opacity=".75"><title>({a:g}, {b:g})</title></circle>' for a, b in zip(xs, ys))
    if fit and len(xs) > 1:
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((a - mx) ** 2 for a in xs) or 1
        k = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / den
        c = my - k * mx
        out += f'<line x1="{sx(min(xs)):.1f}" y1="{sy(c + k * min(xs)):.1f}" x2="{sx(max(xs)):.1f}" y2="{sy(c + k * max(xs)):.1f}" stroke="{PAL[1]}" stroke-width="2" stroke-dasharray="6 4"/>'
    return save(ctx, path, title(title_) + g + out, "scatter plot")


def histogram_chart(ctx, path, values, bins=10, title_="", xlabel=""):
    v = num_list(values)
    lo, hi = min(v), max(v)
    w = (hi - lo) / bins or 1
    cnt = [0] * bins
    for x in v:
        cnt[min(bins - 1, int((x - lo) / w))] += 1
    sx, sy, g = axes(lo, hi if hi > lo else lo + 1, 0, max(cnt), xlabel, "count")
    out = "".join(f'<rect x="{sx(lo + i * w) + 1:.1f}" y="{sy(c):.1f}" width="{max(1, sx(lo + (i + 1) * w) - sx(lo + i * w) - 2):.1f}" height="{sy(0) - sy(c):.1f}" fill="{PAL[0]}" rx="2"><title>{c}</title></rect>' for i, c in enumerate(cnt))
    return save(ctx, path, title(title_) + g + out, "histogram")


def radar_chart(ctx, path, values, labels=None, title_=""):
    s = series_arg(values)
    n = max(len(v) for v in s.values())
    lab = labels_list(labels, n)
    hi = max(max(v) for v in s.values()) or 1
    cx, cy, r = W / 2, H / 2 + 12, 135
    pt = lambda i, f: (cx + r * f * math.sin(2 * math.pi * i / n), cy - r * f * math.cos(2 * math.pi * i / n))  # noqa: E731
    out = "".join(f'<polygon points="{" ".join("%.1f,%.1f" % pt(i, k / 4) for i in range(n))}" fill="none" stroke="{GRID}"/>' for k in range(1, 5))
    for i in range(n):
        x, y = pt(i, 1)
        lx, ly = pt(i, 1.12)
        out += f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{GRID}"/><text x="{lx:.1f}" y="{ly + 4:.1f}" text-anchor="middle" fill="{INK}">{esc(lab[i])}</text>'
    for j, v in enumerate(s.values()):
        out += f'<polygon points="{" ".join("%.1f,%.1f" % pt(i, v[i] / hi) for i in range(len(v)))}" fill="{PAL[j % len(PAL)]}" fill-opacity=".22" stroke="{PAL[j % len(PAL)]}" stroke-width="2"/>'
    return save(ctx, path, title(title_) + out + legend(s), "radar chart")


def heatmap_chart(ctx, path, matrix, row_labels=None, col_labels=None, title_=""):
    m = [num_list(r) for r in matrix]
    R, Cc = len(m), len(m[0])
    lo, hi = min(map(min, m)), max(map(max, m))
    cw, ch = (W - 2 * M) / Cc, (H - 2 * M) / R
    rl, cl = labels_list(row_labels, R), labels_list(col_labels, Cc)
    out = ""
    for i, row in enumerate(m):
        out += f'<text x="{M - 6}" y="{M + i * ch + ch / 2 + 4:.1f}" text-anchor="end" fill="{INK}">{esc(rl[i])}</text>'
        for j, v in enumerate(row):
            t = (v - lo) / ((hi - lo) or 1)
            col = f"rgb({round(240 - 190 * t)},{round(244 - 150 * t)},{round(255 - 60 * t)})"
            out += f'<rect x="{M + j * cw:.1f}" y="{M + i * ch:.1f}" width="{cw:.1f}" height="{ch:.1f}" fill="{col}"/><text x="{M + j * cw + cw / 2:.1f}" y="{M + i * ch + ch / 2 + 4:.1f}" text-anchor="middle" fill="{"#fff" if t > .6 else INK}">{v:g}</text>'
    out += "".join(f'<text x="{M + j * cw + cw / 2:.1f}" y="{M - 8}" text-anchor="middle" fill="{INK}">{esc(cl[j])}</text>' for j in range(Cc))
    return save(ctx, path, title(title_) + out, "heatmap")


def function_plot(ctx, path, expressions, xmin=-10, xmax=10, title_=""):
    exprs = [e.strip() for e in (expressions.split(";") if isinstance(expressions, str) else expressions) if str(e).strip()]
    series, ys = [], []
    for e in exprs:
        pts = []
        for i in range(401):
            x = xmin + (xmax - xmin) * i / 400
            try:
                y = evaluate(e, x=x)
                y = y.real if isinstance(y, complex) else y
                if math.isfinite(y) and abs(y) < 1e9:
                    pts.append((x, y))
                    ys.append(y)
                else:
                    pts.append(None)
            except ToolError:
                pts.append(None)
        series.append(pts)
    if not ys:
        raise ToolError("the expressions have no real values on that interval")
    ys.sort()
    ylo, yhi = ys[int(len(ys) * .01)], ys[int(len(ys) * .99)]
    pad = (yhi - ylo) * .08 or 1
    sx, sy, g = axes(xmin, xmax, ylo - pad, yhi + pad)
    out = ""
    if xmin < 0 < xmax:
        out += f'<line x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{M}" y2="{H - M}" stroke="{MUTED}" stroke-dasharray="3 3"/>'
    if ylo - pad < 0 < yhi + pad:
        out += f'<line x1="{M}" x2="{W - M}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke="{MUTED}" stroke-dasharray="3 3"/>'
    for j, pts in enumerate(series):
        d, pen = "", False
        for p in pts:
            if p is None or not (ylo - pad * 4 <= p[1] <= yhi + pad * 4):
                pen = False
                continue
            d += f"{'L' if pen else 'M'}{sx(p[0]):.1f},{sy(p[1]):.1f} "
            pen = True
        out += f'<path d="{d}" fill="none" stroke="{PAL[j % len(PAL)]}" stroke-width="2.4"/>'
    return save(ctx, path, title(title_ or "y = " + exprs[0]) + g + out + legend(["y = " + e for e in exprs]), "function plot")


def gauge_chart(ctx, path, value, minimum=0, maximum=100, label=""):
    t = min(1, max(0, (value - minimum) / ((maximum - minimum) or 1)))
    cx, cy, r = W / 2, H / 2 + 60, 140
    a = math.pi * (1 - t)
    x, y = cx + r * math.cos(a), cy - r * math.sin(a)
    out = (f'<path d="M{cx - r},{cy} A{r},{r} 0 0 1 {cx + r},{cy}" fill="none" stroke="{GRID}" stroke-width="26" stroke-linecap="round"/>'
           f'<path d="M{cx - r},{cy} A{r},{r} 0 0 1 {x:.1f},{y:.1f}" fill="none" stroke="{PAL[2] if t > .66 else PAL[3] if t > .33 else PAL[1]}" stroke-width="26" stroke-linecap="round"/>'
           f'<text x="{cx}" y="{cy - 10}" text-anchor="middle" font-size="44" font-weight="700" fill="{INK}">{value:g}</text><text x="{cx}" y="{cy + 24}" text-anchor="middle" fill="{MUTED}">{esc(label)}</text>')
    return save(ctx, path, out, "gauge")


def progress_bars(ctx, path, values, labels=None, title_=""):
    v = num_list(values)
    lab = labels_list(labels, len(v))
    out = ""
    for i, x in enumerate(v):
        y = 60 + i * 44
        out += f'<text x="{M}" y="{y}" fill="{INK}">{esc(lab[i])}</text><text x="{W - M}" y="{y}" text-anchor="end" fill="{MUTED}">{x:g}%</text><rect x="{M}" y="{y + 8}" width="{W - 2 * M}" height="12" rx="6" fill="{GRID}"/><rect x="{M}" y="{y + 8}" width="{(W - 2 * M) * min(100, x) / 100:.1f}" height="12" rx="6" fill="{PAL[2] if x >= 90 else PAL[0]}"/>'
    return save(ctx, path, title(title_) + out, "progress bars", W, max(H, 80 + 44 * len(v)))


def sparkline(ctx, path, values):
    v = num_list(values)
    lo, hi = min(v), max(v)
    pts = " ".join(f"{6 + i * 188 / max(1, len(v) - 1):.1f},{34 - (y - lo) / ((hi - lo) or 1) * 28:.1f}" for i, y in enumerate(v))
    p = ctx.path(path if path.endswith(".svg") else path + ".svg")
    p.write_text(svg(f'<polyline points="{pts}" fill="none" stroke="{PAL[0]}" stroke-width="2"/>', 200, 40), encoding="utf-8")
    return {"saved": ctx.produced(p, "image", "sparkline")}


def table_image(ctx, path, rows, title_=""):
    n, cols = len(rows), len(rows[0])
    cw = (W - 2 * 24) / cols
    out = title(title_)
    for i, r in enumerate(rows):
        y = 50 + i * 30
        out += f'<rect x="24" y="{y}" width="{W - 48}" height="30" fill="{PAL[0] if i == 0 else "#f6f8fc" if i % 2 else BG}"/>'
        out += "".join(f'<text x="{32 + j * cw:.1f}" y="{y + 20}" fill="{"#fff" if i == 0 else INK}" font-weight="{600 if i == 0 else 400}">{esc(c)}</text>' for j, c in enumerate(r))
    return save(ctx, path, out, "table", W, 70 + n * 30)


def timeline_chart(ctx, path, events, title_=""):
    """events: list of [year, label]."""
    ev = sorted(([float(a), str(b)] for a, b in events), key=lambda e: e[0])
    lo, hi = ev[0][0], ev[-1][0]
    sx = lambda x: 50 + (x - lo) / ((hi - lo) or 1) * (W - 100)  # noqa: E731
    out = f'<line x1="40" x2="{W - 40}" y1="{H / 2}" y2="{H / 2}" stroke="{MUTED}" stroke-width="3"/>'
    for i, (y, t) in enumerate(ev):
        up = i % 2 == 0
        out += f'<circle cx="{sx(y):.1f}" cy="{H / 2}" r="6" fill="{PAL[i % len(PAL)]}"/><line x1="{sx(y):.1f}" x2="{sx(y):.1f}" y1="{H / 2}" y2="{H / 2 + (-50 if up else 50)}" stroke="{GRID}"/><text x="{sx(y):.1f}" y="{H / 2 + (-78 if up else 74)}" text-anchor="middle" font-weight="600" fill="{INK}">{y:g}</text><text x="{sx(y):.1f}" y="{H / 2 + (-60 if up else 92)}" text-anchor="middle" fill="{MUTED}">{esc(t[:26])}</text>'
    return save(ctx, path, title(title_) + out, "timeline")


for _n, _f, _d, _sig in (
    ("bar_chart", bar_chart, "Bar chart from values (a list, or {series: values}).", "path:str, values:list, labels:str?, title:str?, xlabel:str?, ylabel:str?, horizontal:bool=false"),
    ("line_chart", line_chart, "Line chart from values (a list, or {series: values}).", "path:str, values:list, labels:str?, title:str?, xlabel:str?, ylabel:str?"),
    ("area_chart", lambda *a, **k: line_chart(*a, area=True, **k), "Area chart.", "path:str, values:list, labels:str?, title:str?, xlabel:str?, ylabel:str?"),
    ("pie_chart", pie_chart, "Pie chart with percentages.", "path:str, values:list, labels:str?, title:str?"),
    ("donut_chart", lambda *a, **k: pie_chart(*a, donut=True, **k), "Donut chart with the total in the middle.", "path:str, values:list, labels:str?, title:str?"),
    ("scatter_plot", scatter_chart, "Scatter plot (fit=true adds a trend line).", "path:str, x:list, y:list, title:str?, xlabel:str?, ylabel:str?, fit:bool=false"),
    ("histogram", histogram_chart, "Histogram of numbers.", "path:str, values:list, bins:int=10, title:str?, xlabel:str?"),
    ("radar_chart", radar_chart, "Radar (spider) chart.", "path:str, values:list, labels:str?, title:str?"),
    ("heatmap", heatmap_chart, "Heat-map of a matrix.", "path:str, matrix:list, row_labels:str?, col_labels:str?, title:str?"),
    ("plot_function", function_plot, "Plot one or more functions of x, separated by ';' (e.g. 'sin(x); x^2/10').", "path:str, expressions:str, xmin:float=-10, xmax:float=10, title:str?"),
    ("gauge_chart", gauge_chart, "Speedometer-style gauge.", "path:str, value:float, minimum:float=0, maximum:float=100, label:str?"),
    ("progress_bars", progress_bars, "Horizontal progress bars (0–100).", "path:str, values:list, labels:str?, title:str?"),
    ("sparkline", sparkline, "Tiny inline trend line.", "path:str, values:list"),
    ("table_image", table_image, "Render rows (first row is the header) as a table image.", "path:str, rows:list, title:str?"),
    ("timeline", timeline_chart, "Timeline from [[year, label], …].", "path:str, events:list, title:str?"),
):
    reg(_n, _d + " Saves an SVG into the workspace.", C, _sig, (lambda f: lambda ctx, **kw: f(ctx, **{("title_" if k == "title" else k): v for k, v in kw.items()}))(_f), ctx=True, risk="files", produces="image", tags=("plot", "graph", "visualise", "matplotlib"))
