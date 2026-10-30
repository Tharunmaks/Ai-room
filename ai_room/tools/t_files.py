"""Workspace files, a Python sandbox, archives, and Colab / Kaggle notebook generators."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from .registry import Context, ToolError, reg

F = "files"
MAX_READ = 200_000


def _w(ctx: Context, path: str, content: str, kind: str = "file", note: str = "") -> dict:
    p = ctx.path(path)
    p.write_text(content, encoding="utf-8")
    return {"saved": ctx.produced(p, kind, note), "bytes": p.stat().st_size}


reg("write_file", "Save text to a file in the workspace.", F, "path:str, content:str", lambda ctx, path, content: _w(ctx, path, content), ctx=True, risk="files")
reg("append_file", "Append text to a workspace file.", F, "path:str, content:str", lambda ctx, path, content: (lambda p: (p.open("a", encoding="utf-8").write(content), {"saved": ctx.produced(p)})[1])(ctx.path(path)), ctx=True, risk="files")
reg("read_file", "Read a text file from the workspace.", F, "path:str, max_chars:int=20000", lambda ctx, path, max_chars: ctx.path(path, True).read_text(encoding="utf-8", errors="replace")[:min(max_chars, MAX_READ)], ctx=True, risk="files")
reg("read_lines", "Read a range of lines from a workspace file.", F, "path:str, start:int=1, count:int=50", lambda ctx, path, start, count: "\n".join(ctx.path(path, True).read_text(errors="replace").splitlines()[start - 1:start - 1 + count]), ctx=True, risk="files")
reg("list_files", "List files in the workspace (optionally a sub-folder).", F, "folder:str=.", lambda ctx, folder: sorted(str(p.relative_to(ctx.workspace.resolve())) + ("/" if p.is_dir() else "") for p in ctx.path(folder).resolve().rglob("*"))[:300], ctx=True, risk="files")
reg("file_info", "Size and type of a workspace file.", F, "path:str", lambda ctx, path: (lambda p: {"bytes": p.stat().st_size, "is_dir": p.is_dir(), "suffix": p.suffix, "modified": p.stat().st_mtime})(ctx.path(path, True)), ctx=True, risk="files")
reg("delete_file", "Delete a workspace file.", F, "path:str", lambda ctx, path: (ctx.path(path, True).unlink(), "deleted")[1], ctx=True, risk="files")
reg("copy_file", "Copy a workspace file.", F, "source:str, destination:str", lambda ctx, source, destination: (shutil.copy(ctx.path(source, True), ctx.path(destination)), ctx.produced(ctx.path(destination)))[1], ctx=True, risk="files")
reg("move_file", "Move or rename a workspace file.", F, "source:str, destination:str", lambda ctx, source, destination: (shutil.move(ctx.path(source, True), ctx.path(destination)), "moved")[1], ctx=True, risk="files")
reg("make_folder", "Create a folder in the workspace.", F, "path:str", lambda ctx, path: (ctx.path(path + "/.keep").parent, "created")[1], ctx=True, risk="files")
reg("search_in_files", "Find a text in workspace files.", F, "query:str, folder:str=.", lambda ctx, query, folder: _grep(ctx, query, folder), ctx=True, risk="files")
reg("file_hash", "SHA-256 of a workspace file.", F, "path:str", lambda ctx, path: hashlib.sha256(ctx.path(path, True).read_bytes()).hexdigest(), ctx=True, risk="files")
reg("count_lines_in_file", "Line and word counts of a workspace file.", F, "path:str", lambda ctx, path: (lambda t: {"lines": len(t.splitlines()), "words": len(t.split())})(ctx.path(path, True).read_text(errors="replace")), ctx=True, risk="files")
reg("zip_files", "Zip workspace files or folders into an archive.", F, "paths:list, archive:str=archive.zip", lambda ctx, paths, archive: _zip(ctx, paths, archive), ctx=True, risk="files", produces="file")
reg("unzip_file", "Extract a zip from the workspace into a folder.", F, "archive:str, folder:str=unzipped", lambda ctx, archive, folder: _unzip(ctx, archive, folder), ctx=True, risk="files")
reg("list_zip", "List what is inside a zip file.", F, "archive:str", lambda ctx, archive: zipfile.ZipFile(ctx.path(archive, True)).namelist()[:300], ctx=True, risk="files")
reg("read_csv_file", "Read a CSV file from the workspace as rows.", F, "path:str, limit:int=50", lambda ctx, path, limit: list(csv.DictReader(ctx.path(path, True).open(newline="", encoding="utf-8", errors="replace")))[:limit], ctx=True, risk="files")
reg("write_csv_file", "Write rows (list of lists, first row is the header) to a CSV file.", F, "path:str, rows:list",
    lambda ctx, path, rows: (lambda p: (lambda buf: (csv.writer(buf).writerows(rows), _w(ctx, path, buf.getvalue(), "table"))[1])(io.StringIO()))(None), ctx=True, risk="files")
reg("read_json_file", "Read a JSON file from the workspace.", F, "path:str", lambda ctx, path: json.loads(ctx.path(path, True).read_text()), ctx=True, risk="files")
reg("write_json_file", "Write a value to a JSON file.", F, "path:str, data:dict", lambda ctx, path, data: _w(ctx, path, json.dumps(data, indent=2, ensure_ascii=False), "data"), ctx=True, risk="files")
reg("read_pdf_text", "Extract text from a PDF in the workspace.", F, "path:str, max_pages:int=20", lambda ctx, path, max_pages: _pdf(ctx, path, max_pages), ctx=True, risk="files", needs=("py:pypdf",))

# ---- python
PY = "python"
reg("check_python_syntax", "Check whether Python code parses; report the error line.", PY, "code:str", lambda code: _syntax(code))
reg("python_functions", "List the functions and classes defined in Python code.", PY, "code:str", lambda code: [f"{type(n).__name__}:{n.name}({', '.join(a.arg for a in getattr(n, 'args', ast.arguments(args=[])).args)})" for n in ast.walk(ast.parse(code)) if isinstance(n, (ast.FunctionDef, ast.ClassDef))])
reg("python_imports", "List the modules Python code imports.", PY, "code:str", lambda code: sorted({(n.module or "").split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0] for n in ast.walk(ast.parse(code)) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in (n.names if isinstance(n, ast.Import) else [n])}))
reg("run_python", "Run Python code in a separate process (timeout, empty temp folder inside the workspace). Returns stdout/stderr. Use print() to show results.", PY, "code:str", lambda ctx, code: _run_py(ctx, code), ctx=True, risk="exec", tags=("colab", "jupyter", "script"))
reg("run_python_file", "Run a Python file from the workspace.", PY, "path:str, args:list?", lambda ctx, path, args: _run_file(ctx, path, args or []), ctx=True, risk="exec")
reg("python_eval", "Evaluate a single Python expression (literals, math, list comprehensions) safely via a subprocess.", PY, "expression:str", lambda ctx, expression: _run_py(ctx, f"print(repr({expression}))"), ctx=True, risk="exec")
reg("run_shell_command", "Run a command line in the workspace (no shell; arguments split like a shell).", PY, "command:str", lambda ctx, command: _run_cmd(ctx, __import__("shlex").split(command)), ctx=True, risk="exec")
reg("pip_list", "Installed Python packages (name==version).", PY, "", lambda: sorted(f"{d.metadata['Name']}=={d.version}" for d in __import__("importlib.metadata", fromlist=["x"]).distributions())[:400])
reg("module_available", "Is a Python module importable?", PY, "module:str", lambda module: __import__("importlib.util", fromlist=["x"]).find_spec(module) is not None)
reg("format_python", "Normalise indentation/whitespace of Python code (via ast round-trip).", PY, "code:str", lambda code: ast.unparse(ast.parse(code)))
reg("python_docstring_stub", "Add TODO docstrings to every function lacking one.", PY, "code:str", lambda code: _docstubs(code))


# ---- notebooks (Colab / Kaggle / Jupyter)
def _nb(cells: list[tuple[str, str]]) -> dict:
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(True)}
        if kind == "code":
            c |= {"execution_count": None, "outputs": []}
        out.append(c)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


NOTEBOOKS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "gpu_check": ("Check the GPU and library versions", [("code", "import torch, platform\nprint(platform.python_version(), torch.__version__)\nprint('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')")]),
    "data_analysis": ("Load a CSV and explore it with pandas", [("code", "import pandas as pd\ndf = pd.read_csv('data.csv')\ndf.head()"), ("code", "df.describe(include='all')"), ("code", "df.isna().sum()"), ("code", "df.hist(figsize=(10, 8));")]),
    "classifier": ("Train a scikit-learn classifier", [("code", "import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import classification_report\ndf = pd.read_csv('data.csv')\nX, y = df.drop(columns=['label']), df['label']\nXtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)\nm = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)\nprint(classification_report(yte, m.predict(Xte)))")]),
    "regression": ("Fit and plot a regression", [("code", "import numpy as np, matplotlib.pyplot as plt\nfrom sklearn.linear_model import LinearRegression\nx = np.linspace(0, 10, 50).reshape(-1, 1); y = 3 * x.ravel() + np.random.randn(50)\nm = LinearRegression().fit(x, y)\nplt.scatter(x, y); plt.plot(x, m.predict(x), 'r'); plt.show()\nprint(m.coef_, m.intercept_)")]),
    "pytorch_mnist": ("Train a small PyTorch network on MNIST", [("code", "import torch, torch.nn as nn\nfrom torchvision import datasets, transforms\ndev = 'cuda' if torch.cuda.is_available() else 'cpu'\nds = datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor())\ndl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)\nnet = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(dev)\nopt = torch.optim.Adam(net.parameters(), 1e-3)\nfor epoch in range(3):\n    for x, y in dl:\n        x, y = x.to(dev), y.to(dev)\n        loss = nn.functional.cross_entropy(net(x), y)\n        opt.zero_grad(); loss.backward(); opt.step()\n    print(epoch, loss.item())")]),
    "huggingface_pipeline": ("Run a Hugging Face pipeline", [("code", "!pip -q install transformers\nfrom transformers import pipeline\nclf = pipeline('sentiment-analysis')\nclf(['I love this classroom!', 'This is confusing.'])")]),
    "lora_finetune": ("Fine-tune a small LLM with LoRA (needs a GPU runtime)", [("code", "!pip -q install transformers peft datasets trl accelerate bitsandbytes"), ("code", "from datasets import load_dataset\nfrom peft import LoraConfig\nfrom trl import SFTTrainer, SFTConfig\nds = load_dataset('json', data_files='train.jsonl', split='train')\ncfg = LoraConfig(r=16, lora_alpha=32, target_modules='all-linear', task_type='CAUSAL_LM')\ntr = SFTTrainer(model='Qwen/Qwen2.5-0.5B', train_dataset=ds, peft_config=cfg, args=SFTConfig(output_dir='out', num_train_epochs=1, per_device_train_batch_size=2))\ntr.train()")]),
    "image_generation": ("Generate an image with diffusers (GPU)", [("code", "!pip -q install diffusers transformers accelerate\nimport torch\nfrom diffusers import AutoPipelineForText2Image\npipe = AutoPipelineForText2Image.from_pretrained('stabilityai/sdxl-turbo', torch_dtype=torch.float16).to('cuda')\npipe(prompt='a tiny robot studying in a classroom, watercolor', num_inference_steps=2, guidance_scale=0).images[0]")]),
    "whisper_transcribe": ("Transcribe audio with Whisper", [("code", "!pip -q install openai-whisper\nimport whisper\nwhisper.load_model('base').transcribe('audio.mp3')['text']")]),
    "yt_dlp_download": ("Download a video's audio with yt-dlp (only content you may download)", [("code", "!pip -q install yt-dlp\n!yt-dlp -x --audio-format mp3 -o 'audio.%(ext)s' 'https://www.youtube.com/watch?v=VIDEO_ID'")]),
    "ocr_tesseract": ("OCR text from an image", [("code", "!apt -qq install tesseract-ocr\n!pip -q install pytesseract pillow\nimport pytesseract\nfrom PIL import Image\npytesseract.image_to_string(Image.open('page.png'))")]),
    "opencv_basics": ("Image processing with OpenCV", [("code", "import cv2\nim = cv2.imread('image.jpg')\ng = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)\ne = cv2.Canny(g, 100, 200)\ncv2.imwrite('edges.png', e)")]),
    "time_series": ("Forecast a time series", [("code", "import pandas as pd\nfrom statsmodels.tsa.holtwinters import ExponentialSmoothing\ns = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\nfit = ExponentialSmoothing(s, trend='add', seasonal=None).fit()\nfit.forecast(12)")]),
    "kmeans_clusters": ("Cluster data with k-means", [("code", "import pandas as pd\nfrom sklearn.cluster import KMeans\nX = pd.read_csv('data.csv').select_dtypes('number')\nlabels = KMeans(n_clusters=4, n_init='auto', random_state=0).fit_predict(X)\nX.assign(cluster=labels).groupby('cluster').mean()")]),
    "text_classifier": ("TF-IDF text classifier", [("code", "import pandas as pd\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import make_pipeline\ndf = pd.read_csv('texts.csv')\nm = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)).fit(df['text'], df['label'])\nm.predict(['example sentence'])")]),
    "web_scraper": ("Polite page scraper with requests + BeautifulSoup", [("code", "import requests, time\nfrom bs4 import BeautifulSoup\nr = requests.get('https://example.com', headers={'User-Agent': 'ai-room/0.3 (educational)'}, timeout=20)\nprint(BeautifulSoup(r.text, 'html.parser').title.string)\ntime.sleep(1)")]),
    "wikipedia_api": ("Fetch an article summary from the Wikimedia REST API", [("code", "import requests\nr = requests.get('https://en.wikipedia.org/api/rest_v1/page/summary/Alan_Turing', headers={'User-Agent': 'ai-room/0.3 (educational)'})\nprint(r.json()['extract'])")]),
    "youtube_data_api": ("Search YouTube with the Data API v3 (needs an API key)", [("code", "import requests, os\nkey = os.environ['YOUTUBE_API_KEY']\nr = requests.get('https://www.googleapis.com/youtube/v3/search', params={'part': 'snippet', 'q': 'python tutorial', 'maxResults': 5, 'key': key, 'type': 'video'})\n[(i['id']['videoId'], i['snippet']['title']) for i in r.json()['items']]")]),
    "blender_render": ("Install Blender in the runtime and render a cube headlessly", [("code", "!apt -qq install blender\nopen('scene.py','w').write('import bpy\\nbpy.ops.mesh.primitive_cube_add()\\nbpy.context.scene.render.filepath=\"/content/cube.png\"\\nbpy.ops.render.render(write_still=True)')\n!blender -b -P scene.py")]),
    "ffmpeg_video": ("Make a video from images with ffmpeg", [("code", "!ffmpeg -y -framerate 2 -pattern_type glob -i 'frames/*.png' -c:v libx264 -pix_fmt yuv420p out.mp4")]),
    "streamlit_app": ("A minimal Streamlit app", [("code", "open('app.py','w').write('import streamlit as st\\nst.title(\"Hello\")\\nx = st.slider(\"x\", 0, 10)\\nst.write(x * x)')\n!pip -q install streamlit\n!streamlit run app.py & npx localtunnel --port 8501")]),
    "gradio_demo": ("A Gradio demo", [("code", "!pip -q install gradio\nimport gradio as gr\ngr.Interface(lambda s: s[::-1], 'text', 'text').launch(share=True)")]),
    "sql_sqlite": ("SQL with sqlite3 and pandas", [("code", "import sqlite3, pandas as pd\ncon = sqlite3.connect(':memory:')\npd.DataFrame({'a': [1, 2, 3]}).to_sql('t', con)\npd.read_sql('select sum(a) as total from t', con)")]),
    "kaggle_download": ("Download a Kaggle dataset (needs kaggle.json)", [("code", "!pip -q install kaggle\n!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json\n!kaggle datasets download -d uciml/iris -p data --unzip")]),
    "kaggle_submit": ("Write and submit a Kaggle competition file", [("code", "import pandas as pd\nsub = pd.read_csv('sample_submission.csv')\nsub.iloc[:, 1] = 0\nsub.to_csv('submission.csv', index=False)\n!kaggle competitions submit -c titanic -f submission.csv -m 'first try'")]),
}


def _make_nb(ctx: Context, key: str, path: str | None, platform: str, title: str | None):
    desc, cells = NOTEBOOKS[key]
    cells = [("markdown", f"# {title or desc}\n\nGenerated by AI Room for {platform}. {'Runtime → Change runtime type → GPU if needed.' if platform == 'Colab' else 'Turn on Accelerator / Internet in the notebook settings if needed.'}")] + cells
    nb = _nb(cells)
    if platform == "Colab":
        nb["metadata"]["colab"] = {"provenance": []}
    p = ctx.path(path or f"notebooks/{platform.lower()}_{key}.ipynb")
    p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return {"saved": ctx.produced(p, "notebook", desc), "cells": len(cells), "how_to_run": "Upload at colab.research.google.com → File → Upload notebook" if platform == "Colab" else "Kaggle → Create → New Notebook → File → Import notebook"}


for _k, (_d, _c) in NOTEBOOKS.items():
    for _plat in ("Colab", "Kaggle"):
        reg(f"{_plat.lower()}_notebook_{_k}", f"{_plat} notebook: {_d}. Writes an .ipynb you can upload.", _plat.lower(), "path:str?, title:str?",
            (lambda k, plat: lambda ctx, path, title: _make_nb(ctx, k, path, plat, title))(_k, _plat), ctx=True, risk="files", produces="notebook", tags=("notebook", "jupyter", "google colab" if _plat == "Colab" else "kaggle"))
reg("make_notebook", "Create a Jupyter notebook from cells: a list of [\"code\"|\"markdown\", source] pairs.", "colab", "path:str, cells:list", lambda ctx, path, cells: (lambda p: (p.write_text(json.dumps(_nb([tuple(c) for c in cells]), indent=1)), {"saved": ctx.produced(p, "notebook")})[1])(ctx.path(path)), ctx=True, risk="files", produces="notebook")
reg("notebook_to_script", "Convert a notebook in the workspace into a .py script.", "colab", "path:str", lambda ctx, path: "\n\n".join("".join(c["source"]) if c["cell_type"] == "code" else "\n".join("# " + l for l in "".join(c["source"]).splitlines()) for c in json.loads(ctx.path(path, True).read_text())["cells"]), ctx=True, risk="files")
reg("colab_open_link", "Link that opens a notebook from a public GitHub repo in Colab.", "colab", "user:str, repo:str, path:str, branch:str=main", lambda user, repo, path, branch: f"https://colab.research.google.com/github/{user}/{repo}/blob/{branch}/{path}")
reg("kaggle_kernel_metadata", "Write kernel-metadata.json so `kaggle kernels push` can publish a notebook.", "kaggle", "folder:str, username:str, slug:str, notebook:str, gpu:bool=false, internet:bool=true",
    lambda ctx, folder, username, slug, notebook, gpu, internet: _w(ctx, f"{folder}/kernel-metadata.json", json.dumps({"id": f"{username}/{slug}", "title": slug.replace("-", " ").title(), "code_file": notebook, "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": str(gpu).lower(), "enable_internet": str(internet).lower(), "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, indent=2)), ctx=True, risk="files")
for _sub, _d in (("datasets list -s", "Search Kaggle datasets"), ("competitions list -s", "Search Kaggle competitions"), ("kernels list -s", "Search public Kaggle notebooks")):
    reg("kaggle_" + _sub.split()[0] + "_search", _d + " (needs the kaggle CLI and credentials).", "kaggle", "query:str",
        (lambda sub: lambda ctx, query: _kaggle(ctx, ["kaggle", *sub.split(), query, "--csv"]))(_sub), ctx=True, risk="network", needs=("bin:kaggle",))
reg("kaggle_download_dataset", "Download a Kaggle dataset like 'uciml/iris' into the workspace.", "kaggle", "dataset:str, folder:str=data", lambda ctx, dataset, folder: _kaggle(ctx, ["kaggle", "datasets", "download", "-d", dataset, "-p", str(ctx.path(folder + "/.keep").parent), "--unzip"]), ctx=True, risk="network", needs=("bin:kaggle",))
reg("kaggle_download_competition", "Download a competition's files.", "kaggle", "competition:str, folder:str=data", lambda ctx, competition, folder: _kaggle(ctx, ["kaggle", "competitions", "download", "-c", competition, "-p", str(ctx.path(folder + "/.keep").parent)]), ctx=True, risk="network", needs=("bin:kaggle",))
reg("kaggle_push_notebook", "Publish a notebook folder (with kernel-metadata.json) to Kaggle.", "kaggle", "folder:str", lambda ctx, folder: _kaggle(ctx, ["kaggle", "kernels", "push", "-p", str(ctx.path(folder + "/.keep").parent)]), ctx=True, risk="post", needs=("bin:kaggle",))
reg("kaggle_submit_competition", "Submit a file to a Kaggle competition.", "kaggle", "competition:str, file:str, message:str=submission", lambda ctx, competition, file, message: _kaggle(ctx, ["kaggle", "competitions", "submit", "-c", competition, "-f", str(ctx.path(file, True)), "-m", message]), ctx=True, risk="post", needs=("bin:kaggle",))
reg("kaggle_leaderboard", "Top of a competition's leaderboard.", "kaggle", "competition:str", lambda ctx, competition: _kaggle(ctx, ["kaggle", "competitions", "leaderboard", competition, "--show"]), ctx=True, risk="network", needs=("bin:kaggle",))


# ---- helpers
def _grep(ctx, q, folder):
    out = []
    for p in ctx.path(folder).resolve().rglob("*"):
        if p.is_file() and p.stat().st_size < 2_000_000:
            try:
                for i, l in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                    if q.lower() in l.lower():
                        out.append(f"{p.relative_to(ctx.workspace.resolve())}:{i}: {l.strip()[:160]}")
            except OSError:
                pass
        if len(out) >= 100:
            break
    return out


def _zip(ctx, paths, archive):
    out = ctx.path(archive)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in paths:
            p = ctx.path(rel, True)
            for f in ([p] if p.is_file() else [x for x in p.rglob("*") if x.is_file()]):
                z.write(f, f.relative_to(ctx.workspace.resolve()))
    return {"saved": ctx.produced(out, "archive"), "bytes": out.stat().st_size}


def _unzip(ctx, archive, folder):
    dest = ctx.path(folder + "/.keep").parent.resolve()
    with zipfile.ZipFile(ctx.path(archive, True)) as z:
        for name in z.namelist():
            if (dest / name).resolve() != dest and dest not in (dest / name).resolve().parents:
                raise ToolError("zip contains a path that escapes the folder")
        z.extractall(dest)
        return {"extracted": len(z.namelist()), "folder": str(dest.relative_to(ctx.workspace.resolve()))}


def _pdf(ctx, path, n):
    from pypdf import PdfReader
    r = PdfReader(str(ctx.path(path, True)))
    return "\n".join((pg.extract_text() or "") for pg in r.pages[:n])[:MAX_READ]


def _syntax(code):
    try:
        ast.parse(code)
        return {"valid": True}
    except SyntaxError as e:
        return {"valid": False, "line": e.lineno, "error": e.msg}


def _docstubs(code):
    t = ast.parse(code)
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not ast.get_docstring(n):
            n.body.insert(0, ast.Expr(ast.Constant(f"TODO: describe {n.name}.")))
    return ast.unparse(t)


def _env():
    keep = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "TMPDIR", "SYSTEMROOT")}
    return keep | {"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}


def _run_cmd(ctx, argv, cwd=None):
    try:
        r = subprocess.run(argv, cwd=cwd or ctx.workspace, capture_output=True, text=True, timeout=max(5, ctx.timeout - 2), env=_env(), stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise ToolError(f"timed out after {ctx.timeout:.0f}s")
    except FileNotFoundError:
        raise ToolError(f"'{argv[0]}' is not installed")
    return {"exit_code": r.returncode, "stdout": r.stdout[-4000:], "stderr": r.stderr[-2000:]}


def _run_py(ctx, code):
    with tempfile.TemporaryDirectory(dir=ctx.workspace) as d:
        (Path(d) / "main.py").write_text(code, encoding="utf-8")
        before = {p for p in Path(d).iterdir()}
        r = _run_cmd(ctx, [sys.executable, "-I", "main.py"], cwd=d)
        for p in Path(d).iterdir():
            if p not in before and p.is_file():
                dest = ctx.path(f"outputs/{p.name}")
                shutil.copy(p, dest)
                ctx.produced(dest, "file", "created by run_python")
        return r


def _run_file(ctx, path, args):
    return _run_cmd(ctx, [sys.executable, "-I", str(ctx.path(path, True)), *map(str, args)])


def _kaggle(ctx, argv):
    r = _run_cmd(ctx, argv)
    if r["exit_code"]:
        raise ToolError((r["stderr"] or r["stdout"]).strip()[:500] or "kaggle failed")
    return r["stdout"][-4000:]
