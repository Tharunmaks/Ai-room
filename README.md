# 🎓 AI Room — a classroom where AI learns without training

A student model of **up to 27B parameters** goes through **12 classes** in up to **100 subjects**. Its weights never change — no GPU training, no fine-tuning. It learns the way a person does: it listens to lessons, **writes notes**, asks questions, and **re-reads the right pages of its notebook** before every exam question. It moves up a class only after scoring **at least 90% in every chosen subject**.

**The teacher is not an AI.** There is no teacher model to hallucinate lessons or answer keys. The teacher is made of three things that can be checked:

| The teacher's source | What it provides | Why it can be trusted |
|---|---|---|
| **Exact problem generators** (192 families, plus 53 programming tasks with reference solutions) | Maths, physics, chemistry, finance, logic puzzles, code, CS, languages … | The answer is *computed by code* (or, for puzzles, by exhaustive search). Programming answers are graded by running tests. |
| **A textbook library** | Wikipedia, Simple English Wikipedia, Wikibooks, **and your own books** (txt, md, html, pdf, epub, docx) | Lessons *quote* passages with their source; exam questions are *sentences from the book* with a word blanked out. |
| **Question banks** | Anything you import as JSON / JSON-lines (own format, MMLU, ARC, SciQ, GSM8K, MBPP) | The key comes with the question. |

How the teacher *behaves* is a set of dials — tone, approach, depth, worked examples, practice, encouragement, humour — chosen from six presets (Ms. Patience, Professor Socrates, Dr. Rigor, Mr. Giggles, Dr. Curie, Coach Drill) or set by you. They are rendered by templates, so the same lesson is always phrased the same way.

## How a class works

```
          ┌────────── teacher (no AI) ──────────┐              ┌──────── student (the only AI, ≤ 27B) ────────┐
lesson  → │ cards + worked examples + textbook   │ ───────────→ │ writes notes → notebook                       │
          │ passages + key terms + practice      │              │ asks questions ←→ teacher answers from books  │
          └──────────────────────────────────────┘              └───────────────────────────────────────────────┘
exam    → new questions each time, graded by code ──────────────→ student retrieves its notebook pages (BM25) per question
          ≥ 90% in every subject?  yes → next class
                                    no  → teacher reviews each mistake → student writes a mistake log
                                          → focused re-lesson → fresh exam (up to N attempts)
```

Every exam is newly generated and avoids earlier questions, so the student can't pass by memorising answers; it has to carry methods and facts forward in its notes. After each passed class it condenses its notes into a cheat-sheet.

## The 100 subjects

| | Subjects | What the exams use |
|---|---:|---|
| **Exact practice in all 12 classes** | 79 | generated problems, optionally mixed with textbook questions |
| **Mixed** | 9 | exact problems for some classes, textbooks for the rest |
| **Textbook-only** | 12 | history, philosophy, ethics, literature, art, film, religion, classics, international relations, anatomy, creative writing, English language — fetched or imported |

Categories: Core · Mathematics & Statistics · Physical Sciences · Life & Health Sciences · Engineering · Computing & AI · Social Sciences & Business · Humanities & Arts · World Languages (Spanish, French, German, Mandarin, Hindi — vocabulary and regular grammar from built-in tables). Open any subject in the UI to see its 12-class syllabus.

## Quick start

```bash
pip install -e .                      # add ".[pdf]" to import PDF books
ollama pull qwen2.5:14b               # the student — any model ≤ 27B, via Ollama or an OpenAI-compatible server
ai-room serve                         # open http://127.0.0.1:8765
```

In the web app: **New classroom** → pick the student model, the teacher, the subjects and the rules → **Start class**. Watch lessons appear on the chalkboard, the student's questions and notes, exams marked live, and the report card fill in.

*No model installed?* Choose **Simulated (demo)** as the student to explore the interface. It is shown the answer key and does **not** learn — it exists only so you can look around.

### From the terminal

```bash
ai-room run --student ollama:qwen2.5:14b --persona socratic --subjects math,physics,history,spanish --until all
ai-room library fetch astronomy history        # download textbooks (needs internet; polite to Wikimedia)
ai-room library import history ./my-books      # your own files or folders
ai-room library search history "who was Augustus"
ai-room bank import questions.jsonl --subject biology
```

Textbook-only subjects fetch their books automatically the first time they are needed (switch off with "auto-fetch" in the rules). Fetching reads about 30 articles per subject and is deliberately slow and identified with a descriptive `User-Agent`, as Wikimedia asks. If Wikimedia refuses the connection (it does for busy shared IP addresses) the app says so and you can import books instead.

### On a phone (Termux)

`ai-room serve` runs fine, but a 14B–27B student model does not fit on a phone. Point the student at an OpenAI-compatible server on your computer (`OPENAI_BASE_URL`, or the "Server address" box in the wizard), or use the simulated student to explore. Clone the repo first, then `cd` into it before `pip install -e .`.

## 🛠 The Workshop — 1,200+ tools the student can use

Open **Workshop** in the app, type a goal ("build a 3D snowman in Blender and make an Instagram post about it"), and the student searches the toolbox, calls tools one at a time, reads what came back, and keeps going until the files exist. Everything it makes lands in a workspace folder with a gallery (images, videos, audio, 3D, notebooks).

| Area | Tools | Examples |
|---|---:|---|
| Calculator, maths, statistics, units, finance, physics, chemistry | 220 | `calculator`, `convert_units`, `solve_quadratic`, `loan_payment`, `molar_mass`, `matrix_inverse` |
| Text, data, dates, colours, encoding, hashing | 151 | `csv_summary`, `json_get`, `hash_sha256`, `contrast_ratio`, `age_from_birthdate` |
| Python sandbox, files, archives | 32 | `run_python`, `write_file`, `zip_files`, `read_pdf_text` |
| Charts (SVG) | 15 | `bar_chart`, `plot_function`, `heatmap`, `timeline` |
| Images (Pillow) | 148 | 17 filters, 24 effects, 20 social-media sizes, text cards, memes, certificates, 18 generative-art makers |
| Video & audio (ffmpeg) | 134 | convert, trim, GIFs, 27 video effects, reel/story/Shorts fitting, slide videos, 22 audio effects, melodies |
| 3D meshes (pure Python) | 38 | 22 primitives, OBJ/STL/PLY/glTF export, volume & watertight checks, software renderer, turntable GIFs |
| **Blender** | 214 | scenes as files; 31 shapes, 33 materials, 37 modifiers, lights, 10 cameras, animation, 32 ready-made builds (house, robot, car, solar system…), `bpy` script generation, real rendering |
| Colab & Kaggle | 62 | 25 starter notebooks for each, datasets, competitions, kernels |
| Wikimedia, web, 45 free public APIs | 91 | Wikipedia/Wiktionary/Commons/Wikidata, weather, countries, books, papers, exchange rates |
| YouTube, Instagram, GitHub, Hugging Face | 52 | official APIs only |
| Machine learning & algorithms | 52 | real training (regression, trees, k-NN, small neural net, naive Bayes, k-means), genetic algorithms, Sudoku, graphs |
| Automation (recipes) | 7 | saved pipelines; 13 built in, e.g. `instagram_post_pack`, `blender_showcase`, `youtube_launch_kit` |

**Honest counts.** 1,216 tools are registered; in a typical install 1,181 are ready. The rest say exactly what they need — `blender` / `kaggle` / `yt-dlp` on your PATH, or a key such as `YOUTUBE_API_KEY`. The Workshop shows *needs setup* instead of failing silently. Many tools in a family share code (for example 33 materials, 25 notebook templates × 2 sites, 17 filters): the number counts distinct things the student can ask for, not 1,200 different algorithms.

### How the student uses them

Small models are unreliable at native function-calling, so the protocol is plain JSON in text — one action per turn: `{"tool": …, "args": {…}}`, `{"search": …}` or `{"final": …}`. Each turn the model sees its goal, its last steps, and a **shortlist of tools retrieved with BM25** from all 1,200 (it never reads the whole list). Wrong tool names get "did you mean…", errors are shown to the model so it can repair the call, identical repeated calls are interrupted, and a run stops at its step limit. The model must be ≤ 27B like everywhere else. The *Demo planner* is a keyword rule-set — **not an AI** — so you can explore the Workshop without a model.

### Automation: recipes

A recipe is a saved list of tool calls with `{{parameters}}` and `{{step1}}`-style chaining — it runs with no AI at all. Run one from the **Recipes** tab, run it for a whole list (`recipe_run_batch`), or finish a student run and **save its successful steps as a recipe** so the same job can be repeated for free.

### Blender: what is real and what is not

The Blender tools edit a **scene file** (`scenes/<name>.json`) in pure Python, so they always work. `blender_build_script` writes a `bpy` script; `blender_render` / `blender_run_script` run **real Blender** when it is installed (`blender -b -P …`) and show as *needs setup* until then. Without Blender you still get a software-rendered **preview** (shapes, colours, arrays, mirrors — not lights, glass or subdivision). The generated scripts are compile-checked and executed against a fake `bpy` in the tests; they have **not** been run in a real Blender here, so expect to adjust a few settings across Blender versions (the script handles the EEVEE and Principled-BSDF renames of 4.x).

### "Training" — what it means here

*Local* training tools really change model weights, but only on small data in pure Python (`train_logistic`, `train_tree`, `train_mlp`, `train_text_classifier`, k-means…). Heavy GPU training is **not** run by this app: the Colab and Kaggle tools write ready-to-upload notebooks (LoRA fine-tuning, PyTorch, diffusers, Whisper…). This is separate from the classroom, where the student's own weights never change.

### Safety defaults

* **Permission levels:** `safe` (pure computation), `files` (workspace only), `network` (public pages and APIs) are on. **`exec`** (the student's own Python/shell/Blender code) and **`post`** (publishing to YouTube, Instagram or GitHub) are **off** until you switch them on in *Permissions & keys*. A blocked call tells the student why.
* **Official APIs only.** YouTube uses the Data API v3 with your key; Instagram uses the Graph API for a Business/Creator account. No passwords, no scraping of logged-in sites. Colab has no public API, so the tools write notebooks. `yt-dlp` is optional and for content you are allowed to download.
* Keys are stored in a file readable only by you and are never sent back to the browser.
* The workspace is a folder: tools cannot read or write outside it, zip extraction refuses `../`, page fetching refuses local/private addresses, and files are served with a sandboxing `Content-Security-Policy` so a generated SVG can't run scripts.
* `run_python` runs in a separate process with a timeout, **not** a real sandbox — use a container or VM if you enable `exec` with an untrusted model.
* There is still **no login**: keep the server on `127.0.0.1` unless you add your own protection.

## The 27B limit

Larger students are refused. The size comes from Ollama (`/api/show`) or from the model name (`qwen2.5:14b`, `gemma3:27b`); Mixture-of-Experts models count their *total* parameters (`8x7b` = 56B), and a model whose size can't be determined is refused unless you declare it. Models reported as slightly over 27B (e.g. Gemma 3 27B reports 27.4B) are accepted.

## Bringing your own knowledge

* **Books** — drop files into the Library tab, paste a page URL, or run `ai-room library import`. Passages are split at headings and sentence boundaries, indexed with BM25, and cited by title and URL in lessons.
* **Question banks** — JSON array or JSON-lines. Own format:
  ```json
  {"subject":"biology","type":"mcq","question":"Which organelle makes ATP?","options":["Ribosome","Mitochondrion","Golgi","Nucleus"],"answer":"B","grade":6}
  ```
  `type` is `mcq`, `numeric`, `cloze` or `code` (with `tests` and `function_name`). `grade` is optional. MMLU, ARC, SciQ, GSM8K and MBPP rows are recognised as they are.
* **Licences** — Wikipedia and Wikibooks text is CC BY-SA 4.0; the app keeps the source URL of every article and shows it. Check the licence of anything you import.

## What to expect (honest limits)

* **Learning here is in-context.** The student can only do what its notebook, the lesson in front of it and its own abilities allow. Passing class 12 means "this model, with this notebook, can answer these exams" — not that it gained general intelligence. A stronger student model will climb faster; a weak one may stall (the class is then held and says so).
* **Textbook questions test recall.** They are built by rules from definition and dated-event sentences, so they check whether the student retained *facts and terms*, not deep understanding. Quality depends on the text: well-written reference prose works best. Exact-practice subjects test *method*.
* **Templated problems.** Generated problems vary numbers and wording, but come from templates (192 families); a student can master a family by learning its method — which is the point.
* **Free-text answers are not graded** (that would need a model). Questions are numeric, multiple-choice, fill-in-the-blank or code.
* **Code answers run on your machine** in a separate Python process with a timeout, in an empty folder. That is not a real sandbox: run AI Room in a container or VM if you use untrusted models.
* **Server exposure.** The app binds to `127.0.0.1` by default; it has no login (see the Workshop safety notes — it can write files and, if you allow it, run code). URL import refuses private addresses unless `AI_ROOM_ALLOW_PRIVATE_URLS=1`.

## Project layout

```
ai_room/
  teacher/
    generators.py      problem families + registry          gen_math*.py  gen_science.py  gen_social.py
    gen_cs.py  gen_logic.py  gen_life.py  gen_lang.py        the 192 families (answers computed or brute-forced)
    code_tasks.py      53 programming tasks with reference solutions and generated tests
    library.py         textbook library (SQLite) · chunking · BM25 · importers · question banks
    fetch.py           polite Wikimedia fetcher            knowledge.py   sentences → cloze / multiple-choice questions
    style.py           the teacher's dials and voice       teacher.py     lessons, exams, answers, mistake reviews
  agents.py            the student (notes, questions, recall, reflection)
  classroom.py         lesson → Q&A → exam → review → promotion loop, deterministic grading
  llm.py               student backends (Ollama, OpenAI-compatible, demo) + the 27B guard
  memory.py            notebook, exam records, transcript
  grading.py           answer extraction and matching (numeric, MCQ, cloze, exact, code tests)
  server.py  cli.py    FastAPI app and command line          web/   the single-page UI (no build step)
  workshop_api.py      Workshop HTTP API (tools, runs, files, recipes, permissions)
  tools/
    registry.py        Tool / Param / Context, BM25 tool search, permission + path-escape guards, call() that never raises
    t_math.py t_text.py t_files.py t_svg.py t_image.py t_video.py t_mesh.py t_blender.py t_web.py t_ml.py   the tool families
    recipes.py         automation pipelines          agent.py   the Workshop loop and the rule-based demo planner
tests/                 ~190 offline tests
```

Run the tests with `pip install -e ".[dev]" && pytest`.
