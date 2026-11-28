# 🎓 AI Room — teach, train and build AI on your own computer

AI Room has three parts, all running locally on a CPU:

* **🧪 The Lab**: upload a table (Excel or CSV), a text, or folders of pictures, sounds or videos, and **brand-new, untrained models learn it in seconds**, side by side on every CPU core. They predict columns, write in the style of the text, answer questions from it, and tell pictures, sounds and videos apart. [More below](#-the-lab--train-your-own-ai-from-scratch-in-seconds).
* **🎓 The Classroom**: a language model learns 12 classes in 100 subjects **without training**, by taking notes (described next).
* **🛠 The Workshop**: the student uses 1,200+ tools to make things.

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
pip install -e .                      # add ".[pdf]" to import PDF books, ".[lightgbm]" for stronger trees in the Lab
ollama pull qwen2.5:14b               # the student — any model ≤ 27B, via Ollama or an OpenAI-compatible server
ai-room serve                         # open http://127.0.0.1:8765
```

Open **Lab** to train a model on your own data right away (no student model needed). For the classroom: **New classroom** → pick the student model, the teacher, the subjects and the rules → **Start class**. Watch lessons appear on the chalkboard, the student's questions and notes, exams marked live, and the report card fill in.

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

## 🧪 The Lab — train your own AI from scratch, in seconds

Open **Lab** in the web app, drop in a file (up to **5 TB**) or paste one, press **Train from scratch**. Brand-new models with random weights learn your data on this computer's CPU, with no GPU, no download and no internet, and you watch them learn live. Nothing is pre-trained and nothing leaves the machine. By default the whole job, from reading the file to the saved model, gets **5 minutes per GB of data** (0.3 s per MB, at least half a second). The learners train **side by side, one per CPU core**.

| You upload | It learns to | Then you can |
|---|---|---|
| **A table**: Excel (.xlsx), CSV, TSV, JSON, JSON-lines — or cells pasted from a spreadsheet | predict any column from the others: a category (*spam / not spam*) or a number (*price*) | try single rows in a form, predict a whole Excel or CSV file (you get it back with a new column), or call it over HTTP |
| **Text**: txt, md, html, pdf, docx, epub, code, or a zip of them — or pasted text | write in the style of the text, character by character | let it write from a prompt, and **ask your text**: it finds the passages that answer a question |
| **Pictures**: a .zip with one folder per category (`cats/`, `dogs/` …) | tell the categories apart | drop in a new picture and see what it is and how sure the model is, or sort a whole zip of pictures into a spreadsheet |
| **Sounds**: a .zip with one folder per category (`barks/`, `meows/` …) of WAV, MP3, OGG, FLAC, M4A… | tell the categories apart by ear | drop in a sound **or record one from the microphone**, or sort a whole zip of sounds |
| **Videos**: a .zip with one folder per category (`waving/`, `clapping/` …) of MP4, MOV, WebM, MKV, AVI or animated GIF | tell the categories apart by look, movement and sound | drop in a clip and see what it is, or sort a whole zip of videos |

No data at hand? Eight built-in samples are generated by code: house prices, two spirals, product reviews, customer churn, shapes (pictures), sounds (whistles, drums, chords and chirps), motions (videos of a ball bouncing or sliding, a bar spinning, a circle growing), and AI Room's own source code.

```bash
ai-room serve                                           # http://127.0.0.1:8765 → Lab
ai-room lab train houses.xlsx --target price            # or from the terminal: 5 min per GB (--speed balanced | thorough for longer)
ai-room lab predict <model-id> new_houses.xlsx --out predicted.csv
ai-room lab train pets.zip && ai-room lab predict <model-id> photo.jpg     # pictures: one folder per category
ai-room lab train birdsong.zip && ai-room lab predict <model-id> song.mp3  # sounds, the same way
ai-room lab train moves.zip && ai-room lab predict <model-id> clip.mp4     # videos, the same way
ai-room lab train my_notes.txt && ai-room lab write <model-id> "Dear diary"
ai-room lab ask <model-id> "when is the meeting"
ai-room lab models
```

### Made to be easy

* **Four steps, shown at the top:** add data → choose what to learn → learn → use it. The Lab guesses the column to predict and whether the answer is a category or a number; you can change both.
* **Paste instead of upload:** copy cells from Excel or Google Sheets (with the header row), comma-separated lines, or any text. The Lab tells you whether it sees a table or a text before you press *Use it*.
* **You know how long it will take before you start:** each Train button shows the time allowed for that file, and a bar shows the time used while it learns.
* **See and hear what you uploaded:** picture datasets show example pictures from each folder, sound datasets show spectrograms (pitch from low to high, time from left to right), and video datasets show a frame of a few clips per folder.
* **Try a sound by recording it:** a trained sound model has a *Record 3 seconds* button. The browser turns recordings and MP3/OGG/M4A files into WAV before sending them, so trying sounds needs nothing extra on the server.
* **Results in plain words, with a verdict:** for example "It gets 97.9% of new videos right — 47 of 48 it had never seen. Always guessing the most common category would get 25.0%." A badge (*Excellent / Good / Fair / Weak*) compares the model with guessing, and weak results come with tips on what to try next.
* **Keep, share and redo:** rename a model, download it as a .zip (JSON + NumPy arrays, no code inside), import a model zip someone sent you, or *Train again* with different settings.
* **Helpful refusals:** an old .xls, a single picture, sound or video instead of a zip of folders, an MP4 when ffmpeg is not installed, a column with one value, a file bigger than the free disk space — each gets a message that says what to do instead.

### How it learns so fast

Untrained learners race within the time allowed (**fast: 5 minutes per GB**, the default; **balanced**: at least 30 s; **thorough**: at least 2 min). They run **at the same time, each in its own process on its own core**, and each gets all of the time that is left, so a 4-core computer learns about four times as much in the same time as running them one after another. Each has a lane and a live learning curve in the UI:

| Learner | How it learns | Why it is fast |
|---|---|---|
| **One-shot net** | a wide layer of random features (random Fourier features), with the output layer **solved exactly** by ridge regression | no epochs at all: one linear solve, sized to the time it has. It is usually ready in **0.05–0.3 s** (about 2 s on 400,000 rows) and is often already good |
| **Linear model** | logistic / linear regression with AdamW; reads **text columns** through a sparse, fastText-style layer over up to 262,144 hashed word and word-pair features | only the weights of words in the current batch are touched |
| **Word counts** | for text columns with categories: complement naive Bayes over the same hashed words, with its smoothing chosen on the progress-check rows | counting, not training: one pass over the words |
| **LightGBM** (optional: `pip install -e ".[lightgbm]"`) | Microsoft's histogram gradient boosting in compiled C++; when it is installed it takes the place of the NumPy boosted trees below | C++ on several cores (half of them inside the race, or all of them on its turn when learners take turns); it stops with just enough time to score the test rows, and its validation answers are kept from training rather than worked out again |
| **Boosted trees** | histogram gradient boosting, written for this app in NumPy | bins features into 31–127 values (very common values such as zero get their own bin), **bundles one-hot columns** into one feature, reuses the parent histogram (right child = parent − left), looks at no more than 256 columns per tree on wide data such as pictures, scores the test rows tree by tree so nothing is left to do when the time is up, and stops when the validation loss stalls |
| **Neural network** | a 256-256 ReLU network trained with AdamW, dropout, warm-up and cosine learning rate | mini-batches; it plans its epochs to fit the time it was given, and stops mid-epoch when the time is up |

The rows are split **before** anything is learned: **70%** to learn from, **15%** to check progress (early stopping, choosing the winner), and **15% locked away** and used once, at the end, for the score you see. The winner is the learner, or the average of the best few, whose predictions fit the progress-check rows best (from 1,000 progress-check rows on, the one that is right most often). Every result also shows **what “no learning” scores** (always guessing the most common class, or the average), which columns the model relies on (permutation importance), and for categories a confusion table.

Text is learned by **counting**: a character model with interpolated **modified Kneser–Ney** smoothing looks back 1–10 characters, and picks the length that best predicts held-back text it has not seen. Each context length costs one sort: a (context, next character, previous character) triple is packed into one 64-bit number, so a single NumPy sort yields the counts and the Kneser–Ney continuation counts together. Several context lengths are counted at once on different cores. Questions are answered by **search** (BM25 over passages), so answers are quoted from your text, not made up.

### 5 minutes per GB, and files up to 5 TB

* **Uploads are streamed to disk** in chunks, so their size is limited by the cap (**5 TB per file**, `AI_ROOM_LAB_MAX_MB` to change it) and the free disk space, which is checked before and during the upload. They are not limited by memory. For huge files use `curl -X POST -T big.csv 'http://127.0.0.1:8765/api/lab/datasets?filename=big.csv'`. `-T` streams the file, while `--data-binary @file` would load it all into memory first. Browsers are not built for terabyte uploads.
* **The budget covers the whole job:** reading the file, turning it into numbers, all the learners, the final test, importance and saving. The learners share whatever time is left after reading, and a small reserve is kept for the end. The one-shot net sizes itself to the time, the neural network stops mid-epoch, the boosted trees stop between trees, and a learner that still runs at the end is stopped. In the rare case that reading the data uses up nearly all of the time, the one-shot net still learns, so you always get a real model; the result says so.
* **Reading is fast because it is done column by column.** All-numeric files go through NumPy's C parser (about 58 MB/s here); others through Python's csv module (about 7–18 MB/s). Each *distinct* value of a column is converted once, then mapped to every row in C. Text columns are hashed a whole column at a time, on several cores: each distinct word is hashed once and word pairs are mixed with integer arithmetic. Excel workbooks are read as a stream with the standard library alone (no extra packages). Pictures, sounds and videos are read on all cores at once.
* **Files bigger than memory are sampled honestly.** If the cells won't fit in memory (about a quarter of free RAM; `AI_ROOM_LAB_MAX_CELLS` to change it), the Lab reads rows from random places in the file, which takes minutes even for a terabyte. If cells contain line breaks, where a random seek could land mid-cell, it reads the file once and keeps a uniform sample instead. Texts larger than 32 million characters (`AI_ROOM_LAB_TEXT_CHARS`) are learned from evenly spaced 1 MB pieces. The result always says when a sample was used and how big the file was.
* **What 5 TB means in practice:** at 5 minutes per GB a 5 TB file is *allowed* 25,000 minutes (about 17 days), but the Lab doesn't use that. It learns from a sample that fits in memory. Copying 5 TB to the disk, at disk or network speed, is the slow part, and you need 5 TB of free space.

### Pictures, sounds and videos

Put the files in a .zip with one folder per category; the folder names become the answers (a folder that holds all the others, such as `photos/cats/1.jpg`, is skipped). Every file becomes a fixed list of numbers, and the same untrained learners as for tables learn from those. *What it pays attention to* is shown per group of numbers, and if time runs short while reading, the Lab uses a fair random part of the files and says how many.

| Kind | Numbers per file | What they describe |
|---|---:|---|
| **Picture** | 936 | **shape** (a 24×24 greyscale thumbnail), **colour layout** (an 8×8 colour thumbnail), **edges** (which way edges point in each of 16 regions), **colour mix** (how much of each shade) |
| **Sound** (first 30 s, as mono 16 kHz) | 212 | **tone**: loudness in 40 pitch bands (a mel spectrogram: its average, spread, change and peak); **timbre**: 13 MFCCs, average and spread; **loudness & noisiness**: brightness, bandwidth, roll-off, flatness, zero crossings, loudness levels, length; **changes over time**: how the loudness rises and falls |
| **Video** (8 evenly spaced frames of the first 2 minutes, and its sound track) | 1,172 | **look**: the picture numbers, averaged over the frames; **motion**: where in a 4×4 grid things change, how much, and which way the change moves from frame to frame; **sound**: the sound numbers of its sound track (zeros when there is none) |

WAV sounds and animated GIFs are read with Python and Pillow alone. Other formats (MP3, OGG, FLAC, M4A, AAC, Opus; MP4, MOV, WebM, MKV, AVI, MPEG) are read through **ffmpeg** if it is installed (`apt install ffmpeg`, `brew install ffmpeg`, or ffmpeg.org). One ffmpeg run per video takes small frames and the sound track together, skipping decoding work that 64×64 frames don't need: a 10-second 640×360 MP4 is read in about 0.2 s.

This works well when the look, the sound or the movement tells the categories apart: products, plants, symbols and scans; instruments, machines, alarms, animal calls and single spoken words; gestures, exercises and scenes. It is **not a deep neural network**: it will be far behind one on hard photographs (telling dog breeds apart, say), it does not understand speech, and it does not follow what happens in a long video. Videos are slow to decode, so on the fastest setting a folder of small, highly compressed clips may not all be read in time; the Lab says how many it read, and *Balanced* reads more.

### Measured

These were measured on one 4-core Xeon (15 GB RAM, no GPU) at the default **5 minutes per GB** (0.3 s per MB, at least 0.5 s), **without LightGBM** (what you get from a plain `pip install -e .`; the next table shows what LightGBM adds). Time is the whole job, timed from the stored file to the saved model, including reading the file.

| Data | File | Time allowed | Took | Score on unseen rows | No learning |
|---|---:|---:|---:|---|---|
| Iris | 0.003 MB | 0.5 s | 0.23 s | 90.0% (27 of 30) | 33% |
| Two spirals (sample) | 0.02 MB | 0.5 s | 0.38 s | **100%** | 50% |
| House prices (sample, R²) | 0.08 MB | 0.5 s | 0.46 s | **0.961** | 0 |
| House prices as Excel (.xlsx, R²) | 0.10 MB | 0.5 s | 0.46 s | **0.999** | 0 |
| Customer churn (sample) | 0.14 MB | 0.5 s | 0.46 s | 76.2% | 66% |
| Product reviews (sample, text column) | 0.13 MB | 0.5 s | 0.26 s | **96.3%** | 50% |
| Breast cancer | 0.12 MB | 0.5 s | 0.46 s | **98.8%** | 63% |
| Handwritten digits | 0.27 MB | 0.5 s | 0.49 s | **96.7%** | 10% |
| California housing (R²) | 1.2 MB | 0.5 s | 0.47 s | 0.775 | 0 |
| Shapes: 600 pictures, 3 categories (sample) | 3.4 MB | 1.0 s | 0.96 s | 81.1% (**84.4%** with *balanced*) | 33% |
| Sounds: 400 WAV clips, 4 categories (sample) | 3.7 MB | 1.1 s | 1.01 s | **100%** (60 of 60) | 25% |
| Motions: 320 animated GIFs, 4 categories (sample) | 4.8 MB | 1.4 s | 1.31 s | **97.9%** (47 of 48) | 25% |
| Adult income | 5.3 MB | 1.6 s | 1.56 s | 85.5% | 76% |
| 20 Newsgroups (text column, 20 topics) | 22.5 MB | 6.7 s | 5.20 s | 74.1% | 5% |
| Forest cover type, all 581,012 rows | 75.8 MB | 22.7 s | 20.6 s | **85.7%** | 49% |
| Synthetic, 16 million rows (bigger than memory → 4.9 M sampled) | 1.12 GB | 337 s | **311 s** | **95.0%** | 56% |
| AI Room's own code (text model) | 1.1 MB | 0.5 s | 0.33 s | 2.21 bits per character | 8.75 (random) |
| 20 Newsgroups as plain text (text model) | 22.1 MB | 6.6 s | 6.0 s | 2.15 bits per character (learned 7.4 M of 22 M characters) | 7.03 (random) |

Half the time of the previous default (10 minutes per GB, learners one after another) gave about the same scores: the forest cover type reached 85.7% in 21 s instead of 40 s, the 1.12 GB file 95.0% in 311 s instead of 95.1% in 586 s, and handwritten digits went up from 94.1% to 96.7%. California housing went down (0.800 → 0.775), and the text model learns a third of a 22 MB text instead of all of it, for nearly the same bits per character.

**With LightGBM installed** (`pip install -e ".[lightgbm]"`), same computer and budgets. Scores vary from run to run, so a range is given where several runs were made:

| Data | Without LightGBM | With LightGBM | Took | Winner |
|---|---|---|---:|---|
| Breast cancer | 98.8% | 98.8% | 0.48 s | LightGBM |
| Handwritten digits | 96.7% | 96.7–97.8% | 0.48 s | one-shot net (trees are not the best tool for pixels) |
| California housing (R²) | 0.775 | **0.81–0.85** (8 runs) | 0.49 s | LightGBM |
| Adult income | 85.5% | **87.1%** | 1.52 s | LightGBM |
| Forest cover type, all 581,012 rows | 85.7% | **88.4–89.3%** (4 runs) | 21.7 s | LightGBM |
| Synthetic, 16 million rows (1.12 GB) | 95.0% | 95.1% | 312 s | neural network + LightGBM (LightGBM alone reached 94.9% in 108 s; the NumPy trees needed 234 s for 94.7%) |

On the small samples (iris, spirals, houses, churn) the one-shot net or the linear model still wins and nothing changes, and tables whose only input is text (reviews, newsgroups) don't use the trees at all.

The 1.12 GB file was uploaded with `curl -T`. From the first 4 MB the Lab estimated 15,999,487 rows (there are 16,000,000), read a uniform random sample of 4.9 million rows in 36 s, and learned and tested on it within the time; all four learners finished.

**Compared with scikit-learn 1.9 defaults** on the same rows, split and features. Here the Lab got 5 minutes per GB (at least 0.5 s), and scikit-learn trained on 85% of the rows to the Lab's 70%:

| Data | Lab at 5 min per GB | Lab with LightGBM | Best scikit-learn default |
|---|---|---|---|
| Iris | 90.0% in 0.3 s | 90.0% | 90.0% (boosted trees / logistic) |
| Breast cancer | **98.8%** in 0.5 s | **98.8%** | 96.5% |
| Handwritten digits | **97.8%** in 0.5 s | **96.7–97.8%** | 95.5% (boosted trees, 1.6 s) |
| California housing (R²) | 0.773 in 0.5 s | 0.81–0.85 | **0.858** (boosted trees, 0.3 s) |
| Adult income | 86.1% in 1.6 s | **87.1%** | **87.2%** (boosted trees, 1.3 s) |
| Forest cover type, all 581,012 rows | 85.7% in 21 s (from the file) | **88.4–89.3%** | 78.0% (boosted trees, 4.1 s: its early stopping ends early) |

What this shows, and what it doesn't:

* **Every job finished inside 5 minutes per GB**, from a few kilobytes to 1.1 GB, and the first useful model arrives in well under a second for anything but very large files. When other programs are busy on the same computer, a job can take a little longer than allowed.
* **Short budgets cost some accuracy on small files.** With 30 seconds instead of half a second, digits reached 97.0% and California housing 0.865 instead of 0.775, and the shapes pictures 84.4% instead of 81.1%. Choose **balanced** or **thorough** when accuracy matters more than seconds. Scores also vary a little from run to run, because how far the learners get depends on the time they have.
* On medium tables, scikit-learn's compiled boosted trees are faster than the pure-NumPy Lab, and more accurate on two of the six tables at these budgets. With LightGBM installed, the Lab ties it on adult income and comes close on California housing; scikit-learn there trained on 85% of the rows with no time limit.
* Every job stayed inside its time on a quiet computer, usually with 10–20 ms to spare on the half-second jobs. A busy computer can push one past by a few milliseconds.
* The sound and video samples are made by code and are easy on purpose; real recordings are harder. Their test sets are small (60 and 48 files), so one file is about 2%.
* Test sets of small tables are small: iris has 30 test rows, so one row is 3.3%. Differences of a row or two are luck.
* The text writer is a statistical character model. It copies **style, spelling and layout** convincingly, but it does not understand or reason and knows nothing beyond your text. To get answers, use **Ask your text**.

### Use a trained model from your own code

```bash
curl -s http://127.0.0.1:8765/api/lab/models/<id>/predict -H 'Content-Type: application/json' \
     -d '{"rows": [{"size_m2": 120, "area": "Riverside", "garden": "yes"}]}'
# → {"predictions": [{"prediction": 351199.3}], "seconds": 0.0003}
curl -s "http://127.0.0.1:8765/api/lab/models/<id>/predict_media?filename=bark.mp3" --data-binary @bark.mp3
# → {"prediction": {"prediction": "barks", "confidence": 0.93, "top": […]}, "seconds": 0.08, "media": "audio"}
```

| Endpoint | What it does |
|---|---|
| `POST /api/lab/datasets?filename=…` (file as the body, streamed to disk, up to 5 TB) · `POST /api/lab/samples/{name}` | add a dataset; the reply describes its columns, rows (or categories and files) and time allowed |
| `POST /api/lab/train` `{"dataset", "target", "task", "ignore", "speed"}` | queue a training job (`speed`: `fast` = 5 min per GB, `balanced`, `thorough`) |
| `GET /api/lab/jobs/{id}?after=N` · `POST /api/lab/jobs/{id}/stop` | live progress events; stop and keep what was learned |
| `POST /api/lab/models/{id}/predict` | `{"rows": […]}` for tables; `{"prompt", "length", "temperature"}` or `{"query"}` for text |
| `POST /api/lab/models/{id}/predict_file?filename=…` | send a table (Excel/CSV, any size) or a zip of pictures, sounds or videos, get back a CSV with predictions |
| `POST /api/lab/models/{id}/predict_media?filename=…` (the file as the body) | the category of one picture, sound or video, with the model's confidence in each (`predict_image` still works) |
| `PATCH /api/lab/models/{id}` `{"name"}` · `GET /api/lab/models/{id}/download` · `POST /api/lab/models/import` | rename; download as a .zip; add a downloaded model |
| `GET /api/lab/datasets/{id}/preview.png` | examples from each category: pictures, spectrograms of sounds, or frames of videos |

### Limits

* PDF, DOCX, EPUB and zip files must be read whole, so those are limited to 4 GB; plain text, CSV and JSON-lines can be any size up to the cap. A single JSON array is limited to 2 GB; save bigger data as JSON-lines. Old .xls workbooks are not read (save as .xlsx or CSV), and only the first sheet of a workbook is used.
* Only the linear model and word counts read text columns. The trees, the one-shot net and the neural network use the number, category and date columns, and on tables that have both kinds, the average of the learners combines the two views.
* Learners run side by side in separate processes on Linux. On macOS and Windows they take turns in one process, sharing the time instead (`AI_ROOM_LAB_PARALLEL=0` does the same on Linux); LightGBM then uses every core on its turn. Measured with turns on the same computer: California housing 0.810, adult income 87.1%, forest cover 86.8%, all inside the time.
* LightGBM is imported in the background when AI Room starts (it takes up to a second). A job started in that first second uses the NumPy trees instead. A model trained with LightGBM needs LightGBM installed to be used; it is saved as LightGBM's own text format inside the model file, so loading it still never runs code.
* Sounds other than WAV and videos other than animated GIF need ffmpeg on the computer that trains. Only the first 30 s of each sound and the first 2 minutes of each video are used.
* One job trains at a time; others wait in a queue. Models are saved as JSON + NumPy arrays in `data/lab/` (no pickle), so loading a model never runs code.
* Phones: the Lab needs NumPy (`pkg install python-numpy` in Termux before `pip install -e .`).

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

## 🔧 Real training on a CPU (`ai-room train`)

The classroom itself never changes the student's weights. If you want weights that **actually change**, use the trainer. It fine-tunes a small open model with LoRA on your CPU, using problems from the teacher's exact generators. Every training answer is computed by code, and a test checks that each one is graded correct. Before and after training, the model sits the **same held-out exam** (no overlap with the training problems), graded by the classroom's own grader.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[train]"
ai-room train --model HuggingFaceTB/SmolLM2-135M-Instruct --subjects math --grades 1-3 --examples 1000 --epochs 2
ai-room ask data/trained/model/merged "What is 48 ÷ 6?"
```

**Measured result** (4-core Xeon, no GPU, the command above): held-out exam **5% → 47%**, in 12.9 minutes of training (250 steps, loss 0.64 → 0.10). By topic after training: divisibility 71%, multiplication/division 62%, percentages 57%, fractions 54%, addition/subtraction 37%, order of operations 10%.

What to expect:
* **It is real, and it is small.** A 135M model went from almost never answering in the right form to answering about half correctly. It still makes plenty of mistakes; a spot check of "48 ÷ 6" after training gave a wrong answer.
* **Bigger models learn more but cost much more.** A 360M model is about 3× slower per step; 1.5B is the practical ceiling on a CPU. A 7B model needs about 30 GB of RAM and days. 27B is not feasible on a CPU, so use the Colab/Kaggle notebooks for those.
* **Only exact-practice subjects** can be used as training data, because only they have answers computed by code.
* `--dtype bf16` exists, but on the test CPU it was no faster than fp32 (4.08 vs 4.06 s/step), so fp32 is the default.
* The output is a Hugging Face model folder (`merged/`) plus a small LoRA adapter. To use it in Ollama, convert it to GGUF with llama.cpp's `convert_hf_to_gguf.py`. That conversion step was not tested here.

## 🐘 Big models on a CPU (`ai-room bigmodel`)

Generating a token means reading every weight that token uses, once. So the speed limit is simply
**tokens/second ≤ bytes per second the device can deliver ÷ bytes per token**. llama.cpp already works at that limit: on the test
machine, Qwen2.5-1.5B (1.12 GB) generated 18.0 tokens/s, which is 20 GB/s against a measured memory bandwidth of 20.8 GB/s. No
engine can be 10× faster for the same model on the same CPU; there is nothing left to speed up.

What does make big models usable on a CPU:

* **Mixture-of-experts (MoE) models** read only the experts each token uses. Qwen3-30B-A3B has 30.5B parameters but uses about 3.3B
  per token, so it runs roughly as fast as a 3B dense model. That is the real "several times faster": it comes from the model, not
  the engine.
* **Expert streaming.** When the model is bigger than RAM, its file is memory-mapped: the shared weights stay in RAM and the experts
  each token needs are read from storage, unless they are still cached. It lets the model run at all. It does not make it fast.

Measured on the test machine (4-core Xeon, 15.7 GB RAM, no GPU, llama.cpp through llama-cpp-python):

| model | parameters | file | fits in RAM? | generation speed |
|---|---:|---:|---|---|
| Qwen2.5-1.5B (dense) | 1.5B | 1.1 GB | yes | 18.0 tokens/s |
| Qwen2.5-3B (dense) | 3.1B | 2.1 GB | yes | 9.8 tokens/s |
| Qwen3-30B-A3B (MoE, 3.3B active) | 30.5B | 18.6 GB | **no** | **1.2 tokens/s cold, 6.3 tokens/s warm** |

For the 30B run, 27.1 GB was read from disk (more than the file, because pages that did not fit were read again) and at most
14.3 GB was resident. By the same arithmetic, a dense 27B could manage at most about 0.9 tokens/s on this machine and a dense 70B
about 0.1. 1,200 tokens/s is out of reach on any CPU for models of that size.

```bash
ai-room bigmodel plan --target 1200                     # measures this device; runs on a phone (plain Python)
ai-room bigmodel plan --models 27,70,30.5/3.3           # dense 27B, dense 70B, MoE 30.5B with 3.3B active
ai-room bigmodel bench path/to/model.gguf              # times a real model and reports how much it read from storage
```

* `plan` gives ceilings. Real engines reach most of the in-RAM figure; streaming usually lands below its figure (the cold 30B run
  reached about half), and a 1 GB storage test can read faster than a multi-GB model does.
* `bench` turns off llama.cpp's weight **repacking**. Repacking copies weights into ordinary memory for faster CPU kernels, which
  cannot be streamed: with it on, the 18.6 GB model was killed for running out of memory. Use `--repack` only for models that fit.
* Phones stream from flash storage, which is usually slower than a computer's SSD, and need a llama.cpp build that runs there.
* The classroom's 27B limit counts *total* parameters, so a 30B MoE model is still refused as a student.

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
  train_cpu.py         real LoRA fine-tuning on a CPU with a before/after exam (`ai-room train`, `ai-room ask`)
  boost.py             better answers without training, measured (`ai-room boost`, `ai-room speed`)
  bigmodel.py          what fits on this device and how fast it can go; real benchmarks (`ai-room bigmodel`)
  lab/
    data.py            reading uploads, column kinds, rows → numbers (fitted on training rows only), hashed text
    learners.py        the untrained learners: one-shot net, word counts, histogram boosted trees, neural network / linear model
    automl.py          70/15/15 split, the time-budgeted race, choosing the winner, metrics, importance, save/load
    parallel.py        learners side by side in forked processes (one per core), stopped for sure at the deadline
    lgbm.py            optional LightGBM learner: imported ahead of time, fork-safe, saved as text (no pickle)
    textgen.py         one-pass character model (modified Kneser–Ney) and BM25 passage search
    api.py  cli.py     Lab HTTP API with a training queue (`/api/lab/…`) and `ai-room lab …`
    xlsx.py            Excel workbooks with the standard library      images.py   pictures → 936 features, contact sheets
    media.py           sounds → 212 features, videos → 1,172 (WAV/GIF built in, other formats through ffmpeg), previews
    ingest.py          streamed uploads, head-only inspection, sampling files bigger than memory
    samples.py         the eight built-in practice datasets, generated by code
tests/                 ~270 offline tests
```

Run the tests with `pip install -e ".[dev]" && pytest`.
