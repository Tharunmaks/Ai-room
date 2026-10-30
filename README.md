# 🏫 AI Room — a classroom where AI learns without training

AI Room is a school for language models. A **teacher** model teaches a **student** model (≤ 27B parameters) through **12 classes** of increasing difficulty, in up to **100 subjects**: math, physics, logical thinking, coding, research, and 95 more. The student moves up to the next class only after it scores **≥ 90 % in every subject** you chose.

No GPU training or fine-tuning is involved. The student's weights never change. It learns the way a human student does:

| Human student | AI Room student |
|---|---|
| Listens to the lesson | Reads the teacher's lesson (in-context) |
| Writes notes | Writes study notes into a persistent **notebook** (SQLite) |
| Raises a hand | Asks the teacher follow-up questions; the answers go into the notebook |
| Studies before the exam | Retrieves the most relevant notebook pages (BM25) for every exam question |
| Learns from mistakes | After a failed exam the teacher reviews each mistake and the student writes a **mistake log** of general rules; these notes are ranked highest at retrieval time |
| Makes a cheat-sheet | After passing a class it condenses its notes into a summary for later classes |

Every exam is newly generated and never repeats earlier questions, so the student can't pass by memorising answers. It has to carry the methods forward in its notes.

## Features

- **12 classes**, from *Foundations* to *Mastery*, each with its own syllabus per subject.
- **100 subjects** in 9 categories: Core, Mathematics & Statistics, Physical Sciences, Life & Health Sciences, Engineering, Computing & AI, Social Sciences & Business, Humanities & Arts, and World Languages. The UI has search and a per-category "select all".
- **Promotion rule**: a score at or above the pass mark (90 % by default) in *every* selected subject. Subjects already passed in a class aren't examined again. Failed subjects get a focused re-lesson and a fresh exam, up to N attempts per class.
- **Choose the teacher's behaviour**: Patient, Socratic, Strict, Funny, Researcher, Exam Coach, or write your own personality.
- **The student talks back**: it asks questions during lessons, and you can chat with it (or with the teacher) at any time.
- **Grading you can trust**: numeric answers are checked with tolerance, multiple choice is matched by letter, and code answers run against unit tests. Only free-text answers are graded by the teacher against a rubric. The teacher re-solves numeric and multiple-choice questions on its own and drops any whose answer key it can't reproduce.
- **27B limit enforced**: the student's size is read from Ollama (`/api/show`) or the model name. Students over 27B, MoE models such as `8x7b` (56B total) or `30b-a3b`, and models of unknown size are rejected unless you state the size.
- **Backends**: Ollama, any OpenAI-compatible server (vLLM, llama.cpp, LM Studio), Anthropic Claude (teacher only), and an offline **mock** for demos and tests.

## Quick start

```bash
pip install -e .                 # add ".[anthropic]" to use Claude as the teacher
ollama pull qwen2.5:14b          # student (≤ 27B)
ollama pull gemma3:27b           # teacher (any size or provider)
ai-room serve                    # open http://127.0.0.1:8765
```

Click **New classroom**, pick the student and teacher models, the teacher's behaviour and the subjects, then press **▶ Run class** (one class) or **⏩ Run all 12 classes**.

No models installed? Choose **Mock (offline demo)** for both models to see the whole flow.

### Terminal

```bash
ai-room run --student ollama:qwen2.5:14b --teacher ollama:gemma3:27b \
            --subjects math,physics,logic,code,research,astronomy,spanish \
            --persona socratic --until all
ai-room run --resume <classroom-id> --until all   # continue later
```

Model specs are `backend:model`, where backend is `ollama`, `openai`, `anthropic` or `mock`. For an OpenAI-compatible server, set `OPENAI_BASE_URL` (and `OPENAI_API_KEY` if it needs one). For Claude, set `ANTHROPIC_API_KEY`.

## Tips

- **The teacher matters most.** The student can only learn what the teacher explains correctly, and exam answer keys come from the teacher. Use the strongest teacher you can, such as a 27B+ local model or Claude.
- 10 questions per exam at a 90 % pass mark means at most one mistake per subject. Lower the number of questions for faster runs, or raise the pass mark for a stricter school.
- The notebook works with any model. You can create a new classroom with a different student and compare how fast each one climbs the 12 classes.

## Safety note

Code answers written by the student are run in a separate Python process with a timeout, inside an empty temporary directory. That isn't a real sandbox, so run AI Room in a container or VM if you use untrusted models.

## Layout

```
ai_room/
  llm.py         model backends + 27B guard
  curriculum.py  12 classes, the 10 core subjects, pass mark
  catalog.py     90 more subjects (6 milestones each, spread over 12 classes)
  personas.py    teacher behaviours
  memory.py      notebook, exams and transcript (SQLite) + BM25 retrieval
  agents.py      Teacher and Student prompts
  grading.py     answer extraction, numeric/MCQ checks, code tests
  classroom.py   lessons → Q&A → exam → reflection → promotion
  server.py      FastAPI API + live transcript (SSE)
  web/           single-page UI
tests/           pytest suite (offline, scripted models)
```

Run the tests with `pip install -e ".[dev]" && pytest`.
