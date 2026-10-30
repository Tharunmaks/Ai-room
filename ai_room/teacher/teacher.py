"""The non-AI teacher.

Lessons are assembled from lesson cards (formulas and methods), worked examples
produced by the exact generators, passages from the textbook library and a
key-terms list. Exams mix generated problems, textbook fill-in-the-blanks and
imported question banks. Every answer key is computed or quoted — never guessed.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from ..curriculum import SUBJECTS, difficulty, topic_phrases
from ..memory import tokenize
from .generators import FAMILIES, Family, families_for, generate, seed_for
from .knowledge import Fact, bold_term, facts_from_passages, make_question, split_sentences
from .library import Library, Passage
from .style import TeacherStyle, Voice


class NoCourseMaterial(RuntimeError):
    pass


# words that make a question a question but say nothing about its subject
_ASKING = {"mean", "means", "meaning", "does", "exactly", "explain", "tell", "about", "define", "definition", "describe", "please", "know",
           "word", "term", "work", "works", "use", "used", "using", "difference", "between", "example", "examples", "give", "show", "help",
           "understand", "can", "could", "would", "something", "anything", "thing", "things", "me", "my", "us", "more", "some"}


@dataclass
class Lesson:
    text: str
    concepts: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)


def _opts(q: dict) -> str:
    return "\n".join(q.get("options") or [])


def _show_question(q: dict) -> str:
    return q["question"] + (("\n" + _opts(q)) if q["type"] == "mcq" else "")


def _show_answer(q: dict) -> str:
    if q["type"] == "mcq":
        letter = q["answer"]
        opt = next((o for o in q.get("options", []) if o.startswith(letter)), letter)
        return opt
    if q["type"] == "code":
        return "see the reference solution"
    return q["answer"]


def _worked(q: dict) -> str:
    """A fully worked example: problem, method and answer."""
    if q["type"] == "code":
        return f"{q['question']}\n{q['explanation']}"
    why = f" {q['explanation']}." if q.get("explanation") and q["type"] != "numeric" else ""
    steps = f"\n*Working:* {q['explanation']}" if q["type"] == "numeric" and q.get("explanation") else ""
    return f"{_show_question(q)}{steps}\n**Answer: {_show_answer(q)}**{why}"


class Teacher:
    def __init__(self, library: Library, style: TeacherStyle, student_name: str = "Student", knowledge_share: float = 0.4):
        self.lib = library
        self.style = style
        self.name = style.name
        self.student = student_name
        self.knowledge_share = knowledge_share
        self._facts: dict[tuple, tuple[list[Passage], list[Fact]]] = {}

    # ------------------------------------------------------------ material
    def families(self, subject: str, grade: int, focus: list[str] | None = None) -> list[Family]:
        """The lesson cards taught this class — a stable subset so the exam only covers what was taught."""
        types = set(SUBJECTS[subject].question_types)
        fams = [f for f in families_for(subject, grade) if not (f.id.startswith("code.write") and "code" not in types)]
        limit = 2 + self.style.depth
        if len(fams) > limit:
            rng = random.Random(seed_for("cards", subject, grade))
            keep = set(f.id for f in rng.sample(fams, limit))
            keep |= {c[4:] for c in (focus or []) if c.startswith("fam:")}
            fams = [f for f in fams if f.id in keep]
        return fams

    def _material(self, subject: str, grade: int) -> tuple[list[Passage], list[Fact]]:
        """Passages and facts for a class: those matching its topics first; if the library has little on them
        (e.g. a book you imported that doesn't use the syllabus words) the book's facts are spread over the classes."""
        key = (subject, grade, self.lib.version)
        if key not in self._facts:
            phrases = topic_phrases(subject, grade)
            passages = self.lib.search(subject, " ".join(phrases), k=30, topics=set(phrases), grade=grade)
            facts = facts_from_passages(passages)
            if len(facts) < 6:
                everything = self.lib.passages(subject)[:400]
                pool = [f for f in facts_from_passages(everything) if f.key not in {x.key for x in facts}]
                if len(pool) >= 72:
                    pool = pool[(grade - 1)::12]
                facts = facts + pool[: max(0, 12 - len(facts))]
                have = {p.id for p in passages}
                wanted = {f.passage_id for f in facts}
                passages = passages + [p for p in everything if p.id in wanted and p.id not in have]
            self._facts[key] = (passages, facts)
        return self._facts[key]

    def facts(self, subject: str, grade: int) -> list[Fact]:
        return self._material(subject, grade)[1]

    def taught_facts(self, subject: str, grade: int) -> list[Fact]:
        return self.facts(subject, grade)[: 8 + 4 * self.style.depth]

    def has_material(self, subject: str, grade: int) -> bool:
        return bool(self.families(subject, grade) or self.facts(subject, grade) or self.lib.questions(subject, grade))

    def missing_message(self, subject: str) -> str:
        return (f"There is no course material for {SUBJECTS[subject].name} yet. Open the Library tab and fetch textbooks for it "
                "(or import your own books or a question bank), then run the class again.")

    # ------------------------------------------------------------ lessons
    def lesson(self, subject: str, grade: int, focus: list[str] | None = None) -> Lesson:
        st = self.style
        rng = random.Random(seed_for("lesson", subject, grade, st.tone, st.approach, bool(focus)))
        v = Voice(st, rng, self.student)
        fams = self.families(subject, grade, focus)
        facts = self.taught_facts(subject, grade)
        phrases = topic_phrases(subject, grade)
        passages = self._material(subject, grade)[0][: st.depth]
        if not (fams or facts or passages):
            raise NoCourseMaterial(self.missing_message(subject))

        sub = SUBJECTS[subject]
        out: list[str] = [f"## {sub.name} · Class {grade} — {difficulty(grade)}", v.open(", ".join(phrases)), ""]
        concepts: list[str] = []
        sources: list[dict] = []
        jokes = v.jokes(st.humor)
        if jokes and st.humor >= 2:
            out += [f"> {jokes.pop(0)}", ""]
        if focus:
            out += [f"**{v.wrong()}** Last exam showed gaps, so we revisit those ideas first.", ""]

        out += ["**Today's goals**"] + [f"- {p}" for p in phrases] + [""]

        # ---- core ideas from lesson cards + worked examples
        if fams:
            out += ["### Core ideas", ""]
        for fam in sorted(fams, key=lambda f: (f"fam:{f.id}" not in (focus or []), list(FAMILIES).index(f.id))):
            concepts.append(f"fam:{fam.id}")
            card = fam.card if st.approach != "drill" else fam.card.split("\n")[0]
            n_ex = st.examples + (2 if focus and f"fam:{fam.id}" in focus else 0)
            if st.approach == "drill":
                n_ex = min(n_ex, 1)
            exs = [generate(fam, subject, grade, random.Random(seed_for("ex", subject, grade, fam.id, i))) for i in range(n_ex)]
            out += [f"#### {fam.title}", v.bridge()]
            if st.approach == "socratic" and exs:
                out += ["", "*Before I explain — try this on your own:*", "", _show_question(exs[0]), "", "*…now compare with the method below.*"]
            out += ["", card, ""]
            for i, ex in enumerate(exs, 1):
                out += [f"**Worked example {i}** — {_worked(ex)}", ""]
            if fam.pitfalls:
                out += [f"⚠️ **Watch out:** {fam.pitfalls}", ""]
            if st.praise >= 3 and exs:
                out += [f"*{v.praise()}*", ""]

        # ---- textbook readings
        if passages:
            out += ["### From the textbook", ""]
            for p in passages:
                if st.approach == "socratic":
                    out += [f"*Question to ponder:* what is the main point of “{p.heading or p.title}”?", ""]
                text = p.text if len(p.text) <= 800 else p.text[:800].rsplit(" ", 1)[0] + " …"
                out += [f"> {text}"]
                if st.sources:
                    where = f"*{p.title}* ({p.source})" + (f", “{p.heading}”" if p.heading else "") + (f" · {p.url}" if p.url else "")
                    out += [f"— {where}"]
                out += [""]
                sources.append({"title": p.title, "url": p.url, "source": p.source, "heading": p.heading})

        # ---- key terms
        reviewed = [f for f in facts if f.concept in (focus or [])]
        shown = reviewed + [f for f in facts if f not in reviewed]
        if shown:
            out += ["### Key terms to remember", ""]
            for f in shown:
                concepts.append(f.concept)
                tag = f" ({f.title})" if st.sources else ""
                out += [f"- {bold_term(f)}{tag}"]
                sources.append({"title": f.title, "url": f.url, "source": "", "heading": f.heading})
            out += [""]

        # ---- practice
        practice = self._practice(subject, grade, fams, facts, focus or [], st.practice, rng)
        if practice:
            out += ["### Practice", ""]
            hinted: set[str] = set()
            for i, q in enumerate(practice, 1):
                if st.approach == "socratic":
                    out += [f"{i}. {_show_question(q)}", "   *Think first, then compare:* " + _show_answer(q), ""]
                else:
                    out += [f"{i}. {_show_question(q)}", f"   **Answer: {_show_answer(q)}**", ""]
                c = q.get("concept", "")
                if st.hints and c.startswith("fam:") and c not in hinted and FAMILIES[c[4:]].pitfalls and st.approach != "drill":
                    hinted.add(c)
                    out += [f"   *Hint: reread the ‘Watch out’ note under “{FAMILIES[c[4:]].title}”.*", ""]

        if jokes:
            out += [f"> {jokes.pop(0)}", ""]
        if st.praise >= 2:
            out += [f"*{v.encouragement()}*", ""]
        out += [v.close()]
        seen = set()
        uniq_sources = [s for s in sources if not (s["title"], s["heading"]) in seen and not seen.add((s["title"], s["heading"]))]
        return Lesson("\n".join(out).strip(), list(dict.fromkeys(concepts)), uniq_sources)

    def _practice(self, subject, grade, fams, facts, focus, n, rng) -> list[dict]:
        if n <= 0:
            return []
        qs: list[dict] = []
        pool_fams = [f for f in fams if f"fam:{f.id}" in focus] + [f for f in fams if f"fam:{f.id}" not in focus]
        for i in range(n):
            if pool_fams and (not facts or i % 3 != 2):
                fam = pool_fams[i % len(pool_fams)]
                q = generate(fam, subject, grade, random.Random(seed_for("practice", subject, grade, fam.id, i)))
                if q["type"] != "code":
                    qs.append(q)
            elif facts:
                f = facts[(i * 5) % len(facts)]
                qs.append(make_question(f, facts, random.Random(seed_for("pk", f.key)), prefer_mcq=False))
        return qs[:n]

    # ------------------------------------------------------------ exams
    def exam(self, subject: str, grade: int, count: int, attempt: int, avoid: set[str] | None = None) -> list[dict]:
        avoid = set(avoid or ())
        types = set(SUBJECTS[subject].question_types)
        rng = random.Random(seed_for("exam", subject, grade, attempt))
        fams = self.families(subject, grade)
        facts = self.taught_facts(subject, grade)
        bank = [q for q in self.lib.questions(subject, grade) if q["type"] != "code" or "code" in types]
        if not (fams or facts or bank):
            raise NoCourseMaterial(self.missing_message(subject))

        chosen: list[dict] = []
        texts = set(avoid)

        def add(q: dict | None) -> bool:
            if q and q["question"] not in texts:
                texts.add(q["question"])
                chosen.append(q)
                return True
            return False

        fresh_bank = [q for q in bank if q["question"] not in texts]
        rng.shuffle(fresh_bank)
        n_bank = min(len(fresh_bank), round(count * 0.4)) if (fams or facts) else min(len(fresh_bank), count)
        for q in fresh_bank[:n_bank]:
            add(q)
        remaining = count - len(chosen)
        n_kb = 0 if not facts else (remaining if not fams else round(remaining * self.knowledge_share))
        # textbook questions
        order = list(facts)
        rng.shuffle(order)
        mcq_p = 0.8 if grade <= 5 else 0.5 if grade <= 8 else 0.25
        for f in order:
            if sum(1 for q in chosen if q["concept"].startswith("kb:")) >= n_kb:
                break
            add(make_question(f, facts, rng, prefer_mcq=rng.random() < mcq_p))
        # generated problems, round-robin over the taught cards
        if fams:
            shuffled = list(fams)
            rng.shuffle(shuffled)
            i, tries = 0, 0
            while len(chosen) < count and tries < count * 40:
                fam = shuffled[i % len(shuffled)]
                add(generate(fam, subject, grade, random.Random(rng.getrandbits(48))))
                i += 1
                tries += 1
        # top up from anything left (repeats allowed only as a last resort)
        for pool in (order, fresh_bank[n_bank:]):
            for item in pool:
                if len(chosen) >= count:
                    break
                add(make_question(item, facts, rng, prefer_mcq=rng.random() < .5) if isinstance(item, Fact) else item)
        if len(chosen) < count:
            for f in order:
                if len(chosen) >= count:
                    break
                q = make_question(f, facts, rng, prefer_mcq=rng.random() < .5)
                if q["question"] not in {c["question"] for c in chosen}:
                    chosen.append(q)
        if not chosen:
            raise NoCourseMaterial(self.missing_message(subject))
        rng.shuffle(chosen)
        return chosen[:count]

    # ------------------------------------------------------------ questions from the student / user
    def answer(self, subjects: list[str] | str, grade: int | None, text: str) -> str:
        subjects = [subjects] if isinstance(subjects, str) else subjects
        st = self.style
        rng = random.Random(seed_for("answer", text, st.tone))
        v = Voice(st, rng, self.student)
        q = set(tokenize(text)) - _ASKING
        best_card: tuple[int, Family, str] | None = None
        best_passage: tuple[float, Passage, str] | None = None
        for s in subjects:
            pool = [f for f in FAMILIES.values() if s in f.subjects and (grade is None or f.subjects[s][0] <= grade + 2)]
            for fam in pool:
                overlap = len(q & set(tokenize(fam.title + " " + fam.card)))
                if overlap >= 2 and (best_card is None or overlap > best_card[0]):
                    best_card = (overlap, fam, s)
            for p in self.lib.search(s, text, k=3, grade=grade):
                ov = len(q & set(tokenize(p.text)))
                if ov >= max(1, (len(q) + 1) // 2) and (best_passage is None or p.score > best_passage[0]):  # at least half the content words
                    best_passage = (p.score, p, s)
        parts: list[str] = []
        if best_card and (not best_passage or best_card[0] >= 3):
            fam = best_card[1]
            parts += [f"**{fam.title}**", fam.card]
            if fam.pitfalls and st.hints:
                parts += [f"⚠️ Watch out: {fam.pitfalls}"]
        if best_passage:
            p = best_passage[1]
            sents = split_sentences(p.text)
            scored = sorted(range(len(sents)), key=lambda i: -len(q & set(tokenize(sents[i]))))[:2 + (st.depth > 3)]
            excerpt = " ".join(sents[i] for i in sorted(scored))
            parts += [f"> {excerpt}"]
            if st.sources:
                parts += [f"— *{p.title}* ({p.source})" + (f" · {p.url}" if p.url else "")]
        if not parts:
            return {
                "warm": "That's a good question, but I can't find it in my textbooks yet. You can add a book about it in the Library tab and ask me again.",
                "strict": "That is not covered in my textbooks. Add a source in the Library tab.",
                "playful": "Ha! You've stumped my bookshelf 📚 — nothing about that in my textbooks yet. Try adding one in the Library tab.",
            }.get(st.tone, "I can't find that in my textbooks yet. Add a source in the Library tab and ask again.")
        return "\n\n".join(parts)

    # ------------------------------------------------------------ reviewing mistakes
    def review(self, subject: str, grade: int, mistakes: list[dict]) -> str:
        st = self.style
        rng = random.Random(seed_for("review", subject, grade, len(mistakes)))
        v = Voice(st, rng, self.student)
        out = [f"### Reviewing the mistakes — {SUBJECTS[subject].name}, Class {grade}", "", v.wrong(), ""]
        for i, m in enumerate(mistakes, 1):
            out += [f"**{i}. {m['question'].splitlines()[0][:160]}**", f"Your answer: `{str(m['given'])[:120]}` — correct answer: **{m['expected'][:160]}**"]
            if m.get("feedback"):
                out += [f"Why: {m['feedback'][:500]}"]
            c = m.get("concept", "")
            if c.startswith("fam:") and c[4:] in FAMILIES:
                fam = FAMILIES[c[4:]]
                out += [f"Method — *{fam.title}*: {fam.card.splitlines()[0]}"]
                if fam.pitfalls:
                    out += [f"⚠️ Common mistake: {fam.pitfalls}"]
            out += [""]
        if st.praise >= 1:
            out += [v.encouragement()]
        return "\n".join(out).strip()
