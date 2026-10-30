"""Turn textbook passages into facts and exam questions — with plain rules, no model.

Two kinds of fact are recognised:
  * definitions   "A cell is the basic unit of life."          → term = "cell"
  * dated events  "The treaty was signed in 1648 by ..."        → year = "1648"
Each becomes a fill-in-the-blank question, or a multiple-choice question when
enough other terms exist in the same subject to serve as distractors.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

from .library import Passage

_ABBREV = re.compile(r"\b(Dr|Mr|Mrs|Ms|Prof|St|vs|etc|Fig|Jr|Sr|approx|ca|Inc|Ltd|No)\.|\b([A-Z])\.(?=\s?[A-Z])|\b(e\.g|i\.e|U\.S|U\.K)\.")

STARTERS = {
    "in", "on", "at", "by", "for", "with", "from", "during", "after", "before", "when", "while", "if", "as", "and", "but", "or", "also", "however",
    "although", "because", "since", "this", "that", "these", "those", "it", "he", "she", "they", "there", "here", "some", "many", "most", "other",
    "such", "his", "her", "their", "its", "one", "another", "each", "both", "all", "several", "what", "which", "who", "we", "you", "i", "within",
    "under", "over", "between", "among", "through", "about", "into", "than", "then", "thus", "therefore", "so", "later", "early", "today", "now",
    "first", "second", "third", "finally", "meanwhile", "instead", "yet", "still", "even", "only", "just", "not", "no", "any", "every", "whether",
    "after", "until", "once", "unlike", "like", "despite", "according", "overall", "generally", "usually", "often", "sometimes", "typically",
}
VERBS = r"(?:is|are|was|were|refers to|means|describes|denotes|can be defined as|is called|are called|is known as|are known as)"
_ART = re.compile(rf"^(?P<art>The|A|An) (?P<term>[A-Za-z][\w'’\-]*(?: [\w'’\-]+){{0,3}}?) (?P<verb>{VERBS}) (?P<rest>.+)$")
_BARE = re.compile(rf"^(?P<term>[A-Z][\w'’\-]*(?: [\w'’\-]+){{0,3}}?) (?P<verb>{VERBS}) (?P<rest>.+)$")
_YEAR = re.compile(r"(?<![\d–\-/.,])(1[0-9]{3}|20[0-9]{2})(?![\d–\-/]|\.\d|,\d)")
_BAD_REST = re.compile(r"\b(this|these|those|them|above|below|following|mentioned|aforementioned|such)\b", re.I)
_BAD_TERM = {"which", "who", "what", "that", "there", "one", "ones", "thing", "things", "way", "part", "result", "fact", "idea", "example", "number", "set", "name", "type", "form", "kind"}


@dataclass
class Fact:
    kind: str            # "def" or "year"
    sentence: str
    answer: str
    pre: str             # text before the blank
    post: str            # text after the blank
    key: str
    passage_id: int
    title: str
    url: str
    heading: str
    topic: str = ""

    @property
    def concept(self) -> str:
        return f"kb:{self.key}"


def clean_sentence(s: str) -> str:
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip(" \"'")


def split_sentences(text: str) -> list[str]:
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", "∯"), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“])", protected)
    return [p.replace("∯", ".").strip() for p in parts if p.strip()]


def _hash(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:10]


def facts_from_passage(p: Passage) -> list[Fact]:
    out: list[Fact] = []
    for raw in split_sentences(p.text):
        s = clean_sentence(raw)
        words = s.split()
        if not 8 <= len(words) <= 40 or not s.endswith(".") or s[0].islower() or '"' in s or ":" in s or ";" in s:
            continue
        first = words[0].lower()
        m = _ART.match(s) or (_BARE.match(s) if first not in STARTERS else None)
        if m:
            term = m.group("term").strip()
            rest = m.group("rest")
            low = term.lower()
            wc = len(term.split())
            if (3 <= len(term) <= 40 and wc <= 4 and low not in _BAD_TERM and not any(w in STARTERS for w in low.split()[:1] if not m.groupdict().get("art"))
                    and 5 <= len(rest.split()) <= 36 and not _BAD_REST.search(rest) and low not in rest.lower() and not low.isdigit()
                    and low.split()[-1] not in _BAD_TERM):
                art = m.groupdict().get("art")
                pre = (art + " " if art else "")
                post = f" {m.group('verb')} {rest}"
                out.append(Fact("def", s, term, pre, post, _hash(p.id, s), p.id, p.title, p.url, p.heading, p.topic))
                continue
        years = _YEAR.findall(s)
        lead_year = bool(re.match(r"^In (1[0-9]{3}|20[0-9]{2}), [A-Za-z]", s))  # "In 1648, the treaty ..." keeps its context
        if len(years) == 1 and (first not in STARTERS or lead_year) and len(words) >= 9:
            y = years[0]
            i = s.index(y)
            out.append(Fact("year", s, y, s[:i], s[i + len(y):], _hash(p.id, s), p.id, p.title, p.url, p.heading, p.topic))
    return out


def facts_from_passages(passages: list[Passage]) -> list[Fact]:
    seen, out = set(), []
    for p in passages:
        for f in facts_from_passage(p):
            k = (f.kind, f.answer.lower())
            if k not in seen:
                seen.add(k)
                out.append(f)
    return out


def _distractors(fact: Fact, pool: list[Fact], rng: random.Random, n: int = 3) -> list[str]:
    if fact.kind == "year":
        y = int(fact.answer)
        cands = {y + d for d in (-100, -50, -25, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 50) if 1000 <= y + d <= 2100}
        return [str(v) for v in rng.sample(sorted(cands), min(n, len(cands)))]
    ans = fact.answer.lower()
    same = [f.answer for f in pool if f.kind == "def" and f.answer.lower() != ans and ans not in f.answer.lower() and f.answer.lower() not in ans]
    same = list(dict.fromkeys(same))
    wc = len(fact.answer.split())
    near = [t for t in same if len(t.split()) == wc]
    rng.shuffle(near)
    rng.shuffle(same)
    out = near[:n] + [t for t in same if t not in near][: max(0, n - len(near))]
    return out[:n]


def make_question(fact: Fact, pool: list[Fact], rng: random.Random, prefer_mcq: bool) -> dict:
    blank = "_____"
    text = f"{fact.pre}{blank}{fact.post}"
    source = {"title": fact.title, "url": fact.url, "heading": fact.heading}
    explanation = f"“{fact.sentence}” — {fact.title}" + (f", {fact.heading}" if fact.heading else "")
    wrong = _distractors(fact, pool, rng)
    if prefer_mcq and len(wrong) >= 3:
        opts = [o[:1].upper() + o[1:] for o in wrong[:3] + [fact.answer]]  # same capitalisation, so case gives nothing away
        answer_opt = fact.answer[:1].upper() + fact.answer[1:]
        rng.shuffle(opts)
        letters = "ABCD"
        label = "year" if fact.kind == "year" else "term"
        return {"type": "mcq", "question": f"Which {label} completes this sentence from the textbook?\n{text}", "options": [f"{letters[i]}) {o}" for i, o in enumerate(opts)],
                "answer": letters[opts.index(answer_opt)], "explanation": explanation, "concept": fact.concept, "source": source}
    label = "Fill in the blank with the missing year" if fact.kind == "year" else "Fill in the blank with the missing term"
    return {"type": "cloze", "question": f"{label}:\n{text}", "answer": fact.answer, "explanation": explanation, "concept": fact.concept, "source": source, "exact": fact.kind == "year"}


def bold_term(fact: Fact) -> str:
    """The original sentence with the key term in bold, for 'key terms' lists."""
    return f"{fact.pre}**{fact.answer}**{fact.post}"
