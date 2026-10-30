"""The 12-grade curriculum.

Each subject has one entry per grade. A grade lists the topics the teacher
covers and the kinds of questions allowed on that grade's exam.
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import expand_syllabus, parse_catalog

NUM_GRADES = 12
PASS_MARK = 0.90

DIFFICULTY = [
    "Foundations", "Beginner", "Elementary", "Developing", "Intermediate", "Upper-intermediate",
    "Proficient", "Advanced", "Expert", "Specialist", "Research-level", "Mastery",
]


@dataclass(frozen=True)
class Subject:
    key: str
    name: str
    question_types: tuple[str, ...]  # allowed exam question types
    grades: tuple[str, ...]          # 12 topic lists, one per grade
    category: str = "Core"


SUBJECTS: dict[str, Subject] = {s.key: s for s in [
    Subject("math", "Mathematics", ("numeric", "mcq", "short"), (
        "counting, place value, addition and subtraction of whole numbers",
        "multiplication, division, remainders, order of operations",
        "fractions, decimals, percentages, ratios",
        "negative numbers, exponents, square roots, prime factorisation, GCD/LCM",
        "linear equations, inequalities, coordinate plane, basic geometry (area, perimeter, angles)",
        "systems of equations, quadratics, Pythagorean theorem, probability basics",
        "functions, polynomials, logarithms, sequences and series, trigonometry",
        "limits, derivatives and their rules, optimisation, combinatorics",
        "integrals, techniques of integration, vectors, matrices and determinants",
        "differential equations, eigenvalues, probability distributions, statistics",
        "real analysis (epsilon-delta), abstract algebra (groups, rings), proof techniques",
        "complex analysis, topology basics, measure-theoretic probability, olympiad-style problem solving",
    )),
    Subject("physics", "Physics", ("numeric", "mcq", "short"), (
        "units, measurement, what force, motion and energy are",
        "speed, distance, time; mass vs weight; simple machines",
        "Newton's three laws, friction, density and pressure",
        "work, energy, power, conservation of energy, heat and temperature",
        "kinematics in 1D and 2D, projectile motion, momentum and collisions",
        "circular motion, gravitation, waves and sound, simple harmonic motion",
        "electric charge, Coulomb's law, circuits (Ohm's law, series/parallel), magnetism",
        "thermodynamics laws, ideal gases, geometric optics, electromagnetic induction",
        "rotational dynamics, angular momentum, Lagrangian mechanics intro, AC circuits",
        "Maxwell's equations, special relativity, wave optics",
        "quantum mechanics (Schrodinger equation, operators, hydrogen atom), statistical mechanics",
        "general relativity basics, quantum field theory concepts, condensed matter and particle physics",
    )),
    Subject("logic", "Logical Thinking", ("mcq", "short", "numeric"), (
        "patterns and sequences, sorting and classifying, same/different",
        "if-then reasoning, simple puzzles, true/false statements",
        "logical connectives (and, or, not), truth tables",
        "deductive vs inductive reasoning, syllogisms, Venn diagrams",
        "common fallacies, necessary vs sufficient conditions, counterexamples",
        "knights-and-knaves puzzles, grid/constraint puzzles, logical equivalence",
        "propositional proofs, contrapositive, proof by contradiction",
        "predicate logic, quantifiers, formalising arguments",
        "Bayesian reasoning, base-rate fallacy, causal vs correlational inference",
        "game theory (dominant strategies, Nash equilibrium), decision analysis",
        "modal logic, paradoxes, computability and decidability basics",
        "Gödel's incompleteness ideas, formal systems, hard multi-step reasoning challenges",
    )),
    Subject("code", "Coding", ("code", "mcq", "short"), (
        "what a program is, variables, printing, basic arithmetic in Python",
        "conditionals, loops, simple string handling",
        "functions, lists, dictionaries, basic debugging",
        "string algorithms, nested loops, list comprehensions, reading error messages",
        "recursion, sorting and searching, Big-O notation",
        "classes and objects, exceptions, writing unit tests",
        "stacks, queues, linked lists, hash maps, two-pointer and sliding-window techniques",
        "trees, binary search trees, heaps, graph traversal (BFS/DFS)",
        "dynamic programming, greedy algorithms, shortest paths",
        "concurrency, system design basics, databases and SQL, API design",
        "advanced algorithms (segment trees, network flow, string matching), complexity classes",
        "compilers/interpreters, distributed systems, performance engineering, competitive programming",
    )),
    Subject("research", "Research Skills", ("short", "mcq"), (
        "asking good questions, observation vs opinion",
        "finding information, distinguishing facts from claims",
        "summarising a text, identifying the main idea and supporting evidence",
        "evaluating sources: credibility, bias, primary vs secondary sources",
        "forming hypotheses, independent/dependent variables, controls",
        "designing experiments, sampling, confounding variables",
        "reading scientific papers: abstract, methods, results, limitations",
        "statistical significance, effect sizes, p-hacking, reproducibility",
        "literature review, synthesising conflicting evidence, citation practice",
        "research ethics, pre-registration, peer review, threats to validity",
        "writing a research proposal, choosing methods, identifying open problems",
        "producing original research ideas, critiquing state-of-the-art work, meta-analysis",
    )),
    Subject("chemistry", "Chemistry", ("numeric", "mcq", "short"), (
        "states of matter, mixtures vs pure substances",
        "atoms, elements, the periodic table layout",
        "atomic structure, electrons, ions, isotopes",
        "chemical bonds (ionic, covalent, metallic), simple formulas",
        "balancing equations, the mole, molar mass",
        "stoichiometry, limiting reagents, solutions and concentration",
        "acids and bases, pH, redox reactions",
        "gas laws, thermochemistry, enthalpy, reaction rates",
        "chemical equilibrium, Le Chatelier, electrochemistry",
        "organic chemistry: functional groups, reaction mechanisms",
        "quantum chemistry, spectroscopy, thermodynamics (Gibbs energy, entropy)",
        "advanced synthesis, catalysis, computational chemistry concepts",
    )),
    Subject("biology", "Biology", ("mcq", "short"), (
        "living vs non-living, basic needs of organisms",
        "plants and animals, habitats, life cycles",
        "cells, organs and organ systems",
        "food chains, ecosystems, photosynthesis and respiration basics",
        "cell structure, mitosis, basic genetics (dominant/recessive)",
        "DNA, meiosis, Mendelian inheritance, evolution by natural selection",
        "human physiology: nervous, circulatory, immune systems",
        "molecular biology: transcription, translation, gene regulation",
        "biochemistry: enzymes, metabolism, cell signalling",
        "population genetics, ecology models, developmental biology",
        "genomics, bioinformatics concepts, systems biology",
        "frontier topics: CRISPR, synthetic biology, neuroscience, experimental design critique",
    )),
    Subject("language", "Language & Writing", ("short", "mcq"), (
        "letters, words, simple sentences",
        "nouns, verbs, adjectives, punctuation",
        "paragraphs, topic sentences, reading comprehension",
        "tenses, synonyms/antonyms, narrative writing",
        "persuasive writing, clauses, figurative language",
        "essay structure, thesis statements, citing evidence",
        "rhetoric (ethos, pathos, logos), tone and audience",
        "literary analysis, themes, symbolism",
        "argumentation, counter-arguments, editing for clarity and concision",
        "technical writing, summarising complex documents",
        "stylistics, discourse analysis, translation challenges",
        "publication-quality writing, critique and revision of expert texts",
    )),
    Subject("history", "History & Society", ("mcq", "short"), (
        "past vs present, family and community history",
        "early humans, ancient civilisations (Mesopotamia, Egypt, Indus, China)",
        "classical Greece, Rome, Maurya and Han empires",
        "medieval world, major religions and trade routes",
        "renaissance, age of exploration, scientific revolution",
        "industrial revolution, colonialism, revolutions (American, French)",
        "World War I, interwar period, World War II",
        "decolonisation, the Cold War, civil rights movements",
        "globalisation, economics basics, political systems",
        "historiography, interpreting primary sources",
        "comparative history, causes of long-run development",
        "original historical argument, evaluating competing interpretations",
    )),
    Subject("cs", "Computer Science Theory", ("mcq", "short", "numeric"), (
        "what a computer is, input/output, binary on/off",
        "binary numbers, bits and bytes, simple algorithms as recipes",
        "boolean logic gates, flowcharts",
        "number bases, character encoding, how memory stores data",
        "algorithm analysis, pseudocode, computer architecture basics",
        "operating systems: processes, memory, files",
        "networking: TCP/IP, HTTP, DNS",
        "automata, regular expressions, context-free grammars",
        "Turing machines, decidability, P vs NP",
        "cryptography, information theory, compression",
        "machine learning foundations: optimisation, generalisation, neural networks",
        "transformers and LLMs, distributed computing theory, formal verification",
    )),
]}

for _key, _name, _category, _types, _milestones in parse_catalog():
    if _key in SUBJECTS:
        raise ValueError(f"duplicate subject key {_key}")
    SUBJECTS[_key] = Subject(_key, _name, _types, expand_syllabus(_milestones), _category)

CATEGORIES: list[str] = list(dict.fromkeys(s.category for s in SUBJECTS.values()))

DEFAULT_SUBJECTS = ("math", "physics", "logic", "code", "research")


def topics_for(subject: str, grade: int) -> str:
    """Topic list for a subject at a 1-based grade."""
    return SUBJECTS[subject].grades[grade - 1]


def difficulty(grade: int) -> str:
    return DIFFICULTY[grade - 1]
