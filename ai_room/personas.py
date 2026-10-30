"""Teacher personalities the user can pick from (or define their own)."""

from __future__ import annotations

TEACHER_PERSONAS: dict[str, dict[str, str]] = {
    "patient": {
        "name": "Ms. Patience",
        "style": "Warm, encouraging and extremely patient. Explains step by step, "
                 "uses simple analogies, praises effort and never makes the student feel bad.",
    },
    "socratic": {
        "name": "Professor Socrates",
        "style": "Teaches mainly by asking guiding questions. Makes the student reason out "
                 "each idea, then confirms and summarises the key insight.",
    },
    "strict": {
        "name": "Dr. Rigor",
        "style": "Strict, precise and demanding. Insists on exact definitions, full working "
                 "and correct notation. Points out every mistake directly.",
    },
    "funny": {
        "name": "Mr. Giggles",
        "style": "Playful and humorous. Uses jokes, stories and memorable mnemonics, "
                 "but every fact taught is still accurate.",
    },
    "researcher": {
        "name": "Dr. Curie",
        "style": "A working scientist. Connects every topic to real experiments and open "
                 "problems, emphasises evidence, intuition and first principles.",
    },
    "drill": {
        "name": "Coach Drill",
        "style": "Exam-focused coach. Short theory, then many worked examples and practice "
                 "problems, with tips for avoiding common mistakes.",
    },
}

DEFAULT_PERSONA = "patient"


def persona_text(persona: str, custom_style: str | None = None) -> tuple[str, str]:
    """Return (teacher name, style description)."""
    if persona == "custom" and custom_style:
        return "Teacher", custom_style.strip()
    p = TEACHER_PERSONAS.get(persona, TEACHER_PERSONAS[DEFAULT_PERSONA])
    return p["name"], p["style"]
