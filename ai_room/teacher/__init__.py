"""The non-AI teacher: exact problem generators, a textbook library and rule-based lesson presentation."""

from __future__ import annotations

import importlib

# Importing a module registers its problem families.
for _m in ("gen_math", "gen_math2", "gen_science", "gen_social", "gen_logic", "gen_cs", "gen_life", "gen_lang", "code_tasks"):
    importlib.import_module(f"{__name__}.{_m}")

from .library import Library  # noqa: E402
from .style import PRESETS, TeacherStyle, style_from  # noqa: E402
from .teacher import NoCourseMaterial, Teacher  # noqa: E402

__all__ = ["Library", "NoCourseMaterial", "PRESETS", "Teacher", "TeacherStyle", "style_from"]
