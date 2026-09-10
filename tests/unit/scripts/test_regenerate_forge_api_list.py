"""Focused tests for scripts/regenerate_forge_api_list.py's parsing helpers.

``scripts/`` is not part of the installed package (only ``src/`` is, per
pyproject.toml), so the module under test is loaded by file path rather than
imported by name.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "regenerate_forge_api_list.py"
)
_spec = importlib.util.spec_from_file_location("regenerate_forge_api_list", _SCRIPT_PATH)
_regen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_regen)


class TestClassStemPairs:
    """``class_stem_pairs`` feeds the rename table R3 required be derived
    from Forge's own source rather than hand-typed, so the one property it
    must not depend on is the enum's own ordering -- ordering is Forge's to
    change, not this generator's to rely on.
    """

    def test_the_self_consistent_member_wins_even_when_listed_last(self, tmp_path):
        """Four members share one class, the way ApiType's Charm family does
        (``Charm``, ``CompanionChoose``, ``InternalLegendaryRule``,
        ``InternalIgnoreEffect`` all name ``CharmEffect``), but here the
        self-naming member is listed *last* -- the ordering ApiType.java does
        not currently have, but nothing prevents a future edit from
        producing. Before the ordering-independent fix, first-seen-wins made
        this resolve to ``CompanionChoose``: a wrong rename suggestion for
        the one artifact R3 required be mechanical rather than hand-typed.
        """
        java_file = tmp_path / "ApiType.java"
        java_file.write_text(
            "public enum ApiType {\n"
            "    CompanionChoose (CharmEffect.class),\n"
            "    InternalLegendaryRule (CharmEffect.class),\n"
            "    InternalIgnoreEffect (CharmEffect.class),\n"
            "    Charm (CharmEffect.class),\n"
            "    ;\n",
            encoding="utf-8",
        )

        pairs = _regen.class_stem_pairs(java_file)

        assert pairs["Charm"] == "Charm"

    def test_first_seen_wins_when_no_member_names_the_stem(self, tmp_path):
        """No member of this shared class happens to be named after the
        stem, so there is no self-consistent choice available -- first-seen
        is the documented, deliberate fallback for that case, not an
        accident of implementation.
        """
        java_file = tmp_path / "ApiType.java"
        java_file.write_text(
            "public enum ApiType {\n"
            "    Alpha (SharedEffect.class),\n"
            "    Beta (SharedEffect.class),\n"
            "    ;\n",
            encoding="utf-8",
        )

        pairs = _regen.class_stem_pairs(java_file)

        assert pairs["Shared"] == "Alpha"
