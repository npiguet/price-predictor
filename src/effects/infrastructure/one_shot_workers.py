"""Spawns the Java mains that run once to completion rather than supervised.

Each initializes Forge's GUI-side static state, so each needs the converter's
full-runtime classpath, dependency glob included. Each raises
``FileNotFoundError`` when the connector JAR or java is missing.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from price_predictor.infrastructure.forge_jvm import run_forge_worker

logger = logging.getLogger(__name__)

_PACKAGE = "com.pricepredictor.connector"


def _run(main: str, main_args: list[str]) -> int:
    return run_forge_worker(
        f"{_PACKAGE}.{main}", main_args=main_args, include_dependency_glob=True,
    ).returncode


def write_keyword_definitions(output_path: Path) -> int:
    """Write Forge's keyword table to ``output_path``; return the exit code."""
    return _run("KeywordDefinitionMain", ["--output", str(output_path)])


def write_variant_sidecars(variants_path: Path) -> int:
    """Write a sidecar beside every script in ``variants_path``; return the exit code.

    From Java because a key's ``index_within_kind`` is the trait's position in
    Forge's runtime trait list; a Python numbering would be a second parser
    that drifts silently, since the join fails loudly only on a missing key.
    """
    return _run("VariantSidecarMain", ["--variants-path", str(variants_path)])


def castability_verdicts(names: list[str]) -> dict[str, str]:
    """``card name -> "castable" | "uncastable"`` for every name it judged.

    The verdict only ranks, never drops a card (an uncastable card is exactly
    the one an intervention has to force into play). A name left unanswered is
    absent, which the caller ranks as castable, the safe direction.
    """
    with tempfile.TemporaryDirectory(prefix="effects-consult-") as scratch:
        request = Path(scratch) / "cards.txt"
        response = Path(scratch) / "verdicts.json"
        request.write_text("\n".join(names), encoding="utf-8")
        code = _run(
            "CastabilityMain", ["--cards", str(request), "--output", str(response)],
        )
        if code != 0 or not response.exists():
            logger.warning("CastabilityMain exited with %d; no verdicts", code)
            return {}
        return json.loads(response.read_text(encoding="utf-8"))
