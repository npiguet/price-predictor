"""Spawns the Java mains that run once to completion rather than supervised.

Each initializes Forge's GUI-side static state, so each needs the converter's
full-runtime classpath, dependency glob included. Each raises
``FileNotFoundError`` when the connector JAR or java is missing, and
``ForgeWorkerTimeout`` (a ``TimeoutError``) when the JVM runs past
:data:`ONE_SHOT_TIMEOUT_SECONDS` and is killed.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from price_predictor.infrastructure.forge_jvm import run_forge_worker

logger = logging.getLogger(__name__)

_PACKAGE = "com.pricepredictor.connector"

#: How long a one-shot main may run before it is killed. Each loads Forge's
#: card database once and makes one pass over at most the whole card corpus,
#: minutes at most; an hour is a hang, which nothing else would end.
ONE_SHOT_TIMEOUT_SECONDS = 3600.0


def _run(main: str, main_args: list[str]) -> int:
    return run_forge_worker(
        f"{_PACKAGE}.{main}", main_args=main_args, include_dependency_glob=True,
        timeout=ONE_SHOT_TIMEOUT_SECONDS,
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
        try:
            verdicts = json.loads(response.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("CastabilityMain's answer is unreadable (%s); no verdicts", exc)
            return {}
        if not isinstance(verdicts, dict):
            logger.warning("CastabilityMain's answer is no name-to-verdict map; no verdicts")
            return {}
        return verdicts
