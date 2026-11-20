"""Spawns the Java mains that run once to completion rather than supervised.

Each initializes Forge's GUI-side static state, so each needs the converter's
full-runtime classpath, dependency glob included. Each raises
``FileNotFoundError`` when the connector JAR or java is missing, and
``ForgeWorkerTimeout`` (a ``TimeoutError``) when the JVM runs past its
timeout and is killed (:func:`scaled_timeout`).
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from price_predictor.infrastructure.jvm_processes import run_forge_worker

logger = logging.getLogger(__name__)

_PACKAGE = "com.pricepredictor.connector"

#: How long a one-shot main may run before it is killed, at least: the
#: keyword table's whole timeout, and the floor of the two that scale.
ONE_SHOT_TIMEOUT_SECONDS = 3600.0
#: A Forge start (the card database, the custom cards staged into it), allowed
#: once per run of a main that scales.
FORGE_STARTUP_SECONDS = 600.0
#: Allowed per script a main is handed. Unmeasured: about ten times what one
#: card's conversion or instantiation is expected to cost, so only a hang
#: reaches it.
SECONDS_PER_SCRIPT = 0.1


def scaled_timeout(scripts: int) -> float:
    """The timeout of a main that makes one pass over ``scripts`` scripts or
    names: ``FORGE_STARTUP_SECONDS + SECONDS_PER_SCRIPT × scripts``, never
    below :data:`ONE_SHOT_TIMEOUT_SECONDS`. The variant tree is append-only
    (about 32,000 scripts after one untargeted round, some 2,000 more per
    steered iteration), so a constant would one day kill a pass that was
    merely long."""
    return max(ONE_SHOT_TIMEOUT_SECONDS, FORGE_STARTUP_SECONDS + SECONDS_PER_SCRIPT * scripts)


def _run(main: str, main_args: list[str], *, timeout: float = ONE_SHOT_TIMEOUT_SECONDS) -> int:
    return run_forge_worker(
        f"{_PACKAGE}.{main}", main_args=main_args, include_dependency_glob=True,
        timeout=timeout,
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
    scripts = sum(1 for _ in Path(variants_path).rglob("*.txt"))
    return _run(
        "VariantSidecarMain", ["--variants-path", str(variants_path)],
        timeout=scaled_timeout(scripts),
    )


def castability_verdicts(names: list[str]) -> dict[str, str]:
    """``card name -> "castable" | "uncastable"`` for every name it judged.

    The verdict only ranks, never drops a card (an uncastable card is exactly
    the one an intervention has to force into play). A name left unanswered is
    absent: Forge's card database does not hold it, so no deck can name it,
    and ``collect-coverage`` leaves it out of the unit and the decks. That is
    a separate rule from the verdict. A failed consult answers nothing, and
    then every card ranks as castable, the safe direction.
    """
    with tempfile.TemporaryDirectory(prefix="effects-consult-") as scratch:
        request = Path(scratch) / "cards.txt"
        response = Path(scratch) / "verdicts.json"
        request.write_text("\n".join(names), encoding="utf-8")
        code = _run(
            "CastabilityMain", ["--cards", str(request), "--output", str(response)],
            timeout=scaled_timeout(len(names)),
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
