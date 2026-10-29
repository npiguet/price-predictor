"""Spawns the Java mains that run to completion rather than being supervised.

Each reads or judges something once, with no game to play out, so it is one
blocking JVM launch. Every one of them calls
``ForgeEnvironmentInitializer.initialize()``, which reaches ``GuiBase`` /
``FModel`` in forge-gui and pulls forge-ai subsystems, so each needs the same
full-runtime classpath the converter does, dependency glob included.

Each raises ``FileNotFoundError`` if the connector JAR is missing (the error
names the build command, ``mvn install -DskipTests`` in ``forge-connector``) or
java is not on PATH.
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

    Why the sidecars come from Java at all: a provenance key's
    ``index_within_kind`` is the trait's position in Forge's own runtime trait
    list, and the collectors read that list directly. Numbering the traits in
    Python would be a second implementation of Forge's parser, and the two
    would drift silently — the join only fails loudly when a key is missing,
    not when it points at the wrong line.
    """
    return _run("VariantSidecarMain", ["--variants-path", str(variants_path)])


def castability_verdicts(names: list[str]) -> dict[str, str]:
    """``card name -> "castable" | "uncastable"`` for every name it judged.

    Needs a live Forge because "can this card be cast at all" is a question
    about the engine's own legality rules, not about the card's text.

    The consult **only ranks**. Its verdict never drops a card from deck
    building, because being in a game is the precondition a stage-three
    intervention forks from — a card the consult calls uncastable is exactly
    the one an intervention has to force into play.

    A name the worker does not answer for is simply absent, and the caller
    treats that as ``unknown`` — which ranks it as castable, the safe
    direction, since an under-ranked card only takes longer to cover.
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
