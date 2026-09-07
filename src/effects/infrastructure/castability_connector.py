"""Spawns the Java ``CastabilityMain`` — the coverage collector's consult.

One connector per Java main, the convention every worker in this repo follows.
This one needs a live Forge because "can this card be cast at all" is a question
about the engine's own legality rules, not about the card's text.

The consult **only ranks**. Its verdict never drops a card from deck building,
because being in a game is the precondition a stage-three intervention forks
from — a card the consult calls uncastable is exactly the one an intervention
has to force into play.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

from price_predictor.infrastructure.forge_jvm import (
    build_forge_classpath,
    build_jvm_command,
)

logger = logging.getLogger(__name__)

_MAIN_CLASS = "com.pricepredictor.connector.CastabilityMain"


class CastabilityConnector:
    """Runs ``CastabilityMain`` over a card list and reads its verdicts."""

    def consult(
        self, names: list[str], cards_folder: Path,
    ) -> dict[str, str]:
        """``card name -> "castable" | "uncastable"`` for every name it judged.

        A name the worker does not answer for is simply absent, and the caller
        treats that as ``unknown`` — which ranks it as castable, the safe
        direction, since an under-ranked card only takes longer to cover.
        """
        with tempfile.TemporaryDirectory(prefix="effects-consult-") as scratch:
            request = Path(scratch) / "cards.txt"
            response = Path(scratch) / "verdicts.json"
            request.write_text("\n".join(names), encoding="utf-8")

            cmd = build_jvm_command(
                main_class=_MAIN_CLASS,
                classpath=build_forge_classpath(
                    include_full_runtime=True, include_dependency_glob=True,
                ),
                main_args=[
                    "--cards", str(request),
                    "--output", str(response),
                    "--cards-path", str(cards_folder),
                ],
            )
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0 or not response.exists():
                logger.warning(
                    "CastabilityMain exited with %d; no verdicts",
                    result.returncode,
                )
                return {}
            return json.loads(response.read_text(encoding="utf-8"))
