"""The committed directed-fork counts are the gen-2 rule counts cut down.

``rules/directed-fork-counts.json`` is the part of a gen-2 ``rule-counts``
run's ``rule-counts.json`` that ``scripts/rules_catalogue.py directed-rules``
decides the directed-fork list from (quickstart Stage 1). It needs that run,
which ``output/`` never commits, so it is an integration test, and a missing
run fails it rather than skipping it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[2]
#: Gen-2's rule-counts runs under the artifact layout, newest last by name.
_RUNS = _REPO / "output" / "effects" / "gen2" / "inputs"


def _rules_catalogue():
    scripts = str(_REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import rules_catalogue

    return rules_catalogue


def test_the_committed_counts_are_the_newest_gen2_counts_cut_down():
    runs = sorted(_RUNS.glob("*-rule-counts/rule-counts.json"))
    if not runs:
        pytest.fail(
            f"no gen-2 rule-counts run under {_RUNS} (quickstart Stage 1 writes "
            "one): the committed counts cannot be checked against what they cut",
        )
    rc = _rules_catalogue()
    counts = json.loads(runs[-1].read_text(encoding="utf-8"))
    committed = json.loads((rc.RULES / rc.DIRECTED_COUNTS).read_text(encoding="utf-8"))
    assert committed == rc.decided_from(counts), runs[-1]
