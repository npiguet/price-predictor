"""A segment the root grants adds nothing to the root's amounts (task G13d, finding 1.10).

Lines are real gen-3 sidecar lines of the fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

from effects.domain.value_targets import VALUE_TARGET_NAMES, ValueLine

_SIDECARS = Path(__file__).parents[3] / "fixtures" / "effects" / "gen3-sidecars" / "cardsfolder"


def _stated(card: str, position: int) -> dict[str, float | None]:
    path = _SIDECARS / f"{card}.provenance.json"
    found = json.loads(path.read_text(encoding="utf-8"))["lines"][position]
    targets = ValueLine("", found["script_text"], False).targets
    return {
        name: value if stated else None
        for name, value, stated in zip(VALUE_TARGET_NAMES, targets.values, targets.mask)
    }


def test_a_backup_trigger_that_grants_a_draw_ability_states_no_cards_drawn():
    """The trigger puts a counter and grants ``AB$ Draw`` (SV3) to the creature."""
    stated = _stated("s/scorn_blade_berserker", 0)
    assert stated["cards_drawn"] == 0
    assert stated["counters_placed"] == 1


def test_the_granted_ability_line_states_its_own_draw():
    assert _stated("s/scorn_blade_berserker", 1)["cards_drawn"] == 1
