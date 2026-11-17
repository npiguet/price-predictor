"""Random-seat counterspells: stack-targeting scripts the seat cast, countered or not."""

from __future__ import annotations

import re

from .model import TallyCheck, entities_of, events_of, first_key, is_effect_half, state_of
from .sidecars import Sidecars

STACK_APIS = re.compile(
    r"\b(?:SP|AB|DB)\$ (?:Counter|CopySpellAbility|ChangeTargets|ControlSpell)\b"
)

TARGET_KINDS = ("stack-card-target", "spell-target")


def target_kind(record: dict) -> str:
    """``stack-card-target``: a target that is a stack card (the random seat's draw) while
    ``targets_on_stack`` names no spell; else ``spell-target``."""
    refs = state_of(record).get("refs") or {}
    zones = {e.get("id"): e.get("zone") for e in entities_of(record)}
    targets = refs.get("targets_at_cast") or refs.get("targets") or []
    on_stack_card = any(zones.get(t) == "stack" for t in targets)
    if on_stack_card and not refs.get("targets_on_stack"):
        return "stack-card-target"
    return "spell-target"


def _is_random_seat_live_half(record: dict) -> bool:
    if not is_effect_half(record) or record.get("fork"):
        return False
    return bool(record.get("random_seat"))


class CounterHalves(TallyCheck):
    """Random-seat effect halves of stack-targeting scripts, with and without `spell_countered`."""

    NAMES = tuple(
        f"counter-halves-random-seat-{outcome}[{target}]"
        for target in TARGET_KINDS
        for outcome in ("countered", "not-countered")
    )

    def __init__(self, sidecars: Sidecars) -> None:
        super().__init__()
        self.sidecars = sidecars

    def feed(self, record: dict) -> None:
        if not _is_random_seat_live_half(record):
            return
        if self._script_targets_the_stack(first_key(record)):
            countered = any(e.get("type") == "spell_countered" for e in events_of(record))
            target = target_kind(record)
            self.add(f"counter-halves-random-seat-countered[{target}]", countered)
            self.add(f"counter-halves-random-seat-not-countered[{target}]", not countered)

    def _script_targets_the_stack(self, key: dict) -> bool:
        """Whether any line of the acting card's script holds a stack-targeting API, as the
        review's scan of the card scripts selected them."""
        data = self.sidecars.get(key["script_file"]) if key else None
        return data is not None and any(
            STACK_APIS.search(line.get("script_text") or "") for line in data["lines"]
        )
