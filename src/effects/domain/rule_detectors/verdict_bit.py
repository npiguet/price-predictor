"""Detector family ``verdict_bit``: a playability candidate's verdict bit.

A ``decision`` record lists the candidates a player could play and, for each,
the verdict bits the rules decided. A restricting rule (a sorcery's timing,
"can't be cast", "no legal target") is on its ``fires`` side when a
candidate's bit takes the value the restriction gives it, by default false.

Parameters:

- ``bit`` (required): ``can_play``, ``affordable``, ``has_legal_target``,
  or on gen-3 records ``timing`` or ``failed_check``;
- ``check``: with ``bit = failed_check``, the ``failed_checks`` identifier;
- ``fires_when``: the bit's value on the ``fires`` side, ``false`` by default;
- ``responsible_static``: true to count only candidates a static restricts.

The entity is the verdict head's ``[ACT]``: the candidate is the acting line.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, PlayabilitySubkind, RecordKind
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register

GEN2_BITS = frozenset({"can_play", "affordable", "has_legal_target"})
GEN3_BITS = frozenset({"timing", "failed_check"})
ACT = "[ACT]"


def _bit(candidate, bit: str, check: str | None) -> bool | None:
    if bit == "failed_check":
        failed = getattr(candidate, "failed_checks", None)
        if failed is None:
            return None
        return not any(getattr(f, "check", f) == check for f in failed)
    return getattr(candidate, bit, None)


@dataclass(frozen=True, slots=True)
class VerdictBit:
    id: str = "verdict_bit"
    params: frozenset[str] = frozenset({"bit", "check", "fires_when", "responsible_static"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        bit = params.get("bit")
        if bit == "timing":
            return frozenset({"timing"})
        if bit == "failed_check":
            return frozenset({"failed_checks"})
        return frozenset()

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        bit = params.get("bit")
        if bit not in GEN2_BITS | GEN3_BITS:
            raise ValueError(f"verdict_bit: unknown bit {bit!r}")
        if bit == "failed_check" and not params.get("check"):
            raise ValueError("verdict_bit: a failed_check declaration names its check")
        if record.kind is not RecordKind.PLAYABILITY or (
                record.subkind is not PlayabilitySubkind.DECISION):
            return None
        # A failed_check "bit" reads true where the check passed, so the
        # default false is a candidate the named check failed.
        fires_when = bool(params.get("fires_when", False))
        candidates = [
            c for c in record.payload.candidates
            if not params.get("responsible_static") or c.responsible_static]
        values = [_bit(c, bit, params.get("check")) for c in candidates]
        values = [v for v in values if v is not None]
        if not values:
            return None
        fires = any(v is fires_when for v in values)
        polarity = Polarity.FIRES if fires else Polarity.DOES_NOT_FIRE
        return FamilyMatch(polarity, fires, (ACT,))


FAMILY = register(VerdictBit())
