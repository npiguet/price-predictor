"""The reported checks the specs require and no code builds (FR-110, FR-116).

Each is printed ``[todo]`` on every run, naming what it would measure. Never a
skip: the corpus holds the records these read, so a skip would send an
operator after data that is already there.
"""

from __future__ import annotations

from effects.domain.check_label import CheckLabel
from effects.domain.evaluation_report import CheckName, CheckResult, CheckStatus

_TAXONOMY = CheckName.baseline_comparison("taxonomy")
_NO_STATE = CheckName.baseline_comparison("no-state")

NOT_IMPLEMENTED_CHECKS: dict[str, str] = {
    CheckName.MATCHED_REAL_VS_FORK: (
        "it would compare each fork's observed outcome with the real record "
        "it mirrors"
    ),
    CheckName.PROBE_DIFF: (
        "it would re-run gate 2's canary over real and probe-fork combat "
        "pairs joined by mirror_of, for probed keywords only"
    ),
    CheckName.ROLE_POLARITY: (
        "it would compare the predicted mana-pool sign for {R} in cost "
        "position against effect position"
    ),
    CheckName.SCALING_CALIBRATION: (
        "it would score the predicted sweeper deaths as a function of board size"
    ),
    CheckName.IDENTITY_GAME_DISJOINT: (
        "it would score the identity variant on the game-disjoint split, the "
        "in-distribution memorization ceiling"
    ),
    CheckName.STATE_ONLY_FLOOR: (
        "it would score the state-only variant on every record kind, the floor "
        "each kind's metrics are read against"
    ),
    str(CheckLabel.of(_TAXONOMY, "held-out-prediction")): (
        "it would compare the taxonomy variant with the full encoding on "
        "held-out effect prediction"
    ),
    str(CheckLabel.of(_TAXONOMY, "decodability")): (
        "it would compare the taxonomy variant with the full encoding on the "
        "decodability battery"
    ),
    str(CheckLabel.of(_NO_STATE, "decodability")): (
        "it would compare the no-state variant with the full encoding on the "
        "decodability battery"
    ),
    str(CheckLabel.of(_NO_STATE, CheckName.WARD_CANARY)): (
        "it would run the ward canary over the no-state variant's cache"
    ),
    str(CheckLabel.of(_NO_STATE, CheckName.SCALING_CALIBRATION)): (
        "it would compare the no-state variant with the full encoding on "
        "scaling calibration"
    ),
    CheckName.SHARED_TEXT_STRATUM: (
        "it would score the held-out stratum of texts that also appear on "
        "training cards"
    ),
    CheckName.NOVEL_COMBINATION_STRATUM: (
        "it would score the held-out stratum of novel combinations of seen "
        "sub-abilities"
    ),
    CheckName.NUMERIC_EXTRAPOLATION_STRATUM: (
        "it would score the held-out stratum of a seen effect at an unseen "
        "magnitude"
    ),
}


def not_implemented(name: str) -> CheckResult:
    return CheckResult(
        name, CheckStatus.NOT_IMPLEMENTED,
        f"not implemented; {NOT_IMPLEMENTED_CHECKS[name]}",
    )
