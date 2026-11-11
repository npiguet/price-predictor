"""Collection against a patched checkout (T104).

Skips on stock Forge, which is the ordinary case and not a failure: stage one is
defined as the patch-free stage, and the worker is required to degrade rather
than fail. What this test asserts is the other half — that when the hooks *are*
present, the records say so and the four kinds the patch unlocks appear.

Running it is how an operator confirms a re-applied patch actually took, which
matters because the patch lives in a sibling checkout that is rebuilt
independently and silently reverts on every Forge upgrade.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from effects.infrastructure.shard_io import read_records
from price_predictor.infrastructure.forge_jvm import kill_process_tree
from sealed.infrastructure.match_worker_connector import MatchWorkerConnector
from tests.integration.conftest import require_jar

pytestmark = pytest.mark.integration

_COLLECTION_SECONDS = 240
_POLL_SECONDS = 5

#: The kinds only the patch makes reachable.
_PATCH_UNLOCKED = {"rewrite", "continuous", "trigger", "playability"}


def _require_patched(
    records_dir: Path,
    wants: str | None = None,
    until: Callable[[list], bool] | None = None,
) -> list:
    """Collect until there is something to assert on, or the window closes.

    ``wants`` names a record kind the caller needs. Twenty records is the first
    turn of the first game, which is long before a static has changed anything,
    so a test about continuous records has to keep going or it asserts nothing
    and reports a skip that looks like a pass. ``until`` keeps collecting past
    that until the records hold what the caller's assertion can fail on.
    """
    require_jar()

    process = MatchWorkerConnector().start(
        None,
        run_id=str(uuid.uuid4()),
        best_of=1,
        effect_records_dir=records_dir,
        worker_index=0,
    )
    try:
        deadline = time.monotonic() + _COLLECTION_SECONDS
        while time.monotonic() < deadline:
            time.sleep(_POLL_SECONDS)
            if process.poll() is not None:
                pytest.fail(f"worker exited early with {process.returncode}")
            records = list(read_records(records_dir))
            if until is not None and not until(records):
                continue
            if wants is not None:
                if sum(r.kind.value == wants for r in records) >= 20:
                    return records
            elif len(records) >= 20:
                return records
        return list(read_records(records_dir))
    finally:
        kill_process_tree(process)


def test_a_patched_checkout_stamps_every_record_patched(tmp_path: Path) -> None:
    records = _require_patched(tmp_path / "records")
    if not records:
        pytest.skip("worker produced no records within the collection window")
    modes = {record.mode.value for record in records}
    if modes == {"degraded"}:
        pytest.skip(
            "../forge is unpatched, which is the stage-one case: check out "
            "its effect-record-hooks branch and rebuild to exercise this test"
        )
    assert modes == {"patched"}


def test_a_patched_checkout_reaches_the_four_unlocked_kinds(
    tmp_path: Path,
) -> None:
    records = _require_patched(tmp_path / "records")
    if not records:
        pytest.skip("worker produced no records within the collection window")
    if {record.mode.value for record in records} == {"degraded"}:
        pytest.skip("../forge is unpatched; the four kinds are unreachable")

    kinds = {record.kind.value for record in records}
    reached = kinds & _PATCH_UNLOCKED
    assert reached, f"a patched run reached none of {sorted(_PATCH_UNLOCKED)}; saw {sorted(kinds)}"


def _acting_static_carriers_and_grants(state, key) -> tuple[set[str], list[str]]:
    """The entities holding a static line of the acting key, and the lines it granted elsewhere.

    A line a static grants another card is keyed to that static (a keyword
    line by the granting static, its fragment naming the keyword). The
    carrier's own lines keyed alike are not grants: a Reconfigure Effect's
    static and an Unearth Effect's replacement share their keyword's key.
    """
    carriers = {
        entity.id
        for entity in state.entities
        for line in entity.lines or ()
        if line.kind == "static" and key in line.keys
    }
    granted = [
        f"{entity.name}: {line.kind} {line.fragment or ''}".strip()
        for entity in state.entities
        if entity.id not in carriers
        for line in entity.lines or ()
        if key in line.keys and line.granted and line.in_force
    ]
    return carriers, granted


def _grants_a_keyword_elsewhere(record) -> bool:
    """A continuous record whose static, one card alone holding it, grants another card a keyword.

    Only such a record can fail the line check: a static that grants no line
    to another card leaves none behind to find, suppressed or not.
    """
    if record.kind.value != "continuous":
        return False
    for key in record.ability:
        carriers, _ = _acting_static_carriers_and_grants(record.state, key)
        if len(carriers) == 1 and any(
            contribution.keywords and contribution.entity not in carriers
            for contribution in record.payload.contributions
        ):
            return True
    return False


def test_a_continuous_snapshot_does_not_contain_the_effect_it_labels(
    tmp_path: Path,
) -> None:
    """The premise of the kind: the board is the one the static acted on.

    A snapshot that still carries the static's own work hands the model the
    answer along with the question: a Royal Role's record would show the ward
    it grants and ask for ward as the label. What suppression guarantees is
    read off the board's provenance, not its values. No in-force line the
    acting static granted remains on any card, and no static line lists what
    it affects, in its ``affected`` or in an ``affects`` link. The board is
    read where the acting static's line is held by one card alone.

    Values cannot show it. The board is recomputed with the static's layer
    entries dropped, so a keyword, type or boost the card has from another
    source stays: Wildblood Pack's own trample under Full Moon's Rise, an equal
    +1/+1 from a resolved pump. Boosts, types and colours are sums over their
    sources in the record and name none, so the connector's own tests check
    those channels (``SnapshotCharacterizationTest``). A static whose line a
    second card holds too (two Royal Roles, two tokens of one Dollhouse of
    Horrors) is passed over: the twin's grants carry the same key.
    """
    from effects.domain.records import RecordKind
    from effects.domain.state_snapshot import LineRef

    records = _require_patched(
        tmp_path / "records",
        wants="continuous",
        until=lambda so_far: any(map(_grants_a_keyword_elsewhere, so_far)),
    )
    if not records:
        pytest.skip("worker produced no records within the collection window")
    if {record.mode.value for record in records} == {"degraded"}:
        pytest.skip("../forge is unpatched; continuous records are unreachable")

    continuous = [r for r in records if r.kind is RecordKind.CONTINUOUS]
    if not continuous:
        pytest.skip("no continuous records in the collection window")

    leaks: list[str] = []
    for record in continuous:
        state = record.state
        for entity in state.entities:
            assert entity.lines is not None, (
                f"{record.record_id}: {entity.name} was not written in the gen-3 shape"
            )
            leaks.extend(
                f"{record.record_id}: {entity.name} lists what its {line.kind} line affects"
                for line in entity.lines
                if line.affected is not None
            )
        leaks.extend(
            f"{record.record_id}: an affects link from {link.from_.entity}'s line"
            for link in state.links or ()
            if link.type == "affects" and isinstance(link.from_, LineRef)
        )
        for key in record.ability:
            carriers, granted = _acting_static_carriers_and_grants(state, key)
            if len(carriers) != 1:
                continue
            leaks.extend(f"{record.record_id}: {grant} granted by {key}" for grant in granted)

    if not any(map(_grants_a_keyword_elsewhere, continuous)):
        pytest.skip(
            "no continuous record whose static, one card alone holding it, grants a keyword"
        )
    assert not leaks, (
        f"{len(leaks)} traces of the acting static on the board it acted on; "
        f"suppression is not being applied: {leaks[:5]}"
    )


def test_no_field_listed_as_constant_has_quietly_started_carrying_data(
    tmp_path: Path,
) -> None:
    """The ratchet on the constant-field inventory.

    Asserted in one direction only. Observing a value is evidence that a field
    is wired; *not* observing one is not evidence that it is unwired, because a
    short collection window is also how a rare field looks. So this fails when
    a listed field carries something -- meaning it was implemented and the list
    is stale -- and never fails for a field that merely went unexercised.

    The other direction is the operator's: ``field_coverage`` over a full
    corpus prints every constant field, and one that is not on the list is
    either newly broken or newly rare.
    """
    from effects.application.field_coverage import (
        KNOWN_CONSTANT_FIELDS,
        field_coverage,
    )

    records = _require_patched(tmp_path / "records")
    if not records:
        pytest.skip("worker produced no records within the collection window")

    coverage = field_coverage(records)
    carrying = sorted(
        path for path in KNOWN_CONSTANT_FIELDS if path in coverage and not coverage[path].constant
    )
    assert not carrying, (
        "these fields are listed in KNOWN_CONSTANT_FIELDS but now carry data; "
        f"remove them from the list: {carrying}"
    )


def test_a_patched_checkout_collects_snapshot_tier_three(tmp_path: Path) -> None:
    """Unreferenced stack contents, which stage one does not collect."""
    records = _require_patched(tmp_path / "records")
    if not records:
        pytest.skip("worker produced no records within the collection window")
    if {record.mode.value for record in records} == {"degraded"}:
        pytest.skip("../forge is unpatched; tier 3 is a stage-two tier")

    from effects.domain.state_snapshot import InclusionTier

    assert any(InclusionTier.UNREFERENCED_STACK in record.state.tiers for record in records)
