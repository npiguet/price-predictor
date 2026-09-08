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

import subprocess
import time
import uuid
from pathlib import Path

import pytest

from effects.infrastructure.record_io import read_records
from price_predictor.infrastructure.forge_jvm import resolve_connector_jar
from sealed.infrastructure.match_worker_connector import MatchWorkerConnector

pytestmark = pytest.mark.integration

_COLLECTION_SECONDS = 240
_POLL_SECONDS = 5

#: The kinds only the patch makes reachable.
_PATCH_UNLOCKED = {"rewrite", "continuous", "trigger", "playability"}


def _require_patched(records_dir: Path, wants: str | None = None) -> list:
    """Collect until there is something to assert on, or the window closes.

    ``wants`` names a record kind the caller needs. Twenty records is the first
    turn of the first game, which is long before a static has changed anything,
    so a test about continuous records has to keep going or it asserts nothing
    and reports a skip that looks like a pass.
    """
    try:
        resolve_connector_jar()
    except FileNotFoundError:
        pytest.skip("forge-connector JAR not built (mvn package -DskipTests)")

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
            if wants is not None:
                if sum(r.kind.value == wants for r in records) >= 20:
                    return records
            elif len(records) >= 20:
                return records
        return list(read_records(records_dir))
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def test_a_patched_checkout_stamps_every_record_patched(tmp_path: Path) -> None:
    records = _require_patched(tmp_path / "records")
    if not records:
        pytest.skip("worker produced no records within the collection window")
    modes = {record.mode.value for record in records}
    if modes == {"degraded"}:
        pytest.skip(
            "../forge is unpatched, which is the stage-one case: apply "
            "forge-connector/patches/ and rebuild to exercise this test"
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
    assert reached, (
        f"a patched run reached none of {sorted(_PATCH_UNLOCKED)}; saw {sorted(kinds)}"
    )


#: How much of a static's own work may still show in the board it acted on.
#: Unsuppressed this is 1.0 for the type, colour and keyword channels and 0.85
#: for boosts; suppressed it is a few percent, all of it the overlap below.
_MAX_VISIBLE_CONTRIBUTIONS = 0.25


def test_a_continuous_snapshot_does_not_contain_the_effect_it_labels(
    tmp_path: Path,
) -> None:
    """The premise of the kind: the board is the one the static acted on.

    A snapshot that still carries the static's own work hands the model the
    answer along with the question — an anthem's record would show the +1/+1 in
    the boosts and ask for +1/+1 as the label. Every channel the static wrote
    has to come back out.

    A rate rather than a count, because some contributions genuinely survive.
    The board is recomputed with the static's layer entries dropped, not
    stripped of the tokens it contributed, so a card printed with the keyword
    or type its own static grants keeps it: Wildblood Pack has trample of its
    own when Full Moon's Rise is removed, and Circle of the Moon Druid is still
    a creature. Those are the boards the statics really acted on, and a test
    demanding zero would be demanding a board that never existed.
    """
    from effects.domain.records import RecordKind

    records = _require_patched(tmp_path / "records", wants="continuous")
    if not records:
        pytest.skip("worker produced no records within the collection window")
    if {record.mode.value for record in records} == {"degraded"}:
        pytest.skip("../forge is unpatched; continuous records are unreachable")

    continuous = [r for r in records if r.kind is RecordKind.CONTINUOUS]
    if not continuous:
        pytest.skip("no continuous records in the collection window")

    visible = 0
    total = 0
    for record in continuous:
        entities = {e.id: e for e in record.state.entities}
        for contribution in record.payload.contributions:
            entity = entities.get(contribution.entity)
            if entity is None:
                continue

            if contribution.pt_boost != (0, 0):
                total += 1
                boosts = tuple(entity.pt.boosts) if entity.pt else (0, 0)
                visible += boosts == tuple(contribution.pt_boost)

            shown = {
                k.strip().lower().replace(" ", "_")
                for k in entity.granted_temporary.keywords
            }
            for keyword in contribution.keywords:
                total += 1
                visible += keyword.strip().lower().replace(" ", "_") in shown

            shown = {
                t.lower()
                for t in (*entity.types, *entity.subtypes, *entity.supertypes)
            } | {c.upper() for c in entity.colors}
            for token in (*contribution.types, *contribution.colors):
                # A removal or an overwrite is not invertible from the token,
                # so only additions say anything about suppression.
                if token.startswith(("-", "=")) or token.startswith("all-"):
                    continue
                total += 1
                visible += token.lower() in shown or token.upper() in shown

    if not total:
        pytest.skip("no contributions in the collection window")
    assert visible / total < _MAX_VISIBLE_CONTRIBUTIONS, (
        f"{visible} of {total} contributions are still visible in the board "
        f"their own static acted on; suppression is not being applied"
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
        path for path in KNOWN_CONSTANT_FIELDS
        if path in coverage and not coverage[path].constant
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

    assert any(
        record.state.collected(InclusionTier.UNREFERENCED_STACK)
        for record in records
    )
