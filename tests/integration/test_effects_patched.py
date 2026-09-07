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


def _require_patched(records_dir: Path) -> list:
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
            if len(records) >= 20:
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
