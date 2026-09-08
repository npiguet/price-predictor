"""Fork collection against a real game (T125).

Forks need a live game to copy, so this is where the budget arithmetic meets the
engine. Skips when the JAR is not built or the run produces no forks — stage
three is opt-in and a stock stage-one run takes none, which is correct rather
than a failure.

The property that matters: **a fork discarded by the score check still counts
against its budget.** Without that, a systematically failing copy retries until
the game ends and the run spends its whole simulation budget producing nothing.
"""

from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path

import pytest

from effects.application.evaluate_effect_model import pair_forks
from effects.domain.collection_caps import CollectionCaps
from effects.infrastructure.record_io import read_records
from price_predictor.infrastructure.forge_jvm import resolve_connector_jar
from sealed.infrastructure.match_worker_connector import MatchWorkerConnector

pytestmark = pytest.mark.integration

_COLLECTION_SECONDS = 240
_POLL_SECONDS = 5


def _run(
    records_dir: Path, interventions: int = 0, probe_keywords: str = "",
) -> list:
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
        collection_caps=CollectionCaps(
            interventions_per_game=interventions,
            probes_per_game=4,
            probe_keywords=probe_keywords,
        ).as_system_properties(),
    )
    try:
        deadline = time.monotonic() + _COLLECTION_SECONDS
        while time.monotonic() < deadline:
            time.sleep(_POLL_SECONDS)
            if process.poll() is not None:
                pytest.fail(f"worker exited early with {process.returncode}")
            records = list(read_records(records_dir))
            # A fork run has to reach combat, which is several turns in — the
            # twenty-record mark is the first turn, long before one happens.
            if probe_keywords:
                if any(r.fork for r in records):
                    break
            elif len(records) >= 20:
                break
        return list(read_records(records_dir))
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def test_a_stage_one_run_takes_no_forks(tmp_path: Path) -> None:
    """Correct rather than a failure: stage three is opt-in."""
    records = _run(tmp_path / "records")
    if not records:
        pytest.skip("worker produced no records within the collection window")
    assert not any(record.fork for record in records)
    assert not any(record.interventional for record in records)


def test_a_fork_record_carries_the_forks_own_state(tmp_path: Path) -> None:
    records = _run(tmp_path / "records")
    forks = [r for r in records if r.fork]
    if not forks:
        pytest.skip(
            "no fork records: stage-three collection is opt-in and this run "
            "took none"
        )
    for fork in forks:
        # The intervention changed the board; recording the original would
        # describe a situation the resolution never saw.
        assert fork.state.entities or fork.state.players

    interventional = [f for f in forks if f.interventional]
    for record in interventional:
        assert record.link_id is None, "an intervention has no activation partner"


def test_forks_pair_to_the_records_they_mirror(tmp_path: Path) -> None:
    records = _run(tmp_path / "records")
    pairs = pair_forks(records)
    if not pairs:
        pytest.skip("no matched fork pairs in this run")
    for real, fork in pairs:
        assert fork.mirror_of == real.record_id
        assert fork.game_id == real.game_id, "a fork mirrors a same-game record"


def test_an_intervention_records_what_the_ability_did(tmp_path: Path) -> None:
    """The failure this catches is a fork that copies, score-checks, snapshots
    and writes an empty payload — a record saying an ability was forced without
    saying what it did, which looks like a working collector.

    Needs the budget switched on, which is what makes this the only test in the
    file that takes a fork on purpose.
    """
    records = _run(tmp_path / "records", interventions=2)
    forks = [r for r in records if r.interventional]
    if not forks:
        pytest.skip("no intervention landed within the collection window")

    with_events = [f for f in forks if getattr(f.payload, "events", ())]
    assert with_events, (
        f"{len(forks)} interventions and not one recorded an event; the fork "
        "resolves nothing"
    )
    for record in with_events:
        assert record.fork and record.interventional
        assert record.ability, "an intervention names the line it forced"
        assert record.link_id is None, "an intervention has no activation partner"


def test_an_intervention_forces_something_observation_misses(
    tmp_path: Path,
) -> None:
    """Lands and mana abilities are excluded on purpose: every game plays them,
    so forking to force one spends a game copy to observe the commonest event in
    the corpus."""
    records = _run(tmp_path / "records", interventions=2)
    forks = [r for r in records if r.interventional and r.ability]
    if not forks:
        pytest.skip("no intervention landed within the collection window")
    for record in forks:
        script = record.ability[0].script_file
        assert not script.endswith(
            ("/mountain.txt", "/island.txt", "/swamp.txt", "/plains.txt",
             "/forest.txt")
        ), f"forced a basic land: {script}"


_PROBED = "first_strike,trample,deathtouch"


def test_a_probe_pairs_to_the_combat_it_varies(tmp_path: Path) -> None:
    """A branch that pairs to nothing is unreadable: the whole point is the
    difference between it and the step it forked from."""
    records = _run(tmp_path / "records", probe_keywords=_PROBED)
    probes = [r for r in records if r.fork and not r.interventional]
    if not probes:
        pytest.skip("no probe landed within the collection window")

    paired = {fork.record_id for _real, fork in pair_forks(records)}
    for probe in probes:
        assert probe.record_id in paired, (
            f"probe {probe.record_id} mirrors {probe.mirror_of}, which is not "
            "in the corpus"
        )


def test_a_probe_names_what_it_perturbed(tmp_path: Path) -> None:
    """Keyword and carrier both. A board with two tramplers gives the keyword
    alone two readings, and an evaluator reproducing the perturbation
    model-side would strip the wrong creature."""
    records = _run(tmp_path / "records", probe_keywords=_PROBED)
    probes = [r for r in records if r.fork and not r.interventional]
    if not probes:
        pytest.skip("no probe landed within the collection window")

    for probe in probes:
        assert probe.payload.probed_keyword in _PROBED.split(",")
        assert probe.payload.probed_entity, "no carrier named"


def test_the_probed_creature_actually_lost_the_keyword(tmp_path: Path) -> None:
    """The failure this catches is a strip that silently does nothing: the
    branch then equals the real step, and the counterfactual says the keyword
    was worth nothing."""
    records = _run(tmp_path / "records", probe_keywords=_PROBED)
    probes = [r for r in records if r.fork and not r.interventional]
    if not probes:
        pytest.skip("no probe landed within the collection window")

    checked = 0
    for probe in probes:
        carrier = next(
            (e for e in probe.state.entities
             if e.id == probe.payload.probed_entity),
            None,
        )
        if carrier is None:
            continue
        checked += 1
        assert probe.payload.probed_keyword not in carrier.granted_temporary.keywords, (
            f"{probe.payload.probed_keyword} survived the strip on "
            f"{carrier.name}"
        )
    assert checked, "no probe's carrier appeared in its own snapshot"


def test_an_observed_combat_record_perturbs_nothing(tmp_path: Path) -> None:
    records = _run(tmp_path / "records", probe_keywords=_PROBED)
    observed = [
        r for r in records
        if r.kind.value == "combat" and not r.fork
    ]
    if not observed:
        pytest.skip("worker produced no combat records within the window")
    for record in observed:
        assert record.payload.probed_keyword is None
        assert record.payload.probed_entity is None


def test_no_record_carries_a_precomputed_diff(tmp_path: Path) -> None:
    """FR-042: the real-versus-fork difference is an evaluation-time
    diagnostic, never something the corpus stores and a trainer could read."""
    records = _run(tmp_path / "records")
    if not records:
        pytest.skip("worker produced no records within the collection window")
    for record in records:
        assert "diff" not in record.extra_fields
        assert "real_vs_fork" not in record.extra_fields
