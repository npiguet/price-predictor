"""Fork collection against a real game (T125).

Forks need a live game to copy, so this is where the budget arithmetic meets the
engine. Two worker runs serve the module: a stage-one run, which takes no fork
(stage three is opt-in, and taking none is correct rather than a failure), and
a forking run with interventions and damage-step probes on, played until it has
taken both kinds; each ends by the stop protocol, its game finished. Skips when
the JAR is not built.

The property that matters: **a fork discarded by the score check still counts
against its budget.** Without that, a systematically failing copy retries until
the game ends and the run spends its whole simulation budget producing nothing.
"""

from __future__ import annotations

import subprocess
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from effects.domain.collection_caps import CollectionCaps
from effects.domain.event_schema import EventType
from effects.domain.records import EffectRecord
from effects.infrastructure.shard_io import read_records
from price_predictor.infrastructure.jvm_processes import kill_process_tree
from sealed.infrastructure.match_worker_connector import (
    MatchWorkerConnector,
    WorkerLaunch,
    WorkerRecords,
    WorkerStopProtocol,
)
from tests.integration.conftest import require_jar

pytestmark = pytest.mark.integration

_COLLECTION_SECONDS = 240
_POLL_SECONDS = 5
#: How long a worker asked to stop may take to finish its game.
_STOP_SECONDS = 300
_PROBED = "first_strike,trample,deathtouch"


def pair_forks(records: list[EffectRecord]) -> list[tuple[EffectRecord, EffectRecord]]:
    """Each fork record joined to the real record its ``mirror_of`` names."""
    by_id = {record.record_id: record for record in records}
    return [
        (by_id[record.mirror_of], record)
        for record in records
        if record.fork and record.mirror_of and record.mirror_of in by_id
    ]


def _run(
    records_dir: Path, *, enough: Callable[[list], bool], interventions: int = 0,
    probe_keywords: str = "",
) -> list:
    """Collect until ``enough`` holds of the records or the window closes,
    then stop the worker at its game's end; the records it wrote."""
    require_jar()
    stop_file = records_dir.parent / "worker.stop"
    process = MatchWorkerConnector().start(launch=WorkerLaunch(
        None,
        run_id=str(uuid.uuid4()),
        best_of=1,
        records=WorkerRecords(
            directory=records_dir,
            worker_index=0,
            collection_caps=CollectionCaps(
                interventions_per_game=interventions,
                probes_per_game=4,
                probe_keywords=probe_keywords,
            ).as_system_properties(),
        ),
        stop=WorkerStopProtocol(stop_file=stop_file),
    ))
    try:
        deadline = time.monotonic() + _COLLECTION_SECONDS
        while time.monotonic() < deadline:
            time.sleep(_POLL_SECONDS)
            if process.poll() is not None:
                pytest.fail(f"worker exited early with {process.returncode}")
            if enough(list(read_records(records_dir))):
                break
        stop_file.touch()
        assert process.wait(timeout=_STOP_SECONDS) == 0, "the worker did not stop cleanly"
    except subprocess.TimeoutExpired:
        pytest.fail(f"the worker did not stop within {_STOP_SECONDS} s of the request")
    finally:
        kill_process_tree(process)
    return list(read_records(records_dir))


def _probes(records: list) -> list:
    return [r for r in records if r.fork and not r.interventional]


@pytest.fixture(scope="module")
def stage_one(tmp_path_factory) -> list:
    """A run with no fork budget: the first turns of a game are enough."""
    root = tmp_path_factory.mktemp("stage-one")
    records = _run(root / "records", enough=lambda so_far: len(so_far) >= 20)
    if not records:
        pytest.skip("worker produced no records within the collection window")
    return records


@pytest.fixture(scope="module")
def forked(tmp_path_factory) -> list:
    """A run taking interventions and probes, until it has both: a probe
    needs a combat with a probed keyword, several turns in."""
    root = tmp_path_factory.mktemp("forked")
    return _run(
        root / "records", interventions=2, probe_keywords=_PROBED,
        enough=lambda so_far: any(r.interventional for r in so_far)
        and bool(_probes(so_far)),
    )


def test_a_stage_one_run_takes_no_forks(stage_one) -> None:
    """Correct rather than a failure: stage three is opt-in."""
    records = stage_one
    assert not any(record.fork for record in records)
    assert not any(record.interventional for record in records)


def test_a_fork_record_carries_the_forks_own_state(forked) -> None:
    records = forked
    forks = [r for r in records if r.fork]
    assert forks, "the forking run took no fork within the collection window"
    for fork in forks:
        # The intervention changed the board; recording the original would
        # describe a situation the resolution never saw.
        assert fork.state.entities or fork.state.players

    # A forced resolution writes its effect half alone; a protection fork,
    # played on to the turn's end (FR-058), links its cost halves as a live
    # game does, so a linked interventional record finds its partner there.
    interventional = [f for f in forks if f.interventional]
    linked = {f.link_id for f in interventional if f.link_id is not None}
    for link_id in linked:
        halves = [f for f in interventional if f.link_id == link_id]
        assert len(halves) >= 2, f"interventional half {link_id} has no partner"


def test_forks_pair_to_the_records_they_mirror(forked) -> None:
    records = forked
    pairs = pair_forks(records)
    if not pairs:
        pytest.skip("no probe landed within the collection window")
    for real, fork in pairs:
        assert fork.mirror_of == real.record_id
        assert fork.game_id == real.game_id, "a fork mirrors a same-game record"


def _forced_resolutions(forks: list[EffectRecord]) -> list[EffectRecord]:
    """The unlinked resolution records among ``forks``."""
    return [f for f in forks if f.kind.value == "resolution" and f.link_id is None]


def test_an_intervention_records_what_the_ability_did(forked) -> None:
    """The failure this catches is a fork that copies, score-checks, snapshots
    and writes an empty payload — a record saying an ability was forced without
    saying what it did, which looks like a working collector.

    Needs the budget switched on, which is what makes this the only test in the
    file that takes a fork on purpose.
    """
    records = forked
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
    # A protection fork's records are of every kind and link their halves; the
    # forced resolutions are the unlinked resolution records.
    for record in _forced_resolutions(with_events):
        assert record.ability, "an intervention names the line it forced"


def test_an_intervention_forces_something_observation_misses(forked) -> None:
    """Lands and mana abilities are excluded on purpose: every game plays them,
    so forking to force one spends a game copy to observe the commonest event in
    the corpus.

    A forced resolution is an unlinked interventional resolution, but so is a
    protection fork's mana record: that fork plays its turn on and taps lands as
    a live game does. The two are alike on the record, so mana records are left
    to the connector's ``InterventionSchedulerTest`` and this checks the rest.
    """
    records = forked
    forks = [
        r for r in _forced_resolutions([r for r in records if r.interventional and r.ability])
        if not _only_mana(r)
    ]
    if not forks:
        pytest.skip("no intervention landed within the collection window")
    for record in forks:
        script = record.ability[0].script_file
        assert not script.endswith(
            ("/mountain.txt", "/island.txt", "/swamp.txt", "/plains.txt",
             "/forest.txt")
        ), f"forced a basic land: {script}"


def _only_mana(record: EffectRecord) -> bool:
    """A mana record: a ``mana_produced``, and otherwise only the ``tapped``
    events of what its cost tapped (FR-042c)."""
    events = getattr(record.payload, "events", ())
    return any(event.type is EventType.MANA_PRODUCED for event in events) and all(
        event.type in (EventType.MANA_PRODUCED, EventType.TAPPED) for event in events
    )



def test_a_probe_pairs_to_the_combat_it_varies(forked) -> None:
    """A branch that pairs to nothing is unreadable: the whole point is the
    difference between it and the step it forked from."""
    records = forked
    probes = _probes(records)
    if not probes:
        pytest.skip("no probe landed within the collection window")

    paired = {fork.record_id for _real, fork in pair_forks(records)}
    for probe in probes:
        assert probe.record_id in paired, (
            f"probe {probe.record_id} mirrors {probe.mirror_of}, which is not "
            "in the corpus"
        )


def test_a_probe_names_what_it_perturbed(forked) -> None:
    """Keyword and carrier both. A board with two tramplers gives the keyword
    alone two readings, and an evaluator reproducing the perturbation
    model-side would strip the wrong creature."""
    records = forked
    probes = _probes(records)
    if not probes:
        pytest.skip("no probe landed within the collection window")

    for probe in probes:
        assert probe.payload.probed_keyword in _PROBED.split(",")
        assert probe.payload.probed_entity, "no carrier named"


def _carrier_of(probe: EffectRecord):
    """The entity the probe stripped, or None when its own snapshot lacks it."""
    return next(
        (e for e in probe.state.entities if e.id == probe.payload.probed_entity),
        None,
    )


def _assert_keyword_stripped(probe: EffectRecord, carrier) -> None:
    assert carrier.lines is not None, (
        f"{probe.record_id}: {carrier.name} was not written in the gen-3 shape"
    )
    # Both places a keyword shows: its keyword line, and the overlay's
    # list of every keyword the card has.
    lines = {line.keyword for line in carrier.keyword_lines if line.in_force}
    overlay = set(carrier.granted_temporary.keywords)
    assert probe.payload.probed_keyword not in lines | overlay, (
        f"{probe.payload.probed_keyword} survived the strip on "
        f"{carrier.name}"
    )


def test_the_probed_creature_actually_lost_the_keyword(forked) -> None:
    """The failure this catches is a strip that silently does nothing: the
    branch then equals the real step, and the counterfactual says the keyword
    was worth nothing."""
    records = forked
    probes = _probes(records)
    if not probes:
        pytest.skip("no probe landed within the collection window")

    carriers = [(probe, _carrier_of(probe)) for probe in probes]
    checked = 0
    for probe, carrier in carriers:
        if carrier is None:
            continue
        _assert_keyword_stripped(probe, carrier)
        checked += 1
    assert checked, "no probe's carrier appeared in its own snapshot"


def test_an_observed_combat_record_perturbs_nothing(forked) -> None:
    records = forked
    observed = [
        r for r in records
        if r.kind.value == "combat" and not r.fork
    ]
    if not observed:
        pytest.skip("worker produced no combat records within the window")
    for record in observed:
        assert record.payload.probed_keyword is None
        assert record.payload.probed_entity is None


def test_no_record_carries_a_precomputed_diff(stage_one, forked) -> None:
    """FR-042: the real-versus-fork difference is an evaluation-time
    diagnostic, never something the corpus stores and a trainer could read."""
    for record in stage_one + forked:
        assert "diff" not in record.extra_fields
        assert "real_vs_fork" not in record.extra_fields
