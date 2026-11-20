"""The random-seat pilot against a real Forge (T067, SC-001a).

One worker plays ``match-outcomes``' own matches with every match seating a
random player (``--random-seat-share 1 --random-seat-probability 0.5``) and
effect records on, the way the supervisor starts it, and the test reads what it
left behind:

- no row in ``match-outcomes.txt`` or ``cards-played.txt``, though both writers
  are wired and matches completed (FR-025);
- ``random_seat`` on every record right after ``synthetic``, and varying;
  ``what_if`` right after it on blockers records and nowhere else, a gen-3
  attackers record carrying none (FR-030, FR-031);
- a real attack declaration for every combat, and no legality record that is
  not a real declaration (FR-029);
- one effect half per chosen mode of a resolved charm, each through its
  ``option`` line, and no charm effect half through the bare root (FR-029a,
  SC-001a). On an unpatched checkout the rule is the opposite one, a single
  root half (FR-029e), and that is what is asserted there;
- the worker stopped by the stop protocol (``contracts/cli.md`` § Match worker
  stop protocol): asked through its stop file, it finished the game it was
  playing and exited 0 with the stop line, so every game in its shard ended
  with its game-end marker and no charm's halves are cut short.

The decks are mono-red charms rather than sealed pools, because a pool holds a
charm in a few games out of a hundred and a pilot this short would assert the
modal rule over nothing. Fiery Confluence chooses three with repeats, Abrade and
Fiery Intervention choose one, and Ornithopter gives their destroy modes a
target. Skips when the JAR is not built, like the other collection tests.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from collections import defaultdict
from pathlib import Path

import pytest

from effects.domain.collection_caps import CollectionCaps
from effects.domain.records import Moment, PlayabilitySubkind, RecordKind, TurnActionSubkind
from effects.infrastructure.deck_file import write_deck_file
from effects.infrastructure.record_codec import record_from_dict
from effects.infrastructure.shard_io import ended_games, iter_shard_lines, iter_shards
from price_predictor.infrastructure.forge_classpath import resolve_connector_jar
from price_predictor.infrastructure.jvm_processes import kill_process_tree
from price_predictor.infrastructure.worker_stops import STOPPED_LINE
from sealed.infrastructure.match_worker_connector import (
    MatchWorkerConnector,
    WorkerDecks,
    WorkerLaunch,
    WorkerRecords,
    WorkerStopProtocol,
)

pytestmark = pytest.mark.integration

#: A cold JVM, Forge's card database and a few best-of-one games.
_COLLECTION_SECONDS = 900
_POLL_SECONDS = 10
#: Matches to finish before the outcome files say anything.
_MIN_MATCHES = 3
#: How long the worker, asked to stop, may take to finish the game it is
#: playing: a supervisor's ``--stop-timeout`` default.
_STOP_SECONDS = 600

#: Each charm's script, and how many modes it chooses.
_CHARMS = {
    "fiery_confluence.txt": 3,
    "abrade.txt": 1,
    "fiery_intervention.txt": 1,
}

_DECK = (
    ["Mountain"] * 17
    + ["Fiery Confluence"] * 4
    + ["Abrade"] * 4
    + ["Fiery Intervention"] * 3
    + ["Ornithopter"] * 4
    + ["Goblin Piker"] * 4
    + ["Hill Giant"] * 4
)


def _charm_of(key) -> str | None:
    name = key.script_file.rsplit("/", 1)[-1]
    return name if name in _CHARMS else None


def _read(records_dir: Path) -> list[tuple[list[dict], list]]:
    """Every shard's raw JSON lines and parsed records, in write order."""
    shards = []
    for shard in iter_shards(records_dir):
        raw = [json.loads(line) for line in iter_shard_lines(shard)]
        shards.append((raw, [record_from_dict(item) for item in raw]))
    return shards


def _charm_resolved(shards) -> bool:
    for _, records in shards:
        for record in records:
            if (
                record.kind is RecordKind.RESOLUTION
                and record.moment is Moment.RESOLUTION
                and record.link_id
                and record.ability
                and _charm_of(record.ability[0]) == "fiery_confluence.txt"
            ):
                return True
    return False


def _run_pilot(tmp_path: Path) -> tuple[Path, Path, Path, list, int, str]:
    try:
        resolve_connector_jar()
    except FileNotFoundError:
        pytest.skip("forge-connector JAR not built (mvn install -DskipTests)")

    decks = tmp_path / "decks.txt"
    write_deck_file([_DECK], decks, label="pilot")
    outcomes = tmp_path / "match-outcomes.txt"
    progress = tmp_path / "progress.txt"
    records_dir = tmp_path / "records"
    caps = CollectionCaps(
        random_seat_share=1.0,
        random_seat_probability=0.5,
    )
    stop_file = tmp_path / "worker.stop"
    log_path = tmp_path / "worker.log"
    with open(log_path, "wb") as log:
        process = MatchWorkerConnector().start(
            launch=WorkerLaunch(
                outcomes,
                run_id=str(uuid.uuid4()),
                best_of=1,
                decks=WorkerDecks(side_a=decks, side_b=decks, decks_only=True),
                records=WorkerRecords(
                    directory=records_dir, worker_index=0,
                    collection_caps=caps.as_system_properties(),
                ),
                stop=WorkerStopProtocol(stop_file=stop_file),
                progress_file=progress,
            ),
            log_file=log,
        )
        try:
            deadline = time.monotonic() + _COLLECTION_SECONDS
            while time.monotonic() < deadline:
                time.sleep(_POLL_SECONDS)
                if process.poll() is not None:
                    pytest.fail(f"worker exited early with {process.returncode}")
                matches = (
                    len(progress.read_text(encoding="utf-8").splitlines())
                    if progress.exists() else 0
                )
                if matches >= _MIN_MATCHES and _charm_resolved(_read(records_dir)):
                    break
            # The stop protocol: the worker finishes its game and exits 0.
            stop_file.touch()
            code = process.wait(timeout=_STOP_SECONDS)
        except subprocess.TimeoutExpired:
            pytest.fail(f"the worker did not stop within {_STOP_SECONDS} s of the request")
        finally:
            kill_process_tree(process)
    printed = log_path.read_text(encoding="utf-8", errors="replace")
    return outcomes, progress, records_dir, _read(records_dir), code, printed


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    return _run_pilot(tmp_path_factory.mktemp("random-seat-pilot"))


def test_the_worker_stops_at_a_game_end_when_asked(pilot) -> None:
    *_, code, printed = pilot
    assert code == 0, printed[-4000:]
    line = STOPPED_LINE.search(printed)
    assert line is not None, printed[-4000:]
    assert line.group("reason").startswith("stop file "), line.group(0)
    assert int(line.group("games")) >= 1, line.group(0)


def test_every_game_in_the_shards_ends_with_its_marker(pilot) -> None:
    records_dir, shards = pilot[2], pilot[3]
    if not shards:
        pytest.skip("worker produced no records within the collection window")
    for shard, (_, records) in zip(iter_shards(records_dir), shards, strict=True):
        assert {record.game_id for record in records} == ended_games(shard), (
            f"{shard.name}: a game without its marker"
        )
        last = {record.game_id: record for record in records}
        for game, record in last.items():
            assert record.subkind is TurnActionSubkind.GAME_END, (
                f"{game}'s last record is not its marker: {record.record_id}"
            )
            assert record.mode is records[0].mode, record.record_id
            assert record.state is not None and record.state.players, record.record_id


def test_random_seat_matches_write_no_sealed_row(pilot) -> None:
    outcomes, progress, *_ = pilot
    if not progress.exists():
        pytest.skip("worker completed no match within the collection window")
    assert progress.read_text(encoding="utf-8").splitlines(), "no match finished"
    cards_played = outcomes.parent / "cards-played.txt"
    for path in (outcomes, cards_played):
        assert not path.exists() or not path.read_text(encoding="utf-8").strip(), (
            f"a random-seat match wrote a row to {path.name}"
        )


def test_both_envelope_fields_are_present_in_order(pilot) -> None:
    shards = pilot[3]
    raw = [item for lines, _ in shards for item in lines]
    if not raw:
        pytest.skip("worker produced no records within the collection window")

    # Gen-3 attackers records carry no ``what_if``: blockers alone keep it.
    with_what_if = {PlayabilitySubkind.BLOCKERS.value}
    for item in raw:
        keys = list(item)
        assert "random_seat" in item, item["record_id"]
        assert keys.index("random_seat") == keys.index("synthetic") + 1, keys
        if item.get("subkind") in with_what_if:
            assert keys.index("what_if") == keys.index("random_seat") + 1, keys
        else:
            assert "what_if" not in item, item["record_id"]

    assert {item["random_seat"] for item in raw} == {True, False}
    # The engine's declaration hook reports real declarations only.
    assert {item["what_if"] for item in raw if "what_if" in item} <= {False}


def test_random_seat_follows_the_actor(pilot) -> None:
    """FR-030: within one game, ``random_seat`` is exactly "the actor is the seat"."""
    shards = pilot[3]
    records = [r for _, rs in shards for r in rs]
    seats = defaultdict(set)
    for record in records:
        if record.random_seat:
            seats[record.game_id].add(record.actor_player)
    assert seats, "no record named the random seat"
    for game, players in seats.items():
        assert len(players) == 1, f"{game} has two random seats: {players}"
    for record in records:
        seat = next(iter(seats.get(record.game_id, {None})))
        if seat is not None:
            assert record.random_seat == (record.actor_player == seat), record.record_id


def test_every_real_attack_declaration_is_written(pilot) -> None:
    """FR-029: every real declaration is written, and nothing else is.

    A combat record means attackers were declared that turn, and the engine's
    declaration hook writes a real ``attackers`` record before the combat it
    opens, whichever seat declared. The hook reports declarations only, so no
    legality record is a what-if; a gen-3 attackers record carries no
    ``what_if`` at all, and only a blockers record could say so.
    """
    shards = pilot[3]
    records = [r for _, rs in shards for r in rs]
    real = {
        (r.game_id, r.state.global_.turn)
        for r in records
        if r.subkind is PlayabilitySubkind.ATTACKERS and not r.fork
    }
    combats = {
        (r.game_id, r.state.global_.turn)
        for r in records
        if r.kind is RecordKind.COMBAT and not r.fork
    }
    if not combats:
        pytest.skip("no combat in the collection window")
    missing = sorted(combats - real)
    assert not missing, f"combats with no real attackers record: {missing[:10]}"
    what_ifs = sum(
        r.what_if is True for r in records
        if r.subkind in (PlayabilitySubkind.ATTACKERS, PlayabilitySubkind.BLOCKERS)
    )
    assert not what_ifs, f"{what_ifs} legality records are not real declarations"


def test_a_resolved_charm_has_one_effect_half_per_chosen_mode(pilot) -> None:
    """FR-029a and SC-001a; FR-029e on an unpatched checkout.

    The worker stopped at a game end, so every game is on disk whole
    (``test_every_game_in_the_shards_ends_with_its_marker``): no charm has
    halves cut short, and every one is counted.
    """
    shards = pilot[3]
    if not shards:
        pytest.skip("worker produced no records within the collection window")
    patched = {r.mode.value for _, rs in shards for r in rs} == {"patched"}

    halves: dict[str, list] = defaultdict(list)
    charm_links: dict[str, str] = {}
    for _, records in shards:
        for record in records:
            if record.kind is not RecordKind.RESOLUTION or not record.ability:
                continue
            charm = _charm_of(record.ability[0])
            if charm is None:
                continue
            if record.moment is Moment.ACTIVATION and record.link_id:
                assert all(k.option is None for k in record.ability), (
                    f"a cost half acts through the root line: {record.record_id}"
                )
                charm_links[record.link_id] = charm
            elif record.moment is Moment.RESOLUTION:
                if record.interventional and record.link_id is None:
                    # A forced resolution has no cost half; by FR-029f its
                    # halves still go through the option lines. A protection
                    # fork's halves are linked as live ones (FR-058).
                    if patched:
                        assert all(k.option is not None for k in record.ability), (
                            f"a forked charm half acts through the root: {record.record_id}"
                        )
                    continue
                halves[record.link_id].append(record)

    resolved = {link: charm for link, charm in charm_links.items() if halves.get(link)}
    assert resolved, "no charm resolved in the pilot"
    full_three = False
    for link, charm in resolved.items():
        modal = halves[link]
        if not patched:
            assert len(modal) == 1, f"degraded {charm} wrote {len(modal)} halves"
            assert all(k.option is None for k in modal[0].ability)
            continue
        for half in modal:
            assert all(k.option is not None for k in half.ability), (
                f"a {charm} effect half acts through a key without option: "
                f"{half.record_id}"
            )
        assert 1 <= len(modal) <= _CHARMS[charm], (
            f"{charm} chooses {_CHARMS[charm]} modes; wrote {len(modal)} halves"
        )
        full_three |= charm == "fiery_confluence.txt" and len(modal) == 3
    if patched:
        assert full_three, "no Fiery Confluence resolved with all three of its modes"
