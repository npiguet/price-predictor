"""The random-seat pilot against a real Forge (T067, SC-001a).

One worker plays ``match-outcomes``' own matches with every match seating a
random player (``--random-seat-share 1 --random-seat-probability 0.5``) and
effect records on, the way the supervisor starts it, and the test reads what it
left behind:

- no row in ``match-outcomes.txt`` or ``cards-played.txt``, though both writers
  are wired and matches completed (FR-025);
- ``random_seat`` on every record right after ``synthetic``, ``what_if`` right
  after it on the two legality subkinds and nowhere else, and both varying
  (FR-030, FR-031);
- a real attack declaration for every combat, at a ``--legality-rate`` that
  would keep one in fifty of them if real decisions were sampled (FR-029);
- one effect half per chosen mode of a resolved charm, each through its
  ``option`` line, and no charm effect half through the bare root (FR-029a,
  SC-001a). On an unpatched checkout the rule is the opposite one, a single
  root half (FR-029e), and that is what is asserted there.

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
from effects.domain.records import Moment, PlayabilitySubkind, RecordKind
from effects.infrastructure.deck_file import COVERAGE_SET_CODE, write_deck_file
from effects.infrastructure.record_io import (
    iter_shard_lines,
    iter_shards,
    record_from_dict,
)
from price_predictor.infrastructure.forge_jvm import resolve_connector_jar
from sealed.infrastructure.match_worker_connector import MatchWorkerConnector

pytestmark = pytest.mark.integration

#: A cold JVM, Forge's card database and a few best-of-one games.
_COLLECTION_SECONDS = 900
_POLL_SECONDS = 10
#: Matches to finish before the outcome files say anything.
_MIN_MATCHES = 3
#: Low enough that a sampled real decision would show: one combat in fifty.
_LEGALITY_RATE = 0.02
#: A shard is written in blocks of this many records, and a killed worker
#: truncates the block in flight; a modal resolution whose halves reach into
#: the last complete block may have lost the rest to the truncated one.
_BLOCK = 256

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
        raw = [json.loads(line) for line in iter_shard_lines(shard) if line.strip()]
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


def _run_pilot(tmp_path: Path) -> tuple[Path, Path, Path, list]:
    try:
        resolve_connector_jar()
    except FileNotFoundError:
        pytest.skip("forge-connector JAR not built (mvn install -DskipTests)")

    decks = tmp_path / "decks.txt"
    write_deck_file([_DECK], decks, label="pilot", set_code=COVERAGE_SET_CODE)
    outcomes = tmp_path / "match-outcomes.txt"
    progress = tmp_path / "progress.txt"
    records_dir = tmp_path / "records"
    caps = CollectionCaps(
        legality_rate=_LEGALITY_RATE,
        random_seat_share=1.0,
        random_seat_probability=0.5,
    )
    with open(tmp_path / "worker.log", "wb") as log:
        process = MatchWorkerConnector().start(
            outcomes,
            run_id=str(uuid.uuid4()),
            best_of=1,
            log_file=log,
            side_a_decks_path=decks,
            side_b_decks_path=decks,
            decks_only=True,
            effect_records_dir=records_dir,
            worker_index=0,
            collection_caps=caps.as_system_properties(),
            progress_file=progress,
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
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
    return outcomes, progress, records_dir, _read(records_dir)


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    return _run_pilot(tmp_path_factory.mktemp("random-seat-pilot"))


def test_random_seat_matches_write_no_sealed_row(pilot) -> None:
    outcomes, progress, _, _ = pilot
    if not progress.exists():
        pytest.skip("worker completed no match within the collection window")
    assert progress.read_text(encoding="utf-8").splitlines(), "no match finished"
    cards_played = outcomes.parent / "cards-played.txt"
    for path in (outcomes, cards_played):
        assert not path.exists() or not path.read_text(encoding="utf-8").strip(), (
            f"a random-seat match wrote a row to {path.name}"
        )


def test_both_envelope_fields_are_present_in_order_and_vary(pilot) -> None:
    *_, shards = pilot
    raw = [item for lines, _ in shards for item in lines]
    if not raw:
        pytest.skip("worker produced no records within the collection window")

    legality = {s.value for s in (PlayabilitySubkind.ATTACKERS, PlayabilitySubkind.BLOCKERS)}
    for item in raw:
        keys = list(item)
        assert "random_seat" in item, item["record_id"]
        assert keys.index("random_seat") == keys.index("synthetic") + 1, keys
        if item.get("subkind") in legality:
            assert keys.index("what_if") == keys.index("random_seat") + 1, keys
        else:
            assert "what_if" not in item, item["record_id"]

    assert {item["random_seat"] for item in raw} == {True, False}
    assert {item["what_if"] for item in raw if "what_if" in item} == {True, False}


def test_random_seat_follows_the_actor(pilot) -> None:
    """FR-030: within one game, ``random_seat`` is exactly "the actor is the seat"."""
    *_, shards = pilot
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
    """FR-029: the rate samples what-ifs only.

    A combat record means attackers were declared that turn, by the AI's hook
    or by the random seat itself, and either writes a real ``attackers``
    record before the combat it opens. Sampled at 2%, nearly every combat
    would have none.
    """
    *_, shards = pilot
    records = [r for _, rs in shards for r in rs]
    real = {
        (r.game_id, r.state.global_.turn)
        for r in records
        if r.subkind is PlayabilitySubkind.ATTACKERS and r.what_if is False
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
    sampled = sum(
        r.what_if is True for r in records
        if r.subkind in (PlayabilitySubkind.ATTACKERS, PlayabilitySubkind.BLOCKERS)
    )
    assert sampled, "what-ifs at 2% should still leave some"


def test_a_resolved_charm_has_one_effect_half_per_chosen_mode(pilot) -> None:
    """FR-029a and SC-001a; FR-029e on an unpatched checkout."""
    *_, shards = pilot
    if not shards:
        pytest.skip("worker produced no records within the collection window")
    patched = {r.mode.value for _, rs in shards for r in rs} == {"patched"}

    halves: dict[str, list] = defaultdict(list)
    cut_short: set[str] = set()
    charm_links: dict[str, str] = {}
    for _, records in shards:
        tail = len(records) - _BLOCK
        for index, record in enumerate(records):
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
                if record.interventional:
                    # A forced resolution has no cost half; by FR-029f its
                    # halves still go through the option lines.
                    if patched:
                        assert all(k.option is not None for k in record.ability), (
                            f"a forked charm half acts through the root: {record.record_id}"
                        )
                    continue
                halves[record.link_id].append(record)
                if index >= tail:
                    cut_short.add(record.link_id)

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
        if link not in cut_short:
            assert 1 <= len(modal) <= _CHARMS[charm], (
                f"{charm} chooses {_CHARMS[charm]} modes; wrote {len(modal)} halves"
            )
            full_three |= charm == "fiery_confluence.txt" and len(modal) == 3
    if patched:
        assert full_three, "no Fiery Confluence resolved with all three of its modes"
