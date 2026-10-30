"""Coverage matches never touch the sealed corpora (T103).

Coverage decks are built for coverage rather than as a fair self-play sample, so
a coverage run that appended to ``match-outcomes.txt`` would corrupt what the
scorer trains on — and the corruption would be invisible, because the rows would
look exactly like ordinary self-play rows.

The check is that a records-only worker never creates either file, rather than
that a pre-existing one keeps its bytes: the worker is never told a sealed
path, so a byte comparison of files it cannot see would pass whatever it did.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest

from effects.infrastructure.deck_file import write_deck_file
from effects.infrastructure.shard_io import read_records
from price_predictor.infrastructure.forge_jvm import kill_process_tree
from sealed.infrastructure.match_worker_connector import MatchWorkerConnector
from tests.integration.conftest import POLL_SECONDS, require_jar, run_worker

pytestmark = pytest.mark.integration

#: Short of the default: a records-only worker plays an ordinary sealed game.
_COLLECTION_SECONDS = 180


def _run_records_only(records_dir: Path) -> int:
    """Run a records-only worker until it writes something, then stop it."""
    return len(run_worker(records_dir, seconds=_COLLECTION_SECONDS))


def test_a_records_only_worker_creates_no_sealed_file_at_all(
    tmp_path: Path,
) -> None:
    """Not "created and left empty" — never constructed."""
    require_jar()
    _run_records_only(tmp_path / "records")

    for name in ("match-outcomes.txt", "cards-played.txt"):
        assert not (tmp_path / name).exists()
        assert not list(tmp_path.rglob(name))


def test_the_records_it_writes_are_still_well_formed(tmp_path: Path) -> None:
    """Writing no sealed corpus must not mean writing a broken one."""
    require_jar()
    records_dir = tmp_path / "records"
    if _run_records_only(records_dir) == 0:
        pytest.skip("worker produced no records within the collection window")

    records = list(read_records(records_dir))
    assert records
    assert all(record.game_id for record in records)
    assert all(record.mode.value in ("degraded", "patched") for record in records)


def _deck_of(
    name: str, count: int = 23, lands: tuple[str, ...] = ("Forest",) * 17,
) -> list[str]:
    return [name] * count + list(lands)


#: A mirror of mana-dork-only decks has nothing to press an advantage with,
#: so any one Bo1 game between them can run long -- Forge's AI is in no hurry
#: to trade 1/1s and the game only otherwise ends by decking out ~30 turns in.
#: A lone worker would then make the progress-file assertion hostage to
#: whichever single game it happened to draw. Running several workers against
#: the one shared progress file -- exactly what ``CollectorSupervisor`` does
#: in production, just at a smaller scale -- means only the *fastest* of
#: several games has to finish inside the window. ``ProgressWriter`` is
#: documented to support concurrent appends from exactly this kind of pool.
_WORKER_COUNT = 4


def test_a_named_deck_reaches_the_records(tmp_path: Path) -> None:
    """The whole point of the collector, against a real Forge.

    A coverage deck names the cards it is built to reach. If the worker plays
    that deck, those names appear in effect records; if it falls back to
    sealed self-play -- which is what it did before this plan -- they do not,
    and the run looks identical while collecting something else entirely.

    Two decks, not one: ``pickDeckB`` excludes a deck that is an exact
    content mirror of deck A before accepting one at all. A single deck
    repeated would make every match hit that exclusion and, at the time this
    test was first written, fall through to Forge's own set-based deck
    building -- and a coverage deck's set code is the ``COVERAGE`` sentinel,
    which resolves against no real Forge edition, so every match threw
    ``NullPointerException`` before a game was ever played, forever (a real
    hang: nothing ever reached the progress file, so a bounded round never
    saw its bound). Fixed in ``MatchGenerator.pickDeckB`` to accept a mirror
    instead of reaching Forge under ``decksOnly`` -- see
    ``GeneratedDecksIndex.randomAnyDeckFromSet`` and the Java tests covering
    it (``MatchGeneratorRoutingTest``, ``MatchGeneratorTest``). This test's
    second deck still differs by one basic land, so the fix's own mirror path
    is never what makes this particular test pass -- it exercises the
    ordinary non-mirror branch, the same one every real coverage/variant
    round mostly uses.

    ``Llanowar Elves`` because it is in the converted tree, castable by the
    AI, and does nothing that needs another card present.

    Also closes a gap Task 4 left open: its progress counter
    (``-Dsealed.progress.file``, written per completed match by
    ``ProgressWriter``) was previously only exercised by a unit test calling
    ``recordMatch`` directly. This is the first test to drive real workers'
    ``runForever`` loops end to end and check that the progress file they
    share actually grows.

    On the broken-evidence run this test's own history relies on (an empty
    decks file, since Step 2 predates Tasks 1-6 and no longer fails): that
    proves the harness fails loudly when the worker exits early on a bad
    decks file. It does **not** by itself discriminate the narrower claim in
    this docstring's first paragraph -- that a worker silently ignoring the
    decks file and falling back to ordinary sealed self-play would leave this
    card out. Checked directly rather than assumed: passing ``decksOnly=False``
    against this exact fixture does **not** produce that failure either,
    because ``pickDeckA`` samples from ``side_a_decks_path`` whenever it is
    set, independent of ``decksOnly`` -- deck A still came from the file and
    ``Llanowar Elves`` still reliably appeared within 21s. The regression the
    first paragraph actually describes -- no decks file reaching the worker
    at all -- is what ``_run_records_only`` above already exercises; it never
    sees this card because it never sees any *named* card, only whatever an
    unweighted sealed pool happens to contain.
    """
    require_jar()
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    decks_file = tmp_path / "decks.txt"
    write_deck_file(
        [
            _deck_of("Llanowar Elves"),
            _deck_of("Llanowar Elves", lands=("Forest",) * 16 + ("Plains",)),
        ],
        decks_file,
        label="coverage",
    )
    progress_file = tmp_path / "progress.txt"
    run_id = str(uuid.uuid4())

    processes = [
        MatchWorkerConnector().start(
            None,
            run_id=run_id,
            best_of=1,
            effect_records_dir=records_dir,
            worker_index=worker_index,
            side_a_decks_path=decks_file,
            side_b_decks_path=decks_file,
            decks_only=True,
            progress_file=progress_file,
        )
        for worker_index in range(_WORKER_COUNT)
    ]
    try:
        deadline = time.monotonic() + _COLLECTION_SECONDS
        names: set[str] = set()
        progress_lines = 0
        while time.monotonic() < deadline:
            time.sleep(POLL_SECONDS)
            for process in processes:
                if process.poll() is not None:
                    pytest.fail(f"worker exited early with {process.returncode}")
            names = {
                entity.name
                for record in read_records(records_dir)
                for entity in record.state.entities
            }
            progress_lines = (
                sum(1 for _ in progress_file.open(encoding="utf-8"))
                if progress_file.exists() else 0
            )
            if "Llanowar Elves" in names and progress_lines >= 1:
                break
    finally:
        for process in processes:
            kill_process_tree(process)

    assert "Llanowar Elves" in names, (
        "the decked card never reached a record: the worker is not playing "
        f"the decks file. Saw {len(names)} distinct names."
    )
    assert progress_lines >= 1, (
        "the progress file gained no lines even though a match was awaited: "
        "runForever's loop is not reaching ProgressWriter.write() for every "
        "completed match."
    )
