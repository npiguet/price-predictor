"""The coverage collector: a round plays the computed decks, ends in
its report, and a Ctrl-C ends the run."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from effects.application.collect_coverage import CollectCoverageConfig
from tests.unit.effects.application.collect_coverage_support import _convert_acting_line
from tests.unit.effects.domain.test_coverage import _record
from tests.unit.effects.file_builders import PlainShard, write_plain_shard
from tests.unit.effects.record_builders import make_entity


def _one_card_run(
    tmp_path: Path, *, target_records: int, no_progress_rounds: int,
) -> CollectCoverageConfig:
    """A run over a ``cardsfolder`` under ``tmp_path`` holding the one card
    ``aa``, two decks a round on one worker, its records in ``records``."""
    cards = tmp_path / "cardsfolder"
    cards.mkdir()
    (cards / "aa.txt").write_text(
        "name: aa\nmana cost: {G}\ntypes: creature\n", encoding="utf-8",
    )
    return CollectCoverageConfig(
        effect_records=tmp_path / "records",
        cards_folders=(cards,),
        target_records=target_records,
        decks_per_round=2,
        no_progress_rounds=no_progress_rounds,
        workers=1,
    )


class TestTheRoundPlaysTheComputedDecks:
    """The weights have to reach the table, which is what never happened.

    `play_round` received correctly-computed weights and discarded them, so the
    workers played ordinary sealed self-play: a slower duplicate of
    `match-outcomes` that reaches none of the cards coverage exists for.
    """

    def test_the_decks_played_are_built_from_the_weights(self, tmp_path, fake_supervisor):
        from effects.application import collect_coverage

        cards = tmp_path / "cardsfolder"
        (cards / "a").mkdir(parents=True)
        (cards / "a" / "aa.txt").write_text(
            "name: aa\nmana cost: {G}\ntypes: creature\n", encoding="utf-8",
        )
        (cards / "a" / "bb.txt").write_text(
            "name: bb\nmana cost: {G}\ntypes: creature\n", encoding="utf-8",
        )
        supervisor = fake_supervisor

        # Two candidates, and a recount that plays out over three rounds, so
        # the test can tell "decks track the weights `run` recomputes each
        # round" apart from "decks track a fixture that happens to have one
        # candidate" -- a single-card pool gives `rng.choices` no real choice
        # to make, and a single round never exercises the recompute at all.
        # Round 1 sees no qualifying records yet, so "aa" and "bb" still
        # weigh the same -- uninteresting, and that round's decks are
        # overwritten before the test ever reads them. Round 2's recount
        # reports "aa" at 9 of a target of 10 (nearly satisfied) while "bb"
        # stays at 0, so round 3 computes sharply different weights (aa
        # shortfall 1, bb shortfall 10). Round 3 is also the *last* round:
        # its own recount (still "aa": 10, "bb" untouched) satisfies "aa"
        # and is "bb"'s third consecutive stalled round, so both are done
        # and `run` returns without overwriting round 3's decks -- which are
        # exactly the ones this test reads.
        # Each round's play writes the shard that round produced, which is how
        # the real counter learns anything: it reads shards it has not seen and
        # adds to a running total. Round 1 produces nothing, round 2 nine "aa"
        # records, round 3 the tenth.

        records_dir = tmp_path / "records"
        records_dir.mkdir()
        produced = iter([0, 9, 1])
        # The records act through a converted line, or build-corpus's refusal
        # rule, which the counts apply, refuses them as no-acting-text.
        _convert_acting_line(cards)

        def _play(_decks, _decks_file, spec):
            count = next(produced, 0)
            if not count:
                return
            record = _record([make_entity("E1", "aa")], source="E1")
            write_plain_shard(records_dir, f"{spec.matches}{count}", [record] * count)

        supervisor.play_round.side_effect = _play

        with patch.object(collect_coverage, "consult_castability", return_value={}):
            collect_coverage.run(
                collect_coverage.CollectCoverageConfig(
                    effect_records=tmp_path / "records",
                    cards_folders=(cards,),
                    target_records=10,
                    decks_per_round=150,
                    workers=1,
                )
            )

        decks = supervisor.play_round.call_args.args[0]
        assert decks, "no decks were played"
        assert all(len(deck) == 40 for deck in decks)
        assert supervisor.play_round.call_args.args[2].matches == 150

        flat = [name for deck in decks for name in deck]
        assert flat.count("bb") > flat.count("aa") * 3, (
            "bb is 10 records short of target and aa is only 1 short after "
            "the round-2 recount, so the final round's decks must favour bb "
            "heavily -- a 'deck whatever is in texts, ignore the weights "
            "argument' implementation would split them roughly evenly "
            "instead and fail this assertion"
        )


class TestARoundEndsInItsReport:
    """A round ends in its report, at the level the operator sees.

    ``run``'s ``finally`` calls ``CollectorSupervisor.stop()``, which called a
    pool method that does not exist, so every round ended in
    ``AttributeError``: ``residues(...).render(...)`` (FR-051) never printed
    and ``run`` never returned its code. Every other test in this file mocks
    the supervisor, and a ``MagicMock`` answers to any method name at all --
    so none of them could see it. This one mocks neither the supervisor nor
    the pool, stubbing only ``ForgeWorkerPool.run``, the blocking loop that
    would otherwise spawn real JVMs.
    """

    def _one_card_config(self, tmp_path):
        config = _one_card_run(tmp_path, target_records=1, no_progress_rounds=1)
        (tmp_path / "records").mkdir(exist_ok=True)
        return config

    def _run(self, tmp_path, caplog, pool_run):
        import logging

        from effects.application import collect_coverage
        from price_predictor.infrastructure.forge_worker_pool import ForgeWorkerPool

        config = self._one_card_config(tmp_path)
        with patch.object(ForgeWorkerPool, "run", pool_run), \
             patch.object(collect_coverage, "consult_castability", return_value={}), \
             caplog.at_level(logging.INFO):
            return collect_coverage.run(config)

    def test_the_run_finishes_reports_and_returns_zero(self, tmp_path, caplog):
        played = []

        def pool_run(pool):
            played.append(pool)
            record = _record([make_entity("E1", "aa")], source="E1")
            run_id = pool._spawn_worker.func.__self__.run_id
            write_plain_shard(
                tmp_path / "records", f"r{len(played)}", [record], PlainShard(run_id=run_id),
            )

        code = self._run(tmp_path, caplog, pool_run)

        assert played, "no round was ever played"
        assert code == 0
        assert any("Residues:" in record.getMessage() for record in caplog.records), (
            "the run ended without printing the two residues FR-051 requires"
        )

    @pytest.mark.parametrize("ended", [True, False])
    def test_a_round_whose_last_game_was_cut_credits_nothing_and_says_so(
        self, tmp_path, caplog, ended,
    ):
        """A worker killed mid-game leaves a game without its
        game-end marker, which build-corpus drops whole; counted, it would
        satisfy the card with records the corpus never holds. The round
        report and the run's manifest name the cut game."""
        import json

        from price_predictor.infrastructure.run_manifest import RunCommand, run_recorded

        def pool_run(pool):
            # Kept by build-corpus's refusal rule only when its line converts.
            _convert_acting_line(tmp_path / "cardsfolder")
            record = _record([make_entity("E1", "aa")], source="E1")
            run_id = pool._spawn_worker.func.__self__.run_id
            write_plain_shard(tmp_path / "records", "r1", [record],
                              PlainShard(run_id=run_id, ended=ended))

        code = run_recorded(
            tmp_path / "records", lambda: self._run(tmp_path, caplog, pool_run),
            RunCommand("python -m effects collect-coverage"),
        )

        entry = json.loads(
            (tmp_path / "records" / "run.json").read_text(encoding="utf-8"))["invocations"][0]
        short = 0 if ended else 1
        assert f"{short} castable cards short of 1 records" in caplog.text
        assert entry["cut_games"] == {"before_rounds": 0, "by_round": [short], "total": short}
        assert f"Round 1: {short} cut game(s) (no game-end marker)" in caplog.text
        assert code == 0, "the cut game's records are still records the run wrote"

    def test_a_run_that_wrote_no_record_fails_and_says_where_to_look(self, tmp_path, caplog):
        """Every match failing (a broken deck, a dead worker) played its
        rounds, collected nothing, retired every card and exited 0."""
        played = []

        def pool_run(pool):
            played.append(pool)
            supervisor = pool._spawn_worker.func.__self__
            log = supervisor._worker_logs.path_for(0)
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "ab") as handle:
                handle.write(b"Error generating match: deck cannot be materialized: x\n")

        code = self._run(tmp_path, caplog, pool_run)

        assert played
        assert code == 1
        assert "no new effect record" in caplog.text
        assert "1 unresolved_deck" in caplog.text
        assert any("Residues:" in record.getMessage() for record in caplog.records)


class TestCtrlCEndsTheRun:
    """Nothing in ``effects`` used to read
    ``ForgeWorkerPool.interrupted``, so ``run`` built a fresh pool for the
    next round and Ctrl-C degraded to "skip this round, start another".

    The ``fake_supervisor`` is ``spec``-ed on purpose: a bare ``MagicMock``
    answers to ``interrupted`` whether or not the supervisor exposes it, which
    is the same blindness ``TestARoundEndsInItsReport`` exists for.
    """

    def _two_round_config(self, tmp_path):
        return _one_card_run(tmp_path, target_records=50, no_progress_rounds=5)

    def _run_with(self, tmp_path, supervisor, *, interrupted: bool):
        from effects.application import collect_coverage

        supervisor.interrupted = interrupted
        with patch.object(collect_coverage, "consult_castability", return_value={}):
            code = collect_coverage.run(self._two_round_config(tmp_path))
        return code, supervisor

    def test_an_interrupted_round_is_the_last_round(self, tmp_path, fake_supervisor):
        code, supervisor = self._run_with(tmp_path, fake_supervisor, interrupted=True)

        assert supervisor.play_round.call_count == 1, (
            "a second round was started after the operator interrupted the "
            "first one"
        )
        assert code == 130, "an interrupted run must not report success"

    def test_an_uninterrupted_run_plays_every_round_it_needs(self, tmp_path, fake_supervisor):
        """The control: no game ever shows the card, so it retires after
        ``UNSEEN_RETIRE_FACTOR`` times five written rounds, which ends the run;
        and since those rounds wrote no record, the run fails."""
        from effects.domain.coverage import UNSEEN_RETIRE_FACTOR

        code, supervisor = self._run_with(tmp_path, fake_supervisor, interrupted=False)

        assert supervisor.play_round.call_count == 5 * UNSEEN_RETIRE_FACTOR
        assert code == 1
