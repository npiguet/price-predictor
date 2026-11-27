"""The coverage collector: counting each shard once, across the
name boundary, with progress, in parallel."""

from __future__ import annotations

from pathlib import Path

from effects.domain.coverage import CardCoverage
from tests.unit.effects.application.collect_coverage_support import _holding
from tests.unit.effects.domain.test_coverage import _record
from tests.unit.effects.file_builders import PlainShard, fresh_interpreter_env, write_plain_shard
from tests.unit.effects.record_builders import make_entity


class TestIncrementalCounting:
    """Counting each shard once instead of the whole corpus every round.

    The recount re-read every shard under `--effect-records` after each round.
    Against a corpus of 1,502 shards that measured ~66 minutes per round,
    against ~35 minutes of actual play — so a coverage run spent most of its
    wall clock re-deriving numbers that had not changed.

    Safe because a round's workers are all terminated before `play_round`
    returns (`ForgeWorkerPool._supervisor_loop` calls `_terminate_all` when the
    stop condition fires), and because a shard file is never reopened: its name
    carries the worker's JVM lifetime, so each round's workers write new files.
    A counted shard is therefore final.
    """

    def _shard(self, directory: Path, lifetime: str, cards: list[str]) -> Path:
        """One shard whose every record qualifies for the named card."""
        return write_plain_shard(directory, lifetime, [
            _record([make_entity(f"E-{card}", card)], source=f"E-{card}")
            for card in cards
        ], PlainShard(run_id="a"))

    def test_the_first_pass_counts_every_shard(self, tmp_path) -> None:
        from effects.application.collect_coverage import CoverageCounter

        self._shard(tmp_path, "x", ["alpha", "beta"])
        self._shard(tmp_path, "y", ["alpha"])
        coverage = {n: CardCoverage(n) for n in ("alpha", "beta")}

        CoverageCounter().update(coverage, tmp_path, workers=1)

        assert coverage["alpha"].records == 2
        assert coverage["beta"].records == 1

    def test_a_second_pass_over_the_same_shards_adds_nothing(self, tmp_path) -> None:
        """The bug this replaces: every round re-counted the whole corpus."""
        from effects.application.collect_coverage import CoverageCounter

        self._shard(tmp_path, "x", ["alpha"])
        coverage = {"alpha": CardCoverage("alpha")}
        counter = CoverageCounter()

        counter.update(coverage, tmp_path, workers=1)
        counter.update(coverage, tmp_path, workers=1)

        assert coverage["alpha"].records == 1

    def test_a_new_shard_adds_to_the_running_total(self, tmp_path) -> None:
        from effects.application.collect_coverage import CoverageCounter

        self._shard(tmp_path, "x", ["alpha"])
        coverage = {"alpha": CardCoverage("alpha")}
        counter = CoverageCounter()
        counter.update(coverage, tmp_path, workers=1)

        self._shard(tmp_path, "y", ["alpha", "alpha"])
        counter.update(coverage, tmp_path, workers=1)

        assert coverage["alpha"].records == 3

    def _damaged(self, directory: Path, cards: list[str], *, member_after: bool) -> Path:
        """A gzip shard of one member per line, each card's record a game of
        its own ended by its marker, the second game's record's header
        broken; with ``member_after`` a member follows the broken one. The
        first game ended before the damage; the second's marker is past it."""
        import gzip
        from dataclasses import replace

        plain = write_plain_shard(directory / "plain", "x", [
            replace(_record([make_entity(f"E-{card}", card)], source=f"E-{card}"),
                    game_id=f"a.0-x.{index}", record_id=f"a.0-x.r{index}")
            for index, card in enumerate(cards)
        ], PlainShard(run_id="a"))
        members = [gzip.compress((line + "\n").encode("utf-8"))
                   for line in plain.read_text(encoding="utf-8").splitlines() if line]
        members[2] = b"XX" + members[2][2:]
        if not member_after:
            members = members[:3]
        path = directory / "a.0-z.jsonl.gz"
        path.write_bytes(b"".join(members))
        plain.unlink()
        return path

    def test_a_damaged_shard_is_counted_up_to_the_damage_once(self, tmp_path, caplog) -> None:
        """One damaged shard must not stop a collection run every round: its
        records before the damage count, it is named, and it is not read again."""
        from effects.application.collect_coverage import CoverageCounter

        shard = self._damaged(tmp_path, ["alpha", "beta", "alpha"], member_after=True)
        coverage = {n: CardCoverage(n) for n in ("alpha", "beta")}
        counter = CoverageCounter()
        with caplog.at_level("WARNING"):
            assert counter.update(coverage, tmp_path, workers=1) == 1
        assert (coverage["alpha"].records, coverage["beta"].records) == (1, 0)
        assert len(counter.damaged) == 1 and str(shard) in counter.damaged[0]
        assert any(str(shard) in r.getMessage() for r in caplog.records)
        assert counter.update(coverage, tmp_path, workers=1) == 0
        assert len(counter.damaged) == 1

    def test_a_damaged_last_member_names_no_shard(self, tmp_path) -> None:
        """Nothing follows it, so the read just ends there."""
        from effects.application.collect_coverage import CoverageCounter

        self._damaged(tmp_path, ["alpha", "beta"], member_after=False)
        coverage = {n: CardCoverage(n) for n in ("alpha", "beta")}
        counter = CoverageCounter()
        counter.update(coverage, tmp_path, workers=1)
        assert coverage["alpha"].records == 1 and counter.damaged == []

    def test_the_damaged_shards_go_to_the_run_manifest(self) -> None:
        from unittest.mock import MagicMock

        from effects.application.collector_common import record_damaged_shards

        supervisor = MagicMock()
        record_damaged_shards(supervisor, [])
        supervisor.record.assert_not_called()
        record_damaged_shards(supervisor, ["b: corrupt", "a: corrupt"])
        supervisor.record.assert_called_once_with(corrupt_shards=["a: corrupt", "b: corrupt"])

    def test_it_reports_how_many_shards_it_read(self, tmp_path) -> None:
        """The operator needs to see the round cost fall, not infer it."""
        from effects.application.collect_coverage import CoverageCounter

        self._shard(tmp_path, "x", ["alpha"])
        counter = CoverageCounter()
        coverage = {"alpha": CardCoverage("alpha")}

        assert counter.update(coverage, tmp_path, workers=1) == 1
        assert counter.update(coverage, tmp_path, workers=1) == 0


class TestCountingCrossesTheNameBoundary:
    """Coverage units are converted names; record entities are Forge's.

    `build_coverage_units` keys off `ConvertedTrees.card_files`, which reads names out of
    converted card text — lowercased, `soul echo`. A record's `EntityState.name`
    is Forge's own, `Soul Echo`. Looking one up in the other matches nothing, so
    every card counts zero records forever: nothing is ever satisfied, the
    weights never fall, and cards leave the run only by retiring.

    The same converted-tree/Forge boundary that broke depletion, the split, the
    variant skip and the holdout file. Fifth instance.
    """

    def test_a_printed_case_record_counts_toward_a_converted_case_card(
        self, tmp_path,
    ) -> None:
        from effects.application.collect_coverage import CoverageCounter

        write_plain_shard(tmp_path, "x", [_record([make_entity("E1", "Soul Echo")], source="E1")])
        coverage = {"soul echo": CardCoverage("soul echo")}

        CoverageCounter().update(coverage, tmp_path, workers=1)

        assert coverage["soul echo"].records == 1

    def test_a_card_outside_the_coverage_unit_is_still_ignored(
        self, tmp_path,
    ) -> None:
        from effects.application.collect_coverage import CoverageCounter

        write_plain_shard(tmp_path, "x", [_record([make_entity("E1", "Some Token")], source="E1")])
        coverage = {"soul echo": CardCoverage("soul echo")}

        CoverageCounter().update(coverage, tmp_path, workers=1)

        assert coverage["soul echo"].records == 0


class TestCountingReportsProgress:
    """A silent 35-minute startup is indistinguishable from a hang.

    Moving the corpus scan from per-round to once made the round loop fast and
    the startup long. Without output it looks exactly like the hang this whole
    area kept producing, which is the one shape an operator must never have to
    guess at.
    """

    def _shards(self, directory: Path, count: int) -> None:
        record = _record([make_entity("E1", "aa")], source="E1")
        for n in range(count):
            write_plain_shard(directory, f"{n:03d}", [record])

    def test_it_reports_progress_while_reading(self, tmp_path, caplog) -> None:
        from effects.application.collect_coverage import CoverageCounter

        self._shards(tmp_path, 120)
        coverage = {"aa": CardCoverage("aa")}

        with caplog.at_level("INFO"):
            CoverageCounter(progress_every=50).update(coverage, tmp_path, workers=1)

        progress = [r for r in caplog.messages if "of 120 shard" in r]
        assert len(progress) >= 2, f"expected periodic progress, got {progress}"

    def test_a_short_read_says_nothing(self, tmp_path, caplog) -> None:
        """Counting three shards should not narrate itself."""
        from effects.application.collect_coverage import CoverageCounter

        self._shards(tmp_path, 3)
        coverage = {"aa": CardCoverage("aa")}

        with caplog.at_level("INFO"):
            CoverageCounter(progress_every=50).update(coverage, tmp_path, workers=1)

        assert not [r for r in caplog.messages if "shard" in r and "of 3" in r]

    def test_the_held_out_round_reports_progress_too(self, tmp_path, caplog) -> None:
        """Its first pass reads the same whole corpus, and was silent."""
        from effects.application.coverage_counting import TextCoverageCounter
        from effects.domain.coverage import TextCoverage

        self._shards(tmp_path, 120)
        counter = TextCoverageCounter(
            {"cardsfolder": str(tmp_path / "none")}, _holding({"t"}), progress_every=50,
        )

        with caplog.at_level("INFO"):
            read = counter.update({"t": TextCoverage(text="t")}, tmp_path, workers=1)

        assert read == 120
        progress = [r for r in caplog.messages if "of 120 shard" in r]
        assert len(progress) >= 2, f"expected periodic progress, got {progress}"

    def test_a_counting_worker_never_loads_torch(self, tmp_path) -> None:
        """One pool worker per core, each loading torch's CUDA DLLs, exhausts
        the Windows paging file beside a training job — and counting a shard
        needs nothing from torch."""
        import subprocess
        import sys

        self._shards(tmp_path, 1)
        shard = next(tmp_path.iterdir())
        probe = (
            "import sys\n"
            "from pathlib import Path\n"
            "from effects.application.coverage_counting import count_shard\n"
            f"count_shard(Path({str(shard)!r}))\n"
            "print('torch' in sys.modules)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, check=True, env=fresh_interpreter_env(),
        )
        assert result.stdout.strip() == "False"


class TestParallelCounting:
    """Counting is CPU-bound per shard and the shards are independent.

    Each shard contributes a count per card and nothing else, so the work maps
    cleanly: read shards in parallel, sum the per-shard counters, apply once.
    Summation is associative and commutative, so the result cannot depend on
    how the shards were divided or in what order they finished — which is the
    property worth pinning, since a parallel count that merely *usually* agrees
    with the serial one is worse than no parallelism at all.
    """

    def _shards(self, directory: Path, spec: dict[str, int]) -> None:
        for n, (card, count) in enumerate(spec.items()):
            record = _record([make_entity("E1", card)], source="E1")
            write_plain_shard(directory, f"{n:03d}", [record] * count)

    def test_parallel_and_serial_agree(self, tmp_path) -> None:
        from effects.application.collect_coverage import CoverageCounter

        spec = {f"Card {i}": i + 1 for i in range(12)}
        self._shards(tmp_path, spec)
        expected = {f"card {i}": i + 1 for i in range(12)}

        serial = {n: CardCoverage(n) for n in expected}
        CoverageCounter().update(serial, tmp_path, workers=1)
        parallel = {n: CardCoverage(n) for n in expected}
        CoverageCounter().update(parallel, tmp_path, workers=4)

        assert {n: c.records for n, c in serial.items()} == expected
        assert {n: c.records for n, c in parallel.items()} == expected

    def test_a_card_split_across_shards_sums(self, tmp_path) -> None:
        """The reduce step, which is where a parallel count goes wrong."""
        from effects.application.collect_coverage import CoverageCounter

        record = _record([make_entity("E1", "Soul Echo")], source="E1")
        for n in range(6):
            write_plain_shard(tmp_path, str(n), [record] * 5)

        coverage = {"soul echo": CardCoverage("soul echo")}
        CoverageCounter().update(coverage, tmp_path, workers=3)

        assert coverage["soul echo"].records == 30
