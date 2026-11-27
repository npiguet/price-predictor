"""The coverage collector: held-out exclusion, the held-out text
unit and the held-out round's refusals and caps."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from effects.application.collect_coverage import CollectCoverageConfig
from effects.application.collector_common import DeckHoldout
from effects.domain.event_schema import Event, EventType
from effects.domain.records import ResolutionPayload
from tests.unit.effects.application.collect_coverage_support import (
    _FIXTURES,
    _GEN3_ROOTS,
    _GEN3_SHARD,
    _holding,
    _texts_by_game,
)
from tests.unit.effects.builders import write_cut_gen3_fixture


class TestHeldOutExclusion:
    def test_no_split_holds_nothing_out(self):
        from effects.application.collector_common import load_held_out

        assert load_held_out(None) == frozenset()

    def test_held_out_cards_are_absent_from_the_coverage_unit(self, tmp_path):
        from effects.application.collect_coverage import build_coverage_units
        from effects.infrastructure.converted_trees import ConvertedTrees

        (tmp_path / "s").mkdir()
        for stem, name in (("serra_angel", "Serra Angel"), ("shock", "Shock")):
            (tmp_path / "s" / f"{stem}.txt").write_text(
                f"name: {name}\ntypes: creature\n", encoding="utf-8",
            )
        units = build_coverage_units(
            ConvertedTrees({"cardsfolder": tmp_path}).card_files, frozenset({"shock"}),
        )
        assert set(units) == {"serra angel"}


class TestExclusionsWithoutACheckpoint:
    """The holdout needs no trained model to exist.

    The holdout is derived from the converted tree and two flags, so requiring
    a checkpoint would make coverage collection wait for a training run that
    itself comes after it.
    """

    def test_the_depletion_list_is_read_and_folded(self, tmp_path) -> None:
        listed = tmp_path / "holdout-cards.txt"
        listed.write_text("Soul Echo\nLim-Dûl's Vault\n", encoding="utf-8")

        excluded = DeckHoldout(exclude_cards=listed).exclusions()

        assert excluded == frozenset({"soul echo", "lim-dûl's vault"}), (
            "the file carries Forge's printed case; coverage decks are built "
            "from converted names, which are lowercase"
        )

    def test_neither_source_means_no_exclusions(self) -> None:
        assert DeckHoldout().exclusions() == frozenset()

    def test_both_sources_at_once_is_refused(self, tmp_path) -> None:
        """Two spellings of the same holdout is the bug this feature just had."""
        listed = tmp_path / "holdout-cards.txt"
        listed.write_text("Soul Echo\n", encoding="utf-8")

        with pytest.raises(ValueError, match="one source"):
            DeckHoldout(split_from=tmp_path / "latest.pt", exclude_cards=listed)


# ── the held-out coverage round (FR-035–FR-038) ──────────────


class TestHeldOutTextUnit:
    def test_a_held_out_text_is_one_only_listed_cards_carry(self):
        from effects.domain.coverage import held_out_texts_of

        texts = held_out_texts_of(frozenset({"alpha", "beta"}), {
            "alpha": ["SP$ Rare | Thing$ 1", "SP$ Common"],
            "beta": ["SP$ Rare  | Thing$ 1"],
            "gamma": ["SP$ Common"],
        })
        assert texts == {"SP$ Rare | Thing$ 1": frozenset({"alpha", "beta"})}

    def test_a_listed_cards_line_the_holdout_does_not_hold_is_no_unit(self):
        """A held-out card's other lines are carried by it alone too; under
        ``template`` theirs is printed elsewhere, so no game is ever credited
        to them and the round would deck them until each retired."""
        from effects.application.held_out_round import _held_out_carriers

        held, other = "SP$ Rare | Thing$ 1", "AB$ GainLife | LifeAmount$ 3"
        carriers = _held_out_carriers(
            ["alpha"],
            {"alpha": [held, other], "gamma": ["AB$ GainLife | LifeAmount$ 2"]},
            set(),
            _holding({held}),
        )
        assert carriers == {held: frozenset({"alpha"})}

    def test_a_shard_counts_distinct_games_per_text(self, tmp_path):
        """Counted as build-corpus counts a held-out text's games (FR-053):
        from resolution records only, so a text whose triggers fire but never
        resolve is not satisfied."""
        from effects.application.coverage_counting import count_text_shard

        resolutions, everything = _texts_by_game(_GEN3_SHARD)
        assert resolutions and everything != resolutions
        counted = count_text_shard(_GEN3_SHARD, roots=_GEN3_ROOTS,
                                   held_out=_holding(everything))
        assert counted.games == resolutions
        assert counted.cut_games == 0

    def test_a_cut_game_counts_toward_no_text(self, tmp_path):
        """The fixture without its game-end markers, as a killed worker leaves
        a shard: every game reads as cut, and build-corpus would drop each."""
        from effects.application.coverage_counting import count_text_shard

        cut = write_cut_gen3_fixture(tmp_path / "cut.jsonl.gz")
        resolutions, everything = _texts_by_game(_GEN3_SHARD)
        counted = count_text_shard(cut, roots=_GEN3_ROOTS, held_out=_holding(everything))
        assert counted.games == {}
        assert counted.cut_games >= len(set().union(*resolutions.values())) > 0

    def test_a_game_goes_to_the_first_acting_text_held_out_by_its_unit(
        self, tmp_path,
    ):
        """The text build-corpus credits, not the first text the round lists.

        Under ``template`` the first acting key prints ``NumDmg$ 2``, whose
        template is held out though no listed card prints that exact text;
        build-corpus credits the game to it, so the round's listed ``Draw``
        text, acting second, must not count the game either.
        """
        from effects.application.coverage_counting import (
            TextCoverageCounter,
            count_text_shard,
        )
        from effects.domain.coverage import TextCoverage
        from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
        from tests.unit.effects.builders import make_record, write_ended_shard, write_sidecar

        damage, draw = "SP$ DealDamage | NumDmg$ 2", "SP$ Draw | NumCards$ 1"
        keys = []
        for stem, text in (("variant", damage), ("listed", draw)):
            key = ProvenanceKey(f"cardsfolder/a/{stem}.txt", 0, "spell", 0)
            write_sidecar(
                ProvenanceSidecar(card=stem, script_file=key.script_file, lines=(
                    SidecarLine(0, "spell", (key,), script_text=text, prose_offset=0),
                )),
                tmp_path / "cardsfolder" / "a" / f"{stem}.provenance.json",
            )
            keys.append(key)
        shard = tmp_path / "records" / "run.0-a.jsonl.gz"
        write_ended_shard(shard, [make_record(
            generation=3, ability=tuple(keys), payload=ResolutionPayload(
                events=(Event(type=EventType.CARD_DRAWN, subjects=("P0",)),),
            ),
        )])
        roots = {"cardsfolder": str(tmp_path / "cardsfolder")}
        held_out = _holding({"SP$ DealDamage | NumDmg$ 3", draw})
        coverage = {draw: TextCoverage(text=draw)}

        assert count_text_shard(shard, roots=roots, held_out=held_out).games == {
            damage: {"g1"},
        }
        TextCoverageCounter(roots, held_out).update(coverage, tmp_path / "records", workers=1)
        assert coverage[draw].games == set()

    def test_a_refused_resolution_does_not_count_toward_its_text(self, tmp_path):
        """A record the quality rules refuse never reaches the manifest."""
        from dataclasses import replace

        from effects.application.coverage_counting import count_text_shard
        from effects.domain.record_quality import MAX_EVENTS_PER_RECORD
        from effects.infrastructure.shard_io import read_shard
        from tests.unit.effects.builders import write_ended_shard

        effects = [
            record for record in read_shard(_GEN3_SHARD)
            if isinstance(record.payload, ResolutionPayload)
        ]
        flood = (Event(type=EventType.DAMAGE_DEALT, subjects=("P0",)),) * (
            MAX_EVENTS_PER_RECORD + 1
        )
        clean, flooded = tmp_path / "clean.jsonl.gz", tmp_path / "flooded.jsonl.gz"
        write_ended_shard(clean, effects)
        write_ended_shard(flooded, [
            replace(record, payload=replace(record.payload, events=flood))
            for record in effects
        ])
        _, everything = _texts_by_game(_GEN3_SHARD)
        held_out = _holding(everything)

        assert count_text_shard(clean, roots=_GEN3_ROOTS, held_out=held_out).games
        assert count_text_shard(flooded, roots=_GEN3_ROOTS, held_out=held_out).games == {}

    def test_a_process_reads_each_sidecar_once_across_shards(self, monkeypatch):
        """A sidecar cache per shard re-read every sidecar the shard named."""
        from effects.application import coverage_counting
        from effects.infrastructure import converted_trees, sidecar_io

        reads: list[Path] = []
        real_read = sidecar_io.read_sidecar

        def counting_read(path):
            reads.append(Path(path))
            return real_read(path)

        monkeypatch.setattr(sidecar_io, "read_sidecar", counting_read)
        converted_trees._process_sidecars.cache_clear()
        roots = {
            "cardsfolder": str(_FIXTURES / "gen3-sidecars" / "cardsfolder"),
            "tokenscripts": str(_FIXTURES / "gen3-sidecars" / "tokenscripts"),
        }
        shard = _FIXTURES / "gen3-records.jsonl.gz"
        first = coverage_counting.count_text_shard(shard, roots=roots, held_out=_holding(()))
        read_by_one = len(reads)
        second = coverage_counting.count_text_shard(shard, roots=roots, held_out=_holding(()))
        converted_trees._process_sidecars.cache_clear()

        assert read_by_one > 0
        assert len(reads) == read_by_one
        assert first.games == second.games == {}

    def test_the_round_ends_without_waiting_on_uncastable_texts(self):
        """FR-037 / spec Story 2 scenario 6."""
        from effects.application.held_out_round import text_residues
        from effects.domain.coverage import TextCoverage, awaited_texts, is_complete

        coverage = {
            "done": TextCoverage(text="done", games={f"g{i}" for i in range(5)}),
            "retired": TextCoverage(text="retired", games={"g1"}, retired=True),
            "stuck": TextCoverage(text="stuck", castable=False),
        }
        assert set(awaited_texts(coverage)) == {"done", "retired"}
        assert is_complete(awaited_texts(coverage), 5)
        residues = text_residues(coverage, 5)
        assert residues.under_floor == {"retired": 1}
        assert residues.uncastable == ("stuck",)
        assert "stuck" in residues.render(5)

    def test_decks_hold_only_listed_cards_plus_basics(self):
        """Spec Story 2 scenario 5."""
        import random

        from effects.application.collector_common import build_coverage_decks
        from effects.domain.coverage import (
            DeckCandidate,
            TextCoverage,
            text_deck_weights,
        )

        carriers = {"t1": frozenset({"alpha"}), "t2": frozenset({"beta"})}
        coverage = {"t1": TextCoverage(text="t1"), "t2": TextCoverage(
            text="t2", games={f"g{i}" for i in range(5)},
        )}
        weights = text_deck_weights(coverage, carriers, 5)
        assert weights == {"alpha": 5.0}
        candidates = {
            "alpha": DeckCandidate("alpha", "{R}"), "beta": DeckCandidate("beta", None),
        }
        decks = build_coverage_decks(weights, candidates, 3, rng=random.Random(0))
        basics = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
        assert all(set(deck) <= {"alpha"} | basics for deck in decks)


class TestOnlyCardsRefusals:
    """FR-038 / spec Story 2 scenario 7: refused before any worker starts."""

    @pytest.mark.parametrize("extra", [
        ("--exclude-cards", "holdout.txt"),
        ("--split-from", "models/x/latest.pt"),
    ])
    def test_only_cards_beside_another_holdout_source_is_refused(self, extra):
        from effects.infrastructure.cli import build_parser, collector_config

        args = build_parser().parse_args(
            ["collect-coverage", "--only-cards", "holdout.txt", *extra],
        )
        with pytest.raises(ValueError, match="--only-cards"):
            collector_config(CollectCoverageConfig, args)

    def test_only_cards_beside_a_training_corpus_is_refused(self, tmp_path):
        from effects.infrastructure.cli import build_parser, run_collect_coverage

        corpus = tmp_path / "depleted"
        corpus.mkdir()
        (corpus / "holdout-cards.txt").write_text("Soul Echo\n", encoding="utf-8")
        args = build_parser().parse_args([
            "collect-coverage", "--only-cards", "holdout.txt",
            "--training-corpus", str(corpus),
        ])
        with patch("effects.application.collect_coverage.run") as collect:
            assert run_collect_coverage(args) == 1
        collect.assert_not_called()

    def test_alone_it_selects_the_text_unit(self):
        from effects.infrastructure.cli import build_parser, collector_config

        config = collector_config(CollectCoverageConfig, build_parser().parse_args(
            [
                "collect-coverage", "--only-cards", "holdout.txt", "--min-text-games", "7",
                "--effect-records", "records",
            ],
        ))
        assert config.deck_holdout.only_cards == Path("holdout.txt")
        assert config.min_text_games == 7


class TestHeldOutRoundCaps:
    """FR-059a: every stratum has one record composition,
    so the held-out round collects under the run's own caps. It used to lift
    ``--playability-rate`` to 1.0, which made special decision records about
    a hundred times denser in the card-disjoint stratum than in training."""

    def test_the_held_out_rounds_caps_are_the_runs(self, tmp_path, fake_supervisor):
        from effects.application import collect_coverage, held_out_round
        from effects.domain.collection_caps import CollectionCaps

        cards = tmp_path / "cardsfolder"
        (cards / "a").mkdir(parents=True)
        (cards / "a" / "alpha.txt").write_text(
            "name: alpha\nmana cost: {G}\ntypes: creature\n", encoding="utf-8",
        )
        listed = tmp_path / "holdout-cards.txt"
        listed.write_text("alpha\n", encoding="utf-8")
        (tmp_path / "records").mkdir()
        caps = CollectionCaps(aimed_at=(), playability_rate=0.02)

        with (
            patch.object(held_out_round, "consult_castability", return_value={}),
            patch("effects.infrastructure.converted_trees.load_card_texts",
                  return_value={"alpha": ["SP$ Alpha"]}),
            patch("effects.infrastructure.converted_trees.ConvertedTrees.holdout",
                  return_value=_holding({"SP$ Alpha"})),
        ):
            collect_coverage.run(collect_coverage.CollectCoverageConfig(
                effect_records=tmp_path / "records", cards_folders=(cards,),
                deck_holdout=DeckHoldout(only_cards=listed), min_text_games=0,
                caps=caps, workers=1,
            ))

        assert fake_supervisor.constructor.call_args.args[0].caps == caps
