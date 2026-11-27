"""The variant round (FR-058a): variants staged, decked and played in
rounds weighted by the games each lacks, their games counted from this run's
own shards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from effects.application.collect_variants import CollectVariantsConfig, GeneratedVariant
from effects.domain.records import EffectRecord
from tests.unit.effects.builders import make_record


def _write_plain_sidecars(variant_scripts: Path) -> int:
    """What the sidecar pass leaves: a sidecar of one plain line beside every
    variant script in the tree (the pass itself needs a JVM)."""
    from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
    from effects.infrastructure.sidecar_io import sidecar_path_for
    from tests.unit.effects.builders import write_sidecar

    for script in Path(variant_scripts).glob("*.txt"):
        file = f"variant-scripts/{script.name}"
        write_sidecar(ProvenanceSidecar(card=script.stem, script_file=file, lines=(SidecarLine(
            line_index=0, line_kind="spell", script_text="SP$ Draw | NumCards$ 1",
            provenance=(ProvenanceKey(file, 0, "spell", 0),), prose_offset=0,
        ),)), sidecar_path_for(script))
    return 0


def _empty_card_tree(tmp_path: Path) -> Path:
    """A converted card tree holding no card: no text is held out by it."""
    tree = tmp_path / "cardsfolder"
    tree.mkdir(exist_ok=True)
    return tree


@pytest.fixture(autouse=True)
def _no_real_unstaging(monkeypatch):
    """A run ends by emptying Forge's real custom-cards directory; never here."""
    from effects.application import collect_variants

    monkeypatch.setattr(collect_variants, "unstage_variants", MagicMock(return_value=0))


def _record(**overrides) -> EffectRecord:
    defaults: dict = {"record_id": "r.0.1", "run_id": "r", "game_id": "r.0.1"}
    defaults.update(overrides)
    return make_record(**defaults)


class TestTheVariantRoundPlaysTheVariants:
    def test_the_scripts_are_staged_and_decked(self, tmp_path, fake_supervisor):
        from effects.application import collect_variants

        fake_supervisor.run_id = "run-123"
        source = tmp_path / "cards"
        source.mkdir()
        (source / "bolt.txt").write_text(
            "Name:Bolt\nManaCost:R\nTypes:Instant\n"
            "A:SP$ DealDamage | NumDmg$ 3 | ValidTgts$ Any\n",
            encoding="utf-8",
        )
        records = tmp_path / "records"
        records.mkdir()
        (records / "seed.jsonl").write_text("{}\n" * 100, encoding="utf-8")

        with patch.object(collect_variants, "write_variant_sidecars",
                          side_effect=_write_plain_sidecars), \
             patch.object(collect_variants, "loadable_variants",
                          side_effect=lambda generated, _root: generated):
            # The loadable list is written by the sidecar writer, which is
            # mocked here, so the file it would leave behind does not exist.
            # TestOnlyLoadableVariantsAreDecked covers the filter itself.
            collect_variants.run(
                collect_variants.CollectVariantsConfig(
                    effect_records=records,
                    forge_cards_path=source,
                    variant_scripts=tmp_path / "variants",
                    decks_per_round=3, workers=1,
                    # An empty converted tree holds no text out, rather than
                    # reading the machine's real output tree.
                    cards_folders=(_empty_card_tree(tmp_path),),
                )
            )

        constructed = fake_supervisor.constructor.call_args.args[0]
        assert constructed.variant_scripts == tmp_path / "variants"
        assert fake_supervisor.play_round.call_args.args[0], "no variant decks"
        assert fake_supervisor.play_round.call_args.args[2].matches == 3
        # The run's own decks file (its path is the supervisor's: every
        # worker reloads it at its start, so two runs never share one).
        fake_supervisor.decks_path.assert_called_with("variant")
        assert fake_supervisor.play_round.call_args.args[1] is (
            fake_supervisor.decks_path.return_value
        )


_RUN = "run-1"


def _variant_record(script_file: str, game: str, n: int, **overrides) -> EffectRecord:
    """A resolution acting through ``script_file``'s first spell line."""
    from effects.domain.provenance import ProvenanceKey

    return _record(
        record_id=f"{game}.{n}", game_id=game, run_id=_RUN,
        ability=(ProvenanceKey(script_file, 0, "spell", 0),), **overrides,
    )


def _round_writes(directory: Path, *rounds: list[EffectRecord], run: str = _RUN,
                  ended: bool = True):
    """A stand-in for the workers: the n-th round writes ``rounds[n]`` as a
    shard of this run's, every game ended by its marker unless ``ended`` is
    False, and a round past the last writes nothing."""
    from effects.infrastructure.shard_io import write_shard
    from tests.unit.effects.builders import write_ended_shard

    played = iter(rounds)
    writer = write_ended_shard if ended else write_shard

    def write(*_args, **_kwargs) -> None:
        records = next(played, None)
        if records:
            directory.mkdir(parents=True, exist_ok=True)
            index = len(list(directory.glob(f"{run}.*")))
            writer(directory / f"{run}.0-{index}.jsonl.gz", records)

    return write


def test_a_damaged_variant_shard_is_counted_up_to_the_damage_once(tmp_path):
    """One damaged shard must not stop a collection run every round."""
    import gzip

    from effects.application.collect_variants import VariantGameCounter
    from effects.domain.variant_rounds import VariantGames
    from effects.infrastructure.record_codec import format_record_line
    from tests.unit.effects.builders import with_game_end_markers

    bolt = "variant-scripts/bolt_variant_0a1b2c3d.txt"
    members = [gzip.compress("".join(
        format_record_line(record) + "\n"
        for record in with_game_end_markers([_variant_record(bolt, game, 1)])).encode("utf-8"))
        for game in ("g1", "g2", "g3")]
    members[1] = b"XX" + members[1][2:]
    shard = tmp_path / f"{_RUN}.0-a.jsonl.gz"
    shard.write_bytes(b"".join(members))
    counter = VariantGameCounter(tmp_path, _RUN, workers=1)
    units = {"Bolt": VariantGames(name="Bolt", script_file=bolt)}
    counter.update(units)
    assert units["Bolt"].games == {"g1"}
    assert len(counter.damaged) == 1 and str(shard) in counter.damaged[0]
    counter.update(units)
    assert len(counter.damaged) == 1


@dataclass(frozen=True)
class _RoundEnv:
    """What a round test is run with: its directory, its patcher, its supervisor."""

    tmp_path: Path
    monkeypatch: pytest.MonkeyPatch
    supervisor: MagicMock


@pytest.fixture
def round_env(tmp_path, monkeypatch, fake_supervisor) -> _RoundEnv:
    return _RoundEnv(tmp_path, monkeypatch, fake_supervisor)


class TestTheUntargetedRound:
    """FR-058a: one variant of every eligible script, played in rounds
    weighted by the games each lacks, until each has ``--variant-games``
    games or has retired; its games counted from this run's own shards."""

    BOLT = "variant-scripts/bolt_variant_0a1b2c3d.txt"
    SHOCK = "variant-scripts/shock_variant_99aa00ff.txt"

    def _fixture(self, env, *rounds, ended=True, **config):
        import effects.application.collect_variants as collect_variants

        tmp_path, monkeypatch, supervisor = env.tmp_path, env.monkeypatch, env.supervisor
        variants = []
        for name, card in (("Bolt Variant 0a1b2c3d", "Bolt"), ("Shock Variant 99aa00ff", "Shock")):
            path = tmp_path / "variant-scripts" / f"{name.lower().replace(' ', '_')}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"Name:{name}\nManaCost:R\nTypes:Instant\n", encoding="utf-8")
            variants.append(GeneratedVariant(
                name=name, source_card=card, path=path, perturbation="numeric",
                mana_cost="{R}",
            ))
        monkeypatch.setattr(collect_variants, "generate_variants", lambda *a, **k: variants)
        # The real filter needs a Forge database staged with the variants;
        # TestOnlyLoadableVariantsAreDecked covers it directly.
        monkeypatch.setattr(
            collect_variants, "loadable_variants", lambda generated, _root: generated,
        )
        monkeypatch.setattr(
            collect_variants, "write_variant_sidecars",
            MagicMock(side_effect=_write_plain_sidecars),
        )
        records = tmp_path / "records"
        records.mkdir()
        supervisor.run_id = _RUN
        supervisor.play_round.side_effect = _round_writes(records, *rounds, ended=ended)
        forge_cards = tmp_path / "forge-cards"
        forge_cards.mkdir()
        config.setdefault("cards_folders", (_empty_card_tree(tmp_path),))
        return collect_variants, CollectVariantsConfig(
            effect_records=records, forge_cards_path=forge_cards,
            variant_scripts=tmp_path / "variant-scripts", decks_per_round=4,
            workers=1, **config,
        )

    def _games(self, script_file, *games):
        return [_variant_record(script_file, game, n) for n, game in enumerate(games)]

    def test_rounds_end_once_every_variant_has_its_games(
        self, tmp_path, monkeypatch, fake_supervisor, caplog,
    ):
        import logging

        collect_variants, config = self._fixture(
            _RoundEnv(tmp_path, monkeypatch, fake_supervisor),
            self._games(self.BOLT, "g1", "g2") + self._games(self.SHOCK, "g1"),
            self._games(self.SHOCK, "g3"),
        )
        with caplog.at_level(logging.INFO):
            assert collect_variants.run(config) == 0

        assert fake_supervisor.play_round.call_count == 2
        # Four records and the three games' markers.
        assert "Collected 7 new effect records; 2 of 2 variants played" in caplog.text

    def test_each_round_weights_a_variant_by_the_games_it_lacks(
        self, tmp_path, monkeypatch, fake_supervisor,
    ):
        """After the first round only Shock lacks a game, so it alone is decked."""
        collect_variants, config = self._fixture(
            _RoundEnv(tmp_path, monkeypatch, fake_supervisor),
            self._games(self.BOLT, "g1", "g2") + self._games(self.SHOCK, "g1"),
            self._games(self.SHOCK, "g3"),
        )
        collect_variants.run(config)

        second = fake_supervisor.play_round.call_args_list[1].args[0]
        nonlands = {card for deck in second for card in deck} - {
            "Plains", "Island", "Swamp", "Mountain", "Forest"}
        assert nonlands == {"Shock Variant 99aa00ff"}

    def test_a_game_counts_once_however_many_records_it_holds(
        self, tmp_path, monkeypatch, fake_supervisor,
    ):
        collect_variants, config = self._fixture(
            _RoundEnv(tmp_path, monkeypatch, fake_supervisor),
            self._games(self.BOLT, "g1", "g1", "g1") + self._games(self.SHOCK, "g1", "g2"),
            self._games(self.BOLT, "g2"),
        )
        assert collect_variants.run(config) == 0
        assert fake_supervisor.play_round.call_count == 2

    def test_a_decision_candidate_naming_the_variant_counts_its_game(self):
        from effects.domain.provenance import ProvenanceKey
        from effects.domain.records import (
            Candidate,
            PlayabilityDecisionPayload,
            PlayabilitySubkind,
            RecordKind,
        )
        from effects.domain.variant_rounds import VariantShardGames

        key = ProvenanceKey(self.BOLT, 0, "spell", 0)
        found = VariantShardGames()
        found.add(make_record(
            RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.DECISION, game_id="g7",
            payload=PlayabilityDecisionPayload(candidates=(Candidate(
                ability=(key,), can_play=True, affordable=True, has_legal_target=True,
            ),)),
        ))
        assert found.games == {self.BOLT: {"g7"}}

    def test_a_cut_game_gives_its_variant_no_game(self, tmp_path):
        """A gen-3 game without its game-end marker is one
        build-corpus drops; it must not finish a variant's games."""
        from effects.application.collect_variants import count_variant_shard
        from effects.infrastructure.shard_io import write_shard
        from tests.unit.effects.builders import with_game_end_markers

        ended = with_game_end_markers([_variant_record(self.BOLT, "g1", 0)])
        cut = [_variant_record(self.BOLT, "g2", 0),
               _variant_record(self.SHOCK, "g2", 1)]
        shard = tmp_path / f"{_RUN}.0-a.jsonl.gz"
        write_shard(shard, ended + cut)

        found = count_variant_shard(shard)

        assert found.games == {self.BOLT: {"g1"}}
        assert found.cut_games == 1
        assert found.records == 4, "every record is still read"

    def test_the_round_reports_its_cut_games(self, tmp_path, monkeypatch, fake_supervisor):
        from tests.unit.effects.builders import with_game_end_markers

        records = with_game_end_markers([
            _variant_record(self.BOLT, "g1", 0),
            _variant_record(self.BOLT, "g2", 0),
            _variant_record(self.SHOCK, "g3", 0),
            _variant_record(self.SHOCK, "g4", 0),
        ])
        cut = _variant_record(self.SHOCK, "g5", 0)
        collect_variants, config = self._fixture(
            _RoundEnv(tmp_path, monkeypatch, fake_supervisor), [*records, cut], ended=False,
        )

        assert collect_variants.run(config) == 0
        assert [call.args for call in fake_supervisor.count_cut_games.call_args_list] == [
            (0,), (1,),
        ]

    def test_another_run_s_shard_never_counts(
        self, tmp_path, monkeypatch, fake_supervisor,
    ):
        """python, two runs on one directory: only this run's shards say what
        its variants played."""
        from tests.unit.effects.builders import write_ended_shard

        collect_variants, config = self._fixture(
            _RoundEnv(tmp_path, monkeypatch, fake_supervisor), no_progress_rounds=1,
        )
        write_ended_shard(
            config.effect_records / "other-run.0-a.jsonl.gz",
            self._games(self.BOLT, "x1", "x2") + self._games(self.SHOCK, "x1", "x2"),
        )
        assert collect_variants.run(config) == 1

    def test_a_continued_run_s_games_count_and_its_finished_variants_are_not_replayed(
        self, tmp_path, monkeypatch, fake_supervisor,
    ):
        """An interrupted run restarted with the same seed replayed
        every variant it had finished, doubling their games. ``--continues``
        counts the earlier run's shards first."""
        from tests.unit.effects.builders import write_ended_shard

        rounds = [self._games(self.SHOCK, "g1", "g2")]
        collect_variants, config = self._fixture(
            _RoundEnv(tmp_path, monkeypatch, fake_supervisor), *rounds, continues=("earlier",),
        )
        write_ended_shard(
            config.effect_records / "earlier.0-a.jsonl.gz", self._games(self.BOLT, "x1", "x2"),
        )

        assert collect_variants.run(config) == 0

        decked = {name for call in fake_supervisor.play_round.call_args_list
                  for deck in call.args[0] for name in deck}
        assert "Bolt Variant 0a1b2c3d" not in decked
        assert "Shock Variant 99aa00ff" in decked

    def test_a_run_that_could_deck_nothing_fails_and_says_so(
        self, tmp_path, monkeypatch, fake_supervisor, caplog,
    ):
        """No unfinished variant yields a deck: no round is played, the
        variants still lack their games, and the run must not read as done."""
        collect_variants, config = self._fixture(
            _RoundEnv(tmp_path, monkeypatch, fake_supervisor))
        monkeypatch.setattr(collect_variants, "build_coverage_decks", lambda *a, **k: [])

        assert collect_variants.run(config) == 1
        fake_supervisor.play_round.assert_not_called()
        assert "already has its" not in caplog.text
        assert "2 variant(s) still short" in caplog.text

    def test_a_continued_run_with_nothing_left_plays_no_round_and_succeeds(
        self, tmp_path, monkeypatch, fake_supervisor,
    ):
        from tests.unit.effects.builders import write_ended_shard

        collect_variants, config = self._fixture(
            _RoundEnv(tmp_path, monkeypatch, fake_supervisor), continues=("earlier",),
        )
        write_ended_shard(
            config.effect_records / "earlier.0-a.jsonl.gz",
            self._games(self.BOLT, "x1", "x2") + self._games(self.SHOCK, "x1", "x2"),
        )

        assert collect_variants.run(config) == 0
        fake_supervisor.play_round.assert_not_called()

    def test_a_variant_shown_without_a_new_game_retires(
        self, tmp_path, monkeypatch, fake_supervisor, caplog,
    ):
        """Bolt has its two games; Shock sits on the board in every round's
        games and never acts, so it retires after --no-progress-rounds."""
        import logging

        from effects.domain.provenance import ProvenanceKey
        from tests.unit.effects.builders import lines_of, make_entity, make_snapshot

        shock_on_board = make_snapshot(entities=(make_entity(
            "E9", "Shock Variant 99aa00ff",
            lines=lines_of(printed=(ProvenanceKey(self.SHOCK, 0, "spell", 0),)),
        ),))
        shown = [_variant_record(self.BOLT, "g9", 0, state=shock_on_board)]
        collect_variants, config = self._fixture(
            _RoundEnv(tmp_path, monkeypatch, fake_supervisor),
            self._games(self.BOLT, "g1", "g2"), shown, shown, no_progress_rounds=2,
        )
        with caplog.at_level(logging.INFO):
            assert collect_variants.run(config) == 0

        assert fake_supervisor.play_round.call_count == 3
        assert "1 retired short of 2 games" in caplog.text

    def test_a_round_that_collected_nothing_fails_loudly(
        self, tmp_path, monkeypatch, caplog, fake_supervisor,
    ):
        """A round whose decks all materialized as basics writes no variant
        record; the variants retire unseen and the run exits 1."""
        import logging

        collect_variants, config = self._fixture(
            _RoundEnv(tmp_path, monkeypatch, fake_supervisor), no_progress_rounds=1,
        )
        with caplog.at_level(logging.ERROR):
            code = collect_variants.run(config)

        assert code == 1
        assert "no new effect records" in caplog.text.lower()

    def test_each_generated_variant_is_logged_with_its_source_and_perturbation(
        self, tmp_path, monkeypatch, caplog, fake_supervisor,
    ):
        """So a variant's records can be traced back to what was changed."""
        import logging

        collect_variants, config = self._fixture(
            _RoundEnv(tmp_path, monkeypatch, fake_supervisor), no_progress_rounds=1,
        )
        with caplog.at_level(logging.INFO):
            collect_variants.run(config)

        assert "Bolt Variant 0a1b2c3d, from Bolt: numeric" in caplog.text

    def test_a_step_with_nothing_to_go_on_ends_the_run_with_its_code(
        self, tmp_path, monkeypatch, fake_supervisor,
    ):
        """The sidecar pass's own exit code, not a blanket 1."""
        collect_variants, config = self._fixture(
            _RoundEnv(tmp_path, monkeypatch, fake_supervisor))
        monkeypatch.setattr(
            collect_variants, "write_variant_sidecars", MagicMock(return_value=3),
        )

        assert collect_variants.run(config) == 3
        fake_supervisor.play_round.assert_not_called()

    def test_the_staging_is_claimed_before_anything_is_staged(
        self, tmp_path, monkeypatch, fake_supervisor,
    ):
        """Another command's first JVM leaves a staging alone while its owner
        runs: the owner is recorded before the sidecar pass stages."""
        collect_variants, config = self._fixture(
            _RoundEnv(tmp_path, monkeypatch, fake_supervisor))
        order: list[str] = []
        monkeypatch.setattr(collect_variants, "claim_staging", lambda: order.append("claim"))
        monkeypatch.setattr(
            collect_variants, "write_variant_sidecars",
            lambda _path: order.append("sidecar pass") or 0,
        )

        collect_variants.run(config)

        assert order[:2] == ["claim", "sidecar pass"]

    def test_a_sidecar_pass_past_its_timeout_ends_the_run_with_1(
        self, tmp_path, monkeypatch, fake_supervisor, caplog,
    ):
        """The one-shot JVM is killed at its timeout; the run says so."""
        from price_predictor.infrastructure.jvm_processes import ForgeWorkerTimeout

        collect_variants, config = self._fixture(
            _RoundEnv(tmp_path, monkeypatch, fake_supervisor))
        monkeypatch.setattr(collect_variants, "write_variant_sidecars", MagicMock(
            side_effect=ForgeWorkerTimeout("VariantSidecarMain ran past its 3600 s timeout"),
        ))

        assert collect_variants.run(config) == 1
        fake_supervisor.play_round.assert_not_called()
        assert "ran past its 3600 s timeout" in caplog.text

    @pytest.mark.parametrize("sidecar_code, plays", [(0, True), (0, False), (3, False)])
    def test_the_round_s_staged_variants_are_taken_out_however_it_ends(
        self, round_env, sidecar_code, plays,
    ):
        """FR-058b: the sidecar pass and every worker stage the
        tree into Forge's custom-cards directory, which every later Forge
        startup on the machine reads."""
        rounds = [self._games(self.BOLT, "g1", "g2") + self._games(self.SHOCK, "g1", "g2")]
        collect_variants, config = self._fixture(
            round_env, *(rounds if plays else []), no_progress_rounds=1,
        )
        round_env.monkeypatch.setattr(
            collect_variants, "write_variant_sidecars", MagicMock(return_value=sidecar_code),
        )
        unstage = MagicMock(return_value=1)
        round_env.monkeypatch.setattr(collect_variants, "unstage_variants", unstage)

        collect_variants.run(config)

        unstage.assert_called_once_with()

    def test_an_interrupted_round_does_not_report_success(
        self, tmp_path, monkeypatch, fake_supervisor,
    ):
        """Ctrl-C on the variant side: ``ForgeWorkerPool`` swallows SIGINT, so
        a round the operator stopped returns exactly like one that finished."""
        collect_variants, config = self._fixture(
            _RoundEnv(tmp_path, monkeypatch, fake_supervisor), self._games(self.BOLT, "g1"),
        )
        fake_supervisor.interrupted = True

        assert collect_variants.run(config) == 130
        assert fake_supervisor.play_round.call_count == 1
