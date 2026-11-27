"""``collect-variants`` recomputes each directed rule's fork probability after
every round from the directed forks its shards hold, and hands the next
round's workers the result."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from effects.application.collect_variants import (
    CollectVariantsConfig,
    VariantGameCounter,
    VariantRounds,
    play_variant_rounds,
)
from effects.application.directed_steering import DirectedSteering
from effects.domain.collection_caps import CollectionCaps
from effects.domain.coverage import DeckCandidate
from effects.domain.directed_sides import DirectedForkGames
from effects.domain.provenance import ProvenanceKey
from effects.domain.variant_rounds import VariantGames
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import make_record, write_ended_shard

RULES = Path(__file__).parents[4] / "rules"
_PLAYED = Path(__file__).parents[3] / "fixtures" / "effects" / "directed-702.19ef-combats.jsonl"
BOLT = "variant-scripts/bolt_variant_0a1b2c3d.txt"
RUN = "run-1"


def fork_record(rule: str, game: str):
    """The fixture's real directed record of ``rule``, moved into ``game``."""
    for line in _PLAYED.read_text(encoding="utf-8").splitlines():
        data = json.loads(line)
        if data.get("aimed_at") == [rule]:
            return replace(record_from_dict(data), game_id=game, run_id=RUN,
                           record_id=f"{game}.fork")
    raise AssertionError(f"no directed record of {rule}")


def bolt_record(game: str):
    return make_record(
        record_id=f"{game}.1", game_id=game, run_id=RUN,
        ability=(ProvenanceKey(BOLT, 0, "spell", 0),),
    )


def rules_dir(tmp_path: Path) -> Path:
    """The real catalogue with 702.19e's plain_trample registered beside the
    real registry's pairs: the fixture's played rule is in the catalogue only."""
    target = tmp_path / "rules"
    if not target.exists():
        shutil.copytree(RULES / "scenarios", target / "scenarios")
        listed = (RULES / "directed-fork-variants.tsv").read_text(encoding="utf-8")
        (target / "directed-fork-variants.tsv").write_text(
            listed + "702.19e\tplain_trample\n", encoding="utf-8")
    return target


def config(tmp_path: Path, *rules: str, **overrides) -> CollectVariantsConfig:
    return CollectVariantsConfig(
        effect_records=tmp_path / "records", variant_scripts=tmp_path / "variant-scripts",
        decks_per_round=4, workers=1, rules_dir=rules_dir(tmp_path), variant_games=2, steer_floor=2,
        caps=CollectionCaps(aimed_at=rules, directed_fork_rules=rules),
        **overrides,
    )


class TestSteering:
    def test_a_run_listing_no_rule_sets_nothing(self, tmp_path, fake_supervisor):
        DirectedSteering.of(config(tmp_path)).steer(fake_supervisor, DirectedForkGames())
        fake_supervisor.set_directed_rule_probabilities.assert_not_called()

    def test_before_any_fork_every_listed_rule_is_at_one(self, tmp_path, fake_supervisor, caplog):
        steering = DirectedSteering.of(config(tmp_path, "702.19e", "702.19f"))
        with caplog.at_level(logging.INFO):
            steering.steer(fake_supervisor, DirectedForkGames())
        fake_supervisor.set_directed_rule_probabilities.assert_called_once_with(
            {"702.19e": 1.0, "702.19f": 1.0})
        assert "directed fork probabilities: 702.19e 1.00 (0/2), 702.19f 1.00 (0/2)" in caplog.text

    def test_a_forked_side_lowers_its_rule_only(self, tmp_path, fake_supervisor):
        forks = DirectedForkGames()
        forks.add(fork_record("702.19e", "g1"))
        DirectedSteering.of(config(tmp_path, "702.19e", "702.19f")).steer(fake_supervisor, forks)
        fake_supervisor.set_directed_rule_probabilities.assert_called_once_with(
            {"702.19e": 0.5, "702.19f": 1.0})

    def test_the_count_is_the_steer_floor(self, tmp_path, fake_supervisor):
        forks = DirectedForkGames()
        for game in ("g1", "g2"):
            forks.add(fork_record("702.19e", game))
        DirectedSteering.of(config(tmp_path, "702.19e")).steer(fake_supervisor, forks)
        fake_supervisor.set_directed_rule_probabilities.assert_called_once_with({"702.19e": 0.0})


class TestTheRounds:
    def play(self, tmp_path, supervisor, *rounds):
        """Play rounds whose workers write ``rounds`` (one list of records each)."""
        records = tmp_path / "records"
        records.mkdir(exist_ok=True)
        written = iter(rounds)

        def write(*_args, **_kwargs):
            shard = next(written, None)
            if shard:
                write_ended_shard(records / f"{RUN}.0-{len(list(records.glob('*.gz')))}.jsonl.gz",
                                  shard)
        supervisor.play_round.side_effect = write
        counter = VariantGameCounter(records, RUN, workers=1)
        units = {"Bolt Variant 0a1b2c3d": VariantGames(
            name="Bolt Variant 0a1b2c3d", script_file=BOLT)}
        cands = {"Bolt Variant 0a1b2c3d": DeckCandidate("Bolt Variant 0a1b2c3d", "{R}")}
        play_variant_rounds(VariantRounds(
            units=units, candidates=cands, counter=counter,
            config=config(tmp_path, "702.19e", "702.19f"),
        ), supervisor)

    def test_each_round_is_played_with_the_probabilities_the_rounds_before_it_left(
        self, tmp_path, fake_supervisor,
    ):
        self.play(
            tmp_path, fake_supervisor,
            [bolt_record("g1"), fork_record("702.19e", "g1")],
            [bolt_record("g2"), fork_record("702.19e", "g2")],
        )
        steps = [c for c in fake_supervisor.method_calls
                 if c[0] in ("set_directed_rule_probabilities", "play_round")]
        assert [(c[0], c[1][0] if c[0] == "set_directed_rule_probabilities" else None)
                for c in steps] == [
            ("set_directed_rule_probabilities", {"702.19e": 1.0, "702.19f": 1.0}),
            ("play_round", None),
            ("set_directed_rule_probabilities", {"702.19e": 0.5, "702.19f": 1.0}),
            ("play_round", None),
            ("set_directed_rule_probabilities", {"702.19e": 0.0, "702.19f": 1.0}),
        ]

    def test_a_shard_is_counted_once_across_rounds(self, tmp_path, fake_supervisor):
        """The second round writes nothing: the first shard's fork stays one."""
        self.play(
            tmp_path, fake_supervisor,
            [bolt_record("g1"), fork_record("702.19e", "g1")],
        )
        sent = [c.args[0] for c in
                fake_supervisor.set_directed_rule_probabilities.call_args_list]
        assert sent[1]["702.19e"] == 0.5
        assert all(p["702.19e"] == 0.5 for p in sent[1:])

    def test_a_cut_game_s_fork_counts_for_nothing(self, tmp_path, fake_supervisor):
        from effects.infrastructure.shard_io import write_shard

        records = tmp_path / "records"
        records.mkdir()
        write_shard(records / f"{RUN}.0-0.jsonl.gz", [fork_record("702.19e", "cut")])
        counter = VariantGameCounter(records, RUN, workers=1)
        counter.update({})
        assert counter.directed.games == {}

    def test_the_continued_run_s_forks_count_before_the_first_round(
        self, tmp_path, fake_supervisor,
    ):
        records = tmp_path / "records"
        records.mkdir()
        write_ended_shard(records / "earlier.0-0.jsonl.gz",
                          [fork_record("702.19e", "g1")])
        counter = VariantGameCounter(records, RUN, workers=1, continues=("earlier",))
        counter.seed({})
        assert counter.directed.count("702.19e", {"plain_trample"}) == 1


def test_the_supervisor_keeps_the_probabilities_for_its_workers_and_run_json(tmp_path):
    from effects.infrastructure.collector_connector import CollectorRun, CollectorSupervisor

    supervisor = CollectorSupervisor(CollectorRun(
        worker_count=1, effect_records=tmp_path, run_id=RUN,
        caps=CollectionCaps(directed_fork_rules=("702.19e",)),
    ))
    with patch("effects.infrastructure.collector_connector.annotate_run") as annotate:
        supervisor.set_directed_rule_probabilities({"702.19e": 0.25})
    assert supervisor.caps.as_system_properties()[
        "effect.directed.probability.702.19e"] == "0.25"
    fields = annotate.call_args.kwargs
    assert fields["directed_rule_probabilities"] == {"702.19e": 0.25}
    assert fields["caps"]["directed_rule_probabilities"] == {"702.19e": 0.25}
