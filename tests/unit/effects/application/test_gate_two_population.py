"""Gate 2's scorer over a corpus, and who it counts (FR-119 – FR-122).

The counter and the scorer must describe one population, because the
under-sampled verdict and the agreement verdict are read off the same row.
"""

from __future__ import annotations

import pytest
import torch

from effects.application.gate_two import (
    KeywordScore,
    evaluate_keyword,
    score_keywords,
)
from effects.application.model_loading import Runnable
from effects.domain.damage_step_keywords import (
    KEYWORDS_BY_NAME,
)
from effects.domain.effect_head_layout import (
    FIELD_SLICES,
    PER_ENTITY_FIELDS,
    PER_ENTITY_WIDTH,
    ZONE_OUTCOMES,
)
from effects.domain.evaluation_report import CheckStatus
from effects.domain.input_layout import SlotKind
from effects.domain.records import EffectRecord
from effects.domain.state_snapshot import (
    CombatStatus,
)
from effects.domain.surface_collation import gather_entities
from tests.unit.effects.application.gate_two_support import (
    _FIRST_STRIKE,
    _OTHER,
    _batcher,
    _combat_record,
    _creature,
    _NoEncoder,
    _printed_board,
)

# ── the whole scorer, against a model that reacts to the strip ───────────


_FIELDS = tuple(PER_ENTITY_FIELDS)


class _AbilityCountingModel(torch.nn.Module):
    """A stand-in whose predicted damage and death rise with the record's
    ability count.

    Not a model of anything — it exists so the scorer's two builds differ in a
    way that is a function of the perturbation and nothing else, which is what
    makes an agreement number here a test of the wiring rather than of a
    randomly initialised head. The build that keeps the keyword keeps one more
    ability, so it predicts more damage and a likelier death, which is the way
    first strike moves its opponent.
    """

    def forward(self, inputs):
        """Every slot's "hidden" is its record's ability count."""
        slot_kinds = inputs.slot_kinds
        abilities = (slot_kinds == int(SlotKind.ABILITY)).sum(dim=1).float()
        return abilities[:, None, None].expand(-1, slot_kinds.shape[1], 1)

    def per_entity_at(self, hidden, index):
        abilities = gather_entities(hidden, index)[..., 0]
        row = torch.zeros(*abilities.shape, PER_ENTITY_WIDTH)
        row[..., FIELD_SLICES["damage_taken"][0]] = 0.5 * abilities
        died = FIELD_SLICES["zone_outcome"][0] + ZONE_OUTCOMES.index("died")
        row[..., died] = 0.5 * abilities
        return row


def _runnable() -> Runnable:
    return Runnable(_NoEncoder(), _AbilityCountingModel(), _batcher(), _FIELDS)


class TestScoreKeywords:
    def test_it_scores_the_qualifying_records(self):
        records = [_printed_board(game=f"g{i}") for i in range(3)]
        scores = score_keywords(
            records, (KEYWORDS_BY_NAME["first_strike"],), _runnable(),
        )
        score = scores["first_strike"]
        assert score.qualifying == 3
        assert score.agreeing == 3

    def test_it_reports_agreement_per_field(self):
        records = [_printed_board()]
        scores = score_keywords(
            records, (KEYWORDS_BY_NAME["first_strike"],), _runnable(),
        )
        assert set(scores["first_strike"].per_field) == {
            "opponent.damage_taken", "opponent.zone_outcome/died",
        }

    def test_a_keyword_no_record_qualifies_for_scores_nothing(self):
        scores = score_keywords(
            [_printed_board()], (KEYWORDS_BY_NAME["wither"],), _runnable(),
        )
        assert scores["wither"] == KeywordScore()

    def test_an_empty_corpus_scores_every_row_at_zero(self):
        scores = score_keywords(
            [], (KEYWORDS_BY_NAME["lifelink"],), _runnable(),
        )
        assert scores["lifelink"].qualifying == 0


# ── the evaluator's own wiring ───────────────────────────────────────────


class TestEvaluatorWiring:
    """The gate-2 block must produce a computed agreement, not a zero.

    It reported ``agreeing_records=0`` for every keyword until the scorer
    existed, which routed all eight to a probe whatever the model did — a canary
    that always sings the same note.
    """

    @staticmethod
    def _load(_records):
        return _runnable()

    def test_it_scores_the_game_disjoint_combat_records(self):
        from effects.application.gate_two import keyword_verdicts

        verdicts = keyword_verdicts([_printed_board(game="gd-1")], self._load)

        first_strike = next(v for v in verdicts if v.keyword == "first_strike")
        assert first_strike.qualifying_records == 1
        assert first_strike.direction_agreement == 1.0

    def test_the_reason_lists_agreement_per_field(self):
        from effects.application.gate_two import keyword_verdicts

        verdicts = keyword_verdicts([_printed_board(game="gd-1")], self._load)

        first_strike = next(v for v in verdicts if v.keyword == "first_strike")
        assert "damage_taken 100%" in first_strike.reason
        assert "zone_outcome/died 100%" in first_strike.reason

    def test_no_game_disjoint_combat_record_leaves_every_keyword_at_zero(self):
        from effects.application.gate_two import keyword_verdicts

        def _never(_records):
            raise AssertionError("no records: the model must not be loaded")

        verdicts = keyword_verdicts([], _never)

        assert len(verdicts) == 8
        assert all(v.qualifying_records == 0 for v in verdicts)
        assert all("under-sampled" in v.reason for v in verdicts)

    def test_the_gate_reads_its_own_records_off_the_context(self, tmp_path):
        """Like gate 1, gate 2 is handed the evaluation context and nothing
        else: it reads the checkpoint's game-disjoint games itself."""
        from types import SimpleNamespace

        from effects.application.gate_two import run_gate_two
        from effects.domain.split_provenance import SplitProvenance
        from effects.infrastructure.corpus_store import CorpusStore
        from effects.infrastructure.shard_io import write_shard

        corpus = tmp_path / "corpus"
        write_shard(
            CorpusStore(corpus).game_disjoint_dir / "shard-00001.jsonl.gz",
            [_printed_board(game="gd-1"), _printed_board(game="gd-2")],
        )
        context = SimpleNamespace(
            main=SimpleNamespace(provenance=SplitProvenance(game_disjoint_games=("gd-1",))),
            shards={"game-disjoint": (
                CorpusStore(corpus).game_disjoint_dir / "shard-00001.jsonl.gz",)},
            main_runnable=self._load,
        )

        outcome = run_gate_two(context)

        first_strike = next(v for v in outcome.verdicts if v.keyword == "first_strike")
        assert first_strike.qualifying_records == 1
        assert outcome.check.status is CheckStatus.REPORTED
        assert outcome.check.values == {
            v.keyword: v.direction_agreement for v in outcome.verdicts
        }


# ── who is counted (FR-119, FR-120) ──────────────────────────────────────


def _two_first_strikers() -> EffectRecord:
    """Two first-striking attackers in one combat, each blocked by its own 2/2."""
    creatures = []
    for index in range(2):
        creatures.append(_creature(
            f"A{index}", 2, 2, printed=(_FIRST_STRIKE,),
            combat=CombatStatus(
                attacking="P1", blocked_by=(f"B{index}",), became_blocked=True,
            ),
        ))
        creatures.append(_creature(
            f"B{index}", 2, 2, controller="P1", printed=(_OTHER,),
            combat=CombatStatus(blocking=(f"A{index}",)),
        ))
    return _combat_record(*creatures, substep="first_strike")


def _scores(records):
    return score_keywords(
        records, (KEYWORDS_BY_NAME["first_strike"],), _runnable(),
    )["first_strike"]


class TestPopulation:
    def test_a_fork_record_never_qualifies(self):
        """A damage-step probe re-runs a real combat: a near-duplicate board."""
        import dataclasses

        probe = dataclasses.replace(
            _printed_board(game="gd-1"), record_id="gd-1.2", fork=True,
            mirror_of="gd-1.1",
        )
        assert _scores([probe]).qualifying == 0
        assert _scores([_printed_board(game="gd-1"), probe]).qualifying == 1

    def test_a_records_repeated_row_is_observed_once(self):
        """A curated stratum repeats a record's row for every added copy; its
        agreement counts the record once (FR-068a)."""
        board = _printed_board(game="gd-1")
        score = _scores([board, board, board])
        assert (score.qualifying, score.observations) == (1, 1)

    def test_two_carriers_in_one_record_count_one_qualifying_record(self):
        """FR-119's threshold counts records; agreement averages the carriers."""
        score = _scores([_two_first_strikers()])
        assert score.qualifying == 1
        assert score.observations == 2
        assert score.agreeing == 2

    def test_the_verdict_reads_agreement_over_observations(self):

        verdict = evaluate_keyword(
            KEYWORDS_BY_NAME["first_strike"],
            KeywordScore(qualifying=300, agreeing=300, observations=400),
        )
        assert verdict.qualifying_records == 300
        assert verdict.observations == 400
        assert verdict.direction_agreement == pytest.approx(0.75)


class TestGateTwoRecords:
    def _write(self, directory, *records):
        from effects.infrastructure.shard_io import write_shard

        write_shard(directory / "shard-00001.jsonl.gz", records)

    def test_it_reads_the_recorded_games_of_the_curated_game_disjoint_stratum(
        self, tmp_path,
    ):
        from effects.application.gate_two import gate_two_records
        from effects.domain.split_provenance import SplitProvenance
        from effects.infrastructure.corpus_store import CorpusStore

        corpus = tmp_path / "corpus"
        recorded = _printed_board(game="gd-1")
        other = _printed_board(game="gd-2")
        self._write(CorpusStore(corpus).game_disjoint_dir, recorded, other)
        provenance = SplitProvenance(game_disjoint_games=("gd-1",))

        read = list(gate_two_records(
            provenance, shards=(CorpusStore(corpus).game_disjoint_dir / "shard-00001.jsonl.gz",)))
        assert [r.record_id for r in read] == [recorded.record_id]
