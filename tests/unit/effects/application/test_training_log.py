"""Reading a training log's epoch summaries back."""

from __future__ import annotations

from effects.application.training_log import format_parts, parse_epoch_summaries


def _epoch_line(epoch: int, card: float, parts: str = "") -> str:
    return (f"12:00:00 epoch {epoch} | train 4.0311 | card-disjoint {card:.4f} | "
            f"game-disjoint 2.6789 | gate F1 0.824 | zone acc 0.978 | deviance 0.014{parts}")


def test_reads_the_summary_and_the_fields_breakdown_written_beneath_it() -> None:
    parts = format_parts({"gate": 1.265, "poison_delta": 0.096, "never_varies": 0.001},
                         {"gate": 4.0, "poison_delta": 0.08, "never_varies": 0.0})
    [summary] = parse_epoch_summaries(_epoch_line(5, 2.9648, parts).splitlines())

    assert (summary.epoch, summary.train, summary.card_disjoint, summary.game_disjoint) == (
        5, 4.0311, 2.9648, 2.6789)
    assert (summary.gate_f1, summary.zone_accuracy, summary.deviance) == (0.824, 0.978, 0.014)
    assert summary.fields == {"gate": (1.265, 68.0), "poison_delta": (0.096, -20.0),
                              "never_varies": (0.001, None)}


def test_a_resumed_epoch_keeps_the_line_logged_after_the_restart() -> None:
    lines = [_epoch_line(1, 3.0), _epoch_line(2, 2.9), "resuming from epoch 1",
             _epoch_line(2, 2.8), _epoch_line(3, 2.7)]

    summaries = parse_epoch_summaries(lines)

    assert [(s.epoch, s.card_disjoint) for s in summaries] == [(1, 3.0), (2, 2.8), (3, 2.7)]
    assert all(s.fields == {} for s in summaries)


def test_a_shard_line_is_not_an_epoch_summary() -> None:
    line = "epoch 3 | shard 1/40 a.jsonl.gz | 2000 records | 50 steps | loss 4.1000"

    assert parse_epoch_summaries([line]) == []
