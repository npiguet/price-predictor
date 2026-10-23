"""The training run's log lines: what each number is printed as.

Kept apart from the loop that computes the numbers, so a line's shape can be
read and tested without a run behind it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from effects.application.train_effect_model import MAX_GRAD_NORM
from effects.domain.loss_floor import FLOOR_EPSILON


def format_norms(norms: Mapping[str, float]) -> str:
    """The pre-clip gradient norms, or nothing before the first backward."""
    if not norms:
        return ""
    body = ", ".join(f"{name} {value:.2f}" for name, value in norms.items())
    return f"\n  |g| {body} (clipped per group at {MAX_GRAD_NORM:g})"


def explained(name: str, value: float, floor: Mapping[str, float] | None) -> str:
    """How much of the constant predictor's loss this field's head removed.

    ``1 - loss/floor`` against the best constant prediction for the field
    (FR-127c): zero is the base rate, negative is worse than predicting it. A
    field's loss is a number in nats with no scale of its own, so on its own it
    cannot say whether the head learned the field or learned how often it fires.

    A field whose targets never vary has no deviance to explain and reads
    ``n/a`` rather than dividing by nothing at all.
    """
    if floor is None:
        return ""
    reference = floor.get(name)
    if reference is None or reference < FLOOR_EPSILON:
        return " (n/a)"
    return f" ({round(100.0 * (1.0 - value / reference))}%)"


def format_parts(
    parts: Mapping[str, float], floor: Mapping[str, float] | None = None,
) -> str:
    """Each field's loss term, largest first, and what it explains.

    Ordered by size rather than by name because the question this line answers
    is which term the loss is made of: a single field carrying nearly all of it
    is what a blow-up looks like, and alphabetical order buries that.

    ``floor`` is the card-disjoint constant-predictor floor and is passed only
    for that breakdown; the per-shard training line has no floor of its own —
    its records change every shard — and prints the terms alone.
    """
    if not parts:
        return ""
    ranked = sorted(parts.items(), key=lambda kv: -kv[1])
    body = ", ".join(
        f"{name} {value:.3f}{explained(name, value, floor)}"
        for name, value in ranked
    )
    return f"\n  fields: {body}"


def format_floor(floor: Mapping[str, float]) -> str:
    """The floor itself, ordered like the breakdown it scales."""
    ranked = sorted(floor.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{name} {value:.3f}" for name, value in ranked)


def format_shares(counts: Mapping[str, int], order: Sequence[str] = ()) -> str:
    """``label share%`` per label, in ``order`` first and then largest first."""
    total = sum(counts.values())
    if not total:
        return "none"
    ordered = [label for label in order if label in counts] + sorted(
        (label for label in counts if label not in order),
        key=lambda label: -counts[label],
    )
    return ", ".join(
        f"{label} {100.0 * counts[label] / total:.1f}%" for label in ordered
    )


def format_terms(terms: Mapping[str, float]) -> str:
    """The head terms of the loss, or nothing when there were none."""
    if not terms:
        return ""
    body = ", ".join(f"{name} {value:.3f}" for name, value in terms.items())
    return f"\n  heads: {body}"
