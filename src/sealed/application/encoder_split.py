"""The card-disjoint train/validation split of ``train-encoder``,
stratified by ``score_play`` quartile (FR-018).
"""

from __future__ import annotations

import numpy as np

from sealed.application.encoder_labels import CardLabelMap, CardLabels
from sealed.application.encoder_training_common import RANDOM_SEED, VAL_FRACTION
from sealed.domain.encoder_model import COLOR_ORDER

# ── Train/val split (FR-018, Decision D-4) ──────────────────────────────


def _stratification_value(lbl: CardLabels) -> float | None:
    """Return the FR-018 fallback-chain stratification value for a card.

    Order: ``score_play`` → ``score_draw`` → ``cast_lift`` →
    ``color_lift_W`` → ``color_lift_U`` → ``color_lift_B`` →
    ``color_lift_R`` → ``color_lift_G``. Returns ``None`` when every
    signed cell is empty (the card is bound for the catch-all stratum).
    """
    if lbl.shrunk_score_play is not None:
        return lbl.shrunk_score_play
    if lbl.shrunk_score_draw is not None:
        return lbl.shrunk_score_draw
    if lbl.shrunk_cast_lift is not None:
        return lbl.shrunk_cast_lift
    for color in COLOR_ORDER:
        v = lbl.shrunk_color_lift[color]
        if v is not None:
            return v
    return None


def _split_cards(
    labels: CardLabelMap,
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[list[str], list[str]]:
    """Card-level disjoint split, stratified by ``score_play`` quartile
    with the FR-018 fallback chain for cards whose ``score_play`` cell
    is empty. Cards with no non-empty signed-head cell go into a single
    catch-all stratum.
    """
    if not labels:
        return [], []
    names = sorted(labels.keys())
    # Use NaN as the sentinel for "card has no signed-cell signal" — a
    # plain ``or`` on the FR-018 fallback value misroutes 0.0 (neutral
    # signed score) into the catch-all stratum.
    strat_raw: list[float] = []
    for n in names:
        v = _stratification_value(labels[n])
        strat_raw.append(float("nan") if v is None else v)
    strat_values = np.array(strat_raw, dtype=np.float64)
    catch_all_mask = np.isnan(strat_values)

    # Quartile-stratify the cards whose signed-cell value is non-NaN.
    bin_assignments = np.full(len(names), -1, dtype=np.int64)
    if not catch_all_mask.all():
        finite = strat_values[~catch_all_mask]
        edges = np.unique(np.quantile(finite, [0.25, 0.5, 0.75]))
        bin_assignments[~catch_all_mask] = np.digitize(finite, edges)

    rng = np.random.default_rng(seed)
    train: list[str] = []
    val: list[str] = []
    # Iterate every distinct stratum (each non-negative bin id, plus -1
    # for the catch-all). The deterministic iteration order is the
    # sorted set of bin ids — same `seed` always shuffles the same buckets.
    unique_bins = sorted(set(int(b) for b in bin_assignments.tolist()))
    for b in unique_bins:
        bucket_idxs = np.where(bin_assignments == b)[0]
        rng.shuffle(bucket_idxs)
        n_val = int(round(len(bucket_idxs) * val_fraction))
        val_set = set(bucket_idxs[:n_val].tolist())
        for idx in bucket_idxs:
            (val if idx in val_set else train).append(names[idx])
    # Small-corpus safety: if per-stratum rounding zeroed out the val
    # set entirely, transfer the population's expected total val
    # quota (round(N * val_fraction), at least 1) from train to val.
    # Pick from the largest stratum first so the kept-back cards
    # remain proportional to stratum size.
    if not val and len(train) > 1:
        target = max(1, int(round(len(names) * val_fraction)))
        # Sort train by stratum size descending so we steal from the
        # bigger buckets first. A name → bucket-id dict keeps this linear
        # (the earlier ``names.index(n)`` form was O(n²)).
        name_to_bucket = {names[i]: int(bin_assignments[i]) for i in range(len(names))}
        bucket_sizes = {b: int((bin_assignments == b).sum()) for b in unique_bins}
        train_with_bucket = sorted(
            train, key=lambda n: (-bucket_sizes[name_to_bucket[n]], n),
        )
        moved = set(train_with_bucket[:target])
        val.extend(moved)
        train = [n for n in train if n not in moved]
    train.sort()
    val.sort()
    return train, val
