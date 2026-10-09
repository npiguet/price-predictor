"""The read-out ladder: rung features, the two probes, folds and the share (FR-080–083).

A target that depends on the board is read at six rungs:

====  ==========================================================================
0     the trunk's raw input features, every ``e`` zeroed
1     rung 0 plus the acting ``e``, the target entity's pooled ``e`` and the
      partner's ``e`` (a trigger's cause, a blocker's attacker)
1w    rung 1 with each ``e`` replaced by a fixed random vector per text
1o    rung 0 plus the script values the target depends on
2     the trunk's output at ``[ACT]`` or at the target's ``[CARD]`` slot
3     the model's own head prediction, with no probe
====  ==========================================================================

Rungs 0 to 2 each fit a linear probe and an MLP with hyperparameters fixed here
for every checkpoint, so a difference between two checkpoints is the models'
and not the probe's. Folds are assigned by group and never split one: an
ability text for a target read at ``[ACT]``, a card for a target that pools a
card's lines (FR-082).

Features for every rung come from one batched forward pass per stratum and are
held on the host; the probes then fit on the cached arrays.

Nothing here does work at import time.
"""

from __future__ import annotations

import zlib
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# ── constants fixed in code (research.md § Constants fixed in code) ─────

#: The share is reported only where rung 3 exceeds rung 0 by this much.
MIN_GAP = 0.05
FOLDS = 5
SEED = 42
LINEAR_C = 1.0
RIDGE_ALPHA = 1.0
MLP_HIDDEN = 256
MLP_LAYERS = 2
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_BATCH = 512
MLP_EPOCHS = 30
BOOTSTRAP_RESAMPLES = 1000
CI_LEVEL = 0.95

BOARD_RUNGS: tuple[str, ...] = ("0", "1", "1w", "1o", "2", "3")
PROBED_RUNGS: tuple[str, ...] = ("0", "1", "1w", "1o", "2")
#: A line property's short ladder: ``e``, its width control, ``[ACT]``.
LINE_RUNGS: tuple[str, ...] = ("1", "1w", "2")
PROBE_TYPES: tuple[str, ...] = ("linear", "mlp")


# ── folds, the width control, scores ────────────────────────────────────


def fold_assignment(groups, folds: int = FOLDS, seed: int = SEED) -> np.ndarray:
    """Each item's fold, so that no group falls on both sides of any fold.

    Groups are shuffled once by a seeded permutation of their sorted names and
    dealt round-robin, which keeps the fold sizes even in groups rather than in
    items and makes the assignment a function of the group names alone.
    """
    groups = np.asarray(groups, dtype=object)
    unique = sorted(set(groups.tolist()))
    order = np.random.default_rng(seed).permutation(len(unique))
    fold_of = {unique[index]: rank % folds for rank, index in enumerate(order)}
    return np.array([fold_of[g] for g in groups], dtype=np.int64)


def width_vector(text: str, width: int, seed: int = SEED) -> np.ndarray:
    """Rung 1w's stand-in for ``e``: a fixed random vector per text.

    A function of the text and the seed only, so one text gets one vector across
    records and runs. Under folds split by text it carries nothing from one text
    to another, which is what makes it measure width alone.
    """
    rng = np.random.default_rng([seed, zlib.crc32(text.encode("utf-8"))])
    return rng.standard_normal(width).astype(np.float32)


def score(kind: str, y: np.ndarray, prediction: np.ndarray,
          weight: np.ndarray | None = None) -> float:
    """AUC for a yes/no target, R² for an amount; NaN where undefined."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(prediction)
    y, prediction = y[ok], prediction[ok]
    weight = None if weight is None else np.asarray(weight, dtype=np.float64)[ok]
    if y.size < 2:
        return float("nan")
    if kind == "binary":
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, prediction, sample_weight=weight))
    w = np.ones_like(y) if weight is None else weight
    mean = np.average(y, weights=w)
    total = float((w * (y - mean) ** 2).sum())
    if total <= 0:
        return float("nan")
    return float(1.0 - (w * (y - prediction) ** 2).sum() / total)


#: Standardized features are clipped here, as the embedding probes clip theirs:
#: one extreme value in a test fold would otherwise decide its score alone.
FEATURE_CLIP = 5.0


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return (np.clip((train - mean) / std, -FEATURE_CLIP, FEATURE_CLIP),
            np.clip((test - mean) / std, -FEATURE_CLIP, FEATURE_CLIP))


# ── the two probes ──────────────────────────────────────────────────────


def linear_oof(X: np.ndarray, y: np.ndarray, folds: np.ndarray, kind: str,
               weight: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold predictions of the linear probe, features standardized per fold.

    Logistic regression (L2, C = 1) for a yes/no target, ridge (α = 1) for an
    amount. A training fold holding one class predicts that class's base rate.
    """
    from sklearn.linear_model import LogisticRegression, Ridge

    out = np.full(len(y), np.nan)
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        Xtr, Xte = _standardize(X[train], X[test])
        w = None if weight is None else weight[train]
        if kind == "binary":
            if len(np.unique(y[train])) < 2:
                out[test] = float(y[train].mean()) if train.any() else 0.5
                continue
            model = LogisticRegression(C=LINEAR_C, max_iter=1000)
            model.fit(Xtr, y[train], sample_weight=w)
            out[test] = model.decision_function(Xte)
        else:
            model = Ridge(alpha=RIDGE_ALPHA).fit(Xtr, y[train], sample_weight=w)
            out[test] = model.predict(Xte)
    return out


def mlp_oof(X: np.ndarray, y: np.ndarray, folds: np.ndarray, kind: str,
            weight: np.ndarray | None = None, *, hidden: int = MLP_HIDDEN,
            epochs: int = MLP_EPOCHS, device: str | None = None) -> np.ndarray:
    """Out-of-fold predictions of the MLP probe.

    Two hidden layers of ``hidden`` units with ReLU, AdamW at a fixed learning
    rate and weight decay, fixed batch and epoch counts. Tests pass a small
    ``hidden`` and the CPU; every real run uses the defaults on the GPU.
    """
    import torch
    from torch import nn

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = np.full(len(y), np.nan)
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        Xtr, Xte = _standardize(X[train], X[test])
        ytr = y[train].astype(np.float64)
        if kind == "binary" and len(np.unique(ytr)) < 2:
            out[test] = float(ytr.mean()) if train.any() else 0.5
            continue
        y_mean, y_std = (0.0, 1.0) if kind == "binary" else (ytr.mean(), ytr.std() or 1.0)
        torch.manual_seed(SEED + int(fold))
        layers: list[nn.Module] = []
        width = X.shape[1]
        for _ in range(MLP_LAYERS):
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        layers.append(nn.Linear(width, 1))
        model = nn.Sequential(*layers).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY,
        )
        xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
        yt = torch.tensor((ytr - y_mean) / y_std, dtype=torch.float32, device=device)
        wt = torch.tensor(
            np.ones(len(ytr)) if weight is None else weight[train],
            dtype=torch.float32, device=device,
        )
        generator = torch.Generator().manual_seed(SEED + int(fold))
        for _ in range(epochs):
            order = torch.randperm(len(yt), generator=generator).to(device)
            for start in range(0, len(yt), MLP_BATCH):
                index = order[start : start + MLP_BATCH]
                logits = model(xt[index]).squeeze(-1)
                if kind == "binary":
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        logits, yt[index], weight=wt[index])
                else:
                    loss = (wt[index] * (logits - yt[index]) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            pred = model(torch.tensor(Xte, dtype=torch.float32, device=device))
        prediction = pred.squeeze(-1).cpu().numpy() * y_std + y_mean
        if kind != "binary":
            # An amount is predicted within one span of what training saw: on a
            # small fold the network can extrapolate by orders of magnitude.
            low, high = ytr.min(), ytr.max()
            span = max(high - low, 1.0)
            prediction = np.clip(prediction, low - span, high + span)
        out[test] = prediction
    return out


def probe(X, y, folds, kind, weight=None, probe_type="linear", **mlp) -> np.ndarray:
    if probe_type == "linear":
        return linear_oof(X, y, folds, kind, weight)
    return mlp_oof(X, y, folds, kind, weight, **mlp)


# ── the share ───────────────────────────────────────────────────────────


def share(rung0: float, rung1: float, rung3: float,
          min_gap: float = MIN_GAP) -> float | None:
    """``(rung 1 − rung 0) / (rung 3 − rung 0)``, or None under the minimum gap.

    Below the gap the denominator is near zero and the ratio means nothing, so
    it is not reported at all rather than reported as noise.
    """
    if not all(np.isfinite(v) for v in (rung0, rung1, rung3)):
        return None
    gap = rung3 - rung0
    if gap < min_gap:
        return None
    return float((rung1 - rung0) / gap)


def bootstrap_share(kind, y, oof0, oof1, pred3, groups, weight=None, *,
                    resamples: int = BOOTSTRAP_RESAMPLES,
                    seed: int = SEED) -> tuple[float, float] | None:
    """The share's 95% interval over groups resampled with replacement.

    Resampled by group rather than by item, because items of one text are not
    independent: a text seen in fifty records would otherwise count fifty times.
    A resample whose gap falls under the minimum is skipped, so the interval is
    over resamples where the share is defined.
    """
    groups = np.asarray(groups, dtype=object)
    unique = sorted(set(groups.tolist()))
    members = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        drawn = rng.integers(0, len(unique), len(unique))
        index = np.concatenate([members[unique[d]] for d in drawn])
        w = None if weight is None else weight[index]
        value = share(score(kind, y[index], oof0[index], w),
                      score(kind, y[index], oof1[index], w),
                      score(kind, y[index], pred3[index], w))
        if value is not None:
            values.append(value)
    if not values:
        return None
    tail = (1.0 - CI_LEVEL) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


@dataclass
class LadderInput:
    """One target's items in one stratum: features per rung and the labels."""

    kind: str
    y: np.ndarray
    groups: np.ndarray
    features: dict[str, np.ndarray]
    rung3: np.ndarray | None = None
    weight: np.ndarray | None = None
    #: Why rung 3 is missing, where it is.
    rung3_note: str | None = None
    #: Per-layer trunk outputs, when ``--per-layer`` asked for them.
    layers: list[np.ndarray] = field(default_factory=list)


def run_ladder(data: LadderInput, *, rungs=PROBED_RUNGS, mlp_options=None,
               bootstrap: int = BOOTSTRAP_RESAMPLES) -> dict:
    """Every probed rung's two scores, rung 3's, and each probe type's share.

    The share pairs rung 0 and rung 1 of the **same** probe type with rung 3;
    the two probe types are never mixed in one computation.
    """
    mlp_options = mlp_options or {}
    folds = fold_assignment(data.groups)
    result: dict = {"n": int(len(data.y)), "rungs": {}, "share": None}
    oof: dict[tuple[str, str], np.ndarray] = {}
    for rung in rungs:
        if rung not in data.features:
            continue
        X = data.features[rung]
        scores = {}
        for probe_type in PROBE_TYPES:
            prediction = probe(X, data.y, folds, data.kind, data.weight, probe_type,
                               **(mlp_options if probe_type == "mlp" else {}))
            oof[(rung, probe_type)] = prediction
            scores[probe_type] = score(data.kind, data.y, prediction, data.weight)
        result["rungs"][rung] = scores
    if data.rung3_note is not None:
        result["rung3_unavailable"] = data.rung3_note
    if data.rung3 is not None:
        rung3 = score(data.kind, data.y, data.rung3, data.weight)
        result["rungs"]["3"] = {"model": rung3}
        shares = {}
        for probe_type in PROBE_TYPES:
            if ("0", probe_type) not in oof or ("1", probe_type) not in oof:
                continue
            value = share(result["rungs"]["0"][probe_type],
                          result["rungs"]["1"][probe_type], rung3)
            if value is None:
                shares[probe_type] = None
                continue
            ci = bootstrap_share(
                data.kind, data.y, oof[("0", probe_type)], oof[("1", probe_type)],
                data.rung3, data.groups, data.weight, resamples=bootstrap,
            ) if bootstrap else None
            shares[probe_type] = {"value": value, "ci": list(ci) if ci else None}
        if any(v is not None for v in shares.values()):
            result["share"] = shares
    if data.layers:
        result["per_layer"] = []
        for depth, X in enumerate(data.layers):
            row = {"layer": depth}
            for probe_type in PROBE_TYPES:
                prediction = probe(X, data.y, folds, data.kind, data.weight, probe_type,
                                   **(mlp_options if probe_type == "mlp" else {}))
                row[probe_type] = score(data.kind, data.y, prediction, data.weight)
            result["per_layer"].append(row)
    return result


# ── rung features from one forward pass ─────────────────────────────────


#: Targets whose rung 3 reads the verdict head. Feature 023's trainer never
#: called the verdict loss, so on its checkpoints that head sits at its initial
#: weights; a score read from it measures where a random projection happens to
#: point, not what the model learned.
VERDICT_TARGETS: frozenset[str] = frozenset({"affordable", "fires"})

#: Why rung 3 is missing for a verdict target on such a checkpoint.
UNTRAINED_VERDICT_NOTE = "verdict head never trained by this checkpoint's trainer"


def verdict_head_trained(checkpoint) -> bool:
    """Did the trainer that wrote ``checkpoint`` train the verdict head?

    The gen-2 trainer wires the verdict loss and records its training settings
    in every checkpoint; feature 023's trainer recorded none and never trained
    the head.
    """
    return bool(getattr(checkpoint, "training_settings", None))


def _readout(target: str, outputs, verdict, slot: int, row: int, *,
             verdict_trained: bool = True) -> float:
    """Rung 3: the head's own prediction for one item.

    NaN where the head this target reads was never trained, which
    :func:`ladder_inputs` turns into an unavailable rung 3 rather than a score.
    """
    if target in VERDICT_TARGETS and not verdict_trained:
        return float("nan")
    import torch

    from effects.domain.effect_model import (
        DURATIONS,
        FIELD_SLICES,
        GATE_INDEX,
        VERDICT_BITS,
        ZONE_OUTCOMES,
    )

    vector = outputs[row, slot]
    gate = torch.sigmoid(vector[GATE_INDEX])

    def field(name):
        start, end = FIELD_SLICES[name]
        return vector[start:end]

    if target == "affected":
        value = gate
    elif target == "dies":
        value = gate * torch.softmax(field("zone_outcome"), -1)[ZONE_OUTCOMES.index("died")]
    elif target == "damage_taken":
        value = gate * torch.exp(field("damage_taken")[0])
    elif target == "pt_until_end_of_turn":
        value = torch.softmax(field("pt_duration"), -1)[DURATIONS.index("end_of_turn")]
    elif target == "may_block":
        value = torch.sigmoid(field("blocker_legal")[0])
    elif target in ("legal_target", "legal_target_protected"):
        # The gate, not the `target_legal` field: training sets that field only
        # on legal targets, always to 1, so on an illegal entity its output was
        # never supervised and ranks nothing. Legality is learned through the
        # gate, which a decision record raises exactly on its legal targets.
        value = gate
    elif target == "affordable":
        value = torch.sigmoid(verdict[row, VERDICT_BITS.index("affordable")])
    elif target == "fires":
        value = torch.sigmoid(verdict[row, -1])
    else:
        raise KeyError(target)
    return float(value)


@dataclass
class ExtractedItem:
    """One probe item's features at every rung, and its label."""

    target: str
    label: float
    group: str
    features: dict[str, np.ndarray]
    rung3: float
    layers: list[np.ndarray] = field(default_factory=list)


def _value_vector(text: str | None) -> np.ndarray:
    from effects.domain.value_targets import VALUE_TARGET_NAMES, value_targets

    width = len(VALUE_TARGET_NAMES)
    if not text:
        return np.zeros(2 * width, dtype=np.float32)
    parsed = value_targets(text)
    values = np.array([v if m else 0.0 for v, m in zip(parsed.values, parsed.mask)])
    return np.concatenate([values, np.array(parsed.mask, dtype=np.float64)]).astype(
        np.float32)


def extract_features(probe_model, records: list, items_by_record: dict, *,
                     batch_size: int = 32, per_layer: bool = False,
                     act_hidden: dict | None = None) -> list[ExtractedItem]:
    """Every item's rung features, from one batched forward pass over ``records``.

    ``items_by_record`` maps a record id to its probe items (``labels.RecordItem``).
    ``act_hidden``, when given, accumulates the trunk's ``[ACT]`` output by
    acting text — the line ladder's rung 2 — from the same pass.
    """
    import torch

    from effects.domain.effect_head_input import SlotKind

    model, encoder, batcher = probe_model.model, probe_model.encoder, probe_model.batcher
    e_dim = probe_model.e_dim
    zero_e = np.zeros(e_dim, dtype=np.float32)
    verdict_trained = verdict_head_trained(probe_model.checkpoint)
    captured: list = []
    hooks = []
    if per_layer:
        for layer in model.trunk.layers:
            hooks.append(layer.register_forward_hook(
                lambda _m, _i, output: captured.append(output.detach())))
    extracted: list[ExtractedItem] = []
    try:
        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                chunk = records[start : start + batch_size]
                captured.clear()
                batch, surfaces = batcher.build(chunk, encoder)
                hidden = model(**batch)
                outputs = model.per_entity(hidden)
                verdict = model.verdict(hidden)
                encoded = batcher.encoded
                matrix = (encoded.matrix.float().cpu().numpy()
                          if encoded.matrix is not None else None)
                texts = encoded.texts
                slot_feats = torch.cat(
                    [batch["slot_features"][k] for k in sorted(batch["slot_features"])],
                    dim=-1,
                ).float().cpu().numpy()
                hidden_np = hidden.float().cpu().numpy()
                layer_np = [c.float().cpu().numpy() for c in captured]

                partner_texts = {}
                for record in chunk:
                    for item in items_by_record.get(record.record_id, ()):
                        if item.partner is not None:
                            text = batcher.text_of(item.partner)
                            if text and text not in partner_texts:
                                partner_texts[text] = batcher.sidecars.line_for(
                                    item.partner)
                partner_e = {}
                if partner_texts:
                    rows, extra = batcher.encode_texts(partner_texts, encoder)
                    if extra is not None:
                        extra = extra.float().cpu().numpy()
                        partner_e = {t: extra[r] for t, r in rows.items()}

                for b, (record, surface) in enumerate(zip(chunk, surfaces)):
                    items = items_by_record.get(record.record_id, ())
                    slots = surface.slots
                    card_slot, rows_of = {}, defaultdict(list)
                    player_slot = {}
                    for index, slot in enumerate(slots):
                        if slot.kind is SlotKind.CARD:
                            card_slot[slot.entity_id] = index
                        elif slot.kind is SlotKind.PLAYER:
                            player_slot[slot.player_id] = index
                        elif slot.kind is SlotKind.ABILITY and isinstance(slot.e, int):
                            rows_of[slot.entity_id].append(slot.e)
                    act = slots[1].e
                    act_text = texts[act] if isinstance(act, int) else None
                    if act_hidden is not None and act_text is not None:
                        act_hidden.setdefault(act_text, []).append(hidden_np[b, 1])
                    if not items:
                        continue

                    def e_of(row):
                        return matrix[row] if matrix is not None else zero_e

                    def pooled(entity_id, width=False):
                        rows = rows_of.get(entity_id, [])
                        if not rows:
                            return zero_e
                        if width:
                            return np.mean([width_vector(texts[r], e_dim) for r in rows],
                                           axis=0)
                        return np.mean([e_of(r) for r in rows], axis=0)

                    owner = record.actor_player
                    actor_cards = [i for e, i in card_slot.items()
                                   if _controller(record, e) == owner]
                    other_cards = [i for e, i in card_slot.items()
                                   if _controller(record, e) != owner]
                    players = [player_slot.get(owner)] + [
                        i for p, i in player_slot.items() if p != owner][:1]
                    feature_width = slot_feats.shape[-1]

                    def raw(index):
                        if index is None:
                            return np.zeros(feature_width, dtype=np.float32)
                        return slot_feats[b, index]

                    def mean_raw(indices):
                        if not indices:
                            return np.zeros(feature_width, dtype=np.float32)
                        return slot_feats[b, indices].mean(axis=0)

                    base = np.concatenate([
                        raw(0), raw(1), raw(players[0]),
                        raw(players[1] if len(players) > 1 else None),
                        mean_raw(actor_cards), mean_raw(other_cards),
                    ])
                    acting_e = e_of(act) if isinstance(act, int) else zero_e
                    acting_w = width_vector(act_text, e_dim) if act_text else zero_e
                    for item in items:
                        slot = card_slot.get(item.entity) if item.entity else 1
                        if slot is None:
                            continue
                        entity_raw = raw(slot if item.entity else None)
                        rung0 = np.concatenate([base, entity_raw])
                        partner_text = (batcher.text_of(item.partner)
                                        if item.partner is not None else None)
                        if partner_text is not None:
                            p_e = partner_e.get(partner_text, zero_e)
                            p_w = width_vector(partner_text, e_dim)
                        elif item.partner_entity is not None:
                            p_e = pooled(item.partner_entity)
                            p_w = pooled(item.partner_entity, width=True)
                        else:
                            p_e = p_w = zero_e
                        own_e = pooled(item.entity) if item.entity else zero_e
                        own_w = pooled(item.entity, width=True) if item.entity else zero_e
                        features = {
                            "0": rung0,
                            "1": np.concatenate([rung0, acting_e, own_e, p_e]),
                            "1w": np.concatenate([rung0, acting_w, own_w, p_w]),
                            "1o": np.concatenate([rung0, _value_vector(act_text),
                                                  _value_vector(partner_text)]),
                            "2": hidden_np[b, slot],
                        }
                        extracted.append(ExtractedItem(
                            target=item.target, label=item.label, group=item.group,
                            features={k: v.astype(np.float32) for k, v in features.items()},
                            rung3=_readout(item.target, outputs, verdict, slot, b,
                                           verdict_trained=verdict_trained),
                            layers=[layer[b, slot] for layer in layer_np],
                        ))
    finally:
        for hook in hooks:
            hook.remove()
    return extracted


def _controller(record, entity_id: str) -> str | None:
    entity = record.state.entity(entity_id)
    return None if entity is None else entity.controller


def ladder_inputs(extracted: list[ExtractedItem], kinds: dict[str, str]) -> dict:
    """Group extracted items into one :class:`LadderInput` per target."""
    by_target: dict[str, list[ExtractedItem]] = defaultdict(list)
    for item in extracted:
        by_target[item.target].append(item)
    out = {}
    for target, items in by_target.items():
        rung3 = np.array([i.rung3 for i in items], dtype=np.float64)
        # Every item NaN means the head this target reads was never trained:
        # no rung 3, so no share, rather than a score from random weights.
        untrained = bool(np.isnan(rung3).all())
        out[target] = LadderInput(
            kind=kinds[target],
            y=np.array([i.label for i in items], dtype=np.float64),
            groups=np.array([i.group for i in items], dtype=object),
            features={rung: np.stack([i.features[rung] for i in items])
                      for rung in PROBED_RUNGS},
            rung3=None if untrained else rung3,
            rung3_note=UNTRAINED_VERDICT_NOTE if untrained else None,
            layers=([np.stack([i.layers[d] for i in items])
                     for d in range(len(items[0].layers))]
                    if items[0].layers else []),
        )
    return out
