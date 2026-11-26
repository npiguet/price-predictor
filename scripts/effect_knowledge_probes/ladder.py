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

from effect_knowledge_probes import readout
from effects.domain.probe_readouts import item_score

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

PROBED_RUNGS: tuple[str, ...] = ("0", "1", "1w", "1o", "2")
PROBE_MODELS: tuple[str, ...] = ("linear", "mlp")


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
    records and runs. Where the folds split by acting text it carries nothing from
    one text to another and measures width alone. Card-level targets fold by the
    target's card instead, so there one acting text can sit on both sides of a
    split and the vector names it, letting a probe learn that text's usual answer.
    """
    rng = np.random.default_rng([seed, zlib.crc32(text.encode("utf-8"))])
    return rng.standard_normal(width).astype(np.float32)


#: An item's score, AUC for a yes/no target and R² for an amount; NaN where
#: undefined. The package's, so rung 3 scores as the control score does.
score = item_score


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



@dataclass(frozen=True)
class ProbeProblem:
    """One probe fit's data: features, labels, fold per item, kind and sample weights."""

    X: np.ndarray
    y: np.ndarray
    folds: np.ndarray
    kind: str
    weight: np.ndarray | None = None


@dataclass(frozen=True)
class _Fold:
    """One fold's standardized train and test features and its training labels."""

    Xtr: np.ndarray
    Xte: np.ndarray
    ytr: np.ndarray
    weight: np.ndarray | None
    number: int
    kind: str


@dataclass(frozen=True)
class MlpOptions:
    """What :func:`mlp_oof` takes beside the problem."""

    hidden: int = MLP_HIDDEN
    epochs: int = MLP_EPOCHS
    device: str | None = None


def _out_of_fold(problem: ProbeProblem, fit) -> np.ndarray:
    """Each item's prediction from ``fit`` on the folds that do not hold it."""
    out = np.full(len(problem.y), np.nan)
    for fold in np.unique(problem.folds):
        train, test = problem.folds != fold, problem.folds == fold
        Xtr, Xte = _standardize(problem.X[train], problem.X[test])
        weight = None if problem.weight is None else problem.weight[train]
        out[test] = fit(_Fold(Xtr, Xte, problem.y[train], weight, int(fold), problem.kind))
    return out


def _single_class_rate(fold: _Fold) -> float | None:
    """The base rate of a yes/no training fold holding one class, else None."""
    if fold.kind != "binary" or len(np.unique(fold.ytr)) >= 2:
        return None
    return float(fold.ytr.mean()) if len(fold.ytr) else 0.5


def _linear_fit(fold: _Fold):
    from sklearn.linear_model import LogisticRegression, Ridge

    constant = _single_class_rate(fold)
    if constant is not None:
        return constant
    if fold.kind == "binary":
        model = LogisticRegression(C=LINEAR_C, max_iter=1000)
        model.fit(fold.Xtr, fold.ytr, sample_weight=fold.weight)
        return model.decision_function(fold.Xte)
    model = Ridge(alpha=RIDGE_ALPHA).fit(fold.Xtr, fold.ytr, sample_weight=fold.weight)
    return model.predict(fold.Xte)


def linear_oof(problem: ProbeProblem) -> np.ndarray:
    """Out-of-fold predictions of the linear probe, features standardized per fold.


    Logistic regression (L2, C = 1) for a yes/no target, ridge (α = 1) for an
    amount. A training fold holding one class predicts that class's base rate.
    """
    return _out_of_fold(problem, _linear_fit)


def _target_scale(kind: str, ytr: np.ndarray) -> tuple[float, float]:
    """The mean and spread an amount is trained in; a yes/no target is not scaled."""
    if kind == "binary":
        return 0.0, 1.0
    return ytr.mean(), ytr.std() or 1.0


class _Trainer:
    """One MLP and its optimizer, trained on one fold's tensors."""

    def __init__(self, model, tensors, kind: str) -> None:
        import torch

        self.model = model
        self.xt, self.yt, self.wt = tensors
        self.kind = kind
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY,
        )

    def run(self, epochs: int, generator, device: str) -> None:
        import torch

        for _ in range(epochs):
            order = torch.randperm(len(self.yt), generator=generator).to(device)
            for start in range(0, len(self.yt), MLP_BATCH):
                self._step(order[start : start + MLP_BATCH])

    def _loss(self, logits, index):
        from torch import nn

        if self.kind == "binary":
            return nn.functional.binary_cross_entropy_with_logits(
                logits, self.yt[index], weight=self.wt[index])
        return (self.wt[index] * (logits - self.yt[index]) ** 2).mean()

    def _step(self, index) -> None:
        logits = self.model(self.xt[index]).squeeze(-1)
        loss = self._loss(logits, index)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def _clip_to_training(prediction: np.ndarray, ytr: np.ndarray) -> np.ndarray:
    """An amount is predicted within one span of what training saw: on a small
    fold the network can extrapolate by orders of magnitude."""
    low, high = ytr.min(), ytr.max()
    span = max(high - low, 1.0)
    return np.clip(prediction, low - span, high + span)


class _MlpFit:
    """The MLP probe's fit on one fold, on one device."""

    def __init__(self, options: MlpOptions) -> None:
        import torch

        self.options = options
        self.device = options.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _network(self, width: int):
        from torch import nn

        layers: list[nn.Module] = []
        for _ in range(MLP_LAYERS):
            layers += [nn.Linear(width, self.options.hidden), nn.ReLU()]
            width = self.options.hidden
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers).to(self.device)

    def _tensor(self, values):
        import torch

        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _tensors(self, fold: _Fold, ytr: np.ndarray, scale: tuple[float, float]):
        weight = np.ones(len(ytr)) if fold.weight is None else fold.weight
        return (self._tensor(fold.Xtr), self._tensor((ytr - scale[0]) / scale[1]),
                self._tensor(weight))

    def __call__(self, fold: _Fold):
        import torch

        constant = _single_class_rate(fold)
        if constant is not None:
            return constant
        ytr = fold.ytr.astype(np.float64)
        scale = _target_scale(fold.kind, ytr)
        torch.manual_seed(SEED + fold.number)
        model = self._network(fold.Xtr.shape[1])
        trainer = _Trainer(model, self._tensors(fold, ytr, scale), fold.kind)
        generator = torch.Generator().manual_seed(SEED + fold.number)
        trainer.run(self.options.epochs, generator, self.device)
        with torch.no_grad():
            pred = model(self._tensor(fold.Xte))
        prediction = pred.squeeze(-1).cpu().numpy() * scale[1] + scale[0]
        return prediction if fold.kind == "binary" else _clip_to_training(prediction, ytr)


def mlp_oof(problem: ProbeProblem, options: MlpOptions = MlpOptions()) -> np.ndarray:
    """Out-of-fold predictions of the MLP probe.


    Two hidden layers of ``hidden`` units with ReLU, AdamW at a fixed learning
    rate and weight decay, fixed batch and epoch counts. Tests pass a small
    ``hidden`` and the CPU; every real run uses the defaults on the GPU.
    """
    return _out_of_fold(problem, _MlpFit(options))


def probe(problem: ProbeProblem, probe_type: str = "linear",
          mlp_options: MlpOptions = MlpOptions()) -> np.ndarray:
    """The linear or the MLP probe on ``problem``; ``mlp_options`` go to the MLP only."""
    if probe_type == "linear":
        return linear_oof(problem)
    return mlp_oof(problem, mlp_options)


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


@dataclass(frozen=True)
class BootstrapOptions:
    """What :func:`bootstrap_share` takes beside the columns."""

    resamples: int = BOOTSTRAP_RESAMPLES
    seed: int = SEED


@dataclass(frozen=True)
class BootstrapColumns:
    """The columns a share is read from, one item per row: the labels, rung 0's and
    rung 1's out-of-fold predictions, rung 3's prediction, each item's group and
    the sample weights. :meth:`share_at` reads the share on a resample."""

    kind: str
    y: np.ndarray
    oof0: np.ndarray
    oof1: np.ndarray
    pred3: np.ndarray
    groups: np.ndarray
    weight: np.ndarray | None = None

    def share_at(self, index: np.ndarray) -> float | None:
        w = None if self.weight is None else self.weight[index]
        y = self.y[index]
        return share(score(self.kind, y, self.oof0[index], w),
                     score(self.kind, y, self.oof1[index], w),
                     score(self.kind, y, self.pred3[index], w))


def _resampled_shares(sample: BootstrapColumns, options: BootstrapOptions) -> list[float]:
    """The share on each group resample where it is defined."""
    groups = np.asarray(sample.groups, dtype=object)
    unique = sorted(set(groups.tolist()))
    members = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    rng = np.random.default_rng(options.seed)
    values = []
    for _ in range(options.resamples):
        drawn = rng.integers(0, len(unique), len(unique))
        value = sample.share_at(np.concatenate([members[unique[d]] for d in drawn]))
        if value is not None:
            values.append(value)
    return values


def bootstrap_share(columns: BootstrapColumns,
                    options: BootstrapOptions = BootstrapOptions()) -> tuple[float, float] | None:
    """The share's 95% interval over groups resampled with replacement.

    Resampled by group rather than by item, because items of one text are not
    independent: a text seen in fifty records would otherwise count fifty times.
    A resample whose gap falls under the minimum is skipped, so the interval is
    over resamples where the share is defined.
    """
    values = _resampled_shares(columns, options)
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
    #: Per-layer trunk outputs, when ``--per-layer`` asked for them.
    layers: list[np.ndarray] = field(default_factory=list)


class _Fitter:
    """Both probe types on one target's folds, and their scores."""

    def __init__(self, data: LadderInput, mlp_options: MlpOptions) -> None:
        self.data = data
        self.folds = fold_assignment(data.groups)
        self.mlp_options = mlp_options

    def predictions(self, X: np.ndarray) -> dict[str, np.ndarray]:
        data = self.data
        problem = ProbeProblem(X, data.y, self.folds, data.kind, data.weight)
        return {probe_type: probe(problem, probe_type, self.mlp_options)
                for probe_type in PROBE_MODELS}

    def scores(self, predictions: dict[str, np.ndarray]) -> dict[str, float]:
        data = self.data
        return {probe_type: score(data.kind, data.y, prediction, data.weight)
                for probe_type, prediction in predictions.items()}


def _share_entry(data: LadderInput, scores: tuple, oofs: tuple, bootstrap: int) -> dict | None:
    """One probe type's share and its interval; None under the minimum gap.

    ``scores`` are rung 0's, rung 1's and rung 3's; ``oofs`` rung 0's and rung 1's
    out-of-fold predictions.
    """
    value = share(*scores)
    if value is None:
        return None
    ci = bootstrap_share(
        BootstrapColumns(data.kind, data.y, oofs[0], oofs[1], data.rung3, data.groups,
                         data.weight),
        BootstrapOptions(resamples=bootstrap),
    ) if bootstrap else None
    return {"value": value, "ci": list(ci) if ci else None}


def _shares(data: LadderInput, result: dict, oof: dict, bootstrap: int) -> dict:
    """Each probe type's share, pairing rungs 0 and 1 of that type with rung 3."""
    rung3 = result["rungs"]["3"]["model"]
    shares = {}
    for probe_type in PROBE_MODELS:
        if "0" not in oof or "1" not in oof:
            continue
        scores = (result["rungs"]["0"][probe_type], result["rungs"]["1"][probe_type], rung3)
        oofs = (oof["0"][probe_type], oof["1"][probe_type])
        shares[probe_type] = _share_entry(data, scores, oofs, bootstrap)
    return shares


def _per_layer(data: LadderInput, fitter: _Fitter) -> list[dict]:
    return [{"layer": depth, **fitter.scores(fitter.predictions(X))}
            for depth, X in enumerate(data.layers)]


def run_ladder(data: LadderInput, *, rungs=PROBED_RUNGS,
               mlp_options: MlpOptions = MlpOptions(),
               bootstrap: int = BOOTSTRAP_RESAMPLES) -> dict:
    """Every probed rung's two scores, rung 3's, and each probe type's share.

    The share pairs rung 0 and rung 1 of the **same** probe type with rung 3;
    the two probe types are never mixed in one computation.
    """
    fitter = _Fitter(data, mlp_options)
    oof = {rung: fitter.predictions(data.features[rung])
           for rung in rungs if rung in data.features}
    result: dict = {"n": int(len(data.y)), "rungs": {}, "share": None}
    result["rungs"] = {rung: fitter.scores(predictions) for rung, predictions in oof.items()}
    if data.rung3 is not None:
        result["rungs"]["3"] = {"model": score(data.kind, data.y, data.rung3, data.weight)}
        shares = _shares(data, result, oof, bootstrap)
        if any(v is not None for v in shares.values()):
            result["share"] = shares
    if data.layers:
        result["per_layer"] = _per_layer(data, fitter)
    return result


# ── rung features from one forward pass ─────────────────────────────────


@dataclass(frozen=True)
class ItemAt:
    """Where an item sits in a batch: its ``slot`` and its ``row``."""

    slot: int
    row: int


def _readout(target: str, outputs, verdict, at: ItemAt) -> float:
    """Rung 3: the head's own prediction for one item (:mod:`readout`)."""
    return readout.read(target, outputs, verdict, readout.ReadAt(at.row, at.slot))


@dataclass
class ExtractedItem:
    """One probe item's features at every rung, and its label."""

    target: str
    label: float
    group: str
    features: dict[str, np.ndarray]
    rung3: float
    layers: list[np.ndarray] = field(default_factory=list)
    #: The record the item came from, so a check can group items by record or game.
    record_id: str = ""


def _value_vector(text: str | None) -> np.ndarray:
    from effects.domain.value_targets import VALUE_TARGET_NAMES, value_targets

    width = len(VALUE_TARGET_NAMES)
    if not text:
        return np.zeros(2 * width, dtype=np.float32)
    parsed = value_targets(text)
    values = np.array([v if m else 0.0 for v, m in zip(parsed.values, parsed.mask)])
    return np.concatenate([values, np.array(parsed.mask, dtype=np.float64)]).astype(
        np.float32)


@dataclass
class BatchOutputs:
    """One forward pass over a chunk, on the host: what every rung reads."""

    #: ``(batch, slots, width)`` of the slots' raw input features, every kind.
    slot_features: np.ndarray
    #: The trunk's output per slot.
    hidden: np.ndarray
    #: The batch's encoded ability matrix, one row per text; None when empty.
    e_matrix: np.ndarray | None
    texts: list
    #: The partners' ``e`` by text, encoded beside the batch.
    partner_e: dict


@dataclass
class BatchContext:
    """What every record of a batch is read through: the pass, the batcher, ``e``'s width."""

    outputs: BatchOutputs
    batcher: object
    e_dim: int


def _slot_index(surface) -> tuple[dict[str, int], dict[str, list[int]]]:
    """A surface's slot of each player, and its ability rows by entity."""
    from effects.domain.input_layout import SlotKind

    player_slot: dict[str, int] = {}
    rows_of: dict[str, list[int]] = defaultdict(list)
    for index, slot in enumerate(surface.slots):
        if slot.kind is SlotKind.PLAYER:
            player_slot[slot.player_id] = index
        elif slot.kind is SlotKind.ABILITY and slot.e_row is not None:
            rows_of[slot.entity_id].append(slot.e_row)
    return player_slot, rows_of


class ItemFeatureBuilder:
    """The rung features of the items of one record of a batch (FR-080).

    Built once per record from its surface, then asked once per item:
    :meth:`rung0` the trunk's raw inputs with every ``e`` zeroed, :meth:`rung1`
    those plus the acting, own and partner ``e``, :meth:`rung1w` the same with
    each ``e`` a fixed random vector per text, :meth:`rung1o` rung 0 plus the
    script values the item depends on.
    """

    def __init__(self, record, surface, row: int, context: BatchContext) -> None:
        self.record = record
        self.row = row
        self.outputs = context.outputs
        self.batcher = context.batcher
        self.e_dim = context.e_dim
        self.zero_e = np.zeros(self.e_dim, dtype=np.float32)
        self.card_slot = readout.card_slots(surface)
        self.player_slot, self.rows_of = _slot_index(surface)
        self.act = surface.slots[1].e_row
        self.act_text = self.outputs.texts[self.act] if self.act is not None else None
        self._base = self._board()

    # ── what every item shares ──────────────────────────────────────────

    def _raw(self, index: int | None) -> np.ndarray:
        if index is None:
            return np.zeros(self.outputs.slot_features.shape[-1], dtype=np.float32)
        return self.outputs.slot_features[self.row, index]

    def _mean_raw(self, indices: list[int]) -> np.ndarray:
        if not indices:
            return np.zeros(self.outputs.slot_features.shape[-1], dtype=np.float32)
        return self.outputs.slot_features[self.row, indices].mean(axis=0)

    def _board(self) -> np.ndarray:
        """The global and ``[ACT]`` slots, both players and both sides' mean card."""
        owner = self.record.actor_player
        actor_cards = [i for e, i in self.card_slot.items()
                       if _controller(self.record, e) == owner]
        other_cards = [i for e, i in self.card_slot.items()
                       if _controller(self.record, e) != owner]
        players = [self.player_slot.get(owner)] + [
            i for p, i in self.player_slot.items() if p != owner][:1]
        return np.concatenate([
            self._raw(0), self._raw(1), self._raw(players[0]),
            self._raw(players[1] if len(players) > 1 else None),
            self._mean_raw(actor_cards), self._mean_raw(other_cards),
        ])

    def _e(self, row: int) -> np.ndarray:
        matrix = self.outputs.e_matrix
        return matrix[row] if matrix is not None else self.zero_e

    def _pooled(self, entity_id: str, *, width: bool = False) -> np.ndarray:
        rows = self.rows_of.get(entity_id, [])
        if not rows:
            return self.zero_e
        if width:
            return np.mean([width_vector(self.outputs.texts[r], self.e_dim) for r in rows],
                           axis=0)
        return np.mean([self._e(r) for r in rows], axis=0)

    def _partner_text(self, item) -> str | None:
        return self.batcher.text_of(item.partner) if item.partner is not None else None

    # ── per item ────────────────────────────────────────────────────────

    def slot(self, item) -> int | None:
        """The slot an item is read at: its entity's card, or ``[ACT]``."""
        return self.card_slot.get(item.entity) if item.entity else 1

    def rung0(self, item) -> np.ndarray:
        own = self._raw(self.slot(item) if item.entity else None)
        return np.concatenate([self._base, own])

    def _partner(self, item, *, width: bool) -> np.ndarray:
        text = self._partner_text(item)
        if text is not None:
            return (width_vector(text, self.e_dim) if width
                    else self.outputs.partner_e.get(text, self.zero_e))
        if item.partner_entity is not None:
            return self._pooled(item.partner_entity, width=width)
        return self.zero_e

    def rung1(self, item) -> np.ndarray:
        acting = self._e(self.act) if self.act is not None else self.zero_e
        own = self._pooled(item.entity) if item.entity else self.zero_e
        return np.concatenate([self.rung0(item), acting, own,
                               self._partner(item, width=False)])

    def rung1w(self, item) -> np.ndarray:
        acting = width_vector(self.act_text, self.e_dim) if self.act_text else self.zero_e
        own = self._pooled(item.entity, width=True) if item.entity else self.zero_e
        return np.concatenate([self.rung0(item), acting, own,
                               self._partner(item, width=True)])

    def rung1o(self, item) -> np.ndarray:
        return np.concatenate([self.rung0(item), _value_vector(self.act_text),
                               _value_vector(self._partner_text(item))])


def _note_partner_line(batcher, item, lines: dict) -> None:
    if item.partner is None:
        return
    text = batcher.text_of(item.partner)
    if text and text not in lines:
        lines[text] = batcher.sidecars.line_or_none(item.partner)


def _partner_e(batcher, encoder, chunk, items_by_record) -> dict:
    """The ``e`` of every partner the chunk's items name, encoded beside the batch."""
    partner_lines: dict = {}
    for record in chunk:
        for item in items_by_record.get(record.record_id, ()):
            _note_partner_line(batcher, item, partner_lines)
    if not partner_lines:
        return {}
    rows, extra = batcher.encode_texts(partner_lines, encoder)
    if extra is None:
        return {}
    extra = extra.float().cpu().numpy()
    return {text: extra[row] for text, row in rows.items()}


@dataclass(frozen=True)
class ExtractionOptions:
    """What :func:`extract_features` takes beside the model, records and items."""

    batch_size: int = 32
    per_layer: bool = False
    act_hidden: dict | None = None


class _LayerCapture:
    """Forward hooks on the trunk's layers, when ``--per-layer`` asked for them."""

    def __init__(self, model, enabled: bool) -> None:
        self.captured: list = []
        self._hooks = []
        if enabled:
            for layer in model.trunk.layers:
                self._hooks.append(layer.register_forward_hook(
                    lambda _m, _i, output: self.captured.append(output.detach())))

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()

    def arrays(self) -> list[np.ndarray]:
        return [c.float().cpu().numpy() for c in self.captured]


@dataclass
class _ChunkPass:
    """One chunk's forward pass: the host arrays, the surfaces and the head's reads."""

    host: BatchOutputs
    surfaces: list
    outputs: object
    verdict: object
    layers: list[np.ndarray]


class _Extractor:
    """Reads the items of a batched forward pass, chunk by chunk."""

    def __init__(self, probe_model, items_by_record: dict, options: ExtractionOptions) -> None:
        self.probe_model = probe_model
        self.items_by_record = items_by_record
        self.options = options
        self.capture = _LayerCapture(probe_model.model, options.per_layer)

    def forward(self, chunk) -> _ChunkPass:
        import torch

        model, encoder = self.probe_model.model, self.probe_model.encoder
        batcher = self.probe_model.batcher
        self.capture.captured.clear()
        batch, surfaces, encoded = batcher.build(chunk, encoder)
        hidden = model(batch)
        outputs = model.per_entity(hidden)
        verdict = model.verdict(hidden)
        host = BatchOutputs(
            slot_features=torch.cat(
                [batch.slot_features[k] for k in sorted(batch.slot_features)],
                dim=-1,
            ).float().cpu().numpy(),
            hidden=hidden.float().cpu().numpy(),
            e_matrix=(encoded.matrix.float().cpu().numpy()
                      if encoded.matrix is not None else None),
            texts=encoded.texts,
            partner_e=_partner_e(batcher, encoder, chunk, self.items_by_record),
        )
        return _ChunkPass(host, surfaces, outputs, verdict, self.capture.arrays())

    def chunk_items(self, chunk) -> list[ExtractedItem]:
        done = self.forward(chunk)
        extracted: list[ExtractedItem] = []
        for b, (record, surface) in enumerate(zip(chunk, done.surfaces)):
            extracted += self._record_items(record, surface, b, done)
        return extracted

    def _record_items(self, record, surface, b: int, done: _ChunkPass) -> list[ExtractedItem]:
        context = BatchContext(done.host, self.probe_model.batcher, self.probe_model.e_dim)
        builder = ItemFeatureBuilder(record, surface, b, context)
        act_hidden = self.options.act_hidden
        if act_hidden is not None and builder.act_text is not None:
            act_hidden.setdefault(builder.act_text, []).append(done.host.hidden[b, 1])
        found = []
        for item in self.items_by_record.get(record.record_id, ()):
            slot = builder.slot(item)
            if slot is not None:
                found.append(self._item(builder, item, (slot, b), done))
        return found

    @staticmethod
    def _item(builder, item, at: tuple[int, int], done: _ChunkPass) -> ExtractedItem:
        slot, b = at
        features = {
            "0": builder.rung0(item),
            "1": builder.rung1(item),
            "1w": builder.rung1w(item),
            "1o": builder.rung1o(item),
            "2": done.host.hidden[b, slot],
        }
        return ExtractedItem(
            target=item.target, label=item.label, group=item.group,
            features={k: v.astype(np.float32) for k, v in features.items()},
            rung3=_readout(item.target, done.outputs, done.verdict, ItemAt(slot, b)),
            layers=[layer[b, slot] for layer in done.layers],
            record_id=item.record_id,
        )


def extract_features(probe_model, records: list, items_by_record: dict,
                     options: ExtractionOptions = ExtractionOptions()) -> list[ExtractedItem]:
    """Every item's rung features, from one batched forward pass over ``records``.

    ``items_by_record`` maps a record id to its probe items (``labels.RecordItem``).
    ``options.act_hidden``, when given, accumulates the trunk's ``[ACT]`` output by
    acting text — the line ladder's rung 2 — from the same pass.
    """
    import torch

    extractor = _Extractor(probe_model, items_by_record, options)
    extracted: list[ExtractedItem] = []
    try:
        with torch.no_grad():
            for start in range(0, len(records), options.batch_size):
                extracted += extractor.chunk_items(records[start : start + options.batch_size])
    finally:
        extractor.capture.remove()
    return extracted


def _controller(record, entity_id: str) -> str | None:
    entity = record.state.entity(entity_id)
    return None if entity is None else entity.controller


def _layers_of(items: list[ExtractedItem]) -> list[np.ndarray]:
    """The items' per-layer outputs, stacked per layer; none when not captured."""
    if not items[0].layers:
        return []
    return [np.stack([i.layers[d] for i in items]) for d in range(len(items[0].layers))]


def _ladder_input(kind: str, items: list[ExtractedItem]) -> LadderInput:
    return LadderInput(
        kind=kind,
        y=np.array([i.label for i in items], dtype=np.float64),
        groups=np.array([i.group for i in items], dtype=object),
        features={rung: np.stack([i.features[rung] for i in items])
                  for rung in PROBED_RUNGS},
        rung3=np.array([i.rung3 for i in items], dtype=np.float64),
        layers=_layers_of(items),
    )


def ladder_inputs(extracted: list[ExtractedItem], kinds: dict[str, str]) -> dict:
    """Group extracted items into one :class:`LadderInput` per target."""
    by_target: dict[str, list[ExtractedItem]] = defaultdict(list)
    for item in extracted:
        by_target[item.target].append(item)
    return {target: _ladder_input(kinds[target], items) for target, items in by_target.items()}
