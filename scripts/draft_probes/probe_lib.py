"""Shared instrument for the draft-agent behaviour probes.

Everything in ``scripts/draft_probes/`` is inference-only: it replays recorded
picks from a ``drafts.jsonl`` corpus through a frozen draft-agent checkpoint and
reads the policy logits, optionally after **intervening** on the state (erasing a
token block, rewriting the CONTEXT numbers, transplanting another seat's POOL).

Three pieces:

- :class:`CardTable` — one shared ``name -> row`` embedding table over the
  corpus, built lazily from the ``.npz`` cache, plus the reverse ``row -> name``
  the analyses need (``draft_pick_states`` keeps only rows).
- :func:`iter_corpus_states` — every recorded pick of every seat matching a
  label filter, as :class:`PickSample` (a ``RawPickState`` plus provenance).
- :class:`PolicyRunner` — batched forward passes returning per-PACK logits, with
  an optional per-sample :class:`Intervention`.

The state walk is :func:`draft.application.draft_pick_states.iter_seat_pick_states`,
the same one both trainers use, so a replayed state is bit-identical to the one
the policy saw when it made the pick.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from draft.application.draft_pick_states import (  # noqa: E402
    RawPickState,
    iter_seat_pick_states,
)
from draft.domain.draft_geometry import DraftGeometry, DraftRecord  # noqa: E402
from draft.domain.draft_state import (  # noqa: E402
    NUM_TYPES,
    TYPE_PACK,
    TYPE_PASSED,
    TYPE_POOL,
    TYPE_TAKEN,
)
from draft.infrastructure.draft_record_io import record_from_dict  # noqa: E402
from sealed.infrastructure.converted_card_locator import (  # noqa: E402
    ConvertedCardLocator,
)

DEFAULT_CARDS_PATH = Path("output/cardsfolder-512")

__all__ = [
    "TYPE_PACK",
    "TYPE_PASSED",
    "TYPE_POOL",
    "TYPE_TAKEN",
    "CardTable",
    "Intervention",
    "PickSample",
    "PolicyRunner",
    "collapse_block",
    "drop_types",
    "mean_substitute",
    "retype",
    "transplant_pool",
    "iter_corpus_records",
    "iter_corpus_states",
    "load_agent",
    "set_context",
    "zero_recency",
]


class CardTable:
    """Lazy ``name -> row`` embedding table shared by every state in a run."""

    def __init__(self, cards_path: Path = DEFAULT_CARDS_PATH) -> None:
        self._locator = ConvertedCardLocator(cards_path)
        self._rows: list[np.ndarray] = []
        self.names: list[str] = []
        self._index: dict[str, int | None] = {}
        self.dim: int | None = None
        self.missing: set[str] = set()

    def index(self, name: str) -> int | None:
        """Row of ``name``, loading its ``.npz`` once; ``None`` if it has none."""
        cached = self._index.get(name, -1)
        if cached != -1:
            return cached
        emb = self._locator.load_embedding(name)
        if emb is None:
            self._index[name] = None
            self.missing.add(name)
            return None
        if self.dim is None:
            self.dim = int(emb.shape[0])
        row = len(self._rows)
        self._rows.append(np.asarray(emb, dtype=np.float32))
        self.names.append(name)
        self._index[name] = row
        return row

    def add_vector(self, vec: np.ndarray, name: str = "<synthetic>") -> int:
        """Stage a synthetic card vector (a block mean, an edited card) as a row."""
        if self.dim is None:
            self.dim = int(vec.shape[0])
        row = len(self._rows)
        self._rows.append(np.asarray(vec, dtype=np.float32))
        self.names.append(name)
        return row

    def matrix(self) -> np.ndarray:
        return np.stack(self._rows).astype(np.float32)

    def name(self, row: int) -> str:
        return self.names[row]


@dataclass(slots=True)
class PickSample:
    """One recorded pick: its state, plus who made it and in what draft."""

    draft_id: str
    seat: int
    label: str
    set_code: str
    state: RawPickState
    pack_names: list[str]
    taken_name: str

    @property
    def pack_rows(self) -> np.ndarray:
        """Absolute token indices of the PACK block."""
        return np.flatnonzero(self.state.type_idx == TYPE_PACK)

    @property
    def target(self) -> int:
        """Index into ``pack_names`` of the card actually taken, or -1."""
        try:
            return self.pack_names.index(self.taken_name)
        except ValueError:
            return -1


def iter_corpus_records(
    path: Path, limit: int | None = None,
) -> Iterator[DraftRecord]:
    """Stream ``drafts.jsonl``, tolerating a trailing partial line."""
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue  # trailing partial write
            yield record_from_dict(payload)
            n += 1
            if limit is not None and n >= limit:
                return


def _sample_from_state(
    record: DraftRecord, table: CardTable, seat_idx: int, label: str,
    set_code: str, state,
) -> PickSample | None:
    """Turn a :func:`build_state` ``DraftState`` into a tensorizable sample."""
    idx: list[int] = []
    types: list[int] = []
    packs_ago: list[int] = []
    pick_ago: list[int] = []
    pack_names: list[str] = []
    action_position = -1
    taken_name = state.pack_actions[state.target_index]
    for card in state.cards:
        row = table.index(card.name)
        if row is None:
            continue
        if card.token_type == TYPE_PACK:
            if card.name == taken_name:
                action_position = len(idx)
            pack_names.append(card.name)
        idx.append(row)
        types.append(card.token_type)
        packs_ago.append(card.packs_ago)
        pick_ago.append(card.pick_ago)
    if not pack_names:
        return None
    raw = RawPickState(
        card_idx=np.asarray(idx, dtype=np.int32),
        type_idx=np.asarray(types, dtype=np.int8),
        packs_ago=np.asarray(packs_ago, dtype=np.int8),
        pick_ago=np.asarray(pick_ago, dtype=np.int8),
        pack_number=state.pack_number,
        pick_number=state.pick_number,
        action_position=action_position,
    )
    return PickSample(
        draft_id=record.draft_id, seat=seat_idx, label=label,
        set_code=set_code, state=raw, pack_names=pack_names,
        taken_name=taken_name,
    )


def iter_corpus_states(
    path: Path,
    table: CardTable,
    *,
    labels: Sequence[str] | None = None,
    limit_drafts: int | None = None,
    require_deck: bool = False,
    clocks: Sequence[tuple[int, int]] | None = None,
) -> Iterator[PickSample]:
    """Yield every recorded pick of every seat whose agent label is in ``labels``.

    ``clocks`` restricts the walk to a list of ``(pack, pick)`` pairs. Building a
    state is the dominant cost of a probe that only wants a few of them, so a
    probe reading one pick per booster passes that pick rather than filtering 45
    states per seat afterwards.

    States come from :func:`draft.domain.draft_state.build_state`, the full-record
    oracle the live :class:`OnlineDraftStateTracker` is pinned to. That matters:
    replaying an argmax corpus through its own checkpoint reproduces 100 % of the
    recorded picks this way, against 97.7 % through the *trainers'* walk
    (:func:`iter_seat_pick_states`), which freezes a ``TAKEN`` card's recency at
    the moment it left the pack instead of recomputing it — the two disagree only
    when a card name recurs in a later pack.
    """
    from draft.domain.draft_state import build_state

    wanted = set(labels) if labels is not None else None
    for record in iter_corpus_records(path, limit_drafts):
        geo = DraftGeometry.from_record(record)
        set_code = record.boosters[0].set_code if record.boosters else ""
        for seat_idx, seat in enumerate(record.seats):
            if wanted is not None and seat.agent not in wanted:
                continue
            if require_deck and not seat.deck:
                continue
            wanted_clocks = (
                [(p, i) for p in range(1, geo.packs + 1)
                 for i in range(1, geo.pack_size + 1)]
                if clocks is None else
                [(p, i) for p, i in clocks
                 if 1 <= p <= geo.packs and 1 <= i <= geo.pack_size]
            )
            for p, i in wanted_clocks:
                state = build_state(record, geo, seat_idx, p, i)
                sample = _sample_from_state(
                    record, table, seat_idx, seat.agent, set_code, state,
                )
                if sample is not None:
                    yield sample


Intervention = Callable[[PickSample], RawPickState]
"""Rewrites one sample's state before the forward pass. Must not mutate in place."""


def drop_types(*types: int) -> Intervention:
    """Erase whole token blocks (the PACK block is never droppable)."""
    drop = set(types)
    if TYPE_PACK in drop:
        raise ValueError("cannot drop the PACK block: it holds the actions")

    def apply(sample: PickSample) -> RawPickState:
        st = sample.state
        keep = ~np.isin(st.type_idx, list(drop))
        return replace(
            st,
            card_idx=st.card_idx[keep],
            type_idx=st.type_idx[keep],
            packs_ago=st.packs_ago[keep],
            pick_ago=st.pick_ago[keep],
            action_position=int(np.flatnonzero(keep).tolist().index(
                st.action_position
            )) if st.action_position >= 0 and keep[st.action_position] else -1,
        )

    return apply


def transplant_pool(donor_of: Callable[[PickSample], RawPickState]) -> Intervention:
    """Swap in another seat's POOL block, keeping the pack and the clock fixed.

    The causal form of every "does its pool change its pick" question. A donor
    drawn from the *same* ``(pack, pick)`` keeps the state on-manifold: pool size
    is a deterministic function of the clock, so a same-clock donor arrives with
    the right number of cards and the right recency spread, and only *which*
    cards the seat owns has changed.
    """

    def apply(sample: PickSample) -> RawPickState:
        st = sample.state
        donor = donor_of(sample)
        keep = st.type_idx != TYPE_POOL
        d = donor.type_idx == TYPE_POOL
        n_new = int(d.sum())
        pos = np.flatnonzero(keep).tolist()
        return replace(
            st,
            card_idx=np.concatenate([donor.card_idx[d], st.card_idx[keep]]),
            type_idx=np.concatenate([donor.type_idx[d], st.type_idx[keep]]),
            packs_ago=np.concatenate([donor.packs_ago[d], st.packs_ago[keep]]),
            pick_ago=np.concatenate([donor.pick_ago[d], st.pick_ago[keep]]),
            action_position=(
                n_new + pos.index(st.action_position)
                if st.action_position >= 0 and keep[st.action_position] else -1
            ),
        )

    return apply


def retype(src: int, dst: int) -> Intervention:
    """Relabel a block's token type, keeping every card and count untouched.

    The count-preserving way to ask whether the model uses a distinction rather
    than the cards themselves: POOL→TAKEN scrubs ownership, TAKEN→PASSED scrubs
    fate, and neither changes what the trunk averages over.
    """

    def apply(sample: PickSample) -> RawPickState:
        st = sample.state
        types = st.type_idx.copy()
        types[st.type_idx == src] = dst
        return replace(st, type_idx=types)

    return apply


def mean_substitute(
    *types: int, mean_row: int | Callable[[PickSample], int],
) -> Intervention:
    """Blank a block's card identities, keeping its token count and types.

    Deleting a block also changes how many tokens the trunk averages over, which
    moves every logit on its own. Substituting each of the block's cards for the
    corpus-mean card vector erases *which* cards they are and nothing else — the
    same design the scorer study's ablations use.
    """
    blank = set(types)
    if TYPE_PACK in blank:
        raise ValueError("cannot blank the PACK block: it holds the actions")
    getter = mean_row if callable(mean_row) else (lambda _s: mean_row)

    def apply(sample: PickSample) -> RawPickState:
        st = sample.state
        idx = st.card_idx.copy()
        idx[np.isin(st.type_idx, list(blank))] = getter(sample)
        return replace(st, card_idx=idx)

    return apply


def collapse_block(
    token_type: int, mean_row: int | Callable[[PickSample], int],
) -> Intervention:
    """Replace a whole block with ONE token carrying a summary vector.

    Under a mean-pooling trunk a block reaches the PACK logits only through its
    average, so collapsing it to a single token holding that average should
    barely matter if the average is all the model reads. Pass a constant row for
    a blank (corpus-mean) collapse, or a per-sample callable for the block's own
    mean — the difference between "the model ignores my pool" and "the model
    reads only my pool's average".
    """
    getter = mean_row if callable(mean_row) else (lambda _s: mean_row)

    def apply(sample: PickSample) -> RawPickState:
        st = sample.state
        sel = st.type_idx == token_type
        if not sel.any():
            return st
        first = int(np.flatnonzero(sel)[0])
        keep = ~sel
        keep[first] = True
        idx = st.card_idx.copy()
        idx[first] = getter(sample)
        pos = np.flatnonzero(keep).tolist()
        return replace(
            st,
            card_idx=idx[keep],
            type_idx=st.type_idx[keep],
            packs_ago=st.packs_ago[keep],
            pick_ago=st.pick_ago[keep],
            action_position=(
                pos.index(st.action_position)
                if st.action_position >= 0 and keep[st.action_position] else -1
            ),
        )

    return apply


def set_context(pack_number: int | None = None,
                pick_number: int | None = None) -> Intervention:
    """Rewrite the CONTEXT token's pack/pick index, leaving the cards alone."""

    def apply(sample: PickSample) -> RawPickState:
        st = sample.state
        return replace(
            st,
            pack_number=st.pack_number if pack_number is None else pack_number,
            pick_number=st.pick_number if pick_number is None else pick_number,
        )

    return apply


def zero_recency() -> Intervention:
    """Set every card's ``packs_ago`` / ``pick_ago`` to 0 (all cards look fresh)."""

    def apply(sample: PickSample) -> RawPickState:
        st = sample.state
        return replace(
            st,
            packs_ago=np.zeros_like(st.packs_ago),
            pick_ago=np.zeros_like(st.pick_ago),
        )

    return apply


def load_agent(path: Path, device=None):
    """Load a checkpoint into an eval-mode model on ``device``; return (model, cfg)."""
    import torch

    from draft.domain.draft_agent_model import DraftAgentModel
    from draft.infrastructure.draft_agent_store import DraftAgentStore

    ckpt = DraftAgentStore().load_checkpoint(path)
    model = DraftAgentModel(ckpt.config)
    model.load_state_dict(ckpt.model_state_dict)
    model.eval()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, ckpt.config


class PolicyRunner:
    """Batched policy forward passes over :class:`PickSample` states."""

    def __init__(self, model, table: CardTable, device=None,
                 batch_size: int = 64) -> None:
        import torch

        self._torch = torch
        self._model = model
        self._table = table
        self._device = device or next(model.parameters()).device
        self._batch_size = batch_size
        self._emb: "torch.Tensor | None" = None

    def _embedding_matrix(self):
        torch = self._torch
        if self._emb is None or self._emb.shape[0] != len(self._table.names):
            self._emb = torch.from_numpy(self._table.matrix()).to(self._device)
        return self._emb

    def context_vectors(
        self,
        samples: Sequence[PickSample],
        intervention: Intervention | None = None,
    ) -> np.ndarray:
        """The trunk's CONTEXT-token output, one row per sample.

        The policy head never reads this token — it reads the PACK positions —
        and gen-3/gen-4 carry the critic head untrained, so in the deployed agent
        the CONTEXT token's output is a summary nothing consumes. That makes it
        the natural place to ask what the trunk has worked out about the draft:
        a probe fitted on it is reading the model's own state, not its choice.
        """
        return np.concatenate(
            self._run(samples, intervention, want="context"), axis=0)

    def logits(
        self,
        samples: Sequence[PickSample],
        intervention: Intervention | None = None,
    ) -> list[np.ndarray]:
        """Per-sample PACK logits, in PACK-token order, one array per sample."""
        return self._run(samples, intervention, want="logits")

    def _run(self, samples, intervention, want):
        torch = self._torch
        emb = self._embedding_matrix()
        model = self._model
        out: list[np.ndarray] = []
        for start in range(0, len(samples), self._batch_size):
            chunk = samples[start:start + self._batch_size]
            states = [
                intervention(s) if intervention is not None else s.state
                for s in chunk
            ]
            n_max = max(int(st.card_idx.shape[0]) for st in states)
            b = len(states)
            card_idx = np.zeros((b, n_max), dtype=np.int64)
            type_idx = np.zeros((b, n_max), dtype=np.int64)
            packs_ago = np.zeros((b, n_max), dtype=np.int64)
            pick_ago = np.zeros((b, n_max), dtype=np.int64)
            mask = np.zeros((b, n_max), dtype=bool)
            pack_no = np.zeros(b, dtype=np.int64)
            pick_no = np.zeros(b, dtype=np.int64)
            pack_rows: list[np.ndarray] = []
            for i, st in enumerate(states):
                n = int(st.card_idx.shape[0])
                card_idx[i, :n] = st.card_idx
                type_idx[i, :n] = st.type_idx
                packs_ago[i, :n] = st.packs_ago
                pick_ago[i, :n] = st.pick_ago
                mask[i, :n] = True
                pack_no[i] = st.pack_number
                pick_no[i] = st.pick_number
                pack_rows.append(np.flatnonzero(st.type_idx == TYPE_PACK))

            dev = self._device
            t = lambda a: torch.from_numpy(a).to(dev)  # noqa: E731
            with torch.no_grad():
                card_emb = emb[t(card_idx)]
                if want == "logits":
                    logits, _ = model(
                        card_emb, t(type_idx), t(packs_ago), t(pick_ago),
                        t(mask), t(pack_no), t(pick_no),
                    )
                    logits = logits.float().cpu().numpy()
                    for i, rows in enumerate(pack_rows):
                        out.append(logits[i, rows])
                    continue
                out.append(self._trunk_context(
                    card_emb, t(type_idx), t(packs_ago), t(pick_ago),
                    t(mask), t(pack_no), t(pick_no),
                ))
        return out

    def _trunk_context(self, card_emb, type_idx, packs_ago, pick_ago,
                       card_mask, pack_number, pick_number) -> np.ndarray:
        """Re-run ``DraftAgentModel.forward`` up to the CONTEXT token's output."""
        torch = self._torch
        import torch.nn.functional as F

        model = self._model
        type_onehot = F.one_hot(type_idx, NUM_TYPES).to(card_emb.dtype)
        card_feat = torch.cat(
            [card_emb, type_onehot, model.packs_ago_embed(packs_ago),
             model.pick_ago_embed(pick_ago)], dim=-1,
        )
        x = model.input_projection(card_feat)
        ctx = (model.pack_number_embed(pack_number)
               + model.pick_number_embed(pick_number)).unsqueeze(1)
        x = torch.cat([ctx, x], dim=1)
        ctx_real = torch.zeros(x.size(0), 1, dtype=torch.bool, device=x.device)
        pad = torch.cat([ctx_real, ~card_mask], dim=1)
        for sab in model.sab_layers:
            x = sab(x, attn_padding_mask=pad)
        return x[:, 0, :].float().cpu().numpy()
