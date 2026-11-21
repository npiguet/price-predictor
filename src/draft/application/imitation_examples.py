"""Gen-1 training examples: one ``(draft, seat, pack, pick)`` typed-token example
per pick (FR-030), the corpus loader that builds them over a shared embedding
table, the draft-disjoint split (FR-035) and the critic's target
standardization (FR-032).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np

from draft.application.draft_pick_states import iter_seat_pick_states
from draft.application.draft_training_common import leave_one_out_rewards
from draft.application.imitation_log import _fmt_dur, _log
from draft.domain.draft_geometry import DraftGeometry, DraftRecord
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

RANDOM_SEED = 42  # hardcoded (FR-035): governs init, the draft-disjoint split.
_MISSING_WARN_CAP = 20


@dataclass(slots=True)
class DraftExample:
    """One ``(draft, seat, pack, pick)`` training example (data-model §2).

    To keep millions of examples in RAM, the card embeddings are *not* stored
    here: ``card_idx`` holds int32 rows into a shared embedding table (see
    ``_Loader.table``) materialized per-batch in :func:`_collate`. The small
    per-token int arrays are downcast (type/packs_ago/pick_ago all fit in int8);
    ``pack_mask`` is derived from ``type_idx == TYPE_PACK`` rather than stored.
    """

    draft_index: int             # groups examples for the draft-disjoint split
    card_idx: np.ndarray         # (N,) int32, rows into the shared table
    type_idx: np.ndarray         # (N,) int8
    packs_ago: np.ndarray        # (N,) int8
    pick_ago: np.ndarray         # (N,) int8
    pack_number: int
    pick_number: int
    imitation_active: bool
    target_token: int            # absolute index of the taken PACK token, or -1
    critic_active: bool
    critic_target: float         # raw (pre-standardization) pod-relative reward

    @property
    def n_tokens(self) -> int:
        return int(self.card_idx.shape[0])


class _Loader:
    """Builds ``DraftExample`` objects from a corpus, dropping missing-``.npz`` picks."""

    def __init__(self, locator: ConvertedCardLocator, whitelist: set[str]) -> None:
        self._locator = locator
        self._whitelist = whitelist
        self.embedding_dim: int | None = None
        self.max_packs = 0
        self.max_pack_size = 0
        self._missing: set[str] = set()
        self._missing_total = 0
        # Shared embedding table: each distinct card is loaded once; examples
        # keep only int rows into it (mirrors picker_pools' PreparedPool).
        self._table_rows: list[np.ndarray] = []
        self._name_to_idx: dict[str, int | None] = {}

    def _card_index(self, name: str) -> int | None:
        """Row index of ``name`` in the shared table, loading it once; None if no .npz.

        Counts each distinct missing card once (in ``_missing`` / ``_missing_total``)
        on first lookup.
        """
        idx = self._name_to_idx.get(name, -1)
        if idx != -1:
            return idx
        emb = self._locator.load_embedding(name)
        if emb is None:
            self._name_to_idx[name] = None
            if len(self._missing) < _MISSING_WARN_CAP:
                self._missing.add(name)
            self._missing_total += 1
            return None
        if self.embedding_dim is None:
            self.embedding_dim = int(emb.shape[0])
        new_idx = len(self._table_rows)
        self._table_rows.append(np.asarray(emb, dtype=np.float32))
        self._name_to_idx[name] = new_idx
        return new_idx

    def table(self) -> np.ndarray:
        """The shared ``(num_unique_cards, embedding_dim)`` embedding table."""
        if not self._table_rows:
            return np.zeros((0, self.embedding_dim or 1), dtype=np.float32)
        return np.stack(self._table_rows).astype(np.float32)

    def build(self, records: list[DraftRecord]) -> list[DraftExample]:
        examples: list[DraftExample] = []
        total = len(records)
        log_every = max(200, total // 20)  # ~20 progress lines on a big corpus
        start = time.monotonic()
        last_ri, last_t = 0, start  # windowed-rate anchors
        _log(f"Building examples from {total} drafts...")
        for ri, record in enumerate(records, start=1):
            geo = DraftGeometry.from_record(record)
            self.max_packs = max(self.max_packs, geo.packs)
            self.max_pack_size = max(self.max_pack_size, geo.pack_size)
            rewards = leave_one_out_rewards(record)
            for seat_idx, seat in enumerate(record.seats):
                whitelisted = seat.agent in self._whitelist
                critic_active = rewards[seat_idx] is not None
                if not whitelisted and not critic_active:
                    continue  # neither head learns from this seat
                self._emit_seat(
                    record, geo, ri - 1, seat_idx,
                    whitelisted, critic_active, rewards[seat_idx], examples,
                )
            if ri % log_every == 0 or ri == total:
                now = time.monotonic()
                # Windowed (since-last-log) rate: the cold first chunk reads most
                # of the ~28k unique card .npz files off disk, so a lifetime
                # average would understate the (much faster) cached steady state.
                interval = now - last_t
                rate = (ri - last_ri) / interval if interval > 0 else 0.0
                eta = (total - ri) / rate if rate > 0 else 0.0
                last_ri, last_t = ri, now
                _log(
                    f"  loaded {ri}/{total} drafts -> {len(examples)} examples "
                    f"({rate:.0f} drafts/s, ETA {_fmt_dur(eta)})"
                )
        if self._missing_total:
            sample = ", ".join(sorted(self._missing))
            _log(
                f"Dropped {self._missing_total} distinct cards with no .npz "
                f"embedding (e.g. {sample})"
            )
        return examples

    def _emit_seat(
        self,
        record: DraftRecord,
        geo: DraftGeometry,
        draft_index: int,
        seat: int,
        whitelisted: bool,
        critic_active: bool,
        reward: float | None,
        examples: list[DraftExample],
    ) -> None:
        """Emit one ``DraftExample`` per pick for ``seat``.

        Thin adapter over the shared per-pick state walk
        (:func:`draft.application.draft_pick_states.iter_seat_pick_states`),
        which yields the typed-token state at each pick; this wraps each in a
        ``DraftExample`` with the imitation target + critic label. The walk is
        shared with the gen-2 RL loader so the two cannot drift (research
        §Overlapping vocabulary).
        """
        reward_val = float(reward) if reward is not None else 0.0
        for state in iter_seat_pick_states(
            geo, record.boosters, seat, self._card_index,
        ):
            examples.append(DraftExample(
                draft_index=draft_index,
                card_idx=state.card_idx,
                type_idx=state.type_idx,
                packs_ago=state.packs_ago,
                pick_ago=state.pick_ago,
                pack_number=state.pack_number,
                pick_number=state.pick_number,
                imitation_active=whitelisted and state.action_position >= 0,
                target_token=state.action_position,
                critic_active=critic_active,
                critic_target=reward_val,
            ))


def split_draft_disjoint(
    examples: list[DraftExample], val_fraction: float, seed: int = RANDOM_SEED,
) -> tuple[list[DraftExample], list[DraftExample]]:
    """Partition examples by ``draft_index`` so a draft is entirely train or val.

    Distinct draft indices are shuffled with ``seed`` and the first
    ``val_fraction`` form the held-out set (FR-035).
    """
    ids = sorted({ex.draft_index for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_fraction)
    val_ids = set(ids[:n_val])
    val = [ex for ex in examples if ex.draft_index in val_ids]
    train = [ex for ex in examples if ex.draft_index not in val_ids]
    return train, val


def critic_standardization(train: list[DraftExample]) -> tuple[float, float]:
    """Mean/std of critic targets over the training split (FR-032)."""
    targets = [ex.critic_target for ex in train if ex.critic_active]
    if not targets:
        return 0.0, 1.0
    arr = np.asarray(targets, dtype=np.float64)
    std = float(arr.std())
    return float(arr.mean()), std if std > 1e-8 else 1.0
