"""Live policy inference glue: a pick-request → a chosen card (data-model §2.5).

Wraps one loaded draft-agent checkpoint and turns each external-seat
``PickRequest`` into the card the trained policy would take: reconstruct the
typed-token state with a per-seat :class:`OnlineDraftStateTracker`, embed it
(dropping un-embeddable cards as in training), run one policy forward, mask the
logits to the ``PACK`` positions, and pick (argmax or seeded temperature
sample). Raises :class:`PickFault` when no genuine pick is possible.

Performance (Principle VIII): the model and locator load once; ``.npz``
embeddings are memoized by the locator; tensorization builds one tiny example on
the model device and reads back a single chosen index (one host↔device sync per
pick). Strict single-pick synchrony makes batching inapplicable (plan §Perf).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from draft.domain.draft_state import TYPE_PACK
from draft.domain.online_draft_state import OnlineDraftStateTracker

if TYPE_CHECKING:  # avoid a runtime import cycle with generate_draft_data
    from draft.application.generate_draft_data import PickRequest


class PickFault(Exception):
    """A condition preventing a model seat's genuine pick (data-model §3).

    Carrying it out abandons the in-flight draft; the policy never substitutes a
    Forge/first-card/random choice (SC-002).
    """


class AgentPickService:
    """One loaded checkpoint serving live picks for its mix label."""

    def __init__(
        self,
        checkpoint_path: Path,
        locator,
        *,
        device=None,
        pick_mode: str = "argmax",
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> None:
        import torch

        from draft.domain.draft_agent_model import DraftAgentModel
        from draft.infrastructure.draft_agent_store import DraftAgentStore

        ckpt = DraftAgentStore().load_checkpoint(checkpoint_path)
        model = DraftAgentModel(ckpt.config)
        model.load_state_dict(ckpt.model_state_dict)
        # This constructor owns the model it just built, so it also owns its
        # mode and placement (``from_model`` deliberately does not).
        model.eval()
        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model.to(device)
        self._init(
            model, ckpt.config, locator, device=device,
            pick_mode=pick_mode, temperature=temperature, seed=seed,
        )

    @classmethod
    def from_model(
        cls,
        model,
        config,
        locator,
        *,
        device,
        pick_mode: str = "argmax",
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> "AgentPickService":
        """Wrap an **already-constructed** model instead of loading a checkpoint.

        The service holds the model by reference, so a weight update is visible
        to the very next pick with no copy and no explicit push — which is how
        the online trainer's learner seats are guaranteed to be piloted by the
        current in-training policy (spec 021 FR-012).

        The **caller owns the model**: this does not call ``.eval()``,
        ``.train()``, or ``.to(device)``. The online trainer switches the model
        to ``eval()`` for generation and ``train()`` for the update, and would
        be fighting the service if the service touched mode too.
        """
        service = cls.__new__(cls)
        service._init(
            model, config, locator, device=device,
            pick_mode=pick_mode, temperature=temperature, seed=seed,
        )
        return service

    def _init(
        self,
        model,
        config,
        locator,
        *,
        device,
        pick_mode: str,
        temperature: float,
        seed: int | None,
    ) -> None:
        import torch

        if pick_mode not in ("argmax", "sample"):
            raise ValueError(f"unknown pick mode {pick_mode!r}")

        self._torch = torch
        self._model = model
        self._config = config
        self._device = device
        self._locator = locator
        self._pick_mode = pick_mode
        self._temperature = temperature
        # Per-service RNG so two services seeded identically reproduce picks
        # (SC-007); a CPU generator keeps the single per-pick sync on the readout.
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)
        self._trackers: dict[tuple[str, int], OnlineDraftStateTracker] = {}

    @property
    def config(self):
        return self._config

    def reset_draft(self, draft_id: str) -> None:
        """Discard all trackers for a finished/abandoned draft."""
        for key in [k for k in self._trackers if k[0] == draft_id]:
            del self._trackers[key]

    def pick(self, request: "PickRequest") -> str:
        """Return the policy's chosen card for ``request``; raise on any fault.

        Any tracker/policy error is surfaced as a :class:`PickFault` so the
        supervisor abandons the draft rather than crashing the run (data-model
        §3); a persistent deterministic fault then trips the consecutive-fault
        auto-abort (FR-016).
        """
        try:
            return self._pick(request)
        except PickFault:
            raise
        except Exception as exc:  # tracker/tensor/forward error → abandon, don't crash
            raise PickFault(f"policy error: {exc}") from exc

    def _pick(self, request: "PickRequest") -> str:
        if request.pick_number > self._config.P:
            raise PickFault(
                f"pick_number {request.pick_number} exceeds checkpoint capacity "
                f"P={self._config.P}"
            )
        tracker = self._trackers.get((request.draft_id, request.seat))
        if tracker is None:
            tracker = OnlineDraftStateTracker()
            self._trackers[(request.draft_id, request.seat)] = tracker

        state = tracker.observe(
            pack_number=request.pack_number,
            pick_number=request.pick_number,
            pack=request.pack,
            pod_size=request.pod_size,
        )
        chosen = self._select(state)
        tracker.commit(chosen)
        return chosen

    def _select(self, state) -> str:
        import numpy as np

        torch = self._torch
        dim = self._config.embedding_dim

        embs: list[np.ndarray] = []
        type_ids: list[int] = []
        packs_ago: list[int] = []
        pick_ago: list[int] = []
        pack_positions: list[tuple[int, str]] = []  # (row index, card name)
        for card in state.cards:
            emb = self._locator.load_embedding(card.name)
            if emb is None:
                continue  # dropping individual un-embeddable cards is normal
            row = len(embs)
            embs.append(np.asarray(emb, dtype=np.float32))
            type_ids.append(card.token_type)
            packs_ago.append(card.packs_ago)
            pick_ago.append(card.pick_ago)
            if card.token_type == TYPE_PACK:
                pack_positions.append((row, card.name))

        if not pack_positions:
            raise PickFault("every legal action is un-embeddable")

        n = len(embs)
        device = self._device
        card_emb = torch.from_numpy(np.stack(embs).reshape(1, n, dim)).to(device)
        type_idx = torch.tensor([type_ids], dtype=torch.long, device=device)
        packs_ago_t = torch.tensor([packs_ago], dtype=torch.long, device=device)
        pick_ago_t = torch.tensor([pick_ago], dtype=torch.long, device=device)
        card_mask = torch.ones((1, n), dtype=torch.bool, device=device)
        pack_number = torch.tensor([state.pack_number], dtype=torch.long, device=device)
        pick_number = torch.tensor([state.pick_number], dtype=torch.long, device=device)

        with torch.no_grad():
            policy_logits, _ = self._model(
                card_emb, type_idx, packs_ago_t, pick_ago_t,
                card_mask, pack_number, pick_number,
            )
        rows = [row for row, _ in pack_positions]
        pack_logits = policy_logits[0, rows].cpu()  # single device→host sync

        if self._pick_mode == "sample":
            probs = torch.softmax(pack_logits / self._temperature, dim=-1)
            choice = int(torch.multinomial(probs, 1, generator=self._generator).item())
        else:
            choice = int(torch.argmax(pack_logits).item())
        return pack_positions[choice][1]
