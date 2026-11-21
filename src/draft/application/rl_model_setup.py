"""The gen-2 model's start: actor and critic from ``--checkpoint`` or ``--resume``,
checked against the cache's embedding width, and a frozen KL reference
(research D7).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from draft.domain.draft_agent_model import DraftAgentConfig, DraftAgentModel
from draft.infrastructure.draft_agent_store import DraftAgentStore

if TYPE_CHECKING:
    from draft.application.train_draft_agent_rl import TrainDraftAgentRLConfig



@dataclass
class _Resume:
    model: DraftAgentModel
    reference: DraftAgentModel
    start_epoch: int
    best_val_loss: float
    optimizer_state: dict[str, Any] | None
    lr_decay_count: int
    critic_mean: float
    critic_std: float
    generation: int


def _check_dims(ckpt_config: DraftAgentConfig, embedding_dim: int, cards_path: Path) -> None:
    if ckpt_config.embedding_dim != embedding_dim:
        raise ValueError(
            f"Checkpoint expects {ckpt_config.embedding_dim}-wide card "
            f"embeddings, but the .npz cache under {cards_path} is "
            f"{embedding_dim}-wide. Re-run `sealed encode-cards` with the "
            "encoder this agent was trained on."
        )


def _resume_or_build(
    config: TrainDraftAgentRLConfig, store: DraftAgentStore, embedding_dim: int,
) -> _Resume:
    """Load actor+critic from --checkpoint or --resume; freeze a KL reference.

    The KL reference is a frozen copy of the model weights at run start: the
    reference πₖ on a fresh ``--checkpoint`` run; the resumed-from weights on
    ``--resume`` (research D7). Generation index = reference generation + 1.
    """
    source = config.resume if config.resume is not None else config.checkpoint
    assert source is not None  # CLI guarantees exactly one
    ckpt = store.load_checkpoint(source)
    _check_dims(ckpt.config, embedding_dim, config.embeddings_path or config.cards_path)
    model = DraftAgentModel(ckpt.config)
    model.load_state_dict(ckpt.model_state_dict)
    reference = DraftAgentModel(ckpt.config)
    reference.load_state_dict(ckpt.model_state_dict)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    if config.resume is not None:
        start_epoch = ckpt.epoch + 1
        best = ckpt.best_val_loss
        opt_state = ckpt.optimizer_state_dict or None
        decay = ckpt.lr_decay_count
        # Continuing an RL run: keep this run's generation index.
        generation = int((ckpt.rl_metadata or {}).get("generation", 2))
    else:
        start_epoch = 0
        best = float("inf")
        opt_state = None
        decay = 0
        # Fresh kickoff: reference generation (gen-1 if no rl_metadata) + 1.
        ref_gen = int((ckpt.rl_metadata or {}).get("generation", 1))
        generation = ref_gen + 1
    return _Resume(
        model, reference, start_epoch, best, opt_state, decay,
        ckpt.critic_mean, ckpt.critic_std, generation,
    )
