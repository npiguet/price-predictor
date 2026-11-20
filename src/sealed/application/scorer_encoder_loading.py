"""The Phase B encoder of ``train-scorer``: rebuilt from a resumed
checkpoint or loaded from ``--encoder-checkpoint`` (a sealed or a
price-predictor encoder, told apart by its saved config), its tokenizer,
and the check that its text width matches the ``.npz`` cache."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from price_predictor.domain.entities import TransformerConfig
from price_predictor.domain.tokenizer import MtgTokenizer
from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel
from sealed.application.scorer_resume import ResumeState
from sealed.domain.card_embedding_layout import FEATURE_COUNT

if TYPE_CHECKING:
    from sealed.application.train_scorer import TrainScorerConfig


def _check_encoder_width(
    encoder: torch.nn.Module, embedding_dim: int, config: TrainScorerConfig,
) -> None:
    """Phase B: raise if the encoder's pooled-text width doesn't match the
    text slice of the cached ``.npz`` embeddings the scorer was tuned on —
    typically because ``--encoder-checkpoint`` points at a different encoder
    than the one that produced the cache. Both encoder configs expose
    ``pooled_dim`` (``d_model`` for ``--pool-mode attn``, else ``2 * d_model``)."""
    enc_width = encoder.config.pooled_dim  # type: ignore[attr-defined]
    text_width = embedding_dim - FEATURE_COUNT
    if enc_width != text_width:
        raise ValueError(
            f"Phase B encoder produces a {enc_width}-wide text vector, but the "
            f".npz cache under {config.embeddings_path or config.cards_path} has a "
            f"{text_width}-wide text "
            f"slice ({embedding_dim}-wide total). Pass --encoder-checkpoint for "
            "the encoder that produced the cache, or re-run `sealed encode-cards` "
            "with this encoder."
        )


def _build_phase_b_encoder(
    config: TrainScorerConfig, resume: ResumeState,
) -> tuple[torch.nn.Module, MtgTokenizer, int]:
    """Construct the Phase B encoder + tokenizer + max-seq-len.

    On `--resume <phaseB>.pt`, the resumed checkpoint is self-contained — the
    encoder is rebuilt from `encoder_config` and weights from
    `encoder_state_dict`, no dependency on the encoder file at
    `config.encoder_checkpoint` being unchanged. Otherwise (`--scorer-checkpoint`
    bootstrap), encoder weights come from `config.encoder_checkpoint`.

    Dispatches between the sealed encoder and the price-predictor encoder by
    inspecting the saved config dict — sealed encoder configs carry
    ``n_pool_queries``, price-predictor configs do not.
    """
    from price_predictor.infrastructure.tokenizer_store import load_tokenizer

    if resume.encoder_state_dict is not None and resume.encoder_config is not None:
        encoder, max_seq_len = _build_encoder_from_config(
            resume.encoder_config, resume.encoder_state_dict,
        )
    else:
        encoder, max_seq_len = _load_encoder_dispatch(config.encoder_checkpoint)

    assert config.encoder_checkpoint is not None  # Phase B always names its encoder
    vocab_path = config.encoder_checkpoint.parent / "vocab.txt"
    tokenizer = load_tokenizer(vocab_path)
    return encoder, tokenizer, max_seq_len


def _build_encoder_from_config(
    config_dict: dict[str, Any], state_dict: dict[str, Any],
) -> tuple[torch.nn.Module, int]:
    """Construct a fresh encoder + load weights, dispatching by config shape."""
    if "n_pool_queries" in config_dict:
        from sealed.domain.encoder_model import (
            SealedEncoderConfig,
            SealedEncoderModel,
        )
        cfg = SealedEncoderConfig(**config_dict)
        encoder: torch.nn.Module = SealedEncoderModel(cfg)
        encoder.load_state_dict(state_dict)
        return encoder, cfg.max_seq_len
    cfg_t = TransformerConfig(**config_dict)
    encoder = CardPriceTransformerModel(cfg_t)
    encoder.load_state_dict(state_dict)
    return encoder, cfg_t.max_seq_len


def _load_encoder_dispatch(encoder_path: Path) -> tuple[torch.nn.Module, int]:
    """Load an encoder checkpoint, dispatching between sealed and price formats."""
    raw = torch.load(encoder_path, map_location="cpu", weights_only=False)
    config_dict = raw["config"]
    if "n_pool_queries" in config_dict:
        from sealed.infrastructure.encoder_store import SealedEncoderStore
        encoder, sealed_cfg = SealedEncoderStore().load_encoder(encoder_path)
        return encoder, sealed_cfg.max_seq_len
    from price_predictor.infrastructure.transformer_store import load_model
    encoder, price_cfg = load_model(encoder_path)
    return encoder, price_cfg.max_seq_len
