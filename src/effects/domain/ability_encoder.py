"""The ability encoder: one ability line to its bottleneck vector ``e``.

A transformer pooled through ``[CLS]`` to a small fixed width, so whatever the
effect head needs about an ability has to be in the vector that ships in the
cache. Each token adds a role embedding (FR-069), because ``{R}`` in a cost and
in an effect mean opposite things; only prose carries roles, since the spans
index the prose. A number adds a monotone embedding, so its order is in the
geometry rather than learned per digit. A model architecture is domain in this
project (constitution Principle IV).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from effects.domain.ability_tokenizer import (
    AbilityTokenizer,
    Surface,
    Token,
)
from effects.domain.provenance import Role

#: Fixed order, so an embedding row means the same role across checkpoints.
#: Row 0 is "no role", a token outside every span.
ROLE_ORDER: tuple[Role, ...] = tuple(Role)
NO_ROLE_INDEX = 0

# Defaults of `--encoder-d-model` and `--encoder-layers` (FR-061), and the size
# a checkpoint that records neither loads at.
ENCODER_D_MODEL = 256
ENCODER_N_LAYERS = 4
ENCODER_N_HEADS = 4
FF_MULTIPLIER = 4
DROPOUT = 0.1

MAX_ABILITY_TOKENS = 512


@dataclass(frozen=True, slots=True)
class AbilityEncoderConfig:
    """Architecture of the ability encoder, carried so a checkpoint is
    self-describing. No noise here: the batcher applies it (FR-057)."""

    vocab_size: int
    e_dim: int = 64
    d_model: int = ENCODER_D_MODEL
    n_layers: int = ENCODER_N_LAYERS
    n_heads: int = ENCODER_N_HEADS
    #: 0 means "follow ``d_model``"; resolved in ``__post_init__``.
    ff_dim: int = 0
    dropout: float = DROPOUT
    max_seq_len: int = MAX_ABILITY_TOKENS

    def __post_init__(self) -> None:
        if self.ff_dim == 0 and self.d_model > 0:
            object.__setattr__(self, "ff_dim", self.d_model * FF_MULTIPLIER)
        for name in ("vocab_size", "e_dim", "d_model", "n_layers", "n_heads",
                     "ff_dim", "max_seq_len"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads})"
            )


class MonotoneNumberEmbedding(nn.Module):
    """``base + log1p(n) * direction``: ``log1p`` because 1 against 2 damage
    decides a combat and 17 against 18 rarely does."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.zeros(d_model))
        self.direction = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.base, std=0.02)
        nn.init.normal_(self.direction, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """``values``: ``(...,)`` numbers. Returns ``(..., d_model)``."""
        scaled = torch.log1p(values.clamp(min=0.0)).unsqueeze(-1)
        return self.base + scaled * self.direction


class AbilityEncoder(nn.Module):
    """Tokens of one ability line to its ``e`` vector."""

    def __init__(self, config: AbilityEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        # +1 for "no role": a token outside every span.
        self.role_embedding = nn.Embedding(len(ROLE_ORDER) + 1, config.d_model)
        self.number_embedding = MonotoneNumberEmbedding(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=config.n_layers,
            # norm_first makes torch's nested-tensor fast path inapplicable.
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.to_e = nn.Linear(config.d_model, config.e_dim)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_ids: torch.Tensor | None = None,
        numbers: torch.Tensor | None = None,
        number_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a padded batch of ``(batch, seq)`` lines, ``[CLS]`` first.

        Returns ``(e, hidden)``: the ``(batch, e_dim)`` bottleneck and the
        ``(batch, seq, d_model)`` token outputs the MLM auxiliary reads.
        """
        batch, seq = token_ids.shape
        positions = torch.arange(seq, device=token_ids.device).unsqueeze(0)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        if role_ids is not None:
            hidden = hidden + self.role_embedding(role_ids)
        if numbers is not None:
            numeric = self.number_embedding(numbers)
            if number_mask is not None:
                numeric = numeric * number_mask.unsqueeze(-1).to(numeric.dtype)
            hidden = hidden + numeric
        hidden = self.dropout(hidden)

        padding_mask = attention_mask == 0
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = self.norm(hidden)

        e =self.to_e(hidden[:, 0, :])
        return e, hidden


# ── surfaces (FR-062, stage four) ───────────────────────────────────────


def surface_of(vocab_path) -> Surface:
    """Which surface a vocabulary path implies: never a flag of its own, which
    could disagree with the vocabulary and still look like a working run."""
    from pathlib import Path

    return (
        Surface.SCRIPT
        if "vocab-script" in Path(vocab_path).name
        else Surface.PROSE
    )


def encoding_text(line, prose: str | None, surface: Surface) -> str | None:
    """The text a line is encoded from on ``surface``, falling back to the
    other surface's where its own is missing (a keyword-derived line has no
    script, a variant script no prose)."""
    script = getattr(line, "script_text", None)
    if surface == Surface.SCRIPT:
        return script or prose
    return prose or script


def encoding_role_spans(line, text: str | None, surface: Surface) -> tuple:
    """The role spans a line's encoding text is tokenized with (FR-069): only
    on a prose text, since the spans index the converted prose."""
    if surface != Surface.PROSE or line is None or text is None:
        return ()
    if text == getattr(line, "script_text", None):
        return ()
    return tuple(getattr(line, "role_spans", ()) or ())


def role_index(role: Role | None) -> int:
    if role is None:
        return NO_ROLE_INDEX
    return ROLE_ORDER.index(role) + 1


@dataclass(frozen=True, slots=True)
class EncodedLine:
    """One tokenized line, ready to be batched into the encoder."""

    token_ids: list[int]
    role_ids: list[int]
    numbers: list[float]
    number_mask: list[int]
    #: The line ran past the encoder's window and lost its tail (FR-006).
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.token_ids)


def prepare_line(
    tokenizer: AbilityTokenizer,
    tokens: list[Token],
    *,
    max_len: int = MAX_ABILITY_TOKENS,
) -> EncodedLine:
    """Prefix ``[CLS]`` and lay a token list out as parallel arrays.

    Truncation keeps ``[CLS]`` and drops the tail, and says so, because a tail
    dropped silently is a sub-ability the model never read (FR-006).
    """
    token_ids = [tokenizer.cls_id]
    role_ids = [NO_ROLE_INDEX]
    numbers = [0.0]
    number_mask = [0]
    for token in tokens[: max_len - 1]:
        token_ids.append(token.token_id)
        role_ids.append(role_index(token.role))
        numbers.append(token.number if token.number is not None else 0.0)
        number_mask.append(1 if token.number is not None else 0)
    return EncodedLine(
        token_ids, role_ids, numbers, number_mask,
        truncated=len(tokens) > max_len - 1,
    )


class TruncationLog:
    """Reports each truncated line once per run, by provenance key (FR-006)."""

    def __init__(self, report) -> None:
        """``report(text, keys)`` is called the first time a text truncates."""
        self._report = report
        self.texts: set[str] = set()

    def note(self, text: str, keys=()) -> None:
        if text in self.texts:
            return
        self.texts.add(text)
        self._report(text, tuple(keys))


def log_truncation(logger, command: str):
    """A :class:`TruncationLog` reporter that writes one warning per line."""

    def report(text: str, keys: tuple) -> None:
        named = ", ".join(_key_label(key) for key in keys) or "(no provenance key)"
        logger.warning(
            "%s: truncated at %d tokens: %s — %.80s",
            command, MAX_ABILITY_TOKENS, named, text,
        )

    return report


def _key_label(key) -> str:
    option = getattr(key, "option", None)
    suffix = f" option {option}" if option is not None else ""
    return (
        f"{key.script_file} face {key.face} {key.trait_kind}"
        f"[{key.index_within_kind}]{suffix}"
    )


def collate_lines(
    lines: list[EncodedLine], pad_id: int,
) -> dict[str, torch.Tensor]:
    """Pad a batch of prepared lines to its own longest, as encoder kwargs."""
    width = max((len(line) for line in lines), default=1)
    token_ids, role_ids, numbers, number_mask, attention = [], [], [], [], []
    for line in lines:
        pad = width - len(line)
        token_ids.append(line.token_ids + [pad_id] * pad)
        role_ids.append(line.role_ids + [NO_ROLE_INDEX] * pad)
        numbers.append(line.numbers + [0.0] * pad)
        number_mask.append(line.number_mask + [0] * pad)
        attention.append([1] * len(line) + [0] * pad)
    return {
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "role_ids": torch.tensor(role_ids, dtype=torch.long),
        "numbers": torch.tensor(numbers, dtype=torch.float32),
        "number_mask": torch.tensor(number_mask, dtype=torch.long),
    }
