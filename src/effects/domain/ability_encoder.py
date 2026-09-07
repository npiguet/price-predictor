"""The ability encoder: one ability line to its bottleneck vector ``e``.

A transformer over the line's tokens, pooled through a ``[CLS]`` aggregation
token to a fixed width (`--e-dim`, default 64). The bottleneck is the point of
the whole design: ``e`` is small enough that the effect head cannot route around
it, so whatever the head needs about an ability has to be *in* the vector, and
the vector is what ships in the cache.

Three embeddings are added to each token, not one:

- **token** — the word itself;
- **position** — where it sits in the line;
- **role** — ``cost`` / ``effect`` / ``trigger-condition`` / ``target-spec``, read
  from the sidecar's spans. Without it ``{R}`` in a cost and ``{R}`` in an effect
  are the same input, and they mean opposite things.

Numbers get a fourth: a **monotone numeric embedding**, a shared learned base
vector plus ``log1p(n)`` times a learned direction. A number's magnitude is
ordered, so an embedding table indexed by digit would have to learn that 3 sits
between 2 and 4 from data; this puts it in the geometry.

Torch lives in ``domain`` here for the same reason it does in
``sealed/domain/encoder_model.py`` and ``draft/domain/draft_agent_model.py``: a
model architecture *is* domain in this project, and torch is its notation
(constitution Principle IV, as amended).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from effects.domain.ability_tokenizer import AbilityTokenizer, Token

#: Role vocabulary, in a fixed order so an embedding row means the same thing
#: across checkpoints. Index 0 is "no role" — the token sat outside every span.
ROLE_ORDER: tuple[str, ...] = ("cost", "effect", "trigger-condition", "target-spec")
NO_ROLE_INDEX = 0

# Hardcoded architecture (contracts/cli.md: "Hardcoded, not flags").
ENCODER_D_MODEL = 256
ENCODER_N_LAYERS = 4
ENCODER_N_HEADS = 4
FF_MULTIPLIER = 4
DROPOUT = 0.1

#: Longest line the position table covers. Ability lines are short; a keyword
#: expansion is the case that lengthens them, and none reach this.
MAX_ABILITY_TOKENS = 512


@dataclass(frozen=True, slots=True)
class AbilityEncoderConfig:
    """Architecture of the ability encoder.

    Only ``e_dim`` and ``e_noise`` are flags; the rest are the pinned constants
    above, carried here so a checkpoint is self-describing.
    """

    vocab_size: int
    e_dim: int = 64
    e_noise: float = 0.05
    d_model: int = ENCODER_D_MODEL
    n_layers: int = ENCODER_N_LAYERS
    n_heads: int = ENCODER_N_HEADS
    ff_dim: int = ENCODER_D_MODEL * FF_MULTIPLIER
    dropout: float = DROPOUT
    max_seq_len: int = MAX_ABILITY_TOKENS

    def __post_init__(self) -> None:
        for name in ("vocab_size", "e_dim", "d_model", "n_layers", "n_heads",
                     "ff_dim", "max_seq_len"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads})"
            )
        if self.e_noise < 0.0:
            raise ValueError(f"e_noise must be >= 0, got {self.e_noise}")


class MonotoneNumberEmbedding(nn.Module):
    """``base + log1p(n) * direction`` — an ordered embedding for a number.

    ``log1p`` rather than ``n`` because the useful range is heavily skewed: the
    difference between 1 and 2 damage decides a combat, the difference between
    17 and 18 rarely decides anything.
    """

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
        """Encode a padded batch of lines.

        Args:
            token_ids: ``(batch, seq)``; position 0 must be ``[CLS]``.
            attention_mask: ``(batch, seq)``, 1 for real tokens.
            role_ids: ``(batch, seq)`` indices into :data:`ROLE_ORDER` shifted by
                one, 0 meaning no role. Absent, every token reads "no role".
            numbers: ``(batch, seq)`` numeric values, 0 where not numeric.
            number_mask: ``(batch, seq)``, 1 where ``numbers`` is meaningful.

        Returns:
            ``(e, hidden)`` — the ``(batch, e_dim)`` bottleneck read from the
            ``[CLS]`` position, and the ``(batch, seq, d_model)`` token outputs
            the MLM auxiliary reads.
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

        # [CLS] pooling: position 0 by construction, so no masked reduction is
        # needed and the pooled vector cannot be diluted by padding.
        e = self.to_e(hidden[:, 0, :])
        if self.training and self.config.e_noise > 0.0:
            # Additive noise on the bottleneck. It keeps the head from reading
            # e at a precision the cache cannot reproduce, and widens the
            # neighbourhood of each ability so nearby texts stay nearby.
            e = e + torch.randn_like(e) * self.config.e_noise
        return e, hidden


# ── surfaces (FR-062, stage four) ───────────────────────────────────────

SURFACE_PROSE = "prose"
SURFACE_SCRIPT = "script"


def surface_of(vocab_path) -> str:
    """Which surface a vocabulary path implies.

    The surface follows the loaded ``--vocab-path`` rather than a flag of its
    own, so a stage-one-to-three checkpoint keeps encoding against the
    vocabulary it recorded even after a stage-four rebuild exists. A flag would
    let the two disagree, and the disagreement would look like a working run.
    """
    from pathlib import Path

    return (
        SURFACE_SCRIPT
        if "vocab-script" in Path(vocab_path).name
        else SURFACE_PROSE
    )


def encoding_text(line, prose: str | None, surface: str) -> str | None:
    """The text a line is encoded from, on the given surface.

    Script from stage four, prose before it — and each falls back to the other
    where its own is missing. A keyword-derived line has no Forge script of its
    own, and a variant script has no prose at all, so a surface that dropped
    what it lacked would drop a different slice of the corpus in each direction.
    """
    script = getattr(line, "script_text", None)
    if surface == SURFACE_SCRIPT:
        return script or prose
    return prose or script


def role_index(role: str | None) -> int:
    """The embedding row for a role tag; 0 for a token outside every span."""
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

    def __len__(self) -> int:
        return len(self.token_ids)


def prepare_line(
    tokenizer: AbilityTokenizer,
    tokens: list[Token],
    *,
    max_len: int = MAX_ABILITY_TOKENS,
) -> EncodedLine:
    """Prefix ``[CLS]`` and lay a token list out as parallel arrays.

    Truncation keeps ``[CLS]`` and drops from the tail, so the pooled vector
    always exists even for a line whose keyword expansions overran the window.
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
    return EncodedLine(token_ids, role_ids, numbers, number_mask)


def collate_lines(
    lines: list[EncodedLine], pad_id: int,
) -> dict[str, torch.Tensor]:
    """Pad a batch of prepared lines to the longest, as encoder kwargs.

    Padded per batch rather than to a fixed width, so a batch of short lines
    costs what short lines cost — the corpus's ability texts are mostly one
    clause and occasionally a paragraph.
    """
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
