"""An ability line's tokens laid out for the encoder: ``prepare_line`` to one
:class:`EncodedLine`, ``collate_lines`` to a padded batch, and the once-per-run
truncation report. The stager (``text_staging``) is their caller."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from effects.domain.ability_encoder import (
    MAX_ABILITY_TOKENS,
    NO_ROLE_INDEX,
    ROLE_ORDER,
)
from effects.domain.ability_tokenizer import AbilityTokenizer, Token
from effects.domain.provenance import Role


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
        named = ", ".join(key.label for key in keys) or "(no provenance key)"
        logger.warning(
            "%s: truncated at %d tokens: %s — %.80s",
            command, MAX_ABILITY_TOKENS, named, text,
        )

    return report


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
