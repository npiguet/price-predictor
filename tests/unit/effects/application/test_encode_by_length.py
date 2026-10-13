"""Encoding a batch's texts in length-sorted groups changes no output.

The groups exist for memory alone: padded to the batch's longest text, one
long chain made every line cost its width. Each text must still encode as it
does in one padded batch, in the batch's own row order, gradient included.
"""

from __future__ import annotations

import random

import pytest
import torch

from effects.application import surface_batching
from effects.application.surface_batching import encode_by_length
from effects.domain.ability_encoder import (
    AbilityEncoder,
    AbilityEncoderConfig,
    EncodedLine,
    collate_lines,
)

_VOCAB = 40
_PAD = 0


def _lines(lengths: list[int], seed: int = 3) -> list[EncodedLine]:
    rng = random.Random(seed)
    return [
        EncodedLine(
            token_ids=[1] + [rng.randrange(2, _VOCAB) for _ in range(n - 1)],
            role_ids=[0] * n,
            numbers=[float(rng.randrange(3)) for _ in range(n)],
            number_mask=[rng.randrange(2) for _ in range(n)],
        )
        for n in lengths
    ]


@pytest.fixture
def encoder() -> AbilityEncoder:
    torch.manual_seed(0)
    model = AbilityEncoder(AbilityEncoderConfig(
        vocab_size=_VOCAB, e_dim=8, d_model=16, n_layers=2,
    ))
    return model.eval()


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setattr(surface_batching, "ENCODE_CHUNK", 2)


# Shuffled lengths, so the groups reorder rows and the inverse has work to do.
_LENGTHS = [9, 2, 30, 5, 5, 17, 3, 12]


def test_vectors_match_one_padded_batch(encoder, small_chunks):
    inputs = collate_lines(_lines(_LENGTHS), _PAD)
    whole, _ = encoder(**inputs)
    grouped, hidden = encode_by_length(encoder, inputs, torch.device("cpu"))
    assert hidden is None
    torch.testing.assert_close(grouped, whole, rtol=1e-5, atol=1e-5)


def test_token_outputs_match_at_every_real_position(encoder, small_chunks):
    inputs = collate_lines(_lines(_LENGTHS), _PAD)
    _, whole = encoder(**inputs)
    _, grouped = encode_by_length(encoder, inputs, torch.device("cpu"), hidden=True)
    assert grouped.shape == whole.shape
    real = inputs["attention_mask"].bool()
    torch.testing.assert_close(grouped[real], whole[real], rtol=1e-5, atol=1e-5)


def test_gradient_matches_one_padded_batch(encoder, small_chunks):
    inputs = collate_lines(_lines(_LENGTHS), _PAD)
    weights = torch.randn(len(_LENGTHS), 8, generator=torch.Generator().manual_seed(1))

    whole, _ = encoder(**inputs)
    (whole * weights).sum().backward()
    expected = {n: p.grad.clone() for n, p in encoder.named_parameters() if p.grad is not None}
    encoder.zero_grad()

    grouped, _ = encode_by_length(encoder, inputs, torch.device("cpu"))
    (grouped * weights).sum().backward()
    for name, grad in expected.items():
        torch.testing.assert_close(
            dict(encoder.named_parameters())[name].grad, grad, rtol=1e-4, atol=1e-5,
        )


def test_each_group_is_padded_to_its_own_longest(small_chunks):
    inputs = collate_lines(_lines(_LENGTHS), _PAD)
    chunks = surface_batching._length_chunks(inputs["attention_mask"])
    assert [width for _, width in chunks] == [3, 5, 12, 30]
    assert sorted(torch.cat([rows for rows, _ in chunks]).tolist()) == list(range(8))
