"""The gen-4 sealed encoder, loaded once, and the converted-text reader it encodes."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from probe_paths import ENCODER_CKPT, VOCAB_PATH

from price_predictor.domain.card_text import ConvertedCardText
from price_predictor.infrastructure.tokenizer_store import load_tokenizer
from sealed.infrastructure.encoder_store import SealedEncoderStore

TEXT_DIM = 512          # pooled text vector; the cached .npz is 512 + 32 features


def read_text(path: Path | str) -> str:
    return ConvertedCardText.from_file(Path(path)).text


class EncoderRunner:
    """The gen-4 sealed encoder, loaded once, with batched re-encoding.

    ``encode_texts`` takes *name-stripped* converted card text (a leading
    ``name:`` line is stripped defensively) and returns the 512-dim pooled
    text vector — the first 512 dims of the cached ``.npz``.

    The cache was written by ``encode-cards``, which runs the encoder on
    **CPU, one card at a time, padded to ``max_seq_len``**. That is the
    only configuration that reproduces it bit-exactly: batching or moving
    to CUDA changes the reduction order and shifts the last few bits
    (max |Δ| ≈ 3e-7, ~5e-6 σ of a text dim — irrelevant for probes, fatal
    for an equality check). ``exact=True`` opts into the slow faithful
    path; the default follows ``device``.
    """

    def __init__(
        self, device: str | None = None, checkpoint: Path = ENCODER_CKPT,
        vocab_path: Path = VOCAB_PATH,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.config = SealedEncoderStore().load_encoder(checkpoint)
        self.model.eval().to(self.device)
        self.tokenizer = load_tokenizer(vocab_path)
        self.max_seq_len = self.config.max_seq_len

    def _to(self, device: str) -> None:
        if device != self.device:
            self.model.to(device)
            self.device = device

    @torch.no_grad()
    def encode_texts(
        self, texts: Sequence[str], *, batch_size: int = 64, exact: bool = False,
    ) -> np.ndarray:
        """``(len(texts), 512)`` float32 pooled text vectors."""
        device, batch_size = (("cpu", 1) if exact else (self.device, batch_size))
        self._to(device)
        stripped = [ConvertedCardText(t).without_name_line() for t in texts]
        out = np.empty((len(texts), TEXT_DIM), dtype=np.float32)
        for lo in range(0, len(stripped), batch_size):
            chunk = stripped[lo:lo + batch_size]
            encoded = [self.tokenizer.encode(t, self.max_seq_len) for t in chunk]
            ids = torch.tensor([e[0] for e in encoded], dtype=torch.long, device=device)
            mask = torch.tensor([e[1] for e in encoded], dtype=torch.long, device=device)
            out[lo:lo + len(chunk)] = (
                self.model.encode(ids, mask).cpu().numpy().astype(np.float32)
            )
        return out

    def encode_files(
        self, paths: Sequence[Path], *, batch_size: int = 64, exact: bool = False,
    ) -> np.ndarray:
        return self.encode_texts([read_text(p) for p in paths],
                                 batch_size=batch_size, exact=exact)

    def check_bit_exact(self, paths: Sequence[Path]) -> dict:
        """Re-encode ``paths`` in exact mode and diff against their ``.npz``.

        Returns ``{n, n_exact, max_abs_diff}``; ``n_exact == n`` is the
        contract that lets a counterfactual read-off be attributed to the
        edit rather than to the harness.
        """
        got = self.encode_files(paths, exact=True)
        cached = np.stack([
            np.load(Path(p).with_suffix(".npz"))["embedding"][:TEXT_DIM]
            for p in paths
        ])
        return {
            "n": len(paths),
            "n_exact": int((got == cached).all(axis=1).sum()),
            "max_abs_diff": float(np.abs(got - cached).max()),
        }

