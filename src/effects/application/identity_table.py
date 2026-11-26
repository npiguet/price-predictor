"""The ``identity`` baseline's free ``e`` per ability text."""

from __future__ import annotations

from collections.abc import Iterable

import torch

#: Where a checkpoint's ``extra`` holds the identity baseline's table.
IDENTITY_TABLE_KEY = "identity_table"


class IdentityTable(torch.nn.Module):
    """The ``identity`` baseline's free ``e`` per ability text, indexed exactly.

    One row per text ``ability_text.known_texts`` yields, plus one row every
    other text shares (a counterfactual copy's, say). Never a hash: two texts
    sharing a row is the memorization the baseline exists to measure. The
    texts ride in the checkpoint beside the weights (:meth:`state`).
    """

    UNKNOWN_ROW = 0

    def __init__(self, texts: Iterable[str], e_dim: int) -> None:
        super().__init__()
        self.texts: tuple[str, ...] = tuple(sorted(set(texts)))
        self._rows = {text: row for row, text in enumerate(self.texts, start=1)}
        self.embedding = torch.nn.Embedding(len(self.texts) + 1, e_dim)

    def rows(self, texts: Iterable[str]) -> torch.Tensor:
        """Each text's row, on the host; the unknown row for a text not indexed."""
        return torch.tensor(
            [self._rows.get(text, self.UNKNOWN_ROW) for text in texts],
            dtype=torch.long,
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.embedding(rows)

    def state(self) -> dict:
        return {"texts": list(self.texts), "weight": self.embedding.weight.detach().cpu()}

    @classmethod
    def from_state(cls, state: dict) -> IdentityTable:
        table = cls(state["texts"], int(state["weight"].shape[1]))
        table.embedding.weight.data.copy_(state["weight"])
        return table
