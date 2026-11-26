"""The targets of every loss term beside the per-entity one, on the host: the
shipped heads' (:mod:`effects.application.record_head_targets`) and the
training-only heads', which read the step's texts and no rule mask."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from effects.application.record_head_targets import HeadTargets, shipped_head_targets
from effects.application.surface_batching import PreparedBatch
from effects.domain.aux_heads import Head
from effects.domain.provenance import SidecarLine
from effects.domain.rule_mask import CopyWeights
from effects.domain.script_text import params, segments
from effects.domain.text_head_losses import ApiTargets, ValueTargets
from effects.domain.value_targets import ValueLine


def value_batch(
    lines: Sequence[ValueLine], rows: Mapping[str, int],
) -> ValueTargets:
    """The value head's targets, one row per line (FR-058): ``matrix_rows[i]``
    is the row of the batch's ``e`` matrix line ``i``'s text is encoded at, so two
    lines sharing a text read one ``e`` each toward their own targets.

    Read from each line's script whatever the surface, since prose holds no
    ``$`` parameter; a line with no script, or a keyword line, masks them all.
    """
    targets = [line.targets for line in lines]
    return ValueTargets(
        torch.tensor([row.values for row in targets]),
        torch.tensor([row.mask for row in targets]),
        torch.tensor([rows[line.text] for line in lines], dtype=torch.long),
    )


class ApiTargetBuilder:
    """The script-API head's targets for a batch's texts, with a memo of what
    one sidecar line determines, keyed by that line's text."""

    def __init__(self, api_types: Sequence[str], param_keys: Sequence[str]) -> None:
        self._param_key_count = len(param_keys)
        self._type_index = {name: i for i, name in enumerate(api_types)}
        self._key_index = {name: i for i, name in enumerate(param_keys)}
        self._memo: dict[str, tuple[int, tuple[int, ...]]] = {}

    def batch(self, texts: Sequence[str], lines: Mapping[str, SidecarLine]) -> ApiTargets:
        """The targets for the batch's texts, in row order: the API type
        (``-1`` when none or unknown) and the multi-hot of parameter keys over
        every chained segment."""
        types: list[int] = []
        rows: list[int] = []
        columns: list[int] = []
        for row, text in enumerate(texts):
            api_type, keys = self._target_of(text, lines.get(text))
            types.append(api_type)
            rows += [row] * len(keys)
            columns += keys
        # One indexed write: torch dispatches every element assignment on the
        # host half of the step.
        keys = torch.zeros(len(texts), max(self._param_key_count, 1))
        keys[torch.tensor(rows, dtype=torch.long), torch.tensor(columns, dtype=torch.long)] = 1.0
        return ApiTargets(torch.tensor(types, dtype=torch.long), keys)

    def _target_of(self, text: str, line: SidecarLine | None) -> tuple[int, tuple[int, ...]]:
        """One text's API type and parameter-key columns.

        A text no line of this batch renders (a granted keyword's text,
        ``Protection from white``, which another batch may hold as a card's
        line) has no target in this batch. The memo holds lines' targets only:
        read for a lineless text, or filled from one, it would make a step's
        targets depend on which shards its preparer read before, and a run
        differ by its worker count.
        """
        if line is None:
            return -1, ()
        cached = self._memo.get(text)
        if cached is None:
            cached = self._memo[text] = self._read(line)
        return cached

    def _read(self, line: SidecarLine) -> tuple[int, tuple[int, ...]]:
        found = set(line.script_param_keys)
        for segment in segments(line.script_text or ""):
            found.update(params(segment))
        return (
            self._type_index.get(line.script_api_type, -1),
            tuple(sorted(self._key_index[k] for k in found if k in self._key_index)),
        )


@dataclass(frozen=True)
class HeadTargetBuilder:
    """The targets of every head beside the per-entity one, for a step."""

    #: The training-only heads computed beside the shipped ones.
    heads: frozenset[Head]
    mlm_mask_prob: float
    api: ApiTargetBuilder

    def build(
        self, records, weights: list[CopyWeights], batcher, batch: PreparedBatch,
    ) -> HeadTargets:
        """The shipped heads' targets (FR-060a), each record's fields weighed
        by its rule mask (FR-061), then those of the training-only ``heads``
        (FR-058, FR-060b), which read texts and no mask reaches. The MLM mask
        is the step's last draw from the batcher's generator."""
        targets = shipped_head_targets(records, weights)
        # No text, no matrix for these heads to read.
        if not self.heads or not batch.staged.rows:
            return targets
        if Head.VALUE in self.heads:
            targets[Head.VALUE] = value_batch(batch.value_lines, batch.staged.rows)
        if Head.API in self.heads:
            targets[Head.API] = self.api.batch(batch.staged.ordered, batch.texts)
        if Head.MLM in self.heads:
            targets[Head.MLM] = batcher.draw_mlm_mask(
                batch.staged.encoder_inputs, self.mlm_mask_prob,
            )
        return targets
