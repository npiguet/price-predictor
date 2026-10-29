"""The evaluation baselines, as masks over one pipeline (FR-094).

Torch-free, so the CLI can offer the names as choices and the checkpoint store
can file a baseline under its name without loading the model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from effects.domain.provenance import ProvenanceSidecar, SidecarLine


@dataclass(frozen=True, slots=True)
class BaselineMasks:
    """Which parts of the input a baseline zeroes: masks rather than four
    training paths, so a difference in the numbers is one in the inputs."""

    #: Zero the ``[ACT]`` slot's ``e``: the acting line, or a decision's candidate.
    zero_act_e: bool = False
    #: Zero every entity ability token's ``e``: the board's own abilities.
    zero_context_e: bool = False
    #: Zero the ``[PLAYER]`` and ``[CARD]`` features, and every ``[GLOBAL]``
    #: feature but the record-kind flags.
    zero_state: bool = False
    #: Replace the encoder with a free per-text embedding table.
    identity_embedding: bool = False
    #: Replace ``e`` with an embedding of the sidecar's API type and param keys.
    taxonomy_embedding: bool = False


class Baseline(StrEnum):
    """The shipping model and its four baselines, each writing under its own
    name so a baseline run never overwrites the shipping checkpoint."""

    FULL = "full"
    #: Can memorize what each seen text does and knows nothing of an unseen one.
    IDENTITY = "identity"
    #: The floor every record kind is reported against.
    STATE_ONLY = "state-only"
    #: The average-effect control.
    NO_STATE = "no-state"
    TAXONOMY = "taxonomy"

    @property
    def masks(self) -> BaselineMasks:
        return _MASKS[self]


#: ``no-state`` zeroes the context ``e`` too, or it would read the board
#: through the abilities on it; it keeps ``[ACT]``'s, the ability averaged over.
_MASKS: dict[Baseline, BaselineMasks] = {
    Baseline.FULL: BaselineMasks(),
    Baseline.IDENTITY: BaselineMasks(identity_embedding=True),
    Baseline.STATE_ONLY: BaselineMasks(zero_act_e=True, zero_context_e=True),
    Baseline.NO_STATE: BaselineMasks(zero_state=True, zero_context_e=True),
    Baseline.TAXONOMY: BaselineMasks(taxonomy_embedding=True),
}

BASELINES: tuple[str, ...] = tuple(Baseline)


def baseline_masks(name: str) -> BaselineMasks:
    """The masks the baseline named ``name`` applies."""
    try:
        return Baseline(name).masks
    except ValueError:
        raise ValueError(
            f"unknown --variant {name!r}; known: {list(BASELINES)}"
        ) from None


def taxonomy_vector(line: SidecarLine, e_dim: int) -> np.ndarray:
    """The ``taxonomy`` baseline's stand-in for an encoded ``e``: a
    deterministic hash embedding of the line's API type and parameter keys."""
    vector = np.zeros(e_dim, dtype=np.float32)
    features = [f"api:{line.script_api_type or ''}"]
    features += [f"param:{key}" for key in line.script_param_keys]
    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % e_dim
        sign = 1.0 if digest[4] % 2 else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def taxonomy_matrix(sidecar: ProvenanceSidecar, e_dim: int) -> np.ndarray:
    """The ``taxonomy`` variant's ``(n_lines, e_dim)`` matrix, row-aligned to
    its sidecar, with no encoder at all."""
    if not sidecar.lines:
        return np.zeros((0, e_dim), dtype=np.float32)
    return np.stack([taxonomy_vector(line, e_dim) for line in sidecar.lines])
