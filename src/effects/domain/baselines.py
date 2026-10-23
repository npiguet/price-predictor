"""The evaluation baselines, as masks over one pipeline (FR-094).

Torch-free, so the CLI can offer the names as choices and the checkpoint store
can file a baseline under its name without loading the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class BaselineMasks:
    """Which parts of the input a baseline zeroes.

    Expressed as masks rather than as four training paths, so every baseline
    runs the identical pipeline and a difference in the numbers is a difference
    in the inputs rather than in the code (FR-094).
    """

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
    """The shipping model and its four baselines.

    ``full`` ships; every other baseline writes under its own name, so a
    baseline run can never overwrite the checkpoint the cache was built from.
    Gate 1 is defined against ``identity``, so what each removes is contract
    rather than commentary.
    """

    #: The shipping model: encoder plus effect head.
    FULL = "full"
    #: A free embedding per unique ability text replaces the encoder. It can
    #: memorize what each seen text does and knows nothing about an unseen one.
    IDENTITY = "identity"
    #: Every ``e`` input zeroed: the floor every record kind is reported against.
    STATE_ONLY = "state-only"
    #: Every input zeroed except ``[ACT]`` and the record-kind flag, entity
    #: ability ``e`` tokens included: the average-effect control.
    NO_STATE = "no-state"
    #: ``e`` replaced by an embedding of the sidecar's API type and parameter keys.
    TAXONOMY = "taxonomy"

    @property
    def masks(self) -> BaselineMasks:
        return _MASKS[self]


#: ``no-state`` zeroes the entity ability ``e`` tokens as well as the state
#: features. Leaving them live would let it read the board through the abilities
#: on it, and it would no longer be the average-effect control. It keeps the
#: acting line's ``e``: the average effect is the average *of that ability*.
_MASKS: dict[Baseline, BaselineMasks] = {
    Baseline.FULL: BaselineMasks(),
    Baseline.IDENTITY: BaselineMasks(identity_embedding=True),
    Baseline.STATE_ONLY: BaselineMasks(zero_act_e=True, zero_context_e=True),
    Baseline.NO_STATE: BaselineMasks(zero_state=True, zero_context_e=True),
    Baseline.TAXONOMY: BaselineMasks(taxonomy_embedding=True),
}

#: The names, in the order the CLI offers them.
BASELINES: tuple[str, ...] = tuple(Baseline)


def baseline_masks(name: str) -> BaselineMasks:
    """The masks the baseline named ``name`` applies."""
    try:
        return Baseline(name).masks
    except ValueError:
        raise ValueError(
            f"unknown --variant {name!r}; known: {list(BASELINES)}"
        ) from None
