"""The evaluation baselines, as masks over one pipeline (FR-094).

Torch-free, so the CLI can offer the names as choices and the checkpoint store
can file a variant under its name without loading the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class VariantMasks:
    """Which parts of the input a variant zeroes.

    Expressed as masks rather than as four training paths, so every baseline
    runs the identical pipeline and a difference in the numbers is a difference
    in the inputs rather than in the code (FR-094).
    """

    #: Zero the acting line's and every entity ability token's ``e``.
    zero_e: bool = False
    #: Zero the ``[PLAYER]`` and ``[CARD]`` features and their ability tokens.
    zero_state: bool = False
    #: Replace the encoder with a free per-text embedding table.
    identity_embedding: bool = False
    #: Replace ``e`` with an embedding of the sidecar's API type and param keys.
    taxonomy_embedding: bool = False


class Variant(StrEnum):
    """The shipping model and its four baselines.

    ``full`` ships; every other variant writes under its own name, so a
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
    def masks(self) -> VariantMasks:
        return _MASKS[self]


#: ``no-state`` zeroes the entity ability ``e`` tokens as well as the state
#: features. Leaving them live would let it read the board through the abilities
#: on it, and it would no longer be the average-effect control.
_MASKS: dict[Variant, VariantMasks] = {
    Variant.FULL: VariantMasks(),
    Variant.IDENTITY: VariantMasks(identity_embedding=True),
    Variant.STATE_ONLY: VariantMasks(zero_e=True),
    Variant.NO_STATE: VariantMasks(zero_state=True, zero_e=True),
    Variant.TAXONOMY: VariantMasks(taxonomy_embedding=True),
}

#: The names, in the order the CLI offers them.
VARIANTS: tuple[str, ...] = tuple(Variant)


def variant_masks(variant: str) -> VariantMasks:
    """The masks the variant named ``variant`` applies."""
    try:
        return Variant(variant).masks
    except ValueError:
        raise ValueError(
            f"unknown --variant {variant!r}; known: {list(VARIANTS)}"
        ) from None
