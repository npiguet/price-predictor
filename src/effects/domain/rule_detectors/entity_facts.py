"""What the detector families read off one snapshot entity's state.

Not a family: the entity reads several families share, so "a copy", "a
merged permanent" or "holding a designation" means one thing whichever
family a declaration names. Each read returns None where the record cannot
say: a gen-1 or gen-2 snapshot records no face state, no designations and no
links, and an unreadable fact puts an entity on neither side of a rule.
"""

from __future__ import annotations

from effects.domain.state_snapshot import EntityState, FaceState, LineRef, StateSnapshot

#: The face states a declaration may name (``FaceState``).
FACE_STATES = frozenset(state.value for state in FaceState)


def _is_gen3(entity: EntityState) -> bool:
    # A gen-3 entity always lists its lines; gen-1 and gen-2 list none.
    return entity.lines is not None


def _end_is(end, entity_id: str) -> bool:
    return end == entity_id or (isinstance(end, LineRef) and end.entity == entity_id)


def face_state(entity: EntityState) -> str | None:
    """The entity's face state (``front``, ``back``, ``flipped``, ``melded``,
    ``face_down``); None before gen 3."""
    return entity.face_state


def has_face_state(entity: EntityState, wanted: str) -> bool | None:
    """Whether the entity's face state is ``wanted``, compared without case."""
    found = face_state(entity)
    return None if found is None else found.lower() == wanted.lower()


def has_designation(entity: EntityState, wanted: str) -> bool | None:
    """Whether the entity holds the designation ``wanted`` (``suspected``,
    ``harnessed``, ``solved`` …), compared without case; None before gen 3."""
    if not _is_gen3(entity):
        return None
    return wanted.lower() in {d.lower() for d in entity.designations}


def is_copy(entity: EntityState, state: StateSnapshot) -> bool:
    """Whether the entity is a copy of another object: it names a copy
    source, or a gen-3 ``copy_of`` link leaves it."""
    if entity.copy_source is not None:
        return True
    return any(link.type == "copy_of" and _end_is(link.from_, entity.id)
               for link in state.links or ())


def is_merged(entity: EntityState, state: StateSnapshot) -> bool | None:
    """Whether the entity is a merged permanent: a gen-3 ``merged_with`` link
    reaches it (a mutated permanent's other card, the card a melded permanent
    melded with); None on a snapshot with no links."""
    if state.links is None:
        return None
    return any(link.type == "merged_with" and _end_is(link.to, entity.id)
               for link in state.links)


def card_type_count(entity: EntityState) -> int:
    """How many card types the entity has (an artifact creature has two)."""
    return len({kind.lower() for kind in entity.types})


def has_card_type(entity: EntityState, wanted: str) -> bool:
    """Whether ``wanted`` is among the entity's card types, compared without case."""
    return wanted.lower() in {kind.lower() for kind in entity.types}
