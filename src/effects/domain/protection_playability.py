"""The target-made-illegal and block-made-illegal outcomes of a protection copy
(702.16b, 702.16f): which decisions and block declarations a granted quality
changes.

Torch-free: ``build-corpus``'s survey and write workers call it.
"""

from __future__ import annotations

from effects.domain.protection_model import (
    Opportunity,
    ProtectionOutcome,
    permanent,
    rare_qualities,
)
from effects.domain.provenance import SidecarLookup
from effects.domain.records import PlayabilityBlockersPayload, PlayabilityDecisionPayload
from effects.domain.script_text import params, segments
from effects.domain.state_snapshot import EntityState, FaceState


def target_opportunities(record, payload: PlayabilityDecisionPayload, entities,
                          lines: SidecarLookup | None) -> tuple[Opportunity, ...]:
    """A decision's legal targets, one of them made illegal while another stays."""
    if not payload.candidates:
        return ()
    candidate = payload.candidates[0]
    source = entities.get(record.state.refs.source)
    if source is None or not _removable_target(candidate, source, lines):
        return ()
    qualities = rare_qualities(source)
    return tuple(
        Opportunity(ProtectionOutcome.TARGET_ILLEGAL, target, (source.id,), qualities)
        for target in candidate.legal_targets if permanent(entities, target) is not None
    )


def _removable_target(candidate, source: EntityState, lines: SidecarLookup | None) -> bool:
    """Whether a legal target of ``candidate`` can leave while another stays
    and the verdict holds: a spell or ability with two legal targets, acting
    through the face the snapshot shows, and (where the sidecars are read)
    one required target clause."""
    keys = candidate.ability or ()
    if candidate.land_play or len(candidate.legal_targets) < 2:
        return False
    if not _shows_the_acting_face(source, keys):
        return False
    return lines is None or _one_required_target(keys, lines)


def _one_required_target(keys, lines: SidecarLookup) -> bool:
    """Whether the acting chain has one targeting clause, whose minimum is
    one: removing a legal target from a clause that may take none leaves the
    spell playable, and a second clause (a fight spell's creature on each
    side) may have had that target as its only candidate, which a candidate
    lists no per-clause legality of. A chain the sidecars cannot show is
    refused."""
    line = lines.first_line(keys)
    if line is None or not line.script_text:
        return False
    clauses = [found for found in map(params, segments(line.script_text)) if "ValidTgts" in found]
    return len(clauses) == 1 and clauses[0].get("TargetMin", "1").strip() == "1"


def _shows_the_acting_face(source: EntityState, keys) -> bool:
    """Whether the card's snapshot holds the characteristics of what acts:
    its first face, shown face up, and not a split card, whose card in hand
    shows both halves at once. An Adventure or a back face cast from hand has
    characteristics the snapshot does not hold (Bonecrusher Giant shows the
    Giant while Stomp is cast)."""
    if source.face_state not in (None, FaceState.FRONT):
        return False
    if _is_split_card(source):
        return False
    return all(key.face == 0 for key in keys)


def _is_split_card(source: EntityState) -> bool:
    return " // " in (source.name or "")


def block_opportunities(record, payload: PlayabilityBlockersPayload,
                         entities) -> tuple[Opportunity, ...]:
    """A real block declaration's attacker, protected from one legal blocker."""
    if record.what_if or permanent(entities, payload.anchor_attacker) is None:
        return ()
    return tuple(
        Opportunity(ProtectionOutcome.BLOCK_ILLEGAL, payload.anchor_attacker, (blocker,),
                    rare_qualities(entities[blocker]))
        for blocker in payload.legal_blockers if permanent(entities, blocker) is not None
    )
