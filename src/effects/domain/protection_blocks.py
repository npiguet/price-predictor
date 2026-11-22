"""What a combat record's blocks leave of a protection copy (702.16f, 702.16j).

An attacker granted protection from a quality could not have been blocked by a
creature with it, so a damage-prevented copy of an attacker keeps no quality a
blocker of it has, and no protection from everything.

Torch-free: ``build-corpus``'s survey and write workers call it.
"""

from __future__ import annotations

from dataclasses import replace

from effects.domain.protection_model import Opportunity
from effects.domain.rule_outcomes import Combatant


def keeping_blocks(opportunities: tuple[Opportunity, ...], blocks: dict[str, tuple[str, ...]],
                   entities) -> tuple[Opportunity, ...]:
    """``opportunities`` where a protected attacker's blocks stay legal: each
    without the qualities a blocker of it has (``Combatant`` reads a
    changeling) and without protection from everything; one left with nothing
    is dropped. A protected blocker keeps its opportunity."""
    kept = (_keeping(o, blocks.get(o.protected, ()), entities) for o in opportunities)
    return tuple(o for o in kept if o.qualities or o.everything)


def _keeping(opportunity: Opportunity, blockers: tuple[str, ...], entities) -> Opportunity:
    if not blockers:
        return opportunity
    held = [Combatant.of(entities[b]).characteristics for b in blockers if b in entities]
    return replace(opportunity, everything=False, qualities=tuple(
        q for q in opportunity.qualities if not any(q.covers(c) for c in held)))
