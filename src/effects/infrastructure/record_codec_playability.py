"""The wire form of the playability kind's payloads: a decision's candidates,
an attackers record and a blockers record, with their forbidden entities.

Part of :mod:`effects.infrastructure.record_codec`, whose payload table
(:mod:`effects.infrastructure.record_codec_payloads`) dispatches to them.
"""

from __future__ import annotations

from typing import Any

from effects.domain.gen3_members import Site, owes, require_always
from effects.domain.records import (
    Candidate,
    FailedCheck,
    ForbiddenEntity,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
)
from effects.infrastructure.record_codec_values import (
    _keys_from_json,
    _keys_to_json,
    _tuple_or_none,
)


def _candidate_to_json(candidate: Candidate) -> dict:
    """One candidate; ``timing``, ``failed_checks`` and ``land_play`` only
    where read, as gen-3 writes them."""
    verdict = {
        "can_play": candidate.can_play,
        "affordable": candidate.affordable,
        "has_legal_target": candidate.has_legal_target,
    }
    if candidate.timing is not None:
        verdict["timing"] = candidate.timing
    out = {
        "ability": _keys_to_json(candidate.ability),
        "verdict": verdict,
        "legal_targets": list(candidate.legal_targets),
        "cost_after_adjustment": dict(candidate.cost_after_adjustment),
        "responsible_static": _keys_to_json(candidate.responsible_static),
    }
    if candidate.failed_checks is not None:
        out["failed_checks"] = [
            {"check": failed.check,
             "responsible_static": _keys_to_json(failed.responsible_static)}
            for failed in candidate.failed_checks]
    if candidate.land_play is not None:
        out["land_play"] = candidate.land_play
    return out


@owes(Site.CANDIDATE)
def _candidate_from_json(data: dict) -> Candidate:
    verdict = data["verdict"]
    require_always(Site.VERDICT, verdict)
    failed = data["failed_checks"]
    return Candidate(
        ability=_keys_from_json(data["ability"]),
        can_play=verdict["can_play"],
        affordable=verdict["affordable"],
        has_legal_target=verdict["has_legal_target"],
        legal_targets=tuple(data["legal_targets"]),
        cost_after_adjustment=dict(data["cost_after_adjustment"]),
        responsible_static=_keys_from_json(data["responsible_static"]),
        timing=verdict["timing"],
        failed_checks=tuple(
            _failed_check_from_json(entry) for entry in failed) if failed is not None else None,
        land_play=data["land_play"],
    )


@owes(Site.FAILED_CHECK)
def _failed_check_from_json(data: dict) -> FailedCheck:
    return FailedCheck(check=data["check"],
                       responsible_static=_keys_from_json(data["responsible_static"]))


def _decision_to_json(payload: PlayabilityDecisionPayload) -> dict:
    return {"candidates": [_candidate_to_json(c) for c in payload.candidates]}


def _decision_from_json(data: dict) -> PlayabilityDecisionPayload:
    return PlayabilityDecisionPayload(
        candidates=tuple(_candidate_from_json(c) for c in data["candidates"]),
    )


def _forbidden_to_json(forbidden: tuple[ForbiddenEntity, ...]) -> list[dict]:
    """``responsible_own``, ``responsible_unresolved`` and
    ``required_elsewhere`` after ``responsible_static``, in that order, each
    only when set, as the collector writes them."""
    return [
        {
            "entity": f.entity,
            "responsible_static": _keys_to_json(f.responsible_static),
            **({"responsible_own": True} if f.responsible_own else {}),
            **({"responsible_unresolved": str(f.responsible_unresolved)}
               if f.responsible_unresolved is not None else {}),
            **({"required_elsewhere": True} if f.required_elsewhere else {}),
        }
        for f in forbidden
    ]


def _forbidden_from_json(data: Any) -> tuple[ForbiddenEntity, ...]:
    return tuple(_forbidden_entity_from_json(f) for f in data or ())


@owes(Site.FORBIDDEN)
def _forbidden_entity_from_json(data: dict) -> ForbiddenEntity:
    return ForbiddenEntity(
        entity=data["entity"],
        responsible_static=_keys_from_json(data["responsible_static"]),
        required_elsewhere=bool(data.get("required_elsewhere", False)),
        responsible_own=bool(data.get("responsible_own", False)),
        responsible_unresolved=data.get("responsible_unresolved"),
    )


def _attackers_to_json(payload: PlayabilityAttackersPayload) -> dict:
    """An anchored (gen-3) record writes its anchor's members first."""
    out: dict = {}
    if payload.anchor_attacker is not None:
        out = {
            "anchor_attacker": payload.anchor_attacker,
            "legal_defenders": list(payload.legal_defenders),
            "must_attack": payload.must_attack,
        }
        if payload.tapped_by_attack is not None:
            out["tapped_by_attack"] = payload.tapped_by_attack
    return out | {
        "legal_attackers": list(payload.legal_attackers),
        "forbidden": _forbidden_to_json(payload.forbidden),
    }


def _attackers_from_json(data: dict) -> PlayabilityAttackersPayload:
    return PlayabilityAttackersPayload(
        legal_attackers=tuple(data["legal_attackers"]),
        forbidden=_forbidden_from_json(data["forbidden"]),
        anchor_attacker=data["anchor_attacker"],
        legal_defenders=tuple(data["legal_defenders"]),
        must_attack=data["must_attack"],
        tapped_by_attack=data.get("tapped_by_attack"),
    )


def _blockers_to_json(payload: PlayabilityBlockersPayload) -> dict:
    out = {
        "anchor_attacker": payload.anchor_attacker,
        "legal_blockers": list(payload.legal_blockers),
        "forbidden": _forbidden_to_json(payload.forbidden),
    }
    if payload.must_block is not None:
        out["must_block"] = list(payload.must_block)
    out["min_blockers"] = payload.min_blockers
    return out


def _blockers_from_json(data: dict) -> PlayabilityBlockersPayload:
    return PlayabilityBlockersPayload(
        anchor_attacker=data["anchor_attacker"],
        legal_blockers=tuple(data["legal_blockers"]),
        forbidden=_forbidden_from_json(data["forbidden"]),
        min_blockers=data["min_blockers"],
        must_block=_tuple_or_none(data["must_block"]),
    )
