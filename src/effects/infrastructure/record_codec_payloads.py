"""The wire form of each kind's payload, and the one table that maps every
payload type to its pair of converters.

Part of :mod:`effects.infrastructure.record_codec`; the playability kind's
payloads are :mod:`effects.infrastructure.record_codec_playability`'s.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from effects.domain.gen3_members import Site, owes, payload_site, require_always
from effects.domain.keyword_names import normalize_keyword
from effects.domain.records import (
    PAYLOAD_TYPES,
    ActivationPayload,
    CombatPayload,
    ConditionMet,
    ContinuousPayload,
    Contribution,
    Costs,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
    RewritePayload,
    RewriteResult,
    TriggerPayload,
    TurnActionPayload,
)
from effects.infrastructure.record_codec_playability import (
    _attackers_from_json,
    _attackers_to_json,
    _blockers_from_json,
    _blockers_to_json,
    _decision_from_json,
    _decision_to_json,
)
from effects.infrastructure.record_codec_values import (
    _counters_from_json,
    _event_from_json,
    _events_from_json,
    _events_to_json,
    _keys_from_json,
    _keys_to_json,
    _tuple_or_none,
)

# ── payloads ────────────────────────────────────────────────────────────


def _costs_to_json(costs: Costs) -> dict:
    return {
        "mana_by_color": dict(costs.mana_by_color),
        "tapped": list(costs.tapped),
        "life": costs.life,
        "sacrificed": list(costs.sacrificed),
        "discarded": list(costs.discarded),
        "exiled": list(costs.exiled),
        "returned": list(costs.returned),
        "revealed": list(costs.revealed),
        "exerted": list(costs.exerted),
        "untapped": list(costs.untapped),
        "convoked": list(costs.convoked),
        "delved": list(costs.delved),
        "counters": dict(costs.counters),
    }


def _costs_from_json(data: dict) -> Costs:
    return Costs(
        mana_by_color=dict(data.get("mana_by_color", {})),
        tapped=tuple(data.get("tapped", ())),
        life=data.get("life", 0),
        sacrificed=tuple(data.get("sacrificed", ())),
        discarded=tuple(data.get("discarded", ())),
        exiled=tuple(data.get("exiled", ())),
        counters=_counters_from_json(data.get("counters", {})),
        returned=tuple(data.get("returned", ())),
        revealed=tuple(data.get("revealed", ())),
        exerted=tuple(data.get("exerted", ())),
        untapped=tuple(data.get("untapped", ())),
        convoked=tuple(data.get("convoked", ())),
        delved=tuple(data.get("delved", ())),
    )


def _activation_to_json(payload: ActivationPayload) -> dict:
    return {"events": _events_to_json(payload.events),
            "costs": _costs_to_json(payload.costs)}


def _activation_from_json(data: dict) -> ActivationPayload:
    """A cost half, ``{events, costs}``."""
    return ActivationPayload(
        costs=_costs_from_json(data["costs"]),
        events=_events_from_json(data["events"]),
    )


def _resolution_to_json(payload: ResolutionPayload) -> dict:
    out: dict = {"events": _events_to_json(payload.events)}
    if payload.conditions_met is not None:
        out["conditions_met"] = [
            {"segment": entry.segment, "met": entry.met} for entry in payload.conditions_met]
    if payload.outcome is not None or payload.outcome_written:
        out["outcome"] = payload.outcome.value if payload.outcome is not None else None
    return out


def _resolution_from_json(data: dict) -> ResolutionPayload:
    conditions = data["conditions_met"]
    return ResolutionPayload(
        events=_events_from_json(data["events"]),
        conditions_met=tuple(
            ConditionMet(segment=entry["segment"], met=entry["met"]) for entry in conditions
        ) if conditions is not None else None,
        outcome=ResolutionOutcome(data["outcome"]) if data["outcome"] is not None else None,
        outcome_written="outcome" in data,
    )


def _rewrite_to_json(payload: RewritePayload) -> dict:
    """Gen-3's members after ``replaced_by``, each only where the collector
    writes it: ``incoming_unknown`` when true, the buffered three on a
    replacement that set an ability aside."""
    out = {
        "incoming": payload.incoming.as_dict(),
        "outgoing": payload.outgoing.as_dict() if payload.outgoing else None,
        "result": payload.result.value if payload.result else None,
        "replaced_by": _keys_to_json(payload.replaced_by),
    }
    if payload.incoming_unknown:
        out["incoming_unknown"] = True
    if payload.buffered is not None:
        out["buffered"] = _keys_to_json(payload.buffered)
        out["buffered_ran"] = bool(payload.buffered_ran)
        out["buffered_events"] = _events_to_json(payload.buffered_events)
    return out


def _rewrite_from_json(data: dict) -> RewritePayload:
    outgoing = data["outgoing"]
    result = data["result"]
    buffered = data.get("buffered")
    return RewritePayload(
        incoming=_event_from_json(data["incoming"]),
        outgoing=_event_from_json(outgoing) if outgoing else None,
        result=RewriteResult(result) if result else None,
        replaced_by=_keys_from_json(data["replaced_by"]),
        incoming_unknown=bool(data.get("incoming_unknown", False)),
        buffered=_keys_from_json(buffered) if buffered is not None else None,
        buffered_ran=bool(data.get("buffered_ran")) if buffered is not None else None,
        buffered_events=_events_from_json(data.get("buffered_events")),
    )


def _contribution_to_json(contribution: Contribution) -> dict:
    """``set_pt`` and ``controller`` are written only when set, as the
    collector writes them."""
    out: dict = {"entity": contribution.entity, "pt_boost": list(contribution.pt_boost)}
    if contribution.set_pt is not None:
        out["set_pt"] = list(contribution.set_pt)
    out.update({
        "keywords": list(contribution.keywords), "types": list(contribution.types),
        "colors": list(contribution.colors),
    })
    if contribution.abilities is not None:
        out["abilities"] = list(contribution.abilities)
    if contribution.hidden_keywords is not None:
        out["hidden_keywords"] = list(contribution.hidden_keywords)
    out["name"] = contribution.name
    if contribution.controller is not None:
        out["controller"] = contribution.controller
    return out


@owes(Site.CONTRIBUTION)
def _contribution_from_json(data: dict) -> Contribution:
    return Contribution(
        entity=data["entity"],
        pt_boost=tuple(data["pt_boost"]),
        set_pt=tuple(data["set_pt"]) if data.get("set_pt") is not None else None,
        keywords=tuple(normalize_keyword(k) for k in data["keywords"]),
        types=tuple(data["types"]),
        colors=tuple(data["colors"]),
        abilities=_tuple_or_none(data["abilities"]),
        hidden_keywords=_tuple_or_none(data["hidden_keywords"]),
        name=data["name"],
        controller=data.get("controller"),
    )


def _continuous_to_json(payload: ContinuousPayload) -> dict:
    return {
        "contributions": [_contribution_to_json(c) for c in payload.contributions],
        "board_hash": payload.board_hash,
    }


def _continuous_from_json(data: dict) -> ContinuousPayload:
    return ContinuousPayload(
        contributions=tuple(
            _contribution_from_json(c) for c in data["contributions"]
        ),
        board_hash=data["board_hash"],
    )


def _combat_to_json(payload: CombatPayload) -> dict:
    return {
        "attackers": list(payload.attackers),
        "blocks": {k: list(v) for k, v in payload.blocks.items()},
        "assignment_choices": {
            k: dict(v) for k, v in payload.assignment_choices.items()
        },
        "events": _events_to_json(payload.events),
        "probed_keyword": payload.probed_keyword,
        "probed_entity": payload.probed_entity,
    }


def _is_observed_combat(payload) -> bool:
    return (isinstance(payload, CombatPayload)
            and payload.probed_keyword is None and payload.probed_entity is None)


def _combat_from_json(data: dict) -> CombatPayload:
    return CombatPayload(
        attackers=tuple(data["attackers"]),
        blocks={k: tuple(v) for k, v in data["blocks"].items()},
        assignment_choices={k: dict(v) for k, v in data["assignment_choices"].items()},
        events=_events_from_json(data["events"]),
        probed_keyword=data.get("probed_keyword"),
        probed_entity=data.get("probed_entity"),
    )


def _trigger_to_json(payload: TriggerPayload) -> dict:
    return {"event": payload.event.as_dict(), "fired": payload.fired}


def _trigger_from_json(data: dict) -> TriggerPayload:
    return TriggerPayload(
        event=_event_from_json(data["event"]), fired=bool(data["fired"]),
    )


def _turn_action_to_json(payload: TurnActionPayload) -> dict:
    """The game-end marker's ``records_withheld`` after its (empty) events."""
    out: dict = {"events": _events_to_json(payload.events)}
    if payload.records_withheld is not None:
        out["records_withheld"] = payload.records_withheld
    return out


def _turn_action_from_json(data: dict) -> TurnActionPayload:
    return TurnActionPayload(events=_events_from_json(data["events"]),
                             records_withheld=data.get("records_withheld"))


@dataclass(frozen=True, slots=True)
class _PayloadCodec:
    encode: Callable[[Any], dict]
    decode: Callable[[dict], Any]


#: Every payload type's wire form: the values of ``PAYLOAD_TYPES``, each once.
_PAYLOAD_CODECS: dict[type, _PayloadCodec] = {
    ActivationPayload: _PayloadCodec(_activation_to_json, _activation_from_json),
    ResolutionPayload: _PayloadCodec(_resolution_to_json, _resolution_from_json),
    RewritePayload: _PayloadCodec(_rewrite_to_json, _rewrite_from_json),
    ContinuousPayload: _PayloadCodec(_continuous_to_json, _continuous_from_json),
    CombatPayload: _PayloadCodec(_combat_to_json, _combat_from_json),
    TriggerPayload: _PayloadCodec(_trigger_to_json, _trigger_from_json),
    PlayabilityDecisionPayload: _PayloadCodec(_decision_to_json, _decision_from_json),
    PlayabilityAttackersPayload: _PayloadCodec(_attackers_to_json, _attackers_from_json),
    PlayabilityBlockersPayload: _PayloadCodec(_blockers_to_json, _blockers_from_json),
    TurnActionPayload: _PayloadCodec(_turn_action_to_json, _turn_action_from_json),
}


def _payload_to_json(payload) -> dict:
    codec = _PAYLOAD_CODECS.get(type(payload))
    if codec is None:
        raise TypeError(f"no wire form for payload type {type(payload).__name__}")
    return codec.encode(payload)


def _payload_from_json(
    record: dict, kind: RecordKind, discriminator: Moment | StrEnum | None,
):
    """``record``'s payload, the members its site owes checked first."""
    payload_type = PAYLOAD_TYPES.get((kind, discriminator))
    if payload_type is None:
        # A ValueError, as the record's own discriminator check raises.
        raise ValueError(
            f"no payload is defined for kind={kind.value} with "
            f"moment/subkind={discriminator.value if discriminator else None}"
        )
    data = record["payload"]
    require_always(payload_site(record), data)
    return _PAYLOAD_CODECS[payload_type].decode(data)
