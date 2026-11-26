"""The wire form of one snapshot entity: its lines, P/T, gen-3 facts and status.

Part of :mod:`effects.infrastructure.record_codec`; each block is one
``_x_to_json`` / ``_x_from_json`` pair.
"""

from __future__ import annotations

from effects.domain.gen3_members import Site, owes, require_line_kind
from effects.domain.keyword_names import normalize_keyword
from effects.domain.state_snapshot import (
    CardChoices,
    CardHistory,
    CardLine,
    CombatStatus,
    CostsPaid,
    EntityState,
    GrantedTemporary,
    PowerToughness,
    RealFace,
    StackExtras,
)
from effects.infrastructure.record_codec_values import (
    _absent_or,
    _counters_from_json,
    _keys_from_json,
    _keys_to_json,
    _list_or_none,
    _optional,
    _tuple_or_none,
)


def _pt_boosts_to_json(pt: PowerToughness) -> dict:
    """An entity's ``pt_boosts``; ``base_pt`` is ``pt.base`` written alone."""
    return {"temporary": list(pt.boosts), "counters": list(pt.counters)}


def _pt_from_json(data: dict) -> PowerToughness | None:
    """An entity's ``base_pt`` and ``pt_boosts``, both null on a non-creature."""
    base, boosts = data["base_pt"], data["pt_boosts"]
    if base is None:
        return None
    return PowerToughness(
        base=tuple(base),
        boosts=tuple(boosts.get("temporary", (0, 0))),
        counters=tuple(boosts.get("counters", (0, 0))),
    )


def _line_to_json(line: CardLine) -> dict:
    """One line; the optional members only where set, as Java writes them."""
    out: dict = {"keys": _keys_to_json(line.keys), "kind": line.kind,
                 "granted": line.granted}
    if line.grant is not None:
        out["grant"] = line.grant
    if line.active is not None:
        out["active"] = line.active
    if line.cast_with is not None:
        out["cast_with"] = line.cast_with
    if line.chosen is not None:
        out["chosen"] = list(line.chosen)
    if line.fragment is not None:
        out["fragment"] = line.fragment
    if line.affected is not None:
        out["affected"] = list(line.affected)
    return out


@owes(Site.LINE)
def _line_from_json(data: dict) -> CardLine:
    require_line_kind(data)
    return CardLine(
        keys=_keys_from_json(data["keys"]),
        kind=data["kind"],
        granted=data["granted"],
        grant=data.get("grant"),
        active=data.get("active"),
        cast_with=data.get("cast_with"),
        chosen=_tuple_or_none(data.get("chosen")),
        fragment=data.get("fragment"),
        affected=_tuple_or_none(data.get("affected")),
    )


# ── a gen-3 card's facts ────────────────────────────────────────────────


def _real_face_to_json(face: RealFace) -> dict:
    return {
        "name": face.name, "types": list(face.types), "subtypes": list(face.subtypes),
        "colors": list(face.colors), "mana_value": face.mana_value,
        "base_pt": _list_or_none(face.base_pt),
    }


def _real_face_from_json(data: dict) -> RealFace:
    return RealFace(
        name=data["name"], types=tuple(data.get("types", ())),
        subtypes=tuple(data.get("subtypes", ())), colors=tuple(data.get("colors", ())),
        mana_value=data.get("mana_value", 0), base_pt=_tuple_or_none(data.get("base_pt")),
    )


#: A card's ``chosen`` members in the order the collector writes them, each
#: only when set; the list-valued ones.
_CHOICE_MEMBERS = ("colors", "type", "type2", "number", "player", "cards", "names",
                   "direction", "mode", "even_odd", "sector", "stored_rolls")
_CHOICE_LISTS = frozenset({"colors", "cards", "names", "stored_rolls"})


def _choices_to_json(chosen: CardChoices) -> dict:
    out: dict = {}
    for name in _CHOICE_MEMBERS:
        value = getattr(chosen, name)
        if value is not None:
            out[name] = list(value) if name in _CHOICE_LISTS else value
    return out


def _choices_from_json(data: dict) -> CardChoices:
    return CardChoices(**{
        name: tuple(data[name]) if name in _CHOICE_LISTS else data[name]
        for name in _CHOICE_MEMBERS if data.get(name) is not None
    })


def _costs_paid_to_json(paid: CostsPaid) -> dict:
    return {"mana": dict(paid.mana), "cast_from": paid.cast_from,
            "mana_rider": paid.mana_rider}


def _costs_paid_from_json(data: dict) -> CostsPaid:
    return CostsPaid(mana=dict(data.get("mana", {})), cast_from=data.get("cast_from"),
                     mana_rider=data.get("mana_rider", False))


def _history_to_json(history: CardHistory) -> dict:
    return {
        "entered_this_turn": history.entered_this_turn,
        "attacked_this_turn": history.attacked_this_turn,
        "blocked_this_turn": history.blocked_this_turn,
        "loyalty_activated": history.loyalty_activated,
    }


def _history_from_json(data: dict) -> CardHistory:
    return CardHistory(
        entered_this_turn=data.get("entered_this_turn", False),
        attacked_this_turn=data.get("attacked_this_turn", False),
        blocked_this_turn=data.get("blocked_this_turn", False),
        loyalty_activated=data.get("loyalty_activated", False),
    )


def _combat_status_to_json(combat: CombatStatus) -> dict:
    return {
        "attacking": combat.attacking,
        "blocking": list(combat.blocking),
        "blocked_by": list(combat.blocked_by),
        "became_blocked": combat.became_blocked,
    }


def _combat_status_from_json(data: dict) -> CombatStatus:
    return CombatStatus(
        attacking=data.get("attacking"),
        blocking=tuple(data.get("blocking", ())),
        blocked_by=tuple(data.get("blocked_by", ())),
        became_blocked=data.get("became_blocked", False),
    )


def _granted_temporary_to_json(granted: GrantedTemporary) -> dict:
    out = {
        "keywords": list(granted.keywords),
        "abilities": _keys_to_json(granted.abilities),
    }
    if granted.hidden_keywords is not None:
        out["hidden_keywords"] = list(granted.hidden_keywords)
    return out


def _granted_temporary_from_json(data: dict) -> GrantedTemporary:
    """Keywords in the package's spelling; hidden keywords as Forge writes
    them, which name no keyword."""
    return GrantedTemporary(
        keywords=tuple(normalize_keyword(k) for k in data.get("keywords", ())),
        abilities=_keys_from_json(data.get("abilities")),
        hidden_keywords=_tuple_or_none(data.get("hidden_keywords")),
    )


def _stack_extras_to_json(extras: StackExtras) -> dict:
    return {
        "targets": list(extras.targets),
        "per_target_amounts": dict(extras.per_target_amounts),
        "up_to_counts": dict(extras.up_to_counts),
    }


def _stack_extras_from_json(data: dict) -> StackExtras:
    return StackExtras(
        targets=tuple(data.get("targets", ())),
        per_target_amounts=dict(data.get("per_target_amounts", {})),
        up_to_counts=dict(data.get("up_to_counts", {})),
    )


def _entity_to_json(entity: EntityState) -> dict:
    """One entity, its members in the order ``FE/EntityJson.java`` writes them."""
    pt = entity.pt
    return {
        "id": entity.id, "name": entity.name, "zone": entity.zone,
        "owner": entity.owner, "controller": entity.controller, "face": entity.face,
        "face_state": entity.face_state, "copy_source": entity.copy_source,
        "token_script_id": entity.token_script_id, "types": list(entity.types),
        "subtypes": list(entity.subtypes), "supertypes": list(entity.supertypes),
        "colors": list(entity.colors), "mana_value": entity.mana_value,
        "mana_cost": _absent_or(dict, entity.mana_cost),
        "base_pt": list(pt.base) if pt is not None else None,
        "pt_boosts": _pt_boosts_to_json(pt) if pt is not None else None,
        "printed_loyalty": entity.printed_loyalty,
        "printed_defense": entity.printed_defense,
        "tapped": entity.tapped, "sick": entity.sick, "damage": entity.damage,
        "counters": dict(entity.counters),
        "combat": _optional(_combat_status_to_json, entity.combat),
        "attached_to": entity.attached_to,
        "face_down": entity.face_down,
        "real_face": _absent_or(_real_face_to_json, entity.real_face),
        "designations": list(entity.designations),
        "x_paid": entity.x_paid,
        "chosen": _absent_or(_choices_to_json, entity.chosen),
        "costs_paid": _absent_or(_costs_paid_to_json, entity.costs_paid),
        "history": _absent_or(_history_to_json, entity.history),
        "order_rank": entity.order_rank,
        "granted_temporary": _granted_temporary_to_json(entity.granted_temporary),
        "lines": [_line_to_json(line) for line in entity.lines],
        "stack_extras": _optional(_stack_extras_to_json, entity.stack_extras),
        "role": entity.role,
    }


@owes(Site.ENTITY)
def _entity_from_json(data: dict) -> EntityState:
    """One entity, whose ``printed`` :class:`EntityState` derives from its
    ``lines``."""
    return _entity(
        data, pt=_pt_from_json(data),
        lines=tuple(_line_from_json(line) for line in data["lines"]),
        owner=data["owner"], face_state=data["face_state"],
        mana_cost=_absent_or(dict, data["mana_cost"]),
        printed_loyalty=data["printed_loyalty"],
        printed_defense=data["printed_defense"],
        real_face=_absent_or(_real_face_from_json, data["real_face"]),
        designations=tuple(data["designations"]),
        x_paid=data["x_paid"],
        chosen=_absent_or(_choices_from_json, data["chosen"]),
        costs_paid=_absent_or(_costs_paid_from_json, data["costs_paid"]),
        history=_absent_or(_history_from_json, data["history"]),
        order_rank=data["order_rank"],
        role=data["role"],
    )


def _entity(data: dict, **facts) -> EntityState:
    """An entity's identity, characteristics and status, plus its ``facts``."""
    return EntityState(
        id=data["id"], name=data["name"], zone=data["zone"],
        controller=data["controller"], face=data["face"],
        copy_source=data["copy_source"],
        token_script_id=data["token_script_id"],
        types=tuple(data["types"]),
        subtypes=tuple(data["subtypes"]),
        supertypes=tuple(data["supertypes"]),
        colors=tuple(data["colors"]),
        mana_value=data["mana_value"],
        tapped=data["tapped"], sick=data["sick"],
        damage=data["damage"], counters=_counters_from_json(data["counters"]),
        combat=_optional(_combat_status_from_json, data["combat"]),
        attached_to=data["attached_to"],
        face_down=data["face_down"],
        granted_temporary=_granted_temporary_from_json(data["granted_temporary"]),
        stack_extras=_optional(_stack_extras_from_json, data["stack_extras"]),
        **facts,
    )
