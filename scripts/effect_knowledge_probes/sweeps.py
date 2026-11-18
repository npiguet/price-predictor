"""The board sweeps: edit one input of a real record, read the full model (FR-084).

====================  ===================================  ==========================
sweep                 edit                                 read
====================  ===================================  ==========================
toughness             one target creature's toughness,     that creature's predicted
                      1 to 8                               death
affordability         the acting player's untapped lands,  the predicted affordable
                      none to cost + 2, all basics         verdict
board size            opposing creature count, 0 to 8,     predicted deaths summed
                      by copying a creature entity         over the board
damage                the acting text's ``NumDmg$``,       the toughness-4 target's
                      1 to 8, re-encoded                   predicted death
activated damage      the toughness sweep on an activated  the creature's predicted
                      "any target" ping, its text read     death and affected gate,
                      as written and with "target          per wording
                      creature" in place of "any target"
====================  ===================================  ==========================

The activated-damage sweep measures one wording (FR-074): an activated ability that
deals damage to "any target" (Prodigal Sorcerer's ping) whose target was a creature. The
rule does not care how the target was worded, so a model that has tied the wording to
its target gives the same curve both ways; one that doubts a creature is affected by an
"any target" ping shows it as a lower gate on the text as written. The second reading
re-encodes the acting text with ``ValidTgts$ Any`` as ``ValidTgts$ Creature`` and "any
target" as "target creature" (:func:`target_creature_wording`), through the checkpoint's
own encoder, as the damage sweep re-encodes ``NumDmg$``.

Each edit changes exactly one input and returns a new record; the original is
never touched. An input the record states in two places is edited in both, so
the board never contradicts itself: the affordability sweep's mana is the
player's untapped lands and the production count written beside them. A model
that knows the threshold shows a step at the right value: Lightning Bolt
between toughness 3 and 4.

Nothing here does work at import time.
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np

from effect_knowledge_probes import readout
from effect_knowledge_probes.board_edits import BoardEdit
from effect_knowledge_probes.common import predict
from effects.domain.script_text import root_param
from effects.domain.seeded_draw import id_hash

TOUGHNESS_VALUES = tuple(range(1, 9))
BOARD_SIZES = tuple(range(0, 9))
DAMAGE_VALUES = tuple(range(1, 9))
#: The damage sweep's fixed target toughness: Lightning Bolt's step is 3 → 4.
DAMAGE_TARGET_TOUGHNESS = 4
#: Records per sweep a probing run reads, chosen in ``crc32`` order.
RECORDS_PER_SWEEP = 20
#: The activated-damage sweep's two readings of the acting text.
AS_WRITTEN = "any target"
TARGET_CREATURE = "target creature"
WORDINGS: tuple[str, ...] = (AS_WRITTEN, TARGET_CREATURE)
_ANY_TARGET_RE = re.compile(r"\bany target\b", re.IGNORECASE)
_VALID_ANY_RE = re.compile(r"(ValidTgts\$\s*)Any\b")

_SWEEPER_APIS = frozenset({"DestroyAll", "DamageAll", "SacrificeAll", "ChangeZoneAll"})
_NUMDMG_RE = re.compile(r"(NumDmg\$\s*)(\d+)")
_PIP_RE = re.compile(r"\{([^}]+)\}")

#: The colour each basic land type taps for.
BASIC_LAND_COLOURS: dict[str, str] = {
    "plains": "W", "island": "U", "swamp": "B", "mountain": "R", "forest": "G",
}
_BASIC_LAND_TYPE = {colour: subtype for subtype, colour in BASIC_LAND_COLOURS.items()}
#: A pip one basic land pays alone: a colour, or a hybrid of two.
_LAND_PIP_RE = re.compile(r"[WUBRG](/[WUBRG])?")
#: The colours the collector writes ``untapped_production`` for; never colourless.
_PRODUCTION_COLOURS = ("W", "U", "B", "R", "G")
_MANA_APIS = frozenset({"Mana", "ManaReflected"})


# ── the edits ───────────────────────────────────────────────────────────


def parse_cost(cost: str) -> tuple[list[tuple[str, ...]], int]:
    """``(coloured pips, generic)`` of a ``{2}{R}``-style cost: each coloured
    pip as the colours that pay it.

    ``{2}{R}{W/U}`` is ``([("R",), ("W", "U")], 2)``. Read only for costs
    :func:`land_payable` admits, so a ``{X}`` or ``{2/W}`` pip never reaches it.
    """
    coloured: list[tuple[str, ...]] = []
    generic = 0
    for pip in _PIP_RE.findall(cost or ""):
        if pip.isdigit():
            generic += int(pip)
        else:
            coloured.append(tuple(pip.split("/")))
    return coloured, generic


def cost_pips(cost: str) -> list[str]:
    """A cost as pips: ``"1"`` per point of generic mana, then one colour per
    coloured pip, a hybrid by its first colour."""
    coloured, generic = parse_cost(cost)
    return ["1"] * generic + [pip[0] for pip in coloured]


def candidate_mana(record) -> str:
    """The first candidate's mana cost as the collector wrote it, e.g. ``{2}{R}``."""
    cost = record.payload.candidates[0].cost_after_adjustment
    mana = cost.get("mana") if isinstance(cost, dict) else None
    return mana if isinstance(mana, str) else ""


def candidate_cost(record) -> list[str]:
    return cost_pips(candidate_mana(record))


def land_payable(cost: str) -> bool:
    """Whether basic lands pay ``cost`` one pip each, with a coloured pip among them.

    Generic, coloured and two-colour hybrid pips only. ``{C}`` needs a
    colourless source, ``{S}`` a snow one and ``{X}`` an announced number, and a
    Phyrexian or ``{2/W}`` pip is payable without its colour; on any of them the
    verdict would not step at the land count the sweep reads it against.
    """
    pips = _PIP_RE.findall(cost or "")
    return (all(p.isdigit() or _LAND_PIP_RE.fullmatch(p) for p in pips)
            and any(not p.isdigit() for p in pips))


def basic_land_colour(entity) -> str | None:
    """The one colour a basic land taps for; None for any other permanent.

    A basic given every basic land type (Prismatic Omen) taps for more than one
    colour and counts as no basic here.
    """
    if "land" not in entity.types or "basic" not in entity.supertypes:
        return None
    colours = {BASIC_LAND_COLOURS[t] for t in entity.subtypes if t in BASIC_LAND_COLOURS}
    return colours.pop() if len(colours) == 1 else None


def basic_land(colour: str, entity_id: str, controller: str):
    """An untapped basic land of ``colour``, as the collector writes one.

    Its one line is the card's mana ability, the ``spell`` 1 trait of its
    script (``spell`` 0, the permanent spell every permanent carries, maps to
    no line). Built rather than copied from the record, because a record often
    holds no basic of the colour its cost names — a sweep that needed one would
    drop exactly the boards where the player cannot cast the card.
    """
    from effects.domain.provenance import ProvenanceKey
    from effects.domain.state_snapshot import (
        CardChoices,
        CardHistory,
        CardLine,
        EntityState,
        FaceState,
        GrantedTemporary,
        LineKind,
    )

    land_type = _BASIC_LAND_TYPE[colour]
    mana_ability = ProvenanceKey(f"cardsfolder/{land_type[0]}/{land_type}.txt", 0, "spell", 1)
    return EntityState(
        id=entity_id, name=land_type.capitalize(), zone="battlefield",
        controller=controller, types=("land",), subtypes=(land_type,),
        supertypes=("basic",), granted_temporary=GrantedTemporary(hidden_keywords=()),
        lines=(CardLine(keys=(mana_ability,), kind=LineKind.ACTIVATED, granted=False),),
        owner=controller, face_state=FaceState.FRONT, chosen=CardChoices(),
        history=CardHistory(), order_rank=0,
    )


def set_untapped_lands(record, player_id: str, pips: list[str], amount: int):
    """``record`` with exactly ``amount`` of the player's lands untapped, all basics.

    The model reads a player's mana in two places: the ``untapped_production``
    count on the player, and the lands themselves, each with its ``tapped``
    flag and its mana ability among its ability tokens. An edit to the count
    alone leaves the lands contradicting it, and the model reads the lands; so
    this edits the lands and sets the count to what they make.

    One basic land per mana. Coloured pips are fed first, in pip order (a
    hybrid pip by its first colour), so the verdict should step at exactly the
    cost; the rest is generic. The player's own basics are untapped where they
    fit — a colour's basics for its pips, any basic for generic mana — and a
    fresh basic is added where none is left: of the pip's colour, or of the
    cost's first colour for generic mana. Every other land the player controls
    is tapped, and floating mana is cleared so it cannot pay on the side.
    Mana sources other than lands are left as they are, which is why
    :func:`affordability_ready` keeps a record with an untapped one out of the
    sweep.
    """
    coloured = [pip for pip in pips if pip != "1"]
    wanted = [*coloured[:amount], *[None] * max(0, amount - len(coloured))]
    entities = list(record.state.entities)
    lands = [index for index, entity in enumerate(entities)
             if entity.zone == "battlefield" and entity.controller == player_id
             and "land" in entity.types]
    untapped, added = _plan_lands(entities, lands, wanted, coloured)
    for index in lands:
        entities[index] = replace(entities[index], tapped=index not in untapped)
    _append_basics(entities, added, player_id)
    made = [basic_land_colour(entities[index]) for index in untapped] + added
    production = {colour: made.count(colour) for colour in _PRODUCTION_COLOURS}
    return _with_production(record, entities, player_id, production)


def _can_supply(entity, colour: str | None) -> bool:
    """Whether a basic land taps for ``colour``, or for anything when it is None."""
    land_colour = basic_land_colour(entity)
    return land_colour is not None and colour in (None, land_colour)


def _plan_lands(entities, lands: list[int], wanted: list, coloured: list[str]):
    """``(untapped, added)``: the player's lands to leave untapped, one per wanted
    mana, and the colours of the fresh basics for the mana they cannot supply."""
    untapped: list[int] = []
    added: list[str] = []
    for colour in wanted:
        spare = next((index for index in lands if index not in untapped
                      and _can_supply(entities[index], colour)), None)
        if spare is not None:
            untapped.append(spare)
        else:
            added.append(_fresh_colour(colour, coloured))
    return untapped, added


def _fresh_colour(colour: str | None, coloured: list[str]) -> str:
    """The colour of the basic added for ``colour``; generic mana takes the cost's first."""
    if colour is None and not coloured:
        raise ValueError("generic mana past the player's basics needs a coloured pip")
    return colour or coloured[0]


def _append_basics(entities: list, added: list[str], player_id: str) -> None:
    """Append a fresh untapped basic of each colour in ``added``, on fresh ids."""
    ids = {entity.id for entity in entities}
    for colour in added:
        fresh = next(f"{player_id}~land{k}" for k in range(len(ids) + 1)
                     if f"{player_id}~land{k}" not in ids)
        ids.add(fresh)
        entities.append(basic_land(colour, fresh, player_id))


def _with_production(record, entities: list, player_id: str, production: dict):
    """``record`` on ``entities``, the player's production set and floating mana cleared."""
    players = tuple(
        replace(p, untapped_production=production, floating_mana={})
        if p.id == player_id else p
        for p in record.state.players
    )
    return replace(record, state=replace(record.state, players=players,
                                         entities=tuple(entities)))


def _has_mana_ability(entity, sidecars) -> bool:
    for key in (*entity.ability_keys, *entity.granted_temporary.abilities):
        line = sidecars.line_or_none(key)
        if line is not None and line.script_api_type in _MANA_APIS:
            return True
    return False


def affordability_ready(record, sidecars) -> bool:
    """Whether :func:`set_untapped_lands` edits ``record`` into a consistent board.

    The first candidate's cost must be :func:`land_payable`; the candidate must
    not be an ability of one of the player's lands, which the edit would tap or
    untap along with its mana; and the player must control no untapped mana
    source other than a land. A mana creature or rock the edit leaves alone
    would pay on the side, and the production count cannot be trusted to show
    one, because the collector counts no colourless mana.
    """
    if not _first_candidate_payable(record):
        return False
    mine = [e for e in record.state.entities
            if e.zone == "battlefield" and e.controller == record.actor_player]
    if _candidate_is_land_ability(record, mine):
        return False
    return not any(_has_mana_ability(e, sidecars) for e in mine
                   if "land" not in e.types and not e.tapped)


def _first_candidate_payable(record) -> bool:
    return bool(record.payload.candidates) and land_payable(candidate_mana(record))


def _candidate_is_land_ability(record, mine: list) -> bool:
    """Whether the first candidate is an ability of one of ``mine``'s lands."""
    candidate = set(record.payload.candidates[0].ability)
    return any(candidate & set(e.printed) for e in mine if "land" in e.types)


def opposing_creatures(record) -> list:
    return [e for e in record.state.entities
            if e.zone == "battlefield" and "creature" in e.types
            and e.controller != record.actor_player]


def set_opposing_creatures(record, count: int):
    """``record`` with exactly ``count`` opposing creatures, copies of the first.

    Copies get fresh ids so every one is its own slot; everything else on the
    board is kept.
    """
    opposing = opposing_creatures(record)
    template = opposing[0]
    kept = [e for e in record.state.entities if e not in opposing]
    copies = [
        template if k == 0 else replace(template, id=f"{template.id}~{k}")
        for k in range(count)
    ]
    return BoardEdit(record).with_entities([*kept, *copies]).record


def target_creature_wording(text: str) -> str:
    """An "any target" text read as "target creature": ``ValidTgts$ Any`` as
    ``ValidTgts$ Creature`` on the script surface, the words "any target" as
    "target creature" on both (a description, the converted prose)."""
    text = _VALID_ANY_RE.sub(lambda m: f"{m.group(1)}Creature", text)
    return _ANY_TARGET_RE.sub(lambda m: "Target creature" if m.group(0)[0] == "A"
                              else "target creature", text)


def set_amount(text: str, key: str, amount: int) -> str:
    """The acting text with every literal ``key$`` (``NumDmg``, ``NumCards``) set
    to ``amount``."""
    return re.sub(rf"({key}\$\s*)(\d+)", lambda m: f"{m.group(1)}{amount}", text)


# ── which records each sweep runs on ────────────────────────────────────


def _is_damaged_creature(entity, targets: dict) -> bool:
    from effects.domain.effect_head_layout import HeadField

    entry = targets.get(entity.id)
    if entry is None or not _is_battlefield_creature(entity):
        return False
    return (entry.fields.get(HeadField.DAMAGE_TAKEN) or 0) > 0


def _is_battlefield_creature(entity) -> bool:
    return entity.zone == "battlefield" and "creature" in entity.types


def _damaged_creature(record):
    from effects.domain.effect_targets import derive_targets

    targets = derive_targets(record)
    return next((e.id for e in record.state.entities if _is_damaged_creature(e, targets)),
                None)


def _resolves_an_ability(record) -> bool:
    from effects.domain.records import Moment, RecordKind

    return (record.kind is RecordKind.RESOLUTION and record.moment is Moment.RESOLUTION
            and bool(record.ability))


def _is_any_target_ping(line) -> bool:
    return line.line_kind == "activated" and root_param(
        line.script_text or "", "ValidTgts") == "Any"


class SweepSelector:
    """The real records each sweep edits, chosen in ``crc32`` order of record id.

    Observes records one at a time and keeps only each candidate's row, so the
    choice is the same however the records are read and none is held.
    """

    def __init__(self, sidecars, per_sweep: int = RECORDS_PER_SWEEP) -> None:
        self.sidecars = sidecars
        self.per_sweep = per_sweep
        self._found: dict[str, list[tuple[int, dict]]] = {
            "toughness": [], "affordability": [], "board_size": [], "damage": [],
            "activated_damage": [],
        }

    def _add(self, name: str, order: int, row: dict) -> None:
        self._found[name].append((order, row))

    def observe(self, record) -> None:
        from effects.domain.records import PlayabilityDecisionPayload

        order = id_hash(record.record_id)
        if isinstance(record.payload, PlayabilityDecisionPayload):
            if affordability_ready(record, self.sidecars):
                self._add("affordability", order, {"record_id": record.record_id})
        elif _resolves_an_ability(record):
            self._observe_resolution(record, order)

    def _observe_resolution(self, record, order: int) -> None:
        rid = record.record_id
        line = self.sidecars.line_or_none(record.ability[0])
        script = (line.script_text or "") if line is not None else ""
        api = line.script_api_type if line is not None else None
        target = _damaged_creature(record)
        if target is not None and _NUMDMG_RE.search(script):
            self._observe_damage(order, {"record_id": rid, "entity": target}, line)
        if api in _SWEEPER_APIS and opposing_creatures(record):
            self._add("board_size", order, {"record_id": rid})

    def _observe_damage(self, order: int, row: dict, line) -> None:
        """The damaged creature's row on the sweeps its ability's kind takes."""
        self._add("toughness", order, dict(row))
        if line.line_kind in ("spell", "triggered"):
            self._add("damage", order, {**row, "host": line.line_kind})
        if _is_any_target_ping(line):
            self._add("activated_damage", order, dict(row))

    def result(self) -> dict:
        def first(rows):
            return [row for _, row in sorted(rows, key=lambda pair: pair[0])][: self.per_sweep]

        out = {name: first(rows) for name, rows in self._found.items() if name != "damage"}
        damage = self._found["damage"]
        out["damage"] = (first([r for r in damage if r[1]["host"] == "spell"])
                         + first([r for r in damage if r[1]["host"] == "triggered"]))
        return out


# ── running them ────────────────────────────────────────────────────────


def _points(values, predictions) -> list[dict]:
    return [{"value": int(v), "prediction": p} for v, p in zip(values, predictions)]


def _toughness_sweep(probe_model, record, row: dict) -> list[dict]:
    edited = [BoardEdit(record).set_toughness(row["entity"], t).record
              for t in TOUGHNESS_VALUES]
    dies = readout.at_card("dies", row.get("entity"))
    return _points(TOUGHNESS_VALUES, predict(probe_model, edited, dies))


def _affordability_sweep(probe_model, record, row: dict) -> list[dict]:
    pips = candidate_cost(record)
    values = tuple(range(0, len(pips) + 3))
    edited = [set_untapped_lands(record, record.actor_player, pips, v) for v in values]
    return _points(values, predict(probe_model, edited, readout.of_record("affordable")))


def _board_size_sweep(probe_model, record, row: dict) -> list[dict]:
    edited = [set_opposing_creatures(record, n) for n in BOARD_SIZES]
    deaths = readout.at_every_card("dies")

    def summed(o, v, s, r):
        return float(np.nansum([p for entity, p in deaths(o, v, s, r).items()
                                if _is_opposing(edited[r], entity)]))
    return _points(BOARD_SIZES, predict(probe_model, edited, summed))


def _damage_sweep(probe_model, record, row: dict) -> list[dict]:
    dies = readout.at_card("dies", row.get("entity"))
    base = BoardEdit(record).set_toughness(row["entity"], DAMAGE_TARGET_TOUGHNESS).record
    predictions = []
    batcher = probe_model.batcher
    key = record.ability[0]
    original = batcher.text_of(key)
    for amount in DAMAGE_VALUES:
        with batcher.reading_as(key, set_amount(original, "NumDmg", amount)):
            predictions += predict(probe_model, [base], dies)
    return _points(DAMAGE_VALUES, predictions)


def _activated_damage_sweep(probe_model, record, row: dict) -> list[dict]:
    return activated_damage_sweep(probe_model, record, row["entity"])


_SWEEPS = {
    "toughness": _toughness_sweep,
    "affordability": _affordability_sweep,
    "board_size": _board_size_sweep,
    "damage": _damage_sweep,
    "activated_damage": _activated_damage_sweep,
}


def run_sweep(name: str, probe_model, record, row: dict) -> list[dict]:
    """``[{value, prediction}]`` for one record of one sweep."""
    return _SWEEPS[name](probe_model, record, row)


def activated_damage_sweep(probe_model, record, entity: str) -> list[dict]:
    """``[{value, wording, prediction, affected}]``: the toughness sweep at
    ``entity`` read with the acting text as written and in
    :data:`TARGET_CREATURE` wording, ``prediction`` the predicted death and
    ``affected`` the gate."""
    edited = [BoardEdit(record).set_toughness(entity, t).record for t in TOUGHNESS_VALUES]
    dies, gate = readout.at_card("dies", entity), readout.at_card("affected", entity)

    def both(o, v, s, r):
        return dies(o, v, s, r), gate(o, v, s, r)

    batcher = probe_model.batcher
    key = record.ability[0]
    original = batcher.text_of(key)
    out = []
    for wording in WORDINGS:
        if wording == AS_WRITTEN:
            readings = predict(probe_model, edited, both)
        else:
            with batcher.reading_as(key, target_creature_wording(original)):
                readings = predict(probe_model, edited, both)
        out += [{"value": int(v), "wording": wording, "prediction": p, "affected": a}
                for v, (p, a) in zip(TOUGHNESS_VALUES, readings)]
    return out


def _is_opposing(record, entity_id: str) -> bool:
    entity = record.state.entity(entity_id)
    return (entity is not None and entity.controller != record.actor_player
            and "creature" in entity.types)


def _points_by_value(curves: list[list[dict]]) -> dict[tuple, list[dict]]:
    by_value: dict[tuple, list[dict]] = {}
    for point in (point for curve in curves for point in curve):
        by_value.setdefault((point.get("wording", ""), point["value"]), []).append(point)
    return by_value


def _mean_row(wording: str, value: int, points: list[dict]) -> dict:
    row = {"value": value, "n": len(points),
           "prediction": float(np.nanmean([p["prediction"] for p in points]))}
    if wording:
        row["wording"] = wording
        row["affected"] = float(np.nanmean([p["affected"] for p in points]))
    return row


def summarize(curves: list[list[dict]]) -> list[dict]:
    """The mean prediction per value over a sweep's records; per wording and
    value, the gate's mean beside it, for the activated-damage sweep."""
    return [_mean_row(wording, value, points)
            for (wording, value), points in sorted(_points_by_value(curves).items())]
