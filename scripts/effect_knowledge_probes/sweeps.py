"""The four board sweeps: edit one input of a real record, read the full model (FR-084).

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
====================  ===================================  ==========================

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
import zlib
from dataclasses import replace

import numpy as np

from effect_knowledge_probes.board_edits import BoardEdit
from effect_knowledge_probes.common import predict

TOUGHNESS_VALUES = tuple(range(1, 9))
BOARD_SIZES = tuple(range(0, 9))
DAMAGE_VALUES = tuple(range(1, 9))
#: The damage sweep's fixed target toughness: Lightning Bolt's step is 3 → 4.
DAMAGE_TARGET_TOUGHNESS = 4
#: Records per sweep in a frozen probe set.
RECORDS_PER_SWEEP = 20

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


def cost_pips(cost: str) -> list[str]:
    """A ``{2}{R}``-style cost as pips: one colour symbol per coloured pip,
    ``"1"`` per point of generic mana."""
    pips: list[str] = []
    for pip in _PIP_RE.findall(cost or ""):
        if pip.isdigit():
            pips.extend(["1"] * int(pip))
        elif pip in ("W", "U", "B", "R", "G", "C"):
            pips.append(pip)
        else:
            pips.append(pip[0] if pip[0] in "WUBRG" else "1")
    return pips


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

    Its printed keys are the card's own two traits: ``spell`` 0, the permanent
    spell every permanent carries, which maps to no line, and ``spell`` 1, its
    mana ability. Built rather than copied from the record, because a record
    often holds no basic of the colour its cost names — a sweep that needed one
    would drop exactly the boards where the player cannot cast the card.
    """
    from effects.domain.provenance import ProvenanceKey
    from effects.domain.state_snapshot import EntityState

    land_type = _BASIC_LAND_TYPE[colour]
    script = f"cardsfolder/{land_type[0]}/{land_type}.txt"
    return EntityState(
        id=entity_id, name=land_type.capitalize(), zone="battlefield",
        controller=controller, types=("land",), subtypes=(land_type,),
        supertypes=("basic",),
        printed=tuple(ProvenanceKey(script, 0, "spell", index) for index in (0, 1)),
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
    untapped: list[int] = []
    added: list[str] = []
    for colour in wanted:
        spare = next((index for index in lands if index not in untapped
                      and basic_land_colour(entities[index]) is not None
                      and colour in (None, basic_land_colour(entities[index]))), None)
        if spare is not None:
            untapped.append(spare)
        elif colour is not None or coloured:
            added.append(colour or coloured[0])
        else:
            raise ValueError("generic mana past the player's basics needs a coloured pip")
    for index in lands:
        entities[index] = replace(entities[index], tapped=index not in untapped)
    ids = {entity.id for entity in entities}
    for colour in added:
        fresh = next(f"{player_id}~land{k}" for k in range(len(ids) + 1)
                     if f"{player_id}~land{k}" not in ids)
        ids.add(fresh)
        entities.append(basic_land(colour, fresh, player_id))
    made = [basic_land_colour(entities[index]) for index in untapped] + added
    production = {colour: made.count(colour) for colour in _PRODUCTION_COLOURS}
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
    payload = record.payload
    if not payload.candidates or not land_payable(candidate_mana(record)):
        return False
    player = record.actor_player
    mine = [e for e in record.state.entities
            if e.zone == "battlefield" and e.controller == player]
    candidate = set(payload.candidates[0].ability)
    if any(candidate & set(e.printed) for e in mine if "land" in e.types):
        return False
    return not any(_has_mana_ability(e, sidecars) for e in mine
                   if "land" not in e.types and not e.tapped)


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


def set_damage_amount(text: str, amount: int) -> str:
    """The acting text with every literal ``NumDmg$`` set to ``amount``."""
    return _NUMDMG_RE.sub(lambda m: f"{m.group(1)}{amount}", text)


# ── which records each sweep runs on ────────────────────────────────────


def _damaged_creature(record):
    from effects.domain.effect_head_layout import HeadField
    from effects.domain.effect_targets import derive_targets

    targets = derive_targets(record)
    for entity in record.state.entities:
        entry = targets.get(entity.id)
        if (entry is not None and entity.zone == "battlefield"
                and "creature" in entity.types
                and (entry.fields.get(HeadField.DAMAGE_TAKEN) or 0) > 0):
            return entity.id
    return None


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
        }

    def observe(self, record) -> None:
        from effects.domain.records import Moment, PlayabilityDecisionPayload, RecordKind

        order = zlib.crc32(record.record_id.encode("utf-8"))
        rid = record.record_id
        if isinstance(record.payload, PlayabilityDecisionPayload):
            if affordability_ready(record, self.sidecars):
                self._found["affordability"].append((order, {"record_id": rid}))
            return
        if record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.RESOLUTION:
            return
        if not record.ability:
            return
        line = self.sidecars.line_or_none(record.ability[0])
        script = (line.script_text or "") if line is not None else ""
        api = line.script_api_type if line is not None else None
        target = _damaged_creature(record)
        if target is not None and _NUMDMG_RE.search(script):
            self._found["toughness"].append((order, {"record_id": rid, "entity": target}))
            if line.line_kind in ("spell", "triggered"):
                host = "spell" if line.line_kind == "spell" else "triggered"
                self._found["damage"].append(
                    (order, {"record_id": rid, "entity": target, "host": host}))
        if api in _SWEEPER_APIS and opposing_creatures(record):
            self._found["board_size"].append((order, {"record_id": rid}))

    def result(self) -> dict:
        def first(rows):
            return [row for _, row in sorted(rows, key=lambda pair: pair[0])][: self.per_sweep]

        out = {name: first(rows) for name, rows in self._found.items() if name != "damage"}
        damage = self._found["damage"]
        out["damage"] = (first([r for r in damage if r[1]["host"] == "spell"])
                         + first([r for r in damage if r[1]["host"] == "triggered"]))
        return out


def select_sweep_records(records, sidecars, *, per_sweep: int = RECORDS_PER_SWEEP) -> dict:
    """:class:`SweepSelector` over a whole iterable of records."""
    selector = SweepSelector(sidecars, per_sweep)
    for record in records:
        selector.observe(record)
    return selector.result()


# ── running them ────────────────────────────────────────────────────────


def death(outputs, surface, row, entity_id) -> float:
    """One card's predicted death, gated (:class:`HeadReading`); NaN off the board."""
    from effects.domain.effect_head_layout import ZoneOutcome
    from effects.domain.head_decoding import HeadReading

    for index, slot in enumerate(surface.slots):
        if slot.entity_id == entity_id and slot.kind.name == "CARD":
            return float(HeadReading(outputs[row, index]).p_zone(ZoneOutcome.DIED))
    return float("nan")


def run_sweep(name: str, probe_model, record, row: dict) -> list[dict]:
    """``[{value, prediction}]`` for one record of one sweep."""
    import torch

    from effects.domain.effect_head_layout import VERDICT_BITS

    if name == "toughness":
        values = TOUGHNESS_VALUES
        edited = [BoardEdit(record).set_toughness(row["entity"], t).record for t in values]
        predictions = predict(probe_model, edited,
                              lambda o, v, s, r: death(o, s, r, row["entity"]))
    elif name == "affordability":
        pips = candidate_cost(record)
        values = tuple(range(0, len(pips) + 3))
        edited = [set_untapped_lands(record, record.actor_player, pips, v) for v in values]
        affordable = VERDICT_BITS.index("affordable")
        predictions = predict(probe_model, edited, lambda o, v, s, r: float(
            torch.sigmoid(v[r, affordable])))
    elif name == "board_size":
        values = BOARD_SIZES
        edited = [set_opposing_creatures(record, n) for n in values]

        def summed(o, v, s, r):
            return float(np.nansum([
                death(o, s, r, slot.entity_id) for slot in s.slots
                if slot.kind.name == "CARD" and slot.entity_id is not None
                and _is_opposing(edited[r], slot.entity_id)
            ]))
        predictions = predict(probe_model, edited, summed)
    elif name == "damage":
        values = DAMAGE_VALUES
        base = BoardEdit(record).set_toughness(row["entity"], DAMAGE_TARGET_TOUGHNESS).record
        predictions = []
        batcher = probe_model.batcher
        key = record.ability[0]
        original = batcher.text_of(key)
        for amount in values:
            with batcher.reading_as(key, set_damage_amount(original, amount)):
                predictions += predict(probe_model, [base],
                                       lambda o, v, s, r: death(o, s, r, row["entity"]))
    else:
        raise KeyError(name)
    return [{"value": int(v), "prediction": p} for v, p in zip(values, predictions)]


def _is_opposing(record, entity_id: str) -> bool:
    entity = record.state.entity(entity_id)
    return (entity is not None and entity.controller != record.actor_player
            and "creature" in entity.types)


def summarize(curves: list[list[dict]]) -> list[dict]:
    """The mean prediction per value over a sweep's records."""
    by_value: dict[int, list[float]] = {}
    for curve in curves:
        for point in curve:
            by_value.setdefault(point["value"], []).append(point["prediction"])
    return [{"value": v, "prediction": float(np.nanmean(p)), "n": len(p)}
            for v, p in sorted(by_value.items())]
