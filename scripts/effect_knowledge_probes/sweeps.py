"""The four board sweeps: edit one input of a real record, read the full model (FR-084).

====================  ===================================  ==========================
sweep                 edit                                 read
====================  ===================================  ==========================
toughness             one target creature's toughness,     that creature's predicted
                      1 to 8                               death
affordability         the acting player's untapped         the predicted affordable
                      production, none to cost + 2         verdict
board size            opposing creature count, 0 to 8,     predicted deaths summed
                      by copying a creature entity         over the board
damage                the acting text's ``NumDmg$``,       the toughness-4 target's
                      1 to 8, re-encoded                   predicted death
====================  ===================================  ==========================

Each edit changes exactly one input and returns a new record; the original is
never touched. A model that knows the threshold shows a step at the right value:
Lightning Bolt between toughness 3 and 4.

Nothing here does work at import time.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import replace

import numpy as np

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


# ── the edits ───────────────────────────────────────────────────────────


def _with_entities(record, entities):
    return replace(record, state=replace(record.state, entities=tuple(entities)))


def set_toughness(record, entity_id: str, toughness: int):
    """``record`` with one entity's total toughness set, and its damage cleared.

    The base toughness absorbs the change, so boosts and counters keep their
    own channels; damage is cleared so the toughness is what the ability faces.
    """
    from effects.domain.state_snapshot import PowerToughness

    out = []
    for entity in record.state.entities:
        if entity.id == entity_id:
            pt = entity.pt or PowerToughness(base=(0, 0))
            base_t = toughness - pt.boosts[1] - pt.counters[1]
            entity = replace(entity, pt=replace(pt, base=(pt.base[0], base_t)), damage=0)
        out.append(entity)
    return _with_entities(record, out)


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


def candidate_cost(record) -> list[str]:
    candidate = record.payload.candidates[0]
    cost = candidate.cost_after_adjustment
    mana = cost.get("mana") if isinstance(cost, dict) else None
    return cost_pips(mana if isinstance(mana, str) else "")


def set_production(record, player_id: str, pips: list[str], amount: int):
    """``record`` with the player's untapped production set to ``amount`` mana.

    Coloured pips are fed first, so the verdict should step at exactly the
    cost; whatever is left over is colourless. Floating mana is cleared so it
    cannot pay on the side.
    """
    production: dict[str, int] = {}
    remaining = amount
    for pip in pips:
        if remaining <= 0:
            break
        if pip != "1":
            production[pip] = production.get(pip, 0) + 1
            remaining -= 1
    if remaining > 0:
        production["C"] = production.get("C", 0) + remaining
    players = tuple(
        replace(p, untapped_production=production, floating_mana={})
        if p.id == player_id else p
        for p in record.state.players
    )
    return replace(record, state=replace(record.state, players=players))


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
    return _with_entities(record, [*kept, *copies])


def set_damage(text: str, amount: int) -> str:
    """The acting text with every literal ``NumDmg$`` set to ``amount``."""
    return _NUMDMG_RE.sub(lambda m: f"{m.group(1)}{amount}", text)


# ── which records each sweep runs on ────────────────────────────────────


def _damaged_creature(record):
    from effects.domain.effect_targets import derive_targets

    targets = derive_targets(record)
    for entity in record.state.entities:
        entry = targets.get(entity.id)
        if (entry is not None and entity.zone == "battlefield"
                and "creature" in entity.types
                and (entry.fields.get("damage_taken") or 0) > 0):
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
            if record.payload.candidates and any(p != "1" for p in candidate_cost(record)):
                self._found["affordability"].append((order, {"record_id": rid}))
            return
        if record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.RESOLUTION:
            return
        if not record.ability:
            return
        try:
            line = self.sidecars.line_for(record.ability[0])
        except KeyError:
            line = None
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


def _predict(probe_model, records, read) -> list[float]:
    """One forward over ``records``; ``read(outputs, verdict, surface, row)``."""
    import torch


    with torch.no_grad():
        batch, surfaces = probe_model.batcher.build(records, probe_model.encoder)
        hidden = probe_model.model(**batch)
        outputs = probe_model.model.per_entity(hidden)
        verdict = probe_model.model.verdict(hidden)
    return [read(outputs, verdict, surface, row) for row, surface in enumerate(surfaces)]


def _death(outputs, surface, row, entity_id) -> float:
    import torch

    from effects.domain.effect_model import FIELD_SLICES, GATE_INDEX, ZONE_OUTCOMES

    for index, slot in enumerate(surface.slots):
        if slot.entity_id == entity_id and slot.kind.name == "CARD":
            vector = outputs[row, index]
            start, end = FIELD_SLICES["zone_outcome"]
            died = torch.softmax(vector[start:end], -1)[ZONE_OUTCOMES.index("died")]
            return float(torch.sigmoid(vector[GATE_INDEX]) * died)
    return float("nan")


def run_sweep(name: str, probe_model, record, row: dict) -> list[dict]:
    """``[{value, prediction}]`` for one record of one sweep."""
    import torch

    from effects.domain.effect_model import VERDICT_BITS

    if name == "toughness":
        values = TOUGHNESS_VALUES
        edited = [set_toughness(record, row["entity"], t) for t in values]
        predictions = _predict(probe_model, edited,
                               lambda o, v, s, r: _death(o, s, r, row["entity"]))
    elif name == "affordability":
        pips = candidate_cost(record)
        values = tuple(range(0, len(pips) + 3))
        edited = [set_production(record, record.actor_player, pips, v) for v in values]
        affordable = VERDICT_BITS.index("affordable")
        predictions = _predict(probe_model, edited, lambda o, v, s, r: float(
            torch.sigmoid(v[r, affordable])))
    elif name == "board_size":
        values = BOARD_SIZES
        edited = [set_opposing_creatures(record, n) for n in values]

        def summed(o, v, s, r):
            return float(np.nansum([
                _death(o, s, r, slot.entity_id) for slot in s.slots
                if slot.kind.name == "CARD" and slot.entity_id is not None
                and _is_opposing(edited[r], slot.entity_id)
            ]))
        predictions = _predict(probe_model, edited, summed)
    elif name == "damage":
        values = DAMAGE_VALUES
        base = set_toughness(record, row["entity"], DAMAGE_TARGET_TOUGHNESS)
        predictions = []
        batcher = probe_model.batcher
        key = record.ability[0]
        original = batcher._resolve(key)
        try:
            for amount in values:
                # The acting key's text is the batcher's memo of it; pointing
                # the memo at the edited text is what makes the checkpoint's
                # own encoder encode the edited script, with nothing else moved.
                batcher._text_by_key[key] = (set_damage(original[0], amount), original[1])
                predictions += _predict(probe_model, [base],
                                        lambda o, v, s, r: _death(o, s, r, row["entity"]))
        finally:
            batcher._text_by_key[key] = original
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
