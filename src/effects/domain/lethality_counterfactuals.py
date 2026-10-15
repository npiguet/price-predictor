"""Counterfactual copies of damage and -X/-X resolutions, labelled by the rules (FR-063e).

Forge's AI aims its removal at creatures it can kill, so almost every resolution
that damages or shrinks a creature kills it, and the corpus rarely shows the
amount falling short. A model fits that with "a small creature dies" and never
learns to compare the amount with the creature's toughness. A copy made here
changes one hit creature's toughness and the amount, and is labelled by the
state-based rule: a creature with toughness 0 or less dies, and so does one
with damage at least its toughness (rule 704.5f-g). The rules as written are
the target, not Forge's AI. The amount is a literal in the script, edited in
the copy's text, or the X the caster paid, edited in the copy's references.

Only records whose real outcome already obeys the rule are copied, and only
where nothing else changes what the rule decides:

- the acting chain holds exactly one ``DealDamage``/``DamageAll`` segment with
  a fixed or paid amount, or one ``Pump``/``PumpAll`` lowering toughness by
  one, and no segment remembers, conditions on or replaces anything
  (:data:`BLOCKED_PARAMS`), since those can turn on the death itself;
- every creature on the battlefield the amount hit changes in nothing but the
  amount and its zone, and every player it hit loses exactly the amount;
- the source carries no deathtouch, wither, infect or lifelink, and no hit
  creature carries indestructible, a shield counter or an attachment.

Nothing here does work at import time.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace

from effects.domain.effect_model import ZONE_OUTCOMES
from effects.domain.effect_targets import EntityTargets
from effects.domain.provenance import ProvenanceKey
from effects.domain.value_targets import params, segments

#: The tree a copy's acting key names, so its edited text never shares a key
#: with the real line's.
COUNTERFACTUAL_TREE = "counterfactual"
#: Amounts and toughnesses a copy draws from, uniformly: lethal a little more
#: than half the time.
DRAW_RANGE = (1, 8)
DAMAGE_APIS = frozenset({"DealDamage", "DamageAll"})
SHRINK_APIS = frozenset({"Pump", "PumpAll"})
#: Parameters, in any segment, after which a death can change what else
#: happens: an exile in its place, a follow-up on what was damaged, an amount
#: split or sourced elsewhere. A chain carrying one is not copied.
BLOCKED_PREFIXES = ("Remember", "Condition", "ReplaceDying", "Imprint")
BLOCKED_PARAMS = frozenset({
    "DamageSource", "DividedAsYouChoose", "DivideEvenly", "ExcessSVar",
    "ExcessDamage", "KW",
})
#: What a hit creature's targets may hold, by kind: anything else means
#: another part of the chain touched it.
HIT_FIELDS = {
    "damage": frozenset({"damage_taken", "zone_outcome"}),
    "shrink": frozenset({"toughness_delta", "power_delta", "pt_duration", "zone_outcome"}),
}
#: Keywords on the source that change what damage does.
SOURCE_EXCLUSIONS = frozenset({"deathtouch", "wither", "infect", "lifelink"})

_INT = re.compile(r"^-?\d+$")
_DIED = ZONE_OUTCOMES.index("died")
_STAYED = ZONE_OUTCOMES.index("stayed")


@dataclass(frozen=True)
class Qualified:
    """A record the copies can be made from."""

    kind: str  # "damage" or "shrink"
    amount: int  # damage dealt, or toughness lost
    #: Where the amount is stated: "text" (a literal) or "x" (the X paid).
    stated: str
    symmetric: bool  # a shrink that takes as much power as toughness
    hit: tuple[str, ...]  # the creatures the amount hit, by entity id
    players: tuple[str, ...]  # the players it hit, by player id


@dataclass(frozen=True)
class Counterfactual:
    """One copy: the edited record, its acting text, and its relabelled targets."""

    record: object
    #: The copy's acting key and its edited text, which the batcher's memo
    #: carries for this step; None where the text is the real line's.
    key: ProvenanceKey | None
    text: str | None
    targets: dict[str, EntityTargets]


def toughness_of(entity) -> int:
    pt = entity.pt
    return pt.base[1] + pt.boosts[1] + pt.counters[1]


def lethal(kind: str, amount: int, entity) -> bool:
    """Whether ``amount`` kills ``entity`` under rule 704.5f-g."""
    toughness = toughness_of(entity)
    if kind == "damage":
        return toughness <= 0 or entity.damage + amount >= toughness
    left = toughness - amount
    return left <= 0 or (entity.damage > 0 and entity.damage >= left)


def _blocked(found: dict[str, str]) -> bool:
    for key, value in found.items():
        if key in BLOCKED_PARAMS or key.startswith(BLOCKED_PREFIXES):
            return True
        if key == "Defined" and "Remembered" in value:
            return True
    return False


def _literal(value: str | None) -> int | None:
    return int(value) if value is not None and _INT.match(value) else None


def _amount(found: dict[str, str], record) -> tuple[str, int, str, bool] | None:
    """``(kind, amount, stated, symmetric)`` of one effect segment, or None."""
    api = found.get("SP") or found.get("AB") or found.get("DB")
    x = record.state.refs.x
    if api in DAMAGE_APIS:
        raw = found.get("NumDmg")
        if raw == "X" and x is not None:
            return "damage", x, "x", False
        amount = _literal(raw)
        return None if amount is None else ("damage", amount, "text", False)
    if api in SHRINK_APIS:
        raw, power = found.get("NumDef"), found.get("NumAtt")
        if raw == "-X" and x is not None:
            return "shrink", x, "x", power == "-X"
        shrink = _literal(raw)
        if shrink is None or shrink >= 0:
            return None
        return "shrink", -shrink, "text", power is not None and _literal(power) == shrink
    return None


def qualify(record, line, targets: dict[str, EntityTargets], keywords_of) -> Qualified | None:
    """``record`` as copy material, or None. ``keywords_of(entity)`` names its keywords."""
    from effects.domain.records import Moment, RecordKind

    if record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.RESOLUTION:
        return None
    if line is None or not line.script_text:
        return None
    found_amounts = []
    for segment in segments(line.script_text):
        found = params(segment)
        if _blocked(found):
            return None
        api = found.get("SP") or found.get("AB") or found.get("DB")
        if api in DAMAGE_APIS or api in SHRINK_APIS:
            found_amounts.append(_amount(found, record))
    if len(found_amounts) != 1 or found_amounts[0] is None:
        return None
    kind, amount, stated, symmetric = found_amounts[0]
    if amount <= 0:
        return None

    state = record.state
    entities = {entity.id: entity for entity in state.entities}
    players = {player.id for player in state.players}
    source = entities.get(state.refs.source)
    if source is not None and keywords_of(source) & SOURCE_EXCLUSIONS:
        return None
    attached = {entity.attached_to for entity in state.entities if entity.attached_to}
    hit: list[str] = []
    hit_players: list[str] = []
    for subject, entry in targets.items():
        if not entry.affected:
            continue
        fields = entry.fields
        if subject in players:
            if fields.get("poison_delta"):
                return None
            life = fields.get("life_delta")
            if life is not None and life != 0:
                if kind != "damage" or life != -amount:
                    return None
                hit_players.append(subject)
            continue
        entity = entities.get(subject)
        if entity is None or entity.zone != "battlefield":
            continue
        took = (fields.get("damage_taken") if kind == "damage"
                else fields.get("toughness_delta"))
        if not took:
            continue
        if took != (amount if kind == "damage" else -amount):
            return None
        if "creature" not in entity.types or entity.pt is None:
            return None
        if set(fields) - HIT_FIELDS[kind]:
            return None
        if ("indestructible" in keywords_of(entity) or subject in attached
                or any(name.lower() == "shield" for name in entity.counters)):
            return None
        zone = fields.get("zone_outcome", _STAYED)
        if zone not in (_STAYED, _DIED) or (zone == _DIED) != lethal(kind, amount, entity):
            return None
        hit.append(subject)
    if not hit:
        return None
    return Qualified(kind=kind, amount=amount, stated=stated, symmetric=symmetric,
                     hit=tuple(hit), players=tuple(hit_players))


def edit_text(text: str, qualified: Qualified, amount: int) -> str | None:
    """The acting text with its literal amount set to ``amount``; None where it has none."""
    n = qualified.amount
    if qualified.kind == "damage":
        edited, count = re.subn(rf"(NumDmg\$\s*){n}\b", rf"\g<1>{amount}", text)
        if not count:
            return None
        return re.sub(rf"\b{n} damage\b", f"{amount} damage", edited)
    edited, count = re.subn(rf"(NumDef\$\s*)-{n}\b", rf"\g<1>-{amount}", text)
    if not count:
        return None
    if qualified.symmetric:
        edited = re.sub(rf"(NumAtt\$\s*)-{n}\b", rf"\g<1>-{amount}", edited)
        return edited.replace(f"-{n}/-{n}", f"-{amount}/-{amount}")
    return re.sub(rf"/-{n}\b", f"/-{amount}", edited)


def _relabel(entry: EntityTargets, qualified: Qualified, amount: int, dies: bool
             ) -> EntityTargets:
    fields = dict(entry.fields)
    if qualified.kind == "damage":
        fields["damage_taken"] = float(amount)
    else:
        fields["toughness_delta"] = float(-amount)
        if qualified.symmetric:
            fields["power_delta"] = float(-amount)
    if dies:
        fields["zone_outcome"] = _DIED
    else:
        fields.pop("zone_outcome", None)
    return EntityTargets(affected=True, fields=fields)


def counterfactual(record, qualified: Qualified, targets: dict[str, EntityTargets],
                   text: str | None, rng: random.Random, copy: int) -> Counterfactual:
    """One copy of ``record``: a new amount, a new toughness on one hit creature,
    and every creature and player the amount hit relabelled by the rule.

    A literal amount is edited in the text, which then needs a key of its own; a
    paid X is edited in the references, and where the text names neither the
    amount stays as it was and only the toughness moves.
    """
    low, high = DRAW_RANGE
    drawn = rng.randint(low, high)
    chosen = rng.choice(qualified.hit)
    toughness = rng.randint(low, high)
    amount, key, edited = qualified.amount, None, None
    refs = record.state.refs
    if qualified.stated == "x":
        amount, refs = drawn, replace(refs, x=drawn)
    elif text:
        edited = edit_text(text, qualified, drawn)
        if edited is not None:
            amount = drawn
            original = record.ability[0]
            key = replace(original, script_file=(
                f"{COUNTERFACTUAL_TREE}/{amount}/{original.script_file}"))
    entities = []
    for entity in record.state.entities:
        if entity.id == chosen:
            pt = entity.pt
            base = toughness - pt.boosts[1] - pt.counters[1]
            entity = replace(entity, pt=replace(pt, base=(pt.base[0], base)), damage=0)
        entities.append(entity)
    copied = replace(
        record,
        record_id=f"{record.record_id}~cf{copy}",
        ability=(key,) if key is not None else record.ability,
        state=replace(record.state, entities=tuple(entities), refs=refs),
    )
    by_id = {entity.id: entity for entity in entities}
    relabelled = dict(targets)
    for subject in qualified.hit:
        relabelled[subject] = _relabel(
            targets[subject], qualified, amount,
            lethal(qualified.kind, amount, by_id[subject]))
    for subject in qualified.players:
        fields = dict(targets[subject].fields)
        fields["life_delta"] = float(-amount)
        relabelled[subject] = EntityTargets(affected=True, fields=fields)
    return Counterfactual(record=copied, key=key, text=edited, targets=relabelled)
