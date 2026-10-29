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
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum, StrEnum, auto

from effects.domain.effect_head_layout import ZONE_OUTCOMES
from effects.domain.effect_targets import EntityTargets
from effects.domain.provenance import ProvenanceKey, SourceTree
from effects.domain.script_text import api_of, params, segments

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
#: Keywords on the source that change what damage does.
SOURCE_EXCLUSIONS = frozenset({"deathtouch", "wither", "infect", "lifelink"})

_INT = re.compile(r"^-?\d+$")
_DIED = ZONE_OUTCOMES.index("died")
_STAYED = ZONE_OUTCOMES.index("stayed")


class EffectKind(StrEnum):
    """What the one amount does to a creature it hits.

    A ``StrEnum``, so a member still compares equal to the string it replaced.
    """

    DAMAGE = "damage"  # marks damage on it
    SHRINK = "shrink"  # lowers its toughness, -X/-X style


class Stated(StrEnum):
    """Where the amount is written, which decides where a copy edits it."""

    TEXT = "text"  # a literal in the script, edited in the copy's text
    X = "x"  # the X the caster paid, edited in the copy's references


def _damage_lethal(toughness: int, damage: int, amount: int) -> bool:
    return toughness <= 0 or damage + amount >= toughness


def _shrink_lethal(toughness: int, damage: int, amount: int) -> bool:
    left = toughness - amount
    return left <= 0 or (damage > 0 and damage >= left)


def _prose_damage(n: int, amount: int, symmetric: bool) -> list[tuple[str, str]]:
    return [(rf"\b{n} damage\b", f"{amount} damage")]


def _prose_shrink(n: int, amount: int, symmetric: bool) -> list[tuple[str, str]]:
    if symmetric:
        return [(rf"-{n}/-{n}\b", f"-{amount}/-{amount}")]
    return [(rf"/-{n}\b", f"/-{amount}")]


def _edit_damage(text: str, n: int, amount: int, symmetric: bool) -> str | None:
    edited, count = re.subn(rf"(NumDmg\$\s*){n}\b", rf"\g<1>{amount}", text)
    if not count:
        return None
    return re.sub(rf"\b{n} damage\b", f"{amount} damage", edited)


def _edit_shrink(text: str, n: int, amount: int, symmetric: bool) -> str | None:
    edited, count = re.subn(rf"(NumDef\$\s*)-{n}\b", rf"\g<1>-{amount}", text)
    if not count:
        return None
    if symmetric:
        edited = re.sub(rf"(NumAtt\$\s*)-{n}\b", rf"\g<1>-{amount}", edited)
        return edited.replace(f"-{n}/-{n}", f"-{amount}/-{amount}")
    return re.sub(rf"/-{n}\b", f"/-{amount}", edited)


@dataclass(frozen=True)
class _KindRules:
    """Everything that differs between damage and a shrink, in one place."""

    #: The target field a hit creature records the amount in, and the sign it
    #: carries there: damage taken is positive, a toughness change negative.
    field: str
    sign: int
    #: What a hit creature's targets may hold: anything else means another
    #: part of the chain touched it.
    hit_fields: frozenset[str]
    #: ``(toughness, marked damage, amount) -> dies`` under rule 704.5f-g.
    lethal: Callable[[int, int, int], bool]
    #: ``(text, old amount, new amount, symmetric) -> edited text or None``.
    edit: Callable[[str, int, int, bool], str | None]
    #: ``(old amount, new amount, symmetric) -> (pattern, replacement)`` pairs
    #: that edit the amount where the converted prose states it.
    prose: Callable[[int, int, bool], list[tuple[str, str]]]


_RULES: dict[EffectKind, _KindRules] = {
    EffectKind.DAMAGE: _KindRules(
        field="damage_taken", sign=1,
        hit_fields=frozenset({"damage_taken", "zone_outcome"}),
        lethal=_damage_lethal, edit=_edit_damage, prose=_prose_damage,
    ),
    EffectKind.SHRINK: _KindRules(
        field="toughness_delta", sign=-1,
        hit_fields=frozenset({"toughness_delta", "power_delta", "pt_duration", "zone_outcome"}),
        lethal=_shrink_lethal, edit=_edit_shrink, prose=_prose_shrink,
    ),
}


@dataclass(frozen=True)
class Qualified:
    """A record the copies can be made from."""

    kind: EffectKind
    amount: int  # damage dealt, or toughness lost
    stated: Stated
    symmetric: bool  # a shrink that takes as much power as toughness
    hit: tuple[str, ...]  # the creatures the amount hit, by entity id
    players: tuple[str, ...]  # the players it hit, by player id


@dataclass(frozen=True)
class Counterfactual:
    """One copy: the edited record, its acting text, and its relabelled targets."""

    record: object
    #: The copy's acting key and its edited encoding text, which the batcher's
    #: memo carries for this step; None where the text is the real line's.
    key: ProvenanceKey | None
    text: str | None
    targets: dict[str, EntityTargets]
    #: The edited script, which the value head's targets read whatever the
    #: surface encodes, and the prose's role spans moved with its edit.
    script: str | None = None
    role_spans: tuple = ()


def lethal(kind: str, amount: int, entity) -> bool:
    """Whether ``amount`` kills ``entity`` under rule 704.5f-g."""
    return _RULES[EffectKind(kind)].lethal(entity.pt.total[1], entity.damage, amount)


def _blocked(found: dict[str, str]) -> bool:
    for key, value in found.items():
        if key in BLOCKED_PARAMS or key.startswith(BLOCKED_PREFIXES):
            return True
        if key == "Defined" and "Remembered" in value:
            return True
    return False


def _literal(value: str | None) -> int | None:
    return int(value) if value is not None and _INT.match(value) else None


def _kind_of(api: str | None) -> EffectKind | None:
    if api in DAMAGE_APIS:
        return EffectKind.DAMAGE
    if api in SHRINK_APIS:
        return EffectKind.SHRINK
    return None


@dataclass(frozen=True)
class _Amount:
    kind: EffectKind
    amount: int
    stated: Stated
    symmetric: bool


def _amount(kind: EffectKind, found: dict[str, str], x: int | None) -> _Amount | None:
    """The amount one damage or shrink segment states, or None if it states none."""
    if kind is EffectKind.DAMAGE:
        raw = found.get("NumDmg")
        if raw == "X" and x is not None:
            return _Amount(kind, x, Stated.X, False)
        amount = _literal(raw)
        return None if amount is None else _Amount(kind, amount, Stated.TEXT, False)
    raw, power = found.get("NumDef"), found.get("NumAtt")
    if raw == "-X" and x is not None:
        return _Amount(kind, x, Stated.X, power == "-X")
    shrink = _literal(raw)
    if shrink is None or shrink >= 0:
        return None
    return _Amount(kind, -shrink, Stated.TEXT, power is not None and _literal(power) == shrink)


def _the_one_amount(line, record) -> _Amount | None:
    """The chain's single positive damage or shrink amount, or None.

    None as well where any segment carries a :data:`BLOCKED_PARAMS` parameter,
    or where the chain holds more than one damage or shrink segment.
    """
    found_amounts = []
    for segment in segments(line.script_text):
        found = params(segment)
        if _blocked(found):
            return None
        kind = _kind_of(api_of(found))
        if kind is not None:
            found_amounts.append(_amount(kind, found, record.state.refs.x))
    if len(found_amounts) != 1 or found_amounts[0] is None:
        return None
    return found_amounts[0] if found_amounts[0].amount > 0 else None


class _Hit(Enum):
    """What one affected subject says about a candidate record."""

    MISSED = auto()  # not touched by the amount; neither copied nor refused
    HIT = auto()
    REFUSED = auto()  # something besides the amount changed what the rule decides


def _player_hit(fields: dict, found: _Amount) -> _Hit:
    if fields.get("poison_delta"):
        return _Hit.REFUSED
    life = fields.get("life_delta")
    if life is None or life == 0:
        return _Hit.MISSED
    if found.kind is not EffectKind.DAMAGE or life != -found.amount:
        return _Hit.REFUSED
    return _Hit.HIT


def _creature_hit(entity, fields: dict, found: _Amount, keywords_of, attached) -> _Hit:
    if entity is None or entity.zone != "battlefield":
        return _Hit.MISSED
    rules = _RULES[found.kind]
    took = fields.get(rules.field)
    if not took:
        return _Hit.MISSED
    if took != found.amount * rules.sign:
        return _Hit.REFUSED
    if "creature" not in entity.types or entity.pt is None:
        return _Hit.REFUSED
    if set(fields) - rules.hit_fields:
        return _Hit.REFUSED
    if ("indestructible" in keywords_of(entity) or entity.id in attached
            or any(name.lower() == "shield" for name in entity.counters)):
        return _Hit.REFUSED
    zone = fields.get("zone_outcome", _STAYED)
    if zone not in (_STAYED, _DIED) or (zone == _DIED) != lethal(found.kind, found.amount, entity):
        return _Hit.REFUSED
    return _Hit.HIT


def qualify(record, line, targets: dict[str, EntityTargets], keywords_of) -> Qualified | None:
    """``record`` as copy material, or None. ``keywords_of(entity)`` names its keywords."""
    from effects.domain.records import Moment, RecordKind

    if record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.RESOLUTION:
        return None
    if line is None or not line.script_text:
        return None
    found = _the_one_amount(line, record)
    if found is None:
        return None
    hits = _subjects_hit(record.state, targets, found, keywords_of)
    if hits is None or not hits[0]:
        return None
    hit, hit_players = hits
    return Qualified(kind=found.kind, amount=found.amount, stated=found.stated,
                     symmetric=found.symmetric, hit=tuple(hit), players=tuple(hit_players))


def _subjects_hit(state, targets: dict[str, EntityTargets], found: _Amount, keywords_of
                  ) -> tuple[list[str], list[str]] | None:
    """``(creatures, players)`` the amount hit, or None where the board refuses a copy."""
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
        if subject in players:
            verdict, into = _player_hit(entry.fields, found), hit_players
        else:
            verdict = _creature_hit(
                entities.get(subject), entry.fields, found, keywords_of, attached,
            )
            into = hit
        if verdict is _Hit.REFUSED:
            return None
        if verdict is _Hit.HIT:
            into.append(subject)
    return hit, hit_players


def edit_text(text: str, qualified: Qualified, amount: int) -> str | None:
    """The acting script with its literal amount set to ``amount``; None where it has none."""
    return _RULES[qualified.kind].edit(text, qualified.amount, amount, qualified.symmetric)


def edit_prose(prose: str, spans: tuple, qualified: Qualified, amount: int
               ) -> tuple[str, tuple] | None:
    """The converted prose with its amount set to ``amount``, and its role spans
    moved with any edit that changed its length; None where the prose does not
    state the amount, which leaves a copy with nothing it can edit.
    """
    edits: list[tuple[int, int, str]] = []
    for pattern, replacement in _RULES[qualified.kind].prose(
        qualified.amount, amount, qualified.symmetric,
    ):
        edits += [(m.start(), m.end(), replacement) for m in re.finditer(pattern, prose)]
    if not edits:
        return None
    edits.sort()
    parts, cursor = [], 0
    for start, end, replacement in edits:
        parts += [prose[cursor:start], replacement]
        cursor = end
    edited = "".join(parts) + prose[cursor:]

    def moved(position: int, *, opening: bool) -> int:
        return position + sum(
            len(replacement) - (end - start) for start, end, replacement in edits
            if (end <= position if opening else start < position)
        )

    return edited, tuple(
        replace(span, start=moved(span.start, opening=True),
                end=moved(span.end, opening=False))
        for span in spans
    )


def _relabel(entry: EntityTargets, qualified: Qualified, amount: int, dies: bool
             ) -> EntityTargets:
    fields = dict(entry.fields)
    rules = _RULES[qualified.kind]
    fields[rules.field] = float(amount * rules.sign)
    if qualified.symmetric:
        fields["power_delta"] = float(-amount)
    if dies:
        fields["zone_outcome"] = _DIED
    else:
        fields.pop("zone_outcome", None)
    return replace(entry, affected=True, fields=fields)


def counterfactual(record, qualified: Qualified, targets: dict[str, EntityTargets],
                   line, text: str | None, rng: random.Random, copy: int,
                   ) -> Counterfactual:
    """One copy of ``record``: a new amount, a new toughness on one hit creature,
    and every creature and player the amount hit relabelled by the rule.

    ``line`` is the acting sidecar line and ``text`` what the run encodes it
    from: its script, or on the prose surface its converted prose. A literal
    amount is edited in the script and in that text alike, which then needs a
    key of its own; a paid X is edited in the references; and where the script
    or the encoded text names neither, the amount stays as it was and only the
    toughness moves.
    """
    low, high = DRAW_RANGE
    drawn = rng.randint(low, high)
    chosen = rng.choice(qualified.hit)
    toughness = rng.randint(low, high)
    amount, key, edited, script, spans = qualified.amount, None, None, None, ()
    refs = record.state.refs
    if qualified.stated is Stated.X:
        amount, refs = drawn, replace(refs, x=drawn)
    elif text:
        script = edit_text(line.script_text, qualified, drawn)
        if text == line.script_text:
            edited = script
        else:
            found = edit_prose(text, tuple(getattr(line, "role_spans", ())),
                               qualified, drawn)
            edited, spans = found if found is not None else (None, ())
        if script is not None and edited is not None:
            amount = drawn
            original = record.ability[0]
            # A tree of its own, so the edited text never shares a key with
            # the real line's.
            key = replace(original, script_file=(
                f"{SourceTree.COUNTERFACTUAL}/{amount}/{original.script_file}"))
        else:
            edited, script, spans = None, None, ()
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
        relabelled[subject] = replace(targets[subject], fields=fields)
    return Counterfactual(record=copied, key=key, text=edited, targets=relabelled,
                          script=script, role_spans=spans)
