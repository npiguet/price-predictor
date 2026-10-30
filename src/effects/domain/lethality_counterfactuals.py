"""Counterfactual copies of damage and -X/-X resolutions, labelled by the rules (FR-063e).

A copy changes one hit creature's toughness and the amount, and is labelled
by rule 704.5f-g rather than by Forge's AI. Only records whose real outcome
already obeys the rule are copied, and only where nothing else changes what
the rule decides:

- the acting chain holds exactly one ``DealDamage``/``DamageAll`` segment with
  a fixed or paid amount, or one ``Pump``/``PumpAll`` lowering toughness by
  one, and no segment remembers, conditions on or replaces anything
  (:data:`BLOCKED_PARAMS`), since those can turn on the death itself;
- every creature on the battlefield the amount hit changes in nothing but the
  amount and its zone, and every player it hit loses exactly the amount;
- the source carries no deathtouch, wither, infect or lifelink, and no hit
  creature carries indestructible, a shield counter or an attachment.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum, StrEnum, auto

from effects.domain.damage_step_keywords import KeywordsOf
from effects.domain.effect_head_layout import ZONE_OUTCOMES, HeadField, ZoneOutcome
from effects.domain.effect_targets import EntityTargets
from effects.domain.provenance import ProvenanceKey, RoleSpan, SidecarLine, SourceTree
from effects.domain.records import EffectRecord, Moment, RecordKind
from effects.domain.script_text import api_of, params, segments
from effects.domain.state_snapshot import EntityState, StateSnapshot, normalize_counter_type

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

#: The counter a creature spends instead of being dealt damage or destroyed,
#: so a hit on one carrying it obeys no lethality rule.
SHIELD_COUNTER = "SHIELD"

_INT = re.compile(r"^-?\d+$")
_DIED = ZONE_OUTCOMES.index(ZoneOutcome.DIED)
_STAYED = ZONE_OUTCOMES.index(ZoneOutcome.STAYED)


class EffectKind(StrEnum):
    """What the one amount does to a creature it hits."""

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


def _params_damage(text: str, n: int, amount: int, symmetric: bool) -> str | None:
    edited, count = re.subn(rf"(NumDmg\$\s*){n}\b", rf"\g<1>{amount}", text)
    return edited if count else None


def _params_shrink(text: str, n: int, amount: int, symmetric: bool) -> str | None:
    edited, count = re.subn(rf"(NumDef\$\s*)-{n}\b", rf"\g<1>-{amount}", text)
    if not count:
        return None
    if symmetric:
        edited = re.sub(rf"(NumAtt\$\s*)-{n}\b", rf"\g<1>-{amount}", edited)
    return edited


@dataclass(frozen=True)
class _KindRules:
    """Everything that differs between damage and a shrink, in one place."""

    #: The target field a hit creature records the amount in, and the sign it
    #: carries there: damage taken is positive, a toughness change negative.
    field: HeadField
    sign: int
    #: What a hit creature's targets may hold: anything else means another
    #: part of the chain touched it.
    hit_fields: frozenset[HeadField]
    #: ``(toughness, marked damage, amount) -> dies`` under rule 704.5f-g.
    lethal: Callable[[int, int, int], bool]
    #: ``(script, old amount, new amount, symmetric) -> the script with the
    #: parameters stating the amount edited``, or None where none states it.
    edit_params: Callable[[str, int, int, bool], str | None]
    #: ``(old amount, new amount, symmetric) -> (pattern, replacement)`` pairs
    #: that edit the amount where prose states it: the converted prose, and
    #: the script's own description.
    prose: Callable[[int, int, bool], list[tuple[str, str]]]


_RULES: dict[EffectKind, _KindRules] = {
    EffectKind.DAMAGE: _KindRules(
        field=HeadField.DAMAGE_TAKEN, sign=1,
        hit_fields=frozenset({HeadField.DAMAGE_TAKEN, HeadField.ZONE_OUTCOME}),
        lethal=_damage_lethal, edit_params=_params_damage, prose=_prose_damage,
    ),
    EffectKind.SHRINK: _KindRules(
        field=HeadField.TOUGHNESS_DELTA, sign=-1,
        hit_fields=frozenset({
            HeadField.TOUGHNESS_DELTA, HeadField.POWER_DELTA, HeadField.PT_DURATION,
            HeadField.ZONE_OUTCOME,
        }),
        lethal=_shrink_lethal, edit_params=_params_shrink, prose=_prose_shrink,
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

    record: EffectRecord
    #: The copy's acting key and its edited encoding text, which the batcher's
    #: memo carries for this step; None where the text is the real line's.
    key: ProvenanceKey | None
    text: str | None
    targets: dict[str, EntityTargets]
    #: The edited script, which the value head's targets read whatever the
    #: surface encodes, and the prose's role spans moved with its edit.
    script: str | None = None
    role_spans: tuple[RoleSpan, ...] = ()


def lethal(kind: EffectKind, amount: int, entity: EntityState) -> bool:
    """Whether ``amount`` kills ``entity`` under rule 704.5f-g."""
    return _RULES[kind].lethal(entity.toughness, entity.damage, amount)


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


def _the_one_amount(line: SidecarLine, record: EffectRecord) -> _Amount | None:
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
    if fields.get(HeadField.POISON_DELTA):
        return _Hit.REFUSED
    life = fields.get(HeadField.LIFE_DELTA)
    if life is None or life == 0:
        return _Hit.MISSED
    if found.kind is not EffectKind.DAMAGE or life != -found.amount:
        return _Hit.REFUSED
    return _Hit.HIT


def _creature_hit(entity: EntityState | None, fields: dict, found: _Amount,
                  keywords_of: KeywordsOf, attached: set[str]) -> _Hit:
    if entity is None or not entity.on_battlefield:
        return _Hit.MISSED
    rules = _RULES[found.kind]
    took = fields.get(rules.field)
    if not took:
        return _Hit.MISSED
    if took != found.amount * rules.sign:
        return _Hit.REFUSED
    if not entity.is_creature or entity.pt is None:
        return _Hit.REFUSED
    if set(fields) - rules.hit_fields:
        return _Hit.REFUSED
    if ("indestructible" in keywords_of(entity) or entity.id in attached
            or any(normalize_counter_type(name) == SHIELD_COUNTER for name in entity.counters)):
        return _Hit.REFUSED
    zone = fields.get(HeadField.ZONE_OUTCOME, _STAYED)
    if zone not in (_STAYED, _DIED) or (zone == _DIED) != lethal(found.kind, found.amount, entity):
        return _Hit.REFUSED
    return _Hit.HIT


def qualify(record: EffectRecord, line: SidecarLine | None, targets: dict[str, EntityTargets],
            keywords_of: KeywordsOf) -> Qualified | None:
    """``record`` as copy material, or None. ``keywords_of(entity)`` names its keywords."""
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


def _subjects_hit(state: StateSnapshot, targets: dict[str, EntityTargets], found: _Amount,
                  keywords_of: KeywordsOf) -> tuple[list[str], list[str]] | None:
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
    rules = _RULES[qualified.kind]
    edited = rules.edit_params(text, qualified.amount, amount, qualified.symmetric)
    if edited is None:
        return None
    for pattern, replacement in rules.prose(qualified.amount, amount, qualified.symmetric):
        edited = re.sub(pattern, replacement, edited)
    return edited


def edit_prose(prose: str, spans: tuple[RoleSpan, ...], qualified: Qualified, amount: int
               ) -> tuple[str, tuple[RoleSpan, ...]] | None:
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
        fields[HeadField.POWER_DELTA] = float(-amount)
    if dies:
        fields[HeadField.ZONE_OUTCOME] = _DIED
    else:
        fields.pop(HeadField.ZONE_OUTCOME, None)
    return replace(entry, affected=True, fields=fields)


@dataclass(frozen=True)
class _TextEdit:
    """The amount a copy's labels use, and the text it is written in, if edited.

    ``key`` is None where the copy reads the real line: the amount was paid as
    X, or the text names it nowhere a copy can edit.
    """

    amount: int
    key: ProvenanceKey | None = None
    text: str | None = None
    script: str | None = None
    role_spans: tuple[RoleSpan, ...] = ()


def _edited_text(record: EffectRecord, qualified: Qualified, line: SidecarLine,
                 text: str | None, drawn: int) -> _TextEdit:
    """A literal amount set to ``drawn`` in the script and the encoded text,
    both or neither, so the text never disagrees with the labels."""
    unedited = _TextEdit(qualified.amount)
    if not text:
        return unedited
    script = edit_text(line.script_text, qualified, drawn)
    if script is None:
        return unedited
    if text == line.script_text:
        edited, spans = script, ()
    else:
        found = edit_prose(text, line.role_spans, qualified, drawn)
        if found is None:
            return unedited
        edited, spans = found
    original = record.ability[0]
    # A tree of its own, so the edited text never shares a key with the real line's.
    key = replace(original, script_file=(
        f"{SourceTree.COUNTERFACTUAL}/{drawn}/{original.script_file}"))
    return _TextEdit(drawn, key, edited, script, spans)


def _with_toughness(entities: tuple[EntityState, ...], chosen: str, toughness: int
                    ) -> tuple[EntityState, ...]:
    """``entities`` with ``chosen`` at total toughness ``toughness`` and no marked damage."""
    return tuple(entity.with_total_toughness(toughness) if entity.id == chosen else entity
                 for entity in entities)


def _relabelled_targets(targets: dict[str, EntityTargets], qualified: Qualified, amount: int,
                        entities: tuple[EntityState, ...]) -> dict[str, EntityTargets]:
    """``targets`` with every creature and player the amount hit labelled by the rule."""
    by_id = {entity.id: entity for entity in entities}
    relabelled = dict(targets)
    for subject in qualified.hit:
        relabelled[subject] = _relabel(
            targets[subject], qualified, amount,
            lethal(qualified.kind, amount, by_id[subject]))
    for subject in qualified.players:
        fields = {**targets[subject].fields, HeadField.LIFE_DELTA: float(-amount)}
        relabelled[subject] = replace(targets[subject], fields=fields)
    return relabelled


def counterfactual(record: EffectRecord, qualified: Qualified,
                   targets: dict[str, EntityTargets], line: SidecarLine, text: str | None,
                   rng: random.Random, copy: int) -> Counterfactual:
    """One copy of ``record``: a new amount, a new toughness on one hit creature,
    and every creature and player the amount hit relabelled by the rule.

    ``text`` is what the run encodes ``line`` from. A literal amount is edited
    in the text under a key of its own, a paid X in the references; where the
    text names neither, only the toughness moves.
    """
    low, high = DRAW_RANGE
    drawn = rng.randint(low, high)
    chosen = rng.choice(qualified.hit)
    toughness = rng.randint(low, high)
    refs = record.state.refs
    if qualified.stated is Stated.X:
        edit, refs = _TextEdit(drawn), replace(refs, x=drawn)
    else:
        edit = _edited_text(record, qualified, line, text, drawn)
    entities = _with_toughness(record.state.entities, chosen, toughness)
    copied = replace(
        record,
        record_id=f"{record.record_id}~cf{copy}",
        ability=(edit.key,) if edit.key is not None else record.ability,
        state=replace(record.state, entities=entities, refs=refs),
    )
    return Counterfactual(
        record=copied, key=edit.key, text=edit.text,
        targets=_relabelled_targets(targets, qualified, edit.amount, entities),
        script=edit.script, role_spans=edit.role_spans,
    )
