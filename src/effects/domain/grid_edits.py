"""The damage, combat and evasion edit grids: which records each grid edits,
the edits of every cell, and the rule's answer in each (FR-068, FR-074).

In the package because a training run scores the grids every epoch (the
control score, ``application/control_score.py``) and the package cannot import
``scripts/``; the knowledge probes' grid scripts import their edits from here,
so a probing run and the control score edit the same records the same way.
Torch-free.

Three roles over one set of edits:

- the freeze's :class:`GridSource`\\ s (:func:`grid_sources`): a grid's name,
  its cells, and for each record whether the grid can edit it, which
  entities, and the rule's outcome in every cell
  (:class:`GridQualification`);
- the control score's grid editor (:class:`GridEdits`): one frozen record
  edited into each of its grid's cells, as a :class:`CellRead` naming the
  readout and the entity read;
- the scripts' grid tables, which call the same edit functions.

The grids:

- ``damage``: a single-target damage resolution with a literal amount, its
  amount by its target's toughness (1..8 each), reading the target's death.
  The amount is edited in the encoded text under a key of its own, as a
  lethality copy edits it (:func:`~effects.domain.lethality_counterfactuals.amount_edit`).
- ``combat/<variant>/<blocker|attacker>`` and ``combat-several/…``: a
  one-on-one block (or one pair of a combat of several blocks), the striker's
  power by the struck creature's toughness, under a variant granting one
  creature a damage-step keyword (:data:`VARIANTS`), reading the struck
  creature's death.
- ``evasion``: an attacker keyword (or protection quality) row by a blocker
  trait column, reading the blocker's ``may_block``.

Nothing here does work at import time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np

from effects.domain.board_edits import BoardEdit
from effects.domain.rule_outcomes import (
    COLOUR_WORDS,
    ProtectionQuality,
    QualityKind,
    may_block,
    step_deaths,
    strikes_in_step,
)

#: Damage and toughness of the damage grid, power and toughness of the combat grid.
VALUES: tuple[int, ...] = tuple(range(1, 9))

#: ``entity -> its keywords from both channels`` (``KeywordResolver.keywords_of``).
KeywordsOf = Callable[[object], frozenset[str]]


# ── the ports' values ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GridQualification:
    """A record a grid can edit: the entities the grid edits and reads, and
    the rule's outcome in each of the grid's cells for this record."""

    entities: tuple[str, ...]
    outcomes: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TextOverride:
    """An ability key that reads an edited text, as a lethality copy's does
    (``SurfaceBatcher.override_text``): ``key`` reads ``text`` with ``like``'s
    line, its script and role spans replaced where given."""

    key: object
    text: str
    like: object
    script: str | None = None
    role_spans: tuple | None = None


@dataclass(frozen=True, slots=True)
class CellRead:
    """One grid cell of one grid record: the edited record, the entity read
    and the readout (a ``probe_readouts.READOUTS`` name) read there, and the
    edited ability texts the record's keys read (a damage grid's amount), each
    under a key of its own so the cells of one record can share a chunk."""

    record: object
    entity: str | None
    readout: str
    texts: tuple[TextOverride, ...] = ()


class AbilityTexts(Protocol):
    """What the damage grid's text edit reads of the run's batcher: the text a
    key is encoded from on the run's surface, and its sidecar line."""

    def text_of(self, key) -> str | None: ...

    def line_of(self, key): ...


# ── the combat grid ─────────────────────────────────────────────────────

#: Keywords that change who dies in a damage step; a qualifying pair carries none.
COMBAT_KEYWORDS = frozenset({
    "first_strike", "double_strike", "deathtouch", "indestructible", "wither", "infect",
    "trample", "lifelink", "protection", "banding", "shadow",
})
#: What a combat grid reads: combats with one blocked attacker, or with several.
BLOCKS: tuple[str, ...] = ("one", "several")
#: The frozen combat grids' name prefix for each.
GRID_PREFIX: dict[str, str] = {"one": "combat/", "several": "combat-several/"}
#: ``variant -> (creature granted the keyword or None, keyword, damage step, blocker
#: carries the attacker's first-step damage)``. A record covers one damage step, so a
#: first or double striker is read in the first-strike step, where it deals its damage
#: and the creature it fights deals none, and a double striker's second hit in the
#: regular step, on a blocker already marked with the first.
VARIANTS: dict[str, tuple[str | None, str | None, str, bool]] = {
    "none": (None, None, "regular", False),
    "attacker-deathtouch": ("attacker", "deathtouch", "regular", False),
    "blocker-indestructible": ("blocker", "indestructible", "regular", False),
    "attacker-first-strike": ("attacker", "first_strike", "first_strike", False),
    "blocker-first-strike": ("blocker", "first_strike", "first_strike", False),
    "attacker-double-strike": ("attacker", "double_strike", "first_strike", False),
    "attacker-double-strike-second-hit": ("attacker", "double_strike", "regular", True),
}


def assign_step_damage(record, attacker: str, blocker: str):
    """``record`` with both combatants' damage assignment matching their edited board.

    A combat record's input carries each creature's assignment, the targets it assigns
    to and the damage it assigns, so an edit to a creature's power or to the step has to
    move its assignment with it, or the board and the assignment disagree. In a
    one-on-one block a creature that strikes in this step assigns all its power to the
    other; one that does not strike, or has no power, assigns nothing.
    """
    step = record.state.global_.combat_substep
    choices = {source: dict(targets)
               for source, targets in record.payload.assignment_choices.items()
               if source not in (attacker, blocker)}
    for source, target in ((attacker, blocker), (blocker, attacker)):
        (power, _, _), keywords = side_of(record, source)
        if power > 0 and strikes_in_step(keywords, step):
            choices[source] = {target: power}
    return replace(record, payload=replace(record.payload, assignment_choices=choices))


def side_of(record, entity_id: str):
    """``(power, toughness, marked damage)`` of one entity, and its granted keywords."""
    entity = record.state.entity(entity_id)
    power, toughness = entity.pt.total
    return (power, toughness, entity.damage), frozenset(entity.granted_temporary.keywords)


def grid_block(record, keywords_of: KeywordsOf, blocks: str) -> tuple[str, str] | None:
    """The pair a combat grid of ``blocks`` (:data:`BLOCKS`) edits, or None.

    ``one``: exactly one attacker is blocked, by exactly one blocker blocking
    nothing else. ``several``: two or more blocked attackers, each blocked by
    one creature blocking it alone; the pair is the one whose attacker comes
    first in id order. Either way the record covers the regular damage step,
    and every pair qualifies as :func:`_clean_pair` says.
    """
    from effects.domain.effect_targets import derive_targets
    from effects.domain.records import CombatPayload, RecordKind
    from effects.domain.state_snapshot import CombatSubstep

    if record.kind is not RecordKind.COMBAT or record.fork or record.interventional:
        return None
    if record.state.global_.combat_substep != CombatSubstep.REGULAR:
        return None
    payload = record.payload
    if not isinstance(payload, CombatPayload) or getattr(payload, "probed_keyword", None):
        return None
    blocked = sorted((attacker, tuple(blockers))
                     for attacker, blockers in payload.blocks.items() if blockers)
    if not blocked or (len(blocked) == 1) != (blocks == "one"):
        return None
    if any(len(blockers) != 1 for _, blockers in blocked):
        return None
    targets = derive_targets(record)
    if not all(_clean_pair(record, attacker, blockers[0], keywords_of, targets)
               for attacker, blockers in blocked):
        return None
    attacker_id, (blocker_id,) = blocked[0]
    return attacker_id, blocker_id


def _clean_pair(record, attacker_id: str, blocker_id: str, keywords_of: KeywordsOf,
                targets) -> bool:
    """Whether one blocked pair is a block a grid can edit: two creatures on
    the battlefield, no damage-step keyword on either, the blocker blocking
    this attacker alone, no damage but this step's own, and the real outcome
    the rule's."""
    from effects.domain.effect_head_layout import ZONE_OUTCOMES, HeadField, ZoneOutcome

    attacker = record.state.entity(attacker_id)
    blocker = record.state.entity(blocker_id)
    for entity in (attacker, blocker):
        if (entity is None or entity.pt is None or not entity.on_battlefield
                or not entity.is_creature):
            return False
        if keywords_of(entity) & COMBAT_KEYWORDS:
            return False
    if blocker.combat is None or tuple(blocker.combat.blocking) != (attacker_id,):
        return False
    if attacker.power <= 0 or blocker.power < 0:
        return False
    # Marked damage is either none or this step's own, which the snapshot taken at
    # the first damage event already shows; damage from earlier in the turn would
    # make the cleared board in the grid a different board.
    if attacker.damage not in (0, blocker.power):
        return False
    if blocker.damage not in (0, attacker.power):
        return False
    # The real outcome must obey the rule on the board before the step, so
    # prevention, regeneration or a replacement the board does not show cannot
    # be what the grid reads.
    died = ZONE_OUTCOMES.index(ZoneOutcome.DIED)
    before = BoardEdit(record).clear_damage(attacker_id, blocker_id).record
    expected = step_deaths(side_of(before, attacker_id)[0], side_of(before, blocker_id)[0])
    for entity_id, should_die in zip((attacker_id, blocker_id), expected):
        entry = targets.get(entity_id)
        observed = entry is not None and entry.fields.get(HeadField.ZONE_OUTCOME) == died
        if observed != should_die:
            return False
    return True


def grid_records(record, attacker: str, blocker: str, which: str, variant: str):
    """The 8 x 8 edited records of one combat grid, row by row (the striker's
    power, then the struck creature's toughness), and the rule's answer for each."""
    holder, keyword, step, carried = VARIANTS[variant]
    base = BoardEdit(record).clear_damage(attacker, blocker).set_step(step)
    if keyword is not None:
        base = base.grant_keyword(attacker if holder == "attacker" else blocker, keyword)
    striker, struck = (attacker, blocker) if which == "blocker" else (blocker, attacker)
    rows, rules = [], []
    for power in VALUES:
        row, rule = [], []
        for toughness in VALUES:
            cell = base.set_power(striker, power).set_toughness(struck, toughness)
            if carried:
                cell = cell.set_marked_damage(blocker, side_of(cell.record, attacker)[0][0])
            edited = assign_step_damage(cell.record, attacker, blocker)
            row.append(edited)
            a, a_kw = side_of(edited, attacker)
            b, b_kw = side_of(edited, blocker)
            attacker_dies, blocker_dies = step_deaths(a, b, a_kw, b_kw, step)
            rule.append(blocker_dies if which == "blocker" else attacker_dies)
        rows.append(row)
        rules.append(rule)
    return rows, np.array(rules, dtype=float)


# ── the evasion grid ────────────────────────────────────────────────────

#: Attacker rows that grant protection, each resolved to a quality against the pair
#: (:func:`protection_quality`): ``-blocker`` names the blocker, ``-other`` does not.
PROTECTION_ROWS: tuple[str, ...] = (
    "protection-blocker", "protection-other",
    "protection-type-blocker", "protection-type-other",
    "protection-creatures", "protection-artifacts",
    "protection-monocolored", "protection-multicolored",
    "protection-mana-value-3-or-less", "protection-everything",
)
#: Attacker rows: the keyword the attacker is granted; ``None`` grants nothing.
ATTACKER_KEYWORDS: tuple[str | None, ...] = (
    None, "flying", "horsemanship", "fear", "intimidate", "shadow", "unblockable",
    *PROTECTION_ROWS,
)
#: Creature types a ``protection-type-other`` row picks from, the first that
#: neither creature has: types no limited creature is likely to carry.
OTHER_CREATURE_TYPES: tuple[str, ...] = ("kraken", "leviathan", "octopus", "homarid")
#: The rows whose quality is fixed, whatever the pair.
_FIXED_QUALITIES: dict[str, ProtectionQuality] = {
    "protection-creatures": ProtectionQuality(QualityKind.CARD_TYPE, "creature"),
    "protection-artifacts": ProtectionQuality(QualityKind.CARD_TYPE, "artifact"),
    "protection-monocolored": ProtectionQuality(QualityKind.CONDITION, "monocolored"),
    "protection-multicolored": ProtectionQuality(QualityKind.CONDITION, "multicolored"),
    "protection-mana-value-3-or-less": ProtectionQuality(
        QualityKind.CONDITION, "mana value 3 or less"),
    "protection-everything": ProtectionQuality.everything(),
}
#: Blocker columns: the trait the blocker is given; ``None`` changes nothing.
BLOCKER_TRAITS: tuple[str | None, ...] = (
    None, "flying", "reach", "horsemanship", "shadow", "artifact", "black", "shares-colour",
)
#: Keywords and traits that decide block legality; a qualifying pair carries none.
EVASION_KEYWORDS = frozenset({
    "flying", "reach", "horsemanship", "fear", "intimidate", "shadow", "unblockable",
    "skulk", "menace", "landwalk", "defender",
})


def protection_quality(row: str, attacker, blocker) -> ProtectionQuality:
    """The quality a protection row grants, resolved against the pair: the
    blocker's first colour or creature type for a ``-blocker`` row, a colour
    or creature type neither creature has for an ``-other`` row."""
    if row in _FIXED_QUALITIES:
        return _FIXED_QUALITIES[row]
    if row == "protection-blocker":
        return ProtectionQuality(QualityKind.COLOUR, COLOUR_WORDS[blocker.colors[0]])
    if row == "protection-other":
        taken = {c for c in (*attacker.colors, *blocker.colors)}
        other = next(c for c in COLOUR_WORDS if c not in taken)
        return ProtectionQuality(QualityKind.COLOUR, COLOUR_WORDS[other])
    taken = {t.lower() for t in (*attacker.subtypes, *blocker.subtypes)}
    other = next(t for t in OTHER_CREATURE_TYPES if t not in taken)
    if row == "protection-type-blocker" and blocker.subtypes:
        return ProtectionQuality(QualityKind.CREATURE_TYPE, blocker.subtypes[0].lower())
    if row in ("protection-type-blocker", "protection-type-other"):
        return ProtectionQuality(QualityKind.CREATURE_TYPE, other)
    raise KeyError(row)


def evasion_cell(record, attacker_id: str, blocker_id: str, row: str | None,
                 column: str | None):
    """One evasion grid cell's record, and the rule's answer for it."""
    attacker = record.state.entity(attacker_id)
    board = BoardEdit(record)
    if column in ("flying", "reach", "horsemanship", "shadow"):
        board = board.grant_keyword(blocker_id, column)
    elif column == "artifact":
        board = board.add_type(blocker_id, "artifact")
    elif column == "black":
        board = board.set_colours(blocker_id, ("B",))
    elif column == "shares-colour":
        shared = next(c for c in attacker.colors if c != "B")
        board = board.set_colours(blocker_id, (shared,))
    blocker = board.record.state.entity(blocker_id)
    granted: tuple[ProtectionQuality, ...] = ()
    if row in PROTECTION_ROWS:
        quality = protection_quality(row, attacker, blocker)
        board = board.grant_protection(attacker_id, quality)
        granted = (quality,)
    elif row is not None:
        board = board.grant_keyword(attacker_id, row)
    record = board.record
    a = record.state.entity(attacker_id)
    b = record.state.entity(blocker_id)
    rule = may_block(a.granted_temporary.keywords, a.colors, b.granted_temporary.keywords,
                     b.colors, b.types, attacker_protections=granted,
                     blocker_subtypes=b.subtypes, blocker_mana_value=b.mana_value)
    return record, rule


def evasion_pair(record, keywords_of: KeywordsOf) -> tuple[str, str] | None:
    """``(attacker, blocker)`` for the first qualifying legal blocker of a
    real block-legality record: a legal coloured non-artifact creature, not
    black and sharing no colour with an attacker that has a colour other than
    black, neither carrying an evasion keyword or protection."""
    from effects.domain.records import PlayabilityBlockersPayload

    payload = record.payload
    if not isinstance(payload, PlayabilityBlockersPayload) or record.fork:
        return None
    if getattr(record, "what_if", False):
        return None
    attacker = record.state.entity(payload.anchor_attacker)
    if attacker is None or "creature" not in attacker.types:
        return None
    if keywords_of(attacker) & EVASION_KEYWORDS or any(
            k.startswith("protection") for k in keywords_of(attacker)):
        return None
    if not any(c != "B" for c in attacker.colors if c in COLOUR_WORDS):
        return None
    for blocker_id in payload.legal_blockers:
        blocker = record.state.entity(blocker_id)
        if blocker is None or "creature" not in blocker.types or "artifact" in blocker.types:
            continue
        colours = [c for c in blocker.colors if c in COLOUR_WORDS]
        if not colours or "B" in colours or set(colours) & set(attacker.colors):
            continue
        if keywords_of(blocker) & EVASION_KEYWORDS:
            continue
        return attacker.id, blocker.id
    return None


def label(name: str | None) -> str:
    """An evasion row or column as a cell name spells it."""
    return "none" if name is None else name


# ── the damage grid ─────────────────────────────────────────────────────


def damage_target(lines, keywords_of: KeywordsOf):
    """``choose(record) -> (record, creature) | None`` for single-target
    damage resolutions with a literal amount (the lethality copies'
    qualification); ``lines.line_or_none(key)`` is the acting line."""
    from effects.domain.effect_targets import derive_targets
    from effects.domain.lethality_counterfactuals import qualify

    def choose(record):
        if not record.ability:
            return None
        line = lines.line_or_none(record.ability[0])
        if line is None:
            return None
        qualified = qualify(record, line, derive_targets(record), keywords_of)
        if (qualified is None or qualified.effect.kind != "damage"
                or qualified.effect.stated != "text" or len(qualified.hit) != 1):
            return None
        return record, qualified.hit[0]

    return choose


def damage_cells(record, entity: str, texts: AbilityTexts) -> list[CellRead]:
    """The damage grid's 64 cells of one record, amount by amount and within
    an amount toughness by toughness: the target's toughness set (and its
    marked damage cleared), the acting text's amount edited under a key of its
    own, read at the target for its death. Where the encoded text does not
    state the amount (a prose surface whose line words it otherwise), every
    amount reads the real text, as the grid scripts always have."""
    from effects.domain.lethality_counterfactuals import amount_edit, stated_effect

    key = record.ability[0]
    line = texts.line_of(key)
    effect = None if line is None else stated_effect(line, record)
    edited = [BoardEdit(record).set_toughness(entity, toughness).record
              for toughness in VALUES]
    cells: list[CellRead] = []
    for amount in VALUES:
        edit = None if effect is None else amount_edit(
            record, effect, line, texts.text_of(key), amount)
        overrides: tuple[TextOverride, ...] = ()
        if edit is not None and edit.key is not None:
            overrides = (TextOverride(edit.key, edit.text, key, script=edit.script,
                                      role_spans=edit.role_spans),)
        for board in edited:
            cell = board if not overrides else replace(
                board, ability=(edit.key, *board.ability[1:]))
            cells.append(CellRead(cell, entity, "dies", overrides))
    return cells


# ── the freeze's grid sources ───────────────────────────────────────────


def _square_cells() -> tuple[str, ...]:
    return tuple(f"{a}x{b}" for a in VALUES for b in VALUES)


@dataclass
class _BlockMemo:
    """:func:`grid_block` of the last record asked, shared by every combat
    grid of one kind of block (``blocks``, :data:`BLOCKS`): the freeze asks
    all fourteen of one record in turn."""

    keywords_of: KeywordsOf
    blocks: str = "one"
    _last: tuple = (None, None)

    def __call__(self, record):
        if self._last[0] != record.record_id:
            self._last = (record.record_id, grid_block(record, self.keywords_of, self.blocks))
        return self._last[1]


@dataclass
class CombatGrid:
    """One combat grid: a variant, and whose death it reads (``blocker``: the
    attacker's power against the blocker's toughness; ``attacker``: the
    reverse), over one-on-one blocks or over combats of several
    (``blocks.blocks``). Entities ``(attacker, blocker)`` of the pair edited."""

    variant: str
    which: str
    blocks: _BlockMemo
    cells: tuple[str, ...] = field(default_factory=_square_cells)

    @property
    def name(self) -> str:
        return f"{GRID_PREFIX[self.blocks.blocks]}{self.variant}/{self.which}"

    def qualify(self, record) -> GridQualification | None:
        pair = self.blocks(record)
        if pair is None:
            return None
        attacker, blocker = pair
        _, rule = grid_records(record, attacker, blocker, self.which, self.variant)
        return GridQualification((attacker, blocker), tuple(float(v) for v in rule.ravel()))


@dataclass
class EvasionGrid:
    """The evasion grid: attacker keyword rows by blocker trait columns, read
    at the blocker. Entities ``(attacker, blocker)``."""

    keywords_of: KeywordsOf
    name: str = "evasion"
    cells: tuple[str, ...] = tuple(
        f"{label(row)}|{label(column)}"
        for row in ATTACKER_KEYWORDS for column in BLOCKER_TRAITS)

    def qualify(self, record) -> GridQualification | None:
        pair = evasion_pair(record, self.keywords_of)
        if pair is None:
            return None
        attacker, blocker = pair
        outcomes = tuple(
            float(evasion_cell(record, attacker, blocker, row, column)[1])
            for row in ATTACKER_KEYWORDS for column in BLOCKER_TRAITS)
        return GridQualification((attacker, blocker), outcomes)


@dataclass
class DamageGrid:
    """The damage grid: a single-target damage resolution with a literal
    amount, its amount by its target's toughness, read at the target.
    Entities ``(target,)``."""

    lines: object
    keywords_of: KeywordsOf
    name: str = "damage"
    cells: tuple[str, ...] = field(default_factory=_square_cells)

    def qualify(self, record) -> GridQualification | None:
        from effects.domain.rule_outcomes import damage_lethal

        if record.fork or record.interventional:
            return None
        found = damage_target(self.lines, self.keywords_of)(record)
        if found is None:
            return None
        _, entity_id = found
        damage = record.state.entity(entity_id).damage
        outcomes = tuple(float(damage_lethal(toughness, damage, amount))
                         for amount in VALUES for toughness in VALUES)
        return GridQualification((entity_id,), outcomes)


def grid_sources(lines, keywords_of: KeywordsOf) -> list:
    """Every grid the selection set freezes: damage, the combat grid's
    variants each read at either creature, over one-on-one blocks and over
    combats of several blocked attackers, and evasion. ``lines`` resolves the
    damage grid's acting lines (a ``SidecarCache``)."""
    grids: list = [DamageGrid(lines, keywords_of)]
    for kind in BLOCKS:
        blocks = _BlockMemo(keywords_of, kind)
        grids += [CombatGrid(variant, which, blocks)
                  for variant in VARIANTS for which in ("blocker", "attacker")]
    return [*grids, EvasionGrid(keywords_of)]


# ── the control score's grid editor ─────────────────────────────────────


@dataclass(frozen=True)
class GridEdits:
    """The control score's ``GridEditor``: one frozen record of a grid edited
    into each of the grid's cells, in the frozen cells' order (the order
    :func:`grid_sources` gives them). ``texts`` is the run's batcher, whose
    surface the damage grid's amount is edited on."""

    texts: AbilityTexts

    def cells(self, grid: str, record, entities: Sequence[str]) -> list[CellRead]:
        if grid == "damage":
            (entity,) = entities
            return damage_cells(record, entity, self.texts)
        attacker, blocker = entities
        if grid == "evasion":
            return [CellRead(evasion_cell(record, attacker, blocker, row, column)[0],
                             blocker, "may_block")
                    for row in ATTACKER_KEYWORDS for column in BLOCKER_TRAITS]
        prefix, variant, which = grid.split("/")
        if f"{prefix}/" not in GRID_PREFIX.values():
            raise KeyError(f"no grid {grid!r}")
        rows, _ = grid_records(record, attacker, blocker, which, variant)
        read = blocker if which == "blocker" else attacker
        return [CellRead(cell, read, "dies") for row in rows for cell in row]


__all__ = [
    "ATTACKER_KEYWORDS", "BLOCKER_TRAITS", "BLOCKS", "COMBAT_KEYWORDS", "EVASION_KEYWORDS",
    "GRID_PREFIX", "OTHER_CREATURE_TYPES", "PROTECTION_ROWS", "VALUES", "VARIANTS",
    "AbilityTexts", "CellRead", "CombatGrid", "DamageGrid", "EvasionGrid", "GridEdits",
    "GridQualification", "TextOverride", "assign_step_damage", "damage_cells",
    "damage_target", "evasion_cell", "evasion_pair", "grid_block", "grid_records",
    "grid_sources", "label", "protection_quality", "side_of",
]
