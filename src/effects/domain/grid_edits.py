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
  lethality copy edits it (:func:`~effects.domain.lethality_counterfactuals.amount_edit`),
  so a record qualifies only where the edit takes on both surfaces' texts.
- ``combat/<variant>/<blocker|attacker>`` and ``combat-several/…``: a
  one-on-one block (or one pair of a combat of several blocks), the striker's
  power by the struck creature's toughness, under a variant granting one
  creature a damage-step keyword (:data:`VARIANTS`), reading the struck
  creature's death. Every cell is a board a record could show: a snapshot
  of its damage step, phase and substep both, and for a double striker's
  second hit a blocker the first hit left alive (:class:`Variant`).
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
from effects.domain.effect_head_layout import ZONE_OUTCOMES, HeadField, ZoneOutcome
from effects.domain.effect_targets import derive_targets
from effects.domain.lethality_counterfactuals import (
    AmountText,
    BoardReading,
    amount_edit,
    qualify,
    stated_effect,
)
from effects.domain.records import CombatPayload, PlayabilityBlockersPayload, RecordKind
from effects.domain.rule_detectors.sba_board import game_ends
from effects.domain.rule_outcomes import (
    COLOUR_WORDS,
    BlockKeywords,
    Combatant,
    ProtectionQuality,
    QualityKind,
    damage_lethal,
    has_changeling,
    may_block,
    step_deaths,
)
from effects.domain.state_snapshot import CombatSubstep

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

#: Keywords that change who dies in a damage step; a qualifying pair carries none,
#: nor a protection of any quality (:func:`changes_the_step`).
COMBAT_KEYWORDS = frozenset({
    "first_strike", "double_strike", "deathtouch", "indestructible", "wither", "infect",
    "trample", "lifelink", "banding", "shadow",
})
#: What a combat grid reads: combats with one blocked attacker, or with several.
BLOCKS: tuple[str, ...] = ("one", "several")
#: The frozen combat grids' name prefix for each.
GRID_PREFIX: dict[str, str] = {"one": "combat/", "several": "combat-several/"}


def grid_of(name: str) -> str:
    """The grid a frozen grid named ``name`` is a variant of: ``combat`` for
    every ``combat/…`` and ``combat-several/…`` variant, else the name
    itself (``damage``, ``evasion``)."""
    if name.startswith(tuple(GRID_PREFIX.values())):
        return "combat"
    return name


def changes_the_step(keywords) -> bool:
    """Whether ``keywords`` hold one that changes who dies in a damage step:
    one of :data:`COMBAT_KEYWORDS`, or a protection, spelled by its quality
    (``protection_from_red``), which prevents the damage it names (702.16e)."""
    return bool(keywords & COMBAT_KEYWORDS) or any(
        keyword.startswith("protection") for keyword in keywords)


@dataclass(frozen=True, slots=True)
class Variant:
    """One combat grid variant: the creature granted a damage-step keyword
    (``attacker``, ``blocker``, or None for no grant), the keyword, the damage
    step its cells are snapshots of (``CombatSubstep``'s spelling), and
    whether the blocker carries the attacker's first-strike damage.

    A record covers one damage step, so a first or double striker is read in
    the first-strike step, where it deals its damage and the creature it
    fights deals none, and a double striker's second hit in the regular step,
    on a blocker already marked with the first.
    """

    holder: str | None
    keyword: str | None
    step: str
    carried: bool = False

    def base(self, record, attacker: str, blocker: str) -> BoardEdit:
        """The record's pair cleared of damage, in this variant's step, the
        keyword granted to its holder."""
        board = BoardEdit(record).clear_damage(attacker, blocker).set_step(self.step)
        if self.keyword is None:
            return board
        holder = attacker if self.holder == "attacker" else blocker
        return board.grant_keyword(holder, self.keyword)

    def survives_first_hit(self, attacker_power: int, blocker_toughness: int) -> bool:
        """Whether the blocker is still on the battlefield to strike in the
        step read: a carried first hit at least its toughness destroyed it in
        the first-strike step (704.5g, lethal damage), a board no record shows."""
        return not self.carried or attacker_power < blocker_toughness

    def held_cells(self, which: str) -> tuple[int, ...]:
        """The cells a grid of this variant read at ``which`` holds, as
        indices into its 8 x 8 square (power row by toughness column): every
        cell but, read at the blocker, those whose first hit is lethal, the
        cell's power being the attacker's and its toughness the blocker's."""
        square = [(power, toughness) for power in VALUES for toughness in VALUES]
        return tuple(index for index, (power, toughness) in enumerate(square)
                     if which != "blocker" or self.survives_first_hit(power, toughness))

    def shows(self, record, pair: tuple[str, str], which: str) -> bool:
        """Whether a grid of this variant read at ``which`` can edit the
        record's ``(attacker, blocker)``: read at the attacker, every cell
        keeps the record's own attacker power and blocker toughness, so the
        record qualifies only where the first hit leaves the blocker alive."""
        if which == "blocker":
            return True
        attacker, blocker = pair
        return self.survives_first_hit(record.state.entity(attacker).pt.total[0],
                                       record.state.entity(blocker).pt.total[1])


#: The combat grid's variants by name.
VARIANTS: dict[str, Variant] = {
    "none": Variant(None, None, "regular"),
    "attacker-deathtouch": Variant("attacker", "deathtouch", "regular"),
    "blocker-indestructible": Variant("blocker", "indestructible", "regular"),
    "attacker-first-strike": Variant("attacker", "first_strike", "first_strike"),
    "blocker-first-strike": Variant("blocker", "first_strike", "first_strike"),
    "attacker-double-strike": Variant("attacker", "double_strike", "first_strike"),
    "attacker-double-strike-second-hit": Variant(
        "attacker", "double_strike", "regular", carried=True),
}


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
    first in id order. Either way the record is a live game's combat that
    does not end the game (:func:`_real_combat`), it covers the regular
    damage step, and every pair qualifies as :func:`_clean_pair` says.
    """
    if not _regular_step_combat(record):
        return None
    pairs = _lone_blocks(record.payload, blocks)
    if pairs is None:
        return None
    targets = derive_targets(record)
    if not all(_clean_pair(record, pair, keywords_of, targets) for pair in pairs):
        return None
    return pairs[0]


def _regular_step_combat(record) -> bool:
    """A real combat (:func:`_real_combat`) in the regular damage step, no
    damage-step probe."""
    if not _real_combat(record):
        return False
    if record.state.global_.combat_substep != CombatSubstep.REGULAR:
        return False
    payload = record.payload
    return isinstance(payload, CombatPayload) and not getattr(payload, "probed_keyword", None)


def _lone_blocks(payload: CombatPayload, blocks: str) -> list[tuple[str, str]] | None:
    """Every ``(attacker, blocker)`` of the combat, in attacker id order, when
    the combat is the kind ``blocks`` names and each blocked attacker has one
    blocker; else None."""
    blocked = sorted((attacker, tuple(blockers))
                     for attacker, blockers in payload.blocks.items() if blockers)
    if not blocked or (len(blocked) == 1) != (blocks == "one"):
        return None
    if any(len(blockers) != 1 for _, blockers in blocked):
        return None
    return [(attacker, blockers[0]) for attacker, blockers in blocked]


def _real_combat(record) -> bool:
    """A live game's combat whose changes do not end the game
    (``sba_board.game_ends``, the test ``sba_zone_move`` reads): Forge's
    state-based check returns at the game's end before it destroys any
    creature, so the outcome such a combat shows is not the rule's
    (decision D13)."""
    if record.kind is not RecordKind.COMBAT or record.fork:
        return False
    return not (record.interventional or game_ends(record))


def _clean_pair(record, pair: tuple[str, str], keywords_of: KeywordsOf, targets) -> bool:
    """Whether one blocked ``(attacker, blocker)`` is a block a grid can edit:
    two creatures on the battlefield, no damage-step keyword on either, the
    blocker blocking this attacker alone, no damage but this step's own, and
    the real outcome the rule's."""
    attacker_id, blocker_id = pair
    attacker = record.state.entity(attacker_id)
    blocker = record.state.entity(blocker_id)
    if not all(_plain_creature(entity, keywords_of) for entity in (attacker, blocker)):
        return False
    return (_blocks_alone(blocker, attacker_id) and _both_strike(attacker, blocker)
            and _marked_this_step(attacker, blocker)
            and _outcome_is_the_rules(record, pair, targets))


def _plain_creature(entity, keywords_of: KeywordsOf) -> bool:
    """A creature on the battlefield with a power and toughness, carrying no
    keyword that changes who dies in a damage step (:func:`changes_the_step`)."""
    if entity is None or entity.pt is None:
        return False
    if not (entity.on_battlefield and entity.is_creature):
        return False
    return not changes_the_step(keywords_of(entity))


def _blocks_alone(blocker, attacker_id: str) -> bool:
    """The blocker blocks this attacker and no other."""
    return blocker.combat is not None and tuple(blocker.combat.blocking) == (attacker_id,)


def _both_strike(attacker, blocker) -> bool:
    """The attacker deals damage, and the blocker's power is not negative."""
    return attacker.power > 0 and blocker.power >= 0


def _marked_this_step(attacker, blocker) -> bool:
    """Marked damage is either none or this step's own, which the snapshot
    taken at the first damage event already shows; damage from earlier in the
    turn would make the cleared board in the grid a different board."""
    return attacker.damage in (0, blocker.power) and blocker.damage in (0, attacker.power)


def _outcome_is_the_rules(record, pair: tuple[str, str], targets) -> bool:
    """The real outcome obeys the rule on the board before the step, so
    prevention, regeneration or a replacement the board does not show cannot
    be what the grid reads."""
    died = ZONE_OUTCOMES.index(ZoneOutcome.TO_GRAVEYARD)
    attacker_id, blocker_id = pair
    before = BoardEdit(record).clear_damage(attacker_id, blocker_id).record
    expected = step_deaths(side_of(before, attacker_id)[0], side_of(before, blocker_id)[0])
    return all(_shows_the_death(targets.get(entity_id), should_die, died)
               for entity_id, should_die in zip(pair, expected))


def _shows_the_death(entry, should_die: bool, died: int) -> bool:
    """Whether a creature's targets show the death the rule gives it (or its
    staying): a creature moved where no zone outcome names shows neither."""
    if entry is not None and HeadField.ZONE_OUTCOME in entry.unknown:
        return False
    observed = entry is not None and entry.fields.get(HeadField.ZONE_OUTCOME) == died
    return observed == should_die


def grid_records(record, pair: tuple[str, str], which: str, variant: str):
    """The 8 x 8 edited records of one combat grid over the record's
    ``(attacker, blocker)``, row by row (the striker's power, then the struck
    creature's toughness), and the rule's answer for each; the grid holds the
    cells :meth:`Variant.held_cells` names."""
    chosen = VARIANTS[variant]
    attacker, blocker = pair
    step = chosen.step
    base = chosen.base(record, attacker, blocker)
    striker, struck = (attacker, blocker) if which == "blocker" else (blocker, attacker)
    rows, rules = [], []
    for power in VALUES:
        row, rule = [], []
        for toughness in VALUES:
            cell = base.set_power(striker, power).set_toughness(struck, toughness)
            if chosen.carried:
                cell = cell.set_marked_damage(blocker, side_of(cell.record, attacker)[0][0])
            edited = cell.record
            row.append(edited)
            a, a_kw = side_of(edited, attacker)
            b, b_kw = side_of(edited, blocker)
            attacker_dies, blocker_dies = step_deaths(a, b, BlockKeywords(a_kw, b_kw), step)
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


def _spare_colour(attacker, blocker) -> str | None:
    """The first colour neither creature has, or None when the pair holds all five."""
    taken = {*attacker.colors, *blocker.colors}
    return next((c for c in COLOUR_WORDS if c not in taken), None)


def _spare_creature_type(attacker, blocker) -> str | None:
    """The first of :data:`OTHER_CREATURE_TYPES` neither creature has, or None."""
    taken = {t.lower() for t in (*attacker.subtypes, *blocker.subtypes)}
    return next((t for t in OTHER_CREATURE_TYPES if t not in taken), None)


def protection_quality(row: str, attacker, blocker) -> ProtectionQuality:
    """The quality a protection row grants, resolved against the pair: the
    blocker's first colour or creature type for a ``-blocker`` row, a colour
    or creature type neither creature has for an ``-other`` row (the pair is
    an :func:`evasion_pair` only where both exist)."""
    if row in _FIXED_QUALITIES:
        return _FIXED_QUALITIES[row]
    if row == "protection-blocker":
        first = next(c for c in blocker.colors if c in COLOUR_WORDS)
        return ProtectionQuality(QualityKind.COLOUR, COLOUR_WORDS[first])
    if row == "protection-other":
        return ProtectionQuality(QualityKind.COLOUR, COLOUR_WORDS[_spare_colour(attacker, blocker)])
    return _creature_type_quality(row, attacker, blocker)


def _creature_type_quality(row: str, attacker, blocker) -> ProtectionQuality:
    """A ``protection-type-…`` row's creature type: the blocker's first, or
    for a blocker with none and for the ``-other`` row a spare type neither
    creature has (:func:`_spare_creature_type`); any other row is refused."""
    other = _spare_creature_type(attacker, blocker)
    if row == "protection-type-blocker" and blocker.subtypes:
        return ProtectionQuality(QualityKind.CREATURE_TYPE, blocker.subtypes[0].lower())
    if row in ("protection-type-blocker", "protection-type-other"):
        return ProtectionQuality(QualityKind.CREATURE_TYPE, other)
    raise KeyError(row)


#: Blocker columns granting the blocker the keyword they name.
_KEYWORD_TRAITS: tuple[str, ...] = ("flying", "reach", "horsemanship", "shadow")


def _artifact_trait(board: BoardEdit, attacker, blocker_id: str) -> BoardEdit:
    return board.add_type(blocker_id, "artifact")


def _black_trait(board: BoardEdit, attacker, blocker_id: str) -> BoardEdit:
    return board.set_colours(blocker_id, ("B",))


def _shared_colour_trait(board: BoardEdit, attacker, blocker_id: str) -> BoardEdit:
    """The blocker made the attacker's first colour other than black."""
    shared = next(c for c in attacker.colors if c in COLOUR_WORDS and c != "B")
    return board.set_colours(blocker_id, (shared,))


#: The other blocker columns' edits; a column in neither changes nothing.
_TRAIT_EDITS: dict[str, Callable[[BoardEdit, object, str], BoardEdit]] = {
    "artifact": _artifact_trait,
    "black": _black_trait,
    "shares-colour": _shared_colour_trait,
}


def _with_trait(board: BoardEdit, attacker, blocker_id: str, column: str | None) -> BoardEdit:
    """The board with the blocker given a blocker column's trait."""
    if column in _KEYWORD_TRAITS:
        return board.grant_keyword(blocker_id, column)
    edit = _TRAIT_EDITS.get(column)
    return board if edit is None else edit(board, attacker, blocker_id)


def _granted_protections(row: str | None, attacker, blocker) -> tuple[ProtectionQuality, ...]:
    """The protection an attacker row grants, resolved against the pair; none
    for a keyword row or no row."""
    if row not in PROTECTION_ROWS:
        return ()
    return (protection_quality(row, attacker, blocker),)


def _with_row(board: BoardEdit, attacker_id: str, row: str | None,
              granted: tuple[ProtectionQuality, ...]) -> BoardEdit:
    """The board with the attacker given an attacker row: its protection, or
    its keyword."""
    if granted:
        return board.grant_protection(attacker_id, granted[0])
    if row is None:
        return board
    return board.grant_keyword(attacker_id, row)


def evasion_cell(record, pair: tuple[str, str], row: str | None, column: str | None):
    """One evasion grid cell's record over the record's ``(attacker,
    blocker)``, and the rule's answer for it."""
    attacker_id, blocker_id = pair
    attacker = record.state.entity(attacker_id)
    board = _with_trait(BoardEdit(record), attacker, blocker_id, column)
    blocker = board.record.state.entity(blocker_id)
    granted = _granted_protections(row, attacker, blocker)
    record = _with_row(board, attacker_id, row, granted).record
    a = record.state.entity(attacker_id)
    b = record.state.entity(blocker_id)
    rule = may_block(Combatant.of(a), Combatant.of(b), attacker_protections=granted)
    return record, rule


def evasion_pair(record, keywords_of: KeywordsOf) -> tuple[str, str] | None:
    """``(attacker, blocker)`` for the first qualifying legal blocker of a
    real block-legality record: a legal coloured non-artifact creature, not
    black, no changeling (it has every creature type, 702.73a) and sharing no
    colour with an attacker that has a colour other than black, both on the
    battlefield (not phased out, not a last-known copy), neither carrying an
    evasion keyword or protection, and the pair
    leaving a colour and a spare creature type neither has, which the
    ``-other`` protection rows grant (an Esper attacker against a red-green
    blocker leaves no colour)."""
    attacker = _evasion_attacker(record, keywords_of)
    if attacker is None:
        return None
    blockers = (record.state.entity(blocker_id) for blocker_id in record.payload.legal_blockers)
    blocker = next((b for b in blockers if _evasion_blocker(b, attacker, keywords_of)), None)
    return None if blocker is None else (attacker.id, blocker.id)


def _evasion_attacker(record, keywords_of: KeywordsOf):
    """The anchor attacker of a real block-legality record, where it can head
    an evasion pair (:func:`_plain_attacker`); else None."""
    payload = record.payload
    if not isinstance(payload, PlayabilityBlockersPayload) or record.fork:
        return None
    if getattr(record, "what_if", False):
        return None
    attacker = record.state.entity(payload.anchor_attacker)
    return attacker if _plain_attacker(attacker, keywords_of) else None


def _plain_attacker(attacker, keywords_of: KeywordsOf) -> bool:
    """A creature on the battlefield, with no evasion keyword or protection,
    and a colour other than black."""
    if attacker is None or "creature" not in attacker.types:
        return False
    if not attacker.on_battlefield or _evades(attacker, keywords_of):
        return False
    return any(c != "B" for c in attacker.colors if c in COLOUR_WORDS)


def _evades(attacker, keywords_of: KeywordsOf) -> bool:
    """The attacker carries an evasion keyword or a protection."""
    if keywords_of(attacker) & EVASION_KEYWORDS:
        return True
    return any(k.startswith("protection") for k in keywords_of(attacker))


def _evasion_blocker(blocker, attacker, keywords_of: KeywordsOf) -> bool:
    """A legal blocker that pairs with the attacker (:func:`evasion_pair`)."""
    if not _non_artifact_creature(blocker):
        return False
    if has_changeling(blocker):
        # Every creature type (702.73a): protection from any creature type
        # covers it (702.16f), which no row's label reads.
        return False
    if not _colours_apart(blocker, attacker):
        return False
    if keywords_of(blocker) & EVASION_KEYWORDS:
        return False
    return _leaves_spares(attacker, blocker)


def _non_artifact_creature(blocker) -> bool:
    """A creature on the battlefield that is no artifact."""
    if blocker is None or not blocker.on_battlefield:
        return False
    return "creature" in blocker.types and "artifact" not in blocker.types


def _colours_apart(blocker, attacker) -> bool:
    """The blocker is coloured, not black, and shares no colour with the attacker."""
    colours = [c for c in blocker.colors if c in COLOUR_WORDS]
    if not colours or "B" in colours:
        return False
    return not set(colours) & set(attacker.colors)


def _leaves_spares(attacker, blocker) -> bool:
    """The pair leaves a colour and a spare creature type neither creature has."""
    return (_spare_colour(attacker, blocker) is not None
            and _spare_creature_type(attacker, blocker) is not None)


def label(name: str | None) -> str:
    """An evasion row or column as a cell name spells it."""
    return "none" if name is None else name


# ── the damage grid ─────────────────────────────────────────────────────


def damage_target(lines, keywords_of: KeywordsOf):
    """``choose(record) -> (record, creature) | None`` for single-target
    damage resolutions with a literal amount (the lethality copies'
    qualification) that both surfaces' texts state (:func:`_edits_on_both_surfaces`);
    ``lines.line_or_none(key)`` is the acting line, ``lines.prose_for(key)``
    its prose."""
    def choose(record):
        if not record.ability:
            return None
        key = record.ability[0]
        line = lines.line_or_none(key)
        if line is None:
            return None
        qualified = qualify(record, line, derive_targets(record), BoardReading(keywords_of, lines))
        if not _one_text_damage_hit(qualified) or not _edits_on_both_surfaces(
                record, qualified.effect, line, lines.prose_for(key)):
            return None
        return record, qualified.hit[0]

    return choose


def _one_text_damage_hit(qualified) -> bool:
    """A lethality copy's qualification of damage whose amount the script
    states, hitting one creature."""
    return (qualified is not None and qualified.effect.kind == "damage"
            and qualified.effect.stated == "text" and len(qualified.hit) == 1)


def _edits_on_both_surfaces(record, effect, line, prose: str | None) -> bool:
    """Whether the amount edit takes on the text each surface encodes the
    line from (``ability_encoder.encoding_text``): the script surface reads its
    script, the prose surface its prose, and a line with no prose its script
    reading on both. A prose that does not state the amount (its cost sentence
    alone) would leave every cell reading the real text against outcomes that
    vary by amount."""
    texts = (line.script_text, prose) if prose else (line.script_text,)
    return all(amount_edit(record, AmountText(effect, line, text), effect.amount).key is not None
               for text in texts)


def damage_cells(record, entity: str, texts: AbilityTexts) -> list[CellRead]:
    """The damage grid's 64 cells of one record, amount by amount and within
    an amount toughness by toughness: the target's toughness set (and its
    marked damage cleared), the acting text's amount edited under a key of its
    own, read at the target for its death. Where the encoded text does not
    state the amount, every amount reads the real text; :func:`damage_target`
    qualifies no such record."""
    key = record.ability[0]
    line = texts.line_of(key)
    effect = None if line is None else stated_effect(line, record)
    edited = [BoardEdit(record).set_toughness(entity, toughness).record
              for toughness in VALUES]
    cells: list[CellRead] = []
    for amount in VALUES:
        edit = None if effect is None else amount_edit(
            record, AmountText(effect, line, texts.text_of(key)), amount)
        cells += _amount_cells(edit, key, edited, entity)
    return cells


def _amount_cells(edit, key, boards: list, entity: str) -> list[CellRead]:
    """One amount's cells, toughness by toughness: each board reading the
    amount's edited text under its own key, or the real text where the edit
    did not take."""
    if edit is None or edit.key is None:
        return [CellRead(board, entity, "dies", ()) for board in boards]
    overrides = (TextOverride(edit.key, edit.text, key, script=edit.script,
                              role_spans=edit.role_spans),)
    return [CellRead(replace(board, ability=(edit.key, *board.ability[1:])), entity, "dies",
                     overrides) for board in boards]


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

    @property
    def name(self) -> str:
        return f"{GRID_PREFIX[self.blocks.blocks]}{self.variant}/{self.which}"

    @property
    def cells(self) -> tuple[str, ...]:
        """The ``power x toughness`` cells the variant holds (:meth:`Variant.held_cells`)."""
        square = _square_cells()
        return tuple(square[index] for index in VARIANTS[self.variant].held_cells(self.which))

    def qualify(self, record) -> GridQualification | None:
        pair = self.blocks(record)
        variant = VARIANTS[self.variant]
        if pair is None or not variant.shows(record, pair, self.which):
            return None
        _, rule = grid_records(record, pair, self.which, self.variant)
        flat = rule.ravel()
        return GridQualification(pair, tuple(
            float(flat[index]) for index in variant.held_cells(self.which)))


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
        outcomes = tuple(
            float(evasion_cell(record, pair, row, column)[1])
            for row in ATTACKER_KEYWORDS for column in BLOCKER_TRAITS)
        return GridQualification(pair, outcomes)


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
        if record.fork or record.interventional:
            return None
        found = damage_target(self.lines, self.keywords_of)(record)
        if found is None:
            return None
        _, entity_id = found
        # Every cell sets the toughness and clears the marked damage
        # (:func:`damage_cells`), so the rule reads the cleared board.
        outcomes = tuple(float(damage_lethal(toughness, 0, amount))
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
            return [CellRead(evasion_cell(record, (attacker, blocker), row, column)[0],
                             blocker, "may_block")
                    for row in ATTACKER_KEYWORDS for column in BLOCKER_TRAITS]
        return _combat_cells(grid, record, (attacker, blocker))


def _combat_cells(grid: str, record, pair: tuple[str, str]) -> list[CellRead]:
    """A combat grid's held cells of one record, read at the struck creature."""
    prefix, variant, which = grid.split("/")
    if f"{prefix}/" not in GRID_PREFIX.values():
        raise KeyError(f"no grid {grid!r}")
    rows, _ = grid_records(record, pair, which, variant)
    attacker, blocker = pair
    read = blocker if which == "blocker" else attacker
    square = [cell for row in rows for cell in row]
    return [CellRead(square[index], read, "dies")
            for index in VARIANTS[variant].held_cells(which)]


__all__ = [
    "ATTACKER_KEYWORDS", "BLOCKER_TRAITS", "BLOCKS", "COMBAT_KEYWORDS", "EVASION_KEYWORDS",
    "GRID_PREFIX", "OTHER_CREATURE_TYPES", "PROTECTION_ROWS", "VALUES", "VARIANTS",
    "AbilityTexts", "CellRead", "CombatGrid", "DamageGrid", "EvasionGrid", "GridEdits",
    "GridQualification", "TextOverride", "Variant", "changes_the_step", "damage_cells",
    "damage_target", "evasion_cell", "evasion_pair", "grid_block", "grid_of", "grid_records",
    "grid_sources", "label", "protection_quality", "side_of",
]
