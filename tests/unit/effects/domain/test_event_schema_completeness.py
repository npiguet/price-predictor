"""The event vocabulary, and the connector's Java source that emits it (T025).

Reads the connector's source as text, so it needs no JVM and no ``../forge``
checkout.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from effects.domain.event_schema import (
    ATTRIBUTION_ROOT,
    ATTRIBUTION_UNRESOLVED,
    CAUSE_BEARING_TYPES,
    EVENT_PARAMS,
    Event,
    EventType,
    attribution_kind,
)


class TestVocabulary:
    def test_event_type_values_are_unique_and_snake_case(self):
        values = [t.value for t in EventType]
        assert len(values) == len(set(values))
        assert all(v == v.lower() and " " not in v for v in values)


class TestEventNormalization:
    def test_an_event_accepts_the_params_its_type_normalizes(self):
        event = Event(
            type=EventType.DAMAGE_DEALT,
            subjects=("E12",),
            params={"amount": 3, "combat": True},
        )
        assert event.params["amount"] == 3

    def test_an_event_rejects_a_param_the_schema_does_not_normalize(self):
        with pytest.raises(ValueError, match="does not normalize"):
            Event(type=EventType.DAMAGE_DEALT, params={"amonut": 3})

    def test_a_type_with_no_params_takes_none(self):
        assert Event(type=EventType.LIBRARY_SHUFFLED).params == {}
        with pytest.raises(ValueError, match="does not normalize"):
            Event(type=EventType.LIBRARY_SHUFFLED, params={"count": 1})

    def test_an_event_round_trips_through_its_dict_form(self):
        event = Event(
            type=EventType.ZONE_CHANGE,
            subjects=("E1", "E2"),
            params={"from_zone": "battlefield", "to_zone": "graveyard"},
            duration=None,
            attributed_to="0",
        )
        assert Event.from_dict(event.as_dict()) == event


class TestTheZoneChangeRow:
    """The one slot a `Moved` replacement can move and nothing else.

    A replacement that puts a card second from the top of its library instead
    of into the graveyard changes the destination and the position; without a
    slot for the position, the two halves of the rewrite serialize alike and
    the record is dropped as an identity — which is why `Moved` was 91 of 110
    dropped rewrites and had never once produced a written one.
    """

    def test_a_library_destination_can_say_where_in_the_library(self):
        event = Event(
            type=EventType.ZONE_CHANGE,
            subjects=("E1",),
            params={
                "from_zone": "battlefield", "to_zone": "library",
                "library_position": 1,
            },
        )
        assert event.params["library_position"] == 1

    def test_it_round_trips(self):
        event = Event(
            type=EventType.ZONE_CHANGE,
            params={"to_zone": "library", "library_position": 0},
        )
        assert Event.from_dict(event.as_dict()) == event

    def test_it_is_optional(self):
        """Absent wherever the destination is not a library."""
        assert Event(
            type=EventType.ZONE_CHANGE, params={"to_zone": "graveyard"},
        ).params == {"to_zone": "graveyard"}


class TestTheCauseBearingTypes:
    """The types whose own row asks for a cause, which is what gets measured.

    Every type may carry a ``cause`` — it is a provenance param — so a
    population rate over the whole vocabulary would read near zero on a
    perfectly healthy corpus and mean nothing. The set is derived from the
    table rather than restated, so widening a row's params widens the
    measurement with it.
    """

    def test_the_three_removal_shaped_types_are_in_it(self):
        for event_type in (
            EventType.ZONE_CHANGE, EventType.DESTROYED, EventType.SACRIFICED,
        ):
            assert event_type in CAUSE_BEARING_TYPES

    def test_a_type_that_names_its_source_differently_is_not_in_it(self):
        """``damage_dealt`` carries ``source``; two spellings of one fact
        would leave a reader to guess which one a collector filled."""
        assert EventType.DAMAGE_DEALT not in CAUSE_BEARING_TYPES

    def test_any_type_may_still_carry_a_cause(self):
        """Being outside the set bars nothing — it only says the schema does
        not ask."""
        assert Event(
            type=EventType.CARD_DRAWN, params={"count": 1, "cause": "E3"},
        ).params["cause"] == "E3"


class TestAttributionIsTriState:
    """The three states an ``attributed_to`` may report, and the fourth that
    is not one of them.

    One spelling covered "the root line acted" and "the pointer did not land"
    for a whole collection run, so 96.8% of resolution events said nothing and
    a dead attribution channel looked exactly like a working one on a corpus of
    simple spells. The sentinels separate them; ``None`` stays what a writer
    that predates them left behind, and means *unknown* rather than *root*.
    """

    def test_a_chain_index_reads_as_a_sub_ability(self):
        assert attribution_kind("0") == "sub_ability"
        assert attribution_kind("2") == "sub_ability"

    def test_the_two_sentinels_read_as_themselves(self):
        assert attribution_kind(ATTRIBUTION_ROOT) == "root"
        assert attribution_kind(ATTRIBUTION_UNRESOLVED) == "unresolved"

    def test_null_is_unknown_and_not_the_root(self):
        """The whole point of adding the sentinels, asserted rather than assumed."""
        assert attribution_kind(None) == "absent"

    def test_the_four_states_are_distinct(self):
        seen = {
            attribution_kind(value)
            for value in ("3", ATTRIBUTION_ROOT, ATTRIBUTION_UNRESOLVED, None)
        }
        assert len(seen) == 4

    def test_a_sentinel_is_not_mistakable_for_a_chain_index(self):
        assert all(
            not value.isdigit() for value in (ATTRIBUTION_ROOT, ATTRIBUTION_UNRESOLVED)
        )

    def test_a_sentinel_survives_the_wire_form_unchanged(self):
        """Widening a value set, not redefining the field: it stays a string."""
        event = Event(
            type=EventType.DAMAGE_DEALT,
            subjects=("E1",),
            attributed_to=ATTRIBUTION_UNRESOLVED,
        )
        assert Event.from_dict(event.as_dict()).attributed_to == (
            ATTRIBUTION_UNRESOLVED
        )


#: The connector's effects package, read as text. A type the writer never names
#: cannot appear in a corpus however many games are played, and that is
#: invisible from the corpus itself: "rare" and "unreachable" look identical.
_CONNECTOR_EFFECTS = (
    Path(__file__).resolve().parents[4]
    / "forge-connector" / "src" / "main" / "java" / "com" / "pricepredictor"
    / "connector" / "effects"
)

#: Types with no reference anywhere in the connector's Java source. Distinct from
#: three other shapes a declared type can be missing in, none of which belong in
#: this dict because a reference genuinely exists for each:
#:
#: - ``damage_prevented`` and ``spell_copied``, referenced in dead code that never
#:   runs (``PatchedCollectors.java``, wired, not yet fired).
#: - ``spell_countered`` and ``token_created``, referenced by a live reactive
#:   trigger's own mode name (``"Countered"``, ``"TokenCreated"``/
#:   ``"TokenCreatedOnce"`` -- all real ``TriggerType`` members) -- a card that
#:   *reacts* to a counter or a token appearing, never the resolving ability's own
#:   announcement.
#: - ``dice_rolled`` and ``player_won``, referenced by a live *replacement*
#:   mode name instead -- ``"RollDice"``/``"GameWin"`` are real ``ReplacementType``
#:   members (not ``TriggerType``, despite sitting in the same mode table), reached
#:   from ``rewriteRecord`` rather than ``triggerRecord``. Firing needs a
#:   replacement effect that intercepts the roll or the win in progress, not a
#:   reactive trigger and not the rolling/winning ability's own resolution.
#:
#: All six are wired somewhere, just not for an ability's own resolution to
#: announce what it did -- this scan structurally cannot see that gap, distinguish
#: its two sub-shapes, or distinguish either from a type that fires normally.
#: the event-type check (``corpus_validation.event_checks.EventTypes``), run against a real
#: collection, is the only thing that can. This set shrinks as each remaining type
#: is moved from unreferenced to emitted; a type silently added or removed here
#: fails the test, which is the point.
#:
#: ``regenerated`` and ``library_shuffled`` were originally filed as two more
#: instances of the "referenced but unfired" shape above, and that turned out to
#: be wrong for one of them on closer reading: ``"Regenerated"`` in the mode table
#: matches no real ``TriggerType`` or ``ReplacementType`` member at all -- it is a
#: dead entry, the same defect shape as ``damage_prevented``/``spell_copied``, not
#: a live reference serving a different purpose. ``"Shuffled"`` (for
#: ``library_shuffled``) *is* a real trigger mode, so that half of the original
#: claim held. Both types are now wired directly off the bus instead
#: (``GameEventCardRegenerated``, ``GameEventShuffle`` -- Task 4's pattern,
#: already used for ``energy_change``/``radiation_change``/``speed_changed``/
#: ``day_night_changed``), which needed no schema change: neither type has an
#: ``EVENT_PARAMS`` row, so a bus subscription satisfies the schema on its own.
#: ``EventType.REGENERATED`` also has a second, non-emitting reference this scan
#: finds -- ``EventBuffer.IDEMPOTENT_EVENTS`` -- which was inert while the
#: type had no emitter and is load-bearing now: kept deliberately, not a side
#: effect nobody chose, but not because a second regeneration is impossible --
#: it is not. See the docstring above ``IDEMPOTENT_EVENTS`` itself for why it
#: stays anyway, and the open question a corpus run (not static reasoning) can
#: now settle.
#:
#: Populated again as of the Task 10 fix round, and that is not a regression.
#: The guard this dict feeds used to compare against ``set(EVENT_PARAMS)`` --
#: 57 of the vocabulary's 78 declared types, every type with its own per-type
#: params row -- so the 21 types that carry no params beyond their subjects were
#: invisible to it: referenced or not, checked or not, they could never appear
#: in ``unreachable`` because they were never in the set being subtracted from.
#: Widening the guard to ``set(EventType)`` is what surfaced 15 types; they
#: were unreachable and unchecked for the entire life of this plan, not broken by
#: widening it. Going from ``frozenset()`` to a populated mapping is the set
#: finally being honest about a gap that predates this plan and was invisible the
#: whole time it ran.
#:
#: Task 12 wired all eight highest-reach of those 15 -- ``delayed_trigger_created``,
#: ``replacement_applied``, ``monarch_changed``, ``ring_tempts``,
#: ``removed_from_combat``, ``initiative_taken``, ``text_change``, ``turn_ended``.
#: Six are fully live. Two, ``turn_ended`` and ``text_change``, went back into this
#: dict in Task 12's fix round 1 under a reason this dict had never needed before:
#: **the emitter is wired and correct, and is referenced right here in the
#: connector source -- unlike every other entry below -- but no AI-only corpus can
#: ever reach it**, because the AI refuses to play any of the cards that would
#: fire it. ``AI_CAPABILITY_GAP`` (below) names exactly these two. Conflating this
#: with "no emitter exists" -- this dict's original and only meaning -- would have
#: reproduced the exact ``token_created``/``spell_countered`` shape this whole
#: effort exists to end: a corpus gap with the code that explains it removed from
#: the ledger, at the moment the code was verified correct rather than at the
#: moment it was written. Together all eight account for ~84% of the 15's
#: combined card count. ``combat_ended`` is wired too, through an
#: ``ApiEvents.RULES`` entry for ``EndCombatPhase`` -- leaving the 8 below: 6
#: genuinely unreferenced, 2 wired but AI-unreachable today, live the moment the
#: AI gains these APIs or a human-play corpus is collected.
#:
#: Values are the reason: for the 6 unreferenced entries, the producing Forge
#: effect API(s), that no emission path exists (``ApiEvents.RULES`` in the
#: connector, or an explicit ``EffectRecordOutcomes.note`` call), and how many
#: cards in the sealed/draft pool script that API, read directly from
#: ``forge-gui/res/cardsfolder`` rather than estimated. For the 2
#: ``AI_CAPABILITY_GAP`` entries, the reason instead names the emission path (which
#: exists) and the AI class whose refusal keeps every card that carries it off the
#: stack, so the claim is checkable rather than an assertion.
KNOWN_UNEMITTED: dict[str, str] = {
    "ability_activated":
        "ActivateAbility: no ApiEvents.RULES entry, no outcome-note call; "
        "2 cards script it",
    "game_drawn":
        "GameDrawn: no ApiEvents.RULES entry, no outcome-note call; "
        "2 cards script it",
    "game_restarted":
        "RestartGame: no ApiEvents.RULES entry, no outcome-note call; "
        "1 card scripts it",
    "player_removed":
        "RemoveFromMatch: no ApiEvents.RULES entry, no outcome-note call; "
        "2 cards script it",
    "trigger_fired":
        "No Forge effect API maps to it at all -- no ApiEvents.RULES entry "
        "names it. Its only plausible source is the trigger-fire hook itself, "
        "which already reports through the trigger record kind's own `fired` "
        "flag rather than this type",
    "turn_order_reversed":
        "ReverseTurnOrder: no ApiEvents.RULES entry, no outcome-note call; "
        "3 cards script it",
    "turn_ended":
        "EndTurn: EffectRecordOutcomes.note is wired (EndTurnEffect) and correct "
        "-- but forge-ai's EndTurnAi.canPlay and .chkDrawback both return "
        "CantPlayAi unconditionally, and for the three scripts that bury EndTurn "
        "in a SubAbility$ chain (days_undoing, ultima, hurkyls_final_meditation), "
        "SpellAbilityAi.chkDrawbackWithSubs propagates that refusal to the whole "
        "enclosing spell -- so none of the 9 cards is ever played by an AI-only "
        "corpus. Live the moment the AI gains EndTurn or a human-play corpus is "
        "collected",
    "text_change":
        "ChangeText/ExchangeTextBox: an ApiEvents.RULES entry (ChangeText) and an "
        "EffectRecordOutcomes.note call (ExchangeTextBox) are both wired and "
        "correct -- but neither API has a SpellApiToAi entry, so both fall back "
        "to CannotPlayAi, whose canPlay and chkDrawback also refuse "
        "unconditionally -- so none of the 12 + 2 = 14 cards is ever played by an "
        "AI-only corpus. Live the moment the AI gains these APIs or a human-play "
        "corpus is collected",
}

#: The subset of ``KNOWN_UNEMITTED`` that means something different from the rest
#: of it: **wired, correct, and referenced right here in the connector source**,
#: not absent from it. ``test_no_declared_type_is_unreachable_by_surprise`` has to
#: carve these two out of the textual scan's comparison for exactly that reason --
#: a live reference is what the scan looks for, so ``turn_ended``/``text_change``
#: read as "emitted" to it and would otherwise make the guard's two sets disagree
#: forever. ``test_the_ai_capability_gap_types_stay_wired`` is the guard in the
#: other direction: if either of these two ever stops being referenced, the
#: wiring described in its ``KNOWN_UNEMITTED`` reason has been silently removed,
#: which is a real regression this set exists to catch.
AI_CAPABILITY_GAP: frozenset[str] = frozenset({"turn_ended", "text_change"})


def _emitted_event_types() -> set[str]:
    """Event types whose constants are referenced in the connector's Java source.

    This is a textual static scan—it finds ``EventType.CONSTANT_NAME`` in
    code without distinguishing whether that code is on a live path or dead code,
    and without distinguishing an emitter that builds the type from one that only
    *translates a trigger or replacement mode name* into it for a different
    record kind. A type may be referenced but never fired from an ability's own
    resolution, in three different ways that all look identical to this scan:

    - **Dead reference.** ``damage_prevented`` and ``spell_copied`` were
      referenced in ``PatchedCollectors.java`` code that never runs, across a
      10.1M-record corpus, without firing once. ``"Regenerated"`` in the
      mode table (for ``regenerated``) was the same shape — it matched no real
      Forge mode at all — until a Task 10 fix round wired ``regenerated``
      directly off the bus instead, the same way it wired ``library_shuffled``
      (whose own table entry, unlike ``"Regenerated"``, was a real trigger mode
      that simply served a different purpose — see below).
    - **Reactive trigger, not the ability's own outcome.** ``spell_countered``
      and ``token_created`` are referenced by a live trigger mode name
      (``"Countered"``, ``"TokenCreated"``/``"TokenCreatedOnce"``) that fires
      for a card *reacting* to a counter or a token appearing, never for the
      resolving ability's own announcement.
    - **Replacement mode, not a trigger at all.** ``dice_rolled`` and
      ``player_won`` are referenced by a live *replacement* mode name
      (``"RollDice"``, ``"GameWin"`` — real ``ReplacementType`` members, reached
      from ``rewriteRecord`` rather than ``triggerRecord``), which fires only
      when a replacement effect intercepts the roll or the win in progress.

    Conversely, a type unreferenced in source is truly unreachable. This test
    guards against the under-count direction only—the corpus-empirical gap is
    deeper and cannot be caught by static analysis: some declared types may be
    referenced but unfired, in any of the three shapes above. The unreferenced
    types are recorded in ``KNOWN_UNEMITTED`` and shrink as each is wired and
    emitted; the referenced-but-unfired gap cannot be recorded here at all — a
    reference is a reference regardless of what it is for — which is why
    the event-type check (``corpus_validation.event_checks.EventTypes``), run against a real
    collection, exists as this scan's permanent counterpart rather than a
    one-off check.
    """
    header = (_CONNECTOR_EFFECTS / "EventType.java").read_text(encoding="utf-8")
    constants = dict(re.findall(
        r'^\s*([A-Z_]+)\("([a-z_]+)"\)[,;]', header, re.MULTILINE,
    ))
    emitted: set[str] = set()
    for path in _CONNECTOR_EFFECTS.glob("*.java"):
        if path.name == "EventType.java":
            continue
        text = path.read_text(encoding="utf-8")
        for name, value in constants.items():
            if re.search(rf"EventType\.{name}\b", text):
                emitted.add(value)
    return emitted


class TestEveryDeclaredTypeIsReachable:
    """The vocabulary and the writer are one contract — the *whole* vocabulary.

    A declared type can be unreachable in the corpus, and a mapping from an
    effect API to a type nothing emits would still read as a mapping. The gap has
    two parts, and this class only fully covers one of them as of the Task 10 fix
    round that widened it:

    1. **Never checked.** Types not referenced anywhere in the connector source.
       Recorded in ``KNOWN_UNEMITTED``, each with the producing API and a card
       count read from the pool, and shrinks as each is wired and starts emitting.
       Until this fix round, this class compared against ``set(EVENT_PARAMS)`` —
       57 of the vocabulary's 78 declared types, every type with its own per-type
       params row. The 21 types that carry no params beyond their subjects
       (``delayed_trigger_created``, ``monarch_changed``, and 19 more) were
       invisible to it *by construction*, whichever way the comparison came out:
       "the guard scans for references, not fires" was the founding limitation
       this whole plan named, and a guard that never even looks at a fifth of the
       vocabulary is the same failure mode one level up, inside the guard itself.
       It compares against ``set(EventType)`` now — the full declared vocabulary,
       not a subset of it.
    2. **Referenced but unfired.** Types the guard finds a reference to and
       therefore cannot flag, no matter how wide its domain, because the
       reference is real — just not an outcome emitter for the ability's own
       resolution. A corpus analysis of 10.1M records found two:
       ``damage_prevented`` and ``spell_copied``, both referenced in dead code.
       A Task 10 fix-round measurement against a fresh 54,016-record run found
       four more hiding behind the widening in (1), in two further sub-shapes:
       ``spell_countered``/``token_created`` are referenced by a live
       *reactive-trigger* mode name (a card reacting to the outcome, not the
       ability announcing it), and ``dice_rolled``/``player_won`` by a live
       *replacement* mode name (a replacement effect intercepting the roll or
       the win, reached from ``rewriteRecord`` rather than ``triggerRecord`` —
       not a trigger at all, despite sitting in the same mode table). A fifth
       and sixth, ``regenerated`` and ``library_shuffled``, were provisionally
       filed here too and turned out to need the fix-round's first category
       instead for one of them (``regenerated``'s table entry is dead, not
       live-but-different-purpose) — both are wired directly off the bus now,
       the same pattern as ``energy_change``/``radiation_change``/
       ``speed_changed``/``day_night_changed``, and no longer belong in this
       list at all. The other four -- ``spell_countered``, ``token_created``
       and ``dice_rolled`` via ``EffectRecordOutcomes.note`` from the
       resolving effect itself, ``player_won`` off the bus the same way
       ``regenerated``/``library_shuffled`` are -- were wired by a Task 11 fix
       round; the taxonomy above is unchanged and still true of the mode-table
       references that never fired, it just no longer describes the whole
       story for any of the four. This class, and the guard it backs, could
       not have caught any of it either way: no static scan, however wide, can
       see whether a reference is on a path that actually runs. Only a corpus
       measurement can, which is what makes the event-type check in
       ``corpus_validation/event_checks.py`` this test's permanent counterpart rather than a
       one-off check that stops mattering once the guard is fixed.

    A Task 12 fix round found a **third** category, distinct from both parts
    above and easy to collapse into either: ``turn_ended`` and ``text_change``
    are referenced by a real emitter, for the ability's own resolution, that
    is wired correctly and would fire given the right game state -- but no
    AI-only corpus can ever reach that state, because the AI refuses every
    card that carries these APIs before it resolves one (``EndTurnAi``;
    ``CannotPlayAi`` via the missing ``SpellApiToAi`` entries for
    ``ChangeText``/``ExchangeTextBox``). Unlike part 2's four, the reference
    *is* the real emitter -- nothing here is serving a different purpose --
    so this is not "referenced but unfired" in part 2's sense either. Named
    ``AI_CAPABILITY_GAP`` and kept inside ``KNOWN_UNEMITTED`` (with its own
    carve-out in the guard test below) rather than treated as fully live,
    because a declared type an AI-only corpus can structurally never produce
    needs to say so, the same way an unreferenced one does — leaving it out
    would have reported it covered when no collection at any length could
    ever cover it.
    """

    def test_no_declared_type_is_unreachable_by_surprise(self):
        """``AI_CAPABILITY_GAP`` entries are carved out of both sides on purpose:
        they ARE referenced (the textual scan finds them, correctly), so they
        would never appear in ``unreachable`` no matter how long they stayed in
        ``KNOWN_UNEMITTED`` -- comparing the raw sets would report them "newly
        wired" forever and mask a real drift hiding behind them. Subtracting the
        same known set from both sides keeps this test measuring exactly what it
        always measured (the never-checked types), and
        ``test_the_ai_capability_gap_types_stay_wired`` below covers the two this
        one now deliberately ignores.
        """
        unreachable = set(EventType) - _emitted_event_types()
        never_checked = set(KNOWN_UNEMITTED) - AI_CAPABILITY_GAP
        assert unreachable == never_checked, (
            "declared-but-unemitted types changed; wired: "
            f"{sorted(never_checked - unreachable)}, newly unreachable: "
            f"{sorted(unreachable - never_checked)}"
        )

    def test_the_ai_capability_gap_types_stay_wired(self):
        """The other half of the carve-out above: ``turn_ended``/``text_change``
        must actually be referenced, or their ``KNOWN_UNEMITTED`` reason -- "the
        emitter is wired and correct" -- would be describing code that no longer
        exists, with nothing left to say so. This is the regression
        ``test_no_declared_type_is_unreachable_by_surprise`` was widened to no
        longer be able to see for these two.
        """
        assert AI_CAPABILITY_GAP <= _emitted_event_types(), (
            "an AI_CAPABILITY_GAP type is no longer referenced in the connector "
            "source -- its wiring may have been silently removed"
        )

    def test_the_known_list_names_only_declared_types(self):
        """Against the full vocabulary, not just the params-bearing subset.

        ``EVENT_PARAMS`` would reject this list outright now: most of
        ``KNOWN_UNEMITTED`` is params-less types by construction (that is the
        gap the widened guard exists to see), so checking against
        ``set(EVENT_PARAMS)`` here would fail on every entry rather than catch a
        real mistake.
        """
        assert set(KNOWN_UNEMITTED) <= set(EventType)

    def test_the_ai_capability_gap_is_part_of_known_unemitted(self):
        assert AI_CAPABILITY_GAP <= set(KNOWN_UNEMITTED)


class TestSupersededTypes:
    """A type the record carries under another name is not an event.

    ``cost_adjusted`` is the playability decision's ``cost_after_adjustment``,
    ``damage_assignment_ordered`` is the combat payload's
    ``assignment_choices``, and a name change is a continuous effect's ``name``
    contribution. Emitting them as events would give one fact two spellings and
    let a reader believe a corpus without them was incomplete.
    """

    def test_the_superseded_types_are_out_of_the_vocabulary(self):
        for retired in ("cost_adjusted", "damage_assignment_ordered", "name_change"):
            assert retired not in EVENT_PARAMS
