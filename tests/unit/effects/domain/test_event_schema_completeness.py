"""Every Forge effect API maps to an event type or an explicit exclusion (T025).

Runs against the checked-in class list in
:mod:`effects.domain.forge_effect_apis`, so it needs no JVM and no ``../forge``
checkout. A Forge upgrade that adds an effect API fails here, which is the
point: a new API is a new thing an ability can do, and a corpus that silently
stops describing it cannot be repaired without recollecting.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from effects.domain import event_schema
from effects.domain.event_schema import (
    ATTRIBUTION_ROOT,
    ATTRIBUTION_SENTINELS,
    ATTRIBUTION_UNRESOLVED,
    CAUSE_BEARING_TYPES,
    EFFECT_API_EVENTS,
    EVENT_PARAMS,
    EXCLUDED_EFFECT_APIS,
    SUPERSEDED_EVENT_TYPES,
    Event,
    EventType,
    attribution_kind,
    event_types_for,
)
from effects.domain.forge_effect_apis import (
    EFFECT_APIS,
    GAME_EVENTS,
    REPLACEMENT_TYPES,
    TRIGGER_TYPES,
)

_FORGE_GAME = Path(r"C:/Users/nicol/IdeaProjects/forge") / (
    "forge-game/src/main/java/forge/game"
)

# Matches an enum constant of the shape ``Name (SomeClass.class)`` -- the same
# pattern scripts/regenerate_forge_api_list.py uses to build the checked-in
# lists in the first place. Anchored on the ``.class`` literal rather than a
# bare ``(`` after the name: Forge's own formatting is inconsistent about the
# space before the paren (``FlipCoin(FlipCoinEffect.class)`` has none, most
# members do), and a regex that requires the space silently drops the ones
# that lack it -- the same way the checked-in list silently drops live
# members. Anchoring on ``.class`` instead catches every member regardless of
# that spacing while still skipping each file's own constructor overload,
# whose parameter list is never a bare ``Xxx.class``.
_ENUM_CONSTANT = re.compile(r"^\s+([A-Za-z]+)\s*\(\s*[A-Za-z]+\.class", re.M)


def _live_enum_members(java_file: Path) -> set[str] | None:
    """Live enum member names in ``java_file``, or None with no checkout beside us."""
    if not java_file.exists():
        return None
    return set(_ENUM_CONSTANT.findall(java_file.read_text(encoding="utf-8")))


def _live_game_events() -> set[str] | None:
    """Live ``GameEvent*`` bus-event class stems, or None with no checkout beside us.

    ``GameEvent.java`` itself is the abstract base every real event extends,
    not an event of its own, so it is excluded rather than counted.
    """
    event_dir = _FORGE_GAME / "event"
    if not event_dir.exists():
        return None
    return {p.stem for p in event_dir.glob("GameEvent*.java") if p.stem != "GameEvent"}


class TestTheCheckedInListsMatchForge:
    """Each checked-in list is a snapshot of an enum or a directory, and
    snapshots rot.

    They are checked in so the suite needs no JVM, which is right -- and it
    means nothing compares them to Forge unless a test does. ``EFFECT_APIS``
    had drifted in both directions and the completeness tests below stayed
    green throughout, because they only ever compared the mapping against
    this same checked-in list, never against Forge itself. The other three
    lists were never compared to Forge at all: this same regenerate-and-diff
    turned up a missing live member (``DayTimeChanges``), a phantom entry
    that was never live (``"ReplacementType"``, the enum's own type name) and
    an abstract base class counted as if it were an event (``GameEvent``) --
    all silently, because only a ``>N`` floor (below) ever checked them.
    """

    _CHECKED_IN_VS_LIVE = {
        "EFFECT_APIS": (EFFECT_APIS, _FORGE_GAME / "ability/ApiType.java"),
        "TRIGGER_TYPES": (TRIGGER_TYPES, _FORGE_GAME / "trigger/TriggerType.java"),
        "REPLACEMENT_TYPES": (
            REPLACEMENT_TYPES, _FORGE_GAME / "replacement/ReplacementType.java",
        ),
    }

    @pytest.mark.parametrize(
        "checked_in, java_file", _CHECKED_IN_VS_LIVE.values(),
        ids=list(_CHECKED_IN_VS_LIVE),
    )
    def test_every_checked_in_enum_member_exists_in_forge(self, checked_in, java_file):
        live = _live_enum_members(java_file)
        if live is None:
            pytest.skip("no ../forge checkout beside this one")
        assert set(checked_in) - live == set()

    @pytest.mark.parametrize(
        "checked_in, java_file", _CHECKED_IN_VS_LIVE.values(),
        ids=list(_CHECKED_IN_VS_LIVE),
    )
    def test_every_live_enum_member_is_checked_in(self, checked_in, java_file):
        live = _live_enum_members(java_file)
        if live is None:
            pytest.skip("no ../forge checkout beside this one")
        assert live - set(checked_in) == set()

    def test_every_checked_in_game_event_exists_in_forge(self):
        live = _live_game_events()
        if live is None:
            pytest.skip("no ../forge checkout beside this one")
        assert set(GAME_EVENTS) - live == set()

    def test_every_live_game_event_is_checked_in(self):
        live = _live_game_events()
        if live is None:
            pytest.skip("no ../forge checkout beside this one")
        assert live - set(GAME_EVENTS) == set()


def _count_dot_class_constants(java_file: Path) -> int | None:
    """How many enum constants ``java_file`` declares, counted a different way
    than :data:`_ENUM_CONSTANT` above: strip commented-out lines, then count
    the ones ending ``.class),`` -- no anchoring on leading whitespace then a
    captured name, no ``re`` at all on the matching side. A parsing bug shared
    by both counting strategies (say, Forge one day qualifying a class name as
    ``effects.FooEffect.class``, which ``_ENUM_CONSTANT``'s ``[A-Za-z]+``
    cannot see either) would make :class:`TestTheCheckedInListsMatchForge`
    agree with itself and stay green while a live member goes uncounted on
    both sides at once; this guard is built not to share that blind spot.
    """
    if not java_file.exists():
        return None
    lines = (line.strip() for line in java_file.read_text(encoding="utf-8").splitlines())
    return sum(1 for line in lines if not line.startswith("//") and line.endswith(".class),"))


class TestTheEffectApiCountIsCrossChecked:
    """A count derived from the same regex as the names it counts can share
    that regex's blind spot; this checks ``EFFECT_APIS`` against a total
    computed by an unrelated algorithm instead of trusting the two to agree
    with themselves.
    """

    def test_the_member_count_agrees_with_an_independent_count(self):
        independent = _count_dot_class_constants(_FORGE_GAME / "ability/ApiType.java")
        if independent is None:
            pytest.skip("no ../forge checkout beside this one")
        assert len(EFFECT_APIS) == independent


def _dict_literal_key_count(module_file: Path, name: str) -> int:
    """How many keys ``name``'s dict literal writes out in its own source.

    A repeated key is not a Python error -- the later value silently
    overwrites the earlier one -- so ``len(the_dict)`` can never see it; only
    the source text can.
    """
    tree = ast.parse(module_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and isinstance(node.value, ast.Dict)
        ):
            return len(node.value.keys)
    raise AssertionError(f"{name} is not an annotated dict literal in {module_file}")


class TestTheDictLiteralsHaveNoRepeatedKey:
    """A repeated key is a silent Python no-op, not an error.

    ``test_no_api_is_both_mapped_and_excluded`` (below) catches a name
    present in *both* ``EFFECT_API_EVENTS`` and ``EXCLUDED_EFFECT_APIS``;
    nothing catches one repeated *within* a single dict -- the later entry
    would simply win and the earlier one's event types would vanish with no
    test noticing, the same silent-emptiness this task exists to close,
    relocated from a mis-keyed API to a mis-typed one. Re-keying 33 entries by
    hand, as this task did, is exactly the kind of edit that risks it.
    """

    @pytest.mark.parametrize("name, mapping", [
        ("EFFECT_API_EVENTS", EFFECT_API_EVENTS),
        ("EXCLUDED_EFFECT_APIS", EXCLUDED_EFFECT_APIS),
    ])
    def test_the_source_never_writes_a_key_twice(self, name, mapping):
        written = _dict_literal_key_count(Path(event_schema.__file__), name)
        assert written == len(mapping), (
            f"{name}'s source literal writes {written} keys but the dict "
            f"holds {len(mapping)} -- a duplicate key silently dropped an entry"
        )


class TestCoverage:
    def test_every_effect_api_is_mapped_or_excluded(self):
        covered = set(EFFECT_API_EVENTS) | set(EXCLUDED_EFFECT_APIS)
        missing = sorted(set(EFFECT_APIS) - covered)
        assert not missing, (
            f"{len(missing)} Forge effect APIs are neither mapped in "
            f"EFFECT_API_EVENTS nor excluded with a reason: {missing}"
        )

    def test_nothing_is_mapped_that_forge_does_not_have(self):
        covered = set(EFFECT_API_EVENTS) | set(EXCLUDED_EFFECT_APIS)
        stale = sorted(covered - set(EFFECT_APIS))
        assert not stale, (
            "the vocabulary maps effect APIs that no longer exist in the "
            f"checked-in Forge class list; regenerate or remove: {stale}"
        )

    def test_no_api_is_both_mapped_and_excluded(self):
        both = sorted(set(EFFECT_API_EVENTS) & set(EXCLUDED_EFFECT_APIS))
        assert not both, f"ambiguous coverage for: {both}"

    def test_every_exclusion_states_a_reason(self):
        blank = sorted(k for k, v in EXCLUDED_EFFECT_APIS.items() if len(v) < 15)
        assert not blank, f"exclusions without a usable reason: {blank}"

    def test_every_mapped_api_names_at_least_one_event_type(self):
        empty = sorted(k for k, v in EFFECT_API_EVENTS.items() if not v)
        assert not empty, (
            f"mapped to no event type; these belong in EXCLUDED_EFFECT_APIS "
            f"with a reason: {empty}"
        )

    @pytest.mark.parametrize("api", sorted(EFFECT_API_EVENTS))
    def test_a_mapped_api_resolves_to_known_event_types(self, api: str):
        assert all(isinstance(t, EventType) for t in event_types_for(api))

    @pytest.mark.parametrize("api", sorted(EXCLUDED_EFFECT_APIS))
    def test_an_excluded_api_resolves_to_no_event_type(self, api: str):
        assert event_types_for(api) == ()

    def test_an_unknown_api_fails_loudly(self):
        with pytest.raises(KeyError, match="neither mapped"):
            event_types_for("SomeEffectForgeAddedLastWeek")


class TestVocabulary:
    def test_the_checked_in_forge_lists_are_populated(self):
        """A regeneration that silently produced nothing would pass everything else."""
        assert len(EFFECT_APIS) > 150
        assert len(TRIGGER_TYPES) > 100
        assert len(GAME_EVENTS) > 40
        assert len(REPLACEMENT_TYPES) > 20

    def test_event_type_values_are_unique_and_snake_case(self):
        values = [t.value for t in EventType]
        assert len(values) == len(set(values))
        assert all(v == v.lower() and " " not in v for v in values)

    def test_every_normalized_type_is_a_real_event_type(self):
        """``EVENT_PARAMS``'s keys are well-formed, nothing more.

        Narrower than it sounds next to ``TestEveryDeclaredTypeIsReachable``: this
        only checks that a type with a params row is a real ``EventType`` member,
        not that every declared member is reachable. ``EVENT_PARAMS`` keys are
        written as ``EventType.X`` literals, so this is close to tautological in
        practice — the reachability question is answered by
        ``test_no_declared_type_is_unreachable_by_surprise``, over the full
        vocabulary, not this one.
        """
        assert set(EVENT_PARAMS) <= set(EventType)

    def test_the_vocabulary_is_smaller_than_forges_api_list(self):
        """Types name outcomes, not the script API that produced them."""
        assert len(EventType) < len(EFFECT_APIS)

    def test_the_common_outcome_families_are_all_present(self):
        for name in (
            "ZONE_CHANGE", "DAMAGE_DEALT", "LIFE_CHANGE", "COUNTER_CHANGE",
            "PT_CHANGE", "KEYWORD_CHANGE", "TOKEN_CREATED", "CARD_DRAWN",
        ):
            assert hasattr(EventType, name)


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

    def test_a_continuous_outcome_carries_its_duration(self):
        event = Event(
            type=EventType.PT_CHANGE,
            params={"power_delta": 3, "toughness_delta": 3},
            duration="end_of_turn",
        )
        assert event.duration == "end_of_turn"


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

    def test_it_is_exactly_the_rows_that_declare_a_cause(self):
        assert CAUSE_BEARING_TYPES == {
            event_type for event_type, params in EVENT_PARAMS.items()
            if "cause" in params
        }

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
        assert all(not value.isdigit() for value in ATTRIBUTION_SENTINELS)

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
#: ``event_type_coverage`` in ``validate_corpus.py``, run against a real
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
#: ``EffectEvent.REGENERATED`` also has a second, non-emitting reference this scan
#: finds -- ``BusBracketCollector.IDEMPOTENT_EVENTS`` -- which was inert while the
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
#: Widening the guard to ``set(EventType)`` is what surfaced the 15 below; they
#: were unreachable and unchecked for the entire life of this plan, not broken by
#: widening it. Going from ``frozenset()`` to a populated mapping is the set
#: finally being honest about a gap that predates this plan and was invisible the
#: whole time it ran.
#:
#: Values are the reason: the producing Forge effect API(s), whether an emission
#: path exists (``ApiEvents.RULES`` in the connector, or an explicit
#: ``EffectRecordOutcomes.note`` call -- none of these 15 have either), and how
#: many cards in the sealed/draft pool script that API, read directly from
#: ``forge-gui/res/cardsfolder`` rather than estimated.
KNOWN_UNEMITTED: dict[str, str] = {
    "ability_activated":
        "ActivateAbility: no ApiEvents.RULES entry, no outcome-note call; "
        "2 cards script it",
    "combat_ended":
        "EndCombatPhase: no ApiEvents.RULES entry, no outcome-note call; "
        "1 card scripts it",
    "delayed_trigger_created":
        "DelayedTrigger/ImmediateTrigger: no ApiEvents.RULES entry, no "
        "outcome-note call; 445 + 312 = 757 cards, the largest count of any "
        "type in this dict",
    "game_drawn":
        "GameDrawn: no ApiEvents.RULES entry, no outcome-note call; "
        "2 cards script it",
    "game_restarted":
        "RestartGame: no ApiEvents.RULES entry, no outcome-note call; "
        "1 card scripts it",
    "initiative_taken":
        "TakeInitiative: no ApiEvents.RULES entry, no outcome-note call; "
        "23 cards script it",
    "monarch_changed":
        "BecomeMonarch: no ApiEvents.RULES entry, no outcome-note call; "
        "60 cards script it -- a whole named mechanic (Monarch), not a corner case",
    "player_removed":
        "RemoveFromMatch: no ApiEvents.RULES entry, no outcome-note call; "
        "2 cards script it",
    "removed_from_combat":
        "ChangeCombatants/RemoveFromCombat: no ApiEvents.RULES entry, no "
        "outcome-note call; 9 + 28 = 37 cards",
    "replacement_applied":
        "The six Replace* APIs (ReplaceEffect, ReplaceCounter, ReplaceDamage, "
        "ReplaceMana, ReplaceSplitDamage, ReplaceToken): no ApiEvents.RULES "
        "entry, no outcome-note call; ~304 cards combined",
    "ring_tempts":
        "RingTemptsYou: no ApiEvents.RULES entry, no outcome-note call; "
        "49 cards script it (set-gated: Lord of the Rings only)",
    "text_change":
        "ChangeText/ExchangeTextBox: no ApiEvents.RULES entry, no outcome-note "
        "call; 12 + 2 = 14 cards",
    "trigger_fired":
        "No Forge effect API maps to it at all -- absent from EFFECT_API_EVENTS "
        "entirely. Its only plausible source is the trigger-fire hook itself, "
        "which already reports through the trigger record kind's own `fired` "
        "flag rather than this type",
    "turn_ended":
        "EndTurn: no ApiEvents.RULES entry, no outcome-note call; "
        "9 cards script it",
    "turn_order_reversed":
        "ReverseTurnOrder: no ApiEvents.RULES entry, no outcome-note call; "
        "3 cards script it",
}


def _emitted_event_types() -> set[str]:
    """Event types whose constants are referenced in the connector's Java source.

    This is a textual static scan—it finds ``EffectEvent.CONSTANT_NAME`` in
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
    ``event_type_coverage`` in ``validate_corpus.py``, run against a real
    collection, exists as this scan's permanent counterpart rather than a
    one-off check.
    """
    header = (_CONNECTOR_EFFECTS / "EffectEvent.java").read_text(encoding="utf-8")
    constants = dict(re.findall(
        r'public static final String ([A-Z_]+)\s*=\s*"([a-z_]+)"', header,
    ))
    emitted: set[str] = set()
    for path in _CONNECTOR_EFFECTS.glob("*.java"):
        if path.name == "EffectEvent.java":
            continue
        text = path.read_text(encoding="utf-8")
        for name, value in constants.items():
            if re.search(rf"EffectEvent\.{name}\b", text):
                emitted.add(value)
    return emitted


class TestEveryDeclaredTypeIsReachable:
    """The vocabulary and the writer are one contract — the *whole* vocabulary.

    ``test_every_effect_api_is_mapped_or_excluded`` (in ``TestCoverage``) asserts
    the *map* is total: every Forge effect API names an event type or an explicit
    exclusion. It passed while some declared types were unreachable in the corpus,
    because a mapping to a type nothing emits is still a mapping. That test cannot
    catch the gap; this is the other half. The gap has two parts, and this class
    only fully covers one of them as of the Task 10 fix round that widened it:

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
       list at all. The remaining four are documented with their card counts in
       the spec rather than here, because no static scan — however wide — can
       see this class. Only a corpus measurement can, which is what makes
       ``event_type_coverage`` in ``validate_corpus.py`` this test's permanent
       counterpart rather than a one-off check that stops mattering once the
       guard is fixed.
    """

    def test_no_declared_type_is_unreachable_by_surprise(self):
        unreachable = set(EventType) - _emitted_event_types()
        assert unreachable == set(KNOWN_UNEMITTED), (
            "declared-but-unemitted types changed; wired: "
            f"{sorted(set(KNOWN_UNEMITTED) - unreachable)}, newly unreachable: "
            f"{sorted(unreachable - set(KNOWN_UNEMITTED))}"
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


class TestSupersededTypes:
    """A type the record carries under another name is not an event.

    ``cost_adjusted`` is the playability decision's ``cost_after_adjustment``,
    ``damage_assignment_ordered`` is the combat payload's
    ``assignment_choices``, and a name change is a continuous effect's ``name``
    contribution. Emitting them as events would give one fact two spellings and
    let a reader believe a corpus without them was incomplete.
    """

    def test_the_superseded_types_are_out_of_the_vocabulary(self):
        for retired in SUPERSEDED_EVENT_TYPES:
            assert retired not in EVENT_PARAMS

    def test_each_names_where_the_information_lives(self):
        assert SUPERSEDED_EVENT_TYPES == {
            "cost_adjusted":
                "playability/decision payload, candidates[].cost_after_adjustment",
            "damage_assignment_ordered":
                "combat payload, assignment_choices",
            "name_change":
                "continuous payload, contributions[].name",
        }

    def test_no_effect_api_maps_to_a_superseded_type(self):
        for events in EFFECT_API_EVENTS.values():
            assert not (set(events) & set(SUPERSEDED_EVENT_TYPES))

