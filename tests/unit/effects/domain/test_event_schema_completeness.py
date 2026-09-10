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

#: Types with no reference in the connector's Java source. Distinct from
#: ``damage_prevented`` and ``spell_copied``, which are referenced but never
#: fire in corpus data (wired, not yet fired). This set shrinks as each type
#: is moved from unreferenced to emitted; a type silently added or removed here
#: fails the test, which is the point.
KNOWN_UNEMITTED: frozenset[str] = frozenset({
    "ability_change", "choice_made", "continuous_effect_created",
    "damage_healed", "targets_changed",
})


def _emitted_event_types() -> set[str]:
    """Event types whose constants are referenced in the connector's Java source.

    This is a textual static scan—it finds ``EffectEvent.CONSTANT_NAME`` in
    code without distinguishing whether that code is on a live path or dead code.
    A type may be referenced but never fired: across 10.1M corpus records,
    ``damage_prevented`` and ``spell_copied`` appear zero times despite being
    referenced in ``PatchedCollectors.java``. Conversely, a type unreferenced
    in source is truly unreachable. This test guards against the under-count
    direction only—the corpus-empirical gap is deeper and cannot be caught by
    static analysis: some declared types may be referenced but unfired. The
    unreferenced types are recorded in ``KNOWN_UNEMITTED`` and shrink as each
    is wired and emitted; the referenced-but-unfired gap is separately
    documented here, so both gaps can be tracked independently.
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
    """The vocabulary and the writer are one contract.

    ``test_every_effect_api_is_mapped_or_excluded`` (in ``TestCoverage``) asserts
    the *map* is total: every Forge effect API names an event type or an explicit
    exclusion. It passed while some declared types were unreachable in the corpus,
    because a mapping to a type nothing emits is still a mapping. That test cannot
    catch the gap; this is the other half. The gap has two parts: types not
    referenced in the connector source (unreachable; recorded in ``KNOWN_UNEMITTED``)
    and types referenced but never fired in the corpus (wired but unfired; tracked
    separately). A corpus analysis of 10.1M records found two such wired-but-unfired
    types: ``damage_prevented`` and ``spell_copied``. This test guards against
    unreachable types gaining one unexpectedly, and ``KNOWN_UNEMITTED`` shrinks as
    each is wired and starts emitting.
    """

    def test_no_declared_type_is_unreachable_by_surprise(self):
        unreachable = set(EVENT_PARAMS) - _emitted_event_types()
        assert unreachable == set(KNOWN_UNEMITTED), (
            "declared-but-unemitted types changed; wired: "
            f"{sorted(set(KNOWN_UNEMITTED) - unreachable)}, newly unreachable: "
            f"{sorted(unreachable - set(KNOWN_UNEMITTED))}"
        )

    def test_the_known_list_names_only_declared_types(self):
        assert set(KNOWN_UNEMITTED) <= set(EVENT_PARAMS)


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

