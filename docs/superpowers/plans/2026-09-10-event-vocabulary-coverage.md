# Event Vocabulary Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every event type the schema declares reachable by the collector, so an ability whose effect the corpus cannot describe stops being recorded as an ability that did nothing.

**Architecture:** Three mechanisms, in order of cost. Four types are already broadcast on Forge's event bus and need only a subscription. Twenty-two are produced by a named effect API whose own parameters and results are readable from the resolving `SpellAbility` — those need two new Forge hooks — a notification when a clause resolves, and one an effect calls with an outcome it is about to discard — and one API-keyed emitter table in the connector. Three are superseded by data the record already carries elsewhere and are retired from the vocabulary rather than wired. A guard test asserts the declared set and the emitted set are equal, so the gap cannot reopen.

**Tech Stack:** Java 17 (forge-game, forge-connector), Python 3.14 (src/effects), JUnit 5 in forge-connector, TestNG in forge, pytest in price-predictor, Maven.

**Spec:** `specs/2026-09-05-ability-effect-model.md` § Events, and `specs/023-ability-effect-model/contracts/record-schema.md` § Events. The event vocabulary and its per-type parameters live in `src/effects/domain/event_schema.py`.

## Global Constraints

- **A collection run must not be live while this is built.** Workers load `forge-connector/target/forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar` at spawn and the pool recycles one every 60 seconds, so `mvn package`/`install` mid-run mixes two writer versions inside one corpus. `mvn test` is safe; `package`, `install` and `clean` are not.
- **Forge hooks stay inert.** A hook must add no work and change no behaviour when no listener is installed, must not alter any return value or engine mutation, and gets a TestNG test beside the existing `EffectRecordParamCopyTest` / `EffectRecordReplacementOutcomeTest`.
- **`-Xmx1200m` and the 60-second worker recycle are deliberate containment** for Forge misbehaving after many games in one JVM. Do not change either, or their explanatory comments.
- **Never write into `output/effects/records/` or `output/sealed/`.** Tests use temp directories; measurement runs use `python -m effects collect-coverage`, which writes effect records and never touches the sealed corpora.
- **Test baselines, all of which must still pass:** 638 Java tests (`mvn -B -o test` in `forge-connector`), 1,844 Python tests (`pytest tests/unit/effects tests/unit/sealed`), plus the Forge-side hook tests.
- **The record schema is frozen in one direction:** later stages may add `kind` values and event types but never redefine a field. Removing a declared event type is allowed only because no corpus has ever contained one — state that reason in the commit.
- **Record what the engine did, not what the script said it would do.** A script parameter is
  acceptable only where it *is* the outcome — a phase name, a restriction kind, a zone. Anywhere the
  two can differ (a count, an amount, an identity, a chosen thing), the value comes from the engine:
  `AbilityUtils.calculateAmount` where the number is computed at emit time, the effect's own report
  where the result exists only inside it, and the engine's choke point where one exists. An event
  that describes an intention is worse than a missing one, because nothing downstream can tell them
  apart.
- **Prefer the choke point to the effect.** Where the engine funnels every instance of an outcome
  through one method — `GameAction.reveal`, `GameEntity.staticDamagePrevention` — hook that method
  rather than the effects that call it. One hook is more accurate than ten, covers the paths nobody
  enumerated (costs, replacement effects, triggered abilities), and cannot drift as effects are added.
  This is the lesson `zone_change` already taught: the per-card event beat the per-zone one.
- **`ApiType.name()` is the only key.** The emitter table is looked up with `sa.getApi().name()`, so
  every key must be a member of Forge's `ApiType` enum. `src/effects/domain/forge_effect_apis.py` is
  currently keyed by *effect class* name instead, and **35 of the 180 `EFFECT_API_EVENTS` keys are not
  `ApiType` members** (`ControlGain` for `GainControl`, `CountersPut` for `PutCounter`,
  `ControlExchange` for `ExchangeControl`, and 32 more). A rule keyed that way matches nothing and
  emits nothing, silently. Task 3 re-keys both, and a test asserts every key is a live enum member.
- Comments explain WHY. Javadoc explains design decisions rather than restating signatures.

## Measured Starting Point

From the 10,143,037-record corpus collected 2026-09-09/10 (`output/effects/records/`, 33,649 games):

- **31 of 60 declared event types were ever observed.** The other 29 are not rare — they are unreachable.
- 23 of the 29 have no constant anywhere in the connector; 4 have a constant nothing constructs (`day_night_changed`, `coin_flipped`, `radiation_change`, `speed_changed`); 2 are wired but unfired (`damage_prevented`, `spell_copied`).
- The bus collector subscribes to 16 of Forge's ~55 game events.
- Concrete damage: 16 of 39 Aether Hub resolutions in the corpus carry **no events at all**, and the rest carry only `mana_produced`. The head is being taught that "pay {E}, add one mana of any color" produces no outcome.
- **A result an effect computes is usually not readable afterwards.** `FlipCoin`, `Clash`, `Vote`, `TwoPiles`, `MultiplePiles` and `CopySpellAbility` write their result to the host only when the card's script opts in, and restore the previous remembered objects before the clause ends — verified at `FlipCoinEffect:107-118,154`, `ClashEffect:82`, `VoteEffect:138-140,156,175-179`, `TwoPilesEffect:152-167`, `MultiplePilesEffect:101`, `CopySpellAbilityEffect:206`. Reading the host after the clause therefore finds nothing for them, which is why Task 7 asks the effect instead of diffing its host.
- `test_event_schema_completeness` passes throughout, because it asserts every effect API maps to an event type or an explicit exclusion — that the *map* is total. It never asserts the writer can emit the mapped type.
- Separately, the checked-in API list has drifted from live Forge: **39 names in the list no longer exist in `ApiType`, and 38 live APIs are absent from it**, including `DealDamage`. The list appears to have been generated from effect *class* names rather than enum member names (`OwnershipGain` vs Forge's `GainOwnership`, `DamagePrevent` vs `PreventDamage`).

## File Structure

**forge (branch `effect-record-hooks`)**
- Modify: `forge-game/src/main/java/forge/game/ability/AbilityUtils.java` — the clause listener, at the `resolveApiAbility` wrapper that already owns the sub-ability pointer.
- Create: `forge-game/src/main/java/forge/game/ability/EffectRecordOutcomes.java` — the outcome notification an effect calls with what it computed and is about to discard.
- Modify: `forge-game/src/main/java/forge/game/ability/effects/{FlipCoin,Clash,Vote,TwoPiles,MultiplePiles,CopySpellAbility}Effect.java` — one guarded call each, at the point the result exists.
- Create: `forge-game/src/test/java/forge/game/ability/EffectRecordOutcomeHookTest.java` — the outcome hook reports, stays inert, and survives an observer that throws.
- Create: `forge-game/src/test/java/forge/game/ability/EffectRecordClauseHookTest.java` — the hook's two promises: it reports every clause, and it changes nothing when unlistened.

**price-predictor (branch `023-ability-effect-model`)**
- Create: `forge-connector/src/main/java/com/pricepredictor/connector/effects/ApiEvents.java` — the API-keyed emitter table. One file because the table is one responsibility: turning a resolved clause into the event its API promises.
- Modify: `.../effects/EffectEvent.java` — the missing type constants.
- Modify: `.../effects/BusEvents.java` — factories for the four broadcast types.
- Modify: `.../effects/BusBracketCollector.java` — four `@Subscribe` handlers, and the package-private door the emitter table records through.
- Modify: `.../effects/PatchedCollectors.java` — install the clause hook, call the table.
- Modify: `.../effects/PatchHooks.java` — the new hook joins the patch's description.
- Create/modify tests: `.../effects/ApiEventsTest.java`, `BusEventsTest.java`, `PatchedCollectorTest.java`.
- Modify: `src/effects/domain/event_schema.py` — retire three types, add `SUPERSEDED_EVENT_TYPES`, reconcile API names.
- Modify: `src/effects/domain/forge_effect_apis.py` — regenerated from live Forge.
- Create: `scripts/regenerate_forge_api_list.py` — the generator, so the list is never hand-edited again.
- Modify: `src/effects/application/validate_corpus.py` — event-type coverage as a watched number.
- Modify: `tests/unit/effects/domain/test_event_schema_completeness.py` — the guard test.
- Modify: `specs/2026-09-05-ability-effect-model.md`, `specs/023-ability-effect-model/contracts/record-schema.md`.

---

### Task 1: The guard test that makes the gap visible

The test that would have caught this on day one. It reads the connector's own source and asserts that the set of declared event types minus the set the writer can emit is exactly a checked-in list — so the list shrinks task by task and nobody can add an unreachable type silently.

**Files:**
- Modify: `tests/unit/effects/domain/test_event_schema_completeness.py`
- Reads: `forge-connector/src/main/java/com/pricepredictor/connector/effects/*.java`

**Interfaces:**
- Produces: `KNOWN_UNEMITTED: frozenset[str]` in the test module — every later task removes entries from it.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/effects/domain/test_event_schema_completeness.py`:

```python
import re
from pathlib import Path

#: The connector's effects package, read as text. A type the writer never names
#: cannot appear in a corpus however many games are played, and that is
#: invisible from the corpus itself: "rare" and "unreachable" look identical.
_CONNECTOR_EFFECTS = (
    Path(__file__).resolve().parents[4]
    / "forge-connector" / "src" / "main" / "java" / "com" / "pricepredictor"
    / "connector" / "effects"
)

#: Types the writer cannot emit yet. Shrinks as each is wired; a type added
#: here needs a reason in the same commit.
KNOWN_UNEMITTED: frozenset[str] = frozenset({
    "ability_change", "card_made", "card_revealed", "choice_made",
    "clash_resolved", "coin_flipped", "continuous_effect_created",
    "cost_adjusted", "damage_assignment_ordered", "damage_healed",
    "damage_prevented", "day_night_changed", "dungeon_ventured",
    "energy_change", "name_change", "ownership_change", "permanent_copied",
    "phase_added", "phase_skipped", "piles_made", "radiation_change",
    "restriction_change", "speed_changed", "spell_copied", "targets_changed",
    "turn_added", "turn_skipped", "vote_taken", "x_changed",
})


def _emitted_event_types() -> set[str]:
    """Every event type some collector actually constructs.

    A constant declared in ``EffectEvent`` and referenced nowhere else is a
    name, not a channel: ``day_night_changed`` had a constant for months and no
    code path that built one.
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

    ``test_every_effect_api_maps_to_an_event`` asserts the *map* is total: every
    Forge effect API names an event type or an explicit exclusion. It passed
    while 29 of 60 declared types were unreachable, because a mapping to a type
    nothing emits is still a mapping. This is the other half.
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
```

- [ ] **Step 2: Run it and watch it pass with the gap recorded**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/effects/domain/test_event_schema_completeness.py -v`
Expected: PASS. The 29-name list is the gap, now written down and enforced.

- [ ] **Step 3: Prove the test bites**

Temporarily delete `"energy_change"` from `KNOWN_UNEMITTED`, re-run, and confirm it FAILS with `newly unreachable: []` / `wired: ['energy_change']`. Restore the entry.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/effects/domain/test_event_schema_completeness.py
git commit -m "test(effects): pin which declared event types the writer cannot emit"
```

---

### Task 2: Retire the three superseded types

`cost_adjusted`, `damage_assignment_ordered` and `name_change` map to no effect API (`apis(0)` in `EFFECT_API_EVENTS`) because the information they name is already recorded elsewhere. Wiring them would duplicate a field under a second name.

**Files:**
- Modify: `src/effects/domain/event_schema.py`
- Modify: `tests/unit/effects/domain/test_event_schema_completeness.py`
- Test: `tests/unit/effects/domain/test_event_schema_completeness.py`

**Interfaces:**
- Produces: `SUPERSEDED_EVENT_TYPES: dict[str, str]` — retired type → where the information actually lives.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/effects/domain/test_event_schema_completeness.py -k Superseded -v`
Expected: FAIL with `NameError: name 'SUPERSEDED_EVENT_TYPES' is not defined`.

- [ ] **Step 3: Implement**

In `src/effects/domain/event_schema.py`, delete the `cost_adjusted`, `damage_assignment_ordered` and `name_change` entries from `EVENT_PARAMS`, and add beside it:

```python
#: Event types the vocabulary once declared, and where the fact they named is
#: actually recorded. Retired rather than wired: each would have given one fact
#: two spellings, and a reader comparing a corpus against the vocabulary would
#: have read their absence as a collection failure.
#:
#: Removing a declared type is safe in exactly one direction: no corpus has ever
#: contained one, because nothing could emit them.
SUPERSEDED_EVENT_TYPES: dict[str, str] = {
    "cost_adjusted":
        "playability/decision payload, candidates[].cost_after_adjustment",
    "damage_assignment_ordered":
        "combat payload, assignment_choices",
    "name_change":
        "continuous payload, contributions[].name",
}
```

Then remove those three from `KNOWN_UNEMITTED` in the test module (26 remain).

- [ ] **Step 4: Run the whole effects suite**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/effects -q`
Expected: PASS, 1,844 or more.

- [ ] **Step 5: Commit**

```bash
git add src/effects/domain/event_schema.py tests/unit/effects/domain/test_event_schema_completeness.py
git commit -m "refactor(effects): retire three event types the record already carries"
```

---

### Task 3: Refresh the Forge API list from live Forge

The checked-in list drifted: 39 names it holds are absent from `ApiType`, 38 live APIs are absent from it, and `DealDamage` — the commonest effect in the game — is among the missing. The list was generated from effect class names, not enum member names, which is why the map says `OwnershipGain` where Forge says `GainOwnership`.

**Files:**
- Create: `scripts/regenerate_forge_api_list.py`
- Modify: `src/effects/domain/forge_effect_apis.py` (regenerated)
- Modify: `src/effects/domain/event_schema.py` (rename the two mis-keyed entries)
- Test: `tests/unit/effects/domain/test_event_schema_completeness.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `EFFECT_APIS: frozenset[str]` regenerated; every key of `EFFECT_API_EVENTS` and `EXCLUDED_EFFECT_APIS` is a live `ApiType` member name.

- [ ] **Step 1: Write the failing test**

```python
_FORGE_API_TYPE = Path(r"C:/Users/nicol/IdeaProjects/forge") / (
    "forge-game/src/main/java/forge/game/ability/ApiType.java"
)


def _live_api_names() -> set[str] | None:
    """Forge's own enum member names, or None when no checkout is beside us."""
    if not _FORGE_API_TYPE.exists():
        return None
    return set(re.findall(
        r"^\s+([A-Za-z]+) \(", _FORGE_API_TYPE.read_text(encoding="utf-8"), re.M,
    ))


class TestTheApiListMatchesForge:
    """The checked-in list is a snapshot of an enum, and snapshots rot.

    It is checked in so the suite needs no JVM, which is right — and it means
    nothing compares it to Forge unless a test does. It had drifted by 39 names
    in one direction and 38 in the other, and the completeness test stayed green
    because it compares the list against itself.
    """

    def test_every_checked_in_api_exists_in_forge(self):
        live = _live_api_names()
        if live is None:
            pytest.skip("no ../forge checkout beside this one")
        assert set(EFFECT_APIS) - live == set()

    def test_every_forge_api_is_checked_in(self):
        live = _live_api_names()
        if live is None:
            pytest.skip("no ../forge checkout beside this one")
        assert live - set(EFFECT_APIS) == set()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/effects/domain/test_event_schema_completeness.py -k ApiListMatchesForge -v`
Expected: FAIL, 39 names in the list absent from Forge and 38 the other way.

- [ ] **Step 3: Write the generator**

Create `scripts/regenerate_forge_api_list.py`:

```python
"""Regenerate the checked-in Forge effect-API list from a Forge checkout.

The list exists so the test suite needs no JVM and no sibling checkout. That is
worth keeping and is exactly why it drifts: nothing regenerates it when Forge
moves. Run this after a Forge upgrade; the test beside it fails until you do.

    python scripts/regenerate_forge_api_list.py --forge ../forge
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

HEADER = '''"""Forge's effect APIs, trigger modes, replacement types and game events.

Generated by ``scripts/regenerate_forge_api_list.py`` from a Forge checkout.
Do not hand-edit: the names are enum members, and a hand-fixed name is how the
list came to say ``OwnershipGain`` where Forge says ``GainOwnership``.
"""

from __future__ import annotations

'''


def enum_members(path: Path) -> list[str]:
    return sorted(set(re.findall(r"^\\s+([A-Za-z]+) \\(", path.read_text(encoding="utf-8"), re.M)))


def simple_members(path: Path) -> list[str]:
    return sorted(set(re.findall(r"^\\s+([A-Za-z]+)\\s*[,;]", path.read_text(encoding="utf-8"), re.M)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forge", type=Path, default=Path("../forge"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("src/effects/domain/forge_effect_apis.py"),
    )
    args = parser.parse_args()
    game = args.forge / "forge-game/src/main/java/forge/game"
    blocks = {
        "EFFECT_APIS": enum_members(game / "ability/ApiType.java"),
        "TRIGGER_TYPES": enum_members(game / "trigger/TriggerType.java"),
        "REPLACEMENT_TYPES": simple_members(game / "replacement/ReplacementType.java"),
        "GAME_EVENTS": sorted(
            p.stem for p in (game / "event").glob("GameEvent*.java")
        ),
    }
    lines = [HEADER]
    for name, members in blocks.items():
        lines.append(f"{name}: frozenset[str] = frozenset({{")
        lines.extend(f'    "{m}",' for m in members)
        lines.append("})\n\n")
    args.output.write_text("\n".join(lines), encoding="utf-8")
    for name, members in blocks.items():
        print(f"{name}: {len(members)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Regenerate and reconcile the two renamed keys**

Run: `.venv/Scripts/python.exe scripts/regenerate_forge_api_list.py --forge ../forge`

Then re-key `EFFECT_API_EVENTS` and `EXCLUDED_EFFECT_APIS` in
`src/effects/domain/event_schema.py`. **35 of the 180 mapping keys and 3 of the exclusions are effect
class names rather than `ApiType` members** — `ControlGain`, `CountersPut`, `DamageDeal`, `LifeGain`,
`ZoneExchange`, `ChooseCardName`, `Permanent`, `Replace` and 30 more. Every one of them is looked up by
`ApiType.name()` at runtime, so every one of them matches nothing.

Do not hand-map them. `ApiType.java` already states the pairing — each member names its effect class,
`GainControl (ControlGainEffect.class)` — so the generator emits `class-stem -> member` from the same
parse it already does, and the re-key is mechanical and checkable. Where a class-named key and its
member-named twin both end up present, keep one entry with the member name and the union of their
event types.

The two the earlier draft called out by hand are just two of the thirty-eight:

```python
    "GainOwnership": ("ownership_change",),   # was OwnershipGain, the effect class name
    "PreventDamage": ("damage_prevented",),   # was DamagePrevent, the effect class name
```

- [ ] **Step 5: Map or exclude every newly-visible API**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/effects/domain/test_event_schema_completeness.py -v`

The existing `test_every_effect_api_maps_to_an_event` now fails for each of the 38 APIs that were invisible. For each, add an entry to `EFFECT_API_EVENTS` naming the event types it produces, or to `EXCLUDED_EFFECT_APIS` with the reason it produces none. `DealDamage` and `EachDamage` map to `("damage_dealt",)`; `AddOrRemoveCounter` to `("counter_change",)`; `ExchangeControl`, `ExchangeControlVariant` to `("control_change",)`; `ExchangeLife`, `ExchangeLifeVariant` to `("life_change",)`; `ExchangeZone` to `("zone_change",)`; `ExchangePower` to `("pt_change",)`; `ExchangeTextBox` to `("ability_change",)`; `Cleanup` and `CompanionChoose` are bookkeeping with no game-visible outcome and go in `EXCLUDED_EFFECT_APIS`. Work through the failure list until it is empty.

- [ ] **Step 6: Run the effects suite**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/effects -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/regenerate_forge_api_list.py src/effects/domain/forge_effect_apis.py src/effects/domain/event_schema.py tests/unit/effects/domain/test_event_schema_completeness.py
git commit -m "fix(effects): regenerate the Forge API list and reconcile it with the enum"
```

---

### Task 4: Subscribe the four types Forge already broadcasts

`energy_change`, `radiation_change`, `speed_changed` and `day_night_changed` need no hook: Forge fires an event for each and the collector simply never subscribed. Energy is the valuable one — it is `CounterEnumType.ENERGY`, a *player* counter, so it rides `GameEventPlayerCounters`, and poison only appears in corpora today because it happens to have a dedicated event that is subscribed.

**Files:**
- Modify: `forge-connector/src/main/java/com/pricepredictor/connector/effects/EffectEvent.java`
- Modify: `forge-connector/src/main/java/com/pricepredictor/connector/effects/BusEvents.java`
- Modify: `forge-connector/src/main/java/com/pricepredictor/connector/effects/BusBracketCollector.java`
- Test: `forge-connector/src/test/java/com/pricepredictor/connector/effects/BusEventsTest.java`

**Interfaces:**
- Consumes: `EffectEvent(String type)`, `.subject(String id)`, `.param(String key, Object value)` — the existing builder.
- Produces: `BusEvents.playerCounter`, `BusEvents.radiation`, `BusEvents.speed`, `BusEvents.dayTime`, each returning `EffectEvent` or null.

- [ ] **Step 1: Write the failing test**

Add to `forge-connector/src/test/java/com/pricepredictor/connector/effects/BusEventsTest.java`:

```java
    /**
     * Energy is a player counter, so it arrives on the counters event and not on
     * one of its own. Nothing subscribed to that event, which is why a corpus of
     * ten million records contains no energy at all and Aether Hub reads as an
     * ability that does nothing.
     */
    @Test
    void anEnergyCounterBecomesAnEnergyChangeEvent() {
        EffectEvent event = BusEvents.playerCounter(
                new GameEventPlayerCounters(
                        player, CounterType.get(CounterEnumType.ENERGY), 0, 2));

        assertNotNull(event);
        assertEquals(EffectEvent.ENERGY_CHANGE, event.type());
        assertEquals(2, event.params().get("delta"));
        assertEquals(List.of("P" + player.getId()), event.subjects());
    }

    /** A counter with no vocabulary entry is not guessed at. */
    @Test
    void aPlayerCounterWithNoEventTypeIsSkipped() {
        assertNull(BusEvents.playerCounter(
                new GameEventPlayerCounters(
                        player, CounterType.get(CounterEnumType.EXPERIENCE), 0, 1)));
    }

    @Test
    void daytimeBecomesADayNightEvent() {
        EffectEvent event = BusEvents.dayTime(new GameEventDayTimeChanged(false));

        assertEquals(EffectEvent.DAY_NIGHT_CHANGED, event.type());
        assertEquals("night", event.params().get("to"));
    }

    @Test
    void speedAndRadiationCarryTheirDelta() {
        assertEquals(2, BusEvents.speed(
                new GameEventSpeedChanged(player, 1, 3)).params().get("delta"));
        assertEquals(3, BusEvents.radiation(
                new GameEventPlayerRadiation(player, player, 3)).params().get("delta"));
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `mvn -B -o test -Dtest=BusEventsTest` in `forge-connector`
Expected: FAIL to compile — `cannot find symbol: method playerCounter`.

- [ ] **Step 3: Implement the constants and factories**

In `EffectEvent.java`, beside the existing constants:

```java
    public static final String ENERGY_CHANGE = "energy_change";
```

(`DAY_NIGHT_CHANGED`, `RADIATION_CHANGE` and `SPEED_CHANGED` already exist and have never been used.)

In `BusEvents.java`:

```java
    /**
     * A player counter, for the one kind the vocabulary names.
     *
     * <p>Energy, and only energy: poison has its own bus event and its own
     * factory, and a counter this has no reading for is skipped rather than
     * guessed at — an event type invented here is one no reader accepts.
     */
    static EffectEvent playerCounter(GameEventPlayerCounters event) {
        String type = event.type().getName().toUpperCase(Locale.ROOT);
        if (!"ENERGY".equals(type)) {
            return null;
        }
        return new EffectEvent(EffectEvent.ENERGY_CHANGE)
                .subject("P" + event.receiver().getId())
                .param("delta", event.amount());
    }

    static EffectEvent radiation(GameEventPlayerRadiation event) {
        return new EffectEvent(EffectEvent.RADIATION_CHANGE)
                .subject("P" + event.receiver().getId())
                .param("delta", event.change());
    }

    static EffectEvent speed(GameEventSpeedChanged event) {
        return new EffectEvent(EffectEvent.SPEED_CHANGED)
                .subject("P" + event.player().getId())
                .param("delta", event.newValue() - event.oldValue());
    }

    /** Day and night are a property of the game, so this event names no subject. */
    static EffectEvent dayTime(GameEventDayTimeChanged event) {
        return new EffectEvent(EffectEvent.DAY_NIGHT_CHANGED)
                .param("to", event.daytime() ? "day" : "night");
    }
```

- [ ] **Step 4: Subscribe them**

In `BusBracketCollector.java`, beside the existing handlers:

```java
    /** Energy, and any other player counter the vocabulary learns to name. */
    @Subscribe
    public void onPlayerCounters(GameEventPlayerCounters event) {
        EffectEvent counter = BusEvents.playerCounter(event);
        if (counter != null) {
            record(counter);
        }
    }

    @Subscribe
    public void onRadiation(GameEventPlayerRadiation event) {
        record(BusEvents.radiation(event));
    }

    @Subscribe
    public void onSpeed(GameEventSpeedChanged event) {
        record(BusEvents.speed(event));
    }

    @Subscribe
    public void onDayTime(GameEventDayTimeChanged event) {
        record(BusEvents.dayTime(event));
    }
```

- [ ] **Step 5: Run the tests**

Run: `mvn -B -o test` in `forge-connector`
Expected: PASS, 638 + 4 or more.

- [ ] **Step 6: Shrink the guard list**

Remove `energy_change`, `radiation_change`, `speed_changed`, `day_night_changed` from `KNOWN_UNEMITTED` (22 remain), and run:
`.venv/Scripts/python.exe -m pytest tests/unit/effects/domain/test_event_schema_completeness.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add forge-connector/src/main/java/com/pricepredictor/connector/effects/EffectEvent.java forge-connector/src/main/java/com/pricepredictor/connector/effects/BusEvents.java forge-connector/src/main/java/com/pricepredictor/connector/effects/BusBracketCollector.java forge-connector/src/test/java/com/pricepredictor/connector/effects/BusEventsTest.java tests/unit/effects/domain/test_event_schema_completeness.py
git commit -m "feat(effects): record energy, radiation, speed and day/night"
```

---

### Task 5: The Forge clause hook

Twenty-two of the remaining types are produced by a named effect API and leave no bus trace. What they do leave is their own `SpellAbility` — its parameters, and for several of them a result the effect remembers on the host. The collector needs to be told when a clause resolves.

The hook goes where the sub-ability pointer already lives, and fires **twice**: before the clause and after it. Both are needed — `damage_healed` is the proof, because `HealDamageEffect` calls `gameCard.healDamage()` and an observer that only sees the aftermath finds the damage already zero.

**Files:**
- Modify: `forge-game/src/main/java/forge/game/ability/AbilityUtils.java:1423-1431` (the `resolveApiAbility` wrapper)
- Create: `forge-game/src/test/java/forge/game/ability/EffectRecordClauseHookTest.java`

**Interfaces:**
- Produces: `AbilityUtils.EffectRecordClauseListener` with `onClauseResolving(SpellAbility clause)` and `onClauseResolved(SpellAbility clause, boolean threw)`; installed by `AbilityUtils.setEffectRecordClauseListener(listener)`.

- [ ] **Step 1: Write the failing test**

Create `forge-game/src/test/java/forge/game/ability/EffectRecordClauseHookTest.java` (TestNG — this repo does not use JUnit):

```java
package forge.game.ability;

import org.testng.AssertJUnit;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.List;

/**
 * The clause hook, and its two promises: it reports every clause that resolves,
 * before and after, and it changes nothing when nobody is listening.
 *
 * <p>Both halves of the notification are needed. An effect that heals damage
 * calls {@code healDamage()} and leaves nothing to read afterwards, so an
 * observer that only sees the aftermath records that a card was healed of zero.
 */
public class EffectRecordClauseHookTest {

    @AfterMethod
    public void removeListener() {
        AbilityUtils.setEffectRecordClauseListener(null);
    }

    @Test
    public void testBothHalvesArrivedInOrder() {
        final List<String> seen = new ArrayList<>();
        AbilityUtils.setEffectRecordClauseListener(
                new AbilityUtils.EffectRecordClauseListener() {
                    @Override
                    public void onClauseResolving(forge.game.spellability.SpellAbility clause) {
                        seen.add("before");
                    }

                    @Override
                    public void onClauseResolved(
                            forge.game.spellability.SpellAbility clause, boolean threw) {
                        seen.add(threw ? "threw" : "after");
                    }
                });

        AbilityUtils.notifyClauseForTest(null, false);

        AssertJUnit.assertEquals(List.of("before", "after"), seen);
    }

    @Test
    public void testAThrownClauseIsStillReported() {
        final List<String> seen = new ArrayList<>();
        AbilityUtils.setEffectRecordClauseListener(
                new AbilityUtils.EffectRecordClauseListener() {
                    @Override
                    public void onClauseResolving(forge.game.spellability.SpellAbility clause) {
                        seen.add("before");
                    }

                    @Override
                    public void onClauseResolved(
                            forge.game.spellability.SpellAbility clause, boolean threw) {
                        seen.add(threw ? "threw" : "after");
                    }
                });

        AbilityUtils.notifyClauseForTest(null, true);

        AssertJUnit.assertEquals(List.of("before", "threw"), seen);
    }

    @Test
    public void testNothingHappensWithoutAListener() {
        AbilityUtils.notifyClauseForTest(null, false);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `mvn -B test -pl forge-game -am -Dtest=EffectRecordClauseHookTest -Dsurefire.failIfNoSpecifiedTests=false` in the forge checkout.
Expected: FAIL to compile — `cannot find symbol: class EffectRecordClauseListener`.

Note: a bare `-pl` without `-am` cannot resolve the parent pom's `${revision}`.

- [ ] **Step 3: Implement the hook**

In `AbilityUtils.java`, beside the existing sub-ability pointer (around line 1300):

```java
    /**
     * Notified as each clause of an ability resolves, before and after.
     *
     * <p>Both halves, because an effect's own record of what it did is often
     * gone by the time it returns: {@code HealDamageEffect} calls
     * {@code healDamage()} and leaves zero behind, and an observer with only the
     * aftermath would record every heal as a heal of nothing.
     *
     * <p>The listener must not keep the ability. It is the one resolving, and
     * the engine goes on mutating its targets and its parent the moment this
     * returns.
     */
    public interface EffectRecordClauseListener {
        void onClauseResolving(SpellAbility clause);

        void onClauseResolved(SpellAbility clause, boolean threw);
    }

    private static EffectRecordClauseListener effectRecordClauseListener;

    public static void setEffectRecordClauseListener(
            final EffectRecordClauseListener listener) {
        effectRecordClauseListener = listener;
    }

    /** Package-private seam for the hook's own test; the engine never calls it. */
    static void notifyClauseForTest(final SpellAbility clause, final boolean threw) {
        final EffectRecordClauseListener listener = effectRecordClauseListener;
        if (listener != null) {
            listener.onClauseResolving(clause);
            listener.onClauseResolved(clause, threw);
        }
    }
```

Then extend the existing wrapper — the pointer save/restore stays exactly as it is:

```java
    private static void resolveApiAbility(final SpellAbility sa, final Game game) {
        final SpellAbility previous = EFFECT_RECORD_SUB_ABILITY.get();
        setEffectRecordSubAbility(sa);
        final EffectRecordClauseListener listener = effectRecordClauseListener;
        if (listener != null) {
            listener.onClauseResolving(sa);
        }
        boolean threw = true;
        try {
            resolveApiAbilityBody(sa, game);
            threw = false;
        } finally {
            setEffectRecordSubAbility(previous);
            if (listener != null) {
                listener.onClauseResolved(sa, threw);
            }
        }
    }
```

- [ ] **Step 4: Run the test**

Run: `mvn -B test -pl forge-game -am -Dtest=EffectRecordClauseHookTest -Dsurefire.failIfNoSpecifiedTests=false`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit (in the forge repository)**

```bash
cd ../forge
git add forge-game/src/main/java/forge/game/ability/AbilityUtils.java forge-game/src/test/java/forge/game/ability/EffectRecordClauseHookTest.java
git commit -m "Notify a listener as each clause of an ability resolves"
```

---

### Task 6: The emitter table, and the APIs that only need their own parameters

The mechanism plus the six APIs whose event is fully described by parameters the `SpellAbility` carries — a phase name, a turn count, an X. Everything whose value the effect computes belongs to Task 7 instead, per the accuracy constraint: `MakeCard`, `Detain`, `Goad`, `MustBlock` and `Fog` were all here in an earlier draft and moved. Keyed by API *name* rather than the `ApiType` enum, because `PatchedCollectors` already reads the API as an `Object` and because a Forge rename should degrade to a missing event rather than a compile break. **The name is `ApiType.name()` and nothing else** — see the key-space constraint above; a rule keyed by an effect class name matches nothing and fails silently, which is how 35 of the current mapping's keys are written.

`Fog` is deliberately absent from this table. `FogEffect` prevents nothing when it resolves: it installs a static "prevent all combat damage this turn" effect, and the prevention happens later, per damage event. Recording `damage_prevented` at its resolution would describe an intention. The actual prevention is recorded at the engine's choke point in Task 7, which catches every source of it including this one.

**Files:**
- Create: `forge-connector/src/main/java/com/pricepredictor/connector/effects/ApiEvents.java`
- Modify: `.../effects/EffectEvent.java` (constants)
- Modify: `.../effects/BusBracketCollector.java` (the recording door)
- Modify: `.../effects/PatchedCollectors.java` (install the hook)
- Modify: `.../effects/PatchHooks.java` (the roster)
- Create: `forge-connector/src/test/java/com/pricepredictor/connector/effects/ApiEventsTest.java`

**Interfaces:**
- Consumes: `AbilityUtils.EffectRecordClauseListener` (Task 5), reflectively via `PatchHooks.install`.
- Produces: `ApiEvents.before(SpellAbility)` returning an opaque `Object` memo or null; `ApiEvents.after(SpellAbility, Object memo)` returning `EffectEvent` or null; `BusBracketCollector.recordClauseEvent(EffectEvent)`.

- [ ] **Step 1: Write the failing test**

Create `ApiEventsTest.java`:

```java
package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * An effect whose outcome the engine broadcasts nowhere still describes itself:
 * its own parameters say what it did. These are the APIs that need nothing else.
 */
@ExtendWith(ForgeExtension.class)
class ApiEventsTest {

    @Test
    void anExtraTurnCarriesItsCount() {
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertNotNull(event);
        assertEquals(EffectEvent.TURN_ADDED, event.type());
        assertEquals(1, event.params().get("count"));
    }

    @Test
    void aSkippedPhaseNamesThePhase() {
        SpellAbility sa = TestCards.scriptedAbility("Blinding Angel", "SkipPhase");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.PHASE_SKIPPED, event.type());
        assertNotNull(event.params().get("phase"));
    }

    @Test
    void anApiWithNoEmitterIsSilent() {
        SpellAbility sa = TestCards.scriptedAbility("Lightning Bolt", "DealDamage");

        assertNull(ApiEvents.after(sa, ApiEvents.before(sa)),
                "damage already arrives on the bus; a second event would double-count it");
    }
}
```

Add to `TestCards.java`:

```java
    /** The first ability on a card whose API matches, for the emitter tests. */
    static SpellAbility scriptedAbility(String cardName, String api) {
        Card card = build(cardName);
        for (SpellAbility sa : card.getCurrentState().getSpellAbilities()) {
            if (sa.getApi() != null && api.equals(sa.getApi().name())) {
                return sa;
            }
            for (SpellAbility sub = sa.getSubAbility(); sub != null; sub = sub.getSubAbility()) {
                if (sub.getApi() != null && api.equals(sub.getApi().name())) {
                    return sub;
                }
            }
        }
        throw new AssertionError("no " + api + " ability on " + cardName);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `mvn -B -o test -Dtest=ApiEventsTest` in `forge-connector`
Expected: FAIL to compile — `cannot find symbol: class ApiEvents`.

- [ ] **Step 3: Implement the table**

Create `ApiEvents.java`:

```java
package com.pricepredictor.connector.effects;

import forge.game.ability.AbilityUtils;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;

import java.util.Map;

/**
 * The event an effect API promises, built from the clause that ran.
 *
 * <p>Most of Forge's effects broadcast nothing. Damage and zone changes reach a
 * collector because the engine fires a bus event for them; an extra turn, a
 * prevented damage, a chosen colour reach nobody, and the corpus recorded those
 * abilities as abilities that did nothing. What every one of them does carry is
 * its own {@code SpellAbility}: the parameters it was scripted with, and for
 * some, a result remembered on the host.
 *
 * <p>Keyed by API name rather than by {@code ApiType} so a Forge rename costs a
 * missing event rather than a compile error, and so this file needs no import
 * of an enum whose members move. The names are pinned by
 * {@code test_every_effect_api_maps_to_an_event} on the Python side.
 *
 * <p>An API absent from this table emits nothing. That is the right default:
 * damage, counters, taps and zone changes already arrive on the bus, and a
 * second event for them would double-count an outcome.
 */
final class ApiEvents {

    private ApiEvents() {
    }

    /** What an emitter needs from before the clause ran, or null when nothing. */
    @FunctionalInterface
    private interface Memo {
        Object take(SpellAbility sa, Card host);
    }

    /** The event, given the clause and whatever its memo captured. */
    @FunctionalInterface
    private interface Emitter {
        EffectEvent emit(SpellAbility sa, Card host, Object memo);
    }

    private record Rule(Memo memo, Emitter emitter) {
    }

    private static final Map<String, Rule> RULES = Map.ofEntries(
            Map.entry("AddTurn", new Rule(null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.TURN_ADDED)
                            .subject(playerOf(sa))
                            .param("count", amount(sa, host, "NumTurns", "1")))),
            Map.entry("SkipTurn", new Rule(null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.TURN_SKIPPED)
                            .subject(playerOf(sa))
                            .param("count", amount(sa, host, "NumTurns", "1")))),
            Map.entry("AddPhase", new Rule(null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.PHASE_ADDED)
                            .subject(playerOf(sa))
                            .param("phase", sa.getParamOrDefault("ExtraPhase", "?"))
                            .param("count", amount(sa, host, "NumPhases", "1")))),
            Map.entry("SkipPhase", new Rule(null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.PHASE_SKIPPED)
                            .subject(playerOf(sa))
                            .param("phase", sa.getParamOrDefault("Phase", "?")))),
            Map.entry("ChangeX", new Rule(null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.X_CHANGED)
                            .param("value", amount(sa, host, "Value", "0")))),
            Map.entry("GainOwnership", new Rule(null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.OWNERSHIP_CHANGE)
                            .param("owner", playerOf(sa))))
    );

    /** What the emitter for this clause needs from before it runs, if anything. */
    static Object before(SpellAbility sa) {
        Rule rule = ruleFor(sa);
        if (rule == null || rule.memo() == null) {
            return null;
        }
        return rule.memo().take(sa, sa.getHostCard());
    }

    /** The event this clause promises, or null when its API promises none. */
    static EffectEvent after(SpellAbility sa, Object memo) {
        Rule rule = ruleFor(sa);
        if (rule == null) {
            return null;
        }
        try {
            return rule.emitter().emit(sa, sa.getHostCard(), memo);
        } catch (RuntimeException e) {
            // An emitter reads engine state that a fizzled or redirected clause
            // may have left in a shape it did not expect. A missing event is a
            // gap; a thrown one would end the resolution the observer is only
            // watching.
            return null;
        }
    }

    private static Rule ruleFor(SpellAbility sa) {
        if (sa == null || sa.getApi() == null) {
            return null;
        }
        return RULES.get(sa.getApi().name());
    }

    private static int amount(SpellAbility sa, Card host, String key, String fallback) {
        return AbilityUtils.calculateAmount(host, sa.getParamOrDefault(key, fallback), sa);
    }

    private static String playerOf(SpellAbility sa) {
        return sa.getActivatingPlayer() == null
                ? null : SnapshotBuilder.playerId(sa.getActivatingPlayer());
    }
}
```

Add the missing constants to `EffectEvent.java`:

```java
    public static final String TURN_ADDED = "turn_added";
    public static final String TURN_SKIPPED = "turn_skipped";
    public static final String PHASE_ADDED = "phase_added";
    public static final String PHASE_SKIPPED = "phase_skipped";
    public static final String X_CHANGED = "x_changed";
    public static final String OWNERSHIP_CHANGE = "ownership_change";
    public static final String CARD_MADE = "card_made";
    public static final String RESTRICTION_CHANGE = "restriction_change";
```

(`DAMAGE_PREVENTED` already exists.)

- [ ] **Step 4: Run the emitter test**

Run: `mvn -B -o test -Dtest=ApiEventsTest` in `forge-connector`
Expected: PASS, 3 tests.

- [ ] **Step 5: Wire the table to the bracket**

In `BusBracketCollector.java`, beside the private `record`:

```java
    /**
     * An event an effect API described rather than the bus announced.
     *
     * <p>Package-private because {@code PatchedCollectors} owns the clause hook
     * and this collector owns the open bracket: the event belongs to whatever
     * resolution is in flight, which only this class knows.
     */
    void recordClauseEvent(EffectEvent event) {
        if (event != null) {
            record(event);
        }
    }
```

In `PatchedCollectors.java`, install the hook beside the others:

```java
        if (PatchHooks.install(
                PatchHooks.ABILITY_UTILS, "setEffectRecordClauseListener",
                clauseHandler())) {
            installed.add("clause-events");
        }
```

and the handler:

```java
    /**
     * The events an effect API promises, taken from the clause that ran.
     *
     * <p>The memo is kept per thread and not per collector: a clause can resolve
     * another, and the engine runs games on more than one thread.
     */
    private final ThreadLocal<Deque<Object>> clauseMemos =
            ThreadLocal.withInitial(ArrayDeque::new);

    private InvocationHandler clauseHandler() {
        return (proxy, method, args) -> {
            if (args == null || args.length == 0
                    || !(args[0] instanceof SpellAbility clause)) {
                return null;
            }
            if ("onClauseResolving".equals(method.getName())) {
                clauseMemos.get().push(
                        java.util.Optional.ofNullable(ApiEvents.before(clause)));
                return null;
            }
            if (!"onClauseResolved".equals(method.getName())) {
                return null;
            }
            Deque<Object> memos = clauseMemos.get();
            Object memo = memos.isEmpty() ? null : memos.pop();
            boolean threw = args.length > 1 && Boolean.TRUE.equals(args[1]);
            if (threw) {
                // A clause that threw did not finish, and an event claiming it
                // did would be a fact the game never contained.
                return null;
            }
            Object value = memo instanceof java.util.Optional<?> opt
                    ? opt.orElse(null) : memo;
            bracket.recordClauseEvent(ApiEvents.after(clause, value));
            return null;
        };
    }
```

In `PatchHooks.java`, add the roster entry beside the others:

```java
            new Hook(ABILITY_UTILS, "setEffectRecordClauseListener",
                    "the events an effect API describes but the bus never announces"),
```

- [ ] **Step 6: Run the whole connector suite**

Run: `mvn -B -o test` in `forge-connector`
Expected: PASS, 638 + 7 or more.

- [ ] **Step 7: Shrink the guard list and commit**

Remove `turn_added`, `turn_skipped`, `phase_added`, `phase_skipped`, `x_changed` and `ownership_change` from `KNOWN_UNEMITTED` (16 remain). `card_made`, `restriction_change` and `damage_prevented` are **not** removed here — they are Task 7's, because what a `MakeCard` made, which creatures a `Goad` actually landed on, and how much damage was really prevented are known to the effect and to the engine's prevention path, not to the parameters.

```bash
git add forge-connector/src/main/java/com/pricepredictor/connector/effects forge-connector/src/test/java/com/pricepredictor/connector/effects tests/unit/effects/domain/test_event_schema_completeness.py
git commit -m "feat(effects): an effect API's own parameters describe what it did"
```

---

### Task 7: The outcome hook, for results an effect computes and discards

Six APIs cannot be read from the outside at all, and the first draft of this plan assumed they could. Verified in the engine:

- `FlipCoinEffect:107-118` remembers the winners only when the script declares a `WinSubAbility`, and puts the earlier remembered objects back before the clause ends (`removeRemembered(wonFor); addRemembered(tempRemembered)`); `:154` remembers a count only under `RememberNumber`.
- `ClashEffect:82` remembers the *opponent*, only under `RememberClasher`. Who won is the method's return value and is never written anywhere.
- `VoteEffect:138-140` remembers each voter and removes it again inside the loop; `:156` and `:179` are guarded by `VoteSubAbility` and `RememberVotedObjects`, and `:175-177` calls `clearRemembered()`. The tally is a local `ListMultimap`.
- `TwoPilesEffect:152-167` is guarded by `RememberChosen` and restores the previous list; `MultiplePilesEffect:101` calls `clearRemembered()`.
- `CopySpellAbilityEffect:206` is guarded by `RememberCopies`.

So a before/after diff of the host's remembered list finds nothing for the five richest outcomes on any card that did not opt in, which is most of them. The result exists for a few statements inside the effect and is then gone. The only way to record it is to ask at that moment.

One hook serves all six, and any future effect: a static notification an effect calls with its own account of what it did. It is one interface rather than six because the connector then has one listener to install and one shape to convert, and because an effect-local interface per outcome would be six things to keep inert.

**Files:**
- Create: `forge-game/src/main/java/forge/game/ability/EffectRecordOutcomes.java`
- Modify: `forge-game/src/main/java/forge/game/ability/effects/FlipCoinEffect.java`
- Modify: `forge-game/src/main/java/forge/game/ability/effects/ClashEffect.java`
- Modify: `forge-game/src/main/java/forge/game/ability/effects/VoteEffect.java`
- Modify: `forge-game/src/main/java/forge/game/ability/effects/TwoPilesEffect.java`
- Modify: `forge-game/src/main/java/forge/game/ability/effects/MultiplePilesEffect.java`
- Modify: `forge-game/src/main/java/forge/game/ability/effects/CopySpellAbilityEffect.java`
- Modify: `forge-game/src/main/java/forge/game/ability/effects/VentureEffect.java`
- Modify: `forge-game/src/main/java/forge/game/ability/effects/{CopyPermanent,Clone,MakeCard,Detain,Goad,MustBlock}Effect.java`
- Modify: `forge-game/src/main/java/forge/game/GameAction.java` — the reveal choke point
- Modify: `forge-game/src/main/java/forge/game/GameEntity.java` — the damage-prevention choke point
- Create: `forge-game/src/test/java/forge/game/ability/EffectRecordOutcomeHookTest.java`
- Modify: `.../connector/effects/PatchedCollectors.java`, `.../effects/EffectEvent.java`, `.../effects/PatchHooks.java`
- Test: `.../connector/effects/PatchedCollectorTest.java`

**Interfaces:**
- Produces: `EffectRecordOutcomes.setEffectRecordOutcomeListener(EffectRecordOutcomeListener)`, `EffectRecordOutcomes.isObserved()`, `EffectRecordOutcomes.note(SpellAbility, String, Map<String, Object>)`. The listener is `void onOutcome(SpellAbility sa, String eventType, Map<String, Object> params)` — **argument order is a contract**, because the connector binds it reflectively by position.
- Consumes: `BusBracketCollector.recordClauseEvent(EffectEvent)` (Task 6).

- [ ] **Step 1: Write the failing Forge test**

Create `forge-game/src/test/java/forge/game/ability/EffectRecordOutcomeHookTest.java` (TestNG — this repo does not use JUnit):

```java
package forge.game.ability;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.testng.AssertJUnit;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.Test;

import forge.game.spellability.SpellAbility;

/**
 * The outcome hook's two promises: it reports what an effect computed, and it
 * costs nothing when nobody is listening.
 *
 * <p>The values these effects produce live for a few statements and are then
 * discarded — a coin's result, a clash's winner, a vote's tally — so an
 * observer that reads the host afterwards finds nothing. This is the only
 * point at which they exist.
 */
public class EffectRecordOutcomeHookTest {

    @AfterMethod
    public void clearListener() {
        EffectRecordOutcomes.setEffectRecordOutcomeListener(null);
    }

    @Test
    public void testTheListenerHearsWhatTheEffectReports() {
        final List<String> heard = new ArrayList<>();
        EffectRecordOutcomes.setEffectRecordOutcomeListener(
                (sa, type, params) -> heard.add(type + params));

        EffectRecordOutcomes.note(null, "coin_flipped",
                Map.of("results", List.of("win", "loss")));

        AssertJUnit.assertEquals(1, heard.size());
        AssertJUnit.assertTrue(heard.get(0).startsWith("coin_flipped"));
    }

    @Test
    public void testNothingIsObservedWithoutAListener() {
        AssertJUnit.assertFalse(EffectRecordOutcomes.isObserved());
        EffectRecordOutcomes.note(null, "coin_flipped", Map.of());
    }

    @Test
    public void testAListenerThatThrowsDoesNotEscape() {
        EffectRecordOutcomes.setEffectRecordOutcomeListener((sa, type, params) -> {
            throw new IllegalStateException("observer failure");
        });

        // An observer's failure must not end the resolution it is watching.
        EffectRecordOutcomes.note(null, "coin_flipped", Map.of());
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `mvn -B test -pl forge-game -am -Dtest=EffectRecordOutcomeHookTest -Dsurefire.failIfNoSpecifiedTests=false` in `../forge`
Expected: FAIL — `cannot find symbol: class EffectRecordOutcomes`.

- [ ] **Step 3: Write the hook**

Create `forge-game/src/main/java/forge/game/ability/EffectRecordOutcomes.java`:

```java
package forge.game.ability;

import java.util.Map;

import forge.game.spellability.SpellAbility;

/**
 * What an effect computed and is about to forget.
 *
 * <p>Most of what an effect does reaches an observer some other way: damage,
 * zone changes and counters are on the event bus, and an effect's parameters
 * say what it was scripted to do. A handful of outcomes are neither. A coin's
 * result, a clash's winner, a vote's tally and a pile's contents are computed,
 * used by the same clause, and dropped — Forge remembers them on the host only
 * when the card's script asks it to, and puts the previous remembered objects
 * back when the clause ends. An observer looking afterwards sees nothing.
 *
 * <p>So the effect reports them itself, at the moment they exist. One
 * notification for all of them rather than one interface per outcome: the
 * caller names the event type, which keeps the vocabulary in the collector
 * that consumes it rather than spread across the effects.
 *
 * <p>Inert unless something installs a listener, and nothing in Forge does.
 * {@link #isObserved()} exists so an effect can skip assembling parameters
 * that nobody will read.
 */
public final class EffectRecordOutcomes {

    private EffectRecordOutcomes() {
    }

    /** Notified with one clause's own account of what it did. */
    public interface EffectRecordOutcomeListener {
        void onOutcome(SpellAbility sa, String eventType, Map<String, Object> params);
    }

    private static EffectRecordOutcomeListener effectRecordOutcomeListener;

    public static void setEffectRecordOutcomeListener(
            final EffectRecordOutcomeListener listener) {
        effectRecordOutcomeListener = listener;
    }

    /** Whether anything is listening, so an effect can skip the work if not. */
    public static boolean isObserved() {
        return effectRecordOutcomeListener != null;
    }

    /**
     * Report an outcome, if anyone is listening.
     *
     * <p>An observer's failure is swallowed for the reason the engine swallows
     * nothing else: this call sits in the middle of a resolution that must
     * finish, and a recorder is not worth a game.
     */
    public static void note(final SpellAbility sa, final String eventType,
                            final Map<String, Object> params) {
        final EffectRecordOutcomeListener listener = effectRecordOutcomeListener;
        if (listener == null) {
            return;
        }
        try {
            listener.onOutcome(sa, eventType, params);
        } catch (RuntimeException ignored) {
            // Deliberately swallowed; see the javadoc.
        }
    }
}
```

- [ ] **Step 4: Run the hook test**

Run: `mvn -B test -pl forge-game -am -Dtest=EffectRecordOutcomeHookTest -Dsurefire.failIfNoSpecifiedTests=false`
Expected: PASS, 3 tests.

- [ ] **Step 5: Add the six call sites**

Each is guarded by `isObserved()` so an unlistened engine assembles nothing. Add `import forge.game.ability.EffectRecordOutcomes;` where the class is outside that package, plus `java.util.Map` / `java.util.List` as needed.

`FlipCoinEffect.resolve`, immediately after the loop that fills `countWins` and `countLosses` and before `if (countWins > 0)`:

```java
        if (EffectRecordOutcomes.isObserved()) {
            final List<String> results = new ArrayList<>();
            for (int i = 0; i < countWins; i++) {
                results.add("win");
            }
            for (int i = 0; i < countLosses; i++) {
                results.add("loss");
            }
            EffectRecordOutcomes.note(sa, "coin_flipped", Map.of("results", results));
        }
```

`ClashEffect`, immediately before the `return winner;` that ends the clash (the winner is null on a tie):

```java
        if (EffectRecordOutcomes.isObserved()) {
            EffectRecordOutcomes.note(sa, "clash_resolved",
                    Map.of("won", winner == player));
        }
```

`VoteEffect`, immediately after `runParams.put(AbilityKey.AllVotes, votes);`, where the tally is complete:

```java
        if (EffectRecordOutcomes.isObserved()) {
            final Map<String, Object> tally = new LinkedHashMap<>();
            for (final Map.Entry<Object, Collection<Player>> entry : votes.asMap().entrySet()) {
                tally.put(String.valueOf(entry.getKey()), entry.getValue().size());
            }
            EffectRecordOutcomes.note(sa, "vote_taken",
                    Map.of("options", new ArrayList<>(tally.keySet()), "tally", tally));
        }
```

`TwoPilesEffect`, immediately after the chosen pile is decided (where `chosenPile` is in scope, beside the `RememberChosen` block):

```java
        if (EffectRecordOutcomes.isObserved()) {
            EffectRecordOutcomes.note(sa, "piles_made",
                    Map.of("piles", 2, "chosen", chosenPile.size()));
        }
```

`MultiplePilesEffect`, immediately after `chosen` is decided and before the `clearRemembered()`:

```java
        if (EffectRecordOutcomes.isObserved()) {
            EffectRecordOutcomes.note(sa, "piles_made",
                    Map.of("piles", piles, "chosen", chosen.size()));
        }
```

`CopySpellAbilityEffect`, immediately after `copies` is built (beside the `RememberCopies` block):

```java
        if (EffectRecordOutcomes.isObserved()) {
            EffectRecordOutcomes.note(sa, "spell_copied",
                    Map.of("count", copies.size(),
                            "new_targets", sa.hasParam("MayChooseTarget")));
        }
```

`VentureEffect`, immediately after `dungeon.setCurrentRoom(nextRoom)` — the dungeon card and both
room names are in scope there, and `getDungeonCard` is private, so this is the only place outside
the effect's own body where the pair can be read:

```java
        if (EffectRecordOutcomes.isObserved()) {
            EffectRecordOutcomes.note(sa, "dungeon_ventured",
                    Map.of("dungeon", dungeon.getName(), "room", nextRoom));
        }
```

`CopyPermanentEffect` and `CloneEffect`, immediately after the copies are made, reporting the copies
that exist rather than the number the script asked for — a copy can be replaced, countered or fail:

```java
        if (EffectRecordOutcomes.isObserved() && !copies.isEmpty()) {
            EffectRecordOutcomes.note(sa, "permanent_copied",
                    Map.of("copy_source", copies.get(0).getName(), "count", copies.size()));
        }
```

`MakeCardEffect`, immediately after the loop `for (final Card c : cards)` that puts each made card in
its zone, reporting the cards actually made:

```java
        if (EffectRecordOutcomes.isObserved() && !cards.isEmpty()) {
            EffectRecordOutcomes.note(sa, "card_made",
                    Map.of("card_name", cards.get(0).getName(),
                            "to_zone", String.valueOf(zone),
                            "count", cards.size()));
        }
```

`DetainEffect` (after its `for (final Card c : getTargetCards(sa))` loop), `GoadEffect` (after
`for (final Card tgtC : getDefinedCardsOrTargeted(sa))`) and `MustBlockEffect` (after its loop over
`tgtCards`), each reporting the cards the restriction actually landed on. Detain and Goad resolve
their own affected set — `getDefinedCardsOrTargeted` is not the same as the target list — so the
subjects come from the loop rather than from `sa.getTargets()`:

```java
        if (EffectRecordOutcomes.isObserved() && !affected.isEmpty()) {
            EffectRecordOutcomes.note(sa, "restriction_change",
                    Map.of("restriction", "goad",        // "detain" / "must_block" in the others
                            "value", true,
                            "subjects", affected));      // the Card list the loop just walked
        }
```

- [ ] **Step 5b: Hook the two choke points**

Two outcomes are funnelled through one engine method each, and hooking the funnel is both more
accurate and less code than hooking every effect that reaches it. Neither method belongs to an
effect, so both take the same `EffectRecordOutcomes.note` with a null ability — the bracket
attributes them to whatever is resolving, exactly as it does for a bus event.

`GameAction.reveal(CardCollectionView cards, ZoneType zt, Player cardOwner, boolean dontRevealToOwner, String messagePrefix)` — the widest overload, which the other four delegate to, and the one that
carries the zone. Thirty call sites reach it, including the reveals that are costs rather than
effects:

```java
        if (EffectRecordOutcomes.isObserved() && cards != null && !cards.isEmpty()) {
            EffectRecordOutcomes.note(null, "card_revealed",
                    Map.of("count", cards.size(),
                            "from_zone", zt == null ? "?" : zt.name()));
        }
```

`GameEntity.staticDamagePrevention(int damage, int possiblePrevention, Card source, boolean isCombat, Boolean combatDamagePreventedThisTurn)` — every prevention passes through it, and the amount
prevented is what it removed. Report only when it removed something, at the single `return` the
method ends with (capture the result into a local first):

```java
        if (EffectRecordOutcomes.isObserved() && result < damage) {
            EffectRecordOutcomes.note(null, "damage_prevented",
                    Map.of("amount", damage - result,
                            "source", source == null ? "?" : source.getName()));
        }
```

A replacement effect that prevents damage now shows up twice, and both are true: the `rewrite`
record says a replacement returned `prevented`, and this says how much damage that removed. Note the
relationship in the schema contract so a reader does not treat one as a duplicate of the other.

- [ ] **Step 6: Run the Forge suite for the touched modules**

Run: `mvn -B test -pl forge-game -am -Dsurefire.failIfNoSpecifiedTests=false`
Expected: PASS. The six effects are behaviour-unchanged; only the new test is new.

- [ ] **Step 7: Install the listener in the connector**

In `PatchHooks.java`, add the class constant beside the others and the hook to `REQUIRED`:

```java
    static final String EFFECT_RECORD_OUTCOMES = "forge.game.ability.EffectRecordOutcomes";
```

```java
            new Hook(EFFECT_RECORD_OUTCOMES, "setEffectRecordOutcomeListener",
                    "the outcomes an effect computes and discards: a coin's "
                    + "result, a clash's winner, a vote's tally, a pile's size"),
```

In `PatchedCollectors.java`, install it beside the other listeners and convert:

```java
    /**
     * An effect's own account of what it did.
     *
     * <p>Bound by argument position, because {@code PatchHooks} installs a
     * proxy rather than compiling against the interface: 0 is the ability, 1
     * the event type, 2 the parameters.
     */
    private InvocationHandler outcomeHandler() {
        return (proxy, method, args) -> {
            if (!"onOutcome".equals(method.getName()) || args == null || args.length < 3) {
                return null;
            }
            if (!(args[1] instanceof String type)) {
                return null;
            }
            EffectEvent event = new EffectEvent(type);
            if (args[0] instanceof SpellAbility sa && sa.getHostCard() != null) {
                event.subject(SnapshotBuilder.entityId(sa.getHostCard()));
            }
            if (args[2] instanceof Map<?, ?> params) {
                for (Map.Entry<?, ?> entry : params.entrySet()) {
                    event.param(String.valueOf(entry.getKey()), entry.getValue());
                }
            }
            brackets.recordClauseEvent(event);
            return null;
        };
    }
```

registered alongside the existing installs:

```java
        if (PatchHooks.install(
                PatchHooks.EFFECT_RECORD_OUTCOMES, "setEffectRecordOutcomeListener",
                outcomeHandler())) {
            installed.add("effect-outcome");
        }
```

and uninstalled with the rest:

```java
        PatchHooks.uninstall(
                PatchHooks.EFFECT_RECORD_OUTCOMES, "setEffectRecordOutcomeListener");
```

Add the constants to `EffectEvent.java` (`COIN_FLIPPED` and `SPELL_COPIED` already exist):

```java
    public static final String CLASH_RESOLVED = "clash_resolved";
    public static final String VOTE_TAKEN = "vote_taken";
    public static final String PILES_MADE = "piles_made";
    public static final String DUNGEON_VENTURED = "dungeon_ventured";
```

- [ ] **Step 8: Test the conversion**

Add to `PatchedCollectorTest.java`:

```java
    /**
     * The hook binds by argument position, and no compiler checks that. A
     * changed order in Forge must fail here rather than in a corpus.
     */
    @Test
    void anOutcomeReportBecomesAnEventOfThatType() {
        PatchedCollectors collectors = collectors();
        collectors.outcomeHandlerForTest().invoke(null,
                onOutcomeMethod(),
                new Object[]{null, "vote_taken", Map.of("tally", Map.of("beast", 2))});

        EffectEvent written = collectors.lastClauseEvent();
        assertEquals("vote_taken", written.type());
        assertEquals(Map.of("beast", 2), written.params().get("tally"));
    }
```

Run: `mvn -B -o test` in `forge-connector`
Expected: PASS.

- [ ] **Step 9: Shrink the guard list and commit**

Remove `coin_flipped`, `clash_resolved`, `vote_taken`, `piles_made`, `dungeon_ventured`, `permanent_copied`, `card_made`, `restriction_change` and `card_revealed` from `KNOWN_UNEMITTED` (5 remain: `choice_made`, `damage_healed`, `ability_change`, `continuous_effect_created`, `targets_changed` — all Task 8).

`spell_copied` and `damage_prevented` are **not** in the list and never were: Task 1 measured them as already wired, because their constants are referenced in `PatchedCollectors.java`. They are still this task's work — a constant referenced by code that never runs is why the corpus contains zero of both across 10.1M records — but the proof they now fire is Task 10's measurement, not the guard list. Say so in the commit.

```bash
git -C ../forge add forge-game/src/main/java/forge/game/ability/EffectRecordOutcomes.java forge-game/src/main/java/forge/game/ability/effects forge-game/src/test/java/forge/game/ability/EffectRecordOutcomeHookTest.java
git -C ../forge commit -m "Let an effect report the outcome it computes and discards"
git add forge-connector/src/main/java/com/pricepredictor/connector/effects forge-connector/src/test/java/com/pricepredictor/connector/effects tests/unit/effects/domain/test_event_schema_completeness.py
git commit -m "feat(effects): record the results an effect never writes down"
```


---

### Task 8: Choices, retargets, granted abilities, and healed damage

The last five, and every one of them is readable from outside the effect — which is what separates them from Task 7's. `choice_made` covers fifteen `Choose*` APIs whose result lands in a dedicated accessor on the host and stays there. `damage_healed` needs the before-memo for a value the effect erases. `ability_change` and `continuous_effect_created` describe what an `Animate` or `Effect` clause was scripted to add. `targets_changed` reads another object rather than the host: a spell `ChangeTargets` retargeted is still on the stack afterwards, carrying its new choices, so the record names what those spells point at now.

`permanent_copied` and `card_revealed` are **not** here — they moved to Task 7, because their counts can only be known from the copies and cards that actually exist, and a script parameter would have recorded the intention.

**One limitation, stated rather than hidden.** `ability_change` and `continuous_effect_created` carry the *scripted* text of what an `Animate` or `Effect` clause grants, because what the layer system finally applied is not knowable at the clause. That applied result is already recorded, in the `continuous` record's contributions — so the pair is complete across two record kinds rather than inside one. Say so in the schema contract, so a reader of `ability_change` alone does not mistake it for the applied result.

**Files:**
- Modify: `.../effects/ApiEvents.java`
- Modify: `.../effects/EffectEvent.java`
- Test: `.../effects/ApiEventsTest.java`

- [ ] **Step 1: Write the failing test**

```java
    @Test
    void aChosenColourIsAChoice() {
        SpellAbility sa = TestCards.scriptedAbility("Akroma's Blessing", "ChooseColor");
        Object memo = ApiEvents.before(sa);
        sa.getHostCard().setChosenColors(com.google.common.collect.ImmutableList.of("red"));

        EffectEvent event = ApiEvents.after(sa, memo);

        assertEquals(EffectEvent.CHOICE_MADE, event.type());
        assertEquals("color", event.params().get("choice_kind"));
        assertEquals("red", event.params().get("value"));
    }

    /**
     * HealDamageEffect calls healDamage() and leaves zero behind, so the amount
     * exists only before the clause. This is the case the before-half of the
     * hook was added for.
     */
    @Test
    void healedDamageIsReadFromBeforeTheClause() {
        SpellAbility sa = TestCards.scriptedAbility("Pyramids", "HealDamage");
        Card target = TestCards.build("Grizzly Bears");
        target.setDamage(3);
        sa.resetTargets();
        sa.getTargets().add(target);

        Object memo = ApiEvents.before(sa);
        target.setDamage(0);

        EffectEvent event = ApiEvents.after(sa, memo);

        assertEquals(EffectEvent.DAMAGE_HEALED, event.type());
        assertEquals(3, event.params().get("amount"));
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `mvn -B -o test -Dtest=ApiEventsTest` in `forge-connector`
Expected: FAIL — the two new tests, `null` event.

- [ ] **Step 3: Implement**

In `ApiEvents.java`:

```java
    /** Damage on the clause's targets, before it heals any of it away. */
    private static final Memo DAMAGE_BEFORE = (sa, host) -> {
        int total = 0;
        if (sa.getTargets() != null) {
            for (Card card : sa.getTargets().getTargetCards()) {
                total += card.getDamage();
            }
        }
        return total;
    };

    /** The choice a Choose* clause left on its host, as a kind and a value. */
    private static EffectEvent choice(SpellAbility sa, Card host) {
        if (host == null) {
            return null;
        }
        String kind = null;
        String value = null;
        if (host.getChosenColor() != null) {
            kind = "color";
            value = host.getChosenColor();
        } else if (host.getChosenType() != null && !host.getChosenType().isEmpty()) {
            kind = "type";
            value = host.getChosenType();
        } else if (host.getChosenNumber() != null) {
            kind = "number";
            value = String.valueOf(host.getChosenNumber());
        } else if (host.getNamedCard() != null && !host.getNamedCard().isEmpty()) {
            kind = "card_name";
            value = host.getNamedCard();
        } else if (host.getChosenPlayer() != null) {
            kind = "player";
            value = SnapshotBuilder.playerId(host.getChosenPlayer());
        } else if (host.getChosenCards() != null && !host.getChosenCards().isEmpty()) {
            kind = "cards";
            value = String.valueOf(host.getChosenCards().size());
        }
        if (kind == null) {
            // The clause chose something this has no reading for. An event
            // naming the choice kind "?" would train the head on a distinction
            // the corpus cannot make.
            return null;
        }
        return new EffectEvent(EffectEvent.CHOICE_MADE)
                .param("choice_kind", kind)
                .param("value", value);
    }
```

and the rules, one `Map.entry` per choosing API:

```java
            Map.entry("ChooseColor", new Rule(null, (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseType", new Rule(null, (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseCard", new Rule(null, (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseCardName", new Rule(null, (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseNumber", new Rule(null, (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChoosePlayer", new Rule(null, (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseDirection", new Rule(null, (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseEvenOdd", new Rule(null, (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseSource", new Rule(null, (sa, host, memo) -> choice(sa, host))),
            Map.entry("ChooseGenericEffect", new Rule(null, (sa, host, memo) -> choice(sa, host))),
            Map.entry("HealDamage", new Rule(DAMAGE_BEFORE, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.DAMAGE_HEALED)
                            .param("amount", memo instanceof Integer n ? n : 0))),
            Map.entry("Animate", new Rule(null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.ABILITY_CHANGE)
                            .param("abilities", sa.getParamOrDefault("Abilities", ""))
                            .param("removed", sa.hasParam("RemoveAllAbilities")))),
            Map.entry("AnimateAll", new Rule(null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.ABILITY_CHANGE)
                            .param("abilities", sa.getParamOrDefault("Abilities", ""))
                            .param("removed", sa.hasParam("RemoveAllAbilities")))),
            Map.entry("Effect", new Rule(null, (sa, host, memo) ->
                    new EffectEvent(EffectEvent.CONTINUOUS_EFFECT_CREATED)
                            .param("layers", sa.getParamOrDefault("StaticAbilities", "")))),
            Map.entry("ChangeTargets", new Rule(null, (sa, host, memo) -> retargeted(sa)))
```

with the retarget helper:

```java
    /**
     * What the retargeted spells point at now.
     *
     * <p>Read after the clause rather than reported by it, because a spell it
     * retargeted is still on the stack carrying its new {@code TargetChoices} —
     * this is the one outcome in this group that survives on an object other
     * than the host.
     */
    private static EffectEvent retargeted(SpellAbility sa) {
        List<String> targets = new ArrayList<>();
        if (sa.getTargets() != null) {
            for (SpellAbility changed : sa.getTargets().getTargetSpells()) {
                if (changed.getTargets() == null) {
                    continue;
                }
                for (Card card : changed.getTargets().getTargetCards()) {
                    targets.add(SnapshotBuilder.entityId(card));
                }
            }
        }
        return new EffectEvent(EffectEvent.TARGETS_CHANGED).param("targets", targets);
    }
```

Constants:

```java
    public static final String CHOICE_MADE = "choice_made";
    public static final String DAMAGE_HEALED = "damage_healed";
    public static final String ABILITY_CHANGE = "ability_change";
    public static final String CONTINUOUS_EFFECT_CREATED = "continuous_effect_created";
    public static final String TARGETS_CHANGED = "targets_changed";
```

(`PERMANENT_COPIED` and `CARD_REVEALED` are added in Task 7, with the choke points that fill them.)

- [ ] **Step 4: Run the tests**

Run: `mvn -B -o test` in `forge-connector`
Expected: PASS.

- [ ] **Step 5: Empty the guard list**

Remove the last entries so `KNOWN_UNEMITTED` is `frozenset()`, and run:
`.venv/Scripts/python.exe -m pytest tests/unit/effects/domain/test_event_schema_completeness.py -v`
Expected: PASS with an empty gap — every declared type is now emitted by some path.

- [ ] **Step 6: Commit**

```bash
git add forge-connector/src/main/java/com/pricepredictor/connector/effects forge-connector/src/test/java/com/pricepredictor/connector/effects tests/unit/effects/domain/test_event_schema_completeness.py
git commit -m "feat(effects): record choices, granted abilities and healed damage"
```

---

### Task 9: The continuous record's name channel

`contributions[].name` is declared, is listed in `KNOWN_CONSTANT_FIELDS`, and has never carried a value in any corpus — which is why `name_change` was retired to it in Task 2 rather than wired as an event. The channel has to actually work for that retirement to be honest.

**Files:**
- Modify: `forge-connector/src/main/java/com/pricepredictor/connector/effects/SnapshotBuilder.java`
- Modify: `src/effects/application/field_coverage.py` (drop it from `KNOWN_CONSTANT_FIELDS`)
- Test: `forge-connector/src/test/java/com/pricepredictor/connector/effects/SnapshotBuilderTest.java`

- [ ] **Step 1: Write the failing test**

```java
    /**
     * A continuous effect that renames a permanent contributes that name. The
     * channel is declared in the payload and has never carried a value, which
     * makes "no card was renamed" and "renaming is not collected" the same
     * record.
     */
    @Test
    void aRenamingStaticContributesTheName() {
        Card volrath = card("Volrath's Shapeshifter");
        Card renamed = card("Grizzly Bears");
        renamed.addChangedName("Volrath's Shapeshifter", false,
                volrath.getGame().getNextTimestamp(), 0);

        String json = new SnapshotBuilder(volrath.getGame(), TIERS)
                .contributionJson(renamed, 0);

        assertTrue(json.contains("\"name\":\"Volrath's Shapeshifter\""), json);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `mvn -B -o test -Dtest=SnapshotBuilderTest` in `forge-connector`
Expected: FAIL — the contribution carries `"name":null`.

- [ ] **Step 3: Implement**

In `SnapshotBuilder.java`, where a contribution's channels are written, read the name the static contributed via `card.getChangedCardNames()` keyed by the acting static's timestamp, exactly as the keyword and P/T channels already key by static id, and write it into the `name` slot instead of the literal null.

- [ ] **Step 4: Run the tests**

Run: `mvn -B -o test` in `forge-connector`; then remove `record.payload<ContinuousPayload>.contributions[].name` from `KNOWN_CONSTANT_FIELDS` in `src/effects/application/field_coverage.py` and run `.venv/Scripts/python.exe -m pytest tests/unit/effects -q`.
Expected: PASS both.

- [ ] **Step 5: Commit**

```bash
git add forge-connector/src/main/java/com/pricepredictor/connector/effects/SnapshotBuilder.java forge-connector/src/test/java/com/pricepredictor/connector/effects/SnapshotBuilderTest.java src/effects/application/field_coverage.py
git commit -m "feat(effects): a continuous effect contributes the name it changed"
```

---

### Task 10: Validate, document, and measure

The guard test proves each type is *reachable*. Only a corpus proves it is *reached*.

**Files:**
- Modify: `src/effects/application/validate_corpus.py`
- Modify: `specs/2026-09-05-ability-effect-model.md`, `specs/023-ability-effect-model/contracts/record-schema.md`
- Test: `tests/unit/effects/application/test_validate_corpus.py`

- [ ] **Step 1: Write the failing test**

```python
class TestEventTypeCoverage:
    """Which of the declared event types a corpus actually contains.

    Watched, not judged: a short window legitimately misses rare types, and a
    floor nobody has calibrated would fail every smoke run. What it must not do
    is stay silent — 29 of 60 types were unreachable for the life of the project
    and no report ever said so.
    """

    def test_it_reports_the_types_a_corpus_never_showed(self):
        records = [_record_with_event("zone_change"), _record_with_event("damage_dealt")]

        finding = event_type_coverage(records)

        assert finding.judged is False
        assert "2/" in finding.summary
        assert "energy_change" in finding.detail
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/effects/application/test_validate_corpus.py -k EventTypeCoverage -v`
Expected: FAIL — `ImportError: cannot import name 'event_type_coverage'`.

- [ ] **Step 3: Implement the watched check**

Add `event_type_coverage` to `validate_corpus.py` following the shape of the existing watched checks: count distinct event types over the window, report `seen/declared` with the unseen names sorted, and register it in the check list as watched.

- [ ] **Step 4: Update the contracts**

In the root spec's § Events and in `contracts/record-schema.md`, record what the vocabulary now promises: three types retired to the fields that already carry them (naming `SUPERSEDED_EVENT_TYPES`), the rest reachable, and the two mechanisms — a bus subscription where Forge broadcasts, and the clause hook plus API emitter table where it does not. State plainly that a declared type nothing emits is indistinguishable from a rare one in a corpus, which is why the guard test exists.

- [ ] **Step 5: Build and measure**

With no collection run live:

```bash
cd C:/Users/nicol/IdeaProjects/forge
mvn -B install -DskipTests -pl forge-core,forge-game,forge-ai,forge-gui -am
cd C:/Users/nicol/IdeaProjects/price-predictor/forge-connector
mvn -B test && mvn -B package -DskipTests
cd ..
python -m effects collect-coverage --effect-records %TEMP%/verify-events \
    --decks-per-round 60 --workers 6 --target-records 5 --no-progress-rounds 1
```

Let it collect ~50,000 records, stop it, then run
`python -m effects validate-corpus --effect-records %TEMP%/verify-events` and record which types appeared. Expect the bus four (`energy_change` especially — Aether Hub is the canary) and the parameter-driven APIs; the remembered-result and choice families depend on the decks drawn, so their absence in one coverage run is not a failure.

- [ ] **Step 6: Commit**

```bash
git add src/effects/application/validate_corpus.py tests/unit/effects/application/test_validate_corpus.py specs
git commit -m "feat(effects): report which declared event types a corpus contains"
```

---

## Self-Review

**Spec coverage.** All 29 unobserved types are accounted for: 4 by bus subscription (Task 4), 22 by the clause hook and emitter table (Tasks 5–8), 3 retired with their information located (Task 2). The two drifts found while investigating are covered: the API list (Task 3) and the `contributions[].name` channel (Task 9). The false-assurance gap that hid all of it is closed by the guard test (Task 1) and reported per corpus (Task 10).

**Placeholders.** Every emitter names a real `ApiType` member, verified against `forge-game/src/main/java/forge/game/ability/ApiType.java`; every accessor (`getRemembered`, `getChosenColor`, `getChosenType`, `getChosenNumber`, `getChosenPlayer`, `getChosenCards`, `getNamedCard`, `getCompletedDungeons`, `getDamage`) was checked in `Card.java` / `Player.java`. Two known soft spots, both flagged in their tasks rather than hidden: `dungeon_ventured`'s `room` is written `"?"` because Forge exposes completed dungeons but not the current room, and `Vote`'s `options` reads the script's `VoteType` rather than the engine's tally.

**Card choices in the tests.** Every card named in a test snippet was checked against `forge-gui/res/cardsfolder/` for the API it is used to exercise: Alchemist's Gambit (AddTurn), Blinding Angel (SkipPhase), Aleatory (FlipCoin), Akroma's Blessing (ChooseColor), Pyramids (HealDamage), Lightning Bolt (DealDamage, the negative case). A card added later should be checked the same way — `grep -rlE "(AB|SP|DB)\$ <Api>\b" --include=*.txt .` in the cardsfolder — because `TestCards.scriptedAbility` throws on a card that lacks the API, and a guessed name costs the implementer a build to discover.

**Type consistency.** `ApiEvents.before(SpellAbility)` / `ApiEvents.after(SpellAbility, Object)` are used with those exact signatures in Tasks 6, 7 and 8; `BusBracketCollector.recordClauseEvent(EffectEvent)` is defined in Task 6 and called only there; `KNOWN_UNEMITTED` shrinks 29 → 26 → 22 → 13 → 5 → 0 across Tasks 2, 4, 6, 7 and 8.

**Ordering.** Task 5 (the Forge hook) must precede Tasks 6–8. Task 3 is independent and can run in parallel with Tasks 4–5. Task 9 must not precede Task 2, since Task 2's retirement of `name_change` is only honest once the channel it points at works.
