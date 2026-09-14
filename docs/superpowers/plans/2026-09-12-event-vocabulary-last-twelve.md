# Event Vocabulary — The Last Twelve Types Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the event vocabulary completely — every one of the 78 declared `EventType` members is either emittable by the collector or documented, with measured evidence, as unreachable.

**Architecture:** Six APIs gain an emitter in the connector's `ApiEvents.RULES` table (constants first, since five of the six have no `EffectEvent` constant at all). One type is retired from the vocabulary because the `trigger` record's own `fired` flag already carries it. Three types that *are* wired but whose cards never reach a sealed pool move into `KNOWN_UNEMITTED` with their card lists and the pool evidence. The two AI-capability-gap entries get their reasoning corrected — interventional forks bypass the AI entirely, so the existing explanation is incomplete — and the guard test is tightened so a future undocumented gap fails the suite instead of waiting for a corpus census.

**Tech Stack:** Java 17 (forge-connector, JUnit 5 + `ForgeExtension`), Python 3.14 (`src/effects`, pytest), Forge as a sibling checkout on branch `effect-record-hooks`.

**Spec:** `specs/2026-09-05-ability-effect-model.md` § Events; `specs/023-ability-effect-model/contracts/record-schema.md` § Events. Prior plan: `docs/superpowers/plans/2026-09-10-event-vocabulary-coverage.md` (tasks 1–12, landed — this plan is its remainder).

## Global Constraints

- **`mvn test` only while a collection run is live.** Workers launch from `forge-connector/target/forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar`, loaded at spawn, and the pool recycles a worker every 60 seconds. `mvn package` / `install` / `clean` mid-run hands new workers different code and mixes formats in one corpus. Check for `java.exe` processes before building.
- **Test baselines that must not regress:** 638 Java tests in `forge-connector`, 1,844+ Python tests (`pytest tests/unit/effects tests/unit/sealed`), 252 in `test_event_schema_completeness.py` alone.
- **`-Xmx1200m` and the 60-second worker recycle are deliberate containment** for Forge's late-life misbehaviour. Never change either, and never propose it.
- **Never write under `output/effects/records/` or `output/sealed/`** — those are the user's corpora. `python -m effects collect-coverage --effect-records <tmpdir>` is the contamination-free way to exercise the collector; `match-outcomes` has no output flag and always appends to the sealed corpora.
- A Forge-side change must stay **inert unless a listener is installed**, must not alter engine behaviour, and its test goes in the Forge repo using **TestNG** (the connector uses JUnit 5; do not mix).

---

## Measured Starting Point

Measured against the 6.3 GB / 701-shard corpus `e984ef51-8b0e-4239-9a78-57e1989e2f10` (8,403 matches over 183 sets), scanned 2026-09-12:

- **66 of 78** declared `EventType` members appear in the corpus. **12 do not.**
- Of those 12, **9 are already in `KNOWN_UNEMITTED`** in `tests/unit/effects/domain/test_event_schema_completeness.py:599` with per-type reasons.
- **3 are undocumented**, and all three are *wired*:

  | type | producing API | wired at | why it is absent |
  |---|---|---|---|
  | `ownership_change` | `GainOwnership` | `ApiEvents.java` | 4 cards, all ante: Bronze Tablet, Darkpact, Tempest Efreet, Timmerian Fiends. None appeared in the corpus; sealed pools exclude ante cards. |
  | `radiation_change` | `Radiation`, `InternalRadiation` | `BusEvents.java` (bus path, `onRadiation`) | 22 cards, all Fallout (PIP). **0 of 8,403 matches drew a PIP pool**; PIP is not in the set rotation. |
  | `x_changed` | `ChangeX` | `ApiEvents.java:134` | 2 cards. Unbound Flourishing appeared in 797 snapshots and acted 9 times — **all 9 as `trait=spell idx=0`, the permanent spell cast**. Its triggered ability needs the controller to cast an X spell afterwards, which never happened. Glava, Five-Advents Mage never appeared. |

- The 6 unwired types have **no `ApiEvents.RULES` entry and no outcome-note call**, and five have no `EffectEvent` constant either (only `COMBAT_ENDED` exists).
- `trigger_fired` maps to no Forge effect API at all; the `trigger` record's `fired` flag already carries the same fact.
- `turn_ended` / `text_change` are wired and correct; `KNOWN_UNEMITTED` blames the AI's refusal. That is true for *observation* but incomplete: `PatchedCollectors.collectInterventions()` iterates `actor.getCardsIn(ZoneType.Hand)` and `worthIntervening()` rejects only lands and mana abilities — **an intervention never consults the AI's verdict**. Those 23 cards simply never reached a hand in a 2-per-game intervention budget.

**What "fixed" means here.** Wiring the six closes the *vocabulary* gap: the corpus becomes able to express them. It will not make them appear — Drain Power, Karn Liberated, Aeon Engine, Blacker Lotus and the rest are close to unplayable in AI sealed self-play. The test that proves this work is the static reachability guard, **not** a corpus census. Expect the next run's census to look almost identical, and do not read that as failure.

---

## File Structure

**Connector (Java) — the emitters**
- `forge-connector/src/main/java/com/pricepredictor/connector/effects/EffectEvent.java` — add five `public static final String` constants. One responsibility: the event vocabulary as the writer sees it.
- `forge-connector/src/main/java/com/pricepredictor/connector/effects/ApiEvents.java` — add six `Map.entry(...)` rules to `RULES`. One responsibility: mapping a resolving clause's API to the event it announces.
- `forge-connector/src/test/java/com/pricepredictor/connector/effects/ApiEventsTest.java` — one test per new emitter, in the file that already holds the sibling emitters' tests.

**Python — the vocabulary and its guards**
- `src/effects/domain/event_schema.py` — retire `trigger_fired` from `EventType`.
- `tests/unit/effects/domain/test_event_schema_completeness.py` — `KNOWN_UNEMITTED` gains three entries and loses six; `AI_CAPABILITY_GAP`'s reasons are corrected; one new guard test.
- `tests/unit/effects/infrastructure/test_record_io.py` — a reader test for the retired type, if one references it.

**Docs**
- `specs/023-ability-effect-model/contracts/record-schema.md` — the vocabulary count and the retirement.
- `specs/2026-09-05-ability-effect-model.md` § Events — same.

---

### Task 1: Constants and emitters for the six unwired APIs

**Files:**
- Modify: `forge-connector/src/main/java/com/pricepredictor/connector/effects/EffectEvent.java`
- Modify: `forge-connector/src/main/java/com/pricepredictor/connector/effects/ApiEvents.java` (the `RULES` table, beside the `ChangeX` entry at :134)
- Test: `forge-connector/src/test/java/com/pricepredictor/connector/effects/ApiEventsTest.java`

**Interfaces:**
- Consumes: `ApiEvents.before(SpellAbility)` and `ApiEvents.after(SpellAbility, Object)` — the existing emitter entry points; `EffectEvent.param(String, Object)`; `TestCards.scriptedAbility(String cardName, String apiName)`.
- Produces: `EffectEvent.ABILITY_ACTIVATED`, `.GAME_DRAWN`, `.GAME_RESTARTED`, `.PLAYER_REMOVED`, `.TURN_ORDER_REVERSED` (new `String` constants, values equal to the snake_case type names) plus the six `RULES` entries. Task 3 asserts these exist by name.

**Card evidence for the tests** (read from `forge-gui/res/cardsfolder`, 2026-09-12). `TestCards.scriptedAbility` reaches a root clause and a `SubAbility$` chain hung off one, but **not** a clause that is the `Execute$` of a trigger:

| API | card | clause | reachable by `scriptedAbility`? |
|---|---|---|---|
| `ActivateAbility` | Drain Power | `SP$ ActivateAbility` | yes, root |
| `RestartGame` | Karn Liberated | `AB$ RestartGame` | yes, root |
| `ReverseTurnOrder` | Aeon Engine | `AB$ ReverseTurnOrder` | yes, root |
| `EndCombatPhase` | Mandate of Peace | `DB$ EndCombatPhase` via `SubAbility$` | yes, chain |
| `RemoveFromMatch` | Blacker Lotus | `DB$ RemoveFromMatch` via `SubAbility$` | yes, chain |
| `GameDrawn` | Celestial Convergence | `DB$ GameDrawn` in a trigger's `Execute$` | **no** — see Step 6 |

- [ ] **Step 1: Write the failing tests for the three root-clause APIs**

Add to `ApiEventsTest.java`:

```java
    /** Drain Power: {@code SP$ ActivateAbility}, the API's only root-clause card. */
    @Test
    void anActivatedAbilityAnnouncesItself() {
        SpellAbility sa = TestCards.scriptedAbility("Drain Power", "ActivateAbility");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertNotNull(event);
        assertEquals(EffectEvent.ABILITY_ACTIVATED, event.type());
    }

    /** Karn Liberated's ultimate: {@code AB$ RestartGame}. */
    @Test
    void aRestartedGameAnnouncesItself() {
        SpellAbility sa = TestCards.scriptedAbility("Karn Liberated", "RestartGame");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertNotNull(event);
        assertEquals(EffectEvent.GAME_RESTARTED, event.type());
    }

    /** Aeon Engine: {@code AB$ ReverseTurnOrder | Cost$ T Exile<1/CARDNAME>}. */
    @Test
    void aReversedTurnOrderAnnouncesItself() {
        SpellAbility sa = TestCards.scriptedAbility("Aeon Engine", "ReverseTurnOrder");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertNotNull(event);
        assertEquals(EffectEvent.TURN_ORDER_REVERSED, event.type());
    }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd forge-connector && mvn -B -o test -Dtest=ApiEventsTest`
Expected: FAIL — `cannot find symbol: variable ABILITY_ACTIVATED` (compile error). That is the failure this step wants; five of the six constants do not exist yet.

- [ ] **Step 3: Add the five missing constants**

In `EffectEvent.java`, beside the existing type constants (`COMBAT_ENDED` is already there):

```java
    public static final String ABILITY_ACTIVATED = "ability_activated";
    public static final String GAME_DRAWN = "game_drawn";
    public static final String GAME_RESTARTED = "game_restarted";
    public static final String PLAYER_REMOVED = "player_removed";
    public static final String TURN_ORDER_REVERSED = "turn_order_reversed";
```

- [ ] **Step 4: Add the six rules**

In `ApiEvents.java`, in the `RULES` table beside `Map.entry("ChangeX", ...)` at :134. None of these six declares an `EVENT_PARAMS` row, so each emits its type and its subjects and carries no parameters — do not invent any:

```java
            Map.entry("ActivateAbility", new Rule(EffectEvent.ABILITY_ACTIVATED, null,
                    (sa, host, memo) -> new EffectEvent(EffectEvent.ABILITY_ACTIVATED))),
            Map.entry("EndCombatPhase", new Rule(EffectEvent.COMBAT_ENDED, null,
                    (sa, host, memo) -> new EffectEvent(EffectEvent.COMBAT_ENDED))),
            Map.entry("GameDrawn", new Rule(EffectEvent.GAME_DRAWN, null,
                    (sa, host, memo) -> new EffectEvent(EffectEvent.GAME_DRAWN))),
            Map.entry("RestartGame", new Rule(EffectEvent.GAME_RESTARTED, null,
                    (sa, host, memo) -> new EffectEvent(EffectEvent.GAME_RESTARTED))),
            Map.entry("RemoveFromMatch", new Rule(EffectEvent.PLAYER_REMOVED, null,
                    (sa, host, memo) -> new EffectEvent(EffectEvent.PLAYER_REMOVED))),
            Map.entry("ReverseTurnOrder", new Rule(EffectEvent.TURN_ORDER_REVERSED, null,
                    (sa, host, memo) -> new EffectEvent(EffectEvent.TURN_ORDER_REVERSED))),
```

- [ ] **Step 5: Run the three tests to verify they pass**

Run: `cd forge-connector && mvn -B -o test -Dtest=ApiEventsTest`
Expected: PASS, and the class's existing tests still pass.

- [ ] **Step 6: Add the two SubAbility-chain tests, and prove the GameDrawn case**

`EndCombatPhase` and `RemoveFromMatch` reach their clause through a `SubAbility$` chain, which `scriptedAbility` follows — the sibling tests for Meditate (`SP$ Draw | SubAbility$ DBSkip`) and Full Throttle already rely on that. Add:

```java
    /** Mandate of Peace: {@code DB$ EndCombatPhase} down a {@code SubAbility$} chain. */
    @Test
    void anEndedCombatPhaseAnnouncesItself() {
        SpellAbility sa = TestCards.scriptedAbility("Mandate of Peace", "EndCombatPhase");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertNotNull(event);
        assertEquals(EffectEvent.COMBAT_ENDED, event.type());
    }

    /** Blacker Lotus: {@code DB$ RemoveFromMatch | Defined$ Self}. */
    @Test
    void aRemovedCardAnnouncesItself() {
        SpellAbility sa = TestCards.scriptedAbility("Blacker Lotus", "RemoveFromMatch");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertNotNull(event);
        assertEquals(EffectEvent.PLAYER_REMOVED, event.type());
    }
```

`GameDrawn` has no root-clause or chain card: both Celestial Convergence and Divine Intervention put it in a trigger's `Execute$` SVar, which `scriptedAbility` cannot reach — the same limitation the Blinding Angel comment in this file already records. Do **not** fake a test with a hand-built `SpellAbility`; the rule is covered by Task 3's static guard instead. Add this comment above the `GameDrawn` rule in `ApiEvents.java` so the next reader does not go looking:

```java
            // GameDrawn's two cards (Celestial Convergence, Divine Intervention)
            // both bury it in a trigger's Execute$ SVar, which
            // TestCards.scriptedAbility cannot reach -- so this rule has no
            // emitter test of its own and is covered by the static reachability
            // guard in test_event_schema_completeness.py.
```

- [ ] **Step 7: Run the full connector suite**

Run: `cd forge-connector && mvn -B -o test`
Expected: PASS, >= 638 tests, 0 failures.

- [ ] **Step 8: Commit**

```bash
git add forge-connector/src/main/java/com/pricepredictor/connector/effects/EffectEvent.java \
        forge-connector/src/main/java/com/pricepredictor/connector/effects/ApiEvents.java \
        forge-connector/src/test/java/com/pricepredictor/connector/effects/ApiEventsTest.java
git commit -m "feat(effects): emit the six event types no rule reached"
```

---

### Task 2: Retire `trigger_fired` from the vocabulary

**Files:**
- Modify: `src/effects/domain/event_schema.py` (the `EventType` members)
- Modify: `tests/unit/effects/domain/test_event_schema_completeness.py` (`KNOWN_UNEMITTED`)
- Test: `tests/unit/effects/domain/test_event_schema_completeness.py`

**Interfaces:**
- Consumes: `effects.domain.event_schema.EventType`, `EVENT_PARAMS`.
- Produces: an `EventType` of 77 members. Task 3's guard counts against this.

**Why retire rather than wire:** no Forge effect API maps to `trigger_fired` — it is absent from `EFFECT_API_EVENTS` entirely — and the only thing that could produce it is the trigger-fire hook, which already reports through the `trigger` record kind's own `fired` flag. A second spelling of the same fact is a vocabulary entry that can only ever be dead. Commit `84e1235` retired three types on the same reasoning; follow it.

- [ ] **Step 1: Write the failing test**

In `test_event_schema_completeness.py`:

```python
def test_trigger_fired_is_not_a_declared_type():
    """The trigger record's own `fired` flag carries this, and nothing else can.

    No Forge effect API maps to it, so a declared member could only ever be
    dead vocabulary -- and a dead member is indistinguishable from an unwired
    one to every guard in this file.
    """
    assert "trigger_fired" not in EventType.__args__
    assert "trigger_fired" not in EVENT_PARAMS
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/effects/domain/test_event_schema_completeness.py::test_trigger_fired_is_not_a_declared_type -v`
Expected: FAIL — `assert 'trigger_fired' not in (...)`.

- [ ] **Step 3: Remove the member**

In `src/effects/domain/event_schema.py`, delete `"trigger_fired"` from the `EventType` literal members. Then delete its `KNOWN_UNEMITTED` entry in the test module (the whole `"trigger_fired": "No Forge effect API maps to it at all ..."` item) — a retired type is not an unemitted one.

- [ ] **Step 4: Run the module's tests**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/effects/domain/test_event_schema_completeness.py -q`
Expected: PASS, 252+ tests.

- [ ] **Step 5: Check nothing else referenced it**

Run: `grep -rn "trigger_fired" --include=*.py --include=*.java --include=*.md . | grep -v __pycache__`
Expected: no hits outside this plan and any changelog. If `record_io.py` or a fixture names it, fix that too and re-run `pytest tests/unit/effects`.

- [ ] **Step 6: Commit**

```bash
git add src/effects/domain/event_schema.py tests/unit/effects/domain/test_event_schema_completeness.py
git commit -m "refactor(effects): retire trigger_fired, which the fired flag already carries"
```

---

### Task 3: Document the three reachability-limited types, and make the guard catch the next one

**Files:**
- Modify: `tests/unit/effects/domain/test_event_schema_completeness.py` (`KNOWN_UNEMITTED`, `AI_CAPABILITY_GAP`, one new test)
- Test: same file

**Interfaces:**
- Consumes: `KNOWN_UNEMITTED: dict[str, str]`, `AI_CAPABILITY_GAP: frozenset[str]`, `_emitted_event_types()`.
- Produces: `POOL_UNREACHABLE: frozenset[str]` — the three types that are wired but whose cards no sealed pool deals. Nothing later consumes it; it exists so the reason is checkable and the guard can carve it out the way `AI_CAPABILITY_GAP` already is.

**Why this task matters most:** the six from Task 1 will still not appear in a corpus, and neither will these three. Without a documented, *measured* reason per type, the next person to run a census re-derives all of it. This task is what makes the census answerable by reading rather than by investigating.

- [ ] **Step 1: Write the failing guard test**

```python
#: Wired, correct, and referenced in the connector -- but no sealed pool deals
#: the cards, so no corpus collected this way can contain them. Distinct from
#: AI_CAPABILITY_GAP, where the cards are dealt and the AI refuses to play them.
POOL_UNREACHABLE: frozenset[str] = frozenset({
    "ownership_change", "radiation_change", "x_changed",
})


def test_every_pool_unreachable_type_is_actually_wired():
    """These three are absent for want of cards, not for want of an emitter.

    If one ever stops being referenced in the connector, its reason here
    becomes a lie: it would then be absent because nothing writes it, which is
    a different defect with a different fix. That is what this guards.
    """
    emitted = _emitted_event_types()
    for event_type in sorted(POOL_UNREACHABLE):
        assert event_type in emitted, (
            f"{event_type} is documented as pool-unreachable, which claims an "
            f"emitter exists -- but no EffectEvent constant for it is "
            f"referenced in the connector's Java source"
        )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/effects/domain/test_event_schema_completeness.py::test_every_pool_unreachable_type_is_actually_wired -v`
Expected: FAIL — `NameError: name 'POOL_UNREACHABLE' is not defined` until Step 1's constant is added in the same edit; if you added both together, it passes and you must instead verify by temporarily removing one type from the constant, watching the assertion fire, and restoring it.

- [ ] **Step 3: Record the measured reasons in `KNOWN_UNEMITTED`**

Add the three entries, in the same evidence style the existing nine use — the producing API, the card count read from `forge-gui/res/cardsfolder`, and the pool measurement:

```python
    "ownership_change":
        "GainOwnership: wired in ApiEvents.RULES and correct -- but all 4 cards "
        "that script it (Bronze Tablet, Darkpact, Tempest Efreet, Timmerian "
        "Fiends) are ante cards, which sealed pools exclude. None appeared in "
        "the 6.3 GB e984ef51 corpus. Live only with ante enabled",
    "radiation_change":
        "Radiation/InternalRadiation: wired off the bus (BusEvents.radiation, "
        "BusBracketCollector.onRadiation) and correct -- but all 22 cards that "
        "script it are Fallout (PIP), and 0 of the 8,403 matches in the "
        "e984ef51 corpus drew a PIP pool across 183 sets. Live in "
        "collect-coverage, which decks corpus-wide rather than from pools",
    "x_changed":
        "ChangeX: wired in ApiEvents.RULES (:134) and correct -- but only 2 "
        "cards script it. Unbound Flourishing appeared in 797 snapshots of the "
        "e984ef51 corpus and acted 9 times, every one of them its permanent "
        "spell cast (trait=spell idx=0); its trigger needs the controller to "
        "cast an X spell while it is out, which never happened. Glava, "
        "Five-Advents Mage never appeared",
```

Then delete the six entries Task 1 wired: `ability_activated`, `combat_ended`, `game_drawn`, `game_restarted`, `player_removed`, `turn_order_reversed`.

- [ ] **Step 4: Carve `POOL_UNREACHABLE` out of the textual scan**

`_emitted_event_types()` finds `EffectEvent.CONSTANT` references in Java source, so these three read as *emitted* — exactly as `AI_CAPABILITY_GAP` does. Find the comparison in `test_no_declared_type_is_unreachable_by_surprise` that already subtracts `AI_CAPABILITY_GAP` and subtract `POOL_UNREACHABLE` too, with a comment pointing at the reason. Run the module's tests after the edit; a mismatch here is the test telling you the two sets disagree, not a flake.

- [ ] **Step 5: Run the module's tests**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/effects/domain/test_event_schema_completeness.py -q`
Expected: PASS. `KNOWN_UNEMITTED` now holds 5 entries (9 − 6 wired − 1 retired + 3 documented), and `POOL_UNREACHABLE` holds 3 of them.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/effects/domain/test_event_schema_completeness.py
git commit -m "test(effects): document why three wired types no pool can reach stay absent"
```

---

### Task 4: Correct the AI-capability-gap reasoning

**Files:**
- Modify: `tests/unit/effects/domain/test_event_schema_completeness.py` (the `turn_ended` and `text_change` reasons)
- Test: same file

**Interfaces:**
- Consumes: `AI_CAPABILITY_GAP`, `KNOWN_UNEMITTED`.
- Produces: nothing new — this task corrects prose that is checkable and currently incomplete.

**What is wrong with the current reason:** both entries say the AI's refusal keeps every card carrying the API off the stack. True for observation. But `PatchedCollectors.collectInterventions()` iterates `actor.getCardsIn(ZoneType.Hand)` and forces each candidate through `ForkCollector.intervene`, and `worthIntervening()` rejects only lands, mana abilities, and lines already seen — **it never consults the AI's verdict**. An intervention would resolve an `EndTurn` or `ChangeText` clause the AI refused. So the real reason these are absent is that the 9 + 14 cards never reached a hand within a 2-per-game intervention budget, not that interventions cannot reach them.

- [ ] **Step 1: Extend both reasons**

Append to the `turn_ended` entry, inside the existing string:

```python
        " -- and no interventional fork has drawn one either: "
        "PatchedCollectors.collectInterventions walks the active player's hand "
        "and worthIntervening rejects only lands, mana abilities and "
        "already-seen lines, so an intervention would force this clause past "
        "the AI's refusal. The 9 cards simply never reached a hand inside a "
        "2-per-game budget. Reachable by collect-coverage, which decks "
        "corpus-wide, with --interventions-per-game set"
```

Append the same clause to `text_change`, with `14 cards` in place of `9 cards`.

- [ ] **Step 2: Run the module's tests**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/effects/domain/test_event_schema_completeness.py -q`
Expected: PASS — the reasons are prose, so this confirms nothing else keyed on their text.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/effects/domain/test_event_schema_completeness.py
git commit -m "docs(effects): interventions bypass the AI, so say why the gap types are really absent"
```

---

### Task 5: Update the contracts, then verify against fresh records

**Files:**
- Modify: `specs/023-ability-effect-model/contracts/record-schema.md` (§ Events)
- Modify: `specs/2026-09-05-ability-effect-model.md` (§ Events)
- Test: a `collect-coverage` run into a temporary directory

- [ ] **Step 1: Update both documents**

State the vocabulary size (77 after the retirement), that `trigger_fired` was retired because the `trigger` record's `fired` flag carries it, and that five types are documented as unreachable by AI sealed self-play — two for the AI's refusal, three for want of cards in any pool — with the pointer to `KNOWN_UNEMITTED` for the per-type evidence. Do not restate the evidence in the contract; one copy, in the test module, is the point.

- [ ] **Step 2: Build, with the run stopped**

First: `tasklist //FI "IMAGENAME eq java.exe"` — if any worker is alive, stop here and wait. Then:

```bash
cd /c/Users/nicol/IdeaProjects/price-predictor/forge-connector && mvn -B -o package -DskipTests
```

- [ ] **Step 3: Collect a contamination-free sample**

```bash
cd /c/Users/nicol/IdeaProjects/price-predictor
.venv/Scripts/python.exe -m effects collect-coverage \
    --effect-records /c/Users/nicol/AppData/Local/Temp/verify-vocab \
    --decks-per-round 60 --workers 6 --target-records 5 --no-progress-rounds 1
```

Let it run ~15 minutes, then stop it. `collect-coverage` writes effect records only — it never appends to `output/sealed/`, and it decks corpus-wide, which is the one collection mode that can deal a Fallout or an ante card.

- [ ] **Step 4: Census the sample**

```bash
for f in /c/Users/nicol/AppData/Local/Temp/verify-vocab/*.jsonl.gz; do
  gzip -cd "$f" | grep -oE '"type":"[a-z_]+"'
done | sort | uniq -c | sort -rn
```

Expected: the six from Task 1 may still be absent — that is predicted, not a failure. What this step *is* checking: no type that used to appear has stopped, and nothing emits a type the vocabulary no longer declares. Compare against the 66 types the `e984ef51` corpus contained.

- [ ] **Step 5: Run both suites**

Run: `cd forge-connector && mvn -B -o test` (expect >= 638 passing) then
`.venv/Scripts/python.exe -m pytest tests/unit/effects tests/unit/sealed -q` (expect >= 1,844 passing).

- [ ] **Step 6: Commit**

```bash
git add specs/023-ability-effect-model/contracts/record-schema.md specs/2026-09-05-ability-effect-model.md
git commit -m "docs(specs): the event vocabulary is 77 types, five of them unreachable by design"
```

---

## Self-Review

**Spec coverage.** The 12 missing types each have a task: six wired (Task 1), one retired (Task 2), three documented with measured reasons (Task 3), two with corrected reasoning (Task 4). The contracts follow in Task 5.

**Placeholder scan.** Every code step carries real code; every test names a real card verified present in `forge-gui/res/cardsfolder` on 2026-09-12; the one API with no reachable test card (`GameDrawn`) says so explicitly and routes to the static guard rather than leaving a gap.

**Type consistency.** `EffectEvent.ABILITY_ACTIVATED` / `.GAME_DRAWN` / `.GAME_RESTARTED` / `.PLAYER_REMOVED` / `.TURN_ORDER_REVERSED` are introduced in Task 1 Step 3 and used in Task 1 Steps 1, 4 and 6 under those exact names; `COMBAT_ENDED` already exists and is not redeclared. `POOL_UNREACHABLE` is defined in Task 3 Step 1 and consumed in Steps 4 and 5 of the same task. `KNOWN_UNEMITTED` ends at five entries — the arithmetic is stated in Task 3 Step 5 so an executor can check it.

**One risk worth naming.** Task 1's rules emit parameterless events because none of the six declares an `EVENT_PARAMS` row. If a reviewer wants parameters on any of them (`ability_activated` naming the activated ability, say), that is a vocabulary change: add the `EVENT_PARAMS` row first, in `event_schema.py`, or the reader will drop the parameter silently.
