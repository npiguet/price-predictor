# forge-connector review (2026-11-09): verified findings

Four fresh reviewers each read one slice of `forge-connector/src/main/java/com/pricepredictor/connector/`
(event attribution; channels and plumbing; snapshots and provenance; forks and copier). Each slice's
findings then went to a separate adversarial verifier, who re-traced them in the connector, the Forge
`effect-record-hooks` source and the Python readers. The slice-2 verifier also counted a real full-strength
shard (`output/effects/records-gen3/full-strength/7c8cb070-…2-mw929qz0xvkj.jsonl.gz`). Line numbers are as
of connector HEAD 2acdc5d7 and Forge HEAD 04e72e0518e; re-check them before editing.

Paths:
- connector classes are in `forge-connector/src/main/java/com/pricepredictor/connector/effects/` unless a
  package is given;
- Forge paths are under `../forge/forge-game` or `../forge/forge-ai`;
- Python paths are under `src/effects/`.

Verdicts: CONFIRMED (traced end to end), PARTLY, NEEDS-TEST, REFUTED. "Reaches training" names the model
input or target the defect corrupts.

---

## Slice 1: event attribution

The theme behind A1 to A3: the clause hook assumes one call per clause, made when that clause acts. In
fact, Forge fires the hook pair as follows (`AbilityUtils.resolveObservedClause` /
`resolveApiAbilityUnobserved`, about lines 1440-1480):
1. RESOLVING(clause).
2. `metConditions()`; if it holds, `sa.resolve()` (or `handleUnlessCost`).
3. `resolveSubAbilities(clause)`, which resolves each sub-ability inside the parent's scope.
4. RESOLVED(clause), from a `finally`.

The pair fires whether or not the condition held. For a chain P → S the order is P-RESOLVING, P's effects,
S-RESOLVING, S's effects, S-RESOLVED, P-RESOLVED.

### A3: clauses that did not run still emit their ApiEvents events. CONFIRMED (rank 1 in this slice)
- **Where:** `ApiEvents.ClauseObservation.finish` (about line 130) emits whatever happened.
- **Why it fires:**
  - Forge's hook fires even when `metConditions()` is false (AbilityUtils 1473) and when `handleUnlessCost`
    skips `sa.resolve()`.
  - A `WrappedAbility.resolve` that returns early also fires it: an intervening "if" failed,
    `checkResolvedLimit`, or `confirmTrigger` was declined (WrappedAbility 405-438).
- **Rules affected:**
  - Diff-based rules are safe: ChangeZone, Token, Block, RemoveFromCombat, ChangeCombatants,
    BecomeMonarch, ClassLevelUp, UnlockDoor, Bond.
  - These emit regardless: AddTurn, SkipTurn, AddPhase, SkipPhase, ChangeX, GainOwnership, Effect,
    DelayedTrigger, ImmediateTrigger (gated only on `TriggerAmount`), Replace*, ChangeText, RingTemptsYou,
    Animate (built from params), and HealDamage (whose `amount` is the damage before, not the damage healed).
  - Choose*, Mutate and TakeInitiative can read stale host state.
- **Declined triggers:** `OpenResolution.wasDeclined()` is `declined && modes.isEmpty()` (OpenResolution
  63-65). One spurious event makes a declined trigger read as resolved and writes an effect half.
- **Scripts with a condition or an unless cost** (cardsfolder):

  | API | scripts with a condition or unless cost | scripts in total |
  |---|---|---|
  | ImmediateTrigger | 139 | 323 |
  | Effect | 141 | 1914 |
  | DelayedTrigger | 41 | 475 |
  | Animate | 48 | 1075 |
  | AddTurn | 5 | 66 |

- **Reaches training:**
  - false `effects_created` (`aux_head_targets.py` 84-86);
  - false created-objects effect groups;
  - false `abilities_gained`/`abilities_lost`;
  - false turn-structure counts;
  - `resolution_outcome` `declined` turning into `resolved`.
- **Frequency:** moderate.
- **Fix:** at resolving time, record `metConditions()` and whether an unless cost was paid (`ConditionsMet`
  already reads the first) in the `ClauseObservation`. Skip `finish()` when the clause did not run. Skip
  wrapper clauses (see A2).
- **Tests:**
  - an ETB trigger with `DelayedTrigger`/`Effect` and a declined "you may": expect no
    `delayed_trigger_created`, and outcome `declined`;
  - a kicker spell whose `Condition$ Kicked` creates an Effect, cast unkicked: expect no
    `continuous_effect_created`.

### A5: surveil counts `card_milled` twice. CONFIRMED (rank 2)
- **Mechanism:** `Player.surveil` (Forge `Player.java` 1063-1085) moves each binned card with
  `moveToGraveyard`.
  - Each move publishes `GameEventCardChangeZone` Library→Graveyard, and `BusEvents.libraryMovement`
    (362-376) turns it into `CARD_MILLED`, count 1.
  - `GameEventSurveil` then reaches `BusEvents.surveil` (416-432), which adds `CARD_MILLED` with count =
    `toGraveyard` for the same player.
- **Reaches training:** the `cards_milled` count (additive, silent 0; `effect_targets.py` `_ADDITIVE`)
  reads 2N for N cards binned.
- **Frequency:** high in surveil sets (GRN, MKM, DSK).
- **Fix:** drop the `CARD_MILLED` from `BusEvents.surveil`.
- **Test:** surveil 2 into the graveyard; the sum of `card_milled` counts must be 2.

### L1: every Library→Hand move counts as a draw. CONFIRMED (rank 3)
- **Mechanism:** `BusEvents.libraryMovement` (363) is the only source of `card_drawn`; Forge publishes no
  draw event. Tutors to hand, explore lands and dig-to-hand all add `cards_drawn` +1.
- **Reaches training:** the player `cards_drawn` count (silent 0).
- **Frequency:** moderate to common in sealed.
- **Fix:** tell draws apart. Use Forge's per-player draw counter (`getNumDrawnThisTurn` before and after)
  or the Drawn trigger run parameters. Count a Library→Hand move as a draw only when the draw counter moved.
- **Test:** a tutor-to-hand effect and an explore that reveals a land: `cards_drawn` 0. A real draw:
  `cards_drawn` 1.

### A2: a trigger's root-clause ApiEvents events are written twice. CONFIRMED (rank 4)
- **Mechanism:**
  - `WrappedAbility.getApi` (73-75) returns the inner ability's API, so `resolveInScope` sends the wrapper
    through `resolveApiAbility` and the hook fires for it.
  - `WrappedAbility.resolve` (401-440) then reaches `PlayerControllerAi.playSpellAbilityNoStack` (727-730)
    → `ComputerUtil.playNoStack` (251-268) → `AbilityUtils.resolve(inner)`, which fires a second hook pair.
  - The wrapper delegates params, targets and activator, so its `finish()` emits an identical event.
  - The connector never checks `isWrapper()` (only `ProvenanceResolver` does).
  - None of the ApiEvents types is in `EventBuffer.IDEMPOTENT_EVENTS` (65-69), and nothing dedups events
    on the Python side.
- **Reaches training:**
  - `effects_created` and created-object counts (`aux_head_targets.py` 84-86, 216-223) for Effect and
    DelayedTrigger roots;
  - `abilities_gained` for Animate roots;
  - turn-structure counts for AddTurn/AddPhase roots;
  - `damage_healed` for HealDamage.
  - `token_created` (outcome hook) is not doubled. `choice_made` doubles but nothing reads it.
- **Frequency:** moderate, covering every trigger whose Execute root is Effect, DelayedTrigger or Animate.
- **Fix:** in `ClauseOutcomeRouter.clauseResolving`/`clauseResolved`, skip `sa.isWrapper()` for rule
  observation, but still push `NOTHING` to keep the deque balanced.
- **Test:** an ETB trigger whose root is `DelayedTrigger`: exactly one `delayed_trigger_created`.

### A4: stat events from going onto the stack land after a snapshot that already shows them. CONFIRMED (low frequency)
- **Mechanism:**
  - `GameAction.changeZone` runs `checkStaticAbilities()` with events on (554/567) as the card enters the
    stack zone. The resulting `GameEventCardStatsChanged` goes to StatDiffer → `route` → `Cause.CAST` →
    `CastsUnderWay.hold`.
  - `beginBracket` then takes the snapshot (`snapshots.ofResolving`, line 197), runs
    `rebaselineAtSnapshot`, and only afterwards calls `fileHeldInto` (215; `CastsUnderWay.PreCast.fileInto`
    289-293), filing those stat events into a half whose state already shows them.
  - `MagicStack.push`'s own recheck is `checkStaticAbilities(false)` (554) and publishes nothing.
- **Contract:** `record-schema-delta.md` 251-255 says "stat changes are changes after its own snapshot".
  This violates it.
- **Also:** abandoned `PreCast`s (`dropHeld`/`abandonAllBut` set `preCast = null`) are not counted in
  `DroppedEvents`.
- **Reaches training:** `pt_change`, `keyword_change` and `type_change` targets on cards whose statics read
  hand size, hellbent, or graveyard size when casting from the graveyard.
- **Fix:** count the held stat events as dropped once the snapshot is taken (or apply them before it), and
  count abandoned pre-casts.
- **Test:** cast from hand with a Maro (or a hellbent creature) on the board; the effect half has no
  `pt_change` that is already reflected in `state`.

### A1: a non-last charm mode's ApiEvents events land in the last mode's half. CONFIRMED (low frequency)
- **Mechanism:**
  - `CharmEffect.chainAbilities` (287-310) → `appendSubAbility` = `getTailAbility().setSubAbility(clone)`
    (SpellAbility 1198). So mode k+1 resolves inside mode k's scope.
  - `ClauseOutcomeRouter.clauseResolving` (102-104) → `bracket.beginMode` → `ModeSplit.begin` → `close()`
    drains mode k's half (ModeSplit 75-88) at mode k+1's RESOLVING.
  - Mode k's RESOLVED fires later and files `observed.finish()` (ClauseOutcomeRouter 129-130) into the last
    mode's span.
  - Diff rules for mode k also span the later modes, so they are reported twice.
  - `ModalResolutionTest` (125-132, 195-197) sends RESOLVING and `recordClauseEvent` by hand, so it never
    exercises the nesting.
- **Contract:** `record-schema.md` 110 says "each effect half carries only its mode's clauses' events". This
  violates it.
- **Reaches training:** `continuous_effect_created`, `delayed_trigger_created`, `ability_change`, the
  state-flag and look flags.
- **Fix:** decide the target half at RESOLVING (capture the current mode in the `ClauseObservation`) and
  file there at RESOLVED.
- **Test:** a real Boros Charm-style charm with an Effect or HealDamage first mode, resolved through
  `AbilityUtils.resolve` with the listener installed (not hand-ordered calls). Mode 1's event must be in
  mode 1's half.

### L2: statics that set base P/T are invisible to continuous records. PARTLY (low frequency)
- **Mechanism:** `LayerContributions.addCard` and `EntityJson.Suppressed.of` (about 345-356) read only
  `getPTBoostTable`. The set-P/T tables (layer 7a/7b) are neither reported nor removed.
  - A static whose only effect is set P/T yields no record, which is missing coverage.
  - A static that also changes keywords or types (Frogify-style "loses all abilities and is a 1/1") yields
    a contribution with `pt_boost (0,0)` while the snapshot already shows the 1/1. That corrupts the record.
- **Fix:** together with B1 below.

### A8: a copy parked in the frozen stack drops the resolution's state-based deaths. CONFIRMED (very rare)
- **Mechanism:**
  - `resolveStack` freezes the stack (MagicStack 562), and a copy added mid-resolution without
    `IgnoreFreeze` is parked (345-349).
  - `GameEventSpellResolved` (628) → `endBracket` → `heldResolution.hold`. Then `finishResolving` (655) →
    `unfreezeStack` (160-162) → `push` → `GameEventSpellAbilityCast`.
  - `beginBracket` → `onNextAction()` (186) → `heldResolution.close()` before the state-based loop. Deaths
    become `STATE_BASED_DEATH` and are dropped.
- **Contract:** violates delta 248.
- **Example:** Chain Lightning-style spells.
- **Fix:** don't treat a push made while `StackBrackets.finishing` is set as a next action.

### A6: charm-mode segment indices are offset. CONFIRMED in mechanism, negligible impact
- **Mechanism:** `EventAttribution.attributedTo` (113-136) counts hops to the charm root. The option line's
  chain has no charm segment, so even mode 1 is off by one (Python `_chain_index` = `int(seg)+1`,
  `corpus_curation.py` 313), and later modes add the earlier modes' chain lengths.
- **Readers:** only `turn_step_condition` survey counts and validation.
- **Fix:** count from the mode clone, treating the clone itself as root.

### X1 (new, found by the verifier): every triggered ability's clause events may be stamped `attributed_to: "unresolved"`. NEEDS-TEST
- **Mechanism:** a trigger's executed root is a parentless `AbilitySub`. "DB$" builds an AbilitySub
  (AbilityFactory 88), and TriggerHandler (about 569-596) copies it, anticipating
  `sa instanceof AbilitySub` at 589. `EventAttribution.attributedTo` returns UNRESOLVED when the hop walk
  ends at a null parent (131-135), and `ConditionsMet.of` returns null, so `conditions_met` is always `[]`
  for triggers.
- **Consequence:** `sba_zone_move.py` 150-158 treats an "unresolved" battlefield→graveyard move as a
  state-based death. An ETB "destroy target creature" trigger (common) would then read as SBA deaths to that
  detector.
- **Test:** resolve an ETB destroy trigger in a live game; assert the zone_change's `attributed_to` is
  `"root"` and `conditions_met` lists a conditional clause where there is one.

### Not bugs, or decisions for the user
- **A7, DECISION:** a resolution's state is the cast-time board (`OpenResolution.state` set in
  `beginBracket` 197; the `StackBrackets` Javadoc says so). Responses that resolved in between are missing,
  e.g. Bolt at a 3/3 pumped to 5/5 shows a 3/3 that took 3 and lived. Modal halves read just before their
  mode. Being decided by the user (see README).
- **A9:** a non-combat `player_won` goes to the game-end rescue record. This is documented
  (`record-schema.md` 977) and modelled (`sba_game_loss.py` 21-27). `GAME_LOST` has no silent value, so
  this only leaves supervision missing.
- **L3:** amounts re-evaluated after the clause are negligible (literal `NumTurns$` in 64 of 66 AddTurn
  scripts).
- **L4:** a stale `primaryAbility` after a failed AI payment drops routed CONSEQUENCE events until the next
  `unfreezeStack`. The window is short. Test:
  1. `freezeStack(staleSa)` outside a resolution;
  2. publish a CONSEQUENCE event;
  3. check `DroppedEvents`;
  4. `unfreezeStack()`, then check routing resumes.

### Code quality (slice 1)
- Where an event goes is decided in five places: `fileDamage`, `fileMove`, `route`, `beginMode`,
  `recordClauseCondition`/`noteDeclined`, and `CastsUnderWay`. Name it as one pure `EventDestination`
  function, tested on its own.
- `ClauseObservation` lacks "did it run" and "which half": add both (this fixes A1-A3 in one place).
- `CastsUnderWay` wants a per-card state machine (OnItsWay → Arrived → Pushed → Announced/Abandoned).
- The `IDEMPOTENT_EVENTS` Javadoc essay should be a test.
- `BusBracketCollector.beginBracket` is a long method; extract named steps.

---

## Slice 2: channels and plumbing

### C2: a mana record's state already contains its label. CONFIRMED (rank 1)
- **Mechanism:** Forge `AbilityManaPart.produceMana` calls `manaPool.addMana` (about 220), then the hook
  (225-229, "after the pool has it"). `ManaChannel.handler` (63-66) snapshots inside the hook, and
  `PlayerJson` 46-47 writes `floating_mana`, which `effect_head_input.py` 517-518 reads as a player input.
  The target is `mana_delta_<colour>` (`effect_targets.py` 451). The tap cost has also been paid, so the
  source is already tapped and `untapped_production` has dropped.
- **Data:** in 388 of 402 mana records (96.5%) in a real shard, the actor's `floating_mana` exactly equals
  `mana_produced`.
- **User decision:** read the board before the mana ability activates (source untapped, pool without the
  mana), and add a `tapped` event for the source beside `mana_produced`. The goal is that "tapping a source
  that says add {G} produces {G}" is learnable. The cost half's `costs.mana_by_color` already says what mana
  paid for a spell. No link from a spell to the lands that paid for it is wanted.
- **Implementation note:** this needs a hook point before the activation, or a snapshot taken at
  activation start by the AI or random-seat payment path. Alternatively, subtract the produced mana and
  untap the source in the rendered state; the first is cleaner. `refs.source` is null by design (`ofBoard()`);
  naming the host would also stop context dropout from removing the producing land
  (`effect_head_input.py` 1207-1213).

### C4: decision records carry stale targets, X and modes. CONFIRMED (rank 2)
- **Mechanism:**
  - Candidates are the cards' persistent `SpellAbility` objects (`PriorityCandidates.abilitiesOf`).
  - The AI's `canPlayAndPayForFace` runs `canPlaySa` first (which "sets up targets, defines X",
    `AiController` 886-893), then `canPayCost`, so an unaffordable spell keeps its targets.
  - The random seat's abandoned draws leave targets too (`RandomSeatController.drawPlayChoices`).
  - `PlayabilityChannel.recordDecision` (118-126) → `RefsJson.render` (100-118) reads `getTargets()`, X and
    `getChosenList()` off that object.
  - This contradicts `SnapshotBuilder` 104-106 ("has no targets yet").
- **Data:** 15 of 86 decision records with legal targets carry non-empty `refs.targets`, and in 13 of those
  15 every stale target is legal. 62 of 698 decision records carry an X; 2 carry modes.
- **Reaches training:** `refs.targets` becomes an entity flag and a target count (`effect_head_input.py`
  534, 672), and X and mode count become `[ACT]` inputs. `target_legal` is supervised on these records, so
  this is a partial label leak and a train/inference mismatch.
- **Fix:** decision records render refs with `source = host` only (targets, X, modes and choices empty).
- **Test:** put a target into the persistent `shock.getTargets()`, call `recordDecision`, and assert
  `refs.targets == []`.

### C1: a fork that ends the game loses its lethal combat record and its game-end events. CONFIRMED (rank 3)
- **Mechanism:**
  - Forge fires `GameEventGameFinished` only in `Match.startGame` (`Match.java` 100).
  - Forks play through `ForkPlayForward.playToTheEndOfTheTurn` (109-121), which stops on `isGameOver()`.
  - `BusBracketCollector.finishGame` (910-925), the only game-end caller of `combatStep.flush()` and
    `flushOrphanedEvents()`, runs only from `onGameFinished` (900).
  - The fork's close path (`GameRecording.close` → `PatchedCollectors.close` 226-231 → `Channels.close(false)`
    66-80) flushes only playability and the mana reservoir.
  - `GameCopier` builds the fork with `maingame == null` (100), so `setGameOver` fires
    `GameEventGameOutcome` (`Game.java` 599-601) into the open bracket or orphans, which nothing writes.
  - The killing resolution's held effect half is fine: the state-based-loop end hook fires in a `finally`
    (PhaseHandler 1215-1226).
  - The only test touching this asserts `gameOver()` is false (`ForkPlayForwardTest` 190).
- **Reaches training:** lethal fork combat records are dropped, a bias. Forks never carry `player_won`.
  Directed rule 121.4 (DrawFromEmptyLibrary, variant `empty`) never writes its outcome.
- **Fix:** in `ForkPlayForward.play` (or `GameRecording.close` for a fork), call the bracket's
  `finishGame(idsOnStack())` before closing.
- **Test:** fork a board with an unblockable attacker and the opponent at 1 life, `play()`, and assert the
  held lines include a combat record with `player_won`.

### C5: `affordable` follows the AI's payment planner. CONFIRMED (label); side effects PARTLY
- **Mechanism:** `PlayabilityVerdict.affordable` (160-167) → `ComputerUtilMana.canPayManaCost`. Despite its
  docstring, this honours AI mana reservations through `isManaSourceReserved` (942, 1044-1085):
  - `HELD_MANA_SOURCES_FOR_NEXT_SPELL`, always;
  - held-for-blockers sources, outside declare-blockers and cleanup (set for combat tricks,
    `ComputerUtilCard` 1848, and Fog, `FogAi` 108);
  - main-2 holds.
  It also applies AI-only source filters (life cost, drawbacks, `chooseSacrificeType`).
- **Side effects:**
  - `MyRandom.percentTrue(chanceToReserve)` (1071-1072; Default.ai sets 100) draws from Forge's global
    random stream on every source checked, so the collector perturbs the game's random sequence.
  - `UNPAID_COSTS` grows (read by `ChangeZoneAi` 563).
  - Held-for-blockers checks clear AI memory sets (1058-1063).
- **Reaches training:** false `affordable=false` labels and failed checks while a reservation is active.
- **Fix:** a rules-only mana check over `getAvailableManaSources` without reservations or AI filters, or
  snapshot and restore `AiCardMemory`. Never draw from `MyRandom`.
- **Test:** `AiCardMemory.rememberCard(ai, land, HELD_MANA_SOURCES_FOR_NEXT_SPELL)` for every land, then
  `PlayabilityVerdict.of(castableSpell, ai).affordable()` must be true.

### C3: mana made by a stack resolution gets contradictory labels. CONFIRMED (rare in sealed)
- **Mechanism:** `ManaEffect.java` 187 → `produceMana` fires the mana hook with root = the resolving spell
  or trigger. `ManaChannel` has no `isManaAbility` filter, so it writes a separate mana record. Nothing puts
  `mana_produced` into the resolution's effect half: there is no `GameEventManaPool` subscription, and
  `RunParamNormalizer` maps it only for trigger and replacement params. With `silent=0.0`
  (`effect_head_layout.py` 290-297), the resolution trains `mana_delta = 0` while the mana record says +N.
- **Fix:** `ManaChannel` takes only `isManaAbility()`. The clause or outcome path emits `mana_produced` into
  the open resolution bracket.
- **Test:** Dark Ritual's effect half has `mana_produced {B:3}`, and there is no separate mana record for
  it.

### C7: trigger positives are capped take-first at 24 per line and verdict per game. CONFIRMED (low impact)
- **Mechanism:** `TriggerNegativeSampler.java` 37 and 147-150, checked before the snapshot in
  `TriggerChannel` 58. This biases lines with more than 24 firings toward early boards, and it contradicts
  the mana reservoir's reasoning.
- **Fix:** reservoir-sample per line, or cap negatives only.

### C11: the silenced-hook diagnostic is wrong. CONFIRMED (diagnostic only), plus a real failure it hid
- **Mechanism:** `EffectRecordHook.contain` keeps calling the listener after a failure and only stops
  logging. `MatchWorkerMain` 329 ("their channels now empty") and the Javadoc are wrong.
- **Fix:** count failures per hook and report the counts.
- **Real failure found:** in `output/effects/records-gen3/coverage-held-out/83a758c5-…4.log` the
  priority-settled listener threw an NPE: `Card.getZone()` null inside `isCardAddedThisTurn`. Every later
  failure was unlogged, so the number of lost decision stops is unknown. Investigate it (it may involve
  `Zone.isCardAddedThisTurn` and last-known copies).

### Latent or negligible
- **C6:** in `PhaseBridge.onPhase` (`GameRecording` 311-323), a throw in `continuous().collect()` skips
  that boundary's interventions, probes and directed forks. Guava logs at SEVERE; no occurrence in the
  gen-3 logs. Guard each step anyway.
- **C9:** `RecordShardWriter` counters (96-103) are unsynchronised. AI eval threads that time out keep
  running on Java 20+. No hooked stack was observed. Make them `AtomicLong`.
- **C10:** the game seed is `startMillis + i` (`GamePlayer` 181-201). Workers starting within a few ms share
  streams; no bias.
- **C12:** `ContinuousChannel.boardHash` (124-132) hashes id, name and P/T only; minor. The `staticById`
  null cache (141-159) needs a test: an Animate that grants a static is found at the next boundary.
- **REFUTED:** C8 (stage-1 sampling before de-duplication is the documented design, and stops rarely
  repeat byte for byte).

### Code quality (slice 2)
- `PatchedCollectors` walks its everyMode and patchedOnly lists three times (install, dispatch, close).
  Build one `Map<Hook, ListenerBody>`.
- `PriorityCandidates.abilitiesOf` (about 60 lines) hand-copies `Card.getAllPossibleAbilities`. Pin it with
  a test against Forge's list.
- Stale comments:
  - the `PatchedCollectors` class doc (caps "live here"; mana cap "per text in this JVM");
  - `GameRecording` 103 and `CollectorContext` 14 ("counts its own records");
  - `SnapshotBuilder.ofAbility` ("no targets yet");
  - `src/effects/CLAUDE.md`, which says the random seat writes decision, attackers and blockers records
    (gen-3: none).
- `PhaseSignals.inStateBasedLoop` is read only by tests.
- `RandomSeatController` (630 lines) should be split into `RandomPlay` and `RandomCombat`.
- The game-identity rule is duplicated (`LiveGame.owns`, `BusBracketCollector.belongsToLiveGame`).

---

## Slice 3: snapshots and provenance

### B3: `refs.x` is fabricated or read after the resolution. CONFIRMED (rank 1)
- **Mechanism:**
  - `RefsJson.announcedX` (146-159) calls `acting.getSVar("X")`. Forge's `CardTraitBase.getSVar` (558-563,
    fallback 538-552) falls back to the card's SVars, so any ability on a card defining `SVar:X` gets an
    `x`: triggers, permanent casts, unrelated activations.
  - Live refs are re-rendered at `GameEventSpellResolved` (`BusBracketCollector` 258-302 →
    `HeldResolutionWindow` 102), fork refs at `InterventionFork` 365-369, both after the effect. A `Count$`
    X is recomputed on the post-resolution board, which leaks the outcome.
  - `Count$Random` draws `MyRandom` (AbilityUtils 2844-2849) on every refs render, decision candidates
    included (`SnapshotBuilder` 189).
- **Scope:** 6,401 of 33,682 scripts define `SVar:X:`; 4,511 of those are `Count$`; 7 use `Count$Random`.
- **Reaches training:** `act_features` x value and x-present flag (`effect_head_input.py` 479-480).
- **Fix:** read the announced X only when the acting chain's parameters reference X, read it once at the
  cast with the board, and never evaluate `Count$Random`.
- **Test:** an Elvish Promenade-style "for each" token spell; `refs.x` must equal the pre-resolution count
  (or be absent if not a paid X). A static-only `SVar:X` card's unrelated activation has no `x`.

### B7: a dying host's last-known copy replaces its graveyard card in trigger records. CONFIRMED (common)
- **Mechanism:**
  - `GameEntity.equals` (394-397) compares id and class only.
  - Leaves-the-battlefield triggers are registered from last-known copies (CardZoneTable 112-115;
    GameAction 985), so `trigger.getHostCard()` is the copy.
  - `LiveGame.traitRecord` (178-182) passes it as tier 1, and `SnapshotBuilder.collectEntities` (271-281)
    keeps it and drops the graveyard card with the same id.
  - `getLKICopy` sets only `lastKnownZone`, so `EntityJson.zoneName` writes `"zone": null`.
- **Effect:** the entity shows the old battlefield state with zone null, and no graveyard entity exists. This
  hits every trigger record, fired or negative, of a dies or leaves-play trigger whose host just left.
- **Reaches training:** zone one-hot and status features.
- **Fix:** in tier 1, map each referenced card to `game.getCardState(card)` when the live game holds it.
  Keep the last-known copy only as the `L<id>` entity.
- **Test:** a creature with a dies trigger dies on a TestBoard; exactly one `E<id>` entity, zone
  `"graveyard"`.

### B1: a continuous record's "board without its static" still shows the static's work. CONFIRMED (occasional)
- **Mechanism:**
  - **Set base P/T:** `EntityJson.basePtJson` (130-137) reads live `getNetPowerBreakdown().currentValue`,
    which includes the static's `addNewPT(..., stAb.getId())` (`StaticAbilityContinuous` 678-689). Only
    boosts are removed (153-156).
  - **Animation:** `basePtJson` tests live `isCreature()` while `types` come from `getTypeWithout`, so an
    animated non-creature gets P/T but no Creature type.
  - **Removed abilities:** `StaticAbilityContinuous` 853-857 files `removeAbilities` under the static's id,
    applied inside `CardState.getSpellAbilities()/getTriggers()` (CardState 458-463 →
    `Card.updateSpellAbilities` 3497-3500 → `CardTraitChanges.applySpellAbility` 91-97). `CardLines.Builder`
    (125-135) only holds back lines the static added, so removed ones stay missing (keywords come back via
    `getKeywordsWithout`).
  - **Control:** `controller` is live (EntityJson 76); rare, and not a label.
- **Contract:** base `record-schema.md` 141-143 says the static's entries are dropped from every layer
  channel it wrote.
- **Reaches training:** total P/T thermometers (`state_snapshot` 227-231), controller relations,
  ability-line slots; `continuous` is a trained class.
- **Examples:** Frogify, Kenrith's Transformation and Lignify-type auras; Ensoul Artifact; Spacecraft
  station.
- **Fix:** one `StaticSuppression` object answering every channel (type, colour, keywords, set P/T and
  boosts, controller, traits added and removed). Use the suppressed type for the creature test. Forge may
  need "without" reads for set P/T (filter `setPT` by column) and for trait removal.
- **Test:** `ContinuousChannel` records for Imprisoned in the Moon, Control Magic and Ensoul Artifact;
  inspect `base_pt`, `types` and `lines` on the affected permanent.

### B5: library cards an event names are rendered after the move. PARTLY (accepted part is a decision)
- **Mechanism:** `LibraryCapture` (25-31, 58-72) and `Snapshot` 61-73 capture zone and rank at the
  snapshot, but render every other field from the live object at write time. Forge keeps the same object
  only for a move to the battlefield (GameAction 171-173).
- **Accepted by contract** (delta 151-153): tapped and counters after the move, e.g. a fetched land tapped,
  sick, `entered_this_turn`.
- **Not accepted:** face-down entry (manifest, cloak, manifest dread: empty name, 2/2, `face_state`
  face_down), anthem boosts and keywords, clones entering as copies. The `LibraryCapture` Javadoc's "name,
  types, lines are read right" is false for these.
- **Decision pending** (see README): render with library defaults, which removes both parts.

### B4: played-forward forks list no floating effect drawn before the fork. CONFIRMED (fork records only)
- **Mechanism:** a played-forward fork (directed, protection) builds `GameRecording(fork, …)` →
  `new BusBracketCollector(fork)` → `SnapshotBuilder(fork)` (`GameRecording` 157-169, `BusBracketCollector`
  136, `LiveGame` 42), so `FloatingEffects.of(fork)` starts with empty notes. Layer rows from before the
  fork have no note and are not listed as `F<ts>`.
- **Contract:** delta 132 says a fork's snapshot reads the live game's notes.
- **Why the audit missed it:** `ForkFidelity` (42-43) uses `SnapshotBuilder(fork, live)`, the live notes,
  so it cannot see this.
- **Reaches training:** command-zone card slots (`effect_head_input.py` 865-893), plus `affects` and
  `created_by` links.
- **Fix:** seed the fork's notes from the live game's, or fall back to the live notes for timestamps at or
  before the copy point. Make `ForkFidelity` use the same builder the fork does.
- **Test:** resolve a pump in main 1, take a directed fork later in the turn, and assert its records'
  `floating_effects` hold the `F<ts>`.

### B9: Gift casts never set `cast_with`. CONFIRMED (set-specific)
- **Mechanism:** `OptionalCost.PromiseGift` is named "Promise Gift" while the keyword is "Gift" (Keyword
  96, 278-280), so `CardLines` 188-193 never matches. Gift casts take the optional-cost path
  (GameActionUtil 489-491).
- **Fix:** an explicit `OptionalCost` → `Keyword` map.

### B6: face ordinal shifts after a card has been face down. CONFIRMED (rare)
- **Mechanism:** `FaceOrder.faceOrder` (44-56) numbers by position in `card.getStates()`, an EnumMap in
  which FaceDown is ordinal 1. It is inserted lazily (`Card` 547-551). The converter numbers only printed
  faces (`RulesParser` 115-121). `faceIndex` (36-41) silently returns 0 for an unknown state.
- **Fix:** number over the states the converter emits (skip FaceDown and EmptyRoom), and return null for an
  absent state.
- **Test:** turn a transform DFC face down, then face up, then transform it; `ProvenanceResolver.of(backFaceTrait).face() == 1`.

### B8: clone line keys. REFUTED as stated; NEEDS-TEST for a related problem
- **Refuted part:** every copied line keys to the copier's own script.
- **Related problem:** those keys carry the copied card's indices, so they may name the copier's wrong line,
  e.g. a copied replacement versus the Clone's own "enter as a copy" replacement at index 0.
- **Test:** a Clone copies a creature with an ETB trigger (and one with a replacement). Each line's key
  names the original's script or no key, never `clone.txt` TRIGGER 0 or REPLACEMENT 0.

### Real but read by no model input
- **B2:**
  - an effect card's own `affected` in `floating_effects` (`FloatingEffects` 197-206, 253-281, no
    `withoutStatic`);
  - an aura's grants in `granted_temporary.abilities` (`EntityJson` 222-232).
- **B10:** `combat.attacking` is written with Forge's `toString`; only its presence is read.

### Refuted (B10 parts)
- Secondary state maps to `back`: a design choice.
- "Quadratic notes": the cost is bounded in the common case. A note that never closes pins its `Game`;
  rare, performance only.

### Code quality (slice 3)
- The "board without the static" suppression is scattered (`EntityJson.Suppressed`, `CardLines`,
  `SnapshotLinks`, `LibraryCapture`, not `FloatingEffects`). Make it one `StaticSuppression` with a test that
  walks every entity field.
- The `RefsJson.announcedX` Javadoc promises a check the code doesn't make.
- The `LibraryCapture` Javadoc documents the leak instead of preventing it.
- `FaceOrder.faceIndex` returns a silent 0.
- `EntityJson.combatJson` should write `attacking` as a ref or a boolean.

---

## Slice 4: forks and the copier

What each fork kind does after copying:
- **Damage-step probe:** `DamageStepProbe.runDamageStep` strips a keyword, then runs
  `assignCombatDamage`, `dealAssignedDamage` and the state-based check once. It is taken at the
  FIRST_STRIKE and REGULAR damage phase events.
- **Intervention:** forces one resolution and calls `GameSimulator.resolveStack` (`InterventionFork` 267);
  it never advances phases.
- **Protection fork:** taken at MAIN1 and played to the end of the turn.
- **Directed fork:** played to the end of the turn. Rules are phase-guarded, except meld, aura swap and
  GameLossReplaced.

Every fork's state block is `SnapshotBuilder` on the copy, so any field the copier drops that the snapshot
prints is wrong in every fork record of that board.

### F1: a copied combat forgets who dealt first-strike damage. CONFIRMED (rank 1, set-independent)
- **Mechanism:**
  - `Combat.java` 80 declares `combatantsThatDealtFirstStrikeDamage`; the copy constructor (88-124) never
    copies it.
  - `dealDamageThisPhase` (915-925) returns `!firstStrike && !set.contains(c)` for a non-double-striker.
  - PhaseHandler fires `GameEventTurnPhase` in `advanceToNextPhase` (about 255) before `onPhaseBegin`
    deals the damage (377-385), so a REGULAR-step probe copies a combat whose set is empty.
- **Effect:** every first-strike-only creature deals damage again in the probe fork, the stripped carrier
  and bystanders alike. A `first_strike` probe at the regular step shows the carrier dealing damage the real
  record never shows. Directed and protection forks are not affected (taken before damage); interventions
  deal no combat damage. `ReplaceCardOperation.swapInCombat` already notes this gap.
- **Reaches training:** gate 2's probe records and every regular-step probe of a combat with a first
  striker.
- **Fix:** copy the set in `Combat(Combat, IEntityMap)` through `map.mapCollection`.
- **Test:** a 2/2 first striker attacks into a 1/4. Run the first-strike step live, copy at `COMBAT_DAMAGE`,
  then `assignCombatDamage(false)` and `dealAssignedDamage()` on the copy. The blocker has 2 damage
  (today: 4, and it dies).

### F2: after the monarch or initiative changes hands, every later copy throws. CONFIRMED (rank 2)
- **Mechanism:**
  - `GameAction.becomeMonarch` (2556) and `takeInitiative` (2578) call `removeMonarchEffect` /
    `removeInitiativeEffect` (`Player.java` 3566-3571, 3638-3643). Unlike radiation, keyword, blessing and
    enduring-story, these never null the field.
  - `Zone.remove` (162) does not clear `card.zone`.
  - `Player.mapEffectCard` (2886-2889) then maps it → `GameCopier.find` → "Couldn't map" RuntimeException
    (1024), raised from `makeCopy` 159.
  - `Forks.copyAndVerify` swallows it after spending the budget.
  - The live score's combat simulation copies too, and throws as well.
- **Effect:** zero fork records for the rest of such a game (CN2, CMR, CLB pools; the initiative changes
  hands often in CLB).
- **Existing tests miss it:** `GameSimulationTest` (about 2724) and
  `ForkFidelityTest.aCopyKeepsTheMonarchsEffectCard` cover only a monarch that never changed hands.
- **Fix:** null the field on removal, or have `mapEffectCard` require `effect.getZone().contains(effect)`.
- **Test:** `becomeMonarch(p0)`, `becomeMonarch(p1)`, `new GameCopier(game).makeCopy()` must not throw.
  Repeat with `takeInitiative`.

### F5: card fields the copier never copies. PARTLY (each field confirmed uncopied; impact varies)
- None is set by `GameCopier` or `Card.copyFrom` (8270). `CardCopyService` copies several, but only on the
  LKI path (306-374).
- **Changes records:**
  - Snapshot designations (`CardFacts` 79-96): `bestowed`, `goaded`, `harnessed`, `plotted`, `ring_bearer`,
    `specialized`, `unearthed`.
  - Ring-bearer (LTR): the card flag and `Player.ringBearer` are not copied, and `theRing` is missing from
    `copyEffectCardsToSnapshot` (`Player` 2875-2883). The Ring's statics are dead in the copy, and a new
    tempt makes a second Ring.
  - `classLevel` defaults to 1 (`Card` 246). Level-2 and level-3 traits are disabled (3141) and can be
    levelled again (AFR, BLB; Barbarian Class's `affects` link).
  - Valiant: `targetedFromThisTurn` (`Card` 3760; `MagicStack` 472) is not copied, so Valiant fires again
    (BLB).
  - Play-forwards: goad (`AttackRequirement` 36), detain (`StaticAbilityCantAttackBlock` 45,
    `AbilityActivated` 94), bestow (an aura whose creature dies goes to the graveyard instead of becoming a
    creature).
- **Near-nil effect:**
  - Unearth: the exile replacement is on a copied effect card.
  - Exert: "doesn't untap" acts only beyond the recorded horizon.
  - Per-turn counts (drawn, fought, started untapped, crew count, excess damage, regenerated) and the
    per-source damage map (GameAction 1465).
- **Fix:** one `copyCardState(orig, copy)` for zone-independent state; see also F6.

### F8: a transformed DFC's copy is broken. PARTLY CONFIRMED; the keyword part NEEDS-TEST
- **Mechanism:**
  - `backside` and `transformedTimestamp` are not copied, and `setState` (`Card` 570ff) doesn't set
    `backside`, so `isTransformed()` (1178) is false. `changeCardState("Transform")` (689-692) flips
    `backside` and calls `changeToState(Backside)`, which is a no-op, so the card cannot transform back in a
    fork.
  - `setSVars` (GameCopier 940) writes the back face's SVars into the front state, because `addCard` runs
    before `setState` (line 289).
  - Likely: the back face's keyword cache is never rebuilt (`updateKeywordsCache` at 859 runs on the front;
    `setState(…, false)` recomputes nothing), so a one-shot granted keyword on a transformed DFC is missing
    in the copy. Possibly the audit's Tovolar's Magehunter `lines`/`granted_temporary` difference.
- **Fix:** set the state and `backside` right after `createCardCopy`, before the battlefield branch; copy
  `transformedTimestamp`.
- **Test:** transform a werewolf, pump it with "gains trample until end of turn", and copy. The copy has
  `isTransformed()`, `hasKeyword(TRAMPLE)` and the same CardLines output. `changeCardState("Transform")`
  returns it to Original.

### F3: dungeons are missing from fork states. CONFIRMED (AFR, CLB)
- **Mechanism:**
  - `VentureEffect` 74-75 builds the dungeon from a token PaperCard, then sets its type to DUNGEON.
  - In the copy, `createCardCopy` → `CardFactory.getCard` (184-185) marks it TOKEN, and nothing restores
    the type. `Zone.add` (142) does not list a non-battlefield token, so the card is mapped but not in the
    Command list.
  - `currentRoom` is not copied.
- **Effect:** `dungeons: []` in fork states; a venture in a play-forward restarts from room 1.
- **Fix:** `newCard.setGamePieceType(c.getGamePieceType())` and `setCurrentRoom(c.getCurrentRoom())` in
  `addCard`.
- **Test:** venture once, copy; the copy's Command zone holds the dungeon at the same room.

### F4: mutate and prototype read as clones in the copy; under-cards are lost. CONFIRMED (IKO, BRO)
- **Mechanism:** `mutatedTimestamp` and `prototypeTimestamp` are not copied while clone states are
  (GameCopier 914-916), so `isCloned()` (`Card` 1159-1161) is true and `getCloner()` returns a host.
  MutateEffect moves under-cards to `ZoneType.Merged`, which is not in `ZONES`. `mergedCards` and
  `timesMutated` are not copied either.
- **Effect:** `copy_source` names a cloner (`EntityJson` 243), no mutate components are listed
  (`SnapshotBuilder` 237, `SnapshotLinks` 138), and a mutated creature that dies leaves no under-cards.
- **Fix:** copy both timestamps, the merged pile (mapped) and `timesMutated`, and include the Merged zone.

### F7: a reference to a phased-out card makes the copy throw. PARTLY (rare)
- **Mechanism:** `Game.getCardsIn` filters out phased-out permanents (`Player` 1316, 1336), so they are
  never copied. A reference to one throws (e.g. `getRemembered`, GameCopier 164-176, such as Oubliette), and
  the fork is lost. `SnapshotBuilder` 278 uses the same filter, so `ForkFidelity` is blind to it. Their
  absence alone barely matters within the horizon.
- **Fix:** iterate `getCardsIn(zone, false)` and copy `phasedOut`, `directlyPhasedOut` and
  `wontPhaseInNormal`.

### F6: chosen values and off-battlefield state. CONFIRMED (rare)
- **Mechanism:** counters, `setSVars`, changed SVars and chosen values are copied only inside
  `if (zone == Battlefield)` (GameCopier 839-942). On the battlefield only chosen player, type, type2,
  colours and named cards are copied: number, cards, mode, direction, even/odd, sector, rolls and
  `markedColors` are not. Effect cards (path 794-827) get no chosen values at all.
- **Effect:**
  - the snapshot's `chosen` is wrong (e.g. Circle of Protection: Green's effect loses its chosen card);
  - an effect card's static keyed to a chosen type or colour does nothing in the copy;
  - exiled cards lose their counters (CROAK, suspend time counters).
- **Fix:** move zone-independent state out of the battlefield branch (the same `copyCardState`).

### F12: the copied phase handler's counters are zeroed. CONFIRMED (minor)
- **Mechanism:** `devModeSet` (PhaseHandler 1271-1284) sets only the phase, player and turn. `extraPhases`,
  `nCombatsThisTurn`, `nMainsThisTurn`, `nUpkeepsThisGame` and `skipDamageSteps` (75-93) are lost.
- **Effect:** `isFirstCombat()` is false in forks taken at or after declare attackers (about 49 FirstCombat
  scripts); extra combats are lost.

### F11: `blocked_this_turn` is lost when the attacker no longer exists. CONFIRMED (narrow)
- **Mechanism:** GameCopier 501-507 re-adds only attackers `counterpartCard` maps; a dead token attacker has
  no counterpart.
- **Fix:** keep last-known copies, as `copyAttacksFrom` does.

### F10: a play-forward runs into the next turn's untap step. PARTLY (minor)
- **Mechanism:** the cleanup `mainLoopStep` advances the turn and begins UNTAP in one call (PhaseHandler
  1191-1193).
- **Effect:** the untap turn-action window is opened and never written. Only `continuous().collect()` at
  the new turn's untap boundary writes stray records.
- **Fix:** stop the play-forward when the turn would end (at cleanup with an empty stack).
- **Test:** a play-forward from MAIN2 holds no line with `subkind` untap, phasing or day_night.

### Refuted or negligible
- F9: the shared saddle list cannot be mutated.
- F13: duplicate probes; keywords already de-duplicate.
- F14: the global `MyRandom` swap overlaps only leaked AI threads.

### Code quality (slice 4)
- `ForkBudget`'s "per real resolution" is in practice per candidate line (`InterventionScheduler` 113-117
  passes `candidate.getId()`). Rename it, or key it on the real resolution.
- `ProbeScheduler.writeHeldProbes(String)` guesses the substep; the explicit `(id, DamageStep)` overload is
  unused. Pass the record's substep.
- `Forks.findInFork` duplicates `GameCopier.findSAInCard` (one uses `getSpellAbilities`, the other
  `getAllSpellAbilities`). Use one mapper, the copier's `counterpartAbility`.
- The battlefield-only branch in `GameCopier.addCard` is the root of F5 and F6. Write one
  `copyCardState(orig, copy)`.

### Copier gaps still seen by the fork fidelity audit (run 3, about 700 games, 215 of about 4,500 forks)
Callers: DamageStepProbe 160, DirectedFork 31, ProtectionFork 23. Entries marked † match a finding above.

| Field | Forks | Notes |
|---|---|---|
| `entity.lines` | 78 | e.g. a transformed Tovolar's Magehunter (F8†) |
| `link.affects` live only | 52 | Barbarian Class level (F5†) |
| dungeon entity, owns, controls | 37 | F3† |
| `linked_by_text` | 35 + 17 | Experimental Synthesizer trigger index 0 vs 1; unresolved |
| designations (bestowed, unearthed) | 31 | F5† |
| mutate: `merged_with`, `copy_source`, `copy_of` | 31, 31, 16 | F4† |
| `floating_effects` and `created_by` residue | 24, 20 | Experimental Synthesizer |
| `player.untapped_production` | 23 | a colour live counted as C in the copy; unresolved (`markedColors`? a land's chosen colour?) |
| `pt_boosts` and `granted_temporary` | 17, 12 | Robot token +1/+0, Magehunter trample; F8† or an uncopied condition (F5) |
| `chosen` / `chose` | 13 | F6† |
| CROAK counter on an exiled Forest | 4 | F6† |
| `blocked_this_turn` | 2 | F11† |

Statics are re-applied before the audit and before the probe snapshot (`makeCopy` → `checkStateEffects(true)`,
GameCopier 201), so these are tables or conditions the copier doesn't carry, not timing.
