# Forge `effect-record-hooks` review (2026-11-09): verified findings

Two fresh reviewers read the branch's changes against stock Forge (merge base `53a10372`):
- H1: the engine hooks;
- H2: the copy support.

A separate adversarial verifier then re-traced each slice's findings. Line numbers are as of Forge HEAD
04e72e0518e and connector HEAD 71c7e03b; re-check them before editing.

Paths:
- Forge paths are under `../forge/forge-game` or `../forge/forge-ai` unless noted;
- connector paths are in `forge-connector/src/main/java/com/pricepredictor/connector/effects/`.

Apart from the hooks, the branch's changes keep game behaviour, by H1's check of every changed line
outside listener calls:
- `AiController.canPlayAndPayFor`;
- `collectNonStaticTriggers`;
- Combat's `Boolean.TRUE.equals(band.isBlocked())` (changes only a former NPE path);
- the `Cause` added to Destroyed run params;
- the Card type, colour and keyword refactors;
- `cantAttackStatic` and `cantBlockByStatic`.

No hook draws from `MyRandom`. The trigger, replacement and combat hooks do not fire on AI what-if
queries.

---

## H1: engine hooks

### H1-2: a rewrite record's snapshot already shows the replacement's substitute. CONFIRMED (rank 1)
- **Mechanism:**
  - `ReplacementObserver.observe` (`replacement/ReplacementObserver.java` 94-101) calls the listener in
    `finally`, after `call.run(trace)`.
  - `call.run` has already run `playSpellAbilityNoStack(effectSA)` (`ReplacementHandler.java` 399-407).
  - The connector renders the snapshot inside the listener: `RewriteChannel` 117 →
    `LiveGame.traitRecord` 183 → `snapshots.ofTrait(...).render()`.
- **Concrete leaks:**
  - taplands (`dismal_backwater.txt`: `ReplaceWith$ ETBTapped` → `TapEffect` 74-76 sets tapped before the
    move, so the land in hand renders `tapped: true`);
  - Rest in Peace and other exile-instead effects: the subject is already in exile while `pending_event`
    says it is going to the graveyard;
  - Clone "enter as a copy": the host is already a copy;
  - regeneration: the creature is already tapped with its damage removed;
  - "as it enters, choose": the chosen value is already set.
- **Contract:** violates delta 253 ("its snapshot is the board just before it") and the `ofTrait`
  Javadoc ("about to handle").
- **Reaches training:** the state of every rewrite record that ran a `ReplaceWith$`. Rewrites are 7% of
  the curated mix (`sampling.py` 48). Their verdict target `REPLACEMENT_RESULT` (`aux_head_targets.py`
  88) is given away for optional or conditional replacements (Clone, dredge, "you may").
- **Frequency:** common.
- **Fix (Forge plus connector, game behaviour unchanged):**
  - add a before-callback `onReplacing(effect, runParams)` in `observe`, fired before `call.run`;
  - the connector snapshots there, keeping a stack keyed per call, because replacements nest (an
    exile-instead runs inner Moved replacements);
  - the record moments principle (root spec § 8.14) applies.
- **Test:** play a gain land with the listener installed; the rewrite record's entity for the land has
  `tapped: false` (today `true`).

### H1-4: two `fired=true` trigger records for a once-per-turn trigger that ran once. CONFIRMED (low frequency)
- **Mechanism:**
  - `collectNonStaticTriggers` → `canRunTrigger` (`trigger/TriggerHandler.java` about 451-458) notifies
    `fired = canRunTriggerUnobserved`, which includes `checkActivationLimit` (472), for every waiting
    event in the batch before any runs.
  - `runWaitingTrigger` 390-393 later skips a collected trigger when `!checkActivationLimit()`, because
    `triggerRun()` (611) was counted by the first.
  - `TriggerChannel` 47-66 writes a record per notification.
- **Scope:** per-object modes only: ChangesZone (23 scripts), Sacrificed (7), CounterAdded (4) and
  similar. ChangesZoneAll fires one event per batch.
- **Reaches training:** a false-positive `fired` label on the second simultaneous event (two creatures
  dying in one combat).
- **Fix:** count the already-collected positives (across all waiting events) and report the flag from
  that count, without changing Forge's collected list. Changing the list is not behaviour-preserving:
  `runSingleTrigger` can return before `triggerRun()` (about 577).
- **Test:** a board wipe kills two creatures with a once-per-turn "whenever a creature dies" line in
  play; exactly one trigger record has `fired = true`.

### H1-1: no `blockers` record for attackers attacking a battle. CONFIRMED (missing coverage, MOM only)
- **Mechanism:**
  - `combat/CombatLegalityObserver.java` 73 iterates `combat.getDefendersControlledBy(defending)`, which
    matches `getController() == who` (`Combat.java` 217-226).
  - A Siege is controlled by the attacker. `isPlayerAttacked` (987-1001) and `getDefenderPlayerByAttacker`
    (295-312) map a battle to its protector, so the step runs but the battle's attackers never reach the
    listener.
  - The AI does defend battles (`AiBlockController` 164-165).
  - The connector's only source of blockers records is this callback (`PlayabilityChannel` 180-186,
    197-222).
- **Reaches training:** no `legal_blockers`, `forbidden`, `min_blockers` or `must_block` for attacks on
  battles. 36 battle scripts, nearly all MOM.
- **Fix (pure observation):** iterate `combat.getAttackers()` filtered by
  `getDefenderPlayerByAttacker(a) == defending`.
- **Test:** an opponent attacks a Siege; one blockers record per attacker of the battle.

### H1-3: countering an ability writes `spell_countered` on the ability's host permanent. CONFIRMED (rare)
- **Mechanism:** `ability/effects/CounterEffect.java` 101-108 notes `spellCountered(sa, tgtSACard)`
  whatever `tgtSA` is, with `tgtSACard = tgtSA.getHostCard()`, a permanent. `effect_targets.py` 60-64 maps
  `SPELL_COUNTERED` to `ZoneOutcome.COUNTERED` on each subject. This contradicts delta 259.
- **Scope:** 39 scripts counter abilities, e.g. Siren Stormtamer, Tale's End, Tishana's Tidebinder,
  Overcharged Amalgam, Consign to Memory, Not of This World.
- **Fix:** note only when `tgtSA.isSpell()`.
- **Test:** a Stifle-style counter on an ETB trigger; no `spell_countered` names the battlefield host.

### H1-5: the outcome-hook guard leaves parameter building outside it. PARTLY (latent)
- **Mechanism:** `ability/EffectRecordOutcomes.note` 94-101 builds params outside `contain`, so a null in
  `Map.of` would throw into the resolution (never observed). `batchesOf` runs with no listener
  (`cardMade` calls `getPaperCard().getName()` on every made card), so the Javadoc at 51 is false.
- **Fix:** move `params.get()` inside `contain`, and guard batching with `isObserved()`.
- **Not defects:**
  - `sa = null` outcomes (forks are told apart by thread-locals, and no path delivers another game's
    events);
  - cleanup-step state-based deaths (counted as dropped, the same as in-loop deaths with no held
    resolution);
  - Clash-only reveals (accepted by the contract);
  - optional choices bypassing the confirm hook (a declined replacement is `not_replaced`; negligible);
  - `spellCopied` extra copies (rare, not an input).

### Dead `AiController` playability hook. CONFIRMED (code quality)
`AiController.java` 105-128 and 856-866: nothing installs `setEffectRecordPlayabilityListener`, and it is
not in `PatchHooks.REQUIRED`. The restructured `canPlayAndPayFor` behaves as stock. Revert both to stock.

### Where the connector's attribution findings should be fixed (verifier's verdicts)
- **A2 (a trigger's clause hook fires twice) → Forge: sound and complete.**
  - In `resolveApiAbility` (`ability/AbilityUtils.java` about 1438), skip the notification when
    `sa.isWrapper()`, but keep the scope.
  - The inner ability reaches the hook on every path: `WrappedAbility.resolve` 441 →
    `PlayerControllerAi.playSpellAbilityNoStack` 727-730 → `ComputerUtil.playNoStack` 251-268. This holds
    for the AI, `RandomSeatController`, static triggers through `playTrigger`, and forks.
  - Wrapper early returns then fire nothing, which is correct.
  - Update `EffectRecordClauseHookIntegrationTest`, and run connector X1's test alongside.
- **A3 (clauses that did not run) → Forge: sound, but complete it.**
  - Evaluate `metConditions()` once at its stock position (not ahead of the GIFT `runTrigger(GiveGift)`,
    about 1467).
  - Pass `ran` to `onClauseResolved`. `ran` must also be true for `handleUnlessCost`'s `cost == null`
    path (about 1497-1500), which the formula `alreadyPaid == isSwitched` misses.
  - The connector's `ConditionsMet.of` then stops calling `metConditions()` a second time
    (`ClauseOutcomeRouter` 106-109, which can draw `Count$Random`), and skips `finish()` when `!ran`.
  - WrappedAbility early returns are covered by A2.
- **A1 (charm modes' events in the last mode's half) → Forge: sound.**
  - Fire RESOLVED after the clause's own effect and before `resolveSubAbilities`, in the normal path
    and both unless-cost paths, with RESOLVED in a `finally` around the clause's own effect only.
  - `CharmEffect.resolve` is empty (197-200), so mode chains become sequential.
  - This also removes diff-rule double counting in ordinary P→S chains.
  - Effects that resolve sub-clauses inside their own `resolve()` (Repeat, Branch, GenericChoice) stay
    nested; that is acceptable.
  - The connector fallback (pick the half at RESOLVING) is also sound.
- **C2 (the mana record's board) → a new Forge hook at the top of `CostPayment.payComputerCosts`
  (`cost/CostPayment.java` 178): sound placement, incomplete pairing.**
  - It covers AI payment (`ComputerUtilMana` 770-776), the random seat (`RandomSpellPlayer` 120-122 →
    `AiCostDecision`), `handlePlayingSpellAbility`, `playNoStack` and forks. `predictMana` does not call
    it.
  - The connector must:
    - discard the pending snapshot when payment fails after the hook;
    - pair by mana-ability identity, because activations nest (a filter land paying its own {1} with
      another land);
    - give triggered mana abilities (TapsForMana: Wild Growth, Utopia Sprawl), which produce with no
      activation, a fallback.
- **C3 (mana made by a resolving spell) → connector: sound.** Filter on `isManaAbility()` (CR 605.1b via
  `SpellAbility.isManaAbility()` 393ff), and file Dark Ritual-style `mana_produced` into the resolution's
  effect half.

### Code quality (H1)
- The clause-hook contract is implicit. Hand a ran/skipped/threw value to `onClauseResolved` (A1-A3 in
  one place).
- `CombatLegalityObserver` re-derives who blocks whom from controller equality. Add a `Combat` method
  for the attackers a player may block.
- Hook slots are scattered across nine classes. Use one `EffectRecordHooks` index mirroring
  `PatchHooks.REQUIRED`.
- The trigger "fired" flag is `canRunTrigger`'s answer: document or rename it.

---

## H2: copy support

`GC` = `forge-ai/src/main/java/forge/ai/simulation/GameCopier.java`. Two facts decide what a gap costs:
- A copier crash loses the fork, never the worker: `Forks.copyAndVerify` (connector `Forks.java` 239-256)
  swallows any RuntimeException.
- `GameEntity.equals` compares id and class (about 393), so `cardMap` lookups are by id. A stack card
  and the graveyard card it became map to the same counterpart, which narrows the "host gone" cases.

Sound:
- `CardDamageHistory.copyFrom`, `Zone.copyCardsAddedFrom`, Combat's band copy, `copyMustBlockCardsFrom`,
  `copyControllersFrom`;
- the delayed-trigger lists (no double registration) and the delayed-trigger id reuse;
- the face-down timestamp restore (GC 286-287).

These overlap the connector review's slice 4 and are not repeated here:
- Cryptic Spires is F6 (`markedColors`);
- the phase-handler fields are F12;
- the remembered-object throw is F7's mechanism.

### H2-1: per-card activation, resolution and chosen-mode tables are not copied. CONFIRMED (rank 1, all sets)
- **Mechanism:**
  - The tables are `card/Card.java` 367-379: `numberTurnActivations`, `numberGameActivations`,
    `numberAbilityResolved` and the `chosenModes*` maps.
  - Neither `Card.copyFrom` (8270-8292) nor GC copies them, except the planeswalker loop (GC 899-911,
    which also misses `planeswalkerActivationLimitUsed`). `CardCopyService` 402-404 copies them on the LKI
    path only.
- **Readers:**
  - `Trigger.getActivationsThisTurn` / `getActivationsThisGame` (607-613);
  - `checkResolvedLimit` (352-358);
  - `SpellAbility` 826-834;
  - AI logic (`SpellAbilityAi` 163, `EffectAi` 391, …);
  - the connector's `PlayabilityChecks` 332-361 (boast, exhaust, power-up limits).
  `ForkFidelity` cannot see any of them.
- **Scope:** trigger `ActivationLimit$` 144 scripts, `GameActivationLimit$` 7, activated
  `ActivationLimit$` 161, `ResolvedLimit$` 33, `ChoiceRestriction$` 27, every boast (KHM) and exhaust
  (DFT) card.
- **Effect:**
  - An "only once each turn" trigger or ability that already fired this turn fires again in the fork.
  - Damage-step probes: a once-per-turn combat-damage trigger from the first-strike step fires again in
    the regular-step probe.
  - Play-forwards reuse limited abilities, and their playability records call them playable.
- **Fix:** copy the tables with their keys mapped. They are keyed by the original ability object
  (`ActivationTable.getOriginal`), so this needs the structural ability mapper below.
- **Test:** a creature with `A:AB$ Pump | … | ActivationLimit$ 1`; activate, resolve and copy. The copy's
  ability has `getActivationsThisTurn()==1` and `canPlay()` false.

### H2-MAP: abilities are mapped into the copy by parameter equality, so traits sharing an Execute SVar map to the first one. CONFIRMED (rank 2, all sets)
- **Mechanism:** `counterpartAbility` (GC 734-739) returns the first ability whose API and `mapParams`
  match. Experimental Synthesizer's two triggers both `Execute$ TrigDig`, so trigger 1's ability maps to
  trigger 0's. The same wrong match reaches:
  - `copyExileLinks` (GC 429-430);
  - `copyEffectSource` (GC 356-357);
  - GameCommandCopier commands capturing an ability (GC 687-688).
- **Effect:** fork snapshots name the wrong line (`linked_by_text` index, `created_by`,
  `floating_effects` source lines), as in the audit's Synthesizer rows. Behaviour barely changes, since
  the params are identical, except where a lookup is by identity. Every card whose traits share an
  Execute SVar ("enters or leaves" pairs) is affected.
- **Fix:** one structural mapper (the root's role, its spell-ability or trigger index on the host, then
  the same sub-ability depth). Replace `counterpartAbility`, `findSAInCard` (GC 977) and the connector's
  `Forks.findInFork` (338-355) with it.
- **Test:** make Experimental Synthesizer leave the battlefield, then copy. The exiled card's
  `exiledSA.getRootAbility().getTrigger()` is the host's trigger index 1.

### H2-4: damage dealt this turn through objects that changed zones is lost. CONFIRMED (low impact)
- **Mechanism:** `Game.copyTurnHistoryFrom` (1394-1403) keeps only mapped cards' histories. A zone change
  creates a new Card with a fresh history (`GameAction` 208; only LKI copies share it, `CardCopyService`
  320). So damage this turn by dead or bounced creatures and by resolved spells is dropped from
  `globalDamageHistory`.
- **Readers:** `Count$…DamageThisTurn` (about 12 scripts), `Player.wasDealt…` (8, `PlayerProperty` 107,
  166), prowl (10), `SingleMaxDamageThisTurn`. Unaffected: bloodthirst, `getAssignedDamage`
  (`damageThisTurnLKI` is complete) and the connector's `SnapshotLinks`.
- **Fix:** carry unmapped histories as they are; they are read-only and keyed by identity.
- **Test:** Shock the opponent, copy, and compare `getDamageDoneThisTurn(null,false,"Card","Player",…)`
  (live 1, copy 0 today).

### H2-2: copied effect-card traits keep the live card's state. CONFIRMED in mechanism (rare)
- **Mechanism:** GC 804/807/821/824 call `copy(newCard, true)`. `copyHelper` (`CardTraitBase` 719-727)
  keeps `cardState` whatever `lki` is, and `getSVarFallback` (538-552) reads it before the host.
- **Confirmed effects:**
  - A "prevent the next N" shield reads `ShieldAmount` live (`DamagePreventEffect` 131-134,
    `ReplaceDamageEffect` 37-53) and writes the copy's, so it resets on every hit after the first. A
    shielded attacker blocked by two creatures in a probe shows this.
  - `OriginalHost` resolves to the live card (`CardTraitBase.getOriginalHost` 629-632, 33 scripts).
  - `replacingObjects` is shared with the live ability (`SpellAbility.copy` resets it only when `!lki`,
    1250-1252; written by `ReplacementHandler` 340-345 and `ReplaceDamage` 143-147). The live game
    rewrites the same keys before reading them; only optional keys can go stale.
- **Refuted:** `hasRun` and `otherChoices` are shared (a primitive, and a transient).
- **Same pattern:** battlefield traits in `setChangedCardTraits`, `setChangedCardKeywords` and clone
  states (`Card` 5146, 4304; GC 915).
- **Fix:** `copy(newCard, false, true)`, then `setCardState(newCard.getCurrentState())` on each trait and
  its overriding ability, as `copyDelayedTrigger` does.

### H2-9: `copyEffectSource`'s fallback is a live ability, so `getEffectSource()` returns a live card. CONFIRMED (rare)
- **Mechanism:** `Defined$ EffectSource` then acts on a live card. `handleExiledWith`
  (`SpellAbilityEffect` 1063-1074; `ExiledWithEffectSource`, 10 scripts) can call `addExiledCard` on the
  live card, corrupting the live game. It needs a host with no counterpart (a dead token, a copied
  spell).
- **Fix:** keep the copy's `effectSource` Card when there is no counterpart (and use H2-MAP).

### H2-8: resolved-effect layer rows `Card.copyFrom` misses. CONFIRMED (rare; play-forwards)
- **Rows:**
  - `canBlockAdditional` and `canBlockAny` (`PumpEffect` 79/83, 9 scripts);
  - `cantHaveKeywords` (`AnimateEffectBase` 137, 6);
  - `staticCommandList` (`ControlGainEffect` 203-206, 2);
  - `changedCardManaCost` (perpetual, Alchemy);
  - the text-change rows.
- Statics re-create their own rows; only rows written by resolved effects are lost.

### H2-3: floating mana shared with the live game. CONFIRMED (near zero in sealed)
- **Mechanism:** GC 140-142 shares `Mana` objects whose `AbilityManaPart` holds the live `sourceCard`.
  `createETBCountersEffect` (`GameActionUtil` 797-835) runs through the live `GameAction.changeZone`
  (110-116):
  - it consumes a live card id and timestamp;
  - it runs the live `checkStaticAbilities()`;
  - it registers a fork card's triggers in the live `TriggerHandler`;
  - it fires on the live bus.
  The card itself lands in the fork's Command zone. `addTriggersWhenSpent` (`AbilityManaPart` 362-369)
  registers a live-hosted trigger in the fork.
- **Scope:** 18 scripts, and only when the AI is floating that mana at the fork.
- **Fix:** rebuild each Mana with its source mapped, or refuse pools holding special mana.

### H2-11: a remembered SpellAbility makes the copy throw. CONFIRMED (rare)
`find()` (GC 1005-1024) throws for a `SpellAbility` (a `GameObject`), so the fork is lost. 1
`RememberCounteredSA` script and about 6 that remember a targeted spell. Same mechanism as F7.

### Negligible in sealed
- H2-5: `MagicStack.thisTurnActivated` not copied (12 scripts; cycling FirstTime).
- H2-6: `spellsCastThisGame` counted twice (`copyTurnCountsFrom` plus GC 131-133; 2 scripts read it).
  Delete the loop.
- H2-7: a second Radiation card with a colliding id (PIP only).
- H2-10:
  - `Combat.lkiCache` not copied (no crash; low);
  - waiting triggers and the simultaneous-entry list not copied (low);
  - `hasBeenDealtDeathtouchDamage` (nil);
  - Sideboard (REFUTED: decks have no sideboard);
  - GameCommandCopier rebuilding a captured `Zone` by value (NEEDS-TEST; no capturing command found).

### Still unexplained: the Robot token's +1/+0 and haste. NEEDS-TEST
No mechanism was found by reading (the token path carries the boost and keyword rows, and `PumpAllEffect`
78-91's command captures only mappable objects).
- **Test:** on a TestBoard, Kavaron, Memorial World with 12 charge counters; activate, resolve, copy.
- **Assert:** the copy's Robot has the same `getNetPower()`, `hasKeyword(HASTE)`, `getPTBoostTable()`
  cells and `getChangedCardKeywords()` keys as the live one. Do the same for another creature.

### Code quality (H2)
- Three parameter- or description-based ability mappers (`counterpartAbility`, `findSAInCard`,
  `Forks.findInFork`) should become one structural mapper (H2-MAP).
- Every "no counterpart" path drops silently (commands, delayed triggers, remembered objects, links,
  attacks). Add a per-copy tally that `ForkFidelity` prints.
- One count has two owners (H2-6).
- Live traits are built from lki copies (H2-2).
- Card state copying is battlefield-only (connector F5/F6). Write one `copyCardState(orig, copy)`.
