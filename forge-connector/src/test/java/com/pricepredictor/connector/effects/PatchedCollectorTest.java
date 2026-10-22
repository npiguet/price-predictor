package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.TestCollectors.caps;
import static com.pricepredictor.connector.effects.TestCollectors.recording;
import static com.pricepredictor.connector.effects.TestCollectors.triggerOfMode;
import static com.pricepredictor.connector.effects.TestCollectors.unwired;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What the patched collectors decide around the records they build: which
 * records coalesce, what a continuous contribution carries, when a held
 * probe branch is written, which hooks get installed, and which game a
 * trigger, playability, combat or mana report is admitted from.
 *
 * <p>The record builders have tests of their own: {@link TriggerRecordTest},
 * {@link RewriteContractTest} and {@link OutcomeHandlerTest}.
 */
@ExtendWith(ForgeExtension.class)
class PatchedCollectorTest {

    @TempDir
    Path tempDir;

    private static final Method ON_CONDITION_EVALUATED =
            HookShapes.method(HookShapes.TriggerFire.class, "onConditionEvaluated");
    private static final Method ON_CANDIDATE =
            HookShapes.method(HookShapes.Playability.class, "onCandidate");
    private static final Method ON_ATTACKERS_DECLARABLE =
            HookShapes.method(HookShapes.CombatLegality.class, "onAttackersDeclarable");
    private static final Method ON_BLOCKERS_DECLARABLE =
            HookShapes.method(HookShapes.CombatLegality.class, "onBlockersDeclarable");
    private static final Method ON_MANA_PRODUCED =
            HookShapes.method(HookShapes.Mana.class, "onManaProduced");

    // ── the snapshot depth is the same on every record ─────────────────

    @Test
    void everySnapshotCarriesAllFourTiers() {
        String json = recording(tempDir).triggers().record(
                triggerOfMode("ChangesZone"), Map.of(), true).toJson();

        assertTrue(json.contains("\"tiers\":[1,2,3,4]"), json);
    }

    // ── installing ──────────────────────────────────────────────────────

    /**
     * Exactly the listener hooks the checkout offers get installed, and no
     * others -- and none at all in degraded mode, where install() declines
     * before it looks.
     *
     * <p>Asserted against the checkout rather than against a fixed count: the
     * sibling Forge is patched or not independently of this repository. The
     * expected set is derived from {@link PatchHooks#REQUIRED}, so a listener
     * hook added there is expected here without editing the test. The confirm
     * listener is excluded because the recording installs it with the
     * bracket's handler, not this class.
     */
    @Test
    void everyHookTheCheckoutOffersIsInstalledAndNoOthers() {
        long expected = AttributionMode.detect() == AttributionMode.DEGRADED
                ? 0
                : PatchHooks.REQUIRED.stream()
                        .filter(hook -> hook.method().startsWith("setEffectRecord")
                                && hook.method().endsWith("Listener")
                                && !PatchHooks.PLAYER_CONTROLLER_AI.equals(hook.owner()))
                        .filter(PatchHooks.Hook::present)
                        .count();
        try (PatchedCollectors collector = unwired(CollectionCaps.defaults())) {
            assertEquals(expected, collector.install());
        }
    }

    // ── final-fix-3.md item 2: the other four handlers ───────────────────

    /**
     * final-fix-3.md item 2: {@code TriggerChannel.handler} had no game check at
     * all before this fix. A fork's own trigger evaluations
     * ({@code GameSimulator.resolveStack} runs the forced ability's triggers
     * through the real {@code TriggerHandler}) must not reach the live shard.
     */
    @Test
    void aTriggerEvaluationFromAnotherGameDoesNotReachTheShard() throws Throwable {
        Card forkCard = TestBoard.cardInAnotherGame("Paralyze");
        Trigger forkTrigger = forkCard.getCurrentState().getTriggers().get(0);

        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.triggers().handler().invoke(null, ON_CONDITION_EVALUATED,
                        new Object[]{forkTrigger, Map.of(), true}));

        assertFalse(Files.exists(shard),
                "a fork's own trigger evaluation must not land in the live shard");
    }

    /**
     * final-fix-3.md item 2: {@code playabilityHandler} had no game check
     * either. {@code caps(0, 1.0)} forces {@code playabilityRate} to 1.0 so
     * the sampling gate ahead of the game check cannot flakily hide it.
     */
    @Test
    void aPlayabilityCandidateFromAnotherGameDoesNotReachTheShard() throws Throwable {
        Card forkCard = TestBoard.cardInAnotherGame("Grizzly Bears");
        SpellAbility forkCandidate = AbilityFactory.getAbility(
                "AB$ Pump | Cost$ 1 | NumAtt$ +1 | NumDef$ +1", forkCard);

        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), caps(0, 1.0),
                (bracket, collectors) -> collectors.playability().decisionHandler().invoke(null, ON_CANDIDATE,
                        new Object[]{forkCandidate}));

        assertFalse(Files.exists(shard),
                "a fork's own playability candidate must not land in the live shard");
    }

    /**
     * {@code combatLegalityHandler} checks the game the defender belongs to.
     * The defender is the acting {@code GameEntity} this hook hands over; the
     * candidate/legal lists may be any real card, since only the defender's
     * own game identity is what the check reads.
     */
    @Test
    void anAttackerLegalityAnswerFromAnotherGamesDefenderDoesNotReachTheShard() throws Throwable {
        Player forkDefender = new Player("fork-defender", TestBoard.anotherGame(), 92301);
        Card candidate = TestCards.build("Grizzly Bears");

        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.playability().legalityHandler().invoke(null, ON_ATTACKERS_DECLARABLE,
                        new Object[]{null, forkDefender, List.of(candidate), List.of(candidate)}));

        assertFalse(Files.exists(shard), "another game's defender must not land in the live shard");
    }

    /**
     * final-fix-3.md item 2: {@code manaHandler} had no game check either.
     * Held in {@link ManaChannel}'s reservoir rather than written
     * immediately, so the collector must be closed (which flushes it) before
     * the shard can be read.
     */
    @Test
    void manaProducedByAnotherGamesAbilityDoesNotReachTheShard() throws Throwable {
        Card forkCard = TestBoard.cardInAnotherGame("Llanowar Elves");
        SpellAbility forkAbility = AbilityFactory.getAbility(
                "AB$ Mana | Cost$ T | Produced$ G", forkCard);
        Player forkPlayer = new Player("fork-player", forkCard.getGame(), 92302);

        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.mana().handler().invoke(null, ON_MANA_PRODUCED,
                        new Object[]{forkAbility, forkPlayer, "G"}));

        assertFalse(Files.exists(shard),
                "a fork's own mana production must not land in the live shard");
    }

    // ── final-fix-4.md item 3: a positive test behind each of those gates ──
    //
    // The trigger handler, playabilityHandler, combatLegalityHandler and
    // manaHandler are invoked nowhere in the test tree except the four
    // fork-rejection tests above -- the existing record(...)-style
    // tests call the record builder directly and never reach the handler
    // (InvocationHandler) these gates actually live in. A gate with only a
    // negative test behind it can fail closed for everything, live included,
    // and every one of those tests would still pass. rewriteHandler and
    // confirmHandler have positive coverage in RewriteContractTest and
    // BusBracketCollectorTest respectively, which drive the live handler
    // directly, so only these four need one here.

    @Test
    void aLiveTriggerEvaluationReachesTheShard() throws Throwable {
        Card liveCard = TestCards.build("Paralyze");
        Trigger liveTrigger = liveCard.getCurrentState().getTriggers().get(0);

        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.triggers().handler().invoke(null, ON_CONDITION_EVALUATED,
                        new Object[]{liveTrigger, Map.of(), true}));
        assertTrue(lines.stream().anyMatch(l -> l.contains("\"kind\":\"trigger\"")),
                "a live trigger evaluation must reach the shard: " + lines);
    }

    @Test
    void aLivePlayabilityCandidateReachesTheShard() throws Throwable {
        Card liveCard = TestCards.build("Grizzly Bears");
        SpellAbility liveCandidate = AbilityFactory.getAbility(
                "AB$ Pump | Cost$ 1 | NumAtt$ +1 | NumDef$ +1", liveCard);

        // playabilityRate forced to 1.0, matching final-fix-4.md item 5's own
        // fix: at the default 0.1 this record would be sampled out at random
        // roughly nine times in ten, and a flaky pass here is exactly as
        // uninformative as the vacuous one item 3 exists to close.
        List<String> lines = ShardRun.records(tempDir, TestCards.game(), caps(0, 1.0),
                (bracket, collectors) -> collectors.playability().decisionHandler().invoke(null, ON_CANDIDATE,
                        new Object[]{liveCandidate}));
        assertTrue(lines.stream().anyMatch(l -> l.contains("\"kind\":\"playability\"")),
                "a live playability candidate must reach the shard: " + lines);
    }

    @Test
    void aLiveAttackerLegalityAnswerReachesTheShard() throws Throwable {
        Player liveDefender = new Player("live-defender", TestCards.game(), 93101);
        Card candidate = TestCards.build("Grizzly Bears");

        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.playability().legalityHandler().invoke(null, ON_ATTACKERS_DECLARABLE,
                        new Object[]{null, liveDefender, List.of(candidate), List.of(candidate)}));
        assertTrue(lines.stream().anyMatch(l -> l.contains("\"subkind\":\"attackers\"")),
                "a live defender's legality answer must reach the shard: " + lines);
    }

    @Test
    void aLiveManaProductionReachesTheShard() throws Throwable {
        SpellAbility liveAbility = AbilityFactory.getAbility(
                "AB$ Mana | Cost$ T | Produced$ G", TestCards.build("Llanowar Elves"));
        Player livePlayer = new Player("live-player", TestCards.game(), 93102);

        // manaHandler holds records in the reservoir rather than writing them
        // immediately (ManaChannel); only ManaChannel.flush(),
        // called when the run closes the collectors, writes them out.
        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.mana().handler().invoke(null, ON_MANA_PRODUCED,
                        new Object[]{liveAbility, livePlayer, "G"}));
        assertTrue(lines.stream().anyMatch(l -> l.contains("mana_produced")),
                "live mana production must reach the shard: " + lines);
    }
}
