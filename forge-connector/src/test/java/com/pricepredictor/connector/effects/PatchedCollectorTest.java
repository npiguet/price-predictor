package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.TestCaps.caps;
import static com.pricepredictor.connector.effects.TestCollectors.recording;
import static com.pricepredictor.connector.effects.TestCollectors.triggerOfMode;
import static com.pricepredictor.connector.effects.TestCollectors.unwired;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What the patched collectors decide across their channels: which hooks get
 * installed, how deep a snapshot goes, and which game a trigger, playability,
 * combat or mana report, or a state-based-loop or priority-settled signal, is
 * admitted from.
 *
 * <p>The channels have tests of their own: {@link TriggerRecordTest},
 * {@link RewriteContractTest}, {@link OutcomeHandlerTest},
 * {@link PlayabilityChannelTest}, {@link PriorityCandidatesTest} (which game a
 * priority stop's decision records come from) and {@link ContinuousChannelTest}.
 */
@ExtendWith(ForgeExtension.class)
class PatchedCollectorTest {

    @TempDir
    Path tempDir;

    private static final Method ON_CONDITION_EVALUATED =
            HookShapes.method(HookShapes.TriggerFire.class, "onConditionEvaluated");
    private static final Method ON_ATTACKERS_DECLARABLE =
            HookShapes.method(HookShapes.CombatLegality.class, "onAttackersDeclarable");
    private static final Method ON_MANA_PRODUCED =
            HookShapes.method(HookShapes.Mana.class, "onManaProduced");
    private static final Method ON_PAYING_COSTS =
            HookShapes.method(HookShapes.Payment.class, "onPayingCosts");
    private static final Method ON_COSTS_PAID =
            HookShapes.method(HookShapes.Payment.class, "onCostsPaid");
    private static final Method ON_STATE_BASED_LOOP_BEGIN =
            HookShapes.method(HookShapes.StateBasedLoop.class, "onStateBasedLoopBegin");
    private static final Method ON_STATE_BASED_LOOP_END =
            HookShapes.method(HookShapes.StateBasedLoop.class, "onStateBasedLoopEnd");
    private static final Method ON_PRIORITY_SETTLED =
            HookShapes.method(HookShapes.PrioritySettled.class, "onPrioritySettled");

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
     * others -- and in degraded mode only the confirm listener, the one whose
     * channel needs nothing the mode is decided by.
     *
     * <p>Asserted against the checkout rather than against a fixed count: the
     * sibling Forge is patched or not independently of this repository. The
     * expected set is derived from {@link PatchHooks#REQUIRED}, so a listener
     * hook added there is expected here without editing the test. The confirm
     * listener is counted apart because it is installed in either mode.
     */
    @Test
    void everyHookTheCheckoutOffersIsInstalledAndNoOthers() {
        long confirm = PatchHooks.CONFIRM_LISTENER.present() ? 1 : 0;
        long expected = confirm + (AttributionMode.detect() == AttributionMode.DEGRADED
                ? 0
                : PatchHooks.REQUIRED.stream()
                        .filter(hook -> hook.method().startsWith("setEffectRecord")
                                && hook.method().endsWith("Listener")
                                && hook != PatchHooks.CONFIRM_LISTENER)
                        .filter(PatchHooks.Hook::present)
                        .count());
        // A patched checkout whose hooks all went unfound would agree with a
        // collector that installs nothing.
        if (AttributionMode.detect() == AttributionMode.PATCHED) {
            assertTrue(expected > 0, "a patched checkout offers no listener hook: "
                    + PatchHooks.report());
        }
        try (PatchedCollectors collector = unwired(CollectionCaps.defaults())) {
            assertEquals(expected, collector.install());
        }
    }

    // ── the other three handlers ────────────────────────────────────────

    /**
     * {@code TriggerChannel.handler} checks the game too. A fork's own trigger evaluations
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
     * {@code PlayabilityChannel.legalityHandler} checks the game the defender
     * belongs to.
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
     * {@code ManaChannel.handler} checks the game too.
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
                activate(collectors, forkAbility, forkPlayer, true));

        assertFalse(Files.exists(shard),
                "a fork's own mana production must not land in the live shard");
    }

    // ── a positive test behind each of those gates ──────────────────────
    //
    // The trigger, decision, legality and mana handlers are invoked nowhere
    // in the test tree except the four fork-rejection tests above and these
    // -- the record(...)-style tests call the record builder directly and
    // never reach the handler (InvocationHandler) these gates actually live
    // in. A gate with only a negative test behind it can fail closed for
    // everything, live included, and every one of those tests would still
    // pass. The rewrite handler and
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

        // The mana handler holds records in the reservoir rather than writing
        // them immediately; only ManaChannel.flush(), called when the run
        // closes the collectors, writes them out.
        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) ->
                activate(collectors, liveAbility, livePlayer, true));
        assertEquals(1, lines.size(), lines.toString());
        // The tap its cost made, then the mana (FR-042c).
        assertEquals(List.of("tapped", "mana_produced"), RecordLines.eventTypes(lines.get(0)),
                lines.get(0));
        assertEquals(List.of(Refs.of(liveAbility.getHostCard())),
                RecordLines.events(lines.get(0)).get(0).get("subjects"), lines.get(0));
    }

    /** A mana ability whose payment failed produces no mana and writes no record. */
    @Test
    void aFailedManaPaymentWritesNoRecord() throws Throwable {
        SpellAbility liveAbility = AbilityFactory.getAbility(
                "AB$ Mana | Cost$ T | Produced$ G", TestCards.build("Llanowar Elves"));
        Player livePlayer = new Player("live-player", TestCards.game(), 93103);

        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), (bracket, collectors) ->
                activate(collectors, liveAbility, livePlayer, false));

        assertFalse(Files.exists(shard), "a failed payment's mana record was written");
    }

    /**
     * A mana ability's activation as the engine tells it: its payment, then,
     * if it paid, its mana.
     */
    private static void activate(
            Channels collectors, SpellAbility ability, Player player, boolean paid)
            throws Throwable {
        var payment = collectors.mana().paymentHandler();
        payment.invoke(null, ON_PAYING_COSTS, new Object[]{ability, player, false});
        payment.invoke(null, ON_COSTS_PAID, new Object[]{ability, player, paid});
        if (paid) {
            collectors.mana().handler().invoke(null, ON_MANA_PRODUCED,
                    new Object[]{ability, player, "G"});
        }
    }

    // ── the phase signals ───────────────────────────────────────────────
    //
    // They write no record, so they are driven on a PhaseSignals of their
    // own, over a sink that keeps what reached it.

    /** The signals delivered to a sink, in order. */
    private static final class Delivered implements PhaseSignals.Sink {
        private final List<String> seen = new ArrayList<>();

        @Override
        public void stateBasedLoopBegan() {
            seen.add("begin");
        }

        @Override
        public void stateBasedLoopEnded() {
            seen.add("end");
        }

        @Override
        public void prioritySettled(Player priorityPlayer) {
            seen.add("settled " + priorityPlayer.getName());
        }
    }

    @Test
    void theLiveGameIsInsideItsStateBasedLoopFromItsBeginToItsEnd() throws Throwable {
        Delivered sink = new Delivered();
        PhaseSignals signals = new PhaseSignals(new LiveGame(TestCards.game()), sink);
        PatchHooks.Listener handler = signals.stateBasedLoopHandler();

        handler.invoke(null, ON_STATE_BASED_LOOP_BEGIN, new Object[]{TestCards.game()});
        boolean inside = signals.inStateBasedLoop();
        handler.invoke(null, ON_STATE_BASED_LOOP_END, new Object[]{TestCards.game()});

        assertTrue(inside, "between the begin and the end");
        assertFalse(signals.inStateBasedLoop(), "after the end");
        assertEquals(List.of("begin", "end"), sink.seen);
    }

    /** A fork's phase handler runs its own loop, through the same JVM-static hook. */
    @Test
    void aStateBasedLoopInAnotherGameLeavesTheLiveGameOutsideOne() throws Throwable {
        Delivered sink = new Delivered();
        PhaseSignals signals = new PhaseSignals(new LiveGame(TestCards.game()), sink);

        signals.stateBasedLoopHandler().invoke(null, ON_STATE_BASED_LOOP_BEGIN,
                new Object[]{TestBoard.anotherGame()});

        assertFalse(signals.inStateBasedLoop());
        assertEquals(List.of(), sink.seen);
    }

    @Test
    void priorityThatSettlesInAnotherGameIsNotDeliveredAndTheLiveGamesIs() throws Throwable {
        Delivered sink = new Delivered();
        PhaseSignals signals = new PhaseSignals(new LiveGame(TestCards.game()), sink);
        PatchHooks.Listener handler = signals.prioritySettledHandler();
        Game fork = TestBoard.anotherGame();

        handler.invoke(null, ON_PRIORITY_SETTLED, new Object[]{fork, new Player("fork-seat", fork, 92303)});
        handler.invoke(null, ON_PRIORITY_SETTLED,
                new Object[]{TestCards.game(), new Player("live-seat", TestCards.game(), 93103)});

        assertEquals(List.of("settled live-seat"), sink.seen);
    }
}
