package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.event.GameEventGameFinished;
import forge.game.event.GameEventGameOutcome;
import forge.game.event.GameEventSpellAbilityCast;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.pricepredictor.connector.effects.RecordAbilities.linkOf;
import static com.pricepredictor.connector.effects.RecordAbilities.optionOf;
import static com.pricepredictor.connector.effects.RecordAbilities.scriptOf;
import static com.pricepredictor.connector.effects.RecordEvents.eventTypes;
import static com.pricepredictor.connector.effects.RecordEvents.eventsNaming;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static com.pricepredictor.connector.effects.RecordMoves.deathsOf;
import static com.pricepredictor.connector.effects.RecordMoves.eventsOn;
import static com.pricepredictor.connector.effects.RecordMoves.holdsTheDeathOf;
import static com.pricepredictor.connector.effects.RecordMoves.moves;
import static com.pricepredictor.connector.effects.RecordPicks.only;
import static com.pricepredictor.connector.effects.RecordPicks.records;
import static com.pricepredictor.connector.effects.RecordPicks.withEventOfType;
import static com.pricepredictor.connector.effects.TestBoard.nextStep;
import static com.pricepredictor.connector.effects.TestBoard.spellOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A resolution's effect half carries the deaths its own effects cause.
 *
 * <p>Forge resolves an ability, fires {@code GameEventSpellResolved}, and only
 * at the next priority runs the state-based loop that moves a creature with
 * lethal damage to the graveyard. An effect half written at the resolved event
 * would record a Lightning Bolt at a 2/2 as the damage and never the death.
 *
 * <p>These drive the real engine — the stack, the resolution, and the
 * state-based loop a priority runs ({@link Duel#stateBasedLoop}) — with the
 * game recorded as a worker on the patched checkout records it ({@link
 * ShardRun#hooked}), because what is under test is the order in which the
 * engine publishes and signals, and a hand-fed event sequence would only
 * restate my reading of it. Two cases are about a worker on a stock
 * checkout, with no hook ({@link ShardRun#observed}).
 */
@ExtendWith(ForgeExtension.class)
class StateBasedDeathTest {

    private static final String WHY = "for the resolution starts and the state-based loop's signals";

    @TempDir
    Path tempDir;

    private static SpellAbility targeting(SpellAbility spell, Card target) {
        spell.resetTargets();
        spell.getTargets().add(target);
        return spell;
    }

    /** {@code name}'s spell from {@code duel}'s caster's hand, aimed at {@code target}. */
    private static SpellAbility castAt(Duel duel, String name, Card target) {
        Player caster = duel.caster();
        return targeting(spellOf(TestBoard.inHand(duel.game(), caster, name), caster), target);
    }

    /** The effect halves: the resolution records a cost half links to. */
    private static List<String> effectHalves(List<String> records) {
        return records(records, "resolution").stream()
                .filter(record -> linkOf(record) != null).toList();
    }

    /**
     * The one death of {@code card} in {@code record}, rendered as a
     * state-based action's: no clause produced it, and the line whose
     * resolution it follows, {@code cause}'s, is what caused it.
     */
    private static void assertAStateBasedDeath(String record, Card card, Card cause) {
        List<Map<String, Object>> deaths = deathsOf(record, card);
        assertEquals(1, deaths.size(), record);
        Map<String, Object> death = deaths.get(0);
        assertEquals(Refs.of(cause), params(death).get("cause"), record);
        assertEquals("instant", death.get("duration"), record);
        assertEquals("unresolved", death.get("attributed_to"), record);
    }

    /** Whether {@code record} deals {@code card} damage. */
    private static boolean damages(String record, Card card) {
        return !eventsOn(record, card, "damage_dealt").isEmpty();
    }

    // ── a lethal resolution ─────────────────────────────────────────────

    /**
     * The reproducer: Lightning Bolt at Grizzly Bears writes the damage and
     * the death in one effect half, on a worker with no clause hook.
     */
    @Test
    void aLethalBoltCarriesTheDeathItCausedInItsEffectHalf() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card bears = TestBoard.onBattlefield(game, duel.opponent(), "Grizzly Bears");
        SpellAbility bolt = castAt(duel, "Lightning Bolt", bears);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            duel.castAndResolve(bolt);
            assertTrue(inGraveyard(game, bears), "the bears died");
            nextStep(game);
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), halves.toString());
        String effect = halves.get(0);
        assertTrue(damages(effect, bears), effect);
        assertAStateBasedDeath(effect, bears, bolt.getHostCard());
    }

    /**
     * On a worker without the loop hook the effect half waits for the next
     * action and is written exactly once: here the next action is another
     * cast, whose own half waits in turn.
     */
    @Test
    void aHeldHalfIsWrittenOnceWhenTheNextActionBegins() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card bears = TestBoard.onBattlefield(game, duel.opponent(), "Grizzly Bears");
        Card giant = TestBoard.onBattlefield(game, duel.opponent(), "Hill Giant");
        SpellAbility bolt = castAt(duel, "Lightning Bolt", bears);
        SpellAbility shock = castAt(duel, "Shock", giant);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            duel.castAndResolve(bolt);
            duel.castAndResolve(shock);
            nextStep(game);
            nextStep(game);
        });

        assertEquals(4, records.size(), records.toString());
        List<String> halves = effectHalves(records);
        assertEquals(2, halves.size(), records.toString());
        assertTrue(holdsTheDeathOf(halves.get(0), bears), halves.get(0));
        assertEquals(List.of(), eventsNaming(halves.get(1), Refs.of(bears)), halves.get(1));
        assertFalse(holdsTheDeathOf(halves.get(1), giant), halves.get(1));
    }

    /** The same on a patched worker, where the clause hook files the damage. */
    @Test
    void aPatchedWorkerCarriesTheDeathToo() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Card bears = TestBoard.onBattlefield(duel.game(), duel.opponent(), "Grizzly Bears");
        SpellAbility bolt = castAt(duel, "Lightning Bolt", bears);

        List<String> records = ShardRun.hooked(tempDir, duel.game(), WHY, recording -> {
            duel.castAndResolve(bolt);
            nextStep(duel.game());
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), halves.toString());
        assertTrue(damages(halves.get(0), bears), halves.get(0));
        assertAStateBasedDeath(halves.get(0), bears, bolt.getHostCard());
    }

    /** A creature that survived its damage has no death to carry. */
    @Test
    void aNonLethalShockCarriesNoDeath() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Card giant = TestBoard.onBattlefield(duel.game(), duel.opponent(), "Hill Giant");
        SpellAbility shock = castAt(duel, "Shock", giant);

        List<String> records = ShardRun.hooked(tempDir, duel.game(), WHY, recording -> {
            duel.castAndResolve(shock);
            nextStep(duel.game());
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), records.toString());
        assertTrue(damages(halves.get(0), giant), halves.get(0));
        assertEquals(List.of(), moves(halves.get(0), giant, "battlefield", "graveyard"),
                halves.get(0));
        assertEquals(2, records.size(), "the cost half and one effect half: " + records);
    }

    /**
     * A land played at the next priority is the next action, not a consequence:
     * it writes the held record and joins nothing.
     */
    @Test
    void aLandPlayedAfterwardsDoesNotJoinTheRecord() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card bears = TestBoard.onBattlefield(game, duel.opponent(), "Grizzly Bears");
        SpellAbility bolt = castAt(duel, "Lightning Bolt", bears);
        Card mountain = TestBoard.inHand(game, duel.caster(), "Mountain");

        List<String> records = ShardRun.hooked(tempDir, game, WHY, recording -> {
            duel.castAndResolve(bolt);
            duel.caster().playLand(mountain, null);
            nextStep(game);
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), records.toString());
        assertTrue(holdsTheDeathOf(halves.get(0), bears), halves.get(0));
        for (String record : records) {
            assertEquals(List.of(), eventsNaming(record, Refs.of(mountain)),
                    "the land joined a record: " + record);
        }
    }

    /**
     * A permanent the resolution touched, leaving outside a state-based check
     * — a cost paid for the next spell, a mana ability — is the next action's
     * doing, not this one's consequence.
     */
    @Test
    void aTouchedPermanentLeavingOutsideTheStateBasedCheckDoesNotJoin() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card giant = TestBoard.onBattlefield(game, duel.opponent(), "Hill Giant");
        SpellAbility shock = castAt(duel, "Shock", giant);

        List<String> records = ShardRun.hooked(tempDir, game, WHY, recording -> {
            duel.castAndResolve(shock);
            // Sacrificed as a cost would be: no state-based check is running.
            game.getAction().moveToGraveyard(giant, null);
            nextStep(game);
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), halves.toString());
        assertFalse(holdsTheDeathOf(halves.get(0), giant), halves.get(0));
    }

    /**
     * Forge checks state-based actions before every priority, so the check
     * right after a resolution finds only what that resolution left behind:
     * a creature that dies in it joins the record whether or not one of the
     * resolution's own events named it.
     */
    @Test
    void aDeathInTheFollowingCheckJoinsEvenUntouched() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card giant = TestBoard.onBattlefield(game, duel.opponent(), "Hill Giant");
        Card bystander = TestBoard.onBattlefield(game, duel.opponent(), "Grizzly Bears");
        SpellAbility shock = castAt(duel, "Shock", giant);

        List<String> records = ShardRun.hooked(tempDir, game, WHY, recording -> {
            game.getStack().add(shock);
            game.getStack().resolveStack();
            // Lethal damage the bracket never saw, standing in for what a
            // static the resolution brought does to a creature it never named.
            bystander.setDamage(2);
            duel.stateBasedLoop();
            nextStep(game);
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), halves.toString());
        assertTrue(holdsTheDeathOf(halves.get(0), bystander), halves.get(0));
    }

    /**
     * A check after the next action is that action's: once a land is played
     * the held half is written, and a death in the check that follows the
     * land joins nothing.
     */
    @Test
    void aDeathInACheckAfterTheNextActionDoesNotJoin() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card giant = TestBoard.onBattlefield(game, duel.opponent(), "Hill Giant");
        Card bystander = TestBoard.onBattlefield(game, duel.opponent(), "Grizzly Bears");
        SpellAbility shock = castAt(duel, "Shock", giant);
        Card mountain = TestBoard.inHand(game, duel.caster(), "Mountain");

        List<String> records = ShardRun.hooked(tempDir, game, WHY, recording -> {
            duel.castAndResolve(shock);
            duel.caster().playLand(mountain, null);
            bystander.setDamage(2);
            duel.stateBasedLoop();
            // Otherwise there is no death to join and the test below is vacuous.
            assertTrue(inGraveyard(game, bystander), "the bystander died");
            nextStep(game);
        });

        for (String record : records) {
            assertFalse(holdsTheDeathOf(record, bystander), record);
        }
    }

    /**
     * Dead Weight resolves onto a 2/2, its static shrinks the creature to 0/0
     * once the resolution is over, and the check that follows kills it and
     * sends the Aura after it. Both are the Aura spell's doing, and so is
     * the shrink: made after the spell's snapshot, which shows the 2/2, it
     * is one of the spell's events, as it was in a gen-2 game for a target
     * the collector had read before.
     */
    @Test
    void aMinusAuraKillJoinsTheAuraSpellsRecord() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card bears = TestBoard.onBattlefield(game, duel.opponent(), "Grizzly Bears");
        Card aura = TestBoard.inHand(game, duel.caster(), "Dead Weight");
        SpellAbility deadWeight = targeting(spellOf(aura, duel.caster()), bears);

        List<String> records = ShardRun.hooked(tempDir, game, WHY, recording -> {
            duel.castAndResolve(deadWeight);
            assertTrue(inGraveyard(game, bears), "the bears died");
            nextStep(game);
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), halves.toString());
        String effect = halves.get(0);
        assertTrue(holdsTheDeathOf(effect, bears), effect);
        assertTrue(holdsTheDeathOf(effect, aura), "the Aura fell off: " + effect);
        assertFalse(eventsOn(effect, bears, "pt_change").isEmpty(),
                "the shrink after the snapshot is the spell's event: " + effect);
    }

    /** A static that shrinks on entry kills in the check after its creature resolves. */
    @Test
    void aStaticThatKillsOnEntryJoinsTheCreatureSpellsRecord() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card bears = TestBoard.onBattlefield(game, duel.opponent(), "Grizzly Bears");
        SpellAbility norn = spellOf(
                TestBoard.inHand(game, duel.caster(), "Elesh Norn, Grand Cenobite"), duel.caster());

        List<String> records = ShardRun.hooked(tempDir, game, WHY, recording -> {
            duel.castAndResolve(norn);
            nextStep(game);
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), halves.toString());
        assertTrue(holdsTheDeathOf(halves.get(0), bears), halves.get(0));
    }

    /** The legend rule is a state-based action too, and its loser joins the record. */
    @Test
    void theLegendRulesLoserJoinsTheSecondLegendsRecord() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card first = TestBoard.onBattlefield(game, duel.caster(), "Isamaru, Hound of Konda");
        Card secondCard = TestBoard.inHand(game, duel.caster(), "Isamaru, Hound of Konda");
        SpellAbility second = spellOf(secondCard, duel.caster());

        List<String> records = ShardRun.hooked(tempDir, game, WHY, recording -> {
            duel.castAndResolve(second);
            nextStep(game);
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), halves.toString());
        // The legend its controller chose to lose: the one in the graveyard
        // now. Its death is in the half; the one that stayed has none.
        List<Card> dead = game.getCardsIn(ZoneType.Graveyard).stream()
                .filter(card -> card.getName().equals("Isamaru, Hound of Konda")).toList();
        assertEquals(1, dead.size(), "one Isamaru went to the graveyard: " + dead);
        Card lost = dead.get(0).getId() == first.getId() ? first : secondCard;
        Card kept = lost == first ? secondCard : first;
        assertEquals(1, deathsOf(halves.get(0), lost).size(), halves.get(0));
        assertFalse(holdsTheDeathOf(halves.get(0), kept), halves.get(0));
    }

    /** An Aura falls off when the creature it enchanted dies, in the same check. */
    @Test
    void anAuraFallingOffItsDeadHostJoinsTheRecord() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card bears = TestBoard.onBattlefield(game, duel.opponent(), "Grizzly Bears");
        Card pacifism = TestBoard.onBattlefield(game, duel.caster(), "Pacifism");
        pacifism.attachToEntity(bears, null);
        SpellAbility bolt = castAt(duel, "Lightning Bolt", bears);

        List<String> records = ShardRun.hooked(tempDir, game, WHY, recording -> {
            duel.castAndResolve(bolt);
            nextStep(game);
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), halves.toString());
        assertTrue(holdsTheDeathOf(halves.get(0), bears), halves.get(0));
        assertTrue(holdsTheDeathOf(halves.get(0), pacifism), halves.get(0));
    }

    /**
     * On a modal fork, a state-based death joins the mode half that touched
     * the creature, as the live collector files it. Fiery Confluence's
     * "1 damage to each creature" kills the elf; when the last chosen mode is
     * "2 damage to each opponent", that last half never saw the elf.
     */
    @Test
    void aModalForkFilesTheDeathInTheModeThatTouchedTheCreature() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card elves = TestBoard.onBattlefield(game, duel.opponent(), "Llanowar Elves");
        SpellAbility charm = TestBoard.charmOf(
                TestBoard.inHand(game, duel.caster(), "Fiery Confluence"), duel.caster());
        CollectionCaps caps = TestCaps.caps().manaCap(2000).probes(0).build();

        // A seed whose draw chooses the creature mode and ends on the opponent mode.
        ShardRun.Fork fork = ShardRun.forkedWithClauseHook(
                tempDir, new ShardRun.ForkRun(game, caps, 1L, charm), "to split a fork's modes");
        assertTrue(fork.wrote(), "the forced resolution observed nothing");
        List<String> halves = fork.records();
        assertTrue(halves.size() >= 2, halves.toString());
        assertEquals(1L, optionOf(halves.get(halves.size() - 1)), "the last mode: " + halves);
        int lastTouching = lastHalfOfMode(halves, 0L);
        assertTrue(lastTouching >= 0, "the seed chose no creature mode: " + halves);
        for (int i = 0; i < halves.size(); i++) {
            assertEquals(i == lastTouching ? 1 : 0, deathsOf(halves.get(i), elves).size(),
                    "half " + i + " of " + halves);
        }
    }

    /** The index of the last of {@code halves} through the mode at {@code option}, or -1. */
    private static int lastHalfOfMode(List<String> halves, long option) {
        int last = -1;
        for (int i = 0; i < halves.size(); i++) {
            if (Long.valueOf(option).equals(optionOf(halves.get(i)))) {
                last = i;
            }
        }
        return last;
    }

    // ── the state-based loop (FR-028, FR-029) ───────────────────────────

    private static final String BOLT = "cardsfolder/l/lightning_bolt.txt";
    private static final String FESTERING_GOBLIN = "cardsfolder/f/festering_goblin.txt";
    private static final String DOOMED_DISSENTER = "cardsfolder/d/doomed_dissenter.txt";

    /**
     * The board's dies triggers armed: a card moved onto the battlefield by
     * hand is not in the last-state battlefield a look-back trigger reads.
     */
    private static void armTriggers(Game game) {
        game.getTriggerHandler().resetActiveTriggers(false, null);
        game.copyLastState();
    }

    private static boolean inGraveyard(Game game, Card card) {
        return game.getCardsIn(ZoneType.Graveyard).stream().anyMatch(c -> c.getId() == card.getId());
    }

    /**
     * Play the real main loop until {@code card} has died: a step that gives
     * priority runs the state-based loop, with its signals, before anyone
     * acts. Two AI seats with nothing to cast pass.
     */
    private static void stepUntilDead(Game game, Card card) {
        for (int step = 0; step < 20 && !inGraveyard(game, card); step++) {
            game.getPhaseHandler().mainLoopStep();
        }
        assertTrue(inGraveyard(game, card), card + " died");
    }

    /** The records acting through {@code script}. */
    private static List<String> through(List<String> records, String script) {
        return records.stream().filter(record -> script.equals(scriptOf(record))).toList();
    }

    /**
     * Acceptance 7, on the real loop: a Lightning Bolt kills Festering Goblin,
     * whose dies trigger reaches the stack between two passes of the check.
     * The death sits in the Bolt's effect half; the trigger opens its own
     * bracket, whose records hold no death of that check (SC-004). The
     * trigger's -1/-1 then kills the Llanowar Elves in the check after its
     * own resolution, outside the first one, and that death is the trigger's.
     */
    @Test
    void aBurnSpellKeepsItsKillAndTheDiesTriggerKeepsTheLaterOne() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card goblin = TestBoard.onBattlefield(game, duel.opponent(), "Festering Goblin");
        // The trigger's only legal target once the goblin has died.
        Card elves = TestBoard.onBattlefield(game, duel.caster(), "Llanowar Elves");
        SpellAbility bolt = castAt(duel, "Lightning Bolt", goblin);
        armTriggers(game);

        List<String> records = ShardRun.hooked(tempDir, game, WHY, recording -> {
            game.getStack().add(bolt);
            game.getStack().resolveStack();
            // The loop kills the goblin and puts its trigger on the
            // stack; once both players pass the trigger resolves,
            // and the next loop kills the elves.
            stepUntilDead(game, elves);
            assertTrue(inGraveyard(game, goblin), "the goblin died");
            nextStep(game);
        });

        String boltHalf = only(records, "resolution", BOLT);
        assertTrue(holdsTheDeathOf(boltHalf, goblin), "the kill is the Bolt's: " + boltHalf);
        assertFalse(holdsTheDeathOf(boltHalf, elves), boltHalf);
        // The trigger's own bracket: an effect half, unlinked, a trigger
        // having no cost half.
        String triggerHalf = only(records, "resolution", FESTERING_GOBLIN, "trigger");
        assertEquals(null, linkOf(triggerHalf), triggerHalf);
        assertEquals(List.of(), through(records(records, "activation"), FESTERING_GOBLIN),
                records.toString());
        // The bracket's halves; the trigger record names the death that fired it.
        for (String record : through(records(records, "resolution"), FESTERING_GOBLIN)) {
            assertFalse(holdsTheDeathOf(record, goblin),
                    "a trigger record holds a death of its check: " + record);
        }
        assertTrue(holdsTheDeathOf(triggerHalf, elves),
                "the later death is the trigger's: " + triggerHalf);
    }

    /**
     * Lethal damage marked on {@code victim} as the next item goes on the
     * stack: what a pass of the state-based loop leaves for the loop's next
     * pass, once the triggers the pass set off are on the stack.
     *
     * <p>Public, as every bus subscriber is: Forge's bus reflects over public
     * methods.
     */
    public static final class LethalDamageOnThePush {

        private final Card victim;
        private boolean marked;

        LethalDamageOnThePush(Card victim) {
            this.victim = victim;
        }

        @Subscribe
        public void onPush(GameEventSpellAbilityCast pushed) {
            if (!marked) {
                marked = true;
                victim.setDamage(victim.getNetToughness());
            }
        }
    }

    /**
     * The loop's later passes: a death in the pass after a dies trigger went
     * on the stack is still the loop's, so it joins the held half rather
     * than the trigger's bracket, which is open by then. The engine runs the
     * loop and signals it; lethal damage marked as the trigger goes on the
     * stack gives the second pass a death to find, standing in for whatever
     * the first pass's outcome leaves for the next.
     */
    @Test
    void aDeathInALaterPassOfTheLoopJoinsTheHeldHalfNotTheTrigger() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card dissenter = TestBoard.onBattlefield(game, duel.opponent(), "Doomed Dissenter");
        Card bears = TestBoard.onBattlefield(game, duel.opponent(), "Grizzly Bears");
        SpellAbility bolt = castAt(duel, "Lightning Bolt", dissenter);
        armTriggers(game);

        List<String> records = ShardRun.hooked(tempDir, game, WHY, recording -> {
            game.getStack().add(bolt);
            game.getStack().resolveStack();
            game.subscribeToEvents(new LethalDamageOnThePush(bears));
            duel.stateBasedLoop();
            assertTrue(inGraveyard(game, bears), "the second pass killed the bears");
            // The trigger resolves and makes its Zombie; the loop after.
            duel.resolve();
            nextStep(game);
        });

        String boltHalf = only(records, "resolution", BOLT);
        assertTrue(holdsTheDeathOf(boltHalf, dissenter), boltHalf);
        assertTrue(holdsTheDeathOf(boltHalf, bears), "the second pass is the check's: " + boltHalf);
        String triggerHalf = only(records, "resolution", DOOMED_DISSENTER);
        assertTrue(RecordEvents.events(triggerHalf).stream()
                        .filter(event -> "zone_change".equals(event.get("type")))
                        .anyMatch(move -> "battlefield".equals(params(move).get("to_zone"))),
                "the Zombie is the trigger's: " + triggerHalf);
        // The bracket's halves; the trigger record names the death that fired it.
        for (String record : through(records(records, "resolution"), DOOMED_DISSENTER)) {
            assertFalse(holdsTheDeathOf(record, dissenter), record);
            assertFalse(holdsTheDeathOf(record, bears), record);
        }
    }

    /**
     * FR-029: a state-based death with no held resolution and no open damage
     * step reaches no record and is counted. A -1/-1 counter put on Llanowar
     * Elves during the upkeep, as an upkeep cost would, while a Shock is on
     * the stack: the loop after kills the elves, and the death is neither
     * the Shock's — the open bracket, whose cast did not leave the board in
     * that state — nor anyone else's.
     */
    @Test
    void aStateBasedDeathNothingHeldReachesNoRecordAndIsCounted() throws Throwable {
        Duel duel = Duel.at(PhaseType.UPKEEP);
        Game game = duel.game();
        Card elves = TestBoard.onBattlefield(game, duel.caster(), "Llanowar Elves");
        SpellAbility shock = castAt(duel, "Shock",
                TestBoard.onBattlefield(game, duel.opponent(), "Hill Giant"));

        List<String> records = ShardRun.hooked(tempDir, game, WHY, recording -> {
            game.getStack().add(shock);
            elves.addCounterInternal(CounterEnumType.M1M1, 1, duel.caster(), false, null, null);
            duel.stateBasedLoop();
            assertTrue(inGraveyard(game, elves), "the elves died");
            assertEquals(1L, recording.droppedEvents().counts().get(
                            new DroppedEvents.Key("zone_change", "upkeep")),
                    "the death is counted: " + recording.droppedEvents().counts());
            game.getStack().resolveStack();
            nextStep(game);
        });

        assertFalse(records.isEmpty(), "the Shock wrote its halves");
        for (String record : records) {
            assertFalse(holdsTheDeathOf(record, elves), "the death reached a record: " + record);
        }
    }

    /**
     * Under {@code effect.count.dropped.events} a recorded game appends its
     * counts beside its shard when it ends (FR-041), the loop's signals
     * coming from the engine itself: the priority's loop kills the elves the
     * -1/-1 counter left at 0 toughness, with nothing held.
     */
    @Test
    void aCountingRunAppendsTheGamesDroppedEventsBesideItsShard() throws Throwable {
        Duel duel = Duel.at(PhaseType.UPKEEP);
        Card elves = TestBoard.onBattlefield(duel.game(), duel.caster(), "Llanowar Elves");
        CollectionCaps counting = TestCaps.caps().interventions(0).probes(0)
                .countDroppedEvents().build();

        RecordShardWriter writer = ShardRun.recordedWithClauseHook(
                tempDir, new ShardRun.Hooked(duel.game(), counting, "for the state-based loop's signals"), () -> shrinkToDeath(duel, elves));

        Path counts = GameRecording.droppedEventsFile(writer);
        assertEquals(writer.path().getParent(), counts.getParent());
        List<String> lines = Files.readAllLines(counts);
        assertTrue(lines.stream().anyMatch(line -> line.matches("zone_change\t[a-z0-9_]+\t1")),
                "the death's line: " + lines);
    }

    /** Without the property nothing is appended: the counts are the run's to ask for. */
    @Test
    void aRunThatCountsNothingWritesNoCountsFile() throws Throwable {
        Duel duel = Duel.at(PhaseType.UPKEEP);
        Card elves = TestBoard.onBattlefield(duel.game(), duel.caster(), "Llanowar Elves");
        CollectionCaps silent = TestCaps.caps().interventions(0).probes(0).build();

        RecordShardWriter writer = ShardRun.recordedWithClauseHook(
                tempDir, new ShardRun.Hooked(duel.game(), silent, "for the state-based loop's signals"), () -> shrinkToDeath(duel, elves));

        assertFalse(Files.exists(GameRecording.droppedEventsFile(writer)));
    }

    /** A -1/-1 counter on a 1/1 {@code creature}, and the main loop played until it has died. */
    private static void shrinkToDeath(Duel duel, Card creature) {
        creature.addCounterInternal(CounterEnumType.M1M1, 1, duel.caster(), false, null, null);
        stepUntilDead(duel.game(), creature);
    }

    // ── the end of the game ─────────────────────────────────────────────

    /**
     * The game ending writes the held half with its death, and the winner
     * still reaches the shard after it.
     */
    @Test
    void theGameEndingWritesTheHeldHalfBeforeTheOutcome() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card bears = TestBoard.onBattlefield(game, duel.opponent(), "Grizzly Bears");
        SpellAbility bolt = castAt(duel, "Lightning Bolt", bears);

        List<String> records = ShardRun.hooked(tempDir, game, WHY, recording -> {
            duel.castAndResolve(bolt);
            game.fireEvent(new GameEventGameOutcome(1, List.of("a has won"), "a", "a: 1 b: 0 "));
            game.fireEvent(new GameEventGameFinished());
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), records.toString());
        assertTrue(holdsTheDeathOf(halves.get(0), bears), halves.get(0));
        assertFalse(eventTypes(halves.get(0)).contains("player_won"), halves.get(0));
        List<String> won = withEventOfType(records, "player_won");
        assertEquals(1, won.size(), records.toString());
        assertTrue(records.indexOf(won.get(0)) > records.indexOf(halves.get(0)),
                "the outcome follows the held half: " + records);
    }

    // ── a copy parked while the stack was frozen ────────────────────────

    /**
     * Chain of Acid destroys the Glorious Anthem keeping a damaged Grizzly
     * Bears alive, and the Anthem's controller copies it: the copy, made
     * while the resolution froze the stack, is pushed as Forge finishes the
     * resolution, before the state-based check. That push is not the next
     * action, so the Bears' death still joins Chain of Acid's effect half.
     */
    @Test
    void aCopyPushedAsTheResolutionFinishesKeepsItsDeathInTheHalf() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        TestBoard.inPlay(game, duel.caster(), "Mountain");
        for (int i = 0; i < 4; i++) {
            TestBoard.inPlay(game, duel.opponent(), "Forest");
        }
        Card anthem = TestBoard.inPlay(game, duel.opponent(), "Glorious Anthem");
        Card bears = TestBoard.inPlay(game, duel.opponent(), "Grizzly Bears");
        game.getAction().checkStaticAbilities();
        bears.setDamage(2);
        SpellAbility chain = castAt(duel, "Chain of Acid", anthem);

        List<String> records = ShardRun.recordsWithClauseHook(
                tempDir, game, "to resolve a spell its target's controller copies", () -> {
                    duel.castAndResolve(chain);
                    assertFalse(game.getStack().isEmpty(), "the Anthem's controller copied the spell");
                    assertTrue(inGraveyard(game, bears), "the Bears died");
                    nextStep(game);
                });

        List<String> halves = effectHalves(records).stream()
                .filter(r -> "cardsfolder/c/chain_of_acid.txt".equals(scriptOf(r))).toList();
        assertEquals(1, halves.size(), records.toString());
        assertTrue(holdsTheDeathOf(halves.get(0), bears),
                "the death joins the record: " + halves.get(0));
    }

    // ── a modal resolution ──────────────────────────────────────────────

    /** Fiery Confluence in hand, with the modes at these Choices$ positions chosen. */
    private static SpellAbility charm(Duel duel, int... positions) {
        SpellAbility charm = TestBoard.charmOf(
                TestBoard.inHand(duel.game(), duel.caster(), "Fiery Confluence"), duel.caster());
        TestBoard.chain(charm, positions);
        return charm;
    }

    /** {@code charm} cast and resolved under the hooks, and the next step entered. */
    private List<String> resolvedCharm(Duel duel, SpellAbility charm) throws Throwable {
        return ShardRun.hooked(tempDir, duel.game(), "to split the modes", recording -> {
            duel.castAndResolve(charm);
            nextStep(duel.game());
        });
    }

    /**
     * A death goes to the mode that touched the creature, not to the last
     * mode: Fiery Confluence's "1 damage to each creature" kills the elf, and
     * its "2 damage to each opponent" half, which resolves after, never saw it.
     */
    @Test
    void aModalDeathJoinsTheModeThatTouchedTheCreature() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Card elves = TestBoard.onBattlefield(duel.game(), duel.opponent(), "Llanowar Elves");

        List<String> records = resolvedCharm(duel, charm(duel, 0, 1));

        List<String> halves = effectHalves(records);
        assertEquals(2, halves.size(), records.toString());
        assertEquals(List.of(0L, 1L), halves.stream().map(RecordAbilities::optionOf).toList());
        assertTrue(holdsTheDeathOf(halves.get(0), elves), halves.get(0));
        assertFalse(holdsTheDeathOf(halves.get(1), elves), halves.get(1));
        // The cost half and the link are what they were.
        String cost = records.get(0);
        assertEquals("activation", parse(cost).get("moment"), cost);
        assertNotNull(linkOf(cost), cost);
        for (String half : halves) {
            assertEquals(linkOf(cost), linkOf(half), half);
        }
    }

    /** Touched by two halves, the death goes to the later one, once. */
    @Test
    void aDeathTouchedByTwoModesJoinsTheLastOfThemOnce() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Card bears = TestBoard.onBattlefield(duel.game(), duel.opponent(), "Grizzly Bears");

        List<String> records = resolvedCharm(duel, charm(duel, 0, 0, 1));

        List<String> halves = effectHalves(records);
        assertEquals(3, halves.size(), halves.toString());
        // One half a chosen mode, the repeated mode twice, in the order chosen.
        assertEquals(List.of(0L, 0L, 1L), halves.stream().map(RecordAbilities::optionOf).toList());
        assertEquals(List.of(0, 1, 0),
                halves.stream().map(half -> deathsOf(half, bears).size()).toList(), halves.toString());
    }

    /** The half rule alone: the last half that named the permanent, else the last half. */
    @Test
    void aDeathNoModeNamedJoinsTheLastHalf() {
        List<Set<String>> touched = List.of(Set.of("E1", "E2"), Set.of("E2"), Set.of("P0"));
        assertEquals(0, HeldHalves.halfForDeath(touched, "E1"));
        assertEquals(1, HeldHalves.halfForDeath(touched, "E2"));
        assertEquals(2, HeldHalves.halfForDeath(touched, "E9"));
    }

    // ── forks ───────────────────────────────────────────────────────────

    /**
     * A forced resolution already carries its deaths: {@code
     * GameSimulator.resolveStack} runs the state-based check while the fork's
     * sink is still listening, so the interventional record needs no hold.
     * Flame Slash can only target a creature, so the fork's random target
     * draw has the bears alone to choose.
     */
    @Test
    void anInterventionalBurnSpellAlreadyCarriesItsDeath() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Card bears = TestBoard.onBattlefield(duel.game(), duel.opponent(), "Grizzly Bears");
        Card slashCard = TestBoard.inHand(duel.game(), duel.caster(), "Flame Slash");
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(1).probes(0).build();

        ShardRun.Fork fork = ShardRun.forkedWithClauseHook(tempDir, new ShardRun.ForkRun(duel.game(), caps, 1L, spellOf(slashCard, duel.caster())), "for the fork's own resolution");
        assertTrue(fork.wrote(), "the forced resolution observed nothing");
        List<String> records = fork.records();
        assertEquals(1, records.size(), records.toString());
        String record = records.get(0);
        assertTrue(damages(record, bears), record);
        // Rendered exactly as the observed record renders it.
        assertAStateBasedDeath(record, bears, slashCard);
    }
}
