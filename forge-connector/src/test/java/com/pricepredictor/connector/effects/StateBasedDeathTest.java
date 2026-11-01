package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.event.GameEventGameOutcome;
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
import java.util.Set;

import static com.pricepredictor.connector.effects.RecordLines.count;
import static com.pricepredictor.connector.effects.RecordLines.eventTypes;
import static com.pricepredictor.connector.effects.RecordLines.eventsNaming;
import static com.pricepredictor.connector.effects.RecordLines.linkOf;
import static com.pricepredictor.connector.effects.RecordLines.only;
import static com.pricepredictor.connector.effects.RecordLines.optionOf;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static com.pricepredictor.connector.effects.RecordLines.records;
import static com.pricepredictor.connector.effects.RecordLines.scriptOf;
import static com.pricepredictor.connector.effects.RecordLines.withEventOfType;
import static com.pricepredictor.connector.effects.TestBoard.castAndResolve;
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
 * at the next priority runs the state-based check that moves a creature with
 * lethal damage to the graveyard. An effect half written at the resolved event
 * would record a Lightning Bolt at a 2/2 as the damage and never the death.
 *
 * <p>These drive the real engine — the stack, the resolution, and
 * {@code GameAction.checkStateEffects} — with the collector subscribed to the
 * game's own bus, because what is under test is the order in which the engine
 * publishes, and a hand-fed event sequence would only restate my reading of it.
 */
@ExtendWith(ForgeExtension.class)
class StateBasedDeathTest {

    @TempDir
    Path tempDir;

    private static SpellAbility targeting(SpellAbility spell, Card target) {
        spell.resetTargets();
        spell.getTargets().add(target);
        return spell;
    }

    /** The effect halves: the resolution records a cost half links to. */
    private static List<String> effectHalves(List<String> records) {
        return records(records, "resolution").stream()
                .filter(record -> linkOf(record) != null).toList();
    }

    /** The rendered event that says a card left the battlefield for the graveyard. */
    private static String deathOf(Card card) {
        return "{\"type\":\"zone_change\",\"subjects\":[\"E" + card.getId()
                + "\"],\"params\":{\"from_zone\":\"battlefield\",\"to_zone\":\"graveyard\"";
    }

    // ── a lethal resolution ─────────────────────────────────────────────

    /**
     * The reproducer: Lightning Bolt at Grizzly Bears writes the damage and
     * the death in one effect half, on a worker with no clause hook.
     */
    @Test
    void aLethalBoltCarriesTheDeathItCausedInItsEffectHalf() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.onBattlefield(game, opponent, "Grizzly Bears");
        SpellAbility bolt = targeting(
                spellOf(TestBoard.inHand(game, caster, "Lightning Bolt"), caster), bears);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            castAndResolve(game, bolt);
            assertTrue(game.getCardsIn(ZoneType.Graveyard).stream()
                    .anyMatch(c -> c.getId() == bears.getId()), "the bears died");
            nextStep(game);
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), halves.toString());
        String effect = halves.get(0);
        assertTrue(effect.contains("\"type\":\"damage_dealt\",\"subjects\":[\"E"
                + bears.getId() + "\"]"), effect);
        assertTrue(effect.contains(deathOf(bears)), "the death joins the record: " + effect);
        // A state-based action, not a clause: no clause produced it, and the
        // line whose resolution it follows is what caused it.
        assertTrue(effect.contains(deathOf(bears) + ",\"cause\":\"E"
                + bolt.getHostCard().getId() + "\"},\"duration\":\"instant\","
                + "\"attributed_to\":\"unresolved\"}"), effect);
    }

    /**
     * The effect half waits for the next action and is written exactly once:
     * here the next action is another cast, whose own half waits in turn.
     */
    @Test
    void aHeldHalfIsWrittenOnceWhenTheNextActionBegins() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.onBattlefield(game, opponent, "Grizzly Bears");
        Card giant = TestBoard.onBattlefield(game, opponent, "Hill Giant");
        SpellAbility bolt = targeting(
                spellOf(TestBoard.inHand(game, caster, "Lightning Bolt"), caster), bears);
        SpellAbility shock = targeting(
                spellOf(TestBoard.inHand(game, caster, "Shock"), caster), giant);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            castAndResolve(game, bolt);
            castAndResolve(game, shock);
            nextStep(game);
            nextStep(game);
        });

        assertEquals(4, records.size(), records.toString());
        List<String> halves = effectHalves(records);
        assertEquals(2, halves.size(), records.toString());
        assertTrue(halves.get(0).contains(deathOf(bears)), halves.get(0));
        assertFalse(halves.get(1).contains(deathOf(bears)), halves.get(1));
        assertFalse(halves.get(1).contains(deathOf(giant)), halves.get(1));
    }

    /** The same on a patched worker, where the clause hook files the damage. */
    @Test
    void aPatchedWorkerCarriesTheDeathToo() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.onBattlefield(game, opponent, "Grizzly Bears");
        SpellAbility bolt = targeting(
                spellOf(TestBoard.inHand(game, caster, "Lightning Bolt"), caster), bears);

        List<String> records = ShardRun.recordsWithClauseHook(
                tempDir, game, "", () -> {
                    castAndResolve(game, bolt);
                    nextStep(game);
                });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), halves.toString());
        assertTrue(halves.get(0).contains(deathOf(bears)), halves.get(0));
    }

    /** A creature that survived its damage has no death to carry. */
    @Test
    void aNonLethalShockCarriesNoDeath() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card giant = TestBoard.onBattlefield(game, opponent, "Hill Giant");
        SpellAbility shock = targeting(
                spellOf(TestBoard.inHand(game, caster, "Shock"), caster), giant);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            castAndResolve(game, shock);
            nextStep(game);
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), records.toString());
        assertTrue(halves.get(0).contains("\"type\":\"damage_dealt\""), halves.get(0));
        assertFalse(halves.get(0).contains("\"from_zone\":\"battlefield\""), halves.get(0));
        assertEquals(2, records.size(), "the cost half and one effect half: " + records);
    }

    /**
     * A land played at the next priority is the next action, not a consequence:
     * it writes the held record and joins nothing.
     */
    @Test
    void aLandPlayedAfterwardsDoesNotJoinTheRecord() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.onBattlefield(game, opponent, "Grizzly Bears");
        SpellAbility bolt = targeting(
                spellOf(TestBoard.inHand(game, caster, "Lightning Bolt"), caster), bears);
        Card mountain = TestBoard.inHand(game, caster, "Mountain");

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            castAndResolve(game, bolt);
            caster.playLand(mountain, null);
            nextStep(game);
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), records.toString());
        assertTrue(halves.get(0).contains(deathOf(bears)), halves.get(0));
        for (String record : records) {
            assertEquals(List.of(), eventsNaming(record, "E" + mountain.getId()),
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
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card giant = TestBoard.onBattlefield(game, opponent, "Hill Giant");
        SpellAbility shock = targeting(
                spellOf(TestBoard.inHand(game, caster, "Shock"), caster), giant);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            castAndResolve(game, shock);
            // Sacrificed as a cost would be: no state-based check is running.
            game.getAction().moveToGraveyard(giant, null);
            nextStep(game);
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), halves.toString());
        assertFalse(halves.get(0).contains(deathOf(giant)), halves.get(0));
    }

    /**
     * Forge checks state-based actions before every priority, so the check
     * right after a resolution finds only what that resolution left behind:
     * a creature that dies in it joins the record whether or not one of the
     * resolution's own events named it.
     */
    @Test
    void aDeathInTheFollowingCheckJoinsEvenUntouched() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card giant = TestBoard.onBattlefield(game, opponent, "Hill Giant");
        Card bystander = TestBoard.onBattlefield(game, opponent, "Grizzly Bears");
        SpellAbility shock = targeting(
                spellOf(TestBoard.inHand(game, caster, "Shock"), caster), giant);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            game.getStack().add(shock);
            game.getStack().resolveStack();
            // Lethal damage the bracket never saw, standing in for what a
            // static the resolution brought does to a creature it never named.
            bystander.setDamage(2);
            game.getAction().checkStateEffects(false);
            nextStep(game);
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), halves.toString());
        assertTrue(halves.get(0).contains(deathOf(bystander)), halves.get(0));
    }

    /**
     * A check after the next action is that action's: once a land is played
     * the held half is written, and a death in the check that follows the
     * land joins nothing.
     */
    @Test
    void aDeathInACheckAfterTheNextActionDoesNotJoin() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card giant = TestBoard.onBattlefield(game, opponent, "Hill Giant");
        Card bystander = TestBoard.onBattlefield(game, opponent, "Grizzly Bears");
        SpellAbility shock = targeting(
                spellOf(TestBoard.inHand(game, caster, "Shock"), caster), giant);
        Card mountain = TestBoard.inHand(game, caster, "Mountain");

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            castAndResolve(game, shock);
            caster.playLand(mountain, null);
            bystander.setDamage(2);
            game.getAction().checkStateEffects(false);
            // Otherwise there is no death to join and the test below is vacuous.
            assertTrue(game.getCardsIn(ZoneType.Graveyard).stream()
                    .anyMatch(c -> c.getId() == bystander.getId()), "the bystander died");
            nextStep(game);
        });

        for (String record : records) {
            assertFalse(record.contains(deathOf(bystander)), record);
        }
    }

    /**
     * Dead Weight resolves onto a 2/2, its static shrinks the creature to 0/0
     * once the resolution is over, and the check that follows kills it and
     * sends the Aura after it. Both are the Aura spell's doing; the static's
     * own stat change is not an event of the spell's, and belongs to the
     * static's continuous record.
     */
    @Test
    void aMinusAuraKillJoinsTheAuraSpellsRecord() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.onBattlefield(game, game.getPlayers().get(1), "Grizzly Bears");
        Card aura = TestBoard.inHand(game, caster, "Dead Weight");
        SpellAbility deadWeight = targeting(spellOf(aura, caster), bears);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            castAndResolve(game, deadWeight);
            assertTrue(game.getCardsIn(ZoneType.Graveyard).stream()
                    .anyMatch(c -> c.getId() == bears.getId()), "the bears died");
            nextStep(game);
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), halves.toString());
        String effect = halves.get(0);
        assertTrue(effect.contains(deathOf(bears)), effect);
        assertTrue(effect.contains(deathOf(aura)), "the Aura fell off: " + effect);
        assertFalse(eventsNaming(effect, "E" + bears.getId()).stream()
                        .anyMatch(event -> "pt_change".equals(event.get("type"))),
                "the static's shrink is not the spell's event: " + effect);
    }

    /** A static that shrinks on entry kills in the check after its creature resolves. */
    @Test
    void aStaticThatKillsOnEntryJoinsTheCreatureSpellsRecord() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.onBattlefield(game, game.getPlayers().get(1), "Grizzly Bears");
        SpellAbility norn = spellOf(
                TestBoard.inHand(game, caster, "Elesh Norn, Grand Cenobite"), caster);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            castAndResolve(game, norn);
            nextStep(game);
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), halves.toString());
        assertTrue(halves.get(0).contains(deathOf(bears)), halves.get(0));
    }

    /** The legend rule is a state-based action too, and its loser joins the record. */
    @Test
    void theLegendRulesLoserJoinsTheSecondLegendsRecord() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card first = TestBoard.onBattlefield(game, caster, "Isamaru, Hound of Konda");
        Card secondCard = TestBoard.inHand(game, caster, "Isamaru, Hound of Konda");
        SpellAbility second = spellOf(secondCard, caster);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            castAndResolve(game, second);
            nextStep(game);
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), halves.toString());
        assertTrue(halves.get(0).contains(deathOf(first))
                        || halves.get(0).contains(deathOf(secondCard)),
                "one Isamaru went to the graveyard: " + halves.get(0));
    }

    /** An Aura falls off when the creature it enchanted dies, in the same check. */
    @Test
    void anAuraFallingOffItsDeadHostJoinsTheRecord() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.onBattlefield(game, opponent, "Grizzly Bears");
        Card pacifism = TestBoard.onBattlefield(game, caster, "Pacifism");
        pacifism.attachToEntity(bears, null);
        SpellAbility bolt = targeting(
                spellOf(TestBoard.inHand(game, caster, "Lightning Bolt"), caster), bears);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            castAndResolve(game, bolt);
            nextStep(game);
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), halves.toString());
        assertTrue(halves.get(0).contains(deathOf(bears)), halves.get(0));
        assertTrue(halves.get(0).contains(deathOf(pacifism)), halves.get(0));
    }

    /**
     * On a modal fork, a state-based death joins the mode half that touched
     * the creature, as the live collector files it. Fiery Confluence's
     * "1 damage to each creature" kills the elf; when the last chosen mode is
     * "2 damage to each opponent", that last half never saw the elf.
     */
    @Test
    void aModalForkFilesTheDeathInTheModeThatTouchedTheCreature() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card elves = TestBoard.onBattlefield(game, game.getPlayers().get(1), "Llanowar Elves");
        SpellAbility charm = TestBoard.charmOf(
                TestBoard.inHand(game, caster, "Fiery Confluence"), caster);
        CollectionCaps caps = TestCaps.caps().manaCap(2000).probes(0).build();

        // A seed whose draw chooses the creature mode and ends on the opponent mode.
        ShardRun.Fork fork = ShardRun.forkedWithClauseHook(
                tempDir, game, caps, 1L, charm, "to split a fork's modes");
        assertTrue(fork.wrote(), "the forced resolution observed nothing");
        List<String> halves = fork.records();
        assertTrue(halves.size() >= 2, halves.toString());
        assertEquals(1L, optionOf(halves.get(halves.size() - 1)), "the last mode: " + halves);
        int lastTouching = -1;
        for (int i = 0; i < halves.size(); i++) {
            if (Long.valueOf(0).equals(optionOf(halves.get(i)))) {
                lastTouching = i;
            }
        }
        assertTrue(lastTouching >= 0, "the seed chose no creature mode: " + halves);
        for (int i = 0; i < halves.size(); i++) {
            assertEquals(i == lastTouching ? 1 : 0, count(halves.get(i), deathOf(elves)),
                    "half " + i + " of " + halves);
        }
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
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card goblin = TestBoard.onBattlefield(game, opponent, "Festering Goblin");
        // The trigger's only legal target once the goblin has died.
        Card elves = TestBoard.onBattlefield(game, caster, "Llanowar Elves");
        SpellAbility bolt = targeting(
                spellOf(TestBoard.inHand(game, caster, "Lightning Bolt"), caster), goblin);
        armTriggers(game);

        List<String> records = ShardRun.recordsWithClauseHook(
                tempDir, game, "for the state-based loop's signals", () -> {
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
        assertTrue(boltHalf.contains(deathOf(goblin)), "the kill is the Bolt's: " + boltHalf);
        assertFalse(boltHalf.contains(deathOf(elves)), boltHalf);
        // The trigger's own bracket: a cost half and an effect half, linked.
        String triggerCost = only(records, "activation", FESTERING_GOBLIN, "trigger");
        String triggerHalf = only(records, "resolution", FESTERING_GOBLIN, "trigger");
        assertEquals(linkOf(triggerCost), linkOf(triggerHalf), triggerHalf);
        for (String record : records) {
            if (FESTERING_GOBLIN.equals(scriptOf(record))) {
                assertFalse(record.contains(deathOf(goblin)),
                        "a trigger record holds a death of its check: " + record);
            }
        }
        assertTrue(triggerHalf.contains(deathOf(elves)),
                "the later death is the trigger's: " + triggerHalf);
    }

    /**
     * The loop's later passes: a death in the pass after a dies trigger went
     * on the stack is still the check's, so it joins the held half rather
     * than the trigger's bracket, which is open by then. The loop's signals
     * are given by hand around the engine's own passes, so the second pass
     * has a death to find: lethal damage marked between the two, standing in
     * for whatever the first pass's outcome leaves for the next.
     */
    @Test
    void aDeathInALaterPassOfTheLoopJoinsTheHeldHalfNotTheTrigger() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card dissenter = TestBoard.onBattlefield(game, opponent, "Doomed Dissenter");
        Card bears = TestBoard.onBattlefield(game, opponent, "Grizzly Bears");
        SpellAbility bolt = targeting(
                spellOf(TestBoard.inHand(game, caster, "Lightning Bolt"), caster), dissenter);
        armTriggers(game);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            game.getStack().add(bolt);
            game.getStack().resolveStack();
            collector.phaseSink().stateBasedLoopBegan();
            game.getAction().checkStateEffects(false);
            assertTrue(game.getStack().addAllTriggeredAbilitiesToStack(),
                    "the dies trigger went on the stack between the passes");
            bears.setDamage(2);
            game.getAction().checkStateEffects(false);
            collector.phaseSink().stateBasedLoopEnded();
            assertTrue(inGraveyard(game, bears), "the second pass killed the bears");
            // The trigger resolves and makes its Zombie; the priority after.
            game.getStack().resolveStack();
            collector.phaseSink().stateBasedLoopBegan();
            game.getAction().checkStateEffects(false);
            collector.phaseSink().stateBasedLoopEnded();
            nextStep(game);
        });

        String boltHalf = only(records, "resolution", BOLT);
        assertTrue(boltHalf.contains(deathOf(dissenter)), boltHalf);
        assertTrue(boltHalf.contains(deathOf(bears)), "the second pass is the check's: " + boltHalf);
        String triggerHalf = only(records, "resolution", DOOMED_DISSENTER);
        assertTrue(triggerHalf.contains("\"to_zone\":\"battlefield\""),
                "the Zombie is the trigger's: " + triggerHalf);
        for (String record : records) {
            if (DOOMED_DISSENTER.equals(scriptOf(record))) {
                assertFalse(record.contains(deathOf(dissenter)), record);
                assertFalse(record.contains(deathOf(bears)), record);
            }
        }
    }

    /**
     * FR-029: a state-based death with no held resolution and no open damage
     * step reaches no record and is counted. A -1/-1 counter put on Llanowar
     * Elves during the upkeep, as an upkeep cost would, while a Shock is on
     * the stack: the check after kills the elves, and the death is neither
     * the Shock's — the open bracket, whose cast did not leave the board in
     * that state — nor anyone else's.
     */
    @Test
    void aStateBasedDeathNothingHeldReachesNoRecordAndIsCounted() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.UPKEEP);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card elves = TestBoard.onBattlefield(game, caster, "Llanowar Elves");
        SpellAbility shock = targeting(
                spellOf(TestBoard.inHand(game, caster, "Shock"), caster),
                TestBoard.onBattlefield(game, opponent, "Hill Giant"));

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            game.getStack().add(shock);
            elves.addCounterInternal(CounterEnumType.M1M1, 1, caster, false, null, null);
            collector.phaseSink().stateBasedLoopBegan();
            game.getAction().checkStateEffects(false);
            collector.phaseSink().stateBasedLoopEnded();
            assertTrue(inGraveyard(game, elves), "the elves died");
            assertEquals(1L, collector.droppedEvents().counts().get(
                            new DroppedEvents.Key("zone_change", "upkeep")),
                    "the death is counted: " + collector.droppedEvents().counts());
            game.getStack().resolveStack();
            nextStep(game);
        });

        assertFalse(records.isEmpty(), "the Shock wrote its halves");
        for (String record : records) {
            assertFalse(record.contains(deathOf(elves)), "the death reached a record: " + record);
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
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.UPKEEP);
        Player caster = game.getPlayers().get(0);
        Card elves = TestBoard.onBattlefield(game, caster, "Llanowar Elves");
        CollectionCaps counting = TestCaps.caps().interventions(0).probes(0)
                .countDroppedEvents().build();

        RecordShardWriter writer = ShardRun.recordedWithClauseHook(
                tempDir, game, counting, "for the state-based loop's signals", () -> {
                    elves.addCounterInternal(CounterEnumType.M1M1, 1, caster, false, null, null);
                    stepUntilDead(game, elves);
                });

        Path counts = GameRecording.droppedEventsFile(writer);
        assertEquals(writer.path().getParent(), counts.getParent());
        List<String> lines = Files.readAllLines(counts);
        assertTrue(lines.stream().anyMatch(line -> line.matches("zone_change\t[a-z0-9_]+\t1")),
                "the death's line: " + lines);
    }

    /** Without the property nothing is appended: the counts are the run's to ask for. */
    @Test
    void aRunThatCountsNothingWritesNoCountsFile() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.UPKEEP);
        Player caster = game.getPlayers().get(0);
        Card elves = TestBoard.onBattlefield(game, caster, "Llanowar Elves");
        CollectionCaps silent = TestCaps.caps().interventions(0).probes(0).build();

        RecordShardWriter writer = ShardRun.recordedWithClauseHook(
                tempDir, game, silent, "for the state-based loop's signals", () -> {
                    elves.addCounterInternal(CounterEnumType.M1M1, 1, caster, false, null, null);
                    stepUntilDead(game, elves);
                });

        assertFalse(Files.exists(GameRecording.droppedEventsFile(writer)));
    }

    // ── the end of the game ─────────────────────────────────────────────

    /**
     * The game ending writes the held half with its death, and the winner
     * still reaches the shard after it.
     */
    @Test
    void theGameEndingWritesTheHeldHalfBeforeTheOutcome() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.onBattlefield(game, opponent, "Grizzly Bears");
        SpellAbility bolt = targeting(
                spellOf(TestBoard.inHand(game, caster, "Lightning Bolt"), caster), bears);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            castAndResolve(game, bolt);
            collector.onGameOutcome(new GameEventGameOutcome(
                    1, List.of("a has won"), "a", "a: 1 b: 0 "));
            collector.finishGame(Set.of());
        });

        List<String> halves = effectHalves(records);
        assertEquals(1, halves.size(), records.toString());
        assertTrue(halves.get(0).contains(deathOf(bears)), halves.get(0));
        assertFalse(eventTypes(halves.get(0)).contains("player_won"), halves.get(0));
        List<String> won = withEventOfType(records, "player_won");
        assertEquals(1, won.size(), records.toString());
        assertTrue(records.indexOf(won.get(0)) > records.indexOf(halves.get(0)),
                "the outcome follows the held half: " + records);
    }

    // ── a modal resolution ──────────────────────────────────────────────

    /** Fiery Confluence in hand, with the modes at these Choices$ positions chosen. */
    private static SpellAbility charm(Game game, Player caster, int... positions) {
        SpellAbility charm = TestBoard.charmOf(
                TestBoard.inHand(game, caster, "Fiery Confluence"), caster);
        TestBoard.chain(charm, positions);
        return charm;
    }

    /**
     * A death goes to the mode that touched the creature, not to the last
     * mode: Fiery Confluence's "1 damage to each creature" kills the elf, and
     * its "2 damage to each opponent" half, which resolves after, never saw it.
     */
    @Test
    void aModalDeathJoinsTheModeThatTouchedTheCreature() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card elves = TestBoard.onBattlefield(game, game.getPlayers().get(1), "Llanowar Elves");
        SpellAbility charm = charm(game, caster, 0, 1);

        List<String> records = ShardRun.recordsWithClauseHook(
                tempDir, game, "to split the modes", () -> {
                    castAndResolve(game, charm);
                    nextStep(game);
                });

        List<String> halves = effectHalves(records);
        assertEquals(2, halves.size(), records.toString());
        assertEquals(0L, optionOf(halves.get(0)), halves.get(0));
        assertTrue(halves.get(0).contains(deathOf(elves)), halves.get(0));
        assertEquals(1L, optionOf(halves.get(1)), halves.get(1));
        assertFalse(halves.get(1).contains(deathOf(elves)), halves.get(1));
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
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.onBattlefield(game, game.getPlayers().get(1), "Grizzly Bears");
        SpellAbility charm = charm(game, caster, 0, 0, 1);

        List<String> records = ShardRun.recordsWithClauseHook(
                tempDir, game, "to split the modes", () -> {
                    castAndResolve(game, charm);
                    nextStep(game);
                });

        List<String> halves = effectHalves(records);
        assertEquals(3, halves.size(), halves.toString());
        assertFalse(halves.get(0).contains(deathOf(bears)), halves.get(0));
        assertEquals(1, count(halves.get(1), deathOf(bears)), halves.get(1));
        assertFalse(halves.get(2).contains(deathOf(bears)), halves.get(2));
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
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.onBattlefield(game, game.getPlayers().get(1), "Grizzly Bears");
        Card slashCard = TestBoard.inHand(game, caster, "Flame Slash");
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(1).probes(0).build();

        ShardRun.Fork fork = ShardRun.forked(tempDir, game, caps, 1L, spellOf(slashCard, caster));
        assertTrue(fork.wrote(), "the forced resolution observed nothing");
        List<String> records = fork.records();
        assertEquals(1, records.size(), records.toString());
        String record = records.get(0);
        assertTrue(record.contains("\"type\":\"damage_dealt\",\"subjects\":[\"E"
                + bears.getId() + "\"]"), record);
        // Rendered exactly as the observed record renders it.
        assertTrue(record.contains(deathOf(bears) + ",\"cause\":\"E"
                + slashCard.getId() + "\"},\"duration\":\"instant\","
                + "\"attributed_to\":\"unresolved\"}"), record);
    }
}
