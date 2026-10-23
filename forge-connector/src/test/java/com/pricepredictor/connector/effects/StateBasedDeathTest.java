package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.event.GameEventGameOutcome;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Set;

import static com.pricepredictor.connector.effects.RecordLines.count;
import static com.pricepredictor.connector.effects.RecordLines.eventTypes;
import static com.pricepredictor.connector.effects.RecordLines.linkOf;
import static com.pricepredictor.connector.effects.RecordLines.optionOf;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static com.pricepredictor.connector.effects.RecordLines.records;
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
            assertFalse(record.contains("\"subjects\":[\"E" + mountain.getId() + "\"]"),
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
        assertFalse(effect.contains("\"type\":\"pt_change\",\"subjects\":[\"E" + bears.getId()),
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
        CollectionCaps caps = new CollectionCaps(
                2000, 0.1, 2, 0, List.of(), 0.0, 0.0);

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
        assertEquals(0, BusBracketCollector.halfForDeath(touched, "E1"));
        assertEquals(1, BusBracketCollector.halfForDeath(touched, "E2"));
        assertEquals(2, BusBracketCollector.halfForDeath(touched, "E9"));
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
        CollectionCaps caps = new CollectionCaps(
                2000, 0.1, 1, 0, List.of(), 0.0, 0.0);

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
