package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.event.GameEventGameOutcome;
import forge.game.event.GameEventTurnPhase;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A resolution's effect half carries the deaths its own effects cause.
 *
 * <p>Forge resolves an ability, fires {@code GameEventSpellResolved}, and only
 * at the next priority runs the state-based check that moves a creature with
 * lethal damage to the graveyard. The collector used to write the effect half
 * at the resolved event, so a Lightning Bolt at a 2/2 recorded the damage and
 * never the death: across twelve full-strength shards, none of 515
 * lethal-looking resolution damage events had the death in the same record.
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

    /** The card's own spell, activated by its owner. */
    private static SpellAbility spellOf(Card card, Player owner) {
        SpellAbility spell = card.getFirstSpellAbility();
        assertNotNull(spell, "no spell on " + card);
        spell.setActivatingPlayer(owner);
        return spell;
    }

    private static SpellAbility targeting(SpellAbility spell, Card target) {
        spell.resetTargets();
        spell.getTargets().add(target);
        return spell;
    }

    /**
     * Cast, resolve, and run the state-based check the next priority runs —
     * in the order {@code PhaseHandler.mainLoopStep} runs them.
     */
    private static void castAndResolve(Game game, SpellAbility spell) {
        game.getStack().add(spell);
        game.getStack().resolveStack();
        game.getAction().checkStateEffects(false);
    }

    /** The next real action: a step boundary. */
    private static void nextStep(Game game) {
        game.fireEvent(new GameEventTurnPhase(
                game.getPhaseHandler().getPlayerTurn(), PhaseType.MAIN2, ""));
    }

    private static List<String> effectHalves(List<String> records) {
        List<String> halves = new ArrayList<>();
        for (String record : records) {
            if (record.contains("\"moment\":\"resolution\"")
                    && record.contains("\"link_id\":\"")) {
                halves.add(record);
            }
        }
        return halves;
    }

    /** The rendered event that says a card left the battlefield for the graveyard. */
    private static String deathOf(Card card) {
        return "{\"type\":\"zone_change\",\"subjects\":[\"E" + card.getId()
                + "\"],\"params\":{\"from_zone\":\"battlefield\",\"to_zone\":\"graveyard\"";
    }

    private static int count(String haystack, String needle) {
        int found = 0;
        for (int at = haystack.indexOf(needle); at >= 0; at = haystack.indexOf(needle, at + 1)) {
            found++;
        }
        return found;
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
        SpellAbility shock = targeting(spellOf(TestBoard.inHand(game, caster, "Shock"), caster), giant);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            castAndResolve(game, bolt);
            game.getStack().add(shock);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
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
                tempDir, game, "", (bracket, collectors) -> {
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
        SpellAbility shock = targeting(spellOf(TestBoard.inHand(game, caster, "Shock"), caster), giant);

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
        SpellAbility shock = targeting(spellOf(TestBoard.inHand(game, caster, "Shock"), caster), giant);

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
        SpellAbility shock = targeting(spellOf(TestBoard.inHand(game, caster, "Shock"), caster), giant);

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
        SpellAbility shock = targeting(spellOf(TestBoard.inHand(game, caster, "Shock"), caster), giant);
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
        SpellAbility norn = spellOf(TestBoard.inHand(game, caster, "Elesh Norn, Grand Cenobite"), caster);

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
        int checked = 0;
        for (long seed = 0; seed < 40 && checked == 0; seed++) {
            Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
            Player caster = game.getPlayers().get(0);
            Card elves = TestBoard.onBattlefield(game, game.getPlayers().get(1), "Llanowar Elves");
            SpellAbility charm = TestBoard.charmOf(
                    TestBoard.inHand(game, caster, "Fiery Confluence"), caster);
            CollectionCaps caps = new CollectionCaps(
                    2000, 0.1, 2, 0, List.of(), 0.0, 0.0);

            ShardRun.Fork fork = ShardRun.forkedWithClauseHook(
                    tempDir.resolve("fork-" + seed), game, caps, seed, charm,
                    "to split a fork's modes");
            if (!fork.wrote()) {
                continue;
            }
            List<String> halves = fork.records();
            if (halves.size() < 2 || !halves.get(halves.size() - 1).contains("\"option\":1}")) {
                continue;
            }
            int lastTouching = -1;
            for (int i = 0; i < halves.size(); i++) {
                if (halves.get(i).contains("\"option\":0}")) {
                    lastTouching = i;
                }
            }
            if (lastTouching < 0) {
                continue;
            }
            checked++;
            for (int i = 0; i < halves.size(); i++) {
                assertEquals(i == lastTouching ? 1 : 0, count(halves.get(i), deathOf(elves)),
                        "half " + i + " of " + halves);
            }
        }
        assertTrue(checked > 0, "no seed drew a creature mode followed by a last opponent mode");
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
        assertFalse(halves.get(0).contains("player_won"), halves.get(0));
        int effectAt = records.indexOf(halves.get(0));
        int outcomeAt = -1;
        for (int i = 0; i < records.size(); i++) {
            if (records.get(i).contains("\"type\":\"player_won\"")) {
                outcomeAt = i;
            }
        }
        assertTrue(outcomeAt > effectAt, "the outcome follows the held half: " + records);
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
                tempDir, game, "to split the modes", (bracket, collectors) -> {
                    castAndResolve(game, charm);
                    nextStep(game);
                });

        List<String> halves = effectHalves(records);
        assertEquals(2, halves.size(), records.toString());
        assertTrue(halves.get(0).contains("\"option\":0}"), halves.get(0));
        assertTrue(halves.get(0).contains(deathOf(elves)), halves.get(0));
        assertTrue(halves.get(1).contains("\"option\":1}"), halves.get(1));
        assertFalse(halves.get(1).contains(deathOf(elves)), halves.get(1));
        // The cost half and the link are what they were.
        String cost = records.get(0);
        assertTrue(cost.contains("\"moment\":\"activation\""), cost);
        String link = cost.substring(cost.indexOf("\"link_id\":"), cost.indexOf(",\"mirror_of\""));
        for (String half : halves) {
            assertTrue(half.contains(link), half);
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
                tempDir, game, "to split the modes", (bracket, collectors) -> {
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
     */
    @Test
    void anInterventionalBoltAlreadyCarriesItsDeath() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.onBattlefield(game, game.getPlayers().get(1), "Grizzly Bears");
        Card boltCard = TestBoard.inHand(game, caster, "Lightning Bolt");
        SpellAbility bolt = spellOf(boltCard, caster);
        // A one-creature board: the fork's random target draw has bears or a
        // player to choose from, so try seeds until one picks the bears.
        CollectionCaps caps = new CollectionCaps(
                2000, 0.1, 50, 0, List.of(), 0.0, 0.0);

        List<String> records = new ArrayList<>();
        for (long seed = 0; seed < 50; seed++) {
            ShardRun.Fork fork = ShardRun.forked(
                    tempDir.resolve("fork-" + seed), game, caps, seed, bolt);
            if (fork.wrote()) {
                records.addAll(fork.records());
            }
        }

        String hitTheBears = null;
        for (String record : records) {
            if (record.contains("\"type\":\"damage_dealt\",\"subjects\":[\"E" + bears.getId() + "\"]")) {
                hitTheBears = record;
            }
        }
        assertNotNull(hitTheBears, "no fork targeted the bears: " + records);
        // Rendered exactly as the observed record renders it.
        assertTrue(hitTheBears.contains(deathOf(bears) + ",\"cause\":\"E"
                + boltCard.getId() + "\"},\"duration\":\"instant\","
                + "\"attributed_to\":\"unresolved\"}"), hitTheBears);
    }
}
