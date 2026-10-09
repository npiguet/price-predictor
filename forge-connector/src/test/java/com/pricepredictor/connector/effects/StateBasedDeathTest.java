package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.PatchedCollectors.CollectionCaps;
import forge.StaticData;
import forge.ai.LobbyPlayerAi;
import forge.deck.Deck;
import forge.game.Game;
import forge.game.GameRules;
import forge.game.GameStage;
import forge.game.GameType;
import forge.game.Match;
import forge.game.ability.ApiType;
import forge.game.ability.effects.CharmEffect;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.event.GameEventGameOutcome;
import forge.game.event.GameEventTurnPhase;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.zip.GZIPInputStream;

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

    /** A two-player game in its first main phase, past the start. */
    private static Game game() {
        List<RegisteredPlayer> players = List.of(
                new RegisteredPlayer(new Deck()).setPlayer(new LobbyPlayerAi("a", null)),
                new RegisteredPlayer(new Deck()).setPlayer(new LobbyPlayerAi("b", null)));
        GameRules rules = new GameRules(GameType.Constructed);
        Game game = new Game(players, rules, new Match(rules, players, "StateBasedDeathTest"));
        game.getPhaseHandler().devModeSet(PhaseType.MAIN1, game.getPlayers().get(0));
        // checkStateEffects returns early, with the stack left frozen, on a
        // game that has not started.
        game.setAge(GameStage.Play);
        return game;
    }

    private static Card card(Game game, String name, Player owner) {
        return CardFactory.getCard(
                StaticData.instance().getCommonCards().getCard(name), owner, game);
    }

    private static Card onBattlefield(Game game, String name, Player owner) {
        return game.getAction().moveToPlay(card(game, name, owner), owner, null, null);
    }

    private static Card inHand(Game game, String name, Player owner) {
        return game.getAction().moveTo(ZoneType.Hand, card(game, name, owner), null, null);
    }

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

    private RecordShardWriter writer() {
        return new RecordShardWriter(tempDir, "run", 0, "l1");
    }

    private static List<String> readShard(Path path) throws IOException {
        if (!Files.exists(path)) {
            return List.of();
        }
        try (var gzip = new GZIPInputStream(Files.newInputStream(path));
                var reader = new BufferedReader(
                        new InputStreamReader(gzip, StandardCharsets.UTF_8))) {
            return reader.lines().toList();
        }
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
    void aLethalBoltCarriesTheDeathItCausedInItsEffectHalf() throws IOException {
        Game game = game();
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = onBattlefield(game, "Grizzly Bears", opponent);
        SpellAbility bolt = targeting(
                spellOf(inHand(game, "Lightning Bolt", caster), caster), bears);

        RecordShardWriter writer = writer();
        BusBracketCollector collector = new BusBracketCollector(
                game, writer, "run.0-l1.0", CollectionCaps.defaults());
        game.subscribeToEvents(collector);
        try {
            castAndResolve(game, bolt);
            assertTrue(game.getCardsIn(ZoneType.Graveyard).stream()
                    .anyMatch(c -> c.getId() == bears.getId()), "the bears died");
            nextStep(game);
        } finally {
            writer.close();
        }

        List<String> halves = effectHalves(readShard(writer.path()));
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
    void aHeldHalfIsWrittenOnceWhenTheNextActionBegins() throws IOException {
        Game game = game();
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = onBattlefield(game, "Grizzly Bears", opponent);
        Card giant = onBattlefield(game, "Hill Giant", opponent);
        SpellAbility bolt = targeting(
                spellOf(inHand(game, "Lightning Bolt", caster), caster), bears);
        SpellAbility shock = targeting(spellOf(inHand(game, "Shock", caster), caster), giant);

        RecordShardWriter writer = writer();
        BusBracketCollector collector = new BusBracketCollector(
                game, writer, "run.0-l1.0", CollectionCaps.defaults());
        game.subscribeToEvents(collector);
        try {
            castAndResolve(game, bolt);
            assertEquals(1, collector.recordsWritten(), "only the cost half so far");
            game.getStack().add(shock);
            assertEquals(2, collector.recordsWritten(), "the cast wrote the held half");
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            assertEquals(3, collector.recordsWritten(), "and the shock's own half waits");
            nextStep(game);
            nextStep(game);
        } finally {
            writer.close();
        }

        List<String> records = readShard(writer.path());
        assertEquals(4, records.size(), records.toString());
        List<String> halves = effectHalves(records);
        assertEquals(2, halves.size(), records.toString());
        assertTrue(halves.get(0).contains(deathOf(bears)), halves.get(0));
        assertFalse(halves.get(1).contains(deathOf(bears)), halves.get(1));
        assertFalse(halves.get(1).contains(deathOf(giant)), halves.get(1));
    }

    /** The same on a patched worker, where the clause hook files the damage. */
    @Test
    void aPatchedWorkerCarriesTheDeathToo() throws IOException {
        Game game = game();
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = onBattlefield(game, "Grizzly Bears", opponent);
        SpellAbility bolt = targeting(
                spellOf(inHand(game, "Lightning Bolt", caster), caster), bears);

        RecordShardWriter writer = writer();
        BusBracketCollector collector = new BusBracketCollector(
                game, writer, "run.0-l1.0", CollectionCaps.defaults());
        game.subscribeToEvents(collector);
        try (PatchedCollectors patched = new PatchedCollectors(
                game, writer, "run.0-l1.0", CollectionCaps.defaults(), 1L)) {
            Assumptions.assumeTrue(
                    patched.install() > 0 && patched.installedHooks().contains("clause-events"),
                    "../forge has no clause hook");
            patched.withBracket(collector);
            castAndResolve(game, bolt);
            nextStep(game);
        } finally {
            writer.close();
        }

        List<String> halves = effectHalves(readShard(writer.path()));
        assertEquals(1, halves.size(), halves.toString());
        assertTrue(halves.get(0).contains(deathOf(bears)), halves.get(0));
    }

    /** A creature that survived its damage has no death to carry. */
    @Test
    void aNonLethalShockCarriesNoDeath() throws IOException {
        Game game = game();
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card giant = onBattlefield(game, "Hill Giant", opponent);
        SpellAbility shock = targeting(spellOf(inHand(game, "Shock", caster), caster), giant);

        RecordShardWriter writer = writer();
        BusBracketCollector collector = new BusBracketCollector(
                game, writer, "run.0-l1.0", CollectionCaps.defaults());
        game.subscribeToEvents(collector);
        try {
            castAndResolve(game, shock);
            nextStep(game);
        } finally {
            writer.close();
        }

        List<String> records = readShard(writer.path());
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
    void aLandPlayedAfterwardsDoesNotJoinTheRecord() throws IOException {
        Game game = game();
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = onBattlefield(game, "Grizzly Bears", opponent);
        SpellAbility bolt = targeting(
                spellOf(inHand(game, "Lightning Bolt", caster), caster), bears);
        Card mountain = inHand(game, "Mountain", caster);

        RecordShardWriter writer = writer();
        BusBracketCollector collector = new BusBracketCollector(
                game, writer, "run.0-l1.0", CollectionCaps.defaults());
        game.subscribeToEvents(collector);
        try {
            castAndResolve(game, bolt);
            long held = collector.recordsWritten();
            caster.playLand(mountain, null);
            assertEquals(held + 1, collector.recordsWritten(),
                    "playing the land writes the held effect half");
            nextStep(game);
        } finally {
            writer.close();
        }

        List<String> records = readShard(writer.path());
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
    void aTouchedPermanentLeavingOutsideTheStateBasedCheckDoesNotJoin() throws IOException {
        Game game = game();
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card giant = onBattlefield(game, "Hill Giant", opponent);
        SpellAbility shock = targeting(spellOf(inHand(game, "Shock", caster), caster), giant);

        RecordShardWriter writer = writer();
        BusBracketCollector collector = new BusBracketCollector(
                game, writer, "run.0-l1.0", CollectionCaps.defaults());
        game.subscribeToEvents(collector);
        try {
            castAndResolve(game, shock);
            // Sacrificed as a cost would be: no state-based check is running.
            game.getAction().moveToGraveyard(giant, null);
            nextStep(game);
        } finally {
            writer.close();
        }

        List<String> halves = effectHalves(readShard(writer.path()));
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
    void aDeathInTheFollowingCheckJoinsEvenUntouched() throws IOException {
        Game game = game();
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card giant = onBattlefield(game, "Hill Giant", opponent);
        Card bystander = onBattlefield(game, "Grizzly Bears", opponent);
        SpellAbility shock = targeting(spellOf(inHand(game, "Shock", caster), caster), giant);

        RecordShardWriter writer = writer();
        BusBracketCollector collector = new BusBracketCollector(
                game, writer, "run.0-l1.0", CollectionCaps.defaults());
        game.subscribeToEvents(collector);
        try {
            game.getStack().add(shock);
            game.getStack().resolveStack();
            // Lethal damage the bracket never saw, standing in for what a
            // static the resolution brought does to a creature it never named.
            bystander.setDamage(2);
            game.getAction().checkStateEffects(false);
            nextStep(game);
        } finally {
            writer.close();
        }

        List<String> halves = effectHalves(readShard(writer.path()));
        assertEquals(1, halves.size(), halves.toString());
        assertTrue(halves.get(0).contains(deathOf(bystander)), halves.get(0));
    }

    /**
     * A check after the next action is that action's: once a land is played
     * the held half is written, and a death in the check that follows the
     * land joins nothing.
     */
    @Test
    void aDeathInACheckAfterTheNextActionDoesNotJoin() throws IOException {
        Game game = game();
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card giant = onBattlefield(game, "Hill Giant", opponent);
        Card bystander = onBattlefield(game, "Grizzly Bears", opponent);
        SpellAbility shock = targeting(spellOf(inHand(game, "Shock", caster), caster), giant);
        Card mountain = inHand(game, "Mountain", caster);

        RecordShardWriter writer = writer();
        BusBracketCollector collector = new BusBracketCollector(
                game, writer, "run.0-l1.0", CollectionCaps.defaults());
        game.subscribeToEvents(collector);
        try {
            castAndResolve(game, shock);
            caster.playLand(mountain, null);
            bystander.setDamage(2);
            game.getAction().checkStateEffects(false);
            nextStep(game);
        } finally {
            writer.close();
        }

        for (String record : readShard(writer.path())) {
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
    void aMinusAuraKillJoinsTheAuraSpellsRecord() throws IOException {
        Game game = game();
        Player caster = game.getPlayers().get(0);
        Card bears = onBattlefield(game, "Grizzly Bears", game.getPlayers().get(1));
        Card aura = inHand(game, "Dead Weight", caster);
        SpellAbility deadWeight = targeting(spellOf(aura, caster), bears);

        RecordShardWriter writer = writer();
        BusBracketCollector collector = new BusBracketCollector(
                game, writer, "run.0-l1.0", CollectionCaps.defaults());
        game.subscribeToEvents(collector);
        try {
            castAndResolve(game, deadWeight);
            assertTrue(game.getCardsIn(ZoneType.Graveyard).stream()
                    .anyMatch(c -> c.getId() == bears.getId()), "the bears died");
            nextStep(game);
        } finally {
            writer.close();
        }

        List<String> halves = effectHalves(readShard(writer.path()));
        assertEquals(1, halves.size(), halves.toString());
        String effect = halves.get(0);
        assertTrue(effect.contains(deathOf(bears)), effect);
        assertTrue(effect.contains(deathOf(aura)), "the Aura fell off: " + effect);
        assertFalse(effect.contains("\"type\":\"pt_change\",\"subjects\":[\"E" + bears.getId()),
                "the static's shrink is not the spell's event: " + effect);
    }

    /** A static that shrinks on entry kills in the check after its creature resolves. */
    @Test
    void aStaticThatKillsOnEntryJoinsTheCreatureSpellsRecord() throws IOException {
        Game game = game();
        Player caster = game.getPlayers().get(0);
        Card bears = onBattlefield(game, "Grizzly Bears", game.getPlayers().get(1));
        SpellAbility norn = spellOf(inHand(game, "Elesh Norn, Grand Cenobite", caster), caster);

        RecordShardWriter writer = writer();
        BusBracketCollector collector = new BusBracketCollector(
                game, writer, "run.0-l1.0", CollectionCaps.defaults());
        game.subscribeToEvents(collector);
        try {
            castAndResolve(game, norn);
            nextStep(game);
        } finally {
            writer.close();
        }

        List<String> halves = effectHalves(readShard(writer.path()));
        assertEquals(1, halves.size(), halves.toString());
        assertTrue(halves.get(0).contains(deathOf(bears)), halves.get(0));
    }

    /** The legend rule is a state-based action too, and its loser joins the record. */
    @Test
    void theLegendRulesLoserJoinsTheSecondLegendsRecord() throws IOException {
        Game game = game();
        Player caster = game.getPlayers().get(0);
        Card first = onBattlefield(game, "Isamaru, Hound of Konda", caster);
        Card secondCard = inHand(game, "Isamaru, Hound of Konda", caster);
        SpellAbility second = spellOf(secondCard, caster);

        RecordShardWriter writer = writer();
        BusBracketCollector collector = new BusBracketCollector(
                game, writer, "run.0-l1.0", CollectionCaps.defaults());
        game.subscribeToEvents(collector);
        try {
            castAndResolve(game, second);
            nextStep(game);
        } finally {
            writer.close();
        }

        List<String> halves = effectHalves(readShard(writer.path()));
        assertEquals(1, halves.size(), halves.toString());
        assertTrue(halves.get(0).contains(deathOf(first))
                        || halves.get(0).contains(deathOf(secondCard)),
                "one Isamaru went to the graveyard: " + halves.get(0));
    }

    /** An Aura falls off when the creature it enchanted dies, in the same check. */
    @Test
    void anAuraFallingOffItsDeadHostJoinsTheRecord() throws IOException {
        Game game = game();
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = onBattlefield(game, "Grizzly Bears", opponent);
        Card pacifism = onBattlefield(game, "Pacifism", caster);
        pacifism.attachToEntity(bears, null);
        SpellAbility bolt = targeting(
                spellOf(inHand(game, "Lightning Bolt", caster), caster), bears);

        RecordShardWriter writer = writer();
        BusBracketCollector collector = new BusBracketCollector(
                game, writer, "run.0-l1.0", CollectionCaps.defaults());
        game.subscribeToEvents(collector);
        try {
            castAndResolve(game, bolt);
            nextStep(game);
        } finally {
            writer.close();
        }

        List<String> halves = effectHalves(readShard(writer.path()));
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
    void aModalForkFilesTheDeathInTheModeThatTouchedTheCreature() throws IOException {
        int checked = 0;
        for (long seed = 0; seed < 40 && checked == 0; seed++) {
            Game game = game();
            Player caster = game.getPlayers().get(0);
            Card elves = onBattlefield(game, "Llanowar Elves", game.getPlayers().get(1));
            SpellAbility charm = null;
            for (SpellAbility sa : inHand(game, "Fiery Confluence", caster).getSpellAbilities()) {
                if (sa.getApi() == ApiType.Charm) {
                    charm = sa;
                }
            }
            assertNotNull(charm);
            charm.setActivatingPlayer(caster);
            CollectionCaps caps = new CollectionCaps(
                    2000, 0.1, 2, 0, List.of(), List.of(1, 2, 3), 0.1);

            Path dir = tempDir.resolve("fork-" + seed);
            RecordShardWriter writer = new RecordShardWriter(dir, "run", 0, "l1");
            try (PatchedCollectors patched = new PatchedCollectors(
                    game, writer, "run.0-l1.0", caps, 1L)) {
                Assumptions.assumeTrue(
                        patched.install() > 0 && patched.installedHooks().contains("clause-events"),
                        "../forge has no clause hook to split a fork's modes");
                ForkCollector forks = new ForkCollector(game, writer, "run.0-l1.0", caps, seed);
                patched.withForks(forks);
                forks.intervene(charm, 1);
            } finally {
                writer.close();
            }

            List<String> halves = readShard(writer.path());
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
    void theGameEndingWritesTheHeldHalfBeforeTheOutcome() throws IOException {
        Game game = game();
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = onBattlefield(game, "Grizzly Bears", opponent);
        SpellAbility bolt = targeting(
                spellOf(inHand(game, "Lightning Bolt", caster), caster), bears);

        RecordShardWriter writer = writer();
        BusBracketCollector collector = new BusBracketCollector(
                game, writer, "run.0-l1.0", CollectionCaps.defaults());
        game.subscribeToEvents(collector);
        try {
            castAndResolve(game, bolt);
            collector.onGameOutcome(new GameEventGameOutcome(
                    1, List.of("a has won"), "a", "a: 1 b: 0 "));
            collector.finishGame(Set.of());
        } finally {
            writer.close();
        }

        List<String> records = readShard(writer.path());
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

    /** Chain the modes at these Choices$ positions, as the cast would. */
    private static SpellAbility charm(Game game, Player caster, int... positions) {
        Card confluence = inHand(game, "Fiery Confluence", caster);
        SpellAbility charm = null;
        for (SpellAbility sa : confluence.getSpellAbilities()) {
            if (sa.getApi() == ApiType.Charm) {
                charm = sa;
            }
        }
        assertNotNull(charm);
        charm.setActivatingPlayer(caster);
        List<AbilitySub> choices = charm.getAdditionalAbilityList("Choices");
        List<AbilitySub> chosen = new ArrayList<>();
        for (int position : positions) {
            chosen.add(choices.get(position));
        }
        charm.setChosenList(chosen);
        CharmEffect.chainAbilities(charm, chosen);
        return charm;
    }

    /**
     * A death goes to the mode that touched the creature, not to the last
     * mode: Fiery Confluence's "1 damage to each creature" kills the elf, and
     * its "2 damage to each opponent" half, which resolves after, never saw it.
     */
    @Test
    void aModalDeathJoinsTheModeThatTouchedTheCreature() throws IOException {
        Game game = game();
        Player caster = game.getPlayers().get(0);
        Card elves = onBattlefield(game, "Llanowar Elves", game.getPlayers().get(1));
        SpellAbility charm = charm(game, caster, 0, 1);

        RecordShardWriter writer = writer();
        BusBracketCollector collector = new BusBracketCollector(
                game, writer, "run.0-l1.0", CollectionCaps.defaults());
        game.subscribeToEvents(collector);
        try (PatchedCollectors patched = new PatchedCollectors(
                game, writer, "run.0-l1.0", CollectionCaps.defaults(), 1L)) {
            Assumptions.assumeTrue(
                    patched.install() > 0 && patched.installedHooks().contains("clause-events"),
                    "../forge has no clause hook to split the modes");
            patched.withBracket(collector);
            castAndResolve(game, charm);
            nextStep(game);
        } finally {
            writer.close();
        }

        List<String> records = readShard(writer.path());
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
    void aDeathTouchedByTwoModesJoinsTheLastOfThemOnce() throws IOException {
        Game game = game();
        Player caster = game.getPlayers().get(0);
        Card bears = onBattlefield(game, "Grizzly Bears", game.getPlayers().get(1));
        SpellAbility charm = charm(game, caster, 0, 0, 1);

        RecordShardWriter writer = writer();
        BusBracketCollector collector = new BusBracketCollector(
                game, writer, "run.0-l1.0", CollectionCaps.defaults());
        game.subscribeToEvents(collector);
        try (PatchedCollectors patched = new PatchedCollectors(
                game, writer, "run.0-l1.0", CollectionCaps.defaults(), 1L)) {
            Assumptions.assumeTrue(
                    patched.install() > 0 && patched.installedHooks().contains("clause-events"),
                    "../forge has no clause hook to split the modes");
            patched.withBracket(collector);
            castAndResolve(game, charm);
            nextStep(game);
        } finally {
            writer.close();
        }

        List<String> halves = effectHalves(readShard(writer.path()));
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
    void anInterventionalBoltAlreadyCarriesItsDeath() throws IOException {
        Game game = game();
        Player caster = game.getPlayers().get(0);
        Card bears = onBattlefield(game, "Grizzly Bears", game.getPlayers().get(1));
        Card boltCard = inHand(game, "Lightning Bolt", caster);
        SpellAbility bolt = spellOf(boltCard, caster);
        // A one-creature board: the fork's random target draw has bears or a
        // player to choose from, so try seeds until one picks the bears.
        CollectionCaps caps = new CollectionCaps(
                2000, 0.1, 50, 0, List.of(), List.of(1, 2, 3), 0.1);

        RecordShardWriter writer = writer();
        try (PatchedCollectors patched = new PatchedCollectors(
                game, writer, "run.0-l1.0", caps, 1L)) {
            for (long seed = 0; seed < 50; seed++) {
                ForkCollector forks = new ForkCollector(game, writer, "run.0-l1.0", caps, seed);
                patched.withForks(forks);
                forks.intervene(bolt, (int) seed);
            }
        } finally {
            writer.close();
        }

        List<String> records = readShard(writer.path());
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
