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
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.event.GameEventTurnPhase;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.GZIPInputStream;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A card's first change in power and toughness is recorded.
 *
 * <p>The bus says only that a card's stats changed, and Forge says it after the
 * change is applied, so a differ with no earlier reading of the card has
 * nothing to compare against: the first pump on a creature would leave no
 * {@code pt_change}, and its wearing off would read as a lone -3/-3. These
 * drive the real engine, because what is under test is when the engine
 * publishes relative to when it changes the card.
 */
@ExtendWith(ForgeExtension.class)
class FirstStatChangeTest {

    @TempDir
    Path tempDir;

    /** The rendered start of a +3/+3 on a card. */
    private static String pumpOf(Card card) {
        return "{\"type\":\"pt_change\",\"subjects\":[\"E" + card.getId()
                + "\"],\"params\":{\"power_delta\":3,\"toughness_delta\":3";
    }

    /** A creature that has just entered, then pumped, writes the pump. */
    @Test
    void aCreatureThatJustEnteredRecordsItsFirstPump() throws IOException {
        Game game = game();
        Player caster = game.getPlayers().get(0);
        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        Card bears;
        try {
            game.subscribeToEvents(new BusBracketCollector(game, writer, "run.0-l1.0"));
            bears = onBattlefield(game, "Grizzly Bears", caster);
            SpellAbility growth = spellOf(inHand(game, "Giant Growth", caster), caster);
            growth.resetTargets();
            growth.getTargets().add(bears);
            game.getStack().add(growth);
            game.getStack().resolveStack();
            game.fireEvent(new GameEventTurnPhase(caster, PhaseType.MAIN2, ""));
        } finally {
            writer.close();
        }

        String shard = String.join("\n", readShard(writer.path()));
        assertTrue(shard.contains(pumpOf(bears)), shard);
    }

    /**
     * A fork reports the first change of a card it never saw change before:
     * its sink lives for one resolution, so every card on the copied board
     * is new to it.
     */
    @Test
    void aForkRecordsTheFirstPumpOfACreature() throws IOException {
        Game game = game();
        Player caster = game.getPlayers().get(0);
        Card bears = onBattlefield(game, "Grizzly Bears", caster);
        SpellAbility growth = spellOf(inHand(game, "Giant Growth", caster), caster);
        CollectionCaps caps = new CollectionCaps(2000, 0.1, 1, 0, List.of());

        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        try {
            new ForkCollector(game, writer, "run.0-l1.0", caps, 1L).intervene(growth, 1);
        } finally {
            writer.close();
        }

        String shard = String.join("\n", readShard(writer.path()));
        assertTrue(shard.contains("\"interventional\":true"), "no fork was written: " + shard);
        assertTrue(shard.contains(pumpOf(bears)), shard);
    }

    /** A two-player game in its first main phase, past the start. */
    private static Game game() {
        List<RegisteredPlayer> players = List.of(
                new RegisteredPlayer(new Deck()).setPlayer(new LobbyPlayerAi("a", null)),
                new RegisteredPlayer(new Deck()).setPlayer(new LobbyPlayerAi("b", null)));
        GameRules rules = new GameRules(GameType.Constructed);
        Game game = new Game(players, rules, new Match(rules, players, "FirstStatChangeTest"));
        game.getPhaseHandler().devModeSet(PhaseType.MAIN1, game.getPlayers().get(0));
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

    private static SpellAbility spellOf(Card card, Player owner) {
        SpellAbility spell = card.getFirstSpellAbility();
        spell.setActivatingPlayer(owner);
        return spell;
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
}
