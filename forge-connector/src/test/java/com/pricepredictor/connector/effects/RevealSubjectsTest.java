package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordEvents.events;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static com.pricepredictor.connector.effects.RecordPicks.withEventOfType;
import static com.pricepredictor.connector.effects.TestBoard.castAndResolve;
import static com.pricepredictor.connector.effects.TestBoard.nextStep;
import static com.pricepredictor.connector.effects.TestBoard.spellOf;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * A {@code card_revealed} event names the player who revealed the cards and
 * the zone they were revealed from, whichever zone that is: a reader tells a
 * library's reveal from a hand's by {@code from_zone}, never by whether the
 * event names anybody.
 */
@ExtendWith(ForgeExtension.class)
class RevealSubjectsTest {

    private static final String WHY = "to file a resolution's outcome notes";

    @TempDir
    Path tempDir;

    /** The one {@code card_revealed} event among {@code records}. */
    private static Map<String, Object> theReveal(List<String> records) {
        List<String> revealing = withEventOfType(records, "card_revealed");
        assertEquals(1, revealing.size(), records.toString());
        List<Map<String, Object>> reveals = events(revealing.get(0)).stream()
                .filter(event -> "card_revealed".equals(event.get("type"))).toList();
        assertEquals(1, reveals.size(), revealing.get(0));
        return reveals.get(0);
    }

    /** Mulch: "Reveal the top four cards of your library." Its caster reveals them. */
    @Test
    void aLibraryRevealNamesTheRevealingPlayer() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        for (String name : List.of("Forest", "Grizzly Bears", "Island", "Shock")) {
            game.getAction().moveToLibrary(TestBoard.card(game, caster, name), null);
        }
        SpellAbility mulch = spellOf(TestBoard.inHand(game, caster, "Mulch"), caster);

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            castAndResolve(game, mulch);
            nextStep(game);
        });

        Map<String, Object> reveal = theReveal(records);
        assertEquals(List.of(Refs.of(caster)), reveal.get("subjects"), reveal.toString());
        assertEquals("library", params(reveal).get("from_zone"), reveal.toString());
        assertEquals(4L, ((Number) params(reveal).get("count")).longValue(), reveal.toString());
    }

    /**
     * Duress: "Target opponent reveals their hand." The opponent reveals, so
     * the event names the opponent and not the caster, from the hand.
     */
    @Test
    void aHandRevealNamesTheRevealingPlayer() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        TestBoard.inHand(game, opponent, "Shock");
        TestBoard.inHand(game, opponent, "Grizzly Bears");
        SpellAbility duress = spellOf(TestBoard.inHand(game, caster, "Duress"), caster);
        duress.resetTargets();
        duress.getTargets().add(opponent);

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            castAndResolve(game, duress);
            nextStep(game);
        });

        Card discarded = game.getCardsIn(ZoneType.Graveyard).stream()
                .filter(card -> "Shock".equals(card.getName())).findFirst().orElse(null);
        assertEquals(opponent, discarded == null ? null : discarded.getOwner(),
                "the premise: Duress resolved and took the Shock");
        Map<String, Object> reveal = theReveal(records);
        assertEquals(List.of(Refs.of(opponent)), reveal.get("subjects"), reveal.toString());
        assertEquals("hand", params(reveal).get("from_zone"), reveal.toString());
    }
}
