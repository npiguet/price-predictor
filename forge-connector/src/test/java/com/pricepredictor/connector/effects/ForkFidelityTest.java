package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.ai.simulation.GameCopier;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The fork fidelity audit: a copy compared with the game it came from on
 * everything a record's state says, before anything in the copy is changed.
 */
@ExtendWith(ForgeExtension.class)
class ForkFidelityTest {

    private static Game board() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = live.getPlayers().get(0);
        Player opponent = live.getPlayers().get(1);
        TestBoard.inPlay(live, you, "Grizzly Bears").addCounterInternal(
                CounterEnumType.P1P1, 2, you, false, null, java.util.Map.of());
        TestBoard.inPlay(live, you, "Forest").tap(true, null, null);
        TestBoard.inPlay(live, opponent, "Serra Angel");
        TestBoard.inHand(live, you, "Lightning Bolt");
        TestBoard.inGraveyard(live, opponent, "Giant Growth");
        live.getAction().checkStaticAbilities();
        return live;
    }

    @Test
    void aFaithfulCopyDiffersInNothing() {
        Game live = board();
        Game fork = new GameCopier(live).makeCopy();

        assertEquals(List.of(), ForkFidelity.differences(live, fork));
    }

    @Test
    void aCopyThatDiffersIsNamedByEntityAndField() {
        Game live = board();
        GameCopier copier = new GameCopier(live);
        Game fork = copier.makeCopy();
        Card angel = live.getCardsIn(forge.game.zone.ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Serra Angel")).findFirst().orElseThrow();
        copier.find(angel).tap(true, null, null);

        List<String> differences = ForkFidelity.differences(live, fork);

        assertTrue(differences.stream().anyMatch(line -> line.startsWith(
                "entity " + Refs.of(angel) + " (Serra Angel).tapped: live=false fork=true")),
                differences.toString());
    }

    @Test
    void aPlayerThatDiffersIsNamedByField() {
        Game live = board();
        GameCopier copier = new GameCopier(live);
        Game fork = copier.makeCopy();
        Player you = live.getPlayers().get(0);
        ((Player) copier.find(you)).setLife(13, null);

        List<String> differences = ForkFidelity.differences(live, fork);

        assertTrue(differences.contains("player " + Refs.of(you) + ".life: live=20 fork=13"),
                differences.toString());
    }
}
