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

    /**
     * After combat: a creature attacked, another blocked it, damage was
     * dealt and a creature died. The copier copied none of the turn's combat
     * or zone history, so the copy's creatures had not attacked or blocked,
     * no damage linked them and no creature had died this turn.
     */
    @Test
    void aCopyTakenAfterCombatKeepsTheTurnsHistory() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN2);
        Player you = live.getPlayers().get(0);
        Player opponent = live.getPlayers().get(1);
        Card attacker = TestBoard.inPlay(live, you, "Hill Giant");
        Card blocker = TestBoard.inPlay(live, opponent, "Grizzly Bears");
        Card chump = TestBoard.inPlay(live, opponent, "Llanowar Elves");
        attacker.getDamageHistory().setCreatureAttackedThisCombat(opponent, 0);
        blocker.addBlockedThisTurn(attacker);
        attacker.addBlockedByThisTurn(blocker);
        attacker.getDamageHistory().registerDamage(1, true, attacker, blocker, new java.util.HashMap<>());
        live.getAction().moveTo(forge.game.zone.ZoneType.Graveyard, chump, null, null);
        assertEquals(1, live.getLeftBattlefieldThisTurn().size(), "the chump blocker did not leave");

        Game fork = new GameCopier(live).makeCopy();

        assertEquals(List.of(), ForkFidelity.differences(live, fork));
    }

    /**
     * The monarch is a command-zone effect card holding the designation's
     * triggers; the copy's effect card came out with no lines and no colour.
     */
    @Test
    void aCopyKeepsTheMonarchsEffectCard() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN2);
        live.getAction().becomeMonarch(live.getPlayers().get(0), null);

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
