package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.ai.ability.DamageDealAi;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.util.MyRandom;
import org.apache.commons.lang3.tuple.Pair;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * The AI chaining a damage spell checks the spells in its hand for itself as
 * the activator: Forge prints "Did not have activator set" when an ability
 * reaches the cast restrictions with nobody set as its activator.
 */
@ExtendWith(ForgeExtension.class)
class DamageChainActivatorTest {

    private static final String WARNING = "Did not have activator set";

    @Test
    void chainingAShockBesideALightningBoltChecksTheBoltForTheAi() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN2);
        Player ai = game.getPlayers().get(0);
        TestBoard.inPlay(game, ai, "Mountain");
        TestBoard.inPlay(game, ai, "Mountain");
        Card shock = TestBoard.inHand(game, ai, "Shock");
        Card bolt = TestBoard.inHand(game, ai, "Lightning Bolt");
        SpellAbility shockSpell = TestBoard.spellOf(shock, ai);

        PrintStream out = System.out;
        ByteArrayOutputStream printed = new ByteArrayOutputStream();
        Random was = MyRandom.getRandom();
        MyRandom.setRandom(new Random(7));
        Pair<SpellAbility, Integer> chained = null;
        try {
            System.setOut(new PrintStream(printed, true));
            for (int attempt = 0; attempt < 100 && chained == null; attempt++) {
                chained = DamageDealAi.getDamagingSAToChain(ai, shockSpell, "2");
            }
        } finally {
            System.setOut(out);
            MyRandom.setRandom(was);
        }

        assertNotNull(chained, "the AI chains the Bolt behind the Shock");
        assertEquals(bolt, chained.getLeft().getHostCard());
        assertEquals(ai, chained.getLeft().getActivatingPlayer(), "the Bolt is checked for the AI");
        assertFalse(printed.toString().contains(WARNING), printed.toString());
    }
}
