package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.cost.Cost;
import forge.game.phase.PhaseType;
import forge.game.spellability.AbilityStatic;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Free activations no printed line made, and the bookkeeping of those on the stack. */
@ExtendWith(ForgeExtension.class)
class FruitlessUnkeyedActivationsTest {

    /** An ability with no script line, as one an Aura or Equipment grants is. */
    private static SpellAbility grantedAbility(Card host, String text) {
        SpellAbility ability = new AbilityStatic(host, new Cost("0", true), null) {
            @Override
            public void resolve() {
            }
        };
        ability.setDescription(text);
        host.addSpellAbility(ability);
        return ability;
    }

    /** An activation with no line key is remembered by its card and its text. */
    @Test
    void anUnkeyableActivationIsRememberedByCardAndText() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Card bears = TestBoard.onBattlefield(game, game.getPlayers().get(0), "Grizzly Bears");
        Card other = TestBoard.onBattlefield(game, game.getPlayers().get(0), "Grizzly Bears");
        SpellAbility granted = grantedAbility(bears, "Do nothing.");
        SpellAbility another = grantedAbility(bears, "Do something else.");
        SpellAbility othersGranted = grantedAbility(other, "Do nothing.");
        assertNull(ProvenanceResolver.of(granted), "the board has no line to key it by");
        FruitlessActivations fruitless = new FruitlessActivations(game);

        fruitless.report(granted, null);

        assertTrue(fruitless.isFruitless(granted));
        assertFalse(fruitless.isFruitless(another), "another text on the card is its own ability");
        assertFalse(fruitless.isFruitless(othersGranted), "another card's ability is its own");
    }

    /** Entries of abilities no longer on the stack are dropped when the turn changes. */
    @Test
    void theStackEntriesAreDroppedAtTheTurnChange() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Card bears = TestBoard.onBattlefield(game, game.getPlayers().get(0), "Grizzly Bears");
        FruitlessActivations fruitless = new FruitlessActivations(game);

        fruitless.pending(grantedAbility(bears, "One."), null);
        fruitless.pending(grantedAbility(bears, "Two."), null);
        assertEquals(2, fruitless.trackedOnStack());
        game.getPhaseHandler().devModeSet(PhaseType.MAIN1, game.getPlayers().get(1), 2);
        fruitless.pending(grantedAbility(bears, "Three."), null);

        assertEquals(1, fruitless.trackedOnStack(), "only the one pending since the turn changed");
    }
}
