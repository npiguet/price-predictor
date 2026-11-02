package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** What the random seat is told not to draw again this turn (FR-040). */
@ExtendWith(ForgeExtension.class)
class FruitlessActivationsTest {

    /** Nothing, or an ability change granting and removing nothing, is no change; anything else is. */
    @Test
    void onlyAnEmptyAbilityChangeChangesNothing() {
        EffectEvent empty = new EffectEvent(EventType.ABILITY_CHANGE)
                .param("abilities", "").param("removed", false);
        EffectEvent granted = new EffectEvent(EventType.ABILITY_CHANGE)
                .param("abilities", "ABPump").param("removed", false);
        EffectEvent removed = new EffectEvent(EventType.ABILITY_CHANGE)
                .param("abilities", "").param("removed", true);
        EffectEvent stats = new EffectEvent(EventType.PT_CHANGE).subject("E1");

        assertTrue(FruitlessActivations.changesNothing(List.of()));
        assertTrue(FruitlessActivations.changesNothing(List.of(empty)));
        assertFalse(FruitlessActivations.changesNothing(List.of(granted)));
        assertFalse(FruitlessActivations.changesNothing(List.of(removed)));
        assertFalse(FruitlessActivations.changesNothing(List.of(empty, stats)));
    }

    /** A reported activation is fruitless for its card's line until the turn changes. */
    @Test
    void aReportLastsTheTurn() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Card bears = TestBoard.onBattlefield(game, game.getPlayers().get(0), "Grizzly Bears");
        Card other = TestBoard.onBattlefield(game, game.getPlayers().get(0), "Grizzly Bears");
        String script = "AB$ Animate | Cost$ 0 | Power$ 2 | Toughness$ 2 | Defined$ Self";
        SpellAbility ability = AbilityFactory.getAbility(script, bears);
        bears.addSpellAbility(ability);
        SpellAbility othersAbility = AbilityFactory.getAbility(script, other);
        other.addSpellAbility(othersAbility);
        FruitlessActivations fruitless = new FruitlessActivations(game);

        fruitless.report(bears, ProvenanceResolver.of(ability));

        assertTrue(fruitless.isFruitless(ability));
        assertFalse(fruitless.isFruitless(othersAbility), "another card's activation is its own");
        game.getPhaseHandler().devModeSet(PhaseType.MAIN1, game.getPlayers().get(1), 2);
        assertFalse(fruitless.isFruitless(ability), "cleared at the next turn");
    }
}
