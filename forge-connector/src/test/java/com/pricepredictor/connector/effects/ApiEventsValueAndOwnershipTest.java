package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityFactory;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import static com.pricepredictor.connector.effects.ApiClauseObservations.observe;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The {@link ValueAndOwnershipApiRules}: a card changing owner, and who the
 * new owner is.
 */
@ExtendWith(ForgeExtension.class)
class ApiEventsValueAndOwnershipTest {

    /**
     * Darkpact's clause carries no {@code DefinedPlayer}, so the new owner
     * falls back to the activator -- {@code null} here, since
     * {@link TestCards#game()} is shared and nothing in this suite ever
     * activates an ability as a player. What this pins is that the fallback
     * path still yields the event rather than that it names anyone in particular;
     * {@link ApiEvents} names the {@code DefinedPlayer}-driven branch's real
     * printed counter-examples (Tempest Efreet, Bronze Tablet) in its own
     * javadoc, since neither is reachable without a live target or a prior
     * {@code Remembered} player, which this harness cannot cheaply build.
     */
    @Test
    void anOwnershipChangeWithNoNamedOwnerIsStillAnOwnershipChange() {
        SpellAbility sa = TestCards.scriptedAbility("Darkpact", "GainOwnership");
        Player activator = new Player("activator", TestCards.game(), 91503);
        sa.setActivatingPlayer(activator);

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(EventType.OWNERSHIP_CHANGE, event.type());
        // OwnershipGainEffect's fallback: no DefinedPlayer, the activator.
        assertEquals(Refs.of(activator), event.param(EventParam.OWNER));
    }

    /**
     * {@code SpellAbilityEffect.getPlayers} splits a {@code " & "}-joined
     * {@code Defined}/{@code DefinedPlayer} and
     * unions each token; a rule that read the raw string as one token would
     * hand it to {@code AbilityUtils.getDefinedPlayers} unrecognized, which
     * falls through to its catch-all ({@code
     * players.addAll(game.getPlayersInTurnOrder())}) -- empty on
     * {@link TestCards#game()}'s zero-player game, so {@code ownerOf} would
     * fall back to the activator instead.
     *
     * <p>{@code You & You} could not tell that fallback apart from a correct
     * split, because both halves name the activator either way.
     * {@code Targeted & You} can: split, the owner is the targeted player;
     * unsplit, the whole string matches nothing and {@code ownerOf} falls back
     * to the activator, a different id. Deleting the {@code " & "} split in
     * {@code definedPlayers} makes this fail with the activator's id where the
     * targeted player's was expected.
     */
    @Test
    void ownershipDefinedPlayerSplitsOnAmpersand() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ GainOwnership | ValidTgts$ Card | DefinedPlayer$ Targeted & You",
                TestCards.build("Grizzly Bears"));
        Player activator = new Player("activator", TestCards.game(), 91501);
        Player targeted = new Player("targeted", TestCards.game(), 91502);
        sa.setActivatingPlayer(activator);
        sa.resetTargets();
        sa.getTargets().add(targeted);

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(Refs.of(targeted), event.param(EventParam.OWNER));
    }
}
