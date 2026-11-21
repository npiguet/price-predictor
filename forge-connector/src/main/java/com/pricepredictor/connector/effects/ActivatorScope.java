package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.IdentityHashMap;
import java.util.Map;
import java.util.function.Supplier;

/**
 * What a rules check about a candidate may set on the game, held to be put
 * back after it ({@link PlayabilityVerdict#asActivator}): the candidate's
 * activating player, those of every ability of every card in the game (the
 * mana check sets those of the mana abilities it counts on), and the host's
 * cast ability (Forge's cast-prohibition check sets it).
 *
 * <p>Finding every ability means scanning every card in the game. A scope
 * {@link #inner() opened inside} another holds the same abilities, read as
 * they stand at that moment, without a second scan.
 */
final class ActivatorScope {

    private final SpellAbility candidate;
    private final Player activator;
    /** Each ability's activating player as the scope found it, the candidate's first. */
    private final Map<SpellAbility, Player> activators;
    private final Card host;
    private final SpellAbility castBefore;

    private ActivatorScope(SpellAbility candidate, Player activator, Map<SpellAbility, Player> activators) {
        this.candidate = candidate;
        this.activator = activator;
        this.activators = activators;
        this.host = candidate.getHostCard();
        this.castBefore = host == null ? null : host.getCastSA();
    }

    /** A scope over every ability of every card in {@code activator}'s game. */
    static ActivatorScope over(SpellAbility candidate, Player activator) {
        Map<SpellAbility, Player> activators = startingWith(candidate);
        for (Card card : activator.getGame().getCardsInGame()) {
            for (SpellAbility ability : card.getSpellAbilities()) {
                activators.putIfAbsent(ability, ability.getActivatingPlayer());
            }
        }
        return new ActivatorScope(candidate, activator, activators);
    }

    /**
     * A scope over the candidate alone, with no scan: for a reader that
     * sets nothing on any other ability (a gate reading the candidate's own
     * costs and targets).
     */
    static ActivatorScope alone(SpellAbility candidate, Player activator) {
        return new ActivatorScope(candidate, activator, startingWith(candidate));
    }

    /**
     * A scope inside this one, over the same abilities as they stand now:
     * what {@link #over} would find in its place, since a check run inside
     * this scope adds no card to the game.
     */
    ActivatorScope inner() {
        Map<SpellAbility, Player> now = startingWith(candidate);
        for (SpellAbility ability : activators.keySet()) {
            now.putIfAbsent(ability, ability.getActivatingPlayer());
        }
        return new ActivatorScope(candidate, activator, now);
    }

    private static Map<SpellAbility, Player> startingWith(SpellAbility candidate) {
        Map<SpellAbility, Player> activators = new IdentityHashMap<>();
        activators.put(candidate, candidate.getActivatingPlayer());
        return activators;
    }

    /**
     * Run {@code body} with the activator as the candidate's activating
     * player and no target chosen on any clause the candidate offers
     * ({@link TargetingClauses#withNoneChosen}), then put back everything the
     * scope holds, also when it throws.
     */
    <T> T run(Supplier<T> body) {
        candidate.setActivatingPlayer(activator);
        try {
            return TargetingClauses.withNoneChosen(candidate, body);
        } finally {
            putBack();
        }
    }

    private void putBack() {
        activators.forEach((ability, before) -> {
            if (ability.getActivatingPlayer() != before) {
                ability.setActivatingPlayer(before);
            }
        });
        // The candidate last: an ability of its host may have trickled
        // another player down to it.
        candidate.setActivatingPlayer(activators.get(candidate));
        if (host != null && host.getCastSA() != castBefore) {
            host.setCastSA(castBefore);
        }
    }
}
