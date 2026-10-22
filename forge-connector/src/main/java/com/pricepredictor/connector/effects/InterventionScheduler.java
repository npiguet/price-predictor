package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Which ability nobody played is forced, at a phase boundary.
 *
 * <p>Some abilities appear in no record however long collection runs: the AI
 * judges them bad, cannot afford them, or never draws them into a board where
 * they do anything. Observation cannot reach those, so the corpus forks the
 * game and makes one happen.
 *
 * <p>Driven from a phase boundary rather than from the playability hook, even
 * though that hook knows exactly which candidates were declined. The hook
 * fires inside the AI's own evaluation, and forking the game there means
 * copying a game that is mid-decision and re-entering the rules engine
 * underneath it. A phase boundary is the same board with nothing in flight.
 *
 * <p>Costs a game copy and a stack resolution each, which is why
 * {@code --interventions-per-game} defaults low and why nothing happens at all
 * when it is zero.
 */
final class InterventionScheduler {

    private final Game game;
    private final ForkCollector forks;
    /** Lines already offered an intervention, so a redraw does not repeat one. */
    private final Set<String> interveneSeen = new LinkedHashSet<>();

    InterventionScheduler(Game game, ForkCollector forks) {
        this.game = game;
        this.forks = forks;
    }

    /** Force what the player whose turn it is holds and nobody would play. */
    void collect() {
        Player actor = game.getPhaseHandler() == null
                ? null : game.getPhaseHandler().getPlayerTurn();
        if (actor == null) {
            return;
        }
        for (Card card : actor.getCardsIn(ZoneType.Hand)) {
            for (SpellAbility candidate : card.getSpellAbilities()) {
                if (!worthIntervening(candidate)) {
                    continue;
                }
                if (!forks.mayIntervene(candidate.getId())) {
                    return;
                }
                interveneSeen.add(ManaChannel.abilityText(candidate, ""));
                forks.intervene(candidate, candidate.getId());
            }
        }
    }

    /**
     * Is this line one observation cannot reach anyway?
     *
     * <p>Lands and mana abilities are excluded because every game plays them —
     * forking to force a Mountain onto the battlefield spends a game copy to
     * observe the most common event in the corpus. What is left is the spells
     * and activated abilities the AI passed over, which is the population the
     * intervention exists for.
     *
     * <p>Once per line per game: a card redrawn or returned to hand is the same
     * line, and forcing it twice buys the second copy of one observation.
     */
    private boolean worthIntervening(SpellAbility candidate) {
        if (candidate.isLandAbility() || candidate.isManaAbility()) {
            return false;
        }
        return !interveneSeen.contains(ManaChannel.abilityText(candidate, ""));
    }
}
