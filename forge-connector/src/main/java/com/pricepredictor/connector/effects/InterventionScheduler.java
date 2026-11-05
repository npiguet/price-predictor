package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.LinkedHashSet;
import java.util.Set;
import java.util.function.Predicate;

/**
 * Which ability nobody played is forced, at a phase boundary, and when a
 * permanent is granted a rare protection (FR-058).
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
    private final InterventionFork forks;
    /** Grants the rare protections, or null when the run takes none. */
    private final ProtectionFork protection;
    /** Lines already forced on their holder's own turn, so a redraw does not repeat one. */
    private final Set<String> interveneSeen = new LinkedHashSet<>();
    /** Lines already forced on another player's turn (FR-057), apart from the above. */
    private final Set<String> offTurnSeen = new LinkedHashSet<>();

    InterventionScheduler(Game game, InterventionFork forks) {
        this(game, forks, null);
    }

    /** @param protection grants the rare protections, or null for none */
    InterventionScheduler(Game game, InterventionFork forks, ProtectionFork protection) {
        this.game = game;
        this.forks = forks;
        this.protection = protection;
    }

    /** This boundary's interventions: the forced resolutions, then a protection fork. */
    void collect(PhaseType phase) {
        collect();
        if (protection != null) {
            protection.collect(phase);
        }
    }

    /**
     * Force what the players hold and nobody would play: first, for every
     * player but the one whose turn it is, a line whose branch turns on whose
     * turn it is (FR-057); then everything the player whose turn it is holds.
     *
     * <p>FR-057's lines go first because they are few and a boundary of the
     * other player's turn is the only place they show their other branch: the
     * budget the second pass spends greedily from the game's first boundary on
     * would otherwise never leave one for them. Each line is forced at most
     * once on its holder's turn and once on another's, so both branches can
     * be seen in one game.
     */
    void collect() {
        Player active = game.getPhaseHandler() == null
                ? null : game.getPhaseHandler().getPlayerTurn();
        if (active == null) {
            return;
        }
        for (Player other : game.getPlayers()) {
            if (other != active && !other.hasLost()
                    && !force(other, offTurnSeen, TurnConditions::branchesOnWhoseTurn)) {
                return;
            }
        }
        force(active, interveneSeen, candidate -> true);
    }

    /**
     * Force each line in {@code holder}'s hand {@code wanted} accepts and
     * {@code seen} does not hold yet, for {@code holder}.
     *
     * @return false once the budget refuses a fork, which ends the boundary
     */
    private boolean force(Player holder, Set<String> seen, Predicate<SpellAbility> wanted) {
        for (Card card : holder.getCardsIn(ZoneType.Hand)) {
            for (SpellAbility candidate : card.getSpellAbilities()) {
                if (!worthIntervening(candidate, seen) || !wanted.test(candidate)) {
                    continue;
                }
                if (!forks.mayIntervene(candidate.getId())) {
                    return false;
                }
                seen.add(AbilityLines.key(candidate));
                forks.intervene(candidate, candidate.getId(), holder);
            }
        }
        return true;
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
     * <p>Once per line per game in each set: a card redrawn or returned to
     * hand is the same line, and forcing it twice buys the second copy of one
     * observation.
     */
    private static boolean worthIntervening(SpellAbility candidate, Set<String> seen) {
        if (candidate.isLandAbility() || candidate.isManaAbility()) {
            return false;
        }
        return !seen.contains(AbilityLines.key(candidate));
    }
}
