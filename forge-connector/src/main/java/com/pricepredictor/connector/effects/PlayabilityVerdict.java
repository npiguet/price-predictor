package com.pricepredictor.connector.effects;

import forge.ai.ComputerUtilMana;
import forge.game.ability.AbilityUtils;
import forge.game.ability.ApiType;
import forge.game.cost.Cost;
import forge.game.cost.CostPayment;
import forge.game.player.Player;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;

/**
 * The rules' three answers about a candidate ability, and the one place they
 * are computed for both seats.
 *
 * <p>Every answer is the game's, never the AI's. Two engine calls look like
 * rules checks and are not: {@code ComputerUtilCost.canPayCost} refuses costs
 * the AI dislikes (a planeswalker's last loyalty, by a draw from the shared
 * random stream; a Ward cost; a Casualty sacrifice) and adds the mana its
 * Nether Void heuristics want, and a candidate's chosen targets are whatever
 * the AI's evaluation left on the ability object, which is nothing when it
 * stopped early and an earlier evaluation's when it did not run at all. So
 * affordability is the mana and additional-cost checks those two delegate to,
 * and target legality counts candidates rather than reading choices.
 *
 * <p>Read-only on the game, so observing a candidate cannot change the game
 * that evaluates it.
 */
public record PlayabilityVerdict(boolean canPlay, boolean affordable, boolean hasLegalTarget) {

    /** The verdict for a candidate nobody can be asked about. */
    static final PlayabilityVerdict NONE = new PlayabilityVerdict(false, false, false);

    /**
     * The verdict for {@code candidate} as its activating player stands now.
     *
     * <p>A candidate without an activating player has no one to pay for it or
     * to target with it, and gets {@link #NONE}.
     */
    public static PlayabilityVerdict of(SpellAbility candidate) {
        Player activator = candidate == null ? null : candidate.getActivatingPlayer();
        if (activator == null) {
            return NONE;
        }
        return new PlayabilityVerdict(
                candidate.canPlay(), affordable(candidate, activator), hasLegalTargets(candidate));
    }

    /** Whether the player has the mana and the additional costs, ignoring every AI preference. */
    static boolean affordable(SpellAbility candidate, Player payer) {
        Cost cost = candidate.getPayCosts();
        if (cost == null) {
            return true;
        }
        return ComputerUtilMana.canPayManaCost(cost, candidate, payer, 0, false)
                && CostPayment.canPayAdditionalCosts(cost, candidate, false, payer);
    }

    /**
     * Whether every targeting clause of the ability can reach its minimum.
     *
     * <p>A charm's sub-ability chain is the modes someone chose — the AI's
     * last evaluation, or nobody's yet — not the modes the rules offer, so the
     * walk stops at a charm and asks whether enough of its offered modes could
     * be chosen.
     */
    static boolean hasLegalTargets(SpellAbility ability) {
        for (SpellAbility clause = ability; clause != null; clause = clause.getSubAbility()) {
            if (!clauseCanReachMinimum(clause)) {
                return false;
            }
            if (clause.getApi() == ApiType.Charm) {
                return enoughChoosableModes(clause);
            }
        }
        return true;
    }

    private static boolean clauseCanReachMinimum(SpellAbility clause) {
        return !clause.usesTargeting()
                || clause.getTargetRestrictions().getNumCandidates(clause) >= clause.getMinTargets();
    }

    /**
     * Whether the charm's minimum number of modes can be met by modes whose
     * own targets are reachable (CR 700.2a: a mode with no legal target can't
     * be chosen).
     *
     * <p>Counted here rather than through {@code CharmEffect.makePossibleOptions},
     * which renumbers each offered mode's {@code CharmOrder} as it filters and
     * so would write to the game.
     */
    private static boolean enoughChoosableModes(SpellAbility charm) {
        int choosable = 0;
        for (AbilitySub mode : charm.getAdditionalAbilityList("Choices")) {
            if (clauseCanReachMinimum(mode)) {
                choosable++;
            }
        }
        String minimum = charm.getParamOrDefault("MinCharmNum", charm.getParamOrDefault("CharmNum", "1"));
        int required = AbilityUtils.calculateAmount(charm.getHostCard(), minimum, charm);
        if (charm.hasParam("CanRepeatModes")) {
            required = Math.min(required, 1);
        }
        return choosable >= required;
    }
}
