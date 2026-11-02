package com.pricepredictor.connector.effects;

import forge.game.GameEntity;
import forge.game.ability.ApiType;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;

/**
 * The targeting clauses of a spell ability, and whether each can reach its
 * minimum: the one walk the verdict, the decision record and the random seat
 * share.
 *
 * <p>A clause is one link of a sub-ability chain. A charm's own chain is the
 * modes someone chose — the AI's last evaluation, or nobody's yet — not the
 * modes the rules offer, so each walk says which of the two it follows: the
 * verdict and the record follow the offered modes, the random seat's redraw
 * the chosen ones it is about to cast.
 *
 * <p>Read-only on the ability: counting a clause's candidates never chooses
 * one.
 */
public final class TargetingClauses {

    private TargetingClauses() {
    }

    /**
     * Whether every targeting clause of the ability can reach its minimum.
     *
     * <p>The walk stops at a charm and asks whether enough of its offered
     * modes could be chosen, ignoring whatever chain was left on it.
     */
    public static boolean reachMinimum(SpellAbility ability) {
        for (SpellAbility clause = ability; clause != null; clause = clause.getSubAbility()) {
            if (!canReachMinimum(clause)) {
                return false;
            }
            if (clause.getApi() == ApiType.Charm) {
                return enoughChoosableModes(clause);
            }
        }
        return true;
    }

    /**
     * What the rules let one clause target, not what anyone picked: never
     * {@code getTargets()}, which is the chosen set.
     */
    static List<GameEntity> candidatesOf(SpellAbility clause) {
        if (!clause.usesTargeting() || clause.getTargetRestrictions() == null) {
            return List.of();
        }
        return clause.getTargetRestrictions().getAllCandidates(clause);
    }

    /** Whether one clause is untargeted or has as many candidates as its minimum. */
    static boolean canReachMinimum(SpellAbility clause) {
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
            if (canReachMinimum(mode)) {
                choosable++;
            }
        }
        int required = CharmModes.minimumModes(charm);
        if (charm.hasParam("CanRepeatModes")) {
            required = Math.min(required, 1);
        }
        return choosable >= required;
    }

    /**
     * The clauses of a play about to be cast that target: the root chain, or
     * for a charm the root plus the chosen modes' own chains — the clones the
     * AI chained are rebuilt from those originals at the cast.
     */
    public static List<SpellAbility> chosen(SpellAbility ability) {
        List<SpellAbility> clauses = new ArrayList<>();
        if (ability.getApi() == ApiType.Charm) {
            if (ability.usesTargeting()) {
                clauses.add(ability);
            }
            if (ability.getChosenList() != null) {
                for (AbilitySub mode : ability.getChosenList()) {
                    addTargeting(mode, clauses);
                }
            }
            return clauses;
        }
        addTargeting(ability, clauses);
        return clauses;
    }

    private static void addTargeting(SpellAbility chain, List<SpellAbility> into) {
        for (SpellAbility clause = chain; clause != null; clause = clause.getSubAbility()) {
            if (clause.usesTargeting()) {
                into.add(clause);
            }
        }
    }
}
