package com.pricepredictor.connector.effects;

import forge.game.GameEntity;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.List;

/**
 * A {@link ForcedPlayRun}'s targets, one list per targeting clause of its
 * line: the objects its roles name, empty where no role names the clause
 * (its targets are drawn), null where a role names an object the binding
 * cannot hold yet (the clause is left unchecked).
 */
final class NamedTargets {

    private final Player actor;
    private final List<List<GameEntity>> clauses;

    /**
     * @param actor   the player who makes the play
     * @param clauses per targeting clause, the named objects
     */
    NamedTargets(Player actor, List<List<GameEntity>> clauses) {
        this.actor = actor;
        this.clauses = clauses;
    }

    /**
     * X and targets for {@code ability}, which is not modal: X drawn, each
     * targeting clause's targets the named objects, checked with
     * {@code canTarget}, or drawn where no role names them.
     *
     * @param pay whether X is drawn against what the actor can pay; a trial
     *            announces the least X, reading no mana
     */
    boolean aim(SpellAbility ability, RandomChoices draws, boolean pay) {
        ability.setXManaCostPaid(null);
        draws.announceX(ability, pay ? actor : null, 0);
        List<SpellAbility> targeting = TargetingClauses.chosen(ability);
        if (namesBeyond(targeting.size())) {
            return false;
        }
        for (int i = 0; i < targeting.size(); i++) {
            if (!aimClause(targeting.get(i), namedFor(i), draws)) {
                return false;
            }
        }
        return true;
    }

    /** Whether a clause past the line's first {@code count} is named, or left unchecked. */
    private boolean namesBeyond(int count) {
        return clauses.size() > count
                && clauses.subList(count, clauses.size()).stream().anyMatch(NamedTargets::namesOrUnchecked);
    }

    private static boolean namesOrUnchecked(List<GameEntity> named) {
        return named == null || !named.isEmpty();
    }

    /** The objects named for the line's {@code index}th clause; empty past the named clauses. */
    private List<GameEntity> namedFor(int index) {
        return index < clauses.size() ? clauses.get(index) : List.of();
    }

    /** Aim one clause: unchecked, drawn, or the named objects. */
    private static boolean aimClause(SpellAbility clause, List<GameEntity> named, RandomChoices draws) {
        if (named == null) {
            return true;
        }
        if (named.isEmpty()) {
            return draws.chooseTargetsFor(clause);
        }
        return targetNamed(clause, named);
    }

    /** Set {@code named} as {@code clause}'s targets, each checked with {@code canTarget}. */
    private static boolean targetNamed(SpellAbility clause, List<GameEntity> named) {
        clause.clearTargets();
        for (GameEntity target : named) {
            if (!clause.canTarget(target)) {
                return false;
            }
            clause.getTargets().add(target);
        }
        return clause.isTargetNumberValid();
    }
}
