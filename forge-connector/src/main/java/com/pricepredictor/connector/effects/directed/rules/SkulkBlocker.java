package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.keyword.Keyword;
import forge.game.phase.PhaseType;

import java.util.List;
import java.util.Optional;

/**
 * 702.118b, a creature with skulk can't be blocked by creatures with greater
 * power, on scenario {@code 702.118b-1}: Furtive Homunculus (2/1, skulk)
 * attacks, and B's Centaur Courser (power 3) is not among its legal blockers.
 *
 * <p>The base board, read at a boundary: the declare-blockers step begins,
 * before blockers are declared, with an attacker that has skulk, and its
 * defending player controls an untapped creature with
 * greater power that skulk alone keeps from blocking it. The engine's own
 * question before the declaration writes the blockers record that judges it,
 * whatever the defending player then blocks with. It binds the attacker and
 * that creature.
 *
 * <p>Variant {@code lesser_power} ({@code replace_card}): the creature is
 * Llanowar Elves, whose power 1, with what the creature's boosts and counters
 * add to it, is less than the attacker's, so it may block. The precondition
 * binds only a creature whose Elves stays below the attacker's power and
 * survives what it carries (an attacker of power 1 only where the Elves
 * would carry a boost to power 0 or less), and only an attacker with none of
 * the other evasions that would keep a 1/1 Elf from blocking it, nor any
 * protection.
 */
public final class SkulkBlocker {

    public static final String RULE = "702.118b";
    static final String SCENARIO = "702.118b-1";
    static final String LESSER_POWER = "lesser_power";
    static final String ELVES = "Llanowar Elves";

    private SkulkBlocker() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, SkulkBlocker::bind, new DirectedRule.Variant(LESSER_POWER,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card(Combats.BLOCKER), ELVES))));
    }

    /**
     * The attacker's keywords besides skulk that keep a creature without
     * flying, reach or a colour from blocking it, or that may keep the Elves
     * out as well: a protection (from green, from creatures). Read in a
     * method: a static field of Forge's enum would load it with the registry.
     */
    static List<Keyword> otherEvasions() {
        return List.of(Keyword.FLYING, Keyword.FEAR, Keyword.INTIMIDATE, Keyword.SHADOW,
                Keyword.HORSEMANSHIP, Keyword.LANDWALK, Keyword.PROTECTION);
    }

    static Optional<Binding> bind(Game live) {
        Combat combat = live.getCombat();
        if (!Boards.at(live, PhaseType.COMBAT_DECLARE_BLOCKERS) || combat == null) {
            return Optional.empty();
        }
        return Combats.blockersKeptOut(combat, SkulkBlocker::keptOutByOnlySkulk);
    }

    /**
     * Whether the Elves' stand-in replaces {@code creature}, which the attacker
     * could not be blocked by now: the attacker has skulk and no other evasion
     * ({@link #otherEvasions}), the creature has more power and could block
     * but for skulk, and the Elves in its place have less power than the
     * attacker and survive the replacement.
     */
    private static boolean keptOutByOnlySkulk(Card attacker, Card creature) {
        int power = attacker.getNetPower();
        return hasOnlySkulk(attacker) && creature.getNetPower() > power && CombatUtil.canBlock(creature)
                && !CombatUtil.canBlock(attacker, creature)
                && ReplaceCardOperation.replacementPower(creature, ELVES) < power
                && Boards.replacementSurvives(creature, ELVES);
    }

    private static boolean hasOnlySkulk(Card attacker) {
        return attacker.hasKeyword(Keyword.SKULK) && otherEvasions().stream().noneMatch(attacker::hasKeyword);
    }
}
