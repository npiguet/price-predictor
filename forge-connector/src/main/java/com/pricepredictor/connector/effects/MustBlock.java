package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.card.CardLists;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.keyword.KeywordInterface;
import forge.game.staticability.StaticAbilityMustBlock;

/**
 * Whether a requirement makes one creature block one attacker: a blockers
 * record's {@code must_block} (FR-037).
 *
 * <p>A <b>reimplementation</b>, because Forge has no per-(blocker, attacker)
 * predicate: it checks requirements only against a whole block, in
 * {@code CombatUtil.validateBlocks} and {@code mustBlockAnAttacker}, and only
 * to say whether a declared block obeys them. The sources are the ones those
 * methods read ({@code CombatUtil.java:640-700, 760-870, 907-958}):
 *
 * <ul>
 *   <li>the blocker's {@code getMustBlockCards} (provoke, "blocks CARDNAME
 *       this turn if able");</li>
 *   <li>the attacker's {@code MustBeBlockedBy <valid>} ("must be blocked by an
 *       Eldrazi if able"), while no creature of that kind blocks it;</li>
 *   <li>the attacker's {@code MustBeBlockedByAll:<valid>} ("All Walls able to
 *       block CARDNAME do so.");</li>
 *   <li>the attacker's "All creatures able to block CARDNAME do so." (a
 *       lure);</li>
 *   <li>a {@code MustBlock} static naming the blocker ("blocks each combat if
 *       able", {@code StaticAbilityMustBlock}), which any attacker it could
 *       block satisfies, and so names every attacker.</li>
 * </ul>
 *
 * <p>A requirement is reported whether or not the creature can obey it: a
 * tapped creature a lure names still has the requirement, and the record's
 * legal list says it cannot block. A requirement whose block costs something
 * is none, as rule 509.1c and Forge both have it. The attacker's "must be
 * blocked if able" variants are left out: they require something of the
 * block as a whole, not of any one creature.
 */
final class MustBlock {

    static final String LURE = "All creatures able to block CARDNAME do so.";
    private static final String BLOCKED_BY = "MustBeBlockedBy ";
    private static final String BLOCKED_BY_ALL = "MustBeBlockedByAll";

    private MustBlock() {
    }

    /**
     * Whether a requirement makes {@code blocker} block {@code attacker}.
     *
     * @param combat the combat being blocked, or null when there is none to
     *               read the blocks so far from
     */
    static boolean required(Card blocker, Card attacker, Combat combat) {
        if (CombatUtil.getBlockCost(blocker.getGame(), blocker, attacker) != null) {
            return false;
        }
        return blocker.getMustBlockCards().contains(attacker)
                || attacker.hasStartOfKeyword(LURE)
                || namedByAttacker(blocker, attacker, combat)
                || StaticAbilityMustBlock.blocksEachCombatIfAble(blocker);
    }

    /** {@code MustBeBlockedBy} and {@code MustBeBlockedByAll}, read as {@code CombatUtil} reads them. */
    private static boolean namedByAttacker(Card blocker, Card attacker, Combat combat) {
        for (KeywordInterface instance : attacker.getKeywords()) {
            String keyword = instance.getOriginal();
            if (keyword.startsWith(BLOCKED_BY)) {
                String valid = keyword.substring(BLOCKED_BY.length());
                if (blocker.isValid(valid, null, null, null) && (combat == null
                        || CardLists.getValidCardCount(
                                combat.getBlockers(attacker), valid, null, null, null) == 0)) {
                    return true;
                }
            } else if (keyword.startsWith(BLOCKED_BY_ALL)) {
                String valid = keyword.split(":")[1];
                if (blocker.isValid(valid, null, null, null)) {
                    return true;
                }
            }
        }
        return false;
    }
}
