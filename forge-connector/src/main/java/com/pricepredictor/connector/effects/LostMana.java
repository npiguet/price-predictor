package com.pricepredictor.connector.effects;

import forge.card.MagicColor;
import forge.game.mana.Mana;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.staticability.StaticAbilityUnspentMana;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * A {@code mana_lost}'s {@code mana_by_color}: unspent mana a player's pool
 * lost, or would have lost, counted one per unit under the letter {@code
 * mana_produced} spells its colour with ({@code W U B R G}, {@code C} for
 * colourless whatever Forge keys it under), in that order, a colour with
 * none left out. A unit's restrictions and its snow source change nothing:
 * it counts under its colour.
 */
final class LostMana {

    /** The letters in the order they are written. */
    private static final String ORDER = "WUBRGC";

    private LostMana() {
    }

    /**
     * Each {@link Mana} of {@code units} under its colour's letter; anything
     * else counts under none. A drain's note hands over the mana it emptied.
     */
    static Map<String, Integer> byColor(Iterable<?> units) {
        Map<String, Integer> counts = new TreeMap<>(Comparator.comparingInt(ORDER::indexOf));
        for (Object unit : units) {
            if (unit instanceof Mana mana) {
                counts.merge(MagicColor.toShortString(mana.getColor()), 1, Integer::sum);
            }
        }
        return counts;
    }

    /**
     * The mana {@code player}'s pool would lose now if no replacement applied,
     * by {@code ManaPool.clearPool}'s own rule: a drain (an effect resolving
     * from the stack, {@code DrainMana}) empties the whole pool; a step's or
     * phase's end (500.4, which Forge makes with the stack empty and nothing
     * resolving) keeps the colours a static keeps (Upwelling, Omnath, Locus
     * of Mana) and, before the cleanup step, mana that stays until the turn's
     * end or the combat's.
     */
    static Map<String, Integer> wouldBeLost(Player player) {
        boolean drained = player.getGame().getStack().isResolving();
        Collection<Byte> kept = drained ? Set.of() : StaticAbilityUnspentMana.getManaToKeep(player);
        PhaseHandler phases = player.getGame().getPhaseHandler();
        List<Mana> lost = new ArrayList<>();
        for (Mana mana : player.getManaPool()) {
            boolean stays = kept.contains(mana.getColor()) || (!drained && staysAsTheStepEnds(mana, phases));
            if (!stays) {
                lost.add(mana);
            }
        }
        return byColor(lost);
    }

    /** Mana that outlasts a step's end: until the turn's cleanup, or the combat's end. */
    private static boolean staysAsTheStepEnds(Mana mana, PhaseHandler phases) {
        if (phases.is(PhaseType.CLEANUP)) {
            return false;
        }
        return mana.isPersistentMana() || (mana.isCombatMana() && !phases.is(PhaseType.COMBAT_END));
    }
}
