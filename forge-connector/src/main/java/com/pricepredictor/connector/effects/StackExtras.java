package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A stack entity's {@code stack_extras}: what its item announced, read over
 * the item's whole chain ({@link ChainTargets}), a sub-ability's targets and
 * a targeted spell or ability included.
 *
 * <p>Divided damage and "up to N" counts are choices the caster made that no
 * other field records: a Fireball for 3 split two ways and one aimed at a
 * single creature have the same cost and the same text.
 *
 * <p>An effect half reads its acting item's extras from the targets as cast
 * (608.2b): Forge's check for illegal targets takes a target off the item
 * with its divided share before the half's board is read, and extras read
 * off the thinned item would show the fizzle the half is to predict.
 */
final class StackExtras {

    private StackExtras() {
    }

    /**
     * The extras of {@code item}, or {@code null} JSON where the entity holds
     * no item on the stack.
     *
     * @param reading the item's targets as the record shows them (as cast,
     *                on an effect half), or null for those it holds now
     */
    static String json(SpellAbility item, ChainTargets reading) {
        if (item == null || item.getTargets() == null) {
            return "null";
        }
        ChainTargets targets = reading != null ? reading : ChainTargets.now(item);
        Json.ObjectWriter upTo = Json.object();
        if (usesTargeting(item)) {
            upTo.value("chosen", targets.count());
        }
        return Json.object()
                .raw("targets", Json.stringArray(targets.all()))
                .raw("per_target_amounts", amountsJson(targets))
                .raw("up_to_counts", upTo.render())
                .render();
    }

    /** Whether a link of {@code item}'s chain chooses targets. */
    private static boolean usesTargeting(SpellAbility item) {
        for (SpellAbility link = item; link != null; link = link.getSubAbility()) {
            if (link.getTargetRestrictions() != null) {
                return true;
            }
        }
        return false;
    }

    private static String amountsJson(ChainTargets targets) {
        Map<String, Integer> amounts = new LinkedHashMap<>();
        for (ChainTargets.Link link : targets.links()) {
            addShares(link, amounts);
        }
        Json.ObjectWriter json = Json.object();
        amounts.forEach(json::value);
        return json.render();
    }

    /**
     * Each target's divided share on one link, summed per ref. A target
     * Forge's check removed lost its share with it; a lone such target on a
     * link takes what the others leave of the link's announced total, and
     * two or more cannot be told apart, so none of them is written.
     */
    private static void addShares(ChainTargets.Link link, Map<String, Integer> into) {
        SpellAbility ability = link.ability();
        List<ChainTargets.Target> unshared = new ArrayList<>();
        int shared = 0;
        for (ChainTargets.Target target : link.targets()) {
            Integer share = ability.getDividedValue(target.object());
            if (share == null) {
                unshared.add(target);
            } else {
                into.merge(target.ref(), share, Integer::sum);
                shared += share;
            }
        }
        Integer left = leftOf(ability, shared);
        if (unshared.size() == 1 && left != null) {
            into.merge(unshared.get(0).ref(), left, Integer::sum);
        }
    }

    /** What a divided link's announced total leaves past {@code shared}; null when nothing. */
    private static Integer leftOf(SpellAbility ability, int shared) {
        Integer total = ability.isDividedAsYouChoose() ? ability.getDividedValue() : null;
        return total == null || total <= shared ? null : total - shared;
    }
}
