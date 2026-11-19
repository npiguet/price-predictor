package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.game.spellability.SpellAbility;

import java.util.LinkedHashMap;
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
     * Each target's divided share on one link, summed per ref: the share
     * announced when the item went on the stack ({@link DividedAllocation}),
     * which a target Forge's check removed keeps in the reading as cast.
     */
    private static void addShares(ChainTargets.Link link, Map<String, Integer> into) {
        for (ChainTargets.Target target : link.targets()) {
            if (target.share() != null) {
                into.merge(target.ref(), target.share(), Integer::sum);
            }
        }
    }
}
