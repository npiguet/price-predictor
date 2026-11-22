package com.pricepredictor.connector.effects;

import forge.game.GameObject;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.TargetChoices;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The divided allocation of every link of a stack item's chain as it was
 * announced: each target's share of "N damage divided as you choose".
 *
 * <p>Forge's check for illegal targets ({@code TargetChoices.removeAll}) takes
 * a removed target's share off the item with it, before the resolution hook
 * runs, so the shares are read when the item goes on the stack, live and in a
 * fork, and kept beside the targets as cast ({@link ChainTargets#atCast}).
 * A retarget since moves a share to the new target, which the announcement
 * does not hold: at resolution start the item's current shares override it
 * ({@link #updatedBy}), the announced ones standing for the targets Forge's
 * check removed.
 */
final class DividedAllocation {

    /** No allocation read: every target's share is unknown. */
    static final DividedAllocation NONE = new DividedAllocation(List.of());

    /** Each link's shares by target, root first. */
    private final List<Map<GameObject, Integer>> perLink;

    private DividedAllocation(List<Map<GameObject, Integer>> perLink) {
        this.perLink = perLink;
    }

    /** The shares {@code item}'s chain holds now, link by link. */
    static DividedAllocation of(SpellAbility item) {
        List<Map<GameObject, Integer>> links = new ArrayList<>();
        for (SpellAbility link = item; link != null; link = link.getSubAbility()) {
            links.add(sharesOf(link.getTargets()));
        }
        return new DividedAllocation(List.copyOf(links));
    }

    /**
     * This allocation with each share {@code item}'s chain holds now laid
     * over it: a target that still carries a share takes its current one (a
     * retarget moved it there), and a target the check for illegal targets
     * removed, no longer on the item, keeps its announced one.
     */
    DividedAllocation updatedBy(SpellAbility item) {
        List<Map<GameObject, Integer>> now = of(item).perLink;
        List<Map<GameObject, Integer>> merged = new ArrayList<>();
        for (int index = 0; index < now.size(); index++) {
            Map<GameObject, Integer> shares = new HashMap<>(ofLink(index));
            shares.putAll(now.get(index));
            merged.add(shares);
        }
        return new DividedAllocation(List.copyOf(merged));
    }

    private static Map<GameObject, Integer> sharesOf(TargetChoices chosen) {
        Map<GameObject, Integer> shares = new HashMap<>();
        if (chosen != null) {
            for (GameObject target : chosen) {
                Integer share = chosen.getDividedValue(target);
                if (share != null) {
                    shares.put(target, share);
                }
            }
        }
        return shares;
    }

    /** The shares of the link at {@code index}, root first; none where unread. */
    Map<GameObject, Integer> ofLink(int index) {
        return index < perLink.size() ? perLink.get(index) : Map.of();
    }
}
