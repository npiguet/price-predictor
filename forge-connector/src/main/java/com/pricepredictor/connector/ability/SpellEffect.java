package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.ability.ApiType;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Spell effect ability. Factory walks the sub-ability chain recursively, building
 * a tree of SpellEffect nodes — one per SpellDescription fragment. Nodes without a
 * description are transparent: their children are promoted upward. A spell
 * prints the tree's sentences as one {@link SpellLine}; an activated or
 * triggered ability appends them to its own line.
 */
public record SpellEffect(String descriptionText, List<Ability> subAbilities) implements Ability {

    public SpellEffect {
        Objects.requireNonNull(subAbilities);
        subAbilities = List.copyOf(subAbilities);
    }

    @Override
    public AbilityType type() {
        return AbilityType.SPELL;
    }

    /**
     * Recursively walks a SpellAbility chain, returning 0 or 1 tree nodes.
     * Nodes with no SpellDescription are transparent: their children are promoted.
     * Also walks {@code RepeatSubAbility} (used by RepeatEach/Repeat) alongside the
     * regular {@code SubAbility} chain so those descriptions are not missed.
     */
    public static List<Ability> fromChain(SpellAbility sa) {
        return fromChain(sa, Ancestor.NONE);
    }

    /**
     * The nearest ancestor's cased description ({@code text} null when there
     * is none), against which sub-ability descriptions are compared.
     */
    private record Ancestor(String text) {
        static final Ancestor NONE = new Ancestor(null);
    }

    /**
     * Internal overload that threads {@code parent} (the nearest ancestor's cased description)
     * through the recursive walk. Used to suppress redundant sub-ability descriptions that merely
     * duplicate (or are contained in) text already emitted by the parent.
     */
    private static List<Ability> fromChain(SpellAbility sa, Ancestor parent) {
        if (sa == null) return List.of();

        String rawDesc = ownDescription(sa, parent);
        String stripped = rawDesc == null ? null : AbilityDescription.stripReminderText(rawDesc);
        // The node's own line, or null when it is transparent. A raw description
        // that strips to nothing still counts as one below: an all-reminder
        // SpellDescription covers its DB$ Effect children (Sarah's Wings NoDamage).
        String cased = stripped == null || stripped.isEmpty()
                ? null : AbilityDescription.applyCasing(stripped);
        // A described node's children compare against its own text; a
        // transparent node's against the caller's.
        Ancestor childParent = cased != null ? new Ancestor(cased) : parent;

        List<Ability> children = new ArrayList<>(fromChain(sa.getSubAbility(), childParent));
        if (rawDesc == null) {
            children.addAll(effectSVarChildren(sa, childParent));
        }
        children.addAll(repeatChildren(sa, childParent, cased));

        return cased != null ? List.of(new SpellEffect(cased, children)) : children;
    }

    /**
     * The raw description {@code sa} renders as its own node, or null when it
     * is transparent.
     *
     * <p>A declared description wins; failing one, a literal stack
     * description, unless it only repeats {@code parent}: Forge often
     * copies the parent's SpellDescription onto a sub-ability's
     * StackDescription for stack display (Bifurcate's DBChangeZone).
     */
    private static String ownDescription(SpellAbility sa, Ancestor parent) {
        String declared = SpellAbilityUtils.declaredDescription(sa);
        if (declared != null) return declared;
        String stack = SpellAbilityUtils.literalStackDescription(sa);
        if (stack == null) return null;
        String stackNorm = AbilityDescription.applyCasing(
                AbilityDescription.stripReminderText(stack));
        return stackNorm.equals(parent.text()) ? null : stack;
    }

    /**
     * The lines a transparent {@code DB$ Effect} creates through its
     * {@code Triggers$} and {@code ReplacementEffects$} SVars, promoted to the
     * caller like the rest of a transparent node's children. Empty for any
     * other API.
     */
    private static List<Ability> effectSVarChildren(SpellAbility sa, Ancestor parent) {
        return sa.getApi() == ApiType.Effect
                ? collectEffectDescriptions(sa, parent) : List.of();
    }

    /**
     * The nodes of {@code sa}'s {@code RepeatSubAbility} (RepeatEach, Repeat),
     * which the {@code SubAbility} walk does not reach.
     *
     * <p>A node repeating {@code ownLine}, the line {@code sa} itself renders,
     * is left out: some cards (Hoarder's Greed) copy the root SpellDescription
     * onto the RepeatSubAbility SVar for stack display, and keeping it would
     * emit the line twice.
     */
    private static List<Ability> repeatChildren(SpellAbility sa, Ancestor parent,
                                                String ownLine) {
        SpellAbility repeatSub = sa.getAdditionalAbility("RepeatSubAbility");
        if (repeatSub == null) return List.of();
        List<Ability> result = new ArrayList<>();
        for (Ability rc : fromChain(repeatSub, parent)) {
            if (ownLine != null && ownLine.equals(rc.descriptionText())) continue;
            result.add(rc);
        }
        return result;
    }

    /**
     * The SVar lists of a DB$ Effect that name the lines it creates, and the
     * parameter each SVar's description is read from.
     */
    private enum EffectSVars {
        /**
         * Trigger SVars that fire from a specific zone (TriggerZones$) are also
         * registered as top-level triggers and would duplicate the triggered ability.
         */
        TRIGGERS("Triggers", "TriggerDescription", "TriggerZones$"),
        REPLACEMENTS("ReplacementEffects", "Description", null);

        private final String listParam;
        private final String descriptionParam;
        private final String skipMarker;

        EffectSVars(String listParam, String descriptionParam, String skipMarker) {
            this.listParam = listParam;
            this.descriptionParam = descriptionParam;
            this.skipMarker = skipMarker;
        }

        /** The description {@code svarText} holds, or null when it is empty or marked to skip. */
        private String describe(String svarText) {
            if (svarText == null || svarText.isEmpty()) return null;
            if (skipMarker != null && svarText.contains(skipMarker)) return null;
            return AbilityDescription.normalize(
                    SpellAbilityUtils.svarParam(svarText, descriptionParam));
        }

        /** The descriptions of the SVars {@code sa} names in this list, in order. */
        List<String> descriptions(forge.game.card.Card host, SpellAbility sa) {
            String names = sa.getParam(listParam);
            List<String> result = new ArrayList<>();
            if (names == null || names.isEmpty()) return result;
            for (String name : names.split(",")) {
                String desc = describe(host.getSVar(name.trim()));
                if (desc != null) result.add(desc);
            }
            return result;
        }
    }

    /**
     * For a DB$ Effect SA, collect SpellEffect nodes from Triggers$ and ReplacementEffects$ SVars.
     * Extracts TriggerDescription from trigger SVars and Description from replacement SVars.
     * {@code parentDesc} is the nearest ancestor's cased description; descriptions that duplicate
     * or are contained in it are suppressed.
     */
    private static List<Ability> collectEffectDescriptions(SpellAbility sa, Ancestor parent) {
        List<Ability> result = new ArrayList<>();
        forge.game.card.Card host = sa.getHostCard();
        if (host == null) return result;
        for (EffectSVars source : EffectSVars.values()) {
            for (String desc : source.descriptions(host, sa)) {
                if (!isRedundantDescription(desc, parent.text())) {
                    result.add(new SpellEffect(desc, List.of()));
                }
            }
        }
        return result;
    }

    /**
     * Returns true when the child description should be suppressed because it is
     * already covered by the parent's oracle text, or is a Forge-internal display string.
     */
    private static boolean isRedundantDescription(String childDesc, String parentDesc) {
        // Forge-internal placeholder — "effectsource" is never valid oracle text.
        // Checked unconditionally (before parentDesc null guard) so it fires even when
        // ActivatedAbilityEntry calls fromChain() without threading parentDesc.
        if (childDesc.contains("effectsource")) return true;
        if (parentDesc == null) return false;
        // Exact match — child is identical to the parent's description
        if (parentDesc.equals(childDesc)) return true;
        // Substring — child text is already contained in the parent description
        if (parentDesc.contains(childDesc)) return true;
        // "this card/creature/permanent/spell" in replacement-effect Description$ SVars refers to
        // the same card as CARDNAME in SpellDescription; normalise before substring check.
        String childNorm = childDesc
                .replace("this card", "CARDNAME")
                .replace("this creature", "CARDNAME")
                .replace("this permanent", "CARDNAME")
                .replace("this spell", "CARDNAME");
        if (parentDesc.equals(childNorm) || parentDesc.contains(childNorm)) return true;
        return false;
    }
}
