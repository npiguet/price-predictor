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
 * description are transparent: their children are promoted upward.
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
        return fromChain(sa, null);
    }

    /**
     * Internal overload that threads {@code parentDesc} (the nearest ancestor's cased description)
     * through the recursive walk. Used to suppress redundant sub-ability descriptions that merely
     * duplicate (or are contained in) text already emitted by the parent.
     */
    private static List<Ability> fromChain(SpellAbility sa, String parentDesc) {
        if (sa == null) return List.of();

        String rawDesc = ownDescription(sa, parentDesc);
        String stripped = rawDesc == null ? null : AbilityDescription.stripReminderText(rawDesc);
        // The node's own line, or null when it is transparent. A raw description
        // that strips to nothing still counts as one below: an all-reminder
        // SpellDescription covers its DB$ Effect children (Sarah's Wings NoDamage).
        String cased = stripped == null || stripped.isEmpty()
                ? null : AbilityDescription.applyCasing(stripped);
        // A described node's children compare against its own text; a
        // transparent node's against the caller's.
        String childParentDesc = cased != null ? cased : parentDesc;

        List<Ability> children = new ArrayList<>(fromChain(sa.getSubAbility(), childParentDesc));
        if (rawDesc == null) {
            children.addAll(effectSVarChildren(sa, childParentDesc));
        }
        children.addAll(repeatChildren(sa, childParentDesc, cased));

        return cased != null ? List.of(new SpellEffect(cased, children)) : children;
    }

    /**
     * The raw description {@code sa} renders as its own node, or null when it
     * is transparent.
     *
     * <p>A declared description wins; failing one, a literal stack
     * description, unless it only repeats {@code parentDesc}: Forge often
     * copies the parent's SpellDescription onto a sub-ability's
     * StackDescription for stack display (Bifurcate's DBChangeZone).
     */
    private static String ownDescription(SpellAbility sa, String parentDesc) {
        String declared = SpellAbilityUtils.declaredDescription(sa);
        if (declared != null) return declared;
        String stack = SpellAbilityUtils.literalStackDescription(sa);
        if (stack == null) return null;
        String stackNorm = AbilityDescription.applyCasing(
                AbilityDescription.stripReminderText(stack));
        return stackNorm.equals(parentDesc) ? null : stack;
    }

    /**
     * The lines a transparent {@code DB$ Effect} creates through its
     * {@code Triggers$} and {@code ReplacementEffects$} SVars, promoted to the
     * caller like the rest of a transparent node's children. Empty for any
     * other API.
     */
    private static List<Ability> effectSVarChildren(SpellAbility sa, String parentDesc) {
        return sa.getApi() == ApiType.Effect
                ? collectEffectDescriptions(sa, parentDesc) : List.of();
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
    private static List<Ability> repeatChildren(SpellAbility sa, String parentDesc,
                                                String ownLine) {
        SpellAbility repeatSub = sa.getAdditionalAbility("RepeatSubAbility");
        if (repeatSub == null) return List.of();
        List<Ability> result = new ArrayList<>();
        for (Ability rc : fromChain(repeatSub, parentDesc)) {
            if (ownLine != null && ownLine.equals(rc.descriptionText())) continue;
            result.add(rc);
        }
        return result;
    }

    /**
     * For a DB$ Effect SA, collect SpellEffect nodes from Triggers$ and ReplacementEffects$ SVars.
     * Extracts TriggerDescription from trigger SVars and Description from replacement SVars.
     * {@code parentDesc} is the nearest ancestor's cased description; descriptions that duplicate
     * or are contained in it are suppressed.
     */
    private static List<Ability> collectEffectDescriptions(SpellAbility sa, String parentDesc) {
        List<Ability> result = new ArrayList<>();
        forge.game.card.Card host = sa.getHostCard();
        if (host == null) return result;
        // Trigger SVars that fire from a specific zone (TriggerZones$) are also registered
        // as top-level triggers and would duplicate the triggered ability.
        addSVarDescriptions(host, sa.getParam("Triggers"), "TriggerDescription",
                "TriggerZones$", parentDesc, result);
        addSVarDescriptions(host, sa.getParam("ReplacementEffects"), "Description",
                null, parentDesc, result);
        return result;
    }

    /**
     * One SpellEffect per SVar named in {@code svarNames} (comma-separated), from its
     * {@code descParam} value, skipping SVars containing {@code skipMarker} and
     * descriptions redundant with {@code parentDesc}.
     */
    private static void addSVarDescriptions(forge.game.card.Card host, String svarNames,
                                            String descParam, String skipMarker,
                                            String parentDesc, List<Ability> result) {
        if (svarNames == null || svarNames.isEmpty()) return;
        for (String name : svarNames.split(",")) {
            String svarText = host.getSVar(name.trim());
            if (svarText == null || svarText.isEmpty()) continue;
            if (skipMarker != null && svarText.contains(skipMarker)) continue;
            String desc = extractParam(svarText, descParam);
            if (desc != null && !isRedundantDescription(desc, parentDesc)) {
                result.add(new SpellEffect(desc, List.of()));
            }
        }
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

    /**
     * The normalized value of {@code paramName} in a "Key$ value | ..." SVar
     * text, or null if absent or empty.
     */
    private static String extractParam(String svarText, String paramName) {
        return AbilityDescription.normalize(SpellAbilityUtils.svarParam(svarText, paramName));
    }

}
