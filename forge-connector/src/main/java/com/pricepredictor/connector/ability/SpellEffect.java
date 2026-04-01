package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.ability.ApiType;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Predicate;

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
     * Flattened description text and structured children produced from a single SA chain walk.
     * Callers that need both pieces of information (text to concatenate into a parent description,
     * and structured sub-abilities to keep as children) use this instead of calling
     * {@link #fromChain} and {@link SpellAbilityUtils#expandDiceOutcomes} separately.
     */
    public record Extraction(String flattenedText, List<Ability> children) {
        public static final Extraction EMPTY = new Extraction("", List.of());

        /**
         * Walk {@code sa} once, collecting both flattened description text and dice-outcome
         * children of the given {@code childType}.
         */
        public static Extraction of(SpellAbility sa, AbilityType childType, boolean chaseRepeatSub) {
            String text = SpellEffect.flattenChainText(sa);
            List<Ability> children = SpellAbilityUtils.expandDiceOutcomes(sa, childType, chaseRepeatSub);
            if (text.isEmpty() && children.isEmpty()) return EMPTY;
            return new Extraction(text, children);
        }
    }

    /**
     * Walk the SA chain from {@code sa} and return a single space-joined string of all
     * description fragments, without building an intermediate SpellEffect tree.
     * Returns an empty string when {@code sa} is null or produces no descriptions.
     */
    public static String flattenChainText(SpellAbility sa) {
        List<Ability> tree = fromChain(sa);
        return tree.isEmpty() ? "" : tree.get(0).flattenText();
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
    /**
     * Resolved description for one SA: the stripped text (after reminder-text removal)
     * and whether the SA had any raw description at all (even if it stripped to empty).
     */
    private record ResolvedDescription(String stripped, boolean hadRawDesc) {
        boolean hasDesc() { return stripped != null && !stripped.isEmpty(); }
        String cased() { return AbilityDescription.applyCasing(stripped); }
    }

    /**
     * Resolve the description for {@code sa}: tries SpellDescription, then TriggerDescription,
     * then StackDescription (with guards against Forge-internal references). The {@code parentDesc}
     * is used to reject StackDescriptions that merely duplicate the parent's oracle text.
     */
    private static ResolvedDescription resolveDescription(SpellAbility sa, String parentDesc) {
        String rawDesc = sa.getParam("SpellDescription");
        if (rawDesc == null || rawDesc.isEmpty()) {
            rawDesc = sa.getParam("TriggerDescription");
        }
        // StackDescription fallback — only for literal oracle text strings,
        // not the "SpellDescription" keyword reference, not "None" (no-display sentinel),
        // and not Forge template strings.
        if (rawDesc == null || rawDesc.isEmpty()) {
            String stack = sa.getParam("StackDescription");
            if (stack != null && !stack.isEmpty()
                    && !stack.equals("SpellDescription")
                    && !stack.equals("None")
                    && !stack.contains("REP ")
                    && !stack.contains("{p:")) {
                // Reject if the StackDescription merely duplicates the parent's oracle text
                // (Forge often copies the parent SpellDescription onto sub-ability StackDescription
                // for stack-display purposes — e.g. Bifurcate's DBChangeZone).
                String stackNorm = AbilityDescription.applyCasing(
                        AbilityDescription.stripReminderText(stack));
                if (!stackNorm.equals(parentDesc)) {
                    rawDesc = stack;
                }
            }
        }
        boolean hadRawDesc = rawDesc != null && !rawDesc.isEmpty();
        String stripped = hadRawDesc ? AbilityDescription.stripReminderText(rawDesc) : null;
        return new ResolvedDescription(stripped, hadRawDesc);
    }

    private static List<Ability> fromChain(SpellAbility sa, String parentDesc) {
        if (sa == null) return List.of();

        ResolvedDescription resolved = resolveDescription(sa, parentDesc);
        boolean hadRawDesc = resolved.hadRawDesc();
        boolean hasDesc = resolved.hasDesc();

        // Determine the effective parentDesc to propagate to children:
        //   - if this SA has a description, its children inherit that description
        //   - if this SA is transparent, children inherit the caller's parentDesc
        String effectiveParentDesc = hasDesc ? resolved.cased() : parentDesc;

        List<Ability> children = fromChain(sa.getSubAbility(), effectiveParentDesc);

        // For DB$ Effect with Triggers$ or ReplacementEffects$: collect descriptions
        // from those SVars as additional child nodes. The Effect SA itself is transparent
        // (no SpellDescription), so these get promoted up to the parent caller.
        // Guard: skip when the SA had any SpellDescription (even if all reminder text) —
        // that means its children are already covered by the parent's oracle description.
        if (!hasDesc && !hadRawDesc && sa.getApi() == ApiType.Effect) {
            List<Ability> effectChildren = collectEffectDescriptions(sa, effectiveParentDesc);
            if (!effectChildren.isEmpty()) {
                children = new ArrayList<>(children);
                children.addAll(effectChildren);
            }
        }

        SpellAbility repeatSub = sa.getAdditionalAbility("RepeatSubAbility");
        if (repeatSub != null) {
            List<Ability> repeatChildren = fromChain(repeatSub, effectiveParentDesc);
            if (!repeatChildren.isEmpty()) {
                children = new ArrayList<>(children);
                String parentCased = hasDesc ? resolved.cased() : null;
                for (Ability rc : repeatChildren) {
                    // Skip repeat-sub children whose description duplicates the parent SA.
                    // Some cards (e.g. Hoarder's Greed) copy the root SpellDescription onto
                    // the RepeatSubAbility SVar for stack-display; including it would double-emit.
                    if (parentCased != null && parentCased.equals(rc.descriptionText())) continue;
                    children.add(rc);
                }
            }
        }

        if (hasDesc) {
            return List.of(new SpellEffect(resolved.cased(), children));
        } else {
            return children;
        }
    }

    /**
     * For a DB$ Effect SA, collect SpellEffect nodes from Triggers$ and ReplacementEffects$ SVars.
     * Extracts TriggerDescription from trigger SVars and Description from replacement SVars.
     * {@code parentDesc} is the nearest ancestor's cased description; descriptions that duplicate
     * or are contained in it are suppressed.
     */
    private static List<Ability> collectEffectDescriptions(SpellAbility sa, String parentDesc) {
        forge.game.card.Card host = sa.getHostCard();
        if (host == null) return List.of();

        List<Ability> result = new ArrayList<>();
        // Triggers$ — skip trigger SVars that fire from a specific zone (TriggerZones$): these are
        // also registered as top-level triggers and would duplicate the triggered ability.
        result.addAll(collectSVarDescriptions(sa, "Triggers", "TriggerDescription",
                svar -> !svar.contains("TriggerZones$"), parentDesc));
        result.addAll(collectSVarDescriptions(sa, "ReplacementEffects", "Description",
                svar -> true, parentDesc));
        return result;
    }

    private static List<Ability> collectSVarDescriptions(
            SpellAbility sa, String listParam, String descKey,
            Predicate<String> svarFilter, String parentDesc) {
        String list = sa.getParam(listParam);
        if (list == null || list.isEmpty()) return List.of();
        forge.game.card.Card host = sa.getHostCard();
        if (host == null) return List.of();
        List<Ability> result = new ArrayList<>();
        for (String name : list.split(",")) {
            String svarText = host.getSVar(name.trim());
            if (svarText == null || svarText.isEmpty()) continue;
            if (!svarFilter.test(svarText)) continue;
            String desc = SpellAbilityUtils.extractParam(svarText, descKey);
            if (desc != null && !isRedundantDescription(desc, parentDesc)) {
                result.add(new SpellEffect(desc, List.of()));
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
