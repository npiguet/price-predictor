package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.NonBlankString;
import forge.game.ability.ApiType;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Predicate;

/**
 * Spell effect ability. Factory walks the sub-ability chain recursively, building
 * a tree of SpellEffect nodes — one per SpellDescription fragment. Nodes without a
 * description are transparent: their children are promoted upward.
 */
public record SpellEffect(NonBlankString descriptionText, List<Ability> subAbilities) implements Ability {

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
        return fromChain(sa, Optional.empty());
    }

    /**
     * Internal overload that threads {@code parentDesc} (the nearest ancestor's cased description)
     * through the recursive walk. Used to suppress redundant sub-ability descriptions that merely
     * duplicate (or are contained in) text already emitted by the parent.
     */
    private static List<Ability> fromChain(SpellAbility sa, Optional<NonBlankString> parentDesc) {
        if (sa == null) return List.of();

        SaDescription resolved = SaDescription.resolve(sa, parentDesc);
        boolean hadRawDesc = resolved.hadRawDesc();
        boolean hasDesc = resolved.stripped().isPresent();

        // Determine the effective parentDesc to propagate to children:
        //   - if this SA has a description, its children inherit that description
        //   - if this SA is transparent, children inherit the caller's parentDesc
        Optional<NonBlankString> effectiveParentDesc = resolved.cased().or(() -> parentDesc);

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
                Optional<NonBlankString> parentCased = resolved.cased();
                for (Ability rc : repeatChildren) {
                    // Skip repeat-sub children whose description duplicates the parent SA.
                    // Some cards (e.g. Hoarder's Greed) copy the root SpellDescription onto
                    // the RepeatSubAbility SVar for stack-display; including it would double-emit.
                    if (parentCased.filter(p -> p.equals(rc.descriptionText())).isPresent()) continue;
                    children.add(rc);
                }
            }
        }

        if (hasDesc) {
            return List.of(new SpellEffect(resolved.cased().get(), children));
        }
        return children;
    }

    /**
     * For a DB$ Effect SA, collect SpellEffect nodes from Triggers$ and ReplacementEffects$ SVars.
     * Extracts TriggerDescription from trigger SVars and Description from replacement SVars.
     * {@code parentDesc} is the nearest ancestor's cased description; descriptions that duplicate
     * or are contained in it are suppressed.
     */
    private static List<Ability> collectEffectDescriptions(SpellAbility sa, Optional<NonBlankString> parentDesc) {
        forge.game.card.Card host = sa.getHostCard();
        if (host == null) return List.of();

        List<Ability> result = new ArrayList<>();
        result.addAll(collectSVarDescriptions(sa, "Triggers", "TriggerDescription",
                svar -> !isTopLevelTrigger(svar), parentDesc));
        result.addAll(collectSVarDescriptions(sa, "ReplacementEffects", "Description",
                svar -> true, parentDesc));
        return result;
    }

    /**
     * Returns true when a trigger SVar defines a zone-specific trigger ({@code TriggerZones$}).
     * Such triggers are also registered as top-level triggers on the card, so their descriptions
     * are already emitted by {@code RulesParser.collectTriggers()}.
     * Including them here via Effect SVars would produce duplicates.
     */
    private static boolean isTopLevelTrigger(String svarText) {
        return svarText.contains(ForgeParams.TRIGGER_ZONES_MARKER);
    }

    private static List<Ability> collectSVarDescriptions(
            SpellAbility sa, String listParam, String descKey,
            Predicate<String> svarFilter, Optional<NonBlankString> parentDesc) {
        String list = sa.getParam(listParam);
        if (list == null || list.isEmpty()) return List.of();
        forge.game.card.Card host = sa.getHostCard();
        if (host == null) return List.of();
        List<Ability> result = new ArrayList<>();
        for (String name : list.split(",")) {
            String svarText = host.getSVar(name.trim());
            if (svarText.isEmpty()) continue;
            if (!svarFilter.test(svarText)) continue;
            SpellAbilityUtils.extractParam(svarText, descKey)
                    .filter(desc -> !isRedundantDescription(desc, parentDesc))
                    .ifPresent(desc -> result.add(new SpellEffect(desc, List.of())));
        }
        return result;
    }

    /**
     * Returns true when the child description should be suppressed because it is
     * already covered by the parent's oracle text, or is a Forge-internal display string.
     */
    private static boolean isRedundantDescription(NonBlankString childDesc, Optional<NonBlankString> parentDesc) {
        // Forge-internal placeholder — never valid oracle text. Checked unconditionally
        // (before the parentDesc null guard) so it fires even when ActivatedAbilityEntry
        // calls fromChain() without threading parentDesc.
        if (childDesc.contains(ForgeParams.EFFECT_SOURCE)) return true;
        if (parentDesc.isEmpty()) return false;
        NonBlankString parent = parentDesc.get();
        // Exact match — child is identical to the parent's description.
        if (parent.equals(childDesc)) return true;
        // Substring — child text is already contained in the parent description.
        if (parent.contains(childDesc.value())) return true;
        // Forge replacement-effect Description$ SVars use "this card/creature/permanent/spell"
        // to refer to the host card, while SpellDescription uses the CARDNAME placeholder for
        // the same reference. Normalise before comparison to avoid false negatives.
        String childNorm = childDesc.value()
                .replace("this card", "CARDNAME")
                .replace("this creature", "CARDNAME")
                .replace("this permanent", "CARDNAME")
                .replace("this spell", "CARDNAME");
        return parent.contentEquals(childNorm) || parent.contains(childNorm);
    }

}
