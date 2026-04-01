package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.AbilityDescription;
import forge.game.spellability.SpellAbility;

/**
 * Resolved description for a single SpellAbility.
 *
 * <p>Encapsulates the three-source resolution cascade that determines the oracle
 * text for an SA:
 * <ol>
 *   <li><b>SpellDescription</b> — the primary oracle text param; used when present.</li>
 *   <li><b>TriggerDescription</b> — fallback for SAs that carry trigger text rather
 *       than spell text (e.g. immediate-trigger execute SAs).</li>
 *   <li><b>StackDescription</b> — last resort; accepted only when it is literal oracle
 *       text and not one of Forge's rendering sentinels ({@link ForgeParams#NO_DISPLAY},
 *       {@link ForgeParams#STACK_DESC_REF}, {@link ForgeParams#REP_PREFIX},
 *       {@link ForgeParams#PLAYER_TOKEN}).</li>
 * </ol>
 *
 * <p>Having a single resolution point eliminates the dual implementations that previously
 * existed in {@code SpellEffect.resolveDescription()} and
 * {@code TriggeredAbilityEntry.isTransparentSA()}, which encoded the same logic
 * independently and had begun to diverge.
 *
 * @param stripped     the description text after reminder-text stripping; may be null
 *                     or empty if no usable source was found
 * @param hadRawDesc   true when at least one source contained non-empty text, even if
 *                     that text stripped entirely to nothing (e.g. pure reminder text)
 */
public record SaDescription(String stripped, boolean hadRawDesc) {

    /** True when this SA carries no usable description text. */
    public boolean isEmpty() {
        return stripped == null || stripped.isEmpty();
    }

    /** The description text with casing normalised. Returns null when {@link #isEmpty()}. */
    public String cased() {
        return AbilityDescription.applyCasing(stripped);
    }

    /**
     * Resolve the description for {@code sa} without parent-deduplication.
     * Equivalent to {@code resolve(sa, null)}.
     * Use this overload when checking whether an SA is "transparent" (carries no
     * description of its own), rather than when building a full description tree.
     */
    public static SaDescription resolve(SpellAbility sa) {
        return resolve(sa, null);
    }

    /**
     * Resolve the description for {@code sa}, optionally suppressing StackDescriptions
     * that duplicate {@code parentDesc}.
     *
     * <p>Forge often copies a parent SA's SpellDescription onto a sub-SA's
     * StackDescription for stack-display purposes. Without the parent-dedup guard,
     * that copied text would be emitted twice (once from the parent's SpellDescription,
     * once from the child's StackDescription). The guard compares the normalised forms
     * to catch this case.
     *
     * @param parentDesc the nearest ancestor's cased description, or null if absent
     */
    public static SaDescription resolve(SpellAbility sa, String parentDesc) {
        String rawDesc = sa.getParam("SpellDescription");
        if (rawDesc == null || rawDesc.isEmpty()) {
            rawDesc = sa.getParam("TriggerDescription");
        }
        // StackDescription fallback — accepted only for literal oracle text strings.
        // Forge uses StackDescription for several non-oracle purposes that must be rejected:
        //   "SpellDescription"  → a level-of-indirection marker, not literal text
        //   "None"              → sentinel meaning "don't show on stack"
        //   "REP ..."           → replacement-effect template prefix, not oracle text
        //   "{p:...}"           → player-reference template token, not oracle text
        if (rawDesc == null || rawDesc.isEmpty()) {
            String stack = sa.getParam("StackDescription");
            if (stack != null && !stack.isEmpty()
                    && !stack.equals(ForgeParams.STACK_DESC_REF)
                    && !stack.equals(ForgeParams.NO_DISPLAY)
                    && !stack.contains(ForgeParams.REP_PREFIX)
                    && !stack.contains(ForgeParams.PLAYER_TOKEN)) {
                // Reject if the StackDescription merely duplicates the parent's oracle text.
                // Forge often copies the parent SpellDescription onto sub-ability
                // StackDescription for stack-display — e.g. Bifurcate's DBChangeZone.
                String stackNorm = AbilityDescription.applyCasing(
                        AbilityDescription.stripReminderText(stack));
                if (!stackNorm.equals(parentDesc)) {
                    rawDesc = stack;
                }
            }
        }
        boolean hadRawDesc = rawDesc != null && !rawDesc.isEmpty();
        String stripped = hadRawDesc ? AbilityDescription.stripReminderText(rawDesc) : null;
        return new SaDescription(stripped, hadRawDesc);
    }
}
