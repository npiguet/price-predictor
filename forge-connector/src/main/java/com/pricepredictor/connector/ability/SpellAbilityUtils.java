package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.ability.ApiType;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * Utility methods for walking SpellAbility chains.
 */
public final class SpellAbilityUtils {

    private SpellAbilityUtils() {}

    /**
     * Returns a sequential Stream over the sub-ability chain starting from {@code sa},
     * including {@code sa} itself, stopping at the first null link.
     */
    private static Stream<SpellAbility> walkChain(SpellAbility sa) {
        return Stream.iterate(sa, Objects::nonNull, SpellAbility::getSubAbility);
    }

    /**
     * Walk the SubAbility (and optionally RepeatSubAbility) chain to find the SA that carries
     * {@code ResultSubAbilities}, and return its {@code SpellDescription}.
     * Returns null if no such SA is found.
     */
    public static String findDiceRollDescription(SpellAbility sa) {
        for (SpellAbility cur = sa; cur != null; cur = cur.getSubAbility()) {
            if (cur.getParam("ResultSubAbilities") != null) {
                return cur.getParam("SpellDescription");
            }
            SpellAbility repeatSub = cur.getAdditionalAbility("RepeatSubAbility");
            if (repeatSub != null) {
                String desc = findDiceRollDescription(repeatSub);
                if (desc != null) return desc;
            }
        }
        return null;
    }

    /**
     * Expand {@code ResultSubAbilities} dice-outcome entries into Ability nodes of {@code outputType}.
     * Walks the sub-ability chain starting from {@code sa}; if {@code chaseRepeatSub} is true,
     * also checks the {@code RepeatSubAbility} lateral branch.
     * Returns an empty list when no {@code ResultSubAbilities} param is found.
     */
    public static List<Ability> expandDiceOutcomes(
            SpellAbility sa, AbilityType outputType, boolean chaseRepeatSub) {
        for (SpellAbility cur = sa; cur != null; cur = cur.getSubAbility()) {
            String resultSubAbilities = cur.getParam("ResultSubAbilities");
            if (resultSubAbilities != null && !resultSubAbilities.isEmpty()) {
                List<Ability> result = new ArrayList<>();
                for (String entry : resultSubAbilities.split(",")) {
                    String[] kv = entry.trim().split(":", 2);
                    if (kv.length < 2) continue;
                    String range = kv[0].trim();
                    SpellAbility sub = cur.getAdditionalAbility(range);
                    if (sub == null) continue;
                    String rawDesc = sub.getParam("SpellDescription");
                    if (rawDesc == null || rawDesc.isEmpty()) continue;
                    String normalized = AbilityDescription.normalize(AbilityDescription.replaceVert(rawDesc));
                    if (normalized != null) {
                        result.add(new TextAbility(outputType, AbilityDescription.applyCasing(normalized)));
                    }
                }
                return result;
            }
            if (chaseRepeatSub) {
                SpellAbility repeatSub = cur.getAdditionalAbility("RepeatSubAbility");
                if (repeatSub != null) {
                    List<Ability> fromRepeat = expandDiceOutcomes(repeatSub, outputType, true);
                    if (!fromRepeat.isEmpty()) return fromRepeat;
                }
            }
        }
        return List.of();
    }

    static String findParamInChain(SpellAbility sa, String param) {
        return walkChain(sa)
                .map(cur -> cur.getParam(param))
                .filter(v -> v != null && !v.isEmpty())
                .findFirst()
                .orElse(null);
    }

    /**
     * Result of resolving the ABILITY placeholder in a trigger/chapter description.
     * {@code expandedDescription} has the placeholder replaced; {@code options} are
     * charm or dice-outcome sub-abilities, if any.
     */
    public record AbilityPlaceholderResult(String expandedDescription, List<Ability> options) {}

    /**
     * Resolve the ABILITY placeholder in {@code description} using {@code executeSA}
     * and optional {@code fallbackText}. Returns null when the placeholder is absent or
     * no resolution strategy applies.
     *
     * <p>Resolution cascade:
     * <ol>
     *   <li>Charm: if executeSA (or an ImmediateTrigger nested within it) is a Charm, replace
     *       ABILITY with the synthesized charm header and collect OPTION sub-abilities.</li>
     *   <li>Dice-roll: if executeSA has ResultSubAbilities, replace ABILITY with the
     *       dice-roll SpellDescription and return the dice outcomes as OPTION sub-abilities.</li>
     *   <li>Fallback: if fallbackText is non-null and non-empty, substitute it directly.</li>
     * </ol>
     */
    public static AbilityPlaceholderResult resolveAbilityPlaceholder(
            String description, SpellAbility executeSA, String fallbackText) {
        if (description == null || !description.contains("ABILITY")) return null;

        if (executeSA != null) {
            // Case 1: Charm resolution
            if (executeSA.getApi() == ApiType.Charm) {
                return buildCharmResult(description, executeSA);
            }
            for (SpellAbility cur = executeSA; cur != null; cur = cur.getSubAbility()) {
                if (cur.getApi() == ApiType.ImmediateTrigger) {
                    SpellAbility nestedCharm = cur.getAdditionalAbility("Execute");
                    if (nestedCharm != null && nestedCharm.getApi() == ApiType.Charm) {
                        return buildCharmResult(description, nestedCharm);
                    }
                }
            }
            // Case 2: Dice-roll resolution
            List<Ability> diceOptions = expandDiceOutcomes(executeSA, AbilityType.OPTION, true);
            if (!diceOptions.isEmpty()) {
                String diceDesc = findDiceRollDescription(executeSA);
                if (diceDesc != null) {
                    String expanded = AbilityDescription.normalize(
                            description.replace("ABILITY", AbilityDescription.replaceVert(diceDesc)));
                    if (expanded != null) return new AbilityPlaceholderResult(expanded, diceOptions);
                }
            }
        }

        // Case 3: Direct text substitution
        if (fallbackText != null && !fallbackText.isEmpty()) {
            String expanded = AbilityDescription.normalize(description.replace("ABILITY", fallbackText));
            return new AbilityPlaceholderResult(expanded, List.of());
        }

        return null;
    }

    private static AbilityPlaceholderResult buildCharmResult(String description, SpellAbility charmSA) {
        String header = CharmAbility.synthesizeCharmHeader(charmSA);
        if (header == null) header = "choose one";
        String expanded = AbilityDescription.normalize(description.replace("ABILITY", header));
        List<Ability> options = CharmAbility.optionsFrom(charmSA);
        return new AbilityPlaceholderResult(expanded, options);
    }

    /**
     * Parse a "Key$ value | ..." SVar text and extract the value for the given key.
     * Strips reminder text, normalizes casing; returns null if absent or empty.
     */
    public static String extractParam(String svarText, String paramName) {
        String key = paramName + "$ ";
        int idx = svarText.indexOf(key);
        if (idx < 0) return null;
        int start = idx + key.length();
        int end = svarText.indexOf(" |", start);
        String raw = end >= 0 ? svarText.substring(start, end).trim() : svarText.substring(start).trim();
        if (raw.isEmpty()) return null;
        String stripped = AbilityDescription.stripReminderText(raw);
        String normalized = AbilityDescription.normalize(stripped);
        if (normalized == null || normalized.isEmpty()) return null;
        return AbilityDescription.applyCasing(normalized);
    }
}
