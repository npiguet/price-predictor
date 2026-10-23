package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.CardTraitBase;
import forge.game.keyword.KeywordInterface;

import java.util.ArrayList;
import java.util.List;

/**
 * Class level ability. Stores cost + description as the full text, exposes
 * innerDescription for class dedup matching, and ordinal = level number.
 */
public record ClassLevelAbility(
        String descriptionText,
        String innerDescription,
        int ordinal
) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.LEVEL;
    }

    public static ClassLevelAbility of(KeywordInterface ki) {
        int level = Integer.parseInt(new KeywordSpec(ki.getOriginal()).part(1));

        var it = ki.getAbilities().iterator();
        String cost = it.hasNext() ? it.next().getCostDescription() : null;
        if (cost == null || cost.isEmpty()) {
            return null;
        }
        cost = cost.trim();
        if (cost.endsWith(":")) {
            cost = cost.substring(0, cost.length() - 1).trim();
        }

        String rawDesc = findFirstDescription(ki.getTriggers(), "TriggerDescription");
        if (rawDesc == null) {
            rawDesc = findFirstDescription(ki.getStaticAbilities(), "Description");
        }
        if (rawDesc == null) {
            rawDesc = findFirstDescription(ki.getReplacements(), "Description");
        }
        rawDesc = withoutUndescribedTraits(rawDesc);
        if (rawDesc == null) {
            return null;
        }

        String normalized = AbilityDescription.normalize(rawDesc);
        if (normalized == null) return null;
        String casedCost = AbilityDescription.applyCasing(cost);
        return new ClassLevelAbility(casedCost + ": " + normalized, normalized, level);
    }

    /**
     * The level's description without the placeholder Forge writes for an
     * added trait that has none.
     *
     * <p>Forge builds a level's static itself, in the {@code Class} branch of
     * {@code CardFactoryUtil.addStaticAbility}: it joins the description of
     * every trait the level adds with a line break, reading each one's
     * {@code TriggerDescription$} or {@code Description$} unchecked, so a
     * trait with no text of its own -- a {@code Static$ True} helper trigger,
     * such as the finality-counter trigger on Advanced Floral Invocations'
     * third level -- is appended as the literal {@code null}. Rendered as it
     * stands, that line reads {@code {4}{B/G}{B/G}: null} and pushes the
     * level's real text onto a line of its own.
     *
     * @return the remaining parts in their order and with their separators,
     *         or null where no part has text
     */
    static String withoutUndescribedTraits(String description) {
        if (description == null) {
            return null;
        }
        String separator = description.contains("\r\n") ? "\r\n" : "\n";
        List<String> described = new ArrayList<>();
        for (String part : description.split("\r?\n", -1)) {
            if (!part.isBlank() && !part.strip().equals("null")) {
                described.add(part);
            }
        }
        return described.isEmpty() ? null : String.join(separator, described);
    }

    private static <T extends CardTraitBase> String findFirstDescription(
            Iterable<T> traits, String param) {
        for (T trait : traits) {
            String d = trait.getParam(param);
            if (d != null && !d.isEmpty()) {
                return d;
            }
        }
        return null;
    }
}
