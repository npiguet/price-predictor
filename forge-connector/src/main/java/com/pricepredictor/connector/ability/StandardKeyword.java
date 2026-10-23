package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.cost.Cost;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.keyword.KeywordWithTypeInterface;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Standard keyword ability — handles defined keywords and the undefined keyword fallback.
 * Covers KICKER dual cost, AFFINITY type description, and internal keyword reminder text.
 */
public record StandardKeyword(AbilityType type, String descriptionText) implements Ability {

    public static StandardKeyword of(KeywordInterface ki, Keyword kw) {
        boolean activatable = !ki.getAbilities().isEmpty();
        String title = baseTitle(ki, kw);
        switch (kw) {
            case KICKER -> {
                String secondCost = new KeywordSpec(ki.getOriginal()).rest(2);
                if (secondCost != null) {
                    title = title + " and/or " + new Cost(secondCost, false).toSimpleString();
                }
            }
            case AFFINITY -> {
                if (ki instanceof KeywordWithTypeInterface kti) {
                    title = "Affinity for " + kti.getTypeDescription();
                }
            }
            case CRAFT -> title = buildCraftTitle(ki.getOriginal());
            case PROTECTION -> title = protectionTitle(ki.getOriginal());
            default -> { }
        }

        String reduceCost = extractReduceCostDescription(kw, ki.getOriginal());
        if (reduceCost != null) {
            title = title + ". " + reduceCost;
        }

        AbilityType kwType = AbilityType.classifyKeyword(kw, activatable, !ki.getTriggers().isEmpty());
        return new StandardKeyword(kwType, AbilityDescription.applyCasing(title));
    }

    /**
     * The title a keyword starts from, before {@link #of}'s per-keyword
     * overrides: Forge's own title, or the script's original text where that
     * title is empty.
     *
     * <p>A defined keyword also falls back to the original when its title
     * carries surrounding whitespace, and an internal keyword reads as its
     * reminder text whenever it has one. An undefined keyword has neither
     * rule, and keeps any non-empty title.
     */
    private static String baseTitle(KeywordInterface ki, Keyword kw) {
        String title = ki.getTitle();
        if (kw == Keyword.UNDEFINED) {
            return title == null || title.isEmpty() ? ki.getOriginal() : title;
        }
        if (AbilityType.isInternalKeyword(kw)) {
            String reminder = ki.getReminderText();
            if (reminder != null && !reminder.isEmpty()) {
                return reminder;
            }
        }
        return title == null || title.isEmpty() || !title.trim().equals(title)
                ? ki.getOriginal() : title;
    }

    /**
     * "Protection from &lt;target&gt;", built from the raw script keyword.
     *
     * <p>{@code Protection:Creature} names its target directly, and
     * {@code Protection:Card.MultiColor:multicolored} gives it after a filter,
     * as the human-readable text after the second colon. The colon-free form
     * ({@code Protection from white}) is already oracle text, so the original
     * is the answer. The keyword's title is not: Forge splits the script line
     * on its first space, leaving "from white" as the type description, and
     * Protection.getTitle() then prepends "Protection from " to produce
     * "Protection from from white".
     */
    private static String protectionTitle(String original) {
        KeywordSpec spec = new KeywordSpec(original);
        String target = spec.size() > 2 ? spec.rest(2) : spec.rest(1);
        return target == null ? original : "Protection from " + target.toLowerCase();
    }

    /**
     * Build the terse oracle title for a Craft keyword: "Craft with <typeDesc> <manaCost>".
     *
     * ki.getOriginal() format: "Craft:<costPart>[:<typeDesc>[:<replaceDesc>]]"
     * where costPart contains mana tokens and ExileCtrlOrGrave<…> exile specs.
     *
     * Three sub-patterns for the type description:
     *   - Explicit: parts[2] present and non-empty → use as-is
     *   - Variable count: costPart contains "XMin" → "one or more"
     *   - Simple type: extract first word of TYPE from ExileCtrlOrGrave<N/TYPE.suffix>
     */
    private static String buildCraftTitle(String original) {
        KeywordSpec spec = new KeywordSpec(original);
        if (spec.size() < 2) return original;

        String costPart = spec.part(1);
        String explicitType = spec.part(2);

        // Mana: strip all ExileCtrlOrGrave<…> specs and XMin\d+ constraint tokens
        String mana = costPart
                .replaceAll("ExileCtrlOrGrave<[^>]*>", "")
                .replaceAll("XMin\\d+", "")
                .trim();
        String manaCostStr = new Cost(mana, false).toSimpleString();

        // Type description (three sub-patterns)
        String typeDesc;
        if (explicitType != null && !explicitType.isEmpty()) {
            // Explicit human-readable description in k[2] (Throne of Grim Captain, Eye of Ojer Taq)
            typeDesc = explicitType;
        } else if (costPart.contains("XMin")) {
            // Variable-count exile (Sunbird Standard): "one or more"
            typeDesc = "one or more";
        } else {
            // Simple type: the first word of TYPE in ExileCtrlOrGrave<N/TYPE…>
            Matcher exiled = CRAFT_EXILED_TYPE.matcher(costPart);
            typeDesc = exiled.find() ? exiled.group(1) : "artifact"; // fallback
        }

        return "Craft with " + typeDesc + " " + manaCostStr;
    }

    /**
     * The type a Craft cost exiles: after the first {@code ExileCtrlOrGrave<}
     * and the first {@code /} that follows it, everything up to a {@code .},
     * {@code +}, {@code >}, {@code /} or {@code <}.
     */
    private static final Pattern CRAFT_EXILED_TYPE =
            Pattern.compile("ExileCtrlOrGrave<[^/]*/([^.+>/<]*)");

    /**
     * Extract cost-reduction description from a keyword's original string, or null if absent.
     *
     * Pattern A (Equip): "Equip:cost:::ReduceCost$ SVar:full sentence"
     * Pattern B (Specialize): "Specialize:cost::full sentence.:ReduceCost$ SVar"
     *   → uses the full sentence from parts[3] directly
     *
     * (Adapt and Monstrosity are not members of the {@link Keyword} enum, so
     * no pattern covers them.)
     */
    private static String extractReduceCostDescription(Keyword kw, String original) {
        if (original == null) return null;

        // Pattern A: Equip with ReduceCost$ SVar:human-text format
        int idx = original.indexOf("ReduceCost$ ");
        if (idx >= 0) {
            String after = original.substring(idx + "ReduceCost$ ".length());
            // Format: "SVar:human text" — skip SVar name
            int colon = after.indexOf(':');
            if (colon >= 0) {
                return after.substring(colon + 1);
            }
        }

        // Pattern B: Specialize — full description is in parts[3], ReduceCost$ marker in parts[4]
        if (kw == Keyword.SPECIALIZE) {
            KeywordSpec spec = new KeywordSpec(original);
            if (spec.size() >= 5 && original.contains("ReduceCost$") && !spec.part(3).isEmpty()) {
                return spec.part(3);
            }
        }

        return null;
    }
}
