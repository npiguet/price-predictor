package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.cost.Cost;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.keyword.KeywordWithTypeInterface;

/**
 * Standard keyword ability — handles defined keywords and the undefined keyword fallback.
 * Covers KICKER dual cost, AFFINITY type description, and internal keyword reminder text.
 */
public record StandardKeyword(AbilityType type, String descriptionText) implements Ability {

    public static StandardKeyword of(KeywordInterface ki, Keyword kw) {
        boolean activatable = !ki.getAbilities().isEmpty();
        String title = ki.getTitle();

        if (kw == Keyword.UNDEFINED) {
            if (title == null || title.isEmpty()) {
                title = ki.getOriginal();
            }
        } else {
            if (title == null || title.isEmpty() || !title.trim().equals(title)) {
                title = ki.getOriginal();
            }
            if (AbilityType.isInternalKeyword(kw)) {
                String reminder = ki.getReminderText();
                if (reminder != null && !reminder.isEmpty()) {
                    title = reminder;
                }
            }
            if (kw == Keyword.KICKER) {
                String[] kickerParts = ki.getOriginal().split(":", 3);
                if (kickerParts.length >= 3) {
                    Cost cost2 = new Cost(kickerParts[2], false);
                    title = title + " and/or " + cost2.toSimpleString();
                }
            }
            if (kw == Keyword.AFFINITY && ki instanceof KeywordWithTypeInterface kti) {
                title = "Affinity for " + kti.getTypeDescription();
            }
            if (kw == Keyword.CRAFT) {
                title = buildCraftTitle(ki.getOriginal());
            }
            if (kw == Keyword.PROTECTION) {
                // getOriginal() returns the raw script keyword, e.g.:
                //   "Protection:Creature"         (simple type)
                //   "Protection:Card.MultiColor:multicolored"  (filter with description)
                // Build "Protection from <target>" from it.
                // Color-based protection ("K:Protection from white") already has the
                // correct format in getOriginal(), so no special-casing needed there.
                String original = ki.getOriginal();
                int firstColon = original.indexOf(':');
                if (firstColon >= 0) {
                    String details = original.substring(firstColon + 1);
                    // If the details contain a second colon, the text after it is the
                    // human-readable target (e.g. "Card.MultiColor:multicolored" → "multicolored").
                    int innerColon = details.indexOf(':');
                    if (innerColon >= 0) {
                        details = details.substring(innerColon + 1);
                    }
                    title = "Protection from " + details.toLowerCase();
                }
                // else: getOriginal() had no colon (shouldn't happen), leave title as-is
            }
        }

        String reduceCost = extractReduceCostDescription(kw, ki.getOriginal());
        if (reduceCost != null) {
            title = title + ". " + reduceCost;
        }

        AbilityType kwType = AbilityType.classifyKeyword(kw, activatable, !ki.getTriggers().isEmpty());
        return new StandardKeyword(kwType, AbilityDescription.applyCasing(title));
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
        String[] parts = original.split(":", 4);
        if (parts.length < 2) return original;

        String costPart = parts[1];

        // Mana: strip all ExileCtrlOrGrave<…> specs and XMin\d+ constraint tokens
        String mana = costPart
                .replaceAll("ExileCtrlOrGrave<[^>]*>", "")
                .replaceAll("XMin\\d+", "")
                .trim();
        String manaCostStr = new Cost(mana, false).toSimpleString();

        // Type description (three sub-patterns)
        String typeDesc;
        if (parts.length >= 3 && !parts[2].isEmpty()) {
            // Explicit human-readable description in k[2] (Throne of Grim Captain, Eye of Ojer Taq)
            typeDesc = parts[2];
        } else if (costPart.contains("XMin")) {
            // Variable-count exile (Sunbird Standard): "one or more"
            typeDesc = "one or more";
        } else {
            // Simple type: extract first word before '.' or '+' from ExileCtrlOrGrave<N/TYPE…>
            int ecoIdx = costPart.indexOf("ExileCtrlOrGrave<");
            int slashIdx = ecoIdx >= 0 ? costPart.indexOf('/', ecoIdx) : -1;
            if (slashIdx >= 0) {
                int typeStart = slashIdx + 1;
                int typeEnd = typeStart;
                while (typeEnd < costPart.length()
                        && ".+>/<".indexOf(costPart.charAt(typeEnd)) < 0) {
                    typeEnd++;
                }
                typeDesc = costPart.substring(typeStart, typeEnd);
            } else {
                typeDesc = "artifact"; // fallback
            }
        }

        return "Craft with " + typeDesc + " " + manaCostStr;
    }

    /**
     * Extract cost-reduction description from a keyword's original string, or null if absent.
     *
     * Pattern A (Equip): "Equip:cost:::ReduceCost$ SVar:full sentence"
     * Pattern B (Specialize): "Specialize:cost::full sentence.:ReduceCost$ SVar"
     *   → uses the full sentence from parts[3] directly
     *
     * (Adapt and Monstrosity were removed from the {@link Keyword} enum in
     * recent Forge revisions, so their pattern is no longer reachable.)
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
            String[] parts = original.split(":", 5);
            if (parts.length >= 5 && original.contains("ReduceCost$") && !parts[3].isEmpty()) {
                return parts[3];
            }
        }

        return null;
    }
}
