package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.NonBlankString;
import forge.game.cost.Cost;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.keyword.KeywordWithTypeInterface;

import java.util.Map;
import java.util.Optional;
import java.util.function.BiFunction;

/**
 * Standard keyword ability — handles defined keywords and the undefined keyword fallback.
 * Covers KICKER dual cost, AFFINITY type description, and internal keyword reminder text.
 */
public record StandardKeyword(AbilityType type, NonBlankString descriptionText) implements Ability {

    /** Per-keyword title formatters: (ki, currentTitle) → newTitle. */
    private static final Map<Keyword, BiFunction<KeywordInterface, String, String>> KEYWORD_FORMATTERS =
            Map.of(
                Keyword.KICKER,     StandardKeyword::formatKicker,
                Keyword.AFFINITY,   StandardKeyword::formatAffinity,
                Keyword.CRAFT,      (ki, title) -> buildCraftTitle(ki.getOriginal()),
                Keyword.PROTECTION, StandardKeyword::formatProtection
            );

    public static StandardKeyword of(KeywordInterface ki, Keyword kw) {
        boolean activatable = !ki.getAbilities().isEmpty();
        String title = ki.getTitle();

        if (kw == Keyword.UNDEFINED) {
            if (title == null || title.isEmpty()) {
                title = ki.getOriginal();
            }
        } else {
            // Forge's getTitle() can return an unusable string for some keywords:
            //   - null or empty: no title computed (fall back to raw getOriginal())
            //   - trailing whitespace: Protection.getTitle() returns "Protection from "
            //     because Protection.parse() is empty in forge — fromWhat is never set.
            //     This is a forge bug; the trailing space lets us detect and bypass it.
            if (title == null || title.isEmpty() || !title.trim().equals(title)) {
                title = ki.getOriginal();
            }
            // Internal keywords (MAYFLASHCOST, MAYFLASHSAC) are forge-engine mechanics
            // with no player-facing name. Their reminder text holds the human-readable
            // oracle description (e.g. "you may pay {W} rather than pay this spell's
            // mana cost"), so we substitute reminder text for the title here.
            if (AbilityType.isInternalKeyword(kw)) {
                title = Optional.ofNullable(ki.getReminderText())
                        .filter(r -> !r.isEmpty())
                        .orElse(title);
            }
            var formatter = KEYWORD_FORMATTERS.get(kw);
            if (formatter != null) {
                title = formatter.apply(ki, title);
            }
        }

        String finalTitle = title;
        title = extractReduceCostDescription(kw, ki.getOriginal())
                .map(rc -> finalTitle + ". " + rc)
                .orElse(finalTitle);

        AbilityType kwType = AbilityType.classifyKeyword(kw, activatable, !ki.getTriggers().isEmpty());
        return new StandardKeyword(kwType, AbilityDescription.applyCasing(NonBlankString.require(title)));
    }

    // --- Per-keyword title formatters ---

    /**
     * Format dual-kicker title: "Kicker {cost1} and/or {cost2}".
     *
     * <p>We re-parse {@code getOriginal()} to extract cost2 rather than using
     * {@code ki.getReminderText()}, which returns the full sentence ("You may pay an
     * additional … as you cast this spell."). Stripping that sentence programmatically
     * would be more fragile than extracting the cost directly.
     * Note: {@code Kicker.getTitle()} (inherited from {@code KeywordWithCost}) only
     * renders the first cost, so it cannot be used for dual kicker either.
     */
    private static String formatKicker(KeywordInterface ki, String title) {
        KeywordFields f = KeywordFields.from(ki, 3);
        if (f.hasField(2)) {
            Cost cost2 = new Cost(f.field(2), false);
            return title + " and/or " + cost2.toSimpleString();
        }
        return title;
    }

    private static String formatAffinity(KeywordInterface ki, String title) {
        if (ki instanceof KeywordWithTypeInterface kti) {
            return "Affinity for " + kti.getTypeDescription();
        }
        return title;
    }

    private static String formatProtection(KeywordInterface ki, String title) {
        // getOriginal() returns the raw script keyword, e.g.:
        //   "Protection:Creature"                      (simple type → use field 1)
        //   "Protection:Card.MultiColor:multicolored"  (filter + human-readable → use field 2)
        // The most specific field (last non-empty after keyword name) is always the right one.
        String details = KeywordFields.from(ki, 3).lastNonEmptyField();
        if (!details.isEmpty()) {
            return "Protection from " + details.toLowerCase();
        }
        // getOriginal() had no colon (shouldn't happen), leave title as-is
        return title;
    }

    // --- Craft title construction ---

    /**
     * Build the terse oracle title for a Craft keyword: "Craft with &lt;typeDesc&gt; &lt;manaCost&gt;".
     *
     * <p>Forge's {@code Craft} class already parses the same {@code ki.getOriginal()} string
     * internally (into {@code manaString} and {@code exileString}), but those fields are
     * package-private with no public accessor. Even if exposed, their format is reminder-text
     * oriented ("Exile 2 artifacts from among…"), not the terse oracle title we need here.
     * Manual parsing is therefore the correct approach.
     *
     * <p>{@code ki.getOriginal()} format: {@code "Craft:<costPart>[:<typeDesc>[:<replaceDesc>]]}
     * where {@code costPart} contains mana tokens and {@code ExileCtrlOrGrave<…>} exile specs.
     */
    private static String buildCraftTitle(String original) {
        KeywordFields f = KeywordFields.parse(original, 4);
        if (!f.hasField(1)) return original;
        String costPart = f.field(1);
        return "Craft with " + resolveCraftTypeDescription(costPart, f)
                + " " + extractCraftMana(costPart);
    }

    /**
     * Strip {@code ExileCtrlOrGrave<…>} exile specs and {@code XMin\d+} constraint tokens
     * from {@code costPart}, then parse the remainder as a {@link Cost} and return its
     * simple string representation.
     */
    private static String extractCraftMana(String costPart) {
        String mana = costPart
                .replaceAll("ExileCtrlOrGrave<[^>]*>", "")
                .replaceAll("XMin\\d+", "")
                .trim();
        return new Cost(mana, false).toSimpleString();
    }

    /**
     * Resolve the type description for a Craft keyword, choosing among three sub-patterns:
     * <ol>
     *   <li><b>Explicit</b>: {@code parts[2]} is present and non-empty — use as-is
     *       (e.g. Throne of Grim Captain: "two legendary creatures").</li>
     *   <li><b>Variable count</b>: {@code costPart} contains {@code XMin} — use
     *       {@code "one or more"} (e.g. Sunbird Standard).</li>
     *   <li><b>Simple type</b>: extract the first word before {@code '.'}, {@code '+'}, or
     *       {@code '>'} from the {@code ExileCtrlOrGrave<N/TYPE…>} spec
     *       (e.g. {@code "ExileCtrlOrGrave<2/Artifact>"} → {@code "artifact"}).</li>
     * </ol>
     */
    private static String resolveCraftTypeDescription(String costPart, KeywordFields f) {
        if (f.hasField(2)) {
            return f.field(2);
        }
        if (costPart.contains("XMin")) {
            return "one or more";
        }
        // Extract the type word from ExileCtrlOrGrave<N/TYPE…>
        int ecoIdx = costPart.indexOf("ExileCtrlOrGrave<");
        int slashIdx = ecoIdx >= 0 ? costPart.indexOf('/', ecoIdx) : -1;
        if (slashIdx >= 0) {
            int typeStart = slashIdx + 1;
            int typeEnd = typeStart;
            while (typeEnd < costPart.length()
                    && ".+>/<".indexOf(costPart.charAt(typeEnd)) < 0) {
                typeEnd++;
            }
            return costPart.substring(typeStart, typeEnd);
        }
        return "artifact"; // fallback: no ExileCtrlOrGrave spec found
    }

    // --- Reduce-cost description extraction ---

    /**
     * Extract cost-reduction description from a keyword's original string, or null if absent.
     *
     * <p>Three distinct grammar patterns are handled; each is isolated in its own method:
     * <ul>
     *   <li>Pattern A (generic): {@code "ReduceCost$ SVar:human-text"} anywhere in the string —
     *       used by Equip and other keywords that embed the description in an SVar reference.</li>
     *   <li>Pattern B (Adapt/Monstrosity): 5-field colon-delimited format with the "for each"
     *       description in {@code parts[4]}.</li>
     *   <li>Pattern C (Specialize): full sentence in {@code parts[3]} when the
     *       {@code ReduceCost$} marker is present.</li>
     * </ul>
     */
    private static Optional<String> extractReduceCostDescription(Keyword kw, String original) {
        if (original == null) return Optional.empty();
        return extractReduceCostFromSVar(original)
                .or(() -> extractAdaptMonstrosityReduceCost(kw, original))
                .or(() -> extractSpecializeReduceCost(kw, original));
    }

    /**
     * Pattern A: scan for {@code "ReduceCost$ SVar:human-text"} anywhere in the keyword string.
     * The SVar name is skipped; the human-readable text after the colon is returned.
     * Applies broadly (Equip, any keyword that embeds a ReduceCost$ SVar reference).
     */
    private static Optional<String> extractReduceCostFromSVar(String original) {
        int idx = original.indexOf("ReduceCost$ ");
        if (idx >= 0) {
            String after = original.substring(idx + "ReduceCost$ ".length());
            int colon = after.indexOf(':');
            if (colon >= 0) {
                return Optional.of(after.substring(colon + 1));
            }
        }
        return Optional.empty();
    }

    /**
     * Pattern B: Adapt/Monstrosity 5-field colon-delimited format.
     * When {@code parts[4]} is present and non-empty, synthesises:
     * {@code "This ability costs {1} less to activate for each <parts[4]>."}
     */
    private static Optional<String> extractAdaptMonstrosityReduceCost(Keyword kw, String original) {
        if (kw == Keyword.ADAPT || kw == Keyword.MONSTROSITY) {
            KeywordFields f = KeywordFields.parse(original, 5); // raw String, not KeywordInterface
            if (f.hasField(4)) {
                return Optional.of("This ability costs {1} less to activate for each " + f.field(4) + ".");
            }
        }
        return Optional.empty();
    }

    /**
     * Pattern C: Specialize — full sentence is in {@code parts[3]}, present when the
     * {@code ReduceCost$} marker also appears in the string.
     */
    private static Optional<String> extractSpecializeReduceCost(Keyword kw, String original) {
        if (kw == Keyword.SPECIALIZE) {
            KeywordFields f = KeywordFields.parse(original, 5);
            if (original.contains("ReduceCost$") && f.hasField(3)) {
                return Optional.of(f.field(3));
            }
        }
        return Optional.empty();
    }
}
