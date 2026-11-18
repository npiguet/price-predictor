package com.pricepredictor.connector.ability;

import forge.game.cost.Cost;
import forge.game.keyword.KeywordInterface;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A Craft keyword's fields, read from
 * {@code Craft:<costPart>[:<typeDesc>[:<replaceDesc>]]}, where costPart holds
 * mana tokens and {@code ExileCtrlOrGrave<...>} exile specs.
 */
record Craft(String costPart, String explicitType) {

    /**
     * The type a Craft cost exiles: after the first {@code ExileCtrlOrGrave<}
     * and the first {@code /} that follows it, everything up to a {@code .},
     * {@code +}, {@code >}, {@code /} or {@code <}.
     */
    private static final Pattern EXILED_TYPE =
            Pattern.compile("ExileCtrlOrGrave<[^/]*/([^.+>/<]*)");

    /** The title of {@code ki}, or its original text when that has no cost field. */
    static String titleOf(KeywordInterface ki) {
        KeywordSpec spec = new KeywordSpec(ki.getOriginal());
        return spec.size() < 2
                ? ki.getOriginal() : new Craft(spec.part(1), spec.part(2)).title();
    }

    /** The terse oracle title: "Craft with &lt;typeDesc&gt; &lt;manaCost&gt;". */
    String title() {
        return "Craft with " + typeDescription() + " " + manaCost();
    }

    /** The mana: the cost part without its exile specs and XMin constraint tokens. */
    private String manaCost() {
        String mana = costPart
                .replaceAll("ExileCtrlOrGrave<[^>]*>", "")
                .replaceAll("XMin\\d+", "")
                .trim();
        return new Cost(mana, false).toSimpleString();
    }

    /**
     * What the cost exiles, in three sub-patterns: the explicit description
     * (Throne of Grim Captain, Eye of Ojer Taq); "one or more" for a variable
     * count (Sunbird Standard); else the first word of TYPE in the exile spec.
     */
    private String typeDescription() {
        if (explicitType != null && !explicitType.isEmpty()) return explicitType;
        if (costPart.contains("XMin")) return "one or more";
        Matcher exiled = EXILED_TYPE.matcher(costPart);
        return exiled.find() ? exiled.group(1) : "artifact"; // fallback
    }
}
