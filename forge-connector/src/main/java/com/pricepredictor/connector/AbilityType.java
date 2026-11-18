package com.pricepredictor.connector;

import forge.game.trigger.Trigger;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Categories for ability lines in converted card output.
 */
@Getter
@RequiredArgsConstructor
public enum AbilityType {
    ACTIVATED("activated", true),
    TRIGGERED("triggered", false),
    STATIC("static", false),
    REPLACEMENT("replacement", false),
    CHAPTER("chapter", false) {
        @Override public String formatDescription(String desc) {
            return uppercaseRomanPrefix(desc);
        }
    },
    LEVEL("level", true),
    ALTERNATE_COST("alternate cost", false),
    COST_REDUCTION("cost reduction", false),
    ADDITIONAL_COST("additional cost", false),
    DRAFT("draft", false),
    SPELL("spell", true),
    PLANESWALKER("planeswalker", true) {
        @Override public String formatDescription(String desc) {
            return bracketLoyaltyCost(desc);
        }
    },
    OPTION("option", true),
    TEXT("text", false);

    private final String outputPrefix;
    private final boolean actionable;

    public boolean isCostType() {
        return this == ALTERNATE_COST || this == ADDITIONAL_COST || this == COST_REDUCTION;
    }

    private static final String[] TRIGGER_WORDS = {"when ", "whenever ", "at "};

    /**
     * The line kind a trigger renders as, given its printed prose: a static
     * trigger is a replacement in the printed text, since it changes how an
     * event happens ("as CARDNAME enters") rather than responding to it. One
     * whose prose opens with a trigger word is a triggered ability all the same
     * (a triggered mana ability, rule 605.1b: "whenever you tap a land for
     * mana, ...").
     */
    public static AbilityType forTrigger(Trigger trigger, String prose) {
        return trigger.isStatic() && !opensWithTriggerWord(prose) ? REPLACEMENT : TRIGGERED;
    }

    private static boolean opensWithTriggerWord(String prose) {
        String lower = prose.toLowerCase(Locale.ROOT);
        for (String word : TRIGGER_WORDS) {
            if (lower.startsWith(word)) return true;
        }
        return false;
    }

    /** Apply type-specific formatting to a description. Default is identity. */
    public String formatDescription(String desc) {
        return desc;
    }

    // --- Type-specific formatting patterns ---

    private static final Pattern ROMAN_PREFIX =
            Pattern.compile("^([ivxlcdm]+(?:,\\s*[ivxlcdm]+)*)\\s*\u2014");
    private static final Pattern LOYALTY_COST =
            Pattern.compile("^([+\\-](?:\\d+|X)|0): ", Pattern.CASE_INSENSITIVE);

    static String uppercaseRomanPrefix(String text) {
        Matcher m = ROMAN_PREFIX.matcher(text);
        if (m.find()) {
            return m.group(1).toUpperCase(Locale.ROOT) + text.substring(m.group(1).length());
        }
        return text;
    }

    static String bracketLoyaltyCost(String text) {
        Matcher m = LOYALTY_COST.matcher(text);
        if (m.find()) {
            return "[" + m.group(1) + "]: " + text.substring(m.end());
        }
        return text;
    }
}
