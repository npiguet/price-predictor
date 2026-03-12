package com.pricepredictor.connector;

import forge.game.keyword.Keyword;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.util.EnumMap;
import java.util.EnumSet;
import java.util.Map;
import java.util.Set;
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

    /** Apply type-specific formatting to a description. Default is identity. */
    public String formatDescription(String desc) {
        return desc;
    }

    // --- Type-specific formatting patterns ---

    private static final Pattern ROMAN_PREFIX =
            Pattern.compile("^([ivxlcdm]+(?:,\\s*[ivxlcdm]+)*)\\s*\u2014");
    private static final Pattern LOYALTY_COST =
            Pattern.compile("^([+\\-](?:\\d+|X)): ", Pattern.CASE_INSENSITIVE);

    static String uppercaseRomanPrefix(String text) {
        Matcher m = ROMAN_PREFIX.matcher(text);
        if (m.find()) {
            return m.group(1).toUpperCase() + text.substring(m.group(1).length());
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

    // --- Keyword classification ---

    private static final Map<Keyword, AbilityType> KEYWORD_COST_TYPES = buildKeywordCostTypes();

    private static final Set<Keyword> INTERNAL_KEYWORDS = EnumSet.of(
            Keyword.MAYFLASHCOST, Keyword.MAYFLASHSAC
    );

    private static Map<Keyword, AbilityType> buildKeywordCostTypes() {
        Map<Keyword, AbilityType> map = new EnumMap<>(Keyword.class);
        for (Keyword kw : EnumSet.of(
                Keyword.BESTOW, Keyword.BLITZ, Keyword.DASH, Keyword.DISTURB,
                Keyword.EMERGE, Keyword.ESCAPE, Keyword.EVOKE, Keyword.FLASHBACK,
                Keyword.FORETELL, Keyword.FREERUNNING, Keyword.MADNESS,
                Keyword.MEGAMORPH, Keyword.MORPH, Keyword.DISGUISE,
                Keyword.MORE_THAN_MEETS_THE_EYE, Keyword.MUTATE,
                Keyword.OVERLOAD, Keyword.PLOT, Keyword.PROTOTYPE,
                Keyword.PROWL, Keyword.RETRACE, Keyword.SNEAK,
                Keyword.SPECTACLE, Keyword.SURGE,
                Keyword.AWAKEN, Keyword.EMBALM, Keyword.ENCORE,
                Keyword.ETERNALIZE, Keyword.HARMONIZE, Keyword.IMPENDING,
                Keyword.JUMP_START, Keyword.MIRACLE, Keyword.NINJUTSU,
                Keyword.OFFERING, Keyword.SUSPEND, Keyword.UNEARTH)) {
            map.put(kw, ALTERNATE_COST);
        }
        for (Keyword kw : EnumSet.of(
                Keyword.BUYBACK, Keyword.ENTWINE, Keyword.KICKER,
                Keyword.MULTIKICKER, Keyword.REPLICATE, Keyword.SQUAD,
                Keyword.STRIVE, Keyword.CASUALTY, Keyword.OFFSPRING,
                Keyword.BARGAIN, Keyword.CONSPIRE, Keyword.ESCALATE,
                Keyword.MAYFLASHCOST, Keyword.SPLICE, Keyword.SPREE,
                Keyword.TIERED)) {
            map.put(kw, ADDITIONAL_COST);
        }
        for (Keyword kw : EnumSet.of(
                Keyword.AFFINITY, Keyword.ASSIST, Keyword.CONVOKE,
                Keyword.DELVE, Keyword.IMPROVISE, Keyword.UNDAUNTED)) {
            map.put(kw, COST_REDUCTION);
        }
        return map;
    }

    /** Classify a keyword into an AbilityType. Cost keywords get a fixed type, others by trait presence. */
    public static AbilityType classifyKeyword(Keyword kw, boolean hasAbilities, boolean hasTriggers) {
        AbilityType costType = KEYWORD_COST_TYPES.get(kw);
        if (costType != null) return costType;
        if (hasAbilities) return ACTIVATED;
        if (hasTriggers) return TRIGGERED;
        return STATIC;
    }

    public static boolean isInternalKeyword(Keyword kw) {
        return INTERNAL_KEYWORDS.contains(kw);
    }
}
