package com.pricepredictor.connector;

/**
 * Categories for ability lines in converted card output.
 */
public enum AbilityType {
    KEYWORD_PASSIVE("keyword", false),
    KEYWORD_ACTIVE("keyword", true),
    ACTIVATED("activated", true),
    TRIGGERED("triggered", false),
    STATIC("static", false),
    REPLACEMENT("replacement", false),
    SPELL("spell", true),
    PLANESWALKER("planeswalker", true),
    OPTION("option", true),
    TEXT("text", false);

    private final String outputPrefix;
    private final boolean actionable;

    AbilityType(String outputPrefix, boolean actionable) {
        this.outputPrefix = outputPrefix;
        this.actionable = actionable;
    }

    public String getOutputPrefix() {
        return outputPrefix;
    }

    public boolean isActionable() {
        return actionable;
    }
}
