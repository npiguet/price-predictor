package com.pricepredictor.connector;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

/**
 * Categories for ability lines in converted card output.
 */
@Getter
@RequiredArgsConstructor
public enum AbilityType {
    KEYWORD_PASSIVE("keyword", false),
    KEYWORD_ACTIVE("keyword", true),
    ACTIVATED("activated", true),
    TRIGGERED("triggered", false),
    STATIC("static", false),
    REPLACEMENT("replacement", false),
    CHAPTER("chapter", false),
    LEVEL("level", true),
    SPELL("spell", true),
    PLANESWALKER("planeswalker", true),
    OPTION("option", true),
    TEXT("text", false);

    private final String outputPrefix;
    private final boolean actionable;
}
