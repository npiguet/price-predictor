package com.pricepredictor.connector.ability;

import forge.game.spellability.SpellAbility;

/**
 * How many modes a charm's header says to choose ("one", "up to two", "one or
 * more", "any number", "X"), as Forge's
 * {@code CharmEffect.makeFormatedDescription} counts them; see
 * {@link CharmHeader} for the counts only a game can work out.
 */
final class CharmCount {

    private CharmCount() {}

    /** The count of {@code sa}, a charm. */
    static String of(SpellAbility sa) {
        String numParam = sa.getParamOrDefault("CharmNum", "1").trim();
        int num = parseSimpleInt(numParam);
        if (num >= 0) return counted(sa, num);
        if (isPrintedX(sa, numParam)) return sa.hasParam("MinCharmNum") && minimum(sa) == 0 ? "up to X" : "X";
        return expression(sa);
    }

    /** A number of modes, at least the minimum, as Forge words each pair. */
    private static String counted(SpellAbility sa, int charmNum) {
        int choices = sa.getParamOrDefault("Choices", "").split(",").length;
        int num = sa.hasParam("CanRepeatModes") ? charmNum : Math.min(charmNum, choices);
        int min = sa.hasParam("MinCharmNum") ? minimum(sa) : num;
        if (min < 0 || num == min) return CharmHeader.numeral(num);
        if (min == 0) return num == choices ? "any number" : "up to " + CharmHeader.numeral(num);
        return CharmHeader.numeral(min) + " or " + (choices == 2 ? "both" : "more");
    }

    /**
     * Whether {@code CharmNum$ X} is the X the card prints: an
     * {@code AdditionalDescription$} opening a sentence of its own states the
     * count another way (Atlantis Attacks' SVar X, "choose one. If this spell
     * was cast using teamwork, choose both instead.").
     */
    private static boolean isPrintedX(SpellAbility sa, String numParam) {
        if (!"X".equals(numParam)) return false;
        String desc = sa.getParamOrDefault("AdditionalDescription", "").trim();
        return !desc.startsWith(".") && !CharmHeader.opensASentence(desc);
    }

    /** A count only a game can work out, as its minimum reads. */
    private static String expression(SpellAbility sa) {
        int min = sa.hasParam("MinCharmNum") ? minimum(sa) : -1;
        if (min == 0) return "up to that many";
        return CharmHeader.numeral(Math.max(min, 1));
    }

    /** The {@code MinCharmNum$} as a number, or -1 when it is an expression. */
    private static int minimum(SpellAbility sa) {
        return parseSimpleInt(sa.getParam("MinCharmNum"));
    }

    private static int parseSimpleInt(String s) {
        if (s == null) return -1;
        try {
            return Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            return -1;
        }
    }
}
