package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.AbilityDescription;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.spellability.SpellAbility;
import forge.util.Lang;

import java.util.regex.Pattern;

/**
 * The sentence a charm states its choice with, built as Forge's
 * {@code CharmEffect.makeFormatedDescription} builds it: who chooses, how many
 * modes (a number, {@code X}, a minimum), the {@code ChoiceRestriction$}, "at
 * random", that a mode may be chosen again, an activation limit, then the
 * charm's {@code AdditionalDescription$}. Every path that prints a charm's
 * choice (a modal spell, an activated charm, a trigger's {@code ABILITY}
 * placeholder, a chapter, an Attraction's visit) prints this one.
 *
 * <p>Two departures from Forge, each toward the printed card: no trailing em
 * dash (Forge ends a header followed by bullets with one), and a count Forge
 * can only work out in a game reads as the card prints it. A count named
 * {@code X} that the {@code AdditionalDescription$} does not follow with a
 * sentence of its own is the printed X ("Choose X", "choose up to X, where X
 * is…"); any other expression reads as its minimum ("choose one. If it was
 * kicked, choose both instead.", "choose one or more. Each mode…"), "one"
 * without one, and "up to that many" when the minimum is 0 (Tranquil
 * Frillback).
 */
public final class CharmHeader {

    private static final Pattern TRAILING_EM_DASH = Pattern.compile("\\s*\u2014\\s*$");

    private CharmHeader() {}

    /** The choice sentence of {@code sa}, a charm, without a trailing em dash. */
    public static String of(SpellAbility sa) {
        String pawprint = pawprintHeader(sa);
        StringBuilder header = new StringBuilder(pawprint != null ? pawprint : choice(sa));
        if (pawprint == null) appendLimits(sa, header);
        appendAdditionalDescription(sa, header);
        return TRAILING_EM_DASH.matcher(header).replaceAll("").trim();
    }

    /** A pawprint charm's header, built from its {@code Pawprint} total; null for any other charm. */
    private static String pawprintHeader(SpellAbility sa) {
        if (!sa.hasParam("Pawprint")) return null;
        String header = "Choose up to " + sa.getParam("Pawprint") + " {P} worth of modes.";
        return sa.hasParam("CanRepeatModes")
                ? header + " You may choose the same mode more than once." : header;
    }

    /** Who chooses how many, under which restriction, and whether at random. */
    private static String choice(SpellAbility sa) {
        String chooser = "Opponent".equals(sa.getParam("Chooser")) ? "An opponent chooses " : "Choose ";
        String choice = chooser + CharmCount.of(sa) + restriction(sa);
        return sa.hasParam("Random") ? choice + " at random." : choice;
    }

    /**
     * What Forge appends to a charm's count for its {@code ChoiceRestriction$},
     * with a leading space, or the empty string.
     */
    private static String restriction(SpellAbility sa) {
        return switch (sa.getParamOrDefault("ChoiceRestriction", "")) {
            case "ThisGame" -> " that hasn't been chosen";
            case "ThisTurn" -> " that hasn't been chosen this turn";
            case "YourLastCombat" -> " that wasn't chosen during your last combat";
            default -> "";
        };
    }

    /** The sentences Forge adds after the choice: a repeatable mode, an activation limit. */
    private static void appendLimits(SpellAbility sa, StringBuilder header) {
        if (sa.hasParam("CanRepeatModes")) {
            appendSentence(header, "You may choose the same mode more than once.");
        }
        if ("1".equals(sa.getParam("ActivationLimit"))) {
            appendSentence(header, "Activate only once each turn.");
        }
        if ("1".equals(sa.getParam("GameActivationLimit"))) {
            appendSentence(header, "Activate only once.");
        }
    }

    /**
     * The charm's {@code AdditionalDescription}, without reminder text, after
     * {@code header}, or the header alone.
     *
     * <p>Forge joins it directly when it opens with a period ({@code ". If
     * there are four or more card types..."}), after a comma when it opens
     * "where X", and after a space otherwise; a description opening a sentence
     * of its own follows a period, as the printed card has one there. A Spree
     * or Tiered charm has no header, and a description that states the choice
     * itself needs none.
     */
    private static void appendAdditionalDescription(SpellAbility sa, StringBuilder header) {
        String desc = sa.getParam("AdditionalDescription");
        if (desc == null || desc.isBlank()) return;
        desc = AbilityDescription.stripReminderText(desc).trim();
        if (statesTheChoice(sa, desc)) {
            header.setLength(0);
            header.append(desc);
        } else if (desc.startsWith(".")) {
            appendSentence(header, desc.substring(1).trim());
        } else if (desc.startsWith("where X")) {
            header.append(", ").append(desc);
        } else if (opensASentence(desc)) {
            appendSentence(header, desc);
        } else {
            header.append(' ').append(desc);
        }
    }

    /** Whether the description stands without a header: it opens "Choose", or the card has none. */
    private static boolean statesTheChoice(SpellAbility sa, String desc) {
        Card host = sa.getHostCard();
        return desc.regionMatches(true, 0, "choose", 0, "choose".length())
                || host.hasKeyword(Keyword.SPREE) || host.hasKeyword(Keyword.TIERED);
    }

    /** Whether {@code desc} opens a sentence of its own (a capital), not a continuation. */
    static boolean opensASentence(String desc) {
        return !desc.isEmpty() && Character.isUpperCase(desc.charAt(0));
    }

    /** Append {@code next} as a new sentence, ending the text so far with its period. */
    private static void appendSentence(StringBuilder text, String next) {
        if (text.charAt(text.length() - 1) != '.') text.append('.');
        if (!next.isEmpty()) text.append(' ').append(next);
    }

    /** A numeral as Forge writes one ("one", "three"). */
    static String numeral(int n) {
        return Lang.getNumeral(n);
    }
}
