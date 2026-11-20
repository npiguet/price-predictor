package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.AbilityDescription;
import forge.game.spellability.SpellAbility;

import java.util.Collection;
import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import java.util.StringJoiner;

/**
 * The text of an instant's or sorcery's spell whose chain states nothing,
 * no ability of it carrying a {@code SpellDescription$}: the trait's own
 * {@code Description$} (Garbage Fire), else the sentences of the face's
 * oracle that no other line of the face states (Crashing Wave's effect,
 * beside the waterbend cost its static prints; Tyranid Invasion's whole
 * oracle). Cascade's and rebound's oracle lines are their keywords' own, so
 * Into the Time Vortex's placeholder spell has no text.
 *
 * <p>Without it such a spell printed no line, and a record acting through it
 * read no text.
 */
public final class UndescribedSpellText {

    private UndescribedSpellText() {}

    /**
     * The text of {@code sa}, or nothing when it has none anywhere.
     *
     * @param oracle the face's {@code Oracle:} value, its line breaks written
     *               as {@code \n} escapes
     * @param name   the face's name, written as CARDNAME
     * @param stated the description texts of the face's other lines
     */
    public static Optional<String> of(SpellAbility sa, String oracle, String name, Collection<String> stated) {
        return Optional.ofNullable(AbilityDescription.normalize(sa.getParam("Description")))
                .or(() -> unstatedOracle(sa, oracle, name, stated));
    }

    /** The oracle's lines no other line, nor {@code sa}'s own cost, states, joined. */
    private static Optional<String> unstatedOracle(
            SpellAbility sa, String oracle, String name, Collection<String> stated) {
        if (oracle == null) return Optional.empty();
        Set<String> known = new HashSet<>(stated);
        String cost = SpellAdditionalCost.sentenceOf(sa);
        if (cost != null) known.add(cost);
        StringJoiner unstated = new StringJoiner(" ");
        for (String raw : oracle.replace("\\n", "\n").split("\n")) {
            String line = AbilityDescription.scriptLine(raw, name);
            if (line != null && !isStated(line, known)) unstated.add(line);
        }
        return unstated.length() == 0 ? Optional.empty() : Optional.of(unstated.toString());
    }

    /**
     * Whether a line states {@code line}: an additional-cost line prints the
     * sentence without its "as an additional cost to cast this spell, ".
     */
    private static boolean isStated(String line, Set<String> stated) {
        return stated.contains(line)
                || stated.contains(AbilityDescription.stripAdditionalCostPrefix(line));
    }
}
