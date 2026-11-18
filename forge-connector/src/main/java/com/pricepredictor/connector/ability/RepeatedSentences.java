package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.RenderedLine;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * When a sentence of an ability's chain repeats what its line already states.
 *
 * <p>A line joins its own description and the descriptions of its chain's
 * nodes, and Forge's scripts often restate an effect of the description on a
 * sub-ability (Chill of the Grave's "It doesn't untap ..."), or an additional
 * cost in the spell's description. The line states each such sentence once.
 * A repeat the script writes stays: inside one description (Rally the Horde's
 * three "exile the top card of your library."), or as a sub-ability declaring
 * its parent's description again word for word (Common Bond's two counters).
 */
final class RepeatedSentences {

    private RepeatedSentences() {}

    /**
     * {@code text} followed by the descriptions of {@code nodes} and of every
     * node below them, in order, less each of their sentences that the line
     * already states before that node.
     */
    static String appendMissing(String text, List<Ability> nodes) {
        Line line = new Line(text);
        for (Ability node : nodes) {
            line.add(node, Description.NONE);
        }
        return line.text();
    }

    /**
     * Whether {@code child}, the description of an SVar a {@code DB$ Effect}
     * creates, says nothing {@code ancestor}, its nearest described
     * ancestor's text, does not, or is a Forge-internal display string.
     */
    static boolean restatesAncestor(String child, String ancestor) {
        // "effectsource" is never oracle text; checked without an ancestor too.
        if (child.contains("effectsource")) return true;
        if (ancestor == null) return false;
        return ancestor.contains(child) || ancestor.contains(new Description(child).selfAsCardname());
    }

    /** A line as its nodes are appended to it. */
    private static final class Line {
        private final StringBuilder text;

        Line(String text) {
            this.text = new StringBuilder(text);
        }

        String text() {
            return text.toString();
        }

        /**
         * Append {@code node} and the nodes below it; {@code printedParent} is
         * its parent's description when the line printed it whole.
         */
        void add(Ability node, Description printedParent) {
            // Read as the file prints it: a line break marker is a space there.
            Description own = new Description(RenderedLine.foldLineBreaks(node.descriptionText()));
            List<String> missing = own.sentencesNotIn(text());
            boolean whole = own.sameAs(printedParent) || missing.size() == own.sentences().size();
            append(whole ? own.text() : String.join(" ", missing));
            Description parent = whole ? own : Description.NONE;
            for (Ability child : node.subAbilities()) {
                add(child, parent);
            }
        }

        private void append(String words) {
            if (words.isEmpty()) return;
            if (!text.isEmpty()) text.append(' ');
            text.append(words);
        }
    }

    /** One node's description, as the line would print it. */
    private record Description(String text) {

        static final Description NONE = new Description("");

        private static final Pattern SENTENCE_END = Pattern.compile("(?<=[.!?])\\s+");

        /** Its sentences, in order, a repeated one as often as it is written. */
        List<String> sentences() {
            List<String> sentences = new ArrayList<>();
            for (String sentence : SENTENCE_END.split(text.strip())) {
                if (!sentence.isEmpty()) sentences.add(sentence);
            }
            return sentences;
        }

        /** Its sentences {@code stated} does not hold. */
        List<String> sentencesNotIn(String stated) {
            return sentences().stream().filter(sentence -> !stated.contains(sentence)).toList();
        }

        /** Whether it says word for word what {@code other} says. */
        boolean sameAs(Description other) {
            return !other.text.isEmpty() && text.strip().equals(other.text.strip());
        }

        /**
         * "this card/creature/permanent/spell" in a replacement's
         * {@code Description$} names the card a {@code SpellDescription} calls
         * CARDNAME.
         */
        String selfAsCardname() {
            return text
                    .replace("this card", "CARDNAME")
                    .replace("this creature", "CARDNAME")
                    .replace("this permanent", "CARDNAME")
                    .replace("this spell", "CARDNAME");
        }
    }
}
