package com.pricepredictor.connector.effects;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Prose role spans over one rendered ability line.
 *
 * <p>Character ranges tagged {@code cost}, {@code effect},
 * {@code trigger-condition} or {@code target-spec}, over the converted prose
 * rather than the script. The encoder adds a role embedding per token, which is
 * what lets it learn that {@code {R}} in a cost means the opposite of
 * {@code {R}} in an effect — the role-polarity probe measures exactly that.
 *
 * <p>The rules are structural, not semantic, because the converted format is
 * structural: an activated ability renders as {@code cost: effect}, a triggered
 * one as {@code when/whenever/at <condition>, <effect>}, and a target phrase
 * always begins with the literal word "target". Anything that matches no rule
 * is left untagged rather than guessed at.
 */
public final class RoleSpans {

    private RoleSpans() {
    }

    public static final String COST = "cost";
    public static final String EFFECT = "effect";
    public static final String TRIGGER_CONDITION = "trigger-condition";
    public static final String TARGET_SPEC = "target-spec";

    /** One tagged character range, half-open: {@code [start, end)}. */
    public record Span(int start, int end, String role) {
        public String toJson() {
            return "{\"start\":" + start + ",\"end\":" + end
                    + ",\"role\":" + Json.string(role) + "}";
        }
    }

    private static final String[] TRIGGER_WORDS = {"when ", "whenever ", "at "};

    /**
     * Role spans over {@code text}, the rendered prose of one line without its
     * {@code kind[n]:} prefix.
     *
     * <p>Offsets are relative to {@code text}, so a caller that renders the
     * prefix must add its length. The spans of the first three roles partition
     * the text; {@code target-spec} spans overlap whichever of them contains
     * them, because a target phrase is part of the clause it appears in.
     */
    public static List<Span> of(String text, boolean activated) {
        List<Span> spans = new ArrayList<>();
        if (text == null || text.isEmpty()) return spans;
        String lower = text.toLowerCase(Locale.ROOT);

        int triggerEnd = triggerConditionEnd(lower);
        if (triggerEnd > 0) {
            spans.add(new Span(0, triggerEnd, TRIGGER_CONDITION));
            spans.add(new Span(triggerEnd, text.length(), EFFECT));
        } else if (activated) {
            int costEnd = costEnd(text);
            if (costEnd > 0) {
                spans.add(new Span(0, costEnd, COST));
                spans.add(new Span(costEnd, text.length(), EFFECT));
            } else {
                spans.add(new Span(0, text.length(), EFFECT));
            }
        } else {
            spans.add(new Span(0, text.length(), EFFECT));
        }

        spans.addAll(targetSpans(lower));
        return spans;
    }

    /**
     * End of a leading trigger condition, or -1.
     *
     * <p>A triggered line opens with "when", "whenever" or "at" and separates
     * its condition from its effect with the first top-level comma — top-level
     * meaning outside any parenthesis, so a reminder-text comma does not split
     * the line.
     */
    private static int triggerConditionEnd(String lower) {
        boolean opens = false;
        for (String word : TRIGGER_WORDS) {
            if (lower.startsWith(word)) {
                opens = true;
                break;
            }
        }
        if (!opens) return -1;
        int depth = 0;
        for (int i = 0; i < lower.length(); i++) {
            char c = lower.charAt(i);
            if (c == '(') depth++;
            else if (c == ')') depth = Math.max(0, depth - 1);
            else if (c == ',' && depth == 0) return i + 1;
        }
        return -1;
    }

    /**
     * End of an activated ability's cost, or -1.
     *
     * <p>The converted format renders an activated ability as {@code cost:
     * effect}, so the cost is everything up to the first top-level colon. Mana
     * symbols carry no colon and neither does an ordinary effect clause, so the
     * first one is the separator.
     */
    private static int costEnd(String text) {
        int depth = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c == '(' || c == '{') depth++;
            else if (c == ')' || c == '}') depth = Math.max(0, depth - 1);
            else if (c == ':' && depth == 0) return i + 1;
        }
        return -1;
    }

    private static final String TARGET_WORD = "target ";

    /**
     * Spans covering each "target …" phrase.
     *
     * <p>A phrase runs from the word "target" to whichever comes first: a
     * clause boundary (comma, semicolon, period, end of line) or the next
     * "target" phrase. Bounding on the next phrase is what keeps a line with
     * two targets — "target creature fights target creature you don't control"
     * — from collapsing into one span that swallows both.
     *
     * <p>The bound is where the phrase ends, not where its restriction ends, so
     * a span can carry a trailing verb. Over-inclusion costs the encoder a
     * token of context; missing a target entirely would hide the restriction
     * the model is meant to learn.
     */
    private static List<Span> targetSpans(String lower) {
        List<Span> spans = new ArrayList<>();
        int from = 0;
        while (true) {
            int start = lower.indexOf(TARGET_WORD, from);
            if (start < 0) break;
            if (start > 0 && Character.isLetterOrDigit(lower.charAt(start - 1))) {
                from = start + 1;
                continue;
            }
            int after = start + TARGET_WORD.length();
            int end = lower.indexOf(TARGET_WORD, after);
            if (end < 0) end = lower.length();
            for (int i = after; i < end; i++) {
                char c = lower.charAt(i);
                if (c == ',' || c == ';' || c == '.') {
                    end = i;
                    break;
                }
            }
            while (end > after && Character.isWhitespace(lower.charAt(end - 1))) {
                end--;
            }
            spans.add(new Span(start, end, TARGET_SPEC));
            from = end;
        }
        return spans;
    }
}
