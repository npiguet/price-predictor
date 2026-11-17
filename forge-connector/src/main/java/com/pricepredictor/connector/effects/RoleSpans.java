package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

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

    /** One tagged character range, half-open: {@code [start, end)}. */
    public record Span(int start, int end, SpanRole role) {
        public String toJson() {
            return Json.object()
                    .value("start", start)
                    .value("end", end)
                    .value("role", role)
                    .render();
        }
    }

    private static final String[] TRIGGER_WORDS = {"when ", "whenever ", "at "};

    /**
     * Role spans over {@code text}, the rendered prose of one line without its
     * {@code kind[n]:} prefix.
     *
     * <p>Offsets are relative to {@code text}, so a caller that renders the
     * prefix must add its length. The spans of the first three roles partition
     * the text, except that an ability word before a trigger condition
     * ("landfall — ") stays untagged; {@code target-spec} spans overlap
     * whichever of them contains them, because a target phrase is part of the clause it appears in.
     */
    public static List<Span> of(String text, boolean activated) {
        List<Span> spans = new ArrayList<>();
        if (text == null || text.isEmpty()) return spans;
        String lower = text.toLowerCase(Locale.ROOT);
        spans.addAll(clauseSpans(text, lower, activated));
        spans.addAll(targetSpans(lower));
        return spans;
    }

    /** The target spans alone, for a line whose clauses the caller tags itself. */
    static List<Span> targets(String text) {
        return targetSpans(text.toLowerCase(Locale.ROOT));
    }

    /** The condition, cost and effect spans: what remains once targets are set aside. */
    private static List<Span> clauseSpans(String text, String lower, boolean activated) {
        int length = text.length();
        int start = triggerStart(lower);
        int triggerEnd = start < 0 ? -1 : triggerConditionEnd(lower, start);
        if (triggerEnd > 0) {
            return List.of(new Span(start, triggerEnd, SpanRole.TRIGGER_CONDITION),
                    new Span(triggerEnd, length, SpanRole.EFFECT));
        }
        int costEnd = activated ? costEnd(text) : -1;
        if (costEnd > 0) {
            return List.of(new Span(0, costEnd, SpanRole.COST),
                    new Span(costEnd, length, SpanRole.EFFECT));
        }
        return List.of(new Span(0, length, SpanRole.EFFECT));
    }

    private static final Pattern ABILITY_WORD = Pattern.compile("[a-z][a-z' -]*");
    private static final String ABILITY_WORD_DASH = " \u2014 ";

    /**
     * Where a trigger condition starts, or -1 when the line does not open with
     * one: at 0, or after an ability word ("landfall — whenever ...").
     */
    private static int triggerStart(String lower) {
        if (opensTrigger(lower, 0)) return 0;
        int dash = lower.indexOf(ABILITY_WORD_DASH);
        if (dash <= 0 || !ABILITY_WORD.matcher(lower.substring(0, dash)).matches()) return -1;
        int start = dash + ABILITY_WORD_DASH.length();
        return opensTrigger(lower, start) ? start : -1;
    }

    private static boolean opensTrigger(String lower, int from) {
        for (String word : TRIGGER_WORDS) {
            if (lower.startsWith(word, from)) return true;
        }
        return false;
    }

    /**
     * End of the trigger condition that opens at {@code from}, or -1.
     *
     * <p>The condition ends at the first top-level comma, top-level meaning
     * outside any parenthesis, so a reminder-text comma does not split the line.
     */
    private static int triggerConditionEnd(String lower, int from) {
        return topLevelEnd(lower, from, ',', false);
    }

    /**
     * End of an activated ability's cost, or -1.
     *
     * <p>The converted format renders an activated ability as {@code cost:
     * effect}, so the cost is everything up to the first top-level colon. Mana
     * symbols carry no colon and neither does an ordinary effect clause, so the
     * first one is the separator.
     */
    static int costEnd(String text) {
        return topLevelEnd(text, 0, ':', true);
    }

    /**
     * One past the first {@code separator} at or after {@code from} that sits
     * outside every parenthesis (and, with {@code braces}, every mana symbol),
     * or -1.
     */
    private static int topLevelEnd(String text, int from, char separator, boolean braces) {
        String opening = braces ? "({" : "(";
        String closing = braces ? ")}" : ")";
        int depth = 0;
        for (int i = from; i < text.length(); i++) {
            char c = text.charAt(i);
            if (depth == 0 && c == separator) return i + 1;
            depth = Math.max(0, depth + bracketStep(c, opening, closing));
        }
        return -1;
    }

    private static int bracketStep(char c, String opening, String closing) {
        if (opening.indexOf(c) >= 0) return 1;
        return closing.indexOf(c) >= 0 ? -1 : 0;
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
            int end = phraseEnd(lower, start + TARGET_WORD.length());
            spans.add(new Span(start, end, SpanRole.TARGET_SPEC));
            from = end;
        }
        return spans;
    }

    /** End of the target phrase whose words begin at {@code after}, trailing blanks trimmed. */
    private static int phraseEnd(String lower, int after) {
        int end = lower.indexOf(TARGET_WORD, after);
        if (end < 0) end = lower.length();
        for (int i = after; i < end; i++) {
            if (",;.".indexOf(lower.charAt(i)) >= 0) {
                end = i;
                break;
            }
        }
        while (end > after && Character.isWhitespace(lower.charAt(end - 1))) {
            end--;
        }
        return end;
    }
}
