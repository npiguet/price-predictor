package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityType;

import java.util.ArrayList;
import java.util.List;

/**
 * The role spans of one rendered line, from its prose and the ability that
 * printed it.
 *
 * <p>{@link RoleSpans} reads the roles off the prose. A cost that opens a line
 * as a sentence of its own (a spell's additional cost) has no colon to find
 * it by, so the ability states its length, and that sentence is the
 * {@code cost}, the rest of the line the {@code effect}.
 */
final class LineRoleSpans {

    private LineRoleSpans() {
    }

    static List<RoleSpans.Span> of(String prose, Ability owner) {
        List<RoleSpans.Span> spans = RoleSpans.of(prose, owner.type() == AbilityType.ACTIVATED);
        int costLength = owner.costLength();
        return costLength > 0 ? withCostSentence(spans, costLength, prose.length()) : spans;
    }

    /** The cost sentence and the effect after it, then {@code spans}' target spans. */
    private static List<RoleSpans.Span> withCostSentence(List<RoleSpans.Span> spans,
                                                         int costLength, int length) {
        List<RoleSpans.Span> result = new ArrayList<>(List.of(
                new RoleSpans.Span(0, costLength, SpanRole.COST),
                new RoleSpans.Span(costLength, length, SpanRole.EFFECT)));
        spans.stream().filter(span -> span.role() == SpanRole.TARGET_SPEC).forEach(result::add);
        return result;
    }
}
