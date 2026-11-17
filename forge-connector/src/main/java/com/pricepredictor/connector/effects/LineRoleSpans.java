package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityType;

import java.util.ArrayList;
import java.util.List;

/**
 * The role spans of one rendered line, from its prose and the ability that
 * printed it.
 *
 * <p>{@link RoleSpans} reads the roles off the prose; this class picks which
 * reading a line gets by its kind. A spell's additional cost opens the line as
 * a sentence of its own with no colon to find it by, so the ability states its
 * length. A cost-only line (additional cost, alternate cost, cost reduction) is
 * {@code cost} throughout. A planeswalker's loyalty cost and a level's cost end
 * at the first colon, and what follows reads as an ordinary clause.
 */
final class LineRoleSpans {

    private LineRoleSpans() {
    }

    /** A rendered line: its prose, its kind and the length of its cost sentence, if any. */
    private record Line(String prose, AbilityType type, int costLength) {
        int length() {
            return prose.length();
        }
    }

    static List<RoleSpans.Span> of(String prose, Ability owner) {
        return of(prose, owner.type(), owner.costLength());
    }

    static List<RoleSpans.Span> of(String prose, AbilityType type, int costLength) {
        if (prose == null || prose.isEmpty()) return new ArrayList<>();
        return spansOf(new Line(prose, type, costLength));
    }

    private static List<RoleSpans.Span> spansOf(Line line) {
        if (line.costLength() > 0) return withCostSentence(line);
        if (line.type().isCostType()) return wholeCost(line);
        if (line.type() == AbilityType.PLANESWALKER || line.type() == AbilityType.LEVEL) {
            return costThenClause(line);
        }
        return RoleSpans.of(line.prose(), line.type() == AbilityType.ACTIVATED);
    }

    /** The cost sentence and the effect after it, then the line's target spans. */
    private static List<RoleSpans.Span> withCostSentence(Line line) {
        List<RoleSpans.Span> result = new ArrayList<>(List.of(
                new RoleSpans.Span(0, line.costLength(), SpanRole.COST),
                new RoleSpans.Span(line.costLength(), line.length(), SpanRole.EFFECT)));
        result.addAll(RoleSpans.targets(line.prose()));
        return result;
    }

    private static List<RoleSpans.Span> wholeCost(Line line) {
        List<RoleSpans.Span> result = new ArrayList<>();
        result.add(new RoleSpans.Span(0, line.length(), SpanRole.COST));
        result.addAll(RoleSpans.targets(line.prose()));
        return result;
    }

    /**
     * A loyalty cost ({@code [+1]:}) or a level cost ({@code {1}{B}:}), then the
     * clause after it. A level line may have no cost at all (level 1), so its
     * cost must open with a mana symbol.
     */
    private static List<RoleSpans.Span> costThenClause(Line line) {
        int costEnd = RoleSpans.costEnd(line.prose());
        boolean hasCost = costEnd > 0
                && (line.type() == AbilityType.PLANESWALKER || line.prose().startsWith("{"));
        if (!hasCost) return RoleSpans.of(line.prose(), false);

        List<RoleSpans.Span> result = new ArrayList<>();
        result.add(new RoleSpans.Span(0, costEnd, SpanRole.COST));
        result.addAll(clauseAfter(line, costEnd));
        return result;
    }

    /** The spans of the clause that follows a cost ending at {@code costEnd}. */
    private static List<RoleSpans.Span> clauseAfter(Line line, int costEnd) {
        int lead = 0;
        while (costEnd + lead < line.length() && line.prose().charAt(costEnd + lead) == ' ') lead++;
        int base = costEnd + lead;
        List<RoleSpans.Span> result = new ArrayList<>();
        for (RoleSpans.Span span : RoleSpans.of(line.prose().substring(base), false)) {
            // A bare effect owns the separating space, as after an activated cost.
            boolean opensClause = span.start() == 0 && span.role() == SpanRole.EFFECT;
            int start = opensClause ? costEnd : base + span.start();
            result.add(new RoleSpans.Span(start, base + span.end(), span.role()));
        }
        return result;
    }
}
