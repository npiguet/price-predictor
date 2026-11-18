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
 * {@code cost} throughout. A planeswalker's loyalty cost ends at the first colon and the rest is
 * its effect, a delayed trigger included; a level's cost ends there too and
 * what follows reads as an ordinary clause. A chapter is one effect.
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
        return switch (line.type()) {
            case PLANESWALKER -> loyaltyLine(line);
            case LEVEL -> costThenClause(line);
            case CHAPTER -> allEffect(line);
            case TRIGGERED -> RoleSpans.ofTriggered(line.prose());
            default -> RoleSpans.of(line.prose(), line.type() == AbilityType.ACTIVATED);
        };
    }

    /**
     * A loyalty ability: its cost, then an effect. A "whenever" after the cost
     * is the delayed trigger the ability creates, part of its effect.
     */
    private static List<RoleSpans.Span> loyaltyLine(Line line) {
        int costEnd = RoleSpans.costEnd(line.prose());
        if (costEnd <= 0) return allEffect(line);
        return withCostSentence(new Line(line.prose(), line.type(), costEnd));
    }

    /** A chapter's numerals and what it does, a delayed trigger included, as one effect. */
    private static List<RoleSpans.Span> allEffect(Line line) {
        List<RoleSpans.Span> result = new ArrayList<>();
        result.add(new RoleSpans.Span(0, line.length(), SpanRole.EFFECT));
        result.addAll(RoleSpans.targets(line.prose()));
        return result;
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
     * A level cost ({@code {1}{B}:}), then the clause after it. A level line
     * may have no cost at all (level 1), so its cost must open with a mana symbol.
     */
    private static List<RoleSpans.Span> costThenClause(Line line) {
        int costEnd = RoleSpans.costEnd(line.prose());
        if (costEnd <= 0 || !line.prose().startsWith("{")) {
            return RoleSpans.of(line.prose(), false);
        }

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
            // The clause's first span owns the separating space, as after an activated cost.
            boolean opensClause = span.start() == 0 && span.role() != SpanRole.TARGET_SPEC;
            int start = opensClause ? costEnd : base + span.start();
            result.add(new RoleSpans.Span(start, base + span.end(), span.role()));
        }
        return result;
    }
}
