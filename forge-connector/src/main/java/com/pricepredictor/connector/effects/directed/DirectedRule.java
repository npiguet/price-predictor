package com.pricepredictor.connector.effects.directed;

import forge.game.Game;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * A rule enabled for directed forks (D1): its board precondition and, per
 * contrasting variant of its scenario, the application that makes that
 * variant out of the bound board.
 *
 * <p>The scenario is named because a rule may have several, and two of them
 * may share a variant label: the label is the scenario's, and the registry
 * contract test checks it against that scenario alone.
 *
 * @param rule              the rule id, as {@code --directed-fork-rules} and
 *                          the records' {@code aimed_at} spell it
 * @param scenario          the scenario id whose base variant the
 *                          precondition waits for, e.g. {@code 704.5f-1}
 * @param boardPrecondition judges the live board against the base variant
 * @param variants          variant label to its application, in the order a
 *                          fork draws among them
 */
public record DirectedRule(
        String rule, String scenario, BoardPrecondition boardPrecondition,
        Map<String, Application> variants) {

    public DirectedRule {
        Objects.requireNonNull(rule, "rule");
        Objects.requireNonNull(scenario, "scenario");
        Objects.requireNonNull(boardPrecondition, "boardPrecondition");
        if (variants.isEmpty()) {
            throw new IllegalArgumentException(rule + " registers no variant");
        }
        variants = Collections.unmodifiableMap(new LinkedHashMap<>(variants));
    }

    /** A rule with one variant, the common case. */
    public static DirectedRule of(
            String rule, String scenario, BoardPrecondition precondition,
            String variant, Application application) {
        return new DirectedRule(rule, scenario, precondition, Map.of(variant, application));
    }

    /**
     * The objects the variants act on when the live board meets the base
     * variant, else empty.
     *
     * <p>A precondition that throws reads as not met: it is asked of every
     * board a game reaches, and a card it did not foresee is no reason to
     * stop the live game.
     */
    public Optional<Binding> precondition(Game live) {
        try {
            return Objects.requireNonNullElse(boardPrecondition.bind(live), Optional.empty());
        } catch (RuntimeException e) {
            return Optional.empty();
        }
    }
}
