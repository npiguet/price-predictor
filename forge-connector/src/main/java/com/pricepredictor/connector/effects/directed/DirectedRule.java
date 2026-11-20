package com.pricepredictor.connector.effects.directed;

import forge.game.Game;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * A registration for directed forks (D1): the rules it forks for, its board
 * precondition and, per contrasting variant, the application that makes that
 * variant out of the bound board.
 *
 * <p>Rules that share a board, a binding and an edit are one registration
 * ({@link #alsoFor}): the scheduler draws, budgets and forks it once, and the
 * fork's records name every one of its rules in {@code aimed_at}. Forked once
 * per rule instead, the same board would be forked into near-identical
 * records under each id, each fork spending a slot of the game's budget.
 *
 * <p>Each rule names its scenario, because a rule may have several and two
 * of them may share a variant label: the label is the scenario's, and the
 * registry contract test checks it against that scenario alone.
 *
 * @param scenarios         each rule id, as {@code --directed-fork-rules} and
 *                          the records' {@code aimed_at} spell it, to the
 *                          scenario whose base variant the precondition
 *                          waits for, e.g. {@code 704.5f} to {@code 704.5f-1};
 *                          in the order {@code aimed_at} lists them
 * @param boardPrecondition judges the live board against the base variant
 * @param variants          variant label to its application, in the order a
 *                          fork draws among them
 */
public record DirectedRule(
        Map<String, String> scenarios, BoardPrecondition boardPrecondition,
        Map<String, Application> variants) {

    public DirectedRule {
        if (scenarios.isEmpty()) {
            throw new IllegalArgumentException("a registration names no rule");
        }
        Objects.requireNonNull(boardPrecondition, "boardPrecondition");
        scenarios = Collections.unmodifiableMap(new LinkedHashMap<>(scenarios));
        if (variants.isEmpty()) {
            throw new IllegalArgumentException(String.join("+", scenarios.keySet()) + " registers no variant");
        }
        variants = Collections.unmodifiableMap(new LinkedHashMap<>(variants));
    }

    /** One variant of a registration: its label and the application that makes it. */
    public record Variant(String label, Application application) {
    }

    /** A registration of one rule with one variant, the common case. */
    public static DirectedRule of(
            String rule, String scenario, BoardPrecondition precondition, Variant variant) {
        return new DirectedRule(Map.of(Objects.requireNonNull(rule, "rule"),
                Objects.requireNonNull(scenario, "scenario")), precondition,
                Map.of(variant.label(), variant.application()));
    }

    /**
     * This registration for {@code rule} too, on {@code scenario}: a rule
     * whose board, binding and edit are this registration's.
     *
     * @throws IllegalArgumentException when the registration already names {@code rule}
     */
    public DirectedRule alsoFor(String rule, String scenario) {
        Map<String, String> more = new LinkedHashMap<>(scenarios);
        if (more.putIfAbsent(Objects.requireNonNull(rule, "rule"), Objects.requireNonNull(scenario, "scenario"))
                != null) {
            throw new IllegalArgumentException(rule + " is named twice");
        }
        return new DirectedRule(more, boardPrecondition, variants);
    }

    /** The rule ids, in the order the fork's {@code aimed_at} lists them. */
    public List<String> rules() {
        return List.copyOf(scenarios.keySet());
    }

    /** The scenario {@code rule} waits for, null for a rule the registration does not name. */
    public String scenario(String rule) {
        return scenarios.get(rule);
    }

    /** The rule ids joined by {@code +}, one word, as messages name the registration. */
    public String name() {
        return String.join("+", scenarios.keySet());
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
        return precondition(live, failure -> { });
    }

    /**
     * {@link #precondition(Game)}, {@code threw} told of each exception the
     * precondition throws before it reads as not met, so the throw is
     * counted under its rules rather than lost.
     */
    public Optional<Binding> precondition(Game live, Consumer<RuntimeException> threw) {
        try {
            return Objects.requireNonNullElse(boardPrecondition.bind(live), Optional.empty());
        } catch (RuntimeException e) {
            threw.accept(e);
            return Optional.empty();
        }
    }
}
