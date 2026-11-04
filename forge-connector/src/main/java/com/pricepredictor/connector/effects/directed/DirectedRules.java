package com.pricepredictor.connector.effects.directed;

import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * The rules a directed fork can be taken for, by rule id (D1).
 *
 * <p>The registry lives in Java because a rule's precondition and its edits
 * are Java code; the supervisor learns which rules it holds by running
 * {@code DirectedRulesMain} once at startup (FR-014). A rule listed in
 * {@code --directed-fork-rules} and absent here exits the run before any
 * game.
 *
 * <p>Not {@code VariantRegistry}: that holds a set of card names, where a rule
 * id here maps to behaviour.
 */
public final class DirectedRules {

    /**
     * Every registered rule, in registration order. The rules themselves are
     * registered by T094-T096.
     */
    private static final DirectedRules REGISTERED = of();

    private final Map<String, DirectedRule> rules;

    private DirectedRules(Map<String, DirectedRule> rules) {
        this.rules = Collections.unmodifiableMap(rules);
    }

    /** The rules the worker forks for. */
    public static DirectedRules registered() {
        return REGISTERED;
    }

    /**
     * A registry of {@code rules}, for the registered set and for a test.
     *
     * @throws IllegalArgumentException when two share a rule id
     */
    public static DirectedRules of(DirectedRule... rules) {
        Map<String, DirectedRule> byId = new LinkedHashMap<>();
        for (DirectedRule rule : rules) {
            if (byId.putIfAbsent(rule.rule(), rule) != null) {
                throw new IllegalArgumentException(rule.rule() + " is registered twice");
            }
        }
        return new DirectedRules(byId);
    }

    /** The rule registered under {@code id}, if any. */
    public Optional<DirectedRule> find(String id) {
        return Optional.ofNullable(rules.get(id));
    }

    /** The registered rule ids, in registration order. */
    public List<String> ids() {
        return List.copyOf(rules.keySet());
    }

    /** The registered rules, in registration order. */
    public Collection<DirectedRule> rules() {
        return rules.values();
    }
}
