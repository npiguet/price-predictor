package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.effects.directed.rules.Aftermath;
import com.pricepredictor.connector.effects.directed.rules.Dethrone;
import com.pricepredictor.connector.effects.directed.rules.Enlist;
import com.pricepredictor.connector.effects.directed.rules.GraveyardReplacement;
import com.pricepredictor.connector.effects.directed.rules.Intimidate;
import com.pricepredictor.connector.effects.directed.rules.LifeExchange;
import com.pricepredictor.connector.effects.directed.rules.SimultaneousDamage;
import com.pricepredictor.connector.effects.directed.rules.StunCounter;
import com.pricepredictor.connector.effects.directed.rules.ToughnessZero;
import com.pricepredictor.connector.effects.directed.rules.VanishingInstances;

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
     * Every registered rule, in the catalogue's rule order: one class per
     * rule in {@code directed/rules/}, each naming its scenario, its board
     * precondition and its variants. Together they cover the ten operations
     * (story 3, acceptance 3).
     */
    private static final DirectedRules REGISTERED = of(
            StunCounter.rule(),
            GraveyardReplacement.rule(),
            LifeExchange.rule(),
            Intimidate.rule(),
            VanishingInstances.rule(),
            Dethrone.rule(),
            Aftermath.rule(),
            Enlist.rule(),
            SimultaneousDamage.rule(),
            ToughnessZero.rule());

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
