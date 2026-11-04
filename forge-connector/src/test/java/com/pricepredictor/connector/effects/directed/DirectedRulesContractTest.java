package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.effects.directed.operations.TapOperation;
import org.eclipse.jetty.util.ajax.JSON;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The registry's contract with the catalogue (D1): every registered rule
 * names a scenario that lists it, and every registered variant is a
 * contrasting variant of that scenario classified into the same operation.
 *
 * <p>Read from {@code rules/scenarios/*.json} rather than restated, so a
 * reclassified variant fails here before a directed fork labels its records
 * with an operation the catalogue no longer gives it.
 */
class DirectedRulesContractTest {

    private static final Path SCENARIOS = Path.of("..", "rules", "scenarios");

    /** One scenario as the contract reads it: the rules it lists and each variant's operation. */
    private record Scenario(List<String> rules, Map<String, Optional<String>> operations) {
    }

    /** Every scenario of the catalogue, by id. */
    private static Map<String, Scenario> scenarios() throws IOException {
        Map<String, Scenario> byId = new HashMap<>();
        try (Stream<Path> files = Files.list(SCENARIOS)) {
            for (Path file : files.filter(f -> f.toString().endsWith(".json")).sorted().toList()) {
                Map<?, ?> doc = (Map<?, ?>) JSON.parse(Files.readString(file, StandardCharsets.UTF_8));
                for (Object entry : list(doc.get("scenarios"))) {
                    Map<?, ?> scenario = (Map<?, ?>) entry;
                    List<String> rules = list(scenario.get("rules")).stream().map(String::valueOf).toList();
                    Map<String, Optional<String>> operations = new HashMap<>();
                    for (Object v : list(scenario.get("variants"))) {
                        Map<?, ?> variant = (Map<?, ?>) v;
                        // An unclassifiable variant's operation is an object, the
                        // base's absent: neither is an operation a fork can apply.
                        Object operation = variant.get("operation");
                        operations.put(String.valueOf(variant.get("label")),
                                operation instanceof String name ? Optional.of(name) : Optional.empty());
                    }
                    byId.put(String.valueOf(scenario.get("id")), new Scenario(rules, operations));
                }
            }
        }
        return byId;
    }

    private static List<Object> list(Object value) {
        if (value instanceof Object[] array) {
            return List.of(array);
        }
        if (value instanceof List<?> items) {
            return new ArrayList<>(items);
        }
        return List.of();
    }

    /** What is wrong with {@code registry} against the catalogue, one line each; empty when nothing is. */
    static List<String> disagreements(DirectedRules registry, Map<String, Scenario> scenarios) {
        List<String> wrong = new ArrayList<>();
        for (DirectedRule rule : registry.rules()) {
            Scenario scenario = scenarios.get(rule.scenario());
            if (scenario == null) {
                wrong.add(rule.rule() + ": no scenario " + rule.scenario());
                continue;
            }
            if (!scenario.rules().contains(rule.rule())) {
                wrong.add(rule.rule() + ": scenario " + rule.scenario() + " does not list the rule");
            }
            rule.variants().forEach((label, application) -> {
                Optional<String> classified = scenario.operations().get(label);
                if (classified == null) {
                    wrong.add(rule.rule() + ": scenario " + rule.scenario() + " has no variant " + label);
                } else if (!Objects.equals(classified, Optional.of(application.operation().wireName()))) {
                    wrong.add(rule.rule() + ": variant " + label + " is classified "
                            + classified.orElse("as no operation") + ", registered "
                            + application.operation().wireName());
                }
            });
        }
        return wrong;
    }

    private static DirectedRule rule(String rule, String scenario, String variant, ForkOperation operation) {
        Application application = new Application(operation, binding -> new TapOperation(binding.card("x"), true));
        return DirectedRule.of(rule, scenario, live -> Optional.empty(), variant, application);
    }

    @Test
    void everyRegisteredRuleAgreesWithItsScenario() throws IOException {
        assertEquals(List.of(), disagreements(DirectedRules.registered(), scenarios()));
    }

    /** 704.5f-1's {@code counters} variant is classified {@code counters}. */
    @Test
    void aVariantClassifiedAsRegisteredAgrees() throws IOException {
        assertEquals(List.of(), disagreements(
                DirectedRules.of(rule("704.5f", "704.5f-1", "counters", ForkOperation.COUNTERS)), scenarios()));
    }

    @Test
    void aMissingVariantLabelFails() throws IOException {
        List<String> wrong = disagreements(
                DirectedRules.of(rule("704.5f", "704.5f-1", "no-such-variant", ForkOperation.PT)), scenarios());
        assertEquals(1, wrong.size(), wrong.toString());
        assertTrue(wrong.get(0).contains("has no variant no-such-variant"), wrong.get(0));
    }

    @Test
    void aDifferentOperationFails() throws IOException {
        List<String> wrong = disagreements(
                DirectedRules.of(rule("704.5f", "704.5f-1", "counters", ForkOperation.PT)), scenarios());
        assertEquals(1, wrong.size(), wrong.toString());
        assertTrue(wrong.get(0).contains("classified counters, registered pt"), wrong.get(0));
    }

    /** The base variant is the board a fork waits for, never one it makes. */
    @Test
    void theBaseVariantIsNoRegistrableVariant() throws IOException {
        List<String> wrong = disagreements(
                DirectedRules.of(rule("704.5f", "704.5f-1", "dies", ForkOperation.COUNTERS)), scenarios());
        assertEquals(1, wrong.size(), wrong.toString());
        assertTrue(wrong.get(0).contains("as no operation"), wrong.get(0));
    }

    @Test
    void aScenarioNotListingTheRuleFails() throws IOException {
        List<String> wrong = disagreements(
                DirectedRules.of(rule("702.19b", "704.5f-1", "counters", ForkOperation.COUNTERS)), scenarios());
        assertEquals(List.of("702.19b: scenario 704.5f-1 does not list the rule"), wrong);
    }

    @Test
    void anUnknownScenarioFails() throws IOException {
        assertEquals(List.of("704.5f: no scenario 704.5f-99"), disagreements(
                DirectedRules.of(rule("704.5f", "704.5f-99", "counters", ForkOperation.COUNTERS)), scenarios()));
    }

    @Test
    void aRuleIdIsRegisteredOnce() {
        DirectedRule once = rule("704.5f", "704.5f-1", "counters", ForkOperation.COUNTERS);
        assertThrows(IllegalArgumentException.class, () -> DirectedRules.of(once, once));
    }
}
