package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.directed.operations.TapOperation;
import forge.model.FModel;
import org.eclipse.jetty.util.ajax.JSON;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
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
    private record Scenario(List<String> rules, Map<String, Optional<String>> operations, List<String> cards) {
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
                    List<String> cards = list(scenario.get("cards")).stream().map(String::valueOf).toList();
                    byId.put(String.valueOf(scenario.get("id")), new Scenario(rules, operations, cards));
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

    /** The list the registry's pairs are committed in, which the Python collector reads for each rule's sides. */
    private static final Path VARIANTS = Path.of("..", "rules", "directed-fork-variants.tsv");

    /** Every registered (rule id, variant label) pair as a list line, sorted. */
    static List<String> registeredPairs(DirectedRules registry) {
        return registry.rules().stream()
                .flatMap(rule -> rule.variants().keySet().stream().map(label -> rule.rule() + "	" + label))
                .sorted().toList();
    }

    /** How {@code listed} differs from {@code registered}, one line each; empty when they are equal. */
    static List<String> listDifferences(List<String> registered, List<String> listed) {
        List<String> wrong = new ArrayList<>();
        registered.stream().filter(pair -> !listed.contains(pair))
                .forEach(pair -> wrong.add("registered, not listed: " + pair.replace('	', ' ')));
        listed.stream().filter(pair -> !registered.contains(pair))
                .forEach(pair -> wrong.add("listed, not registered: " + pair.replace('	', ' ')));
        return wrong;
    }

    /**
     * {@code rules/directed-fork-variants.tsv} is the registered pairs, so
     * Python, which cannot see the registry, counts a rule's sides over the
     * variants a fork can make.
     */
    @Test
    void theVariantsListIsTheRegistry() throws IOException {
        List<String> lines = Files.readAllLines(VARIANTS, StandardCharsets.UTF_8);
        assertEquals("rule	variant", lines.get(0));
        List<String> listed = lines.subList(1, lines.size());
        assertEquals(List.of(), listDifferences(registeredPairs(DirectedRules.registered()), listed));
        assertEquals(listed.stream().sorted().toList(), listed, "the list is not sorted");
    }

    @Test
    void aVariantMissingFromTheListIsNamed() {
        assertEquals(List.of("registered, not listed: 704.5f counters", "listed, not registered: 704.5f gone"),
                listDifferences(List.of("704.5f	counters"), List.of("704.5f	gone")));
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

    /**
     * Every rule {@code rules/directed-fork-rules.txt} lists has a
     * registration, which the supervisor's startup check otherwise finds only
     * by running {@code DirectedRulesMain}.
     */
    @Test
    void everyListedRuleIsRegistered() throws IOException {
        List<String> listed = Files.readAllLines(Path.of("..", "rules", "directed-fork-rules.txt"),
                StandardCharsets.UTF_8).stream().map(String::trim).filter(line -> !line.isEmpty()).toList();
        List<String> missing = listed.stream()
                .filter(rule -> DirectedRules.registered().find(rule).isEmpty()).toList();
        assertEquals(List.of(), missing);
    }

    /**
     * TC3: the registry is tied to its scenarios by more than ids and labels.
     * A card a rule class names (the Hill Giant a variant puts in a
     * creature's place, the Lightning Bolt it puts in a library) is one of its
     * scenario's {@code cards}: a rule that drifted to another card would
     * fork a board the scenario does not describe.
     */
    @Nested
    @ExtendWith(ForgeExtension.class)
    class TheCardsARuleNames {

        @Test
        void areItsScenariosCards() throws Exception {
            Map<String, Scenario> scenarios = scenarios();
            Path sources = Path.of("src", "main", "java", "com", "pricepredictor", "connector",
                    "effects", "directed", "rules");
            List<String> wrong = new ArrayList<>();
            int named = 0;
            try (Stream<Path> files = Files.list(sources)) {
                for (Path file : files.sorted().toList()) {
                    String simple = file.getFileName().toString().replace(".java", "");
                    Class<?> type = Class.forName("com.pricepredictor.connector.effects.directed.rules." + simple);
                    Field ruleField;
                    try {
                        ruleField = type.getField("RULE");
                    } catch (NoSuchFieldException e) {
                        continue;  // Boards, Melds: shared reads, no rule
                    }
                    DirectedRule rule = DirectedRules.registered().find((String) ruleField.get(null)).orElseThrow();
                    List<String> cards = scenarios.get(rule.scenario()).cards().stream()
                            .map(name -> name.toLowerCase(Locale.ROOT)).toList();
                    for (Field field : type.getDeclaredFields()) {
                        if (field.getType() != String.class || !Modifier.isStatic(field.getModifiers())) {
                            continue;
                        }
                        field.setAccessible(true);
                        String value = (String) field.get(null);
                        if (FModel.getMagicDb().getCommonCards().getCard(value) == null) {
                            continue;
                        }
                        named++;
                        if (!cards.contains(value.toLowerCase(Locale.ROOT))) {
                            wrong.add(simple + "." + field.getName() + " names " + value
                                    + ", not a card of " + rule.scenario());
                        }
                    }
                }
            }
            assertEquals(List.of(), wrong);
            assertTrue(named >= 15, "the rule classes name " + named + " cards");
        }
    }

    @Test
    void aRuleIdIsRegisteredOnce() {
        DirectedRule once = rule("704.5f", "704.5f-1", "counters", ForkOperation.COUNTERS);
        assertThrows(IllegalArgumentException.class, () -> DirectedRules.of(once, once));
    }
}
