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
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
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
        for (Path file : catalogueFiles()) {
            byId.putAll(scenariosIn(file));
        }
        return byId;
    }

    private static List<Path> catalogueFiles() throws IOException {
        try (Stream<Path> files = Files.list(SCENARIOS)) {
            return files.filter(f -> f.toString().endsWith(".json")).sorted().toList();
        }
    }

    /** The scenarios of one catalogue file, by id. */
    private static Map<String, Scenario> scenariosIn(Path file) throws IOException {
        Map<?, ?> doc = (Map<?, ?>) JSON.parse(Files.readString(file, StandardCharsets.UTF_8));
        Map<String, Scenario> byId = new HashMap<>();
        for (Object entry : list(doc.get("scenarios"))) {
            addScenario(byId, (Map<?, ?>) entry);
        }
        return byId;
    }

    private static void addScenario(Map<String, Scenario> byId, Map<?, ?> scenario) {
        byId.put(String.valueOf(scenario.get("id")), scenarioOf(scenario));
    }

    /** One scenario object of a catalogue file, as the contract reads it. */
    private static Scenario scenarioOf(Map<?, ?> scenario) {
        return new Scenario(texts(scenario.get("rules")), operationsOf(scenario), texts(scenario.get("cards")));
    }

    private static List<String> texts(Object value) {
        return list(value).stream().map(String::valueOf).toList();
    }

    /** Each variant's operation, by label. */
    private static Map<String, Optional<String>> operationsOf(Map<?, ?> scenario) {
        Map<String, Optional<String>> operations = new HashMap<>();
        for (Object v : list(scenario.get("variants"))) {
            Map<?, ?> variant = (Map<?, ?>) v;
            operations.put(String.valueOf(variant.get("label")), operationOf(variant));
        }
        return operations;
    }

    /**
     * The variant's operation name. An unclassifiable variant's operation is an
     * object, the base's absent: neither is an operation a fork can apply.
     */
    private static Optional<String> operationOf(Map<?, ?> variant) {
        return variant.get("operation") instanceof String name ? Optional.of(name) : Optional.empty();
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
            for (String id : rule.rules()) {
                wrong.addAll(disagreements(id, rule, scenarios.get(rule.scenario(id))));
            }
        }
        return wrong;
    }

    /** What is wrong with the registration {@code rule} for its rule {@code id} on {@code scenario}. */
    private static List<String> disagreements(String id, DirectedRule rule, Scenario scenario) {
        String named = rule.scenario(id);
        if (scenario == null) {
            return List.of(id + ": no scenario " + named);
        }
        List<String> wrong = new ArrayList<>();
        if (!scenario.rules().contains(id)) {
            wrong.add(id + ": scenario " + named + " does not list the rule");
        }
        rule.variants().forEach((label, application) -> {
            Optional<String> classified = scenario.operations().get(label);
            if (classified == null) {
                wrong.add(id + ": scenario " + named + " has no variant " + label);
            } else if (!Objects.equals(classified, Optional.of(application.operation().wireName()))) {
                wrong.add(id + ": variant " + label + " is classified "
                        + classified.orElse("as no operation") + ", registered "
                        + application.operation().wireName());
            }
        });
        return wrong;
    }

    /** The list the registry's pairs are committed in, which the Python collector reads for each rule's sides. */
    private static final Path VARIANTS = Path.of("..", "rules", "directed-fork-variants.tsv");

    /**
     * Every registered (rule id, variant label) pair as a list line, sorted:
     * a registration of several rules lists its variants under each.
     */
    static List<String> registeredPairs(DirectedRules registry) {
        return registry.rules().stream()
                .flatMap(rule -> rule.rules().stream().flatMap(id ->
                        rule.variants().keySet().stream().map(label -> id + "	" + label)))
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
        return DirectedRule.of(rule, scenario, live -> Optional.empty(), new DirectedRule.Variant(variant, application));
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
     * The rules {@code rules/directed-fork-rules.txt} lists are the registered
     * rules: a listed rule without a registration the supervisor's startup
     * check otherwise finds only by running {@code DirectedRulesMain}, and a
     * registration no run lists is code no run exercises.
     */
    @Test
    void theListedRulesAreTheRegisteredRules() throws IOException {
        List<String> listed = Files.readAllLines(Path.of("..", "rules", "directed-fork-rules.txt"),
                StandardCharsets.UTF_8).stream().map(String::trim).filter(line -> !line.isEmpty()).toList();
        List<String> registered = DirectedRules.registered().rules().stream()
                .flatMap(rule -> rule.rules().stream()).toList();
        assertEquals(List.of(), listed.stream().filter(rule -> !registered.contains(rule)).toList(),
                "listed, not registered");
        assertEquals(List.of(), registered.stream().filter(rule -> !listed.contains(rule)).toList(),
                "registered, not listed");
    }

    /**
     * TC3: the registry is tied to its scenarios by more than ids and labels.
     * A card a rule class names (the Hill Giant a variant puts in a
     * creature's place, the Lightning Bolt it puts in a library) is one of
     * the cards of the scenarios its registrations name: a rule that drifted
     * to another card would fork a board the scenario does not describe. A
     * class's registrations are those of the rule ids its static strings
     * hold, several for a class registering several rules.
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
                    Map<String, String> strings = strings(
                            Class.forName("com.pricepredictor.connector.effects.directed.rules." + simple));
                    Set<String> cards = scenarioCards(strings.values(), scenarios);
                    // Boards, Melds: shared reads, no rule
                    Map<String, String> namedCards = cards.isEmpty() ? Map.of() : cardsOf(strings);
                    named += namedCards.size();
                    namedCards.forEach((field, card) -> {
                        if (!cards.contains(card.toLowerCase(Locale.ROOT))) {
                            wrong.add(simple + "." + field + " names " + card + ", not a card of its scenarios");
                        }
                    });
                }
            }
            assertEquals(List.of(), wrong);
            assertTrue(named >= 15, "the rule classes name " + named + " cards");
        }
    }

    /** Each static string field of {@code type}, by name. */
    private static Map<String, String> strings(Class<?> type) throws IllegalAccessException {
        Map<String, String> found = new LinkedHashMap<>();
        for (Field field : type.getDeclaredFields()) {
            if (field.getType() == String.class && Modifier.isStatic(field.getModifiers())) {
                field.setAccessible(true);
                found.put(field.getName(), (String) field.get(null));
            }
        }
        return found;
    }

    /** Those of {@code strings} the card database knows. */
    private static Map<String, String> cardsOf(Map<String, String> strings) {
        Map<String, String> cards = new LinkedHashMap<>(strings);
        cards.values().removeIf(value -> FModel.getMagicDb().getCommonCards().getCard(value) == null);
        return cards;
    }

    /** The cards, lower-cased, of every scenario of the registrations of the rule ids among {@code strings}. */
    private static Set<String> scenarioCards(Collection<String> strings, Map<String, Scenario> scenarios) {
        return strings.stream()
                .map(DirectedRules.registered()::find)
                .flatMap(Optional::stream)
                .flatMap(rule -> rule.rules().stream().map(rule::scenario))
                .flatMap(scenario -> scenarios.get(scenario).cards().stream())
                .map(name -> name.toLowerCase(Locale.ROOT))
                .collect(Collectors.toSet());
    }

    @Test
    void aRuleIdIsRegisteredOnce() {
        DirectedRule once = rule("704.5f", "704.5f-1", "counters", ForkOperation.COUNTERS);
        assertThrows(IllegalArgumentException.class, () -> DirectedRules.of(once, once));
        assertThrows(IllegalArgumentException.class, () -> DirectedRules.of(
                once, rule("704.5g", "704.5g-1", "counters", ForkOperation.COUNTERS).alsoFor("704.5f", "704.5f-1")));
        assertThrows(IllegalArgumentException.class, () -> once.alsoFor("704.5f", "704.5f-2"));
    }

    // ── one registration of several rules (finding 6.4) ─────────────────

    /** Each rule of a registration is checked against its own scenario. */
    @Test
    void eachRuleOfARegistrationAgreesWithItsOwnScenario() throws IOException {
        DirectedRule shared = rule("704.5f", "704.5f-1", "counters", ForkOperation.COUNTERS)
                .alsoFor("702.19b", "704.5f-1");
        assertEquals(List.of("702.19b: scenario 704.5f-1 does not list the rule"),
                disagreements(DirectedRules.of(shared), scenarios()));
    }

    /** A registration of several rules lists its variants under each, so Python counts each rule's sides. */
    @Test
    void aRegistrationOfSeveralRulesListsEachRulesPairs() {
        DirectedRule shared = rule("704.5f", "704.5f-1", "counters", ForkOperation.COUNTERS)
                .alsoFor("702.19b", "702.19b-1");
        assertEquals(List.of("702.19b	counters", "704.5f	counters"), registeredPairs(DirectedRules.of(shared)));
    }

    /**
     * The three meld rules and Aura swap's two are one registration each:
     * found under any of their rules, every rule printed by
     * {@code DirectedRulesMain}.
     */
    @Test
    void theMeldAndAuraSwapRulesAreOneRegistrationEach() {
        DirectedRules registry = DirectedRules.registered();
        DirectedRule meld = registry.find("712.14c").orElseThrow();
        DirectedRule auraSwap = registry.find("702.65a").orElseThrow();

        assertEquals(List.of("701.42a", "712.4a", "712.14c"), meld.rules());
        assertEquals(meld, registry.find("701.42a").orElseThrow());
        assertEquals(List.of("701.12e", "702.65a"), auraSwap.rules());
        assertTrue(registry.ids().containsAll(List.of("701.42a", "712.4a", "712.14c", "701.12e", "702.65a")));
        assertEquals(registry.ids().size() - 3, registry.rules().size());
    }
}
