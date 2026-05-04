package com.pricepredictor.connector;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration test asserting that {@link MatchGenerator#generateMatch}
 * returns one {@link CardsPlayedRow} per played game, populated with
 * metadata that aligns with the parent {@link MatchResult}.
 */
@Tag("integration")
@ExtendWith(ForgeExtension.class)
class MatchGeneratorCardsPlayedIntegrationTest {

    private static final String TEST_RUN_ID = "test-run-id";

    private static final Set<String> BASIC_LANDS = Set.of(
            "Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
            "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
            "Snow-Covered Mountain", "Snow-Covered Forest");

    @Test
    void returnsOneRowPerPlayedGame() {
        MatchGenerator generator = MatchGenerator.withDefaultBuilders(TEST_RUN_ID);

        MatchGenerationResult result = generator.generateMatch();

        assertNotNull(result.matchResult());
        assertNotNull(result.cardsPlayedRows());
        assertEquals(result.matchResult().games().length(),
                result.cardsPlayedRows().size(),
                "row count must equal played-game count");
    }

    @Test
    void rowsCarryParentMetadata() {
        MatchGenerator generator = MatchGenerator.withDefaultBuilders(TEST_RUN_ID);

        MatchGenerationResult result = generator.generateMatch();
        MatchResult parent = result.matchResult();

        for (int i = 0; i < result.cardsPlayedRows().size(); i++) {
            CardsPlayedRow row = result.cardsPlayedRows().get(i);
            assertEquals(parent.runId(), row.runId());
            assertEquals(parent.setCode(), row.setCode());
            assertEquals(parent.methodA(), row.methodA());
            assertEquals(parent.methodB(), row.methodB());
            assertEquals(parent.games().charAt(i), row.winner(),
                    "winner at index " + i + " must align with parent games string");
            assertEquals(parent.play().charAt(i), row.starter(),
                    "starter at index " + i + " must align with parent play string");
        }
    }

    @Test
    void basicLandsAbsentFromAllListColumns() {
        MatchGenerator generator = MatchGenerator.withDefaultBuilders(TEST_RUN_ID);
        MatchGenerationResult result = generator.generateMatch();

        for (CardsPlayedRow row : result.cardsPlayedRows()) {
            assertNoBasicsIn(row.cardsPlayedA(), "cards_played_A");
            assertNoBasicsIn(row.cardsPlayedB(), "cards_played_B");
            assertNoBasicsIn(row.cardsNotPlayedA(), "cards_not_played_A");
            assertNoBasicsIn(row.cardsNotPlayedB(), "cards_not_played_B");
        }
    }

    @Test
    void unionEqualsDistinctNonBasicsAndIsDisjoint() {
        MatchGenerator generator = MatchGenerator.withDefaultBuilders(TEST_RUN_ID);
        MatchGenerationResult result = generator.generateMatch();
        MatchResult parent = result.matchResult();

        Set<String> distinctA = distinctNonBasic(parent.deckA());
        Set<String> distinctB = distinctNonBasic(parent.deckB());

        for (CardsPlayedRow row : result.cardsPlayedRows()) {
            assertNoDuplicates(row.cardsPlayedA(), "cards_played_A");
            assertNoDuplicates(row.cardsPlayedB(), "cards_played_B");
            assertNoDuplicates(row.cardsNotPlayedA(), "cards_not_played_A");
            assertNoDuplicates(row.cardsNotPlayedB(), "cards_not_played_B");

            assertDisjoint(row.cardsPlayedA(), row.cardsNotPlayedA(),
                    "cards_played_A and cards_not_played_A must be disjoint");
            assertDisjoint(row.cardsPlayedB(), row.cardsNotPlayedB(),
                    "cards_played_B and cards_not_played_B must be disjoint");

            assertEquals(distinctA, union(row.cardsPlayedA(), row.cardsNotPlayedA()),
                    "deck A union (distinct, non-basic) must equal deck contents");
            assertEquals(distinctB, union(row.cardsPlayedB(), row.cardsNotPlayedB()),
                    "deck B union (distinct, non-basic) must equal deck contents");
        }
    }

    private static void assertNoBasicsIn(List<String> names, String column) {
        for (String name : names) {
            assertFalse(BASIC_LANDS.contains(name),
                    column + " must not contain basic land: " + name);
        }
    }

    private static void assertNoDuplicates(List<String> names, String column) {
        Set<String> seen = new HashSet<>();
        for (String name : names) {
            assertTrue(seen.add(name),
                    column + " must not contain duplicates; offender: " + name);
        }
    }

    private static void assertDisjoint(List<String> a, List<String> b, String message) {
        Set<String> seen = new HashSet<>(a);
        for (String name : b) {
            assertFalse(seen.contains(name), message + "; offender: " + name);
        }
    }

    private static Set<String> distinctNonBasic(List<String> deck) {
        Set<String> distinct = new LinkedHashSet<>();
        for (String name : deck) {
            if (!BASIC_LANDS.contains(name)) {
                distinct.add(name);
            }
        }
        return distinct;
    }

    private static Set<String> union(List<String> a, List<String> b) {
        Set<String> out = new LinkedHashSet<>(a);
        out.addAll(b);
        return out;
    }
}
