package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The closed vocabularies a record is written in, spelled out literally.
 *
 * <p>Each one has a Python twin that refuses a value it does not have, and the
 * corpus it guards is append-only, so a rename on this side alone loses every
 * record collected after it. The literals here are the Python spellings: a
 * change to an enum that does not change this test is a change to the wire.
 */
class WireVocabularyTest {

    private static <E extends Enum<E>> List<String> wireNames(
            E[] values, Function<E, String> wireName) {
        return Arrays.stream(values).map(wireName).toList();
    }

    /** {@code RecordKind} in {@code records.py}. */
    @Test
    void recordKinds() {
        assertEquals(
                List.of("resolution", "rewrite", "continuous", "combat", "trigger", "playability"),
                wireNames(RecordKind.values(), RecordKind::wireName));
    }

    /** {@code Moment} in {@code records.py}. */
    @Test
    void moments() {
        assertEquals(List.of("activation", "resolution"),
                wireNames(Moment.values(), Moment::wireName));
    }

    /** {@code PlayabilitySubkind} in {@code records.py}. */
    @Test
    void playabilitySubkinds() {
        assertEquals(List.of("decision", "attackers", "blockers"),
                wireNames(PlayabilitySubkind.values(), PlayabilitySubkind::wireName));
    }

    /** {@code ResolutionOutcome} in {@code records.py}. */
    @Test
    void activationOutcomes() {
        assertEquals(
                List.of("resolved", "fizzled", "partially_fizzled", "declined", "countered"),
                wireNames(ActivationOutcome.values(), ActivationOutcome::wireName));
    }

    /**
     * A key's {@code trait_kind}: the sidecar and the records join on it, and
     * {@code validate_corpus.py} buckets keys by the {@code keyword} spelling.
     */
    @Test
    void traitKinds() {
        assertEquals(List.of("keyword", "spell", "trigger", "static", "replacement"),
                wireNames(TraitKind.values(), TraitKind::wireName));
    }

    /** {@code ROLES} in {@code provenance.py}, in its order. */
    @Test
    void spanRoles() {
        assertEquals(List.of("cost", "effect", "trigger-condition", "target-spec"),
                wireNames(SpanRole.values(), SpanRole::wireName));
    }

    /** {@code DURATIONS} in {@code effect_head_layout.py}, in its order. */
    @Test
    void eventDurations() {
        assertEquals(List.of("instant", "end_of_turn", "permanent"),
                wireNames(EventDuration.values(), EventDuration::wireName));
    }
}
