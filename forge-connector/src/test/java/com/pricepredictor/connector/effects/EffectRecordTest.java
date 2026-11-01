package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.directed.DirectedTag;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The record envelope, the event shape, the attribution mode and JSON
 * escaping: the parts of a record that need no game.
 */
class EffectRecordTest {

    @TempDir
    Path tempDir;

    // ── the record envelope ─────────────────────────────────────────────

    private EffectRecord record(RecordKind kind) {
        return new EffectRecord(
                "run.0.1", "run", "2026-09-07T00:00:00Z", "run.0.0", kind,
                AttributionMode.DEGRADED);
    }

    private EffectRecord resolution(Moment moment) {
        return EffectRecord.resolution(
                "run.0.1", "run", "2026-09-07T00:00:00Z", "run.0.0", moment,
                AttributionMode.DEGRADED);
    }

    private EffectRecord playability(PlayabilitySubkind subkind) {
        return EffectRecord.playability(
                "run.0.1", "run", "2026-09-07T00:00:00Z", "run.0.0", subkind,
                AttributionMode.DEGRADED);
    }

    @Test
    void anEnvelopeCarriesEveryFrozenField() {
        String json = record(RecordKind.COMBAT).actor("P0").toJson();
        for (String key : List.of(
                "record_id", "generation", "run_id", "timestamp", "game_id", "kind", "moment",
                "subkind", "link_id", "mirror_of", "variant_of", "mode",
                "interventional", "fork", "synthetic", "random_seat", "actor_player",
                "ability", "state", "payload")) {
            assertTrue(json.contains("\"" + key + "\":"), key + " missing from " + json);
        }
    }

    /**
     * {@code generation} is the first member after {@code record_id} on every
     * record (FR-018): its value is what names a gen-3 shard to a reader.
     */
    @Test
    void generationFollowsTheRecordIdOnEveryRecord() {
        for (EffectRecord r : List.of(record(RecordKind.COMBAT), resolution(Moment.RESOLUTION),
                playability(PlayabilitySubkind.DECISION))) {
            assertTrue(r.toJson().startsWith("{\"record_id\":\"run.0.1\",\"generation\":3,"),
                    r.toJson());
        }
    }

    /**
     * {@code aimed_at} and {@code directed} are written only where set, after
     * the gen-2 flags; an empty {@code aimed_at} is written, because a coverage
     * run aimed at no rule says so (FR-009a).
     */
    @Test
    void aimedAtAndDirectedAreWrittenOnlyWhereSet() {
        String plain = resolution(Moment.RESOLUTION).toJson();
        assertFalse(plain.contains("aimed_at"), plain);
        assertFalse(plain.contains("directed"), plain);

        String coverage = resolution(Moment.RESOLUTION).aimedAt(List.of()).toJson();
        assertTrue(coverage.contains("\"random_seat\":false,\"aimed_at\":[],\"actor_player\":"),
                coverage);

        String fork = record(RecordKind.COMBAT).fork(true).aimedAt(List.of("704.5f"))
                .directed(new DirectedTag("tougher", ForkOperation.PT)).toJson();
        assertTrue(fork.contains(
                "\"aimed_at\":[\"704.5f\"],\"directed\":{\"variant\":\"tougher\",\"operation\":\"pt\"},"),
                fork);
    }

    /**
     * The context stamps the run's {@code aimed_at} on every record it writes
     * (FR-009a): a coverage or variant run's rules, an empty list where it
     * names none, and nothing at all on a run aimed at nothing, which is every
     * {@code match-outcomes} run. A directed fork's record keeps the one rule
     * it forked for.
     */
    @Test
    void theContextStampsTheRunsAimedAtOnEveryRecord() throws Exception {
        assertEquals(List.of("\"aimed_at\":[]", "\"aimed_at\":[]"),
                aimedAtWritten(List.of(), null));
        assertEquals(List.of("\"aimed_at\":[\"704.5f\",\"702.19b\"]",
                        "\"aimed_at\":[\"704.5f\",\"702.19b\"]"),
                aimedAtWritten(List.of("704.5f", "702.19b"), null));
        assertEquals(List.of("", ""), aimedAtWritten(null, null));
        // The directed fork's own rule outranks the run's list, and a
        // match-outcomes run writes it on that record alone.
        assertEquals(List.of("\"aimed_at\":[\"704.5f\",\"702.19b\"]", "\"aimed_at\":[\"704.5f\"]"),
                aimedAtWritten(List.of("704.5f", "702.19b"), "704.5f"));
        assertEquals(List.of("", "\"aimed_at\":[\"704.5f\"]"), aimedAtWritten(null, "704.5f"));
    }

    /**
     * The {@code aimed_at} member of the two records a context over {@code
     * runAimedAt} writes — a resolution half, then a combat record, which is
     * a directed fork's for {@code forkRule} when that is set — or "" for a
     * record without one.
     */
    private List<String> aimedAtWritten(List<String> runAimedAt, String forkRule)
            throws Exception {
        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        try {
            CollectorContext context = new CollectorContext(
                    writer, "run.0-l1.0", AttributionMode.DEGRADED, null, runAimedAt);
            context.emit(context.newResolution(Moment.RESOLUTION));
            EffectRecord combat = context.newRecord(RecordKind.COMBAT);
            if (forkRule != null) {
                combat.fork(true).aimedAt(List.of(forkRule))
                        .directed(new DirectedTag("tougher", ForkOperation.PT));
            }
            context.emit(combat);
        } finally {
            writer.close();
        }
        List<String> written = ShardRun.readShard(writer.path()).stream()
                .map(line -> {
                    int start = line.indexOf("\"aimed_at\":");
                    return start < 0 ? "" : line.substring(start, line.indexOf(']', start) + 1);
                })
                .toList();
        java.nio.file.Files.delete(writer.path());
        return written;
    }

    /** The directed-fork signature: a non-interventional fork naming its rule. */
    @Test
    void aDirectedTagNeedsAnAimedNonInterventionalFork() {
        DirectedTag tag = new DirectedTag("tougher", ForkOperation.PT);
        assertThrows(IllegalStateException.class,
                () -> record(RecordKind.COMBAT).fork(true).directed(tag).toJson());
        assertThrows(IllegalStateException.class,
                () -> record(RecordKind.COMBAT).aimedAt(List.of("704.5f")).directed(tag).toJson());
        assertThrows(IllegalStateException.class,
                () -> resolution(Moment.RESOLUTION).fork(true).interventional(true)
                        .aimedAt(List.of("704.5f")).directed(tag).toJson());
    }

    /**
     * {@code random_seat} follows {@code synthetic} on every record, and
     * {@code what_if} follows it on the two legality subkinds alone
     * (record-schema-delta.md): the order is the contract, and the first
     * field's presence is how a reader names a gen-2 shard.
     */
    @Test
    void randomSeatFollowsSyntheticOnEveryRecordAndWhatIfOnlyWhereSet() {
        String plain = resolution(Moment.RESOLUTION).toJson();
        assertTrue(plain.contains("\"synthetic\":false,\"random_seat\":false,\"actor_player\":"),
                plain);
        assertFalse(plain.contains("what_if"), plain);

        String legality = playability(PlayabilitySubkind.ATTACKERS).randomSeat(true).whatIf(false).actor("P1").toJson();
        assertTrue(legality.contains(
                "\"synthetic\":false,\"random_seat\":true,\"what_if\":false,\"actor_player\":\"P1\""),
                legality);
    }

    /**
     * A resolution half is its moment and a legality verdict its subkind: the
     * two kinds are made with them, and the plain constructor refuses both.
     */
    @Test
    void theQualifiedKindsCarryTheirQualifierAndNoOther() {
        String activation = resolution(Moment.ACTIVATION).toJson();
        assertTrue(activation.contains(
                "\"kind\":\"resolution\",\"moment\":\"activation\",\"subkind\":null"),
                activation);
        String blockers = playability(PlayabilitySubkind.BLOCKERS).toJson();
        assertTrue(blockers.contains(
                "\"kind\":\"playability\",\"moment\":null,\"subkind\":\"blockers\""),
                blockers);
        String combat = record(RecordKind.COMBAT).toJson();
        assertTrue(combat.contains(
                "\"kind\":\"combat\",\"moment\":null,\"subkind\":null"), combat);

        String draw = EffectRecord.turnAction(
                "run.0.1", "run", "2026-09-07T00:00:00Z", "run.0.0", TurnActionSubkind.DRAW,
                AttributionMode.DEGRADED).toJson();
        assertTrue(draw.contains(
                "\"kind\":\"turn_action\",\"moment\":null,\"subkind\":\"draw\""), draw);

        for (RecordKind qualified : List.of(
                RecordKind.RESOLUTION, RecordKind.PLAYABILITY, RecordKind.TURN_ACTION)) {
            assertThrows(IllegalArgumentException.class, () -> record(qualified), qualified.wireName());
        }
    }

    @Test
    void aCombatRecordNamesNoActingLine() {
        String json = record(RecordKind.COMBAT).actor("P0").toJson();
        assertTrue(json.contains("\"ability\":null"), json);
    }

    @Test
    void anOrdinaryObservationCarriesNeitherFlag() {
        String json = resolution(Moment.RESOLUTION).toJson();
        assertTrue(json.contains("\"interventional\":false"));
        assertTrue(json.contains("\"fork\":false"));
        assertTrue(json.contains("\"synthetic\":false"));
    }

    @Test
    void aDamageStepProbeIsAForkThatIntervenedInNothing() {
        String json = record(RecordKind.COMBAT)
                .fork(true).interventional(false).mirrorOf("run.0.9").toJson();
        assertTrue(json.contains("\"fork\":true"));
        assertTrue(json.contains("\"interventional\":false"));
        assertTrue(json.contains("\"mirror_of\":\"run.0.9\""));
    }

    @Test
    void anInterventionalResolutionIsBoth() {
        String json = resolution(Moment.RESOLUTION)
                .interventional(true).fork(true).toJson();
        assertTrue(json.contains("\"interventional\":true"));
        assertTrue(json.contains("\"fork\":true"));
    }

    @Test
    void aPartnerlessHalfCarriesNoLinkId() {
        String json = resolution(Moment.ACTIVATION)
                // No ability: a countered spell's costs are still readable in
                // a real game, but the record's shape does not depend on them.
                .payload(RecordPayloads.costPayload(
                        PaidCosts.EMPTY, ActivationOutcome.COUNTERED))
                .toJson();
        assertTrue(json.contains("\"link_id\":null"), json);
        assertTrue(json.contains("\"outcome\":\"countered\""), json);
    }

    @Test
    void theModeIsStampedOnEveryRecord() {
        assertTrue(record(RecordKind.COMBAT).toJson()
                .contains("\"mode\":\"degraded\""));
        assertTrue(new EffectRecord(
                "r", "r", "t", "g", RecordKind.COMBAT,
                AttributionMode.PATCHED).toJson().contains("\"mode\":\"patched\""));
    }

    /**
     * A trigger record names its acting line, and says so in the envelope,
     * with no reason beside it.
     *
     * <p>Guards the wire shape the collector fills: the field rule exempts
     * only {@code combat} and {@code playability} from naming a line, and a
     * trigger is a single acting line.
     */
    @Test
    void aTriggerRecordNamesItsActingLine() {
        String json = record(RecordKind.TRIGGER)
                .ability(new ProvenanceKey.Resolved(new ProvenanceKey(
                        "cardsfolder/g/grave_titan.txt", 0,
                        TraitKind.TRIGGER, 0), null))
                .toJson();

        assertFalse(json.contains("\"ability\":null"), json);
        assertTrue(json.contains("\"trait_kind\":\"trigger\""), json);
        assertTrue(json.contains("cardsfolder/g/grave_titan.txt"), json);
        assertTrue(json.contains("\"ability_unresolved\":null"), json);
    }

    /** And a rewrite record names the replacement effect that rewrote. */
    @Test
    void aRewriteRecordNamesItsReplacement() {
        String json = record(RecordKind.REWRITE)
                .ability(new ProvenanceKey.Resolved(new ProvenanceKey(
                        "cardsfolder/p/paralyze.txt", 0,
                        TraitKind.REPLACEMENT, 0), null))
                .toJson();

        assertTrue(json.contains("\"trait_kind\":\"replacement\""), json);
    }

    /**
     * A line that could not be attributed renders empty rather than absent,
     * and says why it is empty.
     *
     * <p>Empty and absent are different answers -- "a line acted and we could
     * not name it" against "no line acts here", as in a combat record -- and
     * a reader that folds them together cannot measure the first. Nor could
     * it tell "no printed line exists, correctly" from "the resolver
     * regressed" without the reason. The two fields are set from one answer
     * so they cannot be written out of step: a reason beside a named line is
     * a contradiction the reader refuses outright.
     */
    @Test
    void anUnattributableLineIsEmptyRatherThanNullAndSaysWhy() {
        String json = record(RecordKind.TRIGGER)
                .ability(new ProvenanceKey.Resolved(
                        null, UnresolvedReason.ENGINE_EFFECT))
                .toJson();

        assertTrue(json.contains("\"ability\":[]"), json);
        assertTrue(json.contains("\"ability_unresolved\":\"engine_effect\""), json);
    }

    /** A caller that looked and got nothing back still owes a reason. */
    @Test
    void aLookupThatCameBackWithNothingAtAllStillNamesAReason() {
        String json = record(RecordKind.TRIGGER)
                .ability(null)
                .toJson();

        assertTrue(json.contains("\"ability_unresolved\":\"unknown_kind\""), json);
    }

    // ── combat payloads ─────────────────────────────────────────────────

    /** A combat with nobody in it still carries all three combat fields. */
    @Test
    void anEmptyCombatRendersEveryField() {
        assertEquals(
                "{\"attackers\":[],\"blocks\":{},\"assignment_choices\":{},\"events\":[]}",
                RecordPayloads.combatPayload(CombatShape.of(null), List.of()));
    }

    /** A probe branch's perturbation follows the fields it shares with the record. */
    @Test
    void aProbePayloadNamesWhatItPerturbedAfterTheCombat() {
        assertEquals(
                "{\"attackers\":[],\"blocks\":{},\"assignment_choices\":{},\"events\":[],"
                        + "\"probed_keyword\":\"trample\",\"probed_entity\":null}",
                RecordPayloads.probePayload(
                        CombatShape.of(null), List.of(), "trample", null));
    }

    // ── events ──────────────────────────────────────────────────────────

    @Test
    void anEventCarriesItsTypeSubjectsAndParams() {
        String json = new EffectEvent(EventType.DAMAGE_DEALT)
                .subject("E12").param("amount", 3).param("combat", true)
                .toJson();
        assertTrue(json.contains("\"type\":\"damage_dealt\""), json);
        assertTrue(json.contains("\"subjects\":[\"E12\"]"), json);
        assertTrue(json.contains("\"amount\":3"), json);
        assertTrue(json.contains("\"combat\":true"), json);
    }

    @Test
    void aNullParamIsOmittedRatherThanWrittenAsNull() {
        String json = new EffectEvent(EventType.ZONE_CHANGE)
                .subject("E1").param("to_zone", null).toJson();
        assertTrue(json.contains("\"params\":{}"), json);
    }

    @Test
    void aContinuousOutcomeCarriesItsDuration() {
        String json = new EffectEvent(EventType.PT_CHANGE)
                .subject("E1").duration(EventDuration.END_OF_TURN).toJson();
        assertTrue(json.contains("\"duration\":\"end_of_turn\""), json);
    }

    @Test
    void anEventAttributesToItsSubAbilityLinkOrToTheRootLine() {
        assertTrue(new EffectEvent(EventType.DAMAGE_DEALT).toJson()
                .contains("\"attributed_to\":null"));
        assertTrue(new EffectEvent(EventType.DAMAGE_DEALT).attributedTo("0")
                .toJson().contains("\"attributed_to\":\"0\""));
    }

    @Test
    void aMapParamRendersAsAnObject() {
        String json = new EffectEvent(EventType.MANA_PRODUCED)
                .subject("P0")
                .param("mana_by_color", java.util.Map.of("R", 2))
                .toJson();
        assertTrue(json.contains("\"mana_by_color\":{\"R\":2}"), json);
    }

    // ── attribution mode ────────────────────────────────────────────────

    /**
     * Detection agrees with what the checkout actually offers.
     *
     * <p>Asserted as an invariant rather than as a fixed answer, because the
     * sibling checkout is patched or not independently of this repository: a
     * test that hard-codes DEGRADED passes only until someone applies the
     * patches, and then fails without anything being wrong.
     */
    @Test
    void theDetectedModeMatchesTheHooksPresence() {
        boolean hookPresent = PatchHooks
                .find(PatchHooks.ABILITY_UTILS, "getEffectRecordSubAbility")
                .isPresent();
        assertEquals(
                hookPresent ? AttributionMode.PATCHED : AttributionMode.DEGRADED,
                AttributionMode.detect());
    }
}
