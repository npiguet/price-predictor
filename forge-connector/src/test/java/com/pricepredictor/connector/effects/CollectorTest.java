package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The record envelope, the shard writer, and the event shape.
 *
 * <p>These are the parts that need no game. The collectors' behaviour inside a
 * running match — bracket attribution, two records for a first-strike combat, a
 * countered spell's partnerless cost record — is exercised by the integration
 * test, which plays real games; asserting it here would mean mocking Forge's
 * event bus, and a mock that agreed with my reading of the engine would prove
 * only that I read it consistently.
 */
class CollectorTest {

    @TempDir
    Path tempDir;

    // ── the shard writer ────────────────────────────────────────────────

    @Test
    void aShardIsNamedForItsRunAndWorker() {
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run-uuid", 4)) {
            assertEquals(
                    "run-uuid.4.jsonl.gz", writer.path().getFileName().toString());
        }
    }

    @Test
    void recordIdsCarryTheWorkerIndexAndCountUp() {
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 7)) {
            assertEquals("run.7.0", writer.nextRecordId());
            assertEquals("run.7.1", writer.nextRecordId());
        }
    }

    @Test
    void gameIdsCountSeparatelyFromRecordIds() {
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0)) {
            writer.nextRecordId();
            writer.nextRecordId();
            assertEquals("run.0.0", writer.nextGameId());
        }
    }

    @Test
    void twoWorkersProduceDisjointIds() {
        try (RecordShardWriter first = new RecordShardWriter(tempDir, "run", 0);
             RecordShardWriter second = new RecordShardWriter(tempDir, "run", 1)) {
            assertNotEquals(first.nextRecordId(), second.nextRecordId());
        }
    }

    @Test
    void writesAreOneLineEach() throws IOException {
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0)) {
            writer.write("{\"a\":1}");
            writer.write("{\"a\":2}");
        }
        assertEquals(List.of("{\"a\":1}", "{\"a\":2}"), readShard("run.0"));
    }

    @Test
    void aSecondWriterAppendsRatherThanTruncating() throws IOException {
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0)) {
            writer.write("{\"a\":1}");
        }
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0)) {
            writer.write("{\"a\":2}");
        }
        assertEquals(2, readShard("run.0").size());
    }

    @Test
    void aShardIsAConcatenationOfCompleteGzipMembers() throws IOException {
        // Each writer contributes its own member, so appending is valid gzip
        // rather than a second stream glued onto the first. That is what lets a
        // killed worker truncate the last member and leave the rest readable.
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0)) {
            writer.write("{\"a\":1}");
        }
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0)) {
            writer.write("{\"a\":2}");
        }
        byte[] bytes = Files.readAllBytes(
                tempDir.resolve("run.0" + RecordShardWriter.SUFFIX));
        int members = 0;
        for (int i = 0; i + 1 < bytes.length; i++) {
            // gzip's magic number, which starts every member.
            if ((bytes[i] & 0xFF) == 0x1F && (bytes[i + 1] & 0xFF) == 0x8B) {
                members++;
            }
        }
        assertTrue(members >= 2, "expected one member per writer, saw " + members);
    }

    @Test
    void aFullBlockIsFlushedWithoutClosing() throws IOException {
        // The worker loops until it is killed, so a shard that only wrote on
        // close would be empty for the whole run.
        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0);
        for (int i = 0; i < RecordShardWriter.BLOCK_RECORDS; i++) {
            writer.write("{\"a\":" + i + "}");
        }
        assertEquals(RecordShardWriter.BLOCK_RECORDS, readShard("run.0").size());
    }

    /** The shard's lines, decompressed. */
    private List<String> readShard(String stem) throws IOException {
        Path path = tempDir.resolve(stem + RecordShardWriter.SUFFIX);
        try (var gzip = new java.util.zip.GZIPInputStream(Files.newInputStream(path));
                var reader = new java.io.BufferedReader(
                        new java.io.InputStreamReader(gzip, java.nio.charset.StandardCharsets.UTF_8))) {
            return reader.lines().toList();
        }
    }

    @Test
    void theDirectoryIsCreated() {
        Path nested = tempDir.resolve("output/effects/records");
        try (RecordShardWriter writer = new RecordShardWriter(nested, "run", 0)) {
            assertTrue(Files.isDirectory(nested));
            assertNotNull(writer.path());
        }
    }

    private static void assertNotNull(Object value) {
        assertTrue(value != null);
    }

    // ── the record envelope ─────────────────────────────────────────────

    private EffectRecord record(String kind) {
        return new EffectRecord(
                "run.0.1", "run", "2026-09-07T00:00:00Z", "run.0.0", kind,
                AttributionMode.DEGRADED);
    }

    @Test
    void anEnvelopeCarriesEveryFrozenField() {
        String json = record(EffectRecord.KIND_COMBAT).actor("P0").toJson();
        for (String key : List.of(
                "record_id", "run_id", "timestamp", "game_id", "kind", "moment",
                "subkind", "link_id", "mirror_of", "variant_of", "mode",
                "interventional", "fork", "synthetic", "actor_player", "ability",
                "state", "payload")) {
            assertTrue(json.contains("\"" + key + "\":"), key + " missing from " + json);
        }
    }

    @Test
    void aCombatRecordNamesNoActingLine() {
        String json = record(EffectRecord.KIND_COMBAT).actor("P0").toJson();
        assertTrue(json.contains("\"ability\":null"), json);
    }

    @Test
    void anOrdinaryObservationCarriesNeitherFlag() {
        String json = record(EffectRecord.KIND_RESOLUTION).toJson();
        assertTrue(json.contains("\"interventional\":false"));
        assertTrue(json.contains("\"fork\":false"));
        assertTrue(json.contains("\"synthetic\":false"));
    }

    @Test
    void aDamageStepProbeIsAForkThatIntervenedInNothing() {
        String json = record(EffectRecord.KIND_COMBAT)
                .fork(true).interventional(false).mirrorOf("run.0.9").toJson();
        assertTrue(json.contains("\"fork\":true"));
        assertTrue(json.contains("\"interventional\":false"));
        assertTrue(json.contains("\"mirror_of\":\"run.0.9\""));
    }

    @Test
    void anInterventionalResolutionIsBoth() {
        String json = record(EffectRecord.KIND_RESOLUTION)
                .interventional(true).fork(true).toJson();
        assertTrue(json.contains("\"interventional\":true"));
        assertTrue(json.contains("\"fork\":true"));
    }

    @Test
    void aPartnerlessHalfCarriesNoLinkId() {
        String json = record(EffectRecord.KIND_RESOLUTION)
                .moment(EffectRecord.MOMENT_ACTIVATION)
                .payload(EffectRecord.costPayload(EffectRecord.OUTCOME_COUNTERED))
                .toJson();
        assertTrue(json.contains("\"link_id\":null"), json);
        assertTrue(json.contains("\"outcome\":\"countered\""), json);
    }

    @Test
    void theModeIsStampedOnEveryRecord() {
        assertTrue(record(EffectRecord.KIND_COMBAT).toJson()
                .contains("\"mode\":\"degraded\""));
        assertTrue(new EffectRecord(
                "r", "r", "t", "g", EffectRecord.KIND_COMBAT,
                AttributionMode.PATCHED).toJson().contains("\"mode\":\"patched\""));
    }

    @Test
    void anAbilityCanCarrySeveralKeysForAMergedLine() {
        String json = record(EffectRecord.KIND_RESOLUTION)
                .ability(List.of(
                        new ProvenanceKey("cardsfolder/a/x.txt", 0, "trigger", 0),
                        new ProvenanceKey("cardsfolder/a/x.txt", 0, "static", 1)))
                .toJson();
        assertEquals(2, json.split("\"trait_kind\"", -1).length - 1, json);
    }

    // ── events ──────────────────────────────────────────────────────────

    @Test
    void anEventCarriesItsTypeSubjectsAndParams() {
        String json = new EffectEvent(EffectEvent.DAMAGE_DEALT)
                .subject("E12").param("amount", 3).param("combat", true)
                .toJson();
        assertTrue(json.contains("\"type\":\"damage_dealt\""), json);
        assertTrue(json.contains("\"subjects\":[\"E12\"]"), json);
        assertTrue(json.contains("\"amount\":3"), json);
        assertTrue(json.contains("\"combat\":true"), json);
    }

    @Test
    void aNullParamIsOmittedRatherThanWrittenAsNull() {
        String json = new EffectEvent(EffectEvent.ZONE_CHANGE)
                .subject("E1").param("to_zone", null).toJson();
        assertTrue(json.contains("\"params\":{}"), json);
    }

    @Test
    void aContinuousOutcomeCarriesItsDuration() {
        String json = new EffectEvent(EffectEvent.PT_CHANGE)
                .subject("E1").duration("end_of_turn").toJson();
        assertTrue(json.contains("\"duration\":\"end_of_turn\""), json);
    }

    @Test
    void anEventAttributesToItsSubAbilityLinkOrToTheRootLine() {
        assertTrue(new EffectEvent(EffectEvent.DAMAGE_DEALT).toJson()
                .contains("\"attributed_to\":null"));
        assertTrue(new EffectEvent(EffectEvent.DAMAGE_DEALT).attributedTo("0")
                .toJson().contains("\"attributed_to\":\"0\""));
    }

    @Test
    void aMapParamRendersAsAnObject() {
        String json = new EffectEvent(EffectEvent.MANA_PRODUCED)
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
                .find(PatchHooks.TRIGGER_HANDLER, "setEffectRecordCause")
                .present();
        assertEquals(
                hookPresent ? AttributionMode.PATCHED : AttributionMode.DEGRADED,
                AttributionMode.detect());
    }

    @Test
    void theModeIsProbedOnceAndRemembered() {
        assertSame(AttributionMode.detect(), AttributionMode.detect());
    }

    private static void assertSame(Object a, Object b) {
        assertFalse(a != b, "expected the same instance");
    }

    // ── JSON escaping ───────────────────────────────────────────────────

    @Test
    void aQuoteInACardNameIsEscaped() {
        assertEquals("\"Ajani\\\"s\"", Json.string("Ajani\"s"));
    }

    @Test
    void aNewlineInReminderTextIsEscaped() {
        assertEquals("\"a\\nb\"", Json.string("a\nb"));
    }

    @Test
    void nullRendersAsTheLiteral() {
        assertEquals("null", Json.string(null));
    }

    @Test
    void anAccentedCardNameSurvivesUnescaped() {
        assertEquals("\"Lim-Dûl\"", Json.string("Lim-Dûl"));
    }
}
