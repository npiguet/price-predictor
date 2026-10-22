package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The record envelope, the event shape, the attribution mode and JSON
 * escaping: the parts of a record that need no game.
 */
class EffectRecordTest {

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
                "interventional", "fork", "synthetic", "random_seat", "actor_player",
                "ability", "state", "payload")) {
            assertTrue(json.contains("\"" + key + "\":"), key + " missing from " + json);
        }
    }

    /**
     * {@code random_seat} follows {@code synthetic} on every record, and
     * {@code what_if} follows it on the two legality subkinds alone
     * (record-schema-delta.md): the order is the contract, and the first
     * field's presence is how a reader names a gen-2 shard.
     */
    @Test
    void randomSeatFollowsSyntheticOnEveryRecordAndWhatIfOnlyWhereSet() {
        String plain = record(EffectRecord.KIND_RESOLUTION).toJson();
        assertTrue(plain.contains("\"synthetic\":false,\"random_seat\":false,\"actor_player\":"),
                plain);
        assertFalse(plain.contains("what_if"), plain);

        String legality = record(EffectRecord.KIND_PLAYABILITY)
                .subkind("attackers").randomSeat(true).whatIf(false).actor("P1").toJson();
        assertTrue(legality.contains(
                "\"synthetic\":false,\"random_seat\":true,\"what_if\":false,\"actor_player\":\"P1\""),
                legality);
    }

    @Test
    void theActorIsReadableForTheStampAtTheWrite() {
        EffectRecord record = record(EffectRecord.KIND_COMBAT).actor("P0");
        assertEquals("P0", record.actorPlayer());
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
                // No ability: a countered spell's costs are still readable in
                // a real game, but the record's shape does not depend on them.
                .payload(EffectRecord.costPayload(
                        EffectRecord.costsJson(null), EffectRecord.OUTCOME_COUNTERED))
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

    /**
     * A trigger record names its acting line, and says so in the envelope.
     *
     * <p>Guards the wire shape the collector now fills: the whole first corpus
     * carried {@code "ability": null} on all 727,308 trigger records, and the
     * field rule exempts only {@code combat} and {@code playability} -- a
     * trigger is a single acting line.
     */
    @Test
    void aTriggerRecordNamesItsActingLine() {
        String json = record(EffectRecord.KIND_TRIGGER)
                .ability(new ProvenanceKey.Resolved(new ProvenanceKey(
                        "cardsfolder/g/grave_titan.txt", 0,
                        ProvenanceKey.KIND_TRIGGER, 0), null))
                .toJson();

        assertFalse(json.contains("\"ability\":null"), json);
        assertTrue(json.contains("\"trait_kind\":\"trigger\""), json);
        assertTrue(json.contains("cardsfolder/g/grave_titan.txt"), json);
    }

    /** And a rewrite record names the replacement effect that rewrote. */
    @Test
    void aRewriteRecordNamesItsReplacement() {
        String json = record(EffectRecord.KIND_REWRITE)
                .ability(new ProvenanceKey.Resolved(new ProvenanceKey(
                        "cardsfolder/p/paralyze.txt", 0,
                        ProvenanceKey.KIND_REPLACEMENT, 0), null))
                .toJson();

        assertTrue(json.contains("\"trait_kind\":\"replacement\""), json);
    }

    /**
     * A line that could not be attributed renders empty rather than absent.
     *
     * <p>The two are different answers -- "no line acts here" against "a line
     * acted and we could not name it" -- and a reader that folds them together
     * cannot measure the second.
     */
    @Test
    void anUnattributableLineIsEmptyRatherThanNull() {
        assertTrue(record(EffectRecord.KIND_TRIGGER)
                .ability(new ProvenanceKey.Resolved(
                        null, ProvenanceKey.UNRESOLVED_ENGINE_EFFECT))
                .toJson().contains("\"ability\":[]"));
        assertTrue(record(EffectRecord.KIND_COMBAT).toJson()
                .contains("\"ability\":null"));
    }

    /**
     * And an empty line says why it is empty.
     *
     * <p>"No printed line exists, correctly" and "the resolver regressed" were
     * the same row for a whole collection run. The two fields are set from one
     * answer so they cannot be written out of step: a reason beside a named
     * line is a contradiction the reader refuses outright.
     */
    @Test
    void anEmptyLineSaysWhyItIsEmpty() {
        String json = record(EffectRecord.KIND_TRIGGER)
                .ability(new ProvenanceKey.Resolved(
                        null, ProvenanceKey.UNRESOLVED_ENGINE_EFFECT))
                .toJson();

        assertTrue(json.contains("\"ability\":[]"), json);
        assertTrue(json.contains("\"ability_unresolved\":\"engine_effect\""), json);
    }

    @Test
    void aNamedLineCarriesNoReason() {
        String json = record(EffectRecord.KIND_TRIGGER)
                .ability(new ProvenanceKey.Resolved(new ProvenanceKey(
                        "cardsfolder/g/grave_titan.txt", 0,
                        ProvenanceKey.KIND_TRIGGER, 0), null))
                .toJson();

        assertTrue(json.contains("cardsfolder/g/grave_titan.txt"), json);
        assertTrue(json.contains("\"ability_unresolved\":null"), json);
    }

    /** A caller that looked and got nothing back still owes a reason. */
    @Test
    void aLookupThatCameBackWithNothingAtAllStillNamesAReason() {
        String json = record(EffectRecord.KIND_TRIGGER)
                .ability(null)
                .toJson();

        assertTrue(json.contains("\"ability_unresolved\":\"unknown_kind\""), json);
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
                .find(PatchHooks.ABILITY_UTILS, "getEffectRecordSubAbility")
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
