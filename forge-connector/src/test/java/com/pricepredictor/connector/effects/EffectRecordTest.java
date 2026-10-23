package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

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

        for (RecordKind qualified : List.of(RecordKind.RESOLUTION, RecordKind.PLAYABILITY)) {
            assertThrows(IllegalArgumentException.class, () -> record(qualified), qualified.wireName());
        }
    }

    @Test
    void theActorIsReadableForTheStampAtTheWrite() {
        EffectRecord record = record(RecordKind.COMBAT).actor("P0");
        assertEquals("P0", record.actorPlayer());
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
                .payload(EffectRecord.costPayload(
                        EffectRecord.costsJson(null), ActivationOutcome.COUNTERED))
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
        String combat = record(RecordKind.COMBAT).toJson();
        assertTrue(combat.contains("\"ability\":null"), combat);
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
                EffectRecord.combatPayload(CombatShape.of(null).fields(), List.of()));
    }

    /** A probe branch's perturbation follows the fields it shares with the record. */
    @Test
    void aProbePayloadNamesWhatItPerturbedAfterTheCombat() {
        assertEquals(
                "{\"attackers\":[],\"blocks\":{},\"assignment_choices\":{},\"events\":[],"
                        + "\"probed_keyword\":\"trample\",\"probed_entity\":null}",
                EffectRecord.probePayload(
                        CombatShape.of(null).fields(), List.of(), "trample", null));
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
