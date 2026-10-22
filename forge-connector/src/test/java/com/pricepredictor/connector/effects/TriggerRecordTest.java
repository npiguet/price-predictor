package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.CardTraitBase;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.card.CounterType;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static com.pricepredictor.connector.effects.TestCollectors.recording;
import static com.pricepredictor.connector.effects.TestCollectors.replacementOfMode;
import static com.pricepredictor.connector.effects.TestCollectors.triggerOfMode;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What a trigger record says: the line that acted, the event it saw, what
 * caused that event, and how many of the evaluations that did not fire are
 * kept beside the ones that did.
 *
 * <p>Built straight off the record builder the trigger hook calls, because
 * what is under test is what the collector reads off the trait and the run
 * parameters it is handed; the handler around it is
 * {@link PatchedCollectorTest}'s.
 */
@ExtendWith(ForgeExtension.class)
class TriggerRecordTest {

    @TempDir
    Path tempDir;

    private static List<Trigger> triggersOf(Card card) {
        List<Trigger> triggers = new ArrayList<>();
        card.getCurrentState().getTriggers().forEach(triggers::add);
        return triggers;
    }

    /**
     * The defect the whole trigger channel had: 727,308 records that said only
     * "some trigger saw this event". The hook hands over the {@link Trigger}
     * itself and the handler dropped it, so the key and {@code refs.source}
     * were null on every row of the first collected corpus.
     */
    @Test
    void aTriggerRecordNamesItsTriggerLine() {
        Card paralyze = TestCards.build("Paralyze");
        List<Trigger> triggers = triggersOf(paralyze);
        assertEquals(2, triggers.size(), "Paralyze declares two T: lines");

        String json = recording(tempDir)
                .triggers().record(triggers.get(1), Map.of(), true).toJson();

        assertTrue(json.contains("\"trait_kind\":\"trigger\""), json);
        assertTrue(json.contains("cardsfolder/p/paralyze.txt"), json);
        assertTrue(json.contains("\"index_within_kind\":1"),
                "the second T: line, not whichever was found first: " + json);
        assertFalse(json.contains("\"ability\":null"), json);
    }

    /** And it names the permanent whose text acted, which is the join. */
    @Test
    void aTriggerRecordNamesItsHostEntity() {
        Card paralyze = TestCards.build("Paralyze");
        String entity = SnapshotBuilder.entityId(paralyze);

        String json = recording(tempDir)
                .triggers().record(triggersOf(paralyze).get(0), Map.of(), false).toJson();

        assertTrue(json.contains("\"source\":\"" + entity + "\""), json);
        // Tier 1 carries the host whatever zone it sits in. Without that a
        // dies trigger names an entity id absent from state.entities, and the
        // reader cannot follow a source it cannot find.
        assertTrue(json.contains("\"id\":\"" + entity + "\""),
                "the host must be in state.entities: " + json);
    }

    /**
     * A hook whose signature drifted must yield a keyless record rather than a
     * ClassCastException inside a dynamic proxy, which the engine would surface
     * as an UndeclaredThrowableException in the middle of canRunTrigger.
     */
    @Test
    void anArgumentThatIsNotATraitStillProducesARecord() {
        String json = recording(tempDir).triggers().record("not a trait", Map.of(), true).toJson();

        assertTrue(json.contains("\"ability\":[]"),
                "an unattributable line is empty, not null: " + json);
        assertTrue(json.contains("\"source\":null"), json);
        // And it says why it is empty. Nothing that is not one of the five
        // trait kinds can name a printed line, which is a different answer
        // from a resolver that looked and failed.
        assertTrue(json.contains("\"ability_unresolved\":\"unknown_kind\""), json);
    }

    // ── why an empty acting line is empty ──────────────────────────────

    /**
     * An engine-built card names no script file, and the record says which.
     *
     * <p>This is the whole point of the field: 3.1% of resolution records named
     * no acting line and there was no way to ask whether that was The Monarch
     * being The Monarch or the resolver having regressed. {@code engine_effect}
     * is the expected answer and {@code unindexable} is the alarm, and the
     * corpus could not tell them apart because nothing wrote either.
     */
    @Test
    void anEngineBuiltLineSaysEngineEffectOnTheRecord() {
        Card monarch = new Card(TestCards.nextCardId(), TestCards.game());
        monarch.setName("The Monarch");
        monarch.setGamePieceType(forge.card.GamePieceType.EFFECT);
        SpellAbility draw = AbilityFactory.getAbility(
                "DB$ Draw | Defined$ You | NumCards$ 1", monarch);
        monarch.getCurrentState().addSpellAbility(draw);

        String json = recording(tempDir).triggers().record(draw, Map.of(), true).toJson();

        assertTrue(json.contains("\"ability\":[]"), json);
        assertTrue(json.contains("\"ability_unresolved\":\"engine_effect\""), json);
    }

    // ── what a run parameter map says ──────────────────────────────────

    /**
     * The event a trigger record carries, as rendered JSON.
     *
     * <p>Handed a replacement too where the mode under test is the
     * replacement side's spelling: the event is read off the trait's mode and
     * the parameters, whichever kind of trait carries the mode.
     */
    private String eventJson(CardTraitBase trait, Map<String, Object> params) {
        String json = recording(tempDir).triggers().record(trait, params, true).toJson();
        int at = json.indexOf("\"payload\":{\"event\":");
        return json.substring(at + 19, json.indexOf(",\"fired\"", at));
    }

    /**
     * A zone rewrite says which zones, from the values rather than the keys.
     *
     * <p>The whole rewrite channel was information-free because
     * {@code describeParams} read the map's <b>key names</b>: it built both
     * halves out of the mode plus a subject per entry, so incoming and outgoing
     * were byte-identical on 100% of 62,546 records by construction.
     */
    @Test
    void aMoveNamesTheZonesItWentBetween() {
        String json = eventJson(replacementOfMode("Moved"), Map.of(
                "Origin", ZoneType.Hand,
                "Destination", ZoneType.Graveyard,
                "Card", TestCards.build("Mountain")));

        assertTrue(json.contains("\"type\":\"zone_change\""), json);
        assertTrue(json.contains("\"from_zone\":\"hand\""), json);
        assertTrue(json.contains("\"to_zone\":\"graveyard\""), json);
    }

    /**
     * One card named three ways is one subject.
     *
     * <p>{@code Card}, {@code CardLKI} and {@code Affected} routinely all point
     * at the same permanent, and reading every map entry as a subject gave
     * 377 of 691 sampled trigger events a repeated one.
     */
    @Test
    void oneCardNamedThreeWaysIsOneSubject() {
        Card card = TestCards.build("Mountain");
        String json = eventJson(replacementOfMode("Moved"), Map.of(
                "Card", card, "CardLKI", card, "Affected", card));

        assertEquals(
                "[\"" + SnapshotBuilder.entityId(card) + "\"]",
                json.substring(json.indexOf("\"subjects\":") + 11,
                        json.indexOf("]", json.indexOf("\"subjects\":")) + 1),
                json);
    }

    /**
     * {@code cause} is a ref, and never the list of key names it used to be.
     *
     * <p>The schema documents it as the entity or player that caused the event
     * and the collector wrote a comma-joined list of {@code AbilityKey} names —
     * a value that was near-constant per event type and named no cause at all.
     */
    @Test
    void theCauseIsAnEntityRefRatherThanAListOfKeyNames() {
        Card cause = TestCards.build("Lightning Bolt");
        String json = eventJson(replacementOfMode("Moved"), Map.of(
                "Card", TestCards.build("Mountain"),
                "Destination", ZoneType.Graveyard,
                "Cause", cause));

        assertTrue(json.contains(
                "\"cause\":\"" + SnapshotBuilder.entityId(cause) + "\""), json);
        assertFalse(json.contains("\"cause\":\"Card,"), json);
        assertFalse(json.contains("Destination,"), json);
    }

    /** An unmapped mode carries its mode and says nothing it cannot support. */
    @Test
    void anUnmappedModeCarriesOnlyWhatItKnows() {
        String json = eventJson(triggerOfMode("Exploited"), Map.of());

        assertTrue(json.contains("\"type\":\"state_flag_change\""), json);
        assertTrue(json.contains("\"params\":{\"mode\":\"Exploited\"}"),
                json);
    }

    @Test
    void damageIsReadAsAnAmountACombatFlagAndASource() {
        Card source = TestCards.build("Lightning Bolt");
        String json = eventJson(triggerOfMode("DamageDone"), Map.of(
                "DamageAmount", 3,
                "IsCombatDamage", Boolean.FALSE,
                "DamageSource", source,
                "Affected", TestCards.build("Mountain")));

        assertTrue(json.contains("\"amount\":3"), json);
        assertTrue(json.contains("\"combat\":false"), json);
        assertTrue(json.contains(
                "\"source\":\"" + SnapshotBuilder.entityId(source) + "\""), json);
    }

    /** Prevention arrives through the damage keys and is not damage dealt. */
    @Test
    void preventedDamageIsNotRecordedAsDamageDealt() {
        String json = eventJson(triggerOfMode("DamageDone"), Map.of(
                "PreventedAmount", 2,
                "Affected", TestCards.build("Mountain")));

        assertTrue(json.contains("\"type\":\"damage_prevented\""), json);
        assertTrue(json.contains("\"amount\":2"), json);
    }

    /**
     * A counter map is flattened to the type and the number.
     *
     * <p>This is the one shape a counter replacement rewrites in place, so it is
     * also the one the engine-side copy has to reach inside — a collector that
     * reads it correctly still sees identity if the copy is one level deep.
     */
    @Test
    void aCounterMapIsFlattenedToATypeAndADelta() {
        com.google.common.collect.Multiset<CounterType> counters =
                com.google.common.collect.LinkedHashMultiset.create();
        counters.add(CounterType.getType("P1P1"), 2);
        String json = eventJson(triggerOfMode("CounterAdded"), Map.of(
                "CounterMap", Map.of(Optional.empty(), counters),
                "Affected", TestCards.build("Mountain")));

        assertTrue(json.contains("\"type\":\"counter_change\""), json);
        // Named the way the engine names it, which is the same spelling the
        // observed counter events carry -- a reader must not have to know which
        // collector wrote a record to know what "+1/+1" is called.
        assertTrue(json.contains("\"counter_type\":\"+1/+1\""), json);
        assertTrue(json.contains("\"delta\":2"), json);
    }

    /** The replacement side's word for a life total is read as one. */
    @Test
    void theReplacementSpellingOfALifeAmountIsStillADelta() {
        String json = eventJson(replacementOfMode("LifeReduced"), Map.of(
                "Affected", TestCards.build("Mountain"), "Amount", 4));

        assertTrue(json.contains("\"type\":\"life_change\""), json);
        assertTrue(json.contains("\"delta\":4"), json);
    }

    // ── the cause channel ───────────────────────────────────────────────

    private static String eventOf(TestCollectors.Channels collector, Trigger trigger,
            Map<String, Object> runParams) {
        return collector.triggers().record(trigger, runParams, true).toJson();
    }

    /**
     * The cause channel has one spelling, and a missing cause writes nothing.
     *
     * <p>Both halves matter. {@code params.cause} is what tells two attackers
     * each dealing 1 damage to the same player apart, so it has to be the same
     * key holding the same kind of value wherever it is written; and Forge
     * names no cause at all for a phase change or a block, where a placeholder
     * would be a claim rather than a gap.
     */
    @Test
    void theCauseChannelHasOneSpellingAndAnHonestAbsence() {
        assertTrue(new EffectEvent(EffectEvent.TAPPED).cause("E7").toJson()
                .contains("\"cause\":\"E7\""));
        assertFalse(new EffectEvent(EffectEvent.TAPPED).cause(null).toJson()
                .contains("cause"));
    }

    /**
     * An event says what caused it, from whichever key the engine used.
     *
     * <p>{@code causeOf} read {@code Cause} and {@code SpellAbility} only, and
     * over the smoke corpus that answered on 48% of trait-derived events: the
     * modes that keep the causing object under {@code Source}
     * ({@code CounterAdded}, {@code LifeGained}), {@code SourceSA}
     * ({@code BecomesTarget}) or {@code Causer} ({@code Destroyed}) said
     * nothing at all.
     */
    @Test
    void aCounterEventNamesTheCardThatPutTheCounterThere() {
        Card host = TestCards.build("Paralyze");
        Card source = TestCards.build("Sol Ring");

        String json = eventOf(recording(tempDir), triggerOfMode("CounterAdded"),
                Map.of("Card", host, "Source", source,
                        "CounterType", "P1P1", "CounterNum", 1));

        assertTrue(json.contains(
                "\"cause\":\"" + SnapshotBuilder.entityId(source) + "\""), json);
    }

    /** A cause that is a spell is named by the card it is printed on. */
    @Test
    void anAbilityCauseIsNamedByItsHostCard() {
        Card host = TestCards.build("Paralyze");
        Card caster = TestCards.build("Sol Ring");
        SpellAbility cause = AbilityFactory.getAbility(
                "DB$ Draw | Defined$ You | NumCards$ 1", caster);

        String json = eventOf(recording(tempDir), triggerOfMode("ChangesZone"),
                Map.of("Card", host, "Cause", cause,
                        "Origin", "Battlefield", "Destination", "Graveyard"));

        assertTrue(json.contains(
                "\"cause\":\"" + SnapshotBuilder.entityId(caster) + "\""), json);
    }

    /**
     * The chain is ordered, not merged.
     *
     * <p>A map often holds several of these at once — a zone change carries
     * {@code Cause} and whatever the caller put under {@code Source} — and the
     * engine's explicit answer is the one that outranks a source card.
     */
    @Test
    void anExplicitCauseOutranksASofterName() {
        Card host = TestCards.build("Paralyze");
        Card named = TestCards.build("Sol Ring");
        Card softer = TestCards.build("Mox Pearl");

        String json = eventOf(recording(tempDir), triggerOfMode("ChangesZone"),
                Map.of("Card", host, "Cause", named, "Source", softer,
                        "Origin", "Battlefield", "Destination", "Graveyard"));

        assertTrue(json.contains(
                "\"cause\":\"" + SnapshotBuilder.entityId(named) + "\""), json);
    }

    /**
     * Nothing caused an untap step, and the record says so by saying nothing.
     *
     * <p>{@code Card.java:4864} runs the {@code Untaps} trigger on a map that
     * holds the card and no actor of any kind. Naming the host there would put
     * the permanent that untapped into a slot that means "what did this to it".
     */
    @Test
    void anEventNothingCausedCarriesNoCause() {
        Card host = TestCards.build("Paralyze");

        String json = eventOf(
                recording(tempDir), triggerOfMode("Untaps"), Map.of("Card", host));

        assertFalse(json.contains("\"cause\""), json);
    }

    // ── what an event says it was ───────────────────────────────────────

    /**
     * The cast spell is named by the only key the stack puts it under.
     *
     * <p>{@code MagicStack.java:363} builds the {@code SpellCast} map from
     * {@code CardLKI}, {@code Activator} and {@code SpellAbility} and no
     * {@code Card} at all, so all 1,917 {@code SpellCast} events of the smoke
     * corpus named no subject whatever. Last in the key order, because
     * {@code CardLKI} is a copy of the card as it was and the live object is
     * the better answer wherever the map also holds one.
     */
    @Test
    void aCastSpellIsNamedByItsLastKnownCopy() {
        Card cast = TestCards.build("Sol Ring");
        SpellAbility spell = AbilityFactory.getAbility(
                "DB$ Draw | Defined$ You | NumCards$ 1", cast);

        String json = eventOf(recording(tempDir), triggerOfMode("SpellCast"),
                Map.of("CardLKI", cast, "SpellAbility", spell));

        assertTrue(json.contains(
                "\"subjects\":[\"" + SnapshotBuilder.entityId(cast) + "\"]"), json);
    }

    /**
     * A declared attack is an attack, not an unnamed state flag.
     *
     * <p>{@code Attacks} is the third heaviest trait-derived mode in the smoke
     * corpus (2,632 events) and had no entry in the mode table, so every one of
     * them took the generic {@code state_flag_change} and carried nothing but
     * its own name — two creatures attacking the same player were
     * indistinguishable rows. As {@code attackers_declared} it carries the
     * defender, read under the key {@code CombatUtil.checkDeclaredAttacker}
     * actually uses.
     */
    @Test
    void anAttackDeclarationSaysWhoWasAttacked() {
        Card attacker = TestCards.build("Sol Ring");
        Card defender = TestCards.build("Mox Pearl");

        String json = eventOf(recording(tempDir), triggerOfMode("Attacks"),
                Map.of("Attacker", attacker, "Attacked", defender));

        assertTrue(json.contains("\"type\":\"attackers_declared\""), json);
        assertTrue(json.contains(
                "\"defender\":\"" + SnapshotBuilder.entityId(defender) + "\""), json);
    }

    /**
     * A mana trigger says which mana, in the slot the activation channel uses.
     *
     * <p>{@code TapsForMana} was 534 events carrying only their mode. The
     * reading is the same {@code manaByColor} the mana-activation records use,
     * so a tapped land and a replaced mana production are comparable rows
     * rather than two spellings of one fact.
     */
    @Test
    void aManaTriggerSaysWhichManaWasProduced() {
        Card land = TestCards.build("Sol Ring");

        String json = eventOf(recording(tempDir), triggerOfMode("TapsForMana"),
                Map.of("Card", land, "Produced", "G G"));

        assertTrue(json.contains("\"type\":\"mana_produced\""), json);
        assertTrue(json.contains("\"mana_by_color\":{\"G\":2}"), json);
    }

    /** And a mana value it cannot read is absent rather than an empty object. */
    @Test
    void anUnreadableManaValueLeavesTheSlotEmptyRatherThanClaimingNoMana() {
        Card land = TestCards.build("Sol Ring");

        String json = eventOf(recording(tempDir), triggerOfMode("TapsForMana"),
                Map.of("Card", land));

        assertFalse(json.contains("mana_by_color"), json);
    }

    /** Turning a card face up is a face change, and it says which face. */
    @Test
    void turningACardFaceUpNamesTheFaceItTurnedTo() {
        Card card = TestCards.build("Sol Ring");

        String json = eventOf(
                recording(tempDir), triggerOfMode("TurnFaceUp"), Map.of("Card", card));

        assertTrue(json.contains("\"type\":\"face_change\""), json);
        assertTrue(json.contains("\"to_state\":\"face_up\""), json);
    }

    /**
     * Combat's three shapes each name the creature they are about.
     *
     * <p>{@code Blocks} holds {@code Blocker} and {@code Attackers},
     * {@code AttackerBlockedByCreature} holds {@code Attacker} and
     * {@code Blocker}, and {@code AttackersDeclared} holds only the collection —
     * so with none of those three keys in the subject order, all 226 of them in
     * the smoke corpus named no subject at all. The order is what keeps each
     * event about the right creature rather than about whichever key was found
     * first.
     */
    @Test
    void eachCombatShapeIsAboutTheCreatureItIsAbout() {
        Card attacker = TestCards.build("Sol Ring");
        Card blocker = TestCards.build("Mox Pearl");
        String attackerId = SnapshotBuilder.entityId(attacker);
        String blockerId = SnapshotBuilder.entityId(blocker);

        String blocks = eventOf(recording(tempDir), triggerOfMode("Blocks"),
                Map.of("Blocker", blocker, "Attackers", attacker));
        assertTrue(blocks.contains("\"subjects\":[\"" + blockerId + "\"]"), blocks);
        assertTrue(blocks.contains("\"blocked\":\"" + attackerId + "\""), blocks);

        String blocked = eventOf(recording(tempDir),
                triggerOfMode("AttackerBlockedByCreature"),
                Map.of("Attacker", attacker, "Blocker", blocker));
        assertTrue(blocked.contains("\"type\":\"became_blocked\""), blocked);
        assertTrue(blocked.contains("\"subjects\":[\"" + attackerId + "\"]"), blocked);
        assertTrue(blocked.contains("\"blockers\":[\"" + blockerId + "\"]"), blocked);
    }
}
