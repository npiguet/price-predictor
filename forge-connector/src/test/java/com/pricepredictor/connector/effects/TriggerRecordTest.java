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

import static com.pricepredictor.connector.effects.RecordLines.onlyEvent;
import static com.pricepredictor.connector.effects.RecordLines.params;
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
     * A trigger record says which trigger, not only "some trigger saw this
     * event". The hook hands over the {@link Trigger} itself, and the key and
     * {@code refs.source} are read from it.
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
     * <p>This is the whole point of the field: a record naming no acting line
     * cannot otherwise say whether that is The Monarch being The Monarch or the
     * resolver having regressed. {@code engine_effect} is the expected answer
     * and {@code unindexable} is the alarm.
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
     * The event a trigger record carries for {@code trait} seeing {@code
     * runParams}, parsed.
     *
     * <p>Handed a replacement too where the mode under test is the
     * replacement side's spelling: the event is read off the trait's mode and
     * the parameters, whichever kind of trait carries the mode.
     */
    private Map<String, Object> eventOf(CardTraitBase trait, Map<String, Object> runParams) {
        return onlyEvent(recording(tempDir).triggers().record(trait, runParams, true).toJson());
    }

    /**
     * A zone rewrite says which zones, from the values rather than the keys.
     *
     * <p>Read from the map's <b>key names</b> — the mode plus a subject per
     * entry — both halves would be byte-identical by construction, and the
     * whole rewrite channel information-free.
     */
    @Test
    void aMoveNamesTheZonesItWentBetween() {
        Map<String, Object> event = eventOf(replacementOfMode("Moved"), Map.of(
                "Origin", ZoneType.Hand,
                "Destination", ZoneType.Graveyard,
                "Card", TestCards.build("Mountain")));

        assertEquals("zone_change", event.get("type"), event.toString());
        assertEquals("hand", params(event).get("from_zone"), event.toString());
        assertEquals("graveyard", params(event).get("to_zone"), event.toString());
    }

    /**
     * One card named three ways is one subject.
     *
     * <p>{@code Card}, {@code CardLKI} and {@code Affected} routinely all point
     * at the same permanent, so reading every map entry as a subject would
     * repeat it.
     */
    @Test
    void oneCardNamedThreeWaysIsOneSubject() {
        Card card = TestCards.build("Mountain");
        Map<String, Object> event = eventOf(replacementOfMode("Moved"), Map.of(
                "Card", card, "CardLKI", card, "Affected", card));

        assertEquals(List.of(SnapshotBuilder.entityId(card)), event.get("subjects"),
                event.toString());
    }

    /**
     * {@code cause} is a ref, and never a list of key names.
     *
     * <p>The schema documents it as the entity or player that caused the event;
     * a comma-joined list of {@code AbilityKey} names would be near-constant
     * per event type and name no cause at all.
     */
    @Test
    void theCauseIsAnEntityRefRatherThanAListOfKeyNames() {
        Card cause = TestCards.build("Lightning Bolt");
        Map<String, Object> event = eventOf(replacementOfMode("Moved"), Map.of(
                "Card", TestCards.build("Mountain"),
                "Destination", ZoneType.Graveyard,
                "Cause", cause));

        assertEquals(SnapshotBuilder.entityId(cause), params(event).get("cause"),
                event.toString());
        assertFalse(event.toString().contains("Destination,"), event.toString());
    }

    /** An unmapped mode carries its mode and says nothing it cannot support. */
    @Test
    void anUnmappedModeCarriesOnlyWhatItKnows() {
        Map<String, Object> event = eventOf(triggerOfMode("Exploited"), Map.of());

        assertEquals("state_flag_change", event.get("type"), event.toString());
        assertEquals(Map.of("mode", "Exploited"), params(event), event.toString());
    }

    @Test
    void damageIsReadAsAnAmountACombatFlagAndASource() {
        Card source = TestCards.build("Lightning Bolt");
        Map<String, Object> event = eventOf(triggerOfMode("DamageDone"), Map.of(
                "DamageAmount", 3,
                "IsCombatDamage", Boolean.FALSE,
                "DamageSource", source,
                "Affected", TestCards.build("Mountain")));

        assertEquals(3L, params(event).get("amount"), event.toString());
        assertEquals(false, params(event).get("combat"), event.toString());
        assertEquals(SnapshotBuilder.entityId(source), params(event).get("source"),
                event.toString());
    }

    /** Prevention arrives through the damage keys and is not damage dealt. */
    @Test
    void preventedDamageIsNotRecordedAsDamageDealt() {
        Map<String, Object> event = eventOf(triggerOfMode("DamageDone"), Map.of(
                "PreventedAmount", 2,
                "Affected", TestCards.build("Mountain")));

        assertEquals("damage_prevented", event.get("type"), event.toString());
        assertEquals(2L, params(event).get("amount"), event.toString());
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
        Map<String, Object> event = eventOf(triggerOfMode("CounterAdded"), Map.of(
                "CounterMap", Map.of(Optional.empty(), counters),
                "Affected", TestCards.build("Mountain")));

        assertEquals("counter_change", event.get("type"), event.toString());
        // Named the way the engine names it, which is the same spelling the
        // observed counter events carry -- a reader must not have to know which
        // collector wrote a record to know what "+1/+1" is called.
        assertEquals("+1/+1", params(event).get("counter_type"), event.toString());
        assertEquals(2L, params(event).get("delta"), event.toString());
    }

    /** The replacement side's word for a life total is read as one. */
    @Test
    void theReplacementSpellingOfALifeAmountIsStillADelta() {
        Map<String, Object> event = eventOf(replacementOfMode("LifeReduced"), Map.of(
                "Affected", TestCards.build("Mountain"), "Amount", 4));

        assertEquals("life_change", event.get("type"), event.toString());
        assertEquals(4L, params(event).get("delta"), event.toString());
    }

    // ── the cause channel ───────────────────────────────────────────────

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
     * <p>Not only {@code Cause} and {@code SpellAbility}: the modes that keep
     * the causing object under {@code Source} ({@code CounterAdded},
     * {@code LifeGained}), {@code SourceSA} ({@code BecomesTarget}) or
     * {@code Causer} ({@code Destroyed}) would otherwise say nothing at all.
     */
    @Test
    void aCounterEventNamesTheCardThatPutTheCounterThere() {
        Card host = TestCards.build("Paralyze");
        Card source = TestCards.build("Sol Ring");

        Map<String, Object> event = eventOf(triggerOfMode("CounterAdded"),
                Map.of("Card", host, "Source", source,
                        "CounterType", "P1P1", "CounterNum", 1));

        assertEquals(SnapshotBuilder.entityId(source), params(event).get("cause"),
                event.toString());
    }

    /** A cause that is a spell is named by the card it is printed on. */
    @Test
    void anAbilityCauseIsNamedByItsHostCard() {
        Card host = TestCards.build("Paralyze");
        Card caster = TestCards.build("Sol Ring");
        SpellAbility cause = AbilityFactory.getAbility(
                "DB$ Draw | Defined$ You | NumCards$ 1", caster);

        Map<String, Object> event = eventOf(triggerOfMode("ChangesZone"),
                Map.of("Card", host, "Cause", cause,
                        "Origin", "Battlefield", "Destination", "Graveyard"));

        assertEquals(SnapshotBuilder.entityId(caster), params(event).get("cause"),
                event.toString());
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

        Map<String, Object> event = eventOf(triggerOfMode("ChangesZone"),
                Map.of("Card", host, "Cause", named, "Source", softer,
                        "Origin", "Battlefield", "Destination", "Graveyard"));

        assertEquals(SnapshotBuilder.entityId(named), params(event).get("cause"),
                event.toString());
    }

    /**
     * Nothing caused an untap step, and the record says so by saying nothing.
     *
     * <p>{@code Card.untap} runs the {@code Untaps} trigger on a map that
     * holds the card and no actor of any kind. Naming the host there would put
     * the permanent that untapped into a slot that means "what did this to it".
     */
    @Test
    void anEventNothingCausedCarriesNoCause() {
        Card host = TestCards.build("Paralyze");

        Map<String, Object> event = eventOf(triggerOfMode("Untaps"), Map.of("Card", host));

        assertFalse(event.toString().contains("cause"), event.toString());
    }

    // ── what an event says it was ───────────────────────────────────────

    /**
     * The cast spell is named by the only key the stack puts it under.
     *
     * <p>{@code MagicStack.add} builds the {@code SpellCast} map from
     * {@code CardLKI}, {@code Activator} and {@code SpellAbility} and no
     * {@code Card} at all, so without it a {@code SpellCast} event names no
     * subject whatever. Last in the key order, because
     * {@code CardLKI} is a copy of the card as it was and the live object is
     * the better answer wherever the map also holds one.
     */
    @Test
    void aCastSpellIsNamedByItsLastKnownCopy() {
        Card cast = TestCards.build("Sol Ring");
        SpellAbility spell = AbilityFactory.getAbility(
                "DB$ Draw | Defined$ You | NumCards$ 1", cast);

        Map<String, Object> event = eventOf(triggerOfMode("SpellCast"),
                Map.of("CardLKI", cast, "SpellAbility", spell));

        assertEquals(List.of(SnapshotBuilder.entityId(cast)), event.get("subjects"),
                event.toString());
    }

    /**
     * A declared attack is an attack, not an unnamed state flag.
     *
     * <p>{@code Attacks} is one of the heaviest trait-derived modes. Under the
     * generic {@code state_flag_change} it would carry nothing but its own
     * name — two creatures attacking the same player would be
     * indistinguishable rows. As {@code attackers_declared} it carries the
     * defender, read under the key {@code CombatUtil.checkDeclaredAttacker}
     * actually uses.
     */
    @Test
    void anAttackDeclarationSaysWhoWasAttacked() {
        Card attacker = TestCards.build("Sol Ring");
        Card defender = TestCards.build("Mox Pearl");

        Map<String, Object> event = eventOf(triggerOfMode("Attacks"),
                Map.of("Attacker", attacker, "Attacked", defender));

        assertEquals("attackers_declared", event.get("type"), event.toString());
        assertEquals(SnapshotBuilder.entityId(defender), params(event).get("defender"),
                event.toString());
    }

    /**
     * A mana trigger says which mana, in the slot the activation channel uses.
     *
     * <p>Without it a {@code TapsForMana} event carries only its mode. The
     * reading is the same {@code manaByColor} the mana-activation records use,
     * so a tapped land and a replaced mana production are comparable rows
     * rather than two spellings of one fact.
     */
    @Test
    void aManaTriggerSaysWhichManaWasProduced() {
        Card land = TestCards.build("Sol Ring");

        Map<String, Object> event = eventOf(triggerOfMode("TapsForMana"),
                Map.of("Card", land, "Produced", "G G"));

        assertEquals("mana_produced", event.get("type"), event.toString());
        assertEquals(Map.of("G", 2L), params(event).get("mana_by_color"), event.toString());
    }

    /** And a mana value it cannot read is absent rather than an empty object. */
    @Test
    void anUnreadableManaValueLeavesTheSlotEmptyRatherThanClaimingNoMana() {
        Card land = TestCards.build("Sol Ring");

        Map<String, Object> event = eventOf(triggerOfMode("TapsForMana"),
                Map.of("Card", land));

        assertFalse(params(event).containsKey("mana_by_color"), event.toString());
    }

    /** Turning a card face up is a face change, and it says which face. */
    @Test
    void turningACardFaceUpNamesTheFaceItTurnedTo() {
        Card card = TestCards.build("Sol Ring");

        Map<String, Object> event = eventOf(triggerOfMode("TurnFaceUp"), Map.of("Card", card));

        assertEquals("face_change", event.get("type"), event.toString());
        assertEquals("face_up", params(event).get("to_state"), event.toString());
    }

    /**
     * Combat's three shapes each name the creature they are about.
     *
     * <p>{@code Blocks} holds {@code Blocker} and {@code Attackers},
     * {@code AttackerBlockedByCreature} holds {@code Attacker} and
     * {@code Blocker}, and {@code AttackersDeclared} holds only the collection —
     * so with none of those three keys in the subject order, every one of them
     * would name no subject at all. The order is what keeps each
     * event about the right creature rather than about whichever key was found
     * first.
     */
    @Test
    void eachCombatShapeIsAboutTheCreatureItIsAbout() {
        Card attacker = TestCards.build("Sol Ring");
        Card blocker = TestCards.build("Mox Pearl");
        String attackerId = SnapshotBuilder.entityId(attacker);
        String blockerId = SnapshotBuilder.entityId(blocker);

        Map<String, Object> blocks = eventOf(triggerOfMode("Blocks"),
                Map.of("Blocker", blocker, "Attackers", attacker));
        assertEquals(List.of(blockerId), blocks.get("subjects"), blocks.toString());
        assertEquals(attackerId, params(blocks).get("blocked"), blocks.toString());

        Map<String, Object> blocked = eventOf(
                triggerOfMode("AttackerBlockedByCreature"),
                Map.of("Attacker", attacker, "Blocker", blocker));
        assertEquals("became_blocked", blocked.get("type"), blocked.toString());
        assertEquals(List.of(attackerId), blocked.get("subjects"), blocked.toString());
        assertEquals(List.of(blockerId), params(blocked).get("blockers"), blocked.toString());
    }
}
