package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.PatchedCollectors.CollectionCaps;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.event.GameEventCardChangeZone;
import forge.game.spellability.SpellAbility;
import forge.game.zone.Zone;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The four things a bus record could not say.
 *
 * <p>Each of these was measured on a 55,296-record corpus collected from the
 * previous version of this collector, and each was a wiring gap rather than a
 * format one: {@code outcome} was the literal {@code resolved} on all 4,989
 * activation records and 6.1% of link halves had no partner; 54% of
 * {@code zone_change} events said {@code to_zone=stack} and 13.68% of events in
 * a record repeated another one in it; {@code combat} and {@code resolution}
 * records were snapshotted one tier shallower than every other kind; and
 * {@code refs.modes} was filled on 125 records out of 55,296.
 *
 * <p>The bus itself is not exercised here — driving Forge's event bus needs a
 * running match, which is the integration test's job. What is exercised is
 * everything below the two lines that read the stack, against real abilities
 * built by {@code AbilityFactory} from real cards, which is where all four
 * defects lived.
 */
@ExtendWith(ForgeExtension.class)
class BusBracketCollectorTest {

    @TempDir
    Path tempDir;

    private RecordShardWriter writer;

    private BusBracketCollector collector(CollectionCaps caps) {
        writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        return new BusBracketCollector(TestCards.game(), writer, "run.0-l1.0", caps);
    }

    private BusBracketCollector collector() {
        return collector(CollectionCaps.defaults());
    }

    /** The records written so far, one parsed line each. */
    private List<String> written() throws IOException {
        writer.close();
        Path path = tempDir.resolve("run.0-l1" + RecordShardWriter.SUFFIX);
        if (!Files.exists(path)) {
            return List.of();
        }
        try (var gzip = new java.util.zip.GZIPInputStream(Files.newInputStream(path));
                var reader = new java.io.BufferedReader(new java.io.InputStreamReader(
                        gzip, java.nio.charset.StandardCharsets.UTF_8))) {
            return reader.lines().toList();
        }
    }

    /** A targeted burn spell, the shape a partial fizzle happens to. */
    private static SpellAbility damageAbility() {
        Card host = TestCards.build("Lightning Bolt");
        return AbilityFactory.getAbility(
                "SP$ DealDamage | Cost$ R | ValidTgts$ Any | TgtPrompt$ Choose"
                        + " | NumDmg$ 3", host);
    }

    // ── outcome ─────────────────────────────────────────────────────────

    /**
     * A cast writes nothing yet, because its outcome is not knowable yet.
     *
     * <p>This is the whole defect in one assertion. The cost half carries
     * {@code outcome}, and whether a spell resolved, fizzled or was countered
     * does not exist at the moment it is cast — so a record written here can
     * only say a literal, which is what all 4,989 activation records said.
     */
    @Test
    void aCastIsHeldRatherThanWrittenWithAGuessedOutcome() throws IOException {
        BusBracketCollector collector = collector();

        collector.beginBracket(damageAbility());

        assertEquals(0, collector.recordsWritten());
        assertEquals(1, collector.unresolvedActivations());
        assertEquals(List.of(), written());
    }

    /**
     * A resolution stamps the held half and writes both, joined by one link.
     */
    @Test
    void aResolutionWritesBothHalvesUnderOneLink() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();

        collector.beginBracket(bolt);
        collector.endBracket(bolt.getId(), false);

        List<String> records = written();
        assertEquals(2, records.size(), records.toString());
        String activation = records.get(0);
        String resolution = records.get(1);
        assertTrue(activation.contains("\"moment\":\"activation\""), activation);
        assertTrue(activation.contains("\"outcome\":\"resolved\""), activation);
        assertTrue(resolution.contains("\"moment\":\"resolution\""), resolution);

        String link = "run.0-l1.0.link." + bolt.getId();
        assertTrue(activation.contains("\"link_id\":\"" + link + "\""), activation);
        assertTrue(resolution.contains("\"link_id\":\"" + link + "\""), resolution);
    }

    /**
     * A fizzle writes the cost half alone, and issues no link at all.
     *
     * <p>{@code MagicStack.resolveStack} does not resolve a spell whose targets
     * are all gone, so there is no effect half to link to — and a {@code link_id}
     * on a fizzled activation is precisely the 306 dangling halves. The costs
     * were still paid, so the record itself must still be written.
     */
    @Test
    void aFizzleWritesThePartnerlessCostHalfAndNoEffectHalf() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();

        collector.beginBracket(bolt);
        collector.endBracket(bolt.getId(), true);

        List<String> records = written();
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"outcome\":\"fizzled\""), records.get(0));
        assertTrue(records.get(0).contains("\"link_id\":null"), records.get(0));
        assertTrue(records.get(0).contains("\"moment\":\"activation\""), records.get(0));
    }

    /**
     * A held cast that has left the stack without resolving was countered.
     *
     * <p>Forge publishes the same removal event for a spell that resolved and
     * for one a counterspell took away, so the question is asked of the stack
     * instead: what is no longer on it and did not resolve was removed.
     */
    @Test
    void aSpellGoneFromTheStackIsWrittenOffAsCountered() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility survives = damageAbility();
        SpellAbility countered = damageAbility();

        collector.beginBracket(survives);
        collector.beginBracket(countered);
        int written = collector.writeOffRemoved(Set.of(survives.getId()));

        assertEquals(1, written);
        assertEquals(1, collector.unresolvedActivations(), "the survivor is still held");
        List<String> records = written();
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"outcome\":\"countered\""), records.get(0));
        assertTrue(records.get(0).contains("\"link_id\":null"), records.get(0));
    }

    /**
     * Losing some targets but not all is a partial fizzle.
     *
     * <p>{@code MagicStack.hasFizzled} strips the targets that became illegal
     * before the spell resolves, so the count at resolution against the count at
     * the cast is the engine's own answer. Losing every target without the engine
     * calling it a fizzle is {@code CantFizzle}, where the spell still does what
     * it does.
     */
    @Test
    void losingSomeTargetsButNotAllIsAPartialFizzle() {
        SpellAbility bolt = damageAbility();
        bolt.resetTargets();
        bolt.getTargets().add(TestCards.build("Grizzly Bears"));
        bolt.getTargets().add(TestCards.build("Grizzly Bears"));

        assertEquals(EffectRecord.OUTCOME_RESOLVED,
                BusBracketCollector.outcomeOf(2, bolt));

        bolt.getTargets().remove(bolt.getTargets().get(0));
        assertEquals(EffectRecord.OUTCOME_PARTIALLY_FIZZLED,
                BusBracketCollector.outcomeOf(2, bolt));

        bolt.resetTargets();
        assertEquals(EffectRecord.OUTCOME_RESOLVED,
                BusBracketCollector.outcomeOf(2, bolt));
    }

    /** An untargeted spell never reads as partially fizzled. */
    @Test
    void anUntargetedSpellIsSimplyResolved() {
        SpellAbility gain = AbilityFactory.getAbility(
                "SP$ GainLife | Cost$ W | Defined$ You | LifeAmount$ 3",
                TestCards.build("Healing Salve"));

        assertEquals(EffectRecord.OUTCOME_RESOLVED,
                BusBracketCollector.outcomeOf(0, gain));
    }

    // ── zone changes ────────────────────────────────────────────────────

    private static Zone zone(ZoneType type) {
        return new Zone(type, TestCards.game());
    }

    /**
     * A card put onto the stack is a cast, not an outcome of one.
     *
     * <p>It has a record already — the activation half, with what was paid. As a
     * zone_change it lands in whichever bracket is open, so an opponent's
     * response reads as something the resolving ability did; it was 54% of the
     * whole channel.
     */
    @Test
    void aMoveOntoTheStackIsNotRecordedAsAnOutcome() {
        assertNull(BusEvents.cardMoved(new GameEventCardChangeZone(
                TestCards.build("Lightning Bolt"),
                zone(ZoneType.Hand), zone(ZoneType.Stack))));
    }

    /** A move off the stack is an outcome, and now says where it came from. */
    @Test
    void aMoveOffTheStackKeepsBothOfItsEnds() {
        EffectEvent moved = BusEvents.cardMoved(new GameEventCardChangeZone(
                TestCards.build("Grizzly Bears"),
                zone(ZoneType.Stack), zone(ZoneType.Battlefield)));

        assertNotNull(moved);
        String json = moved.toJson();
        assertTrue(json.contains("\"from_zone\":\"stack\""), json);
        assertTrue(json.contains("\"to_zone\":\"battlefield\""), json);
    }

    /**
     * The same transition told twice in one record is filed once.
     *
     * <p>A card is on the battlefield or it is not, so a second identical
     * {@code zone_change} carries nothing the first did not.
     */
    @Test
    void aRepeatedStateTransitionIsFiledOnce() {
        List<EffectEvent> into = new ArrayList<>();
        Set<String> seen = new LinkedHashSet<>();

        assertTrue(BusBracketCollector.fileEvent(zoneChange(), into, seen));
        assertFalse(BusBracketCollector.fileEvent(zoneChange(), into, seen));

        assertEquals(1, into.size());
    }

    /**
     * A repeated quantity is not, because there the repeat is the information.
     *
     * <p>Two creatures each dealing one damage to the same blocker render two
     * identical events, and collapsing them would turn two damage into one.
     */
    @Test
    void aRepeatedQuantityIsFiledEveryTime() {
        List<EffectEvent> into = new ArrayList<>();
        Set<String> seen = new LinkedHashSet<>();

        assertTrue(BusBracketCollector.fileEvent(oneDamage(), into, seen));
        assertTrue(BusBracketCollector.fileEvent(oneDamage(), into, seen));

        assertEquals(2, into.size());
    }

    private static EffectEvent zoneChange() {
        return new EffectEvent(EffectEvent.ZONE_CHANGE)
                .subject("E7").param("from_zone", "hand")
                .param("to_zone", "battlefield");
    }

    private static EffectEvent oneDamage() {
        return new EffectEvent(EffectEvent.DAMAGE_DEALT)
                .subject("E7").param("amount", 1).param("combat", true);
    }

    // ── snapshot depth ──────────────────────────────────────────────────

    /**
     * The collector snapshots at the run's depth, not at one of its own.
     *
     * <p>Checked where the vector lands rather than where it was read. Hardcoded
     * here, it made {@code state.tiers} say which collector wrote a record —
     * {@code combat} and {@code resolution} at {@code [1,2]} against
     * {@code [1,2,3]} everywhere else — which is collection metadata the schema
     * keeps out of a model's inputs.
     */
    @Test
    void theSnapshotDepthIsTheRunsAndNotTheCollectorsOwn() {
        assertArrayEquals(
                new int[]{1, 2, 3},
                collector(CollectionCaps.defaults()).snapshotTiers());

        CollectionCaps deep = new CollectionCaps(
                1, 0.1, 2, 2, List.of(), List.of(1, 2, 3, 4), 0.1);
        assertArrayEquals(new int[]{1, 2, 3, 4}, collector(deep).snapshotTiers());
    }

    /** And a caller that passes no caps still gets the run's, not a default here. */
    @Test
    void aCollectorBuiltWithoutCapsStillReadsTheRunsDepth() {
        writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        String previous = System.getProperty("effect.snapshot.tiers");
        System.setProperty("effect.snapshot.tiers", "1,2,3,4");
        try {
            assertArrayEquals(
                    new int[]{1, 2, 3, 4},
                    new BusBracketCollector(TestCards.game(), writer, "g")
                            .snapshotTiers());
        } finally {
            if (previous == null) {
                System.clearProperty("effect.snapshot.tiers");
            } else {
                System.setProperty("effect.snapshot.tiers", previous);
            }
        }
    }

    // ── resolution-time refs ────────────────────────────────────────────

    /**
     * The effect half's refs are read when the bracket closes, not at the cast.
     *
     * <p>A charm's chosen mode, an announced X and a named card are set by the
     * engine while the ability resolves. Read at the cast they are all absent,
     * which is why {@code refs.modes} was filled on 125 records out of 55,296;
     * read here they are there, on a board that is still the pre-event one.
     */
    @Test
    void theEffectHalfCarriesRefsReadAtResolution() throws IOException {
        BusBracketCollector collector = collector();
        SpellAbility bolt = damageAbility();
        Card victim = TestCards.build("Grizzly Bears");

        collector.beginBracket(bolt);
        // The engine chooses during resolution; this stands in for that.
        bolt.resetTargets();
        bolt.getTargets().add(victim);
        collector.endBracket(bolt.getId(), false);

        List<String> records = written();
        assertEquals(2, records.size(), records.toString());
        assertFalse(records.get(0).contains("\"targets\":[\"E" + victim.getId() + "\"]"),
                "the cost half describes the cast, before any of this existed");
        assertTrue(records.get(1).contains("\"targets\":[\"E" + victim.getId() + "\"]"),
                records.get(1));
    }
}
