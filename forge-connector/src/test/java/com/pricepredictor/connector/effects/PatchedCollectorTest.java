package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.PatchedCollectors.Contribution;
import forge.card.CardChangedType;
import forge.card.CardType;
import forge.card.ColorSet;
import forge.card.RemoveType;
import forge.card.StateChangedType;
import forge.card.WordChangedType;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.ability.AbilityFactory;
import forge.game.ability.AbilityUtils;
import forge.game.card.Card;
import forge.game.card.CounterType;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.replacement.ReplacementEffect;
import forge.game.replacement.ReplacementResult;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerHandler;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The caps and the hook lookup.
 *
 * <p>The collectors' behaviour inside a running match needs a patched checkout
 * and a real game, which is the integration test's job. What is testable here
 * is what the collectors decide before any game runs: whether a hook exists,
 * whether a mana ability is still under its cap, and whether a static has
 * already been recorded on this board.
 */
@ExtendWith(ForgeExtension.class)
class PatchedCollectorTest {

    @TempDir
    Path tempDir;

    private static CollectionCaps caps(int manaCap, double playabilityRate) {
        return new CollectionCaps(
                manaCap, playabilityRate, 2, 2, List.of(), 0.0, 0.0);
    }

    private static PatchedCollectors collectors(CollectionCaps caps) {
        return collectors(caps, 42L);
    }

    private static PatchedCollectors collectors(CollectionCaps caps, long seed) {
        return new PatchedCollectors(null, null, "game-1", caps, seed);
    }

    // ── the shared caps ─────────────────────────────────────────────────

    @Test
    void theDefaultCapsAreTheContracts() {
        CollectionCaps defaults = CollectionCaps.defaults();
        // One record per mana ability per game: this collector is built per
        // game, and a Mountain's dozen taps observe a board that barely moved.
        assertEquals(1, defaults.manaCap());
        assertEquals(0.1, defaults.playabilityRate());
        assertEquals(2, defaults.interventionsPerGame());
        assertEquals(2, defaults.probesPerGame());
        // A worker started without the property probes nothing; the eight
        // keywords a run probes by default are the supervisor's to send.
        assertTrue(defaults.probeKeywords().isEmpty());
    }

    /**
     * The caps arrive as system properties the supervisor sets.
     *
     * <p>Every one is a per-worker-process quantity the Python side cannot
     * observe, so the flags are inert unless this reads them — which is exactly
     * the state they were in before: parsed by the CLI and never sent.
     */
    @Test
    void capsAreReadFromSystemProperties() {
        Map<String, String> properties = Map.of(
                "effect.mana.cap", "7",
                "effect.playability.rate", "0.5",
                "effect.interventions.per.game", "3",
                "effect.probes.per.game", "4",
                "effect.probe.keywords", "wither, infect");
        properties.forEach(System::setProperty);
        try {
            CollectionCaps caps = CollectionCaps.fromSystemProperties();
            assertEquals(7, caps.manaCap());
            assertEquals(0.5, caps.playabilityRate());
            assertEquals(3, caps.interventionsPerGame());
            assertEquals(4, caps.probesPerGame());
            assertEquals(List.of("wither", "infect"), caps.probeKeywords());
        } finally {
            properties.keySet().forEach(System::clearProperty);
        }
    }

    @Test
    void anUnsetPropertyKeepsItsDefault() {
        // A worker started without instrumentation sets none of them, and one
        // started by an older supervisor may set only some.
        CollectionCaps caps = CollectionCaps.fromSystemProperties();
        assertEquals(CollectionCaps.defaults(), caps);
    }

    @Test
    void aMalformedPropertyKeepsItsDefaultRatherThanFailing() {
        System.setProperty(CollectionCaps.MANA_CAP, "not-a-number");
        PrintStream realErr = System.err;
        ByteArrayOutputStream warnings = new ByteArrayOutputStream();
        System.setErr(new PrintStream(warnings, true, StandardCharsets.UTF_8));
        try {
            assertEquals(
                    CollectionCaps.defaults().manaCap(),
                    CollectionCaps.fromSystemProperties().manaCap());
        } finally {
            System.setErr(realErr);
            System.clearProperty(CollectionCaps.MANA_CAP);
        }
        String warning = warnings.toString(StandardCharsets.UTF_8).strip();
        assertEquals(1, warning.lines().count(), warning);
        assertTrue(warning.contains(CollectionCaps.MANA_CAP) && warning.contains("not-a-number"),
                warning);
    }

    @Test
    void probesAreOffUntilKeywordsAreNamed() {
        assertFalse(CollectionCaps.defaults().probesEnabled());
        assertTrue(new CollectionCaps(
                2000, 0.1, 2, 2, List.of("wither"), 0.0, 0.0)
                .probesEnabled());
    }

    // ── the mana reservoir ──────────────────────────────────────────────

    @Test
    void theFirstActivationsFillTheReservoir() {
        ManaReservoir reservoir = new ManaReservoir(3, new Random(0));
        assertEquals(0, reservoir.slot("Mana$ G"));
        assertEquals(1, reservoir.slot("Mana$ G"));
        assertEquals(2, reservoir.slot("Mana$ G"));
    }

    @Test
    void theReservoirIsPerUniqueManaAbilityText() {
        // A basic land taps a dozen times a game; a rare mana ability must not
        // be starved by it.
        ManaReservoir reservoir = new ManaReservoir(1, new Random(0));
        assertEquals(0, reservoir.slot("Mana$ G"));
        assertEquals(0, reservoir.slot("Mana$ Any | Amount$ 2"));
    }

    @Test
    void aCapOfZeroRecordsNothing() {
        assertEquals(-1, new ManaReservoir(0, new Random(0)).slot("Mana$ G"));
    }

    /**
     * Every activation is equally likely to survive.
     *
     * <p>The property the reservoir exists for. Taking the first would make
     * every mana record describe turn one against an empty board, because that
     * is when a land's first tap happens — a bias introduced before collection
     * has even started, into the one input the model conditions on.
     */
    @Test
    void everyActivationIsEquallyLikelyToSurvive() {
        int activations = 8;
        int trials = 20000;
        int[] survivors = new int[activations];
        for (int trial = 0; trial < trials; trial++) {
            ManaReservoir reservoir = new ManaReservoir(1, new Random(PatchedCollectors.scramble(trial)));
            int held = -1;
            for (int i = 0; i < activations; i++) {
                if (reservoir.slot("Mana$ G") >= 0) {
                    held = i;
                }
            }
            survivors[held]++;
        }
        double expected = (double) trials / activations;
        for (int i = 0; i < activations; i++) {
            assertTrue(
                    Math.abs(survivors[i] - expected) < expected * 0.15,
                    "activation " + i + " survived " + survivors[i]
                            + " times, expected about " + expected);
        }
    }

    @Test
    void aLaterActivationCanReplaceAnEarlierOne() {
        // Otherwise the reservoir is just "keep the first", renamed.
        boolean replaced = false;
        for (int seed = 0; seed < 50 && !replaced; seed++) {
            ManaReservoir reservoir = new ManaReservoir(1, new Random(PatchedCollectors.scramble(seed)));
            reservoir.slot("Mana$ G");
            replaced = reservoir.slot("Mana$ G") == 0;
        }
        assertTrue(replaced, "no seed ever replaced the first activation");
    }

    // ── continuous coalescing ───────────────────────────────────────────

    /**
     * A static's record is keyed by the static and the board: one repeating
     * both is refused, and a change in either is a record of its own.
     */
    @ParameterizedTest(name = "{0}")
    @CsvSource({
            "a static is recorded once per stable board, anthem, board-1, false",
            "a changed board gets a new record, anthem, board-2, true",
            "two statics on one board each get a record, crusade, board-1, true"})
    void aContinuousRecordIsKeyedByItsStaticAndBoard(
            String why, String staticKey, String board, boolean allowed) {
        PatchedCollectors collector = collectors(caps(2000, 1.0));
        assertTrue(collector.allowContinuousRecord("anthem", "board-1"));
        assertEquals(allowed, collector.allowContinuousRecord(staticKey, board), why);
    }

    @Test
    void continuousRecordsTakeNoNumericCap() {
        // Coalescing per stable board is itself the cap (FR-029).
        PatchedCollectors collector = collectors(caps(1, 1.0));
        for (int board = 0; board < 100; board++) {
            assertTrue(collector.allowContinuousRecord("anthem", "board-" + board));
        }
    }

    // ── continuous type and colour channels ─────────────────────────────

    /**
     * A stand-in for the patch's own {@code Card.CardColor}.
     *
     * <p>The real one is nested in {@code Card} and only exists as a public type
     * on a patched checkout, which this module does not compile against. The
     * collector reads both components reflectively, so any public record with
     * the same two accessors exercises the same path — and a test that named the
     * real type would not compile on a stock checkout.
     */
    public record ColorChange(ColorSet color, boolean additional) {
    }

    private static String tokens(Contribution into, String field) {
        String json = into.toJson();
        int at = json.indexOf("\"" + field + "\":[");
        return json.substring(at + field.length() + 4, json.indexOf(']', at));
    }

    @Test
    void anAddedTypeIsBareAndARemovedOneIsDashed() {
        Contribution into = new Contribution("E1");
        PatchedCollectors.typeTokens(new CardChangedType(
                new CardType(List.of("Creature", "Elf"), true),
                new CardType(List.of("Land"), true),
                false, Set.of()), into);
        assertEquals("\"creature\",\"elf\",\"-land\"", tokens(into, "types"));
    }

    @Test
    void removingAWholeClassOfTypeIsNamedForTheClass() {
        // "loses all creature types" names no type, so it cannot be spelled as
        // a list of removals.
        Contribution into = new Contribution("E1");
        PatchedCollectors.typeTokens(new CardChangedType(
                null, null, true,
                Set.of(RemoveType.CreatureTypes, RemoveType.SuperTypes)), into);
        assertEquals(
                Set.of("\"all-creature-types\"", "\"-all-creature-types\"",
                        "\"-all-super-types\""),
                Set.of(tokens(into, "types").split(",")));
    }

    @Test
    void aStateChangeMarksTheTypeLineAsSetRatherThanAdded() {
        Contribution into = new Contribution("E1");
        PatchedCollectors.typeTokens(
                new StateChangedType(new CardType(List.of("Land"), true)), into);
        assertEquals("\"=\",\"land\"", tokens(into, "types"));
    }

    @Test
    void aTextChangeIsARemovalAndAnAdditionInOneEntry() {
        Contribution into = new Contribution("E1");
        PatchedCollectors.typeTokens(new WordChangedType("Forest", "Island"), into);
        assertEquals("\"-forest\",\"island\"", tokens(into, "types"));
    }

    @Test
    void anAdditionalColourIsJustTheLetter() {
        Contribution into = new Contribution("E1");
        PatchedCollectors.colorTokens(
                new ColorChange(ColorSet.fromNames("green"), true), into);
        assertEquals("\"G\"", tokens(into, "colors"));
    }

    @Test
    void aReplacingColourIsMarked() {
        Contribution into = new Contribution("E1");
        PatchedCollectors.colorTokens(
                new ColorChange(ColorSet.fromNames("green"), false), into);
        assertEquals("\"=\",\"G\"", tokens(into, "colors"));
    }

    @Test
    void replacingTheColourWithNothingIsColourless() {
        // An empty colour set and an empty token list are different answers:
        // one turns the permanent colourless, the other says nothing happened.
        Contribution into = new Contribution("E1");
        PatchedCollectors.colorTokens(
                new ColorChange(ColorSet.fromMask(0), false), into);
        assertEquals("\"=\",\"C\"", tokens(into, "colors"));
    }

    @Test
    void anUnreadableColourEntryContributesNothingRatherThanAReplacement() {
        // Guessing here would report every colour change as an overwrite.
        Contribution into = new Contribution("E1");
        PatchedCollectors.colorTokens(new Object(), into);
        assertEquals("", tokens(into, "colors"));
    }

    @Test
    void aStaticThatWritesTheSameTypeTwiceSaysItOnce() {
        // One static can write to more than one type layer.
        Contribution into = new Contribution("E1");
        CardChangedType change = new CardChangedType(
                new CardType(List.of("Creature"), true), null, false, Set.of());
        PatchedCollectors.typeTokens(change, into);
        PatchedCollectors.typeTokens(change, into);
        assertEquals("\"creature\"", tokens(into, "types"));
    }

    // ── continuous name channel ──────────────────────────────────────────

    /** {@code "name"} is the last field {@link Contribution#toJson()} writes. */
    private static String nameField(Contribution into) {
        String json = into.toJson();
        int at = json.indexOf("\"name\":");
        return json.substring(at + "\"name\":".length(), json.length() - 1);
    }

    /**
     * The channel this task exists for: {@code contributions[].name} is
     * declared and listed in {@code KNOWN_CONSTANT_FIELDS} because it had
     * never carried a value in any corpus.
     *
     * <p>Goes through the real {@code Card.addChangedName} and the real,
     * reflectively-read {@code Card.getChangedCardNames} rather than a
     * hand-built stand-in, because the defect this guards against is exactly
     * the gap between "compiles against a plausible shape" and "the engine
     * ever produces this".
     *
     * <p>Two competing entries, not one: with a single rename in the table,
     * {@code card.getName()} -- which walks every entry and returns the last
     * overwrite -- answers the same string as that one static's own cell, so
     * a {@code contribution(...).name(card.getName())} implementation
     * (whole-card, not this cell) would pass unnoticed. The later static
     * ({@code winningStatic}) is what {@code getName()} reports; asserting
     * that the earlier one ({@code supersededStatic}) still carries its own,
     * superseded name is what only a per-cell read can produce. This also
     * covers two statics renaming the same card, which otherwise had no test
     * at all.
     */
    @Test
    void aChangedNameContributesTheNewName() {
        Card card = TestCards.build("Grizzly Bears");
        long supersededStatic = 1111L;
        long winningStatic = 2222L;
        card.addChangedName("Superseded-By-Later-Static", false, 3L, supersededStatic);
        card.addChangedName("Winning-Name", false, 7L, winningStatic);
        // The premise the whole-card shortcut would rely on, and get wrong
        // for the superseded static below.
        assertEquals("Winning-Name", card.getName());

        Map<Long, Map<String, Contribution>> byStatic = new LinkedHashMap<>();
        PatchedCollectors.nameContributions(card, byStatic, "E1");

        assertEquals("\"Superseded-By-Later-Static\"",
                nameField(byStatic.get(supersededStatic).get("E1")));
        assertEquals("\"Winning-Name\"",
                nameField(byStatic.get(winningStatic).get("E1")));
    }

    /**
     * The channel's other half: most statics contribute no name, and null is
     * the correct, meaningful answer there -- not an empty string, and not a
     * dropped key.
     */
    @Test
    void aCardWithNoNameChangeRendersNullName() {
        Card card = TestCards.build("Grizzly Bears");
        Map<Long, Map<String, Contribution>> byStatic = new LinkedHashMap<>();
        PatchedCollectors.nameContributions(card, byStatic, "E1");
        assertTrue(byStatic.isEmpty(), "no static wrote a name, so nothing is filed");
        assertEquals("null", nameField(new Contribution("E1")));
    }

    @Test
    void anUnreadableNameEntryContributesNoName() {
        // Guessing here would report every unrelated object as a rename.
        Contribution into = new Contribution("E1");
        PatchedCollectors.nameTokens(new Object(), into);
        assertEquals("null", nameField(into));
    }

    // ── the trigger and rewrite records ──────────────────

    /**
     * A collector over a game nothing is played in.
     *
     * <p>Enough for a snapshot and a record id, which is all these need: what
     * is under test is what the collector reads off the trait the hook hands
     * it, and that read is the same whether or not a turn has been taken.
     */
    private PatchedCollectors recording() {
        return new PatchedCollectors(
                TestCards.game(),
                new RecordShardWriter(tempDir, "run", 0, "l1"),
                "run.0-l1.0", CollectionCaps.defaults(), 1L);
    }

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

        String json = recording()
                .triggerRecord(triggers.get(1), Map.of(), true).toJson();

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

        String json = recording()
                .triggerRecord(triggersOf(paralyze).get(0), Map.of(), false).toJson();

        assertTrue(json.contains("\"source\":\"" + entity + "\""), json);
        // Tier 1 carries the host whatever zone it sits in. Without that a
        // dies trigger names an entity id absent from state.entities, and the
        // reader cannot follow a source it cannot find.
        assertTrue(json.contains("\"id\":\"" + entity + "\""),
                "the host must be in state.entities: " + json);
    }

    /**
     * The other half of the same omission: 62,546 rewrite records that named
     * no replacement effect, on the one hook every replacement passes through.
     */
    @Test
    void aRewriteRecordNamesItsReplacementEffect() {
        // Rest in Peace rather than Paralyze: its replacement redirects a move,
        // so this is the in-place rewrite the pair was always able to express,
        // and every other field on the record can be read beside it.
        Card rest = TestCards.build("Rest in Peace");
        ReplacementEffect replacement =
                rest.getCurrentState().getReplacementEffects().iterator().next();

        String json = recording().rewriteRecord(
                replacement,
                Map.of("Card", rest, "Destination", ZoneType.Graveyard),
                Map.of("Card", rest, "Destination", ZoneType.Exile),
                ReplacementResult.Replaced, null).toJson();

        assertTrue(json.contains("\"kind\":\"rewrite\""), json);
        assertTrue(json.contains("\"trait_kind\":\"replacement\""), json);
        assertTrue(json.contains("cardsfolder/r/rest_in_peace.txt"), json);
        assertTrue(json.contains(
                "\"source\":\"" + SnapshotBuilder.entityId(rest) + "\""), json);
        // And it says what the replacement actually did, which is the whole of
        // the rewrite channel's defect: a graveyard destination in, an exile
        // destination out.
        assertTrue(json.contains("\"incoming\":{\"type\":\"zone_change\""), json);
        assertTrue(json.contains("\"to_zone\":\"graveyard\""), json);
        assertTrue(json.contains("\"to_zone\":\"exile\""), json);
    }

    /**
     * A prevention that changes no parameter is a record, not a loss.
     *
     * <p>Paralyze's "doesn't untap" is the shape the whole contract is for: the
     * engine skips the event rather than rewriting a value, so both halves read
     * alike and the record used to be dropped. It was 87% of the channel. The
     * result is what makes the row readable without an outgoing — {@code skipped}
     * says outright what an identity pair could only imply — so the record is
     * written, and its outgoing is null rather than a copy of its incoming.
     */
    @Test
    void aPreventionThatRewritesNothingIsStillARecord() {
        Card paralyze = TestCards.build("Paralyze");
        ReplacementEffect prevention =
                paralyze.getCurrentState().getReplacementEffects().iterator().next();

        EffectRecord record = recording().rewriteRecord(
                prevention, Map.of("Card", paralyze), Map.of("Card", paralyze),
                ReplacementResult.Skipped, null);

        assertNotNull(record, "a skipped replacement is this channel's negative");
        String json = record.toJson();
        assertTrue(json.contains("\"result\":\"skipped\""), json);
        assertTrue(json.contains("\"outgoing\":null"),
                "null, never a copy of incoming: " + json);
        assertTrue(json.contains("\"replaced_by\":[]"),
                "a skip returns above the ability that would have run: " + json);
    }

    /**
     * A hook whose signature drifted must yield a keyless record rather than a
     * ClassCastException inside a dynamic proxy, which the engine would surface
     * as an UndeclaredThrowableException in the middle of canRunTrigger.
     */
    @Test
    void anArgumentThatIsNotATraitStillProducesARecord() {
        String json = recording().triggerRecord("not a trait", Map.of(), true).toJson();

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
     * A record that names its line carries no reason, and the reverse.
     *
     * <p>The two fields are one statement in two halves. A reason beside a
     * named line is a contradiction the Python reader refuses outright, so the
     * collector may never write both.
     */
    @Test
    void aRecordThatNamesItsLineCarriesNoReason() {
        Card rest = TestCards.build("Rest in Peace");
        ReplacementEffect replacement =
                rest.getCurrentState().getReplacementEffects().iterator().next();

        String json = recording().rewriteRecord(
                replacement,
                Map.of("Card", rest, "Destination", ZoneType.Graveyard),
                Map.of("Card", rest, "Destination", ZoneType.Exile),
                ReplacementResult.Replaced, null).toJson();

        assertTrue(json.contains("cardsfolder/r/rest_in_peace.txt"), json);
        assertTrue(json.contains("\"ability_unresolved\":null"), json);
    }

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

        String json = recording().triggerRecord(draw, Map.of(), true).toJson();

        assertTrue(json.contains("\"ability\":[]"), json);
        assertTrue(json.contains("\"ability_unresolved\":\"engine_effect\""), json);
    }

    // ── per-clause attribution ─────────────────────────

    /** A three-clause line: gain life, then draw, then scry. */
    private static SpellAbility threeClauseAbility() {
        Card host = TestCards.build("Fountain of Youth");
        host.setSVar("DBDraw",
                "DB$ Draw | Defined$ You | NumCards$ 1 | SubAbility$ DBScry");
        host.setSVar("DBScry", "DB$ Scry | Defined$ You | ScryNum$ 1");
        SpellAbility root = AbilityFactory.getAbility(
                "AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1"
                        + " | SubAbility$ DBDraw", host);
        assertNotNull(root.getSubAbility(), "the chain has to be three deep");
        assertNotNull(root.getSubAbility().getSubAbility());
        return root;
    }

    /**
     * Each clause of a multi-clause line is its own index.
     *
     * <p>The engine's pointer was set only around the line the stack resolved,
     * so every clause after the first was attributed to the root: 298 of 74,952
     * events carried an {@code attributed_to} at all, and those were the dozen
     * branching effects that resolve a sub-ability through
     * {@code AbilityUtils.resolve}. "Put a counter" and "draw a card" are
     * exactly the distinction the per-entity head learns, so a whole line
     * attributed to its root teaches neither.
     */
    @Test
    void eachClauseOfAMultiClauseAbilityAttributesToItself() {
        SpellAbility root = threeClauseAbility();
        AbilitySub first = (AbilitySub) root.getSubAbility();
        AbilitySub second = (AbilitySub) first.getSubAbility();

        assertEquals("0", EventAttribution.attributedTo(root, first));
        assertEquals("1", EventAttribution.attributedTo(root, second));
    }

    /**
     * The root acting and no pointer at all are different answers.
     *
     * <p>Both used to be null, which is why the 3.2% of events carrying an
     * attribution could not be read: a single-clause ability legitimately has
     * no sub-ability and the root is the right answer, and that looked exactly
     * like the pointer failing to land. Separating them is what turns the rate
     * into a measurement instead of a number.
     */
    @Test
    void theRootActingAndNoPointerAreDifferentAnswers() {
        SpellAbility root = threeClauseAbility();

        assertEquals(EventAttribution.ROOT,
                EventAttribution.attributedTo(root, root));
        assertEquals(EventAttribution.UNRESOLVED,
                EventAttribution.attributedTo(root, null));
        assertNotEquals(EventAttribution.ROOT, EventAttribution.UNRESOLVED);
    }

    /** Neither sentinel can be mistaken for a chain index. */
    @Test
    void neitherSentinelReadsAsAClauseIndex() {
        for (String sentinel
                : List.of(EventAttribution.ROOT, EventAttribution.UNRESOLVED)) {
            assertThrows(NumberFormatException.class,
                    () -> Integer.parseInt(sentinel), sentinel);
        }
    }

    /**
     * A clause with no line above it names nothing a reader could join to.
     *
     * <p>Its index counts clauses of a chain this collector never saw, so the
     * number would address a line in some other script. Saying so is the third
     * state, not the root.
     */
    @Test
    void aDetachedClauseCannotBePlacedAndSaysSo() {
        SpellAbility root = threeClauseAbility();
        AbilitySub orphan = (AbilitySub) root.getSubAbility();
        orphan.setParent(null);

        assertEquals(EventAttribution.UNRESOLVED,
                EventAttribution.attributedTo(root, orphan));
    }

    /** And attribution never returns null, whatever it is handed. */
    @Test
    void attributionAlwaysNamesOneOfTheThreeStates() {
        SpellAbility root = threeClauseAbility();

        for (Object pointer
                : new Object[]{null, "not an ability", root, root.getSubAbility()}) {
            assertNotNull(EventAttribution.attributedTo(root, pointer),
                    String.valueOf(pointer));
        }
    }

    /** And an event stamped inside a clause carries that clause's index. */
    @Test
    void anEventCarriesTheClauseThatProducedIt() {
        SpellAbility root = threeClauseAbility();
        AbilitySub second = (AbilitySub) root.getSubAbility().getSubAbility();

        String json = EventAttribution.stamp(
                new EffectEvent(EffectEvent.CARD_DRAWN).subject("P0"),
                root, second).toJson();

        assertTrue(json.contains("\"attributed_to\":\"1\""), json);
    }

    /**
     * A clause's own {@code Duration$} outranks the line's.
     *
     * <p>Both fields come from one read of the pointer, so they cannot describe
     * different instants of a nested resolution.
     */
    @Test
    void aClauseDeclaringItsOwnDurationOutranksTheLine() {
        Card host = TestCards.build("Fountain of Youth");
        host.setSVar("DBPump",
                "DB$ Pump | Defined$ Self | NumAtt$ 1 | Duration$ Permanent");
        SpellAbility root = AbilityFactory.getAbility(
                "AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1"
                        + " | SubAbility$ DBPump", host);

        assertEquals(EventAttribution.PERMANENT,
                EventAttribution.duration(root, root.getSubAbility()));
        assertEquals(EventAttribution.INSTANT,
                EventAttribution.duration(root, root));
    }

    /**
     * An undeclared duration means what Forge does with it. A pump with no
     * {@code Duration$} ends at end of turn — Giant Growth — and a counter
     * lasts; only an outcome with no lasting state is instant.
     */
    @Test
    void anUndeclaredDurationIsTheOneForgeApplies() {
        Card host = TestCards.build("Grizzly Bears");
        SpellAbility pump = AbilityFactory.getAbility(
                "SP$ Pump | ValidTgts$ Creature | NumAtt$ +3 | NumDef$ +3", host);
        SpellAbility counter = AbilityFactory.getAbility(
                "SP$ PutCounter | Defined$ Self | CounterType$ P1P1 | CounterNum$ 1", host);

        assertEquals(EventAttribution.END_OF_TURN,
                EventAttribution.duration(pump, pump, EffectEvent.PT_CHANGE));
        assertEquals(EventAttribution.PERMANENT,
                EventAttribution.duration(counter, counter, EffectEvent.PT_CHANGE));
        assertEquals(EventAttribution.INSTANT,
                EventAttribution.duration(pump, pump, EffectEvent.DAMAGE_DEALT));

        String json = EventAttribution.stamp(
                new EffectEvent(EffectEvent.PT_CHANGE).subject("E1"), pump, pump).toJson();
        assertTrue(json.contains("\"duration\":\"end_of_turn\""), json);
    }


    // ── what a run parameter map says ──────────────────────────────────

    /**
     * A stand-in for a trait whose only readable property is its mode.
     *
     * <p>The collector reads the mode reflectively, so anything with the
     * accessor exercises the same path. Using one here keeps these tests about
     * the parameter reading rather than about which Forge trait happens to
     * declare a given mode.
     */
    public record FakeTrait(String mode) {
        public String getMode() {
            return mode;
        }
    }

    /** The event a trigger record carries, as rendered JSON. */
    private String eventJson(String mode, Map<String, Object> params) {
        String json = recording().triggerRecord(new FakeTrait(mode), params, true).toJson();
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
        String json = eventJson("Moved", Map.of(
                "Origin", ZoneType.Hand,
                "Destination", ZoneType.Graveyard,
                "Card", TestCards.build("Mountain")));

        assertTrue(json.contains("\"type\":\"zone_change\""), json);
        assertTrue(json.contains("\"from_zone\":\"hand\""), json);
        assertTrue(json.contains("\"to_zone\":\"graveyard\""), json);
    }

    /** And a rewrite that changed the destination is two different halves. */
    @Test
    void aRewrittenDestinationMakesTheTwoHalvesDiffer() {
        Card card = TestCards.build("Mountain");
        Map<String, Object> before = new LinkedHashMap<>(Map.of(
                "Origin", ZoneType.Battlefield, "Destination", ZoneType.Graveyard,
                "Card", card));
        Map<String, Object> after = new LinkedHashMap<>(before);
        after.put("Destination", ZoneType.Exile);

        EffectRecord record = recording().rewriteRecord(
                new FakeTrait("Moved"), before, after,
                ReplacementResult.Updated, null);

        assertNotNull(record, "a replacement that redirected a card is a real rewrite");
        String json = record.toJson();
        assertTrue(json.contains("\"to_zone\":\"graveyard\""), json);
        assertTrue(json.contains("\"to_zone\":\"exile\""), json);
    }

    /**
     * A rewrite whose halves read alike says so with a null, not with a copy.
     *
     * <p>Writing the copy is what made the channel unreadable: 87% of the
     * candidates were a substitution whose two halves were byte-identical
     * because Forge substitutes by running another ability rather than by
     * editing the map. Null says "not rewritten" outright, and the corpus
     * validator judges an identity pair at zero rather than letting one parse.
     */
    @Test
    void aRewriteThatChangedNoObservableParameterCarriesNoOutgoing() {
        Map<String, Object> params = Map.of("Card", TestCards.build("Mountain"));

        String json = recording().rewriteRecord(
                new FakeTrait("Moved"), params, params,
                ReplacementResult.Replaced, null).toJson();

        assertTrue(json.contains("\"outgoing\":null"), json);
        assertFalse(json.contains("\"outgoing\":{"),
                "never a copy of incoming: " + json);
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
        String json = eventJson("Moved", Map.of(
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
        String json = eventJson("Moved", Map.of(
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
        String json = eventJson("SomethingForgeCallsThis", Map.of());

        assertTrue(json.contains("\"type\":\"state_flag_change\""), json);
        assertTrue(json.contains("\"params\":{\"mode\":\"SomethingForgeCallsThis\"}"),
                json);
    }

    @Test
    void damageIsReadAsAnAmountACombatFlagAndASource() {
        Card source = TestCards.build("Lightning Bolt");
        String json = eventJson("DamageDone", Map.of(
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
        String json = eventJson("DamageDone", Map.of(
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
        String json = eventJson("CounterAdded", Map.of(
                "CounterMap", Map.of(Optional.empty(), counters),
                "Affected", TestCards.build("Mountain")));

        assertTrue(json.contains("\"type\":\"counter_change\""), json);
        // Named the way the engine names it, which is the same spelling the
        // observed counter events carry -- a reader must not have to know which
        // collector wrote a record to know what "+1/+1" is called.
        assertTrue(json.contains("\"counter_type\":\"+1/+1\""), json);
        assertTrue(json.contains("\"delta\":2"), json);
    }

    // ── the trigger negative sample ────────────────────────────────────

    /**
     * Drive one mode's evaluations and count what the writer was offered.
     *
     * @param burst how many non-fired evaluations follow each firing
     */
    private static int[] driveTrigger(
            PatchedCollectors collector, String mode, int burst, int firings) {
        int[] kept = new int[2];
        for (int firing = 0; firing < firings; firing++) {
            if (collector.offerTriggerEvaluation(mode, true)) {
                collector.keepTriggerEvaluation(mode, true);
                kept[0]++;
            }
            for (int i = 0; i < burst; i++) {
                if (collector.offerTriggerEvaluation(mode, false)) {
                    collector.keepTriggerEvaluation(mode, false);
                    kept[1]++;
                }
            }
        }
        return kept;
    }

    /**
     * The kept negatives track the kept positives, whatever the population is.
     *
     * <p>The first corpus used a fixed 2% rate and landed at 74:26 rather than
     * 1:1, because a fixed rate cannot hold a ratio: the true population was
     * 1:17.7 and it varies by mode, board and turn. Two modes with very
     * different burst lengths are driven here, and both have to come out level
     * — a fixed rate fails this by construction, at 1:0.8 for the short burst
     * and 1:0.06 for the long one.
     */
    @Test
    void negativesAreKeptAtAboutOneToOneWhateverTheBurstLength() {
        PatchedCollectors collector = collectors(caps(1, 1.0));
        int[] rare = driveTrigger(collector, "ChangesZone", 40, 200);
        int[] common = driveTrigger(collector, "TapsForMana", 3, 200);

        for (int[] kept : List.of(rare, common)) {
            assertTrue(kept[0] > 0 && kept[1] > 0, "nothing was kept");
            double ratio = (double) kept[1] / kept[0];
            assertTrue(Math.abs(ratio - 1.0) < 0.1,
                    "kept " + kept[0] + " fired against " + kept[1]
                            + " not fired, a ratio of " + ratio);
        }
    }

    /**
     * The ratio holds over what reaches the shard, not over what was offered.
     *
     * <p>This is the 58:42 the corpus measured while the unit test above read
     * 1:1. The difference is here: an offered negative is refused downstream by
     * the duplicate check or the per-line flood cap far more often than a
     * positive is, and under the old rate the deficit that leaves standing was
     * repaid only at {@code deficit / estimated run length} — a lag a game's
     * handful of firings per mode never works off. Three refusals in four is
     * harsher than the corpus and the kept sets still come out level.
     */
    @Test
    void negativesStayLevelEvenWhenMostOfThemAreRefusedDownstream() {
        PatchedCollectors collector = collectors(caps(1, 1.0));
        java.util.Random refusals = new java.util.Random(7);
        int[] kept = new int[2];
        for (int firing = 0; firing < 400; firing++) {
            if (collector.offerTriggerEvaluation("ChangesZone", true)) {
                collector.keepTriggerEvaluation("ChangesZone", true);
                kept[0]++;
            }
            for (int i = 0; i < 20; i++) {
                if (!collector.offerTriggerEvaluation("ChangesZone", false)) {
                    continue;
                }
                // The record was built and then refused: nothing is kept, and
                // the deficit it was drawn against is still owed.
                if (refusals.nextInt(4) != 0) {
                    continue;
                }
                collector.keepTriggerEvaluation("ChangesZone", false);
                kept[1]++;
            }
        }
        double ratio = (double) kept[1] / kept[0];
        assertTrue(Math.abs(ratio - 1.0) < 0.1,
                "kept " + kept[0] + " fired against " + kept[1]
                        + " not fired, a ratio of " + ratio);
    }

    /**
     * A deficit that cannot be filled costs a few offers, not all of them.
     *
     * <p>The price of offering on the deficit alone: a mode whose every acting
     * line has hit its per-line cap owes a negative for ever, and each offer
     * builds a snapshot on the hook that fires most often in the collector.
     * The backoff is what keeps that bounded.
     */
    @Test
    void aDeficitThatCanNeverBeFilledStopsSpendingWorkOnItself() {
        PatchedCollectors collector = collectors(caps(1, 1.0));
        collector.offerTriggerEvaluation("ChangesZone", true);
        collector.keepTriggerEvaluation("ChangesZone", true);

        int offered = 0;
        for (int i = 0; i < 20_000; i++) {
            if (collector.offerTriggerEvaluation("ChangesZone", false)) {
                offered++;   // built, refused downstream, never kept
            }
        }
        assertTrue(offered > 0, "the standing deficit is never abandoned");
        assertTrue(offered < 100,
                "20,000 unfillable evaluations built " + offered + " records");
    }

    /** A mode's balance is its own, so a busy mode cannot starve a quiet one. */
    @Test
    void oneModeDoesNotSpendAnothersBudget() {
        PatchedCollectors collector = collectors(caps(1, 1.0));
        driveTrigger(collector, "ChangesZone", 40, 200);
        int[] quiet = driveTrigger(collector, "Untaps", 2, 5);

        assertTrue(quiet[1] > 0,
                "a quiet mode kept no negatives after a busy one ran");
    }

    /** A fired evaluation is always offered; only negatives are sampled. */
    @Test
    void everyFiredEvaluationIsOffered() {
        PatchedCollectors collector = collectors(caps(1, 1.0));
        for (int i = 0; i < 50; i++) {
            assertTrue(collector.offerTriggerEvaluation("ChangesZone", true));
        }
    }

    /** And a mode that never fires contributes no negatives, by construction. */
    @Test
    void aModeThatNeverFiresKeepsNoNegatives() {
        PatchedCollectors collector = collectors(caps(1, 1.0));
        for (int i = 0; i < 500; i++) {
            assertFalse(collector.offerTriggerEvaluation("Untaps", false));
        }
    }

    /**
     * A replacement that halved a draw is a rewrite, not an identity pair.
     *
     * <p>The mode table was written from the trigger vocabulary, so 31 of
     * Forge's 42 {@code ReplacementType}s rendered as the generic outcome
     * carrying only their mode -- and an event with no value in it cannot
     * differ from itself. Every rewrite of a draw, a mill, a token count or a
     * life total was therefore dropped as an identity pair by construction,
     * whatever the replacement actually did.
     */
    @Test
    void aRewrittenQuantityOnTheReplacementSideIsNoLongerAnIdentityPair() {
        Card card = TestCards.build("Mountain");
        Map<String, Object> before = new LinkedHashMap<>(Map.of(
                "Affected", card, "Number", 3));
        Map<String, Object> after = new LinkedHashMap<>(before);
        after.put("Number", 1);

        EffectRecord record = recording().rewriteRecord(
                new FakeTrait("DrawCards"), before, after,
                ReplacementResult.Updated, null);

        assertNotNull(record, "a replacement that halved a draw is a real rewrite");
        String json = record.toJson();
        assertTrue(json.contains("\"type\":\"card_drawn\""), json);
        assertTrue(json.contains("\"count\":3"), json);
        assertTrue(json.contains("\"count\":1"), json);
    }

    /** And the replacement side's word for a life total is read as one. */
    @Test
    void theReplacementSpellingOfALifeAmountIsStillADelta() {
        String json = eventJson("LifeReduced", Map.of(
                "Affected", TestCards.build("Mountain"), "Amount", 4));

        assertTrue(json.contains("\"type\":\"life_change\""), json);
        assertTrue(json.contains("\"delta\":4"), json);
    }

    // ── what the rewrite channel wrote ─────────────────────────────────

    /**
     * A record with no outgoing names the parameters that moved under it.
     *
     * <p>The count alone says the channel filled; this says whether the rows in
     * it are as thin as they look or whether the normaliser cannot express what
     * the replacement changed, and it hands over the key that would close the
     * gap.
     */
    @Test
    void aRecordWithNoOutgoingNamesTheParametersTheNormaliserDidNotRead() {
        RewriteTally tally = new RewriteTally();
        Card card = TestCards.build("Mountain");
        tally.record("Moved", "replaced", false, false,
                Map.of("Card", card, "LibraryPosition", 0),
                Map.of("Card", card, "LibraryPosition", -1));

        String summary = tally.summary();
        assertTrue(summary.contains("Moved.LibraryPosition=1"), summary);
        assertTrue(summary.contains(
                "1 carried no outgoing while a raw parameter moved"), summary);
    }

    /** And one that moved nothing at all is not counted against the reader. */
    @Test
    void aRecordThatMovedNothingIsNoNormaliserGap() {
        RewriteTally tally = new RewriteTally();
        Map<String, Object> same = Map.of("Card", TestCards.build("Mountain"));
        tally.record("Untap", "skipped", false, false, same, same);

        String summary = tally.summary();
        assertTrue(summary.contains(
                "0 carried no outgoing while a raw parameter moved: none"), summary);
    }

    /**
     * The headline is the volume and the results, because that is the question.
     *
     * <p>The channel held 34 records in 1.88M against a 7% share of the
     * mixture, and the operator needs to know in the first hour whether it
     * filled. A breakdown that is all {@code ?} is a Forge jar whose listener
     * predates the result argument, which reads identically to a wired one on
     * every other line of the summary.
     */
    @Test
    void theSummaryLeadsWithTheVolumeAndTheResults() {
        RewriteTally tally = new RewriteTally();
        Map<String, Object> same = Map.of("Card", TestCards.build("Mountain"));
        tally.record("Moved", "updated", true, true, same, same);
        for (int i = 0; i < 9; i++) {
            tally.record("Moved", "not_replaced", false, false, same, same);
        }

        String summary = tally.summary();
        assertTrue(summary.contains(
                "10 records, 1 rewrote a parameter in place (10.0%), "
                        + "1 named the ability that ran instead"), summary);
        assertTrue(summary.contains("by result: not_replaced=9, updated=1"), summary);
        assertTrue(summary.contains("by mode and result: Moved.not_replaced=9"),
                summary);
    }

    /** A listener that never learnt the result says so rather than guessing. */
    @Test
    void aResultlessRecordIsItsOwnRowInTheBreakdown() {
        RewriteTally tally = new RewriteTally();
        Map<String, Object> same = Map.of("Card", TestCards.build("Mountain"));
        tally.record("Moved", null, false, false, same, same);

        assertTrue(tally.summary().contains("by result: ?=1"), tally.summary());
    }

    /**
     * A worker that saw no replacement says nothing.
     *
     * <p>Whether the hook is installed at all is {@code PatchHooks.report()}'s
     * line, and repeating it once a game would be noise in every worker log.
     */
    @Test
    void aGameWithNoReplacementPrintsNothing() {
        RewriteTally tally = new RewriteTally();
        java.io.PrintStream out = System.out;
        java.io.ByteArrayOutputStream captured = new java.io.ByteArrayOutputStream();
        System.setOut(new java.io.PrintStream(captured));
        try {
            tally.endOfGame();
        } finally {
            System.setOut(out);
        }
        assertEquals("", captured.toString());
    }

    /** The live path feeds the tally, which is what makes it a live number. */
    @Test
    void everyReplacementReachesTheWorkerTally() {
        Card card = TestCards.build("Mountain");
        Map<String, Object> before = new LinkedHashMap<>(Map.of(
                "Origin", ZoneType.Battlefield, "Destination", ZoneType.Graveyard,
                "Card", card));
        Map<String, Object> after = new LinkedHashMap<>(before);
        after.put("Destination", ZoneType.Exile);
        // Deltas rather than absolutes: the tally is the worker's, so every
        // other test in this class has already contributed to it.
        long recordsBefore = PatchedCollectors.JVM_REWRITES.records();
        PatchedCollectors collector = recording();

        collector.rewriteRecord(
                new FakeTrait("Moved"), before, after,
                ReplacementResult.Updated, null);
        collector.rewriteRecord(
                new FakeTrait("Moved"), before, before,
                ReplacementResult.NotReplaced, null);

        // Two records for two replacements: the one that declined is counted
        // too, because it is no longer dropped.
        assertEquals(recordsBefore + 2, PatchedCollectors.JVM_REWRITES.records());
    }

    // ── coalescing the kinds that had none ─────────────────────────────

    /**
     * A decision is keyed by what it says and the board it was asked on: the
     * same answer on an unchanged board is written once, and a board that
     * moved gets its own record.
     */
    @ParameterizedTest(name = "{0}")
    @CsvSource({
            "a decision on an unchanged board is written once, '{life:20}', false",
            "a board that moved gets its own decision record, '{life:19}', true"})
    void aDecisionIsKeyedByItsPayloadAndBoard(String why, String board, boolean allowed) {
        PatchedCollectors collector = collectors(caps(2000, 1.0));
        assertTrue(collector.allowDistinctRecord("decision", "{payload}", "{life:20}"));
        assertEquals(allowed, collector.allowDistinctRecord("decision", "{payload}", board), why);
    }

    /**
     * Two copies of one anthem render one record.
     *
     * <p>The per-board coalescer cannot see this: the two statics have different
     * ids, so it treats them as different questions, while their key, their
     * contributions and their suppressed board are all the same.
     */
    @Test
    void twoStaticsThatRenderIdenticallyAreWrittenOnce() {
        PatchedCollectors collector = collectors(caps(2000, 1.0));
        assertTrue(collector.allowContinuousRecord("11", "board-1"));
        assertTrue(collector.allowContinuousRecord("12", "board-1"));
        assertTrue(collector.allowDistinctRecord("continuous", "{anthem}", "{board}"));
        assertFalse(collector.allowDistinctRecord("continuous", "{anthem}", "{board}"));
    }

    /**
     * The legality cap keys on the snapshot as well as the payload (FR-028).
     *
     * <p>The AI's re-asks within one combat still collapse -- the board does
     * not move while it evaluates, so their snapshots render identically --
     * while the same answer on a board that moved is kept, which the
     * payload-only key of gen-1 dropped as a repeat (spec Story 3 scenario 10).
     */
    @Test
    void aLegalityAnswerCoalescesOnPayloadAndSnapshotTogether() {
        PatchedCollectors collector = collectors(caps(2000, 1.0));
        assertTrue(collector.allowDistinctRecord("blockers", "{payload}", "{board-1}"));
        assertFalse(collector.allowDistinctRecord("blockers", "{payload}", "{board-1}"));
        assertTrue(collector.allowDistinctRecord("blockers", "{payload}", "{board-2}"));
        assertTrue(collector.allowDistinctRecord("attackers", "{payload}", "{board-1}"));
    }

    /** One acting line is worth a couple of dozen looks in a game, not hundreds. */
    @Test
    void oneTriggerLineIsCappedPerGameAndPerVerdict() {
        PatchedCollectors collector = collectors(caps(2000, 1.0));
        int written = 0;
        for (int i = 0; i < 100; i++) {
            if (collector.allowTriggerLine("[key]", true)) {
                written++;
            }
        }
        assertEquals(24, written);
        // The other verdict has its own budget: "fired" and "did not fire" are
        // different observations of the same line.
        assertTrue(collector.allowTriggerLine("[key]", false));
    }

    // ── the snapshot depth is the same on every record ─────────────────

    @Test
    void everySnapshotCarriesAllFourTiers() {
        PatchedCollectors collector = new PatchedCollectors(
                TestCards.game(),
                new RecordShardWriter(tempDir, "run", 0, "l1"),
                "run.0-l1.0", CollectionCaps.defaults(), 1L);

        String json = collector.triggerRecord(
                new FakeTrait("Moved"), Map.of(), true).toJson();

        assertTrue(json.contains("\"tiers\":[1,2,3,4]"), json);
    }

    // ── held probe branches ────────────────────────────────────────────

    private static ForkCollector.HeldProbe branch(String substep) {
        return new ForkCollector.HeldProbe(
                "trample", "E1", "{}", List.of(), "P0", "\"combat\":{}", substep,
                ForkCollector.TURN_UNKNOWN);
    }

    private PatchedCollectors withForks(RecordShardWriter writer) {
        CollectionCaps probing = new CollectionCaps(
                1, 1.0, 2, 2, List.of("trample"), 0.0, 0.0);
        PatchedCollectors collector = new PatchedCollectors(
                TestCards.game(), writer, "run.0-l1.0", probing, 1L);
        collector.withForks(
                new ForkCollector(null, writer, "run.0-l1.0", probing, 1L));
        return collector;
    }

    /**
     * A branch is completed only against its own damage step.
     *
     * <p>In the first corpus it was not: a branch forked at the first-strike
     * step, where nothing was ever assigned, survived that step and was written
     * against the regular step's record — so the sampled fork record whose phase
     * was {@code combat_first_strike_damage} carried a {@code mirror_of} whose
     * phase was {@code combat_damage}, and gate 2's difference compared a step
     * that did not happen with one that did.
     */
    @Test
    void aBranchIsCompletedOnlyAgainstItsOwnDamageStep() throws Exception {
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1")) {
            PatchedCollectors collector = withForks(writer);
            collector.hold(branch(DamageStep.FIRST_STRIKE.wireName()));
            collector.hold(branch(DamageStep.REGULAR.wireName()));

            collector.writeHeldProbes("run.0-l1.9", DamageStep.REGULAR.wireName());
            // The first-strike branch must still be waiting for its own step.
            collector.writeHeldProbes("run.0-l1.10", DamageStep.FIRST_STRIKE.wireName());
        }

        List<String> lines = ShardRun.readShard(
                tempDir.resolve("run.0-l1" + RecordShardWriter.SUFFIX));
        assertEquals(2, lines.size(), lines.toString());
        assertTrue(lines.get(0).contains("\"mirror_of\":\"run.0-l1.9\""), lines.get(0));
        assertTrue(lines.get(1).contains("\"mirror_of\":\"run.0-l1.10\""), lines.get(1));
    }

    /** With no substep named, the oldest held step is the one being written. */
    @Test
    void theOldestHeldStepIsTheOneACombatRecordCompletes() throws Exception {
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1")) {
            PatchedCollectors collector = withForks(writer);
            collector.hold(branch(DamageStep.FIRST_STRIKE.wireName()));
            collector.hold(branch(DamageStep.REGULAR.wireName()));

            collector.writeHeldProbes("run.0-l1.9");
            // One branch is left, and it is the regular step's.
            collector.writeHeldProbes("run.0-l1.10", DamageStep.FIRST_STRIKE.wireName());
            collector.writeHeldProbes("run.0-l1.11", DamageStep.REGULAR.wireName());
        }

        List<String> lines = ShardRun.readShard(
                tempDir.resolve("run.0-l1" + RecordShardWriter.SUFFIX));
        assertEquals(2, lines.size(), lines.toString());
        assertTrue(lines.get(0).contains("\"mirror_of\":\"run.0-l1.9\""), lines.get(0));
        assertTrue(lines.get(1).contains("\"mirror_of\":\"run.0-l1.11\""), lines.get(1));
    }

    /** A branch whose step produced no record is dropped rather than mispaired. */
    @Test
    void aBranchFromAStepThatWroteNoRecordIsDropped() throws Exception {
        try (RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1")) {
            PatchedCollectors collector = withForks(writer);
            collector.hold(branch(DamageStep.FIRST_STRIKE.wireName()));

            assertEquals(1, collector.discardStaleProbes(DamageStep.REGULAR.wireName()));
            collector.writeHeldProbes("run.0-l1.9", DamageStep.FIRST_STRIKE.wireName());
        }

        assertFalse(Files.exists(tempDir.resolve("run.0-l1" + RecordShardWriter.SUFFIX)));
    }

    // ── hook lookup ─────────────────────────────────────────────────────

    @Test
    void anAbsentHookReportsAbsentRatherThanThrowing() {
        assertFalse(PatchHooks.find(
                "forge.game.trigger.TriggerHandler", "noSuchHook").present());
    }

    @Test
    void aMissingClassReportsAbsentRatherThanThrowing() {
        assertFalse(PatchHooks.find("forge.game.NotAClass", "anything").present());
    }

    @Test
    void readingAnAbsentHookYieldsNull() {
        assertEquals(null, PatchHooks.readStatic(
                "forge.game.trigger.TriggerHandler", "noSuchHook"));
    }

    @Test
    void installingAnAbsentHookIsANoOpRatherThanAFailure() {
        assertFalse(PatchHooks.install(
                "forge.game.trigger.TriggerHandler", "setNoSuchListener",
                (proxy, method, args) -> null));
    }

    @Test
    void uninstallingAnAbsentHookDoesNotThrow() {
        PatchHooks.uninstall(
                "forge.game.trigger.TriggerHandler", "setNoSuchListener");
    }

    /**
     * Exactly the listener hooks the checkout offers get installed, and no
     * others -- and none at all in degraded mode, where install() declines
     * before it looks.
     *
     * <p>Asserted against the checkout rather than against a fixed count: the
     * sibling Forge is patched or not independently of this repository. The
     * expected set is derived from {@link PatchHooks#REQUIRED}, so a listener
     * hook added there is expected here without editing the test. The confirm
     * listener is excluded because the bracket collector installs it, not this
     * class.
     */
    @Test
    void everyHookTheCheckoutOffersIsInstalledAndNoOthers() {
        long expected = AttributionMode.detect() == AttributionMode.DEGRADED
                ? 0
                : PatchHooks.REQUIRED.stream()
                        .filter(hook -> hook.method().startsWith("setEffectRecord")
                                && hook.method().endsWith("Listener")
                                && !PatchHooks.PLAYER_CONTROLLER_AI.equals(hook.owner()))
                        .filter(PatchHooks.Hook::present)
                        .count();
        try (PatchedCollectors collector = collectors(CollectionCaps.defaults())) {
            assertEquals(expected, collector.install());
        }
    }

    /**
     * A patched checkout has every required hook, not just the one the mode
     * probes: an engine refactor that drops or renames a setter would
     * otherwise leave its channel empty while the count above still agrees
     * with itself.
     */
    @Test
    void aPatchedCheckoutHasEveryRequiredHook() {
        if (AttributionMode.detect() == AttributionMode.PATCHED) {
            assertEquals(List.of(), PatchHooks.missing(), PatchHooks.report());
        }
    }

    /**
     * The read-only sub-ability hook answers null outside a resolution.
     *
     * <p>True on a stock checkout because the method does not exist, and true
     * on a patched one because no sub-ability is resolving on this thread —
     * the test asserts the caller-visible behaviour that holds either way.
     */
    @Test
    void theSubAbilityHookReadsNullOutsideAResolution() {
        assertEquals(null, PatchHooks.currentSubAbility());
    }

    // ── the cause channel ───────────────────────────────────────────────

    /**
     * A trigger of a chosen mode, on a host card the mode does not care about.
     *
     * <p>Parsed rather than taken off a real card, because what is under test
     * is the reading of one mode's run parameters and finding a printed card
     * per mode would tie the test to the set list.
     */
    private static Trigger triggerOfMode(String mode) {
        return TriggerHandler.parseTrigger(
                "Mode$ " + mode + " | TriggerZones$ Battlefield"
                        + " | TriggerDescription$ under test",
                TestCards.build("Paralyze"), true);
    }

    private static String eventOf(PatchedCollectors collector, Trigger trigger,
            Map<String, Object> runParams) {
        return collector.triggerRecord(trigger, runParams, true).toJson();
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

        String json = eventOf(recording(), triggerOfMode("CounterAdded"),
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

        String json = eventOf(recording(), triggerOfMode("ChangesZone"),
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

        String json = eventOf(recording(), triggerOfMode("ChangesZone"),
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
                recording(), triggerOfMode("Untaps"), Map.of("Card", host));

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

        String json = eventOf(recording(), triggerOfMode("SpellCast"),
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

        String json = eventOf(recording(), triggerOfMode("Attacks"),
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

        String json = eventOf(recording(), triggerOfMode("TapsForMana"),
                Map.of("Card", land, "Produced", "G G"));

        assertTrue(json.contains("\"type\":\"mana_produced\""), json);
        assertTrue(json.contains("\"mana_by_color\":{\"G\":2}"), json);
    }

    /** And a mana value it cannot read is absent rather than an empty object. */
    @Test
    void anUnreadableManaValueLeavesTheSlotEmptyRatherThanClaimingNoMana() {
        Card land = TestCards.build("Sol Ring");

        String json = eventOf(recording(), triggerOfMode("TapsForMana"),
                Map.of("Card", land));

        assertFalse(json.contains("mana_by_color"), json);
    }

    /** Turning a card face up is a face change, and it says which face. */
    @Test
    void turningACardFaceUpNamesTheFaceItTurnedTo() {
        Card card = TestCards.build("Sol Ring");

        String json = eventOf(
                recording(), triggerOfMode("TurnFaceUp"), Map.of("Card", card));

        assertTrue(json.contains("\"type\":\"face_change\""), json);
        assertTrue(json.contains("\"to_state\":\"face_up\""), json);
    }

    // ── what the rewrite channel cannot express ─────────────────────────

    /**
     * The tally separates an unreadable change from no change at all.
     *
     * <p>The smoke run reported 110 drops, {@code Moved=91} among them, and
     * {@code 69 dropped with no raw parameter difference at all} — three lines
     * needing arithmetic across them to reach the one fact that decides what to
     * do next: 62 of Moved's 91 changed nothing whatever, while 29 changed only
     * the entry-counter table, which {@code zone_change} has no slot for. Those
     * rows are all written now, so the question is no longer which to drop but
     * which of them a reader is being short-changed on.
     */
    @Test
    void theTallySeparatesAnUnreadableChangeFromNoChange() {
        RewriteTally tally = new RewriteTally();

        tally.record("Moved", "replaced", false, true,
                Map.of("Destination", "Battlefield"),
                Map.of("Destination", "Battlefield"));
        tally.record("Moved", "replaced", false, true,
                Map.of("CounterMap", "{}"),
                Map.of("CounterMap", "{P1P1=1}"));

        String summary = tally.summary();
        assertTrue(summary.contains(
                "1 carried no outgoing while a raw parameter moved"), summary);
        assertTrue(summary.contains("of those, by mode: Moved=1"), summary);
        assertTrue(summary.contains("Moved.CounterMap=1"), summary);
    }

    /**
     * A replacement whose mode only the replacement side spells is still typed.
     *
     * <p>{@code ReplacementType} names {@code Untap} where {@code TriggerType}
     * names {@code Untaps}, and the mode table was written from the trigger
     * vocabulary — so an untap replacement took the generic type, carried
     * nothing but its own name, and could not differ from itself however much
     * it changed. Paralyze's is that mode. Its record still carries no outgoing,
     * because a prevention changes no parameter at all, but the incoming half is
     * now typed rather than generic.
     */
    @Test
    void anUntapReplacementIsSpelledTheReplacementSideWay() {
        Card paralyze = TestCards.build("Paralyze");
        ReplacementEffect prevention =
                paralyze.getCurrentState().getReplacementEffects().iterator().next();

        assertEquals("Untap", String.valueOf(prevention.getMode()));
        String json = recording().rewriteRecord(
                prevention, Map.of("Card", paralyze), Map.of("Card", paralyze),
                ReplacementResult.Skipped, null).toJson();

        assertTrue(json.contains("\"mode\":\"Untap\""), json);
        assertTrue(json.contains("\"outgoing\":null"), json);
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

        String blocks = eventOf(recording(), triggerOfMode("Blocks"),
                Map.of("Blocker", blocker, "Attackers", attacker));
        assertTrue(blocks.contains("\"subjects\":[\"" + blockerId + "\"]"), blocks);
        assertTrue(blocks.contains("\"blocked\":\"" + attackerId + "\""), blocks);

        String blocked = eventOf(recording(),
                triggerOfMode("AttackerBlockedByCreature"),
                Map.of("Attacker", attacker, "Blocker", blocker));
        assertTrue(blocked.contains("\"type\":\"became_blocked\""), blocked);
        assertTrue(blocked.contains("\"subjects\":[\"" + attackerId + "\"]"), blocked);
        assertTrue(blocked.contains("\"blockers\":[\"" + blockerId + "\"]"), blocked);
    }

    // ── outcome hook argument order (Task 7) ─────────────────────────────

    /**
     * Forge's outcome listener, copied argument for argument rather than
     * imported: the connector compiles against stock Forge, where
     * {@code EffectRecordOutcomes.EffectRecordOutcomeListener} does not
     * exist, and a test that imported it would stop this module building on
     * an unpatched checkout. See {@code ClauseContractTest.ListenerShape} for
     * the same pattern on the clause hook.
     */
    public interface OutcomeListenerShape {
        void onOutcome(SpellAbility sa, String eventType, Map<String, Object> params);
    }

    @BeforeEach
    void resetUnknownOutcomeTypeLatch() throws ReflectiveOperationException {
        // Static, and shared with the production path: without a reset here,
        // whichever test trips it first leaves every test after it in this
        // class unable to see its own "did this print" outcome.
        var latch = PatchedCollectors.class.getDeclaredField("UNKNOWN_OUTCOME_TYPE_REPORTED");
        latch.setAccessible(true);
        ((java.util.concurrent.atomic.AtomicBoolean) latch.get(null)).set(false);
    }

    private static Method methodNamed(Class<?> shape, String name) {
        for (Method method : shape.getMethods()) {
            if (method.getName().equals(name)) {
                return method;
            }
        }
        throw new AssertionError(shape + " declares no " + name);
    }

    /**
     * The stand-in above is what Forge declares, position by position.
     *
     * <p>Reflective, so an unpatched checkout skips the assertion rather than
     * failing it -- the same degradation every hook lookup makes.
     */
    @Test
    void theOutcomeStandInIsForgesOwnDeclaration() {
        PatchHooks.Lookup setter = PatchHooks.find(
                PatchHooks.EFFECT_RECORD_OUTCOMES, "setEffectRecordOutcomeListener");
        Assumptions.assumeTrue(setter.present(), "the checkout lacks the outcome hook");
        Class<?> declared = setter.method().getParameterTypes()[0];
        Class<?>[] theirs = methodNamed(declared, "onOutcome").getParameterTypes();
        Class<?>[] ours = methodNamed(OutcomeListenerShape.class, "onOutcome").getParameterTypes();
        assertEquals(ours.length, theirs.length, "onOutcome arity");
        for (int i = 0; i < theirs.length; i++) {
            assertEquals(ours[i], theirs[i], "onOutcome argument " + i);
        }
    }

    /**
     * The hook binds by argument position, and no compiler checks that. A
     * reorder in Forge is invisible to both compilers and must fail here
     * rather than in a corpus -- which is why this uses a real, distinct
     * {@link SpellAbility} rather than null for position 0: a null would not
     * notice a swap between the ability and either of the other two
     * arguments, only {@link #theOutcomeStandInIsForgesOwnDeclaration} would,
     * and that test is reflective and skips on an unpatched checkout.
     */
    @Test
    void anOutcomeReportBecomesAnEventOfThatType() throws Throwable {
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");
        PatchedCollectors collectors = recording();

        collectors.outcomeHandler().invoke(null,
                methodNamed(OutcomeListenerShape.class, "onOutcome"),
                new Object[]{sa, "vote_taken", Map.of("tally", Map.of("beast", 2))});

        EffectEvent written = collectors.lastClauseEvent();
        assertEquals("vote_taken", written.type());
        assertEquals(Map.of("beast", 2), written.params().get("tally"));
    }

    /**
     * An outcome that names no subjects has none: a coin flip or a token
     * made is not something done to the card whose ability did it, so the
     * acting host is not a default.
     */
    @Test
    void anOutcomeNamingNoSubjectsHasNone() throws Throwable {
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");
        PatchedCollectors collectors = recording();

        collectors.outcomeHandler().invoke(null,
                methodNamed(OutcomeListenerShape.class, "onOutcome"),
                new Object[]{sa, "coin_flipped", Map.of("won", true)});

        EffectEvent written = collectors.lastClauseEvent();
        assertEquals("coin_flipped", written.type());
        assertEquals(List.of(), written.subjects(),
                "the acting host is not a subject: " + written.toJson());
    }

    /**
     * A {@code token_created} note hands over a sample token, which becomes
     * the token's script id and characteristics as the board would render
     * that token -- keywords included, spelled the way an entity's are --
     * and never reaches the event itself.
     */
    @Test
    void aTokenSampleBecomesItsScriptIdAndCharacteristics() throws Throwable {
        Card bat = TestCards.token("b_1_1_bat_flying");
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");
        PatchedCollectors collectors = recording();

        collectors.outcomeHandler().invoke(null,
                methodNamed(OutcomeListenerShape.class, "onOutcome"),
                new Object[]{sa, "token_created", Map.of("sample", bat, "count", 2)});

        EffectEvent written = collectors.lastClauseEvent();
        assertEquals("token_created", written.type());
        assertEquals(Set.of("token_script_id", "characteristics", "count"),
                written.params().keySet(), written.toJson());
        assertEquals("b_1_1_bat_flying", written.params().get("token_script_id"));
        assertEquals(SnapshotBuilder.tokenScriptIdOf(bat), written.params().get("token_script_id"));
        assertEquals(2, written.params().get("count"));
        @SuppressWarnings("unchecked")
        Map<String, Object> traits = (Map<String, Object>) written.params().get("characteristics");
        assertEquals(1, traits.get("power"));
        assertEquals(1, traits.get("toughness"));
        assertEquals(bat.getType().toString(), traits.get("types"));
        assertEquals(List.of("flying"), traits.get("keywords"));
        assertEquals(SnapshotBuilder.keywordNames(bat.getKeywords()), traits.get("keywords"));
        assertFalse(written.toJson().contains("sample"), written.toJson());
        assertEquals(List.of(), written.subjects());
    }

    /**
     * {@code damage_prevented} names who the damage would have been dealt
     * to, a player or a creature, as the event's subject.
     */
    @Test
    void damagePreventedNamesItsRecipient() throws Throwable {
        Card creature = TestCards.build("Grizzly Bears");
        Player player = new Player("recipient", TestCards.game(), 93201);
        PatchedCollectors collectors = recording();
        Method onOutcome = methodNamed(OutcomeListenerShape.class, "onOutcome");

        collectors.outcomeHandler().invoke(null, onOutcome,
                new Object[]{null, "damage_prevented",
                        Map.of("amount", 2, "subjects", List.of(creature))});
        assertEquals(List.of(SnapshotBuilder.entityId(creature)),
                collectors.lastClauseEvent().subjects());
        assertNull(collectors.lastClauseEvent().params().get("subjects"));

        collectors.outcomeHandler().invoke(null, onOutcome,
                new Object[]{null, "damage_prevented",
                        Map.of("amount", 2, "subjects", List.of(player))});
        assertEquals(List.of(SnapshotBuilder.playerId(player)),
                collectors.lastClauseEvent().subjects());

        collectors.outcomeHandler().invoke(null, onOutcome,
                new Object[]{null, "damage_prevented",
                        Map.of("amount", 2, "subjects", List.of())});
        assertEquals(List.of(), collectors.lastClauseEvent().subjects());
    }

    /**
     * A reveal from a library names the players it came from; any other
     * reveal arrives with an empty list and names nobody.
     */
    @Test
    void cardRevealedNamesTheLibrariesItCameFrom() throws Throwable {
        Player first = new Player("first", TestCards.game(), 93202);
        Player second = new Player("second", TestCards.game(), 93203);
        PatchedCollectors collectors = recording();
        Method onOutcome = methodNamed(OutcomeListenerShape.class, "onOutcome");

        collectors.outcomeHandler().invoke(null, onOutcome,
                new Object[]{null, "card_revealed",
                        Map.of("count", 2, "from_zone", "Library",
                                "subjects", List.of(first, second))});
        EffectEvent fromLibraries = collectors.lastClauseEvent();
        assertEquals(List.of(SnapshotBuilder.playerId(first), SnapshotBuilder.playerId(second)),
                fromLibraries.subjects());
        assertEquals(Set.of("count", "from_zone"), fromLibraries.params().keySet());

        collectors.outcomeHandler().invoke(null, onOutcome,
                new Object[]{null, "card_revealed",
                        Map.of("count", 1, "from_zone", "Hand", "subjects", List.of())});
        assertEquals(List.of(), collectors.lastClauseEvent().subjects());
    }

    /**
     * {@code "subjects"} in the params map is not a generic parameter, even
     * though Detain/Goad/MustBlock report it as a params entry (see
     * {@code EffectRecordOutcomes.note} call sites in
     * {@code GoadEffect}/{@code DetainEffect}/{@code MustBlockEffect}): the
     * schema's {@code restriction_change} row normalizes only
     * {@code restriction} and {@code value}, so a raw {@code subjects} entry
     * left in {@code params} would fail the per-type param allow-list on the
     * Python side. The handler converts it to the event's own top-level
     * subject refs instead.
     *
     * <p>Exact match, not {@code containsAll}: {@code restriction_change}'s
     * acting card (the host, "Alchemist's Gambit" here) must not appear in
     * the subject list alongside the two cards actually goaded, and
     * {@code containsAll} cannot tell "exactly these two" from "these two,
     * plus an extra false one".
     */
    @Test
    void restrictionChangeSubjectsBecomeEventSubjectsNotAParam() throws Throwable {
        Card goaded1 = TestCards.build("Sol Ring");
        Card goaded2 = TestCards.build("Mox Pearl");
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");
        PatchedCollectors collectors = recording();

        collectors.outcomeHandler().invoke(null,
                methodNamed(OutcomeListenerShape.class, "onOutcome"),
                new Object[]{sa, "restriction_change",
                        Map.of("restriction", "goad", "value", true,
                                "subjects", List.of(goaded1, goaded2))});

        EffectEvent written = collectors.lastClauseEvent();
        assertEquals("restriction_change", written.type());
        assertEquals("goad", written.params().get("restriction"));
        assertEquals(Boolean.TRUE, written.params().get("value"));
        assertNull(written.params().get("subjects"), "subjects must not leak into params");
        assertEquals(List.of(
                SnapshotBuilder.entityId(goaded1), SnapshotBuilder.entityId(goaded2)),
                written.subjects(),
                "exactly the two goaded cards -- not the acting host as well");
    }

    /**
     * An outcome type outside {@link PatchedCollectors#outcomeHandler()}'s
     * known set is exactly the failure this whole plan exists to end,
     * recreated one level up: before this check existed, all nine of this
     * task's types were referenced only inside a javadoc comment, which
     * satisfies the completeness guard's textual scan just as well as real
     * code -- so a deleted {@code note(...)} call or a misspelled event name
     * would go dark and the guard built to notice would say nothing. A
     * recognized type still writes an event; an unrecognized one does not,
     * and is reported exactly once.
     */
    @Test
    void anUnrecognizedOutcomeTypeWritesNoEventAndIsReportedOnce() throws Throwable {
        PatchedCollectors collectors = recording();
        Method onOutcome = methodNamed(OutcomeListenerShape.class, "onOutcome");

        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream original = System.err;
        System.setErr(new PrintStream(captured, true));
        try {
            collectors.outcomeHandler().invoke(null, onOutcome,
                    new Object[]{null, "dungeon_entered", Map.of()});
            collectors.outcomeHandler().invoke(null, onOutcome,
                    new Object[]{null, "dungeon_entered", Map.of()});
            collectors.outcomeHandler().invoke(null, onOutcome,
                    new Object[]{null, "another_unknown_type", Map.of()});
        } finally {
            System.setErr(original);
        }

        assertNull(collectors.lastClauseEvent(), "an unrecognized type must not become an event");
        String output = captured.toString();
        long lineCount = output.isBlank() ? 0 : output.strip().lines().count();
        assertEquals(1, lineCount, "exactly one report despite three unrecognized types: " + output);
        assertTrue(output.contains("dungeon_entered"), "the report names the first unrecognized type: " + output);
    }

    /**
     * {@code damage_prevented}'s {@code "source"} is converted to an entity
     * ref, matching every other producer of a {@code source} key
     * ({@code damage_dealt}, and the trigger-derived {@code damage_prevented}
     * via {@code subjectOf}). Forge passes the {@code Card} itself rather
     * than a display name for exactly this conversion -- a display name
     * would silently fail a join against those refs, and collide two sources
     * sharing a name.
     */
    @Test
    void damagePreventedSourceBecomesAnEntityRef() throws Throwable {
        Card source = TestCards.build("Grizzly Bears");
        PatchedCollectors collectors = recording();

        collectors.outcomeHandler().invoke(null,
                methodNamed(OutcomeListenerShape.class, "onOutcome"),
                new Object[]{null, "damage_prevented",
                        Map.of("amount", 3, "source", source)});

        EffectEvent written = collectors.lastClauseEvent();
        assertEquals("damage_prevented", written.type());
        assertEquals(3, written.params().get("amount"));
        assertEquals(SnapshotBuilder.entityId(source), written.params().get("source"),
                "a ref like damage_dealt's, not a display name");
    }

    // ── F2: another game's ability must not reach this game's bracket ────

    /**
     * F2: an outcome reported for an ability that belongs to a DIFFERENT
     * {@code Game} -- exactly the shape {@code ForkCollector.forceResolution}
     * produces, since {@code GameSimulator.resolveStack} runs the real
     * resolution pipeline against a {@code GameCopier} clone under
     * {@code AIOption.USE_FULL_SIMULATION} regardless of how this game's own
     * lobby seats were registered (ruling R31) -- must not reach the live
     * game's bracket. {@code GameCopier} builds every copied {@code Card}
     * against a new {@code Game} object, so a fork's ability's own host card
     * already carries a different {@code Game} than this collector's.
     *
     * <p>{@code vote_taken} and {@code coin_flipped} (the concrete types
     * named in final-fix-1.md F2) are both outcome-hook types, so this
     * exercises {@link PatchedCollectors#outcomeHandler()} rather than the
     * clause hook -- {@code ClauseContractTest} covers the clause hook's own
     * half of the same fix.
     *
     * <p>{@code lastClauseEvent()} is asserted non-null first: it is set
     * unconditionally, before the (now gated) delivery to the bracket, so
     * this also proves the event was genuinely built and rejected by the
     * game-identity check -- not merely never reached because of some other,
     * unrelated failure this test would then wrongly credit to the fix.
     */
    @Test
    void anOutcomeFromAnotherGameDoesNotReachThisGamesBracket() throws Throwable {
        Card forkHost = TestBoard.cardInAnotherGame("Grizzly Bears");
        SpellAbility forkAbility = AbilityFactory.getAbility(
                "SP$ AddTurn | ValidTgts$ Player | NumTurns$ 1", forkHost);

        SpellAbility liveAbility = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");
        List<String> records = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) -> {
            bracket.beginBracket(liveAbility);

            collectors.outcomeHandler().invoke(null,
                    methodNamed(OutcomeListenerShape.class, "onOutcome"),
                    new Object[]{forkAbility, "vote_taken", Map.of("tally", Map.of("beast", 2))});

            assertNotNull(collectors.lastClauseEvent(),
                    "the event must still be built -- only its delivery to "
                            + "the live bracket is gated");
            assertEquals("vote_taken", collectors.lastClauseEvent().type());

            bracket.endBracket(liveAbility.getId(), false);
            bracket.flushHeldResolution();
        });

        String shard = String.join("\n", records);
        assertFalse(shard.contains("vote_taken"),
                "a fork's outcome must not land in the live game's record: " + shard);
    }

    // ── final-fix-3.md item 1 / final-fix-4.md item 1: a null-ability ────
    // ── outcome's own game signal, at the shapes production reaches ──────

    /**
     * The critical regression. {@code GameAction.reveal}'s {@code
     * card_revealed} and {@code ReplacementHandler}'s {@code
     * damage_prevented} (rulings R10, R11/R12) report through {@code
     * EffectRecordOutcomes.note} with a literal {@code null} ability,
     * <em>always</em> -- F2's {@code args[0] instanceof SpellAbility} gate
     * (final-fix-1.md) makes that {@code instanceof} false unconditionally,
     * which silently dropped both channels on the live game, not just on a
     * fork.
     *
     * <p>Every pre-existing null-ability outcome test (e.g. {@code
     * damagePreventedSourceBecomesAnEntityRef} above) asserts on {@link
     * PatchedCollectors#lastClauseEvent()}, which is set unconditionally
     * above the delivery gate and so cannot see this class of bug -- that is
     * exactly why 689/689 stayed green while the channel was dead. This
     * asserts on what actually reaches the shard instead, the way {@link
     * #anOutcomeFromAnotherGameDoesNotReachThisGamesBracket} already does for
     * the ability-bearing case.
     *
     * <p>final-fix-4.md item 1: replaces the previous version of this test,
     * which opened a resolution bracket for an ability nothing was actually
     * resolving -- a synthetic composite ("pointer null" and "bracket open")
     * production never produces. Real combat damage reaches this method with
     * <em>no</em> bracket open at all, which is the shape used here.
     */
    @Test
    void aNullAbilityOutcomeDuringLiveCombatDamageReachesTheShard() throws Throwable {
        // Genuinely mid combat-damage step: the one production shape a live
        // damage_prevented reaches with neither the fork flag nor the
        // resolving-clause pointer set, since Combat.dealAssignedDamage is a
        // turn-based action and no resolution bracket is ever open around it.
        // Its own game rather than the shared TestCards.game(): the phase is
        // what routes an event into the combat bracket, and a shared game left
        // mid combat-damage would route every other test's events there too.
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DAMAGE);
        List<String> records = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            collectors.outcomeHandler().invoke(null,
                    methodNamed(OutcomeListenerShape.class, "onOutcome"),
                    new Object[]{null, "damage_prevented", Map.of("amount", 3)});

            // Flushes the combat bracket the null-ability report above opened,
            // the way BusBracketCollectorTest.theCombatThatEndedTheGameIsWritten
            // does for the same reason: a game ending in combat damage never
            // reaches the ordinary phase-boundary flush.
            bracket.finishGame(Set.of());
        });

        String shard = String.join("\n", records);
        assertTrue(shard.contains("damage_prevented"),
                "a live combat-damage prevention must still reach the shard: " + shard);
    }

    /**
     * final-fix-4.md item 1 (Critical guard gap): pins the resolving-clause
     * pointer branch directly, which nothing did before this test existed.
     * {@code return !ForkCollector.isForkRunningOnThisThread();} -- the exact
     * single-signal version the report built and discarded to demonstrate the
     * naive-fail-open trap -- passed every test in this class, because no
     * test ever set the pointer: the combat-damage test above leaves it null
     * by construction, and the probe test below only ever exercises the flag.
     * So did dropping the {@link PatchedCollectors#belongsToLiveGame(CardTraitBase)}
     * delegate and returning {@code true} whenever the pointer is merely
     * non-null. This resolves a real ability -- through {@code
     * AbilityUtils.resolve}, not a synthesized call -- so the pointer is
     * genuinely set the way {@code ForkCollector.forceResolution}'s own
     * forced ability sets it, and installs a real {@code
     * EffectRecordClauseListener} (the same reflective seam {@link
     * PatchedCollectors#install()} itself uses) to fire the null-ability
     * report from {@code onClauseResolving}, while the pointer names the
     * fork's own ability.
     *
     * <p>The live game's own bracket is opened for an unrelated live ability
     * around the fork's resolution, purely so the buffered event has
     * somewhere to flush to -- {@code BusBracketCollector} only ever writes
     * {@code bracketEvents} on a matching {@code endBracket}, discarding them
     * unflushed otherwise (confirmed by hand: this test read back an empty
     * shard under both the correct implementation and the naive one before
     * this bracket was added, which is not evidence of anything). This
     * mirrors {@code anOutcomeFromAnotherGameDoesNotReachThisGamesBracket}'s
     * own live bracket, opened for the same reason. It answers a different
     * question than which ability the pointer names: the pointer is set (and
     * read) entirely inside {@code AbilityUtils.resolve(forkAbility)}, which
     * runs nested inside this bracket but neither reads nor writes it.
     */
    @Test
    void aForksAbilitysNullAbilityOutcomeIsDroppedWhileItIsGenuinelyResolving() throws Throwable {
        Card forkHost = TestBoard.cardInAnotherGame("Grizzly Bears");
        Game fork = forkHost.getGame();
        Player forkActor = new Player("fork-actor", fork, 93001);
        Player forkTarget = new Player("fork-target", fork, 93002);
        SpellAbility forkAbility = AbilityFactory.getAbility(
                "SP$ AddTurn | ValidTgts$ Player | NumTurns$ 1", forkHost);
        forkAbility.setActivatingPlayer(forkActor);
        forkAbility.resetTargets();
        forkAbility.getTargets().add(forkTarget);
        SpellAbility flushAbility = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");

        List<String> records = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) -> {
            bracket.beginBracket(flushAbility);

            boolean installed = PatchHooks.install(PatchHooks.ABILITY_UTILS,
                    "setEffectRecordClauseListener", (proxy, method, args) -> {
                        if ("onClauseResolving".equals(method.getName())) {
                            try {
                                collectors.outcomeHandler().invoke(null,
                                        methodNamed(OutcomeListenerShape.class, "onOutcome"),
                                        new Object[]{null, "damage_prevented", Map.of("amount", 3)});
                            } catch (Throwable t) {
                                throw new RuntimeException(t);
                            }
                        }
                        return null;
                    });
            Assumptions.assumeTrue(installed, "the checkout lacks the clause hook");
            try {
                AbilityUtils.resolve(forkAbility);
            } finally {
                PatchHooks.uninstall(PatchHooks.ABILITY_UTILS, "setEffectRecordClauseListener");
            }
            bracket.endBracket(flushAbility.getId(), false);
            bracket.flushHeldResolution();
        });

        String shard = String.join("\n", records);
        assertFalse(shard.contains("damage_prevented"),
                "a fork ability's own null-ability outcome, reported while it "
                        + "is genuinely resolving, must not reach the live shard: " + shard);
    }

    /**
     * The admitting half of the same pin: a <em>live</em> ability's own
     * null-ability outcome, reported while the pointer genuinely names it
     * (not null, not a fork), must be admitted. Where the fork test above
     * fires from {@code intervene}'s own phase-boundary timing (no live
     * resolution in flight), this fires while that same live ability's own
     * bracket is open -- production's shape for a live reveal or damage
     * prevention nested inside a live spell's own resolution.
     */
    @Test
    void aLiveAbilitysNullAbilityOutcomeIsAdmittedWhileItIsGenuinelyResolving() throws Throwable {
        SpellAbility liveAbility = AbilityFactory.getAbility(
                "SP$ AddTurn | ValidTgts$ Player | NumTurns$ 1", TestCards.build("Grizzly Bears"));
        Player liveActor = new Player("live-actor", TestCards.game(), 93003);
        Player liveTarget = new Player("live-target", TestCards.game(), 93004);
        liveAbility.setActivatingPlayer(liveActor);
        liveAbility.resetTargets();
        liveAbility.getTargets().add(liveTarget);

        List<String> records = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) -> {
            bracket.beginBracket(liveAbility);

            boolean installed = PatchHooks.install(PatchHooks.ABILITY_UTILS,
                    "setEffectRecordClauseListener", (proxy, method, args) -> {
                        if ("onClauseResolving".equals(method.getName())) {
                            try {
                                collectors.outcomeHandler().invoke(null,
                                        methodNamed(OutcomeListenerShape.class, "onOutcome"),
                                        new Object[]{null, "damage_prevented", Map.of("amount", 3)});
                            } catch (Throwable t) {
                                throw new RuntimeException(t);
                            }
                        }
                        return null;
                    });
            Assumptions.assumeTrue(installed, "the checkout lacks the clause hook");
            try {
                AbilityUtils.resolve(liveAbility);
            } finally {
                PatchHooks.uninstall(PatchHooks.ABILITY_UTILS, "setEffectRecordClauseListener");
            }
            bracket.endBracket(liveAbility.getId(), false);
            bracket.flushHeldResolution();
        });

        String shard = String.join("\n", records);
        assertTrue(shard.contains("damage_prevented"),
                "a live ability's own null-ability outcome, reported while it "
                        + "is genuinely resolving, must reach the shard: " + shard);
    }

    /**
     * The trap the finding warns against: failing open on {@code null}
     * readmits exactly what the gate exists to exclude, because {@code
     * ForkCollector.probe}'s own combat-damage counterfactual fires this same
     * {@code null}-ability {@code damage_prevented} report -- {@code
     * Combat.dealAssignedDamage} always passes a null cause, live or forked.
     * A probe's damage step is a turn-based action with no ability resolving
     * and nothing on the resolving-clause pointer, which is why {@link
     * ForkCollector#isForkRunningOnThisThread()} exists. FORK_RUNNING
     * is set directly here, without needing a real combat to
     * fork. No live bracket is open, matching a probe's own
     * timing: like {@code intervene}, it does not run nested inside a live
     * resolution.
     */
    @Test
    void aNullAbilityOutcomeFromAProbesOwnCombatDoesNotReachTheLiveShard() throws Throwable {
        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), (bracket, collectors) -> {
            var forkRunning = ForkCollector.class.getDeclaredField("FORK_RUNNING");
            forkRunning.setAccessible(true);
            @SuppressWarnings("unchecked")
            ThreadLocal<Boolean> flag = (ThreadLocal<Boolean>) forkRunning.get(null);
            flag.set(true);
            try {
                collectors.outcomeHandler().invoke(null,
                        methodNamed(OutcomeListenerShape.class, "onOutcome"),
                        new Object[]{null, "damage_prevented", Map.of("amount", 3)});
            } finally {
                flag.set(false);
            }

            assertNotNull(collectors.lastClauseEvent(),
                    "the event must still be built -- only its delivery to "
                            + "the live bracket is gated");

            // The event reaches the bracket's buffer, which only a flush
            // writes; without one the shard is empty whether or not the gate
            // exists.
            bracket.finishGame(Set.of());
        });

        assertFalse(Files.exists(shard),
                "a probe's own combat-damage report must not land in the live game's record");
    }

    // ── final-fix-3.md item 2: the other four handlers ───────────────────

    private interface TriggerFireListenerShape {
        void onConditionEvaluated(Trigger trigger, Map<Object, Object> runParams, boolean fired);
    }

    private interface PlayabilityListenerShape {
        void onCandidate(SpellAbility candidate);
    }

    /** Forge's {@code CombatLegalityObserver.EffectRecordCombatListener}, position by position. */
    private interface CombatLegalityListenerShape {
        void onAttackersDeclarable(Player decider, GameEntity defender,
                List<Card> candidates, List<Card> legal);

        void onBlockersDeclarable(Player decider, Card attacker,
                List<Card> candidates, List<Card> legal, int minBlockers);
    }

    private interface ManaListenerShape {
        void onManaProduced(SpellAbility ability, Player player, String produced);
    }

    /**
     * final-fix-3.md item 2: {@code triggerFireHandler} had no game check at
     * all before this fix. A fork's own trigger evaluations
     * ({@code GameSimulator.resolveStack} runs the forced ability's triggers
     * through the real {@code TriggerHandler}) must not reach the live shard.
     */
    @Test
    void aTriggerEvaluationFromAnotherGameDoesNotReachTheShard() throws Throwable {
        Card forkCard = TestBoard.cardInAnotherGame("Paralyze");
        Trigger forkTrigger = forkCard.getCurrentState().getTriggers().get(0);

        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.triggerFireHandler().invoke(null,
                        methodNamed(TriggerFireListenerShape.class, "onConditionEvaluated"),
                        new Object[]{forkTrigger, Map.of(), true}));

        assertFalse(Files.exists(shard),
                "a fork's own trigger evaluation must not land in the live shard");
    }

    /**
     * final-fix-3.md item 2: {@code playabilityHandler} had no game check
     * either. {@code caps(0, 1.0)} forces {@code playabilityRate} to 1.0 so
     * the sampling gate ahead of the game check cannot flakily hide it.
     */
    @Test
    void aPlayabilityCandidateFromAnotherGameDoesNotReachTheShard() throws Throwable {
        Card forkCard = TestBoard.cardInAnotherGame("Grizzly Bears");
        SpellAbility forkCandidate = AbilityFactory.getAbility(
                "AB$ Pump | Cost$ 1 | NumAtt$ +1 | NumDef$ +1", forkCard);

        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), caps(0, 1.0),
                (bracket, collectors) -> collectors.playabilityHandler().invoke(null,
                        methodNamed(PlayabilityListenerShape.class, "onCandidate"),
                        new Object[]{forkCandidate}));

        assertFalse(Files.exists(shard),
                "a fork's own playability candidate must not land in the live shard");
    }

    /**
     * {@code combatLegalityHandler} checks the game the defender belongs to.
     * The defender is the acting {@code GameEntity} this hook hands over; the
     * candidate/legal lists may be any real card, since only the defender's
     * own game identity is what the check reads.
     */
    @Test
    void anAttackerLegalityAnswerFromAnotherGamesDefenderDoesNotReachTheShard() throws Throwable {
        Player forkDefender = new Player("fork-defender", TestBoard.anotherGame(), 92301);
        Card candidate = TestCards.build("Grizzly Bears");

        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.combatLegalityHandler().invoke(null,
                        methodNamed(CombatLegalityListenerShape.class, "onAttackersDeclarable"),
                        new Object[]{null, forkDefender, List.of(candidate), List.of(candidate)}));

        assertFalse(Files.exists(shard), "another game's defender must not land in the live shard");
    }

    /**
     * final-fix-3.md item 2: {@code manaHandler} had no game check either.
     * Held in {@link PatchedCollectors#manaReservoir} rather than written
     * immediately, so the collector must be closed (which flushes it) before
     * the shard can be read.
     */
    @Test
    void manaProducedByAnotherGamesAbilityDoesNotReachTheShard() throws Throwable {
        Card forkCard = TestBoard.cardInAnotherGame("Llanowar Elves");
        SpellAbility forkAbility = AbilityFactory.getAbility(
                "AB$ Mana | Cost$ T | Produced$ G", forkCard);
        Player forkPlayer = new Player("fork-player", forkCard.getGame(), 92302);

        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.manaHandler().invoke(null,
                        methodNamed(ManaListenerShape.class, "onManaProduced"),
                        new Object[]{forkAbility, forkPlayer, "G"}));

        assertFalse(Files.exists(shard),
                "a fork's own mana production must not land in the live shard");
    }

    // ── final-fix-4.md item 3: a positive test behind each of those gates ──
    //
    // triggerFireHandler, playabilityHandler, combatLegalityHandler and
    // manaHandler are invoked nowhere in the test tree except the four
    // fork-rejection tests above -- the existing triggerRecord(...)-style
    // tests call the record builder directly and never reach the handler
    // (InvocationHandler) these gates actually live in. A gate with only a
    // negative test behind it can fail closed for everything, live included,
    // and every one of those tests would still pass. rewriteHandler and
    // confirmHandler already have positive coverage elsewhere in this class
    // and in BusBracketCollectorTest respectively (this file's own
    // theHandlerReadsEachArgumentOffItsOwnPosition and friends predate item 2
    // and already drive the live handler directly), so only these four need
    // one here.

    @Test
    void aLiveTriggerEvaluationReachesTheShard() throws Throwable {
        Card liveCard = TestCards.build("Paralyze");
        Trigger liveTrigger = liveCard.getCurrentState().getTriggers().get(0);

        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.triggerFireHandler().invoke(null,
                        methodNamed(TriggerFireListenerShape.class, "onConditionEvaluated"),
                        new Object[]{liveTrigger, Map.of(), true}));
        assertTrue(lines.stream().anyMatch(l -> l.contains("\"kind\":\"trigger\"")),
                "a live trigger evaluation must reach the shard: " + lines);
    }

    @Test
    void aLivePlayabilityCandidateReachesTheShard() throws Throwable {
        Card liveCard = TestCards.build("Grizzly Bears");
        SpellAbility liveCandidate = AbilityFactory.getAbility(
                "AB$ Pump | Cost$ 1 | NumAtt$ +1 | NumDef$ +1", liveCard);

        // playabilityRate forced to 1.0, matching final-fix-4.md item 5's own
        // fix: at the default 0.1 this record would be sampled out at random
        // roughly nine times in ten, and a flaky pass here is exactly as
        // uninformative as the vacuous one item 3 exists to close.
        List<String> lines = ShardRun.records(tempDir, TestCards.game(), caps(0, 1.0),
                (bracket, collectors) -> collectors.playabilityHandler().invoke(null,
                        methodNamed(PlayabilityListenerShape.class, "onCandidate"),
                        new Object[]{liveCandidate}));
        assertTrue(lines.stream().anyMatch(l -> l.contains("\"kind\":\"playability\"")),
                "a live playability candidate must reach the shard: " + lines);
    }

    @Test
    void aLiveAttackerLegalityAnswerReachesTheShard() throws Throwable {
        Player liveDefender = new Player("live-defender", TestCards.game(), 93101);
        Card candidate = TestCards.build("Grizzly Bears");

        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.combatLegalityHandler().invoke(null,
                        methodNamed(CombatLegalityListenerShape.class, "onAttackersDeclarable"),
                        new Object[]{null, liveDefender, List.of(candidate), List.of(candidate)}));
        assertTrue(lines.stream().anyMatch(l -> l.contains("\"subkind\":\"attackers\"")),
                "a live defender's legality answer must reach the shard: " + lines);
    }

    @Test
    void aLiveManaProductionReachesTheShard() throws Throwable {
        SpellAbility liveAbility = AbilityFactory.getAbility(
                "AB$ Mana | Cost$ T | Produced$ G", TestCards.build("Llanowar Elves"));
        Player livePlayer = new Player("live-player", TestCards.game(), 93102);

        // manaHandler holds records in the reservoir rather than writing them
        // immediately (PatchedCollectors#manaReservoir); only flushMana(),
        // called when the run closes the collectors, writes them out.
        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) ->
                collectors.manaHandler().invoke(null,
                        methodNamed(ManaListenerShape.class, "onManaProduced"),
                        new Object[]{liveAbility, livePlayer, "G"}));
        assertTrue(lines.stream().anyMatch(l -> l.contains("mana_produced")),
                "live mana production must reach the shard: " + lines);
    }

    /**
     * final-fix-4.md item 5: a fork candidate rejected on identity must not
     * have drawn from the shared {@code sampler} on its way to being
     * rejected -- checking identity before the {@code caps.playabilityRate()}
     * draw is what this pins, not just what the diff shows.
     *
     * <p>Both collectors share one seed (1) and one {@code playabilityRate}
     * (0.5). The sampler is <em>not</em> {@code new Random(1L)}: it is
     * {@code new Random(scramble(seed))}, and {@code scramble} is SplitMix64's
     * finalizer, so the draws are those of {@code Random(scramble(1L))} --
     * {@code 0.9245, 0.1142}, not {@code Random(1L)}'s {@code 0.7309, 0.4101}.
     * An earlier version of this comment cited the latter; the conclusion held
     * either way, which is exactly why the wrong constants survived. Do not
     * try to reproduce them against the real {@code sampler}.
     *
     * <p>A solo live candidate draws only the first (0.9245 &gt; 0.5, sampled
     * out). If the identity check ran <em>after</em> the draw, the fork
     * candidate ahead of it would consume that first draw before being dropped
     * on identity, and the live candidate behind it would draw the second
     * (0.1142 &le; 0.5, sampled in) -- a live record appearing only because a fork candidate
     * happened to precede it. Checked first, the fork candidate never reaches
     * the sampler at all, and the live candidate's own draw -- and therefore
     * its own outcome -- is identical whether or not the fork one ran ahead
     * of it.
     */
    @Test
    void aRejectedForkCandidateDoesNotShiftTheLiveSampleStream() throws Throwable {
        Card liveCard = TestCards.build("Grizzly Bears");
        SpellAbility liveCandidate = AbilityFactory.getAbility(
                "AB$ Pump | Cost$ 1 | NumAtt$ +1 | NumDef$ +1", liveCard);
        CollectionCaps halfRate = new CollectionCaps(
                CollectionCaps.defaults().manaCap(), 0.5,
                CollectionCaps.defaults().interventionsPerGame(), CollectionCaps.defaults().probesPerGame(),
                CollectionCaps.defaults().probeKeywords(), 0.0, 0.0);

        Path solo = ShardRun.shardOf(tempDir.resolve("solo"), TestCards.game(), halfRate,
                (bracket, collectors) -> collectors.playabilityHandler().invoke(null,
                        methodNamed(PlayabilityListenerShape.class, "onCandidate"),
                        new Object[]{liveCandidate}));
        boolean soloAdmitted = admitsAPlayabilityRecord(solo);

        Card forkCard = TestBoard.cardInAnotherGame("Grizzly Bears");
        SpellAbility forkCandidate = AbilityFactory.getAbility(
                "AB$ Pump | Cost$ 1 | NumAtt$ +1 | NumDef$ +1", forkCard);
        Path paired = ShardRun.shardOf(tempDir.resolve("paired"), TestCards.game(), halfRate,
                (bracket, collectors) -> {
                    collectors.playabilityHandler().invoke(null,
                            methodNamed(PlayabilityListenerShape.class, "onCandidate"),
                            new Object[]{forkCandidate});
                    collectors.playabilityHandler().invoke(null,
                            methodNamed(PlayabilityListenerShape.class, "onCandidate"),
                            new Object[]{liveCandidate});
                });
        boolean pairedAdmitted = admitsAPlayabilityRecord(paired);

        assertEquals(soloAdmitted, pairedAdmitted,
                "a rejected fork candidate ahead of it must not change whether the "
                        + "live candidate behind it is sampled in: solo=" + soloAdmitted
                        + " paired=" + pairedAdmitted);
    }

    /** Whether a run sampled a playability record in; one that sampled none wrote no shard. */
    private static boolean admitsAPlayabilityRecord(Path shard) throws IOException {
        return Files.exists(shard) && ShardRun.readShard(shard).stream()
                .anyMatch(l -> l.contains("\"kind\":\"playability\""));
    }

    // ── gen-2: declarations and the deciding player ─────────────────────
    //
    // The engine reports legality from the turn-based declaration itself and
    // names the deciding player, so every record is a real declaration
    // (what_if=false) whatever the step, and FR-030a's actor is the player
    // the engine names; FR-030 stamps random_seat off that actor. These
    // build a two-player game so the deciding player is a real seat.

    private static List<String> recordsOf(List<String> lines) {
        return lines.stream().filter(l -> l.contains("\"kind\":\"playability\"")).toList();
    }

    /**
     * An attackers record is a real declaration made by the player the
     * engine names, and a candidate the rules keep out is listed as
     * forbidden rather than left out.
     */
    @Test
    void anAttackDeclarationIsRealAndNamesTheDeciderTheEngineReports() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_ATTACKERS);
        Player active = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        Card ready = TestBoard.inPlay(game, active, "Grizzly Bears");
        Card tapped = TestBoard.inPlay(game, active, "Runeclaw Bear");
        tapped.setTapped(true);

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            collectors.combatLegalityHandler().invoke(null,
                    methodNamed(CombatLegalityListenerShape.class, "onAttackersDeclarable"),
                    new Object[]{active, other, List.of(ready, tapped), List.of(ready)});
        });

        List<String> records = recordsOf(lines);
        assertEquals(1, records.size(), records.toString());
        String real = records.get(0);
        assertTrue(real.contains("\"subkind\":\"attackers\""), real);
        assertTrue(real.contains("\"what_if\":false"), real);
        assertTrue(real.contains("\"actor_player\":\"" + SnapshotBuilder.playerId(active) + "\""),
                "the deciding player is the one the engine names: " + real);
        assertTrue(real.contains("\"random_seat\":false"), real);
        assertTrue(real.contains("\"legal_attackers\":[\"" + SnapshotBuilder.entityId(ready) + "\"]"),
                real);
        assertTrue(real.contains("\"forbidden\":[{\"entity\":\""
                + SnapshotBuilder.entityId(tapped) + "\""), real);
    }

    /**
     * Nothing about the step or whose turn it is reclassifies a declaration:
     * the engine reports only real ones, so a report outside the
     * declare-attackers step is still written as real.
     */
    @Test
    void aDeclarationIsRealWhateverTheStep() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        Card theirs = TestBoard.inPlay(game, other, "Grizzly Bears");

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            collectors.combatLegalityHandler().invoke(null,
                    methodNamed(CombatLegalityListenerShape.class, "onAttackersDeclarable"),
                    new Object[]{other, active, List.of(theirs), List.of(theirs)});
        });

        List<String> records = recordsOf(lines);
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"what_if\":false"), records.get(0));
        assertTrue(records.get(0).contains(
                "\"actor_player\":\"" + SnapshotBuilder.playerId(other) + "\""), records.get(0));
    }

    /**
     * FR-030a's blockers row: a blockers record names the blocking player
     * the engine reports, with the attacker it anchors on and how many
     * blockers that attacker takes.
     */
    @Test
    void aBlockDeclarationIsRealAndNamesTheBlockingPlayer() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player attacking = game.getPlayers().get(0);
        Player blocking = game.getPlayers().get(1);
        Card attacker = TestBoard.inPlay(game, attacking, "Grizzly Bears");
        Card blocker = TestBoard.inPlay(game, blocking, "Hill Giant");
        Combat combat = new Combat(attacking);
        combat.addAttacker(attacker, blocking);
        game.getPhaseHandler().setCombat(combat);

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            collectors.combatLegalityHandler().invoke(null,
                    methodNamed(CombatLegalityListenerShape.class, "onBlockersDeclarable"),
                    new Object[]{blocking, attacker, List.of(blocker), List.of(blocker), 1});
        });

        List<String> records = recordsOf(lines);
        assertEquals(1, records.size(), records.toString());
        String real = records.get(0);
        assertTrue(real.contains("\"subkind\":\"blockers\""), real);
        assertTrue(real.contains("\"what_if\":false"), real);
        assertTrue(real.contains("\"actor_player\":\"" + SnapshotBuilder.playerId(blocking) + "\""),
                "the deciding player is the blocking player: " + real);
        assertTrue(real.contains("\"anchor_attacker\":\"" + SnapshotBuilder.entityId(attacker) + "\""),
                real);
        assertTrue(real.contains("\"min_blockers\":1"), real);
    }

    /**
     * Every declaration on a board that moved is written, and the same
     * declaration on an unchanged board is written once (FR-028).
     */
    @Test
    void everyDeclarationOnAMovedBoardIsWrittenAndARepeatIsNot() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_ATTACKERS);
        Player active = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        Card mine = TestBoard.inPlay(game, active, "Grizzly Bears");

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            Method declarable = methodNamed(CombatLegalityListenerShape.class, "onAttackersDeclarable");
            for (int turn = 0; turn < 20; turn++) {
                // A life total that moves is enough to make each snapshot new.
                active.setLife(20 - turn, null);
                collectors.combatLegalityHandler().invoke(null, declarable,
                        new Object[]{active, other, List.of(mine), List.of(mine)});
            }
            collectors.combatLegalityHandler().invoke(null, declarable,
                    new Object[]{active, other, List.of(mine), List.of(mine)});
        });

        assertEquals(20, recordsOf(lines).size());
    }

    /**
     * Spec Story 3 scenario 6 on the hook path: a record whose actor is the
     * random seat says so.
     */
    @Test
    void randomSeatIsStampedOffTheActor() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_ATTACKERS);
        Player active = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        Card mine = TestBoard.inPlay(game, active, "Grizzly Bears");

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            collectors.setRandomSeatPlayerId(SnapshotBuilder.playerId(active));
            collectors.combatLegalityHandler().invoke(null,
                    methodNamed(CombatLegalityListenerShape.class, "onAttackersDeclarable"),
                    new Object[]{active, other, List.of(mine), List.of(mine)});
        });

        List<String> records = recordsOf(lines);
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"random_seat\":true"), records.get(0));
    }

    /**
     * The combat stand-in is what Forge declares, position by position.
     * Reflective, so an unpatched checkout skips the assertion.
     */
    @Test
    void theCombatLegalityStandInIsForgesOwnDeclaration() {
        PatchHooks.Lookup setter = PatchHooks.find(
                PatchHooks.COMBAT_LEGALITY_OBSERVER, "setEffectRecordCombatListener");
        Assumptions.assumeTrue(setter.present(), "the checkout lacks the combat hook");
        Class<?> declared = setter.method().getParameterTypes()[0];
        for (String name : List.of("onAttackersDeclarable", "onBlockersDeclarable")) {
            Class<?>[] theirs = methodNamed(declared, name).getParameterTypes();
            Class<?>[] ours = methodNamed(CombatLegalityListenerShape.class, name).getParameterTypes();
            assertArrayEquals(ours, theirs, name);
        }
    }

    /**
     * FR-030a's decision row: a candidate evaluated on the opponent's turn
     * names the candidate's controller, not the active player.
     */
    @Test
    void aDecisionRecordNamesTheCandidatesController() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player other = game.getPlayers().get(1);
        Card theirs = TestBoard.inPlay(game, other, "Grizzly Bears");
        SpellAbility candidate = AbilityFactory.getAbility(
                "AB$ Pump | Cost$ 1 | NumAtt$ +1 | NumDef$ +1", theirs);
        candidate.setActivatingPlayer(other);

        List<String> lines = ShardRun.records(tempDir, game, caps(0, 1.0), (bracket, collectors) -> {
            collectors.playabilityHandler().invoke(null,
                    methodNamed(PlayabilityListenerShape.class, "onCandidate"),
                    new Object[]{candidate});
        });

        List<String> records = recordsOf(lines);
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains(
                "\"actor_player\":\"" + SnapshotBuilder.playerId(other) + "\""), records.get(0));
        assertFalse(records.get(0).contains("what_if"),
                "a decision record carries no what_if: " + records.get(0));
    }

    /** The two knobs of the random seat travel like every other cap. */
    @Test
    void theRandomSeatKnobsAreReadFromSystemProperties() {
        System.setProperty("effect.random.seat.share", "0.125");
        System.setProperty("effect.random.seat.probability", "0.3");
        try {
            CollectionCaps caps = CollectionCaps.fromSystemProperties();
            assertEquals(0.125, caps.randomSeatShare());
            assertEquals(0.3, caps.randomSeatProbability());
            assertTrue(caps.randomSeatEnabled());
        } finally {
            System.clearProperty("effect.random.seat.share");
            System.clearProperty("effect.random.seat.probability");
        }
        assertEquals(0.0, CollectionCaps.defaults().randomSeatShare());
        assertFalse(CollectionCaps.defaults().randomSeatEnabled());
    }
}
