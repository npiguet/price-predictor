package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.PatchedCollectors.CollectionCaps;
import com.pricepredictor.connector.effects.PatchedCollectors.Contribution;
import forge.card.CardChangedType;
import forge.card.CardType;
import forge.card.ColorSet;
import forge.card.RemoveType;
import forge.card.StateChangedType;
import forge.card.WordChangedType;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
class PatchedCollectorTest {

    private static CollectionCaps caps(int manaCap, double playabilityRate) {
        return new CollectionCaps(manaCap, playabilityRate, 2, 2, List.of());
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
        System.setProperty("effect.mana.cap", "not-a-number");
        try {
            assertEquals(
                    CollectionCaps.defaults().manaCap(),
                    CollectionCaps.fromSystemProperties().manaCap());
        } finally {
            System.clearProperty("effect.mana.cap");
        }
    }

    @Test
    void probesAreOffUntilKeywordsAreNamed() {
        assertFalse(CollectionCaps.defaults().probesEnabled());
        assertTrue(new CollectionCaps(2000, 0.1, 2, 2, List.of("wither"))
                .probesEnabled());
    }

    // ── the mana reservoir ──────────────────────────────────────────────

    @Test
    void theFirstActivationsFillTheReservoir() {
        PatchedCollectors collector = collectors(caps(3, 1.0));
        assertEquals(0, collector.manaReservoirSlot("Mana$ G"));
        assertEquals(1, collector.manaReservoirSlot("Mana$ G"));
        assertEquals(2, collector.manaReservoirSlot("Mana$ G"));
    }

    @Test
    void theReservoirIsPerUniqueManaAbilityText() {
        // A basic land taps a dozen times a game; a rare mana ability must not
        // be starved by it.
        PatchedCollectors collector = collectors(caps(1, 1.0));
        assertEquals(0, collector.manaReservoirSlot("Mana$ G"));
        assertEquals(0, collector.manaReservoirSlot("Mana$ Any | Amount$ 2"));
    }

    @Test
    void aCapOfZeroRecordsNothing() {
        assertEquals(-1, collectors(caps(0, 1.0)).manaReservoirSlot("Mana$ G"));
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
            PatchedCollectors collector = collectors(caps(1, 1.0), trial);
            int held = -1;
            for (int i = 0; i < activations; i++) {
                if (collector.manaReservoirSlot("Mana$ G") >= 0) {
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
            PatchedCollectors collector = collectors(caps(1, 1.0), seed);
            collector.manaReservoirSlot("Mana$ G");
            replaced = collector.manaReservoirSlot("Mana$ G") == 0;
        }
        assertTrue(replaced, "no seed ever replaced the first activation");
    }

    // ── continuous coalescing ───────────────────────────────────────────

    @Test
    void aStaticIsRecordedOncePerStableBoard() {
        PatchedCollectors collector = collectors(caps(2000, 1.0));
        assertTrue(collector.allowContinuousRecord("anthem", "board-1"));
        assertFalse(collector.allowContinuousRecord("anthem", "board-1"));
    }

    @Test
    void aChangedBoardGetsANewRecord() {
        PatchedCollectors collector = collectors(caps(2000, 1.0));
        assertTrue(collector.allowContinuousRecord("anthem", "board-1"));
        assertTrue(collector.allowContinuousRecord("anthem", "board-2"));
    }

    @Test
    void twoStaticsOnOneBoardEachGetARecord() {
        PatchedCollectors collector = collectors(caps(2000, 1.0));
        assertTrue(collector.allowContinuousRecord("anthem", "board-1"));
        assertTrue(collector.allowContinuousRecord("crusade", "board-1"));
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
     * Exactly the hooks the checkout offers get installed, and no others.
     *
     * <p>Asserted against the checkout rather than against a fixed count: the
     * sibling Forge is patched or not independently of this repository, so a
     * test expecting zero passes only until someone applies the patches. What
     * must hold either way is that install() finds what is there — a worker
     * degrades on a stock checkout and collects fully on a patched one.
     */
    @Test
    void everyHookTheCheckoutOffersIsInstalledAndNoOthers() {
        int available = 0;
        if (PatchHooks.find(PatchHooks.REPLACEMENT_HANDLER,
                "setEffectRecordListener").present()) {
            available++;
        }
        if (PatchHooks.find(PatchHooks.TRIGGER_HANDLER,
                "setEffectRecordTriggerListener").present()) {
            available++;
        }
        if (PatchHooks.find(PatchHooks.AI_CONTROLLER,
                "setEffectRecordPlayabilityListener").present()) {
            available++;
        }
        if (PatchHooks.find(PatchHooks.ABILITY_MANA_PART,
                "setEffectRecordManaListener").present()) {
            available++;
        }
        if (PatchHooks.find(PatchHooks.AI_CONTROLLER,
                "setEffectRecordCombatListener").present()) {
            available++;
        }
        try (PatchedCollectors collector = collectors(CollectionCaps.defaults())) {
            assertEquals(available, collector.install());
            assertEquals(available, collector.installedHooks().size());
        }
    }

    /**
     * The two read-only hooks answer null outside a resolution.
     *
     * <p>True on a stock checkout because the method does not exist, and true
     * on a patched one because no trigger or sub-ability is resolving on this
     * thread — the test asserts the caller-visible behaviour that holds either
     * way.
     */
    @Test
    void theCauseAndSubAbilityHooksReadNullOutsideAResolution() {
        assertEquals(null, PatchHooks.currentTriggerCause());
        assertEquals(null, PatchHooks.currentSubAbility());
    }
}
