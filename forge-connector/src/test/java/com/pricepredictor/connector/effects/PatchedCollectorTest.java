package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.PatchedCollectors.CollectionCaps;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

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
        return new PatchedCollectors(null, null, "game-1", caps, 42L);
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

    // ── the mana cap ────────────────────────────────────────────────────

    @Test
    void aManaAbilityIsRecordedUpToItsCap() {
        PatchedCollectors collector = collectors(caps(3, 1.0));
        for (int i = 0; i < 3; i++) {
            assertTrue(collector.allowManaRecord("Mana$ G"), "record " + i);
        }
        assertFalse(collector.allowManaRecord("Mana$ G"));
    }

    @Test
    void theCapIsPerUniqueManaAbilityText() {
        // A basic land's tap ability resolves thousands of times a run; a rare
        // mana ability must not be starved by it.
        PatchedCollectors collector = collectors(caps(1, 1.0));
        assertTrue(collector.allowManaRecord("Mana$ G"));
        assertFalse(collector.allowManaRecord("Mana$ G"));
        assertTrue(collector.allowManaRecord("Mana$ Any | Amount$ 2"));
    }

    @Test
    void theCapIsPerWorkerProcess() {
        // Two collectors are two JVM-local counters; neither sees the other's.
        assertTrue(collectors(caps(1, 1.0)).allowManaRecord("Mana$ G"));
        assertTrue(collectors(caps(1, 1.0)).allowManaRecord("Mana$ G"));
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
