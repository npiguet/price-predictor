package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.ai.PlayerControllerAi;
import forge.game.ability.AbilityFactory;
import forge.game.player.Player;
import forge.game.spellability.AbilityManaPart;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

/** What a game's recording leaves behind on the engine's static hooks. */
@ExtendWith(ForgeExtension.class)
class GameRecordingTest {

    @TempDir
    Path tempDir;

    /**
     * Every hook comes down even when writing what is held throws.
     *
     * <p>The hooks are JVM statics: one left installed would hand the next
     * game's callbacks to this game's channels. A mana record is held until
     * close, so a shard closed under it makes the flush throw.
     */
    @Test
    void closeTakesEveryHookDownEvenWhenTheFlushThrows() throws Exception {
        Assumptions.assumeTrue(AttributionMode.detect() == AttributionMode.PATCHED
                        && PatchHooks.MANA_LISTENER.present()
                        && PatchHooks.CONFIRM_LISTENER.present(),
                "../forge has no mana or confirm hook");
        SpellAbility mana = AbilityFactory.getAbility(
                "AB$ Mana | Cost$ T | Produced$ G", TestCards.build("Llanowar Elves"));
        Player producer = new Player("live-player", TestCards.game(), 93201);
        RecordShardWriter writer = ShardRun.writer(tempDir);
        GameRecording recording = new GameRecording(
                TestCards.game(), writer, ShardRun.GAME_ID,
                TestCaps.caps().everyDecision().interventions(0).probes(0).build(), 1L, null);

        recording.open();
        Object manaListener = installed(AbilityManaPart.class, "MANA_HOOK");
        assertNotNull(manaListener, "open installs the mana hook");
        assertNotNull(installed(PlayerControllerAi.class, "CONFIRM_HOOK"),
                "open installs the confirm hook");
        Object paymentListener = installed(forge.game.cost.CostPayment.class, "PAYMENT_HOOK");
        assertNotNull(paymentListener, "open installs the payment hook");
        callback(paymentListener, "onPayingCosts").invoke(paymentListener, mana, producer, false);
        callback(paymentListener, "onCostsPaid").invoke(paymentListener, mana, producer, true);
        callback(manaListener, "onManaProduced").invoke(manaListener, mana, producer, "G");
        writer.close();

        assertThrows(IllegalStateException.class, recording::close);
        assertNull(installed(AbilityManaPart.class, "MANA_HOOK"),
                "the mana hook outlived a close that threw");
        assertNull(installed(PlayerControllerAi.class, "CONFIRM_HOOK"),
                "the confirm hook outlived a close that threw");
    }

    /**
     * A steered game's records carry its two decks' rules as {@code aimed_at},
     * de-duplicated and sorted, in place of the run's list (FR-058d); a game
     * of the same run whose decks carry none keeps the run's.
     */
    @Test
    void aSteeredGamesRecordsCarryItsDecksRules() throws Throwable {
        Assumptions.assumeTrue(AttributionMode.detect() == AttributionMode.PATCHED
                        && PatchHooks.MANA_LISTENER.present(),
                "../forge has no mana hook");

        assertEquals(java.util.List.of("\"aimed_at\":[\"702.19b\",\"704.5f\"]"),
                aimedAtOfAManaRecord(DeckRules.of("704.5f", "702.19b"), tempDir.resolve("a")));
        assertEquals(java.util.List.of("\"aimed_at\":[\"704.5f\"]"),
                aimedAtOfAManaRecord(DeckRules.of(null, "704.5f"), tempDir.resolve("b")));
        assertEquals(java.util.List.of("\"aimed_at\":[]"),
                aimedAtOfAManaRecord(DeckRules.NONE, tempDir.resolve("c")));
    }

    /**
     * The {@code aimed_at} member of every record a game of a variant run
     * (aimed at no rule) writes when its decks carry {@code decks} and a mana
     * ability is played in it.
     */
    private static java.util.List<String> aimedAtOfAManaRecord(DeckRules decks, Path dir)
            throws Throwable {
        java.nio.file.Files.createDirectories(dir);
        SpellAbility mana = AbilityFactory.getAbility(
                "AB$ Mana | Cost$ T | Produced$ G", TestCards.build("Llanowar Elves"));
        Player producer = new Player("live-player", TestCards.game(), 93202);
        RecordShardWriter writer = ShardRun.writer(dir);
        GameRecording recording = new GameRecording(
                TestCards.game(), writer, ShardRun.GAME_ID,
                TestCaps.caps().everyDecision().interventions(0).probes(0).aimedAt().build(),
                1L, null, ForkSeats.NONE, decks);
        try {
            recording.open();
            Object paymentListener = installed(forge.game.cost.CostPayment.class, "PAYMENT_HOOK");
            Object manaListener = installed(AbilityManaPart.class, "MANA_HOOK");
            callback(paymentListener, "onPayingCosts").invoke(paymentListener, mana, producer, false);
            callback(paymentListener, "onCostsPaid").invoke(paymentListener, mana, producer, true);
            callback(manaListener, "onManaProduced").invoke(manaListener, mana, producer, "G");
        } finally {
            recording.close();
            writer.close();
        }
        java.util.List<String> lines = ShardRun.readShard(writer.path());
        assertFalse(lines.isEmpty(), "the mana ability wrote no record");
        return lines.stream()
                .map(line -> {
                    int start = line.indexOf("\"aimed_at\":");
                    return start < 0 ? "" : line.substring(start, line.indexOf(']', start) + 1);
                })
                .distinct()
                .toList();
    }

    /**
     * A close after an {@code open()} that threw part way takes down what was
     * installed and does not trip over what never was, which is what lets the
     * caller open inside the try whose finally closes.
     */
    @Test
    void closeAfterAnOpenThatStoppedPartWayTakesDownWhatItInstalled() throws Exception {
        Assumptions.assumeTrue(AttributionMode.detect() == AttributionMode.PATCHED
                        && PatchHooks.MANA_LISTENER.present()
                        && PatchHooks.CONFIRM_LISTENER.present(),
                "../forge has no mana or confirm hook");
        RecordShardWriter writer = ShardRun.writer(tempDir);
        GameRecording recording = new GameRecording(
                TestCards.game(), writer, ShardRun.GAME_ID, CollectionCaps.defaults(), 1L, null);
        // What an open() that threw after its first hook leaves behind.
        PatchHooks.CONFIRM_LISTENER.install((proxy, method, args) -> null);
        assertNotNull(installed(PlayerControllerAi.class, "CONFIRM_HOOK"));

        recording.close();
        writer.close();

        assertNull(installed(PlayerControllerAi.class, "CONFIRM_HOOK"),
                "the confirm hook outlived the close");
        assertNull(installed(AbilityManaPart.class, "MANA_HOOK"));
    }

    /** The listener a patched engine class's hook holds, or null when none is. */
    private static Object installed(Class<?> owner, String hookField) throws Exception {
        Field field;
        try {
            field = owner.getDeclaredField(hookField);
        } catch (NoSuchFieldException e) {
            Assumptions.abort(owner.getSimpleName() + " has no " + hookField);
            return null;
        }
        field.setAccessible(true);
        Object hook = field.get(null);
        return hook.getClass().getMethod("get").invoke(hook);
    }

    /** The listener interface's method of that name, which the proxy answers. */
    private static Method callback(Object listener, String name) {
        return Arrays.stream(listener.getClass().getInterfaces())
                .flatMap(type -> Arrays.stream(type.getMethods()))
                .filter(method -> method.getName().equals(name))
                .findFirst()
                .orElseThrow();
    }

    /**
     * A phase boundary's collection steps are guarded each on its own (C6): a
     * step that throws costs its own records at that boundary, and every
     * step after it still runs.
     */
    @Test
    void aBoundaryStepThatThrowsLeavesTheStepsAfterItRunning() {
        java.util.List<String> ran = new java.util.ArrayList<>();

        GameRecording.eachGuarded(
                new GameRecording.BoundaryStep("continuous", () -> {
                    ran.add("continuous");
                    throw new IllegalStateException("a static this board cannot render");
                }),
                new GameRecording.BoundaryStep("interventions", () -> ran.add("interventions")),
                new GameRecording.BoundaryStep("probes", () -> ran.add("probes")),
                new GameRecording.BoundaryStep("directed forks", () -> ran.add("directed forks")));

        org.junit.jupiter.api.Assertions.assertEquals(
                java.util.List.of("continuous", "interventions", "probes", "directed forks"), ran);
    }
}
