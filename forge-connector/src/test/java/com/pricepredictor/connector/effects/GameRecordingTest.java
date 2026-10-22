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
import java.util.List;

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
        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        GameRecording recording = new GameRecording(
                TestCards.game(), writer, "run.0-l1.0",
                new CollectionCaps(1, 1.0, 0, 0, List.of(), 0.0, 0.0), 1L, null);

        recording.open();
        Object manaListener = installed(AbilityManaPart.class, "MANA_HOOK");
        assertNotNull(manaListener, "open installs the mana hook");
        assertNotNull(installed(PlayerControllerAi.class, "CONFIRM_HOOK"),
                "open installs the confirm hook");
        callback(manaListener, "onManaProduced").invoke(manaListener, mana, producer, "G");
        writer.close();

        assertThrows(IllegalStateException.class, recording::close);
        assertNull(installed(AbilityManaPart.class, "MANA_HOOK"),
                "the mana hook outlived a close that threw");
        assertNull(installed(PlayerControllerAi.class, "CONFIRM_HOOK"),
                "the confirm hook outlived a close that threw");
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
}
