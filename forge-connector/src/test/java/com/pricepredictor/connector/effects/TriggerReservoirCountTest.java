package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import forge.game.trigger.Trigger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * What a line's fired-trigger reservoir counts: a firing Forge ran, never an
 * evaluation that was not run.
 */
@ExtendWith(ForgeExtension.class)
class TriggerReservoirCountTest {

    private static final Method ON_CONDITION_EVALUATED =
            HookShapes.method(HookShapes.TriggerFire.class, "onConditionEvaluated");
    private static final Method ON_TRIGGER_RUN =
            HookShapes.method(HookShapes.TriggerFire.class, "onTriggerRun");

    @TempDir
    Path tempDir;

    /**
     * Many evaluations of one line that Forge never runs leave its count at
     * zero: the first firing that is run is the reservoir's first, kept for
     * certain, where counted at evaluation it would be kept with probability
     * {@code cap/n}.
     */
    @Test
    void evaluationsNeverRunLeaveTheReservoirsCountUnchanged() throws Throwable {
        Card card = TestCards.build("Garruk's Uprising");
        List<Trigger> triggers = new ArrayList<>();
        card.getCurrentState().getTriggers().forEach(triggers::add);
        Trigger trigger = triggers.get(0);

        List<String> lines = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) -> {
            var handler = collectors.triggers().handler();
            for (int i = 0; i < 20 * TriggerNegativeSampler.POSITIVES_PER_LINE; i++) {
                handler.invoke(null, ON_CONDITION_EVALUATED,
                        new Object[]{trigger, new HashMap<>(), true});
            }
            Map<?, ?> run = new HashMap<>();
            handler.invoke(null, ON_CONDITION_EVALUATED, new Object[]{trigger, run, true});
            handler.invoke(null, ON_TRIGGER_RUN, new Object[]{trigger, run, null});
        });

        assertEquals(1, lines.size(), lines.toString());
    }
}
