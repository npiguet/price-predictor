package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.ai.ComputerUtil;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import static com.pricepredictor.connector.effects.RecordLines.eventsNaming;
import static com.pricepredictor.connector.effects.RecordLines.records;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What a cost payment publishes reaches no record: the paid ability's cost
 * half holds what was paid in its {@code costs}, never an event. Each such
 * event is counted as dropped (FR-041), the ones outside the cost half's six
 * channels included — a pain land's damage to its controller here.
 */
@ExtendWith(ForgeExtension.class)
class PaymentEventsTest {

    @TempDir
    Path tempDir;

    @Test
    void aPainLandsDamageWhilePayingIsCountedAsDropped() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        TestBoard.inPlay(game, caster, "Shivan Reef");
        Card shock = TestBoard.inHand(game, caster, "Shock");
        SpellAbility spell = TestBoard.spellOf(shock, caster);
        spell.resetTargets();
        spell.getTargets().add(opponent);
        AtomicReference<Map<DroppedEvents.Key, Long>> counted = new AtomicReference<>();

        List<String> all = ShardRun.observed(tempDir, game, collector -> {
            assertTrue(ComputerUtil.handlePlayingSpellAbility(caster, spell, null),
                    "the AI cast the Shock, paying with the Reef's red");
            assertEquals(19, caster.getLife(), "the Reef dealt 1 damage to its controller");
            counted.set(Map.copyOf(collector.droppedEvents().counts()));
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            TestBoard.nextStep(game);
        });

        for (String record : records(all, "resolution")) {
            assertEquals(List.of(), eventsNaming(record, "P" + caster.getId()),
                    "nothing the Reef dealt reaches a record: " + record);
        }
        assertEquals(1L, counted.get().get(new DroppedEvents.Key("damage_dealt", "main1")),
                "the Reef's damage, counted as dropped: " + counted.get());
    }
}
