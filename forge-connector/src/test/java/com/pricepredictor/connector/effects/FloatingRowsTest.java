package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The resolved effects a snapshot's {@code floating_effects} lists by
 * timestamp, beyond the layer tables: a resolved control change and a
 * resolved can't-have row (C2 #8).
 */
@ExtendWith(ForgeExtension.class)
class FloatingRowsTest {

    private static final String WHY = "for the clause notes of floating effects";

    @TempDir
    Path tempDir;

    /** The floating effects of the board of the one cost half acting through {@code script}. */
    private static List<Map<String, Object>> floatingOf(List<String> all, String script) {
        String later = RecordLines.only(all, "activation", script);
        Map<String, Object> state = asMap(parse(later).get("state"));
        return ((List<?>) asMap(state.get("global")).get("floating_effects")).stream()
                .map(RecordLines::asMap).toList();
    }

    /**
     * Blatant Thievery's control change, which writes no layer-table row,
     * is a floating effect of its clause, lasting indefinitely, on the
     * creature it took. (Act of Treason's haste shares the change's
     * timestamp, so its row already showed.)
     */
    @Test
    void aResolvedControlChangeIsAFloatingEffect() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, opponent, "Grizzly Bears");
        Card thievery = TestBoard.inHand(game, caster, "Blatant Thievery");
        Card shock = TestBoard.inHand(game, caster, "Shock");

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.cast(game, thievery, caster, bears);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.cast(game, shock, caster, opponent);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.nextStep(game);
        });

        assertEquals(caster, bears.getController(), "Blatant Thievery took the bears");
        List<Map<String, Object>> floating = floatingOf(all, "cardsfolder/s/shock.txt");
        assertTrue(floating.stream().anyMatch(f -> String.valueOf(f.get("id")).startsWith("F")
                        && ((List<?>) f.get("affected")).contains(Refs.of(bears))
                        && "permanent".equals(f.get("duration"))
                        && String.valueOf(f.get("source_lines")).contains("blatant_thievery")),
                "the control change is a floating effect: " + floating);
    }

    /** A static's control change (Control Magic) is no floating effect. */
    @Test
    void aStaticsControlChangeIsNoFloatingEffect() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, opponent, "Grizzly Bears");
        Card magic = TestBoard.inHand(game, caster, "Control Magic");
        Card shock = TestBoard.inHand(game, caster, "Shock");

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.cast(game, magic, caster, bears);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.cast(game, shock, caster, opponent);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.nextStep(game);
        });

        assertEquals(caster, bears.getController(), "Control Magic took the bears");
        List<Map<String, Object>> floating = floatingOf(all, "cardsfolder/s/shock.txt");
        assertTrue(floating.stream().noneMatch(f -> String.valueOf(f.get("id")).startsWith("F")),
                "no timestamped floating effect: " + floating);
    }
}
