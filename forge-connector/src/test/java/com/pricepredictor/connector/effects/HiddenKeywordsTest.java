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
import static com.pricepredictor.connector.effects.RecordLines.events;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static com.pricepredictor.connector.effects.RecordLines.records;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Forge's hidden keywords — "target creature can't block this turn", a
 * {@code KW$ HIDDEN} pump — in the effect half that grants one, in the
 * snapshots after it and among the floating effects (V2 #14, X1 #10,
 * C2 #8).
 */
@ExtendWith(ForgeExtension.class)
class HiddenKeywordsTest {

    private static final String WHY = "for the clause notes of floating effects";
    private static final String CANT_BLOCK = "CARDNAME can't block.";

    @TempDir
    Path tempDir;

    @Test
    void cantBlockThisTurnIsAnEventAnEntityFactAndAFloatingEffect() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, opponent, "Grizzly Bears");
        Card abandon = TestBoard.inHand(game, caster, "Abandon the Post");
        Card shock = TestBoard.inHand(game, caster, "Shock");

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.cast(game, abandon, caster, bears);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            // A later record's board: Shock's cost half, at the opponent.
            TestBoard.cast(game, shock, caster, opponent);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.nextStep(game);
        });

        String half = RecordLines.only(all, "resolution", "cardsfolder/a/abandon_the_post.txt");
        Map<String, Object> gained = events(half).stream()
                .filter(e -> "keyword_change".equals(e.get("type"))
                        && Boolean.TRUE.equals(params(e).get("hidden")))
                .findFirst().orElseThrow(() -> new AssertionError(half));
        assertEquals(List.of(Refs.of(bears)), gained.get("subjects"), half);
        assertEquals(List.of(CANT_BLOCK), params(gained).get("keywords"), half);
        assertEquals(false, params(gained).get("removed"), half);

        String later = RecordLines.only(all, "activation", "cardsfolder/s/shock.txt");
        Map<String, Object> state = asMap(parse(later).get("state"));
        Map<String, Object> entity = ((List<?>) state.get("entities")).stream()
                .map(RecordLines::asMap).filter(e -> Refs.of(bears).equals(e.get("id")))
                .findFirst().orElseThrow(() -> new AssertionError(later));
        assertEquals(List.of(CANT_BLOCK),
                asMap(entity.get("granted_temporary")).get("hidden_keywords"), later);
        List<?> floating = (List<?>) asMap(state.get("global")).get("floating_effects");
        assertTrue(floating.stream().map(RecordLines::asMap)
                        .anyMatch(f -> ((List<?>) f.get("affected")).contains(Refs.of(bears))
                                && String.valueOf(f.get("source_lines")).contains("abandon_the_post")),
                "the grant is a floating effect of Abandon the Post's: " + floating);
        assertTrue(records(all, "resolution").size() >= 2, all.toString());
    }
}
