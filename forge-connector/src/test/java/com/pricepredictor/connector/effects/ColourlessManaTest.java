package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.ai.ComputerUtil;
import forge.game.Game;
import forge.game.ability.ApiType;
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

import static com.pricepredictor.connector.effects.RecordEvents.payload;
import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static com.pricepredictor.connector.effects.RecordPicks.only;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Colourless mana counted under {@code C} on boards the engine really plays
 *: Forge keys a colourless unit as {@code ManaAtom.COLORLESS}, not
 * {@code MagicColor.COLORLESS}, and a floating pool, a cost half's {@code
 * mana_by_color} and a cast entity's {@code costs_paid.mana} count it under
 * {@code C}, one per unit, as {@code mana_produced} and {@code mana_lost} do.
 */
@ExtendWith(ForgeExtension.class)
class ColourlessManaTest {

    private static final String WHY = "for the mana records and the cast's halves";
    private static final String MOUNTAIN = "cardsfolder/m/mountain.txt";
    private static final String THINK_TWICE = "cardsfolder/t/think_twice.txt";

    @TempDir
    Path tempDir;

    /** Sol Ring tapped for {C}{C}, then a Mountain: the Mountain's snapshot floats {C}{C}. */
    @Test
    void solRingsManaFloatsUnderC() throws Throwable {
        Game game = game();
        Player you = you(game);
        Card ring = TestBoard.inPlay(game, you, "Sol Ring");
        Card mountain = TestBoard.inPlay(game, you, "Mountain");

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            tapForMana(game, you, ring);
            tapForMana(game, you, mountain);
        });

        String mana = only(all, "resolution", MOUNTAIN);
        assertEquals(Map.of("W", 0L, "U", 0L, "B", 0L, "R", 0L, "G", 0L, "C", 2L),
                asMap(player(mana, you).get("floating_mana")), mana);
    }

    /**
     * Think Twice paid with an Island and a Wastes: its cost half counts the
     * {C} that paid the generic, and so does the spell's {@code costs_paid}
     * on its effect half's board.
     */
    @Test
    void aGenericCostPaidWithColourlessCountsItUnderC() throws Throwable {
        List<String> all = thinkTwicePaidWith("Island", "Wastes");

        assertEquals(Map.of("U", 1L, "C", 1L), costHalfMana(all), all.toString());
        assertEquals(Map.of("W", 0L, "U", 1L, "B", 0L, "R", 0L, "G", 0L, "C", 1L),
                costsPaidMana(all), all.toString());
    }

    /** Think Twice paid with two Islands: the coloured count, and no {@code C}. */
    @Test
    void aColouredPaymentCountsUnderItsColour() throws Throwable {
        List<String> all = thinkTwicePaidWith("Island", "Island");

        assertEquals(Map.of("U", 2L), costHalfMana(all), all.toString());
        assertEquals(Map.of("W", 0L, "U", 2L, "B", 0L, "R", 0L, "G", 0L, "C", 0L),
                costsPaidMana(all), all.toString());
    }

    // ── the boards ──────────────────────────────────────────────────────

    /** Think Twice cast from the hand as the AI casts it, paid with the two lands, and resolved. */
    private List<String> thinkTwicePaidWith(String first, String second) throws Throwable {
        Game game = game();
        Player you = you(game);
        TestBoard.inPlay(game, you, first);
        TestBoard.inPlay(game, you, second);
        game.getAction().moveToLibrary(TestBoard.card(game, you, "Forest"), 0, null);
        SpellAbility spell = TestBoard.spellOf(TestBoard.inHand(game, you, "Think Twice"), you);
        return ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            assertTrue(ComputerUtil.handlePlayingSpellAbility(you, spell, null), "the AI cast Think Twice");
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            TestBoard.nextStep(game);
        });
    }

    private static void tapForMana(Game game, Player you, Card land) {
        SpellAbility mana = TestBoard.ability(land, ApiType.Mana);
        mana.setActivatingPlayer(you);
        assertTrue(ComputerUtil.playNoStack(you, mana, game, false), "the AI tapped " + land);
    }

    private static Game game() {
        return TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
    }

    private static Player you(Game game) {
        return game.getPlayers().get(0);
    }

    // ── the records ─────────────────────────────────────────────────────

    private static Map<String, Object> costHalfMana(List<String> all) {
        String cost = only(all, "activation", THINK_TWICE);
        return asMap(asMap(payload(cost).get("costs")).get("mana_by_color"));
    }

    /** The {@code costs_paid.mana} of the Think Twice its effect half's board holds on the stack. */
    private static Map<String, Object> costsPaidMana(List<String> all) {
        String effect = only(all, "resolution", THINK_TWICE);
        Map<String, Object> spell = stateMember(effect, "entities", entity -> "stack".equals(entity.get("zone")));
        return asMap(asMap(spell.get("costs_paid")).get("mana"));
    }

    private static Map<String, Object> player(String record, Player player) {
        return stateMember(record, "players", entry -> Refs.of(player).equals(entry.get("id")));
    }

    private static Map<String, Object> stateMember(
            String record, String block, java.util.function.Predicate<Map<String, Object>> which) {
        List<Map<String, Object>> found = ((List<?>) asMap(parse(record).get("state")).get(block)).stream()
                .map(RecordLines::asMap)
                .filter(which)
                .toList();
        assertEquals(1, found.size(), record);
        return found.get(0);
    }
}
