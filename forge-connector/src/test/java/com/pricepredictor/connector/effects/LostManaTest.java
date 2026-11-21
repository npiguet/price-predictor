package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.card.MagicColor;
import forge.card.mana.ManaAtom;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.mana.Mana;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordAbilities.scriptOf;
import static com.pricepredictor.connector.effects.RecordEvents.events;
import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static com.pricepredictor.connector.effects.RecordPicks.only;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * A {@code mana_lost}'s {@code mana_by_color} on boards the engine really
 * plays (task B51): a drain's effect half names the mana the pool lost, and
 * a {@code LoseMana} replacement's rewrite record names the mana that would
 * have been lost, read off the pool before the replacement converts it.
 */
@ExtendWith(ForgeExtension.class)
class LostManaTest {

    private static final String WHY = "for the drain's effect half and the rewrite records";
    private static final String MANA_SHORT = "cardsfolder/m/mana_short.txt";
    private static final String HORIZON_STONE = "cardsfolder/h/horizon_stone.txt";

    @TempDir
    Path tempDir;

    /** Mana Short at a player floating {R}{G}: its effect half names that player and the two mana. */
    @Test
    void aDrainsEffectHalfNamesTheManaThePoolLost() throws Throwable {
        Game game = game();
        Player drained = game.getPlayers().get(1);
        Card land = TestBoard.inPlay(game, drained, "Forest");
        floating(drained, land, MagicColor.RED, MagicColor.GREEN);

        Map<String, Object> lost = onlyManaLost(manaShortAt(game, drained));

        assertEquals(List.of(Refs.of(drained)), lost.get("subjects"), lost.toString());
        assertEquals(Map.of("R", 1L, "G", 1L), params(lost).get("mana_by_color"), lost.toString());
        assertEquals(0, drained.getManaPool().totalMana(), "the pool was emptied");
    }

    /** Colourless mana, which Forge keeps under its own atom, is spelled {@code C}, one per unit. */
    @Test
    void drainedColourlessManaIsSpelledC() throws Throwable {
        Game game = game();
        Player drained = game.getPlayers().get(1);
        Card land = TestBoard.inPlay(game, drained, "Wastes");
        floating(drained, land, (byte) ManaAtom.COLORLESS, (byte) ManaAtom.COLORLESS);

        Map<String, Object> lost = onlyManaLost(manaShortAt(game, drained));

        assertEquals(Map.of("C", 2L), params(lost).get("mana_by_color"), lost.toString());
    }

    /**
     * Horizon Stone and {U}{U} floating as main phase 1 ends: the
     * replacement's record names the {U}{U} that would have been lost, and no
     * outgoing event, since the mana became colourless instead of leaving.
     */
    @Test
    void aLoseManaRewriteNamesTheManaThatWouldHaveBeenLost() throws Throwable {
        Game game = game();
        Player you = game.getPlayers().get(0);
        Card stone = TestBoard.inPlay(game, you, "Horizon Stone");
        floating(you, stone, MagicColor.BLUE, MagicColor.BLUE);

        Map<String, Object> rewrite = onlyRewrite(ShardRun.recordsWithClauseHook(
                tempDir, game, WHY, () -> pastTheStep(game)));

        Map<String, Object> incoming = asMap(rewrite.get("incoming"));
        assertEquals("mana_lost", incoming.get("type"), rewrite.toString());
        assertEquals(List.of(Refs.of(you)), incoming.get("subjects"), rewrite.toString());
        assertEquals(Map.of("U", 2L), params(incoming).get("mana_by_color"), rewrite.toString());
        assertNull(rewrite.get("outgoing"), rewrite.toString());
        assertEquals(2, you.getManaPool().getAmountOfColor((byte) ManaAtom.COLORLESS), "the mana became colourless");
    }

    /**
     * Omnath, Locus of Mana keeps green mana as steps end: of {G}{U} at the
     * step's end, only the {U} would have been lost.
     */
    @Test
    void aStepsEndLeavesOutTheManaAStaticKeeps() throws Throwable {
        Game game = game();
        Player you = game.getPlayers().get(0);
        Card stone = TestBoard.inPlay(game, you, "Horizon Stone");
        TestBoard.inPlay(game, you, "Omnath, Locus of Mana");
        floating(you, stone, MagicColor.GREEN, MagicColor.BLUE);

        Map<String, Object> rewrite = onlyRewrite(ShardRun.recordsWithClauseHook(
                tempDir, game, WHY, () -> pastTheStep(game)));

        assertEquals(Map.of("U", 1L), params(asMap(rewrite.get("incoming"))).get("mana_by_color"),
                rewrite.toString());
    }

    /**
     * Mana Short at a Horizon Stone controller floating {G}{U} beside Omnath:
     * a drain empties green too, so the rewrite names both; the mana became
     * colourless, so the drain's effect half lost nothing and holds no
     * {@code mana_lost}.
     */
    @Test
    void aDrainAReplacementTurnsColourlessLosesNothing() throws Throwable {
        Game game = game();
        Player drained = game.getPlayers().get(1);
        Card stone = TestBoard.inPlay(game, drained, "Horizon Stone");
        TestBoard.inPlay(game, drained, "Omnath, Locus of Mana");
        floating(drained, stone, MagicColor.GREEN, MagicColor.BLUE);

        List<String> all = manaShortAt(game, drained);

        assertEquals(List.of(), manaLost(only(all, "resolution", MANA_SHORT)), all.toString());
        Map<String, Object> rewrite = onlyRewrite(all);
        assertEquals(Map.of("G", 1L, "U", 1L), params(asMap(rewrite.get("incoming"))).get("mana_by_color"),
                rewrite.toString());
        assertEquals(2, drained.getManaPool().totalMana(), "the mana stayed, colourless");
    }

    // ── the boards ──────────────────────────────────────────────────────

    /** Mana Short cast at {@code drained} and resolved: every record written. */
    private List<String> manaShortAt(Game game, Player drained) throws Throwable {
        Player caster = game.getPlayers().get(0);
        Card spell = TestBoard.inHand(game, caster, "Mana Short");
        return ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.cast(game, spell, caster, drained);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.nextStep(game);
        });
    }

    /** The engine plays on until main phase 1 has ended, emptying the pools (500.4). */
    private static void pastTheStep(Game game) {
        PhaseHandler phases = game.getPhaseHandler();
        for (int step = 0; step < 10 && phases.is(PhaseType.MAIN1); step++) {
            phases.mainLoopStep();
        }
        assertFalse(phases.is(PhaseType.MAIN1), "the game left main phase 1");
    }

    private static void floating(Player player, Card source, byte... colors) {
        for (byte color : colors) {
            player.getManaPool().addMana(new Mana(color, source, null, player));
        }
    }

    private static Game game() {
        return TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
    }

    // ── the records ─────────────────────────────────────────────────────

    private static List<Map<String, Object>> manaLost(String record) {
        return events(record).stream().filter(event -> "mana_lost".equals(event.get("type"))).toList();
    }

    private static Map<String, Object> onlyManaLost(List<String> all) {
        List<Map<String, Object>> lost = manaLost(only(all, "resolution", MANA_SHORT));
        assertEquals(1, lost.size(), all.toString());
        return lost.get(0);
    }

    /** The payload of the one rewrite record Horizon Stone's replacement wrote. */
    private static Map<String, Object> onlyRewrite(List<String> all) {
        List<Map<String, Object>> rewrites = all.stream()
                .filter(record -> "rewrite".equals(parse(record).get("kind"))
                        && HORIZON_STONE.equals(scriptOf(record)))
                .map(RecordEvents::payload)
                .toList();
        assertEquals(1, rewrites.size(), all.toString());
        return rewrites.get(0);
    }
}
