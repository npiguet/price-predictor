package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The command-zone objects of a snapshot's global block (FR-024, C4):
 * floating effects, pending delayed triggers and dungeons, each read off a
 * real resolution.
 *
 * <p>The resolutions run with every patched collector installed, because a
 * timestamped floating effect is joined to its line only through the note the
 * clause hook takes; on a checkout without the hook the tests skip.
 */
@ExtendWith(ForgeExtension.class)
class FloatingEffectsTest {

    @TempDir
    Path dir;

    /** "Creatures without flying can't block this turn" is an effect card affecting them. */
    @Test
    void creaturesCantBlockThisTurnIsAnEffectCardKeyedToItsSpell() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.opponent);
        Card angel = board.battlefield("Serra Angel", board.opponent);

        resolveWithHook(board, board.hand("Falter", board.you), null);

        Map<String, Object> falter = onlyFloating(board, "spell");
        assertTrue(Refs.isEntity((String) falter.get("id")), falter.toString());
        assertEquals("cardsfolder/f/falter.txt", onlyKey(falter).get("script_file"));
        assertEquals("end_of_turn", falter.get("duration"));
        List<?> affected = (List<?>) falter.get("affected");
        assertTrue(affected.contains(Refs.of(bears)), falter.toString());
        assertTrue(!affected.contains(Refs.of(angel)), "the angel flies: " + falter);
        assertNull(falter.get("shield_left"));
    }

    /** A prevention shield counts down from its amount, read off the effect card's SVar. */
    @Test
    void aPreventionShieldCarriesTheAmountLeft() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);

        resolveWithHook(board, board.hand("Mending Hands", board.you), bears);

        Map<String, Object> shield = onlyFloating(board, "spell");
        assertEquals(4L, shield.get("shield_left"));
        assertEquals("cardsfolder/m/mending_hands.txt", onlyKey(shield).get("script_file"));
    }

    /**
     * Feral Lightning's "exile them at the beginning of the next end step" is a
     * pending delayed trigger keyed to the spell, remembering the tokens.
     */
    @Test
    void aDelayedEndStepTriggerIsKeyedToTheSpellThatMadeIt() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Assumptions.assumeTrue(PatchHooks.PENDING_DELAYED_TRIGGERS.present(),
                "../forge has no pending-delayed-trigger accessor");

        resolveWithHook(board, board.hand("Feral Lightning", board.you), null);

        List<Map<String, Object>> delayed = list(global(board).get("delayed_triggers"));
        assertEquals(1, delayed.size(), delayed.toString());
        Map<String, Object> exile = delayed.get(0);
        assertTrue(((String) exile.get("id")).startsWith("D"), exile.toString());
        assertEquals("cardsfolder/f/feral_lightning.txt", onlyKey(exile).get("script_file"));
        List<?> remembered = (List<?>) exile.get("remembered");
        assertEquals(3, remembered.size(), "the three tokens: " + exile);
        for (Card token : board.game.getCardsIn(ZoneType.Battlefield)) {
            assertTrue(remembered.contains(Refs.of(token)), exile.toString());
        }
        assertTrue(List.of("end_of_turn", "permanent").contains(exile.get("duration")));
    }

    /**
     * Kick in the Door's haste is a timestamped change on the creature, its
     * "can't be blocked by Walls" an effect card remembering it, and its
     * venture puts its caster in a dungeon's first room.
     */
    @Test
    void aPumpsTimestampIsAFloatingEffectAndAVentureADungeon() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);

        resolveWithHook(board, board.hand("Kick in the Door", board.you), bears);

        List<Map<String, Object>> floating = list(global(board).get("floating_effects"));
        Map<String, Object> haste = floating.stream()
                .filter(f -> ((String) f.get("id")).startsWith("F")).findFirst().orElseThrow();
        assertEquals(List.of(Refs.of(bears)), haste.get("affected"));
        assertEquals("cardsfolder/k/kick_in_the_door.txt", onlyKey(haste).get("script_file"));
        assertEquals("end_of_turn", haste.get("duration"));
        Map<String, Object> walls = floating.stream()
                .filter(f -> Refs.isEntity((String) f.get("id"))).findFirst().orElseThrow();
        assertEquals(List.of(Refs.of(bears)), walls.get("remembered"));

        List<Map<String, Object>> dungeons = list(global(board).get("dungeons"));
        assertEquals(1, dungeons.size(), dungeons.toString());
        Map<String, Object> dungeon = dungeons.get(0);
        assertEquals(Refs.of(board.you), dungeon.get("player"));
        Card card = board.game.getCardsIn(ZoneType.Command).stream()
                .filter(c -> c.getType().isDungeon()).findFirst().orElseThrow();
        assertEquals(Refs.of(card), dungeon.get("dungeon"));
        assertEquals(card.getCurrentRoom(), dungeon.get("room"));
        assertTrue(dungeon.get("room") != null, dungeon.toString());
    }

    /** The monarch's effect card was made by no ability: it lasts until taken away. */
    @Test
    void aDesignationsEffectCardIsPermanentAndKeyedToNoLine() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.game.getAction().becomeMonarch(board.you, null);

        List<Map<String, Object>> floating = list(global(board).get("floating_effects"));

        assertEquals(1, floating.size(), floating.toString());
        assertEquals(List.of(), floating.get(0).get("source_lines"));
        assertEquals("permanent", floating.get(0).get("duration"));
    }

    /** A timestamped change no clause note covers names no line, and is not listed. */
    @Test
    void aPumpNoNoteCoversIsNotListed() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        TestBoard.cast(board.game, board.hand("Giant Growth", board.you), board.you, bears);
        board.game.getStack().resolveStack();

        assertEquals(List.of(), global(board).get("floating_effects"));
        assertEquals(List.of(), global(board).get("dungeons"));
    }

    @ParameterizedTest
    @CsvSource(nullValues = "null", value = {
            "null, end_of_turn",
            "Permanent, permanent",
            "UntilEndOfCombat, end_of_turn",
            "UntilYourNextTurn, end_of_turn",
            "UntilHostLeavesPlay, until_leaves",
            "UntilHostLeavesPlayOrEOT, end_of_turn",
            "UntilLoseControlOfHost, until_leaves",
            "AsLongAsControl, until_leaves",
            "UntilUntaps, until_leaves",
    })
    void aDurationIsOneOfThree(String declared, String bucket) {
        assertEquals(bucket, FloatingEffects.bucket(declared));
    }

    // ── helpers ─────────────────────────────────────────────────────────

    /**
     * Cast a card's spell (at {@code target}, if any) and resolve it with
     * every patched collector installed.
     */
    private void resolveWithHook(CharacterizationBoard board, Card card, Card target) {
        Assumptions.assumeTrue(
                AttributionMode.detect() == AttributionMode.PATCHED
                        && PatchHooks.CLAUSE_LISTENER.present(),
                "../forge has no clause hook");
        Game game = board.game;
        RecordShardWriter writer = ShardRun.writer(dir);
        try (PatchedCollectors collectors = new PatchedCollectors(
                game, TestCollectors.context(writer, ShardRun.GAME_ID),
                TestCaps.caps().interventions(0).probes(0).build(), 1L,
                TestCollectors.bracket(game, writer, ShardRun.GAME_ID))) {
            collectors.install();
            Card onStack = game.getAction().moveToStack(card, null);
            SpellAbility spell = TestBoard.spellOf(onStack, board.you);
            spell.resetTargets();
            if (target != null) {
                spell.getTargets().add(target);
            }
            game.getStack().add(spell);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            game.getAction().checkStaticAbilities();
        } finally {
            writer.close();
        }
    }

    private static Map<String, Object> global(CharacterizationBoard board) {
        return RecordLines.asMap(RecordLines.parse(
                new SnapshotBuilder(board.game).ofBoard().render()).get("global"));
    }

    /** The one floating effect, which is keyed to a line of this trait kind. */
    private static Map<String, Object> onlyFloating(CharacterizationBoard board, String traitKind) {
        List<Map<String, Object>> floating = list(global(board).get("floating_effects"));
        assertEquals(1, floating.size(), floating.toString());
        assertEquals(traitKind, onlyKey(floating.get(0)).get("trait_kind"));
        return floating.get(0);
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> list(Object value) {
        return (List<Map<String, Object>>) value;
    }

    private static Map<String, Object> onlyKey(Map<String, Object> object) {
        List<?> keys = (List<?>) object.get("source_lines");
        assertEquals(1, keys.size(), object.toString());
        return RecordLines.asMap(keys.get(0));
    }
}
