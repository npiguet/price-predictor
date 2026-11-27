package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.spellability.OptionalCost;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.SnapshotReading.entityIn;
import static com.pricepredictor.connector.effects.SnapshotReading.keywordLine;
import static com.pricepredictor.connector.effects.SnapshotReading.kinds;
import static com.pricepredictor.connector.effects.SnapshotReading.line;
import static com.pricepredictor.connector.effects.SnapshotReading.lines;
import static com.pricepredictor.connector.effects.SnapshotReading.linesOf;
import static com.pricepredictor.connector.effects.SnapshotReading.onlyKey;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Feature 025 story 2's acceptance 1 to 5 and the face-down case: what a card slot's
 * {@code lines} carries, read off the rendered entity; and a continuous
 * record's board, where the removed static leaves no trace.
 */
@ExtendWith(ForgeExtension.class)
class SnapshotLinesTest {

    /**
     * Acceptance 1: Guile's static, trigger and replacement are each a
     * line keyed into its own script, and the static carries
     * {@code active}, which holds on the battlefield.
     */
    @Test
    void aCreaturesStaticTriggerAndReplacementAreLinesWithTheirKeys() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card guile = board.battlefield("Guile", board.you);

        List<Map<String, Object>> lines = linesOf(board, guile);

        assertEquals(List.of("trigger", "static", "replacement"), kinds(lines));
        for (Map<String, Object> line : lines) {
            Map<String, Object> key = onlyKey(line);
            assertEquals("cardsfolder/g/guile.txt", key.get("script_file"), line.toString());
            assertEquals(line.get("kind"), key.get("trait_kind"), line.toString());
            assertEquals(0L, key.get("index_within_kind"), line.toString());
            assertEquals(false, line.get("granted"), line.toString());
        }
        assertEquals(true, line(lines, "static").get("active"));
        assertFalse(line(lines, "trigger").containsKey("active"),
                "active is a static's: " + lines);
    }

    /** A static whose condition does not hold says so: no threshold, no flying. */
    @Test
    void aStaticWhoseConditionFailsIsInactive() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card enforcer = board.battlefield("Mystic Enforcer", board.you);

        assertEquals(false, line(linesOf(board, enforcer), "static").get("active"));
    }

    /**
     * Acceptance 2: under Humility a creature has lost all its abilities,
     * and the live lists Forge plays from carry none; neither does its
     * plain cast, which is no line of its own.
     */
    @Test
    void aCreatureThatLostAllAbilitiesListsNone() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card guile = board.battlefield("Guile", board.you);
        board.battlefield("Humility", board.opponent);
        board.game.getAction().checkStaticAbilities();

        assertEquals(List.of(), linesOf(board, guile));
    }

    /**
     * Acceptance 3: flying from Levitation is a keyword line on the
     * creature, granted, keyed to the static that grants it.
     */
    @Test
    void aKeywordAnotherPermanentsStaticGrantsIsListedGranted() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        board.battlefield("Levitation", board.you);
        board.game.getAction().checkStaticAbilities();

        Map<String, Object> flying = line(linesOf(board, bears), "keyword");
        assertEquals("Flying", flying.get("fragment"));
        assertEquals(true, flying.get("granted"));
        assertEquals("static", flying.get("grant"));
        Map<String, Object> key = onlyKey(flying);
        assertEquals("cardsfolder/l/levitation.txt", key.get("script_file"));
        assertEquals("static", key.get("trait_kind"));
    }

    /**
     * Acceptance 4: ward {2} and annihilator 2 are keyword lines carrying
     * their instance's fragment, value included, and keyed to the keyword
     * the card prints. Annihilator's trigger is that line, not one more.
     */
    @Test
    void wardAndAnnihilatorAreKeywordLinesWithTheirValues() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card spawn = board.battlefield("Aboleth Spawn", board.you);
        Card crusher = board.battlefield("Ulamog's Crusher", board.you);

        Map<String, Object> ward = keywordLine(linesOf(board, spawn), "Ward:2");
        assertEquals(false, ward.get("granted"));
        assertEquals("keyword", onlyKey(ward).get("trait_kind"));
        List<Map<String, Object>> crusherLines = linesOf(board, crusher);
        Map<String, Object> annihilator = keywordLine(crusherLines, "Annihilator:2");
        assertEquals("cardsfolder/u/ulamogs_crusher.txt",
                onlyKey(annihilator).get("script_file"));
        assertEquals(List.of("static", "keyword"), kinds(crusherLines),
                "the keyword's own trigger is its line: " + crusherLines);
    }

    /** Acceptance 5: the protection line is the instance's, naming its quality. */
    @Test
    void protectionFromGoblinsNamesGoblins() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card guide = board.battlefield("Foothill Guide", board.you);

        keywordLine(linesOf(board, guide), "Protection:Goblin");
    }

    /**
     * A face-down permanent's lines are its face-down state's, and its real
     * face rides beside them, every line flagged inactive.
     */
    @Test
    void aFaceDownPermanentCarriesItsRealFaceAsInactiveLines() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card guide = board.battlefield("Foothill Guide", board.you);
        guide.turnFaceDown(true);

        List<Map<String, Object>> lines = linesOf(board, guide);

        Map<String, Object> protection = keywordLine(lines, "Protection:Goblin");
        assertEquals(false, protection.get("active"), lines.toString());
        Map<String, Object> morph = keywordLine(lines, "Morph:W");
        assertEquals(false, morph.get("active"), lines.toString());
        assertEquals("cardsfolder/f/foothill_guide.txt", onlyKey(morph).get("script_file"));
    }

    /** A spell on the stack cast with its kicker says so on the kicker's line. */
    @Test
    void aKickedSpellOnTheStackMarksItsKickerLine() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card onStack = board.game.getAction().moveToStack(
                board.hand("Burst Lightning", board.you), null);
        SpellAbility burst = board.spell(onStack, board.you);
        burst.addOptionalCost(OptionalCost.Kicker1);
        burst.resetTargets();
        burst.getTargets().add(board.opponent);
        board.game.getStack().add(burst);

        List<Map<String, Object>> lines = linesOf(board, onStack);

        assertEquals(1L, keywordLine(lines, "Kicker:4").get("cast_with"), lines.toString());
        assertFalse(line(lines, "spell").containsKey("cast_with"), lines.toString());
    }

    /**
     * A spell cast with its gift promised says so on its Gift line: the
     * optional cost is named "Promise Gift", the keyword "Gift" (B9).
     */
    @Test
    void aSpellCastWithItsGiftPromisedMarksItsGiftLine() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.opponent);
        Card onStack = board.game.getAction().moveToStack(
                board.hand("Blooming Blast", board.you), null);
        SpellAbility blast = board.spell(onStack, board.you);
        blast.addOptionalCost(OptionalCost.PromiseGift);
        blast.resetTargets();
        blast.getTargets().add(bears);
        board.game.getStack().add(blast);

        List<Map<String, Object>> lines = linesOf(board, onStack);

        assertEquals(1L, keywordLine(lines, "Gift").get("cast_with"), lines.toString());
    }

    /**
     * Every optional cost Forge has is mapped to its keyword or known to
     * be in no keyword line, so a cost Forge adds is not silently unread.
     */
    @Test
    void everyOptionalCostIsMappedOrKnownToHaveNoLine() {
        java.util.Set<OptionalCost> noLine = java.util.EnumSet.of(
                OptionalCost.ReduceW, OptionalCost.ReduceU, OptionalCost.ReduceB,
                OptionalCost.ReduceR, OptionalCost.ReduceG, OptionalCost.AltCost,
                OptionalCost.Flash, OptionalCost.Generic);
        for (OptionalCost cost : OptionalCost.values()) {
            assertTrue(noLine.contains(cost) != CardLines.OPTIONAL_COST_KEYWORDS.containsKey(cost),
                    cost.name());
        }
    }

    /** A charm on the stack names the modes it chose on its own line. */
    @Test
    void aCharmOnTheStackNamesItsChosenModes() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card onStack = board.game.getAction().moveToStack(
                board.hand("Boros Charm", board.you), null);
        SpellAbility charm = TestBoard.charmOf(onStack, board.you);
        // Mode 1, permanents you control gain indestructible: no target to choose.
        TestBoard.chain(charm, 1);
        board.game.getStack().add(charm);

        List<Map<String, Object>> lines = linesOf(board, onStack);

        assertEquals(List.of(1L), line(lines, "spell").get("chosen"), lines.toString());
    }

    /** Face up, the same lines are in force and carry no flag. */
    @Test
    void aFaceUpPermanentsLinesAreNotFlaggedInactive() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card guide = board.battlefield("Foothill Guide", board.you);

        for (Map<String, Object> line : linesOf(board, guide)) {
            assertFalse(line.containsKey("active"), line.toString());
        }
    }

    // ── a continuous record's board (the removed static leaves no trace) ─

    /**
     * The board without Levitation lists no line of Levitation's on the
     * creature: neither the flying keyword line nor the CantBlockBy static
     * Forge builds for flying, which is the static the record is about.
     */
    @Test
    void theRemovedStaticsKeywordLeavesNoLineBehind() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        Card levitation = board.battlefield("Levitation", board.you);
        board.game.getAction().checkStaticAbilities();
        StaticAbility levitating = levitation.getStaticAbilities().get(0);

        Map<String, Object> without = entityIn(
                new SnapshotBuilder(board.game).ofBoardWithout(levitating).render(), bears);
        Map<String, Object> with = entityIn(
                new SnapshotBuilder(board.game).ofBoard().render(), bears);

        assertEquals(List.of(), without.get("lines"));
        assertEquals(List.of("keyword"), kinds(lines(with)), with.toString());
    }

    /**
     * Nor does any static say which permanents it affects: that is the
     * record's answer, and hiding only the removed static's list would
     * point at it.
     */
    @Test
    void noStaticListsAffectedPermanents() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        Card levitation = board.battlefield("Levitation", board.you);
        board.game.getAction().checkStaticAbilities();
        StaticAbility levitating = levitation.getStaticAbilities().get(0);

        Map<String, Object> without = line(lines(entityIn(
                new SnapshotBuilder(board.game).ofBoardWithout(levitating).render(),
                levitation)), "static");
        Map<String, Object> with = line(linesOf(board, levitation), "static");

        assertFalse(without.containsKey("affected"), without.toString());
        assertEquals(List.of(Refs.of(bears)), with.get("affected"));
    }
}
