package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A dungeon's rooms and an Attraction's Prize are lines of their own
 * (finding R2-3). A {@code K:Dungeon} keyword builds one {@code RoomEntered}
 * trigger per room: the dungeon prints one line per room, the room's
 * {@code TriggerDescription} ({@code Room — effect}, the "Leads to" note
 * stripped as reminder text), in room order, each declaring the
 * {@code built} key of its room's trigger, and the keyword's bare key on no
 * line, as a saga's chapters do. A {@code K:Prize} keyword builds one
 * {@code ClaimPrize} trigger and prints its description as its one line,
 * under the bare key.
 */
@ExtendWith(ForgeExtension.class)
class DungeonRoomLinesTest {

    private static final Path GOLDEN_TOKENS = Path.of("src/test/resources/golden-024/tokenscripts");
    private static final String LOST_MINE = "tokenscripts/lost_mine_of_phandelver.txt";
    private static final String UNDERCITY = "tokenscripts/undercity.txt";

    private final RulesParser parser = new RulesParser();

    private ConvertedScript token(String scriptFile) {
        String fileName = scriptFile.substring(scriptFile.indexOf('/') + 1);
        return ConvertedScript.of(
                parser.parseScript(ConvertedScript.source(GOLDEN_TOKENS, fileName), scriptFile),
                scriptFile);
    }

    private static ProvenanceKey keyword(String scriptFile, int index) {
        return new ProvenanceKey(scriptFile, 0, TraitKind.KEYWORD, index);
    }

    /** The lines from {@code first} on, each with the keys it declares. */
    private static List<String> linesWithKeys(ConvertedScript converted, int first) {
        List<String> lines = new ArrayList<>();
        for (ProvenanceSidecar.Line line : converted.sidecar().lines()) {
            if (line.lineIndex() < first) continue;
            lines.add(converted.texts().get(line.lineIndex()) + " <- "
                    + line.provenance().stream().map(ProvenanceKey::describe).toList());
        }
        return lines;
    }

    private static String roomLine(String text, ProvenanceKey dungeon, int room) {
        return "triggered: " + text + " <- "
                + List.of(dungeon.withBuilt(TraitKind.TRIGGER, room).describe());
    }

    @Test
    void lostMineOfPhandelverPrintsALinePerRoomEachDeclaringItsRoomsTrigger() {
        ConvertedScript mine = token(LOST_MINE);
        ProvenanceKey dungeon = keyword(LOST_MINE, 0);

        assertEquals(List.of(
                roomLine("cave entrance — scry 1.", dungeon, 0),
                roomLine("goblin lair — create a 1/1 red goblin creature token.", dungeon, 1),
                roomLine("mine tunnels — create a treasure token.", dungeon, 2),
                roomLine("storeroom — put a +1/+1 counter on target creature.", dungeon, 3),
                roomLine("dark pool — each opponent loses 1 life and you gain 1 life.", dungeon, 4),
                roomLine("fungi cavern — target creature gets -4/-0 until your next turn.", dungeon, 5),
                roomLine("temple of dumathoin — draw a card.", dungeon, 6)),
                linesWithKeys(mine, 2));
        assertEquals(List.of(), mine.sidecar().droppedKeys());
    }

    /** A room line's script is its own room's trigger chain, not the keyword's text. */
    @Test
    void aRoomLinesScriptIsItsRoomsTriggerChain() {
        ConvertedScript mine = token(LOST_MINE);

        ProvenanceSidecar.Line goblinLair = mine.sidecar().lines().get(1);
        String script = goblinLair.script().scriptText();

        assertTrue(script.contains("ValidRoom$ Goblin Lair"), script);
        assertTrue(script.contains("DB$ Token"), script);
        assertFalse(script.contains("Dungeon:"), script);
    }

    /**
     * Undercity's nine rooms, then its static: the rooms take Forge's room
     * names ({@code RoomName$ Archive} where the Oracle text says
     * "Archives").
     */
    @Test
    void undercityPrintsItsNineRoomsThenItsStatic() {
        ConvertedScript undercity = token(UNDERCITY);
        ProvenanceKey dungeon = keyword(UNDERCITY, 0);

        assertEquals(List.of(
                roomLine("secret entrance — search your library for a basic land card, reveal it,"
                        + " put it into your hand, then shuffle.", dungeon, 0),
                roomLine("forge — put two +1/+1 counters on target creature.", dungeon, 1),
                roomLine("lost well — scry 2.", dungeon, 2),
                roomLine("trap! — target player loses 5 life.", dungeon, 3),
                roomLine("arena — goad target creature.", dungeon, 4),
                roomLine("stash — create a treasure token.", dungeon, 5),
                roomLine("archive — draw a card.", dungeon, 6),
                roomLine("catacombs — create a 4/1 black skeleton creature token with menace.", dungeon, 7),
                roomLine("throne of the dead three — reveal the top ten cards of your library. put a"
                        + " creature card from among them onto the battlefield with three +1/+1"
                        + " counters on it. it gains hexproof until your next turn. then shuffle.",
                        dungeon, 8),
                "static: you can't enter this dungeon unless you \"venture into CARDNAME.\" <- "
                        + List.of(keyword(UNDERCITY, 1).describe())),
                linesWithKeys(undercity, 2));
    }

    /** Pick-a-Beeble's Prize line is its trigger's description, under the keyword's bare key. */
    @Test
    void aPrizeLineIsItsTriggersDescriptionUnderTheBareKey() {
        ConvertedScript beeble = ConvertedScript.fromFile(
                parser, ConvertedScript.GOLDEN_024_CARDS, "p/pick_a_beeble.txt");
        ProvenanceKey prize = keyword("cardsfolder/p/pick_a_beeble.txt", 0);

        ProvenanceSidecar.Line line = beeble.sidecar().lines().get(0);

        assertEquals("triggered: prize — create two treasure tokens, then sacrifice CARDNAME"
                + " and open an attraction.", beeble.texts().get(line.lineIndex()));
        assertEquals(List.of(prize), line.provenance());
        assertTrue(line.script().scriptText().contains("Mode$ ClaimPrize"), line.script().scriptText());
        assertTrue(line.script().scriptText().contains("DB$ OpenAttraction"), line.script().scriptText());
    }
}
