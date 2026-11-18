package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.card.CardStateName;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

/**
 * A Class level line and a Room door line always say whether they are in
 * force (finding 9.12, root spec § 9.2): every {@code Class:N} keyword line
 * carries {@code active}, true once the Class has reached level N; a Room
 * lists both doors' lines, each {@code active} while its door is unlocked
 * and {@code false} while it is locked.
 */
@ExtendWith(ForgeExtension.class)
class ClassAndRoomLinesTest {

    private static final String MONK_CLASS = "Monk Class";
    private static final String POOL_ROOM = "Bottomless Pool // Locker Room";

    // ── Class levels ────────────────────────────────────────────────────

    @Test
    void atLevelOneNeitherLevelUpLineIsInForce() {
        List<Map<String, Object>> lines = monkClassAt(1);

        assertEquals(false, classLine(lines, 2).get("active"), lines.toString());
        assertEquals(false, classLine(lines, 3).get("active"), lines.toString());
        assertInstanceOf(Boolean.class, levelOneLine(lines).get("active"), lines.toString());
    }

    @Test
    void atLevelTwoOnlyLevelThreeIsNotInForce() {
        List<Map<String, Object>> lines = monkClassAt(2);

        assertEquals(true, classLine(lines, 2).get("active"), lines.toString());
        assertEquals(false, classLine(lines, 3).get("active"), lines.toString());
    }

    @Test
    void atLevelThreeBothLevelUpLinesAreInForce() {
        List<Map<String, Object>> lines = monkClassAt(3);

        assertEquals(true, classLine(lines, 2).get("active"), lines.toString());
        assertEquals(true, classLine(lines, 3).get("active"), lines.toString());
    }

    /**
     * The trigger level 2 adds is printed on the Class, beside its
     * {@code Class:2} line under the same key, not a granted line.
     */
    @Test
    void aReachedLevelsTriggerIsPrintedBesideItsLevelLine() {
        List<Map<String, Object>> lines = monkClassAt(2);

        Map<String, Object> trigger = only(lines, line -> "trigger".equals(line.get("kind")));

        assertEquals(classLine(lines, 2).get("keys"), trigger.get("keys"), lines.toString());
        assertEquals(false, trigger.get("granted"), lines.toString());
    }

    // ── Room doors ──────────────────────────────────────────────────────

    @Test
    void aRoomWithOneDoorUnlockedListsBothDoors() {
        List<Map<String, Object>> lines = roomWithUnlocked(List.of(CardStateName.LeftSplit));

        assertEquals(true, doorLine(lines, 0).get("active"), lines.toString());
        assertEquals(false, doorLine(lines, 1).get("active"), lines.toString());
    }

    @Test
    void aRoomWithTheOtherDoorUnlockedListsBothDoors() {
        List<Map<String, Object>> lines = roomWithUnlocked(List.of(CardStateName.RightSplit));

        assertEquals(false, doorLine(lines, 0).get("active"), lines.toString());
        assertEquals(true, doorLine(lines, 1).get("active"), lines.toString());
    }

    @Test
    void aRoomWithBothDoorsUnlockedHasBothInForce() {
        List<Map<String, Object>> lines = roomWithUnlocked(
                List.of(CardStateName.LeftSplit, CardStateName.RightSplit));

        assertEquals(true, doorLine(lines, 0).get("active"), lines.toString());
        assertEquals(true, doorLine(lines, 1).get("active"), lines.toString());
    }

    @Test
    void aRoomWithBothDoorsLockedListsBothDoorsNotInForce() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card room = roomOnBattlefield(board);

        List<Map<String, Object>> lines = linesOf(board, room);

        assertEquals(false, doorLine(lines, 0).get("active"), lines.toString());
        assertEquals(false, doorLine(lines, 1).get("active"), lines.toString());
    }

    /** A face-down Room carries its real doors, neither in force. */
    @Test
    void aFaceDownRoomListsBothDoorsNotInForce() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card room = roomOnBattlefield(board);
        room.unlockRoom(board.you, CardStateName.LeftSplit);
        room.turnFaceDown();

        List<Map<String, Object>> lines = linesOf(board, room);

        assertEquals(false, doorLine(lines, 0).get("active"), lines.toString());
        assertEquals(false, doorLine(lines, 1).get("active"), lines.toString());
    }

    /** Off the battlefield a Room has both halves (709.4): both doors are in force. */
    @Test
    void aRoomInHandHasBothDoorsInForce() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card room = board.hand(POOL_ROOM, board.you);

        List<Map<String, Object>> lines = linesOf(board, room);

        assertEquals(true, doorLine(lines, 0).get("active"), lines.toString());
        assertEquals(true, doorLine(lines, 1).get("active"), lines.toString());
    }

    /** A Room spell has only the half cast (709.3): the other door is not in force. */
    @Test
    void aRoomCastAsOneDoorHasOnlyThatDoorInForce() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card room = board.hand(POOL_ROOM, board.you);
        SpellAbility lockerRoom = room.getAllSpellAbilities().stream()
                .filter(sa -> sa.getCardState().getStateName() == CardStateName.RightSplit)
                .findFirst().orElseThrow();
        lockerRoom.setActivatingPlayer(board.you);
        Card spell = board.game.getAction().moveToStack(room, lockerRoom);

        List<Map<String, Object>> lines = linesOf(board, spell);

        assertEquals(false, doorLine(lines, 0).get("active"), lines.toString());
        assertEquals(true, doorLine(lines, 1).get("active"), lines.toString());
    }

    // ── helpers ─────────────────────────────────────────────────────────

    private static List<Map<String, Object>> monkClassAt(int level) {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card monk = board.battlefield(MONK_CLASS, board.you);
        monk.setClassLevel(level);
        board.game.getAction().checkStaticAbilities();
        return linesOf(board, monk);
    }

    private static List<Map<String, Object>> roomWithUnlocked(List<CardStateName> doors) {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card room = roomOnBattlefield(board);
        for (CardStateName door : doors) {
            room.unlockRoom(board.you, door);
        }
        board.game.getAction().checkStaticAbilities();
        return linesOf(board, room);
    }

    /**
     * The Room put onto the battlefield from its owner's hand, not cast: both
     * doors locked. Forge locks them as a card enters from a zone, which a
     * card entering from none, as {@link CharacterizationBoard#battlefield}'s
     * does, skips.
     */
    private static Card roomOnBattlefield(CharacterizationBoard board) {
        Card room = board.hand(POOL_ROOM, board.you);
        return board.game.getAction().moveToPlay(room, board.you, null, null);
    }

    /** The {@code Class:N} keyword line of level {@code level}. */
    private static Map<String, Object> classLine(List<Map<String, Object>> lines, int level) {
        return only(lines, line -> "keyword".equals(line.get("kind"))
                && String.valueOf(line.get("fragment")).startsWith("Class:" + level + ":"));
    }

    /** Monk Class's level-1 line: its cost-reduction static. */
    private static Map<String, Object> levelOneLine(List<Map<String, Object>> lines) {
        return only(lines, line -> "static".equals(line.get("kind")));
    }

    /** The one line of the door printed as face {@code face}: each door has one trigger. */
    private static Map<String, Object> doorLine(List<Map<String, Object>> lines, int face) {
        return only(lines, line -> faceOf(line) == face);
    }

    @SuppressWarnings("unchecked")
    private static long faceOf(Map<String, Object> line) {
        List<Map<String, Object>> keys = (List<Map<String, Object>>) line.get("keys");
        assertEquals(1, keys.size(), line.toString());
        return (Long) keys.get(0).get("face");
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> linesOf(CharacterizationBoard board, Card card) {
        return (List<Map<String, Object>>) RecordLines.parse(
                EntityJson.render(board.game, card, StaticSuppression.NONE)).get("lines");
    }

    private static Map<String, Object> only(List<Map<String, Object>> lines,
                                            Predicate<Map<String, Object>> matches) {
        List<Map<String, Object>> found = lines.stream().filter(matches).toList();
        assertEquals(1, found.size(), lines.toString());
        return found.get(0);
    }
}
