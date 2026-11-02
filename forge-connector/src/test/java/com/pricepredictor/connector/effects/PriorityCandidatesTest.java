package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.TestCaps.caps;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Both players' candidates at a priority stop and the decision records
 * written of them (FR-030, FR-033), driven through the priority-settled
 * signal the engine sends before the priority holder chooses.
 */
@ExtendWith(ForgeExtension.class)
class PriorityCandidatesTest {

    @TempDir
    Path tempDir;

    private static final Method ON_PRIORITY_SETTLED =
            HookShapes.method(HookShapes.PrioritySettled.class, "onPrioritySettled");

    /** Every candidate sampled in, so a record's absence means none was written. */
    private static final CollectionCaps EVERY_CANDIDATE =
            caps().manaCap(0).everyDecision().build();

    private static void settle(Channels collectors, Game game, Player priority) throws Throwable {
        collectors.phase().prioritySettledHandler().invoke(
                null, ON_PRIORITY_SETTLED, new Object[]{game, priority});
    }

    private static List<String> decisions(List<String> lines) {
        return lines.stream().filter(l -> l.contains("\"subkind\":\"decision\"")).toList();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> candidateOf(String record) {
        return RecordLines.asMap(((List<Object>) RecordLines.payload(record).get("candidates")).get(0));
    }

    /** Whether the record's candidate is a line of {@code scriptFile}. */
    @SuppressWarnings("unchecked")
    private static boolean names(String record, String scriptFile) {
        return ((List<Object>) candidateOf(record).get("ability")).stream()
                .anyMatch(key -> scriptFile.equals(RecordLines.asMap(key).get("script_file")));
    }

    /**
     * A's main phase, A holding priority, B holding a sorcery and the lands
     * to pay for it.
     */
    private static Game aSorceryInTheHandOfThePlayerWithoutPriority() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        TestBoard.inPlay(game, a, "Forest");
        TestBoard.inPlay(game, a, "Forest");
        TestBoard.inHand(game, a, "Grizzly Bears");
        TestBoard.inPlay(game, b, "Island");
        TestBoard.inPlay(game, b, "Island");
        TestBoard.inPlay(game, b, "Island");
        TestBoard.inHand(game, b, "Divination");
        return game;
    }

    /**
     * Acceptance 8: the sorcery of the player without priority is evaluated,
     * and carries {@code timing = false}, {@code can_play = false} and a
     * {@code casting_speed} entry; the priority holder's candidates are
     * evaluated beside it.
     */
    @Test
    void bothPlayersAreEvaluatedAndASorceryOffTurnFailsCastingSpeed() throws Throwable {
        Game game = aSorceryInTheHandOfThePlayerWithoutPriority();
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);

        List<String> records = decisions(ShardRun.records(tempDir, game, EVERY_CANDIDATE,
                (bracket, collectors) -> settle(collectors, game, a)));

        String divination = records.stream()
                .filter(r -> names(r, "cardsfolder/d/divination.txt")).findFirst().orElseThrow(
                        () -> new AssertionError("no record of B's sorcery: " + records));
        assertTrue(divination.contains("\"actor_player\":\"" + Refs.of(b) + "\""), divination);
        Map<String, Object> candidate = candidateOf(divination);
        Map<String, Object> verdict = RecordLines.asMap(candidate.get("verdict"));
        assertEquals(false, verdict.get("timing"), divination);
        assertEquals(false, verdict.get("can_play"), divination);
        assertEquals(List.of(Map.of("check", "casting_speed", "responsible_static", List.of())),
                candidate.get("failed_checks"), divination);

        String bears = records.stream()
                .filter(r -> names(r, "cardsfolder/g/grizzly_bears.txt")).findFirst().orElseThrow(
                        () -> new AssertionError("no record of A's creature spell: " + records));
        assertTrue(bears.contains("\"actor_player\":\"" + Refs.of(a) + "\""), bears);
        assertEquals(true, RecordLines.asMap(candidateOf(bears).get("verdict")).get("can_play"));
    }

    /** A fork's phase handler fires the same JVM-static signal: it enumerates nothing here. */
    @Test
    void aPriorityStopInAnotherGameWritesNothing() throws Throwable {
        Game game = aSorceryInTheHandOfThePlayerWithoutPriority();
        Game fork = TestBoard.anotherGame();

        Path shard = ShardRun.shardOf(tempDir, game, EVERY_CANDIDATE, (bracket, collectors) ->
                settle(collectors, fork, new Player("fork-seat", fork, 92304)));

        assertFalse(Files.exists(shard));
    }

    /**
     * The priority holder draws from the stop its records were written of:
     * the very candidates, read once, so its plays and its records cannot
     * disagree; asked again, it gets the board evaluated afresh.
     */
    @Test
    void thePriorityHolderTakesTheCandidatesItsRecordsWereWrittenOf() throws Throwable {
        Game game = aSorceryInTheHandOfThePlayerWithoutPriority();
        Player a = game.getPlayers().get(0);
        List<String> payloads = new ArrayList<>();
        List<PriorityCandidates.Candidate> taken = new ArrayList<>();
        List<PriorityCandidates.Candidate> again = new ArrayList<>();

        List<String> records = decisions(ShardRun.records(tempDir, game, EVERY_CANDIDATE,
                (bracket, collectors) -> {
                    settle(collectors, game, a);
                    taken.addAll(collectors.candidates().takeFor(a));
                    again.addAll(collectors.candidates().takeFor(a));
                }));
        for (PriorityCandidates.Candidate candidate : taken) {
            candidate.read().ifPresent(read -> payloads.add(read.payloadJson()));
        }

        assertFalse(taken.isEmpty());
        assertTrue(taken.stream().allMatch(c -> c.player() == a));
        for (String payload : payloads) {
            assertTrue(records.stream().anyMatch(r -> r.contains("\"payload\":" + payload)),
                    "a taken candidate is one a record was written of: " + payload);
        }
        assertEquals(taken.size(), again.size());
        for (int i = 0; i < taken.size(); i++) {
            assertNotSame(taken.get(i), again.get(i), "the second ask is a fresh enumeration");
        }
        PriorityCandidates.Candidate bears = taken.stream()
                .filter(c -> c.ability().getHostCard().getName().equals("Grizzly Bears"))
                .findFirst().orElseThrow();
        assertSame(bears.read().orElseThrow(), bears.read().orElseThrow(), "read once");
        assertTrue(bears.playable());
    }

    /** The stop of another player's priority is not the seat's to take. */
    @Test
    void anotherPlayersStopIsNotTaken() throws Throwable {
        Game game = aSorceryInTheHandOfThePlayerWithoutPriority();
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        List<PriorityCandidates.Candidate> forB = new ArrayList<>();
        List<PriorityCandidates.Candidate> forA = new ArrayList<>();

        ShardRun.records(tempDir, game, EVERY_CANDIDATE, (bracket, collectors) -> {
            settle(collectors, game, a);
            forB.addAll(collectors.candidates().takeFor(b));
            forA.addAll(collectors.candidates().takeFor(a));
        });

        assertTrue(forB.stream().allMatch(c -> c.player() == b));
        assertTrue(forB.stream().noneMatch(PriorityCandidates.Candidate::playable),
                "a sorcery on the other player's turn");
        assertTrue(forA.stream().anyMatch(PriorityCandidates.Candidate::playable),
                "A's own stop was left for A");
    }

    // ── land plays (FR-033a) ────────────────────────────────────────────

    /** The decision record whose snapshot's source is {@code card}. */
    private static String recordOf(List<String> records, Card card) {
        return records.stream()
                .filter(r -> Refs.of(card).equals(RecordLines.asMap(RecordLines.asMap(
                        RecordLines.parse(r).get("state")).get("refs")).get("source")))
                .findFirst().orElseThrow(
                        () -> new AssertionError("no record of " + card + ": " + records));
    }

    @SuppressWarnings("unchecked")
    private static List<String> failedChecks(String record) {
        return ((List<Object>) candidateOf(record).get("failed_checks")).stream()
                .map(c -> (String) RecordLines.asMap(c).get("check")).toList();
    }

    private static Map<String, Object> verdictOf(String record) {
        return RecordLines.asMap(candidateOf(record).get("verdict"));
    }

    /**
     * Story 2 acceptance 13: a second land after one played is a land-play
     * candidate past its limit, and no mana ability is a candidate.
     */
    @Test
    void aSecondLandIsALandPlayPastTheLimit() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        a.playLand(TestBoard.inHand(game, a, "Forest"), null);
        Card mountain = TestBoard.inHand(game, a, "Mountain");

        List<String> records = decisions(ShardRun.records(tempDir, game, EVERY_CANDIDATE,
                (bracket, collectors) -> settle(collectors, game, a)));

        String second = recordOf(records, mountain);
        assertEquals(true, candidateOf(second).get("land_play"), second);
        assertEquals(false, verdictOf(second).get("can_play"), second);
        assertEquals(true, verdictOf(second).get("affordable"), second);
        assertEquals(true, verdictOf(second).get("has_legal_target"), second);
        assertEquals(List.of("land_play_limit"), failedChecks(second));
        assertEquals(List.of(), candidateOf(second).get("legal_targets"));
        assertEquals(Map.of(), candidateOf(second).get("cost_after_adjustment"));
        for (Player player : game.getPlayers()) {
            assertTrue(PriorityCandidates.candidatesOf(game, player).stream()
                    .noneMatch(c -> c.ability().isManaAbility()), "a mana ability is no candidate");
        }
    }

    /** Exploration raises the limit: the same second land is playable. */
    @Test
    void explorationMakesTheSecondLandPlayable() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        TestBoard.inPlay(game, a, "Exploration");
        game.getAction().checkStaticAbilities();
        a.playLand(TestBoard.inHand(game, a, "Forest"), null);
        Card mountain = TestBoard.inHand(game, a, "Mountain");

        PriorityCandidates.Candidate second = PriorityCandidates.candidatesOf(game, a).stream()
                .filter(c -> c.ability().getHostCard() == mountain).findFirst().orElseThrow();

        assertTrue(second.read().orElseThrow().landPlay());
        assertTrue(second.playable(), second.read().orElseThrow().verdict().toString());
    }

    /**
     * Story 2 acceptance 14: the land of the player without priority fails
     * its timing, whose turn and casting speed both.
     */
    @Test
    void aLandOfThePlayerWithoutPriorityFailsWhoseTurnAndCastingSpeed() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        Card island = TestBoard.inHand(game, b, "Island");

        List<String> records = decisions(ShardRun.records(tempDir, game, EVERY_CANDIDATE,
                (bracket, collectors) -> settle(collectors, game, a)));

        String land = recordOf(records, island);
        assertTrue(land.contains("\"actor_player\":\"" + Refs.of(b) + "\""), land);
        assertEquals(true, candidateOf(land).get("land_play"), land);
        assertEquals(false, verdictOf(land).get("timing"), land);
        assertEquals(false, verdictOf(land).get("can_play"), land);
        assertEquals(List.of("casting_speed", "whose_turn"), failedChecks(land));
    }

    /** A modal double-faced card's land face is a land-play candidate; its spell face is not. */
    @Test
    void aModalDoubleFacedCardsLandFaceIsALandPlay() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card awakening = TestBoard.inHand(game, a, "Agadeem's Awakening");

        List<PriorityCandidates.Candidate> on = PriorityCandidates.candidatesOf(game, a).stream()
                .filter(c -> c.ability().getHostCard() == awakening).toList();

        PriorityCandidates.Candidate land = on.stream()
                .filter(c -> c.ability().isLandAbility()).findFirst().orElseThrow(
                        () -> new AssertionError("no land face: " + on));
        assertTrue(land.read().orElseThrow().landPlay());
        assertTrue(land.playable(), land.read().orElseThrow().verdict().toString());
        assertTrue(on.stream().anyMatch(c -> c.ability().isSpell()
                && !c.read().orElseThrow().landPlay()), "the spell face: " + on);
    }

    /** Every spell candidate carries {@code land_play = false}. */
    @Test
    void aSpellCandidateIsNoLandPlay() throws Throwable {
        Game game = aSorceryInTheHandOfThePlayerWithoutPriority();
        Player a = game.getPlayers().get(0);

        List<String> records = decisions(ShardRun.records(tempDir, game, EVERY_CANDIDATE,
                (bracket, collectors) -> settle(collectors, game, a)));

        String bears = records.stream()
                .filter(r -> names(r, "cardsfolder/g/grizzly_bears.txt")).findFirst().orElseThrow();
        assertEquals(false, candidateOf(bears).get("land_play"), bears);
    }

    // ── read-only ───────────────────────────────────────────────────────

    /**
     * What a verdict could write: the board a snapshot renders, every card's
     * cast ability and every spell ability's activating player.
     */
    private static String digest(Game game) {
        StringBuilder digest = new StringBuilder(new SnapshotBuilder(game).ofBoard().render());
        for (Card card : game.getCardsInGame()) {
            digest.append('\n').append(card.getId())
                    .append(" cast=").append(System.identityHashCode(card.getCastSA()));
            for (SpellAbility ability : card.getSpellAbilities()) {
                digest.append(" | ").append(ability.getId())
                        .append(" by ").append(ability.getActivatingPlayer());
            }
        }
        return digest.toString();
    }

    /**
     * Enumerating and evaluating both players' candidates leaves the game as
     * it was: spells and abilities, a charm, a static prohibition, a
     * split-second spell on the stack, in both seats' hands and on both
     * sides of the battlefield.
     */
    @Test
    void theGameIsUnchangedByEnumeration() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        for (String land : List.of("Mountain", "Mountain", "Island", "Forest")) {
            TestBoard.inPlay(game, a, land);
            TestBoard.inPlay(game, b, land);
        }
        TestBoard.inPlay(game, a, "Grand Abolisher");
        TestBoard.inPlay(game, a, "Prodigal Sorcerer");
        TestBoard.inPlay(game, b, "Llanowar Elves");
        TestBoard.inPlay(game, b, "Jace Beleren");
        TestBoard.inHand(game, a, "Fiery Intervention");
        TestBoard.inHand(game, a, "Prey Upon");
        TestBoard.inHand(game, b, "Lightning Bolt");
        TestBoard.inHand(game, b, "Divination");
        TestBoard.inGraveyard(game, b, "Grizzly Bears");
        // Land plays: in hand on both sides, past the limit, a modal
        // double-faced land face, one a static forbids, one a static permits.
        TestBoard.inPlay(game, a, "Exploration");
        TestBoard.inPlay(game, b, "Aggressive Mining");
        TestBoard.inPlay(game, a, "Crucible of Worlds");
        TestBoard.inGraveyard(game, a, "Swamp");
        game.getAction().checkStaticAbilities();
        a.playLand(TestBoard.inHand(game, a, "Plains"), null);
        a.playLand(TestBoard.inHand(game, a, "Plains"), null);
        TestBoard.inHand(game, a, "Agadeem's Awakening");
        TestBoard.inHand(game, a, "Forest");
        TestBoard.inHand(game, b, "Island");
        TestBoard.cast(game, TestBoard.inHand(game, a, "Sudden Shock"), a, b);
        String before = digest(game);

        int read = 0;
        for (Player player : game.getPlayers()) {
            for (PriorityCandidates.Candidate candidate : PriorityCandidates.candidatesOf(game, player)) {
                if (candidate.read().isPresent()) {
                    read++;
                }
            }
        }

        assertTrue(read > 5, "the board offers candidates to evaluate: " + read);
        assertEquals(before, digest(game));
        assertEquals(1, game.getStack().size());
        assertEquals(4, a.getCardsIn(ZoneType.Hand).size());
        assertEquals(3, b.getCardsIn(ZoneType.Hand).size());
        assertEquals(2, a.getLandsPlayedThisTurn());
        assertEquals(1, a.getCardsIn(ZoneType.Graveyard).size());
    }
}
