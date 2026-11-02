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
            caps().manaCap(0).playabilityRate(1.0).build();

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
        assertEquals(2, a.getCardsIn(ZoneType.Hand).size());
        assertEquals(2, b.getCardsIn(ZoneType.Hand).size());
    }
}
