package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static com.pricepredictor.connector.effects.TestCollectors.caps;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What the playability channel records of a decision or a declaration: the
 * sample stream a fork's candidate must not shift, and the deciding player
 * each record names.
 *
 * <p>Which game a candidate or a declaration is admitted from is in
 * {@link PatchedCollectorTest}, beside the other hooks' identity checks.
 */
@ExtendWith(ForgeExtension.class)
class PlayabilityChannelTest {

    @TempDir
    Path tempDir;

    private static final Method ON_CANDIDATE =
            HookShapes.method(HookShapes.Playability.class, "onCandidate");
    private static final Method ON_ATTACKERS_DECLARABLE =
            HookShapes.method(HookShapes.CombatLegality.class, "onAttackersDeclarable");
    private static final Method ON_BLOCKERS_DECLARABLE =
            HookShapes.method(HookShapes.CombatLegality.class, "onBlockersDeclarable");

    /**
     * A fork candidate rejected on identity must not have drawn from the
     * shared {@code sampler} on its way to being rejected -- checking identity
     * before the {@code caps.playabilityRate()} draw is what this pins.
     *
     * <p>Both collectors share one seed (1) and one {@code playabilityRate}
     * (0.5). The sampler is <em>not</em> {@code new Random(1L)}: it is
     * {@code new Random(scramble(seed))}, and {@code scramble} is SplitMix64's
     * finalizer, so the draws are those of {@code Random(scramble(1L))} --
     * {@code 0.9245, 0.1142}, not {@code Random(1L)}'s {@code 0.7309, 0.4101};
     * the conclusion holds either way, so do not reproduce the latter against
     * the real {@code sampler}.
     *
     * <p>A solo live candidate draws only the first (0.9245 &gt; 0.5, sampled
     * out). If the identity check ran <em>after</em> the draw, the fork
     * candidate ahead of it would consume that first draw before being dropped
     * on identity, and the live candidate behind it would draw the second
     * (0.1142 &le; 0.5, sampled in) -- a live record appearing only because a fork candidate
     * happened to precede it. Checked first, the fork candidate never reaches
     * the sampler at all, and the live candidate's own draw -- and therefore
     * its own outcome -- is identical whether or not the fork one ran ahead
     * of it.
     */
    @Test
    void aRejectedForkCandidateDoesNotShiftTheLiveSampleStream() throws Throwable {
        Card liveCard = TestCards.build("Grizzly Bears");
        SpellAbility liveCandidate = AbilityFactory.getAbility(
                "AB$ Pump | Cost$ 1 | NumAtt$ +1 | NumDef$ +1", liveCard);
        CollectionCaps halfRate = new CollectionCaps(
                CollectionCaps.defaults().manaCap(), 0.5,
                CollectionCaps.defaults().interventionsPerGame(), CollectionCaps.defaults().probesPerGame(),
                CollectionCaps.defaults().probeKeywords(), 0.0, 0.0);

        Path solo = ShardRun.shardOf(tempDir.resolve("solo"), TestCards.game(), halfRate,
                (bracket, collectors) -> collectors.playability().decisionHandler().invoke(null, ON_CANDIDATE,
                        new Object[]{liveCandidate}));
        boolean soloAdmitted = admitsAPlayabilityRecord(solo);

        Card forkCard = TestBoard.cardInAnotherGame("Grizzly Bears");
        SpellAbility forkCandidate = AbilityFactory.getAbility(
                "AB$ Pump | Cost$ 1 | NumAtt$ +1 | NumDef$ +1", forkCard);
        Path paired = ShardRun.shardOf(tempDir.resolve("paired"), TestCards.game(), halfRate,
                (bracket, collectors) -> {
                    collectors.playability().decisionHandler().invoke(null, ON_CANDIDATE,
                            new Object[]{forkCandidate});
                    collectors.playability().decisionHandler().invoke(null, ON_CANDIDATE,
                            new Object[]{liveCandidate});
                });
        boolean pairedAdmitted = admitsAPlayabilityRecord(paired);

        assertEquals(soloAdmitted, pairedAdmitted,
                "a rejected fork candidate ahead of it must not change whether the "
                        + "live candidate behind it is sampled in: solo=" + soloAdmitted
                        + " paired=" + pairedAdmitted);
    }

    /** Whether a run sampled a playability record in; one that sampled none wrote no shard. */
    private static boolean admitsAPlayabilityRecord(Path shard) throws IOException {
        return Files.exists(shard) && ShardRun.readShard(shard).stream()
                .anyMatch(l -> l.contains("\"kind\":\"playability\""));
    }

    // ── gen-2: declarations and the deciding player ─────────────────────
    //
    // The engine reports legality from the turn-based declaration itself and
    // names the deciding player, so every record is a real declaration
    // (what_if=false) whatever the step, and FR-030a's actor is the player
    // the engine names; FR-030 stamps random_seat off that actor. These
    // build a two-player game so the deciding player is a real seat.

    private static List<String> recordsOf(List<String> lines) {
        return lines.stream().filter(l -> l.contains("\"kind\":\"playability\"")).toList();
    }

    /**
     * An attackers record is a real declaration made by the player the
     * engine names, and a candidate the rules keep out is listed as
     * forbidden rather than left out.
     */
    @Test
    void anAttackDeclarationIsRealAndNamesTheDeciderTheEngineReports() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_ATTACKERS);
        Player active = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        Card ready = TestBoard.inPlay(game, active, "Grizzly Bears");
        Card tapped = TestBoard.inPlay(game, active, "Runeclaw Bear");
        tapped.setTapped(true);

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            collectors.playability().legalityHandler().invoke(null, ON_ATTACKERS_DECLARABLE,
                    new Object[]{active, other, List.of(ready, tapped), List.of(ready)});
        });

        List<String> records = recordsOf(lines);
        assertEquals(1, records.size(), records.toString());
        String real = records.get(0);
        assertTrue(real.contains("\"subkind\":\"attackers\""), real);
        assertTrue(real.contains("\"what_if\":false"), real);
        assertTrue(real.contains("\"actor_player\":\"" + Refs.of(active) + "\""),
                "the deciding player is the one the engine names: " + real);
        assertTrue(real.contains("\"random_seat\":false"), real);
        assertTrue(real.contains("\"legal_attackers\":[\"" + Refs.of(ready) + "\"]"),
                real);
        assertTrue(real.contains("\"forbidden\":[{\"entity\":\""
                + Refs.of(tapped) + "\""), real);
    }

    /**
     * Nothing about the step or whose turn it is reclassifies a declaration:
     * the engine reports only real ones, so a report outside the
     * declare-attackers step is still written as real.
     */
    @Test
    void aDeclarationIsRealWhateverTheStep() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        Card theirs = TestBoard.inPlay(game, other, "Grizzly Bears");

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            collectors.playability().legalityHandler().invoke(null, ON_ATTACKERS_DECLARABLE,
                    new Object[]{other, active, List.of(theirs), List.of(theirs)});
        });

        List<String> records = recordsOf(lines);
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"what_if\":false"), records.get(0));
        assertTrue(records.get(0).contains(
                "\"actor_player\":\"" + Refs.of(other) + "\""), records.get(0));
    }

    /**
     * FR-030a's blockers row: a blockers record names the blocking player
     * the engine reports, with the attacker it anchors on and how many
     * blockers that attacker takes.
     */
    @Test
    void aBlockDeclarationIsRealAndNamesTheBlockingPlayer() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player attacking = game.getPlayers().get(0);
        Player blocking = game.getPlayers().get(1);
        Card attacker = TestBoard.inPlay(game, attacking, "Grizzly Bears");
        Card blocker = TestBoard.inPlay(game, blocking, "Hill Giant");
        Combat combat = new Combat(attacking);
        combat.addAttacker(attacker, blocking);
        game.getPhaseHandler().setCombat(combat);

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            collectors.playability().legalityHandler().invoke(null, ON_BLOCKERS_DECLARABLE,
                    new Object[]{blocking, attacker, List.of(blocker), List.of(blocker), 1});
        });

        List<String> records = recordsOf(lines);
        assertEquals(1, records.size(), records.toString());
        String real = records.get(0);
        assertTrue(real.contains("\"subkind\":\"blockers\""), real);
        assertTrue(real.contains("\"what_if\":false"), real);
        assertTrue(real.contains("\"actor_player\":\"" + Refs.of(blocking) + "\""),
                "the deciding player is the blocking player: " + real);
        assertTrue(real.contains("\"anchor_attacker\":\"" + Refs.of(attacker) + "\""),
                real);
        assertTrue(real.contains("\"min_blockers\":1"), real);
    }

    /**
     * Every declaration on a board that moved is written, and the same
     * declaration on an unchanged board is written once (FR-028).
     */
    @Test
    void everyDeclarationOnAMovedBoardIsWrittenAndARepeatIsNot() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_ATTACKERS);
        Player active = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        Card mine = TestBoard.inPlay(game, active, "Grizzly Bears");

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            for (int turn = 0; turn < 20; turn++) {
                // A life total that moves is enough to make each snapshot new.
                active.setLife(20 - turn, null);
                collectors.playability().legalityHandler().invoke(null, ON_ATTACKERS_DECLARABLE,
                        new Object[]{active, other, List.of(mine), List.of(mine)});
            }
            collectors.playability().legalityHandler().invoke(null, ON_ATTACKERS_DECLARABLE,
                    new Object[]{active, other, List.of(mine), List.of(mine)});
        });

        assertEquals(20, recordsOf(lines).size());
    }

    /**
     * Spec Story 3 scenario 6 on the hook path: a record whose actor is the
     * random seat says so.
     */
    @Test
    void randomSeatIsStampedOffTheActor() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_ATTACKERS);
        Player active = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        Card mine = TestBoard.inPlay(game, active, "Grizzly Bears");

        List<String> lines = ShardRun.recordsWithRandomSeat(tempDir, game,
                Refs.of(active), (bracket, collectors) -> {
            collectors.playability().legalityHandler().invoke(null, ON_ATTACKERS_DECLARABLE,
                    new Object[]{active, other, List.of(mine), List.of(mine)});
        });

        List<String> records = recordsOf(lines);
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"random_seat\":true"), records.get(0));
    }

    /**
     * FR-030a's decision row: a candidate evaluated on the opponent's turn
     * names the candidate's controller, not the active player.
     */
    @Test
    void aDecisionRecordNamesTheCandidatesController() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player other = game.getPlayers().get(1);
        Card theirs = TestBoard.inPlay(game, other, "Grizzly Bears");
        SpellAbility candidate = AbilityFactory.getAbility(
                "AB$ Pump | Cost$ 1 | NumAtt$ +1 | NumDef$ +1", theirs);
        candidate.setActivatingPlayer(other);

        List<String> lines = ShardRun.records(tempDir, game, caps(0, 1.0), (bracket, collectors) -> {
            collectors.playability().decisionHandler().invoke(null, ON_CANDIDATE,
                    new Object[]{candidate});
        });

        List<String> records = recordsOf(lines);
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains(
                "\"actor_player\":\"" + Refs.of(other) + "\""), records.get(0));
        assertFalse(records.get(0).contains("what_if"),
                "a decision record carries no what_if: " + records.get(0));
    }
}
