package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static com.pricepredictor.connector.effects.TestCaps.caps;
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

    private static final Method ON_PRIORITY_SETTLED =
            HookShapes.method(HookShapes.PrioritySettled.class, "onPrioritySettled");
    private static final Method ON_ATTACKERS_DECLARABLE =
            HookShapes.method(HookShapes.CombatLegality.class, "onAttackersDeclarable");
    private static final Method ON_BLOCKERS_DECLARABLE =
            HookShapes.method(HookShapes.CombatLegality.class, "onBlockersDeclarable");

    /**
     * A fork's priority stop must not draw from the shared {@code sampler}
     * on its way to being rejected -- checking the game before any candidate
     * is sampled is what this pins.
     *
     * <p>Both collectors share one seed (1) and one {@code playabilityRate}
     * (0.8). The sampler is {@code SeedStreams.random(1, SAMPLER)}, whose
     * first two draws are {@code 0.8179} and {@code 0.7792}: at rate 0.8 the
     * first is sampled out (0.8179 &ge; 0.8) and the second in (0.7792 &lt;
     * 0.8). A solo run settling the live game twice shows the second premise
     * (it admits a record).
     *
     * <p>The live board offers one candidate, which draws only the first
     * (sampled out). If a fork's stop ahead of it drew for the fork's
     * candidate before being dropped, the live candidate behind it would draw
     * the second (sampled in) -- a live record appearing only because a fork's
     * stop happened to precede it.
     */
    @Test
    void aForkPriorityStopDoesNotShiftTheLiveSampleStream() throws Throwable {
        // Stage 2 keeps whatever stage 1 does, so the stream is stage 1's alone.
        CollectionCaps rate = caps().playabilityRate(0.8).candidateRates(0.8, 0.8).build();
        Game live = oneCandidate(TestBoard.twoPlayerGame(PhaseType.MAIN1));
        Player seat = live.getPlayers().get(0);
        Game fork = oneCandidate(TestBoard.twoPlayerGame(PhaseType.MAIN1));

        Path solo = ShardRun.shardOf(tempDir.resolve("solo"), live, rate,
                (bracket, collectors) -> settle(collectors, live, seat));
        boolean soloAdmitted = admitsAPlayabilityRecord(solo);

        Path twice = ShardRun.shardOf(tempDir.resolve("twice"), live, rate,
                (bracket, collectors) -> {
                    settle(collectors, live, seat);
                    settle(collectors, live, seat);
                });
        boolean twiceAdmitted = admitsAPlayabilityRecord(twice);

        Path paired = ShardRun.shardOf(tempDir.resolve("paired"), live, rate,
                (bracket, collectors) -> {
                    settle(collectors, fork, fork.getPlayers().get(0));
                    settle(collectors, live, seat);
                });
        boolean pairedAdmitted = admitsAPlayabilityRecord(paired);

        assertFalse(soloAdmitted, "premise: alone, the live game's first draw is sampled out");
        assertTrue(twiceAdmitted, "premise: the sampler's second draw is sampled in, so a leaked "
                + "draw would admit the live record");
        assertEquals(soloAdmitted, pairedAdmitted,
                "a fork's stop ahead of it must not change whether the live candidate "
                        + "behind it is sampled in: solo=" + soloAdmitted
                        + " paired=" + pairedAdmitted);
    }

    /** A board whose first seat has one candidate, a creature spell in hand. */
    private static Game oneCandidate(Game game) {
        TestBoard.inHand(game, game.getPlayers().get(0), "Grizzly Bears");
        return game;
    }

    private static void settle(Channels collectors, Game game, Player priority) throws Throwable {
        collectors.phase().prioritySettledHandler().invoke(
                null, ON_PRIORITY_SETTLED, new Object[]{game, priority});
    }

    /** Whether a run sampled a playability record in; one that sampled none wrote no shard. */
    private static boolean admitsAPlayabilityRecord(Path shard) throws IOException {
        return Files.exists(shard) && ShardRun.readShard(shard).stream()
                .anyMatch(l -> l.contains("\"kind\":\"playability\""));
    }

    // ── declarations and the deciding player ────────────────────────────
    //
    // The engine reports legality from the turn-based declaration itself and
    // names the deciding player, so every record is a real declaration
    // whatever the step -- an attackers record carries no what_if at all
    // (FR-037), a blockers record what_if=false -- and FR-030a's actor is the
    // player the engine names; FR-030 stamps random_seat off that actor.
    // These build a two-player game so the deciding player is a real seat.
    // An attackers declaration nothing ended is written when the collectors
    // close; AttackersRecordTest covers the declaration's end.

    private static List<String> recordsOf(List<String> lines) {
        return lines.stream().filter(l -> l.contains("\"kind\":\"playability\"")).toList();
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
        assertFalse(records.get(0).contains("what_if"), records.get(0));
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
     * A blocker Pacifism keeps from blocking at all is forbidden by that
     * static, named through Forge's {@code cantBlockStatic} (F3 #2): the
     * pair-wise lookup finds no static for it.
     */
    @Test
    void aPacifiedBlockerNamesPacifismAsItsResponsibleStatic() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player attacking = game.getPlayers().get(0);
        Player blocking = game.getPlayers().get(1);
        Card attacker = TestBoard.inPlay(game, attacking, "Grizzly Bears");
        Card blocker = TestBoard.inPlay(game, blocking, "Hill Giant");
        Card pacifism = TestBoard.inPlay(game, attacking, "Pacifism");
        pacifism.attachToEntity(blocker, null);
        Combat combat = new Combat(attacking);
        combat.addAttacker(attacker, blocking);
        game.getPhaseHandler().setCombat(combat);

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) ->
                collectors.playability().legalityHandler().invoke(null, ON_BLOCKERS_DECLARABLE,
                        new Object[]{blocking, attacker, List.of(blocker), List.of(), 1}));

        String record = recordsOf(lines).get(0);
        assertTrue(record.contains("\"responsible_static\":[{\"script_file\":\"cardsfolder/p/pacifism.txt\""),
                record);
        assertFalse(record.contains("required_elsewhere"), record);
    }

    /**
     * A blocker a lure makes block another attacker is kept from blocking
     * this one by a requirement, not a prohibition (F3 #7): its forbidden
     * entry says {@code required_elsewhere}.
     */
    @Test
    void aBlockerALureCallsElsewhereIsRequiredElsewhere() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player attacking = game.getPlayers().get(0);
        Player blocking = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, attacking, "Grizzly Bears");
        Card unicorn = TestBoard.inPlay(game, attacking, "Prized Unicorn");
        Card blocker = TestBoard.inPlay(game, blocking, "Hill Giant");
        Combat combat = new Combat(attacking);
        combat.addAttacker(bears, blocking);
        combat.addAttacker(unicorn, blocking);
        game.getPhaseHandler().setCombat(combat);

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) ->
                collectors.playability().legalityHandler().invoke(null, ON_BLOCKERS_DECLARABLE,
                        new Object[]{blocking, bears, List.of(blocker), List.of(), 1}));

        String record = recordsOf(lines).get(0);
        assertTrue(record.contains("\"required_elsewhere\":true"), record);
        assertTrue(record.contains("\"responsible_static\":[]"), record);
    }

    /**
     * A lured creature the restrictions keep from the anchor anyway is not
     * required elsewhere (509.1c): a ground creature against a flying
     * anchor, while a lure attacks too, is forbidden by the flying alone.
     */
    @Test
    void aLuredBlockerAFlyerShutsOutIsNotRequiredElsewhere() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player attacking = game.getPlayers().get(0);
        Player blocking = game.getPlayers().get(1);
        Card drake = TestBoard.inPlay(game, attacking, "Wind Drake");
        Card unicorn = TestBoard.inPlay(game, attacking, "Prized Unicorn");
        Card blocker = TestBoard.inPlay(game, blocking, "Hill Giant");
        Combat combat = new Combat(attacking);
        combat.addAttacker(drake, blocking);
        combat.addAttacker(unicorn, blocking);
        game.getPhaseHandler().setCombat(combat);

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) ->
                collectors.playability().legalityHandler().invoke(null, ON_BLOCKERS_DECLARABLE,
                        new Object[]{blocking, drake, List.of(blocker), List.of(), 1}));

        String record = recordsOf(lines).get(0);
        assertFalse(record.contains("required_elsewhere"), record);
        assertTrue(record.contains("\"responsible_static\":[{\"script_file\":\"cardsfolder/w/wind_drake.txt\""),
                "the flyer's keyword line keeps the blocker out: " + record);
    }

    /**
     * A creature kept from blocking by a "can't block this turn" effect
     * (Panic's hidden keyword, no static) is forbidden with an empty
     * {@code responsible_static}: no static is to blame.
     */
    @Test
    void aBlockerACantBlockEffectStopsNamesNoStatic() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player attacking = game.getPlayers().get(0);
        Player blocking = game.getPlayers().get(1);
        Card attacker = TestBoard.inPlay(game, attacking, "Grizzly Bears");
        Card blocker = TestBoard.inPlay(game, blocking, "Hill Giant");
        blocker.addHiddenExtrinsicKeywords(game.getNextTimestamp(), 0,
                List.of("CARDNAME can't block."));
        Combat combat = new Combat(attacking);
        combat.addAttacker(attacker, blocking);
        game.getPhaseHandler().setCombat(combat);

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) ->
                collectors.playability().legalityHandler().invoke(null, ON_BLOCKERS_DECLARABLE,
                        new Object[]{blocking, attacker, List.of(blocker), List.of(), 1}));

        String record = recordsOf(lines).get(0);
        assertTrue(record.contains("\"responsible_static\":[]"), record);
        assertFalse(record.contains("required_elsewhere"), record);
    }

    /**
     * The blockers record Forge's own verdict gives: the candidates
     * {@code CombatUtil.canBlock(anchor, candidate, combat)} admits are
     * legal, as the engine's declaration has them.
     */
    private int declarations;

    private String blockersRecord(Game game, Player attacking, Player blocking, Card anchor,
            List<Card> attackers, List<Card> candidates) throws Throwable {
        Combat combat = new Combat(attacking);
        for (Card attacker : attackers) {
            combat.addAttacker(attacker, blocking);
        }
        game.getPhaseHandler().setCombat(combat);
        List<Card> legal = candidates.stream()
                .filter(c -> forge.game.combat.CombatUtil.canBlock(anchor, c, combat)).toList();
        List<String> lines = ShardRun.records(
                tempDir.resolve("declaration-" + declarations++), game,
                (bracket, collectors) -> collectors.playability().legalityHandler().invoke(
                        null, ON_BLOCKERS_DECLARABLE,
                        new Object[]{blocking, anchor, candidates, legal, 1}));
        return recordsOf(lines).get(0);
    }

    /**
     * Another attacker that "must be blocked if able", "by exactly one
     * creature" or "by two or more creatures" while that is unmet makes
     * Forge hold every able blocker to it, so a creature free to block the
     * anchor otherwise is refused: a requirement elsewhere, not a
     * prohibition.
     */
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(strings = {
            "CARDNAME must be blocked if able.",
            "CARDNAME must be blocked by exactly one creature if able.",
            "CARDNAME must be blocked by two or more creatures if able."})
    void aMustBeBlockedAttackerElsewhereMakesTheBlockerRequiredElsewhere(String keyword)
            throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player attacking = game.getPlayers().get(0);
        Player blocking = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, attacking, "Grizzly Bears");
        // Canopy Stalker, Nacatl War Pride, Gorm the Great print these.
        Card required = TestBoard.inPlay(game, attacking, "Runeclaw Bear");
        required.addChangedCardKeywords(
                List.of(keyword), null, false, game.getNextTimestamp(), null);
        Card blocker = TestBoard.inPlay(game, blocking, "Hill Giant");

        String record = blockersRecord(
                game, attacking, blocking, bears, List.of(bears, required), List.of(blocker));

        assertTrue(record.contains("\"legal_blockers\":[]"), "Forge refuses the block: " + record);
        assertTrue(record.contains("\"forbidden\":[{\"entity\":\"" + Refs.of(blocker)
                + "\",\"responsible_static\":[],\"required_elsewhere\":true}]"), record);
    }

    /** A creature printing "can't attack or block" names that keyword line, its own. */
    @Test
    void aBlockerItsOwnKeywordKeepsOutNamesThatLineAsItsOwn() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player attacking = game.getPlayers().get(0);
        Player blocking = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, attacking, "Grizzly Bears");
        Card grakk = TestBoard.inPlay(game, blocking, "Grakk the Pacifist");

        String record = blockersRecord(
                game, attacking, blocking, bears, List.of(bears), List.of(grakk));

        assertTrue(record.contains("\"forbidden\":[{\"entity\":\"" + Refs.of(grakk)
                + "\",\"responsible_static\":[{\"script_file\":\"cardsfolder/g/grakk_the_pacifist.txt\""
                + ",\"face\":0,\"trait_kind\":\"keyword\",\"index_within_kind\":0}]"
                + ",\"responsible_own\":true}]"), record);
    }

    /**
     * "Can't attack or block alone" keeps a creature out while its
     * controller has no other creature, naming its own line; with another
     * creature beside it, it may block.
     */
    @Test
    void aBlockerAloneItsOwnKeywordKeepsOutNamesThatLine() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player attacking = game.getPlayers().get(0);
        Player blocking = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, attacking, "Grizzly Bears");
        Card flunkies = TestBoard.inPlay(game, blocking, "Mogg Flunkies");

        String alone = blockersRecord(
                game, attacking, blocking, bears, List.of(bears), List.of(flunkies));
        assertTrue(alone.contains("\"responsible_static\":[{\"script_file\":"
                + "\"cardsfolder/m/mogg_flunkies.txt\""), alone);
        assertTrue(alone.contains("\"responsible_own\":true"), alone);

        Card friend = TestBoard.inPlay(game, blocking, "Hill Giant");
        String together = blockersRecord(
                game, attacking, blocking, bears, List.of(bears), List.of(flunkies, friend));
        assertTrue(together.contains("\"forbidden\":[]"), together);
    }

    /** Pacifism is another card's static: the entry names it and is not the creature's own. */
    @Test
    void aStaticElsewhereIsNotTheCreaturesOwn() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player attacking = game.getPlayers().get(0);
        Player blocking = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, attacking, "Grizzly Bears");
        Card blocker = TestBoard.inPlay(game, blocking, "Hill Giant");
        TestBoard.inPlay(game, attacking, "Pacifism").attachToEntity(blocker, null);

        String record = blockersRecord(
                game, attacking, blocking, bears, List.of(bears), List.of(blocker));

        assertTrue(record.contains("cardsfolder/p/pacifism.txt"), record);
        assertFalse(record.contains("responsible_own"), record);
        assertFalse(record.contains("responsible_unresolved"), record);
    }

    /**
     * A static the resolver cannot key (an engine-built effect card's, no
     * script and no ability that made it) still explains the forbidden
     * blocker: no key, and the reason there is none. An effect a resolved
     * spell left (Ember Gale's) keys to that spell's line.
     */
    @Test
    void aBlockerAnUnkeyableStaticKeepsOutSaysWhyItHasNoKey() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player attacking = game.getPlayers().get(0);
        Player blocking = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, attacking, "Grizzly Bears");
        Card blocker = TestBoard.inPlay(game, blocking, "Hill Giant");
        Card effect = new Card(game.nextCardId(), game);
        effect.setName("Engine Effect");
        effect.setOwner(attacking);
        effect.setGamePieceType(forge.card.GamePieceType.EFFECT);
        effect.addStaticAbility("Mode$ CantBlock | ValidCard$ Creature.OppCtrl | EffectZone$ Command"
                + " | Description$ Creatures your opponents control can't block.");
        game.getAction().moveTo(forge.game.zone.ZoneType.Command, effect, null, null);
        assertTrue(forge.game.staticability.StaticAbilityCantAttackBlock.cantBlockStatic(blocker)
                != null, "the effect's static keeps the blocker out");

        String record = blockersRecord(
                game, attacking, blocking, bears, List.of(bears), List.of(blocker));

        assertTrue(record.contains("\"responsible_static\":[],\"responsible_unresolved\":\"engine_effect\""),
                record);
    }

    /** An attacker printing "can't attack or block" names that keyword line too. */
    @Test
    void anAttackerItsOwnKeywordKeepsOutNamesThatLine() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_ATTACKERS);
        Player active = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        Card grakk = TestBoard.inPlay(game, active, "Grakk the Pacifist");

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) ->
                collectors.playability().legalityHandler().invoke(null, ON_ATTACKERS_DECLARABLE,
                        new Object[]{active, other, List.of(grakk), List.of()}));

        String record = recordsOf(lines).get(0);
        assertTrue(record.contains("\"forbidden\":[{\"entity\":\"" + Refs.of(grakk)
                + "\",\"responsible_static\":[{\"script_file\":\"cardsfolder/g/grakk_the_pacifist.txt\""),
                record);
        assertTrue(record.contains("\"responsible_own\":true"), record);
    }

    /**
     * Every declaration on a board that moved is written, and the same
     * declaration on an unchanged board is written once (FR-028). A defender
     * asked about a second time is a new declaration.
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
     * FR-030a's decision row: a candidate of the player without priority
     * names that player, not the active one, and carries no {@code what_if}.
     */
    @Test
    void aDecisionRecordNamesTheCandidatesPlayer() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        TestBoard.inHand(game, other, "Lightning Bolt");

        List<String> lines = ShardRun.records(tempDir, game,
                caps().manaCap(0).everyDecision().build(),
                (bracket, collectors) -> settle(collectors, game, active));

        List<String> records = recordsOf(lines);
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains(
                "\"actor_player\":\"" + Refs.of(other) + "\""), records.get(0));
        assertFalse(records.get(0).contains("what_if"),
                "a decision record carries no what_if: " + records.get(0));
    }
}
