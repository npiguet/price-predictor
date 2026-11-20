package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The kept-record cases (FR-010c) played forward: they play with no
 * decision candidate sampled, live or in the fork, and find the bound
 * card's candidates, or its trigger's not-fired evaluations, in the fork's
 * records all the same.
 */
@ExtendWith(ForgeExtension.class)
class DirectedForkPlayedForwardKeptRecordsTest extends DirectedForkPlayedForwardSupport {

    // ── kept records (FR-010c) ──────────────────────────────────────────
    //
    // Each case plays with no decision candidate sampled, live or in the
    // fork: the fork's records still hold the bound card's candidates, which
    // its variant keeps, and the live game's hold none.

    /** A case's board: the game and the card whose records the case reads. */
    record Setup(Game game, Card host) {
    }

    /**
     * Play {@code board} with no decision sampled, so only the candidates the
     * rule's fork keeps are written, and return the fork's records; when
     * dumping, play a second board with every candidate written, so the live
     * game's side is dumped too.
     */
    private List<String> keptFork(java.util.function.Supplier<Setup> board, PhaseType step, String rule,
                                  String variant) throws Throwable {
        Setup setup = board.get();
        List<String> all = played(setup.game(), step, rule, Decisions.NONE);
        for (String record : all) {
            if (!Boolean.TRUE.equals(parse(record).get("fork"))) {
                assertFalse("decision".equals(parse(record).get("subkind")),
                        "no live decision candidate is sampled at rate 0: " + record);
            }
        }
        if (DUMP != null) {
            played(board.get().game(), step, rule, Decisions.EVERY);
        }
        return forkOf(all, rule, variant);
    }

    /** The candidates of the decision records among {@code records} whose card is {@code host}. */
    private static List<Map<String, Object>> candidatesOf(List<String> records, Card host) {
        List<Map<String, Object>> found = new ArrayList<>();
        for (String record : records) {
            Map<String, Object> parsed = parse(record);
            if (!"decision".equals(parsed.get("subkind"))) {
                continue;
            }
            Object source = RecordLines.asMap(RecordLines.asMap(parsed.get("state")).get("refs")).get("source");
            if (ref(host).equals(source) && RecordLines.payload(record).get("candidates") instanceof List<?> list) {
                list.forEach(candidate -> found.add(RecordLines.asMap(candidate)));
            }
        }
        return found;
    }

    /** A verdict bit of each candidate. */
    private static List<Object> bits(List<Map<String, Object>> candidates, String bit) {
        return candidates.stream().map(candidate -> RecordLines.asMap(candidate.get("verdict")).get(bit)).toList();
    }

    /** The failed checks of each candidate, by identifier. */
    private static List<List<Object>> failedChecks(List<Map<String, Object>> candidates) {
        List<List<Object>> found = new ArrayList<>();
        for (Map<String, Object> candidate : candidates) {
            List<Object> checks = new ArrayList<>();
            if (candidate.get("failed_checks") instanceof List<?> list) {
                list.forEach(check -> checks.add(RecordLines.asMap(check).get("check")));
            }
            found.add(checks);
        }
        return found;
    }

    /**
     * {@code add_permanent}, 601.2c: the active player holds Undo ("return
     * two target creatures") with Grizzly Bears the only creature as its
     * first main phase begins. The live Undo has no legal choice of targets;
     * the fork's opponent also controls Craw Wurm, and the fork's kept
     * decision records show Undo with a legal choice of targets.
     */
    @Test
    void aSecondCreatureGivesTheTwoTargetSpellALegalChoice() throws Throwable {
        Card[] undo = new Card[1];
        List<String> fork = keptFork(() -> {
            Game game = bare(PhaseType.MAIN1);
            Player active = game.getPlayers().get(0);
            lands(game, active, "Island", 3);
            permanent(game, game.getPlayers().get(1), "Grizzly Bears");
            undo[0] = TestBoard.inHand(game, active, "Undo");
            return new Setup(game, undo[0]);
        }, PhaseType.MAIN1, "601.2c", "two-creatures");

        List<Object> verdicts = bits(candidatesOf(fork, undo[0]), "has_legal_target");
        assertFalse(verdicts.isEmpty(), "a kept fork decision record names Undo");
        assertTrue(verdicts.contains(true), "Undo has a legal choice of targets in the fork: " + verdicts);
    }

    /** 405.6d: a face-down Exalted Angel over four Plains; the fork taps one, so the morph cost cannot be paid. */
    @Test
    void aTappedPlainsLeavesTheMorphCostUnpaid() throws Throwable {
        Card[] angel = new Card[1];
        List<String> fork = keptFork(() -> {
            Game game = bare(PhaseType.MAIN1);
            Player active = game.getPlayers().get(0);
            lands(game, active, "Plains", 4);
            angel[0] = TestBoard.card(game, active, "Exalted Angel");
            angel[0].turnFaceDownNoUpdate();
            active.getZone(forge.game.zone.ZoneType.Battlefield).add(angel[0]);
            angel[0].setSickness(false);
            angel[0].setGameTimestamp(game.getNextTimestamp());
            return new Setup(game, angel[0]);
        }, PhaseType.MAIN1, "405.6d", "cannot_pay");

        List<Object> playable = bits(candidatesOf(fork, angel[0]), "can_play");
        assertFalse(playable.isEmpty(), "a kept candidate turns the Angel face up");
        assertTrue(playable.contains(false), playable.toString());
    }

    /** 702.126a: Reverse Engineer over two Islands and three Ornithopters; the fork taps one, so improvise is a mana short. */
    @Test
    void aTappedArtifactLeavesImproviseShort() throws Throwable {
        Card[] engineer = new Card[1];
        List<String> fork = keptFork(() -> {
            Game game = bare(PhaseType.MAIN1);
            Player active = game.getPlayers().get(0);
            lands(game, active, "Island", 2);
            lands(game, active, "Ornithopter", 3);
            engineer[0] = TestBoard.inHand(game, active, "Reverse Engineer");
            return new Setup(game, engineer[0]);
        }, PhaseType.MAIN1, "702.126a", "one_artifact_tapped");

        List<Object> affordable = bits(candidatesOf(fork, engineer[0]), "affordable");
        assertFalse(affordable.isEmpty(), "kept candidates of Reverse Engineer");
        assertTrue(affordable.contains(false), affordable.toString());
    }

    /** 702.127a: Cut // Ribbons in A's graveyard; the fork has it in hand, where the aftermath half fails its zone check. */
    @Test
    void theAftermathHalfInHandFailsItsZoneCheck() throws Throwable {
        Card[] cut = new Card[1];
        List<String> fork = keptFork(() -> {
            Game game = bare(PhaseType.MAIN1);
            cut[0] = TestBoard.inGraveyard(game, game.getPlayers().get(0), "Cut // Ribbons");
            return new Setup(game, cut[0]);
        }, PhaseType.MAIN1, "702.127a", "from_hand");

        List<List<Object>> checks = failedChecks(candidatesOf(fork, cut[0]));
        assertTrue(checks.stream().anyMatch(failed -> failed.contains("zone")), checks.toString());
    }

    /** 702.187b: Electro's Bolt discarded this turn over two Mountains; the fork takes one away, so the mayhem cast is unplayable. */
    @Test
    void oneMountainLeavesMayhemUnplayable() throws Throwable {
        Card[] bolt = new Card[1];
        List<String> fork = keptFork(() -> {
            Game game = bare(PhaseType.MAIN2);
            Player active = game.getPlayers().get(0);
            lands(game, active, "Mountain", 2);
            bolt[0] = TestBoard.inGraveyard(game, active, "Electro's Bolt");
            bolt[0].setDiscarded(true);
            return new Setup(game, bolt[0]);
        }, PhaseType.MAIN2, "702.187b", "one_mountain");

        List<Object> playable = bits(candidatesOf(fork, bolt[0]), "can_play");
        assertTrue(playable.contains(false), playable.toString());
    }

    /** 702.188a: Spider-Man, Web-Slinger over a Plains and a tapped Grizzly Bears; the fork untaps the Bears, so nothing can be returned. */
    @Test
    void anUntappedCreatureLeavesWebSlingingUnplayable() throws Throwable {
        Card[] spider = new Card[1];
        List<String> fork = keptFork(() -> {
            Game game = bare(PhaseType.MAIN2);
            Player active = game.getPlayers().get(0);
            permanent(game, active, "Plains");
            permanent(game, active, "Grizzly Bears").setTapped(true);
            spider[0] = TestBoard.inHand(game, active, "Spider-Man, Web-Slinger");
            return new Setup(game, spider[0]);
        }, PhaseType.MAIN2, "702.188a", "creature_untapped");

        List<Object> playable = bits(candidatesOf(fork, spider[0]), "can_play");
        assertTrue(playable.contains(false), playable.toString());
    }

    /** 718.3a: Blitz Automaton over three Mountains; the fork takes one away, so the prototype cast is unplayable. */
    @Test
    void twoMountainsLeaveThePrototypeUnplayable() throws Throwable {
        Card[] automaton = new Card[1];
        List<String> fork = keptFork(() -> {
            Game game = bare(PhaseType.MAIN1);
            Player active = game.getPlayers().get(0);
            lands(game, active, "Mountain", 3);
            automaton[0] = TestBoard.inHand(game, active, "Blitz Automaton");
            return new Setup(game, automaton[0]);
        }, PhaseType.MAIN1, "718.3a", "two_mountains");

        List<Object> playable = bits(candidatesOf(fork, automaton[0]), "can_play");
        assertTrue(playable.contains(false), playable.toString());
    }

    /**
     * 702.84a: Dregscape Zombie in A's graveyard over a Swamp; the fork has it
     * in hand, where unearth fails its zone check. The fork keeps the Zombie's
     * candidates at every priority stop, but writes each (candidate, failed
     * checks) pair once: one record of unearth failing its zone check alone.
     */
    @Test
    void unearthFromHandFailsItsZoneCheck() throws Throwable {
        Card[] zombie = new Card[1];
        List<String> fork = keptFork(() -> {
            Game game = bare(PhaseType.MAIN1);
            Player active = game.getPlayers().get(0);
            permanent(game, active, "Swamp");
            zombie[0] = TestBoard.inGraveyard(game, active, "Dregscape Zombie");
            return new Setup(game, zombie[0]);
        }, PhaseType.MAIN1, "702.84a", "card_in_hand");

        List<Map<String, Object>> candidates = candidatesOf(fork, zombie[0]);
        List<List<Object>> checks = failedChecks(candidates);
        assertEquals(1, checks.stream().filter(List.of("zone")::equals).count(), checks.toString());
        List<List<Object>> pairs = new ArrayList<>();
        for (int i = 0; i < candidates.size(); i++) {
            pairs.add(List.of(candidates.get(i).get("ability"), checks.get(i)));
        }
        assertEquals(pairs.stream().distinct().toList(), pairs, "each (candidate, failed checks) pair is written once");
    }

    /** The not-fired trigger records among {@code records} whose trigger's host is {@code host}. */
    private static List<String> notFired(List<String> records, Card host) {
        List<String> found = new ArrayList<>();
        for (String record : records) {
            Map<String, Object> parsed = parse(record);
            Object source = RecordLines.asMap(RecordLines.asMap(parsed.get("state")).get("refs")).get("source");
            if ("trigger".equals(parsed.get("kind")) && ref(host).equals(source)
                    && Boolean.FALSE.equals(RecordLines.payload(record).get("fired"))) {
                found.add(record);
            }
        }
        return found;
    }

    /**
     * 608.2p, inverted: Narci, Fable Singer beside History of Benalia at one
     * lore counter as the first main phase begins. The live chapter II leaves
     * Narci unfired; the fork's Saga has two counters, so the final chapter
     * resolves and Narci's trigger fires. Its evaluations against the fork's
     * other resolutions, which do not fire, the fork keeps although its
     * sampler holds no firing to repay (FR-010c).
     */
    @Test
    void theFinalChapterFiresTheResolutionTrigger() throws Throwable {
        Card[] narci = new Card[1];
        List<String> fork = keptFork(() -> {
            Game game = bare(PhaseType.MAIN1);
            Player active = game.getPlayers().get(0);
            narci[0] = permanent(game, active, "Narci, Fable Singer");
            permanent(game, active, "History of Benalia").setCounters(CounterEnumType.LORE, 1);
            return new Setup(game, narci[0]);
        }, PhaseType.MAIN1, "608.2p", "final-chapter");

        assertFalse(fired(fork, narci[0]).isEmpty(), "Narci's trigger fires in the fork");
        assertFalse(notFired(fork, narci[0]).isEmpty(), "Narci's trigger evaluated and not fired, kept");
    }

    /**
     * 702.105a: Enraged Revolutionary (A at 15 life) as the declare-attackers
     * step begins, B at 20; the fork's B has 15, tied with A, and the
     * Revolutionary is made to attack B (FR-010b), so dethrone fires.
     */
    @Test
    void attackingThePlayerTiedForTheMostLifeFiresDethrone() throws Throwable {
        Card[] revolutionary = new Card[1];
        List<String> fork = keptFork(() -> {
            Game game = bare(PhaseType.MAIN1);
            Player active = game.getPlayers().get(0);
            revolutionary[0] = permanent(game, active, "Enraged Revolutionary");
            active.setLife(15, null);
            return new Setup(game, revolutionary[0]);
        }, null, "702.105a", "tied");

        assertFalse(fired(fork, revolutionary[0]).isEmpty(), "dethrone fires in the fork");
    }
}
