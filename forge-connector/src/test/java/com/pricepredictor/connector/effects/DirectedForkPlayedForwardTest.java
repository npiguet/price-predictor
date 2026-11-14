package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.directed.KeywordEdit;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.combat.Combat;
import forge.game.event.GameEventGameFinished;
import forge.game.keyword.Keyword;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.events;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * One registered rule per operation family played forward (review TC3's
 * "real fix"): a live game set up as the rule's base, recorded by a
 * {@code GameRecording} that takes the rule's directed fork at the boundary
 * ({@code DirectedFork.take}), the live turn played out and the fork played
 * to its end; the fork's records then show the variant's point happening,
 * not only the edited board. Each case picks a rule whose point is the
 * engine's own (a turn-based action, a trigger, combat damage, the
 * collector's own reading of a priority stop's candidates), so neither
 * seat's choices decide it.
 *
 * <p>Two cases read a record no other kind of point writes: the engine's
 * own question before a declaration of blockers (702.118b's blockers
 * record), and the game's end at a state-based check (104.4a's two losses
 * live, and the fork's one beside a win).
 *
 * <p>The kept-record cases (FR-010c) play with no decision candidate
 * sampled, live or in the fork, and find the bound card's candidates, or
 * its trigger's not-fired evaluations, in the fork's records all the same.
 * The inverted cases (FR-014a) bind the common board and find the rare
 * fact's record in the fork. A few cases' point follows a seat's own cast or
 * attack (104.3d, 702.105a, 702.124j, 702.131a), which Forge's AI makes on
 * these boards; a forced play (FR-010b) would make it certain. Run with {@code -Ddirected.records.dump=DIR}, every
 * case's records are copied to {@code DIR} for tagging with the Python
 * detectors.
 *
 * <p>{@code pt} is played forward by
 * {@code ForkRecordsTest.theRegisteredToughnessZeroRuleForksATougherCreatureToTheEndOfTheTurn}.
 * The families left out ({@code tap}, {@code colour_type}) have no
 * registered rule whose point happens without a seat's choice: a spell
 * cast, an attack, a block.
 */
@ExtendWith(ForgeExtension.class)
class DirectedForkPlayedForwardTest {

    @TempDir
    Path tempDir;

    /** A started two-seat game in {@code step} of the first seat's third turn. */
    private static Game board(PhaseType step) {
        Game game = ForkPlayForwardTest.board(TestBoard.twoPlayerGame(PhaseType.MAIN1));
        game.getPhaseHandler().devModeSet(step, game.getPlayers().get(0), 3);
        return game;
    }

    /** A permanent since the turn began, with a timestamp of its own. */
    private static Card permanent(Game game, Player owner, String name) {
        Card card = TestBoard.inPlay(game, owner, name);
        card.setGameTimestamp(game.getNextTimestamp());
        card.setSickness(false);
        return card;
    }

    /**
     * Record the game from the boundary beginning {@code step}, taking
     * {@code rule}'s directed fork there, and play the live turn out; the
     * fork's records, each checked to name the rule.
     */
    private List<String> forkOf(Game game, PhaseType step, String rule, String variant) throws Throwable {
        return forkOf(game, step, rule, variant, Decisions.SAMPLED);
    }

    /** {@link #forkOf(Game, PhaseType, String, String)}, decision candidates sampled as {@code decisions} says. */
    private List<String> forkOf(Game game, PhaseType step, String rule, String variant, Decisions decisions)
            throws Throwable {
        List<String> forked = new ArrayList<>();
        for (String record : played(game, step, rule, decisions)) {
            Map<String, Object> parsed = parse(record);
            if (Boolean.TRUE.equals(parsed.get("fork"))) {
                assertEquals(List.of(rule), parsed.get("aimed_at"), record);
                assertEquals(variant, RecordLines.asMap(parsed.get("directed")).get("variant"), record);
                forked.add(record);
            }
        }
        assertFalse(forked.isEmpty(), rule + ": the fork was kept and recorded");
        return forked;
    }

    /**
     * Every record of the game recorded from the boundary beginning
     * {@code step}, {@code rule}'s directed fork taken there, and the live
     * turn played out; with {@code step} null, the live game reaches the
     * boundary itself, its turn-based actions included. A live game the turn
     * ended writes its end, as {@code Match.startGame} has it do.
     */
    private List<String> played(Game game, PhaseType step, String rule, Decisions decisions) throws Throwable {
        TestCaps built = TestCaps.caps().manaCap(2000).interventions(0).probes(0);
        built = switch (decisions) {
            case SAMPLED -> built;
            case NONE -> built.playabilityRate(0.0).candidateRates(0.0, 0.0);
            case EVERY -> built.everyDecision();
        };
        CollectionCaps caps = built.directed(1, 1.0, rule).build();
        // A directory of its own per game: one test may play several.
        RecordShardWriter shard = ShardRun.writer(
                java.nio.file.Files.createDirectories(tempDir.resolve("game-" + DUMPED.incrementAndGet())));
        try (GameRecording recording = new GameRecording(game, shard, ShardRun.GAME_ID, caps, 7L, null)) {
            recording.open();
            game.getTriggerHandler().resetActiveTriggers();
            if (step != null) {
                TestBoard.enterStep(game, step);
            }
            ForkPlayForwardTest.playTheTurnOut(game);
            if (game.isGameOver()) {
                game.fireEvent(new GameEventGameFinished());
            }
        } finally {
            shard.close();
        }
        List<String> records = ShardRun.readShard(shard.path());
        dump(rule, records);
        return records;
    }

    /** How a case's decision candidates are sampled, live and in the fork alike. */
    enum Decisions {
        /** At the defaults' rates. */
        SAMPLED,
        /** At rate 0: only the candidates a fork keeps (FR-010c) are written. */
        NONE,
        /** At rate 1: every candidate is written. */
        EVERY
    }

    /**
     * Where each case's records are copied when the property
     * {@code directed.records.dump} names a directory: one
     * {@code <rule>-<n>.jsonl} per game played, for tagging with the Python
     * detectors (the brief's check that both sides of a rule are tagged).
     */
    private static final String DUMP = System.getProperty("directed.records.dump");
    private static final java.util.concurrent.atomic.AtomicInteger DUMPED = new java.util.concurrent.atomic.AtomicInteger();

    private static void dump(String rule, List<String> records) throws java.io.IOException {
        if (DUMP == null) {
            return;
        }
        Path dir = Path.of(DUMP);
        java.nio.file.Files.createDirectories(dir);
        java.nio.file.Files.write(dir.resolve(rule + "-" + DUMPED.incrementAndGet() + ".jsonl"), records,
                java.nio.charset.StandardCharsets.UTF_8);
    }

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

    /** The fork's records among {@code all}, each checked to name the rule and the variant. */
    private static List<String> forkOf(List<String> all, String rule, String variant) {
        List<String> forked = new ArrayList<>();
        for (String record : all) {
            Map<String, Object> parsed = parse(record);
            if (Boolean.TRUE.equals(parsed.get("fork"))) {
                assertEquals(List.of(rule), parsed.get("aimed_at"), record);
                assertEquals(variant, RecordLines.asMap(parsed.get("directed")).get("variant"), record);
                forked.add(record);
            }
        }
        assertFalse(forked.isEmpty(), rule + ": the fork was kept and recorded");
        return forked;
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

    /** Every event of {@code records} of {@code type} naming {@code ref}. */
    private static List<Map<String, Object>> eventsOf(List<String> records, String type, String ref) {
        List<Map<String, Object>> found = new ArrayList<>();
        for (String record : records) {
            for (Map<String, Object> event : events(record)) {
                if (type.equals(event.get("type")) && event.get("subjects") instanceof List<?> subjects
                        && subjects.contains(ref)) {
                    found.add(event);
                }
            }
        }
        return found;
    }

    private static String ref(Card card) {
        return "E" + card.getId();
    }

    /**
     * {@code counters}, 122.1d: a tapped Grizzly Bears with a stun counter
     * as the untap step begins. The fork takes the counter away, so the
     * step untaps the Bears, where the live game removes the counter instead.
     */
    @Test
    void aStunCounterTakenAwayLetsThePermanentUntap() throws Throwable {
        Game game = board(PhaseType.UNTAP);
        Card bears = game.getCardsIn(forge.game.zone.ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElseThrow();
        bears.setTapped(true);
        bears.setCounters(CounterEnumType.STUN, 1);

        List<String> fork = forkOf(game, PhaseType.UNTAP, "122.1d", "no_stun");

        assertFalse(eventsOf(fork, "untapped", ref(bears)).isEmpty(), "the Bears untap in the fork");
    }

    /**
     * {@code keyword}, 702.63c: Maelstrom Djinn with vanishing twice and two
     * time counters as the upkeep begins. The live game's two triggers take
     * both counters and sacrifice it; the fork's one instance takes one, and
     * the Djinn stays.
     */
    @Test
    void oneInstanceOfVanishingTakesOneCounter() throws Throwable {
        Game game = board(PhaseType.UPKEEP);
        Player active = game.getPlayers().get(0);
        Card djinn = permanent(game, active, "Maelstrom Djinn");
        KeywordEdit.add(game, djinn, "Vanishing");
        KeywordEdit.add(game, djinn, "Vanishing");
        djinn.setCounters(CounterEnumType.TIME, 2);
        assertEquals(2, djinn.getAmountOfKeyword(Keyword.VANISHING));

        List<String> fork = forkOf(game, PhaseType.UPKEEP, "702.63c", "one_instance");

        List<Map<String, Object>> removed = eventsOf(fork, "counter_change", ref(djinn));
        assertEquals(1, removed.size(), "one trigger, one counter: " + removed);
        assertTrue(eventsOf(fork, "sacrificed", ref(djinn)).isEmpty(), "the Djinn stays in the fork");
    }

    /**
     * {@code life}, 104.2b, inverted: Felidar Sovereign's controller at 39
     * life as the upkeep begins. The live trigger does not fire; the fork's
     * 40 life fires it, it resolves and wins the game, the {@code player_won}
     * in the resolution's effect half 104.2b-1's declaration reads.
     */
    @Test
    void fortyLifeWinsTheGameInTheFork() throws Throwable {
        Game game = board(PhaseType.UPKEEP);
        Player active = game.getPlayers().get(0);
        permanent(game, active, "Felidar Sovereign");
        active.setLife(39, null);

        List<String> fork = forkOf(game, PhaseType.UPKEEP, "104.2b", "forty");

        List<String> won = new ArrayList<>();
        for (String record : fork) {
            if (RecordLines.eventTypes(record).contains("player_won")) {
                won.add(parse(record).get("kind") + "/" + parse(record).get("moment"));
            }
        }
        assertEquals(List.of("resolution/resolution"), won, "the win is in the trigger's effect half");
        assertFalse(game.isGameOver(), "the live game goes on");
    }

    /**
     * {@code replace_card}, 510.2: A's Grizzly Bears blocked by B's Grizzly
     * Bears as the first-strike damage step begins. The live creatures trade
     * in the regular step; the fork's blocker is Youthful Knight, which kills
     * the attacker in the first-strike step and takes no damage.
     */
    @Test
    void aFirstStrikeBlockerInTheBlockersPlaceKillsFirst() throws Throwable {
        Game game = board(PhaseType.COMBAT_FIRST_STRIKE_DAMAGE);
        Player active = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        Card attacker = game.getCardsIn(forge.game.zone.ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElseThrow();
        attacker.setSickness(false);
        Card blocker = permanent(game, defending, "Grizzly Bears");
        TestBoard.blockedCombat(game, active, defending, attacker, blocker);

        List<String> fork = forkOf(game, PhaseType.COMBAT_FIRST_STRIKE_DAMAGE, "510.2",
                "knight_in_the_bears_place");

        List<Map<String, Object>> moves = eventsOf(fork, "zone_change", ref(attacker));
        assertTrue(moves.stream().anyMatch(move -> "graveyard".equals(params(move).get("to_zone"))),
                "the attacker dies in the fork: " + moves);
        String knight = null;
        for (String record : fork) {
            for (Map<String, Object> event : events(record)) {
                if ("damage_dealt".equals(event.get("type"))
                        && List.of(ref(attacker)).equals(event.get("subjects"))) {
                    knight = (String) params(event).get("source");
                }
            }
        }
        assertTrue(knight != null && !knight.equals(ref(blocker)), "a new card dealt the attacker's damage");
        assertTrue(eventsOf(fork, "damage_dealt", knight).isEmpty(), "the Knight is dealt no damage");
    }

    /**
     * {@code move_card}, 704.5b: the active player's library holds one card
     * as its upkeep begins. The live draw step draws it and the game goes
     * on; the fork's card is in the graveyard, so the draw step's draw is
     * attempted from the empty library and the player loses the game.
     */
    @Test
    void aDrawStepFromAnEmptyLibraryLosesTheGame() throws Throwable {
        Game game = board(PhaseType.UPKEEP);
        Player active = game.getPlayers().get(0);
        List<Card> library = List.copyOf(active.getCardsIn(forge.game.zone.ZoneType.Library));
        for (Card card : library.subList(1, library.size())) {
            game.getAction().exile(card, null, null);
        }
        assertEquals(1, active.getCardsIn(forge.game.zone.ZoneType.Library).size());

        List<String> fork = forkOf(game, PhaseType.UPKEEP, "704.5b", "empty");

        List<String> kinds = new ArrayList<>();
        Map<String, Object> lost = null;
        for (String record : fork) {
            for (Map<String, Object> event : events(record)) {
                if ("player_lost".equals(event.get("type"))
                        && List.of(Refs.of(active)).equals(event.get("subjects"))) {
                    lost = event;
                    kinds.add(parse(record).get("kind") + "/" + parse(record).get("moment"));
                }
            }
        }
        assertTrue(lost != null, "the active player loses in the fork: " + fork);
        assertEquals("Milled", params(lost).get("reason"), lost.toString());
        assertFalse(active.hasLost(), "the live player draws its last card and plays on");
        assertEquals(List.of("resolution/resolution"), kinds,
                "the loss is in an effect half, where 704.5b-1's declaration reads it");
    }

    // ── kept records (FR-010c) ──────────────────────────────────────────
    //
    // Each case plays with no decision candidate sampled, live or in the
    // fork: the fork's records still hold the bound card's candidates, which
    // its variant keeps, and the live game's hold none.

    /** A started two-seat game in {@code step} of the first seat's third turn, both libraries ten Forests. */
    private static Game bare(PhaseType step) {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        game.getPhaseHandler().devModeSet(step, active, 3);
        for (Player player : game.getPlayers()) {
            for (int i = 0; i < 10; i++) {
                game.getAction().moveToLibrary(TestBoard.card(game, player, "Forest"), null, null);
            }
        }
        return game;
    }

    private static Card lands(Game game, Player owner, String name, int count) {
        Card first = null;
        for (int i = 0; i < count; i++) {
            Card land = permanent(game, owner, name);
            first = first == null ? land : first;
        }
        return first;
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

    /** 702.84a: Dregscape Zombie in A's graveyard over a Swamp; the fork has it in hand, where unearth fails its zone check. */
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

        List<List<Object>> checks = failedChecks(candidatesOf(fork, zombie[0]));
        assertTrue(checks.stream().anyMatch(failed -> failed.contains("zone")), checks.toString());
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
     * 608.2p: Narci, Fable Singer beside History of Benalia at two lore
     * counters as the first main phase begins. The live final chapter fires
     * Narci; the fork's Saga has one counter, so chapter II resolves and
     * Narci's trigger is evaluated and does not fire, a negative the fork
     * keeps although its sampler holds no firing to repay.
     */
    @Test
    void aNonFinalChapterLeavesTheResolutionTriggerUnfired() throws Throwable {
        Card[] narci = new Card[1];
        List<String> fork = keptFork(() -> {
            Game game = bare(PhaseType.MAIN1);
            Player active = game.getPlayers().get(0);
            narci[0] = permanent(game, active, "Narci, Fable Singer");
            permanent(game, active, "History of Benalia").setCounters(CounterEnumType.LORE, 2);
            return new Setup(game, narci[0]);
        }, PhaseType.MAIN1, "608.2p", "not-final-chapter");

        assertFalse(notFired(fork, narci[0]).isEmpty(), "Narci's trigger evaluated and not fired, kept");
    }

    /**
     * 702.105a: Enraged Revolutionary (A at 15 life) as the declare-attackers
     * step begins, B at 20; the fork's B has 14, so if the Revolutionary
     * attacks, dethrone is evaluated and does not fire. Whether it attacks is
     * its seat's decision.
     */
    @Test
    void attackingThePlayerWithLessLifeLeavesDethroneUnfired() throws Throwable {
        Card[] revolutionary = new Card[1];
        List<String> fork = keptFork(() -> {
            Game game = bare(PhaseType.MAIN1);
            Player active = game.getPlayers().get(0);
            revolutionary[0] = permanent(game, active, "Enraged Revolutionary");
            active.setLife(15, null);
            return new Setup(game, revolutionary[0]);
        }, null, "702.105a", "less-life");

        assertFalse(notFired(fork, revolutionary[0]).isEmpty(), "dethrone evaluated and not fired, kept");
    }

    /** The subjects of every event of {@code record} of {@code type}. */
    private static List<Object> subjectsOf(String record, String type) {
        List<Object> found = new ArrayList<>();
        for (Map<String, Object> event : events(record)) {
            if (type.equals(event.get("type"))) {
                found.add(event.get("subjects"));
            }
        }
        return found;
    }

    /**
     * {@code life}, 104.4a: Plague Spitter's controller and its opponent at
     * 1 life as the untap step begins. In the live upkeep the trigger takes
     * both players' last life, and the game ends with two losses and no
     * winner, a draw; the fork, taken as the upkeep begins, has the opponent
     * at 2, and its game ends with the active player's loss beside the
     * opponent's win. Both ends are in a resolution's effect half, where
     * 104.4a-1's declaration reads them.
     */
    @Test
    void bothPlayersAtOneLifeDrawAndTheForksOpponentAtTwoWins() throws Throwable {
        Game game = board(PhaseType.UNTAP);
        Player active = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        permanent(game, active, "Plague Spitter");
        active.setLife(1, null);
        opponent.setLife(1, null);

        List<String> records = played(game, null, "104.4a", Decisions.SAMPLED);

        List<List<Object>> ends = new ArrayList<>();
        for (String record : records) {
            List<Object> lost = subjectsOf(record, "player_lost");
            if (!lost.isEmpty()) {
                Map<String, Object> parsed = parse(record);
                assertEquals("resolution/resolution", parsed.get("kind") + "/" + parsed.get("moment"), record);
                ends.add(List.of(Boolean.TRUE.equals(parsed.get("fork")), lost, subjectsOf(record, "player_won")));
            }
        }
        assertEquals(List.of(
                List.of(true, List.of(List.of(Refs.of(active))), List.of(List.of(Refs.of(opponent)))),
                List.of(false, List.of(List.of(Refs.of(active)), List.of(Refs.of(opponent))), List.of())),
                ends, "the fork's end has a winner, the live end none");
    }

    /**
     * {@code replace_card}, 702.118b: Furtive Homunculus (2/1, skulk)
     * attacks and B's Centaur Courser could block it but for its power, as
     * the declare-blockers step begins. The fork's Courser is Llanowar
     * Elves, which the blockers record of the engine's question before the
     * declaration lists among the Homunculus's legal blockers, whatever B
     * then blocks with.
     */
    @Test
    void anElfOfLesserPowerInTheCoursersPlaceIsALegalBlocker() throws Throwable {
        Game game = board(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player active = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        Card homunculus = permanent(game, active, "Furtive Homunculus");
        Card courser = permanent(game, defending, "Centaur Courser");
        Combat combat = new Combat(active);
        combat.addAttacker(homunculus, defending);
        game.getPhaseHandler().setCombat(combat);

        List<String> fork = forkOf(game, PhaseType.COMBAT_DECLARE_BLOCKERS, "702.118b", "lesser_power");

        List<Object> legal = new ArrayList<>();
        for (String record : fork) {
            if ("blockers".equals(parse(record).get("subkind"))
                    && ref(homunculus).equals(RecordLines.payload(record).get("anchor_attacker"))) {
                legal.add(RecordLines.payload(record).get("legal_blockers"));
            }
        }
        assertEquals(1, legal.size(), "one blockers record for the Homunculus: " + legal);
        assertTrue(legal.get(0) instanceof List<?> blockers && blockers.size() == 1
                && !blockers.contains(ref(courser)), "the Elves, a new card, may block: " + legal);
    }

    /**
     * Every record of a live turn whose {@code attacker} attacks B's Jace
     * Beleren unblocked beside a Grizzly Bears attacking B, Jace having left
     * the battlefield before the damage step (removed by whatever removed it),
     * so the attacker attacks Forge's placeholder (506.4c). The live game
     * leaves the declare-blockers step and begins the damage step itself,
     * {@code rule}'s directed fork taken at that boundary.
     */
    /** The Grizzly Bears attacking B beside the attacker of {@link #goneWalkerTurn}. */
    private String bearsRef;

    private List<String> goneWalkerTurn(String attackerName, String rule) throws Throwable {
        Game game = board(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player active = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        Card attacker = permanent(game, active, attackerName);
        Card bears = permanent(game, active, "Grizzly Bears");
        bearsRef = ref(bears);
        Card jace = permanent(game, defending, "Jace Beleren");
        Combat combat = new Combat(active);
        combat.addAttacker(attacker, jace);
        combat.addAttacker(bears, defending);
        combat.setBlocked(attacker, false);
        combat.setBlocked(bears, false);
        game.getPhaseHandler().setCombat(combat);
        game.getAction().moveToGraveyard(jace, null);
        assertTrue(Combat.isRemovedDefender(combat.getDefenderByAttacker(attacker)));
        return played(game, null, rule, Decisions.SAMPLED);
    }

    /**
     * The combat damage {@code records} (of the fork, or of the live game)
     * dealt to player {@code P1} by anything but the Bears beside the attacker.
     */
    private List<Map<String, Object>> combatDamageToB(List<String> records, boolean fork) {
        List<Map<String, Object>> found = new ArrayList<>();
        for (String record : records) {
            Map<String, Object> parsed = parse(record);
            if (Boolean.TRUE.equals(parsed.get("fork")) != fork || !"combat".equals(parsed.get("kind"))) {
                continue;
            }
            for (Map<String, Object> event : eventsOf(List.of(record), "damage_dealt", "P1")) {
                if (Boolean.TRUE.equals(params(event).get("combat")) && !bearsRef.equals(params(event).get("source"))) {
                    found.add(event);
                }
            }
        }
        return found;
    }

    /** Whether {@code records} hold a combat record (of the fork, or of the live game). */
    private static boolean hasCombat(List<String> records, boolean fork) {
        return records.stream().map(RecordLines::parse).anyMatch(parsed ->
                Boolean.TRUE.equals(parsed.get("fork")) == fork && "combat".equals(parsed.get("kind")));
    }

    /**
     * {@code replace_card}, 702.19e: Thrasta, Tempest's Roar attacks a
     * planeswalker that has left combat and deals its 7 to B; the fork's
     * Colossal Dreadmaw, with plain trample, deals none.
     */
    @Test
    void aDreadmawInThrastasPlaceDealsNothingOnceThePlaneswalkerIsGone() throws Throwable {
        List<String> records = goneWalkerTurn("Thrasta, Tempest's Roar", "702.19e");

        assertTrue(hasCombat(records, true) && hasCombat(records, false), "both games recorded the step");
        List<Map<String, Object>> live = combatDamageToB(records, false);
        assertEquals(1, live.size(), "Thrasta's damage reaches B live: " + live);
        assertEquals(7, ((Number) params(live.get(0)).get("amount")).intValue());
        assertEquals(List.of(), combatDamageToB(records, true), "the Dreadmaw's reaches nobody");
    }

    /**
     * {@code replace_card}, 702.19f: Craw Wurm attacks a planeswalker that
     * has left combat and deals no combat damage; the fork's Thrasta,
     * Tempest's Roar deals its 7 to B.
     */
    @Test
    void thrastaInTheWurmsPlaceHitsBOnceThePlaneswalkerIsGone() throws Throwable {
        List<String> records = goneWalkerTurn("Craw Wurm", "702.19f");

        assertTrue(hasCombat(records, true) && hasCombat(records, false), "both games recorded the step");
        assertEquals(List.of(), combatDamageToB(records, false), "the Wurm's damage reaches nobody live");
        List<Map<String, Object>> fork = combatDamageToB(records, true);
        assertEquals(1, fork.size(), "Thrasta's reaches B in the fork: " + fork);
        assertEquals(7, ((Number) params(fork.get(0)).get("amount")).intValue());
    }

    // ── inverted variants (DF-A item 3) ─────────────────────────────────

    /** The fork's records among {@code records} holding an event of {@code type}, as kind/moment. */
    private static List<String> holding(List<String> records, String type) {
        List<String> found = new ArrayList<>();
        for (String record : records) {
            if (RecordLines.eventTypes(record).contains(type)) {
                found.add(parse(record).get("kind") + "/" + parse(record).get("moment"));
            }
        }
        return found;
    }

    /**
     * {@code add_permanent}, 614.17d, inverted: Zombify in hand and Craw Wurm
     * in A's graveyard, no Grafdigger's Cage; the fork adds the Cage. Forge's
     * AI reads the Cage and, in the games played, did not cast Zombify under
     * it, so the fork writes no prevented move until its variant forces the
     * cast (FR-010b); the case checks the fork's board alone.
     */
    @Test
    void theForksBoardHoldsTheCage() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Swamp", 4);
        TestBoard.inHand(game, active, "Zombify");
        TestBoard.inGraveyard(game, active, "Craw Wurm");
        permanent(game, game.getPlayers().get(1), "Hill Giant");

        List<String> all = played(game, PhaseType.MAIN1, "614.17d", Decisions.EVERY);

        List<String> fork = forkOf(all, "614.17d", "cage");
        assertTrue(fork.stream().anyMatch(record -> record.contains("\"name\":\"Grafdigger's Cage\"")),
                "the fork's board holds the Cage");
    }

    /**
     * {@code move_card}, 702.124j, inverted: Pir, Imaginative Rascal in hand
     * over a library of Forests. The fork's top card is Toothy, Imaginary
     * Friend, which Pir's partner-with trigger finds, if Pir is cast.
     */
    @Test
    void thePartnerOnTopIsFound() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Forest", 3);
        TestBoard.inHand(game, active, "Pir, Imaginative Rascal");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "702.124j", "named_card_in_library");

        assertTrue(fork.stream().anyMatch(record -> "resolution".equals(parse(record).get("kind"))
                        && record.contains("\"from_zone\":\"library\",\"to_zone\":\"hand\"")),
                "the partner-with trigger finds Toothy in the fork");
    }

    /**
     * {@code add_permanent}, 702.131a, inverted: nine permanents and Secrets
     * of the Golden City in hand. The fork adds a tapped Island; the spell
     * resolving gives the city's blessing, if it is cast.
     */
    @Test
    void aTenthPermanentGivesTheBlessing() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Island", 8);
        permanent(game, active, "Grizzly Bears");
        TestBoard.inHand(game, active, "Secrets of the Golden City");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "702.131a", "ten_permanents");

        assertTrue(holding(fork, "state_flag_change").contains("resolution/resolution"),
                "the spell's resolution gives the blessing in the fork");
    }

    /**
     * {@code counters}, 104.3d, inverted: B at 8 poison counters and Ichor
     * Rats in A's hand. The fork's B has 9, so the Rats' enters trigger, if
     * they are cast, gives B its tenth and B loses.
     */
    @Test
    void aNinthPoisonCounterLetsTheNextOneKill() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Swamp", 3);
        TestBoard.inHand(game, active, "Ichor Rats");
        game.getPlayers().get(1).setCounters(CounterEnumType.POISON, 8, null, false);

        List<String> fork = forkOf(game, PhaseType.MAIN1, "104.3d", "nine");

        assertTrue(holding(fork, "player_lost").contains("resolution/resolution"),
                "B's tenth poison counter loses it the game in the fork");
    }

    /**
     * {@code replace_card}, 701.42a, 712.4a and 712.14c, inverted: Graf Rats
     * beside Hill Giant as the first main phase begins. The fork's Giant is
     * Midnight Scavengers, so at the beginning of combat Graf Rats' trigger
     * melds the two into Chittering Host, with no seat's choice.
     */
    @Test
    void thePartnerInTheGiantsPlaceMelds() throws Throwable {
        for (String rule : List.of("701.42a", "712.4a", "712.14c")) {
            Game game = bare(PhaseType.MAIN1);
            Player active = game.getPlayers().get(0);
            permanent(game, active, "Graf Rats");
            permanent(game, active, "Hill Giant");

            List<String> fork = forkOf(game, PhaseType.MAIN1, rule, "meld_pair");

            assertFalse(fork.stream().filter(record -> record.contains("\"to_zone\":\"exile\"")).toList().isEmpty(),
                    rule + ": the pair is exiled to meld");
        }
    }
}
