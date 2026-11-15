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
 * fact's record in the fork. The forced-play cases (FR-010b) find the play
 * their variant forces made, and what it did. Run with {@code -Ddirected.records.dump=DIR}, every
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
     * {@code replace_card}, 702.1c: Concerted Effort beside White Knight and
     * Grizzly Bears as the upkeep begins. The live trigger gives the Bears the
     * Knight's first strike and protection from black; the fork's Hill Giant
     * in the Knight's place has no listed keyword, so the trigger resolves
     * and grants nothing.
     */
    @Test
    void aGiantInTheKnightsPlaceLeavesConcertedEffortNothingToShare() throws Throwable {
        Game game = bare(PhaseType.UNTAP);
        Player active = game.getPlayers().get(0);
        Card effort = permanent(game, active, "Concerted Effort");
        permanent(game, active, "White Knight");
        Card bears = permanent(game, active, "Grizzly Bears");

        // The live game reaches the upkeep itself, so its trigger fires there too.
        List<String> all = played(game, null, "702.1c", Decisions.SAMPLED);

        List<String> live = all.stream().filter(record -> !Boolean.TRUE.equals(parse(record).get("fork"))).toList();
        assertFalse(eventsOf(live, "keyword_change", ref(bears)).isEmpty(),
                "the live Bears share the Knight's keywords");
        List<String> fork = forkOf(all, "702.1c", "no_keywords");
        List<String> resolved = resolutionsOf(fork, effort);
        assertFalse(resolved.isEmpty(), "Concerted Effort's trigger resolves in the fork");
        assertTrue(resolved.stream().noneMatch(record -> RecordLines.eventTypes(record).contains("keyword_change")),
                "and grants no keyword there");
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

    /** The fired trigger records among {@code records} whose trigger's host is {@code host}. */
    private static List<String> fired(List<String> records, Card host) {
        List<String> found = new ArrayList<>();
        for (String record : records) {
            Map<String, Object> parsed = parse(record);
            Object source = RecordLines.asMap(RecordLines.asMap(parsed.get("state")).get("refs")).get("source");
            if ("trigger".equals(parsed.get("kind")) && ref(host).equals(source)
                    && Boolean.TRUE.equals(RecordLines.payload(record).get("fired"))) {
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

    /** The Grizzly Bears attacking B beside the attacker of {@link #goneWalkerTurn}. */
    private String bearsRef;

    /**
     * Every record of a live turn whose {@code attacker} attacks B's Jace
     * Beleren unblocked beside a Grizzly Bears attacking B, Jace having left
     * the battlefield before the damage step (removed by whatever removed it),
     * so the attacker attacks Forge's placeholder (506.4c). The live game
     * leaves the declare-blockers step and begins the damage step itself,
     * {@code rule}'s directed fork taken at that boundary.
     */
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
     * {@code replace_card}, 702.19c, inverted: Colossal Dreadmaw attacks B's
     * Jace Beleren (3 loyalty) unblocked, and all its damage stays on Jace;
     * the fork's Thrasta, Tempest's Roar, in its place, has trample over
     * planeswalkers, and its damage beyond Jace's loyalty reaches B.
     */
    @Test
    void thrastaInTheDreadmawsPlaceTramplesOverThePlaneswalker() throws Throwable {
        Game game = board(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player active = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        Card dreadmaw = permanent(game, active, "Colossal Dreadmaw");
        Card bears = game.getCardsIn(forge.game.zone.ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElseThrow();
        bearsRef = ref(bears);
        Card jace = permanent(game, defending, "Jace Beleren");
        jace.setCounters(CounterEnumType.LOYALTY, 3);
        Combat combat = new Combat(active);
        combat.addAttacker(dreadmaw, jace);
        combat.setBlocked(dreadmaw, false);
        game.getPhaseHandler().setCombat(combat);

        List<String> records = played(game, null, "702.19c", Decisions.SAMPLED);

        assertTrue(hasCombat(records, true) && hasCombat(records, false), "both games recorded the step");
        assertEquals(List.of(), combatDamageToB(records, false), "the Dreadmaw's damage stays on Jace live");
        List<Map<String, Object>> fork = combatDamageToB(records, true);
        assertEquals(1, fork.size(), "Thrasta's excess reaches B in the fork: " + fork);
        assertEquals(4, ((Number) params(fork.get(0)).get("amount")).intValue());
    }

    /**
     * {@code replace_card}, 702.19f: Craw Wurm attacks a planeswalker that
     * has left combat and deals no combat damage; the fork's Thrasta,
     * Tempest's Roar deals its 7 to B. The same fork is 702.19e's
     * {@code fires} side (an attacker with trample over planeswalkers whose
     * planeswalker left combat), which is why 702.19e has no fork of its own.
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

    /**
     * {@code replace_card}, 701.44d, inverted: Hakbal of the Surging Soul is
     * A's only Merfolk as its first main phase begins, beside A's Grizzly
     * Bears. Live, Hakbal's beginning-of-combat trigger has Hakbal alone
     * explore and reveal one card; the fork's Merfolk Branchwalker in the
     * Bears' place explores too, so the trigger's resolution reveals two.
     */
    @Test
    void aBranchwalkerInTheBearsPlaceExploresBesideHakbal() throws Throwable {
        Game game = board(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        permanent(game, active, "Hakbal of the Surging Soul");

        List<String> records = played(game, PhaseType.MAIN1, "701.44d", Decisions.SAMPLED);

        assertEquals(List.of(1), revealsOfExplores(records, false), "Hakbal explores alone live");
        assertEquals(List.of(2), revealsOfExplores(records, true), "two explore in the fork");
    }

    /**
     * Per resolution revealing cards among the fork's or the live game's
     * records (on this board only Hakbal's trigger reveals), the cards it
     * revealed.
     */
    private static List<Integer> revealsOfExplores(List<String> records, boolean fork) {
        List<Integer> found = new ArrayList<>();
        for (String record : records) {
            Map<String, Object> parsed = parse(record);
            if (Boolean.TRUE.equals(parsed.get("fork")) != fork || !"resolution".equals(parsed.get("moment"))) {
                continue;
            }
            List<String> types = RecordLines.eventTypes(record);
            if (types.contains("card_revealed")) {
                found.add((int) types.stream().filter("card_revealed"::equals).count());
            }
        }
        return found;
    }

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
     * in A's graveyard, no Grafdigger's Cage; the fork adds the Cage and
     * forces the cast of Zombify (FR-010b), which Forge's AI does not make
     * under the Cage: the Wurm's move to the battlefield is prevented.
     */
    @Test
    void theCagePreventsTheForcedReanimation() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Swamp", 4);
        TestBoard.inHand(game, active, "Zombify");
        TestBoard.inGraveyard(game, active, "Craw Wurm");
        permanent(game, game.getPlayers().get(1), "Hill Giant");

        List<String> all = played(game, PhaseType.MAIN1, "614.17d", Decisions.EVERY);

        List<String> fork = forkOf(all, "614.17d", "cage");
        assertTrue(fork.stream().anyMatch(record -> "rewrite".equals(parse(record).get("kind"))
                        && record.contains("\"result\":\"prevented\"")),
                "the move to the battlefield is prevented in the fork");
    }

    /**
     * {@code move_card}, 702.124j, inverted: Pir, Imaginative Rascal in hand
     * over a library of Forests. The fork's top card is Toothy, Imaginary
     * Friend, which Pir's partner-with trigger finds, Pir's cast forced.
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
     * of the Golden City in hand. The fork adds a tapped Island and forces
     * the cast; the spell resolving gives the city's blessing.
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
     * Rats in A's hand. The fork's B has 9 and A is made to cast the Rats,
     * whose enters trigger gives B its tenth: B loses.
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

    // ── forced plays (FR-010b) ──────────────────────────────────────────
    //
    // Each case's variant forces the play its point follows, which Forge's
    // AI makes rarely or never on the fork's board; the fork's records show
    // the play made and what it did.

    /** A started two-seat game in {@code step} of the first seat's third turn, both libraries empty. */
    private static Game empty(PhaseType step) {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        game.getPhaseHandler().devModeSet(step, game.getPlayers().get(0), 3);
        return game;
    }

    /** {@code names} at the bottom of {@code owner}'s library in order. */
    private static List<Card> library(Game game, Player owner, String... names) {
        List<Card> cards = new ArrayList<>();
        for (String name : names) {
            cards.add(game.getAction().moveToLibrary(TestBoard.card(game, owner, name), -1, null, null));
        }
        return cards;
    }

    /** The fork's resolution records whose source is {@code card}. */
    private static List<String> resolutionsOf(List<String> fork, Card card) {
        List<String> found = new ArrayList<>();
        for (String record : fork) {
            Map<String, Object> parsed = parse(record);
            Object source = RecordLines.asMap(RecordLines.asMap(parsed.get("state")).get("refs")).get("source");
            if ("resolution".equals(parsed.get("kind")) && ref(card).equals(source)) {
                found.add(record);
            }
        }
        return found;
    }

    /**
     * {@code add_permanent}, 111.10v, inverted: a Mutagen token beside
     * Crustacean Commando, A's one Island tapped. The fork adds an untapped
     * Island and forces the token's activation, which puts its counter.
     */
    @Test
    void anIslandLetsTheMutagenTokenPutItsCounter() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        permanent(game, active, "Crustacean Commando");
        Card token = game.getAction().moveToPlay(forge.game.card.CardFactory.getCard(
                forge.StaticData.instance().getAllTokens().getToken("c_a_mutagen_sac"), active, game),
                active, null, null);
        token.setGameTimestamp(game.getNextTimestamp());
        token.setSickness(false);
        permanent(game, active, "Island").setTapped(true);

        List<String> fork = forkOf(game, PhaseType.MAIN1, "111.10v", "mana_available");

        assertTrue(resolutionsOf(fork, token).stream()
                .anyMatch(record -> RecordLines.eventTypes(record).contains("counter_change")),
                "the token's ability puts its counter in the fork");
    }

    /**
     * {@code add_permanent}, 119.7, inverted: Tree of Redemption, A at 5
     * life. The fork adds Leyline of Punishment and A is made to activate
     * the Tree: the exchange, which would raise A's life, does not happen.
     */
    @Test
    void anExchangeAPlayerWhoCannotGainLifeWouldGainFromChangesNothing() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        Card tree = permanent(game, active, "Tree of Redemption");
        active.setLife(5, null);

        List<String> fork = forkOf(game, PhaseType.MAIN1, "119.7", "leyline");

        List<String> resolved = resolutionsOf(fork, tree);
        assertFalse(resolved.isEmpty(), "the exchange resolves in the fork");
        assertTrue(resolved.stream().noneMatch(record -> RecordLines.eventTypes(record).contains("life_change")),
                "and changes no life");
    }

    /**
     * {@code life}, 701.12g: Tree of Perdition, B at 7. The fork's B has 20
     * and A is made to activate the Tree at B: the exchange lowers B's life.
     */
    @Test
    void anExchangeThatLowersTheOpponentsLifeHappens() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        Card tree = permanent(game, active, "Tree of Perdition");
        game.getPlayers().get(1).setLife(7, null);

        List<String> fork = forkOf(game, PhaseType.MAIN1, "701.12g", "opponent_above");

        assertTrue(resolutionsOf(fork, tree).stream()
                        .anyMatch(record -> RecordLines.eventTypes(record).contains("life_change")),
                "the exchange changes the life of B in the fork");
    }

    /**
     * {@code add_permanent}, 701.39a, inverted: Cached Defenses over three
     * Forests, A controlling no creature. The fork adds Grizzly Bears and A
     * is made to cast it: the bolster puts its counters on the Bears.
     */
    @Test
    void aBolsterWithACreaturePutsItsCounters() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Forest", 3);
        Card defenses = TestBoard.inHand(game, active, "Cached Defenses");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "701.39a", "has_creature");

        assertTrue(resolutionsOf(fork, defenses).stream()
                        .anyMatch(record -> RecordLines.eventTypes(record).contains("counter_change")),
                "Cached Defenses puts its counters in the fork");
    }

    /**
     * {@code move_card}, 702.71a: Fleshwrither over three Swamps, Hill Giant
     * the library's one creature of mana value 4. The fork's is Craw Wurm,
     * and its A is made to transfigure: Fleshwrither is sacrificed and
     * nothing enters.
     */
    @Test
    void aTransfigureFindingNothingPutsNothingOntoTheBattlefield() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Swamp", 3);
        Card fleshwrither = permanent(game, active, "Fleshwrither");
        library(game, active, "Hill Giant");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "702.71a", "no_matching_value");

        assertEquals(List.of(ref(fleshwrither)), ForcedPlayTest.costCards(fork, "sacrificed"),
                "Fleshwrither is sacrificed for the cost");
        assertTrue(fork.stream().noneMatch(record -> record.contains("\"from_zone\":\"library\",\"to_zone\":\"battlefield\"")),
                "nothing enters from the library");
    }

    /**
     * {@code move_card}, 702.53a: Dimir Infiltrator over an Island, a Swamp
     * and a Forest, Grizzly Bears the library's one card of mana value 2. The
     * fork's is Centaur Courser, and its A is made to transmute: the
     * Infiltrator is discarded and the search finds nothing.
     */
    @Test
    void aTransmuteFindingNothingPutsNothingIntoTheHand() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        for (String land : List.of("Island", "Swamp", "Forest")) {
            permanent(game, active, land);
        }
        Card infiltrator = TestBoard.inHand(game, active, "Dimir Infiltrator");
        library(game, active, "Grizzly Bears");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "702.53a", "no_match");

        assertEquals(List.of(ref(infiltrator)), ForcedPlayTest.costCards(fork, "discarded"),
                "the Infiltrator is discarded for the cost");
        assertTrue(fork.stream().noneMatch(record -> record.contains("\"from_zone\":\"library\",\"to_zone\":\"hand\"")),
                "the search finds nothing");
    }

    /**
     * {@code add_permanent}, 118.12b, inverted, forcing a "you may" answer:
     * A's upkeep begins with a basic land in its library and no Elfhame
     * Sanctuary. Live, A draws in its draw step; the fork's Sanctuary
     * triggers, the forced yes searches, and the draw step is skipped.
     */
    @Test
    void anAddedSanctuarysForcedSearchSkipsTheDraw() throws Throwable {
        Game game = empty(PhaseType.UPKEEP);
        Player active = game.getPlayers().get(0);
        library(game, active, "Shock", "Forest", "Shock", "Shock", "Shock", "Shock");
        library(game, game.getPlayers().get(1), "Shock", "Shock", "Shock");

        List<String> fork = forkOf(game, PhaseType.UPKEEP, "118.12b", "sanctuary_added");

        assertTrue(holding(fork, "phase_skipped").contains("resolution/resolution"),
                "the search's resolution skips the draw step in the fork");
    }

    /**
     * {@code counters}, 205.1a, forcing a cast: A holds Song of the Dryads
     * and three Forests, and B controls Grizzly Bears, as A's first main
     * phase begins. The fork's Bears has a +1/+1 counter, and the forced Song
     * makes it a colorless Forest land: the continuous record of the Song's
     * static removes every card type it had.
     */
    @Test
    void aForcedSongOfTheDryadsSetsTheBearsType() throws Throwable {
        Game game = empty(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Forest", 3);
        Card bears = permanent(game, game.getPlayers().get(1), "Grizzly Bears");
        TestBoard.inHand(game, active, "Song of the Dryads");
        library(game, active, "Forest", "Forest", "Forest");
        library(game, game.getPlayers().get(1), "Forest", "Forest", "Forest");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "205.1a", "counter-kept");

        assertTrue(fork.stream().anyMatch(record -> "continuous".equals(parse(record).get("kind"))
                        && record.contains("-all-card-types") && record.contains(ref(bears))),
                "the Song's static sets the Bears' card type in the fork: " + holding(fork, "attached"));
    }

    /**
     * {@code add_permanent}, 111.10p, forcing a cast: A holds Ellivere of
     * the Wild Court and four lands, beside its Grizzly Bears, as its first
     * main phase begins. The fork's A also controls Seal of Fire; the forced
     * Ellivere's enters trigger attaches a Virtuous Role to the Bears, whose
     * static boosts it.
     */
    @Test
    void aForcedElliveresRoleBoostsTheBears() throws Throwable {
        Game game = empty(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Forest", 3);
        lands(game, active, "Plains", 1);
        permanent(game, active, "Grizzly Bears");
        TestBoard.inHand(game, active, "Ellivere of the Wild Court");
        library(game, active, "Forest", "Forest", "Forest");
        library(game, game.getPlayers().get(1), "Forest", "Forest", "Forest");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "111.10p", "two_enchantments");

        assertTrue(holding(fork, "token_created").contains("resolution/resolution"),
                "Ellivere's trigger creates the Role in the fork");
        assertTrue(fork.stream().anyMatch(record -> "continuous".equals(parse(record).get("kind"))
                && record.contains("Virtuous")), "the Role's static is recorded in the fork");
    }

    /**
     * {@code add_permanent}, 701.23f, inverted, forcing an activation: A
     * controls Archivist of Oghma and B an untapped Evolving Wilds as A's
     * first main phase begins. The fork's A also controls Aven Mindcensor;
     * B is made to activate the Wilds, searches the top four cards of its
     * library, and Archivist's trigger fires.
     */
    @Test
    void aSearchLimitedByTheMindcensorStillFiresArchivist() throws Throwable {
        Game game = empty(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        Player searcher = game.getPlayers().get(1);
        Card archivist = permanent(game, active, "Archivist of Oghma");
        permanent(game, searcher, "Evolving Wilds");
        library(game, active, "Island", "Island", "Island");
        library(game, searcher, "Grizzly Bears", "Grizzly Bears", "Grizzly Bears", "Grizzly Bears", "Forest");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "701.23f", "mindcensor_added");

        assertFalse(fired(fork, archivist).isEmpty(), "Archivist's trigger fires on the limited search");
    }

    /**
     * {@code move_card}, 702.133a, forcing a cast: Radical Idea in A's
     * graveyard, two Islands, as A's first main phase begins. The fork's
     * Radical Idea is in A's hand and is cast from there; as it resolves it
     * goes to the graveyard, not to exile.
     */
    @Test
    void aJumpStartCardCastFromHandGoesToTheGraveyard() throws Throwable {
        Game game = empty(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Island", 2);
        Card idea = TestBoard.card(game, active, "Radical Idea");
        game.getAction().moveToGraveyard(idea, null);
        library(game, active, "Plains", "Plains", "Plains", "Plains", "Plains", "Plains");
        library(game, game.getPlayers().get(1), "Forest", "Forest", "Forest");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "702.133a", "cast_from_hand");

        List<String> moved = new ArrayList<>();
        for (String record : fork) {
            for (Map<String, Object> event : events(record)) {
                if ("zone_change".equals(event.get("type")) && "stack".equals(params(event).get("from_zone"))) {
                    moved.add(String.valueOf(params(event).get("to_zone")));
                }
            }
        }
        assertTrue(moved.contains("graveyard") && !moved.contains("exile"),
                "the spell cast from hand leaves the stack to the graveyard: " + moved);
    }

    /**
     * {@code remove_permanent}, 607.2k: Changeling Hero over five Plains,
     * Grizzly Bears A's only other creature. The fork's A controls none and
     * is made to cast the Hero: its champion trigger exiles nothing and it is
     * sacrificed.
     */
    @Test
    void aChampionWithNothingToChampionIsSacrificed() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Plains", 5);
        permanent(game, active, "Grizzly Bears");
        Card hero = TestBoard.inHand(game, active, "Changeling Hero");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "607.2k", "no_creature_to_champion");

        assertTrue(eventsOf(fork, "zone_change", ref(hero)).stream()
                        .anyMatch(move -> "graveyard".equals(params(move).get("to_zone"))),
                "the Hero is sacrificed in the fork");
    }

    /** Arcanum Wings on Grizzly Bears over three Islands, {@code inHand} in the hand of A. */
    private static Game wingsBoard(String inHand) {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        Card bears = permanent(game, active, "Grizzly Bears");
        permanent(game, active, "Arcanum Wings").attachToEntity(bears, null);
        lands(game, active, "Island", 3);
        TestBoard.inHand(game, active, inHand);
        return game;
    }

    /** The fork's resolutions of the Aura swap of the Arcanum Wings on the battlefield of {@code game}. */
    private static List<String> swaps(Game game, List<String> fork) {
        Card wings = game.getCardsIn(forge.game.zone.ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Arcanum Wings")).findFirst().orElseThrow();
        return resolutionsOf(fork, wings);
    }

    /**
     * {@code move_card}, 701.12e and 702.65a, inverted: Arcanum Wings on
     * Grizzly Bears, no Aura in hand. The fork's hand holds Holy Strength in
     * the hand's card's place, and its A is made to activate the swap: Holy
     * Strength moves from the hand and is attached to the Bears.
     */
    @Test
    void aSwapWithAnAuraInHandExchangesAndAttaches() throws Throwable {
        for (String rule : List.of("701.12e", "702.65a")) {
            Game game = wingsBoard(rule.equals("701.12e") ? "Giant Growth" : "Grizzly Bears");

            List<String> fork = forkOf(game, PhaseType.MAIN1, rule, "aura_in_hand");

            List<String> swapped = swaps(game, fork);
            assertTrue(swapped.stream().anyMatch(record -> RecordLines.eventTypes(record).contains("attached")),
                    rule + ": the swap attaches Holy Strength in the fork");
            assertTrue(swapped.stream().anyMatch(record -> RecordLines.eventTypes(record).contains("zone_change")),
                    rule + ": and moves it from the hand");
        }
    }

    /**
     * {@code move_card}, 702.65b: Arcanum Wings on Grizzly Bears, Wild Growth
     * (enchant land) in hand. The fork's is Holy Strength, and its A is made
     * to activate the swap: the exchange happens.
     */
    @Test
    void aSwapWithAnAuraThatFitsExchanges() throws Throwable {
        Game game = wingsBoard("Wild Growth");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "702.65b", "can_enchant");

        assertTrue(swaps(game, fork).stream().anyMatch(record -> RecordLines.eventTypes(record).contains("zone_change")),
                "the swap moves the Auras in the fork");
    }

    /**
     * {@code pt}, 704.5f: Disfigure in the hand of A over a Swamp, the Drudge
     * Skeletons of B a 1/1. The fork's Skeletons is a 1/3 and A is made to
     * cast Disfigure at it: it survives the -2/-2.
     */
    @Test
    void aTougherCreatureSurvivesTheForcedDisfigure() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        permanent(game, active, "Swamp");
        Card disfigure = TestBoard.inHand(game, active, "Disfigure");
        Card skeletons = permanent(game, game.getPlayers().get(1), "Drudge Skeletons");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "704.5f", "tougher");

        assertFalse(resolutionsOf(fork, disfigure).isEmpty(), "Disfigure resolves in the fork");
        assertTrue(eventsOf(fork, "zone_change", ref(skeletons)).isEmpty(), "the Skeletons survive");
    }

    /**
     * 701.24e: the Cosi's Trickster of B; the Demonic Tutor of A over two
     * Swamps and a two-card library. The fork's library holds one card and A
     * is made to cast the Tutor: the library shuffled after the search is
     * empty, and the Trickster's trigger is evaluated.
     */
    @Test
    void aTutorShufflesTheEmptiedLibrary() throws Throwable {
        Game game = empty(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        permanent(game, game.getPlayers().get(1), "Cosi's Trickster");
        lands(game, active, "Swamp", 2);
        Card tutor = TestBoard.inHand(game, active, "Demonic Tutor");
        library(game, active, "Forest", "Island");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "701.24e", "zero-cards");

        assertFalse(resolutionsOf(fork, tutor).isEmpty(), "the Tutor resolves in the fork");
    }

    /**
     * 701.44b: Merfolk Branchwalker over two Forests beside Wildgrowth
     * Walker, the library of A empty and a Forest in its graveyard. The
     * fork's Forest is the library's one card and A is made to cast the
     * Branchwalker.
     */
    @Test
    void aBranchwalkerExploresOntoTheOneCard() throws Throwable {
        Game game = empty(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Forest", 2);
        permanent(game, active, "Wildgrowth Walker");
        Card branchwalker = TestBoard.inHand(game, active, "Merfolk Branchwalker");
        TestBoard.inGraveyard(game, active, "Forest");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "701.44b", "land_on_top");

        assertFalse(resolutionsOf(fork, branchwalker).isEmpty(), "the Branchwalker is cast in the fork");
    }

    /**
     * 701.50f: Echo Inspector and one other card in the hand of A under the
     * Spirit of the Labyrinth of B, A having drawn, beside Glorious Purpose.
     * The fork's other card is in the library and A is made to cast the
     * Inspector, which connives with nothing to discard.
     */
    @Test
    void anInspectorConnivesWithNothingToDiscard() throws Throwable {
        Game game = empty(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Island", 4);
        permanent(game, active, "Glorious Purpose");
        permanent(game, game.getPlayers().get(1), "Spirit of the Labyrinth");
        Card inspector = TestBoard.inHand(game, active, "Echo Inspector");
        library(game, active, "Forest", "Island", "Island");
        active.drawCard();

        List<String> fork = forkOf(game, PhaseType.MAIN1, "701.50f", "nothing_to_discard");

        assertFalse(resolutionsOf(fork, inspector).isEmpty(), "the Inspector is cast in the fork");
    }

    /**
     * 701.57b: Geological Appraiser over four Mountains beside Val, Marooned
     * Surveyor, no nonland card of mana value 3 or less in the library. The
     * fork's library holds Grizzly Bears and A is made to cast the Appraiser.
     */
    @Test
    void anAppraiserDiscoversTheBears() throws Throwable {
        Game game = empty(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Mountain", 4);
        permanent(game, active, "Val, Marooned Surveyor");
        Card appraiser = TestBoard.inHand(game, active, "Geological Appraiser");
        library(game, active, "Forest", "Hill Giant", "Island", "Island", "Island");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "701.57b", "card_found");

        assertFalse(resolutionsOf(fork, appraiser).isEmpty(), "the Appraiser is cast in the fork");
    }

    /**
     * {@code replace_card}, 702.103e, inverted: Hopeful Eidolon over four
     * Plains, Grizzly Bears the battlefield's one creature. The fork's is
     * Skulking Ghost and A is made to cast the Eidolon bestowed, its target
     * drawn, the Ghost: the Ghost is sacrificed as it becomes the target, and
     * the Eidolon resolves as a creature, attached to nothing.
     */
    @Test
    void aBestowedEidolonWhoseTargetIsSacrificedEntersAsACreature() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Plains", 4);
        permanent(game, active, "Grizzly Bears");
        Card eidolon = TestBoard.inHand(game, active, "Hopeful Eidolon");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "702.103e", "skulking_ghost");

        List<String> resolved = resolutionsOf(fork, eidolon);
        assertTrue(resolved.stream().anyMatch(record -> RecordLines.eventTypes(record).contains("zone_change")),
                "the Eidolon enters in the fork");
        assertTrue(resolved.stream().noneMatch(record -> RecordLines.eventTypes(record).contains("attached")),
                "attached to nothing");
    }

    /**
     * {@code move_card}, 702.170c, inverted: Kellan Joins Up over a Forest, a
     * Plains and an Island, Hill Giant the hand's other card. The fork's is
     * Grizzly Bears and A is made to cast Kellan: its trigger exiles the
     * Bears, which become plotted.
     */
    @Test
    void kellanPlotsTheBears() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        for (String land : List.of("Forest", "Plains", "Island")) {
            permanent(game, active, land);
        }
        Card kellan = TestBoard.inHand(game, active, "Kellan Joins Up");
        TestBoard.inHand(game, active, "Hill Giant");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "702.170c", "small_card_in_hand");

        assertTrue(resolutionsOf(fork, kellan).stream().anyMatch(record -> events(record).stream()
                        .anyMatch(event -> "zone_change".equals(event.get("type"))
                                && "exile".equalsIgnoreCase(String.valueOf(params(event).get("to_zone"))))),
                "Kellan's trigger exiles a card in the fork");
    }
}
