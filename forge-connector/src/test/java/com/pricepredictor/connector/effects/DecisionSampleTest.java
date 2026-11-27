package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;

import static com.pricepredictor.connector.effects.TestCaps.caps;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Two-stage decision sampling (FR-033b): the bucket each kind of candidate
 * falls in, and the rate each bucket is written at.
 */
@ExtendWith(ForgeExtension.class)
class DecisionSampleTest {

    @TempDir
    Path tempDir;

    private static final Method ON_PRIORITY_SETTLED =
            HookShapes.method(HookShapes.PrioritySettled.class, "onPrioritySettled");

    private static Game mainPhase() {
        return TestBoard.twoPlayerGame(PhaseType.MAIN1);
    }

    private static Player first(Game game) {
        return game.getPlayers().get(0);
    }

    private static Player second(Game game) {
        return game.getPlayers().get(1);
    }

    /** Two Forests in play and Grizzly Bears in hand, all {@code player}'s. */
    private static Card twoForestsAndBearsInHand(Game game, Player player) {
        TestBoard.inPlay(game, player, "Forest");
        TestBoard.inPlay(game, player, "Forest");
        return TestBoard.inHand(game, player, "Grizzly Bears");
    }

    private static void assertChecksAndSample(
            DecisionCandidate read, List<PlayabilityCheck> expected, DecisionSample sample) {
        assertEquals(expected, checks(read));
        assertEquals(sample, read.sample());
    }

    private static void assertCheckAndSample(
            DecisionCandidate read, PlayabilityCheck check, DecisionSample sample) {
        assertTrue(checks(read).contains(check), checks(read).toString());
        assertEquals(sample, read.sample());
    }

    private static void assertPlayableAndSample(DecisionCandidate read, DecisionSample sample) {
        assertTrue(read.verdict().playable());
        assertEquals(sample, read.sample());
    }

    /** The candidates of {@code player} on {@code card}. */
    private static List<PriorityCandidates.Candidate> on(Game game, Player player, Card card) {
        return PriorityCandidates.candidatesOf(game, player).stream()
                .filter(c -> c.ability().getHostCard() == card).toList();
    }

    /** The one candidate of {@code player} on {@code card} that {@code which} picks. */
    private static DecisionCandidate read(
            Game game, Player player, Card card, Predicate<SpellAbility> which) {
        List<PriorityCandidates.Candidate> picked = on(game, player, card).stream()
                .filter(c -> which.test(c.ability())).toList();
        assertEquals(1, picked.size(), "candidates on " + card + ": " + on(game, player, card));
        return picked.get(0).read().orElseThrow();
    }

    private static DecisionCandidate onlyRead(Game game, Player player, Card card) {
        return read(game, player, card, sa -> true);
    }

    private static List<PlayabilityCheck> checks(DecisionCandidate read) {
        return read.verdict().failedChecks().stream()
                .map(PlayabilityVerdict.FailedCheck::check).toList();
    }

    // ── the buckets ─────────────────────────────────────────────────────

    /** A permanent's own cast, on the battlefield: the bulk of every stop. */
    @Test
    void aPermanentsOwnCastIsAZoneCandidate() {
        Game game = mainPhase();
        Player a = first(game);
        Card bears = TestBoard.inPlay(game, a, "Grizzly Bears");

        DecisionCandidate read = onlyRead(game, a, bears);

        assertCheckAndSample(read, PlayabilityCheck.ZONE, DecisionSample.ZONE);
    }

    /** A creature in hand without the mana: its usual zone, failing only cost. */
    @Test
    void aSpellInHandFailingOnlyCostIsPlain() {
        Game game = mainPhase();
        Player a = first(game);
        Card bears = TestBoard.inHand(game, a, "Grizzly Bears");

        DecisionCandidate read = onlyRead(game, a, bears);

        assertChecksAndSample(read, List.of(PlayabilityCheck.COST), DecisionSample.PLAIN);
    }

    /** A sorcery on the other player's turn: timing is bulk. */
    @Test
    void aSorceryOffTurnIsPlain() {
        Game game = mainPhase();
        Player b = second(game);
        Card divination = TestBoard.inHand(game, b, "Divination");

        DecisionCandidate read = onlyRead(game, b, divination);

        assertCheckAndSample(read, PlayabilityCheck.CASTING_SPEED, DecisionSample.PLAIN);
    }

    /** A playable spell from the hand is plain too. */
    @Test
    void aPlayableSpellInHandIsPlain() {
        Game game = mainPhase();
        Player a = first(game);
        Card bears = twoForestsAndBearsInHand(game, a);

        DecisionCandidate read = onlyRead(game, a, bears);

        assertPlayableAndSample(read, DecisionSample.PLAIN);
    }

    /** A static prohibition is special. */
    @Test
    void aSpellAStaticForbidsIsSpecial() {
        Game game = mainPhase();
        Player a = first(game);
        Player b = second(game);
        TestBoard.inPlay(game, a, "Grand Abolisher");
        TestBoard.inPlay(game, b, "Mountain");
        Card bolt = TestBoard.inHand(game, b, "Lightning Bolt");

        DecisionCandidate read = onlyRead(game, b, bolt);

        assertChecksAndSample(read, List.of(PlayabilityCheck.STATIC_PROHIBITION), DecisionSample.SPECIAL);
    }

    /**
     * Think Twice in the graveyard: its flashback is special (an alternative
     * cost, outside the hand); its ordinary cast fails zone.
     */
    @Test
    void aFlashbackFromTheGraveyardIsSpecialAndItsOrdinaryCastIsAZoneCandidate() {
        Game game = mainPhase();
        Player a = first(game);
        Card thinkTwice = TestBoard.inGraveyard(game, a, "Think Twice");

        DecisionCandidate flashback =
                read(game, a, thinkTwice, sa -> sa.getAlternativeCost() != null);
        DecisionCandidate ordinary =
                read(game, a, thinkTwice, sa -> sa.getAlternativeCost() == null);

        assertFalse(checks(flashback).contains(PlayabilityCheck.ZONE), checks(flashback).toString());
        assertEquals(DecisionSample.SPECIAL, flashback.sample());
        assertTrue(checks(ordinary).contains(PlayabilityCheck.ZONE), checks(ordinary).toString());
        assertEquals(DecisionSample.ZONE, ordinary.sample());
    }

    /**
     * Leyline of Anticipation lets B cast a creature on A's turn: its timing
     * holds through the static alone, which makes it special.
     */
    @Test
    void aSpellWithFlashOnlyFromAStaticIsSpecial() {
        Game game = mainPhase();
        Player b = second(game);
        TestBoard.inPlay(game, b, "Leyline of Anticipation");
        Card bears = twoForestsAndBearsInHand(game, b);

        DecisionCandidate read = onlyRead(game, b, bears);

        assertTrue(read.verdict().timing(), "the Leyline grants flash");
        assertEquals(DecisionSample.SPECIAL, read.sample());
    }

    /** Without the Leyline the same creature is plain: timing fails, which is bulk. */
    @Test
    void theSameSpellWithoutTheStaticIsPlain() {
        Game game = mainPhase();
        Player b = second(game);
        Card bears = twoForestsAndBearsInHand(game, b);

        DecisionCandidate read = onlyRead(game, b, bears);

        assertFalse(read.verdict().timing());
        assertEquals(DecisionSample.PLAIN, read.sample());
    }

    /** Cycling: an activated ability from the hand, outside its usual zone. */
    @Test
    void anActivatedAbilityFromTheHandIsSpecial() {
        Game game = mainPhase();
        Player a = first(game);
        Card angel = TestBoard.inHand(game, a, "Angel of the God-Pharaoh");

        DecisionCandidate cycling = read(game, a, angel, sa -> !sa.isSpell());

        assertFalse(checks(cycling).contains(PlayabilityCheck.ZONE), checks(cycling).toString());
        assertEquals(DecisionSample.SPECIAL, cycling.sample());
    }

    // ── land plays ──────────────────────────────────────────────

    @Test
    void aLandInHandOnItsOwnersTurnIsPlain() {
        Game game = mainPhase();
        Player a = first(game);
        Card forest = TestBoard.inHand(game, a, "Forest");

        DecisionCandidate read = onlyRead(game, a, forest);

        assertTrue(read.landPlay());
        assertPlayableAndSample(read, DecisionSample.PLAIN);
    }

    /** Rule 305.3 is a land's timing: off turn it fails whose_turn and casting_speed, and is bulk. */
    @Test
    void aLandOffTurnIsPlain() {
        Game game = mainPhase();
        Player b = second(game);
        Card forest = TestBoard.inHand(game, b, "Forest");

        DecisionCandidate read = onlyRead(game, b, forest);

        assertChecksAndSample(read, List.of(PlayabilityCheck.CASTING_SPEED, PlayabilityCheck.WHOSE_TURN), DecisionSample.PLAIN);
    }

    /** A second land in one turn fails the land-play limit, which is special. */
    @Test
    void aLandPastTheLimitIsSpecial() {
        Game game = mainPhase();
        Player a = first(game);
        a.playLand(TestBoard.inHand(game, a, "Forest"), null);
        Card second = TestBoard.inHand(game, a, "Forest");

        DecisionCandidate read = onlyRead(game, a, second);

        assertChecksAndSample(read, List.of(PlayabilityCheck.LAND_PLAY_LIMIT), DecisionSample.SPECIAL);
    }

    @Test
    void aLandOnTheBattlefieldIsAZoneCandidate() {
        Game game = mainPhase();
        Player a = first(game);
        Card forest = TestBoard.inPlay(game, a, "Forest");

        DecisionCandidate read = onlyRead(game, a, forest);

        assertTrue(read.landPlay());
        assertChecksAndSample(read, List.of(PlayabilityCheck.ZONE), DecisionSample.ZONE);
    }

    /**
     * A land on the battlefield after the turn's land drop fails the limit
     * beside zone; zone decides, so it stays a zone candidate.
     */
    @Test
    void aLandOnTheBattlefieldPastTheLimitIsStillAZoneCandidate() {
        Game game = mainPhase();
        Player a = first(game);
        Card forest = a.playLand(TestBoard.inHand(game, a, "Forest"), null);

        DecisionCandidate read = onlyRead(game, a, forest);

        assertChecksAndSample(read, List.of(PlayabilityCheck.LAND_PLAY_LIMIT, PlayabilityCheck.ZONE), DecisionSample.ZONE);
    }

    /** Another player's permanent's ability fails the activator check: a zone candidate. */
    @Test
    void anotherPlayersAbilityIsAZoneCandidate() {
        Game game = mainPhase();
        Player a = first(game);
        Player b = second(game);
        Card sorcerer = TestBoard.inPlay(game, b, "Prodigal Sorcerer");

        DecisionCandidate ping = read(game, a, sorcerer, sa -> !sa.isSpell());

        assertTrue(checks(ping).contains(PlayabilityCheck.SCRIPT_RESTRICTION),
                checks(ping).toString());
        assertEquals(DecisionSample.ZONE, ping.sample());
        assertEquals(DecisionSample.PLAIN,
                read(game, b, sorcerer, sa -> !sa.isSpell()).sample(), "its controller's");
    }

    /** A static prohibition stays special beside a failed zone. */
    @Test
    void aStaticProhibitionBesideAFailedZoneIsSpecial() {
        Game game = mainPhase();
        Player a = first(game);
        Player b = second(game);
        TestBoard.inPlay(game, a, "Grand Abolisher");
        Card bears = TestBoard.inPlay(game, b, "Grizzly Bears");

        DecisionCandidate read = onlyRead(game, b, bears);

        assertTrue(checks(read).containsAll(List.of(
                PlayabilityCheck.ZONE, PlayabilityCheck.STATIC_PROHIBITION)), checks(read).toString());
        assertEquals(DecisionSample.SPECIAL, read.sample());
    }

    /** Crucible of Worlds permits a land from the graveyard: a may-play static, special. */
    @Test
    void aLandAStaticLetsBePlayedFromTheGraveyardIsSpecial() {
        Game game = mainPhase();
        Player a = first(game);
        TestBoard.inPlay(game, a, "Crucible of Worlds");
        Card forest = TestBoard.inGraveyard(game, a, "Forest");
        game.getAction().checkStaticAbilities();

        DecisionCandidate permitted = read(game, a, forest, sa -> sa.getMayPlay() != null);
        DecisionCandidate ordinary = read(game, a, forest, sa -> sa.getMayPlay() == null);

        assertTrue(permitted.verdict().playable(), permitted.verdict().toString());
        assertEquals(DecisionSample.SPECIAL, permitted.sample());
        assertEquals(DecisionSample.ZONE, ordinary.sample());
    }

    // ── the rates ───────────────────────────────────────────────────────

    /** How many of {@code trials} candidates of {@code bucket} both stages keep. */
    private int written(CollectionCaps caps, DecisionSample bucket, int trials) throws Throwable {
        Game game = mainPhase();
        int[] kept = {0};
        ShardRun.shardOf(tempDir.resolve(bucket.name() + caps.hashCode()), game, caps,
                (bracket, collectors) -> {
                    PlayabilityChannel channel = collectors.playability();
                    for (int i = 0; i < trials; i++) {
                        if (channel.sampled() && channel.keeps(bucket)) {
                            kept[0]++;
                        }
                    }
                });
        return kept[0];
    }

    /** Within five standard deviations of the binomial mean. */
    private static void assertNear(double rate, int trials, int observed, String what) {
        double mean = rate * trials;
        double sd = Math.sqrt(trials * rate * (1 - rate));
        assertTrue(Math.abs(observed - mean) <= 5 * sd,
                what + ": " + observed + " of " + trials + ", expected about " + mean);
    }

    /**
     * Each bucket is written at its own rate, both stages together, under the
     * game's seeded sampler.
     */
    @Test
    void eachBucketIsWrittenAtItsRateOfAllCandidates() throws Throwable {
        CollectionCaps rates = caps().playabilityRate(0.5).candidateRates(new TestCaps.CandidateRates(0.05, 0.1)).build();
        int trials = 40_000;

        assertNear(0.5, trials, written(rates, DecisionSample.SPECIAL, trials), "special");
        assertNear(0.05, trials, written(rates, DecisionSample.ZONE, trials), "zone");
        assertNear(0.1, trials, written(rates, DecisionSample.PLAIN, trials), "plain");
    }

    /** At the defaults, stage 2 keeps a zone candidate at 1 in 100 and a plain one at 1 in 10. */
    @Test
    void theDefaultsWriteBulkCandidatesAtTheirRates() throws Throwable {
        CollectionCaps defaults = CollectionCaps.defaults();
        int trials = 400_000;

        assertNear(0.0001, trials, written(defaults, DecisionSample.ZONE, trials), "zone");
        assertNear(0.001, trials, written(defaults, DecisionSample.PLAIN, trials), "plain");
        assertNear(0.01, trials, written(defaults, DecisionSample.SPECIAL, trials), "special");
    }

    /**
     * A special candidate is written whenever stage 1 keeps it, and stage 2
     * draws nothing for it: the same seed's stage-1 draws keep exactly the
     * same candidates with and without the stage-2 question asked.
     */
    @Test
    void aSpecialCandidateIsWrittenWheneverStageOneKeepsIt() throws Throwable {
        CollectionCaps rates = caps().playabilityRate(0.3).candidateRates(new TestCaps.CandidateRates(0.0, 0.0)).build();
        int trials = 2_000;
        StringBuilder stageOneOnly = new StringBuilder();
        StringBuilder bothStages = new StringBuilder();
        ShardRun.shardOf(tempDir.resolve("one"), mainPhase(), rates,
                (bracket, collectors) -> {
                    for (int i = 0; i < trials; i++) {
                        stageOneOnly.append(collectors.playability().sampled() ? '1' : '0');
                    }
                });
        ShardRun.shardOf(tempDir.resolve("two"), mainPhase(), rates,
                (bracket, collectors) -> {
                    PlayabilityChannel channel = collectors.playability();
                    for (int i = 0; i < trials; i++) {
                        bothStages.append(channel.sampled() && channel.keeps(DecisionSample.SPECIAL)
                                ? '1' : '0');
                    }
                });

        assertEquals(stageOneOnly.toString(), bothStages.toString());
        assertTrue(stageOneOnly.indexOf("1") >= 0);
    }

    /** A bulk rate of zero writes nothing of its bucket, however often stage 1 keeps it. */
    @Test
    void aBulkRateOfZeroWritesNothing() throws Throwable {
        CollectionCaps rates = caps().playabilityRate(1.0).candidateRates(new TestCaps.CandidateRates(0.0, 0.0)).build();

        assertEquals(0, written(rates, DecisionSample.ZONE, 10_000));
        assertEquals(0, written(rates, DecisionSample.PLAIN, 10_000));
        assertEquals(10_000, written(rates, DecisionSample.SPECIAL, 10_000));
    }

    // ── through a priority stop ─────────────────────────────────────────

    /** The failed checks of every decision record of a stop. */
    @SuppressWarnings("unchecked")
    private List<Set<String>> writtenChecks(String run, Game game, CollectionCaps caps)
            throws Throwable {
        Player a = first(game);
        List<String> records = ShardRun.records(tempDir.resolve(run), game, caps,
                (bracket, collectors) -> collectors.phase().prioritySettledHandler().invoke(
                        null, ON_PRIORITY_SETTLED, new Object[]{game, a}));
        return records.stream()
                .filter(r -> r.contains("\"subkind\":\"decision\""))
                .map(r -> {
                    Map<String, Object> candidate = RecordLines.asMap(((List<Object>)
                            RecordEvents.payload(r).get("candidates")).get(0));
                    return Set.copyOf(((List<Object>) candidate.get("failed_checks")).stream()
                            .map(c -> (String) RecordLines.asMap(c).get("check")).toList());
                })
                .toList();
    }

    /**
     * A board with a candidate of each bucket and no unusual zone: a plain
     * creature in hand, a zone-failing one on the battlefield, and an instant
     * Grand Abolisher forbids.
     */
    private static Game aBoardWithEachBucket() {
        Game game = mainPhase();
        Player a = first(game);
        Player b = second(game);
        TestBoard.inHand(game, a, "Grizzly Bears");
        TestBoard.inPlay(game, a, "Grand Abolisher");
        TestBoard.inHand(game, b, "Lightning Bolt");
        return game;
    }

    private static final Set<String> BULK = Set.of("zone", "cost", "casting_speed");

    /** Special by its checks, on a board where every card a player could activate is its own. */
    private static boolean special(Set<String> checks) {
        return checks.contains("static_prohibition")
                || (!checks.contains("zone") && !BULK.containsAll(checks));
    }

    /** With both bulk rates at zero, a stop writes its special candidates only. */
    @Test
    void aStopWritesOnlyItsSpecialCandidatesWhenTheBulkRatesAreZero() throws Throwable {
        List<Set<String>> written = writtenChecks("special", aBoardWithEachBucket(),
                caps().manaCap(0).playabilityRate(1.0).candidateRates(new TestCaps.CandidateRates(0.0, 0.0)).build());

        assertFalse(written.isEmpty(), "the forbidden Bolt is special");
        assertTrue(written.stream().allMatch(DecisionSampleTest::special), written.toString());
    }

    /** With the zone rate at one, the zone candidates join them; still no plain one. */
    @Test
    void aStopWritesItsZoneCandidatesAtAZoneRateOfOne() throws Throwable {
        List<Set<String>> written = writtenChecks("zone", aBoardWithEachBucket(),
                caps().manaCap(0).playabilityRate(1.0).candidateRates(new TestCaps.CandidateRates(1.0, 0.0)).build());

        assertTrue(written.stream().anyMatch(c -> !special(c) && c.contains("zone")),
                written.toString());
        assertTrue(written.stream().anyMatch(DecisionSampleTest::special), written.toString());
        assertTrue(written.stream().noneMatch(c -> !special(c) && !c.contains("zone")),
                "no plain candidate: " + written);
    }
}
