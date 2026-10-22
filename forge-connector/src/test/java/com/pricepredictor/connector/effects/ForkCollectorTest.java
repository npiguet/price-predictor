package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.StaticData;
import forge.ai.LobbyPlayerAi;
import forge.deck.Deck;
import forge.game.Game;
import forge.game.GameRules;
import forge.game.GameStage;
import forge.game.GameType;
import forge.game.Match;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.zip.GZIPInputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The fork budgets and the record shapes they produce.
 *
 * <p>Forking a live game needs one, which is the integration test's job. What is
 * testable here is the budget arithmetic — and it is worth testing precisely,
 * because the failure mode is a run that spends its whole simulation budget on
 * forks of one board.
 */
@ExtendWith(ForgeExtension.class)
class ForkCollectorTest {

    @TempDir
    Path tempDir;

    private static CollectionCaps caps(
            int interventions, int probes, List<String> keywords) {
        return new CollectionCaps(
                2000, 0.1, interventions, probes, keywords, 0.0, 0.0);
    }

    private RecordShardWriter writer;

    private ForkCollector collector(CollectionCaps caps) {
        writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        return new ForkCollector(null, writer, "run.0-l1.0", caps, 42L);
    }

    /** How many records the collector has written to its shard so far. */
    private int recordsInShard() throws IOException {
        writer.close();
        return readShard(tempDir.resolve("run.0-l1" + RecordShardWriter.SUFFIX)).size();
    }

    /**
     * The same, over a game whose turn can be read.
     *
     * <p>No turn is ever taken in the shared test game, so its phase handler
     * reads turn 0 — which is all the pairing guard needs: a branch that says
     * it came from any other turn disagrees with it.
     */
    private ForkCollector collectorOverAGame(CollectionCaps caps) {
        writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        return new ForkCollector(TestCards.game(), writer, "run.0-l1.0", caps, 42L);
    }

    // ── the per-game budget ─────────────────────────────────────────────

    @Test
    void aFreshGameMayIntervene() {
        assertTrue(collector(caps(2, 2, List.of())).mayIntervene(1));
    }

    @Test
    void anExhaustedInterventionBudgetStopsFurtherForks() {
        ForkCollector collector = collector(caps(0, 2, List.of()));
        assertFalse(collector.mayIntervene(1));
    }

    // ── the fixed per-resolution cap ────────────────────────────────────

    @Test
    void atMostTwoForksMayTargetOneRealResolution() {
        assertEquals(2, ForkCollector.MAX_FORKS_PER_RESOLUTION);
    }

    @Test
    void thePerResolutionCapIsIndependentOfTheFlag() {
        // A generous --interventions-per-game must not let one board absorb
        // the whole budget: two counterfactuals for it are already both worth
        // having, and a third costs a game's simulation for a third variation.
        ForkCollector collector = collector(caps(100, 2, List.of()));
        collector.intervene(null, 7);
        collector.intervene(null, 7);
        assertFalse(collector.mayIntervene(7));
        assertTrue(collector.mayIntervene(8));
    }

    @Test
    void aDiscardedForkStillCountsAgainstItsBudget() {
        // Otherwise a systematically failing copy would retry until the game
        // ended. A null game makes every copy fail.
        ForkCollector collector = collector(caps(1, 2, List.of()));
        assertFalse(collector.intervene(null, 1));
        assertFalse(collector.mayIntervene(1));
    }

    @Test
    void aFailedForkWritesNoRecord() throws IOException {
        ForkCollector collector = collector(caps(2, 2, List.of()));
        collector.intervene(null, 1);
        assertEquals(0, recordsInShard());
    }

    /**
     * The intervention resolves the ability rather than only forking.
     *
     * <p>The failure this guards against is a collector that copies the game,
     * score-checks it, snapshots it, and writes a record saying an ability was
     * forced without saying what it did — which is what this class did before
     * the resolution was written, and which looks like a working collector.
     *
     * <p>Only the wiring is checkable here, because forking needs a live game;
     * {@code tests/integration/test_effects_forks.py} asserts against a real
     * one that the records carry events.
     */
    @Test
    void interveningResolvesRatherThanOnlyCopying() throws Exception {
        var method = ForkCollector.class.getDeclaredMethod(
                "forceResolution",
                forge.game.Game.class,
                forge.game.spellability.SpellAbility.class,
                forge.game.player.Player.class,
                SnapshotBuilder.class);
        assertTrue(method != null, "no resolution step on the intervention path");
        // Returns the events it observed, so an empty payload can only mean the
        // ability genuinely did nothing observable -- carried beside where each
        // chosen mode began, which is what splits a modal one per mode (FR-029f).
        var events = method.getReturnType().getDeclaredMethod("events");
        assertEquals(List.class, events.getReturnType());
    }

    // ── the probe switch ────────────────────────────────────────────────

    @Test
    void noProbeIsTakenWithTheFlagUnset() {
        // Whatever the build state: the flag is the runtime switch, and it is
        // deliberately separate from the build decision gate 2 drives.
        ForkCollector collector = collector(caps(2, 2, List.of()));
        assertFalse(collector.mayProbe("wither"));
    }

    @Test
    void onlyTheNamedKeywordsAreProbed() {
        ForkCollector collector = collector(caps(2, 2, List.of("wither")));
        assertTrue(collector.mayProbe("wither"));
        assertFalse(collector.mayProbe("lifelink"));
    }

    @Test
    void theProbeBudgetIsSpentByTakingProbes() {
        // Spent even by a fork that fails: a null game makes every copy fail.
        ForkCollector collector = collector(caps(2, 1, List.of("wither")));
        assertNull(collector.probe("wither", TestCards.build("Grizzly Bears"), false));
        assertFalse(collector.mayProbe("wither"));
    }

    @Test
    void aProbeForAnUnnamedKeywordSpendsNothing() throws IOException {
        ForkCollector collector = collector(caps(2, 1, List.of("wither")));
        assertNull(collector.probe("lifelink", TestCards.build("Grizzly Bears"), false));
        assertTrue(collector.mayProbe("wither"));
        assertEquals(0, recordsInShard());
    }

    // ── the held branch knows which step it came from ───────────────────

    /**
     * A branch carries the damage step it forked from.
     *
     * <p>Without it the branch is completed against whichever combat record
     * comes next, and in the first corpus that was systematically the wrong one:
     * every first-strike-step fork was empty, its step wrote no real record, and
     * the branch was written against the regular step's.
     */
    @Test
    void aHeldBranchNamesItsDamageStep() {
        ForkCollector.HeldProbe held = new ForkCollector.HeldProbe(
                "trample", "E1", "{}", List.of(), "P0", "\"combat\":{}",
                DamageStep.FIRST_STRIKE.wireName(), ForkCollector.TURN_UNKNOWN);

        assertEquals("first_strike", held.substep());
        // Spelled as the bracket collector spells it, because the two classes
        // have to agree about which step they are describing.
        assertEquals("regular", DamageStep.REGULAR.wireName());
    }

    // ── the held branch knows *when* it came from ───────────────────────

    /**
     * A branch and the record it mirrors describe the same turn, or neither is
     * written.
     *
     * <p>The substep alone was not enough. A branch whose own damage step wrote
     * no combat record kept waiting for the next record of that substep, which
     * in the smoke corpus arrived up to <b>eight turns</b> later: 55 of 1,069
     * probe records mirrored a record from a different turn, and every mirrored
     * pair whose turns disagreed was one of those. Because the branch's own
     * snapshot is honest — the copier carries the turn across, and not one of
     * the 1,788 interventional forks ever named a stale one — relabelling it
     * would make the state block lie, so the branch is dropped instead.
     *
     * <p>Dropped still spends the fork's budget: the simulation was spent
     * either way.
     */
    @Test
    void aBranchFromAnEarlierTurnIsNotWrittenAgainstThisOnesRecord() throws IOException {
        ForkCollector collector = collectorOverAGame(caps(2, 2, List.of("trample")));

        assertFalse(collector.writeHeldProbe(branch(7), "run.0-l1.9"),
                "a branch seven turns older than its mirror is not a pair");
        assertEquals(0, recordsInShard());
    }

    /** A branch from the turn the record was written on is written. */
    @Test
    void aBranchFromThisTurnIsWritten() throws IOException {
        ForkCollector collector = collectorOverAGame(caps(2, 2, List.of("trample")));

        assertTrue(collector.writeHeldProbe(branch(0), "run.0-l1.9"));
        assertEquals(1, recordsInShard());
    }

    /**
     * A branch with no turn to check is written, as it always was.
     *
     * <p>{@link ForkCollector#TURN_UNKNOWN} is what a hand-built branch carries
     * and what a collector with no live game reads, and the guard fires on
     * neither: there is no turn to disagree with.
     */
    @Test
    void aBranchWithNoTurnIsPairedAsBefore() {
        ForkCollector collector = collectorOverAGame(caps(2, 2, List.of("trample")));
        ForkCollector.HeldProbe held = new ForkCollector.HeldProbe(
                "trample", "E1", "{}", List.of(), "P0", "\"combat\":{}",
                DamageStep.REGULAR.wireName(), ForkCollector.TURN_UNKNOWN);

        assertEquals(ForkCollector.TURN_UNKNOWN, held.turn());
        assertTrue(collector.writeHeldProbe(held, "run.0-l1.9"));
    }

    /** A branch from a named turn, otherwise like every other hand-built one. */
    private static ForkCollector.HeldProbe branch(int turn) {
        return new ForkCollector.HeldProbe(
                "trample", "E1", "{}", List.of(), "P0", "\"combat\":{}",
                DamageStep.REGULAR.wireName(), turn);
    }

    // ── a probe branch reads damage as the observed record does ─────────

    /**
     * A probe branch's combat damage carries the combat flag and the named
     * cause the observed combat record carries.
     *
     * <p>Gate 2 reads the branch against the record it mirrors, so the two
     * must render one damage the same way. The blocker's damage to the
     * attacker is the same in both branches — the stripped keyword is the
     * attacker's trample — so its rendered event must match verbatim.
     */
    @Test
    void aProbeBranchsCombatDamageReadsAsTheObservedRecordReadsIt() throws IOException {
        Game game = gameInADamageStep();
        Player attacking = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        Card dreadmaw = onBattlefield(game, "Colossal Dreadmaw", attacking);
        Card bears = onBattlefield(game, "Grizzly Bears", defending);
        Combat combat = new Combat(attacking);
        combat.addAttacker(dreadmaw, defending);
        combat.addBlocker(dreadmaw, bears);
        // What the declare-blockers step leaves behind.
        combat.setBlocked(dreadmaw, true);
        combat.orderBlockersForDamageAssignment();
        combat.orderAttackersForDamageAssignment();
        game.getPhaseHandler().setCombat(combat);

        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        ForkCollector.HeldProbe held;
        try {
            ForkCollector forks = new ForkCollector(
                    game, writer, "run.0-l1.0", caps(0, 1, List.of("trample")), 1L);
            held = forks.probe("Trample", dreadmaw, false);
            assertNotNull(held, "the probe fork was discarded");

            BusBracketCollector observed = new BusBracketCollector(game, writer, "run.0-l1.0");
            game.subscribeToEvents(observed);
            assertTrue(combat.assignCombatDamage(false), "the live step assigned no damage");
            combat.dealAssignedDamage();
            observed.finishGame(Set.of());
        } finally {
            writer.close();
        }

        String record = readShard(writer.path()).stream()
                .filter(r -> r.contains("\"kind\":\"combat\"")).findFirst().orElseThrow();
        String onTheBears = damageTo(held.events(), bears).toJson();
        assertTrue(onTheBears.contains("\"combat\":true"), onTheBears);
        assertTrue(onTheBears.contains("\"cause\":\"E" + dreadmaw.getId() + "\""), onTheBears);
        String onTheDreadmaw = damageTo(held.events(), dreadmaw).toJson();
        assertTrue(record.contains(onTheDreadmaw),
                "the branch renders " + onTheDreadmaw + " and the observed record " + record);
        // The stripped keyword is the branch's input, not something its step did.
        assertTrue(held.events().stream()
                        .noneMatch(e -> EffectEvent.KEYWORD_CHANGE.equals(e.type())),
                held.events().stream().map(EffectEvent::toJson).toList().toString());
    }

    // ── a forced resolution's state is the board it acted on ────────────

    /**
     * A forced targeted spell's record names the target it resolved at, as
     * an observed resolution record does.
     *
     * <p>The fork draws its targets itself, so a state rendered before that
     * draw holds an ability with no target: Disfigure resolving at the one
     * creature on the board, recorded as aimed at nothing.
     */
    @Test
    void aForcedTargetedSpellsRecordNamesItsTarget() throws IOException {
        Game game = gameIn(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = onBattlefield(game, "Grizzly Bears", game.getPlayers().get(1));
        // In hand, where the fork's copier can find it.
        Card disfigure = game.getAction().moveTo(ZoneType.Hand,
                CardFactory.getCard(StaticData.instance().getCommonCards()
                        .getCard("Disfigure"), caster, game),
                null, null);
        SpellAbility spell = disfigure.getFirstSpellAbility();
        spell.setActivatingPlayer(caster);

        RecordShardWriter writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        try {
            ForkCollector forks = new ForkCollector(
                    game, writer, "run.0-l1.0", caps(1, 0, List.of()), 1L);
            assertTrue(forks.intervene(spell, 1), "the forced resolution observed nothing");
        } finally {
            writer.close();
        }

        List<String> records = readShard(writer.path());
        assertEquals(1, records.size(), records.toString());
        String target = "\"E" + bears.getId() + "\"";
        assertTrue(records.get(0).contains("\"refs\":{\"targets\":[" + target + "]"),
                records.get(0));
    }

    private static EffectEvent damageTo(List<EffectEvent> events, Card card) {
        return events.stream()
                .filter(e -> EffectEvent.DAMAGE_DEALT.equals(e.type())
                        && e.subjects().contains("E" + card.getId()))
                .findFirst().orElseThrow(() -> new AssertionError(
                        "no damage to " + card + " in "
                                + events.stream().map(EffectEvent::toJson).toList()));
    }

    /** A two-player game in the regular damage step, past the start. */
    private static Game gameInADamageStep() {
        return gameIn(PhaseType.COMBAT_DAMAGE);
    }

    /** A two-player game in {@code phase}, past the start. */
    private static Game gameIn(PhaseType phase) {
        List<RegisteredPlayer> players = List.of(
                new RegisteredPlayer(new Deck()).setPlayer(new LobbyPlayerAi("a", null)),
                new RegisteredPlayer(new Deck()).setPlayer(new LobbyPlayerAi("b", null)));
        GameRules rules = new GameRules(GameType.Constructed);
        Game game = new Game(players, rules, new Match(rules, players, "ForkCollectorTest"));
        game.getPhaseHandler().devModeSet(phase, game.getPlayers().get(0));
        game.setAge(GameStage.Play);
        return game;
    }

    private static Card onBattlefield(Game game, String name, Player owner) {
        Card card = CardFactory.getCard(
                StaticData.instance().getCommonCards().getCard(name), owner, game);
        return game.getAction().moveToPlay(card, owner, null, null);
    }

    private static List<String> readShard(Path path) throws IOException {
        if (!Files.exists(path)) {
            return List.of();
        }
        try (var gzip = new GZIPInputStream(Files.newInputStream(path));
                var reader = new BufferedReader(
                        new InputStreamReader(gzip, StandardCharsets.UTF_8))) {
            return reader.lines().toList();
        }
    }
}
