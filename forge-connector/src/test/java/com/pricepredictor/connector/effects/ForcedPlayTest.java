package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RandomSeatController;
import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.BoardPrecondition;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.DirectedRules;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
import com.pricepredictor.connector.effects.directed.operations.LifeOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.combat.Combat;
import forge.game.event.GameEventGameFinished;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.function.Function;

import static com.pricepredictor.connector.effects.RecordLines.events;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A directed fork's forced play (FR-010b), piece by piece: a live game set
 * up at a boundary, a test rule whose variant forces one play, the fork taken
 * there by the game's recording and played to the end of the turn; the
 * fork's records show the play made as named, and a play that cannot be made
 * ends the fork as a counted refusal with nothing written.
 *
 * <p>Each test rule's edit sets the opponent's life to 19, a fact no play
 * here reads, so what the fork does differently from the live turn is the
 * forced play's.
 */
@ExtendWith(ForgeExtension.class)
class ForcedPlayTest {
    private static final Role<Player> R_PLAYER = Role.player("player");
    private static final Role<Card> R_BOLT = Role.card("bolt");
    private static final Role<Card> R_OUTLET = Role.card("outlet");
    private static final Role<Card> R_SANCTUARY = Role.card("sanctuary");
    private static final Role<Card> R_SPELL = Role.card("spell");
    private static final Role<Card> R_BEARS = Role.card("bears");
    private static final Role<Card> R_MADE = Role.card("made");
    private static final Role<Player> R_OPPONENT = Role.player("opponent");
    private static final Role<Card> R_WALKER = Role.card("walker");
    private static final Role<Card> R_GIANT = Role.card("giant");
    private static final Role<Card> R_VICTIM = Role.card("victim");
    private static final Role<Card> R_PROWLER = Role.card("prowler");


    @TempDir
    Path tempDir;

    private static int games;

    /** A started two-seat game in {@code step} of the first seat's third turn, both libraries ten Forests. */
    private static Game bare(PhaseType step) {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        game.getPhaseHandler().devModeSet(step, game.getPlayers().get(0), 3);
        for (Player player : game.getPlayers()) {
            for (int i = 0; i < 10; i++) {
                game.getAction().moveToLibrary(TestBoard.card(game, player, "Forest"), null, null);
            }
        }
        return game;
    }

    private static Card permanent(Game game, Player owner, String name) {
        Card card = TestBoard.inPlay(game, owner, name);
        card.setGameTimestamp(game.getNextTimestamp());
        card.setSickness(false);
        return card;
    }

    private static Player you(Game game) {
        return game.getPlayers().get(0);
    }

    private static Player opponent(Game game) {
        return game.getPlayers().get(1);
    }

    /** A test rule at {@code step} binding {@code roles}, forcing {@code play}; its edit sets B's life to 19. */
    private static DirectedRule rule(PhaseType step, Function<Game, Binding> roles, ForcedPlay play) {
        return rule(step, roles, new Application(ForkOperation.LIFE,
                fork -> new LifeOperation(fork.player("opponent"), 19)).forcing(play));
    }

    private static DirectedRule rule(PhaseType step, Function<Game, Binding> roles, Application application) {
        BoardPrecondition precondition = live -> live.getPhaseHandler().is(step)
                ? Optional.of(roles.apply(live).with("player", you(live)).with("opponent", opponent(live)))
                : Optional.empty();
        return DirectedRule.of("T.1", "T.1-1", precondition, new DirectedRule.Variant("forced", application));
    }

    /**
     * Every record of the live turn recorded from the boundary beginning
     * {@code step} ({@code null}: the live game reaches it itself), the
     * rule's directed fork taken there.
     */
    private List<String> played(Game game, PhaseType step, DirectedRule rule) throws Throwable {
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(0).probes(0)
                .directed(1, 1.0, rule.name()).build();
        RecordShardWriter shard = ShardRun.writer(Files.createDirectories(tempDir.resolve("game-" + ++games)));
        try (GameRecording recording = new GameRecording(
                GameRecordingRequest.of(game, shard, ShardRun.GAME_ID, caps)
                        .seeded(7L)
                        .withForkSeats(ForkSeats.NONE)
                        .withDirected(DirectedRules.of(rule), new DirectedForkTally()))) {
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
        // A game whose one fork was refused, and that wrote nothing live, leaves no shard.
        return Files.exists(shard.path()) ? ShardRun.readShard(shard.path()) : List.of();
    }

    /** The cards the costs paid in {@code records} took, under {@code part} ({@code sacrificed}, {@code discarded}). */
    static List<Object> costCards(List<String> records, String part) {
        List<Object> cards = new ArrayList<>();
        for (String record : records) {
            if (RecordLines.payload(record).get("costs") instanceof Map<?, ?> costs
                    && costs.get(part) instanceof List<?> list) {
                cards.addAll(list);
            }
        }
        return cards;
    }

    private static List<String> forkOf(List<String> records) {
        return records.stream().filter(record -> Boolean.TRUE.equals(parse(record).get("fork"))).toList();
    }

    private static String ref(Card card) {
        return "E" + card.getId();
    }

    /** The params of every event of {@code type} in {@code records} whose subjects hold {@code subject}. */
    private static List<Map<String, Object>> eventsOn(List<String> records, String type, String subject) {
        List<Map<String, Object>> found = new ArrayList<>();
        for (String record : records) {
            for (Map<String, Object> event : events(record)) {
                if (type.equals(event.get("type")) && event.get("subjects") instanceof List<?> subjects
                        && subjects.contains(subject)) {
                    found.add(params(event));
                }
            }
        }
        return found;
    }

    private static long refusals(String reason) {
        return ForkRefusals.counts().getOrDefault("directed " + reason, 0L);
    }

    // ── targets, answers and attacks by role ────────────────────────────

    /** A cast's target named by role: the Bolt the AI would aim at the Bears goes at the Giant. */
    @Test
    void aCastIsAimedAtTheNamedTarget() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        permanent(game, you(game), "Mountain");
        Card bolt = TestBoard.inHand(game, you(game), "Lightning Bolt");
        permanent(game, opponent(game), "Grizzly Bears");
        Card giant = permanent(game, opponent(game), "Hill Giant");

        List<String> fork = forkOf(played(game, PhaseType.MAIN1, rule(PhaseType.MAIN1,
                live -> Binding.empty().with("bolt", bolt).with("giant", giant),
                ForcedPlay.cast(R_PLAYER, R_BOLT).targeting(R_GIANT))));

        List<Map<String, Object>> damage = eventsOn(fork, "damage_dealt", ref(giant));
        assertEquals(1, damage.size(), "the Bolt hits the Giant in the fork: " + damage);
        assertEquals(3, ((Number) damage.get(0).get("amount")).intValue());
    }

    /** Elfhame Sanctuary's "you may search", answered: yes skips the draw step, no does not. */
    @Test
    void aYouMayIsAnsweredAsNamed() throws Throwable {
        for (boolean yes : new boolean[] {true, false}) {
            Game game = bare(PhaseType.UPKEEP);
            Card sanctuary = permanent(game, you(game), "Elfhame Sanctuary");

            List<String> fork = forkOf(played(game, PhaseType.UPKEEP, rule(PhaseType.UPKEEP,
                    live -> Binding.empty().with("sanctuary", sanctuary),
                    ForcedPlay.answer(R_PLAYER, R_SANCTUARY, yes))));

            boolean skipped = fork.stream().anyMatch(record -> RecordLines.eventTypes(record).contains("phase_skipped"));
            assertEquals(yes, skipped, "answered " + yes + ": " + fork.size() + " fork records");
        }
    }

    /** An attack the AI would not make, into a Craw Wurm, added to its declaration. */
    @Test
    void anAttackIsAddedToTheDeclaration() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Card bears = permanent(game, you(game), "Grizzly Bears");
        permanent(game, opponent(game), "Craw Wurm");

        List<String> fork = forkOf(played(game, null, rule(PhaseType.COMBAT_DECLARE_ATTACKERS,
                live -> Binding.empty().with("bears", bears),
                ForcedPlay.attack(R_PLAYER, R_BEARS, R_OPPONENT))));

        boolean dealt = false;
        for (String record : fork) {
            for (Map<String, Object> event : events(record)) {
                dealt |= "damage_dealt".equals(event.get("type")) && ref(bears).equals(params(event).get("source"));
            }
        }
        assertTrue(dealt, "the Bears attack in the fork and deal combat damage");
    }

    /**
     * An answer matches its line, not only its source: a confirmation another
     * line of the source asks (Goblin Bombardment's activation, the answer
     * naming the source's triggers) is not the play's and leaves it unmade.
     */
    @Test
    void anAnswerIsGivenOnlyToItsLine() {
        Game game = bare(PhaseType.MAIN1);
        Player you = you(game);
        Card bombardment = permanent(game, you, "Goblin Bombardment");
        ForcedPlayRun run = ForcedPlayRun.start(new Application(ForkOperation.LIFE,
                        fork -> new LifeOperation(you, 20)).forcing(ForcedPlay.answer(R_PLAYER, R_OUTLET, true)),
                Binding.empty().with("player", you).with("outlet", bombardment), game);
        SpellAbility activation = bombardment.getSpellAbilities().stream()
                .filter(SpellAbility::isActivatedAbility).findFirst().orElseThrow();

        assertEquals(null, run.answer(you, activation));
        assertFalse(run.made());
    }

    /** The AI aims the forced creature at another defender: the attack is moved to the named one. */
    @Test
    void anAttackTheSeatAimsElsewhereIsMovedToTheNamedDefender() {
        Game game = bare(PhaseType.COMBAT_DECLARE_ATTACKERS);
        Player you = you(game);
        Card bears = permanent(game, you, "Grizzly Bears");
        Card walker = permanent(game, opponent(game), "Ajani Goldmane");
        walker.setCounters(CounterEnumType.LOYALTY, 4);
        ForcedPlayRun run = ForcedPlayRun.start(new Application(ForkOperation.LIFE,
                        fork -> new LifeOperation(you, 20)).forcing(ForcedPlay.attack(R_PLAYER, R_BEARS, R_WALKER)),
                Binding.empty().with("player", you).with("bears", bears).with("walker", walker), game);
        Combat combat = new Combat(you);
        combat.addAttacker(bears, opponent(game));

        run.declared(you, combat);

        assertSame(walker, combat.getDefenderByAttacker(bears));
        assertTrue(run.made());
    }

    // ── refusals ────────────────────────────────────────────────────────

    /** A named target the line cannot take is refused on the live board, before the fork: no budget, no record. */
    @Test
    void aTargetTheLineCannotTakeIsRefusedBeforeTheFork() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        permanent(game, you(game), "Swamp");
        Card disfigure = TestBoard.inHand(game, you(game), "Disfigure");
        permanent(game, opponent(game), "Grizzly Bears");
        long before = refusals(ForkRefusal.FORCED_NO_LINE);

        List<String> records = played(game, PhaseType.MAIN1, rule(PhaseType.MAIN1,
                live -> Binding.empty().with("spell", disfigure),
                ForcedPlay.cast(R_PLAYER, R_SPELL).targeting(R_OPPONENT)));

        assertEquals(List.of(), forkOf(records));
        assertTrue(refusals(ForkRefusal.FORCED_NO_LINE) > before, "counted as a refusal");
    }

    /**
     * FR-010b's promise for a refusal found before the fork: it spends none of
     * {@code --directed-forks-per-game}, is tallied as a refusal and not as a
     * fork, and leaves the rule its one fork of the game, which a later
     * boundary whose board offers the play takes.
     */
    @Test
    void aRefusalFoundBeforeTheForkSpendsNoBudgetAndLeavesTheRuleItsFork() {
        Game game = bare(PhaseType.MAIN1);
        permanent(game, you(game), "Swamp");
        Card disfigure = TestBoard.inHand(game, you(game), "Disfigure");
        Card bears = permanent(game, opponent(game), "Grizzly Bears");
        boolean[] aimable = {false};
        DirectedRule rule = rule(PhaseType.MAIN1,
                live -> aimable[0]
                        ? Binding.empty().with("spell", disfigure).with("victim", bears)
                        : Binding.empty().with("spell", disfigure).with("victim", opponent(live)),
                ForcedPlay.cast(R_PLAYER, R_SPELL).targeting(R_VICTIM));
        CollectionCaps caps = TestCaps.caps().interventions(0).probes(0).directed(1, 1.0, rule.name()).build();
        ForkBudget budget = new ForkBudget(caps);
        DirectedForkTally tally = new DirectedForkTally();
        List<String> forked = new ArrayList<>();
        DirectedForkScheduler scheduler = new DirectedForkScheduler(game,
                new DirectedForkScheduler.Directing(caps, DirectedRules.of(rule), 7L), budget,
                tally, (taken, variant, application, binding) -> forked.add(taken.name()));
        long before = refusals(ForkRefusal.FORCED_NO_LINE);

        assertEquals(0, scheduler.collect(PhaseType.MAIN1), "no fork where Disfigure cannot take its target");
        assertEquals(before + 1, refusals(ForkRefusal.FORCED_NO_LINE), "counted as a refusal");
        assertTrue(budget.mayDirect(), "the budget is untouched");
        assertEquals(0, tally.get(rule.name(), DirectedForkTally.Count.TAKEN), "no fork in the lifetime tally");
        assertEquals(List.of(), forked);

        aimable[0] = true;
        assertEquals(1, scheduler.collect(PhaseType.MAIN1), "the rule's one fork is taken later");
        assertEquals(List.of(rule.name()), forked);
    }

    /**
     * A role the edit makes ({@link Application#making}) is on the copy only:
     * the live dry run skips what names it, a targeting clause or the whole
     * line when it is the source, and still checks the rest of the
     * play: the source, its line and the other named targets.
     */
    @Test
    void theDryRunSkipsOnlyWhatARoleTheEditMakesNames() {
        Game game = bare(PhaseType.MAIN1);
        permanent(game, you(game), "Swamp");
        Card disfigure = TestBoard.inHand(game, you(game), "Disfigure");
        Card bears = permanent(game, opponent(game), "Grizzly Bears");
        Binding live = Binding.empty().with("player", you(game)).with("opponent", opponent(game))
                .with("spell", disfigure).with("bears", bears);
        Function<ForcedPlay, Application> made = play -> new Application(ForkOperation.LIFE,
                fork -> new LifeOperation(fork.player("opponent"), 19)).forcing(play)
                .making("made", (fork, copy) -> fork.card("bears"));

        assertEquals(Optional.of(ForkRefusal.FORCED_NO_LINE), ForcedPlayRun.dryRun(
                        made.apply(ForcedPlay.cast(R_PLAYER, R_SPELL).targeting(R_OPPONENT)), live, game),
                "Disfigure cannot take a player, whatever card the edit makes");
        assertEquals(Optional.of(ForkRefusal.FORCED_NO_LINE), ForcedPlayRun.dryRun(
                        made.apply(ForcedPlay.cast(R_PLAYER, R_BEARS).targeting(R_MADE)), live, game),
                "the Bears in play are no spell to cast");
        assertEquals(Optional.empty(), ForcedPlayRun.dryRun(
                        made.apply(ForcedPlay.cast(R_PLAYER, R_SPELL).targeting(R_MADE)), live, game),
                "the made target is checked on the copy");
        assertEquals(Optional.empty(), ForcedPlayRun.dryRun(
                        made.apply(ForcedPlay.cast(R_PLAYER, R_MADE)), live, game),
                "the made source's line is checked on the copy");
    }

    /** A charm is no line a forced play makes (its modes would be asked afresh): refused before the fork. */
    @Test
    void aForcedCharmHasNoLegalLine() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        permanent(game, you(game), "Mountain");
        permanent(game, you(game), "Plains");
        Card charm = TestBoard.inHand(game, you(game), "Boros Charm");
        long before = refusals(ForkRefusal.FORCED_NO_LINE);

        List<String> records = played(game, PhaseType.MAIN1, rule(PhaseType.MAIN1,
                live -> Binding.empty().with("spell", charm),
                ForcedPlay.cast(R_PLAYER, R_SPELL)));

        assertEquals(List.of(), forkOf(records));
        assertEquals(before + 1, refusals(ForkRefusal.FORCED_NO_LINE));
    }

    /** The edit takes the play's source away: refused on the edited copy, nothing written. */
    @Test
    void aSourceTheEditRemovesIsRefusedInTheFork() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Card bombardment = permanent(game, you(game), "Goblin Bombardment");
        permanent(game, you(game), "Grizzly Bears");
        long before = refusals(ForkRefusal.FORCED_NO_LINE);

        List<String> records = played(game, PhaseType.MAIN1, rule(PhaseType.MAIN1,
                live -> Binding.empty().with("outlet", bombardment),
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("outlet")))
                        .forcing(ForcedPlay.activate(R_PLAYER, R_OUTLET, ForcedPlay.Line.api("DealDamage")))));

        assertEquals(List.of(), forkOf(records));
        assertEquals(before + 1, refusals(ForkRefusal.FORCED_NO_LINE));
    }

    /** A play its step passes without, the mana never there: refused, nothing written. */
    @Test
    void aPlayNeverPlayableIsRefusedAsNotMade() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Card defenses = TestBoard.inHand(game, you(game), "Cached Defenses");
        long before = refusals(ForkRefusal.FORCED_NOT_MADE);

        List<String> records = played(game, PhaseType.MAIN1, rule(PhaseType.MAIN1,
                live -> Binding.empty().with("spell", defenses),
                ForcedPlay.cast(R_PLAYER, R_SPELL)));

        assertEquals(List.of(), forkOf(records));
        assertEquals(before + 1, refusals(ForkRefusal.FORCED_NOT_MADE));
    }

    /**
     * An answer with no step whose question is never asked (Elfhame
     * Sanctuary's upkeep search, the fork taken in the first main phase):
     * refused once the turn is over, nothing written.
     */
    @Test
    void anAnswerNeverAskedIsRefusedOnceTheTurnIsOver() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Card sanctuary = permanent(game, you(game), "Elfhame Sanctuary");
        long before = refusals(ForkRefusal.FORCED_NOT_MADE);

        List<String> records = played(game, PhaseType.MAIN1, rule(PhaseType.MAIN1,
                live -> Binding.empty().with("sanctuary", sanctuary),
                ForcedPlay.answer(R_PLAYER, R_SANCTUARY, true)));

        assertEquals(List.of(), forkOf(records));
        assertEquals(before + 1, refusals(ForkRefusal.FORCED_NOT_MADE));
    }

    /**
     * A refusal flagged in the turn's last main-loop step (the opponent's
     * forced Disfigure in the end step, at a creature whose ward {2} its one
     * Swamp leaves the AI unable to pay), after which the fork steps no
     * more: thrown once the turn is over, nothing written.
     */
    @Test
    void aRefusalFlaggedInTheLastStepIsThrownOnceTheTurnIsOver() throws Throwable {
        Game game = bare(PhaseType.END_OF_TURN);
        permanent(game, opponent(game), "Swamp");
        Card disfigure = TestBoard.inHand(game, opponent(game), "Disfigure");
        Card prowler = permanent(game, you(game), "Cackling Prowler");
        long before = refusals(ForkRefusal.FORCED_COST_NOT_PAID);

        List<String> records = played(game, PhaseType.END_OF_TURN, rule(PhaseType.END_OF_TURN,
                live -> Binding.empty().with("spell", disfigure).with("prowler", prowler),
                ForcedPlay.cast(R_OPPONENT, R_SPELL).targeting(R_PROWLER)));

        assertEquals(List.of(), forkOf(records));
        assertEquals(before + 1, refusals(ForkRefusal.FORCED_COST_NOT_PAID));
    }

    /**
     * A line the rules let the actor afford but the AI's own check does not
     * (Disfigure at a creature with ward {2}, one Swamp): refused as not paid
     * before the card moves, the Disfigure still in hand.
     */
    @Test
    void aLineTheAiCannotPayForIsRefusedBeforeItsCardMoves() {
        Game game = bare(PhaseType.MAIN1);
        Player you = you(game);
        permanent(game, you, "Swamp");
        Card disfigure = TestBoard.inHand(game, you, "Disfigure");
        Card prowler = permanent(game, opponent(game), "Cackling Prowler");
        ForcedPlayRun run = ForcedPlayRun.start(new Application(ForkOperation.LIFE,
                        fork -> new LifeOperation(you, 20))
                        .forcing(ForcedPlay.cast(R_PLAYER, R_SPELL).targeting(R_PROWLER)),
                Binding.empty().with("player", you).with("spell", disfigure).with("prowler", prowler), game);
        run.attach(new Random(5L));

        List<SpellAbility> chosen = you.getController().chooseSpellAbilityToPlay();
        assertNotNull(chosen, "the rules let the actor cast it");
        assertFalse(you.getController().playChosenSpellAbility(chosen.get(0)));

        assertEquals(ForkRefusal.FORCED_COST_NOT_PAID, run.refusal());
        assertFalse(run.made());
        assertTrue(game.getCardState(disfigure).isInZone(ZoneType.Hand), "the Disfigure never moved");
        assertTrue(game.getStack().isEmpty());
    }

    // ── the random seat ─────────────────────────────────────────────────

    /**
     * A random seat that draws every decision consults the forced play first:
     * at its priority it plays the forced line, cost paid by the AI's own
     * choice (the Bears, its one creature, which the line does not target),
     * and is a random seat again once the play is made.
     */
    @Test
    void aRandomSeatPlaysTheForcedLineFirst() {
        Game game = bare(PhaseType.MAIN1);
        Player you = you(game);
        Card bombardment = permanent(game, you, "Goblin Bombardment");
        Card bears = permanent(game, you, "Grizzly Bears");
        permanent(game, you, "Mountain");
        TestBoard.inHand(game, you, "Lightning Bolt");
        RandomSeatController seat = new RandomSeatController(game, you, you.getLobbyPlayer(), 1.0, Random::new);
        you.dangerouslySetController(seat);
        seat.attach(game, null, null, new Random(3L));
        Binding binding = Binding.empty().with("player", you).with("outlet", bombardment)
                .with("opponent", opponent(game));
        ForcedPlayRun run = ForcedPlayRun.start(new Application(ForkOperation.LIFE,
                        fork -> new LifeOperation(you, 20))
                        .forcing(ForcedPlay.activate(R_PLAYER, R_OUTLET, ForcedPlay.Line.api("DealDamage"))
                                .targeting(R_OPPONENT)),
                binding, game);
        run.attach(new Random(5L));
        assertSame(seat, you.getController(), "a random seat takes the play itself");

        List<SpellAbility> chosen = seat.chooseSpellAbilityToPlay();
        assertNotNull(chosen);
        assertEquals(bombardment, chosen.get(0).getHostCard());
        assertTrue(seat.playChosenSpellAbility(chosen.get(0)));
        assertTrue(run.made());
        assertTrue(bears.isInZone(ZoneType.Graveyard) || game.getCardState(bears).isInZone(ZoneType.Graveyard),
                "the Bears paid the cost");
        assertEquals(null, run.choose(you), "made once");
    }

    /** A plain AI seat is replaced, in the game the play is attached to only, by a forcing seat. */
    @Test
    void aPlainAiSeatIsReplacedByAForcingSeat() {
        Game game = bare(PhaseType.MAIN1);
        Player you = you(game);
        Card bears = permanent(game, you, "Grizzly Bears");
        ForcedPlayRun run = ForcedPlayRun.start(new Application(ForkOperation.LIFE,
                        fork -> new LifeOperation(you, 20))
                        .forcing(ForcedPlay.attack(R_PLAYER, R_BEARS, R_OPPONENT)),
                Binding.empty().with("player", you).with("bears", bears).with("opponent", opponent(game)), game);
        run.attach(new Random(5L));
        assertTrue(you.getController() instanceof ForcingSeatController, you.getController().toString());
        assertFalse(opponent(game).getController() instanceof ForcingSeatController);
        ForkRefusal refusal = assertThrows(ForkRefusal.class, run::finish);
        assertEquals(ForkRefusal.FORCED_NOT_MADE, refusal.reason());
    }
}
