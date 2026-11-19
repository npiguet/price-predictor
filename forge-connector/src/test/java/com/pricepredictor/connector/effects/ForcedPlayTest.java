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
import com.pricepredictor.connector.effects.directed.operations.LifeOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.ability.AbilityUtils;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.combat.Combat;
import forge.game.cost.Cost;
import forge.game.event.GameEventGameFinished;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.PlayerCollection;
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
        return DirectedRule.of("T.1", "T.1-1", precondition, "forced", application);
    }

    /**
     * Every record of the live turn recorded from the boundary beginning
     * {@code step} ({@code null}: the live game reaches it itself), the
     * rule's directed fork taken there.
     */
    private List<String> played(Game game, PhaseType step, DirectedRule rule) throws Throwable {
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(0).probes(0)
                .directed(1, 1.0, rule.rule()).build();
        RecordShardWriter shard = ShardRun.writer(Files.createDirectories(tempDir.resolve("game-" + ++games)));
        try (GameRecording recording = new GameRecording(game, shard, ShardRun.GAME_ID, caps, 7L, null,
                ForkSeats.NONE, DirectedRules.of(rule), new DirectedForkTally())) {
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

    // ── targets, costs, answers and attacks by role ─────────────────────

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
                ForcedPlay.cast("player", "bolt").targeting("giant"))));

        List<Map<String, Object>> damage = eventsOn(fork, "damage_dealt", ref(giant));
        assertEquals(1, damage.size(), "the Bolt hits the Giant in the fork: " + damage);
        assertEquals(3, ((Number) damage.get(0).get("amount")).intValue());
    }

    /** A cost's sacrifice named by role: the Giant, not the Bears the AI would pick. */
    @Test
    void aCostSacrificesTheNamedCard() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Card bombardment = permanent(game, you(game), "Goblin Bombardment");
        Card bears = permanent(game, you(game), "Grizzly Bears");
        Card giant = permanent(game, you(game), "Hill Giant");

        List<String> fork = forkOf(played(game, PhaseType.MAIN1, rule(PhaseType.MAIN1,
                live -> Binding.empty().with("outlet", bombardment).with("giant", giant),
                ForcedPlay.activate("player", "outlet", ForcedPlay.Line.api("DealDamage")).paying("giant"))));

        assertEquals(List.of(ref(giant)), costCards(fork, "sacrificed"), "the Giant is sacrificed in the fork");
        assertFalse(costCards(fork, "sacrificed").contains(ref(bears)), "the Bears stay");
    }

    /** Elfhame Sanctuary's "you may search", answered: yes skips the draw step, no does not. */
    @Test
    void aYouMayIsAnsweredAsNamed() throws Throwable {
        for (boolean yes : new boolean[] {true, false}) {
            Game game = bare(PhaseType.UPKEEP);
            Card sanctuary = permanent(game, you(game), "Elfhame Sanctuary");

            List<String> fork = forkOf(played(game, PhaseType.UPKEEP, rule(PhaseType.UPKEEP,
                    live -> Binding.empty().with("sanctuary", sanctuary),
                    ForcedPlay.answer("player", "sanctuary", yes))));

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
                ForcedPlay.attack("player", "bears", "opponent"))));

        boolean dealt = false;
        for (String record : fork) {
            for (Map<String, Object> event : events(record)) {
                dealt |= "damage_dealt".equals(event.get("type")) && ref(bears).equals(params(event).get("source"));
            }
        }
        assertTrue(dealt, "the Bears attack in the fork and deal combat damage");
    }

    /**
     * Preferred Selection's "you may sacrifice it and pay {2}{G}{G}", which
     * Forge's AI never pays, answered yes: the fork pays it, the Selection is
     * sacrificed.
     */
    @Test
    void aForcedYesToAYouMayPayIsPaidWhateverTheAiWouldChoose() throws Throwable {
        Game game = bare(PhaseType.UPKEEP);
        Card selection = permanent(game, you(game), "Preferred Selection");
        for (int i = 0; i < 4; i++) {
            permanent(game, you(game), "Forest");
        }

        List<String> fork = forkOf(played(game, PhaseType.UPKEEP, rule(PhaseType.UPKEEP,
                live -> Binding.empty().with("selection", selection),
                ForcedPlay.answer("player", "selection", true))));

        assertTrue(eventsOn(fork, "zone_change", ref(selection)).stream()
                        .anyMatch(move -> "graveyard".equals(move.get("to_zone"))),
                "the Selection is sacrificed in the fork: " + fork.size() + " fork records");
    }

    /** A random seat pays a forced yes to a "you may pay" too, where its AI would not. */
    @Test
    void aRandomSeatPaysAForcedYes() {
        Game game = bare(PhaseType.UPKEEP);
        Player you = you(game);
        Card selection = permanent(game, you, "Preferred Selection");
        for (int i = 0; i < 4; i++) {
            permanent(game, you, "Forest");
        }
        RandomSeatController seat = new RandomSeatController(game, you, you.getLobbyPlayer(), 1.0, Random::new);
        you.dangerouslySetController(seat);
        seat.attach(game, null, null, new Random(3L));
        ForcedPlayRun run = ForcedPlayRun.start(new Application(ForkOperation.LIFE,
                        fork -> new LifeOperation(you, 20)).forcing(ForcedPlay.answer("player", "selection", true)),
                Binding.empty().with("player", you).with("selection", selection), game);
        run.attach(new Random(5L));
        SpellAbility look = selection.getTriggers().get(0).ensureAbility();
        look.setActivatingPlayer(you);
        SpellAbility bottom = look.getSubAbility();
        Cost cost = AbilityUtils.calculateUnlessCost(bottom, bottom.getParam("UnlessCost"), true);

        assertTrue(seat.payCostToPreventEffect(cost, bottom, false, new PlayerCollection(you)), "paid");
        assertTrue(run.made());
        assertTrue(game.getCardState(selection).isInZone(ZoneType.Graveyard), "the Selection was sacrificed");
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
                        fork -> new LifeOperation(you, 20)).forcing(ForcedPlay.answer("player", "outlet", true)),
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
                        fork -> new LifeOperation(you, 20)).forcing(ForcedPlay.attack("player", "bears", "walker")),
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
                ForcedPlay.cast("player", "spell").targeting("opponent")));

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
                ForcedPlay.cast("player", "spell").targeting("victim"));
        CollectionCaps caps = TestCaps.caps().interventions(0).probes(0).directed(1, 1.0, rule.rule()).build();
        ForkBudget budget = new ForkBudget(caps);
        DirectedForkTally tally = new DirectedForkTally();
        List<String> forked = new ArrayList<>();
        DirectedForkScheduler scheduler = new DirectedForkScheduler(game, caps, DirectedRules.of(rule), budget,
                tally, 7L, (taken, variant, application, binding) -> forked.add(taken.rule()));
        long before = refusals(ForkRefusal.FORCED_NO_LINE);

        assertEquals(0, scheduler.collect(PhaseType.MAIN1), "no fork where Disfigure cannot take its target");
        assertEquals(before + 1, refusals(ForkRefusal.FORCED_NO_LINE), "counted as a refusal");
        assertTrue(budget.mayDirect(), "the budget is untouched");
        assertEquals(0, tally.get(rule.rule(), DirectedForkTally.Count.TAKEN), "no fork in the lifetime tally");
        assertEquals(List.of(), forked);

        aimable[0] = true;
        assertEquals(1, scheduler.collect(PhaseType.MAIN1), "the rule's one fork is taken later");
        assertEquals(List.of(rule.rule()), forked);
    }

    /**
     * A role the edit makes ({@link Application#making}) is on the copy only:
     * the live dry run skips what names it, a targeting clause, a cost or the
     * whole line when it is the source, and still checks the rest of the
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
                        made.apply(ForcedPlay.cast("player", "spell").targeting("opponent").paying("made")), live, game),
                "Disfigure cannot take a player, whatever card the edit makes");
        assertEquals(Optional.of(ForkRefusal.FORCED_NO_LINE), ForcedPlayRun.dryRun(
                        made.apply(ForcedPlay.cast("player", "bears").targeting("made")), live, game),
                "the Bears in play are no spell to cast");
        assertEquals(Optional.empty(), ForcedPlayRun.dryRun(
                        made.apply(ForcedPlay.cast("player", "spell").targeting("made")), live, game),
                "the made target is checked on the copy");
        assertEquals(Optional.empty(), ForcedPlayRun.dryRun(
                        made.apply(ForcedPlay.cast("player", "made")), live, game),
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
                ForcedPlay.cast("player", "spell")));

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
                        .forcing(ForcedPlay.activate("player", "outlet", ForcedPlay.Line.api("DealDamage")))));

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
                ForcedPlay.cast("player", "spell")));

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
                ForcedPlay.answer("player", "sanctuary", true)));

        assertEquals(List.of(), forkOf(records));
        assertEquals(before + 1, refusals(ForkRefusal.FORCED_NOT_MADE));
    }

    /**
     * A refusal flagged in the turn's last main-loop step (the opponent's
     * forced activation at its last priority of the end step, its named cost
     * card one the cost cannot take), after which the fork steps no more:
     * thrown once the turn is over, nothing written.
     */
    @Test
    void aRefusalFlaggedInTheLastStepIsThrownOnceTheTurnIsOver() throws Throwable {
        Game game = bare(PhaseType.END_OF_TURN);
        Card bombardment = permanent(game, opponent(game), "Goblin Bombardment");
        permanent(game, opponent(game), "Grizzly Bears");
        Card forest = permanent(game, opponent(game), "Forest");
        long before = refusals(ForkRefusal.FORCED_COST_NOT_PAID);

        List<String> records = played(game, PhaseType.END_OF_TURN, rule(PhaseType.END_OF_TURN,
                live -> Binding.empty().with("outlet", bombardment).with("forest", forest),
                ForcedPlay.activate("opponent", "outlet", ForcedPlay.Line.api("DealDamage")).paying("forest")));

        assertEquals(List.of(), forkOf(records));
        assertEquals(before + 1, refusals(ForkRefusal.FORCED_COST_NOT_PAID));
    }

    /**
     * A forced yes to a "you may pay" whose cost cannot be paid (Preferred
     * Selection with no land): refused, nothing written.
     */
    @Test
    void aForcedYesThatCannotBePaidIsRefused() throws Throwable {
        Game game = bare(PhaseType.UPKEEP);
        Card selection = permanent(game, you(game), "Preferred Selection");
        long before = refusals(ForkRefusal.FORCED_COST_NOT_PAID);

        List<String> records = played(game, PhaseType.UPKEEP, rule(PhaseType.UPKEEP,
                live -> Binding.empty().with("selection", selection),
                ForcedPlay.answer("player", "selection", true)));

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
                        .forcing(ForcedPlay.cast("player", "spell").targeting("prowler")),
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

    /** A named cost card the cost cannot take (a land for a creature's sacrifice): the cost is not paid. */
    @Test
    void aNamedCostCardTheCostCannotTakeLeavesTheCostUnpaid() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Card bombardment = permanent(game, you(game), "Goblin Bombardment");
        permanent(game, you(game), "Grizzly Bears");
        Card forest = permanent(game, you(game), "Forest");
        long before = refusals(ForkRefusal.FORCED_COST_NOT_PAID);

        List<String> records = played(game, PhaseType.MAIN1, rule(PhaseType.MAIN1,
                live -> Binding.empty().with("outlet", bombardment).with("forest", forest),
                ForcedPlay.activate("player", "outlet", ForcedPlay.Line.api("DealDamage")).paying("forest")));

        assertEquals(List.of(), forkOf(records));
        assertEquals(before + 1, refusals(ForkRefusal.FORCED_COST_NOT_PAID));
    }

    // ── the random seat ─────────────────────────────────────────────────

    /**
     * A random seat that draws every decision consults the forced play first:
     * at its priority it plays the forced line, cost paid, and is a random
     * seat again once the play is made.
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
        Binding binding = Binding.empty().with("player", you).with("outlet", bombardment).with("bears", bears);
        ForcedPlayRun run = ForcedPlayRun.start(new Application(ForkOperation.LIFE,
                        fork -> new LifeOperation(you, 20))
                        .forcing(ForcedPlay.activate("player", "outlet", ForcedPlay.Line.api("DealDamage"))
                                .paying("bears")),
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
                        .forcing(ForcedPlay.attack("player", "bears", "opponent")),
                Binding.empty().with("player", you).with("bears", bears).with("opponent", opponent(game)), game);
        run.attach(new Random(5L));
        assertTrue(you.getController() instanceof ForcingSeatController, you.getController().toString());
        assertFalse(opponent(game).getController() instanceof ForcingSeatController);
        ForkRefusal refusal = assertThrows(ForkRefusal.class, run::finish);
        assertEquals(ForkRefusal.FORCED_NOT_MADE, refusal.reason());
    }
}
