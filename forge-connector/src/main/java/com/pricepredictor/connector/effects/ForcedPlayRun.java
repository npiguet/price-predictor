package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.RandomSpellPlayer;
import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import forge.ai.AiCostDecision;
import forge.ai.ComputerUtilCost;
import forge.ai.PlayerControllerAi;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardCollectionView;
import forge.game.card.CardLists;
import forge.game.card.CardPredicates;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.cost.Cost;
import forge.game.cost.CostDiscard;
import forge.game.cost.CostExile;
import forge.game.cost.CostPayment;
import forge.game.cost.CostSacrifice;
import forge.game.cost.PaymentDecision;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.PlayerController;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Random;

/**
 * One directed fork's {@link ForcedPlay} (FR-010b), as its forcing seat makes
 * it: the play's roles read in the fork's copy, and whether it has been made.
 *
 * <p>Consulted first by the actor's controller in the copy, a
 * {@link ForcingSeatController} or a random seat ({@link ForcedPlaySeat}), at
 * the decisions the play is made in: at a priority of the fork's step where
 * the line is playable by the rules ({@link PlayabilityVerdict}) and its named
 * targets are legal, the line; when the line is played, its X and targets,
 * and its cost paid through the engine's own cast path
 * ({@link RandomSpellPlayer#play}) with the named cards as what it sacrifices,
 * discards or exiles; a confirmation or a "you may pay" the source's named
 * line asks, the play's answer; a declaration of attackers, the named attack
 * added to the seat's own. Every other decision is the seat's.
 *
 * <p>It never throws inside the engine: a play that cannot be made flags its
 * refusal, and {@code ForkPlayForward} throws it between main-loop steps
 * ({@link #check}) and after the turn ({@link #finish}), before any line of
 * the fork is written. Its draws (an unnamed clause's targets, X) come from
 * the fork's seed.
 */
public final class ForcedPlayRun {

    private final ForcedPlay play;
    private final Game game;
    private final Player actor;
    private final int sourceId;
    /** The step the play is made in; an answer's is null, any step of the turn. */
    private final PhaseType step;
    private final int turn;
    private final List<List<GameEntity>> targets;
    private final List<Card> costs;
    private final GameEntity defender;
    private RandomChoices choices;
    /** The line the actor's last priority chose, until it is played. */
    private SpellAbility pending;
    private boolean made;
    /** Why the play cannot be made, once known. */
    private String refusal;

    private ForcedPlayRun(ForcedPlay play, Binding fork, Game game) {
        this.play = play;
        this.game = game;
        this.actor = fork.player(play.actor());
        this.sourceId = fork.card(play.source()).getId();
        this.step = switch (play.kind()) {
            case ANSWER -> null;
            case ATTACK -> PhaseType.COMBAT_DECLARE_ATTACKERS;
            case CAST, ACTIVATE -> game.getPhaseHandler().getPhase();
        };
        this.turn = game.getPhaseHandler().getTurn();
        this.targets = new ArrayList<>();
        for (List<String> clause : play.targets()) {
            this.targets.add(clause.stream().map(role -> entity(fork, role)).toList());
        }
        this.costs = play.costs().stream().map(fork::card).toList();
        this.defender = play.defender() == null ? null : entity(fork, play.defender());
    }

    /**
     * The forced play of {@code application}'s fork, its roles read in the
     * fork's binding after the edit, or null when the variant forces none.
     *
     * @throws ForkRefusal when the edited copy offers the play no line, no
     *                     source or no defender ({@link ForkRefusal#FORCED_NO_LINE})
     */
    public static ForcedPlayRun start(Application application, Binding fork, Game game) {
        ForcedPlay play = application.forced();
        if (play == null) {
            return null;
        }
        ForcedPlayRun run = new ForcedPlayRun(play, fork, game);
        if (!run.possible()) {
            throw new ForkRefusal(ForkRefusal.FORCED_NO_LINE);
        }
        return run;
    }

    /**
     * Why the variant's forced play cannot be made on the live board its
     * precondition bound, before the fork is taken, or empty when it may be:
     * its source holds the line, the named targets are legal for their
     * clauses (tried on a copy of the line, with throwaway draws for the
     * clauses named by no role), the named attack is legal. Mana is not
     * asked: a variant may give the mana the play needs.
     */
    public static Optional<String> dryRun(Application application, Binding live, Game game) {
        if (application.forced() == null) {
            return Optional.empty();
        }
        try {
            ForcedPlayRun trial = new ForcedPlayRun(application.forced(), live, game);
            return trial.possible() ? Optional.empty() : Optional.of(ForkRefusal.FORCED_NO_LINE);
        } catch (RuntimeException e) {
            // A role the binding lacks is the registry's error, which the
            // fork's own guard reports as a failure.
            return Optional.empty();
        }
    }

    /** The play this run makes. */
    public ForcedPlay play() {
        return play;
    }

    /** The player who makes the play, in the fork. */
    public Player actor() {
        return actor;
    }

    /** Whether the play has been made. */
    public boolean made() {
        return made;
    }

    /** Why the play cannot be made, or null while it still may be. */
    public String refusal() {
        return refusal;
    }

    private Card source() {
        return game.findById(sourceId);
    }

    /**
     * Whether the board offers the play at all: the source and its line, the
     * named targets legal on a trial copy of the line, or the named attack.
     */
    private boolean possible() {
        Card source = source();
        if (source == null) {
            return false;
        }
        return switch (play.kind()) {
            case CAST, ACTIVATE -> lines(source).stream().anyMatch(this::aimable);
            case ANSWER -> true;
            case ATTACK -> defender != null && source.isCreature() && source.isInPlay()
                    && source.getController() == actor && CombatUtil.canAttack(source, defender);
        };
    }

    /** Every ability of the actor's candidates hosted by {@code source} that is the play's line. */
    private List<SpellAbility> lines(Card source) {
        List<SpellAbility> lines = new ArrayList<>();
        for (PriorityCandidates.Candidate candidate : PriorityCandidates.candidatesOf(game, actor)) {
            if (isTheLine(candidate.ability(), source.getId()) && !modal(candidate.ability())) {
                lines.add(candidate.ability());
            }
        }
        return lines;
    }

    /** Whether {@code ability} is the play's line: hosted by the source and chosen by its {@link ForcedPlay.Line}. */
    private boolean isTheLine(SpellAbility ability, int source) {
        return ability.getHostCard() != null && ability.getHostCard().getId() == source
                && play.line().matches(ability);
    }

    /**
     * Whether {@code ability} is modal, a line no cast or activation is forced
     * to make: the engine's cast ({@code CharmEffect.makeChoices} in
     * {@link RandomSpellPlayer#play}) asks a charm's modes afresh, which would
     * undo the modes and named targets drawn here, so a forced charm has no
     * legal line.
     */
    private static boolean modal(SpellAbility ability) {
        return ability.getApi() == ApiType.Charm;
    }

    /** Whether the line's named targets are legal, tried on a copy with throwaway draws. */
    private boolean aimable(SpellAbility line) {
        try {
            SpellAbility trial = line.copy(line.getHostCard(), actor, true);
            trial.setActivatingPlayer(actor);
            return aim(trial, new RandomChoices(new Random(0L)), false);
        } catch (RuntimeException e) {
            return false;
        }
    }

    // ── the seat's decisions ────────────────────────────────────────────

    /**
     * Attach the play to the actor's controller in the fork: a seat that
     * knows forced plays takes it, a plain AI is replaced by a
     * {@link ForcingSeatController}, in the copy only.
     *
     * @param random the fork's own draws for the play
     */
    void attach(Random random) {
        this.choices = new RandomChoices(random);
        PlayerController controller = actor.getController();
        if (controller instanceof ForcedPlaySeat seat) {
            seat.force(this);
        } else if (controller instanceof PlayerControllerAi ai) {
            actor.dangerouslySetController(new ForcingSeatController(game, actor, ai, this));
        } else {
            throw new IllegalStateException("the forced seat " + actor + " is not an AI");
        }
    }

    /**
     * The line to play at {@code player}'s priority, or null when this
     * priority is not the play's: another player's, another step's, the play
     * made or refused, or its line not yet playable by the rules.
     */
    public List<SpellAbility> choose(Player player) {
        if (!play.playsALine() || player != actor || made || refusal != null || !inStep()) {
            return null;
        }
        Card source = source();
        if (source == null) {
            return null;
        }
        for (PriorityCandidates.Candidate candidate : PriorityCandidates.candidatesOf(game, actor)) {
            SpellAbility ability = candidate.ability();
            if (isTheLine(ability, sourceId) && !modal(ability) && candidate.playable() && aimable(ability)) {
                ability.setActivatingPlayer(actor);
                pending = ability;
                List<SpellAbility> chosen = new ArrayList<>(1);
                chosen.add(ability);
                return chosen;
            }
        }
        return null;
    }

    /**
     * Play {@code ability} if it is the line this run chose: its X and
     * targets set, its cost paid through the engine's own cast path.
     *
     * @return whether it was played, or null when it is not the run's
     */
    public Boolean playChosen(Player player, SpellAbility ability) {
        if (player != actor || ability != pending) {
            return null;
        }
        pending = null;
        if (!aim(ability, choices, true)) {
            refusal = ForkRefusal.FORCED_NO_LINE;
            return false;
        }
        // The rules said the line is affordable; the AI pays the mana, and
        // what it holds back can leave it short. Asked before the card moves.
        boolean affordable;
        try {
            affordable = ComputerUtilCost.canPayCost(ability, actor, false);
        } catch (RuntimeException e) {
            affordable = false;
        }
        if (!affordable || !RandomSpellPlayer.play(actor, ability, new RoleCostDecision(actor, ability, costs), null)) {
            refusal = ForkRefusal.FORCED_COST_NOT_PAID;
            return false;
        }
        made = true;
        return true;
    }

    /**
     * The play's answer to a confirmation {@code ability} asks of
     * {@code player}, reported through the engine's confirm hook as an AI
     * answer is, or null when the question is not the play's.
     */
    public Boolean answer(Player player, SpellAbility ability) {
        if (!asks(player, ability)) {
            return null;
        }
        made = true;
        return noted(ability, play.answer());
    }

    /**
     * The play's answer to a "you may pay" {@code ability} asks of
     * {@code player}, or null when the question is not the play's. A no
     * declines. A yes pays {@code cost} whenever it can be paid, whatever the
     * AI would have chosen: the AI's own payment, the named cards as what it
     * sacrifices, discards or exiles; one that cannot be paid, or whose
     * payment fails, flags {@link ForkRefusal#FORCED_COST_NOT_PAID}. Not
     * reported through the confirm hook, which the AI's own payment does not
     * report either.
     *
     * @return whether the cost was paid
     */
    public Boolean pay(Player player, Cost cost, SpellAbility ability) {
        if (!asks(player, ability)) {
            return null;
        }
        made = true;
        if (!play.answer()) {
            return false;
        }
        boolean paid;
        try {
            paid = ComputerUtilCost.canPayCost(cost, ability, actor, true)
                    && new CostPayment(cost, ability).payComputerCosts(
                            new RoleCostDecision(actor, ability, costs, true));
        } catch (RuntimeException e) {
            paid = false;
        }
        if (!paid) {
            refusal = ForkRefusal.FORCED_COST_NOT_PAID;
        }
        return paid;
    }

    /**
     * Whether {@code ability}'s question to {@code player} is the play's: an
     * answer not yet given, asked of its actor by its line on its source.
     * Only the game's own thread asks: Forge's AI evaluates its candidates on
     * a thread of its own, which resolves, pays and confirms nothing.
     */
    private boolean asks(Player player, SpellAbility ability) {
        return play.kind() == ForcedPlay.Kind.ANSWER && player == actor && !made && refusal == null
                && ability != null && isTheLine(ability, sourceId);
    }

    /**
     * Add the named attack to {@code player}'s declaration, once the seat
     * has declared its own; a declaration the engine rejects with it flags
     * the refusal.
     */
    public void declared(Player player, Combat combat) {
        if (play.kind() != ForcedPlay.Kind.ATTACK || player != actor || made || refusal != null) {
            return;
        }
        Card attacker = source();
        if (attacker == null || !attacker.isInPlay()) {
            refusal = ForkRefusal.FORCED_NO_LINE;
            return;
        }
        GameEntity declared = combat.getDefenderByAttacker(attacker);
        if (declared != defender) {
            if (declared != null) {
                combat.removeFromCombat(attacker);
            }
            if (!CombatUtil.canAttack(attacker, defender)) {
                refusal = ForkRefusal.FORCED_NO_LINE;
                return;
            }
            combat.addAttacker(attacker, defender);
            if (!CombatUtil.validateAttackers(combat)) {
                combat.removeFromCombat(attacker);
                refusal = ForkRefusal.FORCED_NO_LINE;
                return;
            }
        }
        made = true;
    }

    // ── the fork's checks ───────────────────────────────────────────────

    /**
     * Between main-loop steps: throw the refusal flagged, or the play not
     * made once its step has passed.
     */
    void check() {
        if (refusal != null) {
            throw new ForkRefusal(refusal);
        }
        if (!made && step != null && passed()) {
            throw new ForkRefusal(ForkRefusal.FORCED_NOT_MADE);
        }
    }

    /** After the turn: throw the refusal flagged, or the play not made. */
    void finish() {
        if (refusal != null) {
            throw new ForkRefusal(refusal);
        }
        if (!made) {
            throw new ForkRefusal(ForkRefusal.FORCED_NOT_MADE);
        }
    }

    private boolean inStep() {
        return game.getPhaseHandler().getTurn() == turn && game.getPhaseHandler().is(step);
    }

    private boolean passed() {
        return game.getPhaseHandler().getTurn() != turn
                || game.getPhaseHandler().getPhase().ordinal() > step.ordinal();
    }

    // ── choices ─────────────────────────────────────────────────────────

    /**
     * X and targets for {@code ability}, which is not modal: X drawn, each
     * targeting clause's targets the named objects, checked with
     * {@code canTarget}, or drawn where no role names them.
     *
     * @param pay whether X is drawn against what the actor can pay; a trial
     *            announces the least X, reading no mana
     */
    private boolean aim(SpellAbility ability, RandomChoices draws, boolean pay) {
        ability.setXManaCostPaid(null);
        draws.announceX(ability, pay ? actor : null, 0);
        List<SpellAbility> clauses = TargetingClauses.chosen(ability);
        if (targets.size() > clauses.size()
                && targets.subList(clauses.size(), targets.size()).stream().anyMatch(named -> !named.isEmpty())) {
            return false;
        }
        for (int i = 0; i < clauses.size(); i++) {
            SpellAbility clause = clauses.get(i);
            List<GameEntity> named = i < targets.size() ? targets.get(i) : List.of();
            if (named.isEmpty()) {
                if (!draws.chooseTargetsFor(clause)) {
                    return false;
                }
                continue;
            }
            clause.clearTargets();
            for (GameEntity target : named) {
                if (!clause.canTarget(target)) {
                    return false;
                }
                clause.getTargets().add(target);
            }
            if (!clause.isTargetNumberValid()) {
                return false;
            }
        }
        return true;
    }

    private static GameEntity entity(Binding binding, String role) {
        Card card = binding.cards().get(role);
        return card != null ? card : binding.player(role);
    }

    // ── the confirm hook ────────────────────────────────────────────────

    /** {@code PlayerControllerAi.noteEffectRecordConfirm}, private: the hook every AI answer is reported through. */
    private static final Method NOTE_CONFIRM;

    static {
        Method note;
        try {
            note = PlayerControllerAi.class.getDeclaredMethod("noteEffectRecordConfirm", SpellAbility.class,
                    boolean.class);
            note.setAccessible(true);
        } catch (ReflectiveOperationException | RuntimeException e) {
            note = null;
        }
        NOTE_CONFIRM = note;
    }

    /**
     * Report {@code answer} to {@code ability}'s question through the
     * engine's confirm hook, as the AI's own answers are, so the fork's
     * records read a forced answer as a live one.
     */
    static boolean noted(SpellAbility ability, boolean answer) {
        if (NOTE_CONFIRM == null) {
            throw new IllegalStateException("this Forge's PlayerControllerAi reports no confirmation");
        }
        try {
            return (Boolean) NOTE_CONFIRM.invoke(null, ability, answer);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        } catch (InvocationTargetException e) {
            if (e.getCause() instanceof RuntimeException runtime) {
                throw runtime;
            }
            throw new IllegalStateException(e.getCause());
        }
    }

    // ── the cost ────────────────────────────────────────────────────────

    /**
     * The AI's cost decision with what a sacrifice, discard or exile takes
     * named by role: the named cards the part may legally take, as many as
     * it takes; a part of the source itself, or with no card named, is the
     * AI's. A part the named cards cannot pay is not paid, so the play is
     * refused rather than made with another card. Mana stays the AI's.
     */
    static final class RoleCostDecision extends AiCostDecision {

        private final List<Card> named;

        RoleCostDecision(Player ai, SpellAbility sa, List<Card> named) {
            this(ai, sa, named, false);
        }

        /** @param effect whether the cost is an effect's ("unless ... pays"), not a play's */
        RoleCostDecision(Player ai, SpellAbility sa, List<Card> named, boolean effect) {
            super(ai, sa, effect);
            this.named = Objects.requireNonNull(named, "named");
        }

        @Override
        public PaymentDecision visit(CostSacrifice cost) {
            if (named.isEmpty() || cost.payCostFromSource()) {
                return super.visit(cost);
            }
            CardCollection valid = CardLists.getValidCards(player.getCardsIn(ZoneType.Battlefield),
                    cost.getType().split(";"), player, source, ability);
            return pick(CardLists.filter(valid, CardPredicates.canBeSacrificedBy(ability, isEffect())),
                    cost.getAbilityAmount(ability));
        }

        @Override
        public PaymentDecision visit(CostDiscard cost) {
            if (named.isEmpty() || cost.payCostFromSource()) {
                return super.visit(cost);
            }
            CardCollectionView hand = player.getCardsIn(ZoneType.Hand);
            CardCollection valid = CardLists.getValidCards(hand, cost.getType().split(";"), player, source, ability);
            return pick(CardLists.filter(valid, c -> c.canBeDiscardedBy(ability, isEffect())),
                    cost.getAbilityAmount(ability));
        }

        @Override
        public PaymentDecision visit(CostExile cost) {
            if (named.isEmpty() || cost.payCostFromSource()) {
                return super.visit(cost);
            }
            CardCollection valid = CardLists.getValidCards(player.getGame().getCardsIn(cost.getFrom()),
                    cost.getType().split(";"), player, source, ability);
            return pick(CardLists.filter(valid, CardPredicates.canExiledBy(ability, isEffect())),
                    cost.getAbilityAmount(ability));
        }

        /** {@code count} of the named cards {@code valid} holds, or null when it holds fewer. */
        private PaymentDecision pick(CardCollection valid, int count) {
            List<Card> chosen = named.stream()
                    .map(card -> player.getGame().getCardState(card))
                    .filter(valid::contains).limit(count).toList();
            if (count < 0 || chosen.size() < count) {
                return null;
            }
            return PaymentDecision.card(new CardCollection(chosen));
        }
    }
}
