package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.RandomSpellPlayer;
import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.Role;
import forge.ai.AiCostDecision;
import forge.ai.ComputerUtilCost;
import forge.ai.PlayerControllerAi;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.PlayerController;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.Set;

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
 * ({@link RandomSpellPlayer#play}) with the AI's choices; a confirmation the
 * source's named line asks, the play's answer; a declaration of attackers,
 * the named attack added to the seat's own. Every other decision is the
 * seat's.
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
    private final NamedTargets targets;
    private final GameEntity defender;
    /**
     * The roles the binding cannot hold yet, a dry run's made roles
     * ({@link Application#makes}): what names them is left unchecked. Empty
     * in a fork.
     */
    private final Set<String> unknown;
    private RandomChoices choices;
    /** The line the actor's last priority chose, until it is played. */
    private SpellAbility pending;
    private boolean made;
    /** Why the play cannot be made, once known. */
    private String refusal;

    private ForcedPlayRun(ForcedPlay play, Binding fork, Game game) {
        this(play, fork, game, Set.of());
    }

    private ForcedPlayRun(ForcedPlay play, Binding fork, Game game, Set<String> unknown) {
        this.play = play;
        this.game = game;
        this.unknown = unknown;
        this.actor = fork.player(play.actor());
        this.sourceId = unknown.contains(play.source().name()) ? -1 : fork.card(play.source()).getId();
        this.step = stepOf(play.kind(), game);
        this.turn = game.getPhaseHandler().getTurn();
        this.targets = new NamedTargets(actor, namedTargets(play, fork, unknown));
        this.defender = namedDefender(play, fork, unknown);
    }

    /** The step a play of {@code kind} is made in, read in {@code game}'s current one. */
    private static PhaseType stepOf(ForcedPlay.Kind kind, Game game) {
        return switch (kind) {
            case ANSWER -> null;
            case ATTACK -> PhaseType.COMBAT_DECLARE_ATTACKERS;
            case CAST, ACTIVATE -> game.getPhaseHandler().getPhase();
        };
    }

    /** Per targeting clause, the named objects; a clause naming an unknown role is null: left unchecked. */
    private static List<List<GameEntity>> namedTargets(ForcedPlay play, Binding fork, Set<String> unknown) {
        List<List<GameEntity>> targets = new ArrayList<>();
        for (List<Role<?>> clause : play.targets()) {
            targets.add(clause.stream().anyMatch(role -> unknown.contains(role.name())) ? null
                    : clause.stream().map(role -> entity(fork, role.name())).toList());
        }
        return targets;
    }

    /** The named defender, or null when the play names none or names an unknown role. */
    private static GameEntity namedDefender(ForcedPlay play, Binding fork, Set<String> unknown) {
        if (play.defender() == null || unknown.contains(play.defender().name())) {
            return null;
        }
        return entity(fork, play.defender().name());
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
     * asked: a variant may give the mana the play needs. A role the edit
     * makes ({@link Application#makes}) has no live card: what names it (the
     * line, when it is the source; a targeting clause; the defender)
     * is checked on the copy only, and the rest of the play here.
     */
    public static Optional<String> dryRun(Application application, Binding live, Game game) {
        if (application.forced() == null) {
            return Optional.empty();
        }
        try {
            ForcedPlayRun trial = new ForcedPlayRun(application.forced(), live, game, application.makes());
            return trial.possible() ? Optional.empty() : Optional.of(ForkRefusal.FORCED_NO_LINE);
        } catch (RuntimeException e) {
            // A role neither bound nor made is the registry's error, which
            // the fork's own guard reports as a failure.
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
        if (unknown.contains(play.source().name())) {
            return true;
        }
        Card source = source();
        if (source == null) {
            return false;
        }
        return switch (play.kind()) {
            case CAST, ACTIVATE -> lines(source).stream().anyMatch(this::aimable);
            case ANSWER -> true;
            case ATTACK -> actorsCreatureInPlay(source) && reachesDefender(source);
        };
    }

    private boolean actorsCreatureInPlay(Card source) {
        return source.isCreature() && source.isInPlay() && source.getController() == actor;
    }

    /** Whether {@code attacker} may attack the named defender; an unknown one is left unchecked. */
    private boolean reachesDefender(Card attacker) {
        return unknown.contains(play.defender().name())
                || defender != null && CombatUtil.canAttack(attacker, defender);
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
            return targets.aim(trial, new RandomChoices(new Random(0L)), false);
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
        if (!awaitsLine(player) || source() == null) {
            return null;
        }
        SpellAbility ability = playableLine();
        if (ability == null) {
            return null;
        }
        ability.setActivatingPlayer(actor);
        pending = ability;
        List<SpellAbility> chosen = new ArrayList<>(1);
        chosen.add(ability);
        return chosen;
    }

    /** Whether {@code player}'s priority may be the play's line: its actor's, in its step, the play open. */
    private boolean awaitsLine(Player player) {
        return play.playsALine() && awaits(player) && inStep();
    }

    /** Whether {@code player} is the actor and the play is neither made nor refused. */
    private boolean awaits(Player player) {
        return player == actor && !made && refusal == null;
    }

    /** The first of the actor's candidates that is the line, playable by the rules, its named targets legal; or null. */
    private SpellAbility playableLine() {
        for (PriorityCandidates.Candidate candidate : PriorityCandidates.candidatesOf(game, actor)) {
            if (playableLine(candidate)) {
                return candidate.ability();
            }
        }
        return null;
    }

    private boolean playableLine(PriorityCandidates.Candidate candidate) {
        SpellAbility ability = candidate.ability();
        return isTheLine(ability, sourceId) && !modal(ability) && candidate.playable() && aimable(ability);
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
        if (!targets.aim(ability, choices, true)) {
            refusal = ForkRefusal.FORCED_NO_LINE;
            return false;
        }
        if (!affordable(ability) || !RandomSpellPlayer.play(actor, ability, new AiCostDecision(actor, ability, false), null)) {
            refusal = ForkRefusal.FORCED_COST_NOT_PAID;
            return false;
        }
        made = true;
        return true;
    }

    /**
     * Whether the AI can pay {@code ability}'s cost. The rules said the line
     * is affordable; the AI pays the mana, and what it holds back can leave
     * it short. Asked before the card moves.
     */
    private boolean affordable(SpellAbility ability) {
        try {
            return ComputerUtilCost.canPayCost(ability, actor, false);
        } catch (RuntimeException e) {
            return false;
        }
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
     * Whether {@code ability}'s question to {@code player} is the play's: an
     * answer not yet given, asked of its actor by its line on its source.
     * Only the game's own thread asks: Forge's AI evaluates its candidates on
     * a thread of its own, which resolves, pays and confirms nothing.
     */
    private boolean asks(Player player, SpellAbility ability) {
        return play.kind() == ForcedPlay.Kind.ANSWER && awaits(player)
                && ability != null && isTheLine(ability, sourceId);
    }

    /**
     * Add the named attack to {@code player}'s declaration, once the seat
     * has declared its own; a declaration the engine rejects with it flags
     * the refusal.
     */
    public void declared(Player player, Combat combat) {
        if (play.kind() != ForcedPlay.Kind.ATTACK || !awaits(player)) {
            return;
        }
        Card attacker = source();
        if (attacker == null || !attacker.isInPlay()) {
            refusal = ForkRefusal.FORCED_NO_LINE;
            return;
        }
        GameEntity declared = combat.getDefenderByAttacker(attacker);
        if (declared != defender && !redeclared(attacker, declared, combat)) {
            refusal = ForkRefusal.FORCED_NO_LINE;
            return;
        }
        made = true;
    }

    /**
     * Move {@code attacker}, declared at {@code declared} (or not at all),
     * to the named defender; whether the engine accepts the declaration. A
     * rejected one leaves the attacker out of combat.
     */
    private boolean redeclared(Card attacker, GameEntity declared, Combat combat) {
        if (declared != null) {
            combat.removeFromCombat(attacker);
        }
        if (!CombatUtil.canAttack(attacker, defender)) {
            return false;
        }
        combat.addAttacker(attacker, defender);
        if (CombatUtil.validateAttackers(combat)) {
            return true;
        }
        combat.removeFromCombat(attacker);
        return false;
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
        if (overdue()) {
            throw new ForkRefusal(ForkRefusal.FORCED_NOT_MADE);
        }
    }

    /** Whether the play, made in a step, is not made and its step has passed. */
    private boolean overdue() {
        return !made && step != null && passed();
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

    // ── roles ───────────────────────────────────────────────────────────

    /** The card {@code binding} holds for {@code role}, else its player. */
    private static GameEntity entity(Binding binding, String role) {
        Card card = binding.cards().get(role);
        return card != null ? card : binding.player(role);
    }

    // ── the confirm hook ────────────────────────────────────────────────

    /**
     * Report {@code answer} to {@code ability}'s question through the
     * engine's confirm hook ({@link PatchHooks#NOTE_CONFIRM}), as the AI's
     * own answers are, so the fork's records read a forced answer as a live
     * one. Throws on a Forge without the hook, which the report names.
     */
    static boolean noted(SpellAbility ability, boolean answer) {
        return (Boolean) PatchHooks.NOTE_CONFIRM.call(ability, answer);
    }
}
