package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.ForcedPlayRun;
import com.pricepredictor.connector.effects.ForcedPlaySeat;
import com.pricepredictor.connector.effects.FruitlessActivations;
import com.pricepredictor.connector.effects.LentActivator;
import com.pricepredictor.connector.effects.MandatoryChoiceAi;
import com.pricepredictor.connector.effects.PriorityCandidates;
import com.pricepredictor.connector.effects.RandomChoices;
import com.pricepredictor.connector.effects.TargetingClauses;
import forge.LobbyPlayer;
import forge.ai.ComputerUtilAbility;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.combat.Combat;
import forge.game.player.Player;
import forge.game.player.PlayerActionConfirmMode;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.TargetChoices;
import forge.game.trigger.WrappedAbility;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.function.Supplier;

/**
 * The random seat: a Forge AI that, with probability {@code P} at each of
 * four decision points, draws uniformly from the legal options instead
 * (024 FR-022).
 *
 * <p>Every outcome of an ordinary game is one Forge chose, so the model never saw what
 * happens when an ability is aimed at a target Forge would not pick or a
 * creature attacks into a board Forge would hold it back from. This seat
 * supplies those outcomes, and it supplies them <em>off-policy</em> rather
 * than as a different policy: on every draw that fails it is the ordinary
 * AI, so a match with one random seat is still mostly a Forge game, with a
 * few decisions the engine's own legality is the only judge of.
 *
 * <p>The four overrides are {@link #chooseSpellAbilityToPlay},
 * {@link #playChosenSpellAbility}, {@link #declareAttackers} and
 * {@link #declareBlockers}, plus {@link #chooseTargetsFor} for the clauses
 * whose targeting the engine asks for on its own. Everything else — land drops, mana payment, mulligans, what a
 * resolving effect asks its controller — is the AI's (024 FR-023), but for a
 * mandatory single choice the AI declines, which the seat draws among the
 * options offered ({@link MandatoryChoiceAi}). The nearest
 * relative is {@code InterventionFork}, which draws the same choices for an
 * ability it forces on a fork; it acts only <em>inside</em> a fork and this
 * acts only outside one, through the same test inverted: a controller is
 * live only once {@link GamePlayer} has attached it to the game it created.
 * The controller a fork's {@code GameCopier} clone builds for its copy of
 * this seat is attached only when the fork is played on to the end of the
 * turn ({@link GamePlayer#attachForkSeats}); in an intervention or a probe it
 * never is, so every override there defers to {@code super}.
 *
 * <p>The seat writes no record of its own. The {@code decision} records come
 * from {@link PriorityCandidates}, which enumerates both seats' candidates at
 * every priority stop, and the seat draws its plays from that same stop, by
 * the verdicts its records carry, so the two never disagree on
 * {@code can_play} (FR-033, FR-040). The {@code attackers} and
 * {@code blockers} records come from the engine's turn-based declaration
 * itself, before any controller declares, so the seat's declarations are
 * recorded like every other.
 *
 * <p>In a directed fork whose variant forces a play of this seat (FR-010b),
 * the seat consults the play first ({@link ForcedPlaySeat}): at the
 * decisions the play is made in it answers as the play says, attached or
 * not, and everywhere else it is the random seat it would be.
 */
public final class RandomSeatController extends MandatoryChoiceAi implements ForcedPlaySeat {

    private final double probability;
    /**
     * The seat's draws for the game it plays in, drawn from the match's
     * source when the live game attaches it and never for a fork, so a fork
     * leaves later games' seats as they would have been.
     */
    private final Supplier<Random> liveRandom;
    /** The seat's draws, set when it is attached; null before. */
    private Random random;
    private RandomChoices choices;

    /** The game this seat plays for real, or null for a fork's clone. */
    private Game liveGame;
    /** The game's priority stops, or null when the run collects no records. */
    private PriorityCandidates candidates;
    /**
     * The free activations that changed nothing this turn, or null when the
     * run collects no records (FR-040).
     */
    private FruitlessActivations fruitless;

    /** The play decision's draw, carried from one engine call to the next; set when attached. */
    private RandomPlayDraw playDraw;
    /** The seat's attack and block draws; set when attached. */
    private RandomCombatDeclarations combatDraws;

    /** A directed fork's forced play of this seat, or null. */
    private ForcedPlayRun forced;

    /**
     * @param liveRandom the seat's draws for the live game, asked for once,
     *                   when {@link GamePlayer} attaches it
     */
    public RandomSeatController(
            Game game, Player player, LobbyPlayer lobbyPlayer,
            double probability, Supplier<Random> liveRandom) {
        super(game, player, lobbyPlayer);
        this.probability = probability;
        this.liveRandom = liveRandom;
    }

    /**
     * Make this seat live for {@code game}, drawing its plays from
     * {@code candidates}.
     *
     * <p>Called by {@link GamePlayer} after {@code match.createGame()}, which
     * is the first moment both exist; the seat's draws come from the match's
     * source now. A controller never attached — the one an intervention or a
     * probe clones — stays an ordinary AI.
     *
     * @param candidates null when the run writes no effect records; the seat
     *                   still plays at random, enumerating its own candidates
     * @param fruitless  the collector's report of free activations that
     *                   changed nothing this turn, or null with no records
     */
    public void attach(Game game, PriorityCandidates candidates, FruitlessActivations fruitless) {
        attach(game, candidates, fruitless, liveRandom.get());
    }

    /**
     * Make this seat live for {@code game} with draws of its own: a fork's
     * copy of the seat, played on with the seat choosing afresh (D4), takes
     * a {@code Random} seeded from the game's fork seed rather than one from
     * the match's source.
     */
    public void attach(
            Game game, PriorityCandidates candidates, FruitlessActivations fruitless, Random random) {
        this.liveGame = game;
        this.candidates = candidates;
        this.fruitless = fruitless;
        this.random = random;
        this.choices = new RandomChoices(random);
        this.playDraw = new RandomPlayDraw(random);
        this.combatDraws = new RandomCombatDeclarations(player, random);
    }

    @Override
    public void force(ForcedPlayRun run) {
        this.forced = run;
    }

    /** Whether this controller is the live seat rather than a fork's copy. */
    boolean isLive() {
        return liveGame != null && player.getGame() == liveGame;
    }

    private boolean draw() {
        return random.nextDouble() < probability;
    }

    /**
     * A mandatory choice the AI declined, drawn uniformly from the offered
     * options when live; a fork's unattached copy takes the first, as the AI
     * it is there.
     */
    @Override
    protected int optionIndex(int size) {
        return isLive() ? random.nextInt(size) : super.optionIndex(size);
    }

    /**
     * Every priority is a decision point with its own draw (024 FR-022a).
     *
     * <p>Land first (024 FR-022d): while a land drop is open and a land is
     * playable, no draw is made and the AI decides the priority, which keeps
     * land drops the AI's without the seat having to understand land choice.
     * Then the draw. On success the seat picks uniformly among what is
     * playable — never passing while anything is, and never a mana ability —
     * and passes when nothing is.
     */
    @Override
    public List<SpellAbility> chooseSpellAbilityToPlay() {
        List<SpellAbility> line = forced == null ? null : forced.choose(player);
        if (line != null) {
            return line;
        }
        if (!isLive()) {
            return super.chooseSpellAbilityToPlay();
        }
        if (playDraw.beginPriority()) {
            return null;
        }
        if (landDropPending() || !draw()) {
            return super.chooseSpellAbilityToPlay();
        }
        List<SpellAbility> candidates = playableCandidates();
        if (candidates.isEmpty()) {
            return null;
        }
        List<SpellAbility> chosen = new ArrayList<>(1);
        chosen.add(playDraw.draw(candidates));
        return chosen;
    }

    /**
     * Whether the AI should decide this priority because a land could be
     * played (024 FR-022d): the drop is unused this turn, it is a main phase with
     * an empty stack, and a land in hand may be played — all of which
     * {@code getAvailableLandsToPlay} already checks.
     */
    private boolean landDropPending() {
        CardCollection lands = ComputerUtilAbility.getAvailableLandsToPlay(
                player.getGame(), player);
        if (lands == null) {
            return false;
        }
        // The hand's lands come back already checked against the drop; a land
        // another zone lets the seat play (the top of the library, an exiled
        // card) comes back on its permission alone, and without this check
        // would hold the draw off for the rest of a turn whose drop is used.
        for (Card land : lands) {
            if (player.canPlayLand(land, false, land.getFirstSpellAbility())) {
                return true;
            }
        }
        return false;
    }

    /**
     * What the rules let this seat play now (024 FR-022a): the candidates of this
     * priority stop whose {@link com.pricepredictor.connector.effects.PlayabilityVerdict}
     * says they can be played, paid for and aimed — the stop the decision
     * records were written of when the run records, so the seat plays only
     * what its records call playable. Of those it leaves out a land play,
     * which stays the AI's ({@link #landDropPending}, FR-033a), and a free
     * activation that changed nothing earlier this turn (FR-040): still
     * playable, and still recorded so, but not drawn again until the turn
     * ends. A play whose life payment would leave the seat at 0 or less is
     * left out the same way ({@link LifeCostGate}), as is an attach whose only
     * legal target is what its host is attached to ({@link RepeatAttachGate}).
     */
    private List<SpellAbility> playableCandidates() {
        List<PriorityCandidates.Candidate> stop = candidates == null
                ? PriorityCandidates.candidatesOf(player.getGame(), player)
                : candidates.takeFor(player);
        List<SpellAbility> playable = new ArrayList<>();
        for (PriorityCandidates.Candidate candidate : stop) {
            // A script whose check throws on this board reads as unplayable,
            // as the AI would have skipped it.
            if (seatMayDraw(candidate)) {
                // The verdict put the ability's activating player back; the
                // seat is about to play it, which is the seat's to write.
                candidate.ability().setActivatingPlayer(player);
                playable.add(candidate.ability());
            }
        }
        return playable;
    }

    /** Whether the seat may draw {@code candidate}: playable, no land, not barred (see above). */
    private boolean seatMayDraw(PriorityCandidates.Candidate candidate) {
        SpellAbility ability = candidate.ability();
        return candidate.playable()
                && !ability.isLandAbility()
                && LentActivator.holds(ability, player, () -> notBarred(ability));
    }

    /** The gates, read inside the seat's lend of the activating player ({@link LentActivator}). */
    private boolean notBarred(SpellAbility ability) {
        return !LifeCostGate.wouldLeaveNoLife(ability, player)
                && !RepeatAttachGate.attachesOnlyToWhatItIsOn(ability)
                && (fruitless == null || !fruitless.isFruitless(ability));
    }

    /**
     * Cast or activate the chosen play.
     *
     * <p>A play the random draw picked makes every choice at random with no
     * further gate (024 FR-022c): modes, targets and X are drawn before the cast,
     * redrawn up to {@link RandomChoices#MAX_DRAWS} times when a draw leaves
     * a clause short of targets, and the additional costs are drawn as the
     * cast pays them. A play that cannot be completed is abandoned and the
     * draw repeats over the remaining candidates; when none remains the seat
     * passes at its next priority. A play the AI chose keeps the AI's own
     * choices, with its targets alone under their own draw (024 FR-022).
     */
    @Override
    public boolean playChosenSpellAbility(SpellAbility sa) {
        Boolean played = forced == null ? null : forced.playChosen(player, sa);
        if (played != null) {
            return played;
        }
        if (!isLive()) {
            return super.playChosenSpellAbility(sa);
        }
        return playDraw.claims(sa) ? playDrawn(sa) : playAiChoice(sa);
    }

    /** A play the AI chose, with its targets alone under their own draw. */
    private boolean playAiChoice(SpellAbility sa) {
        if (!sa.isLandAbility() && draw()) {
            redrawTargets(sa);
        }
        return super.playChosenSpellAbility(sa);
    }

    /**
     * The play the random draw picked, then each candidate redrawn over the
     * rest while one cannot be completed; with none completed the seat owes
     * a pass.
     */
    private boolean playDrawn(SpellAbility sa) {
        for (SpellAbility current = sa; current != null;
                current = playDraw.nextAfterAbandon(current)) {
            if (playAtRandom(current)) {
                return true;
            }
        }
        playDraw.exhausted();
        return false;
    }

    /**
     * One candidate, drawn and cast.
     *
     * <p>The draws happen before the cast and are retried on their own; the
     * cast itself is tried once, because past the move to the stack a
     * failure is the engine's to recover from, not a draw to repeat.
     */
    private boolean playAtRandom(SpellAbility sa) {
        for (int attempt = 0; attempt < RandomChoices.MAX_DRAWS; attempt++) {
            if (!drawPlayChoices(sa)) {
                continue;
            }
            // The candidate was judged affordable by the rules alone; the AI
            // pays the mana, and what it holds back or what the cost itself
            // sacrifices can leave it short. Asked before the card moves to
            // the stack, so a play the AI would fail to pay is a draw to
            // repeat (an X drawn lower may fit), never a card left in the
            // stack zone.
            if (!affordableByTheAi(sa)) {
                continue;
            }
            try {
                return RandomSpellPlayer.play(
                        player, sa,
                        new RandomSpellPlayer.RandomCostDecision(player, sa, random), null);
            } catch (RuntimeException e) {
                // The AI reaches the same states on the same cards; a throw
                // here is a card the engine cannot play, not a draw to repeat.
                System.err.println("Random seat: " + sa.getHostCard() + " could not be played ("
                        + e + "); abandoning it");
                return false;
            }
        }
        return false;
    }

    /** Whether the AI, which pays the seat's mana, can pay {@code sa} as drawn. */
    private boolean affordableByTheAi(SpellAbility sa) {
        try {
            return forge.ai.ComputerUtilCost.canPayCost(sa, player, false);
        } catch (RuntimeException e) {
            return false;
        }
    }

    /**
     * Modes, targets and X for a randomly drawn play, all drawn fresh.
     *
     * <p>A charm's modes are drawn as the chosen list and their targets set
     * on the charm's own {@code Choices$} entries, which is where the cast's
     * {@code makeChoices} clones them from — a clone copies its original's
     * targets, so the draw reaches the mode that resolves. The charm's chain
     * is cleared first because the AI may have chained an earlier evaluation's
     * modes onto this same ability object.
     */
    private boolean drawPlayChoices(SpellAbility sa) {
        sa.setXManaCostPaid(null);
        if (sa.getApi() == ApiType.Charm && !drawCharmModes(sa)) {
            return false;
        }
        // X before the targets (601.2b, 601.2c): a divided amount of X, and
        // a bound across the targets that reads X (MaxTotalTargetCMC), are
        // read as the targets are drawn.
        choices.announceX(sa, player, 0);
        return drawModeTargets(sa) && choices.chooseTargets(sa);
    }

    /** A charm's modes, drawn afresh as its chosen list; false when none can be drawn. */
    private boolean drawCharmModes(SpellAbility charm) {
        charm.setChosenList(null);
        charm.setSubAbility(null);
        Optional<List<AbilitySub>> modes = choices.drawModes(charm);
        if (modes.isEmpty()) {
            return false;
        }
        charm.setChosenList(modes.get());
        return true;
    }

    /** The targets of each chosen mode, in order; false at the first a draw leaves short. */
    private boolean drawModeTargets(SpellAbility sa) {
        List<AbilitySub> modes = sa.getChosenList();
        if (modes == null) {
            return true;
        }
        for (AbilitySub mode : modes) {
            if (!choices.chooseTargets(mode)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Redraw the targets of a play the AI chose (024 FR-022), keeping the AI's
     * where the draw cannot complete.
     */
    private void redrawTargets(SpellAbility sa) {
        Map<SpellAbility, TargetChoices> kept = new LinkedHashMap<>();
        for (SpellAbility clause : TargetingClauses.chosen(sa)) {
            kept.put(clause, clause.getTargets());
        }
        boolean drawn = true;
        for (SpellAbility clause : kept.keySet()) {
            if (!choices.chooseTargetsFor(clause)) {
                drawn = false;
                break;
            }
        }
        if (!drawn) {
            kept.forEach(SpellAbility::setTargets);
        }
    }

    /**
     * Targets for one clause whose targeting the engine asks for separately:
     * {@code SpellAbility.setupTargets}, a {@code TargetingPlayer} trigger,
     * a generic choice's random pick.
     *
     * <p>One clause, not the chain from it: {@code setupTargets} walks the
     * chain itself and calls this per targeting clause, so drawing the rest
     * here would be redrawn the moment it reached them. Most of the AI's
     * triggers never come here — {@code PlayerControllerAi.playTrigger}
     * targets them inside {@code doTrigger} — and stay the AI's: 024 FR-022's
     * target decision point is the targets of a play.
     */
    @Override
    public boolean chooseTargetsFor(SpellAbility currentAbility) {
        if (drewTargetsFor(currentAbility)) {
            return true;
        }
        return super.chooseTargetsFor(currentAbility);
    }

    /**
     * Whether the live seat drew to choose {@code clause}'s targets at random
     * and the draw completed; a clause that targets nothing takes no draw.
     */
    private boolean drewTargetsFor(SpellAbility clause) {
        if (!isLive() || !clause.usesTargeting()) {
            return false;
        }
        return draw() && choices.chooseTargetsFor(clause);
    }

    /**
     * Attacks drawn at random when the live seat draws to (024 FR-022b,
     * {@link RandomCombatDeclarations#declareAttackers}), else the AI's; a
     * forced attack is added after either.
     */
    @Override
    public void declareAttackers(Player attacker, Combat combat) {
        declareOwnAttackers(attacker, combat);
        if (forced != null) {
            forced.declared(attacker, combat);
        }
    }

    /**
     * A forced play's answer to an optional trigger of its source, else the
     * AI's.
     */
    @Override
    public boolean confirmTrigger(WrappedAbility wrapper) {
        Boolean answer = forced == null || wrapper.isMandatory()
                ? null : forced.answer(player, wrapper.getWrappedAbility());
        return answer != null ? answer : super.confirmTrigger(wrapper);
    }

    /** A forced play's answer to its source's "you may", else the AI's. */
    @Override
    public boolean confirmAction(SpellAbility sa, PlayerActionConfirmMode mode, String message,
                                 List<String> options, Card cardToShow, Map<String, Object> params) {
        Boolean answer = forced == null ? null : forced.answer(player, sa);
        return answer != null ? answer : super.confirmAction(sa, mode, message, options, cardToShow, params);
    }

    /** The seat's own declaration, before a forced attack is added. */
    private void declareOwnAttackers(Player attacker, Combat combat) {
        if (drawsToDeclare(attacker)) {
            combatDraws.declareAttackers(combat, () -> super.declareAttackers(attacker, combat));
            return;
        }
        super.declareAttackers(attacker, combat);
    }

    /**
     * Blocks drawn at random when the live seat draws to (024 FR-022b,
     * {@link RandomCombatDeclarations#declareBlockers}), else the AI's.
     */
    @Override
    public void declareBlockers(Player defender, Combat combat) {
        if (drawsToDeclare(defender)) {
            combatDraws.declareBlockers(combat, () -> super.declareBlockers(defender, combat));
            return;
        }
        super.declareBlockers(defender, combat);
    }

    /**
     * Whether the live seat declares at random for {@code declaring}: its own
     * declaration, and the draw succeeds; a fork's copy or another player's
     * declaration takes no draw.
     */
    private boolean drawsToDeclare(Player declaring) {
        if (!isLive() || declaring != player) {
            return false;
        }
        return draw();
    }
}
