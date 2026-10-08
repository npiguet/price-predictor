package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.PatchedCollectors;
import com.pricepredictor.connector.effects.RandomChoices;
import forge.LobbyPlayer;
import forge.ai.ComputerUtilAbility;
import forge.ai.ComputerUtilCost;
import forge.ai.PlayerControllerAi;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.cost.Cost;
import forge.game.player.Player;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.TargetChoices;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * The random seat: a Forge AI that, with probability {@code P} at each of
 * four decision points, draws uniformly from the legal options instead
 * (FR-022).
 *
 * <p>Every gen-1 outcome is one Forge chose, so the model never saw what
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
 * resolving effect asks its controller — is the AI's (FR-023). The nearest
 * relative is {@code ForkCollector}, which draws the same choices for an
 * ability it forces on a fork; it acts only <em>inside</em> a fork and this
 * acts only outside one, through the same test inverted: a controller is
 * live only once {@link GamePlayer} has attached it to the game it created,
 * and the controller a fork's {@code GameCopier} clone builds for its copy of
 * this seat is never attached, so every override there defers to
 * {@code super}.
 *
 * <p>When a draw succeeds the AI's decision code does not run, so the hooks
 * that would have written the {@code decision}, {@code attackers} and
 * {@code blockers} records never fire; the seat writes them itself through
 * {@link PatchedCollectors}' public emitters, from the lists it drew from
 * (FR-022d), stamping its own declarations {@code what_if = false}.
 */
public final class RandomSeatController extends PlayerControllerAi {

    private final double probability;
    private final Random random;
    private final RandomChoices choices;

    /** The game this seat plays for real, or null for a fork's clone. */
    private Game liveGame;
    /** Where the seat's own records go, or null when the run collects none. */
    private PatchedCollectors collectors;

    /** The play the last successful draw picked, until it is played. */
    private SpellAbility randomlyDrawn;
    /** The other playable candidates of that draw, for a redraw (FR-022c). */
    private List<SpellAbility> remainingCandidates = new ArrayList<>();
    /** Set when every candidate of a draw was abandoned: the next priority passes. */
    private boolean passNextPriority;

    public RandomSeatController(
            Game game, Player player, LobbyPlayer lobbyPlayer,
            double probability, Random random) {
        super(game, player, lobbyPlayer);
        this.probability = probability;
        this.random = random;
        this.choices = new RandomChoices(random);
    }

    /**
     * Make this seat live for {@code game}, writing its own records to
     * {@code collectors}.
     *
     * <p>Called by {@link GamePlayer} after {@code match.createGame()}, which
     * is the first moment both exist. A controller never attached — the one
     * a fork clones — stays an ordinary AI.
     *
     * @param collectors null when the run writes no effect records; the seat
     *                   still plays at random, it just reports nothing
     */
    public void attach(Game game, PatchedCollectors collectors) {
        this.liveGame = game;
        this.collectors = collectors;
    }

    /** Whether this controller is the live seat rather than a fork's copy. */
    boolean isLive() {
        return liveGame != null && player.getGame() == liveGame;
    }

    private boolean draw() {
        return random.nextDouble() < probability;
    }

    /** Whether the last play decision was a random draw, for a test of the gate. */
    boolean lastDrawWasRandom() {
        return randomlyDrawn != null;
    }

    // ── the play decision ───────────────────────────────────────────────

    /**
     * Every priority is a decision point with its own draw (FR-022a).
     *
     * <p>Land first (FR-022d): while a land drop is open and a land is
     * playable, no draw is made and the AI decides the priority, which keeps
     * land drops the AI's without the seat having to understand land choice.
     * Then the draw. On success the seat picks uniformly among what is
     * playable — never passing while anything is, and never a mana ability —
     * and passes when nothing is.
     */
    @Override
    public List<SpellAbility> chooseSpellAbilityToPlay() {
        if (!isLive()) {
            return super.chooseSpellAbilityToPlay();
        }
        randomlyDrawn = null;
        if (passNextPriority) {
            // Every candidate of the last draw was abandoned (FR-022c): the
            // engine asked again because the play returned false, and the
            // answer is a pass, not another draw.
            passNextPriority = false;
            return null;
        }
        if (landDropPending() || !draw()) {
            return super.chooseSpellAbilityToPlay();
        }
        List<SpellAbility> candidates = playableCandidates();
        if (candidates.isEmpty()) {
            return null;
        }
        remainingCandidates = candidates;
        randomlyDrawn = candidates.get(random.nextInt(candidates.size()));
        List<SpellAbility> chosen = new ArrayList<>(1);
        chosen.add(randomlyDrawn);
        return chosen;
    }

    /**
     * Whether the AI should decide this priority because a land could be
     * played (FR-022d): the drop is unused this turn, it is a main phase with
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
     * What the rules let this seat play now: the AI's own filter — can play,
     * timing, affordable — over everything but lands and mana abilities
     * (FR-022a), each candidate reported as a {@code decision} record through
     * the hook's own emitter (FR-022d).
     */
    private List<SpellAbility> playableCandidates() {
        Game game = player.getGame();
        List<SpellAbility> all = ComputerUtilAbility.getSpellAbilities(
                ComputerUtilAbility.getAvailableCards(game, player), player);
        List<SpellAbility> playable = new ArrayList<>();
        for (SpellAbility sa : all) {
            if (sa.isLandAbility() || sa.isManaAbility()) {
                continue;
            }
            boolean canPlay;
            boolean timing;
            boolean affordable;
            boolean hasTarget;
            try {
                sa.setActivatingPlayer(player);
                canPlay = sa.canPlay();
                timing = sa.canCastTiming(player);
                affordable = ComputerUtilCost.canPayCost(sa, player, false);
                hasTarget = hasLegalTargets(sa);
            } catch (RuntimeException e) {
                // A script whose restriction throws on this board is one the
                // AI would have skipped the same way; it is not playable.
                continue;
            }
            if (collectors != null) {
                collectors.recordCandidate(player, sa, canPlay, affordable, hasTarget);
            }
            if (canPlay && timing && affordable && hasTarget) {
                playable.add(sa);
            }
        }
        return playable;
    }

    /**
     * Whether every targeting clause of the chain can reach its minimum.
     * A charm's modes are not in the chain yet and are checked when drawn.
     */
    private static boolean hasLegalTargets(SpellAbility ability) {
        for (SpellAbility clause = ability; clause != null; clause = clause.getSubAbility()) {
            if (clause.usesTargeting()
                    && clause.getTargetRestrictions().getNumCandidates(clause)
                            < clause.getMinTargets()) {
                return false;
            }
        }
        return true;
    }

    /**
     * Cast or activate the chosen play.
     *
     * <p>A play the random draw picked makes every choice at random with no
     * further gate (FR-022c): modes, targets and X are drawn before the cast,
     * redrawn up to {@link RandomChoices#MAX_DRAWS} times when a draw leaves
     * a clause short of targets, and the additional costs are drawn as the
     * cast pays them. A play that cannot be completed is abandoned and the
     * draw repeats over the remaining candidates; when none remains the seat
     * passes at its next priority. A play the AI chose keeps the AI's own
     * choices, with its targets alone under their own draw (FR-022).
     */
    @Override
    public boolean playChosenSpellAbility(SpellAbility sa) {
        if (!isLive()) {
            return super.playChosenSpellAbility(sa);
        }
        if (sa != randomlyDrawn) {
            if (!sa.isLandAbility() && draw()) {
                redrawTargets(sa);
            }
            return super.playChosenSpellAbility(sa);
        }
        randomlyDrawn = null;
        SpellAbility current = sa;
        while (current != null) {
            if (playAtRandom(current)) {
                return true;
            }
            final SpellAbility abandoned = current;
            remainingCandidates.removeIf(candidate -> candidate == abandoned);
            current = remainingCandidates.isEmpty()
                    ? null
                    : remainingCandidates.get(random.nextInt(remainingCandidates.size()));
        }
        passNextPriority = true;
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
        if (sa.getApi() == ApiType.Charm) {
            sa.setChosenList(null);
            sa.setSubAbility(null);
            List<AbilitySub> modes = choices.drawModes(sa);
            if (modes == null) {
                return false;
            }
            sa.setChosenList(modes);
            for (AbilitySub mode : modes) {
                if (!choices.chooseTargets(mode)) {
                    return false;
                }
            }
        }
        if (!choices.chooseTargets(sa)) {
            return false;
        }
        choices.announceX(sa, player, 0);
        return true;
    }

    /**
     * Redraw the targets of a play the AI chose (FR-022), keeping the AI's
     * where the draw cannot complete.
     */
    private void redrawTargets(SpellAbility sa) {
        Map<SpellAbility, TargetChoices> kept = new LinkedHashMap<>();
        for (SpellAbility clause : targetingClausesOf(sa)) {
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
     * The clauses of a play that target: the root chain, or for a charm the
     * root plus the chosen modes' own chains — the clones the AI chained are
     * rebuilt from those originals at the cast.
     */
    private static List<SpellAbility> targetingClausesOf(SpellAbility sa) {
        List<SpellAbility> clauses = new ArrayList<>();
        if (sa.getApi() == ApiType.Charm) {
            if (sa.usesTargeting()) {
                clauses.add(sa);
            }
            if (sa.getChosenList() != null) {
                for (AbilitySub mode : sa.getChosenList()) {
                    addTargetingClauses(mode, clauses);
                }
            }
            return clauses;
        }
        addTargetingClauses(sa, clauses);
        return clauses;
    }

    private static void addTargetingClauses(SpellAbility chain, List<SpellAbility> into) {
        for (SpellAbility clause = chain; clause != null; clause = clause.getSubAbility()) {
            if (clause.usesTargeting()) {
                into.add(clause);
            }
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
     * targets them inside {@code doTrigger} — and stay the AI's: FR-022's
     * target decision point is the targets of a play.
     */
    @Override
    public boolean chooseTargetsFor(SpellAbility currentAbility) {
        if (isLive() && currentAbility.usesTargeting() && draw()
                && choices.chooseTargetsFor(currentAbility)) {
            return true;
        }
        return super.chooseTargetsFor(currentAbility);
    }

    // ── attacks ─────────────────────────────────────────────────────────

    /**
     * Each creature that can attack attacks with probability one half, at a
     * defender drawn among those it may attack (FR-022b).
     *
     * <p>The engine validates the whole declaration — requirements, maximums,
     * "can't attack alone" — and a declaration it rejects is redrawn; after
     * {@link RandomChoices#MAX_DRAWS} the AI declares. On success an
     * {@code attackers} record per defender says who could have attacked
     * and who was legal against it, stamped a real decision (FR-022d).
     */
    @Override
    public void declareAttackers(Player attacker, Combat combat) {
        if (!isLive() || attacker != player || !draw()) {
            super.declareAttackers(attacker, combat);
            return;
        }
        CardCollection mine = player.getCreaturesInPlay();
        Map<GameEntity, List<Card>> legalByDefender = new LinkedHashMap<>();
        for (GameEntity defender : combat.getDefenders()) {
            List<Card> legal = new ArrayList<>();
            for (Card creature : mine) {
                if (CombatUtil.canAttack(creature, defender)) {
                    legal.add(creature);
                }
            }
            legalByDefender.put(defender, legal);
        }
        if (collectors != null) {
            legalByDefender.forEach((defender, legal) ->
                    collectors.recordAttackers(player, defender, mine, legal, false));
        }
        for (int attempt = 0; attempt < RandomChoices.MAX_DRAWS; attempt++) {
            combat.clearAttackers();
            for (Card creature : mine) {
                List<GameEntity> options = new ArrayList<>();
                legalByDefender.forEach((defender, legal) -> {
                    if (legal.contains(creature)) {
                        options.add(defender);
                    }
                });
                if (!options.isEmpty() && random.nextBoolean()) {
                    combat.addAttacker(creature, options.get(random.nextInt(options.size())));
                }
            }
            removeUnpayableAttackers(combat);
            if (CombatUtil.validateAttackers(combat)) {
                return;
            }
        }
        combat.clearAttackers();
        super.declareAttackers(attacker, combat);
    }

    /**
     * Attackers whose attack cost the seat cannot pay are withdrawn before
     * validation, as the AI withdraws its own: the decision to pay a
     * Propaganda tax is made at declaration (CR 508.1d), and an attacker
     * left in that cannot pay is removed by the engine after the fact.
     */
    private void removeUnpayableAttackers(Combat combat) {
        Game game = player.getGame();
        for (Card attacker : new ArrayList<>(combat.getAttackers())) {
            Cost attackCost = CombatUtil.getAttackCost(
                    game, attacker, combat.getDefenderByAttacker(attacker));
            if (attackCost == null) {
                continue;
            }
            SpellAbility fakeSA = new SpellAbility.EmptySa(attacker, attacker.getController());
            fakeSA.setCardState(attacker.getCurrentState());
            fakeSA.setPayCosts(attackCost);
            fakeSA.setSVar("X", "0");
            if (!ComputerUtilCost.canPayCost(attackCost, fakeSA, player, true)) {
                combat.removeFromCombat(attacker);
            }
        }
    }

    // ── blocks ──────────────────────────────────────────────────────────

    /**
     * Each creature that can block picks uniformly from no block plus the
     * attackers it may legally block (FR-022b).
     *
     * <p>{@code CombatUtil.validateBlocks} is the engine's own check and the
     * engine never calls it, so the seat calls it before returning; a
     * declaration it rejects is redrawn, then left to the AI. On success a
     * {@code blockers} record per attacker says who could have blocked it
     * and how many it takes, stamped a real decision (FR-022d).
     */
    @Override
    public void declareBlockers(Player defender, Combat combat) {
        if (!isLive() || defender != player || !draw()) {
            super.declareBlockers(defender, combat);
            return;
        }
        List<Card> candidates = new ArrayList<>();
        for (Card creature : player.getCreaturesInPlay()) {
            if (CombatUtil.canBlock(creature, combat)) {
                candidates.add(creature);
            }
        }
        List<Card> attackers = new ArrayList<>();
        for (GameEntity defended : combat.getDefendersControlledBy(player)) {
            attackers.addAll(combat.getAttackersOf(defended));
        }
        Map<Card, List<Card>> legalByAttacker = new LinkedHashMap<>();
        for (Card attacker : attackers) {
            List<Card> legal = new ArrayList<>();
            for (Card blocker : candidates) {
                if (CombatUtil.canBlock(attacker, blocker, combat)) {
                    legal.add(blocker);
                }
            }
            legalByAttacker.put(attacker, legal);
        }
        if (collectors != null) {
            legalByAttacker.forEach((attacker, legal) -> collectors.recordBlockers(
                    player, attacker, candidates, legal,
                    CombatUtil.getMinNumBlockersForAttacker(attacker, player), false));
        }
        for (int attempt = 0; attempt < RandomChoices.MAX_DRAWS; attempt++) {
            clearBlocks(combat);
            for (Card blocker : candidates) {
                List<Card> options = new ArrayList<>();
                options.add(null);
                legalByAttacker.forEach((attacker, legal) -> {
                    if (legal.contains(blocker)) {
                        options.add(attacker);
                    }
                });
                Card attacker = options.get(random.nextInt(options.size()));
                if (attacker != null) {
                    combat.addBlocker(attacker, blocker);
                }
            }
            if (CombatUtil.validateBlocks(combat, player) == null) {
                return;
            }
        }
        clearBlocks(combat);
        super.declareBlockers(defender, combat);
    }

    private void clearBlocks(Combat combat) {
        for (Card blocker : new ArrayList<>(combat.getAllBlockers())) {
            if (blocker.getController() != player) {
                continue;
            }
            for (Card attacker : new ArrayList<>(combat.getAttackersBlockedBy(blocker))) {
                combat.removeBlockAssignment(attacker, blocker);
            }
        }
    }
}
