package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.cost.Cost;
import forge.game.cost.CostUntap;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.AbilityStatic;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.function.Predicate;

/**
 * Which ability nobody played is forced, at a step's first priority, and
 * when a permanent is granted a rare protection (FR-058), as a first main
 * phase begins.
 *
 * <p>Some abilities appear in no record however long collection runs: the AI
 * judges them bad, cannot afford them, or never draws them into a board where
 * they do anything. Observation cannot reach those, so the corpus forks the
 * game and makes one happen.
 *
 * <p>Forced at the first moment of a step that priority settles with the
 * stack empty ({@link #prioritySink}), never at the step's phase event:
 * Forge publishes that event before the step's turn-based action (the draw,
 * the attack, the combat damage) and also for a step the turn then skips,
 * so a line forced there resolved on a board no player is given priority
 * on. Nor from the playability hook, even though it knows exactly which
 * candidates were declined: it fires inside the AI's own evaluation, and
 * forking the game there means copying a game that is mid-decision and
 * re-entering the rules engine underneath it. Priority settles after the
 * state-based check and before the player holding it chooses: the same
 * board with nothing in flight.
 *
 * <p>Costs a game copy and a stack resolution each, which is why
 * {@code --interventions-per-game} defaults low and why nothing happens at all
 * when it is zero.
 *
 * <p>Spread over the game rather than spent where it starts ({@link
 * #spreadOverTheGame}): each game turn draws, once, whether it takes one of the
 * budget's interventions, with a chance that spreads what is left of the
 * budget over the turns left until a typical game's end ({@link
 * #TYPICAL_GAME_TURNS}); a drawn intervention not yet forced is carried to
 * the next boundary that has a line to force. A boundary forces one line at
 * most, drawn at random from the fork's stream among those it can force, and
 * only a line its own restrictions let its holder play there but for its
 * cost ({@link #playableNow}).
 */
final class InterventionScheduler {

    /**
     * What forces a line: a fork of the game with the line resolved, under the
     * game's fork budget ({@link InterventionFork}).
     */
    interface Forcing {

        /** Whether the budget lets {@link #intervene} take another fork of this resolution. */
        boolean mayIntervene(int realResolutionId);

        /**
         * Put a boundary's lines in the order it tries them, drawn from the
         * fork's own stream: the first it can force is then a draw at random
         * among those it can force. Left in zone order by default.
         */
        default void shuffle(List<SpellAbility> lines) {
        }

        /**
         * Whether {@code ability} has the modes and targets a forced
         * resolution needs, for {@code holder}, on the board as it stands:
         * one that has none would spend a fork on a resolution that cannot
         * happen. Asked of the live board without changing it.
         */
        default boolean hasChoices(SpellAbility ability, Player holder) {
            return true;
        }

        /**
         * Fork the game and force {@code ability}, resolved for {@code holder}.
         *
         * @return whether a record was written
         */
        boolean intervene(SpellAbility ability, int realResolutionId, Player holder);
    }

    /**
     * Whether a boundary may force a line: asked once per boundary with the
     * game's turn, and told when a line was forced there.
     */
    interface Gate {
        /** Whether this boundary, {@code phase} of game turn {@code turn}, may force a line. */
        boolean open(int turn, PhaseType phase);

        /** A line was forced at the boundary the gate last opened. */
        default void forced() {
        }
    }

    /**
     * The game turns a typical game lasts, counting each player's turn: what
     * the budget is spread over. A game that lasts longer forces what is
     * left one turn at a time from this turn on; a shorter one may end with
     * some of it unspent.
     */
    static final int TYPICAL_GAME_TURNS = 20;

    /** A gate every boundary passes, for a caller that forces at each one. */
    static final Gate EVERY_BOUNDARY = (turn, phase) -> true;

    /**
     * The steps an intervention drawn for a turn may open at, in turn order:
     * those that give priority and where a fork's score check can pass. One
     * is drawn with the intervention, and the gate opens from that step on,
     * so instant-speed lines are forced across the turn rather than all at
     * its first step.
     */
    static final List<PhaseType> OPENING_STEPS = List.of(
            PhaseType.UPKEEP, PhaseType.DRAW, PhaseType.MAIN1,
            PhaseType.COMBAT_DECLARE_ATTACKERS, PhaseType.COMBAT_DECLARE_BLOCKERS,
            PhaseType.COMBAT_DAMAGE, PhaseType.COMBAT_END, PhaseType.MAIN2, PhaseType.END_OF_TURN);

    /**
     * The worker's gate: the game's {@code budget} spread over its turns,
     * from a generator seeded by the game's seed alone, never the live game's
     * random stream.
     *
     * <p>Each game turn draws once, at its first boundary (or, for a turn
     * with no boundary asked, at the next), whether it takes an
     * intervention: with chance {@code left / (TYPICAL_GAME_TURNS - turn +
     * 1)}, at least the whole of it from the typical game's last turn on,
     * where {@code left} is the budget neither forced nor already drawn; and
     * for one drawn, the step of {@link #OPENING_STEPS} it opens at. An
     * intervention drawn and not yet forced opens every boundary from that
     * step on until one forces a line, so a step or a turn whose boundaries
     * offer nothing to force carries it forward rather than losing it.
     */
    static Gate spreadOverTheGame(long seed, int budget) {
        return new SpreadGate(SeedStreams.random(seed, SeedStreams.Consumer.SPREAD_GATE), budget);
    }

    /** {@link #spreadOverTheGame}'s gate. */
    static final class SpreadGate implements Gate {
        private final Random random;
        private final int budget;
        /** The last game turn drawn for. */
        private int drawnThrough;
        /** Interventions drawn and not yet forced: each its turn and the step it opens at. */
        private final java.util.ArrayDeque<int[]> pending = new java.util.ArrayDeque<>();
        private int forced;

        SpreadGate(Random random, int budget) {
            this.random = random;
            this.budget = budget;
        }

        @Override
        public boolean open(int turn, PhaseType phase) {
            while (drawnThrough < turn) {
                drawnThrough++;
                draw(drawnThrough);
            }
            int[] first = pending.peekFirst();
            return first != null && (turn > first[0]
                    || phase != null && phase.ordinal() >= OPENING_STEPS.get(first[1]).ordinal());
        }

        private void draw(int turn) {
            int left = budget - forced - pending.size();
            double chance = left <= 0 ? 0.0
                    : Math.min(1.0, left / (double) Math.max(1, TYPICAL_GAME_TURNS - turn + 1));
            // Both drawn whatever the chance, so a turn's draws depend on the
            // seed and the turn alone.
            boolean drawn = random.nextDouble() < chance;
            int step = random.nextInt(OPENING_STEPS.size());
            if (drawn) {
                pending.addLast(new int[] {turn, step});
            }
        }

        @Override
        public void forced() {
            pending.pollFirst();
            forced++;
        }
    }

    /**
     * Told of each line forced and its holder, as the scheduler forces it,
     * for a test that plays real matches; null in a worker.
     */
    static volatile java.util.function.BiConsumer<SpellAbility, Player> forcedWatch;

    private final Game game;
    private final Forcing forks;
    /** Whether a boundary forces a line: asked once per boundary. */
    private final Gate gate;
    /** Grants the rare protections, or null when the run takes none. */
    private final ProtectionFork protection;
    /**
     * Whether either deck of the game carries a rule (FR-058d): a steered
     * game forces the variant cards its players hold first (FR-058e).
     */
    private final boolean steered;
    /** Lines already forced on their holder's own turn, so a redraw does not repeat one. */
    private final Set<String> interveneSeen = new LinkedHashSet<>();
    /** Lines already forced on another player's turn (FR-057), apart from the above. */
    private final Set<String> offTurnSeen = new LinkedHashSet<>();
    /**
     * Lines refused for reading a cost a forced line never pays ({@link
     * InterventionFork#readsAnUnpaidCost}), so each is counted once a game.
     */
    private final Set<String> unpaidCostRefused = new LinkedHashSet<>();
    /** Whether a step began whose first priority with the stack empty has not come yet. */
    private boolean stepAwaitsPriority;

    /** An unsteered game's scheduler forcing at every boundary and granting no protection. */
    InterventionScheduler(Game game, Forcing forks) {
        this(game, forks, null, false, EVERY_BOUNDARY);
    }

    /**
     * @param protection grants the rare protections, or null for none
     * @param steered    whether either deck of the game carries a rule
     * @param gate       whether a boundary forces a line, asked once per
     *                   boundary ({@link #spreadOverTheGame} in a worker)
     */
    InterventionScheduler(Game game, Forcing forks, ProtectionFork protection, boolean steered,
                          Gate gate) {
        this.game = game;
        this.forks = forks;
        this.protection = protection;
        this.steered = steered;
        this.gate = gate;
    }

    /**
     * A step began (its phase event): its first priority with the stack
     * empty may force a line. Published also for a step the turn skips, which
     * gives no priority and so forces nothing.
     */
    void stepBegan() {
        stepAwaitsPriority = true;
    }

    /**
     * The protection fork of the step beginning {@code phase}, taken at its
     * phase event; none where the score check cannot pass ({@link
     * Forks#scoreCheckable}).
     */
    void protect(PhaseType phase) {
        if (protection != null && Forks.scoreCheckable(phase)) {
            protection.collect(phase);
        }
    }

    /**
     * Where the game's {@link PhaseSignals} deliver the priority-settled
     * signal: the step's first with the stack empty forces a line, guarded
     * as a phase boundary's step is ({@link GameRecording#eachGuarded}).
     */
    PhaseSignals.Sink prioritySink() {
        return new PhaseSignals.Sink() {
            @Override
            public void prioritySettled(Player priorityPlayer) {
                GameRecording.eachGuarded(new GameRecording.BoundaryStep(
                        "interventions", InterventionScheduler.this::prioritySettled));
            }
        };
    }

    /**
     * Priority settled: at the step's first time with the stack empty, the
     * step's forced resolution. The step's turn-based action has happened,
     * and a line the rules let be played only with the stack empty can be
     * played there.
     */
    void prioritySettled() {
        if (!stepAwaitsPriority || !game.getStack().isEmpty()) {
            return;
        }
        stepAwaitsPriority = false;
        collect(game.getPhaseHandler().getPhase());
    }

    /**
     * The forced resolution of a step's first priority in {@code phase}.
     * None where the score check cannot pass ({@link Forks#scoreCheckable}),
     * which would spend the game's budget, and mark the line forced, for no
     * record.
     */
    void collect(PhaseType phase) {
        if (Forks.scoreCheckable(phase)) {
            collect();
        }
    }

    /**
     * Force one line the players hold and nobody would play, if this boundary
     * draws one ({@link #gate}): in a steered game, first among the lines of
     * the variant cards each player holds, the player whose turn it is first
     * (FR-058e); then, for every player but the one whose turn it is, a line
     * whose branch turns on whose turn it is (FR-057); then anything the
     * player whose turn it is holds.
     *
     * <p>A line without the modes or targets its resolution needs on this
     * board is passed over and not marked forced, so a removal spell held
     * over an empty board is forced once a target has arrived. A line whose
     * resolution reads a cost a forced line never pays (ninjutsu, a
     * sacrificed creature's power) is passed over too, wherever it is held,
     * counted once a game under its refusal's reason ({@link
     * InterventionFork#unpaidCostRefusal}): it takes neither the boundary
     * nor one of the gate's planned interventions.
     *
     * <p>A steered game's variants go first because they are what its decks
     * were built for: the rule its decks aim at is short of records, and the
     * passes after spend the budget on whatever else the players hold. A
     * variant line forced on its holder's turn counts as forced there, and one
     * forced on another player's turn as forced off-turn, so neither later
     * pass forces it again.
     *
     * <p>FR-057's lines go first because they are few and a boundary of the
     * other player's turn is the only place they show their other branch: the
     * last pass, over everything the active player holds, would otherwise
     * take every boundary that draws a line. Each line is forced at most
     * once on its holder's turn and once on another's, so both branches can
     * be seen in one game.
     */
    void collect() {
        Player active = game.getPhaseHandler() == null
                ? null : game.getPhaseHandler().getPlayerTurn();
        if (active == null || !gateOpen()) {
            return;
        }
        if (steered && forceVariants(active) == Pass.DONE) {
            return;
        }
        if (forceOffTurn(active, TurnConditions::branchesOnWhoseTurn) == Pass.DONE) {
            return;
        }
        force(active, interveneSeen, candidate -> true);
    }

    /** Whether the gate opens at this boundary: asked once a boundary. */
    private boolean gateOpen() {
        return gate.open(game.getPhaseHandler().getTurn(), game.getPhaseHandler().getPhase());
    }

    /**
     * A pass over the lines {@code wanted} accepts of each player still in
     * the game but {@code active}, off their own turn.
     */
    private Pass forceOffTurn(Player active, Predicate<SpellAbility> wanted) {
        for (Player other : game.getPlayers()) {
            if (stillPlaying(other, active) && force(other, offTurnSeen, wanted) == Pass.DONE) {
                return Pass.DONE;
            }
        }
        return Pass.GO_ON;
    }

    private static boolean stillPlaying(Player other, Player active) {
        return other != active && !other.hasLost();
    }

    /** What a pass over a player's lines ends with. */
    private enum Pass {
        /** A line was forced, or the budget refused one: the boundary is over. */
        DONE,
        /** Nothing was forced: the next pass may force a line. */
        GO_ON
    }

    /**
     * The steered game's first pass: the variant cards' lines each player
     * holds, the active player's on its own turn, the others' off it.
     */
    private Pass forceVariants(Player active) {
        if (force(active, interveneSeen, InterventionScheduler::ofAVariant) == Pass.DONE) {
            return Pass.DONE;
        }
        return forceOffTurn(active, InterventionScheduler::ofAVariant);
    }

    /** Whether the line is a staged variant card's ({@link VariantRegistry}). */
    private static boolean ofAVariant(SpellAbility candidate) {
        return candidate.getHostCard() != null && ScriptLocator.isVariant(candidate.getHostCard());
    }

    /**
     * Force a line {@code holder} holds in the {@link #FORCED_ZONES} that
     * {@code wanted} accepts, {@code seen} does not hold yet, that its
     * restrictions let {@code holder} play at this boundary, and that has the
     * choices its resolution needs, for {@code holder}: the first of them in
     * the order the fork's stream draws ({@link Forcing#shuffle}), so one
     * drawn at random among them. The first in hand order spent most of the
     * budget on creature spells the AI casts anyway.
     */
    private Pass force(Player holder, Set<String> seen, Predicate<SpellAbility> wanted) {
        List<SpellAbility> lines = offered(holder, seen, wanted);
        forks.shuffle(lines);
        for (SpellAbility candidate : lines) {
            // A line with no choices is not marked: the board may offer a target later.
            if (playableNow(candidate, holder) && forks.hasChoices(candidate, holder)) {
                return force(candidate, holder, seen);
            }
        }
        return Pass.GO_ON;
    }

    /** The lines {@code holder} holds that a pass may force, in zone order. */
    private List<SpellAbility> offered(Player holder, Set<String> seen, Predicate<SpellAbility> wanted) {
        List<SpellAbility> lines = new ArrayList<>();
        for (ZoneType zone : FORCED_ZONES) {
            for (Card card : holder.getCardsIn(zone)) {
                lines.addAll(forcedFrom(card, zone));
            }
        }
        lines.removeIf(candidate -> !offerable(candidate, seen, wanted));
        return lines;
    }

    private boolean offerable(SpellAbility candidate, Set<String> seen, Predicate<SpellAbility> wanted) {
        return worthIntervening(candidate, seen) && wanted.test(candidate) && !refusedForAnUnpaidCost(candidate);
    }

    /**
     * Whether the line reads a cost a forced line never pays ({@link
     * InterventionFork#readsAnUnpaidCost}): refused here, as a game-restart
     * line is, not as a fork, so the boundary and the gate's planned
     * intervention stay for another line; counted once a game.
     */
    private boolean refusedForAnUnpaidCost(SpellAbility candidate) {
        String refusal = InterventionFork.unpaidCostRefusal(candidate);
        if (refusal == null) {
            return false;
        }
        if (unpaidCostRefused.add(lineKey(candidate))) {
            ForkRefusals.count(InterventionFork.KIND, refusal);
        }
        return true;
    }

    /** Force {@code candidate} for {@code holder} if the budget lets it: the boundary is over either way. */
    private Pass force(SpellAbility candidate, Player holder, Set<String> seen) {
        if (!forks.mayIntervene(candidate.getId())) {
            return Pass.DONE;
        }
        seen.add(lineKey(candidate));
        gate.forced();
        java.util.function.BiConsumer<SpellAbility, Player> watch = forcedWatch;
        if (watch != null) {
            watch.accept(candidate, holder);
        }
        forks.intervene(candidate, candidate.getId(), holder);
        return Pass.DONE;
    }

    /**
     * Whether {@code holder} could play {@code candidate} here but for its
     * cost: every check of the line's playability verdict ({@link
     * PlayabilityVerdict}) passes but the cost's. That is its timing (a
     * sorcery-speed line only in a main phase of {@code holder}'s turn with
     * the stack empty), whose turn and which step it may be played in
     * ({@code PlayerTurn$}, {@code ActivationPhases$}: a forecast only in its
     * holder's upkeep), its script's conditions ({@code IsPresent$},
     * threshold), its activation limits, the zone its card must be in and a
     * static forbidding it. A spell whose mana cost is no cost (Ancestral
     * Vision) is cast only through an effect, so it is not forced either.
     * Never at the untap or cleanup step. A line whose verdict cannot be read
     * is not forced.
     */
    static boolean playableNow(SpellAbility candidate, Player holder) {
        if (!givesPriority(phaseOf(holder)) || cannotBePlayedAtAll(candidate)) {
            return false;
        }
        try {
            return PlayabilityVerdict.of(candidate, holder).failedChecks().stream()
                    .allMatch(failed -> failed.check() == PlayabilityCheck.COST);
        } catch (RuntimeException e) {
            return false;
        }
    }

    /** Whether no cost accounts for what bars the line: a spell with no mana cost, or its source ({@link #barredBySource}). */
    private static boolean cannotBePlayedAtAll(SpellAbility candidate) {
        return unpayable(candidate) || barredBySource(candidate);
    }

    /**
     * Whether its source bars the line, which no cost accounts for: Forge
     * models summoning sickness (302.6) inside the {@code {T}} and {@code {Q}}
     * costs' {@code canPay}, so the verdict fails only {@code cost} for a
     * sick creature, and a detained or suppressed source fails only {@code
     * cost} when an additional cost cannot be paid either.
     */
    private static boolean barredBySource(SpellAbility candidate) {
        Card host = candidate.getHostCard();
        Cost cost = candidate.getPayCosts();
        if (host == null || !candidate.isActivatedAbility()) {
            return false;
        }
        boolean sickAndTaps = cost != null && host.isAbilitySick()
                && (cost.hasTapCost() || cost.hasSpecificCostType(CostUntap.class));
        return sickAndTaps || host.isDetained() || candidate.isSuppressed();
    }

    /** The step {@code holder}'s game is in, or null where it has none. */
    private static PhaseType phaseOf(Player holder) {
        Game game = holder == null ? null : holder.getGame();
        return game == null || game.getPhaseHandler() == null ? null : game.getPhaseHandler().getPhase();
    }

    /** Whether a step may give priority: not the untap step, nor, as a rule, the cleanup step. */
    private static boolean givesPriority(PhaseType phase) {
        return phase != null && phase != PhaseType.UNTAP && phase != PhaseType.CLEANUP;
    }

    /** Whether the line is a spell whose mana cost is "no cost" (118.6). */
    private static boolean unpayable(SpellAbility candidate) {
        Cost cost = candidate.getPayCosts();
        return cost != null && PlayabilityVerdict.unpayable(candidate, cost);
    }

    /**
     * The line a candidate is, for forcing each once a game: its printed
     * line's provenance key, so two lines of one card with the same effect
     * API are two lines and a card redrawn is the same one; for a line that
     * keys to none, its API, its host's name and its text.
     *
     * <p>Without its holder or controller: the key is what keeps a game from
     * spending its budget on a second observation of one printed line, and
     * the same line held by the other player, or a second copy of the card,
     * resolves the same text on a board of the same game. A mirror match
     * whose players hold the same lines therefore forces each once, and may
     * leave budget unspent, which takes no fork.
     */
    static String lineKey(SpellAbility candidate) {
        ProvenanceKey.Resolved resolved;
        try {
            resolved = ProvenanceResolver.resolve(candidate);
        } catch (RuntimeException e) {
            resolved = ProvenanceKey.Resolved.UNKNOWN;
        }
        return resolved.key() != null
                ? resolved.key().toString()
                : AbilityLines.key(candidate) + "|" + candidate.getDescription();
    }

    /**
     * The zones whose lines are forced: the hand, what is already on the
     * battlefield, the graveyard and exile. A permanent's activated
     * abilities — a planeswalker's loyalty abilities among them — and a
     * graveyard's are where the AI passes most often, and a hand-only list
     * never reached them. The order spends no budget first: a boundary's
     * lines are shuffled from the fork's stream.
     */
    static final List<ZoneType> FORCED_ZONES = List.of(
            ZoneType.Hand, ZoneType.Battlefield, ZoneType.Graveyard, ZoneType.Exile);

    /**
     * The lines {@code card} offers from {@code zone}: in the hand all of them,
     * as before; anywhere else only those whose activation zone is that zone
     * (Forge's own rule, {@code SpellAbilityRestriction}: the {@code
     * ActivationZone$} or {@code AdditionalActivationZone$}), so a graveyard
     * card's flashback or return-to-hand ability is forced and its printed
     * spell is not. On the battlefield a permanent's own spell is never one:
     * it has been cast already.
     */
    static List<SpellAbility> forcedFrom(Card card, ZoneType zone) {
        List<SpellAbility> lines = new ArrayList<>();
        for (SpellAbility candidate : card.getSpellAbilities()) {
            if (zone == ZoneType.Battlefield && candidate.isSpell()) {
                continue;
            }
            if (activatesFrom(candidate, zone)) {
                lines.add(candidate);
            }
        }
        return lines;
    }

    /**
     * Whether {@code candidate} is played from {@code zone}. A card lists
     * every one of its lines wherever it is, so a permanent's activated
     * abilities are listed in the hand too; forced there, one resolved with
     * its host in the hand, a board no activation reaches. A spell declares
     * no zone of its own and is cast from the hand.
     */
    private static boolean activatesFrom(SpellAbility candidate, ZoneType zone) {
        ZoneType declared = candidate.getRestrictions() == null ? null : candidate.getRestrictions().getZone();
        if (zone == ZoneType.Hand && castFromHand(candidate, declared)) {
            return true;
        }
        return zone == declared
                || zone.name().equalsIgnoreCase(candidate.getParamOrDefault("AdditionalActivationZone", ""));
    }

    /** Whether {@code candidate} is a spell cast from the hand: one declaring no zone, or the hand. */
    private static boolean castFromHand(SpellAbility candidate, ZoneType declared) {
        return candidate.isSpell() && (declared == null || declared == ZoneType.Hand);
    }

    /**
     * Is this line one observation cannot reach anyway?
     *
     * <p>Lands and mana abilities are excluded because every game plays them —
     * forking to force a Mountain onto the battlefield spends a game copy to
     * observe the most common event in the corpus. What is left is the spells
     * and activated abilities the AI passed over, which is the population the
     * intervention exists for. A special action (116.2: suspending,
     * foretelling or plotting a card; Forge's {@code AbilityStatic}) uses no
     * stack and is never forced.
     *
     * <p>A line that restarts the game (Karn Liberated's) is never forced:
     * Forge's restart resets the JVM-wide trigger id counter
     * ({@code Trigger.resetIDs}), so a fork forcing one would hand the live
     * game's next triggers ids its existing ones hold.
     *
     * <p>Once per line per game in each set: a card redrawn or returned to
     * hand is the same line, and forcing it twice buys the second copy of one
     * observation.
     */
    static boolean worthIntervening(SpellAbility candidate, Set<String> seen) {
        return !neverForced(candidate) && !seen.contains(lineKey(candidate));
    }

    /** A land play, a mana ability, a special action or a game restart. */
    private static boolean neverForced(SpellAbility candidate) {
        return candidate.isLandAbility() || candidate.isManaAbility()
                || candidate instanceof AbilityStatic || restartsTheGame(candidate);
    }

    /** Whether a clause of the line's chain restarts the game. */
    static boolean restartsTheGame(SpellAbility candidate) {
        for (SpellAbility clause = candidate; clause != null; clause = clause.getSubAbility()) {
            if (clause.getApi() == ApiType.RestartGame) {
                return true;
            }
        }
        return false;
    }
}
