package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.function.Predicate;

/**
 * Which ability nobody played is forced, at a phase boundary, and when a
 * permanent is granted a rare protection (FR-058).
 *
 * <p>Some abilities appear in no record however long collection runs: the AI
 * judges them bad, cannot afford them, or never draws them into a board where
 * they do anything. Observation cannot reach those, so the corpus forks the
 * game and makes one happen.
 *
 * <p>Driven from a phase boundary rather than from the playability hook, even
 * though that hook knows exactly which candidates were declined. The hook
 * fires inside the AI's own evaluation, and forking the game there means
 * copying a game that is mid-decision and re-entering the rules engine
 * underneath it. A phase boundary is the same board with nothing in flight.
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
 * most, and only a line its own timing rules let its holder play there
 * ({@link #playableNow}): a sorcery-speed line in a main phase of its holder's
 * turn with the stack empty, an instant-speed one at any boundary that gives
 * priority.
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

    /** Kept apart from the other streams the same game seed seeds. */
    private static final long SEED_SALT = 0x2545F4914F6CDD1DL;

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
        return new SpreadGate(new Random(GameRecording.scramble(seed ^ SEED_SALT)), budget);
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
     * This boundary's interventions: the forced resolutions, then a protection
     * fork. None where the score check cannot pass ({@link Forks#scoreCheckable}),
     * which would spend the game's budget, and mark the line forced, for no record.
     */
    void collect(PhaseType phase) {
        if (!Forks.scoreCheckable(phase)) {
            return;
        }
        collect();
        if (protection != null) {
            protection.collect(phase);
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
     * over an empty board is forced once a target has arrived.
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
        if (active == null
                || !gate.open(game.getPhaseHandler().getTurn(), game.getPhaseHandler().getPhase())) {
            return;
        }
        if (steered && forceVariants(active) == Pass.DONE) {
            return;
        }
        for (Player other : game.getPlayers()) {
            if (other != active && !other.hasLost()
                    && force(other, offTurnSeen, TurnConditions::branchesOnWhoseTurn) == Pass.DONE) {
                return;
            }
        }
        force(active, interveneSeen, candidate -> true);
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
        for (Player other : game.getPlayers()) {
            if (other != active && !other.hasLost()
                    && force(other, offTurnSeen, InterventionScheduler::ofAVariant) == Pass.DONE) {
                return Pass.DONE;
            }
        }
        return Pass.GO_ON;
    }

    /** Whether the line is a staged variant card's ({@link VariantRegistry}). */
    private static boolean ofAVariant(SpellAbility candidate) {
        return candidate.getHostCard() != null && ScriptLocator.isVariant(candidate.getHostCard());
    }

    /**
     * Force the first line {@code holder} holds in the {@link #FORCED_ZONES}
     * that {@code wanted} accepts, {@code seen} does not hold yet, that its
     * timing lets {@code holder} play at this boundary, and that has the
     * choices its resolution needs, for {@code holder}.
     */
    private Pass force(Player holder, Set<String> seen, Predicate<SpellAbility> wanted) {
        for (ZoneType zone : FORCED_ZONES) {
            for (Card card : holder.getCardsIn(zone)) {
                for (SpellAbility candidate : forcedFrom(card, zone)) {
                    if (!worthIntervening(candidate, seen) || !wanted.test(candidate)
                            || !playableNow(candidate, holder)) {
                        continue;
                    }
                    if (!forks.mayIntervene(candidate.getId())) {
                        return Pass.DONE;
                    }
                    if (!forks.hasChoices(candidate, holder)) {
                        // Not marked: the board may offer a target later.
                        continue;
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
            }
        }
        return Pass.GO_ON;
    }

    /**
     * Whether {@code holder} could play {@code candidate} at this boundary by
     * its own timing: a sorcery-speed line (a sorcery, a creature, a
     * planeswalker's loyalty ability, an ability restricted to sorcery speed)
     * only in a main phase of {@code holder}'s turn with the stack empty, an
     * instant-speed one (an instant, flash, an activated ability) wherever
     * the step gives priority, which the untap and cleanup steps do not
     * ({@code SpellAbility.canCastTiming}, read without changing anything).
     * A line whose timing cannot be read is not forced.
     */
    static boolean playableNow(SpellAbility candidate, Player holder) {
        if (holder == null || holder.getGame() == null || holder.getGame().getPhaseHandler() == null) {
            return false;
        }
        PhaseType phase = holder.getGame().getPhaseHandler().getPhase();
        if (phase == null || phase == PhaseType.UNTAP || phase == PhaseType.CLEANUP) {
            return false;
        }
        try {
            return candidate.canCastTiming(candidate.getHostCard(), holder);
        } catch (RuntimeException e) {
            return false;
        }
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
     * The zones whose lines are forced, in the order the budget is spent on
     * them: the hand first, as it always was, so a run's spells keep the
     * budget they had; then what is already on the battlefield; then the
     * graveyard and exile. A permanent's activated abilities — a
     * planeswalker's loyalty abilities among them — and a graveyard's are
     * where the AI passes most often, and a hand-only list never reached them.
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
        if (zone == ZoneType.Hand && candidate.isSpell() && (declared == null || declared == ZoneType.Hand)) {
            return true;
        }
        return zone == declared
                || zone.name().equalsIgnoreCase(candidate.getParamOrDefault("AdditionalActivationZone", ""));
    }

    /**
     * Is this line one observation cannot reach anyway?
     *
     * <p>Lands and mana abilities are excluded because every game plays them —
     * forking to force a Mountain onto the battlefield spends a game copy to
     * observe the most common event in the corpus. What is left is the spells
     * and activated abilities the AI passed over, which is the population the
     * intervention exists for.
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
        if (candidate.isLandAbility() || candidate.isManaAbility() || restartsTheGame(candidate)) {
            return false;
        }
        return !seen.contains(lineKey(candidate));
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
