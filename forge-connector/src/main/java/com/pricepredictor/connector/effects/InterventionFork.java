package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.Random;
import java.util.function.Supplier;

/**
 * An <b>interventional resolution</b>: a fork that forces an ability Forge's
 * AI never chose to use.
 *
 * <p>Some abilities are never cast in any game the AI plays — it judges them
 * bad, or cannot afford them, or never draws them into a board where they do
 * anything — and no amount of observation will produce a record for them. The
 * fork resolves the ability with chosen targets and modes, routing an
 * unaffordable candidate through the play-without-paying path, and writes the
 * <b>effect half only</b>: there was no real activation, so there is no cost
 * half to link to. The fork is recorded by the live bracket's own code
 * ({@link ForcedResolution}).
 */
final class InterventionFork implements InterventionScheduler.Forcing {

    /** A refusal's reason: the forced ability never reached the copy's stack (a special action). */
    static final String NOT_ON_THE_STACK = "nothing reached the stack";

    /** A refusal's reason: the forced ability reached the stack, never began to resolve and did nothing. */
    static final String NEVER_RESOLVED = "never resolved";

    /**
     * A refusal's reason: the line's resolution reads, unguarded, a cost the
     * fork does not pay ({@link #readsAnUnpaidCost}).
     */
    static final String READS_AN_UNPAID_COST = "reads a cost it never paid";

    private final Game game;
    /** The live game's recording: its hooks and its shard. */
    private final GameRecording live;
    private final ForkBudget budget;
    /** The draws a forced resolution makes, from the game's seed. */
    private final RandomChoices choices;
    /** The game's seed, which seeds a fork recording's sampling stream. */
    private final long seed;

    /**
     * @param game   the live game
     * @param live   its recording, which records each fork
     * @param budget the game's fork budget
     * @param seed   the game's seed
     */
    InterventionFork(Game game, GameRecording live, ForkBudget budget, long seed) {
        this.game = game;
        this.live = live;
        this.budget = budget;
        this.seed = seed;
        // Seeded so a rerun of a game forces the same choices. Scrambled as
        // the sampler's seed is: games are seeded consecutively, and the first
        // draw here is a mode or target count, which an unscrambled seed would
        // make nearly the same in every game.
        this.choices = new RandomChoices(new Random(GameRecording.scramble(seed)));
    }

    /**
     * Whether {@code line}'s resolution reads what its cost paid where Forge
     * reads it without a guard, so a forced line, which pays nothing, throws
     * there: ninjutsu and a sneak cast take the attacker returned to hand
     * ({@code getPaidList("Returned")}) and the defender it attacked. Refused
     * before the fork is taken, as a line the engine cannot resolve off its
     * normal path.
     *
     * <p>Crewing and saddling are forced: each adds its paid list of tapped
     * creatures to the vehicle's or mount's list for the turn, and a forced
     * line's list is null, which Forge stores into an empty list and adds to
     * a held one as nothing ({@code FCollection.addAll} ignores a null), so
     * neither a first nor a second crew or saddle in a turn throws.
     */
    static boolean readsAnUnpaidCost(SpellAbility line) {
        if (line == null) {
            return false;
        }
        SpellAbility root = line.getRootAbility();
        return root.isNinjutsu() || root.isSneak();
    }

    /** Whether the budget lets {@link #intervene} take another fork of this resolution. */
    @Override
    public boolean mayIntervene(int realResolutionId) {
        return budget.mayIntervene(realResolutionId);
    }

    /**
     * Fork the game, force {@code ability} to resolve, and record the effect.
     *
     * @param realResolutionId the real resolution this fork varies, for the
     *                         per-resolution cap
     * @return whether a record was written
     */
    boolean intervene(SpellAbility ability, int realResolutionId) {
        return intervene(ability, realResolutionId, () -> ability.getActivatingPlayer() != null
                ? ability.getActivatingPlayer()
                : game.getPhaseHandler().getPlayerTurn());
    }

    /**
     * The same, the ability resolved for {@code holder}: the player whose hand
     * holds it, who need not be the player whose turn it is (FR-057).
     */
    @Override
    public boolean intervene(SpellAbility ability, int realResolutionId, Player holder) {
        return intervene(ability, realResolutionId, () -> holder);
    }

    private boolean intervene(SpellAbility ability, int realResolutionId, Supplier<Player> perspective) {
        if (!budget.mayIntervene(realResolutionId)) {
            return false;
        }
        if (readsAnUnpaidCost(ability)) {
            // Before the fork is taken: it would spend the budget and a copy
            // on a line whose forced resolution throws. The scheduler passes
            // such a line over before it asks (InterventionScheduler.force).
            ForkRefusals.count("intervention", READS_AN_UNPAID_COST);
            return false;
        }
        Forks.ForkRun run = new Forks.ForkRun(
                "intervention", game, () -> budget.spendIntervention(realResolutionId), perspective);
        Boolean written = run.take((verified, snapshots) -> {
            SpellAbility forked = Forks.mappedAbility(verified.copier(), ability);
            Player actor = forked == null
                    ? null : Forks.mappedPlayer(verified.copier(), verified.perspective());
            if (forked == null || actor == null) {
                throw new ForkRefusal(ForkRefusal.NO_COUNTERPART);
            }
            return forceResolution(verified, forked, actor);
        }, (verified, resolution) -> {
            live.writeForkLines(resolution.outcome().stream().map(EffectRecord::toJson).toList());
            return Boolean.TRUE;
        });
        return Boolean.TRUE.equals(written);
    }

    /**
     * Put the ability on the fork's stack and resolve it, recording what it did.
     *
     * <p>Placed on the stack directly rather than played through the AI's cost
     * machinery, because <b>the cost is the reason the record is missing</b>. An
     * ability the AI never used is usually one it could never afford, and
     * routing the fork through cost payment fails on exactly the population the
     * intervention exists to reach — leaving lands and other free plays, which
     * observation already covers. The corpus wants what the effect does, and
     * the cost half is recorded from real activations elsewhere.
     *
     * <p>The board still holds what a cost would have used up: the mana
     * sources a payment would have tapped, the mana it would have spent from
     * the pool, and anything sacrificed, discarded, exiled or tapped as a
     * cost. And nothing was paid into the ability, so a resolution that reads
     * what was paid reads nothing.
     *
     * <p>The modes, targets and X are drawn at random from the legal options,
     * out of the fork's own seeded source, rather than through the AI. That
     * is the same decision the cost bypass makes and for the same reason: the
     * corpus wants what the effect <b>does</b>, and asking the AI which mode
     * is good would reproduce exactly the selection bias the intervention
     * exists to escape. Seeded so a rerun of a game makes the same choice.
     * Choosing none at all is not an option: a Lava Axe pushed with no target
     * resolves into nothing.
     *
     * <p>A resolution that began is written whatever it did, as a live one
     * is: its halves empty where it resolved and did nothing observable or
     * fizzled. One that never began to resolve wrote no half, and is nothing
     * to write.
     *
     * @throws ForkRefusal when its choices cannot be drawn, nothing reached
     *         the stack, or it never began to resolve
     */
    private ForcedResolution forceResolution(Forks.VerifiedFork verified, SpellAbility ability, Player actor) {
        // Not guarded here: a forced resolution reaches states ordinary play
        // does not, and Forks.ForkRun discards the fork when a card throws.
        if (!drawForcedChoices(ability, actor)) {
            throw new ForkRefusal(ForkRefusal.NO_CHOICES);
        }
        ForcedResolution resolution = new ForcedResolution(live, verified, seed).force(ability, actor);
        if (resolution.outcome().isEmpty()) {
            throw new ForkRefusal(NEVER_RESOLVED);
        }
        return resolution;
    }

    /**
     * Make the forced ability's choices for {@code actor}: its modes, its
     * targets and its X, drawn at random from the legal options.
     *
     * @return whether a mode and a target could be chosen wherever the
     *         ability needs one
     */
    private boolean drawForcedChoices(SpellAbility ability, Player actor) {
        ability.setActivatingPlayer(actor);
        return drawChoices(choices, ability, actor);
    }

    /**
     * Modes, then X, then targets, the order a cast takes them (601.2b-c):
     * a clause dividing X among its targets splits the X just announced, and
     * one drawn before it split nothing.
     *
     * @param payer whose leftover mana bounds X, or null to announce the floor
     */
    private static boolean drawChoices(RandomChoices from, SpellAbility ability, Player payer) {
        if (!from.chooseModes(ability)) {
            return false;
        }
        // Floor one: an X of zero resolves into the same silence a
        // missing target does, and the fork never pays, so the leftover mana of
        // the forked actor is what bounds the draw above it.
        from.announceX(ability, payer, 1);
        return from.chooseTargets(ability);
    }

    /**
     * Whether {@code ability} has a mode and targets for {@code holder} on
     * the live board: the forced choices tried on a copy of it, with draws
     * of their own, so neither the live ability nor any stream is touched.
     * A copy that cannot be tried is let through, to the fork's own guard.
     */
    @Override
    public boolean hasChoices(SpellAbility ability, Player holder) {
        try {
            SpellAbility trial = ability.copy(ability.getHostCard(), holder, true);
            trial.setActivatingPlayer(holder);
            return drawChoices(new RandomChoices(new Random(0L)), trial, null);
        } catch (RuntimeException e) {
            return true;
        }
    }
}
