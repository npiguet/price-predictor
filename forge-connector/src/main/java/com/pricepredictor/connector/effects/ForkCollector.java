package com.pricepredictor.connector.effects;

import forge.ai.ComputerUtil;
import forge.ai.simulation.GameCopier;
import forge.ai.simulation.GameSimulator;
import forge.ai.simulation.GameStateEvaluator;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.ability.AbilityUtils;
import forge.game.ability.ApiType;
import forge.game.ability.effects.CharmEffect;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.StringJoiner;

/**
 * Stage three: records for what observation cannot reach.
 *
 * <p>Two kinds of fork, and the flags tell them apart on their own.
 *
 * <p>An <b>interventional resolution</b> forces an ability Forge's AI never
 * chose to use. Some abilities are never cast in any game the AI plays — it
 * judges them bad, or cannot afford them, or never draws them into a board
 * where they do anything — and no amount of observation will produce a record
 * for them. The fork resolves the ability with chosen targets and modes,
 * routing an unaffordable candidate through the play-without-paying path, and
 * writes the <b>effect half only</b>: there was no real activation, so there is
 * no cost half to link to.
 *
 * <p>A <b>damage-step probe</b> re-runs one combat with a keyword stripped, to
 * isolate what that keyword did. It is built only for keywords gate 2 routed to
 * it, and taken only when {@code --probe-keywords} names one — the flag is the
 * runtime switch, separate from the build decision, so a checkout that has the
 * machinery still takes no fork unless asked.
 *
 * <p><b>Every fork is score-checked against the live game before any
 * perturbation.</b> A copy that already disagrees with the game it came from is
 * measuring the copier rather than the ability, and a discarded fork still
 * counts against its budget — otherwise a systematically-failing copy would
 * retry forever.
 */
public final class ForkCollector {

    /** At most this many forks may target one real resolution (FR-040). */
    public static final int MAX_FORKS_PER_RESOLUTION = 2;

    /**
     * The two damage steps, spelled as the bracket collector spells them.
     *
     * <p>A held branch and the real record it mirrors must agree about which
     * step they describe, and the two are written by different classes.
     */
    public static final String SUBSTEP_FIRST_STRIKE = "first_strike";
    public static final String SUBSTEP_REGULAR = "regular";

    private final Game game;
    private final RecordShardWriter writer;
    private final String gameId;
    private final AttributionMode mode;
    private final int interventionsPerGame;
    private final int probesPerGame;
    private final List<String> probeKeywords;
    /** The run's snapshot depth, the same on every record this writes. */
    private final int[] snapshotTiers;
    private final Random random;

    private int interventionsUsed;
    private int probesUsed;
    private final Map<Integer, Integer> forksPerResolution = new HashMap<>();
    private int discarded;
    private long recordsWritten;

    public ForkCollector(
            Game game, RecordShardWriter writer, String gameId,
            PatchedCollectors.CollectionCaps caps, long seed) {
        this.game = game;
        this.writer = writer;
        this.gameId = gameId;
        this.mode = AttributionMode.detect();
        this.interventionsPerGame = caps.interventionsPerGame();
        this.probesPerGame = caps.probesPerGame();
        this.probeKeywords = List.copyOf(caps.probeKeywords());
        // The run's tier vector, not this collector's own idea of one. Choosing
        // it here is what made state.tiers a perfect proxy for the
        // interventional flag: tier 4 appeared on the 55,661 interventional
        // records of the first corpus and on nothing else, so a model handed
        // the tier list could read collection metadata the schema forbids as
        // an input.
        this.snapshotTiers = caps.snapshotTierArray();
        // Seeded so both branches of a probe see the same shuffles: a
        // difference between them has to be the keyword, not the draw.
        this.random = new Random(seed);
    }

    public long recordsWritten() {
        return recordsWritten;
    }

    /** Forks created and thrown away by the score check. */
    public int discardedForks() {
        return discarded;
    }

    // ── budgets ─────────────────────────────────────────────────────────

    /**
     * Whether another interventional fork may be taken.
     *
     * <p>Two budgets, both binding: the per-game one from
     * {@code --interventions-per-game}, and a fixed cap of
     * {@value #MAX_FORKS_PER_RESOLUTION} forks per real resolution that no flag
     * can raise. The second exists because forking the same resolution twice
     * already gives both counterfactuals worth having; a third would cost a
     * game's simulation for a third variation on one board.
     */
    public boolean mayIntervene(int realResolutionId) {
        if (interventionsUsed >= interventionsPerGame) {
            return false;
        }
        return forksPerResolution.getOrDefault(realResolutionId, 0)
                < MAX_FORKS_PER_RESOLUTION;
    }

    /**
     * Whether another probe fork may be taken.
     *
     * <p>With {@code --probe-keywords} unset, never — whatever the build state.
     * The flag is the runtime switch; whether the probe code exists at all is
     * the build decision gate 2 drives, and the two are deliberately separate
     * so a checkout that has the machinery does not start using it by default.
     */
    public boolean mayProbe(String keyword) {
        // Normalized before the check: the flag names keywords the way the
        // corpus spells them (first_strike) and a caller holding a card's
        // keyword has Forge's (First Strike). Comparing the two directly is
        // what made gate 2 report zero observations of first strike, and it
        // would have made every probe silently decline here.
        if (keyword == null || probeKeywords.isEmpty()
                || !probeKeywords.contains(
                        PatchedCollectors.normalizeKeyword(keyword))) {
            return false;
        }
        return probesUsed < probesPerGame;
    }

    // ── the score check ─────────────────────────────────────────────────

    /**
     * Verify a fresh copy scores the same as the game it came from.
     *
     * <p>Run <b>before any perturbation</b>: a copy that already disagrees is
     * measuring the copier, and a record from it would be noise dressed as a
     * counterfactual. A mismatch logs a warning, discards the fork, and
     * <b>still counts against the budget</b> — otherwise a systematically
     * failing copy would retry until the game ended.
     */
    public boolean scoreCheck(Game fork, Player perspective) {
        try {
            GameStateEvaluator evaluator = new GameStateEvaluator();
            int original = evaluator.getScoreForGameState(game, perspective).value;
            Player forkPerspective = matchingPlayer(fork, perspective);
            if (forkPerspective == null) {
                return false;
            }
            int copied = evaluator
                    .getScoreForGameState(fork, forkPerspective).value;
            if (original != copied) {
                System.err.println(
                        "Effect records: discarding a fork whose score "
                                + copied + " disagrees with the live game's "
                                + original + "; it would measure the copier "
                                + "rather than the ability");
                return false;
            }
            return true;
        } catch (RuntimeException e) {
            System.err.println(
                    "Effect records: fork score check failed (" + e.getMessage()
                            + "); discarding");
            return false;
        }
    }

    private static Player matchingPlayer(Game fork, Player original) {
        for (Player candidate : fork.getPlayers()) {
            if (candidate.getId() == original.getId()) {
                return candidate;
            }
        }
        return null;
    }

    // ── interventional resolutions ──────────────────────────────────────

    /**
     * Fork the game, force {@code ability} to resolve, and record the effect.
     *
     * @param realResolutionId the real resolution this fork varies, for the
     *                         per-resolution cap
     * @return whether a record was written
     */
    public boolean intervene(SpellAbility ability, int realResolutionId) {
        if (!mayIntervene(realResolutionId)) {
            return false;
        }
        // Counted before the score check, so a discarded fork still spends its
        // budget.
        interventionsUsed++;
        forksPerResolution.merge(realResolutionId, 1, Integer::sum);

        GameCopier copier;
        Game fork;
        try {
            // Both inside the guard: the copier's own constructor reads the
            // game, so a game it cannot read fails here rather than at makeCopy.
            copier = new GameCopier(game);
            fork = copier.makeCopy();
        } catch (RuntimeException e) {
            discarded++;
            return false;
        }
        Player perspective = ability.getActivatingPlayer() != null
                ? ability.getActivatingPlayer()
                : game.getPhaseHandler().getPlayerTurn();
        if (perspective == null || !scoreCheck(fork, perspective)) {
            discarded++;
            return false;
        }

        SpellAbility forked = findInFork(copier, fork, ability);
        Player actor = forked == null ? null : mappedPlayer(copier, perspective);
        if (forked == null || actor == null) {
            discarded++;
            return false;
        }

        // The fork's own state, not the live game's, and read *before* the
        // resolution: the record's state is the board the ability acted on, and
        // reading it after would hand the model the answer.
        SnapshotBuilder snapshots = new SnapshotBuilder(fork, snapshotTiers);
        String state = snapshots.toJson(forked, referencedOf(forked));

        List<EffectEvent> events = forceResolution(fork, forked, actor);
        if (events == null || events.isEmpty()) {
            // An empty event list is not an observation. "The ability resolved
            // and did nothing" and "the ability never got to act" render the
            // same record, and the first corpus could not tell them apart —
            // 24 of the 88 sampled interventions were empty and every one of
            // them was a targeted spell that resolved with no target. A
            // counterfactual nobody can read is worse than one that was never
            // written, so this drops it and counts it.
            discarded++;
            return false;
        }

        emit(new EffectRecord(
                writer.nextRecordId(), writer.runId(),
                RecordShardWriter.timestamp(), gameId,
                EffectRecord.KIND_RESOLUTION, mode)
                .moment(EffectRecord.MOMENT_RESOLUTION)
                .interventional(true)
                .fork(true)
                // No link_id: the effect half is written alone, because there
                // was no real activation to pair it with.
                .actor(SnapshotBuilder.playerId(perspective))
                // The *live* ability's keys, not the fork's. A provenance key
                // names a printed line, which the copy shares — but keying off
                // the copy would make the record's identity depend on a game
                // that no longer exists.
                .ability(keysOf(ability))
                .state(state)
                .payload(EffectRecord.eventsPayload(events)));
        return true;
    }

    /**
     * Put the ability on the fork's stack and resolve it, collecting what it did.
     *
     * <p>Placed on the stack directly rather than played through the AI's cost
     * machinery, because <b>the cost is the reason the record is missing</b>. An
     * ability the AI never used is usually one it could never afford, and
     * routing the fork through cost payment fails on exactly the population the
     * intervention exists to reach — leaving lands and other free plays, which
     * observation already covers. The corpus wants what the effect does, and
     * the cost half is recorded from real activations elsewhere.
     *
     * <p>Resolution drains the whole stack, so a trigger the effect put there
     * is part of what the effect did.
     *
     * @return the events, empty if the ability resolved and did nothing
     *         observable, or null if it could not be resolved at all
     */
    private List<EffectEvent> forceResolution(
            Game fork, SpellAbility ability, Player actor) {
        ForkEventSink sink = new ForkEventSink(fork);
        fork.subscribeToEvents(sink);
        try {
            ability.setActivatingPlayer(actor);
            chooseModes(ability);
            if (!chooseTargets(ability)) {
                return null;
            }
            announceX(ability);
            fork.copyLastState();
            fork.getStack().add(ability);
            GameSimulator.resolveStack(fork, actor.getWeakestOpponent());
            return sink.events();
        } catch (RuntimeException | StackOverflowError e) {
            // A forced resolution reaches states ordinary play does not — an
            // ability resolving with no legal target, a cost that was never
            // paid — and a card that throws is one this fork cannot describe.
            // Discarding is right; failing the worker is not.
            return null;
        }
    }


    /**
     * Pick this fork's modes, where the line is modal.
     *
     * <p>At random from the legal options, out of the fork's own seeded source,
     * rather than through the AI. That is the same decision the cost bypass
     * above makes and for the same reason: the corpus wants what the effect
     * <b>does</b>, and asking the AI which mode is good would reproduce exactly
     * the selection bias the intervention exists to escape. Seeded so both
     * branches of a probe and a rerun of a game make the same choice.
     */
    void chooseModes(SpellAbility ability) {
        if (ability.getApi() != ApiType.Charm || ability.getChosenList() != null) {
            return;
        }
        List<AbilitySub> options = CharmEffect.makePossibleOptions(ability);
        if (options == null || options.isEmpty()) {
            return;
        }
        int wanted = AbilityUtils.calculateAmount(
                ability.getHostCard(),
                ability.getParamOrDefault("CharmNum", "1"), ability);
        wanted = Math.max(1, Math.min(wanted, options.size()));
        List<AbilitySub> shuffled = new ArrayList<>(options);
        Collections.shuffle(shuffled, random);
        List<AbilitySub> chosen = new ArrayList<>(shuffled.subList(0, wanted));
        ability.setChosenList(chosen);
        CharmEffect.chainAbilities(ability, chosen);
    }

    /**
     * Give every targeting clause of the chain legal targets.
     *
     * <p>The feature spec says an intervention resolves "with chosen targets and
     * modes" and the first implementation chose neither: it set the activating
     * player, pushed the ability and resolved it. A "deal 5 damage to target
     * player" that resolves with no target does nothing, which is why every one
     * of the 88 sampled interventional records had an empty {@code refs.targets}
     * and why the empty ones were Lava Axe, Disintegrate, Drain Life, Mind
     * Control and their like.
     *
     * <p>The whole {@code getSubAbility()} chain, not just the root: a sub-ability
     * targets independently, and a chain whose second clause has no target
     * resolves into the same silence.
     *
     * @return false when some clause has no legal target at all, which is an
     *         intervention with no counterfactual to record rather than one
     *         that did nothing
     */
    boolean chooseTargets(SpellAbility ability) {
        for (SpellAbility clause = ability; clause != null;
                clause = clause.getSubAbility()) {
            if (!clause.usesTargeting()) {
                continue;
            }
            clause.resetTargets();
            List<GameEntity> candidates =
                    clause.getTargetRestrictions().getAllCandidates(clause);
            List<GameEntity> shuffled = new ArrayList<>(candidates);
            Collections.shuffle(shuffled, random);
            for (GameEntity candidate : shuffled) {
                if (!clause.canAddMoreTarget()) {
                    break;
                }
                clause.getTargets().add(candidate);
            }
            if (!clause.isMinTargetChosen()) {
                return false;
            }
        }
        return true;
    }

    /**
     * Announce an X the fork never paid.
     *
     * <p>An X spell whose X is null resolves as an X of zero, which is the same
     * silence a missing target produces. One is the smallest value that makes
     * the effect happen at all; what value the corpus actually wants is an open
     * question, and a bigger one would have to be justified against mana the
     * fork deliberately never paid.
     */
    void announceX(SpellAbility ability) {
        if (ability.costHasX() && ability.getXManaCostPaid() == null) {
            ability.setXManaCostPaid(1);
        }
    }

    /**
     * The fork's copy of a live ability, or null when it cannot be matched.
     *
     * <p>The host card maps directly through the copier. The ability itself
     * does not, so it is matched by description among the copy's — the same
     * approach Forge's own simulator takes.
     */
    private static SpellAbility findInFork(
            GameCopier copier, Game fork, SpellAbility ability) {
        Card host;
        try {
            host = copier.find(ability.getHostCard());
        } catch (RuntimeException e) {
            return null;
        }
        if (host == null) {
            return null;
        }
        String description = ability.getDescription();
        for (SpellAbility candidate : host.getSpellAbilities()) {
            if (candidate.getDescription().equals(description)) {
                return candidate;
            }
        }
        return null;
    }

    private static Player mappedPlayer(GameCopier copier, Player player) {
        try {
            return copier.find(player);
        } catch (RuntimeException e) {
            return null;
        }
    }

    // ── damage-step probes ──────────────────────────────────────────────

    /**
     * Record a probe branch as an ordinary combat record.
     *
     * <p>{@code fork = true}, {@code interventional = false} — which is what
     * distinguishes a probe from an intervention by flags alone — and
     * {@code mirror_of} naming the real record it varies.
     *
     * <p>The real-versus-fork <b>difference is never computed here</b> and never
     * becomes a training target: it is a diagnostic about the model, computed at
     * evaluation time, and training on it would teach the model to reproduce its
     * own errors.
     */
    public boolean recordProbeBranch(
            String keyword, String mirrorOfRecordId, String snapshotJson,
            String payloadJson, String actorPlayerId) {
        if (!mayProbe(keyword)) {
            return false;
        }
        probesUsed++;
        emit(new EffectRecord(
                writer.nextRecordId(), writer.runId(),
                RecordShardWriter.timestamp(), gameId,
                EffectRecord.KIND_COMBAT, mode)
                .fork(true)
                .interventional(false)
                .mirrorOf(mirrorOfRecordId)
                .actor(actorPlayerId)
                .state(snapshotJson)
                .payload(payloadJson));
        return true;
    }

    /**
     * Re-run this damage step without one creature's keyword.
     *
     * <p>Called <b>before</b> the real step, which is the only moment the fork
     * can diverge: the phase event fires ahead of the turn-based action that
     * deals the damage, so the copy is taken from a board where nothing has
     * been dealt yet. The fork then assigns and deals its own damage with the
     * keyword gone, and what the bus publishes is what that keyword was worth.
     *
     * <p>The branch is <b>held rather than written</b>. It mirrors the real
     * combat record, and that record does not exist until the step it forked
     * ahead of has finished — so the caller completes the pairing once the id
     * is known. Reconstructing the pairing from board state instead would break
     * on two identical-looking combats in one turn.
     *
     * @return the held branch, or null when the fork could not be taken
     */
    public HeldProbe probe(String keyword, Card carrier, boolean firstStrike) {
        if (!mayProbe(keyword) || carrier == null) {
            return null;
        }
        // Counted before the score check, like an intervention: a copy that
        // systematically fails must not retry until the game ends.
        probesUsed++;

        GameCopier copier;
        Game fork;
        try {
            copier = new GameCopier(game);
            fork = copier.makeCopy();
        } catch (RuntimeException e) {
            discarded++;
            return null;
        }
        Player perspective = game.getPhaseHandler() == null
                ? null : game.getPhaseHandler().getPlayerTurn();
        if (perspective == null || !scoreCheck(fork, perspective)) {
            discarded++;
            return null;
        }

        Card forkCarrier;
        try {
            forkCarrier = copier.find(carrier);
        } catch (RuntimeException e) {
            forkCarrier = null;
        }
        if (forkCarrier == null || fork.getCombat() == null) {
            discarded++;
            return null;
        }

        // The state is the board as the step began, with the keyword already
        // gone: that is the input the counterfactual answers for.
        SnapshotBuilder snapshots = new SnapshotBuilder(fork, snapshotTiers);
        ForkEventSink sink = new ForkEventSink(fork);
        fork.subscribeToEvents(sink);
        String state;
        CombatShape shape;
        try {
            // Through the layer system rather than off the printed list: the
            // keyword may be printed, equipped or granted until end of turn,
            // and a removal layer takes it away in all three cases the way the
            // rules would.
            forkCarrier.addChangedCardKeywords(
                    null, List.of(keyword), false,
                    fork.getNextTimestamp(), null);
            state = snapshots.toJson(null, List.of());
            fork.getCombat().removeAbsentCombatants();
            // Who is blocking whom, before the damage removes any of them; the
            // assignment only exists after. Gate 2 reads this branch against
            // the record it mirrors, so both are read the same way.
            shape = CombatShape.of(fork.getCombat());
            if (fork.getCombat().assignCombatDamage(firstStrike)) {
                shape.addAssignment(fork.getCombat(), null);
                fork.getCombat().dealAssignedDamage();
                // And then let the deaths happen. Damage alone kills nothing;
                // a creature with lethal damage on it leaves the battlefield
                // during state-based actions, and stopping before them made the
                // branch systematically miss every death the real step had —
                // biasing the difference gate 2 computes towards "the keyword
                // changed nothing". The same call Forge's own simulator makes
                // after resolving a stack.
                fork.getAction().checkStateEffects(false, new HashSet<>());
            }
        } catch (RuntimeException | StackOverflowError e) {
            // A stripped keyword reaches combat states ordinary play does not.
            // Discarding is right; failing the worker is not.
            discarded++;
            return null;
        }
        return new HeldProbe(
                PatchedCollectors.normalizeKeyword(keyword),
                SnapshotBuilder.entityId(carrier), state, sink.events(),
                SnapshotBuilder.playerId(perspective), shape.fields(),
                firstStrike ? SUBSTEP_FIRST_STRIKE : SUBSTEP_REGULAR);
    }

    /**
     * A probe branch waiting for the record it mirrors.
     *
     * <p>Holds what the fork observed until the real combat record has an id.
     * The probe's own budget was already spent taking the fork, so a branch
     * that is never completed still counted — the cost was the simulation, not
     * the write.
     *
     * <p>{@code substep} names the damage step this branched from, spelled the
     * way the bracket collector spells it. Without it a branch taken at the
     * first-strike step was completed against whichever combat record came
     * next: in the first corpus that was the <b>regular</b> step's record, so
     * gate 2 was reading a counterfactual for a step that did not happen
     * against a real record for a different one.
     */
    public record HeldProbe(
            String keyword, String carrier, String state,
            List<EffectEvent> events, String actor, String combatFields,
            String substep) {

        /**
         * The branch's payload, naming what was perturbed.
         *
         * <p>Both the keyword and the creature it came from: a board with two
         * tramplers gives the keyword alone two readings, and the evaluator
         * reproducing this perturbation model-side would strip the wrong one.
         *
         * <p>{@code combatFields} is the fork's own combat, rendered when the
         * branch was taken. It is the counterfactual's half of what gate 2
         * compares: the observed record says who blocked whom and how the
         * damage was split with the keyword, and this says the same about the
         * board without it.
         */
        String payload() {
            StringJoiner rendered = new StringJoiner(",", "[", "]");
            for (EffectEvent event : events) {
                rendered.add(event.toJson());
            }
            return "{" + combatFields
                    + ",\"events\":" + rendered
                    + ",\"probed_keyword\":" + Json.string(keyword)
                    + ",\"probed_entity\":" + Json.string(carrier) + "}";
        }
    }

    /** Write a held branch, now that the record it mirrors has an id. */
    public boolean writeHeldProbe(HeldProbe held, String mirrorOfRecordId) {
        if (held == null || mirrorOfRecordId == null) {
            return false;
        }
        emit(new EffectRecord(
                writer.nextRecordId(), writer.runId(),
                RecordShardWriter.timestamp(), gameId,
                EffectRecord.KIND_COMBAT, mode)
                .fork(true)
                .interventional(false)
                .mirrorOf(mirrorOfRecordId)
                .actor(held.actor())
                .state(held.state())
                .payload(held.payload()));
        return true;
    }

    /**
     * The seeded source both probe branches draw from.
     *
     * <p>Forge's own restore of the random source is not in a {@code finally},
     * so a branch that throws would leave the live game with the probe's
     * sequence. Callers must install this and restore the original in a
     * {@code finally} of their own.
     */
    public Random seededRandom() {
        return random;
    }

    // ── plumbing ────────────────────────────────────────────────────────

    /** The acting line, or the reason there is none — see the sibling in
     *  {@link BusBracketCollector}: a fork's record is as entitled to say why
     *  it has no line as a real one. */
    private static ProvenanceKey.Resolved keysOf(SpellAbility ability) {
        if (ability == null) {
            return new ProvenanceKey.Resolved(null, ProvenanceKey.UNRESOLVED_UNKNOWN_KIND);
        }
        return ProvenanceKey.resolve(ability);
    }

    private static List<Card> referencedOf(SpellAbility ability) {
        List<Card> referenced = new ArrayList<>();
        if (ability != null && ability.getHostCard() != null) {
            referenced.add(ability.getHostCard());
        }
        return referenced;
    }

    private void emit(EffectRecord record) {
        writer.write(record.toJson());
        recordsWritten++;
    }
}
