package com.pricepredictor.connector.effects;

import forge.game.CardTraitBase;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.Map;

/**
 * The game a recording is for, and what every patched hook asks of it.
 *
 * <p>The patched hooks are plain JVM statics with no {@code Game} reference
 * of their own, and a fork fires them too: a forced resolution resolves a
 * forked ability through the same {@code AbilityUtils.resolveApiAbility} the
 * hooks are installed against. A fork's collectors keep its <em>bus</em>
 * events out of this game's records by subscribing to the fork's own
 * {@code Game}, but the hooks are
 * not bus subscriptions — without the checks here, a fork's
 * {@code vote_taken} or {@code coin_flipped} lands in the live game's bracket
 * with nothing marking it as hypothetical.
 *
 * <p>Game identity is the check because it needs no cooperation from the
 * fork's side: {@code GameCopier} builds every copied {@code Card} against a
 * new {@code Game}, so a fork's clause's own host card already carries the
 * answer. Compared by reference, as {@code SpellAbility.setActivatingPlayer}
 * compares players: a copied {@code Game} is never the live one.
 *
 * <p>Each hook checks in its own handler, immediately before its delivery
 * gate (the bracket for clauses and outcomes, sampling and dedup for the
 * others), rather than in {@link BusBracketCollector#recordClauseEvent},
 * which receives only the built {@code EffectEvent} and not the ability that
 * produced it.
 */
final class LiveGame {

    private final Game game;
    private final SnapshotBuilder snapshots;
    /**
     * The thread running the collector's own enumeration of this game's
     * candidates, or null: a flag scoped to one thread, so a query the AI
     * makes of this game on a thread of its own is not taken for it, and to
     * this game, so a fork, which has a {@code LiveGame} of its own, never
     * inherits it.
     */
    private volatile Thread enumerating;

    LiveGame(Game game) {
        this.game = game;
        this.snapshots = new SnapshotBuilder(game);
    }

    Game game() {
        return game;
    }

    /**
     * Run {@code enumeration}, the collector's own rules queries of this
     * game, as invisible to the channels (finding 8.5).
     *
     * <p>A query reads a candidate on a stand-in card (a morph's face-down
     * cast, a face-down permanent turned up, a spell with an alternate
     * host), and Forge then re-applies every static to the live game and
     * runs its state triggers: each state trigger is evaluated, and one whose
     * condition holds is run and queued for the stack. None of that is the
     * game's. The trigger channel drops every evaluation and run reported
     * while {@link #enumeratingHere()} holds, and the triggers the
     * enumeration queued are taken off again before it ends
     * ({@link QueuedTriggers}), whatever it threw.
     */
    void enumerate(Runnable enumeration) {
        QueuedTriggers queued = QueuedTriggers.of(game.getStack());
        enumerating = Thread.currentThread();
        try {
            enumeration.run();
        } finally {
            try {
                queued.dropQueuedSince();
            } finally {
                enumerating = null;
            }
        }
    }

    /** Whether this thread is running the collector's own enumeration of this game. */
    boolean enumeratingHere() {
        return enumerating == Thread.currentThread();
    }

    /** How this game's state is rendered into a record. */
    SnapshotBuilder snapshots() {
        return snapshots;
    }

    /**
     * Whether a clause, replacement, trigger or mana ability belongs to this
     * game.
     *
     * <p>A null host means "not live": the real resolution path always has a
     * host card ({@code AbilityUtils.resolveApiAbilityUnobserved} dereferences
     * it unconditionally), so this only picks the safer default for a state
     * that should not occur.
     *
     * <p>Typed {@link CardTraitBase} rather than {@link SpellAbility} so the
     * rewrite and trigger hooks, whose acting argument is a
     * {@code ReplacementEffect} or a {@code Trigger}, reuse this same check.
     */
    boolean owns(CardTraitBase ability) {
        if (ability == null) {
            return false;
        }
        Card host = ability.getHostCard();
        return host != null && host.getGame() == game;
    }

    /**
     * Whether an ability the cast or resolution hook hands over is this
     * game's: as {@link #owns(CardTraitBase)}, for an ability the hook has
     * already found non-null, except that a {@code LiveGame} with no game
     * owns none, not even an ability whose host has no game either (a
     * collector exercised against a real ability without a game in
     * progress).
     */
    boolean ownsStackAbility(SpellAbility ability) {
        Card host = ability.getHostCard();
        return host != null && game != null && host.getGame() == game;
    }

    /**
     * Whether a refusal the confirm hook reports naming no ability is this
     * game's: not while a fork runs on this thread, unless it is the fork
     * played on that this game's recording records.
     *
     * <p>Not {@link #ownsNullAbilityOutcome}: a refusal carries no params to
     * name a game, and the resolving-clause pointer is not asked.
     */
    boolean ownsNullAbilityRefusal() {
        return !anotherGameForksHere();
    }

    /**
     * Whether a fork other than this game is running on this thread: one
     * the thread's fork flag marks, and not this game played on with
     * collectors of its own.
     */
    private boolean anotherGameForksHere() {
        return Forks.isForkRunningOnThisThread() && Forks.playingForwardOnThisThread() != game;
    }

    /**
     * Whether a hook's argument is a trait positively identified as another
     * game's — the one case a trait-handing hook drops.
     *
     * <p>Not simply the negation of {@link #owns(CardTraitBase)}: an argument
     * that is not a trait at all — null, or whatever a drifted hook signature
     * passes in its place — cannot be identified, so it is kept and its record
     * names no line. Dropping it would make a signature change look like a
     * game in which nothing triggered.
     *
     * <p>Checked first in each handler, before the evaluation can spend any
     * sampling or dedup budget: a fork's evaluations must not skew the live
     * game's bookkeeping any more than they may reach its shard.
     */
    boolean foreign(Object hookArgument) {
        return hookArgument instanceof CardTraitBase trait && !owns(trait);
    }

    /**
     * As {@link #owns(CardTraitBase)}, for the combat-legality hook: the
     * attacker or defender Forge hands it is a {@code Card} or {@code Player}
     * directly, never a trait with a host, so the game comes straight off the
     * {@code GameEntity} itself.
     */
    boolean owns(GameEntity entity) {
        return entity != null && entity.getGame() == game;
    }

    /** Whether an outcome belongs to this game: by its ability, or by its params where it has none. */
    boolean ownsOutcome(SpellAbility ability, Map<?, ?> params) {
        return ability != null ? owns(ability) : ownsNullAbilityOutcome(params);
    }

    /**
     * The game the objects an outcome's params name belong to, or null when
     * they name none: the {@code "source"} card, else the first of the
     * {@code "subjects"} with a game. The objects decide first because they
     * are the evidence; the thread's signals are a guess for an outcome that
     * names no object (a bare amount).
     */
    private static Game gameNamedBy(Map<?, ?> params) {
        Game ofSource = gameOf(params.get("source"));
        if (ofSource != null) {
            return ofSource;
        }
        if (params.get("subjects") instanceof Iterable<?> subjects) {
            for (Object subject : subjects) {
                Game ofSubject = gameOf(subject);
                if (ofSubject != null) {
                    return ofSubject;
                }
            }
        }
        return null;
    }

    private static Game gameOf(Object named) {
        return named instanceof GameEntity entity ? entity.getGame() : null;
    }

    /**
     * Whether a null-ability outcome belongs to this game.
     *
     * <p>When a {@code source} or {@code subjects} entry names a game, that
     * decides. Otherwise:
     *
     * <p>The outcome hook's null-ability sources — {@code GameAction.reveal}'s
     * six-argument funnel and
     * {@code ReplacementHandler.runSingleReplaceDamageEffect}'s two
     * {@code damage_prevented} sites — pass a literal {@code null} ability on
     * the live game and on a fork alike, so {@link #owns(CardTraitBase)} has
     * nothing to compare, and keeping every one would readmit exactly what it
     * keeps out. Two signals instead, in this order:
     *
     * <ol>
     *   <li>{@link Forks#isForkRunningOnThisThread()}, set around the
     *   whole of both fork mechanisms, so it is true for every null-ability
     *   outcome a fork produces, whichever segment of it fires. First because
     *   it does not depend on the call being nested inside a resolution.
     *   <li>The resolving-clause pointer ({@link PatchHooks#currentSubAbility()}),
     *   which every resolution, live or forked, sets before running its body:
     *   an independent second signal, for a fork that resolves an ability
     *   outside {@code intervene} and {@code probe}.
     * </ol>
     *
     * <p>The first gives way to a fork {@link Forks#playingForward played on}
     * with collectors of its own: while it plays, a null-ability outcome on
     * this thread is the fork's turn-based action, and the fork's collectors
     * own it as the live game's own its turn-based actions.
     *
     * <p>Neither fires for a real combat-damage prevention on the live game —
     * {@code Combat.dealAssignedDamage} is a turn-based action, outside any
     * resolution and any fork — so that falls through to "belongs".
     *
     * <p>Not keyed on {@code damage_prevented}'s own {@code source} param: the
     * flag leaves no fork for it to catch, and {@code card_revealed} carries no
     * card to key on.
     */
    boolean ownsNullAbilityOutcome(Map<?, ?> params) {
        Game named = gameNamedBy(params);
        if (named != null) {
            return named == game;
        }
        // A fork played on to the end of the turn is a game of its own for
        // the span: its collectors own its turn-based outcomes, the live
        // game's own none.
        if (anotherGameForksHere()) {
            return false;
        }
        if (PatchHooks.currentSubAbility() instanceof SpellAbility resolving) {
            return owns(resolving);
        }
        return true;
    }

    /**
     * A new record about a trait — a trigger or a replacement — about to
     * handle an event, with everything but its payload set.
     *
     * <p>The actor is the host's controller ({@link #controllerOf}), the acting
     * line is the trait's own, and the state is {@link SnapshotBuilder#ofTrait}.
     * Shared so the trigger and rewrite channels describe a trait the same way;
     * the payload and the gates around the write differ per channel and stay
     * there.
     *
     * @param trait   the trait as the hook handed it over; anything that is not
     *                a {@link CardTraitBase} names no line and no host
     * @param pending the event as the trait received it
     */
    EffectRecord traitRecord(
            CollectorContext context, RecordKind kind, Object trait, EffectEvent pending) {
        return traitRecord(context, kind, trait, lineOf(trait), pending);
    }

    /**
     * The same, with the acting line already resolved by a caller that gated
     * on it before paying for the snapshot.
     *
     * @param line {@link #lineOf}{@code (trait)}
     */
    EffectRecord traitRecord(
            CollectorContext context, RecordKind kind, Object trait,
            ProvenanceKey.Resolved line, EffectEvent pending) {
        return traitRecordOn(context, kind, trait, line, traitState(trait, pending));
    }

    /**
     * The board a trait record carries, read now: {@link
     * SnapshotBuilder#ofTrait} for the trait's host and the event it is about
     * to handle.
     */
    RecordState traitState(Object trait, EffectEvent pending) {
        return traitSnapshot(trait, pending).state();
    }

    /** The same board, unrendered: a replacement's, read before it ran, rendered when it is written. */
    Snapshot traitSnapshot(Object trait, EffectEvent pending) {
        return snapshots.ofTrait(hostOf(trait), pending);
    }

    /**
     * The same record on a board read earlier ({@link #traitState}): a
     * replacement's, read before it ran.
     */
    EffectRecord traitRecordOn(
            CollectorContext context, RecordKind kind, Object trait,
            ProvenanceKey.Resolved line, RecordState state) {
        return context.newRecord(kind)
                .actor(controllerOf(hostOf(trait)))
                .ability(line)
                .state(state);
    }

    private static Card hostOf(Object trait) {
        return trait instanceof CardTraitBase acting ? acting.getHostCard() : null;
    }

    /** The printed line a hook's trait argument names, or why it names none. */
    static ProvenanceKey.Resolved lineOf(Object trait) {
        return ProvenanceResolver.resolve(trait instanceof CardTraitBase ctb ? ctb : null);
    }

    /**
     * The player whose line acted, falling back to the active player.
     *
     * <p>A triggered or replacement ability belongs to its host's controller,
     * which is often not whoever's turn it is: a death trigger fires on the
     * opponent's turn as often as on its own. Forge decides it the same way —
     * {@code TriggerHandler.runSingleTriggerUnobserved} defaults the controller to
     * the host card's.
     */
    String controllerOf(Card source) {
        return playerIdOrActive(source == null ? null : source.getController());
    }

    /** The deciding player's id, or the active player's where none is known. */
    String playerIdOrActive(Player decider) {
        return decider == null ? activePlayerId() : Refs.of(decider);
    }

    String activePlayerId() {
        var phase = game == null ? null : game.getPhaseHandler();
        if (phase == null || phase.getPlayerTurn() == null) {
            return null;
        }
        return Refs.of(phase.getPlayerTurn());
    }
}
