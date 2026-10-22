package com.pricepredictor.connector.effects;

import forge.game.CardTraitBase;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

/**
 * The game a recording is for, and what every patched hook asks of it.
 *
 * <p>The patched hooks are plain JVM statics with no {@code Game} reference
 * of their own — whichever resolution fires them last wins.
 * {@code ForkCollector.forceResolution} fires them too:
 * {@code GameSimulator.resolveStack} builds its own {@code PlayerControllerAi}
 * and unconditionally calls
 * {@code setUseSimulation(AIOption.USE_FULL_SIMULATION)}, regardless of how
 * this game's own two lobby seats were registered, so a forked ability
 * resolves through the exact same {@code AbilityUtils.resolveApiAbility} the
 * hooks are installed against. {@link ForkEventSink} keeps a fork's own
 * <em>bus</em> events out of this game's records by subscribing to the fork's
 * own {@code Game} instead of this one, but the hooks are not bus
 * subscriptions, so that separation never reaches them — without the checks
 * here, a fork's {@code vote_taken} or {@code coin_flipped} lands in the live
 * game's bracket with nothing marking it as hypothetical.
 *
 * <p>Game identity is the check because it needs no cooperation from
 * {@code ForkCollector} or {@code GameSimulator}, neither of which know the
 * channels exist: {@code GameCopier} builds every copied {@code Card} against
 * a new {@code Game} object ({@code new Card(id, paperCard, newGame)}), so a
 * fork's clause's own host card already carries the answer. Compared by
 * reference, the same way {@code SpellAbility.setActivatingPlayer} itself
 * already compares players ("don't use equals because player might be from
 * simulation") — a copied {@code Game} is never {@code equals} to the live one
 * by accident, only by being the same object.
 *
 * <p>Each hook checks in its own handler, immediately before its delivery
 * gate (the bracket for clauses and outcomes, sampling and dedup for the
 * others), rather than inside {@link BusBracketCollector#recordClauseEvent}:
 * that method only ever receives the already-built {@code EffectEvent}, not
 * the {@code SpellAbility} that produced it, so checking there would mean
 * widening a documented, two-call-site contract just to re-derive something
 * both callers already have in scope.
 */
final class LiveGame {

    private final Game game;
    private final SnapshotBuilder snapshots;

    LiveGame(Game game) {
        this.game = game;
        this.snapshots = new SnapshotBuilder(game);
    }

    Game game() {
        return game;
    }

    /** How this game's state is rendered into a record. */
    SnapshotBuilder snapshots() {
        return snapshots;
    }

    /**
     * Whether a clause, replacement, trigger or mana ability belongs to this
     * game.
     *
     * <p>A trait with no identifiable host is treated as not belonging rather
     * than let through by default: on the real resolution path a host card is
     * always present (the same assumption
     * {@code AbilityUtils.resolveApiAbilityUnobserved} already makes,
     * unconditionally dereferencing it), so this branch is never live in
     * production and only ever chooses the safer of two defaults for a state
     * that should not occur.
     *
     * <p>Typed {@link CardTraitBase} rather than {@link SpellAbility}:
     * {@code SpellAbility}, {@code ReplacementEffect} and {@code Trigger} all
     * extend it and all declare {@code getHostCard()} on it, so the rewrite and
     * trigger hooks — whose acting argument is a
     * {@code ReplacementEffect}/{@code Trigger}, never a {@code SpellAbility} —
     * reuse this same check rather than a near-duplicate.
     */
    boolean owns(CardTraitBase ability) {
        if (ability == null) {
            return false;
        }
        Card host = ability.getHostCard();
        return host != null && host.getGame() == game;
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

    /**
     * Whether a null-ability outcome belongs to this game.
     *
     * <p>The outcome hook's two production null-ability sources —
     * {@code GameAction.reveal}'s six-argument funnel and
     * {@code ReplacementHandler.runSingleReplaceDamageEffect}'s two
     * {@code damage_prevented} sites — call {@code EffectRecordOutcomes.note}
     * with a literal {@code null} ability, always, on the live game exactly as
     * on a fork: {@link #owns(CardTraitBase)} has nothing to compare, and
     * simply keeping every null-ability outcome would readmit exactly what
     * that method exists to keep out.
     *
     * <p>Two signals, checked in this order:
     *
     * <ol>
     *   <li>{@link ForkCollector#isForkRunningOnThisThread()}. Set around the
     *   whole of {@code ForkCollector.intervene} and {@code
     *   ForkCollector.probe} — both of the recording's only two fork
     *   mechanisms — so "a fork is running on this thread" is by construction
     *   true for every null-ability outcome a fork can produce today,
     *   whichever of the two produces it and whichever segment of either it
     *   fires from. Tried first because, unlike the pointer below, it does not
     *   depend on the call happening to be nested inside an ability
     *   resolution.
     *   <li>The resolving-clause pointer ({@code
     *   AbilityUtils.getEffectRecordSubAbility}, read via {@link
     *   PatchHooks#currentSubAbility()}), when the flag is false. Every
     *   ability resolves through {@code AbilityUtils.resolve}/{@code
     *   resolveApiAbility}, live or forked alike, and both set this pointer to
     *   the ability currently resolving before running its body. Kept as a
     *   second, independent signal although the flag alone is sufficient for
     *   both current fork mechanisms: insurance against a future one that
     *   resolves an ability without going through {@code intervene}/{@code
     *   probe}. When the flag is true this branch is redundant with it (both
     *   cover the resolving case), not a second opinion; it only does
     *   independent work when the flag is false.
     * </ol>
     *
     * <p>Neither signal fires for a real combat-damage prevention on the live
     * game — {@code Combat.dealAssignedDamage} is a turn-based action, never
     * inside an ability resolution and never inside a fork — which is the
     * intended "belongs to the live game" default this method falls through
     * to.
     *
     * <p><b>Why not also use {@code damage_prevented}'s own {@code "source"}
     * param</b> (a raw {@code Card}, held by the outcome hook before it is
     * converted to a ref): with the flag covering the entirety of both fork
     * mechanisms, there is no remaining path where {@code source} would catch a
     * fork this method's two signals miss — it would be a third check for a gap
     * that is already closed, not a narrower one. It is also not general:
     * {@code card_revealed}'s params carry no card at all (just a count and a
     * zone name), so a {@code source}-keyed check could only ever cover one of
     * the two known null-ability types, where the flag and the pointer both
     * cover either.
     */
    boolean ownsNullAbilityOutcome() {
        if (ForkCollector.isForkRunningOnThisThread()) {
            return false;
        }
        if (PatchHooks.currentSubAbility() instanceof SpellAbility resolving) {
            return owns(resolving);
        }
        return true;
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
        if (source == null || source.getController() == null) {
            return activePlayerId();
        }
        return SnapshotBuilder.playerId(source.getController());
    }

    /** The deciding player's id, or the active player's where none is known. */
    String playerIdOrActive(Player decider) {
        return decider == null ? activePlayerId() : SnapshotBuilder.playerId(decider);
    }

    String activePlayerId() {
        var phase = game == null ? null : game.getPhaseHandler();
        if (phase == null || phase.getPlayerTurn() == null) {
            return null;
        }
        return SnapshotBuilder.playerId(phase.getPlayerTurn());
    }
}
