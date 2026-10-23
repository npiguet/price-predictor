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
 * of their own, and a fork fires them too: {@code GameSimulator.resolveStack}
 * resolves a forked ability through the same
 * {@code AbilityUtils.resolveApiAbility} the hooks are installed against.
 * {@link ForkEventSink} keeps a fork's <em>bus</em> events out of this game's
 * records by subscribing to the fork's own {@code Game}, but the hooks are
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
     * <p>The outcome hook's null-ability sources — {@code GameAction.reveal}'s
     * six-argument funnel and
     * {@code ReplacementHandler.runSingleReplaceDamageEffect}'s two
     * {@code damage_prevented} sites — pass a literal {@code null} ability on
     * the live game and on a fork alike, so {@link #owns(CardTraitBase)} has
     * nothing to compare, and keeping every one would readmit exactly what it
     * keeps out. Two signals instead, in this order:
     *
     * <ol>
     *   <li>{@link ForkCollector#isForkRunningOnThisThread()}, set around the
     *   whole of both fork mechanisms, so it is true for every null-ability
     *   outcome a fork produces, whichever segment of it fires. First because
     *   it does not depend on the call being nested inside a resolution.
     *   <li>The resolving-clause pointer ({@link PatchHooks#currentSubAbility()}),
     *   which every resolution, live or forked, sets before running its body:
     *   an independent second signal, for a fork that resolves an ability
     *   outside {@code intervene} and {@code probe}.
     * </ol>
     *
     * <p>Neither fires for a real combat-damage prevention on the live game —
     * {@code Combat.dealAssignedDamage} is a turn-based action, outside any
     * resolution and any fork — so that falls through to "belongs".
     *
     * <p>Not keyed on {@code damage_prevented}'s own {@code source} param: the
     * flag leaves no fork for it to catch, and {@code card_revealed} carries no
     * card to key on.
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
        return playerIdOrActive(source == null ? null : source.getController());
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
