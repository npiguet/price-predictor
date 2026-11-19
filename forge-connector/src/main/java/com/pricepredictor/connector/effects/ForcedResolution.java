package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import com.pricepredictor.connector.effects.directed.KeptRecords;
import forge.ai.AIOption;
import forge.ai.PlayerControllerAi;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.event.GameEventSpellAbilityCast;
import forge.game.player.Player;
import forge.game.player.PlayerController;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.SpellAbilityStackInstance;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * One forced resolution, recorded by the live bracket's own code.
 *
 * <p>The fork is recorded as {@link ForkPlayForward} records a fork it plays
 * on: by a fork-bound {@link GameRecording} whose {@link BusBracketCollector}
 * hears the fork's bus, and to which the live game's hooks hand the fork's
 * callbacks ({@link GameRecording#routingTo}). So a forced effect half is
 * filed by the rules a live one is: the board at the resolution's start, the
 * modes' halves, the conditions, the refusals, the spell's move off the
 * stack, the state-based check's deaths and what they take off, the steps an
 * effect ends the turn into, the mana an unless cost was paid with.
 *
 * <p>Of everything the fork's recording writes, only the forced item's
 * effect halves are kept ({@link #keep}): the halves under its cost half's
 * link, one per chosen mode of a modal item. The cost half describes a
 * payment the fork never made, and the records of what the cast triggered
 * and of what the resolution caused are those items' own, as they are live.
 * A kept half loses the link, since its cost half is not written, and is
 * marked {@code interventional} and {@code fork}, with what the copy left out.
 *
 * <p>The stack is resolved only until the forced item has left it, the
 * state-based check after it run and the triggers it left put on the stack:
 * from there nothing more is the forced item's, and the recording's close
 * writes the half the next action would have closed.
 */
final class ForcedResolution {

    private final GameRecording live;
    private final Forks.VerifiedFork verified;
    /** Seeds the fork recording's sampling stream, whose records are not kept. */
    private final long seed;
    /** The forced item's link, which its effect halves carry; null until it is on the stack. */
    private String link;
    private final List<EffectRecord> halves = new ArrayList<>();

    /**
     * @param live     the live game's recording, whose hooks route to the fork's
     * @param verified the fork the item is forced in
     * @param seed     seeds the fork recording's sampling stream
     */
    ForcedResolution(GameRecording live, Forks.VerifiedFork verified, long seed) {
        this.live = live;
        this.verified = verified;
        this.seed = seed;
    }

    /**
     * Put {@code ability}, its choices drawn, on the fork's stack and resolve
     * it for {@code actor}, recording the fork as it goes.
     *
     * @return this, its {@link #outcome} read
     * @throws ForkRefusal when nothing reached the stack
     */
    ForcedResolution force(SpellAbility ability, Player actor) {
        Game fork = verified.fork();
        fork.copyLastState();
        GameRecording recording = live.forkRecording(fork, this::keep, line -> { }, seed, KeptRecords.NONE);
        try {
            recording.open();
            live.routingTo(recording, () -> Forks.playingForward(fork, () -> {
                resolve(fork, ability, actor);
                return null;
            }));
        } finally {
            // Writes the half the next action would have closed.
            recording.close();
        }
        return this;
    }

    /**
     * The forced item's effect halves, in the order written: one, or one per
     * chosen mode; none when it never began to resolve.
     */
    List<EffectRecord> outcome() {
        return List.copyOf(halves);
    }

    /**
     * The fork recording's flags, which every record it writes is handed
     * before it is rendered: a forced item's effect half is kept, stamped as
     * a fork's, and everything else is not.
     */
    private void keep(EffectRecord record) {
        if (record.isEffectHalfOf(link)) {
            verified.stamp(record.linkId(null).interventional(true).fork(true));
            halves.add(record);
        }
    }

    /**
     * Push the forced item and resolve the fork's stack until it has left,
     * the opponent deciding as the simulator's AI, as {@code
     * GameSimulator.resolveStack} has it decide, and the actor as its own
     * seat, the plain AI with its mandatory choices made
     * ({@link MandatoryChoiceAi#deciding}).
     *
     * @throws ForkRefusal when nothing reached the stack, or a forced Aura
     *         spell resolved and stayed in the stack zone
     */
    private void resolve(Game fork, SpellAbility ability, Player actor) {
        SpellAbility onStack = FirstCast.push(fork, ability);
        if (onStack == null) {
            // Mostly a special action: foretell, plot and suspend are
            // AbilityStatic, which MagicStack.add resolves on the spot
            // without a push, and live opens no bracket for one either.
            // Otherwise a frozen stack parked it or its targeting was
            // refused, and nothing will resolve.
            throw new ForkRefusal(InterventionFork.NOT_ON_THE_STACK);
        }
        link = live.linkOf(onStack);
        Player opponent = actor.getWeakestOpponent();
        PlayerControllerAi simulated = new PlayerControllerAi(fork, opponent, opponent.getLobbyPlayer());
        simulated.getAi().setUseSimulation(AIOption.USE_FULL_SIMULATION);
        PlayerController deciding = MandatoryChoiceAi.deciding(actor);
        actor.runWithController(() -> opponent.runWithController(
                () -> untilItHasLeft(fork, onStack.getId()), simulated), deciding);
        if (!fork.isGameOver() && strandedAura(fork, onStack)) {
            throw new ForkRefusal(InterventionFork.AURA_LEFT_ON_THE_STACK);
        }
    }

    /**
     * Resolve the top of the stack, then the state-based check and the
     * triggers, until the forced item is no longer on the stack or the game
     * is over.
     */
    private static void untilItHasLeft(Game fork, int forcedId) {
        Set<Card> affected = new HashSet<>();
        fork.getAction().checkStateEffects(false, affected);
        fork.getStack().addAllTriggeredAbilitiesToStack();
        while (holds(fork, forcedId) && !fork.isGameOver()) {
            // A belt (FR-010d): the copier copies no stack, so the copy's
            // holds only the forced line, never a restart, and what its cast
            // triggered.
            Forks.refuseARestart(fork);
            fork.getStack().resolveStack();
            fork.getAction().checkStateEffects(false, affected);
            fork.getStack().addAllTriggeredAbilitiesToStack();
        }
    }

    /**
     * Whether {@code forced} is an Aura spell whose card is still in the
     * stack zone: its attachment chose nothing, so it neither entered the
     * battlefield nor fizzled into the graveyard.
     */
    static boolean strandedAura(Game fork, SpellAbility forced) {
        Card card = fork.getCardState(forced.getHostCard());
        return forced.isSpell() && card.isAura() && card.isInZone(ZoneType.Stack);
    }

    private static boolean holds(Game fork, int abilityId) {
        for (SpellAbilityStackInstance instance : fork.getStack()) {
            SpellAbility held = instance.getSpellAbility();
            if (held != null && held.getId() == abilityId) {
                return true;
            }
        }
        return false;
    }

    /**
     * The first ability cast onto the fork's stack, read off the stack as
     * the live bracket reads it: an activated ability goes on as a copy of
     * the line it was handed. The copy is taken at a phase boundary with
     * nothing in flight, so the first cast is the forced one.
     */
    static final class FirstCast {
        private final Game fork;
        private SpellAbility onStack;

        private FirstCast(Game fork) {
            this.fork = fork;
        }

        /** Put {@code ability} on {@code fork}'s stack: what it put there, or null when nothing reached it. */
        static SpellAbility push(Game fork, SpellAbility ability) {
            FirstCast cast = new FirstCast(fork);
            fork.subscribeToEvents(cast);
            // The call a real cast ends in, so a spell's card moves to the
            // stack zone first, as casting moves it.
            fork.getStack().addAndUnfreeze(ability);
            return cast.onStack;
        }

        @Subscribe
        public void onCast(GameEventSpellAbilityCast event) {
            if (onStack == null) {
                onStack = StackActions.topAbility(fork.getStack());
            }
        }
    }
}
