package com.pricepredictor.connector.effects;

import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.StringJoiner;

/**
 * The live collector's cost halves, held from each cast until its outcome is
 * known, and written exactly once with that outcome.
 *
 * <p>The record cannot be written at the cast, because {@code outcome} is a
 * field of the cost half and the answer to it does not exist until the spell
 * resolves or leaves the stack without resolving. Written at the cast, every
 * activation would read {@code resolved}, and a fizzled spell would issue a
 * {@code link_id} whose effect half never comes.
 *
 * <p>A list rather than a map because it is the stack: two spells can be
 * waiting at once and the inner one resolves first, so a lookup runs from the
 * newest end. Nothing here has been written yet — see
 * {@link PendingActivation} for why the record has to wait.
 */
final class ActivationLedger {

    private final CollectorContext context;

    /** Casts whose outcome is not known yet, oldest first. */
    private final List<PendingActivation> casts = new ArrayList<>();

    ActivationLedger(CollectorContext context) {
        this.context = context;
    }

    /**
     * A cast that has been observed but whose outcome is not settled.
     *
     * <p>Holds what was read rather than the {@link SpellAbility}, because
     * what was paid lives on the ability object and its next activation clears
     * it — the reason {@link PaidCosts#of} must be called at the cast and
     * nowhere later. The record id and the timestamp are taken at
     * the cast too, so a held record still says when the spell was cast rather
     * than when it stopped being on the stack.
     */
    private record PendingActivation(
            int abilityId,
            String recordId,
            String timestamp,
            PaidCosts costs,
            String actor,
            ProvenanceKey.Resolved ability,
            String state,
            String linkId,
            int targetsAtCast) {
    }

    /**
     * Hold the cost half of a cast that has just been announced.
     *
     * <p>The costs are read <b>now</b> — {@link PaidCosts} says why they are
     * unreadable a moment later.
     *
     * @param state the board at the cast, rendered
     */
    void hold(SpellAbility ability, String actor, ProvenanceKey.Resolved keys, String state) {
        casts.add(new PendingActivation(
                ability.getId(), context.nextRecordId(),
                RecordShardWriter.timestamp(), PaidCosts.of(ability),
                actor, keys, state,
                context.gameId() + ".link." + ability.getId(), targetCount(ability)));
    }

    /** Whether nothing is held. */
    boolean isEmpty() {
        return casts.isEmpty();
    }

    /** Whether a cast of this ability id is held: its cast event has been seen. */
    boolean holds(int abilityId) {
        return find(abilityId).isPresent();
    }

    /**
     * A cast of an ability whose earlier cast is still held, written off as
     * any removal is.
     *
     * <p>Defensive: no real game reaches it. One ability is on the stack at
     * most once — an activation and a trigger are each a fresh object, a copy
     * has an id of its own — and a card that leaves the stack is rebuilt by
     * {@code CardCopyService.copyCard} with fresh spell ability ids, so a
     * card cast again always casts a new id. Were an id ever reused while its
     * earlier cast was held, that cast would otherwise be settled by the
     * later one's resolution.
     */
    void writeOffEarlierCast(int abilityId) {
        take(abilityId).ifPresent(
                earlier -> emit(earlier, ActivationOutcome.COUNTERED, false));
    }

    /**
     * What became of a cast that has left the stack through a resolution.
     *
     * @param resolved the resolved ability, or null when no bracket was held
     *                 for it
     */
    ActivationOutcome settledOutcome(
            int abilityId, boolean fizzled, boolean declined, SpellAbility resolved) {
        if (fizzled) {
            return ActivationOutcome.FIZZLED;
        }
        if (declined) {
            return ActivationOutcome.DECLINED;
        }
        return outcomeOf(
                find(abilityId).map(PendingActivation::targetsAtCast).orElse(0), resolved);
    }

    /**
     * Write a held cost half now that its outcome is known.
     *
     * <p>{@code linked} is separate from the outcome because they answer
     * different questions: the outcome says what became of the spell, and
     * {@code link_id} promises that an effect half carrying the same id exists.
     * A resolution this collector holds no bracket for -- an item whose
     * bracket was written off before it resolved -- is a {@code resolved}
     * activation with no link, which is honest; issuing the link anyway would
     * leave it dangling.
     *
     * @return the link the effect half may claim, or null when there is to be
     *         no effect half or nothing was held for this ability
     */
    String settle(int abilityId, ActivationOutcome outcome, boolean linked) {
        Optional<PendingActivation> found = take(abilityId);
        if (found.isEmpty()) {
            return null;
        }
        PendingActivation activation = found.get();
        emit(activation, outcome, linked);
        return linked ? activation.linkId() : null;
    }

    /**
     * Write off every held cast that has left the stack without resolving.
     *
     * <p>Counterspells, {@code CostExileFromStack}, a spell bounced off the
     * stack and {@code MagicStack.clear} all take an entry away with no event
     * this collector can tell apart from a resolution -- {@code finishResolving}
     * publishes the same removal for a spell that resolved normally. So the
     * question is asked of the stack itself, at every moment the stack can have
     * changed: a held cast that is no longer on it and did not resolve was
     * removed, and {@code countered} is what the schema calls that.
     *
     * <p>It is the nearest of the five outcomes rather than an exact one. A
     * spell exiled off the stack by a cost was not countered in the rules sense;
     * what the corpus needs to know, and what all of these share, is that the
     * cost was paid and no effect followed.
     *
     * @param onStack the ability ids still on the stack
     * @return how many held casts were written off
     */
    int writeOffRemoved(Set<Integer> onStack) {
        int removed = 0;
        for (Iterator<PendingActivation> it = casts.iterator(); it.hasNext();) {
            PendingActivation cast = it.next();
            if (!onStack.contains(cast.abilityId())) {
                it.remove();
                emit(cast, ActivationOutcome.COUNTERED, false);
                removed++;
            }
        }
        return removed;
    }

    /**
     * The game has ended: drop every cast still held, counted and named on
     * stderr.
     *
     * <p>What is still held after the last write-off is genuinely still on
     * the stack: it neither resolved nor was removed, and {@code outcome} has
     * no member for that. So it is dropped rather than stamped with one of the
     * five outcomes it did not have.
     *
     * @return how many held casts were abandoned rather than written
     */
    long abandonAll() {
        List<PendingActivation> abandoned = List.copyOf(casts);
        casts.clear();
        if (!abandoned.isEmpty()) {
            StringJoiner names = new StringJoiner(", ");
            for (PendingActivation activation : abandoned) {
                names.add(activation.linkId());
            }
            System.err.println(
                    "effect records: " + context.gameId() + " ended with " + abandoned.size()
                            + " cast(s) still on the stack, dropped because"
                            + " outcome has no member for a game that stopped: "
                            + names);
        }
        return abandoned.size();
    }

    /**
     * Whether a resolution lost some of its targets on the way.
     *
     * <p>{@code MagicStack.hasFizzled} strips the targets that became illegal
     * from the ability before resolving it, and only then does the ability
     * resolve — so comparing the count with the one read at the cast is the
     * engine's own answer to "did part of this spell do nothing". Nothing else
     * in the record distinguishes a Lightning Helix that hit from one whose
     * creature had already died.
     *
     * <p>Only the root line's targets are compared. A partial fizzle confined to
     * a sub-ability reads as {@code resolved}, which understates rather than
     * invents. Losing <b>every</b> target without the engine calling it a fizzle
     * is the {@code CantFizzle} case (Gilded Drake), where the spell still does
     * what it does: that is {@code resolved} too.
     */
    static ActivationOutcome outcomeOf(int targetsAtCast, SpellAbility resolved) {
        if (resolved == null) {
            return ActivationOutcome.RESOLVED;
        }
        int surviving = targetCount(resolved);
        return surviving > 0 && surviving < targetsAtCast
                ? ActivationOutcome.PARTIALLY_FIZZLED
                : ActivationOutcome.RESOLVED;
    }

    private static int targetCount(SpellAbility ability) {
        return ability == null || ability.getTargets() == null
                ? 0 : ability.getTargets().size();
    }

    /** The held cast for an ability id, newest first. */
    private Optional<PendingActivation> find(int abilityId) {
        for (int i = casts.size() - 1; i >= 0; i--) {
            if (casts.get(i).abilityId() == abilityId) {
                return Optional.of(casts.get(i));
            }
        }
        return Optional.empty();
    }

    /** The held cast for an ability id, no longer held. */
    private Optional<PendingActivation> take(int abilityId) {
        Optional<PendingActivation> found = find(abilityId);
        found.ifPresent(casts::remove);
        return found;
    }

    private void emit(PendingActivation activation, ActivationOutcome outcome, boolean linked) {
        context.emit(context.newResolution(
                Moment.ACTIVATION, activation.recordId(), activation.timestamp())
                .actor(activation.actor())
                .ability(activation.ability())
                .linkId(linked ? activation.linkId() : null)
                .state(activation.state())
                .payload(RecordPayloads.costPayload(activation.costs(), outcome)));
    }
}
