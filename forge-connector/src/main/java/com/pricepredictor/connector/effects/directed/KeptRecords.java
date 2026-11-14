package com.pricepredictor.connector.effects.directed;

import forge.game.CardTraitBase;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;

import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The records a directed fork keeps whatever the samplers say (FR-010c): the
 * decision candidates of the cards its variant names, and the not-fired
 * evaluations of the triggers those cards host.
 *
 * <p>Most directed forks reach their rule's point with no choice, and the
 * record that shows it is a decision candidate written at a sampling rate
 * ({@code --playability-rate} and the candidate rates), or a trigger's
 * "evaluated, did not fire" record, which the trigger-negative sampler
 * offers only to repay a firing of the same mode earlier in the same
 * recording (a fork's recording is fresh). A fork of such a rule would
 * otherwise come back with nothing its detector can tag.
 *
 * <p>Cards are named by id, which a card keeps through the fork's zone
 * changes; a card a variant puts in another's place has a new id, and is
 * named by the variant after the edit if at all.
 *
 * @param candidateHosts the ids of the cards whose decision candidates are kept
 * @param triggerHosts   the ids of the cards whose triggers' not-fired
 *                       evaluations are kept
 */
public record KeptRecords(Set<Integer> candidateHosts, Set<Integer> triggerHosts) {

    /** Nothing kept: every record of the fork is sampled as a live game's is. */
    public static final KeptRecords NONE = new KeptRecords(Set.of(), Set.of());

    public KeptRecords {
        candidateHosts = Set.copyOf(candidateHosts);
        triggerHosts = Set.copyOf(triggerHosts);
    }

    /** The decision candidates of {@code cards}. */
    public static KeptRecords candidatesOf(Card... cards) {
        return new KeptRecords(ids(cards), Set.of());
    }

    /** The not-fired evaluations of the triggers {@code cards} host. */
    public static KeptRecords triggersOf(Card... cards) {
        return new KeptRecords(Set.of(), ids(cards));
    }

    /** This and {@code more} together. */
    public KeptRecords and(KeptRecords more) {
        return new KeptRecords(
                Stream.concat(candidateHosts.stream(), more.candidateHosts.stream()).collect(Collectors.toSet()),
                Stream.concat(triggerHosts.stream(), more.triggerHosts.stream()).collect(Collectors.toSet()));
    }

    /** Whether nothing is kept. */
    public boolean isEmpty() {
        return candidateHosts.isEmpty() && triggerHosts.isEmpty();
    }

    /** Whether {@code candidate}'s decision record is written whatever the decision sample draws. */
    public boolean keepsCandidate(SpellAbility candidate) {
        return !candidateHosts.isEmpty() && candidate != null && hostedBy(candidate.getHostCard(), candidateHosts);
    }

    /**
     * Whether a not-fired evaluation of {@code trigger} is offered for writing
     * whatever the trigger-negative sampler's balance says.
     */
    public boolean keepsNegative(Object trigger) {
        return !triggerHosts.isEmpty() && trigger instanceof CardTraitBase trait
                && hostedBy(trait.getHostCard(), triggerHosts);
    }

    private static boolean hostedBy(Card host, Set<Integer> ids) {
        return host != null && ids.contains(host.getId());
    }

    private static Set<Integer> ids(Card... cards) {
        return Stream.of(cards).map(card -> Objects.requireNonNull(card, "card").getId())
                .collect(Collectors.toSet());
    }
}
