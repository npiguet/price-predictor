package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

import forge.game.card.Card;

import java.util.function.Predicate;

/**
 * Forge's closed list of marks a permanent carries apart from its
 * characteristics, as an entity's {@code designations} spell them.
 *
 * <p>Constants are in wire-name order, so a card's list is written sorted.
 * {@code DESIGNATIONS} in {@code effect_head_input.py} (and its twin in {@code
 * rule_detectors/declared_values.py}) is the other half of this vocabulary, in
 * the same order.
 */
enum Designation implements WireNamed {
    BESTOWED("bestowed", Card::isBestowed),
    CLOAKED("cloaked", Card::isCloaked),
    COMMANDER("commander", Card::isCommander),
    FORETOLD("foretold", Card::isForetold),
    GOADED("goaded", Card::isGoaded),
    HARNESSED("harnessed", Card::isHarnessed),
    MANIFESTED("manifested", Card::isManifested),
    MONSTROUS("monstrous", Card::isMonstrous),
    PHASED_OUT("phased_out", Card::isPhasedOut),
    PLOTTED("plotted", Card::isPlotted),
    RENOWNED("renowned", Card::isRenowned),
    RING_BEARER("ring_bearer", Card::isRingBearer),
    SADDLED("saddled", Card::isSaddled),
    SOLVED("solved", Card::isSolved),
    SPECIALIZED("specialized", Card::isSpecialized),
    SUSPECTED("suspected", Card::isSuspected),
    UNEARTHED("unearthed", Card::isUnearthed),
    WARPED("warped", Card::isWarped);

    private final String wireName;
    private final Predicate<Card> held;

    Designation(String wireName, Predicate<Card> held) {
        this.wireName = wireName;
        this.held = held;
    }

    /** The name an entity's {@code designations} list carries. */
    @Override
    public String wireName() {
        return wireName;
    }

    /** Whether {@code card} holds this designation. */
    boolean heldBy(Card card) {
        return held.test(card);
    }
}
