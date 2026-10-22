package com.pricepredictor.connector.effects;

import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;

/**
 * The two combat damage steps.
 *
 * <p>A first-strike combat deals damage twice, and each step writes its own
 * combat record, so a record names the one it describes. A probe branch and
 * the observed record it mirrors are written by different classes and have to
 * agree on that name, which is why it is spelled once, here.
 */
public enum DamageStep {
    FIRST_STRIKE("first_strike"),
    REGULAR("regular");

    private final String wireName;

    DamageStep(String wireName) {
        this.wireName = wireName;
    }

    /** The {@code combat_substep} a record carries. */
    public String wireName() {
        return wireName;
    }

    /**
     * The damage step the game is in, or null outside both.
     *
     * <p>Both steps are named: Forge's first-strike step is {@code
     * COMBAT_FIRST_STRIKE_DAMAGE}, and a test for {@code COMBAT_DAMAGE} alone
     * sent the step's creature damage, life changes and deaths to the ordinary
     * bracket.
     */
    static DamageStep of(PhaseHandler phase) {
        if (phase == null) {
            return null;
        }
        PhaseType step = phase.getPhase();
        if (step == PhaseType.COMBAT_FIRST_STRIKE_DAMAGE) {
            return FIRST_STRIKE;
        }
        return step == PhaseType.COMBAT_DAMAGE ? REGULAR : null;
    }
}
