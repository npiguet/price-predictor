package com.pricepredictor.connector;

import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * One random play decision of the {@link RandomSeatController}, from the
 * draw at a priority to the play that completes or the pass owed when none
 * does (024 FR-022c).
 *
 * <p>The decision spans two engine calls, which is why it is state rather
 * than a local: {@link RandomSeatController#chooseSpellAbilityToPlay} draws a
 * candidate and {@link RandomSeatController#playChosenSpellAbility} plays it,
 * redrawing over the rest when it cannot be completed. When every candidate
 * is abandoned the play returns false, the engine asks for a play again at
 * the same priority, and the answer then is a pass, not another draw.
 */
final class RandomPlayDraw {
    private final Random random;
    /** The play the last successful draw picked, until it is played. */
    private SpellAbility drawn;
    /** The other playable candidates of that draw, for a redraw. */
    private List<SpellAbility> remaining = new ArrayList<>();
    /** Set when every candidate of a draw was abandoned: the next priority passes. */
    private boolean passOwed;

    RandomPlayDraw(Random random) {
        this.random = random;
    }

    /**
     * Begin a priority: drop a pick the engine never played, and consume
     * the pass an exhausted draw owes.
     *
     * @return whether this priority must pass
     */
    boolean beginPriority() {
        drawn = null;
        if (!passOwed) {
            return false;
        }
        passOwed = false;
        return true;
    }

    /** Pick one of {@code candidates}, none empty, keeping them all for a redraw. */
    SpellAbility draw(List<SpellAbility> candidates) {
        remaining = candidates;
        drawn = pick();
        return drawn;
    }

    /**
     * Whether {@code sa} is the play the last draw picked; claiming it
     * clears the pick, so the engine's next call is the AI's again.
     */
    boolean claims(SpellAbility sa) {
        if (sa != drawn) {
            return false;
        }
        drawn = null;
        return true;
    }

    /** The next candidate once {@code abandoned} could not be completed, or null when none remain. */
    SpellAbility nextAfterAbandon(SpellAbility abandoned) {
        remaining.removeIf(candidate -> candidate == abandoned);
        return remaining.isEmpty() ? null : pick();
    }

    /** Every candidate was abandoned: the seat passes at its next priority. */
    void exhausted() {
        passOwed = true;
    }

    private SpellAbility pick() {
        return remaining.get(random.nextInt(remaining.size()));
    }
}
