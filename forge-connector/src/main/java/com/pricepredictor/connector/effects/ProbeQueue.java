package com.pricepredictor.connector.effects;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

/**
 * Probe branches waiting for the combat record each mirrors.
 *
 * <p>A branch is taken before its damage step and the record it mirrors is
 * written after it, so the branch waits here, in the order the steps happen.
 * A branch and its record have to describe the same step: one taken at the
 * first-strike step is completed against that step's record, and one whose
 * step wrote no record is dropped rather than paired with the next.
 */
final class ProbeQueue {

    private final List<DamageStepProbe.HeldProbe> held = new ArrayList<>();

    void hold(DamageStepProbe.HeldProbe branch) {
        held.add(branch);
    }

    boolean isEmpty() {
        return held.isEmpty();
    }

    /** The damage step of the oldest branch waiting, which comes first. */
    DamageStep oldestSubstep() {
        return held.get(0).substep();
    }

    /**
     * Take the branches of one damage step out of the queue, keeping the rest
     * in order.
     */
    List<DamageStepProbe.HeldProbe> take(DamageStep substep) {
        return removeIf(branch -> branch.substep() == substep);
    }

    /** Drop the branches of every other damage step, and say how many went. */
    int discardOtherThan(DamageStep substep) {
        return removeIf(branch -> branch.substep() != substep).size();
    }

    private List<DamageStepProbe.HeldProbe> removeIf(Predicate<DamageStepProbe.HeldProbe> test) {
        List<DamageStepProbe.HeldProbe> removed = new ArrayList<>();
        List<DamageStepProbe.HeldProbe> remaining = new ArrayList<>();
        for (DamageStepProbe.HeldProbe branch : held) {
            (test.test(branch) ? removed : remaining).add(branch);
        }
        held.clear();
        held.addAll(remaining);
        return removed;
    }
}
