package com.pricepredictor.connector.effects;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Which trigger evaluations reach the writer: every firing, about as many
 * non-firings per mode, and no flood from one line.
 *
 * <p>Non-fired negatives are what teach the condition; without them the model
 * would learn the base rate of a trigger type firing and nothing about when.
 * A turn evaluates far more conditions than it fires, so the negatives are
 * sampled to roughly 1:1.
 *
 * <p>Per mode, and per game — one sampler is built per game. A mode that never
 * fires contributes no negatives, which is the intended reading of "negatives
 * drawn from same-event-type evaluations". The aggregate ratio is therefore
 * near 1:1 rather than exactly it, and the per-mode ratios are the ones to
 * validate.
 */
final class TriggerNegativeSampler {

    /** Negatives kept per positive kept, per trigger mode. */
    private static final double TARGET_NEGATIVE_RATIO = 1.0;
    /**
     * Negatives a mode may be offered at full rate before it backs off.
     *
     * <p>Only reached by a mode whose offered negatives are being refused
     * downstream -- by the duplicate check or the per-line flood cap -- so it
     * is a bound on wasted snapshot building, not a sampling parameter.
     */
    private static final int NEGATIVE_OFFERS_AT_FULL_RATE = 4;
    /** No negative is ever kept with probability below this. */
    private static final double MIN_NEGATIVE_RATE = 1.0 / 4096.0;
    /** Records one acting line may write per verdict in one game. */
    private static final int TRIGGER_RECORDS_PER_LINE = 24;

    /** The game's shared sampling stream; see {@link PatchedCollectors}. */
    private final Random sampler;
    /** Each trigger mode's written evaluations and pending offers, this game. */
    private final Map<String, ModeBalance> triggerBalance = new HashMap<>();
    /** Records written per acting line and verdict, for the flood cap. */
    private final Map<LineVerdict, Integer> triggerPerLine = new HashMap<>();

    TriggerNegativeSampler(Random sampler) {
        this.sampler = sampler;
    }

    /**
     * Whether this evaluation should be offered to the writer.
     *
     * <p>A fired evaluation always is. A non-fired one is offered whenever this
     * mode's kept negatives are behind its kept positives -- the deficit is the
     * whole rule, with no rate in front of it.
     *
     * <p>No sampling rate decides it. <b>A fixed rate cannot hold a ratio</b>:
     * the population ratio varies by mode, board size and turn. A probability in
     * front of the deficit is a <b>lagging</b> controller: it catches only
     * some of the negatives it is owed, and one game gives a mode too few
     * firings for the carried deficit ever to be repaid. Offering on the
     * deficit alone has no lag.
     *
     * <p>Bunching behind the firing is not a loss worth a controller. The
     * negatives that reach the shard are spread by
     * {@link RecordDedup#allowDistinct} instead: two consecutive non-fired
     * evaluations on an unmoved board render the same record and the second is
     * refused, so this walks forward until it finds one that differs. The
     * negative it lands on is then the one closest in time to the positive it
     * balances, which is the more useful contrast rather than the less.
     *
     * <p>What the deterministic rule needs in exchange is a bound on wasted
     * work, because a refusal downstream leaves the deficit standing and the
     * next evaluation is offered again. A mode whose negatives are all being
     * refused -- every acting line at its per-line cap, say -- would otherwise
     * build a snapshot per evaluation of the busiest hook in the collector.
     * {@link ModeBalance#offersSinceWrite} counts offers since that mode last
     * wrote anything and decays the rate harmonically past
     * {@link #NEGATIVE_OFFERS_AT_FULL_RATE}, so an unfillable deficit costs a
     * few offers rather than all of them.
     */
    boolean offer(String mode, boolean fired) {
        if (fired) {
            return true;
        }
        ModeBalance balance = triggerBalance.computeIfAbsent(mode, m -> new ModeBalance());
        if (balance.fired * TARGET_NEGATIVE_RATIO - balance.notFired <= 0.0) {
            return false;
        }
        balance.offersSinceWrite++;
        if (balance.offersSinceWrite <= NEGATIVE_OFFERS_AT_FULL_RATE) {
            return true;
        }
        double rate = Math.max(
                MIN_NEGATIVE_RATE,
                (double) NEGATIVE_OFFERS_AT_FULL_RATE / balance.offersSinceWrite);
        return sampler.nextDouble() < rate;
    }

    /**
     * Count a written evaluation against its mode's balance.
     *
     * <p>Called only for a record that reached the shard, which is what makes
     * the ratio hold over the corpus rather than over the offers: a negative
     * refused as a duplicate has not been drawn, and the deficit it leaves
     * standing is repaid by the next evaluation of its mode.
     */
    void keep(String mode, boolean fired) {
        ModeBalance balance = triggerBalance.computeIfAbsent(mode, m -> new ModeBalance());
        if (fired) {
            balance.fired++;
        } else {
            balance.notFired++;
        }
        // A written record of either verdict opens a fresh window: a positive
        // has just created a debt and a negative has just paid one, and either
        // way the offers spent finding this record are spent.
        balance.offersSinceWrite = 0;
    }

    /**
     * A flood cap per acting line, beside the duplicate check.
     *
     * <p>The duplicate check collapses records that say the same thing; this
     * bounds the ones that differ only in a board that moved a little. One
     * line's condition is worth a couple of dozen looks in a game, not the
     * hundreds an aura on a creature that keeps being pumped would write.
     */
    boolean allowLine(String abilityJson, boolean fired) {
        LineVerdict line = new LineVerdict(abilityJson, fired);
        int written = triggerPerLine.getOrDefault(line, 0);
        if (written >= TRIGGER_RECORDS_PER_LINE) {
            return false;
        }
        triggerPerLine.put(line, written + 1);
        return true;
    }

    /** One trigger mode's written evaluations, and the negatives offered since. */
    private static final class ModeBalance {
        private int fired;
        private int notFired;
        /** Negatives offered since this mode last wrote a record of either verdict. */
        private int offersSinceWrite;
    }

    /** One acting line's verdict, the key the flood cap counts by. */
    private record LineVerdict(String abilityJson, boolean fired) {
    }
}
