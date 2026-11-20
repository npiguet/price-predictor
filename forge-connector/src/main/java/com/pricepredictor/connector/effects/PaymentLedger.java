package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.spellability.SpellAbility;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * The counter and life events a cost payment publishes, kept per paid ability
 * until its cast event reads them into its cost half ({@link PaidCosts}).
 *
 * <p>The payment is told from the engine's own stack of payments
 * ({@code game.costPaymentStack}), so a mana ability paid inside a spell's
 * payment keeps its own (those events are the mana record's, taken before they
 * get here).
 *
 * <p>Kept by the id of the ability that paid: an activated ability on the
 * stack is a fresh copy of the one that paid ({@code MagicStack.add}), which
 * names it as its original; a spell is its own.
 */
final class PaymentLedger {

    /** Entries kept before the unread ones of cancelled payments are let go. */
    private static final int CAP = 64;

    /** What one payment has moved so far. */
    private static final class Moved {
        final Map<String, Integer> counters = new TreeMap<>();
        int life;
    }

    private final Game game;
    private final Map<Integer, Moved> payments = new HashMap<>();

    PaymentLedger(Game game) {
        this.game = game;
    }

    /** Keep {@code event} if it is a counter or a life change made while paying. */
    void note(EffectEvent event) {
        SpellAbility paid = payingNow();
        if (paid == null || !(event.param(EventParam.DELTA) instanceof Integer delta)) {
            return;
        }
        if (payments.size() > CAP) {
            payments.clear();
        }
        if (event.type() == EventType.COUNTER_CHANGE) {
            movedBy(paid).counters
                    .merge(String.valueOf(event.param(EventParam.COUNTER_TYPE)), delta, Integer::sum);
        } else if (event.type() == EventType.LIFE_CHANGE && delta < 0) {
            movedBy(paid).life -= delta;
        }
    }

    /** The ability whose costs are being paid now, from the top of the payment stack. */
    private SpellAbility payingNow() {
        var top = game.costPaymentStack.peek();
        return top == null || top.payment() == null ? null : top.payment().getAbility();
    }

    private Moved movedBy(SpellAbility paid) {
        return payments.computeIfAbsent(paid.getId(), key -> new Moved());
    }

    /** The ability whose payment {@code onStack} is the result of. */
    private static SpellAbility paidBy(SpellAbility onStack) {
        SpellAbility original = onStack.getOriginalAbility();
        return original == null ? onStack : original;
    }

    /** What {@code ability}'s payment did, read once. */
    PaymentTally take(SpellAbility ability) {
        Moved moved = payments.remove(paidBy(ability).getId());
        if (moved == null) {
            return PaymentTally.NONE;
        }
        Map<String, Integer> net = new TreeMap<>(moved.counters);
        net.values().removeIf(count -> count == 0);
        return new PaymentTally(net, moved.life);
    }
}
