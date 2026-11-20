package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
import com.pricepredictor.connector.effects.directed.operations.CountersOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 608.2p, an ability that triggers when another ability resolves triggers
 * after that resolution, on scenario {@code 608.2p-1}: History of Benalia's
 * final chapter resolves and Narci, Fable Singer's trigger on a final chapter
 * resolving fires.
 *
 * <p>The base board, read at a boundary: the precombat main phase begins
 * (before its lore counters are added), the active player controls a
 * permanent whose trigger watches the final chapter ability of a Saga resolve
 * (Narci), and a Saga with at least one lore counter whose next lore counter
 * reaches the chapter before its final one, so the chapter that resolves is
 * not the final one and the trigger does not fire. It binds the Saga and the
 * watching permanent.
 *
 * <p>Variant {@code final-chapter} ({@code counters}): the Saga has one lore
 * counter more, so the main phase's lore counter reaches its final chapter,
 * which resolves, and the trigger fires: the side of the rule the counts lack.
 *
 * <p>The fork also keeps the not-fired evaluations of the triggers the
 * watching permanent hosts whatever the trigger-negative sampler's balance
 * says (FR-010c): a fork's recording is fresh, so the sampler holds no
 * earlier firing of the mode for a negative to repay, and the watcher's
 * evaluations against the fork's other resolutions are the near misses.
 */
public final class FinalChapterResolves {

    public static final String RULE = "608.2p";
    static final String SCENARIO = "608.2p-1";
    static final String FINAL_CHAPTER = "final-chapter";
    static final CounterEnumType LORE = CounterEnumType.LORE;

    private static final Role<Card> SAGA = Role.card("saga");
    private static final Role<Card> WATCHER = Role.card("watcher");

    private FinalChapterResolves() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, FinalChapterResolves::bind, new DirectedRule.Variant(FINAL_CHAPTER,
                new Application(ForkOperation.COUNTERS, fork -> {
                    Card saga = fork.card(SAGA);
                    return new CountersOperation(saga, LORE, saga.getCounters(LORE) + 1);
                }).keepingNegativesOf(WATCHER)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.MAIN1)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        Optional<Card> watcher = active.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getTriggers().stream().anyMatch(FinalChapterResolves::watchesFinalChapter))
                .findFirst();
        if (watcher.isEmpty()) {
            return Optional.empty();
        }
        return active.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.isSaga() && card.hasChapter())
                .filter(card -> card.getCounters(LORE) >= 1
                        && card.getCounters(LORE) == card.getFinalChapterNr() - 2)
                .findFirst()
                .map(saga -> Binding.empty().with(SAGA, saga).with(WATCHER, watcher.get()));
    }

    /** Whether the trigger fires on a Saga's final chapter ability resolving. */
    static boolean watchesFinalChapter(Trigger trigger) {
        String valid = trigger.getParam("ValidSpellAbility");
        return trigger.getMode() == TriggerType.AbilityResolves && valid != null && valid.contains("LastChapter");
    }
}
