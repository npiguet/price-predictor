package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
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
 * (Narci), and a Saga whose next lore counter reaches its final chapter. It
 * binds the Saga.
 *
 * <p>Variant {@code not-final-chapter} ({@code counters}): the Saga has one
 * lore counter fewer, so the chapter that resolves is not the final one and
 * the trigger does not fire.
 */
public final class FinalChapterResolves {

    public static final String RULE = "608.2p";
    static final String SCENARIO = "608.2p-1";
    static final String NOT_FINAL_CHAPTER = "not-final-chapter";
    static final CounterEnumType LORE = CounterEnumType.LORE;

    private FinalChapterResolves() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, FinalChapterResolves::bind, NOT_FINAL_CHAPTER,
                new Application(ForkOperation.COUNTERS, fork -> {
                    Card saga = fork.card("saga");
                    return new CountersOperation(saga, LORE, saga.getCounters(LORE) - 1);
                }));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.MAIN1)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        boolean watched = active.getCardsIn(ZoneType.Battlefield).stream()
                .flatMap(card -> card.getTriggers().stream())
                .anyMatch(FinalChapterResolves::watchesFinalChapter);
        if (!watched) {
            return Optional.empty();
        }
        return active.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.isSaga() && card.hasChapter())
                .filter(card -> card.getCounters(LORE) >= 1
                        && card.getCounters(LORE) == card.getFinalChapterNr() - 1)
                .findFirst()
                .map(saga -> Binding.empty().with("saga", saga));
    }

    /** Whether the trigger fires on a Saga's final chapter ability resolving. */
    static boolean watchesFinalChapter(Trigger trigger) {
        String valid = trigger.getParam("ValidSpellAbility");
        return trigger.getMode() == TriggerType.AbilityResolves && valid != null && valid.contains("LastChapter");
    }
}
