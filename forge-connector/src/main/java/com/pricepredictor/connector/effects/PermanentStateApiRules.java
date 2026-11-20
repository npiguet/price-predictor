package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.ApiEvents.Rule;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;

import static com.pricepredictor.connector.effects.ApiEvents.put;

/**
 * The {@link ApiEvents} rules for a permanent's own state: Class levels,
 * doors, pairs, merges.
 */
final class PermanentStateApiRules {

    private PermanentStateApiRules() {
    }

    /**
     * The states of a permanent no bus event announces and no stats change
     * shows, each a {@code state_flag_change} naming it: a Class gaining a
     * level ({@code class_level}, {@code value} the level reached), a Room's
     * door unlocking ({@code door_unlocked}), soulbond pairing two creatures
     * ({@code paired}, on both), and a mutating spell merging into a
     * creature ({@code merged}, on the merged permanent). Each read as the
     * state after the clause against the state before it, so a clause that
     * changed nothing reports nothing.
     */
    static void permanentStateRules(Map<String, Rule<?>> rules) {
        put(rules, Rule.one(EventType.STATE_FLAG_CHANGE,
                        (sa, host) -> host.getClassLevel(),
                        (sa, host, before) -> before == null || host.getClassLevel() <= before
                                ? Optional.empty()
                                : Optional.of(new EffectEvent(EventType.STATE_FLAG_CHANGE)
                                        .subject(Refs.of(host))
                                        .flag(StateFlag.CLASS_LEVEL, host.getClassLevel()))),
                "ClassLevelUp");
        put(rules, Rule.one(EventType.STATE_FLAG_CHANGE, PermanentStateApiRules::unlockedDoors,
                        (sa, host, before) -> doorsUnlocked(host, before)),
                "UnlockDoor");
        put(rules, Rule.one(EventType.STATE_FLAG_CHANGE, PermanentStateApiRules::pairedCreatures,
                        (sa, host, before) -> newlyPaired(host, before)),
                "Bond");
        put(rules, Rule.one(EventType.STATE_FLAG_CHANGE, null,
                        (sa, host, memo) -> merged(host)),
                "Mutate");
    }

    /** How many doors each permanent on the battlefield has unlocked, by id. */
    private static Map<Integer, Integer> unlockedDoors(SpellAbility sa, Card host) {
        Map<Integer, Integer> unlocked = new HashMap<>();
        for (Card card : host.getGame().getCardsIn(ZoneType.Battlefield)) {
            unlocked.put(card.getId(), doors(card));
        }
        return unlocked;
    }

    private static Optional<EffectEvent> doorsUnlocked(Card host, Map<Integer, Integer> before) {
        if (before == null) {
            return Optional.empty();
        }
        return flagged(host, StateFlag.DOOR_UNLOCKED,
                card -> doors(card) > before.getOrDefault(card.getId(), 0));
    }

    /**
     * {@code state_flag_change} {@code flag} on every permanent on the
     * battlefield that {@code changed} holds for; empty when it holds for none.
     */
    private static Optional<EffectEvent> flagged(Card host, StateFlag flag, Predicate<Card> changed) {
        EffectEvent event = new EffectEvent(EventType.STATE_FLAG_CHANGE)
                .flag(flag, true);
        for (Card card : host.getGame().getCardsIn(ZoneType.Battlefield)) {
            if (changed.test(card)) {
                event.subject(Refs.of(card));
            }
        }
        return event.subjects().isEmpty() ? Optional.empty() : Optional.of(event);
    }

    private static int doors(Card card) {
        return card.getUnlockedRooms() == null ? 0 : card.getUnlockedRooms().size();
    }

    /** The ids of every paired permanent on the battlefield. */
    private static Set<Integer> pairedCreatures(SpellAbility sa, Card host) {
        Set<Integer> paired = new HashSet<>();
        for (Card card : host.getGame().getCardsIn(ZoneType.Battlefield)) {
            if (card.isPaired()) {
                paired.add(card.getId());
            }
        }
        return paired;
    }

    private static Optional<EffectEvent> newlyPaired(Card host, Set<Integer> before) {
        if (before == null) {
            return Optional.empty();
        }
        return flagged(host, StateFlag.PAIRED, card -> card.isPaired() && !before.contains(card.getId()));
    }

    /**
     * {@code merged} on the permanent a mutating spell merged into: {@code
     * MutateEffect} sets the spell's {@code getMergedToCard} to it, the one
     * permanent the merge leaves, whichever card went on top.
     */
    private static Optional<EffectEvent> merged(Card host) {
        Card into = host == null ? null : host.getMergedToCard();
        if (into == null) {
            return Optional.empty();
        }
        return Optional.of(new EffectEvent(EventType.STATE_FLAG_CHANGE)
                .subject(Refs.of(into))
                .flag(StateFlag.MERGED, true));
    }
}
