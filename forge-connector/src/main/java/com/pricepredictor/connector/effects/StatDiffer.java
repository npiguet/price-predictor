package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Changes to a card's computed characteristics, as events.
 *
 * <p>Forge's bus publishes what an effect <i>did</i> only for the outcomes that
 * have their own event: damage, counters, zone changes, taps. A pump, an anthem
 * recompute, an animation and a colour change all publish the same thing — the
 * card's stats changed — and say nothing about what changed. So the whole
 * "gets +2/+2 and gains flying" family, which is most of what a limited deck
 * does, produced records whose event list was empty or nearly so.
 *
 * <p>This closes that by remembering each card's computed characteristics and
 * describing the difference when the engine says they moved. Computed rather
 * than printed, for the reason every other characteristic in a snapshot is: the
 * model has to predict what happens to the 5/5 an anthem made, not to the 4/4
 * on the card.
 *
 * <p>The remembered value updates on every diff, so two effects in one bracket
 * each report their own share rather than the second reporting both.
 */
final class StatDiffer {

    /** One card's computed characteristics, as of the last time it was read. */
    private record Stats(
            int power, int toughness, Set<String> keywords,
            Set<String> types, Set<String> colors, String controller) {
    }

    private final Map<Integer, Stats> last = new HashMap<>();

    /**
     * What changed on this card since it was last read, as events.
     *
     * <p>A card seen for the first time reports nothing: there is no earlier
     * value to have moved from, and reporting its whole characteristic set as a
     * change would make every card that enters look like it was animated.
     */
    List<EffectEvent> diff(Card card) {
        if (card == null) {
            return List.of();
        }
        Stats now = read(card);
        Stats before = last.put(card.getId(), now);
        if (before == null || before.equals(now)) {
            return List.of();
        }
        String subject = SnapshotBuilder.entityId(card);
        List<EffectEvent> events = new ArrayList<>();

        if (before.power() != now.power() || before.toughness() != now.toughness()) {
            events.add(new EffectEvent(EffectEvent.PT_CHANGE)
                    .subject(subject)
                    .param("power_delta", now.power() - before.power())
                    .param("toughness_delta", now.toughness() - before.toughness()));
        }
        addKeywordEvents(events, subject, before.keywords(), now.keywords());
        addTypeEvent(events, subject, before.types(), now.types());
        addColorEvent(events, subject, before.colors(), now.colors());
        if (!java.util.Objects.equals(before.controller(), now.controller())) {
            events.add(new EffectEvent(EffectEvent.CONTROL_CHANGE)
                    .subject(subject)
                    .param("controller", now.controller()));
        }
        return events;
    }

    /**
     * Remember a card's characteristics without reporting a change.
     *
     * <p>Used when a card arrives on the battlefield, so that its first real
     * change is a change rather than its whole self.
     */
    void prime(Card card) {
        if (card != null) {
            last.put(card.getId(), read(card));
        }
    }

    /** Forget a card, so a re-entering permanent is a new object. */
    void forget(int cardId) {
        last.remove(cardId);
    }

    private static void addKeywordEvents(
            List<EffectEvent> events, String subject,
            Set<String> before, Set<String> now) {
        Set<String> gained = difference(now, before);
        Set<String> lost = difference(before, now);
        if (!gained.isEmpty()) {
            events.add(new EffectEvent(EffectEvent.KEYWORD_CHANGE)
                    .subject(subject)
                    .param("keywords", List.copyOf(gained))
                    .param("removed", false));
        }
        if (!lost.isEmpty()) {
            events.add(new EffectEvent(EffectEvent.KEYWORD_CHANGE)
                    .subject(subject)
                    .param("keywords", List.copyOf(lost))
                    .param("removed", true));
        }
    }

    private static void addTypeEvent(
            List<EffectEvent> events, String subject,
            Set<String> before, Set<String> now) {
        Set<String> added = difference(now, before);
        Set<String> removed = difference(before, now);
        if (added.isEmpty() && removed.isEmpty()) {
            return;
        }
        events.add(new EffectEvent(EffectEvent.TYPE_CHANGE)
                .subject(subject)
                .param("types_added", List.copyOf(added))
                .param("types_removed", List.copyOf(removed)));
    }

    private static void addColorEvent(
            List<EffectEvent> events, String subject,
            Set<String> before, Set<String> now) {
        Set<String> added = difference(now, before);
        Set<String> removed = difference(before, now);
        if (added.isEmpty() && removed.isEmpty()) {
            return;
        }
        // colors_removed is this side's addition to the type's normalization:
        // the head has a colors_lost field and the event carried no way to say
        // a colour went away.
        events.add(new EffectEvent(EffectEvent.COLOR_CHANGE)
                .subject(subject)
                .param("colors", List.copyOf(added))
                .param("colors_removed", List.copyOf(removed)));
    }

    private static Set<String> difference(Set<String> from, Set<String> without) {
        Set<String> result = new LinkedHashSet<>(from);
        result.removeAll(without);
        return result;
    }

    private static Stats read(Card card) {
        Set<String> keywords = new LinkedHashSet<>();
        for (KeywordInterface keyword : card.getKeywords()) {
            String original = keyword.getOriginal();
            if (original != null && !original.isEmpty()) {
                keywords.add(original.toLowerCase(Locale.ROOT));
            }
        }
        Set<String> types = new LinkedHashSet<>();
        for (var type : card.getType().getCoreTypes()) {
            types.add(SnapshotBuilder.typeName(type));
        }
        for (var supertype : card.getType().getSupertypes()) {
            types.add(SnapshotBuilder.typeName(supertype));
        }
        for (String subtype : card.getType().getSubtypes()) {
            types.add(SnapshotBuilder.typeName(subtype));
        }
        return new Stats(
                card.getNetPower(), card.getNetToughness(), keywords, types,
                new LinkedHashSet<>(SnapshotBuilder.colorLetters(card.getColor())),
                card.getController() == null
                        ? null : SnapshotBuilder.playerId(card.getController()));
    }
}
