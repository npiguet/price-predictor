package com.pricepredictor.connector.effects;

import forge.game.event.GameEventCardCounters;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventCardTapped;
import forge.game.event.GameEventPlayerDamaged;
import forge.game.event.GameEventPlayerLivesChanged;
import forge.game.event.GameEventPlayerPoisoned;
import forge.game.event.GameEventZone;

import java.util.Locale;

/**
 * One Forge bus event as one {@link EffectEvent}.
 *
 * <p>Pure shaping, with no opinion about where the result belongs. Two
 * collectors need it and route the results differently: the bracket collector
 * sorts each event into the resolution or the combat bracket, while the fork
 * collector appends everything one forced resolution produced. Shared because
 * the alternative is two readings of the same bus event that drift — and a
 * drift here is a record that describes the wrong thing while looking fine.
 */
final class BusEvents {

    private BusEvents() {
    }

    static EffectEvent cardDamaged(GameEventCardDamaged event, boolean combat) {
        return new EffectEvent(EffectEvent.DAMAGE_DEALT)
                .subject("E" + event.card().getId())
                .param("amount", event.amount())
                .param("combat", combat);
    }

    static EffectEvent playerDamaged(GameEventPlayerDamaged event) {
        return new EffectEvent(EffectEvent.DAMAGE_DEALT)
                .subject("P" + event.target().getId())
                .param("amount", event.amount())
                .param("combat", event.combat());
    }

    static EffectEvent lifeChanged(GameEventPlayerLivesChanged event) {
        return new EffectEvent(EffectEvent.LIFE_CHANGE)
                .subject("P" + event.player().getId())
                .param("delta", event.newLives() - event.oldLives());
    }

    static EffectEvent poisoned(GameEventPlayerPoisoned event) {
        return new EffectEvent(EffectEvent.POISON_CHANGE)
                .subject("P" + event.receiver().getId())
                .param("delta", event.amount());
    }

    static EffectEvent counters(GameEventCardCounters event) {
        return new EffectEvent(EffectEvent.COUNTER_CHANGE)
                .subject("E" + event.card().getId())
                .param("counter_type", event.type().getName().toUpperCase(Locale.ROOT))
                .param("delta", event.newValue() - event.oldValue());
    }

    static EffectEvent tapped(GameEventCardTapped event) {
        return new EffectEvent(
                event.tapped() ? EffectEvent.TAPPED : EffectEvent.UNTAPPED)
                .subject("E" + event.card().getId());
    }

    /**
     * A zone change, or null when the event names no card.
     *
     * <p>The bus's zone event names the zone rather than the card, so the
     * destination comes from the event and the subject from the card it moved.
     */
    static EffectEvent zone(GameEventZone event) {
        if (event.card() == null) {
            return null;
        }
        return new EffectEvent(EffectEvent.ZONE_CHANGE)
                .subject("E" + event.card().getId())
                .param("to_zone", event.zoneType() == null
                        ? null
                        : event.zoneType().name().toLowerCase(Locale.ROOT));
    }
}
