package com.pricepredictor.connector.effects;

import forge.game.event.GameEventCardAttachment;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventCardCounters;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventCardTapped;
import forge.game.event.GameEventPlayerDamaged;
import forge.game.event.GameEventPlayerLivesChanged;
import forge.game.event.GameEventPlayerPoisoned;
import forge.game.event.GameEventScry;
import forge.game.event.GameEventSurveil;
import forge.game.event.GameEventZone;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;
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

    /**
     * A draw, a discard or a mill, named for what it is.
     *
     * <p>Forge publishes no event for any of the three: each is a card moving
     * between two zones, and which of the three it is depends on where it came
     * from. The head counts all three per player, so the movement has to be
     * classified here or those counters never fire.
     *
     * <p>The subject is the <b>player</b>, not the card, because that is whose
     * count it is. A zone-change event for the card itself is recorded
     * separately by {@link #zone}.
     */
    static EffectEvent libraryMovement(GameEventCardChangeZone event) {
        if (event.card() == null || event.card().getOwner() == null
                || event.from() == null || event.to() == null) {
            return null;
        }
        ZoneType from = event.from().zoneType();
        ZoneType to = event.to().zoneType();
        if (from == null || to == null) {
            return null;
        }
        String type = switch (to) {
            case Hand -> from == ZoneType.Library ? EffectEvent.CARD_DRAWN : null;
            case Graveyard -> switch (from) {
                case Hand -> EffectEvent.CARD_DISCARDED;
                case Library -> EffectEvent.CARD_MILLED;
                default -> null;
            };
            default -> null;
        };
        if (type == null) {
            return null;
        }
        return new EffectEvent(type)
                .subject("P" + event.card().getOwner().getId())
                .param("count", 1);
    }

    /** An aura or equipment moving, or null when it names no card. */
    static EffectEvent attachment(GameEventCardAttachment event) {
        if (event.equipment() == null) {
            return null;
        }
        boolean attaching = event.newTarget() != null;
        return new EffectEvent(
                attaching ? EffectEvent.ATTACHED : EffectEvent.UNATTACHED)
                .subject("E" + event.equipment().getId());
    }

    /**
     * Scry, as the two things it is: cards looked at, and a library reordered.
     *
     * <p>The head's library-event channel is a multi-hot over what happened to
     * a library, and a scry is both.
     */
    static List<EffectEvent> scry(GameEventScry event) {
        int count = event.toTop() + event.toBottom();
        if (event.player() == null || count <= 0) {
            return List.of();
        }
        String player = "P" + event.player().getId();
        return List.of(
                new EffectEvent(EffectEvent.CARD_LOOKED_AT)
                        .subject(player).param("count", count),
                new EffectEvent(EffectEvent.LIBRARY_REORDERED)
                        .subject(player).param("count", count));
    }

    /** Surveil: cards looked at, and however many went to the graveyard. */
    static List<EffectEvent> surveil(GameEventSurveil event) {
        int count = event.toLibrary() + event.toGraveyard();
        if (event.player() == null || count <= 0) {
            return List.of();
        }
        String player = "P" + event.player().getId();
        List<EffectEvent> events = new ArrayList<>();
        events.add(new EffectEvent(EffectEvent.CARD_LOOKED_AT)
                .subject(player).param("count", count));
        events.add(new EffectEvent(EffectEvent.LIBRARY_REORDERED)
                .subject(player).param("count", count));
        if (event.toGraveyard() > 0) {
            events.add(new EffectEvent(EffectEvent.CARD_MILLED)
                    .subject(player).param("count", event.toGraveyard()));
        }
        return events;
    }
}
