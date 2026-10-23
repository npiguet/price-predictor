package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.card.CardType;
import forge.game.card.Card;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** What a change to a card's computed characteristics reports. */
@ExtendWith(ForgeExtension.class)
class StatDifferTest {

    @Test
    void aFirstReadReportsNothing() {
        Card bears = TestCards.build("Grizzly Bears");
        assertTrue(new StatDiffer().diff(bears).isEmpty());
    }

    @Test
    void aTypeChangeNamesWhatJoinedAndWhatLeftTheTypeLine() {
        Card bears = TestCards.build("Grizzly Bears");
        StatDiffer differ = new StatDiffer();
        differ.prime(bears);
        bears.addChangedCardTypes(
                new CardType(List.of("Legendary", "Artifact", "Golem"), true),
                new CardType(List.of("Bear"), true),
                false, Set.of(), 1L, 0L, false, false);

        List<EffectEvent> events = differ.diff(bears);

        assertEquals(1, events.size());
        // Core, then super, then sub: the order the type line flattens in.
        EffectEvent event = events.get(0);
        assertEquals(EventType.TYPE_CHANGE, event.type());
        assertEquals(List.of(Refs.of(bears)), event.subjects());
        assertEquals(
                Map.of("types_added", List.of("artifact", "legendary", "golem"),
                        "types_removed", List.of("bear")),
                event.params());
        assertEquals(List.of("types_added", "types_removed"), List.copyOf(event.params().keySet()));
    }

    @Test
    void aColourChangeNamesTheColoursGainedAndLost() {
        Card bears = TestCards.build("Grizzly Bears");
        StatDiffer differ = new StatDiffer();
        differ.prime(bears);
        bears.setColor("blue", "red");

        List<EffectEvent> events = differ.diff(bears);

        assertEquals(1, events.size());
        EffectEvent event = events.get(0);
        assertEquals(EventType.COLOR_CHANGE, event.type());
        assertEquals(List.of(Refs.of(bears)), event.subjects());
        assertEquals(
                Map.of("colors", List.of("U", "R"), "colors_removed", List.of("G")),
                event.params());
        assertEquals(List.of("colors", "colors_removed"), List.copyOf(event.params().keySet()));
    }
}
