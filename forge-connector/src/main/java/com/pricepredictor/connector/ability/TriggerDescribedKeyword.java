package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.KeywordRendering;
import forge.game.keyword.KeywordInterface;
import forge.game.trigger.Trigger;

/**
 * A keyword whose text is that of the triggers Forge builds for it: a
 * dungeon's rooms ({@code K:Dungeon}) and an Attraction's Prize
 * ({@code K:Prize}). The keyword's own script text names SVars and states
 * nothing; each trigger's {@code TriggerDescription$} is the card's line.
 */
public final class TriggerDescribedKeyword {

    private TriggerDescribedKeyword() {}

    /**
     * A dungeon's room lines: one per {@code RoomEntered} trigger, in the
     * keyword's room order, each the room's description ({@code Room —
     * effect}, the "Leads to" note going as reminder text) and printed by
     * its room's trigger, as a saga's chapter lines are by theirs.
     */
    public static KeywordRendering rooms(KeywordInterface ki) {
        KeywordRendering.Builder rooms = new KeywordRendering.Builder();
        for (Trigger room : ki.getTriggers()) {
            Ability line = lineOf(room);
            if (line == null) continue;
            rooms.add(line);
            rooms.printed(room, line);
        }
        return rooms.build();
    }

    /** Prize's one line: its {@code ClaimPrize} trigger's description ({@code Prize — effect}). */
    public static KeywordRendering prize(KeywordInterface ki) {
        Trigger claim = ki.getTriggers().stream().findFirst().orElse(null);
        return KeywordRendering.ofTrait(claim == null ? null : lineOf(claim), claim);
    }

    /** The triggered line {@code trigger}'s description prints, or null for none. */
    private static Ability lineOf(Trigger trigger) {
        String text = AbilityDescription.normalize(trigger.getParam("TriggerDescription"));
        return text == null ? null : new TextAbility(AbilityType.TRIGGERED, text);
    }
}
