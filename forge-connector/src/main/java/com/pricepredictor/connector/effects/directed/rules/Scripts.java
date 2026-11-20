package com.pricepredictor.connector.effects.directed.rules;

import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.trigger.Trigger;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * What a card's ability scripts say, read as the parameter maps Forge parses
 * them to: the script a SVar holds, the one a trigger executes, the API it
 * names, a number or a comma-separated list in it. Read only.
 */
final class Scripts {

    private Scripts() {
    }

    /** Whether a comma-separated script value lists {@code item}. */
    static boolean lists(String value, String item) {
        return value != null && List.of(value.split(",")).contains(item);
    }

    /** The parameters of the ability script {@code host} holds as {@code svar}, empty when it holds none. */
    static Map<String, String> script(Card host, String svar) {
        if (svar == null || !host.hasSVar(svar)) {
            return Map.of();
        }
        return AbilityFactory.getMapParams(host.getSVar(svar));
    }

    /** The parameters of the ability {@code trigger} executes, empty when it names none. */
    static Map<String, String> executed(Trigger trigger) {
        return trigger.hasParam("Execute") ? script(trigger.getHostCard(), trigger.getParam("Execute")) : Map.of();
    }

    /** The API a script's parameters name, whether it is a spell, an ability or a sub-ability. */
    static String api(Map<String, String> script) {
        for (String kind : List.of("SP", "AB", "DB")) {
            if (script.containsKey(kind)) {
                return script.get(kind);
            }
        }
        return "";
    }

    /** {@code value} as a number, or empty when it is an X, a count or absent. */
    static Optional<Integer> number(String value) {
        if (value == null) {
            return Optional.empty();
        }
        try {
            return Optional.of(Integer.parseInt(value.trim()));
        } catch (NumberFormatException e) {
            return Optional.empty();
        }
    }
}
