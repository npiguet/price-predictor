package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * 111.10p, a Virtuous Role token gives the creature it enchants +1/+1 for
 * each enchantment its controller controls, on scenario {@code 111.10p-1}: A
 * casts Ellivere of the Wild Court, whose enters trigger attaches a Virtuous
 * Role to A's Grizzly Bears.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card it has the lands to cast whose enters trigger
 * creates a Virtuous Role attached to another target creature it controls
 * (Ellivere), and it controls exactly one creature, the trigger's only
 * target. It binds that creature and the card.
 *
 * <p>Variant {@code no_other_creature} ({@code remove_permanent}): the
 * creature is not on the battlefield, so the trigger has no legal target and
 * no Role is created.
 */
public final class VirtuousRole {

    public static final String RULE = "111.10p";
    static final String SCENARIO = "111.10p-1";
    static final String NO_OTHER_CREATURE = "no_other_creature";
    static final String ROLE = "role_virtuous";

    private VirtuousRole() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, VirtuousRole::bind, NO_OTHER_CREATURE,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("creature"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> creatures = active.getCreaturesInPlay();
        if (creatures.size() != 1) {
            return Optional.empty();
        }
        return active.getCardsIn(ZoneType.Hand).stream()
                .filter(card -> Boards.affordable(active, card) && attachesRoleOnEntering(card))
                .findFirst()
                .map(card -> Binding.empty().with("creature", creatures.get(0)).with("card", card));
    }

    /** Whether the card's enters trigger creates a Virtuous Role attached to another creature its controller controls. */
    static boolean attachesRoleOnEntering(Card card) {
        for (Trigger trigger : card.getTriggers()) {
            if (trigger.getMode() != TriggerType.ChangesZone
                    || !"Battlefield".equals(trigger.getParam("Destination"))
                    || !"Card.Self".equals(trigger.getParam("ValidCard"))) {
                continue;
            }
            Map<String, String> token = Boards.executed(trigger);
            if (ROLE.equals(token.get("TokenScript")) && "Targeted".equals(token.get("AttachedTo"))
                    && token.getOrDefault("ValidTgts", "").contains("Other")) {
                return true;
            }
        }
        return false;
    }
}
