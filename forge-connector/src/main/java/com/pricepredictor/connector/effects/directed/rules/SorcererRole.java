package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.Map;
import java.util.Optional;

/**
 * 111.10n, a Sorcerer Role token gives the creature it enchants +1/+1 and a
 * scry-on-attack ability, on scenario {@code 111.10n-1}: at the beginning of
 * combat A pays {1} for Spellbook Vendor's trigger and attaches a Sorcerer
 * Role to the Vendor.
 *
 * <p>The base board, read at a boundary: the active player's precombat main
 * phase begins, it controls a permanent whose beginning-of-combat trigger
 * lets it pay mana to create a Sorcerer Role (Spellbook Vendor), and it
 * controls exactly one untapped permanent that makes mana, the last mana it
 * has for that payment. It binds that mana source.
 *
 * <p>Variant {@code cannot_pay} ({@code remove_permanent}): the mana source is
 * not on the battlefield, as A has two Plains instead of three, so no mana
 * remains at the beginning of combat and no Role is created.
 */
public final class SorcererRole {

    public static final String RULE = "111.10n";
    static final String SCENARIO = "111.10n-1";
    static final String CANNOT_PAY = "cannot_pay";
    static final String ROLE = "role_sorcerer";

    private SorcererRole() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, SorcererRole::bind, CANNOT_PAY,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("mana source"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.MAIN1)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        boolean paysForRole = active.getCardsIn(ZoneType.Battlefield).stream()
                .anyMatch(SorcererRole::paysForRoleAtCombat);
        if (!paysForRole) {
            return Optional.empty();
        }
        return Boards.soleManaSource(active).map(source -> Binding.empty().with("mana source", source));
    }

    /** Whether a beginning-of-combat trigger of the card lets its controller pay to create a Sorcerer Role. */
    static boolean paysForRoleAtCombat(Card card) {
        for (Trigger trigger : card.getTriggers()) {
            if (trigger.getMode() != TriggerType.Phase || !"BeginCombat".equals(trigger.getParam("Phase"))
                    || !"You".equals(trigger.getParam("ValidPlayer"))) {
                continue;
            }
            Map<String, String> paid = Boards.executed(trigger);
            if (ApiType.ImmediateTrigger.name().equals(Boards.api(paid)) && paid.containsKey("Cost")
                    && ROLE.equals(Boards.script(card, paid.get("Execute")).get("TokenScript"))) {
                return true;
            }
        }
        return false;
    }
}
