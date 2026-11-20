package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
import com.pricepredictor.connector.effects.directed.operations.AddPermanentOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;

import java.util.Map;
import java.util.Optional;

/**
 * 111.10p, a Virtuous Role token gives the creature it enchants +1/+1 for
 * each enchantment its controller controls, on scenario {@code 111.10p-1}:
 * A casts Ellivere of the Wild Court, whose enters trigger creates a Virtuous
 * Role attached to A's Grizzly Bears.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, and it holds a creature card it has the lands for whose enters
 * trigger creates a Virtuous Role attached to another creature it controls
 * (Ellivere of the Wild Court), and it controls a creature that trigger can
 * target. It binds the active player and the card.
 *
 * <p>Variant {@code two_enchantments} ({@code add_permanent}): the player
 * also controls Seal of Fire, so the Role counts two enchantments. The fork
 * forces the card's cast (FR-010b): its trigger's Role and the Role's static
 * are the rule's {@code fires} side, which the counts lack (45 records); the
 * AI casts the card on its own only now and then.
 */
public final class VirtuousRole {

    public static final String RULE = "111.10p";
    static final String SCENARIO = "111.10p-1";
    static final String TWO_ENCHANTMENTS = "two_enchantments";
    static final String SEAL_OF_FIRE = "Seal of Fire";
    /** The token script of a Virtuous Role, as Forge names it. */
    private static final String VIRTUOUS_ROLE = "role_virtuous";

    private static final Role<Player> CASTER = Role.player("caster");
    private static final Role<Card> CARD = Role.card("card");

    private VirtuousRole() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, VirtuousRole::bind, new DirectedRule.Variant(TWO_ENCHANTMENTS,
                new Application(ForkOperation.ADD_PERMANENT,
                        fork -> AddPermanentOperation.of(SEAL_OF_FIRE, fork.player(CASTER), false))
                        .forcing(ForcedPlay.cast(CASTER, CARD))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        if (active.getCreaturesInPlay().isEmpty()) {
            return Optional.empty();
        }
        return Scan.handCard(active, card -> card.isCreature() && makesAVirtuousRoleAsItEnters(card))
                .map(card -> Binding.empty().with(CASTER, active).with(CARD, card));
    }

    /**
     * Whether the card's enters trigger creates a Virtuous Role token
     * attached to a creature it targets: a {@code ChangesZone} trigger to the
     * battlefield of the card itself executing a {@code Token} of
     * {@code TokenScript$ role_virtuous} with {@code AttachedTo$ Targeted}.
     */
    static boolean makesAVirtuousRoleAsItEnters(Card card) {
        return Scan.unconditionedEntersTrigger(card, ApiType.Token.name()).stream()
                .anyMatch(VirtuousRole::createsAnAttachedVirtuousRole);
    }

    private static boolean createsAnAttachedVirtuousRole(Map<String, String> script) {
        return VIRTUOUS_ROLE.equals(script.get("TokenScript"))
                && "Targeted".equals(script.get("AttachedTo"));
    }
}
