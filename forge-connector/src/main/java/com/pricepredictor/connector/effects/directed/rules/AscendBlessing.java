package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
import com.pricepredictor.connector.effects.directed.operations.AddPermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.Map;
import java.util.Optional;

/**
 * 702.131a, a resolving instant or sorcery with ascend gives its controller
 * the city's blessing when it controls ten permanents, on scenario
 * {@code 702.131a-1}: A controls nine Islands and casts Secrets of the Golden
 * City, which resolves without the blessing; at ten permanents A would get
 * it.
 *
 * <p>The base board, read at a boundary: a main phase of the active
 * player's turn begins with the stack empty; the active player controls
 * exactly nine permanents, phased-out ones included as Forge counts its
 * battlefield zone for ascend, has no city's blessing, and holds an instant
 * or sorcery with ascend it has the untapped lands for. It binds the player
 * and the spell. The live game, at nine, is
 * the common side; the fork adds the rare fact.
 *
 * <p>Variant {@code ten_permanents} ({@code add_permanent}): the player also
 * controls a tapped Island, its tenth permanent, which makes no mana this
 * turn and so changes nothing else the turn can play; the spell resolving
 * gives the blessing, the state flag change 702.131a-1's declaration reads.
 * The fork forces the spell's cast (FR-010b): Forge's AI holds an instant
 * for the opponent's turn, after the fork has stopped.
 */
public final class AscendBlessing {

    public static final String RULE = "702.131a";
    static final String SCENARIO = "702.131a-1";
    static final String TEN_PERMANENTS = "ten_permanents";
    static final String ISLAND = Cards.ISLAND;
    static final int NINE = 9;

    private static final Role<Player> PLAYER = Role.player("player");
    private static final Role<Card> SPELL = Role.card("spell");

    private AscendBlessing() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, AscendBlessing::bind, TEN_PERMANENTS,
                new Application(ForkOperation.ADD_PERMANENT, fork -> new AddPermanentOperation(
                        ISLAND, fork.player(PLAYER), fork.player(PLAYER), true, false, Map.of()))
                        .forcing(ForcedPlay.cast(PLAYER, SPELL)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live) || !live.getStack().isEmpty()) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        // AbilityUtils.resolvePreAbilities counts the battlefield zone,
        // phased-out permanents included.
        if (active.getZone(ZoneType.Battlefield).size() != NINE || active.hasBlessing()) {
            return Optional.empty();
        }
        Optional<Card> spell = Scan.handCard(active,
                card -> (card.isSorcery() || card.isInstant()) && card.hasKeyword(Keyword.ASCEND));
        return spell.map(card -> Binding.empty().with(PLAYER, active).with(SPELL, card));
    }
}
