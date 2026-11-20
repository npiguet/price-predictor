package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.PlayabilityVerdict;
import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.AddPermanentOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.Optional;
import java.util.stream.Stream;

/**
 * 111.10v, a Mutagen token is a colourless Mutagen artifact with "{1}, {T},
 * sacrifice this token: put a +1/+1 counter on target creature; activate only
 * as a sorcery", on scenario {@code 111.10v-1}: A pays {1} with its third
 * Island and puts the counter on Crustacean Commando.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it controls an untapped Mutagen token whose counter ability it
 * cannot pay for ({@link PlayabilityVerdict#affordable}), as A with two
 * Islands spent, and a creature is on the battlefield for the counter. It
 * binds the active player and the token.
 *
 * <p>Variant {@code mana_available} ({@code add_permanent}): the active
 * player also controls an untapped Island, the mana the {1} needs. The fork
 * forces the token's activation (FR-010b), its target drawn among the
 * creatures, so its resolution puts the counter.
 */
public final class MutagenToken {

    public static final String RULE = "111.10v";
    static final String SCENARIO = "111.10v-1";
    static final String MANA_AVAILABLE = "mana_available";
    static final String ISLAND = Cards.ISLAND;

    private MutagenToken() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, MutagenToken::bind, MANA_AVAILABLE,
                new Application(ForkOperation.ADD_PERMANENT,
                        fork -> AddPermanentOperation.of(ISLAND, fork.player("player"), false))
                        .forcing(ForcedPlay.activate("player", "token", ForcedPlay.Line.api("PutCounter"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live) || live.getCardsIn(ZoneType.Battlefield).stream().noneMatch(Card::isCreature)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        return active.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> !card.isTapped() && isMutagen(card))
                .filter(card -> counterAbility(card).noneMatch(ability ->
                        PlayabilityVerdict.of(ability, active).affordable()))
                .findFirst()
                .map(token -> Binding.empty().with("player", active).with("token", token));
    }

    /** Whether the card is a Mutagen token with its counter ability. */
    static boolean isMutagen(Card card) {
        return card.isToken() && card.getType().hasSubtype("Mutagen") && counterAbility(card).findAny().isPresent();
    }

    private static Stream<SpellAbility> counterAbility(Card card) {
        return card.getSpellAbilities().stream().filter(ability -> ability.getApi() == ApiType.PutCounter);
    }
}
