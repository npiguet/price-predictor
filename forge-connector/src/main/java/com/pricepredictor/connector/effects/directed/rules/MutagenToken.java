package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 111.10v, a Mutagen token is a colourless Mutagen artifact with "{1}, {T},
 * sacrifice this token: put a +1/+1 counter on target creature; activate only
 * as a sorcery", on scenario {@code 111.10v-1}: A pays {1} with its third
 * Island and puts the counter on Crustacean Commando.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it controls an untapped Mutagen token, a creature is on the
 * battlefield for the counter, and the active player controls exactly one
 * untapped permanent that makes mana, the last mana it has for the {1}. It
 * binds the token and that mana source.
 *
 * <p>Variant {@code no_mana} ({@code remove_permanent}): the mana source is not
 * on the battlefield, as A has two Islands, both spent, instead of three, so
 * the token's ability cannot be activated.
 */
public final class MutagenToken {

    public static final String RULE = "111.10v";
    static final String SCENARIO = "111.10v-1";
    static final String NO_MANA = "no_mana";

    private MutagenToken() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, MutagenToken::bind, NO_MANA,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("mana source"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live) || live.getCardsIn(ZoneType.Battlefield).stream().noneMatch(Card::isCreature)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        Optional<Card> token = active.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> !card.isTapped() && isMutagen(card))
                .findFirst();
        Optional<Card> source = Boards.soleManaSource(active);
        if (token.isEmpty() || source.isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(Binding.empty().with("token", token.get()).with("mana source", source.get()));
    }

    /** Whether the card is a Mutagen token with its counter ability. */
    static boolean isMutagen(Card card) {
        return card.isToken() && card.getType().hasSubtype("Mutagen")
                && card.getSpellAbilities().stream().anyMatch(ability -> ability.getApi() == ApiType.PutCounter);
    }
}
