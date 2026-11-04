package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.replacement.ReplacementEffect;
import forge.game.replacement.ReplacementType;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * 701.24f, two shuffles of one library at the same time fire shuffle
 * triggers twice, on scenario {@code 701.24f-1}: B's Barter in Blood makes A
 * sacrifice two Darksteel Colossi, each shuffled into A's library by its own
 * replacement, and B's Cosi's Trickster triggers once per shuffle.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a spell it has the mana for that makes players sacrifice
 * two or more creatures at once (Barter in Blood), a player controls a
 * permanent that triggers when an opponent shuffles their library, and an
 * opponent of theirs controls exactly two creatures, each of which is
 * shuffled into its owner's library instead of going to a graveyard. It binds
 * one of those two.
 *
 * <p>Variant {@code one-colossus} ({@code replace_card}): Grizzly Bears is in
 * that creature's place, so the sacrifice shuffles the library once and the
 * trigger fires once.
 */
public final class SimultaneousShuffles {

    public static final String RULE = "701.24f";
    static final String SCENARIO = "701.24f-1";
    static final String ONE_COLOSSUS = "one-colossus";
    static final String BEARS = "Grizzly Bears";

    private SimultaneousShuffles() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, SimultaneousShuffles::bind, ONE_COLOSSUS,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card("creature"), BEARS)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)
                || Boards.castableInHand(Boards.active(live), SimultaneousShuffles::massSacrifice).isEmpty()) {
            return Optional.empty();
        }
        for (Player watcher : live.getPlayers()) {
            if (!SmallLibraryShuffle.watchesOpponentsShuffles(watcher)) {
                continue;
            }
            for (Player opponent : watcher.getOpponents()) {
                List<Card> creatures = List.copyOf(opponent.getCreaturesInPlay());
                if (creatures.size() == 2 && creatures.stream().allMatch(SimultaneousShuffles::shufflesIntoLibrary)) {
                    return Optional.of(Binding.empty().with("creature", creatures.get(0)));
                }
            }
        }
        return Optional.empty();
    }

    /**
     * Whether the spell makes players other than its caster alone sacrifice
     * two or more creatures at once: an edict for two ({@code Defined$
     * Player}) or a sacrifice of all.
     */
    static boolean massSacrifice(SpellAbility spell) {
        if (spell.getApi() == ApiType.SacrificeAll) {
            return true;
        }
        if (spell.getApi() != ApiType.Sacrifice) {
            return false;
        }
        String defined = spell.getParam("Defined");
        return defined != null && (defined.contains("Player") || defined.contains("Opponent"))
                && Boards.number(spell.getParam("Amount")).orElse(0) >= 2;
    }

    /** Whether the card is shuffled into its owner's library instead of going to a graveyard (Darksteel Colossus). */
    static boolean shufflesIntoLibrary(Card card) {
        return ownGraveyardReplacement(card).map(replacing -> "True".equals(replacing.get("Shuffle")))
                .orElse(false);
    }

    /**
     * The parameters of the ability that puts the card into its owner's
     * library instead of its going to a graveyard, when one of its own
     * replacement effects does so.
     */
    static Optional<Map<String, String>> ownGraveyardReplacement(Card card) {
        for (ReplacementEffect replacement : card.getReplacementEffects()) {
            if (replacement.getMode() != ReplacementType.Moved
                    || !Boards.lists(replacement.getParam("Destination"), "Graveyard")
                    || !"Card.Self".equals(replacement.getParam("ValidCard"))) {
                continue;
            }
            Map<String, String> replacing = Boards.script(card, replacement.getParam("ReplaceWith"));
            if (ApiType.ChangeZone.name().equals(Boards.api(replacing))
                    && "Library".equals(replacing.get("Destination"))) {
                return Optional.of(replacing);
            }
        }
        return Optional.empty();
    }
}
