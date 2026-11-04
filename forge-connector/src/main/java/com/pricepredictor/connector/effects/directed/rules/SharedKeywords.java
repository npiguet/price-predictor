package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * 702.1c, "the same is true for" grants each listed keyword, with its
 * variants, to the creatures an effect names when one of them has it, on
 * scenario {@code 702.1c-1}: Concerted Effort's upkeep trigger gives Grizzly
 * Bears the first strike and protection from black White Knight has.
 *
 * <p>The base board, read at a boundary: the upkeep begins, a player controls
 * a permanent whose beginning-of-upkeep trigger grants its creatures the
 * listed keywords any of them has (Concerted Effort, a {@code PumpAll} with
 * {@code SharedKeywordsZone$}), and of the two or more creatures that player
 * controls exactly one has any of the listed keywords, so the others would
 * gain it. It binds that creature.
 *
 * <p>Variant {@code no_keywords} ({@code replace_card}): Hill Giant is in that
 * creature's place, so no creature has a listed keyword and none is granted.
 */
public final class SharedKeywords {

    public static final String RULE = "702.1c";
    static final String SCENARIO = "702.1c-1";
    static final String NO_KEYWORDS = "no_keywords";
    static final String GIANT = "Hill Giant";

    private SharedKeywords() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, SharedKeywords::bind, NO_KEYWORDS,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card("creature"), GIANT)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.UPKEEP)) {
            return Optional.empty();
        }
        for (Player player : live.getPlayers()) {
            for (Card permanent : player.getCardsIn(ZoneType.Battlefield)) {
                for (Trigger trigger : permanent.getTriggers()) {
                    Optional<Card> holder = sharedKeywords(trigger).flatMap(listed -> soleHolder(player, listed));
                    if (holder.isPresent()) {
                        return Optional.of(Binding.empty().with("creature", holder.get()));
                    }
                }
            }
        }
        return Optional.empty();
    }

    /** The keywords a beginning-of-upkeep trigger shares among its controller's creatures, if it is one. */
    static Optional<Set<Keyword>> sharedKeywords(Trigger trigger) {
        if (trigger.getMode() != TriggerType.Phase || !"Upkeep".equals(trigger.getParam("Phase"))) {
            return Optional.empty();
        }
        Map<String, String> pump = Boards.executed(trigger);
        if (!ApiType.PumpAll.name().equals(Boards.api(pump)) || !pump.containsKey("SharedKeywordsZone")
                || pump.get("KW") == null) {
            return Optional.empty();
        }
        return Optional.of(Keyword.setValueOf(pump.get("KW")));
    }

    /** The one creature {@code player} controls with any of {@code listed}, when it controls another. */
    private static Optional<Card> soleHolder(Player player, Set<Keyword> listed) {
        List<Card> creatures = List.copyOf(player.getCreaturesInPlay());
        List<Card> holders = creatures.stream()
                .filter(creature -> listed.stream().anyMatch(creature::hasKeyword))
                .toList();
        return creatures.size() >= 2 && holders.size() == 1 ? Optional.of(holders.get(0)) : Optional.empty();
    }
}
