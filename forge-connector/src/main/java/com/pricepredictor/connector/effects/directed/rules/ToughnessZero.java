package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.BoardEdit;
import com.pricepredictor.connector.effects.directed.operations.PtOperation;
import forge.ai.ComputerUtilCard;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.List;
import java.util.Optional;

/**
 * 704.5f, a creature with toughness 0 or less is put into its owner's
 * graveyard, on scenario {@code 704.5f-1}: A holds Disfigure in its main
 * phase and B's Drudge Skeletons is a 1/1, so the -2/-2 takes it to
 * toughness -1 and it dies whatever its regeneration shield.
 *
 * <p>The base board, read at a boundary: a main phase begins, a player holds
 * a spell that gives a target creature -N/-N (a pump with a negative
 * toughness number, the card's largest: {@link #SHRINKING}), can cast it now
 * ({@link Boards#castableAtItsSpeed}: a sorcery only as the active player)
 * and has the mana for it, colours included, and an opponent of that player
 * controls a creature of toughness 1 to N the spell can target, which the
 * spell would take to 0 or less. It binds the creature Forge's AI rates best
 * among those, the one its removal would most likely aim at, the spell's
 * card and its holder. Whether the live game casts the spell is its own
 * decision; the fork forces the cast at the bound creature (FR-010b), which
 * Forge's AI does not make at a creature the spell no longer kills.
 *
 * <p>Variant {@code tougher} ({@code pt}): the creature's base toughness rises
 * by N, so the spell leaves it at toughness 1 or more, as Drudge Skeletons is
 * a 1/3 in the scenario's variant.
 */
public final class ToughnessZero {

    public static final String RULE = "704.5f";
    static final String SCENARIO = "704.5f-1";
    static final String TOUGHER = "tougher";
    /**
     * A spell of the card that gives a target creature the card's largest
     * fixed -N/-N ({@link #toughnessLoss}): what the precondition finds and
     * the fork casts, the half of a split card that shrinks.
     */
    static final ForcedPlay.Line SHRINKING = ForcedPlay.Line.spell("toughness loss", spell -> {
        int loss = toughnessLoss(spell);
        return loss > 0 && loss == toughnessLoss(spell.getHostCard());
    });

    private ToughnessZero() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, ToughnessZero::bind, TOUGHER,
                new Application(ForkOperation.PT, ToughnessZero::tougher)
                        .forcing(ForcedPlay.cast("holder", "spell").playing(SHRINKING).targeting("creature")));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        for (Player holder : live.getPlayers()) {
            for (SpellAbility spell : Scan.castableSpellsInHand(holder, SHRINKING)) {
                Optional<Binding> binding = Boards.castableAtItsSpeed(holder, spell.getHostCard())
                        ? killing(holder, spell) : Optional.empty();
                if (binding.isPresent()) {
                    return binding;
                }
            }
        }
        return Optional.empty();
    }

    /**
     * The binding of {@code holder}'s {@code spell} and the opposing creature
     * it would kill that Forge's AI rates best, among those of toughness 1 to
     * the spell's N it can target; empty when there is none.
     */
    private static Optional<Binding> killing(Player holder, SpellAbility spell) {
        int loss = toughnessLoss(spell);
        SpellAbility cast = Mana.asPlayedBy(spell, holder);
        List<Card> killed = holder.getOpponents().stream()
                .flatMap(opponent -> opponent.getCreaturesInPlay().stream())
                .filter(creature -> creature.getNetToughness() >= 1 && creature.getNetToughness() <= loss)
                .filter(cast::canTarget)
                .toList();
        if (killed.isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(Binding.empty()
                .with("holder", holder)
                .with("creature", ComputerUtilCard.getBestCreatureAI(killed))
                .with("spell", spell.getHostCard()));
    }

    /**
     * The largest toughness one of the card's targeted -N/-N spells takes
     * away ({@link #toughnessLoss(SpellAbility)}).
     */
    static int toughnessLoss(Card card) {
        return card.getSpells().stream().mapToInt(ToughnessZero::toughnessLoss).max().orElse(0);
    }

    /**
     * The toughness a targeted -N/-N spell takes away, 0 for any other spell:
     * only a number written into the script counts, since an X is chosen on
     * casting.
     */
    static int toughnessLoss(SpellAbility spell) {
        if (spell.getApi() != ApiType.Pump || !spell.usesTargeting()) {
            return 0;
        }
        return Scripts.number(spell.getParam("NumDef")).map(def -> Math.max(0, -def)).orElse(0);
    }

    private static BoardEdit tougher(Binding fork) {
        Card creature = fork.card("creature");
        return new PtOperation(creature, null,
                creature.getCurrentToughness() + toughnessLoss(fork.card("spell")));
    }
}
