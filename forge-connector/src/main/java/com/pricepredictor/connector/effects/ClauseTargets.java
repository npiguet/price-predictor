package com.pricepredictor.connector.effects;

import forge.game.ability.AbilityUtils;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;

/**
 * The players and cards a resolving clause acts on, read the way the clause's
 * own effect reads them.
 *
 * <p>Forge's readings live in {@code SpellAbilityEffect} ({@code getPlayers},
 * {@code getTargetCards}, {@code getCardsfromTargets}), which are
 * {@code protected} to its effect classes, so they are mirrored here: one
 * place for every {@link ApiEvents} rule that names who or what a clause
 * touched.
 */
final class ClauseTargets {

    private ClauseTargets() {
    }

    /**
     * Every player {@code AddTurn}, {@code SkipTurn} and {@code SkipPhase}
     * actually affect -- the ones their own {@code resolve()} iterates, not
     * whoever activated them.
     *
     * <p>All three iterate {@code SpellAbilityEffect.getPlayers(false,
     * "Defined", sa)}, which is private to its package and so mirrored here:
     * <b>targeting wins over {@code Defined$}</b> whenever the ability
     * targets at all. The activator is not a safe default: most
     * {@code SkipPhase} scripts, and {@code empty_city_ruse.txt}'s
     * {@code Opponent} target, name someone else.
     *
     * <p>{@link ApiEvents} captures it as a memo, before the clause
     * resolves: none of the three changes {@code sa}'s targets, so either
     * side reads the same, and the memo is what pins the clause hook's
     * push/pop pairing against a non-null value.
     */
    static List<Player> affectedPlayers(SpellAbility sa) {
        if (sa.usesTargeting()) {
            List<Player> targeted = new ArrayList<>();
            sa.getTargets().getTargetPlayers().forEach(targeted::add);
            return targeted;
        }
        return definedPlayers(sa, sa.getParamOrDefault("Defined", "You"));
    }

    /**
     * {@code AbilityUtils.getDefinedPlayers} for one {@code Defined$}/
     * {@code DefinedPlayer$} value, extended to the {@code " & "}-joined
     * multi-value form {@code SpellAbilityEffect.getPlayers} also accepts:
     * {@code String[] def =
     * sa.getParamOrDefault(definedParam, "You").split(" & ")}, one lookup per
     * token, unioned. No card in the cardsfolder uses this today for either
     * {@code Defined$} on these three rules or {@code DefinedPlayer$} on
     * {@code GainOwnership}, but an approximation this cheap to close is not
     * worth keeping.
     */
    static List<Player> definedPlayers(SpellAbility sa, String def) {
        List<Player> players = new ArrayList<>();
        for (String token : def.split(" & ")) {
            players.addAll(AbilityUtils.getDefinedPlayers(sa.getHostCard(), token, sa));
        }
        return players;
    }

    /**
     * The cards a clause with no dedicated targeting concept of its own
     * acts on -- {@link #affectedPlayers}'s gate, mirrored for cards rather
     * than players, because {@code HealDamage} is exactly that shape:
     * {@code SpellAbilityEffect.getCards} (which {@code
     * HealDamageEffect.resolve()} calls through {@code getTargetCards(sa)})
     * reads {@code sa.getTargets()} only when {@code sa.usesTargeting()}
     * is true, and otherwise falls back to a {@code " & "}-split {@code
     * Defined$}, defaulted to {@code "Self"} -- the same default used here, not
     * a different one invented for this file.
     */
    static List<Card> affectedCards(SpellAbility sa) {
        if (sa.usesTargeting()) {
            List<Card> targeted = new ArrayList<>();
            sa.getTargets().getTargetCards().forEach(targeted::add);
            return targeted;
        }
        return definedCards(sa, sa.getParamOrDefault("Defined", "Self"));
    }

    /** {@link #definedPlayers}, mirrored for cards via {@code AbilityUtils.getDefinedCards}. */
    static List<Card> definedCards(SpellAbility sa, String def) {
        List<Card> cards = new ArrayList<>();
        for (String token : def.split(" & ")) {
            cards.addAll(AbilityUtils.getDefinedCards(sa.getHostCard(), token, sa));
        }
        return cards;
    }

    /**
     * {@code SpellAbilityEffect.getCardsfromTargets(sa)}, mirrored
     * the way {@link #affectedCards} already mirrors {@code getTargetCards}.
     *
     * <p>The {@code getTargetSpells()} union below is NOT dead for today's
     * cardsfolder. Seven of {@code ChangeText}'s seventeen
     * occurrences are {@code ValidTgts$ Card | TgtZone$ Stack,Battlefield} --
     * {@code artificial_evolution}, {@code sleight_of_mind} and {@code
     * magical_hack} among them -- which can target a spell on the stack, and
     * {@code TargetChoices} stores a spell target as the {@code SpellAbility}
     * itself ({@code getTargetCards()} filters to {@code Card.class}
     * instances only, per {@code TargetChoices.java}; a targeted spell is a
     * {@code SpellAbility}, not one). For that shape this union is not an
     * extra path -- it is the <em>only</em> one that produces a subject at
     * all; {@link #affectedCards}'s targeting branch reads {@code
     * getTargetCards()} alone and would see nothing.
     */
    static List<Card> cardsFromTargets(SpellAbility sa) {
        List<Card> cards = new ArrayList<>(affectedCards(sa));
        if (sa.getTargets() != null) {
            for (SpellAbility spellTarget : sa.getTargets().getTargetSpells()) {
                cards.add(spellTarget.getHostCard());
            }
        }
        return cards;
    }
}
