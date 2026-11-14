package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.BoardEdit;
import com.pricepredictor.connector.effects.directed.operations.LifeOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.zone.ZoneType;

import java.util.Optional;
import java.util.Set;

/**
 * 702.105a, dethrone triggers when its creature attacks the player with the
 * most life or tied for the most life, on scenario {@code 702.105a-1}: A (15
 * life) attacks B (20 life) with Enraged Revolutionary, and dethrone puts a
 * +1/+1 counter on it.
 *
 * <p>The base board, read at a boundary: the declare-attackers step begins,
 * before attackers are declared, the active player controls an untapped
 * creature with dethrone that can attack this turn, and an opponent has the
 * most life or is tied for it, while every other player has 2 life or more
 * and more than the active player's creatures could deal this turn
 * ({@link #attackReach}), so the variant's cut leaves the opponent out of
 * reach of a lethal attack the base board did not offer. The reach is an
 * upper bound only where nothing can grow the attack once attackers are
 * declared, so the board binds nothing when the active player has an
 * attack-time trigger other than dethrone's own (exalted, battle cry, a
 * creature growing as it attacks or is blocked), a pump or damage ability
 * on a permanent, or an instant-speed pump or damage spell it has the lands
 * for ({@link #canGrowTheAttack}). It binds the creature and that opponent.
 *
 * <p>Variant {@code less-life} ({@code life}): the opponent has one life less
 * than the most any other player has, so attacking it does not trigger
 * dethrone. The other players having 2 or more keeps the opponent alive, and
 * having more than the attack's reach keeps the attack from killing it.
 *
 * <p>The fork keeps the not-fired evaluations of the triggers the creature hosts
 * whatever the trigger-negative sampler's balance says (FR-010c): a fork's
 * recording is fresh, so the sampler holds no earlier firing of the mode
 * for a negative to repay. Whether the creature attacks is its seat's own
 * decision.
 */
public final class Dethrone {

    public static final String RULE = "702.105a";
    static final String SCENARIO = "702.105a-1";
    static final String LESS_LIFE = "less-life";

    private Dethrone() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Dethrone::bind, LESS_LIFE,
                new Application(ForkOperation.LIFE, Dethrone::lessLife).keepingNegativesOf("attacker"));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.at(live, PhaseType.COMBAT_DECLARE_ATTACKERS)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        Optional<Card> attacker = Boards.untappedCreatures(active).stream()
                .filter(creature -> creature.hasKeyword(Keyword.DETHRONE) && !creature.isSick())
                .findFirst();
        if (attacker.isEmpty()) {
            return Optional.empty();
        }
        if (canGrowTheAttack(active)) {
            return Optional.empty();
        }
        int reach = attackReach(active);
        for (Player opponent : active.getOpponents()) {
            int most = Boards.mostLifeBesides(opponent);
            if (opponent.getLife() >= most && most >= 2 && most - 1 > reach) {
                return Optional.of(Binding.empty()
                        .with("attacker", attacker.get())
                        .with("opponent", opponent));
            }
        }
        return Optional.empty();
    }

    /**
     * The most combat damage {@code active}'s creatures could deal this turn
     * when nothing grows the attack ({@link #canGrowTheAttack}): the power of
     * each untapped creature that can attack, one more for each with dethrone,
     * whose counter would come before damage, and twice that for a double
     * striker, which deals it in both damage steps.
     */
    static int attackReach(Player active) {
        return Boards.untappedCreatures(active).stream()
                .filter(creature -> !creature.isSick())
                .mapToInt(creature -> (Math.max(0, creature.getNetCombatDamage())
                        + (creature.hasKeyword(Keyword.DETHRONE) ? 1 : 0))
                        * (creature.hasDoubleStrike() ? 2 : 1))
                .sum();
    }

    /** The APIs a trigger, an ability or a spell grows an attack's damage with. */
    private static final Set<ApiType> GROWING = Set.of(ApiType.Pump, ApiType.PumpAll, ApiType.PutCounter,
            ApiType.PutCounterAll, ApiType.DealDamage, ApiType.DamageAll, ApiType.LoseLife);

    /**
     * Whether something {@code active} has could add to the attack's damage
     * after the declare-attackers step begins, which {@link #attackReach}
     * leaves out: a trigger of an attack or a block on one of its permanents
     * other than a dethrone instance (exalted, battle cry, bushido), a
     * non-mana activated ability of one of its permanents with a pump,
     * counter or damage effect (firebreathing), or a spell it holds and has
     * the lands for that it can cast in combat (an instant or a card with
     * flash) with one.
     */
    static boolean canGrowTheAttack(Player active) {
        for (Card permanent : active.getCardsIn(ZoneType.Battlefield)) {
            for (Trigger trigger : permanent.getTriggers()) {
                if (trigger.getMode().name().startsWith("Attack") && !trigger.isKeyword(Keyword.DETHRONE)) {
                    return true;
                }
            }
            for (SpellAbility ability : permanent.getSpellAbilities()) {
                if (ability.isActivatedAbility() && !ability.isManaAbility() && grows(ability)) {
                    return true;
                }
            }
        }
        return active.getCardsIn(ZoneType.Hand).stream()
                .filter(card -> card.isInstant() || card.hasKeyword(Keyword.FLASH))
                .filter(card -> Boards.affordable(active, card))
                .anyMatch(card -> card.getSpells().stream().anyMatch(Dethrone::grows));
    }

    /** Whether a clause of the ability's chain has one of the {@link #GROWING} APIs. */
    private static boolean grows(SpellAbility ability) {
        for (SpellAbility clause = ability; clause != null; clause = clause.getSubAbility()) {
            if (GROWING.contains(clause.getApi())) {
                return true;
            }
        }
        return false;
    }

    private static BoardEdit lessLife(Binding fork) {
        Player opponent = fork.player("opponent");
        return new LifeOperation(opponent, Boards.mostLifeBesides(opponent) - 1);
    }
}
