package com.pricepredictor.connector.effects;

import forge.card.MagicColor;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.cost.Cost;
import forge.game.cost.CostPayment;
import forge.game.cost.CostUntap;
import forge.game.player.Player;
import forge.game.spellability.AbilityManaPart;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbilityActivateAbilityAsIfHaste;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;

/**
 * A player's {@code untapped_production}: how much of each colour the player
 * could still make, per colour, on the board a snapshot shows.
 *
 * <p>Counted over the untapped permanents the player controls on that board,
 * one per mana ability that can be activated now and can produce the colour.
 * It is what separates a board that can answer a threat from one that only
 * looks like it can: a summoning-sick creature's {T} ability, or one its
 * restrictions forbid (once each turn and already activated, a phase it may
 * not be activated in, a condition, a sacrifice with nothing to sacrifice),
 * counts nothing. An ability that makes one mana of any colour counts once
 * for each, so the numbers are an upper bound rather than a sum that can be
 * spent.
 *
 * <p>On a continuous record's board ({@link StaticSuppression}) the static's
 * work is left out: the mana abilities it granted (Chromatic Lantern's), the
 * ones it took away back, the permanents it took control of with their
 * controller before it, and the haste it gave a creature that came under its
 * controller's control this turn. Forge's own activation checks still read
 * the live board, so the permanent's live controller is the one asked
 * whether it could activate the ability.
 */
final class UntappedProduction {

    private UntappedProduction() {
    }

    /** What {@code player} could still make on {@code board}. */
    static ManaCounts of(Game game, Player player, StaticSuppression board) {
        List<Byte> producible = new ArrayList<>();
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            if (!card.isTapped() && player.equals(board.controller(card))) {
                producible.addAll(producibleBy(card, board));
            }
        }
        return ManaCounts.tally(producible);
    }

    /** The colours each of one permanent's mana abilities on {@code board} can make now. */
    private static List<Byte> producibleBy(Card card, StaticSuppression board) {
        List<Byte> producible = new ArrayList<>();
        for (SpellAbility ability : board.spellAbilities(card.getCurrentState())) {
            if (isManaAbility(ability) && !sickOnThisBoard(ability, board)) {
                producible.addAll(colorsOf(ability, card.getController()));
            }
        }
        return producible;
    }

    private static boolean isManaAbility(SpellAbility ability) {
        return ability.isManaAbility() && ability.getManaPart() != null;
    }

    /**
     * Whether the ability taps or untaps a host that is summoning sick on a
     * continuous record's board though not on the live one, its haste the
     * suppressed static's. The live board's own sickness is Forge's check.
     */
    private static boolean sickOnThisBoard(SpellAbility ability, StaticSuppression board) {
        Card host = ability.getHostCard();
        return board.active() && tapsItsHost(ability.getPayCosts()) && board.sick(host)
                && !StaticAbilityActivateAbilityAsIfHaste.canActivate(host);
    }

    private static boolean tapsItsHost(Cost cost) {
        return cost != null && (cost.hasTapCost() || cost.hasSpecificCostType(CostUntap.class));
    }

    /**
     * The colours one mana ability can make when {@code activator} activates it.
     *
     * <p>An ability's activating player is whoever last looked at it, and the
     * opponent's AI lists every battlefield's abilities as its own candidates:
     * read as it was left, Bloom Tender's "each colour among permanents you
     * control" counted the opponent's colours. Its X is whatever an earlier
     * activation or AI evaluation announced, which a copy's ability, rebuilt
     * from the card, does not carry: Mage-Ring Network's "remove X storage
     * counters" read with a leftover X it can no longer pay counted nothing
     * live and one in the copy, so it is read with no X announced, as X = 0.
     * Reading a special mana ability also writes its express choice. All
     * three are put back, so the read changes nothing in the game. None when
     * the player cannot activate it now.
     */
    private static List<Byte> colorsOf(SpellAbility ability, Player activator) {
        AbilityManaPart part = ability.getManaPart();
        SpellAbility root = ability.getRootAbility();
        Player previous = ability.getActivatingPlayer();
        String choice = part.getExpressChoice();
        Integer x = root.getXManaCostPaid();
        ability.setActivatingPlayer(activator);
        root.setXManaCostPaid(null);
        try {
            return activatableNow(ability, activator) ? producedColors(ability, part) : List.of();
        } finally {
            ability.setActivatingPlayer(previous);
            part.setExpressChoice(choice);
            root.setXManaCostPaid(x);
        }
    }

    private static List<Byte> producedColors(SpellAbility ability, AbilityManaPart part) {
        List<Byte> colors = new ArrayList<>();
        for (byte symbol : MagicColor.WUBRGC) {
            if (part.canProduce(ManaCounts.letter(symbol), ability)) {
                colors.add(symbol);
            }
        }
        return colors;
    }

    /**
     * Whether {@code player}, set as the ability's activator, could activate
     * it now by the rules alone: Forge's own checks of its restrictions
     * (activation limits, {@code ActivationPhases$}, conditions), of the
     * statics forbidding activation, and of its non-mana costs ({T} on a
     * summoning-sick creature, a sacrifice or counter with nothing to pay it
     * with), as {@link ManaAffordability} reads a source. No AI check, no AI
     * card memory, no draw from {@code MyRandom}. The ability's own mana cost
     * is not checked: other sources pay it.
     */
    private static boolean activatableNow(SpellAbility ability, Player player) {
        return ability.canPlay() && ability.checkRestrictions(player)
                && CostPayment.canPayAdditionalCosts(ability.getPayCosts(), ability, false, player)
                && ability.metConditions();
    }
}
