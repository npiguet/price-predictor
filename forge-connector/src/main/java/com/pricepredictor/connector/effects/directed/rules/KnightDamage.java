package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.StaticData;
import forge.ai.ComputerUtilCombat;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.item.IPaperCard;

/**
 * Whether the card {@code replace_card} puts in a blocker's place would
 * destroy an attacker with its combat damage (510.2): the attacker can be
 * destroyed at all (702.12b), and the damage is not prevented (protection,
 * 702.16e, or a prevention effect), read as the AI predicts damage.
 */
final class KnightDamage {

    /** The id of a probe card, which no zone holds and the game never allots. */
    private static final int PROBE_ID = -1;

    private KnightDamage() {
    }

    /**
     * Whether {@code cardName} in {@code blocker}'s place deals {@code attacker}
     * lethal combat damage: its power ({@link ReplaceCardOperation#replacementPower})
     * after prevention reaches the attacker's toughness less its marked damage.
     */
    static boolean destroys(Game live, Card blocker, String cardName, Card attacker) {
        if (!attacker.canBeDestroyed()) {
            return false;
        }
        int power = ReplaceCardOperation.replacementPower(blocker, cardName);
        int dealt = ComputerUtilCombat.predictDamageTo(attacker, power, probe(live, blocker, cardName), true);
        return dealt >= attacker.getNetToughness() - attacker.getDamage();
    }

    /** A card named {@code cardName} owned by the blocker's owner, in no zone, standing for the replacement as a damage source. */
    private static Card probe(Game live, Card blocker, String cardName) {
        IPaperCard paper = StaticData.instance().getCommonCards().getCard(cardName);
        return CardFactory.getCard(paper, blocker.getOwner(), PROBE_ID, live);
    }
}
