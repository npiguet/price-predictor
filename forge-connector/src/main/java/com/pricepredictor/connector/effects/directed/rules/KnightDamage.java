package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.StaticData;
import forge.card.ICardFace;
import forge.ai.ComputerUtilCombat;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.trackable.Tracker;
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

    /**
     * A card standing for {@code cardName} as a damage source, owned by the
     * blocker's owner and in no zone: its name, type and colour, which
     * protection and prevention effects read. It is built with a tracker of
     * its own, as {@code CardFactory.getCard} would register its views with
     * the live game's, and a precondition leaves the live game as it was.
     */
    private static Card probe(Game live, Card blocker, String cardName) {
        IPaperCard paper = StaticData.instance().getCommonCards().getCard(cardName);
        ICardFace face = paper.getRules().getMainPart();
        Card probe = new Card(PROBE_ID, paper, live, new Tracker());
        probe.setOwner(blocker.getOwner());
        CardState state = probe.getCurrentState();
        state.setName(face.getName());
        state.setType(face.getType());
        state.setColor(face.getColor());
        return probe;
    }
}
