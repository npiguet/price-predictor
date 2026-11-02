package com.pricepredictor.connector.effects.directed;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Adding a keyword to a card, or taking one away, through the layer system:
 * the edit the damage-step probe strips a combatant's keyword with, and the
 * {@code keyword} fork operation's in both directions.
 *
 * <p>Through a keyword-changing layer rather than off the printed list: the
 * keyword may be printed, equipped or granted until end of turn, and a
 * removal layer takes it away in all three cases the way the rules would. The
 * edit carries a fresh timestamp, so it applies after every effect already on
 * the card, and no static ability: nothing removes it when the statics are
 * checked again.
 */
public final class KeywordEdit {

    private KeywordEdit() {
    }

    /**
     * Forge's spelling of a keyword in the one every reader uses.
     *
     * <p>Mirrors {@code effects.domain.state_snapshot.normalize_keyword}: the
     * parameter after {@code :} goes, spaces become underscores. Two spellings
     * of first strike is exactly the mismatch that made gate 2 report zero
     * observations of it.
     */
    public static String normalize(String original) {
        int parameter = original.indexOf(':');
        String base = parameter < 0 ? original : original.substring(0, parameter);
        return base.trim().toLowerCase(Locale.ROOT).replace(' ', '_');
    }

    /**
     * Take {@code keyword}, spelled as the card carries it, away from {@code card}.
     *
     * @param game the game the card is in, whose timestamp the edit takes
     */
    public static void remove(Game game, Card card, String keyword) {
        card.addChangedCardKeywords(
                null, List.of(keyword), false, game.getNextTimestamp(), null);
    }

    /**
     * Give {@code card} the keyword, spelled as a card script spells it
     * ({@code Flying}, {@code Ward:2}).
     */
    public static void add(Game game, Card card, String keyword) {
        card.addChangedCardKeywords(
                List.of(keyword), null, false, game.getNextTimestamp(), null);
    }

    /**
     * The card's own spellings of every keyword it carries whose normalized
     * form is {@code keyword}'s, in the order the card lists them: what
     * {@link #remove} takes for a keyword named in either spelling.
     */
    public static List<String> carried(Card card, String keyword) {
        String wanted = normalize(keyword);
        List<String> originals = new ArrayList<>();
        for (KeywordInterface carried : card.getKeywords()) {
            String original = carried.getOriginal();
            if (original != null && !original.isEmpty()
                    && normalize(original).equals(wanted)
                    && !originals.contains(original)) {
                originals.add(original);
            }
        }
        return originals;
    }
}
