package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.ForkRefusal;
import com.pricepredictor.connector.effects.directed.ForkOperationSupport.Board;
import com.pricepredictor.connector.effects.directed.operations.KeywordOperation;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Set;

import static com.pricepredictor.connector.effects.directed.BoardChanges.id;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The {@code keyword} operation changes the card's lines alone, and refuses
 * a removal of a keyword the card lacks and a keyword the card cannot have.
 *
 * <p>The board each test edits and how a check reads it: {@link ForkOperationSupport}.
 */
@ExtendWith(ForgeExtension.class)
class ForkOperationKeywordTest {

    @Test
    void keywordAddedChangesTheLinesAlone() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        // A keyword no static grants reads as granted until the turn ends:
        // the line and the temporary grant name the one keyword.
        assertEquals(Set.of(id(bears) + ".lines", id(bears) + ".granted_temporary"),
                board.edit(KeywordOperation.adding(bears, "Flying")));
        assertTrue(bears.hasKeyword(Keyword.FLYING));
    }

    @Test
    void keywordRemovedByTheCorpusSpelling() {
        Board board = Board.main();
        Card angel = board.creature(board.you, "Serra Angel");
        assertEquals(Set.of(id(angel) + ".lines", id(angel) + ".granted_temporary"),
                board.edit(KeywordOperation.removing(angel, "flying")));
        assertFalse(angel.hasKeyword(Keyword.FLYING));
        assertTrue(angel.hasKeyword(Keyword.VIGILANCE));
    }

    @Test
    void removingAKeywordTheCardLacksIsRefused() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        assertThrows(com.pricepredictor.connector.effects.ForkRefusal.class,
                () -> KeywordOperation.removing(bears, "flying").apply(board.game));
    }

    /**
     * A keyword the card cannot have (a can't-have effect) is refused rather
     * than silently not added, so a fork tagged with the edit is discarded.
     */
    @Test
    void aKeywordTheCardCannotHaveIsRefused() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        bears.addCantHaveKeyword(board.game.getNextTimestamp(), 0L,
                List.of(forge.game.keyword.Keyword.FLYING));
        bears.updateKeywordsCache();

        assertThrows(com.pricepredictor.connector.effects.ForkRefusal.class,
                () -> KeywordEdit.add(board.game, bears, "Flying"));
    }
}
