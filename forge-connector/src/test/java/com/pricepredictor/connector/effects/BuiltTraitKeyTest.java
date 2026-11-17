package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.trigger.WrappedAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * A trait a keyword instance built keys to its keyword's line plus
 * {@code built}: the trait's kind and its position in that instance's list
 * of the kind (finding 1.1). Without it every chapter of a saga named its
 * keyword's one key and joined chapter I's text.
 *
 * <p>A keyword's own line, the one a board lists per keyword instance, stays
 * the bare key.
 */
@ExtendWith(ForgeExtension.class)
class BuiltTraitKeyTest {

    private static final String BENALIA = "cardsfolder/h/history_of_benalia.txt";

    private static Player you(Game game) {
        return game.getPlayers().get(0);
    }

    /** The chapter trigger of History of Benalia for {@code chapter}, by its own {@code Chapter$}. */
    private static Trigger chapter(Card saga, int chapter) {
        for (Trigger trigger : saga.getCurrentState().getTriggers()) {
            if (trigger.getKeyword() != null && String.valueOf(chapter).equals(trigger.getParam("Chapter"))) {
                return trigger;
            }
        }
        throw new AssertionError("no chapter " + chapter + " trigger on " + saga.getName());
    }

    /** The keyword instance whose script text starts with {@code text}. */
    private static KeywordInterface keywordOf(CardState state, String text) {
        for (KeywordInterface each : state.getIntrinsicKeywords()) {
            if (each.getOriginal().startsWith(text)) {
                return each;
            }
        }
        throw new AssertionError("no " + text + " on " + state.getCard().getName());
    }

    private static SpellAbility cyclingOf(Card card) {
        for (SpellAbility sa : card.getCurrentState().getSpellAbilities()) {
            if (sa.getKeyword() != null && sa.getKeyword().getKeyword() == Keyword.CYCLING) {
                return sa;
            }
        }
        throw new AssertionError("no cycling ability on " + card.getName());
    }

    private static ProvenanceKey keywordKey(String scriptFile, int index) {
        return new ProvenanceKey(scriptFile, 0, TraitKind.KEYWORD, index);
    }

    @Test
    void aSagasChaptersKeyTheirKeywordWithWhichTriggerEachIs() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Card benalia = TestBoard.onBattlefield(game, you(game), "History of Benalia");

        assertEquals(keywordKey(BENALIA, 0).withBuilt(TraitKind.TRIGGER, 0),
                ProvenanceResolver.of(chapter(benalia, 1)), "chapter I");
        assertEquals(keywordKey(BENALIA, 0).withBuilt(TraitKind.TRIGGER, 2),
                ProvenanceResolver.of(chapter(benalia, 3)), "chapter III");
    }

    /** The shape a resolving chapter reaches a record in: its ability wrapped around the trigger. */
    @Test
    void aResolvingChapterKeysToItsOwnTrigger() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Card benalia = TestBoard.onBattlefield(game, you(game), "History of Benalia");
        Trigger third = chapter(benalia, 3);

        WrappedAbility wrapped = new WrappedAbility(third, third.ensureAbility(), you(game));

        assertEquals(keywordKey(BENALIA, 0).withBuilt(TraitKind.TRIGGER, 2),
                ProvenanceResolver.of(wrapped));
    }

    /** One line per keyword instance on a board: the saga's own line names no chapter. */
    @Test
    void aSagasKeywordLineIsTheBareKey() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        CardState state = TestBoard.onBattlefield(game, you(game), "History of Benalia").getCurrentState();

        assertEquals(keywordKey(BENALIA, 0),
                ProvenanceResolver.keyOfKeyword(state, keywordOf(state, "Chapter")));
    }

    /** Cycling builds an activated ability, a {@code spell} trait, and the stack activates a copy of it. */
    @Test
    void aCyclingAbilityKeysItsKeywordWithTheFirstSpellItBuilt() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Card censor = TestBoard.inHand(game, you(game), "Censor");
        SpellAbility cycling = cyclingOf(censor);
        ProvenanceKey expected = keywordKey("cardsfolder/c/censor.txt",
                FaceOrder.keywordIndex(censor.getCurrentState(), cycling.getKeyword()))
                .withBuilt(TraitKind.SPELL, 0);

        SpellAbility onStack = cycling.copy(censor, you(game), false, true);
        onStack.setOriginalAbility(cycling);

        assertEquals(expected, ProvenanceResolver.of(cycling), "the ability as built");
        assertEquals(expected, ProvenanceResolver.of(onStack), "the copy the stack activates");
    }

    /** A copy with no way back to a member of the keyword's lists keeps the bare key: no index is guessed. */
    @Test
    void aKeywordTraitInNoneOfItsKeywordsListsKeepsTheBareKey() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Card censor = TestBoard.inHand(game, you(game), "Censor");
        SpellAbility cycling = cyclingOf(censor);

        SpellAbility orphan = cycling.copy(censor, you(game), false, true);

        ProvenanceKey key = ProvenanceResolver.of(orphan);
        assertEquals(TraitKind.KEYWORD, key.traitKind());
        assertNull(key.built());
    }

    /** Flying builds no trait, and its line carries no {@code built}. */
    @Test
    void flyingHasNoBuilt() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        CardState state = TestBoard.onBattlefield(game, you(game), "Serra Angel").getCurrentState();

        ProvenanceKey key = ProvenanceResolver.keyOfKeyword(state, keywordOf(state, "Flying"));

        assertEquals(keywordKey("cardsfolder/s/serra_angel.txt", 0), key);
        assertNull(key.built());
    }
}
