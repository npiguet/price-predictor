package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.ForcedPlayRun;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A directed rule's precondition binds only boards where the variant's
 * forced play, or the fact its edit takes away, works (finding 6.7): the
 * mana a cast needs is asked colours included, a target is one the spell can
 * take, a reanimation spell can return the creature card that is there, mana
 * is counted per source rather than per land, a meld trigger can still fire
 * this turn, and a split card's forced cast is the half the precondition
 * matched. Each board is a real {@code TestBoard} game.
 */
@ExtendWith(ForgeExtension.class)
class PlayablePreconditionsTest {

    private final Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
    private final Player you = game.getPlayers().get(0);
    private final Player opponent = game.getPlayers().get(1);

    private Card permanent(Player owner, String name) {
        Card card = TestBoard.inPlay(game, owner, name);
        card.setGameTimestamp(game.getNextTimestamp());
        return card;
    }

    private void permanents(Player owner, String name, int count) {
        for (int i = 0; i < count; i++) {
            permanent(owner, name);
        }
    }

    // ── mana, colours included ──────────────────────────────────────────

    /** Disfigure ({B}) over a Forest: one land for a spell of mana value 1, but not the colour. */
    @Test
    void offColourLandsDoNotBindTheSpell() {
        permanent(you, "Forest");
        TestBoard.inHand(game, you, "Disfigure");
        permanent(opponent, "Drudge Skeletons");
        assertTrue(ToughnessZero.bind(game).isEmpty());

        permanent(you, "Swamp");
        assertTrue(ToughnessZero.bind(game).isPresent(), "with the Swamp the spell is castable");
    }

    // ── targets ─────────────────────────────────────────────────────────

    /** Gladecover Scout (1/1, hexproof) would die to Disfigure, which cannot target it. */
    @Test
    void toughnessZeroBindsOnlyACreatureTheSpellCanTarget() {
        permanent(you, "Swamp");
        TestBoard.inHand(game, you, "Disfigure");
        permanent(opponent, "Gladecover Scout");
        assertTrue(ToughnessZero.bind(game).isEmpty());

        Card skeletons = permanent(opponent, "Drudge Skeletons");
        assertEquals(skeletons, ToughnessZero.bind(game).orElseThrow().card("creature"));
    }

    /** Call a Surprise Witness returns a creature card of mana value 3 or less: not the Craw Wurm. */
    @Test
    void cantEnterBattlefieldBindsOnlyASpellThatCanReturnTheCreatureThere() {
        permanents(you, "Plains", 2);
        TestBoard.inHand(game, you, "Call a Surprise Witness");
        TestBoard.inGraveyard(game, you, "Craw Wurm");
        assertTrue(CantEnterBattlefield.bind(game).isEmpty());

        TestBoard.inGraveyard(game, you, "Grizzly Bears");
        assertTrue(CantEnterBattlefield.bind(game).isPresent(), "the Bears is a card the spell returns");
    }

    /**
     * Arcanum Wings on Black Knight (protection from white), Wild Growth in
     * hand: the Holy Strength the variant puts in hand fails on the Knight as
     * Wild Growth does, so the fork would show no exchange.
     */
    @Test
    void auraSwapCantEnchantBindsNothingOnAHostWithProtection() {
        permanents(you, "Island", 3);
        Card knight = permanent(you, "Black Knight");
        Card wings = permanent(you, "Arcanum Wings");
        wings.attachToEntity(knight, null);
        TestBoard.inHand(game, you, "Wild Growth");
        assertTrue(AuraSwapCantEnchant.bind(game).isEmpty());
    }

    // ── mana per source ─────────────────────────────────────────────────

    /**
     * Reverse Engineer ({3}{U}{U}) over two Islands and three Ornithopters
     * binds; a Llanowar Elves beside them pays the mana the tapped Ornithopter
     * takes away, so the spell stays castable and the board binds nothing.
     */
    @Test
    void improviseCountsAManaCreatureAsMana() {
        permanents(you, "Island", 2);
        permanents(you, "Ornithopter", 3);
        TestBoard.inHand(game, you, "Reverse Engineer");
        assertTrue(Improvise.bind(game).isPresent());

        permanent(you, "Llanowar Elves");
        assertTrue(Improvise.bind(game).isEmpty());
    }

    /**
     * Reverse Engineer over two Forests and three Ornithopters: the mana
     * value adds up, but the Forests cannot pay {U}{U}, so the spell is not
     * castable on the base board and the fork's tapped artifact would change
     * nothing.
     */
    @Test
    void improviseBindsNothingWhoseColoursTheLandsCannotPay() {
        permanents(you, "Forest", 2);
        permanents(you, "Ornithopter", 3);
        TestBoard.inHand(game, you, "Reverse Engineer");
        assertTrue(Improvise.bind(game).isEmpty());
    }

    /** A face-down Exalted Angel (morph {2}{W}{W}) beside Plains and an Azorius Chancery, which makes two mana. */
    private Card angelOver(int plains) {
        permanents(you, "Plains", plains);
        permanent(you, "Azorius Chancery");
        Card angel = TestBoard.card(game, you, "Exalted Angel");
        angel.turnFaceDownNoUpdate();
        you.getZone(ZoneType.Battlefield).add(angel);
        angel.setSickness(false);
        angel.setGameTimestamp(game.getNextTimestamp());
        return angel;
    }

    @Test
    void morphFaceUpCountsABounceLandAsTwoMana() {
        Card angel = angelOver(2);
        assertEquals(angel, MorphFaceUp.bind(game).orElseThrow().card("permanent"),
                "two Plains and the Chancery make exactly the four mana");
    }

    @Test
    void morphFaceUpBindsNothingWhenABounceLandLeavesManaToSpare() {
        angelOver(3);
        assertTrue(MorphFaceUp.bind(game).isEmpty(), "three Plains and the Chancery make five");
    }

    // ── a keyword's cost, colours and sources ───────────────────────────

    private void inLibrary(Player owner, String name) {
        game.getAction().moveTo(ZoneType.Library, TestBoard.card(game, owner, name), null, null);
    }

    /** Dregscape Zombie in the graveyard (unearth {B}, 702.84a, Unearth): a Forest makes one land's mana, not the colour. */
    @Test
    void unearthNeedsTheColourOfItsCost() {
        TestBoard.inGraveyard(game, you, "Dregscape Zombie");
        permanent(you, "Forest");
        assertTrue(Unearth.bind(game).isEmpty());

        permanent(you, "Swamp");
        assertTrue(Unearth.bind(game).isPresent());
    }

    /**
     * Dimir Infiltrator in hand (transmute {1}{U}{B}, 702.53a, Transmute) with
     * a Grizzly Bears of its mana value in the library: three lands, one
     * short of the colours; and a Dimir Aqueduct, which makes {U}{B} alone,
     * counts as the two mana it makes.
     */
    @Test
    void transmuteNeedsTheColoursAndCountsABounceLandAsTwo() {
        TestBoard.inHand(game, you, "Dimir Infiltrator");
        inLibrary(you, "Grizzly Bears");
        permanent(you, "Island");
        permanents(you, "Forest", 2);
        assertTrue(ManaValueSearch.transmute().precondition(game).isEmpty(), "no black");

        permanent(you, "Swamp");
        assertTrue(ManaValueSearch.transmute().precondition(game).isPresent());
    }

    @Test
    void transmuteBindsOnABounceLandAndAForest() {
        TestBoard.inHand(game, you, "Dimir Infiltrator");
        inLibrary(you, "Grizzly Bears");
        permanent(you, "Forest");
        assertTrue(ManaValueSearch.transmute().precondition(game).isEmpty(), "one source");

        permanent(you, "Dimir Aqueduct");
        assertTrue(ManaValueSearch.transmute().precondition(game).isPresent(), "{U}{B} and {G} pay {1}{U}{B}");
    }

    /** Hopeful Eidolon (bestow {3}{W}, 702.103a, Bestow) over four Islands and Grizzly Bears. */
    @Test
    void bestowNeedsTheColourOfItsCost() {
        TestBoard.inHand(game, you, "Hopeful Eidolon");
        permanent(you, "Grizzly Bears");
        permanents(you, "Island", 4);
        assertTrue(BestowIllegalTarget.bind(game).isEmpty());

        permanent(you, "Plains");
        assertTrue(BestowIllegalTarget.bind(game).isPresent());
    }

    /** Spider-Man, Web-Slinger in hand (web-slinging {W}, 702.188a, Web-slinging) over a tapped Grizzly Bears. */
    @Test
    void webSlingingNeedsTheColourOfItsCost() {
        TestBoard.inHand(game, you, "Spider-Man, Web-Slinger");
        permanent(you, "Grizzly Bears").setTapped(true);
        permanent(you, "Island");
        assertTrue(WebSlinging.bind(game).isEmpty());

        permanent(you, "Plains");
        assertTrue(WebSlinging.bind(game).isPresent());
    }

    /** Arcanum Wings (aura swap {2}{U}, 702.65a, Aura swap) on Grizzly Bears, Giant Growth in hand, over three Plains. */
    @Test
    void auraSwapNeedsTheColourOfItsCost() {
        Card bears = permanent(you, "Grizzly Bears");
        permanent(you, "Arcanum Wings").attachToEntity(bears, null);
        TestBoard.inHand(game, you, "Giant Growth");
        permanents(you, "Plains", 3);
        assertTrue(AuraSwap.bind(game).isEmpty());

        permanent(you, "Island");
        assertTrue(AuraSwap.bind(game).isPresent());
    }

    /**
     * Krosan Verge ({2}, {T}, sacrifice: search, 701.23f, Search) under an Archivist of Oghma: the land
     * taps for its own cost, so the {2} comes from the other sources: a Forest makes one, an Azorius Chancery two.
     */
    @Test
    void searchUnderMindcensorCountsTheOtherSourcesMana() {
        permanent(you, "Archivist of Oghma");
        permanent(opponent, "Krosan Verge");
        permanent(opponent, "Forest");
        assertTrue(SearchUnderMindcensor.bind(game).isEmpty(), "one other mana");

        permanent(opponent, "Azorius Chancery");
        assertTrue(SearchUnderMindcensor.bind(game).isPresent(), "three other mana");
    }

    @Test
    void searchUnderMindcensorCountsABounceLandAsTwo() {
        permanent(you, "Archivist of Oghma");
        permanent(opponent, "Krosan Verge");
        permanent(opponent, "Azorius Chancery");
        assertTrue(SearchUnderMindcensor.bind(game).isPresent(), "the Chancery's two mana pay {2}");
    }

    // ── meld triggers ───────────────────────────────────────────────────

    /**
     * Titania, Voice of Gaea melds at the beginning of upkeep, which has
     * passed at the first main phase: a Forest in Argoth's place melds nothing
     * this turn.
     */
    @Test
    void titaniaAtTheFirstMainPhaseDoesNotBind() {
        permanent(you, "Forest");
        permanent(you, "Titania, Voice of Gaea");
        for (int i = 0; i < 4; i++) {
            TestBoard.inGraveyard(game, you, "Forest");
        }
        assertTrue(Meld.bind(game).isEmpty());
    }

    /** Titania's meld trigger checks four land cards in its controller's graveyard ({@code CheckSVar$ X}). */
    @Test
    void aMeldTriggersSVarCheckIsRead() {
        Card titania = permanent(you, "Titania, Voice of Gaea");
        Trigger meld = titania.getTriggers().stream()
                .filter(trigger -> trigger.getMode() == TriggerType.Phase).findFirst().orElseThrow();
        for (int i = 0; i < 3; i++) {
            TestBoard.inGraveyard(game, you, "Forest");
        }
        assertFalse(Melds.svarsHold(meld));
        TestBoard.inGraveyard(game, you, "Forest");
        assertTrue(Melds.svarsHold(meld));
    }

    // ── split cards ─────────────────────────────────────────────────────

    /**
     * Life // Death over a Forest and two Swamps, Grizzly Bears in the
     * graveyard: both halves are castable, and 614.17d's forced cast is
     * Death, the half that reanimates.
     */
    @Test
    void aSplitCardWithOneMatchingHalfCastsThatHalf() {
        permanent(you, "Forest");
        permanents(you, "Swamp", 2);
        Card lifeDeath = TestBoard.inHand(game, you, "Life // Death");
        TestBoard.inGraveyard(game, you, "Grizzly Bears");
        Binding binding = CantEnterBattlefield.bind(game).orElseThrow();
        assertEquals(lifeDeath, binding.card("card"));

        Application cage = CantEnterBattlefield.rule().variants().get(CantEnterBattlefield.CAGE);
        List<SpellAbility> chosen = ForcedPlayRun.start(cage, binding, game).choose(you);
        assertNotNull(chosen, "the forced cast has a playable line");
        assertTrue(CantEnterBattlefield.reanimates(chosen.get(0)),
                "the forced cast is Death, not Life: " + chosen.get(0));
    }
}
