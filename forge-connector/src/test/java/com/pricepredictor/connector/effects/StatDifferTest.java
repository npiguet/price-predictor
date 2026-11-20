package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.card.CardType;
import forge.game.card.Card;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** What a change to a card's computed characteristics reports. */
@ExtendWith(ForgeExtension.class)
class StatDifferTest {

    @Test
    void aFirstReadReportsNothing() {
        Card bears = TestCards.build("Grizzly Bears");
        assertTrue(new StatDiffer().diff(bears).isEmpty());
    }

    @Test
    void aTypeChangeNamesWhatJoinedAndWhatLeftTheTypeLine() {
        Card bears = TestCards.build("Grizzly Bears");
        StatDiffer differ = new StatDiffer();
        differ.prime(bears);
        bears.addChangedCardTypes(
                new CardType(List.of("Legendary", "Artifact", "Golem"), true),
                new CardType(List.of("Bear"), true),
                false, Set.of(), 1L, 0L, false, false);

        List<EffectEvent> events = differ.diff(bears);

        assertEquals(1, events.size());
        // Core, then super, then sub: the order the type line flattens in.
        EffectEvent event = events.get(0);
        assertEquals(EventType.TYPE_CHANGE, event.type());
        assertEquals(List.of(Refs.of(bears)), event.subjects());
        assertEquals(
                Map.of("types_added", List.of("artifact", "legendary", "golem"),
                        "types_removed", List.of("bear")),
                event.params());
        assertEquals(List.of("types_added", "types_removed"), List.copyOf(event.params().keySet()));
    }

    @Test
    void aColourChangeNamesTheColoursGainedAndLost() {
        Card bears = TestCards.build("Grizzly Bears");
        StatDiffer differ = new StatDiffer();
        differ.prime(bears);
        bears.setColor("blue", "red");

        List<EffectEvent> events = differ.diff(bears);

        assertEquals(1, events.size());
        EffectEvent event = events.get(0);
        assertEquals(EventType.COLOR_CHANGE, event.type());
        assertEquals(List.of(Refs.of(bears)), event.subjects());
        assertEquals(
                Map.of("colors", List.of("U", "R"), "colors_removed", List.of("G")),
                event.params());
        assertEquals(List.of("colors", "colors_removed"), List.copyOf(event.params().keySet()));
    }
    @Test
    void aPumpMovesThePowerButNotTheBase() {
        Card bears = TestCards.build("Grizzly Bears");
        StatDiffer differ = new StatDiffer();
        differ.prime(bears);
        bears.addPTBoost(2, 2, 1L, 0L);

        List<EffectEvent> events = differ.diff(bears);

        assertEquals(1, events.size());
        assertEquals(EventType.PT_CHANGE, events.get(0).type());
        assertEquals(Map.of("power_delta", 2, "toughness_delta", 2), events.get(0).params());
    }

    /**
     * A setting effect (layer 7b) names the base it set, as the snapshot's
     * {@code base_pt} spells it, beside the change it made to the total.
     */
    @Test
    void aSettingEffectNamesTheBaseItSet() {
        Card bears = TestCards.build("Grizzly Bears");
        StatDiffer differ = new StatDiffer();
        differ.prime(bears);
        bears.addNewPT(5, 5, 1L, 0L);

        List<EffectEvent> events = differ.diff(bears);

        assertEquals(1, events.size());
        assertEquals(EventType.PT_CHANGE, events.get(0).type());
        assertEquals(
                Map.of("power_delta", 3, "toughness_delta", 3, "set_pt", List.of(5, 5)),
                events.get(0).params());
    }

    /** A new base and a boost that cancel out still moved the base. */
    @Test
    void aBaseThatMovesUnderAnUnchangedTotalStillReportsTheBase() {
        Card bears = TestCards.build("Grizzly Bears");
        bears.addPTBoost(2, 2, 1L, 0L);
        StatDiffer differ = new StatDiffer();
        differ.prime(bears);
        bears.addNewPT(4, 4, 2L, 0L);
        bears.removePTBoost(1L, 0L);

        List<EffectEvent> events = differ.diff(bears);

        assertEquals(1, events.size());
        assertEquals(
                Map.of("power_delta", 0, "toughness_delta", 0, "set_pt", List.of(4, 4)),
                events.get(0).params());
    }

    // ── keywords, hidden keywords, abilities, control ───────────────────

    /** Keywords gained are listed sorted, the same on every JVM (V2 #16). */
    @Test
    void gainedKeywordsAreSorted() {
        forge.game.Game game = TestBoard.startedTwoPlayerGame(forge.game.phase.PhaseType.MAIN1);
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(0), "Grizzly Bears");
        StatDiffer differ = new StatDiffer();
        differ.prime(bears);
        bears.addChangedCardKeywords(List.of("Vigilance", "Flying", "Trample"), List.of(), false,
                game.getNextTimestamp(), null);

        EffectEvent event = only(differ.diff(bears), EventType.KEYWORD_CHANGE);
        assertEquals(List.of("flying", "trample", "vigilance"), event.param(EventParam.KEYWORDS));
        assertEquals(false, event.param(EventParam.REMOVED));
    }

    /** A second instance of a stacking keyword is a keyword gained (C2 #7). */
    @Test
    void aSecondInstanceOfAKeywordIsAChange() {
        forge.game.Game game = TestBoard.startedTwoPlayerGame(forge.game.phase.PhaseType.MAIN1);
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(0), "Grizzly Bears");
        bears.addChangedCardKeywords(List.of("Bushido:1"), List.of(), false,
                game.getNextTimestamp(), null);
        StatDiffer differ = new StatDiffer();
        differ.prime(bears);
        bears.addChangedCardKeywords(List.of("Bushido:1"), List.of(), false,
                game.getNextTimestamp(), null);

        EffectEvent event = only(differ.diff(bears), EventType.KEYWORD_CHANGE);
        assertEquals(List.of("bushido:1"), event.param(EventParam.KEYWORDS));
    }

    /** A hidden keyword is a keyword change of its own, flagged hidden (X1 #10). */
    @Test
    void aHiddenKeywordIsAHiddenKeywordChange() {
        forge.game.Game game = TestBoard.startedTwoPlayerGame(forge.game.phase.PhaseType.MAIN1);
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(0), "Grizzly Bears");
        StatDiffer differ = new StatDiffer();
        differ.prime(bears);
        long timestamp = game.getNextTimestamp();
        bears.addHiddenExtrinsicKeywords(timestamp, 0, List.of("CARDNAME can't block."));

        EffectEvent gained = only(differ.diff(bears), EventType.KEYWORD_CHANGE);
        assertEquals(Map.of("keywords", List.of("CARDNAME can't block."), "removed", false,
                "hidden", true), gained.params());

        bears.removeHiddenExtrinsicKeywords(timestamp, 0);
        EffectEvent lost = only(differ.diff(bears), EventType.KEYWORD_CHANGE);
        assertEquals(true, lost.param(EventParam.REMOVED));
        assertEquals(true, lost.param(EventParam.HIDDEN));
    }

    /**
     * An ability a static grants is an ability change naming its line, and
     * one taken away an ability change marked removed (C2 #7).
     */
    @Test
    void anAbilityGrantedAndTakenAwayIsAnAbilityChange() {
        forge.game.Game game = TestBoard.startedTwoPlayerGame(forge.game.phase.PhaseType.MAIN1);
        forge.game.player.Player you = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, you, "Grizzly Bears");
        StatDiffer differ = new StatDiffer();
        differ.prime(bears);
        Card rite = TestBoard.inPlay(game, you, "Cryptolith Rite");
        game.getAction().checkStaticAbilities();

        EffectEvent gained = only(differ.diff(bears), EventType.ABILITY_CHANGE);
        assertEquals(false, gained.param(EventParam.REMOVED));
        assertTrue(String.valueOf(gained.param(EventParam.ABILITIES))
                .startsWith("(cardsfolder/c/cryptolith_rite.txt)"), gained.params().toString());

        game.getAction().moveToGraveyard(rite, null);
        game.getAction().checkStaticAbilities();
        EffectEvent lost = only(differ.diff(bears), EventType.ABILITY_CHANGE);
        assertEquals(Map.of("abilities", "", "removed", true), lost.params());
    }

    /** Re-applying the same static changes nothing: abilities are read by their text. */
    @Test
    void aStaticReappliedIsNoAbilityChange() {
        forge.game.Game game = TestBoard.startedTwoPlayerGame(forge.game.phase.PhaseType.MAIN1);
        forge.game.player.Player you = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, you, "Grizzly Bears");
        TestBoard.inPlay(game, you, "Cryptolith Rite");
        game.getAction().checkStaticAbilities();
        StatDiffer differ = new StatDiffer();
        differ.prime(bears);
        game.getAction().checkStaticAbilities();

        assertTrue(differ.diff(bears).isEmpty());
    }

    /**
     * A forgotten card is a new object: its next reading is a first reading
     * and reports nothing, however it changed while forgotten; one not
     * forgotten reports the change.
     */
    @Test
    void aForgottenCardsNextReadingIsAFirstReading() {
        forge.game.Game game = TestBoard.startedTwoPlayerGame(forge.game.phase.PhaseType.MAIN1);
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(0), "Grizzly Bears");
        Card other = TestBoard.inPlay(game, game.getPlayers().get(0), "Hill Giant");
        StatDiffer differ = new StatDiffer();
        differ.prime(bears);
        differ.prime(other);

        differ.forget(bears.getId());
        for (Card card : java.util.List.of(bears, other)) {
            card.addPTBoost(2, 2, game.getNextTimestamp(), 0);
        }

        assertTrue(differ.diff(bears).isEmpty(), "forgotten: a first reading");
        assertTrue(!differ.diff(other).isEmpty(), "remembered: the pump is a change");
        assertTrue(differ.diff(other).isEmpty(), "and is reported once");
    }

    /**
     * A rebaseline returns what changed since the last reading, unreported
     * so far, and measures from the new reading on: the same change is not
     * reported again.
     */
    @Test
    void aRebaselineReturnsTheSwallowedChangesAndMeasuresFromThere() {
        forge.game.Game game = TestBoard.startedTwoPlayerGame(forge.game.phase.PhaseType.MAIN1);
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(0), "Grizzly Bears");
        StatDiffer differ = new StatDiffer();
        differ.prime(bears);
        bears.addPTBoost(1, 1, game.getNextTimestamp(), 0);

        java.util.List<EffectEvent> swallowed = differ.rebaseline(java.util.List.of(bears));

        assertTrue(!swallowed.isEmpty(), "the pump the snapshot already shows");
        assertTrue(differ.diff(bears).isEmpty(), "measured from the rebaseline");
    }

    /** A control change names the new controller (TC2 #9). */
    @Test
    void aControlChangeNamesTheNewController() {
        forge.game.Game game = TestBoard.startedTwoPlayerGame(forge.game.phase.PhaseType.MAIN1);
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(0), "Grizzly Bears");
        StatDiffer differ = new StatDiffer();
        differ.prime(bears);
        forge.game.player.Player thief = game.getPlayers().get(1);
        bears.addTempController(thief, game.getNextTimestamp());

        EffectEvent event = only(differ.diff(bears), EventType.CONTROL_CHANGE);
        assertEquals(Refs.of(thief), event.param(EventParam.CONTROLLER));
    }

    /** An Animate's Name$ is a name change, naming the name now (follow-up item 10). */
    @Test
    void aNameAnEffectGivesIsANameChange() {
        forge.game.Game game = TestBoard.startedTwoPlayerGame(forge.game.phase.PhaseType.MAIN1);
        forge.game.player.Player you = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, you, "Grizzly Bears");
        StatDiffer differ = new StatDiffer();
        differ.prime(bears);
        forge.game.spellability.SpellAbility rename = forge.game.ability.AbilityFactory.getAbility(
                "AB$ Animate | Cost$ 0 | Defined$ Self | Name$ Pigeon | Duration$ Permanent", bears);
        rename.setActivatingPlayer(you);
        rename.resolve();

        EffectEvent event = only(differ.diff(bears), EventType.NAME_CHANGE);
        assertEquals(Map.of("name", "Pigeon"), event.params());
    }

    /** A mana cost an effect sets is a mana cost change: the cost now and its value. */
    @Test
    void aManaCostAnEffectSetsIsAManaCostChange() {
        forge.game.Game game = TestBoard.startedTwoPlayerGame(forge.game.phase.PhaseType.MAIN1);
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(0), "Grizzly Bears");
        StatDiffer differ = new StatDiffer();
        differ.prime(bears);
        bears.addChangedManaCost(new forge.card.mana.ManaCost(
                new forge.card.mana.ManaCostParser("5 G")), false, game.getNextTimestamp(), 0L);

        EffectEvent event = only(differ.diff(bears), EventType.MANA_COST_CHANGE);
        assertEquals(Map.of("generic", 5, "G", 1), event.param(EventParam.MANA_COST));
        assertEquals(6, event.param(EventParam.MANA_VALUE));
    }

    /** A transform's new name is the face change's, not a name change of its own. */
    @Test
    void aFaceChangesNameIsNoNameChange() {
        forge.game.Game game = TestBoard.startedTwoPlayerGame(forge.game.phase.PhaseType.MAIN1);
        Card avacyn = TestBoard.inPlay(game, game.getPlayers().get(0), "Archangel Avacyn");
        StatDiffer differ = new StatDiffer();
        differ.prime(avacyn);
        avacyn.changeCardState("Transform", null, null);

        assertTrue(differ.diff(avacyn).stream().noneMatch(e -> e.type() == EventType.NAME_CHANGE
                || e.type() == EventType.MANA_COST_CHANGE));
    }

    private static EffectEvent only(List<EffectEvent> events, EventType type) {
        List<EffectEvent> found = events.stream().filter(e -> e.type() == type).toList();
        assertEquals(1, found.size(), events.toString());
        return found.get(0);
    }
}
