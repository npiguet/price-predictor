package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.LayerContributions.Contribution;
import forge.card.CardChangedType;
import forge.card.CardType;
import forge.card.ColorSet;
import forge.card.RemoveType;
import forge.card.StateChangedType;
import forge.card.WordChangedType;
import forge.game.card.Card;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** What a continuous contribution carries, read off one layer table entry at a time. */
@ExtendWith(ForgeExtension.class)
class LayerContributionsTest {

    // ── continuous type and colour channels ─────────────────────────────

    /**
     * A stand-in for the patch's own {@code Card.CardColor}.
     *
     * <p>The real one is nested in {@code Card} and only exists as a public type
     * on a patched checkout, which this module does not compile against. The
     * collector reads both components reflectively, so any public record with
     * the same two accessors exercises the same path — and a test that named the
     * real type would not compile on a stock checkout.
     */
    public record ColorChange(ColorSet color, boolean additional) {
    }

    private static String tokens(Contribution into, String field) {
        String json = into.toJson();
        int at = json.indexOf("\"" + field + "\":[");
        return json.substring(at + field.length() + 4, json.indexOf(']', at));
    }

    @Test
    void anAddedTypeIsBareAndARemovedOneIsDashed() {
        Contribution into = new Contribution("E1");
        LayerContributions.typeTokens(new CardChangedType(
                new CardType(List.of("Creature", "Elf"), true),
                new CardType(List.of("Land"), true),
                false, Set.of()), into);
        assertEquals("\"creature\",\"elf\",\"-land\"", tokens(into, "types"));
    }

    @Test
    void removingAWholeClassOfTypeIsNamedForTheClass() {
        // "loses all creature types" names no type, so it cannot be spelled as
        // a list of removals.
        Contribution into = new Contribution("E1");
        LayerContributions.typeTokens(new CardChangedType(
                null, null, true,
                Set.of(RemoveType.CreatureTypes, RemoveType.SuperTypes)), into);
        assertEquals(
                Set.of("\"all-creature-types\"", "\"-all-creature-types\"",
                        "\"-all-super-types\""),
                Set.of(tokens(into, "types").split(",")));
    }

    @Test
    void aStateChangeMarksTheTypeLineAsSetRatherThanAdded() {
        Contribution into = new Contribution("E1");
        LayerContributions.typeTokens(
                new StateChangedType(new CardType(List.of("Land"), true)), into);
        assertEquals("\"=\",\"land\"", tokens(into, "types"));
    }

    @Test
    void aTextChangeIsARemovalAndAnAdditionInOneEntry() {
        Contribution into = new Contribution("E1");
        LayerContributions.typeTokens(new WordChangedType("Forest", "Island"), into);
        assertEquals("\"-forest\",\"island\"", tokens(into, "types"));
    }

    @Test
    void anAdditionalColourIsJustTheLetter() {
        Contribution into = new Contribution("E1");
        LayerContributions.colorTokens(
                new ColorChange(ColorSet.fromNames("green"), true), into);
        assertEquals("\"G\"", tokens(into, "colors"));
    }

    @Test
    void aReplacingColourIsMarked() {
        Contribution into = new Contribution("E1");
        LayerContributions.colorTokens(
                new ColorChange(ColorSet.fromNames("green"), false), into);
        assertEquals("\"=\",\"G\"", tokens(into, "colors"));
    }

    @Test
    void replacingTheColourWithNothingIsColourless() {
        // An empty colour set and an empty token list are different answers:
        // one turns the permanent colourless, the other says nothing happened.
        Contribution into = new Contribution("E1");
        LayerContributions.colorTokens(
                new ColorChange(ColorSet.fromMask(0), false), into);
        assertEquals("\"=\",\"C\"", tokens(into, "colors"));
    }

    @Test
    void anUnreadableColourEntryContributesNothingRatherThanAReplacement() {
        // Guessing here would report every colour change as an overwrite.
        Contribution into = new Contribution("E1");
        LayerContributions.colorTokens(new Object(), into);
        assertEquals("", tokens(into, "colors"));
    }

    @Test
    void aStaticThatWritesTheSameTypeTwiceSaysItOnce() {
        // One static can write to more than one type layer.
        Contribution into = new Contribution("E1");
        CardChangedType change = new CardChangedType(
                new CardType(List.of("Creature"), true), null, false, Set.of());
        LayerContributions.typeTokens(change, into);
        LayerContributions.typeTokens(change, into);
        assertEquals("\"creature\"", tokens(into, "types"));
    }

    // ── continuous name channel ──────────────────────────────────────────

    /** {@code "name"} is the last field {@link Contribution#toJson()} writes. */
    private static String nameField(Contribution into) {
        String json = into.toJson();
        int at = json.indexOf("\"name\":");
        return json.substring(at + "\"name\":".length(), json.length() - 1);
    }

    /**
     * {@code contributions[].name} carries the name a static gives a card;
     * without this channel the declared field never carries a value.
     *
     * <p>Goes through the real {@code Card.addChangedName} and the real,
     * reflectively-read {@code Card.getChangedCardNames} rather than a
     * hand-built stand-in, because the defect this guards against is exactly
     * the gap between "compiles against a plausible shape" and "the engine
     * ever produces this".
     *
     * <p>Two competing entries, not one: with a single rename in the table,
     * {@code card.getName()} -- which walks every entry and returns the last
     * overwrite -- answers the same string as that one static's own cell, so
     * a {@code contribution(...).name(card.getName())} implementation
     * (whole-card, not this cell) would pass unnoticed. The later static
     * ({@code winningStatic}) is what {@code getName()} reports; asserting
     * that the earlier one ({@code supersededStatic}) still carries its own,
     * superseded name is what only a per-cell read can produce. This also
     * covers two statics renaming the same card, which otherwise had no test
     * at all.
     */
    @Test
    void aChangedNameContributesTheNewName() {
        Card card = TestCards.build("Grizzly Bears");
        long supersededStatic = 1111L;
        long winningStatic = 2222L;
        card.addChangedName("Superseded-By-Later-Static", false, 3L, supersededStatic);
        card.addChangedName("Winning-Name", false, 7L, winningStatic);
        // The premise the whole-card shortcut would rely on, and get wrong
        // for the superseded static below.
        assertEquals("Winning-Name", card.getName());

        LayerContributions contributions = new LayerContributions();
        contributions.addNames(card, "E1");
        Map<Long, Map<String, Contribution>> byStatic = contributions.byStatic();

        assertEquals("\"Superseded-By-Later-Static\"",
                nameField(byStatic.get(supersededStatic).get("E1")));
        assertEquals("\"Winning-Name\"",
                nameField(byStatic.get(winningStatic).get("E1")));
    }

    /**
     * The channel's other half: most statics contribute no name, and null is
     * the correct, meaningful answer there -- not an empty string, and not a
     * dropped key.
     */
    @Test
    void aCardWithNoNameChangeRendersNullName() {
        Card card = TestCards.build("Grizzly Bears");
        LayerContributions contributions = new LayerContributions();
        contributions.addNames(card, "E1");
        Map<Long, Map<String, Contribution>> byStatic = contributions.byStatic();
        assertTrue(byStatic.isEmpty(), "no static wrote a name, so nothing is filed");
        assertEquals("null", nameField(new Contribution("E1")));
    }

    @Test
    void anUnreadableNameEntryContributesNoName() {
        // Guessing here would report every unrelated object as a rename.
        Contribution into = new Contribution("E1");
        LayerContributions.nameTokens(new Object(), into);
        assertEquals("null", nameField(into));
    }
}
