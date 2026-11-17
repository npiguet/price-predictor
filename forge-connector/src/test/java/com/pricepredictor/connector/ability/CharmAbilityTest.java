package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.ForgeEnvironmentInitializer;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RenderedLine;
import com.pricepredictor.connector.RulesParser;
import com.pricepredictor.connector.effects.ProvenanceKey;
import com.pricepredictor.connector.effects.ProvenanceSidecar;
import com.pricepredictor.connector.effects.RoleSpans;
import com.pricepredictor.connector.effects.SpanRole;
import com.pricepredictor.connector.effects.TraitKind;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * How a charm prints: its root line, its cost and its modes.
 *
 * <p>A mode's additional cost, as the card prints it: Forge writes a cost
 * space-separated ({@code ModeCost$ 3 W W}) and the converted text is read by
 * a tokenizer for which braces delimit one atom. One pair of braces around the
 * whole cost would therefore produce a distinct vocabulary entry per
 * multi-shard mode cost — {@code {3 W W}}, {@code {B B}}, {@code {5 U}} — each
 * unrelated to the {@code {3}} and {@code {W}} every other cost in the corpus
 * is spelled with, and each seen a handful of times.
 *
 * <p>A charm's root renders by what the charm is: an activated charm is an
 * {@code activated} line opening with its cost, and a trigger whose execute is
 * a charm prints one {@code option} line per mode whether or not its
 * description holds the {@code ABILITY} placeholder.
 */
@ExtendWith(ForgeExtension.class)
class CharmAbilityTest {

    private static final Path CARDS_FOLDER = ForgeEnvironmentInitializer.findCardsFolder();

    @Test
    void aMultiShardCostBecomesOneSymbolPerShard() {
        assertEquals("{3}{W}{W}", CharmAbility.manaCostSymbols("3 W W"));
    }

    @Test
    void aSingleShardCostIsUnchanged() {
        // The common case: most cards using ModeCost have a one-shard cost.
        assertEquals("{1}", CharmAbility.manaCostSymbols("1"));
        assertEquals("{0}", CharmAbility.manaCostSymbols("0"));
    }

    @Test
    void aHybridShardStaysOneSymbol() {
        // The slash is inside a shard, not between two of them.
        assertEquals("{2}{W/U}", CharmAbility.manaCostSymbols("2 W/U"));
    }

    @Test
    void surroundingSpaceIsNotAShard() {
        assertEquals("{3}{W}", CharmAbility.manaCostSymbols("  3   W  "));
    }

    @Test
    void anActivatedCharmIsAnActivatedLineOpeningWithItsCost() {
        Converted bow = Converted.of("b/bow_of_nylea.txt");
        List<String> texts = bow.texts();
        int root = texts.indexOf("activated[1]: {1}{G}, {T}: choose one");
        assertTrue(root >= 0, "the charm's root states its cost: " + texts);
        assertEquals(List.of(
                "option[2]: put a +1/+1 counter on target creature.",
                "option[3]: CARDNAME deals 2 damage to target creature with flying.",
                "option[4]: you gain 3 life.",
                "option[5]: put up to four target cards from your graveyard"
                        + " on the bottom of your library in any order."),
                texts.subList(root + 1, root + 5));
    }

    @Test
    void anActivatedCharmsCostIsItsCostSpan() {
        ProvenanceSidecar.Line root = Converted.of("b/bow_of_nylea.txt").lineKind("activated").get(0);
        RoleSpans.Span cost = root.roleSpans().get(0);
        assertEquals(SpanRole.COST, cost.role(), "spans: " + root.roleSpans());
        assertEquals(0, cost.start());
        assertEquals("{1}{G}, {T}:".length(), cost.end());
    }

    @Test
    void aCharmRootWithAContinuingDescriptionOpensWithItsHeader() {
        // AdditionalDescription$ ". If there are…" continues the header Forge
        // writes before it; alone, it left the root a bare period.
        assertEquals("spell[1]: delirium — choose one. if there are four or more card types"
                        + " among cards in your graveyard, choose one or more instead.",
                Converted.of("l/lets_play_a_game.txt").texts().get(3));
        for (String card : List.of("l/lets_play_a_game.txt", "w/wail_of_the_forgotten.txt")) {
            List<String> prose = Converted.of(card).prose();
            assertTrue(prose.stream().noneMatch(text -> text.startsWith(".")),
                    card + " has a line opening with a period: " + prose);
        }
    }

    @Test
    void aTriggeredCharmWithoutThePlaceholderPrintsItsModes() {
        Converted riders = Converted.of("v/varchilds_war_riders.txt");
        assertEquals(List.of(
                "option[1]: have an opponent create a 1/1 red survivor creature token"
                        + " for each age counter on CARDNAME.",
                "option[2]: sacrifice CARDNAME."),
                riders.texts().stream().filter(t -> t.startsWith("option")).toList());
        assertEquals(List.of(Set.of("trigger 0 option 0"), Set.of("trigger 0 option 1")),
                riders.optionKeys());
    }

    @Test
    void aSharedCharmsModesAreKeyedUnderEveryTriggerItsLineCarries() {
        // Two triggers, one Execute$: one line, and each mode under both roots.
        Converted four = Converted.of("t/the_fantastic_four.txt");
        assertEquals(4, four.lineKind("option").size(), "modes: " + four.texts());
        assertEquals(List.of(
                Set.of("trigger 0 option 0", "trigger 1 option 0"),
                Set.of("trigger 0 option 1", "trigger 1 option 1"),
                Set.of("trigger 0 option 2", "trigger 1 option 2"),
                Set.of("trigger 0 option 3", "trigger 1 option 3")),
                four.optionKeys());
    }

    @Test
    void anActivatedCharmsHeaderStatesItsChoiceRestriction() {
        // Forge words ChoiceRestriction$ ThisTurn "that hasn't been chosen this turn".
        assertEquals(List.of("activated[1]: {1}: choose one that hasn't been chosen this turn"),
                Converted.of("k/kargan_intimidator.txt").texts().stream()
                        .filter(t -> t.startsWith("activated")).toList());
    }

    @Test
    void aRootsAdditionalDescriptionFollowsTheChooseHeaderWithoutALeadingPeriod() {
        assertEquals("spell[1]: choose one. if you control a commander as you cast"
                        + " this spell, you may choose both.",
                rootOf("k/klauths_will.txt"));
        assertEquals("spell[1]: choose one. if you control a commander as you cast"
                        + " this spell, you may choose both.",
                rootOf("j/jeskas_will.txt"));
        assertEquals("spell[1]: choose one. if you control a commander as you cast"
                        + " this spell, you may choose both.",
                rootOf("n/nexus_mentality.txt"));
        assertEquals("spell[1]: choose one. if you control a nonland permanent you don't own"
                        + " as you cast this spell, you may choose two instead.",
                rootOf("w/weave_the_nightmare.txt"));
    }

    @Test
    void aTieredCharmsDescriptionStandsAloneAsForgeWritesIt() {
        // Forge's own description of a Tiered charm has no "Choose" header.
        assertTrue(rootOf("v/vincents_limit_break.txt").startsWith("spell[1]: until end of turn"),
                rootOf("v/vincents_limit_break.txt"));
    }

    @Test
    void aChapterCharmPrintsOneKeyedOptionLinePerMode() {
        Converted sisters = Converted.of("s/summon_magus_sisters.txt");
        List<String> modes = sisters.texts().stream().filter(t -> t.startsWith("option")).toList();
        assertEquals(3, modes.size(), "modes: " + sisters.texts());
        assertEquals(3, sisters.optionKeys().size());
        for (Set<String> keys : sisters.optionKeys()) {
            assertTrue(keys.stream().anyMatch(k -> k.matches(".* option [0-2]")), keys.toString());
        }
    }

    @Test
    void aModalSpellWithNoDescriptionStillPrintsItsRootKeyedWithTheSpell() {
        Converted doomsday = Converted.of("d/doomsday_confluence.txt");
        assertTrue(doomsday.texts().contains("spell[1]: choose one"), "root: " + doomsday.texts());
        assertEquals(3, doomsday.lineKind("option").size());
        assertTrue(doomsday.sidecar().droppedKeys().isEmpty(),
                "dropped: " + doomsday.sidecar().droppedKeys());
    }

    private static String rootOf(String card) {
        return Converted.of(card).texts().stream().filter(t -> t.startsWith("spell")).findFirst().orElseThrow();
    }

    /** A real Forge script converted with its sidecar, as BatchConverter writes both. */
    private record Converted(List<RenderedLine> rendered, ProvenanceSidecar sidecar) {

        static Converted of(String relativePath) {
            String scriptFile = "cardsfolder/" + relativePath;
            RulesParser.ParsedCard parsed = new RulesParser().parseScript(lines(relativePath), scriptFile);
            List<RenderedLine> rendered = parsed.card().renderLines();
            return new Converted(rendered, ProvenanceSidecar.build(
                    parsed.card().faces().get(0).name(), scriptFile, rendered, parsed.recorders()));
        }

        private static List<String> lines(String relativePath) {
            try {
                return Files.readAllLines(CARDS_FOLDER.resolve(relativePath));
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        List<String> texts() {
            return rendered.stream().map(RenderedLine::text).toList();
        }

        /** Each line's prose, after its {@code kind[n]: } prefix. */
        List<String> prose() {
            return rendered.stream().map(RenderedLine::prose).toList();
        }

        List<ProvenanceSidecar.Line> lineKind(String kind) {
            return sidecar.lines().stream().filter(line -> line.lineKind().equals(kind)).toList();
        }

        /** Each option line's keys, as {@code trigger 0 option 1}. */
        List<Set<String>> optionKeys() {
            return lineKind("option").stream()
                    .map(line -> line.provenance().stream().map(Converted::shortKey)
                            .collect(Collectors.toSet()))
                    .toList();
        }

        private static String shortKey(ProvenanceKey key) {
            String root = (key.traitKind() == TraitKind.TRIGGER ? "trigger" : key.traitKind().toString())
                    + " " + key.indexWithinKind();
            return key.option() == null ? root : root + " option " + key.option();
        }
    }
}
