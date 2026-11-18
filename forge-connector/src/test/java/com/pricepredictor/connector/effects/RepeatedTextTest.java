package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeEnvironmentInitializer;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.nio.file.Path;
import java.util.List;
import java.util.function.Predicate;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The converter's one duplicate rule on real cards: a sentence a line already
 * states is not appended to it again, a repeat the card's script writes
 * stays, and a line dropped as another's duplicate leaves its keys on the line
 * that stays.
 */
@ExtendWith(ForgeExtension.class)
class RepeatedTextTest {

    private static final Path CARDS = ForgeEnvironmentInitializer.findCardsFolder();

    private final RulesParser parser = new RulesParser();

    private ConvertedScript convert(String relativePath) {
        return ConvertedScript.fromFile(parser, CARDS, relativePath);
    }

    private static ProvenanceKey key(String relativePath, TraitKind kind, int index) {
        return new ProvenanceKey("cardsfolder/" + relativePath, 0, kind, index);
    }

    private static ProvenanceSidecar.Line line(ConvertedScript card, Predicate<String> kind) {
        return card.sidecar().lines().stream()
                .filter(l -> kind.test(l.lineKind())).findFirst().orElseThrow();
    }

    @Test
    void aSubAbilityRestatingTheSpellsDescriptionIsPrintedOnce() {
        // Chill of the Grave's DBPump says again "It doesn't untap ..." of its root's description.
        List<String> lines = convert("c/chill_of_the_grave.txt").texts();
        assertEquals("spell[1]: tap target creature. it doesn't untap during its controller's"
                + " next untap step. draw a card.", lines.get(3));
    }

    @Test
    void anAdditionalCostTheChainRestatesIsPrintedOnce() {
        ConvertedScript mask = convert("m/mask_of_the_mimic.txt");
        String cost = "as an additional cost to cast this spell, sacrifice a creature.";
        assertEquals("spell[1]: " + cost + " search your library for a card with the same name"
                + " as target nontoken creature, put that card onto the battlefield, then shuffle.",
                mask.texts().get(3));
        List<RoleSpans.Span> spans = line(mask, "spell"::equals).roleSpans();
        assertEquals(new RoleSpans.Span(0, cost.length(), SpanRole.COST), spans.get(0));
    }

    @Test
    void aSpellStatingOnlyItsAdditionalCostKeepsItsKeyedLine() {
        // Warlord's Elite's spell description is its additional cost, which the line opens with.
        String path = "w/warlords_elite.txt";
        ConvertedScript elite = convert(path);
        String cost = "as an additional cost to cast this spell, tap two untapped artifacts,"
                + " creatures, and/or lands you control.";
        assertEquals("spell[1]: " + cost, elite.texts().get(4));
        ProvenanceSidecar.Line spell = line(elite, "spell"::equals);
        assertEquals(List.of(key(path, TraitKind.SPELL, 0)), spell.provenance());
        assertEquals(List.of(new RoleSpans.Span(0, cost.length(), SpanRole.COST)), spell.roleSpans());
    }

    @Test
    void aRepeatTheScriptWritesStays() {
        assertEquals("spell[1]: put a +1/+1 counter on target creature. put a +1/+1 counter on"
                + " target creature.", convert("c/common_bond.txt").texts().get(3));
        String pump = "target creature gets +3/+3 until end of turn.";
        assertEquals("spell[1]: " + pump + " " + pump + " " + pump,
                convert("b/bounty_of_might.txt").texts().get(3));
    }

    @Test
    void aRestatedSecondaryTriggersKeysStayOnTheLineThatStatesIt() {
        String queen = "c/coffin_queen.txt";
        assertEquals(List.of(key(queen, TraitKind.SPELL, 1), key(queen, TraitKind.TRIGGER, 0),
                        key(queen, TraitKind.TRIGGER, 1), key(queen, TraitKind.TRIGGER, 2)),
                line(convert(queen), k -> k.startsWith("activated")).provenance());
        String wellspring = "w/wellspring.txt";
        assertEquals(List.of(key(wellspring, TraitKind.TRIGGER, 0),
                        key(wellspring, TraitKind.TRIGGER, 1)),
                line(convert(wellspring), "triggered"::equals).provenance());
    }

    @Test
    void aClassLevelOneLineKeepsTheKeyOfTheTraitItRetypes() {
        String path = "a/advanced_floral_invocations.txt";
        ConvertedScript invocations = convert(path);
        assertEquals("level[1]: landfall — whenever a land you control enters, mill two cards.",
                invocations.texts().get(3));
        assertEquals(List.of(key(path, TraitKind.TRIGGER, 0)),
                line(invocations, "level"::equals).provenance());
    }

    @Test
    void aChoiceOfAMergedTriggerIsKeyedUnderBothTriggers() {
        // Appa's "enters or attacks" trigger is two triggers on one line.
        String path = "a/appa_loyal_sky_bison.txt";
        List<ProvenanceKey> first = line(convert(path), "option"::equals).provenance();
        assertEquals(List.of(key(path, TraitKind.TRIGGER, 0).withOption(0),
                key(path, TraitKind.TRIGGER, 1).withOption(0)), first);
    }
}
