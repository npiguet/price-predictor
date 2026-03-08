package com.pricepredictor.connector;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for CardScriptConverter using Forge's CardRules.Reader.
 * Requires the full Forge transitive dependency chain on the classpath.
 * Run via: mvn test -Pintegration
 */
@Tag("integration")
@ExtendWith(ForgeExtension.class)
class CardScriptConverterTest {

    private final CardScriptConverter converter = new CardScriptConverter();

    private MultiCard convert(String... lines) {
        return converter.convertCard(Arrays.asList(lines), "test.txt");
    }

    // --- US1: Basic card conversion ---

    @Test
    void vanillaCreature() {
        MultiCard result = convert(
                "Name:Grizzly Bears",
                "ManaCost:1 G",
                "Types:Creature Bear",
                "PT:2/2",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        assertEquals("grizzly bears", card.name());
        assertTrue(card.manaCost().contains("{G}"));
        assertEquals("creature bear", card.types());
        assertEquals("2/2", card.powerToughness());
        assertTrue(card.abilities().isEmpty());
        assertNull(card.loyalty());
    }

    @Test
    void passiveKeywords() {
        MultiCard result = convert(
                "Name:Serra Angel",
                "ManaCost:3 W W",
                "Types:Creature Angel",
                "PT:4/4",
                "K:Flying",
                "K:Vigilance",
                "Oracle:Flying, vigilance");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> keywords = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.KEYWORD_PASSIVE).toList();
        assertEquals(2, keywords.size());
        assertEquals("keyword: flying", keywords.get(0).formatLine());
        assertEquals("keyword: vigilance", keywords.get(1).formatLine());
        assertNull(keywords.get(0).actionNumber());
    }

    @Test
    void activatedAbility() {
        MultiCard result = convert(
                "Name:Llanowar Elves",
                "ManaCost:G",
                "Types:Creature Elf Druid",
                "PT:1/1",
                "A:AB$ Mana | Cost$ T | Produced$ G | SpellDescription$ Add {G}.",
                "Oracle:{T}: Add {G}.");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> activated = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ACTIVATED).toList();
        assertEquals(1, activated.size());
        String line = activated.get(0).formatLine();
        assertTrue(line.startsWith("activated[1]:"), "Expected activated[1]: but got: " + line);
        assertTrue(line.contains("{T}"), "Expected {T} in: " + line);
        assertTrue(line.contains("add {G}"), "Expected 'add {G}' in: " + line);
    }

    @Test
    void triggeredAbility() {
        MultiCard result = convert(
                "Name:Thragtusk",
                "ManaCost:4 G",
                "Types:Creature Beast",
                "PT:5/3",
                "T:Mode$ ChangesZone | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ Life | TriggerDescription$ When CARDNAME enters, you gain 5 life.",
                "SVar:Life:DB$ GainLife | Defined$ You | LifeAmount$ 5",
                "Oracle:When Thragtusk enters, you gain 5 life.");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> triggered = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.TRIGGERED).toList();
        assertEquals(1, triggered.size());
        String line = triggered.get(0).formatLine();
        assertTrue(line.startsWith("triggered:"));
        assertTrue(line.contains("CARDNAME"));
        assertTrue(line.contains("gain 5 life"));
    }

    @Test
    void staticAbility() {
        MultiCard result = convert(
                "Name:Blood Moon",
                "ManaCost:2 R",
                "Types:Enchantment",
                "S:Mode$ Continuous | Affected$ Land.nonBasic | AddType$ Mountain | RemoveLandTypes$ True | Description$ Nonbasic lands are Mountains.",
                "Oracle:Nonbasic lands are Mountains.");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> statics = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.STATIC).toList();
        assertEquals(1, statics.size());
        assertTrue(statics.get(0).formatLine().contains("nonbasic lands are mountains"));
    }

    @Test
    void replacementEffect() {
        MultiCard result = convert(
                "Name:Rest in Peace",
                "ManaCost:1 W",
                "Types:Enchantment",
                "R:Event$ Moved | ActiveZones$ Battlefield | Destination$ Graveyard | ValidCard$ Card | ReplaceWith$ Exile | Description$ If a card or token would be put into a graveyard from anywhere, exile it instead.",
                "SVar:Exile:DB$ ChangeZone | Origin$ All | Destination$ Exile",
                "Oracle:If a card or token would be put into a graveyard from anywhere, exile it instead.");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> replacements = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.REPLACEMENT).toList();
        assertEquals(1, replacements.size());
        assertTrue(replacements.get(0).formatLine().startsWith("replacement:"));
    }

    @Test
    void engineMetadataExcluded() {
        MultiCard result = convert(
                "Name:Test Card",
                "ManaCost:1",
                "Types:Creature Human",
                "PT:1/1",
                "AI:RemoveDeck:Random",
                "DeckHints:Ability$LifeGain",
                "SVar:AnotherSVar:DB$ Something",
                "Oracle:");
        String output = OutputFormatter.formatMultiCard(result);
        assertFalse(output.contains("AI:"));
        assertFalse(output.contains("DeckHints"));
        assertFalse(output.contains("SVar"));
    }

    @Test
    void actionCounterIncrementsAcrossMixedTypes() {
        MultiCard result = convert(
                "Name:Versatile Card",
                "ManaCost:2 W",
                "Types:Creature Human",
                "PT:2/2",
                "K:Cycling:2",
                "A:AB$ GainLife | Cost$ T | LifeAmount$ 1 | SpellDescription$ Gain 1 life.",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> actionable = card.abilities().stream()
                .filter(a -> a.actionNumber() != null).toList();
        assertTrue(actionable.size() >= 2, "Expected at least 2 actionable abilities");
        // Action numbers should be sequential
        for (int i = 0; i < actionable.size() - 1; i++) {
            assertTrue(actionable.get(i).actionNumber() < actionable.get(i + 1).actionNumber());
        }
    }

    @Test
    void textCasingCorrect() {
        MultiCard result = convert(
                "Name:Llanowar Elves",
                "ManaCost:G",
                "Types:Creature Elf Druid",
                "PT:1/1",
                "A:AB$ Mana | Cost$ T | Produced$ G | SpellDescription$ Add {G}.",
                "Oracle:{T}: Add {G}.");
        String output = OutputFormatter.formatMultiCard(result);
        assertTrue(output.contains("llanowar elves"), "Name should be lowercase");
        assertTrue(output.contains("{G}"), "Brace symbols should be uppercase");
        assertTrue(output.contains("{T}"), "Tap symbol should be uppercase");
    }

    @Test
    void variableXStaysUppercase() {
        // Standalone X in various positions
        assertEquals("-X: deal X damage", CardScriptConverter.applyTextCasing("-X: Deal X damage"));
        assertEquals("+X/+0 until end of turn", CardScriptConverter.applyTextCasing("+X/+0 until end of turn"));
        assertEquals("where X is the number", CardScriptConverter.applyTextCasing("Where X is the number"));

        // X inside words must stay lowercase
        assertEquals("exile target creature", CardScriptConverter.applyTextCasing("Exile target creature"));
        assertEquals("next end step", CardScriptConverter.applyTextCasing("Next end step"));
        assertEquals("tax each opponent", CardScriptConverter.applyTextCasing("Tax each opponent"));

        // X inside braces handled separately
        assertEquals("{X}{R}", CardScriptConverter.applyTextCasing("{X}{R}"));

        // Mixed: brace X and standalone X in same text
        assertEquals("pay {X}, where X is the number of counters",
                CardScriptConverter.applyTextCasing("Pay {X}, where X is the number of counters"));
    }

    @Test
    void noCostCardOmitsManaCostLine() {
        MultiCard result = convert(
                "Name:Ancestral Vision",
                "ManaCost:no cost",
                "Types:Sorcery",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        assertNull(card.manaCost());
        String output = OutputFormatter.formatCard(card);
        assertFalse(output.contains("mana cost:"));
    }

    @Test
    void textPropertyIncluded() {
        MultiCard result = convert(
                "Name:Test Card",
                "ManaCost:1",
                "Types:Creature Human",
                "PT:1/1",
                "Text:This is flavor text.",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        assertNotNull(card.text());
        String output = OutputFormatter.formatCard(card);
        assertTrue(output.contains("text: this is flavor text."));
    }

    // --- US2: Complex card types ---

    @Test
    void planeswalkerAbilities() {
        MultiCard result = convert(
                "Name:Jace Beleren",
                "ManaCost:1 U U",
                "Types:Legendary Planeswalker Jace",
                "Loyalty:3",
                "A:AB$ Draw | Cost$ AddCounter<2/LOYALTY> | Defined$ Player | NumCards$ 1 | Planeswalker$ True | SpellDescription$ Each player draws a card.",
                "A:AB$ Draw | Cost$ SubCounter<1/LOYALTY> | Defined$ Targeted | NumCards$ 1 | Planeswalker$ True | ValidTgts$ Player | SpellDescription$ Target player draws a card.",
                "A:AB$ Mill | Cost$ SubCounter<10/LOYALTY> | Defined$ Targeted | NumCards$ 20 | Planeswalker$ True | ValidTgts$ Player | SpellDescription$ Target player mills twenty cards.",
                "Oracle:+2: Each player draws a card.\\n-1: Target player draws a card.\\n-10: Target player mills twenty cards.");
        ConvertedCard card = result.faces().get(0);
        assertEquals("3", card.loyalty());
        List<AbilityLine> pwAbilities = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.PLANESWALKER).toList();
        assertEquals(3, pwAbilities.size());
        assertTrue(pwAbilities.get(0).formatLine().contains("[+2]:"), "Expected [+2]: but got: " + pwAbilities.get(0).formatLine());
        assertTrue(pwAbilities.get(1).formatLine().contains("[-1]:"), "Expected [-1]: but got: " + pwAbilities.get(1).formatLine());
        assertTrue(pwAbilities.get(2).formatLine().contains("[-10]:"), "Expected [-10]: but got: " + pwAbilities.get(2).formatLine());
        assertEquals(1, (int) pwAbilities.get(0).actionNumber());
        assertEquals(2, (int) pwAbilities.get(1).actionNumber());
        assertEquals(3, (int) pwAbilities.get(2).actionNumber());
    }

    @Test
    void sagaChapterAbilities() {
        MultiCard result = convert(
                "Name:The Eldest Reborn",
                "ManaCost:4 B",
                "Types:Enchantment Saga",
                "K:Chapter:3:ChI,ChII,ChIII",
                "SVar:ChI:DB$ Sacrifice | Defined$ OpponentNonTgtAP | SacValid$ Creature.OppCtrl,Planeswalker.OppCtrl | SpellDescription$ Each opponent sacrifices a creature or planeswalker.",
                "SVar:ChII:DB$ Discard | Defined$ OpponentNonTgtAP | NumCards$ 1 | SpellDescription$ Each opponent discards a card.",
                "SVar:ChIII:DB$ ChangeZone | Origin$ Graveyard | Destination$ Battlefield | ChangeType$ Creature.inYourYard,Planeswalker.inYourYard | SpellDescription$ Put target creature or planeswalker card from a graveyard onto the battlefield under your control.",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> chapters = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.CHAPTER).toList();
        assertEquals(3, chapters.size(), "Expected 3 chapter abilities but got: " + card.abilities());
        String ch1 = chapters.get(0).formatLine();
        String ch2 = chapters.get(1).formatLine();
        String ch3 = chapters.get(2).formatLine();
        assertTrue(ch1.startsWith("chapter:"), "Expected chapter: prefix but got: " + ch1);
        assertTrue(ch1.contains("I \u2014"), "Expected 'I \u2014' in: " + ch1);
        assertTrue(ch2.contains("II \u2014"), "Expected 'II \u2014' in: " + ch2);
        assertTrue(ch3.contains("III \u2014"), "Expected 'III \u2014' in: " + ch3);
        assertTrue(ch1.contains("sacrifices"), "Expected sacrifice text in: " + ch1);
        assertNull(chapters.get(0).actionNumber(), "Chapters should not have action numbers");
    }

    @Test
    void battleCardWithDefense() {
        MultiCard result = convert(
                "Name:Invasion of Kamigawa",
                "ManaCost:3 U",
                "Types:Battle Siege",
                "Defense:4",
                "T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ TrigTap | TriggerDescription$ When CARDNAME enters, tap target artifact or creature an opponent controls and put a stun counter on it.",
                "SVar:TrigTap:DB$ Tap | ValidTgts$ Artifact.OppCtrl,Creature.OppCtrl | SubAbility$ DBCounter | TgtPrompt$ Select target artifact or creature an opponent controls",
                "SVar:DBCounter:DB$ PutCounter | Defined$ Targeted | CounterType$ Stun | CounterNum$ 1",
                "AlternateMode:DoubleFaced",
                "Oracle:",
                "ALTERNATE",
                "Name:Rooftop Saboteurs",
                "ManaCost:no cost",
                "Colors:blue",
                "Types:Creature Moonfolk Ninja",
                "PT:2/3",
                "K:Flying",
                "T:Mode$ DamageDone | ValidSource$ Card.Self | ValidTarget$ Player,Battle | CombatDamage$ True | Execute$ TrigDraw | TriggerDescription$ Whenever CARDNAME deals combat damage to a player or battle, draw a card.",
                "SVar:TrigDraw:DB$ Draw",
                "Oracle:");
        assertEquals("transform", result.layout());
        assertEquals(2, result.faces().size());

        // Front face: Battle with defense
        ConvertedCard front = result.faces().get(0);
        assertEquals("invasion of kamigawa", front.name());
        assertEquals("4", front.defense());
        assertNull(front.powerToughness());
        assertNull(front.loyalty());
        String frontOutput = OutputFormatter.formatCard(front);
        assertTrue(frontOutput.contains("defense: 4"), "Expected defense line in output");

        // Back face: creature, no defense
        ConvertedCard back = result.faces().get(1);
        assertEquals("rooftop saboteurs", back.name());
        assertNull(back.defense());
        assertEquals("2/3", back.powerToughness());
    }

    @Test
    void transformCard() {
        MultiCard result = convert(
                "Name:Daring Sleuth",
                "ManaCost:1 U",
                "Types:Creature Human Rogue",
                "PT:2/1",
                "AlternateMode:DoubleFaced",
                "Oracle:",
                "ALTERNATE",
                "Name:Bearer of Overwhelming Truths",
                "Types:Creature Human Wizard",
                "PT:3/2",
                "Oracle:");
        assertEquals("transform", result.layout());
        assertEquals(2, result.faces().size());
        assertEquals("daring sleuth", result.faces().get(0).name());
        assertEquals("bearer of overwhelming truths", result.faces().get(1).name());
    }

    @Test
    void spellEffect() {
        MultiCard result = convert(
                "Name:Lightning Bolt",
                "ManaCost:R",
                "Types:Instant",
                "A:SP$ DealDamage | ValidTgts$ Creature,Player,Planeswalker | NumDmg$ 3 | SpellDescription$ CARDNAME deals 3 damage to any target.",
                "Oracle:Lightning Bolt deals 3 damage to any target.");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> spells = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.SPELL).toList();
        assertEquals(1, spells.size());
        assertTrue(spells.get(0).formatLine().contains("CARDNAME deals 3 damage"));
        assertEquals(1, (int) spells.get(0).actionNumber());
        // No additional cost for a simple spell
        long additionalCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).count();
        assertEquals(0, additionalCosts, "Lightning Bolt should have no additional cost");
    }

    @Test
    void spellWithAdditionalCost() {
        MultiCard result = convert(
                "Name:Abandon Hope",
                "ManaCost:X 1 B",
                "Types:Sorcery",
                "A:SP$ Discard | Cost$ X 1 B Discard<X/Card/card> | ValidTgts$ Opponent | Mode$ LookYouChoose | NumCards$ X | SpellDescription$ Look at target opponent's hand and choose X cards from it. That player discards those cards.",
                "SVar:X:Count$xPaid",
                "Oracle:As an additional cost to cast this spell, discard X cards.\\nLook at target opponent's hand and choose X cards from it. That player discards those cards.");
        ConvertedCard card = result.faces().get(0);

        // Additional cost on its own line
        List<AbilityLine> costs = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, costs.size(), "Expected 1 additional cost line but got: " + card.abilities());
        String costLine = costs.get(0).formatLine();
        assertTrue(costLine.startsWith("additional cost:"),
                "Expected 'additional cost:' prefix but got: " + costLine);
        assertTrue(costLine.contains("discard X"),
                "Expected discard text in: " + costLine);
        assertFalse(costLine.contains("as an additional cost to cast"),
                "Prefix should be stripped, but got: " + costLine);

        // Spell effect on its own line
        List<AbilityLine> spells = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.SPELL).toList();
        assertEquals(1, spells.size(), "Expected 1 spell line");
        assertTrue(spells.get(0).formatLine().contains("look at target opponent's hand"),
                "Expected spell text in: " + spells.get(0).formatLine());
    }

    @Test
    void cleaveEmittedAsAlternateCost() {
        // Alchemist's Gambit: Cleave is a NonBasicSpell alternate cost
        MultiCard result = convert(
                "Name:Alchemist's Gambit",
                "ManaCost:1 R R",
                "Types:Sorcery",
                "A:SP$ AddTurn | NumTurns$ 1 | ExtraTurnDelayedTrigger$ DBDelTrig | ExtraTurnDelayedTriggerExcute$ TrigEffect | SubAbility$ DBExile | StackDescription$ {p:You} takes an extra turn after this one. During that turn, damage can't be prevented. At the beginning of that turn's end step, {p:You} loses the game. | SpellDescription$ Take an extra turn after this one. [At the beginning of that turn's end step, you lose the game.]",
                "SVar:DBDelTrig:ThisTurn$ True | Static$ True | Mode$ Phase | Phase$ Upkeep | TriggerDescription$ During that turn, damage can't be prevented.",
                "SVar:TrigEffect:DB$ Effect | Defined$ You | StaticAbilities$ STCantPrevent | Triggers$ EndLose",
                "SVar:STCantPrevent:Mode$ CantPreventDamage | Description$ Damage can't be prevented.",
                "SVar:EndLose:Mode$ Phase | Phase$ End of Turn | Execute$ TrigLose | TriggerDescription$ At the beginning of that turn's end step, you lose the game.",
                "SVar:TrigLose:DB$ LosesGame | Defined$ You",
                "A:SP$ AddTurn | Cost$ 4 U U R | NumTurns$ 1 | ExtraTurnDelayedTrigger$ DBDelTrig | ExtraTurnDelayedTriggerExcute$ TrigEffect2 | PrecostDesc$ Cleave | CostDesc$ {4}{U}{U}{R} | NonBasicSpell$ True | SubAbility$ DBExile | StackDescription$ {p:You} takes an extra turn after this one. During that turn, damage can't be prevented. | SpellDescription$ (You may cast this spell for its cleave cost. If you do, remove the words in square brackets.)",
                "SVar:TrigEffect2:DB$ Effect | Defined$ You | StaticAbilities$ STCantPrevent",
                "SVar:DBExile:DB$ ChangeZone | Origin$ Stack | Destination$ Exile | SpellDescription$ Exile CARDNAME.",
                "Oracle:Cleave {4}{U}{U}{R} (You may cast this spell for its cleave cost. If you do, remove the words in square brackets.)\\nTake an extra turn after this one. During that turn, damage can't be prevented. [At the beginning of that turn's end step, you lose the game.]\\nExile Alchemist's Gambit.");
        ConvertedCard card = result.faces().get(0);

        // Cleave should appear as alternate cost
        List<AbilityLine> altCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ALTERNATE_COST).toList();
        assertEquals(1, altCosts.size(), "Expected 1 alternate cost line but got: " + card.abilities());
        String altLine = altCosts.get(0).formatLine();
        assertTrue(altLine.contains("cleave"), "Expected 'cleave' in: " + altLine);
        assertTrue(altLine.contains("{4}{U}{U}{R}"), "Expected '{4}{U}{U}{R}' in: " + altLine);
        // No buggy trailing characters
        assertFalse(altLine.matches(".*\\{R\\}\\d.*"), "No trailing digits after cost in: " + altLine);

        // Real spell lines should still have correct action numbers
        List<AbilityLine> spells = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.SPELL).toList();
        assertTrue(spells.size() >= 1, "Expected at least 1 spell line but got: " + spells);
        List<Integer> actionNumbers = spells.stream()
                .map(AbilityLine::actionNumber).toList();
        for (int i = 0; i < actionNumbers.size() - 1; i++) {
            assertEquals(actionNumbers.get(i) + 1, (int) actionNumbers.get(i + 1),
                    "Action numbers should be sequential: " + actionNumbers);
        }
    }

    @Test
    void creatureWithAdditionalCost() {
        MultiCard result = convert(
                "Name:Abhorrent Oculus",
                "ManaCost:2 U",
                "Types:Creature Eye",
                "PT:5/5",
                "A:SP$ PermanentCreature | Cost$ 2 U ExileFromGrave<6/Card>",
                "K:Flying",
                "T:Mode$ Phase | Phase$ Upkeep | ValidPlayer$ Opponent | TriggerZones$ Battlefield | Execute$ TrigDread | TriggerDescription$ At the beginning of each opponent's upkeep, manifest dread.",
                "SVar:TrigDread:DB$ ManifestDread",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);

        // Additional cost should appear even without SpellDescription
        List<AbilityLine> costs = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, costs.size(), "Expected 1 additional cost line but got: " + card.abilities());
        assertTrue(costs.get(0).formatLine().contains("exile"),
                "Expected exile text in: " + costs.get(0).formatLine());
    }

    @Test
    void additionalCostFromRaiseCostStatic() {
        MultiCard result = convert(
                "Name:Aether Tide",
                "ManaCost:X U",
                "Types:Sorcery",
                "S:Mode$ RaiseCost | ValidCard$ Card.Self | Activator$ You | Type$ Spell | Cost$ Discard<X/Creature/creature(s)> | EffectZone$ All | Description$ As an additional cost to cast this spell, discard X creature cards.",
                "A:SP$ ChangeZone | Cost$ X U | TargetMin$ X | TargetMax$ X | Origin$ Battlefield | Destination$ Hand | ValidTgts$ Creature | TgtPrompt$ Select X target creatures | SpellDescription$ Return X target creatures to their owners' hands.",
                "SVar:X:Count$xPaid",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);

        // RaiseCost static should become additional cost, not static
        List<AbilityLine> costs = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, costs.size(), "Expected 1 additional cost line but got: " + card.abilities());
        String costLine = costs.get(0).formatLine();
        assertTrue(costLine.contains("discard X creature cards"),
                "Expected discard text in: " + costLine);
        assertFalse(costLine.contains("as an additional cost to cast"),
                "Prefix should be stripped, but got: " + costLine);

        // No static lines should remain
        long staticCount = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.STATIC).count();
        assertEquals(0, staticCount, "RaiseCost should not appear as static");

        // Spell effect on its own line
        List<AbilityLine> spells = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.SPELL).toList();
        assertEquals(1, spells.size(), "Expected 1 spell line");
        assertTrue(spells.get(0).formatLine().contains("return X target creatures"),
                "Expected spell text in: " + spells.get(0).formatLine());

        // Additional cost should appear before spell in ability list
        int costIdx = card.abilities().indexOf(costs.get(0));
        int spellIdx = card.abilities().indexOf(spells.get(0));
        assertTrue(costIdx < spellIdx, "Additional cost should appear before spell abilities");
    }

    @Test
    void optionalAdditionalCost() {
        MultiCard result = convert(
                "Name:Analyze the Pollen",
                "ManaCost:G",
                "Types:Sorcery",
                "S:Mode$ OptionalCost | EffectZone$ All | ValidCard$ Card.Self | ValidSA$ Spell | Cost$ CollectEvidence<8> | Description$ As an additional cost to cast this spell, you may collect evidence 8. (Exile cards with total mana value 8 or greater from your graveyard.)",
                "A:SP$ ChangeZone | Origin$ Library | Destination$ Hand | ChangeType$ Land.Basic | SpellDescription$ Search your library for a basic land card.",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);

        // OptionalCost static should become additional cost
        List<AbilityLine> costs = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, costs.size(), "Expected 1 additional cost line but got: " + card.abilities());
        String costLine = costs.get(0).formatLine();
        assertTrue(costLine.contains("you may collect evidence 8"),
                "Expected 'you may' description in: " + costLine);
        assertFalse(costLine.contains("as an additional cost to cast"),
                "Prefix should be stripped, but got: " + costLine);

        // No static lines should remain
        long staticCount = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.STATIC).count();
        assertEquals(0, staticCount, "OptionalCost should not appear as static");

        // Additional cost should appear before spell
        List<AbilityLine> spells = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.SPELL).toList();
        assertEquals(1, spells.size());
        int costIdx = card.abilities().indexOf(costs.get(0));
        int spellIdx = card.abilities().indexOf(spells.get(0));
        assertTrue(costIdx < spellIdx, "Additional cost should appear before spell abilities");
    }

    @Test
    void alternateAdditionalCost() {
        MultiCard result = convert(
                "Name:Annihilating Glare",
                "ManaCost:B",
                "Types:Sorcery",
                "K:AlternateAdditionalCost:Sac<1/Creature;Artifact/artifact or creature>:4",
                "A:SP$ Destroy | ValidTgts$ Creature,Planeswalker | TgtPrompt$ Select target creature or planeswalker | SpellDescription$ Destroy target creature or planeswalker.",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);

        // AlternateAdditionalCost keyword should become additional cost line
        List<AbilityLine> costs = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, costs.size(), "Expected 1 additional cost line but got: " + card.abilities());
        String costLine = costs.get(0).formatLine();
        assertTrue(costLine.contains("sacrifice"), "Expected sacrifice in: " + costLine);
        assertTrue(costLine.contains("or"), "Expected 'or' joining alternatives in: " + costLine);

        // No keyword lines should remain for this
        long keywordCount = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.KEYWORD_PASSIVE
                        || a.type() == AbilityType.KEYWORD_ACTIVE).count();
        assertEquals(0, keywordCount, "AlternateAdditionalCost should not appear as keyword");

        // Spell effect present
        List<AbilityLine> spells = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.SPELL).toList();
        assertEquals(1, spells.size());
        assertTrue(spells.get(0).formatLine().contains("destroy target creature or planeswalker"));

        // Additional cost before spell
        int costIdx = card.abilities().indexOf(costs.get(0));
        int spellIdx = card.abilities().indexOf(spells.get(0));
        assertTrue(costIdx < spellIdx, "Additional cost should appear before spell abilities");
    }

    @Test
    void pawprintCharmOptions() {
        MultiCard result = convert(
                "Name:Season of the Burrow",
                "ManaCost:3 W W",
                "Types:Sorcery",
                "A:SP$ Charm | Choices$ DBToken,DBExile,DBReanimate | CharmNum$ 5 | MinCharmNum$ 0 | CanRepeatModes$ True | Pawprint$ 5",
                "SVar:DBToken:DB$ Token | Pawprint$ 1 | TokenScript$ w_1_1_rabbit | SpellDescription$ Create a 1/1 white Rabbit creature token.",
                "SVar:DBExile:DB$ ChangeZone | Pawprint$ 2 | Origin$ Battlefield | Destination$ Exile | ValidTgts$ Permanent.nonLand | RememberLKI$ True | SubAbility$ DBDraw | SpellDescription$ Exile target nonland permanent. Its controller draws a card.",
                "SVar:DBDraw:DB$ Draw | Defined$ RememberedController | NumCards$ 1 | SubAbility$ DBCleanup",
                "SVar:DBCleanup:DB$ Cleanup | ClearRemembered$ True",
                "SVar:DBReanimate:DB$ ChangeZone | Pawprint$ 3 | ValidTgts$ Permanent.YouCtrl+cmcLE3 | Origin$ Graveyard | Destination$ Battlefield | WithCountersType$ Indestructible | SpellDescription$ Return target permanent card with mana value 3 or less from your graveyard to the battlefield with an indestructible counter on it.",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);

        // Charm generates a spell line from the combined choice descriptions
        List<AbilityLine> spells = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.SPELL).toList();
        assertEquals(1, spells.size(), "Expected 1 spell line but got: " + card.abilities());

        List<AbilityLine> options = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.OPTION).toList();
        assertEquals(3, options.size(), "Expected 3 options but got: " + card.abilities());

        // Each option should have paw print cost prefix
        String o1 = options.get(0).formatLine();
        assertTrue(o1.contains("{P} \u2014"), "Expected {P} cost in: " + o1);
        assertTrue(o1.contains("create a 1/1"), "Expected token text in: " + o1);

        String o2 = options.get(1).formatLine();
        assertTrue(o2.contains("{P}{P} \u2014"), "Expected {P}{P} cost in: " + o2);
        assertTrue(o2.contains("exile target"), "Expected exile text in: " + o2);

        String o3 = options.get(2).formatLine();
        assertTrue(o3.contains("{P}{P}{P} \u2014"), "Expected {P}{P}{P} cost in: " + o3);
        assertTrue(o3.contains("return target"), "Expected reanimate text in: " + o3);
    }

    @Test
    void classEnchantmentLevels() {
        MultiCard result = convert(
                "Name:Artificer Class",
                "ManaCost:1 U",
                "Types:Enchantment Class",
                "S:Mode$ ReduceCost | EffectZone$ Battlefield | ValidCard$ Card.Artifact | Activator$ You | Type$ Spell | OnlyFirstSpell$ True | Amount$ 1 | Description$ The first artifact spell you cast each turn costs {1} less to cast.",
                "K:Class:2:1 U:AddTrigger$ TriggerClassLevel",
                "SVar:TriggerClassLevel:Mode$ ClassLevelGained | ClassLevel$ 2 | ValidCard$ Card.Self | TriggerZones$ Battlefield | Execute$ TrigDigUntil | Secondary$ True | TriggerDescription$ When this Class becomes level 2, reveal cards from the top of your library until you reveal an artifact card. Put that card into your hand and the rest on the bottom of your library in a random order.",
                "SVar:TrigDigUntil:DB$ DigUntil | Valid$ Artifact | FoundDestination$ Hand | RevealedDestination$ Library | RevealedLibraryPosition$ -1 | RevealRandomOrder$ True",
                "K:Class:3:5 U:AddTrigger$ TriggerEndTurn",
                "SVar:TriggerEndTurn:Mode$ Phase | Phase$ End of Turn | ValidPlayer$ You | TriggerZones$ Battlefield | Execute$ CopyArtifact | Secondary$ True | TriggerDescription$ At the beginning of your end step, create a token that's a copy of target artifact you control.",
                "SVar:CopyArtifact:DB$ CopyPermanent | ValidTgts$ Artifact.YouCtrl | TgtPrompt$ Select target artifact you control to copy",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        assertEquals("enchantment class", card.types());

        List<AbilityLine> levels = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.LEVEL).toList();
        assertEquals(3, levels.size(), "Expected 3 level abilities but got: " + card.abilities());

        // Level 1: no cost prefix
        String l1 = levels.get(0).formatLine();
        assertTrue(l1.startsWith("level[1]:"), "Expected level[1]: but got: " + l1);
        assertTrue(l1.contains("first artifact spell"), "Expected level 1 text in: " + l1);
        assertFalse(l1.contains("{1}{U}:"), "Level 1 should not have a cost prefix");

        // Level 2: cost prefix
        String l2 = levels.get(1).formatLine();
        assertTrue(l2.startsWith("level[2]:"), "Expected level[2]: but got: " + l2);
        assertTrue(l2.contains("{1}{U}:"), "Expected cost {1}{U}: in: " + l2);
        assertTrue(l2.contains("reveal cards"), "Expected level 2 text in: " + l2);

        // Level 3: cost prefix
        String l3 = levels.get(2).formatLine();
        assertTrue(l3.startsWith("level[3]:"), "Expected level[3]: but got: " + l3);
        assertTrue(l3.contains("{5}{U}:"), "Expected cost {5}{U}: in: " + l3);
        assertTrue(l3.contains("create a token"), "Expected level 3 text in: " + l3);

        // No activated or static lines should remain
        long activatedCount = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ACTIVATED).count();
        long staticCount = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.STATIC).count();
        assertEquals(0, activatedCount, "No activated lines expected for Class card");
        assertEquals(0, staticCount, "No static lines expected for Class card");
    }

    @Test
    void classWithEtbReplacementLevel1() {
        MultiCard result = convert(
                "Name:Bard Class",
                "ManaCost:R G",
                "Types:Enchantment Class",
                "K:ETBReplacement:Other:AddExtraCounter:Mandatory:Battlefield:Creature.Legendary+YouCtrl+Other",
                "SVar:AddExtraCounter:DB$ PutCounter | ETB$ True | Defined$ ReplacedCard | CounterType$ P1P1 | CounterNum$ 1 | SpellDescription$ Legendary creatures you control enter with an additional +1/+1 counter on them.",
                "K:Class:2:R G:AddStaticAbility$ SReduceCost",
                "SVar:SReduceCost:Mode$ ReduceCost | ValidCard$ Legendary | Type$ Spell | Activator$ You | Amount$ 1 | Color$ R G | IgnoreGeneric$ True | Secondary$ True | Description$ Legendary spells you cast cost {R}{G} less to cast.",
                "K:Class:3:3 R G:AddTrigger$ TriggerCast",
                "SVar:TriggerCast:Mode$ SpellCast | ValidCard$ Legendary | ValidActivatingPlayer$ You | TriggerZones$ Battlefield | Execute$ TrigImpulsiveDraw | Secondary$ True | TriggerDescription$ Whenever you cast a legendary spell, exile the top two cards of your library. You may play them this turn.",
                "SVar:TrigImpulsiveDraw:DB$ Dig",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);

        List<AbilityLine> levels = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.LEVEL).toList();
        assertEquals(3, levels.size(), "Expected 3 level abilities but got: " + card.abilities());

        // Level 1 comes from ETBReplacement (not a static)
        String l1 = levels.get(0).formatLine();
        assertTrue(l1.startsWith("level[1]:"), "Expected level[1]: but got: " + l1);
        assertTrue(l1.contains("legendary creatures"), "Expected level 1 text in: " + l1);

        // No raw ETBReplacement keyword lines
        long rawKeywords = card.abilities().stream()
                .filter(a -> (a.type() == AbilityType.KEYWORD_PASSIVE
                        || a.type() == AbilityType.KEYWORD_ACTIVE)
                        && a.description().contains("etbreplacement")).count();
        assertEquals(0, rawKeywords, "ETBReplacement should not appear as raw keyword");
    }

    @Test
    void etbReplacementOnNonClassCard() {
        MultiCard result = convert(
                "Name:Flickering Ward",
                "ManaCost:W",
                "Types:Enchantment Aura",
                "K:Enchant creature",
                "K:ETBReplacement:Other:ChooseColor",
                "SVar:ChooseColor:DB$ ChooseColor | Defined$ You | SpellDescription$ As CARDNAME enters, choose a color.",
                "A:AB$ ChangeZone | Cost$ W | Origin$ Battlefield | Destination$ Hand | SpellDescription$ Return CARDNAME to its owner's hand.",
                "S:Mode$ Continuous | Affected$ Creature.EnchantedBy | AddKeyword$ Protection:ChosenColor | Description$ Enchanted creature has protection from the chosen color. This effect doesn't remove CARDNAME.",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);

        // ETBReplacement should be a replacement line, not a raw keyword
        long rawKeywords = card.abilities().stream()
                .filter(a -> (a.type() == AbilityType.KEYWORD_PASSIVE
                        || a.type() == AbilityType.KEYWORD_ACTIVE)
                        && a.description().contains("etbreplacement")).count();
        assertEquals(0, rawKeywords, "ETBReplacement should not appear as raw keyword");

        List<AbilityLine> replacements = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.REPLACEMENT).toList();
        assertTrue(replacements.stream().anyMatch(r -> r.description().contains("choose a color")),
                "Expected 'choose a color' replacement but got: " + replacements);
    }

    @Test
    void companionKeywordIncludesRestriction() {
        MultiCard result = convert(
                "Name:Gyruda Doom of Depths",
                "ManaCost:4 U B",
                "Types:Legendary Creature Demon Kraken",
                "PT:6/6",
                "K:Companion:Card.cmcM20:Your starting deck contains only cards with even mana value.",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> keywords = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.KEYWORD_PASSIVE).toList();
        assertTrue(keywords.stream().anyMatch(k ->
                        k.description().contains("companion") && k.description().contains("even mana value")),
                "Expected companion with deck restriction but got: " + keywords);
    }

    @Test
    void doubleKickerIncludesBothCosts() {
        MultiCard result = convert(
                "Name:Archangel of Wrath",
                "ManaCost:2 W W",
                "Types:Creature Angel",
                "PT:3/4",
                "K:Kicker:B:R",
                "K:Flying",
                "K:Lifelink",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        // Kicker should be reclassified as additional cost with both costs
        List<AbilityLine> costs = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, costs.size(), "Expected 1 additional cost line but got: " + card.abilities());
        String costLine = costs.get(0).formatLine();
        assertTrue(costLine.contains("kicker"), "Expected 'kicker' in: " + costLine);
        assertTrue(costLine.contains("{B}"), "Expected '{B}' in: " + costLine);
        assertTrue(costLine.contains("{R}"), "Expected '{R}' in: " + costLine);
        assertTrue(costLine.contains("and/or"), "Expected 'and/or' in: " + costLine);
    }

    @Test
    void giftKeywordIncludesParameter() {
        MultiCard result = convert(
                "Name:Dawn's Truce",
                "ManaCost:1 W",
                "Types:Instant",
                "K:Gift",
                "SVar:GiftAbility:DB$ Draw | Defined$ Promised | GiftDescription$ a card",
                "A:SP$ Pump | Defined$ You & Valid Permanent.YouCtrl | KW$ Hexproof | SubAbility$ DBPumpAll | SpellDescription$ You and permanents you control gain hexproof until end of turn.",
                "SVar:DBPumpAll:DB$ PumpAll | ValidCards$ Permanent.YouCtrl | KW$ Indestructible | ConditionZone$ Stack | ConditionPresent$ Card.Self+PromisedGift | ConditionCompare$ EQ1",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> keywords = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.KEYWORD_PASSIVE).toList();
        // Gift should include its parameter
        assertTrue(keywords.stream().anyMatch(k -> k.description().contains("gift a card")),
                "Expected 'gift a card' keyword but got: " + keywords);
    }

    // --- Alternate cost keyword classification ---

    @Test
    void ninjutsuClassifiedAsAlternateCost() {
        MultiCard result = convert(
                "Name:Yuffie Materia Hunter",
                "ManaCost:2 R",
                "Types:Legendary Creature Human Ninja",
                "PT:3/3",
                "K:Ninjutsu:1 R",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> altCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ALTERNATE_COST).toList();
        assertEquals(1, altCosts.size(), "Expected 1 alternate cost for Ninjutsu but got: " + card.abilities());
        assertTrue(altCosts.get(0).description().contains("ninjutsu"),
                "Expected 'ninjutsu' in: " + altCosts.get(0).description());
        assertTrue(altCosts.get(0).description().contains("{1}{R}"),
                "Expected '{1}{R}' in: " + altCosts.get(0).description());
    }

    @Test
    void embalmClassifiedAsAlternateCost() {
        MultiCard result = convert(
                "Name:Vizier of Many Faces",
                "ManaCost:2 U U",
                "Types:Creature Shapeshifter Cleric",
                "PT:0/0",
                "K:ETBReplacement:Copy:DBCopy:Optional",
                "SVar:DBCopy:DB$ Clone | Choices$ Creature.Other | Embalm$ True | RemoveCost$ True | SetColor$ White | AddTypes$ Zombie | SpellDescription$ You may have CARDNAME enter as a copy of any creature on the battlefield.",
                "K:Embalm:3 U U",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> altCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ALTERNATE_COST).toList();
        assertEquals(1, altCosts.size(), "Expected 1 alternate cost for Embalm but got: " + card.abilities());
        assertTrue(altCosts.get(0).description().contains("embalm"),
                "Expected 'embalm' in: " + altCosts.get(0).description());
        assertTrue(altCosts.get(0).description().contains("{3}{U}{U}"),
                "Expected '{3}{U}{U}' in: " + altCosts.get(0).description());
    }

    @Test
    void miracleClassifiedAsAlternateCost() {
        MultiCard result = convert(
                "Name:Zephyrim",
                "ManaCost:3 W",
                "Types:Creature Human Warrior",
                "PT:3/3",
                "K:Squad:2",
                "K:Flying",
                "K:Vigilance",
                "K:Miracle:1 W",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> altCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ALTERNATE_COST).toList();
        assertEquals(1, altCosts.size(), "Expected 1 alternate cost for Miracle but got: " + card.abilities());
        assertTrue(altCosts.get(0).description().contains("miracle"),
                "Expected 'miracle' in: " + altCosts.get(0).description());
        // Squad should be additional cost, not alternate
        List<AbilityLine> addCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, addCosts.size(), "Expected 1 additional cost for Squad");
    }

    @Test
    void unearthClassifiedAsAlternateCost() {
        MultiCard result = convert(
                "Name:Yotian Frontliner",
                "ManaCost:1",
                "Types:Artifact Creature Soldier",
                "PT:1/1",
                "T:Mode$ Attacks | ValidCard$ Card.Self | Execute$ TrigPump | TriggerDescription$ Whenever CARDNAME attacks, another target creature you control gets +1/+1 until end of turn.",
                "SVar:TrigPump:DB$ Pump | ValidTgts$ Creature.YouCtrl+Other | TgtPrompt$ Select another target creature you control | NumAtt$ +1 | NumDef$ +1",
                "K:Unearth:W",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> altCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ALTERNATE_COST).toList();
        assertEquals(1, altCosts.size(), "Expected 1 alternate cost for Unearth but got: " + card.abilities());
        assertTrue(altCosts.get(0).description().contains("unearth"),
                "Expected 'unearth' in: " + altCosts.get(0).description());
    }

    @Test
    void awakenClassifiedAsAlternateCost() {
        MultiCard result = convert(
                "Name:Sheer Drop",
                "ManaCost:2 W",
                "Types:Sorcery",
                "A:SP$ Destroy | ValidTgts$ Creature.tapped | TgtPrompt$ Select target tapped creature | SpellDescription$ Destroy target tapped creature.",
                "K:Awaken:3:5 W",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> altCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ALTERNATE_COST).toList();
        assertEquals(1, altCosts.size(), "Expected 1 alternate cost for Awaken but got: " + card.abilities());
        assertTrue(altCosts.get(0).description().contains("awaken"),
                "Expected 'awaken' in: " + altCosts.get(0).description());
    }

    @Test
    void suspendClassifiedAsAlternateCost() {
        MultiCard result = convert(
                "Name:Wheel of Fate",
                "ManaCost:no cost",
                "Colors:red",
                "Types:Sorcery",
                "K:Suspend:4:1 R",
                "A:SP$ Discard | Mode$ Hand | Defined$ Player | SubAbility$ DBDraw | SpellDescription$ Each player discards their hand, then draws seven cards.",
                "SVar:DBDraw:DB$ Draw | Defined$ Player | NumCards$ 7",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> altCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ALTERNATE_COST).toList();
        assertEquals(1, altCosts.size(), "Expected 1 alternate cost for Suspend but got: " + card.abilities());
        assertTrue(altCosts.get(0).description().contains("suspend"),
                "Expected 'suspend' in: " + altCosts.get(0).description());
    }

    @Test
    void jumpStartClassifiedAsAlternateCost() {
        MultiCard result = convert(
                "Name:Surge of Acclaim",
                "ManaCost:1 U",
                "Types:Instant",
                "K:Jump-start",
                "A:SP$ Charm | CharmNum$ X | Choices$ DBSeek1,DBSeek2",
                "SVar:DBSeek1:DB$ Seek | Type$ Card.nonLand | SpellDescription$ Seek a nonland card.",
                "SVar:DBSeek2:DB$ Seek | Type$ Card.nonLand | SpellDescription$ Seek a nonland card.",
                "SVar:X:Count$MaxSpeed.2.1",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> altCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ALTERNATE_COST).toList();
        assertEquals(1, altCosts.size(), "Expected 1 alternate cost for Jump-start but got: " + card.abilities());
        assertTrue(altCosts.get(0).description().contains("jump-start"),
                "Expected 'jump-start' in: " + altCosts.get(0).description());
    }

    // --- Additional cost keyword classification ---

    @Test
    void bargainClassifiedAsAdditionalCost() {
        MultiCard result = convert(
                "Name:Troublemaker Ouphe",
                "ManaCost:1 G",
                "Types:Creature Ouphe",
                "PT:2/2",
                "K:Bargain",
                "T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self+bargained | Execute$ TrigExile | TriggerDescription$ When CARDNAME enters, if it was bargained, exile target artifact or enchantment an opponent controls.",
                "SVar:TrigExile:DB$ ChangeZone | IsCurse$ True | ValidTgts$ Artifact.OppCtrl,Enchantment.OppCtrl | TgtPrompt$ Select target artifact or enchantment an opponent controls | Origin$ Battlefield | Destination$ Exile",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> addCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, addCosts.size(), "Expected 1 additional cost for Bargain but got: " + card.abilities());
        assertTrue(addCosts.get(0).description().contains("bargain"),
                "Expected 'bargain' in: " + addCosts.get(0).description());
    }

    @Test
    void conspireClassifiedAsAdditionalCost() {
        MultiCard result = convert(
                "Name:Traitor's Roar",
                "ManaCost:4 BR",
                "Types:Sorcery",
                "A:SP$ Tap | ValidTgts$ Creature.untapped | TgtPrompt$ Select target untapped creature | SubAbility$ DBDamage | SpellDescription$ Tap target untapped creature. It deals damage equal to its power to its controller.",
                "SVar:DBDamage:DB$ DealDamage | Defined$ TargetedController | DamageSource$ Targeted | NumDmg$ X",
                "SVar:X:Targeted$CardPower",
                "K:Conspire",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> addCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, addCosts.size(), "Expected 1 additional cost for Conspire but got: " + card.abilities());
        assertTrue(addCosts.get(0).description().contains("conspire"),
                "Expected 'conspire' in: " + addCosts.get(0).description());
    }

    @Test
    void spliceClassifiedAsAdditionalCost() {
        MultiCard result = convert(
                "Name:Wear Away",
                "ManaCost:G G",
                "Types:Instant Arcane",
                "K:Splice:Arcane:3 G",
                "A:SP$ Destroy | ValidTgts$ Artifact,Enchantment | TgtPrompt$ Select target artifact or enchantment | SpellDescription$ Destroy target artifact or enchantment.",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> addCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, addCosts.size(), "Expected 1 additional cost for Splice but got: " + card.abilities());
        assertTrue(addCosts.get(0).description().contains("splice"),
                "Expected 'splice' in: " + addCosts.get(0).description());
    }

    @Test
    void spreeClassifiedAsAdditionalCost() {
        MultiCard result = convert(
                "Name:Unfortunate Accident",
                "ManaCost:B",
                "Types:Instant",
                "K:Spree",
                "A:SP$ Charm | Choices$ DBMurder,DBRecruit | MinCharmNum$ 1 | CharmNum$ 2",
                "SVar:DBMurder:DB$ Destroy | ModeCost$ 2 B | ValidTgts$ Creature | SpellDescription$ Destroy target creature.",
                "SVar:DBRecruit:DB$ Token | ModeCost$ 1 | TokenScript$ r_1_1_mercenary_tappump | TokenOwner$ You | SpellDescription$ Create a 1/1 red Mercenary creature token.",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> addCosts = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.ADDITIONAL_COST).toList();
        assertEquals(1, addCosts.size(), "Expected 1 additional cost for Spree but got: " + card.abilities());
        assertTrue(addCosts.get(0).description().contains("spree"),
                "Expected 'spree' in: " + addCosts.get(0).description());
    }

    // --- Cost reduction keyword classification ---

    @Test
    void affinityClassifiedAsCostReduction() {
        MultiCard result = convert(
                "Name:Sojourner's Companion",
                "ManaCost:7",
                "Types:Artifact Creature Salamander",
                "PT:4/4",
                "K:Affinity:Artifact",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> reductions = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.COST_REDUCTION).toList();
        assertEquals(1, reductions.size(), "Expected 1 cost reduction for Affinity but got: " + card.abilities());
        assertTrue(reductions.get(0).description().contains("affinity"),
                "Expected 'affinity' in: " + reductions.get(0).description());
        assertEquals("cost reduction: affinity for artifacts", reductions.get(0).formatLine());
    }

    @Test
    void delveClassifiedAsCostReduction() {
        MultiCard result = convert(
                "Name:Will of the Naga",
                "ManaCost:4 U U",
                "Types:Instant",
                "K:Delve",
                "A:SP$ Tap | TargetMin$ 0 | TargetMax$ 2 | ValidTgts$ Creature | SpellDescription$ Tap up to two target creatures.",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> reductions = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.COST_REDUCTION).toList();
        assertEquals(1, reductions.size(), "Expected 1 cost reduction for Delve but got: " + card.abilities());
        assertTrue(reductions.get(0).description().contains("delve"),
                "Expected 'delve' in: " + reductions.get(0).description());
    }

    @Test
    void improviseClassifiedAsCostReduction() {
        MultiCard result = convert(
                "Name:Whir of Invention",
                "ManaCost:X U U U",
                "Types:Instant",
                "K:Improvise",
                "A:SP$ ChangeZone | Origin$ Library | Destination$ Battlefield | ChangeType$ Artifact.cmcLEX | ChangeNum$ 1 | SpellDescription$ Search your library for an artifact card with mana value X or less, put it onto the battlefield, then shuffle.",
                "SVar:X:Count$xPaid",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> reductions = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.COST_REDUCTION).toList();
        assertEquals(1, reductions.size(), "Expected 1 cost reduction for Improvise but got: " + card.abilities());
        assertTrue(reductions.get(0).description().contains("improvise"),
                "Expected 'improvise' in: " + reductions.get(0).description());
    }

    @Test
    void convokeClassifiedAsCostReduction() {
        MultiCard result = convert(
                "Name:Test Convoke Spell",
                "ManaCost:3 W W",
                "Types:Sorcery",
                "K:Convoke",
                "A:SP$ Destroy | ValidTgts$ Creature | SpellDescription$ Destroy target creature.",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> reductions = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.COST_REDUCTION).toList();
        assertEquals(1, reductions.size(), "Expected 1 cost reduction for Convoke but got: " + card.abilities());
        assertTrue(reductions.get(0).description().contains("convoke"),
                "Expected 'convoke' in: " + reductions.get(0).description());
    }

    // --- Cost lines sort to top ---

    @Test
    void costReductionSortsToTop() {
        MultiCard result = convert(
                "Name:Whir of Invention",
                "ManaCost:X U U U",
                "Types:Instant",
                "K:Improvise",
                "A:SP$ ChangeZone | Origin$ Library | Destination$ Battlefield | ChangeType$ Artifact.cmcLEX | ChangeNum$ 1 | SpellDescription$ Search your library for an artifact card with mana value X or less, put it onto the battlefield, then shuffle.",
                "SVar:X:Count$xPaid",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        // Cost reduction should come before spell abilities
        List<AbilityLine> abilities = card.abilities();
        assertTrue(abilities.size() >= 2, "Expected at least 2 abilities");
        assertEquals(AbilityType.COST_REDUCTION, abilities.get(0).type(),
                "Cost reduction should be first ability but got: " + abilities);
    }

    @Test
    void allCostTypesSortBeforeOtherAbilities() {
        // Card with alternate cost, additional cost, cost reduction, and a spell
        MultiCard result = convert(
                "Name:Zephyrim",
                "ManaCost:3 W",
                "Types:Creature Human Warrior",
                "PT:3/3",
                "K:Flying",
                "K:Squad:2",
                "K:Miracle:1 W",
                "Oracle:");
        ConvertedCard card = result.faces().get(0);
        List<AbilityLine> abilities = card.abilities();
        // Find the first non-cost ability
        int firstNonCostIdx = -1;
        int lastCostIdx = -1;
        for (int i = 0; i < abilities.size(); i++) {
            AbilityType t = abilities.get(i).type();
            if (t == AbilityType.ALTERNATE_COST || t == AbilityType.ADDITIONAL_COST
                    || t == AbilityType.COST_REDUCTION) {
                lastCostIdx = i;
            } else if (firstNonCostIdx == -1) {
                firstNonCostIdx = i;
            }
        }
        if (lastCostIdx >= 0 && firstNonCostIdx >= 0) {
            assertTrue(lastCostIdx < firstNonCostIdx,
                    "All cost lines should appear before non-cost lines but got: " + abilities);
        }
    }

    @Test
    void layoutDetection() {
        // Transform
        MultiCard transform = convert("Name:A", "ManaCost:1", "Types:Creature Human", "PT:1/1",
                "AlternateMode:DoubleFaced", "Oracle:", "ALTERNATE",
                "Name:B", "Types:Creature Werewolf", "PT:2/2", "Oracle:");
        assertEquals("transform", transform.layout());

        // Split
        MultiCard split = convert("Name:Fire", "ManaCost:1 R", "Types:Instant",
                "AlternateMode:Split", "Oracle:", "ALTERNATE",
                "Name:Ice", "ManaCost:1 U", "Types:Instant", "Oracle:");
        assertEquals("split", split.layout());

        // Adventure
        MultiCard adv = convert("Name:Bonecrusher Giant", "ManaCost:2 R", "Types:Creature Giant",
                "PT:4/3", "AlternateMode:Adventure", "Oracle:", "ALTERNATE",
                "Name:Stomp", "ManaCost:1 R", "Types:Instant Adventure", "Oracle:");
        assertEquals("adventure", adv.layout());
    }
}
