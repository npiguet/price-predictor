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
        List<AbilityLine> keywords = card.abilities().stream()
                .filter(a -> a.type() == AbilityType.KEYWORD_ACTIVE
                        || a.type() == AbilityType.KEYWORD_PASSIVE).toList();
        // Kicker should include both costs
        assertTrue(keywords.stream().anyMatch(k ->
                        k.description().contains("kicker") && k.description().contains("{B}")
                                && k.description().contains("{R}") && k.description().contains("and/or")),
                "Expected kicker with both costs but got: " + keywords);
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
