package com.pricepredictor.connector;

import com.pricepredictor.connector.ability.ActivatedAbilityEntry;
import com.pricepredictor.connector.ability.AlternateCostSpell;
import com.pricepredictor.connector.ability.ChapterAbility;
import com.pricepredictor.connector.ability.CharmAbility;
import com.pricepredictor.connector.ability.ClassLevelAbility;
import com.pricepredictor.connector.ability.CompanionKeyword;
import com.pricepredictor.connector.ability.EtbReplacementAbility;
import com.pricepredictor.connector.ability.GiftKeyword;
import com.pricepredictor.connector.ability.ReplacementAbilityEntry;
import com.pricepredictor.connector.ability.SpellAdditionalCost;
import com.pricepredictor.connector.ability.SpellEffect;
import com.pricepredictor.connector.ability.StandardKeyword;
import com.pricepredictor.connector.ability.StaticAbilityEntry;
import com.pricepredictor.connector.ability.TextAbility;
import com.pricepredictor.connector.ability.TriggeredAbilityEntry;
import forge.card.CardRarity;
import forge.card.CardRules;
import forge.card.CardSplitType;
import forge.card.CardStateName;
import forge.card.ICardFace;
import forge.card.mana.ManaCost;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.cost.Cost;
import forge.game.keyword.Companion;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.Trigger;
import forge.game.Game;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.item.PaperCard;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Parses Forge card scripts into domain objects (MultiCard/CardFace/Ability).
 * Acts as a router, delegating to variant Ability implementations in the ability sub-package.
 */
public class RulesParser {

    private final CardRules.Reader reader = new CardRules.Reader();
    private int nextCardId = 1;

    /**
     * Parse a card script and build domain objects for all faces.
     */
    public MultiCard parseScript(List<String> scriptLines, String filename) {
        reader.reset();
        for (String line : scriptLines) {
            if (line.isEmpty() || line.charAt(0) == '#') {
                continue;
            }
            reader.parseLine(line);
        }
        CardRules rules = reader.getCard();
        return parseRules(rules);
    }

    /**
     * Parse CardRules into a MultiCard.
     */
    public MultiCard parseRules(CardRules rules) {
        CardSplitType splitType = rules.getSplitType();
        Card card = buildFullCard(rules);

        if (splitType == CardSplitType.None) {
            return MultiCard.singleFace(parseFace(card, rules.getMainPart()));
        }

        String layout = splitType.name().toLowerCase();
        List<CardFace> faces = new ArrayList<>();

        CardStateName mainState = (splitType == CardSplitType.Split)
                ? CardStateName.LeftSplit : CardStateName.Original;
        card.setState(mainState, false);
        faces.add(parseFace(card, rules.getMainPart()));

        if (splitType == CardSplitType.Specialize) {
            for (Map.Entry<CardStateName, ICardFace> e : rules.getSpecializeParts().entrySet()) {
                if (e.getValue() != null) {
                    card.setState(e.getKey(), false);
                    faces.add(parseFace(card, e.getValue()));
                }
            }
        } else {
            ICardFace otherFace = rules.getOtherPart();
            if (otherFace != null) {
                card.setState(splitType.getChangedStateName(), false);
                faces.add(parseFace(card, otherFace));
            }
        }

        return MultiCard.multiFace(layout, faces);
    }

    /**
     * Parse a single card face. The Card must already be in the correct state.
     */
    CardFace parseFace(Card card, ICardFace face) {
        List<Ability> abilities = new ArrayList<>();
        boolean isClass = face.getType().toString().contains("Class");
        Set<String> classLevelDescriptions = new HashSet<>();

        // --- Keywords — route to variants ---
        for (KeywordInterface ki : card.getKeywords()) {
            Keyword kw = ki.getKeyword();
            if (kw == Keyword.UNDEFINED) {
                routeUndefinedKeyword(ki, abilities, classLevelDescriptions);
            } else if (kw == Keyword.GIFT) {
                abilities.add(GiftKeyword.of(ki, card));
            } else if (kw == Keyword.COMPANION && ki instanceof Companion comp) {
                abilities.add(CompanionKeyword.of(ki, comp));
            } else {
                abilities.add(StandardKeyword.of(ki, kw));
            }
        }

        // --- Spell abilities — route to variants ---
        for (SpellAbility sa : card.getSpellAbilities()) {
            if (sa.getKeyword() != null) continue;
            if (sa.getApi() == ApiType.Charm) {
                abilities.addAll(CharmAbility.fromSpellAbility(sa));
            } else if (sa.isSpell() && "True".equals(sa.getParam("NonBasicSpell"))) {
                addIfNotNull(abilities, AlternateCostSpell.of(sa));
            } else if (sa.isActivatedAbility()) {
                addIfNotNull(abilities, ActivatedAbilityEntry.of(sa));
            } else if (sa.isSpell()) {
                addIfNotNull(abilities, SpellAdditionalCost.of(sa));
                abilities.addAll(SpellEffect.fromChain(sa));
            }
        }

        // --- Traits — direct wrapping ---
        for (Trigger t : card.getTriggers()) {
            addIfNotNull(abilities, TriggeredAbilityEntry.of(t));
        }
        for (StaticAbility s : card.getStaticAbilities()) {
            addIfNotNull(abilities, StaticAbilityEntry.of(s));
        }
        for (ReplacementEffect r : card.getReplacementEffects()) {
            addIfNotNull(abilities, ReplacementAbilityEntry.of(r));
        }

        // --- Synthetic land mana ---
        String landDesc = buildLandManaDescription(face);
        if (landDesc != null) {
            abilities.add(new TextAbility(AbilityType.ACTIVATED, landDesc));
        }

        // --- Post-processing ---
        if (isClass) {
            abilities = applyClassPostProcessing(abilities, classLevelDescriptions);
        }
        abilities = sortCostsFirst(abilities);

        // --- Build CardFace ---
        String name = AbilityDescription.applyCasing(face.getName());
        ManaCost manaCost = face.getManaCost();
        String manaCostStr = (manaCost == null || manaCost == ManaCost.NO_COST)
                ? null : manaCost.getSimpleString();

        String typeLine = formatTypeLine(face);
        String pt = (face.getPower() != null && face.getToughness() != null)
                ? face.getPower() + "/" + face.getToughness() : null;
        String loyalty = nullIfEmpty(face.getInitialLoyalty());
        String defense = nullIfEmpty(face.getDefense());

        String text = nullIfEmpty(face.getNonAbilityText());
        if (text != null) text = AbilityDescription.applyCasing(text);

        return new CardFace(name, manaCostStr, typeLine, pt, loyalty, defense, null, text, abilities);
    }

    // --- Undefined keyword routing ---

    private void routeUndefinedKeyword(KeywordInterface ki, List<Ability> abilities,
                                       Set<String> classLevelDescriptions) {
        String original = ki.getOriginal();

        if (original.startsWith("CARDNAME ") || original.startsWith("NICKNAME ")) {
            abilities.add(new TextAbility(AbilityType.STATIC, AbilityDescription.applyCasing(original)));
        } else if (original.startsWith("Chapter:")) {
            abilities.addAll(ChapterAbility.fromKeyword(ki));
        } else if (original.startsWith("Class:")) {
            ClassLevelAbility level = ClassLevelAbility.of(ki);
            if (level != null) {
                classLevelDescriptions.add(level.innerDescription());
                abilities.add(level);
            }
        } else if (original.startsWith("etbCounter:") || original.startsWith("ETBReplacement:")) {
            abilities.addAll(EtbReplacementAbility.fromKeyword(ki));
        } else if (original.startsWith("AlternateAdditionalCost:")) {
            String desc = buildAlternateAdditionalCostDescription(original);
            abilities.add(new TextAbility(AbilityType.ADDITIONAL_COST, AbilityDescription.applyCasing(desc)));
        } else {
            abilities.add(StandardKeyword.of(ki, Keyword.UNDEFINED));
        }
    }

    // --- Post-processing ---

    private List<Ability> applyClassPostProcessing(List<Ability> abilities,
                                                   Set<String> classLevelDescriptions) {
        List<Ability> result = new ArrayList<>(abilities);

        result.removeIf(a ->
                a.type() != AbilityType.LEVEL
                        && classLevelDescriptions.contains(a.descriptionText()));

        for (int i = 0; i < result.size(); i++) {
            Ability a = result.get(i);
            if (a.type() == AbilityType.STATIC || a.type() == AbilityType.TRIGGERED
                    || a.type() == AbilityType.REPLACEMENT) {
                result.set(i, new TextAbility(AbilityType.LEVEL, a.descriptionText(), 1));
            }
        }

        result.sort((a, b) -> {
            boolean aLevel = a.type() == AbilityType.LEVEL;
            boolean bLevel = b.type() == AbilityType.LEVEL;
            if (aLevel && bLevel) {
                return Integer.compare(a.ordinal(), b.ordinal());
            }
            if (aLevel) return -1;
            if (bLevel) return 1;
            return 0;
        });

        return result;
    }

    private List<Ability> sortCostsFirst(List<Ability> abilities) {
        List<Ability> result = new ArrayList<>(abilities);
        result.sort((a, b) -> {
            boolean aCost = a.type().isCostType();
            boolean bCost = b.type().isCostType();
            if (aCost == bCost) return 0;
            return aCost ? -1 : 1;
        });
        return result;
    }

    // --- Helpers ---

    private static String buildAlternateAdditionalCostDescription(String original) {
        String[] costParts = original.split(":", 2)[1].split(":");
        StringBuilder desc = new StringBuilder();
        for (int n = 0; n < costParts.length; n++) {
            Cost cost = new Cost(costParts[n], false);
            String costText = cost.toSimpleString();
            if (cost.isOnlyManaCost()) {
                desc.append("pay ");
            }
            desc.append(costText.substring(0, 1).toLowerCase()).append(costText.substring(1));
            if (n + 2 == costParts.length) {
                desc.append(costParts.length > 2 ? ", or " : " or ");
            } else if (n + 1 < costParts.length) {
                desc.append(", ");
            }
        }
        return desc.toString();
    }

    private static class DummyGameHolder {
        static final Game INSTANCE = createDummyGame();

        private static Game createDummyGame() {
            GameRules rules = new GameRules(GameType.Constructed);
            Match match = new Match(rules, List.of(), "DummyMatch");
            return new Game(List.of(), rules, match);
        }
    }

    private Card buildFullCard(CardRules rules) {
        PaperCard paperCard = new PaperCard(rules, "UNK", CardRarity.Common);
        return CardFactory.getCard(paperCard, null, nextCardId++, DummyGameHolder.INSTANCE);
    }

    private static final Map<String, String> LAND_TYPE_MANA = Map.of(
            "Plains",   "{W}",
            "Island",   "{U}",
            "Swamp",    "{B}",
            "Mountain", "{R}",
            "Forest",   "{G}"
    );

    static String buildLandManaDescription(ICardFace face) {
        List<String> symbols = new ArrayList<>();
        for (Map.Entry<String, String> entry : LAND_TYPE_MANA.entrySet()) {
            if (face.getType().hasSubtype(entry.getKey())) {
                symbols.add(entry.getValue());
            }
        }
        if (symbols.isEmpty()) {
            return null;
        }
        StringBuilder sb = new StringBuilder("{T}: add ");
        for (int i = 0; i < symbols.size(); i++) {
            sb.append(symbols.get(i));
            if (i + 2 == symbols.size()) {
                sb.append(" or ");
            } else if (i + 1 < symbols.size()) {
                sb.append(", ");
            }
        }
        return sb.toString();
    }

    private static String formatTypeLine(ICardFace face) {
        String typeStr = face.getType().toString();
        typeStr = typeStr.replace(" - ", " ");
        return typeStr.toLowerCase();
    }

    private static String nullIfEmpty(String s) {
        return (s == null || s.isEmpty()) ? null : s;
    }

    private static void addIfNotNull(List<Ability> list, Ability item) {
        if (item != null) list.add(item);
    }
}
