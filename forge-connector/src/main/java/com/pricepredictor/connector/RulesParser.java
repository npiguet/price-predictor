package com.pricepredictor.connector;

import forge.card.CardRarity;
import forge.card.CardRules;
import forge.card.CardSplitType;
import forge.card.CardStateName;
import forge.card.ICardFace;
import forge.card.mana.ManaCost;
import forge.game.CardTraitBase;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.cost.Cost;
import forge.game.keyword.Companion;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;
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
import java.util.function.Function;

/**
 * Parses Forge card scripts into domain objects (MultiCard/CardFace/Ability).
 * The only class that touches Forge classes directly.
 */
public class RulesParser {

    private static final String ADDITIONAL_COST_PREFIX = "as an additional cost to cast this spell, ";

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
        ActionCounter counter = new ActionCounter(0);
        List<Ability> abilities = new ArrayList<>();
        boolean isClass = face.getType().toString().contains("Class");
        Set<String> classLevelDescriptions = new HashSet<>();

        // --- Keywords ---
        abilities.addAll(extractKeywordAbilities(card, face, counter, classLevelDescriptions));

        // --- Spell abilities (from A: lines) ---
        abilities.addAll(extractSpellAbilities(card, counter));

        // --- Triggers, Statics, Replacements (from T:/S:/R: lines) ---
        abilities.addAll(extractTraitAbilities(card.getTriggers(), "TriggerDescription", AbilityType.TRIGGERED));
        abilities.addAll(extractTraitAbilities(card.getStaticAbilities(), "Description", AbilityType.STATIC));
        abilities.addAll(extractTraitAbilities(card.getReplacementEffects(), "Description", AbilityType.REPLACEMENT));

        // --- Implicit mana abilities from basic land types ---
        abilities.addAll(extractLandManaAbilities(face, counter));

        // --- Class post-processing ---
        if (isClass) {
            abilities = applyClassPostProcessing(abilities, classLevelDescriptions);
        }

        // --- Sort costs first ---
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

    // --- Extraction methods ---

    private List<Ability> extractKeywordAbilities(Card card, ICardFace face, ActionCounter counter,
                                                  Set<String> classLevelDescriptions) {
        List<Ability> abilities = new ArrayList<>();
        for (KeywordInterface ki : card.getKeywords()) {
            Keyword kw = ki.getKeyword();

            if (kw == Keyword.UNDEFINED) {
                abilities.addAll(handleUndefinedKeyword(ki, counter, classLevelDescriptions));
                continue;
            }

            if (kw == Keyword.GIFT) {
                String giftTitle = ki.getTitle();
                for (SpellAbility sa : card.getSpellAbilities()) {
                    if (sa.hasAdditionalAbility("GiftAbility")) {
                        String giftDesc = sa.getAdditionalAbility("GiftAbility")
                                .getParam("GiftDescription");
                        if (giftDesc != null && !giftDesc.isEmpty()) {
                            giftTitle = giftTitle + " " + giftDesc;
                        }
                        break;
                    }
                }
                abilities.add(new Ability(AbilityType.STATIC, AbilityDescription.ofCased(giftTitle), null));
                continue;
            }

            if (kw == Keyword.COMPANION && ki instanceof Companion comp) {
                String compDesc = comp.getDescription();
                if (compDesc != null && !compDesc.isEmpty()) {
                    compDesc = AbilityDescription.stripReminderText(compDesc);
                }
                String compTitle = (compDesc != null && !compDesc.isEmpty())
                        ? ki.getTitle() + " \u2014 " + compDesc : ki.getTitle();
                abilities.add(new Ability(AbilityType.STATIC, AbilityDescription.ofCased(compTitle), null));
                continue;
            }

            boolean activatable = !ki.getAbilities().isEmpty();
            String title = ki.getTitle();
            if (title == null || title.isEmpty() || !title.trim().equals(title)) {
                title = ki.getOriginal();
            }
            if (AbilityType.isInternalKeyword(kw)) {
                String reminder = ki.getReminderText();
                if (reminder != null && !reminder.isEmpty()) {
                    title = reminder;
                }
            }

            if (kw == Keyword.KICKER) {
                String[] kickerParts = ki.getOriginal().split(":", 3);
                if (kickerParts.length >= 3) {
                    Cost cost2 = new Cost(kickerParts[2], false);
                    title = title + " and/or " + cost2.toSimpleString();
                }
            }

            if (kw == Keyword.AFFINITY) {
                String[] affinityParts = ki.getOriginal().split(":", 3);
                if (affinityParts.length >= 2) {
                    String typeName = affinityParts.length >= 3
                            ? affinityParts[2] : affinityParts[1];
                    title = "Affinity for " + typeName + "s";
                }
            }

            AbilityType kwType = AbilityType.classifyKeyword(kw, activatable, !ki.getTriggers().isEmpty());
            abilities.add(new Ability(kwType, AbilityDescription.ofCased(title), counter.nextIfActionable(kwType)));
        }
        return abilities;
    }

    private List<Ability> extractSpellAbilities(Card card, ActionCounter counter) {
        List<Ability> abilities = new ArrayList<>();
        for (SpellAbility sa : card.getSpellAbilities()) {
            if (sa.getKeyword() != null) {
                continue;
            }

            if (sa.getApi() == ApiType.Charm) {
                abilities.addAll(extractCharmAbility(sa, counter));
                continue;
            }

            if (sa.isSpell() && "True".equals(sa.getParam("NonBasicSpell"))) {
                String precost = sa.getParam("PrecostDesc");
                String costDesc = sa.getParam("CostDesc");
                if (precost != null && costDesc != null) {
                    abilities.add(new Ability(AbilityType.ALTERNATE_COST,
                            AbilityDescription.ofCased(precost + " " + costDesc), null));
                }
                continue;
            }

            if (sa.isSpell()) {
                String costDesc = sa.getCostDescription();
                if (costDesc != null) {
                    costDesc = costDesc.trim();
                    costDesc = AbilityDescription.stripReminderText(costDesc);
                    if (costDesc != null && !costDesc.isEmpty()) {
                        String normalized = AbilityDescription.applyCasing(costDesc);
                        if (normalized.startsWith(ADDITIONAL_COST_PREFIX)) {
                            normalized = normalized.substring(ADDITIONAL_COST_PREFIX.length());
                        }
                        if (!normalized.isEmpty()) {
                            abilities.add(new Ability(AbilityType.ADDITIONAL_COST,
                                    new AbilityDescription(normalized), null));
                        }
                    }
                }
            }

            if (sa.isActivatedAbility()) {
                if (sa.getParam("SpellDescription") == null
                        || sa.getParam("SpellDescription").isEmpty()) {
                    continue;
                }
                String desc = sa.getDescription();
                SpellAbility sub = sa.getSubAbility();
                while (sub != null) {
                    String subDesc = sub.getParam("SpellDescription");
                    if (subDesc != null && !subDesc.isEmpty()) {
                        desc = desc + " " + subDesc;
                    }
                    sub = sub.getSubAbility();
                }
                AbilityType type = sa.isPwAbility() ? AbilityType.PLANESWALKER : AbilityType.ACTIVATED;
                AbilityDescription abilityDesc = AbilityDescription.of(desc).withTypeFormatting(type);
                abilities.add(new Ability(type, abilityDesc, counter.next()));
            } else if (sa.isSpell()) {
                List<String> descs = collectParamInChain(sa, "SpellDescription");
                for (String spellDesc : descs) {
                    String stripped = AbilityDescription.stripReminderText(spellDesc);
                    if (stripped == null || stripped.isEmpty()) {
                        continue;
                    }
                    abilities.add(new Ability(AbilityType.SPELL,
                            AbilityDescription.ofCased(stripped), counter.next()));
                }
            }
        }
        return abilities;
    }

    /**
     * Extract a charm ability with its choices modeled as sub-abilities.
     */
    private List<Ability> extractCharmAbility(SpellAbility sa, ActionCounter counter) {
        String charmDesc = sa.getParam("SpellDescription");
        if (charmDesc != null && !charmDesc.isEmpty()) {
            charmDesc = AbilityDescription.stripReminderText(charmDesc);
        }
        if ((charmDesc == null || charmDesc.isEmpty()) && sa.hasParam("Pawprint")) {
            String total = sa.getParam("Pawprint");
            charmDesc = "Choose up to " + total + " {P} worth of modes.";
            if ("True".equals(sa.getParam("CanRepeatModes"))) {
                charmDesc += " You may choose the same mode more than once.";
            }
        }

        // Collect charm choices as sub-abilities
        List<Ability> choiceSubs = new ArrayList<>();
        var choices = sa.getAdditionalAbilityList("Choices");
        if (choices != null) {
            for (var choice : choices) {
                String choiceDesc = findParamInChain(choice, "SpellDescription");
                if (choiceDesc != null) {
                    choiceDesc = AbilityDescription.stripReminderText(choiceDesc);
                }
                String pawprint = choice.getParam("Pawprint");
                if (pawprint != null) {
                    choiceDesc = "{P}".repeat(Integer.parseInt(pawprint))
                            + " \u2014 " + choiceDesc;
                }
                choiceSubs.add(new Ability(AbilityType.OPTION,
                        AbilityDescription.ofCased(choiceDesc), counter.next()));
            }
        }

        List<Ability> result = new ArrayList<>();
        if (charmDesc != null && !charmDesc.isEmpty()) {
            result.add(new Ability(AbilityType.SPELL,
                    AbilityDescription.ofCased(charmDesc), counter.next(), choiceSubs));
        } else {
            // No charm description — add choices as top-level abilities
            result.addAll(choiceSubs);
        }
        return result;
    }

    private List<Ability> extractTraitAbilities(Iterable<? extends CardTraitBase> traits,
                                                String descParam, AbilityType type) {
        List<Ability> abilities = new ArrayList<>();
        Set<String> seenDescriptions = new HashSet<>();
        for (CardTraitBase trait : traits) {
            if (trait.getKeyword() != null) {
                continue;
            }
            AbilityType effectiveType = type;
            if ("True".equals(trait.getParam("Static"))) {
                effectiveType = AbilityType.REPLACEMENT;
            }
            String raw = trait.getParam(descParam);
            if (raw == null || raw.isEmpty()) {
                continue;
            }
            String normalized = AbilityDescription.applyCasing(AbilityDescription.stripReminderText(raw));
            if (normalized.isEmpty()) {
                continue;
            }
            String mode = trait.getParam("Mode");
            if (type == AbilityType.STATIC
                    && ("RaiseCost".equals(mode) || "OptionalCost".equals(mode))) {
                String validCard = trait.getParam("ValidCard");
                String affected = trait.getParam("Affected");
                boolean selfCost = "Card.Self".equals(validCard) || "Card.Self".equals(affected);
                if (selfCost) {
                    effectiveType = AbilityType.ADDITIONAL_COST;
                    if (normalized.startsWith(ADDITIONAL_COST_PREFIX)) {
                        normalized = normalized.substring(ADDITIONAL_COST_PREFIX.length());
                    }
                }
            }
            if (!seenDescriptions.add(normalized)) {
                continue;
            }
            abilities.add(new Ability(effectiveType, new AbilityDescription(normalized), null));
        }
        return abilities;
    }

    private List<Ability> extractLandManaAbilities(ICardFace face, ActionCounter counter) {
        String desc = buildLandManaDescription(face);
        if (desc == null) {
            return List.of();
        }
        return List.of(new Ability(AbilityType.ACTIVATED, new AbilityDescription(desc), counter.next()));
    }

    // --- Post-processing ---

    private List<Ability> applyClassPostProcessing(List<Ability> abilities,
                                                   Set<String> classLevelDescriptions) {
        List<Ability> result = new ArrayList<>(abilities);

        result.removeIf(a ->
                a.type() != AbilityType.LEVEL
                        && classLevelDescriptions.contains(a.description().text()));

        for (int i = 0; i < result.size(); i++) {
            Ability a = result.get(i);
            if (a.type() == AbilityType.STATIC || a.type() == AbilityType.TRIGGERED
                    || a.type() == AbilityType.REPLACEMENT) {
                result.set(i, new Ability(AbilityType.LEVEL, a.description(), 1));
            }
        }

        result.sort((a, b) -> {
            boolean aLevel = a.type() == AbilityType.LEVEL;
            boolean bLevel = b.type() == AbilityType.LEVEL;
            if (aLevel && bLevel) {
                return Integer.compare(a.actionNumber(), b.actionNumber());
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

    // --- Private helpers ---

    private List<Ability> handleUndefinedKeyword(KeywordInterface ki, ActionCounter counter,
                                                 Set<String> classLevelDescriptions) {
        List<Ability> abilities = new ArrayList<>();
        String original = ki.getOriginal();

        if (original.startsWith("CARDNAME ") || original.startsWith("NICKNAME ")) {
            abilities.add(new Ability(AbilityType.STATIC, AbilityDescription.ofCased(original), null));
            return abilities;
        }

        if (original.startsWith("Chapter:")) {
            abilities.addAll(emitKeywordTraits(ki.getTriggers(), t -> t.getParam("TriggerDescription"),
                    AbilityType.CHAPTER));
            return abilities;
        }
        if (original.startsWith("Class:")) {
            emitClassLevel(ki, abilities, classLevelDescriptions);
            return abilities;
        }
        if (original.startsWith("etbCounter:") || original.startsWith("ETBReplacement:")) {
            abilities.addAll(emitAllKeywordTraits(ki.getReplacements(), t -> t.getParam("Description"),
                    AbilityType.REPLACEMENT));
            return abilities;
        }
        if (original.startsWith("AlternateAdditionalCost:")) {
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
            abilities.add(new Ability(AbilityType.ADDITIONAL_COST,
                    AbilityDescription.ofCased(desc.toString()), null));
            return abilities;
        }

        boolean activatable = !ki.getAbilities().isEmpty();
        String title = ki.getTitle();
        if (title == null || title.isEmpty()) {
            title = original;
        }
        AbilityType type = AbilityType.classifyKeyword(Keyword.UNDEFINED, activatable, !ki.getTriggers().isEmpty());
        abilities.add(new Ability(type, AbilityDescription.ofCased(title), counter.nextIfActionable(type)));
        return abilities;
    }

    private <T extends CardTraitBase> List<Ability> emitKeywordTraits(
            Iterable<T> traits, Function<T, String> descExtractor, AbilityType type) {
        return emitKeywordTraitsImpl(traits, descExtractor, type, true);
    }

    private <T extends CardTraitBase> List<Ability> emitAllKeywordTraits(
            Iterable<T> traits, Function<T, String> descExtractor, AbilityType type) {
        return emitKeywordTraitsImpl(traits, descExtractor, type, false);
    }

    private <T extends CardTraitBase> List<Ability> emitKeywordTraitsImpl(
            Iterable<T> traits, Function<T, String> descExtractor,
            AbilityType type, boolean skipSecondary) {
        List<Ability> abilities = new ArrayList<>();
        for (T trait : traits) {
            if (skipSecondary && "True".equals(trait.getParam("Secondary"))) {
                continue;
            }
            String desc = descExtractor.apply(trait);
            if (desc == null || desc.isEmpty()) {
                continue;
            }
            String normalized = AbilityDescription.applyCasing(AbilityDescription.stripReminderText(desc));
            if (normalized.isEmpty()) {
                continue;
            }
            AbilityDescription abilityDesc = new AbilityDescription(type.formatDescription(normalized));
            abilities.add(new Ability(type, abilityDesc, null));
        }
        return abilities;
    }

    private void emitClassLevel(KeywordInterface ki, List<Ability> abilities,
                                Set<String> classLevelDescriptions) {
        String original = ki.getOriginal();
        int level = Integer.parseInt(original.split(":", 3)[1]);

        var it = ki.getAbilities().iterator();
        String cost = it.hasNext() ? it.next().getCostDescription() : null;
        if (cost == null || cost.isEmpty()) {
            return;
        }
        cost = cost.trim();
        if (cost.endsWith(":")) {
            cost = cost.substring(0, cost.length() - 1).trim();
        }

        String rawDesc = findFirstDescription(ki.getTriggers(), "TriggerDescription");
        if (rawDesc == null) {
            rawDesc = findFirstDescription(ki.getStaticAbilities(), "Description");
        }
        if (rawDesc == null) {
            rawDesc = findFirstDescription(ki.getReplacements(), "Description");
        }
        if (rawDesc == null) {
            return;
        }

        AbilityDescription desc = AbilityDescription.of(rawDesc);
        classLevelDescriptions.add(desc.text());
        String casedCost = AbilityDescription.applyCasing(cost);
        abilities.add(new Ability(AbilityType.LEVEL,
                new AbilityDescription(casedCost + ": " + desc.text()), level));
    }

    private <T extends CardTraitBase> String findFirstDescription(Iterable<T> traits, String param) {
        for (T trait : traits) {
            String d = trait.getParam(param);
            if (d != null && !d.isEmpty()) {
                return d;
            }
        }
        return null;
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

    private static String buildLandManaDescription(ICardFace face) {
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

    private static List<String> collectParamInChain(SpellAbility sa, String param) {
        List<String> values = new ArrayList<>();
        String value = sa.getParam(param);
        if (value != null && !value.isEmpty()) {
            values.add(value);
        }
        SpellAbility sub = sa.getSubAbility();
        while (sub != null) {
            value = sub.getParam(param);
            if (value != null && !value.isEmpty()) {
                values.add(value);
            }
            sub = sub.getSubAbility();
        }
        return values;
    }

    private static String findParamInChain(SpellAbility sa, String param) {
        String value = sa.getParam(param);
        if (value != null && !value.isEmpty()) {
            return value;
        }
        SpellAbility sub = sa.getSubAbility();
        while (sub != null) {
            value = sub.getParam(param);
            if (value != null && !value.isEmpty()) {
                return value;
            }
            sub = sub.getSubAbility();
        }
        return null;
    }
}
