package com.pricepredictor.connector;

import com.esotericsoftware.minlog.Log;
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
import forge.game.keyword.KeywordInterface;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.item.PaperCard;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Converts Forge card scripts into LLM-friendly text format.
 */
public class CardScriptConverter {

    private static final Pattern BRACE_SYMBOL = Pattern.compile("\\{[^}]+\\}");
    private static final Pattern REMINDER_TEXT = Pattern.compile("\\s*\\([^)]*\\)");
    private static final Pattern PLACEHOLDER_WORD = Pattern.compile("\\b(cardname|nickname|alternate)\\b");

    private final CardRules.Reader reader = new CardRules.Reader();
    private int nextCardId = 1;

    /**
     * Parse a card script and convert all faces.
     */
    public MultiCard convertCard(List<String> scriptLines, String filename) {
        reader.reset();
        for (String line : scriptLines) {
            if (line.isEmpty() || line.charAt(0) == '#') continue;
            reader.parseLine(line);
        }
        CardRules rules = reader.getCard();
        return convertRules(rules);
    }

    /**
     * Convert parsed CardRules into a MultiCard.
     * Returns null for meld cards whose other part is not available (requires StaticData).
     */
    public MultiCard convertRules(CardRules rules) {
        CardSplitType splitType = rules.getSplitType();

        // Meld cards without other part crash in CardFactory.readCard (StaticData not initialized)
        if (splitType == CardSplitType.Meld && rules.getOtherPart() == null) {
            Log.info("CardScriptConverter", "Skipping meld card without other part: " + rules.getName());
            return null;
        }

        Card card = buildFullCard(rules);

        if (splitType == CardSplitType.None) {
            return MultiCard.singleFace(convertFace(card, rules.getMainPart()));
        }

        String layout = splitType.name().toLowerCase();
        List<ConvertedCard> faces = new ArrayList<>();

        // Main face state: LeftSplit for split cards, Original for everything else
        CardStateName mainState = (splitType == CardSplitType.Split)
                ? CardStateName.LeftSplit : CardStateName.Original;
        card.setState(mainState, false);
        faces.add(convertFace(card, rules.getMainPart()));

        // Other face(s)
        if (splitType == CardSplitType.Specialize) {
            for (Map.Entry<CardStateName, ICardFace> e : rules.getSpecializeParts().entrySet()) {
                if (e.getValue() != null) {
                    card.setState(e.getKey(), false);
                    faces.add(convertFace(card, e.getValue()));
                }
            }
        } else if (rules.getOtherPart() != null) {
            card.setState(splitType.getChangedStateName(), false);
            faces.add(convertFace(card, rules.getOtherPart()));
        }

        return MultiCard.multiFace(layout, faces);
    }

    /**
     * Convert a single card face into a ConvertedCard.
     * The Card must already be in the correct state for this face.
     */
    public ConvertedCard convertFace(Card card, ICardFace face) {
        int actionCounter = 0;
        List<AbilityLine> abilities = new ArrayList<>();

        // --- Keywords (from parsed KeywordInterface objects) ---
        for (KeywordInterface ki : card.getKeywords()) {
            String original = ki.getOriginal();

            // Plaintext K: lines starting with CARDNAME/NICKNAME are rules text, not keywords
            if (original.startsWith("CARDNAME ") || original.startsWith("NICKNAME ")) {
                abilities.add(new AbilityLine(AbilityType.STATIC, applyTextCasing(original), null));
                continue;
            }

            // Chapter keywords — emit trigger descriptions as chapter lines
            if (original.startsWith("Chapter:")) {
                for (Trigger t : ki.getTriggers()) {
                    if ("True".equals(t.getParam("Secondary"))) continue;
                    String desc = t.getParam("TriggerDescription");
                    if (desc != null && !desc.isEmpty()) {
                        desc = stripReminderText(desc);
                        abilities.add(new AbilityLine(AbilityType.CHAPTER, applyTextCasing(desc), null));
                    }
                }
                continue;
            }

            // Class keywords — emit level-up activated abilities (skip level 1)
            if (original.startsWith("Class:")) {
                for (SpellAbility sa : ki.getAbilities()) {
                    String desc = stripReminderText(sa.getDescription());
                    if (desc != null && !desc.isEmpty()) {
                        actionCounter++;
                        abilities.add(new AbilityLine(AbilityType.ACTIVATED, applyTextCasing(desc), actionCounter));
                    }
                }
                continue;
            }

            // etbCounter keywords — emit replacement descriptions
            if (original.startsWith("etbCounter:")) {
                for (ReplacementEffect re : ki.getReplacements()) {
                    String desc = re.getParam("Description");
                    if (desc != null && !desc.isEmpty()) {
                        abilities.add(new AbilityLine(AbilityType.REPLACEMENT, applyTextCasing(desc), null));
                    }
                }
                continue;
            }

            // Regular keywords — use trait system to classify as active or passive
            boolean activatable = !ki.getAbilities().isEmpty();
            String title = ki.getTitle();
            if (title == null || title.isEmpty()) {
                title = original;
            }

            if (activatable) {
                actionCounter++;
                abilities.add(new AbilityLine(AbilityType.KEYWORD_ACTIVE, applyTextCasing(title), actionCounter));
            } else {
                abilities.add(new AbilityLine(AbilityType.KEYWORD_PASSIVE, applyTextCasing(title), null));
            }
        }

        // --- Abilities (from A: lines, already parsed on the card) ---
        for (SpellAbility sa : card.getSpellAbilities()) {
            if (sa.getKeyword() != null) continue; // skip keyword-derived abilities
            String spellDesc = sa.getParam("SpellDescription");
            if (spellDesc == null || spellDesc.isEmpty()) continue;

            if (sa.getApi() == ApiType.Charm) {
                actionCounter++;
                String desc = stripReminderText(sa.getDescription());
                abilities.add(new AbilityLine(AbilityType.SPELL,
                        applyTextCasing(desc), actionCounter));
                var choices = sa.getAdditionalAbilityList("Choices");
                if (choices != null) {
                    for (var choice : choices) {
                        String choiceDesc = choice.getParam("SpellDescription");
                        if (choiceDesc == null) choiceDesc = "(no description)";
                        choiceDesc = stripReminderText(choiceDesc);
                        actionCounter++;
                        abilities.add(new AbilityLine(AbilityType.OPTION,
                                applyTextCasing(choiceDesc), actionCounter));
                    }
                }
            } else if (sa.isActivatedAbility()) {
                String desc = stripReminderText(sa.getDescription());
                actionCounter++;
                AbilityType type = sa.isPwAbility() ? AbilityType.PLANESWALKER : AbilityType.ACTIVATED;
                abilities.add(new AbilityLine(type, applyTextCasing(desc), actionCounter));
            } else if (sa.isSpell()) {
                actionCounter++;
                String desc = stripReminderText(sa.getDescription());
                abilities.add(new AbilityLine(AbilityType.SPELL,
                        applyTextCasing(desc), actionCounter));
            }
        }

        // --- Triggers, Statics, Replacements (from T:/S:/R: lines, already parsed on the card) ---
        addDescribedTraits(abilities, card.getTriggers(), "TriggerDescription", AbilityType.TRIGGERED, face.getName());
        addDescribedTraits(abilities, card.getStaticAbilities(), "Description", AbilityType.STATIC, face.getName());
        addDescribedTraits(abilities, card.getReplacementEffects(), "Description", AbilityType.REPLACEMENT, face.getName());

        // --- Build ConvertedCard ---
        String name = applyTextCasing(face.getName());
        ManaCost manaCost = face.getManaCost();
        String manaCostStr = (manaCost == null || manaCost == ManaCost.NO_COST)
                ? null : manaCost.getSimpleString();

        String typeLine = formatTypeLine(face);
        String pt = (face.getPower() != null && face.getToughness() != null)
                ? face.getPower() + "/" + face.getToughness() : null;
        String loyalty = face.getInitialLoyalty();
        if (loyalty != null && loyalty.isEmpty()) loyalty = null;

        String colors = null;

        String text = face.getNonAbilityText();
        if (text != null && text.isEmpty()) text = null;
        if (text != null) text = applyTextCasing(text);

        return new ConvertedCard(name, manaCostStr, typeLine, pt, loyalty, colors, text, abilities);
    }

    // --- Helper methods ---

    private void addDescribedTraits(List<AbilityLine> abilities,
                                    Iterable<? extends CardTraitBase> traits,
                                    String descParam, AbilityType type,
                                    String faceName) {
        for (CardTraitBase trait : traits) {
            if ("True".equals(trait.getParam("Secondary")) || "True".equals(trait.getParam("Static"))) continue;
            String desc = trait.getParam(descParam);
            if (desc == null || desc.isEmpty()) {
                Log.warn("CardScriptConverter", "[" + faceName + "] missing description for " + type.name().toLowerCase());
                continue;
            }
            abilities.add(new AbilityLine(type, applyTextCasing(stripReminderText(desc)), null));
        }
    }

    /**
     * Build a fully-parsed Card from CardRules using Forge's CardFactory.
     */
    private Card buildFullCard(CardRules rules) {
        PaperCard paperCard = new PaperCard(rules, "UNK", CardRarity.Common);
        return CardFactory.getCard(paperCard, null, nextCardId++, null);
    }

    private String formatTypeLine(ICardFace face) {
        String typeStr = face.getType().toString();
        typeStr = typeStr.replace(" - ", " ");
        return typeStr.toLowerCase();
    }

    /**
     * Apply text casing rules: lowercase all text, then restore CARDNAME/NICKNAME/ALTERNATE
     * and brace symbols to uppercase.
     */
    static String applyTextCasing(String text) {
        if (text == null) return null;

        // Protect brace symbols by extracting them
        List<String> braceSymbols = new ArrayList<>();
        Matcher m = BRACE_SYMBOL.matcher(text);
        StringBuilder sb = new StringBuilder();
        int lastEnd = 0;
        while (m.find()) {
            braceSymbols.add(m.group().toUpperCase());
            sb.append(text, lastEnd, m.start());
            sb.append("\0BRACE").append(braceSymbols.size() - 1).append('\0');
            lastEnd = m.end();
        }
        sb.append(text, lastEnd, text.length());

        // Lowercase everything
        String lowered = sb.toString().toLowerCase();

        // Restore brace symbols (uppercase)
        for (int i = 0; i < braceSymbols.size(); i++) {
            lowered = lowered.replace("\0brace" + i + "\0", braceSymbols.get(i));
        }

        // Restore placeholders to uppercase (word-boundary-aware to avoid corrupting substrings)
        lowered = PLACEHOLDER_WORD.matcher(lowered).replaceAll(mr -> mr.group().toUpperCase());

        return lowered;
    }

    private String stripReminderText(String text) {
        if (text == null) return null;
        return REMINDER_TEXT.matcher(text).replaceAll("").trim();
    }
}
