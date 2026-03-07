package com.pricepredictor.connector;

import com.esotericsoftware.minlog.Log;
import forge.card.CardRules;
import forge.card.CardSplitType;
import forge.card.CardType;
import forge.card.ICardFace;
import forge.card.mana.ManaCost;
import forge.game.ability.AbilityFactory;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardFactoryUtil;
import forge.game.keyword.KeywordInterface;
import forge.game.replacement.ReplacementEffect;
import forge.game.replacement.ReplacementHandler;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerHandler;
import forge.util.Lang;
import forge.util.Localizer;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.ResourceBundle;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Converts Forge card scripts into LLM-friendly text format.
 */
public class CardScriptConverter {

    private static final Pattern BRACE_SYMBOL = Pattern.compile("\\{[^}]+\\}");
    private static final Pattern REMINDER_TEXT = Pattern.compile("\\s*\\([^)]*\\)");
    private static final Pattern PLACEHOLDER_WORD = Pattern.compile("\\b(cardname|nickname|alternate)\\b");

    static {
        ensureForgeInitialized();
    }

    private final CardRules.Reader reader = new CardRules.Reader();

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
     */
    public MultiCard convertRules(CardRules rules) {
        CardSplitType splitType = rules.getSplitType();

        if (splitType == CardSplitType.None) {
            ConvertedCard face = convertFace(rules.getMainPart());
            return MultiCard.singleFace(face);
        }

        String layout = splitType.name().toLowerCase();
        List<ConvertedCard> faces = new ArrayList<>();
        for (ICardFace face : rules.getAllFaces()) {
            faces.add(convertFace(face));
        }
        return MultiCard.multiFace(layout, faces);
    }

    /**
     * Convert a single card face into a ConvertedCard.
     */
    public ConvertedCard convertFace(ICardFace face) {
        int actionCounter = 0;
        List<AbilityLine> abilities = new ArrayList<>();
        Card card = buildCard(face);

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

        // --- Abilities (A: lines) ---
        for (String ability : face.getAbilities()) {
            try {
                SpellAbility sa = AbilityFactory.getAbility(ability, card);
                String spellDesc = sa.getParam("SpellDescription");
                if (spellDesc == null || spellDesc.isEmpty()) {
                    continue;
                }

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
            } catch (Throwable e) {
                Log.warn("CardScriptConverter",
                        "[" + face.getName() + "] failed to parse ability: " + e.getMessage());
            }
        }

        // --- Triggers (from T: lines, already parsed on the card) ---
        for (Trigger t : card.getTriggers()) {
            if ("True".equals(t.getParam("Secondary")) || "True".equals(t.getParam("Static"))) continue;
            String desc = t.getParam("TriggerDescription");
            if (desc == null || desc.isEmpty()) {
                Log.warn("CardScriptConverter", "[" + face.getName() + "] missing description for triggered");
                continue;
            }
            abilities.add(new AbilityLine(AbilityType.TRIGGERED, applyTextCasing(stripReminderText(desc)), null));
        }

        // --- Statics (from S: lines, already parsed on the card) ---
        for (StaticAbility sa : card.getStaticAbilities()) {
            if ("True".equals(sa.getParam("Secondary")) || "True".equals(sa.getParam("Static"))) continue;
            String desc = sa.getParam("Description");
            if (desc == null || desc.isEmpty()) {
                Log.warn("CardScriptConverter", "[" + face.getName() + "] missing description for static");
                continue;
            }
            abilities.add(new AbilityLine(AbilityType.STATIC, applyTextCasing(stripReminderText(desc)), null));
        }

        // --- Replacements (from R: lines, already parsed on the card) ---
        for (ReplacementEffect re : card.getReplacementEffects()) {
            if ("True".equals(re.getParam("Secondary")) || "True".equals(re.getParam("Static"))) continue;
            String desc = re.getParam("Description");
            if (desc == null || desc.isEmpty()) {
                Log.warn("CardScriptConverter", "[" + face.getName() + "] missing description for replacement");
                continue;
            }
            abilities.add(new AbilityLine(AbilityType.REPLACEMENT, applyTextCasing(stripReminderText(desc)), null));
        }

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

    private static void ensureForgeInitialized() {
        try {
            Localizer localizer = Localizer.getInstance();
            Field rbField = Localizer.class.getDeclaredField("resourceBundle");
            rbField.setAccessible(true);
            if (rbField.get(localizer) == null) {
                ResourceBundle dummyBundle = new ResourceBundle() {
                    @Override
                    protected Object handleGetObject(String key) { return key; }
                    @Override
                    public Enumeration<String> getKeys() { return Collections.emptyEnumeration(); }
                };
                rbField.set(localizer, dummyBundle);
                Field ebField = Localizer.class.getDeclaredField("englishBundle");
                ebField.setAccessible(true);
                if (ebField.get(localizer) == null) {
                    ebField.set(localizer, dummyBundle);
                }
            }
        } catch (Exception ignored) {
        }
        try {
            if (Lang.getInstance() == null) {
                Lang.createInstance("en-US");
            }
        } catch (Exception ignored) {
        }
    }

    /**
     * Build a fully-populated Card from an ICardFace, replicating the essential
     * steps of CardFactory.readCardFace() + CardFactoryUtil.setupKeywordedAbilities().
     */
    private Card buildCard(ICardFace face) {
        Card card = new Card(0, null, null);
        card.setName(face.getName());

        // Type, mana cost, P/T, loyalty — needed for trait resolution
        card.setType(new CardType(face.getType()));
        card.setManaCost(face.getManaCost());
        if (face.getIntPower() != Integer.MAX_VALUE) {
            card.setBasePower(face.getIntPower());
            card.setBasePowerString(face.getPower());
        }
        if (face.getIntToughness() != Integer.MAX_VALUE) {
            card.setBaseToughness(face.getIntToughness());
            card.setBaseToughnessString(face.getToughness());
        }
        card.getCurrentState().setBaseLoyalty(face.getInitialLoyalty());
        card.getCurrentState().setBaseDefense(face.getDefense());

        // SVars first — keywords and abilities reference them
        for (Map.Entry<String, String> v : face.getVariables()) {
            card.setSVar(v.getKey(), v.getValue());
        }

        // Parsed traits — same order as CardFactory.readCardFace()
        for (String r : face.getReplacements())
            card.addReplacementEffect(ReplacementHandler.parseReplacement(r, card, true));
        for (String s : face.getStaticAbilities())
            card.addStaticAbility(s);
        for (String t : face.getTriggers())
            card.addTrigger(TriggerHandler.parseTrigger(t, card, true));

        // Keywords after SVars (Forge requirement), no traits yet
        // Use CardState directly to avoid Card.updateKeywords() which triggers CardView updates
        card.getCurrentState().addIntrinsicKeywords(face.getKeywords(), false);
        card.updateKeywordsCache();

        // Abilities (A: lines) — parse and add without triggering view updates
        for (String rawAbility : face.getAbilities()) {
            try {
                SpellAbility sa = AbilityFactory.getAbility(rawAbility, card);
                card.addSpellAbility(sa, false);
                sa.setIntrinsic(true);
                sa.setCardState(card.getCurrentState());
            } catch (Exception e) {
                Log.warn("CardScriptConverter",
                        "[" + face.getName() + "] failed to parse ability in buildCard: " + e.getMessage());
            }
        }

        // Create keyword traits (Chapter triggers, Cycling abilities, etbCounter replacements, etc.)
        CardFactoryUtil.setupKeywordedAbilities(card);

        card.setText(face.getNonAbilityText());
        return card;
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
