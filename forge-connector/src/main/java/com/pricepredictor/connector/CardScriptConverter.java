package com.pricepredictor.connector;

import com.esotericsoftware.minlog.Log;
import forge.card.CardRules;
import forge.card.CardSplitType;
import forge.card.ICardFace;
import forge.card.mana.ManaCost;
import forge.game.CardTraitBase;
import forge.game.ability.AbilityFactory;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardFactoryUtil;
import forge.game.card.CardState;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.replacement.ReplacementHandler;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
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
    private static final String[] ROMAN_NUMERALS = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"};

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
        Card hostCard = createHostCard(face);

        // --- Keywords ---
        for (String kw : face.getKeywords()) {
            if (kw.startsWith("Chapter:") || kw.startsWith("Class:") || kw.startsWith("etbCounter:")) {
                List<AbilityLine> specialLines = handleSpecialKeyword(kw, hostCard, actionCounter);
                for (AbilityLine line : specialLines) {
                    if (line.actionNumber() != null && line.actionNumber() > actionCounter) {
                        actionCounter = line.actionNumber();
                    }
                    abilities.add(line);
                }
                continue;
            }
            // Plaintext K: lines starting with CARDNAME/NICKNAME are rules text, not keywords
            if (kw.startsWith("CARDNAME ") || kw.startsWith("NICKNAME ")) {
                abilities.add(new AbilityLine(AbilityType.STATIC, applyTextCasing(kw), null));
                continue;
            }

            // Parse once — if Forge doesn't recognize it, treat as passive keyword text
            KeywordInterface ki;
            try {
                ki = Keyword.getInstance(kw);
            } catch (Exception e) {
                abilities.add(new AbilityLine(AbilityType.KEYWORD_PASSIVE,
                        applyTextCasing(kw), null));
                continue;
            }

            // Use Forge's trait system to determine if keyword creates activatable abilities
            boolean activatable = false;
            try {
                ki.createTraits(hostCard, true);
                activatable = !ki.getAbilities().isEmpty();
            } catch (Exception ignored) {
                // If trait creation fails (e.g., missing game context), default to passive
            }

            String title = ki.getTitle();
            if (activatable) {
                actionCounter++;
                abilities.add(new AbilityLine(AbilityType.KEYWORD_ACTIVE,
                        applyTextCasing(title), actionCounter));
            } else {
                abilities.add(new AbilityLine(AbilityType.KEYWORD_PASSIVE,
                        applyTextCasing(title), null));
            }
        }

        // --- Abilities (A: lines) ---
        for (String ability : face.getAbilities()) {
            try {
                SpellAbility sa = AbilityFactory.getAbility(ability, hostCard);
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

        // --- Triggers, Statics, Replacements ---
        extractTraitDescriptions(face.getTriggers(), "TriggerDescription",
                (raw, host) -> TriggerHandler.parseTrigger(raw, host, true),
                AbilityType.TRIGGERED, hostCard, face.getName(), abilities);

        extractTraitDescriptions(face.getStaticAbilities(), "Description",
                (raw, host) -> StaticAbility.create(raw, host, host.getCurrentState(), true),
                AbilityType.STATIC, hostCard, face.getName(), abilities);

        extractTraitDescriptions(face.getReplacements(), "Description",
                (raw, host) -> ReplacementHandler.parseReplacement(raw, host, true),
                AbilityType.REPLACEMENT, hostCard, face.getName(), abilities);

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

    // --- Trait description extraction (common pattern for T:, S:, R: lines) ---

    @FunctionalInterface
    private interface TraitParser {
        CardTraitBase parse(String raw, Card host) throws Exception;
    }

    private void extractTraitDescriptions(Iterable<String> rawLines, String descKey,
            TraitParser parser, AbilityType type, Card hostCard,
            String faceName, List<AbilityLine> abilities) {
        for (String raw : rawLines) {
            try {
                CardTraitBase trait = parser.parse(raw, hostCard);
                if ("True".equals(trait.getParam("Secondary"))
                        || "True".equals(trait.getParam("Static"))) {
                    continue;
                }
                String description = trait.getParam(descKey);
                if (description == null || description.isEmpty()) {
                    Log.warn("CardScriptConverter",
                            "[" + faceName + "] missing description for " + type.getOutputPrefix());
                    continue;
                }
                description = stripReminderText(description);
                abilities.add(new AbilityLine(type, applyTextCasing(description), null));
            } catch (Throwable e) {
                Log.warn("CardScriptConverter",
                        "[" + faceName + "] failed to parse " + type.getOutputPrefix() + ": " + e.getMessage());
            }
        }
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

    private Card createHostCard(ICardFace face) {
        Card card = new Card(0, null, null);
        card.setName(face.getName());
        CardState state = card.getCurrentState();
        for (Map.Entry<String, String> entry : face.getVariables()) {
            state.setSVar(entry.getKey(), entry.getValue());
        }
        return card;
    }

    private String formatTypeLine(ICardFace face) {
        String typeStr = face.getType().toString();
        typeStr = typeStr.replace(" - ", " ");
        return typeStr.toLowerCase();
    }

    private List<AbilityLine> handleSpecialKeyword(String kw, Card hostCard, int startCounter) {
        List<AbilityLine> lines = new ArrayList<>();

        if (kw.startsWith("etbCounter:")) {
            try {
                var re = CardFactoryUtil.makeEtbCounter(kw, hostCard.getCurrentState(), true);
                String desc = re.getParam("Description");
                if (desc != null && !desc.isEmpty()) {
                    lines.add(new AbilityLine(AbilityType.REPLACEMENT,
                            applyTextCasing(desc), null));
                }
            } catch (Exception e) {
                Log.warn("CardScriptConverter",
                        "[" + hostCard.getName() + "] failed to parse etbCounter: " + e.getMessage());
            }
        } else if (kw.startsWith("Chapter:")) {
            String[] parts = kw.split(":");
            if (parts.length >= 3) {
                String[] svarNames = parts[2].split(",");
                CardState state = hostCard.getCurrentState();
                for (int i = 0; i < svarNames.length; i++) {
                    String svarValue = state.getSVar(svarNames[i].trim());
                    if (svarValue == null) continue;
                    try {
                        String desc = extractDescriptionFromSVar(svarValue, hostCard);
                        if (desc != null) {
                            desc = stripReminderText(desc);
                            String roman = toRoman(i + 1);
                            lines.add(new AbilityLine(AbilityType.CHAPTER,
                                    applyTextCasing(roman + " \u2014 " + desc), null));
                        }
                    } catch (Exception e) {
                        Log.warn("CardScriptConverter",
                                "[" + hostCard.getName() + "] failed to parse chapter SVar: " + e.getMessage());
                    }
                }
            }
        } else if (kw.startsWith("Class:")) {
            String[] parts = kw.split(":");
            if (parts.length >= 4) {
                String level = parts[1];
                String cost = parts[2];
                if (!"1".equals(level)) {
                    startCounter++;
                    String costDisplay;
                    try {
                        costDisplay = new forge.game.cost.Cost(cost, true).toSimpleString();
                    } catch (Exception e) {
                        costDisplay = cost;
                    }
                    lines.add(new AbilityLine(AbilityType.ACTIVATED,
                            applyTextCasing(costDisplay + ": level " + level),
                            startCounter));
                }
            }
        }

        return lines;
    }

    /**
     * Parse a SVar value through the appropriate Forge parser and extract its description.
     * DB$/SP$/AB$ SVars are parsed as abilities; Mode$ SVars are parsed as triggers.
     */
    private String extractDescriptionFromSVar(String svarValue, Card hostCard) {
        if (svarValue.contains("Mode$")) {
            var trigger = TriggerHandler.parseTrigger(svarValue, hostCard, true);
            return trigger.getParam("TriggerDescription");
        }
        SpellAbility sa = AbilityFactory.getAbility(svarValue, hostCard);
        return sa.getParam("SpellDescription");
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

    private static String toRoman(int chapter) {
        if (chapter >= 1 && chapter <= ROMAN_NUMERALS.length) {
            return ROMAN_NUMERALS[chapter - 1];
        }
        return String.valueOf(chapter);
    }

    private String stripReminderText(String text) {
        if (text == null) return null;
        return REMINDER_TEXT.matcher(text).replaceAll("").trim();
    }
}
