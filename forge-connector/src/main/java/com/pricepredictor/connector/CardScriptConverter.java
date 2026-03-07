package com.pricepredictor.connector;

import com.esotericsoftware.minlog.Log;
import forge.card.CardRules;
import forge.card.CardSplitType;
import forge.card.ICardFace;
import forge.card.mana.ManaCost;
import forge.game.ability.AbilityFactory;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.cost.Cost;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerHandler;
import forge.util.FileSection;
import forge.util.Lang;
import forge.util.Localizer;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
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
            ConvertedCard face = convertFace(rules.getMainPart(), buildSvarMap(rules.getMainPart()));
            return MultiCard.singleFace(face);
        }

        String layout = mapLayout(splitType);
        List<ConvertedCard> faces = new ArrayList<>();
        for (ICardFace face : rules.getAllFaces()) {
            faces.add(convertFace(face, buildSvarMap(face)));
        }
        return MultiCard.multiFace(layout, faces);
    }

    /**
     * Convert a single card face into a ConvertedCard.
     */
    public ConvertedCard convertFace(ICardFace face, Map<String, String> svars) {
        int actionCounter = 0;
        List<AbilityLine> abilities = new ArrayList<>();
        Card hostCard = createHostCard(face, svars);

        // --- Keywords ---
        for (String kw : face.getKeywords()) {
            if (kw.startsWith("Chapter:") || kw.startsWith("Class:") || kw.startsWith("etbCounter:")) {
                // Special K: lines handled separately
                List<AbilityLine> specialLines = handleSpecialKeyword(kw, svars, actionCounter);
                for (AbilityLine line : specialLines) {
                    if (line.actionNumber() != null && line.actionNumber() > actionCounter) {
                        actionCounter = line.actionNumber();
                    }
                    abilities.add(line);
                }
                continue;
            }
            // Plaintext K: line (starts with card name or similar text)
            if (!isStandardKeyword(kw)) {
                abilities.add(new AbilityLine(AbilityType.STATIC, applyTextCasing(kw), null));
                continue;
            }

            try {
                KeywordInterface ki = Keyword.getInstance(kw);
                String title = ki.getTitle();
                if (KeywordClassifier.isActivatable(ki.getKeyword().toString())) {
                    actionCounter++;
                    abilities.add(new AbilityLine(AbilityType.KEYWORD_ACTIVE,
                            applyTextCasing(title), actionCounter));
                } else {
                    abilities.add(new AbilityLine(AbilityType.KEYWORD_PASSIVE,
                            applyTextCasing(title), null));
                }
            } catch (Exception e) {
                // Fallback: treat as passive keyword
                abilities.add(new AbilityLine(AbilityType.KEYWORD_PASSIVE,
                        applyTextCasing(kw), null));
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
                    // Charm/modal spell — parent line + choices
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
                    if (sa.isPwAbility()) {
                        // Planeswalker: +N: desc → [+N]: desc
                        desc = desc.replaceFirst("^([+-]?\\d+):", "[$1]:");
                        actionCounter++;
                        abilities.add(new AbilityLine(AbilityType.PLANESWALKER,
                                applyTextCasing(desc), actionCounter));
                    } else {
                        actionCounter++;
                        abilities.add(new AbilityLine(AbilityType.ACTIVATED,
                                applyTextCasing(desc), actionCounter));
                    }
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

        // --- Triggers (T: lines) ---
        for (String trigger : face.getTriggers()) {
            try {
                Trigger trig = TriggerHandler.parseTrigger(trigger, hostCard, true);
                if ("True".equals(trig.getParam("Static"))
                        || "True".equals(trig.getParam("Secondary"))) {
                    continue;
                }
                String description = trig.getParam("TriggerDescription");
                if (description == null || description.isEmpty()) {
                    Log.warn("CardScriptConverter",
                            "[" + face.getName() + "] missing description for trigger");
                    continue;
                }
                description = stripReminderText(description);
                abilities.add(new AbilityLine(AbilityType.TRIGGERED,
                        applyTextCasing(description), null));
            } catch (Throwable e) {
                Log.warn("CardScriptConverter",
                        "[" + face.getName() + "] failed to parse trigger: " + e.getMessage());
            }
        }

        // --- Statics (S: lines) ---
        for (String staticAbility : face.getStaticAbilities()) {
            try {
                StaticAbility stab = StaticAbility.create(staticAbility, hostCard,
                        hostCard.getCurrentState(), true);
                if ("True".equals(stab.getParam("Secondary"))) {
                    continue;
                }
                String description = stab.getParam("Description");
                if (description == null || description.isEmpty()) {
                    Log.warn("CardScriptConverter",
                            "[" + face.getName() + "] missing description for static ability");
                    continue;
                }
                description = stripReminderText(description);
                abilities.add(new AbilityLine(AbilityType.STATIC,
                        applyTextCasing(description), null));
            } catch (Throwable e) {
                Log.warn("CardScriptConverter",
                        "[" + face.getName() + "] failed to parse static ability: " + e.getMessage());
            }
        }

        // --- Replacements (R: lines) ---
        for (String replacement : face.getReplacements()) {
            Map<String, String> params = FileSection.parseToMap(replacement,
                    FileSection.DOLLAR_SIGN_KV_SEPARATOR);
            String description = params.get("Description");
            if (description == null || description.isEmpty()) {
                Log.warn("CardScriptConverter",
                        "[" + face.getName() + "] missing description for replacement");
                continue;
            }
            description = stripReminderText(description);
            abilities.add(new AbilityLine(AbilityType.REPLACEMENT,
                    applyTextCasing(description), null));
        }

        // --- Build ConvertedCard ---
        String name = applyTextCasing(face.getName());
        ManaCost manaCost = face.getManaCost();
        String manaCostStr = (manaCost == null || manaCost == ManaCost.NO_COST)
                ? null : manaCost.getSimpleString();
        // ManaCost symbols should stay uppercase — getSimpleString() already produces them uppercase
        // but the rest of the cost string should be lowercased (rare edge case)

        String typeLine = formatTypeLine(face);
        String pt = (face.getPower() != null && face.getToughness() != null)
                ? face.getPower() + "/" + face.getToughness() : null;
        String loyalty = face.getInitialLoyalty();
        if (loyalty != null && loyalty.isEmpty()) loyalty = null;

        // Colors: only include if explicitly set (differs from mana cost derived color)
        String colors = null; // We skip explicit color override for now — rarely used

        String text = face.getNonAbilityText();
        if (text != null && text.isEmpty()) text = null;
        if (text != null) text = applyTextCasing(text);

        return new ConvertedCard(name, manaCostStr, typeLine, pt, loyalty, colors, text, abilities);
    }

    // --- Helper methods ---

    private static void ensureForgeInitialized() {
        try {
            // Initialize Localizer with a dummy ResourceBundle so ZoneType enum can load
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
            // Initialize Lang so AbilityFactory can format valid target descriptions
            if (Lang.getInstance() == null) {
                Lang.createInstance("en-US");
            }
        } catch (Exception ignored) {
        }
    }

    private Card createHostCard(ICardFace face, Map<String, String> svars) {
        Card card = new Card(0, null, null);
        card.setName(face.getName());
        CardState state = card.getCurrentState();
        for (Map.Entry<String, String> entry : svars.entrySet()) {
            state.setSVar(entry.getKey(), entry.getValue());
        }
        return card;
    }

    private Map<String, String> buildSvarMap(ICardFace face) {
        Map<String, String> svars = new HashMap<>();
        for (Map.Entry<String, String> entry : face.getVariables()) {
            svars.put(entry.getKey(), entry.getValue());
        }
        return svars;
    }

    private String formatTypeLine(ICardFace face) {
        // Get the full type string and lowercase it, removing the dash separator
        String typeStr = face.getType().toString();
        typeStr = typeStr.replace(" - ", " ");
        return typeStr.toLowerCase();
    }

    private String formatCost(String costStr) {
        if (costStr == null || costStr.isEmpty()) return "";
        try {
            Cost cost = new Cost(costStr, true);
            return cost.toSimpleString();
        } catch (Exception e) {
            return costStr;
        }
    }

    private boolean isStandardKeyword(String kw) {
        // Standard keywords don't start with card-name-like patterns
        // They follow patterns like "Flying", "Equip:2", "Protection from red"
        // Plaintext K: lines typically start with CARDNAME or a long phrase
        if (kw.startsWith("CARDNAME ") || kw.startsWith("NICKNAME ")) return false;
        // Check if Forge recognizes it
        try {
            Keyword.getInstance(kw);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private List<AbilityLine> handleSpecialKeyword(String kw, Map<String, String> svars, int startCounter) {
        List<AbilityLine> lines = new ArrayList<>();

        if (kw.startsWith("etbCounter:")) {
            // K:etbCounter:Type:N → triggered line
            String[] parts = kw.split(":");
            if (parts.length >= 3) {
                String counterType = parts[1].toLowerCase();
                String count = parts.length >= 4 ? parts[2] : "1";
                lines.add(new AbilityLine(AbilityType.TRIGGERED,
                        applyTextCasing("enters with " + count + " " + counterType +
                                (Integer.parseInt(count) > 1 ? " counters" : " counter")),
                        null));
            }
        } else if (kw.startsWith("Chapter:")) {
            // K:Chapter:N:SVar1,...,SvarN → triggered lines for each chapter
            String[] parts = kw.split(":");
            if (parts.length >= 3) {
                String svarList = parts[2];
                for (String svarName : svarList.split(",")) {
                    svarName = svarName.trim();
                    String svarValue = svars.get(svarName);
                    if (svarValue != null) {
                        Map<String, String> svarParams = FileSection.parseToMap(svarValue,
                                FileSection.DOLLAR_SIGN_KV_SEPARATOR);
                        String desc = svarParams.get("TriggerDescription");
                        if (desc == null) desc = svarParams.get("SpellDescription");
                        if (desc != null) {
                            desc = stripReminderText(desc);
                            lines.add(new AbilityLine(AbilityType.TRIGGERED,
                                    applyTextCasing(desc), null));
                        }
                    }
                }
            }
        } else if (kw.startsWith("Class:")) {
            // K:Class:level:cost:effect → activated lines for level 2+
            // Level 1 abilities are regular abilities on the face
            // For now, treat as activated with level cost
            String[] parts = kw.split(":");
            if (parts.length >= 4) {
                String level = parts[1];
                String cost = parts[2];
                String effect = parts.length >= 5 ? parts[3] : "";
                if (!"1".equals(level)) {
                    startCounter++;
                    String costDisplay = formatCost(cost);
                    lines.add(new AbilityLine(AbilityType.ACTIVATED,
                            applyTextCasing(costDisplay + ": level " + level),
                            startCounter));
                }
            }
        }

        return lines;
    }

    /**
     * Apply text casing rules: lowercase all text, then restore CARDNAME/NICKNAME/ALTERNATE
     * and brace symbols to uppercase.
     */
    static String applyTextCasing(String text) {
        if (text == null) return null;

        // First, protect brace symbols by extracting them
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

        // Restore placeholders to uppercase
        lowered = lowered.replace("cardname", "CARDNAME");
        lowered = lowered.replace("nickname", "NICKNAME");
        lowered = lowered.replace("alternate", "ALTERNATE");

        return lowered;
    }

    private String stripReminderText(String text) {
        if (text == null) return null;
        return REMINDER_TEXT.matcher(text).replaceAll("").trim();
    }

    private static String mapLayout(CardSplitType splitType) {
        return switch (splitType) {
            case Transform -> "transform";
            case Split -> "split";
            case Adventure -> "adventure";
            case Modal -> "modal";
            case Flip -> "flip";
            default -> splitType.name().toLowerCase();
        };
    }
}
