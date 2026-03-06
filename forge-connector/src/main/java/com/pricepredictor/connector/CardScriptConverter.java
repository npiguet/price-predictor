package com.pricepredictor.connector;

import com.esotericsoftware.minlog.Log;
import forge.card.CardRules;
import forge.card.CardSplitType;
import forge.card.ICardFace;
import forge.card.mana.ManaCost;
import forge.game.cost.Cost;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.util.FileSection;

import java.util.ArrayList;
import java.util.HashMap;
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
            Map<String, String> params = FileSection.parseToMap(ability,
                    FileSection.DOLLAR_SIGN_KV_SEPARATOR);

            boolean isPlaneswalker = "True".equals(params.get("Planeswalker"));
            String costStr = params.get("Cost");
            String description = params.get("SpellDescription");

            if (description == null || description.isEmpty()) {
                // SP$ PermanentCreature and similar engine-internal abilities have no description
                // — skip them rather than producing placeholder output
                continue;
            }

            description = stripReminderText(description);

            if (ability.startsWith("AB$ ") || (ability.contains("AB$") && ability.indexOf("AB$") < 5)) {
                if (isPlaneswalker && costStr != null) {
                    // Planeswalker loyalty ability
                    String loyaltyPrefix = extractLoyaltyPrefix(costStr);
                    actionCounter++;
                    abilities.add(new AbilityLine(AbilityType.PLANESWALKER,
                            applyTextCasing(loyaltyPrefix + ": " + description), actionCounter));
                } else {
                    // Regular activated ability
                    String costDisplay = formatCost(costStr);
                    actionCounter++;
                    abilities.add(new AbilityLine(AbilityType.ACTIVATED,
                            applyTextCasing(costDisplay + ": " + description), actionCounter));
                }
            } else if (ability.startsWith("SP$ Charm") || ability.contains("SP$ Charm")) {
                // Charm/modal spell — parent line
                actionCounter++;
                String chooseText = extractChooseText(params);
                abilities.add(new AbilityLine(AbilityType.SPELL,
                        applyTextCasing(chooseText), actionCounter));
                // Resolve choices
                String choices = params.get("Choices");
                if (choices != null) {
                    for (String svarName : choices.split(",")) {
                        svarName = svarName.trim();
                        String choiceSvar = svars.get(svarName);
                        if (choiceSvar != null) {
                            Map<String, String> choiceParams = FileSection.parseToMap(choiceSvar,
                                    FileSection.DOLLAR_SIGN_KV_SEPARATOR);
                            String choiceDesc = choiceParams.get("SpellDescription");
                            if (choiceDesc == null) choiceDesc = "(no description)";
                            choiceDesc = stripReminderText(choiceDesc);
                            actionCounter++;
                            abilities.add(new AbilityLine(AbilityType.OPTION,
                                    applyTextCasing(choiceDesc), actionCounter));
                        }
                    }
                }
            } else if (ability.startsWith("SP$ ") || ability.contains("SP$")) {
                // Spell effect
                actionCounter++;
                abilities.add(new AbilityLine(AbilityType.SPELL,
                        applyTextCasing(description), actionCounter));
            }
        }

        // --- Triggers (T: lines) ---
        for (String trigger : face.getTriggers()) {
            Map<String, String> params = FileSection.parseToMap(trigger,
                    FileSection.DOLLAR_SIGN_KV_SEPARATOR);
            if ("True".equals(params.get("Static")) || "True".equals(params.get("Secondary"))) {
                continue;
            }
            String description = params.get("TriggerDescription");
            if (description == null || description.isEmpty()) {
                Log.warn("CardScriptConverter",
                        "[" + face.getName() + "] missing description for trigger");
                continue;
            }
            description = stripReminderText(description);
            abilities.add(new AbilityLine(AbilityType.TRIGGERED,
                    applyTextCasing(description), null));
        }

        // --- Statics (S: lines) ---
        for (String staticAbility : face.getStaticAbilities()) {
            Map<String, String> params = FileSection.parseToMap(staticAbility,
                    FileSection.DOLLAR_SIGN_KV_SEPARATOR);
            if ("True".equals(params.get("Secondary"))) {
                continue;
            }
            String description = params.get("Description");
            if (description == null || description.isEmpty()) {
                Log.warn("CardScriptConverter",
                        "[" + face.getName() + "] missing description for static ability");
                continue;
            }
            description = stripReminderText(description);
            abilities.add(new AbilityLine(AbilityType.STATIC,
                    applyTextCasing(description), null));
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

    private String extractLoyaltyPrefix(String costStr) {
        // Parse loyalty cost: AddCounter<N/LOYALTY> -> [+N], SubCounter<N/LOYALTY> -> [-N]
        if (costStr.contains("AddCounter<")) {
            int start = costStr.indexOf("AddCounter<") + "AddCounter<".length();
            int slash = costStr.indexOf('/', start);
            if (slash > start) {
                String n = costStr.substring(start, slash);
                return "[+" + n + "]";
            }
        }
        if (costStr.contains("SubCounter<")) {
            int start = costStr.indexOf("SubCounter<") + "SubCounter<".length();
            int slash = costStr.indexOf('/', start);
            if (slash > start) {
                String n = costStr.substring(start, slash);
                return "[-" + n + "]";
            }
        }
        // Fallback: try to format via Cost class
        return "[0]";
    }

    private String extractChooseText(Map<String, String> params) {
        // Default "choose one —"; could be "choose two —" etc.
        String chooseNum = params.get("ChNum");
        if (chooseNum != null) {
            try {
                int n = Integer.parseInt(chooseNum);
                String word = switch (n) {
                    case 1 -> "one";
                    case 2 -> "two";
                    case 3 -> "three";
                    case 4 -> "four";
                    default -> String.valueOf(n);
                };
                return "choose " + word + " —";
            } catch (NumberFormatException ignored) {}
        }
        return "choose one —";
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
