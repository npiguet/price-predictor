package com.pricepredictor.connector;

import com.esotericsoftware.minlog.Log;
import forge.StaticData;
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
import forge.item.PaperCard;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Converts Forge card scripts into LLM-friendly text format.
 */
public class CardScriptConverter {

    private static final Pattern BRACE_SYMBOL = Pattern.compile("\\{[^}]+\\}");
    private static final Pattern REMINDER_TEXT = Pattern.compile("\\s*\\([^)]*\\)");
    private static final Pattern PLACEHOLDER_WORD = Pattern.compile("\\b(cardname|nickname|alternate)\\b");
    private static final Pattern VARIABLE_X = Pattern.compile("(?<![a-z])x(?![a-z])");
    private static final Pattern ROMAN_PREFIX = Pattern.compile("^([ivxlcdm]+(?:,\\s*[ivxlcdm]+)*)\\s*\u2014");
    private static final Pattern LOYALTY_COST = Pattern.compile("^([+\\-](?:\\d+|X)): ", Pattern.CASE_INSENSITIVE);
    private static final String ADDITIONAL_COST_PREFIX = "As an additional cost to cast this spell, ";

    // Keywords that represent alternate casting costs (replace the mana cost)
    // or alternate deployment (put onto battlefield / create token from graveyard)
    private static final Set<Keyword> ALTERNATE_COST_KEYWORDS = EnumSet.of(
            Keyword.BESTOW, Keyword.BLITZ, Keyword.DASH, Keyword.DISTURB,
            Keyword.EMERGE, Keyword.ESCAPE, Keyword.EVOKE, Keyword.FLASHBACK,
            Keyword.FORETELL, Keyword.FREERUNNING, Keyword.MADNESS,
            Keyword.MEGAMORPH, Keyword.MORPH, Keyword.DISGUISE,
            Keyword.MORE_THAN_MEETS_THE_EYE, Keyword.MUTATE,
            Keyword.OVERLOAD, Keyword.PLOT, Keyword.PROTOTYPE,
            Keyword.PROWL, Keyword.RETRACE, Keyword.SNEAK,
            Keyword.SPECTACLE, Keyword.SURGE,
            Keyword.AWAKEN,       // cast for awaken cost → also animates a land
            Keyword.EMBALM,       // create token copy from graveyard
            Keyword.ENCORE,       // create token copies from graveyard
            Keyword.ETERNALIZE,   // create 4/4 token copy from graveyard
            Keyword.HARMONIZE,    // cast from graveyard for harmonize cost
            Keyword.IMPENDING,    // cast for impending cost (enters with time counters)
            Keyword.JUMP_START,   // cast from graveyard (discard a card)
            Keyword.MIRACLE,      // cast for miracle cost when drawn
            Keyword.NINJUTSU,     // put onto battlefield from hand
            Keyword.OFFERING,     // sacrifice a [type] + pay the mana difference
            Keyword.SUSPEND,      // pay cost to exile with time counters, then free-cast
            Keyword.UNEARTH       // return from graveyard to battlefield
    );

    // Keywords that represent additional casting costs (paid on top of mana cost)
    private static final Set<Keyword> ADDITIONAL_COST_KEYWORDS = EnumSet.of(
            Keyword.BUYBACK, Keyword.ENTWINE, Keyword.KICKER,
            Keyword.MULTIKICKER, Keyword.REPLICATE, Keyword.SQUAD,
            Keyword.STRIVE, Keyword.CASUALTY, Keyword.OFFSPRING,
            Keyword.BARGAIN,      // sacrifice artifact/enchantment/token
            Keyword.CONSPIRE,     // tap two creatures that share a color
            Keyword.ESCALATE,     // pay for each mode beyond the first
            Keyword.MAYFLASHCOST, // pay extra to cast as though it had flash
            Keyword.SPLICE,       // pay splice cost to add effects to another spell
            Keyword.SPREE,        // choose one or more additional costs
            Keyword.TIERED        // choose one additional cost
    );

    // Keywords that reduce the effective casting cost
    private static final Set<Keyword> COST_REDUCTION_KEYWORDS = EnumSet.of(
            Keyword.AFFINITY,     // costs {1} less per [type] you control
            Keyword.ASSIST,       // another player can pay up to X
            Keyword.CONVOKE,      // tap creatures to pay for {1} or colored mana
            Keyword.DELVE,        // exile from graveyard to pay for {1} each
            Keyword.IMPROVISE,    // tap artifacts to pay for {1} each
            Keyword.UNDAUNTED     // costs {1} less per opponent
    );

    private final CardRules.Reader reader = new CardRules.Reader();
    private int nextCardId = 1;

    /**
     * Parse a card script and convert all faces.
     */
    public MultiCard convertCard(List<String> scriptLines, String filename) {
        reader.reset();
        for (String line : scriptLines) {
            if (line.isEmpty() || line.charAt(0) == '#') {
                continue;
            }
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
        } else {
            ICardFace otherFace = rules.getOtherPart();
            if (otherFace == null && !rules.getMeldWith().isEmpty()) {
                // Meld half without inline back face — resolve via StaticData
                otherFace = StaticData.instance().getCommonCards()
                        .getRules(rules.getMeldWith()).getOtherPart();
            }
            if (otherFace != null) {
                card.setState(splitType.getChangedStateName(), false);
                faces.add(convertFace(card, otherFace));
            }
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
        boolean isClass = face.getType().toString().contains("Class");
        Set<String> classLevelDescriptions = new HashSet<>();

        // --- Keywords (from parsed KeywordInterface objects) ---
        for (KeywordInterface ki : card.getKeywords()) {
            Keyword kw = ki.getKeyword();

            if (kw == Keyword.UNDEFINED) {
                actionCounter = handleUndefinedKeyword(ki, abilities, actionCounter, classLevelDescriptions);
                continue;
            }

            // Gift keyword stores its description in the GiftAbility additional ability
            // on the card's first spell ability (set up by CardFactoryUtil), not on the
            // keyword itself (which uses SimpleKeyword).
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
                abilities.add(new AbilityLine(AbilityType.KEYWORD_PASSIVE, applyTextCasing(giftTitle), null));
                continue;
            }

            // Companion stores deck restriction in a separate field, not in getTitle()
            if (kw == Keyword.COMPANION && ki instanceof Companion comp) {
                String compDesc = comp.getDescription();
                if (compDesc != null && !compDesc.isEmpty()) {
                    compDesc = stripReminderText(compDesc);
                }
                String compTitle = (compDesc != null && !compDesc.isEmpty())
                        ? ki.getTitle() + " \u2014 " + compDesc : ki.getTitle();
                abilities.add(new AbilityLine(AbilityType.KEYWORD_PASSIVE, applyTextCasing(compTitle), null));
                continue;
            }

            // Known keywords — classify as active or passive based on generated traits
            boolean activatable = !ki.getAbilities().isEmpty();
            String title = ki.getTitle();
            if (title == null || title.isEmpty()) {
                title = ki.getOriginal();
            }

            // Kicker with two costs: getTitle() only includes the first cost.
            // Parse the second cost from the original string (format: "Kicker:cost1:cost2")
            if (kw == Keyword.KICKER) {
                String[] kickerParts = ki.getOriginal().split(":", 3);
                if (kickerParts.length >= 3) {
                    Cost cost2 = new Cost(kickerParts[2], false);
                    title = title + " and/or " + cost2.toSimpleString();
                }
            }

            // Affinity: Forge title is "Affinity <type>" but Oracle text reads "Affinity for <type>s"
            if (kw == Keyword.AFFINITY) {
                String[] affinityParts = ki.getOriginal().split(":", 3);
                if (affinityParts.length >= 2) {
                    // Third part is the display name (e.g. "outlaw"), otherwise use the type
                    String typeName = affinityParts.length >= 3
                            ? affinityParts[2] : affinityParts[1];
                    title = "Affinity for " + typeName + "s";
                }
            }

            // Reclassify casting-cost keywords
            if (ALTERNATE_COST_KEYWORDS.contains(kw)) {
                abilities.add(new AbilityLine(AbilityType.ALTERNATE_COST,
                        applyTextCasing(title), null));
                continue;
            }
            if (ADDITIONAL_COST_KEYWORDS.contains(kw)) {
                abilities.add(new AbilityLine(AbilityType.ADDITIONAL_COST,
                        applyTextCasing(title), null));
                continue;
            }
            if (COST_REDUCTION_KEYWORDS.contains(kw)) {
                abilities.add(new AbilityLine(AbilityType.COST_REDUCTION,
                        applyTextCasing(title), null));
                continue;
            }

            // Regular keywords — classify as active or passive
            if (activatable) {
                actionCounter++;
                abilities.add(new AbilityLine(AbilityType.KEYWORD_ACTIVE, applyTextCasing(title), actionCounter));
            } else {
                abilities.add(new AbilityLine(AbilityType.KEYWORD_PASSIVE, applyTextCasing(title), null));
            }
        }

        // --- Abilities (from A: lines, already parsed on the card) ---
        for (SpellAbility sa : card.getSpellAbilities()) {
            if (sa.getKeyword() != null) {
                continue; // skip keyword-derived abilities
            }

            if (sa.getApi() == ApiType.Charm) {
                // Use SpellDescription (the main charm text only), not getDescription()
                // which concatenates all choice descriptions into one long string.
                String charmDesc = sa.getParam("SpellDescription");
                if (charmDesc != null && !charmDesc.isEmpty()) {
                    charmDesc = stripReminderText(charmDesc);
                }
                // Pawprint charms lack SpellDescription — synthesize from Pawprint$ + CanRepeatModes$
                if ((charmDesc == null || charmDesc.isEmpty()) && sa.hasParam("Pawprint")) {
                    String total = sa.getParam("Pawprint");
                    charmDesc = "Choose up to " + total + " {P} worth of modes.";
                    if ("True".equals(sa.getParam("CanRepeatModes"))) {
                        charmDesc += " You may choose the same mode more than once.";
                    }
                }
                if (charmDesc != null && !charmDesc.isEmpty()) {
                    actionCounter++;
                    abilities.add(new AbilityLine(AbilityType.SPELL,
                            applyTextCasing(charmDesc), actionCounter));
                }
                var choices = sa.getAdditionalAbilityList("Choices");
                if (choices != null) {
                    for (var choice : choices) {
                        String choiceDesc = choice.getParam("SpellDescription");
                        if (choiceDesc == null) {
                            choiceDesc = "(no description)";
                        }
                        choiceDesc = stripReminderText(choiceDesc);
                        String pawprint = choice.getParam("Pawprint");
                        if (pawprint != null) {
                            choiceDesc = "{P}".repeat(Integer.parseInt(pawprint))
                                    + " \u2014 " + choiceDesc;
                        }
                        actionCounter++;
                        abilities.add(new AbilityLine(AbilityType.OPTION,
                                applyTextCasing(choiceDesc), actionCounter));
                    }
                }
                continue;
            }

            // Skip additional cost extraction for alternate cost spells (e.g. Cleave)
            if (sa.isSpell() && "True".equals(sa.getParam("NonBasicSpell"))) {
                String precost = sa.getParam("PrecostDesc");
                String costDesc = sa.getParam("CostDesc");
                if (precost != null && costDesc != null) {
                    abilities.add(new AbilityLine(AbilityType.ALTERNATE_COST,
                            applyTextCasing(precost + " " + costDesc), null));
                }
                continue;
            }

            // For spell-type abilities, emit additional cost as a separate line.
            // Strip the "As an additional cost to cast this spell, " prefix since
            // the "additional cost:" label already conveys that meaning.
            if (sa.isSpell()) {
                String costDesc = sa.getCostDescription();
                if (costDesc != null) {
                    costDesc = costDesc.trim();
                    if (costDesc.startsWith(ADDITIONAL_COST_PREFIX)) {
                        costDesc = costDesc.substring(ADDITIONAL_COST_PREFIX.length());
                    }
                    costDesc = stripReminderText(costDesc);
                }
                if (costDesc != null && !costDesc.isEmpty()) {
                    abilities.add(new AbilityLine(AbilityType.ADDITIONAL_COST,
                            applyTextCasing(costDesc), null));
                }
            }

            String spellDesc = sa.getParam("SpellDescription");
            if (spellDesc == null || spellDesc.isEmpty()) {
                continue;
            }

            if (sa.isActivatedAbility()) {
                String desc = stripReminderText(sa.getDescription());
                actionCounter++;
                AbilityType type;
                if (sa.isPwAbility()) {
                    type = AbilityType.PLANESWALKER;
                    desc = bracketLoyaltyCost(desc);
                } else {
                    type = AbilityType.ACTIVATED;
                }
                abilities.add(new AbilityLine(type, applyTextCasing(desc), actionCounter));
            } else if (sa.isSpell()) {
                String desc = stripReminderText(spellDesc);
                if (desc.isEmpty()) {
                    continue;
                }
                actionCounter++;
                abilities.add(new AbilityLine(AbilityType.SPELL,
                        applyTextCasing(desc), actionCounter));
            }
        }

        // --- Triggers, Statics, Replacements (from T:/S:/R: lines, already parsed on the card) ---
        addDescribedTraits(abilities, card.getTriggers(), "TriggerDescription", AbilityType.TRIGGERED, face.getName());
        addDescribedTraits(abilities, card.getStaticAbilities(), "Description", AbilityType.STATIC, face.getName());
        addDescribedTraits(abilities, card.getReplacementEffects(), "Description", AbilityType.REPLACEMENT, face.getName());

        // --- Class post-processing: merge level costs with effect descriptions ---
        if (isClass) {
            // Remove any statics/triggers/replacements that duplicate level 2+ descriptions
            abilities.removeIf(a ->
                    a.type() != AbilityType.LEVEL
                    && classLevelDescriptions.contains(a.description()));

            // Convert remaining non-keyword, non-level abilities to LEVEL[1]
            for (int i = 0; i < abilities.size(); i++) {
                AbilityLine a = abilities.get(i);
                if (a.type() == AbilityType.STATIC || a.type() == AbilityType.TRIGGERED
                        || a.type() == AbilityType.REPLACEMENT) {
                    abilities.set(i, new AbilityLine(AbilityType.LEVEL, a.description(), 1));
                }
            }

            // Sort LEVEL abilities by level number (level 1 first)
            abilities.sort((a, b) -> {
                boolean aLevel = a.type() == AbilityType.LEVEL;
                boolean bLevel = b.type() == AbilityType.LEVEL;
                if (aLevel && bLevel) {
                    return Integer.compare(a.actionNumber(), b.actionNumber());
                }
                // LEVEL abilities come first, then any remaining non-level abilities
                if (aLevel) return -1;
                if (bLevel) return 1;
                return 0;
            });
        }

        // --- Move cost lines to the top ---
        abilities.sort((a, b) -> {
            boolean aCost = a.type() == AbilityType.ADDITIONAL_COST
                    || a.type() == AbilityType.ALTERNATE_COST
                    || a.type() == AbilityType.COST_REDUCTION;
            boolean bCost = b.type() == AbilityType.ADDITIONAL_COST
                    || b.type() == AbilityType.ALTERNATE_COST
                    || b.type() == AbilityType.COST_REDUCTION;
            if (aCost == bCost) return 0;
            return aCost ? -1 : 1;
        });

        // --- Build ConvertedCard ---
        String name = applyTextCasing(face.getName());
        ManaCost manaCost = face.getManaCost();
        String manaCostStr = (manaCost == null || manaCost == ManaCost.NO_COST)
                ? null : manaCost.getSimpleString();

        String typeLine = formatTypeLine(face);
        String pt = (face.getPower() != null && face.getToughness() != null)
                ? face.getPower() + "/" + face.getToughness() : null;
        String loyalty = face.getInitialLoyalty();
        if (loyalty != null && loyalty.isEmpty()) {
            loyalty = null;
        }
        String defense = face.getDefense();
        if (defense != null && defense.isEmpty()) {
            defense = null;
        }

        String colors = null;

        String text = face.getNonAbilityText();
        if (text != null && text.isEmpty()) {
            text = null;
        }
        if (text != null) {
            text = applyTextCasing(text);
        }

        return new ConvertedCard(name, manaCostStr, typeLine, pt, loyalty, defense, colors, text, abilities);
    }

    // --- Helper methods ---

    private void addDescribedTraits(List<AbilityLine> abilities,
                                    Iterable<? extends CardTraitBase> traits,
                                    String descParam, AbilityType type,
                                    String faceName) {
        for (CardTraitBase trait : traits) {
            if (trait.getKeyword() != null) {
                // Keyword-generated traits are already emitted by the keyword loop
                // and handleUndefinedKeyword. Without this skip, saga chapter triggers
                // (e.g. "A Golden Opportunity") appear twice: once as chapter: from
                // the K:Chapter handler, and again as triggered: from
                // card.getTriggers() which includes the same Forge-generated objects.
                continue;
            }
            if ("True".equals(trait.getParam("Secondary")) || "True".equals(trait.getParam("Static"))) {
                continue;
            }
            String desc = trait.getParam(descParam);
            if (desc == null || desc.isEmpty()) {
                // Expected for multi-line statics where only the first S: line carries the
                // Description (e.g. Vow of Duty: Continuous + CantAttack), and for AI-only
                // hint statics like AddSVar$ AITap (e.g. Well of Discovery).
                continue;
            }
            desc = applyTextCasing(stripReminderText(desc));
            if (desc.isEmpty()) {
                continue;
            }
            // RaiseCost/OptionalCost statics are additional casting costs ONLY when they
            // apply to this card itself (ValidCard$ Card.Self or Affected$ Card.Self).
            // Otherwise they are static abilities that tax or modify other spells' costs.
            AbilityType effectiveType = type;
            String mode = trait.getParam("Mode");
            if (type == AbilityType.STATIC
                    && ("RaiseCost".equals(mode) || "OptionalCost".equals(mode))) {
                String validCard = trait.getParam("ValidCard");
                String affected = trait.getParam("Affected");
                boolean selfCost = "Card.Self".equals(validCard) || "Card.Self".equals(affected);
                if (selfCost) {
                    effectiveType = AbilityType.ADDITIONAL_COST;
                    if (desc.startsWith(applyTextCasing(ADDITIONAL_COST_PREFIX))) {
                        desc = desc.substring(applyTextCasing(ADDITIONAL_COST_PREFIX).length());
                    }
                }
            }
            abilities.add(new AbilityLine(effectiveType, desc, null));
        }
    }

    /**
     * Handle UNDEFINED keywords — Chapter, Class, etbCounter, and plaintext rules text.
     * These are not recognized by the Keyword enum and require string-based classification.
     * Returns the updated actionCounter.
     */
    private int handleUndefinedKeyword(KeywordInterface ki, List<AbilityLine> abilities,
                                       int actionCounter, Set<String> classLevelDescriptions) {
        String original = ki.getOriginal();

        // Plaintext K: lines starting with CARDNAME/NICKNAME are rules text, not keywords
        if (original.startsWith("CARDNAME ") || original.startsWith("NICKNAME ")) {
            abilities.add(new AbilityLine(AbilityType.STATIC, applyTextCasing(original), null));
            return actionCounter;
        }

        if (original.startsWith("Chapter:")) {
            return emitKeywordTraits(ki.getTriggers(), t -> t.getParam("TriggerDescription"),
                    AbilityType.CHAPTER, false, abilities, actionCounter);
        }
        if (original.startsWith("Class:")) {
            emitClassLevel(ki, abilities, classLevelDescriptions);
            return actionCounter;
        }
        if (original.startsWith("etbCounter:") || original.startsWith("ETBReplacement:")) {
            return emitKeywordTraits(ki.getReplacements(), t -> t.getParam("Description"),
                    AbilityType.REPLACEMENT, false, abilities, actionCounter);
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
            abilities.add(new AbilityLine(AbilityType.ADDITIONAL_COST,
                    applyTextCasing(desc.toString()), null));
            return actionCounter;
        }

        // Fallback: treat as regular keyword
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
        return actionCounter;
    }

    /**
     * Emit ability lines from keyword-generated traits (triggers, abilities, replacements).
     * Returns the updated actionCounter.
     */
    private <T extends CardTraitBase> int emitKeywordTraits(
            Iterable<T> traits, Function<T, String> descExtractor,
            AbilityType type, boolean numbered,
            List<AbilityLine> abilities, int actionCounter) {
        for (T trait : traits) {
            if ("True".equals(trait.getParam("Secondary"))) {
                continue;
            }
            String desc = descExtractor.apply(trait);
            if (desc == null || desc.isEmpty()) {
                continue;
            }
            desc = stripReminderText(desc);
            desc = applyTextCasing(desc);
            if (type == AbilityType.CHAPTER) {
                desc = uppercaseRomanPrefix(desc);
            }
            Integer num = numbered ? ++actionCounter : null;
            abilities.add(new AbilityLine(type, desc, num));
        }
        return actionCounter;
    }

    /**
     * Emit a single Class level ability line.
     * Extracts the level number and Forge-formatted cost from the keyword's activated ability,
     * then finds the effect description from the keyword's triggers, statics, or replacements.
     */
    private void emitClassLevel(KeywordInterface ki, List<AbilityLine> abilities,
                                Set<String> classLevelDescriptions) {
        String original = ki.getOriginal();
        // Format: "Class:level:cost:params"
        int level = Integer.parseInt(original.split(":", 3)[1]);

        // Get the Forge-formatted cost string from the level-up activated ability.
        // SpellAbility.getCostDescription() formats the Cost domain object properly,
        // handling mana, tap, sacrifice, life payment, etc.
        String cost = null;
        for (SpellAbility sa : ki.getAbilities()) {
            cost = sa.getCostDescription();
            break;
        }
        if (cost == null || cost.isEmpty()) {
            return;
        }
        // getCostDescription() appends ": " for ability costs — trim trailing whitespace
        cost = cost.trim();
        // Remove trailing colon if present (we add our own separator)
        if (cost.endsWith(":")) {
            cost = cost.substring(0, cost.length() - 1).trim();
        }

        // Find the effect description from the keyword's own traits
        String desc = findFirstDescription(ki.getTriggers(), "TriggerDescription");
        if (desc == null) {
            desc = findFirstDescription(ki.getStaticAbilities(), "Description");
        }
        if (desc == null) {
            desc = findFirstDescription(ki.getReplacements(), "Description");
        }
        if (desc == null) {
            return;
        }

        desc = stripReminderText(desc);
        desc = applyTextCasing(desc);
        classLevelDescriptions.add(desc);
        String fullDesc = cost + ": " + desc;
        abilities.add(new AbilityLine(AbilityType.LEVEL, applyTextCasing(fullDesc), level));
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
        if (text == null) {
            return null;
        }

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

        // Restore variable X to uppercase (standalone X not inside words like "exile")
        lowered = VARIABLE_X.matcher(lowered).replaceAll("X");

        return lowered;
    }

    /**
     * Uppercase roman numeral prefix before an em dash (e.g. "i, ii — ..." → "I, II — ...").
     */
    static String uppercaseRomanPrefix(String text) {
        Matcher m = ROMAN_PREFIX.matcher(text);
        if (m.find()) {
            return m.group(1).toUpperCase() + text.substring(m.group(1).length());
        }
        return text;
    }

    /**
     * Wrap the loyalty cost prefix in square brackets to match Oracle text format.
     * E.g. "+2: Each player draws a card." → "[+2]: Each player draws a card."
     */
    static String bracketLoyaltyCost(String text) {
        Matcher m = LOYALTY_COST.matcher(text);
        if (m.find()) {
            return "[" + m.group(1) + "]: " + text.substring(m.end());
        }
        return text;
    }

    private String stripReminderText(String text) {
        if (text == null) {
            return null;
        }
        return REMINDER_TEXT.matcher(text).replaceAll("").trim();
    }
}
