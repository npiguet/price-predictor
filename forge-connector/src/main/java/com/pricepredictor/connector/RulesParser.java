package com.pricepredictor.connector;

import com.pricepredictor.connector.ability.ActivatedAbilityEntry;
import com.pricepredictor.connector.ability.AlternateCostSpell;
import com.pricepredictor.connector.ability.ChapterAbility;
import com.pricepredictor.connector.ability.CharmAbility;
import com.pricepredictor.connector.ability.ClassLevelAbility;
import com.pricepredictor.connector.ability.CompanionKeyword;
import com.pricepredictor.connector.ability.EtbReplacementAbility;
import com.pricepredictor.connector.ability.GiftKeyword;
import com.pricepredictor.connector.ability.HauntKeyword;
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

        // Pre-pass: capture front-face Oracle, Name, and Draft lines.
        // We stop at ALTERNATE so we never accidentally use a back-face oracle.
        String frontOracle = null;
        String frontName = null;
        List<String> frontDraftLines = new ArrayList<>();
        boolean inAlternate = false;
        for (String line : scriptLines) {
            String trimmed = line.trim();
            if ("ALTERNATE".equals(trimmed)) { inAlternate = true; continue; }
            if (inAlternate) continue;
            if (trimmed.startsWith("Oracle:") && frontOracle == null) {
                frontOracle = trimmed.substring(7).trim();
            }
            if (trimmed.startsWith("Name:") && frontName == null) {
                frontName = trimmed.substring(5).trim();
            }
            if (trimmed.startsWith("Draft:")) {
                frontDraftLines.add(trimmed.substring(6).trim());
            }
        }

        for (String line : scriptLines) {
            if (line.isEmpty() || line.charAt(0) == '#') {
                continue;
            }
            reader.parseLine(line);
        }
        CardRules rules = reader.getCard();
        MultiCard result = parseRules(rules);

        // Apply oracle fallback: if the primary face produced no ability lines and no
        // non-ability text (e.g. Draft/conspiracy cards whose game-engine abilities are
        // not represented as standard Forge abilities), emit the Oracle text directly.
        // Skip oracle fallback when draft lines are present: draft-only cards (e.g. Cogwork
        // Librarian) produce their content via applyDraftLines(); firing the oracle fallback
        // too would duplicate those lines as both draft: and text: entries.
        if (frontOracle != null && !frontOracle.isEmpty() && frontDraftLines.isEmpty()) {
            result = applyOracleFallbackIfNeeded(result, frontOracle, frontName);
        }
        // Prepend Draft: lines (if any) as TEXT abilities on the primary face.
        // The Forge game engine ignores Draft fields, so they never appear in game abilities,
        // but they are part of the card's oracle text and must be included in the output.
        if (!frontDraftLines.isEmpty()) {
            result = applyDraftLines(result, frontDraftLines, frontName);
        }
        return result;
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
            } else if (kw == Keyword.HAUNT) {
                abilities.addAll(HauntKeyword.of(ki, card));
            } else if (kw == Keyword.GIFT) {
                abilities.add(GiftKeyword.of(ki, card));
            } else if (kw == Keyword.COMPANION && ki instanceof Companion comp) {
                abilities.add(CompanionKeyword.of(ki, comp));
            } else {
                abilities.add(StandardKeyword.of(ki, kw));
            }
        }

        // --- Spell abilities — route to variants ---
        boolean isAlternateFace = face.getType().hasSubtype("Adventure") || face.getType().hasSubtype("Omen");
        for (SpellAbility sa : card.getSpellAbilities()) {
            if (sa.getKeyword() != null) continue;
            // Adventure/Omen SAs belong to the Secondary state; skip them when processing the main face.
            // We check both sa.isAdventure()/isOmen() (which tests the SA's own CardStateName) and
            // the state-name mismatch (fallback in case getCardStateName() cannot resolve the type).
            if (!isAlternateFace) {
                if (sa.isAdventure() || sa.isOmen()) continue;
                if (sa.getCardState() != null
                        && sa.getCardState().getStateName() == CardStateName.Secondary) continue;
            }
            if (sa.getApi() == ApiType.Charm) {
                abilities.addAll(CharmAbility.fromSpellAbility(sa));
            } else if (sa.isSpell() && "True".equals(sa.getParam("NonBasicSpell"))) {
                addIfNotNull(abilities, AlternateCostSpell.of(sa));
            } else if (sa.isActivatedAbility()) {
                addIfNotNull(abilities, ActivatedAbilityEntry.of(sa));
            } else if (sa.isSpell()) {
                addIfNotNull(abilities, SpellAdditionalCost.of(sa));
                abilities.addAll(SpellEffect.fromChain(sa));
                abilities.addAll(diceOutcomesFromSA(sa));
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
        // Remove abilities whose description duplicates an earlier one.
        // This eliminates the spurious second trigger that Forge registers for
        // the "enters or attacks" pattern (two T: lines, identical TriggerDescription).
        abilities = deduplicateByDescription(abilities);
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
        } else if (original.startsWith("Visit:")) {
            // Attraction visit keyword — the trigger's TriggerDescription already contains
            // "Visit — [effect text]", which after normalization matches the oracle.
            for (Trigger t : ki.getTriggers()) {
                String tDesc = t.getParam("TriggerDescription");
                if (tDesc == null || "Blank".equals(tDesc)) continue;
                String normalized = AbilityDescription.normalize(tDesc);
                if (normalized == null || normalized.isEmpty()) continue;
                List<Ability> children = SpellEffect.fromChain(t.getOverridingAbility());
                abilities.add(new TriggeredAbilityEntry(AbilityType.TRIGGERED,
                        AbilityDescription.applyCasing(normalized), children));
            }
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

    /**
     * Expand {@code ResultSubAbilities} dice-outcome entries into SpellEffect nodes.
     * Each entry has the form {@code "range:SvarName"} (e.g. {@code "1-9:DBTapAll"}).
     * The SVar's SpellDescription is used as the description text; VERT placeholders
     * are replaced so "1—9 VERT Tap all…" becomes "1—9 | Tap all…".
     */
    private static List<Ability> diceOutcomesFromSA(SpellAbility sa) {
        String resultSubAbilities = sa.getParam("ResultSubAbilities");
        if (resultSubAbilities == null || resultSubAbilities.isEmpty()) return List.of();

        List<Ability> result = new ArrayList<>();
        for (String entry : resultSubAbilities.split(",")) {
            String[] kv = entry.trim().split(":", 2);
            if (kv.length < 2) continue;
            String range = kv[0].trim();
            SpellAbility sub = sa.getAdditionalAbility(range);
            if (sub == null) continue;
            String rawDesc = sub.getParam("SpellDescription");
            if (rawDesc == null || rawDesc.isEmpty()) continue;
            String desc = AbilityDescription.replaceVert(rawDesc);
            String normalized = AbilityDescription.normalize(desc);
            if (normalized != null) {
                result.add(new SpellEffect(AbilityDescription.applyCasing(normalized), List.of()));
            }
        }
        return result;
    }

    /** Remove abilities whose descriptionText duplicates an earlier entry. */
    private static List<Ability> deduplicateByDescription(List<Ability> abilities) {
        Set<String> seen = new HashSet<>();
        List<Ability> result = new ArrayList<>();
        for (Ability a : abilities) {
            String desc = a.descriptionText();
            if (desc == null || seen.add(desc)) {
                result.add(a);
            }
        }
        return result;
    }

    /**
     * If the primary face of {@code card} produced no ability lines and no non-ability
     * text, re-emit it with the front-face Oracle text split into {@code TEXT} ability
     * lines.  This handles Draft/conspiracy cards whose game-engine representation in
     * Forge carries no standard spell/trigger/static abilities.
     */
    private static MultiCard applyOracleFallbackIfNeeded(
            MultiCard card, String rawOracle, String cardName) {
        CardFace first = card.faces().get(0);
        if (!first.abilities().isEmpty() || first.text() != null) {
            return card; // already has content — fallback not needed
        }

        // Replace literal \n escape sequences with real newlines, then substitute
        // the card name with the CARDNAME placeholder used throughout the output.
        String oracleText = rawOracle.replace("\\n", "\n");
        if (cardName != null && !cardName.isEmpty()) {
            oracleText = oracleText.replace(cardName, "CARDNAME");
        }

        List<Ability> fallback = new ArrayList<>();
        for (String line : oracleText.split("\n")) {
            String stripped = AbilityDescription.stripReminderText(line.trim());
            if (stripped != null && !stripped.isEmpty()) {
                fallback.add(new TextAbility(AbilityType.TEXT,
                        AbilityDescription.applyCasing(stripped)));
            }
        }
        if (fallback.isEmpty()) return card;

        CardFace newFirst = new CardFace(first.name(), first.manaCost(), first.types(),
                first.powerToughness(), first.loyalty(), first.defense(),
                first.colors(), first.text(), fallback);
        List<CardFace> newFaces = new ArrayList<>(card.faces());
        newFaces.set(0, newFirst);
        return new MultiCard(card.layout(), newFaces);
    }

    /**
     * Prepend {@code Draft:} script lines as TEXT abilities on the primary face.
     * Draft instructions are part of the card's oracle text but are invisible to the
     * Forge game engine, so they must be injected from the raw script.
     */
    private static MultiCard applyDraftLines(
            MultiCard card, List<String> draftLines, String cardName) {
        CardFace first = card.faces().get(0);
        List<Ability> draftAbilities = new ArrayList<>();
        for (String raw : draftLines) {
            String text = (cardName != null && !cardName.isEmpty())
                    ? raw.replace(cardName, "CARDNAME") : raw;
            String stripped = AbilityDescription.stripReminderText(text.trim());
            if (stripped != null && !stripped.isEmpty()) {
                draftAbilities.add(new TextAbility(AbilityType.DRAFT,
                        AbilityDescription.applyCasing(stripped)));
            }
        }
        if (draftAbilities.isEmpty()) return card;

        List<Ability> combined = new ArrayList<>(draftAbilities);
        combined.addAll(first.abilities());
        CardFace newFirst = new CardFace(first.name(), first.manaCost(), first.types(),
                first.powerToughness(), first.loyalty(), first.defense(),
                first.colors(), first.text(), combined);
        List<CardFace> newFaces = new ArrayList<>(card.faces());
        newFaces.set(0, newFirst);
        return new MultiCard(card.layout(), newFaces);
    }
}
