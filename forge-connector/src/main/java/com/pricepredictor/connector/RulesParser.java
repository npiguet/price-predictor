package com.pricepredictor.connector;

import com.pricepredictor.connector.ability.ActivatedAbilityEntry;
import com.pricepredictor.connector.ability.AlternateAdditionalCostAbility;
import com.pricepredictor.connector.ability.ForgeParams;
import com.pricepredictor.connector.ability.AlternateCostSpell;
import com.pricepredictor.connector.ability.ChapterAbility;
import com.pricepredictor.connector.ability.CharmAbility;
import com.pricepredictor.connector.ability.ClassLevelAbility;
import com.pricepredictor.connector.ability.CompanionKeyword;
import com.pricepredictor.connector.ability.EtbReplacementAbility;
import com.pricepredictor.connector.ability.GiftKeyword;
import com.pricepredictor.connector.ability.HauntKeyword;
import com.pricepredictor.connector.ability.ReplacementAbilityEntry;
import com.pricepredictor.connector.ability.SpellAbilityUtils;
import com.pricepredictor.connector.ability.SpellAdditionalCost;
import com.pricepredictor.connector.ability.SpellEffect;
import com.pricepredictor.connector.ability.StandardKeyword;
import com.pricepredictor.connector.ability.StaticAbilityEntry;
import com.pricepredictor.connector.ability.TextAbility;
import com.pricepredictor.connector.ability.TriggeredAbilityEntry;
import com.pricepredictor.connector.ability.VisitAbility;
import com.pricepredictor.connector.ability.OpeningHandAbility;
import forge.card.CardRarity;
import forge.card.CardRules;
import forge.card.CardSplitType;
import forge.card.CardStateName;
import forge.card.ICardFace;
import forge.card.mana.ManaCost;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardFactory;
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Parses Forge card scripts into domain objects (MultiCard/CardFace/Ability).
 * Acts as a router, delegating to variant Ability implementations in the ability sub-package.
 */
public class RulesParser {

    private final CardRules.Reader reader = new CardRules.Reader();
    private int nextCardId = 1;

    /**
     * Dispatch table for UNDEFINED keyword prefixes that produce List&lt;Ability&gt; results.
     * CARDNAME/NICKNAME and Class are handled separately (different signatures or side-effects).
     */
    private static final Map<String, Function<KeywordInterface, List<Ability>>> KEYWORD_ROUTES;
    static {
        KEYWORD_ROUTES = new LinkedHashMap<>();
        KEYWORD_ROUTES.put("Chapter:", ChapterAbility::fromKeyword);
        KEYWORD_ROUTES.put("etbCounter:", EtbReplacementAbility::fromKeyword);
        KEYWORD_ROUTES.put("ETBReplacement:", EtbReplacementAbility::fromKeyword);
        KEYWORD_ROUTES.put("AlternateAdditionalCost:", AlternateAdditionalCostAbility::fromKeyword);
        KEYWORD_ROUTES.put("Visit:", VisitAbility::fromKeyword);
        KEYWORD_ROUTES.put("MayEffectFromOpeningHand:", OpeningHandAbility::fromKeyword);
        KEYWORD_ROUTES.put("MayEffectFromOpeningDeck:", OpeningHandAbility::fromKeyword);
    }

    // Script field prefixes used in the pre-pass below.
    private static final String ORACLE_KEY = "Oracle:";
    private static final String NAME_KEY   = "Name:";
    private static final String DRAFT_KEY  = "Draft:";

    /**
     * Parse a card script and build domain objects for all faces.
     */
    public MultiCard parseScript(List<String> scriptLines, String filename) {
        reader.reset();

        // Pre-pass: capture front-face Oracle, Name, and Draft lines before ALTERNATE.
        //
        // CardRules.Reader doesn't expose the front-face oracle text as a distinct field
        // after multi-face parsing, so we capture it here from the raw script. Two uses:
        //   1. Oracle fallback: Draft/conspiracy cards whose Forge game-engine representation
        //      carries no standard spell/trigger/static abilities need the raw oracle text
        //      emitted directly as TEXT ability lines.
        //   2. Draft injection: Draft: lines are invisible to the Forge game engine but are
        //      part of the card's oracle text and must appear in the output.
        //
        // We stop at ALTERNATE to avoid using a back-face oracle for the front face
        // on transform/modal double-faced cards.
        String frontOracle = null;
        String frontName = null;
        List<String> frontDraftLines = new ArrayList<>();
        boolean inAlternate = false;
        for (String line : scriptLines) {
            String trimmed = line.trim();
            if ("ALTERNATE".equals(trimmed)) { inAlternate = true; continue; }
            if (inAlternate) continue;
            if (trimmed.startsWith(ORACLE_KEY) && frontOracle == null) {
                frontOracle = trimmed.substring(ORACLE_KEY.length()).trim();
            }
            if (trimmed.startsWith(NAME_KEY) && frontName == null) {
                frontName = trimmed.substring(NAME_KEY.length()).trim();
            }
            if (trimmed.startsWith(DRAFT_KEY)) {
                frontDraftLines.add(trimmed.substring(DRAFT_KEY.length()).trim());
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

    /** Keywords collected from a card face: ability nodes and Class level description strings. */
    private record CollectedKeywords(List<Ability> abilities, Set<NonBlankString> classLevelDescriptions) {}

    /**
     * Parse a single card face. The Card must already be in the correct state.
     */
    CardFace parseFace(Card card, ICardFace face) {
        boolean isClass = face.getType().toString().contains("Class");
        boolean isAlternateFace = face.getType().hasSubtype("Adventure") || face.getType().hasSubtype("Omen");

        CollectedKeywords kw = collectKeywords(card);
        List<Ability> abilities = new ArrayList<>(kw.abilities());
        abilities.addAll(collectSpellAbilities(card, isAlternateFace));
        abilities.addAll(collectTriggers(card));
        abilities.addAll(collectStaticsAndReplacements(card));

        buildLandManaDescription(face)
                .map(desc -> new TextAbility(AbilityType.ACTIVATED, desc))
                .ifPresent(abilities::add);

        if (isClass) {
            abilities = applyClassPostProcessing(abilities, kw.classLevelDescriptions());
        }
        // Remove abilities whose description duplicates an earlier one.
        // This eliminates the spurious second trigger that Forge registers for
        // the "enters or attacks" pattern (two T: lines, identical TriggerDescription).
        abilities = deduplicateByDescription(abilities);
        abilities = sortCostsFirst(abilities);

        return buildCardFace(face, abilities);
    }

    private CollectedKeywords collectKeywords(Card card) {
        List<Ability> abilities = new ArrayList<>();
        Set<NonBlankString> classLevelDescriptions = new HashSet<>();
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
        return new CollectedKeywords(abilities, classLevelDescriptions);
    }

    private List<Ability> collectSpellAbilities(Card card, boolean isAlternateFace) {
        List<Ability> abilities = new ArrayList<>();
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
                AlternateCostSpell.of(sa).ifPresent(abilities::add);
            } else if (sa.isActivatedAbility()) {
                ActivatedAbilityEntry.of(sa).ifPresent(abilities::add);
            } else if (sa.isSpell()) {
                // Skip SAs that are handled by the triggers or replacements loops.
                // ImmediateTrigger SAs fire immediately on enter and are also registered
                // as triggers; ETBReplacement SAs are also replacement effects.
                // Both appear in card.getSpellAbilities() AND in getTriggers()/
                // getReplacementEffects(), so without this guard they double-emit.
                if (sa.getApi() == ApiType.ImmediateTrigger) continue;
                String mode = sa.getParam("Mode");
                if (ForgeParams.ETB_REPLACEMENT_MODE.equals(mode)) continue;
                SpellAdditionalCost.of(sa).ifPresent(abilities::add);
                abilities.addAll(SpellEffect.fromChain(sa));
                abilities.addAll(SpellAbilityUtils.expandDiceOutcomes(sa, AbilityType.SPELL, false));
            }
        }
        return abilities;
    }

    private List<Ability> collectTriggers(Card card) {
        // Deduplicate "attacks or blocks" trigger pairs.
        //
        // For cards with "whenever CARDNAME attacks or blocks", Forge registers two triggers
        // that share the same Execute SVar: a primary trigger ("attacks") whose
        // TriggerDescription says "attacks or blocks", and a Secondary trigger ("blocks")
        // pointing at the same Execute SVar. The primary description already covers both
        // halves, so including the secondary trigger would emit a duplicate ability line.
        //
        // Pass 1: collect execute SVar names from all non-secondary triggers.
        // Pass 2: skip any secondary trigger whose Execute SVar was seen in pass 1.
        Set<String> primaryExecuteSVars = card.getTriggers().stream()
                .filter(t -> !"True".equalsIgnoreCase(t.getParam("Secondary")))
                .map(t -> t.getParam("Execute"))
                .filter(Objects::nonNull)
                .collect(Collectors.toSet());

        List<Ability> abilities = new ArrayList<>();
        for (Trigger t : card.getTriggers()) {
            String exec = t.getParam("Execute");
            if ("True".equalsIgnoreCase(t.getParam("Secondary"))
                    && exec != null && primaryExecuteSVars.contains(exec)) {
                continue; // secondary of an "attacks or blocks" pair — skip duplicate
            }
            TriggeredAbilityEntry.of(t).ifPresent(abilities::add);
        }
        return abilities;
    }

    private List<Ability> collectStaticsAndReplacements(Card card) {
        List<Ability> abilities = new ArrayList<>();
        for (StaticAbility s : card.getStaticAbilities()) {
            StaticAbilityEntry.of(s).ifPresent(abilities::add);
        }
        for (ReplacementEffect r : card.getReplacementEffects()) {
            ReplacementAbilityEntry.of(r).ifPresent(abilities::add);
        }
        return abilities;
    }

    private static CardFace buildCardFace(ICardFace face, List<Ability> abilities) {
        NonBlankString name = AbilityDescription.applyCasing(NonBlankString.require(face.getName()));
        ManaCost manaCost = face.getManaCost();
        Optional<NonBlankString> manaCostStr = (manaCost == null || manaCost == ManaCost.NO_COST)
                ? Optional.empty() : NonBlankString.of(manaCost.getSimpleString());

        NonBlankString typeLine = formatTypeLine(face);
        Optional<NonBlankString> pt = (face.getPower() != null && face.getToughness() != null)
                ? NonBlankString.of(face.getPower() + "/" + face.getToughness()) : Optional.empty();
        Optional<NonBlankString> loyalty = NonBlankString.of(face.getInitialLoyalty());
        Optional<NonBlankString> defense = NonBlankString.of(face.getDefense());

        // Strip [Developer's note: …] brackets — Forge-internal metadata, not oracle content.
        Optional<NonBlankString> text = Optional.ofNullable(face.getNonAbilityText())
                .filter(t -> !t.isEmpty())
                .map(t -> t.replaceAll("(?i)\\[Developer's note:[^]]*]", "").strip())
                .flatMap(NonBlankString::of)
                .map(AbilityDescription::applyCasing);

        return new CardFace(name, manaCostStr, typeLine, pt, loyalty, defense, Optional.empty(), text, abilities);
    }

    // --- Undefined keyword routing ---

    private void routeUndefinedKeyword(KeywordInterface ki, List<Ability> abilities,
                                       Set<NonBlankString> classLevelDescriptions) {
        String original = ki.getOriginal();

        if (original.startsWith("CARDNAME ") || original.startsWith("NICKNAME ")) {
            abilities.add(new TextAbility(AbilityType.STATIC, AbilityDescription.applyCasing(NonBlankString.require(original))));
            return;
        }
        if (original.startsWith("Class:")) {
            ClassLevelAbility.of(ki).ifPresent(level -> {
                classLevelDescriptions.add(level.innerDescription());
                abilities.add(level);
            });
            return;
        }
        for (var entry : KEYWORD_ROUTES.entrySet()) {
            if (original.startsWith(entry.getKey())) {
                abilities.addAll(entry.getValue().apply(ki));
                return;
            }
        }
        abilities.add(StandardKeyword.of(ki, Keyword.UNDEFINED));
    }

    // --- Post-processing ---

    private static List<Ability> applyClassPostProcessing(List<Ability> abilities,
                                                          Set<NonBlankString> classLevelDescriptions) {
        List<Ability> result = removeClassDuplicates(abilities, classLevelDescriptions);
        result = retypeToLevel(result);
        return sortLevelFirst(result);
    }

    /**
     * Remove abilities whose description text duplicates the inner description of a LEVEL
     * ability. Class cards embed triggers and static effects inside level blocks; Forge also
     * emits those same effects as top-level abilities. The inner text of each ClassLevelAbility
     * is tracked so those duplicates can be stripped here.
     */
    private static List<Ability> removeClassDuplicates(List<Ability> abilities,
                                                        Set<NonBlankString> classLevelDescriptions) {
        List<Ability> result = new ArrayList<>(abilities);
        result.removeIf(a ->
                a.type() != AbilityType.LEVEL
                        && classLevelDescriptions.contains(a.descriptionText()));
        return result;
    }

    /**
     * Retype STATIC, TRIGGERED, and REPLACEMENT abilities to LEVEL at ordinal 1.
     * Class cards declare their level-1 effects as ordinary triggers/statics in Forge's
     * game engine, but in oracle text they belong inside the level block. Reclassifying
     * them as LEVEL ensures they render in the correct position.
     */
    private static List<Ability> retypeToLevel(List<Ability> abilities) {
        List<Ability> result = new ArrayList<>(abilities);
        for (int i = 0; i < result.size(); i++) {
            Ability a = result.get(i);
            if (a.type() == AbilityType.STATIC || a.type() == AbilityType.TRIGGERED
                    || a.type() == AbilityType.REPLACEMENT) {
                result.set(i, new TextAbility(AbilityType.LEVEL, a.descriptionText(), 1));
            }
        }
        return result;
    }

    /**
     * Sort LEVEL abilities first (ordered by their ordinal), then all other abilities in
     * their original relative order. This is a stable partition: non-LEVEL abilities
     * retain their existing sequence.
     */
    private static List<Ability> sortLevelFirst(List<Ability> abilities) {
        List<Ability> result = new ArrayList<>(abilities);
        result.sort((a, b) -> {
            boolean aLevel = a.type() == AbilityType.LEVEL;
            boolean bLevel = b.type() == AbilityType.LEVEL;
            if (aLevel && bLevel) return Integer.compare(a.ordinal(), b.ordinal());
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

    static Optional<NonBlankString> buildLandManaDescription(ICardFace face) {
        List<String> symbols = LAND_TYPE_MANA.entrySet().stream()
                .filter(e -> face.getType().hasSubtype(e.getKey()))
                .map(Map.Entry::getValue)
                .toList();
        if (symbols.isEmpty()) return Optional.empty();
        return Optional.of(NonBlankString.require("{T}: add " + AbilityDescription.joinDisjunction(symbols)));
    }

    private static NonBlankString formatTypeLine(ICardFace face) {
        String typeStr = face.getType().toString();
        typeStr = typeStr.replace(" - ", " ");
        return NonBlankString.require(typeStr.toLowerCase());
    }

    /** Remove abilities whose descriptionText duplicates an earlier entry. */
    private static List<Ability> deduplicateByDescription(List<Ability> abilities) {
        Set<NonBlankString> seen = new HashSet<>();
        List<Ability> result = new ArrayList<>();
        for (Ability a : abilities) {
            if (seen.add(a.descriptionText())) {
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
        if (!first.abilities().isEmpty() || first.text().isPresent()) {
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
            AbilityDescription.normalize(line.trim())
                    .ifPresent(n -> fallback.add(new TextAbility(AbilityType.TEXT, n)));
        }
        if (fallback.isEmpty()) return card;

        return card.withPrimaryFace(face -> face.withAbilities(fallback));
    }

    /**
     * Prepend {@code Draft:} script lines as TEXT abilities on the primary face.
     * Draft instructions are part of the card's oracle text but are invisible to the
     * Forge game engine, so they must be injected from the raw script.
     */
    private static MultiCard applyDraftLines(
            MultiCard card, List<String> draftLines, String cardName) {
        List<Ability> draftAbilities = new ArrayList<>();
        for (String raw : draftLines) {
            String text = (cardName != null && !cardName.isEmpty())
                    ? raw.replace(cardName, "CARDNAME") : raw;
            AbilityDescription.normalize(text.trim())
                    .ifPresent(n -> draftAbilities.add(new TextAbility(AbilityType.DRAFT, n)));
        }
        if (draftAbilities.isEmpty()) return card;

        return card.withPrimaryFace(face -> {
            List<Ability> combined = new ArrayList<>(draftAbilities);
            combined.addAll(face.abilities());
            return face.withAbilities(combined);
        });
    }
}
