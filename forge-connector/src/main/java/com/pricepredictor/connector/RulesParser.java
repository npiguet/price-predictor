package com.pricepredictor.connector;

import com.google.common.base.Strings;
import com.pricepredictor.connector.ability.ActivatedAbilityEntry;
import com.pricepredictor.connector.ability.AlternateCostSpell;
import com.pricepredictor.connector.ability.ChapterAbility;
import com.pricepredictor.connector.ability.CharmAbility;
import com.pricepredictor.connector.ability.ClassLevelAbility;
import com.pricepredictor.connector.ability.CompanionKeyword;
import com.pricepredictor.connector.ability.DiceOutcomes;
import com.pricepredictor.connector.ability.EtbReplacementAbility;
import com.pricepredictor.connector.ability.GiftKeyword;
import com.pricepredictor.connector.ability.HauntKeyword;
import com.pricepredictor.connector.ability.KeywordSpec;
import com.pricepredictor.connector.ability.ReplacementAbilityEntry;
import com.pricepredictor.connector.ability.SpellAdditionalCost;
import com.pricepredictor.connector.ability.SpellEffect;
import com.pricepredictor.connector.ability.StandardKeyword;
import com.pricepredictor.connector.ability.StaticAbilityEntry;
import com.pricepredictor.connector.ability.TextAbility;
import com.pricepredictor.connector.ability.TriggeredAbilityEntry;
import com.pricepredictor.connector.effects.MissingSVarReport;
import com.pricepredictor.connector.effects.ProvenanceKey;
import com.pricepredictor.connector.effects.ProvenanceRecorder;
import com.pricepredictor.connector.effects.TraitKind;
import forge.card.CardRarity;
import forge.card.CardRules;
import forge.card.CardSplitType;
import forge.card.CardStateName;
import forge.card.ICardFace;
import forge.card.mana.ManaCost;
import forge.game.CardTraitBase;
import forge.game.ability.AbilityFactory;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.cost.Cost;
import forge.game.keyword.Companion;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.item.PaperCard;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.regex.Pattern;

/**
 * Parses Forge card scripts into domain objects (MultiCard/CardFace/Ability).
 * Acts as a router, delegating to variant Ability implementations in the ability sub-package.
 */
public class RulesParser {

    /** A {@code [Developer's note: …]} bracket: Forge-internal metadata, not oracle content. */
    private static final Pattern DEVELOPER_NOTE =
            Pattern.compile("(?i)\\[Developer's note:[^]]*]");

    private final CardRules.Reader reader = new CardRules.Reader();
    private int nextCardId = 1;
    /**
     * The undefined SVars every recorder of this parser reports into: a run's
     * worth, drained by whoever converts a tree, so the operator sees one block
     * rather than a line per card.
     */
    private final MissingSVarReport missingSVars = new MissingSVarReport();

    /** The undefined-SVar report the recorded faces have filled so far. */
    public MissingSVarReport missingSVars() {
        return missingSVars;
    }

    /** A parsed card together with the per-face provenance recorders. */
    public record ParsedCard(MultiCard card, List<ProvenanceRecorder> recorders) {
    }

    /**
     * Parse a card script, optionally recording provenance.
     *
     * <p>Recording is additive: the rendered text is identical whether or not
     * a script file is supplied, which is what lets the sidecar ship without
     * reconverting a corpus.
     *
     * @param scriptFile the tree-prefixed script path keys should name, or null
     *                   to skip recording
     */
    public ParsedCard parseScript(List<String> scriptLines, String filename,
                                  String scriptFile) {
        List<ProvenanceRecorder> recorders = new ArrayList<>();
        MultiCard card = parseScriptInternal(scriptLines, filename, recorders,
                scriptFile);
        return new ParsedCard(card, scriptFile == null ? List.of() : recorders);
    }

    private MultiCard parseScriptInternal(List<String> scriptLines, String filename,
                                          List<ProvenanceRecorder> recorders,
                                          String scriptFile) {
        reader.reset();

        ScriptHeader header = ScriptHeader.of(scriptLines);

        for (String line : scriptLines) {
            if (line.isEmpty() || line.charAt(0) == '#') {
                continue;
            }
            reader.parseLine(line);
        }
        CardRules rules = reader.getCard();
        MultiCard result = parseRules(rules, recorders, scriptFile);

        // Apply oracle fallback: if the primary face produced no ability lines and no
        // non-ability text (e.g. Draft/conspiracy cards whose game-engine abilities are
        // not represented as standard Forge abilities), emit the Oracle text directly.
        // Skip oracle fallback when draft lines are present: draft-only cards (e.g. Cogwork
        // Librarian) produce their content via applyDraftLines(); firing the oracle fallback
        // too would duplicate those lines as both draft: and text: entries.
        if (header.oracle() != null && !header.oracle().isEmpty()
                && header.draftLines().isEmpty()) {
            result = applyOracleFallbackIfNeeded(result, header.oracle(), header.name());
        }
        // Prepend Draft: lines (if any) as TEXT abilities on the primary face.
        // The Forge game engine ignores Draft fields, so they never appear in game abilities,
        // but they are part of the card's oracle text and must be included in the output.
        if (!header.draftLines().isEmpty()) {
            result = applyDraftLines(result, header.draftLines(), header.name());
        }
        return result;
    }

    /**
     * Parse CardRules into a MultiCard, optionally recording provenance.
     *
     * @param recorders one recorder per face is appended here, in face order
     * @param scriptFile the tree-prefixed script path keys should name, or
     *                   null to record nothing
     */
    private MultiCard parseRules(CardRules rules, List<ProvenanceRecorder> recorders,
                                String scriptFile) {
        CardSplitType splitType = rules.getSplitType();
        Card card = buildFullCard(rules);

        if (splitType == CardSplitType.None) {
            return MultiCard.singleFace(
                    parseFace(card, rules.getMainPart(),
                            nextRecorder(recorders), 0, scriptFile));
        }

        String layout = splitType.name().toLowerCase();
        List<CardFace> faces = new ArrayList<>();

        CardStateName mainState = (splitType == CardSplitType.Split)
                ? CardStateName.LeftSplit : CardStateName.Original;
        card.setState(mainState, false);
        faces.add(parseFace(card, rules.getMainPart(),
                nextRecorder(recorders), 0, scriptFile));

        if (splitType == CardSplitType.Specialize) {
            // In the state enum's order, not the parts map's: CardRules holds
            // them in a HashMap keyed by enum constants, whose order changes
            // from one JVM to the next, while the collectors number faces in
            // the live card's EnumMap order (ProvenanceKey.faceOrder). A face
            // index taken from the map's order resolved a record to another
            // face's lines, and the converted text came out reordered.
            Map<CardStateName, ICardFace> parts = rules.getSpecializeParts();
            for (CardStateName state : CardStateName.values()) {
                ICardFace part = parts.get(state);
                if (part != null) {
                    card.setState(state, false);
                    faces.add(parseFace(card, part,
                            nextRecorder(recorders), faces.size(), scriptFile));
                }
            }
        } else {
            ICardFace otherFace = rules.getOtherPart();
            if (otherFace != null) {
                card.setState(splitType.getChangedStateName(), false);
                faces.add(parseFace(card, otherFace,
                        nextRecorder(recorders), faces.size(), scriptFile));
            }
        }

        return MultiCard.multiFace(layout, faces);
    }

    /** Append and return a fresh recorder for the next face. */
    private ProvenanceRecorder nextRecorder(List<ProvenanceRecorder> recorders) {
        ProvenanceRecorder recorder = new ProvenanceRecorder(missingSVars);
        recorders.add(recorder);
        return recorder;
    }

    /**
     * One face's parse: the face, where its provenance goes, and the ability
     * list each phase of {@link #parseFace} appends to.
     */
    private static final class FaceParse {
        final Card card;
        final ICardFace face;
        final ProvenanceRecorder recorder;
        final int faceIndex;
        /** The tree-prefixed script path keys name, or null when not recording. */
        final String scriptFile;
        List<Ability> abilities = new ArrayList<>();
        /** Each Class level's inner text, so its duplicate trait entry can be removed. */
        final Set<String> classLevelDescriptions = new HashSet<>();
        /** The synthetic mana line of a basic-typed face, or null. */
        final String landManaDescription;
        /**
         * Runtime mana abilities of a basic-typed face, skipped by the spell
         * phase and rendered once by the synthetic land-mana line. Kept with
         * their keys so that line can claim them: without the claim the land's
         * mana ability is a dropped key, and every activation record that names
         * it joins to nothing.
         */
        final List<SpellAbility> landManaAbilities = new ArrayList<>();
        final List<ProvenanceKey> landManaKeys = new ArrayList<>();

        FaceParse(Card card, ICardFace face, ProvenanceRecorder recorder,
                  int faceIndex, String scriptFile) {
            this.card = card;
            this.face = face;
            this.recorder = recorder;
            this.faceIndex = faceIndex;
            this.scriptFile = scriptFile;
            this.landManaDescription = buildLandManaDescription(face);
        }

        /**
         * The key of this face's {@code index}-th trait of {@code kind}, or null
         * when not recording. Every recorder call takes a null key as "nothing
         * to record", which is what lets the phases record unconditionally.
         */
        ProvenanceKey key(TraitKind kind, int index) {
            return scriptFile == null ? null
                    : new ProvenanceKey(scriptFile, faceIndex, kind, index);
        }

        /** The abilities appended since the list had {@code before} of them. */
        List<Ability> since(int before) {
            return List.copyOf(abilities.subList(before, abilities.size()));
        }
    }

    /**
     * Parse a single card face, optionally recording provenance.
     *
     * <p>Recording happens here rather than inside the ability implementations
     * because this is the only place both the runtime trait and its index
     * within its kind are in scope, and {@code index_within_kind} is half the
     * provenance key. Each phase records the abilities it appended by
     * bracketing the list, so a factory that returns one ability and one that
     * returns several are handled identically.
     *
     * @param recorder    collects trait-to-ability attribution
     * @param faceIndex   this face's ordinal in the converter's emission order
     * @param scriptFile  the tree-prefixed path of the card's script, or null
     *                    to record nothing
     */
    CardFace parseFace(Card card, ICardFace face, ProvenanceRecorder recorder,
                       int faceIndex, String scriptFile) {
        FaceParse p = new FaceParse(card, face, recorder, faceIndex, scriptFile);
        parseKeywords(p);
        parseSpellAbilities(p);
        parseTriggers(p);
        parseTraits(p, card.getStaticAbilities(), TraitKind.STATIC,
                StaticAbilityEntry::of);
        parseTraits(p, card.getReplacementEffects(), TraitKind.REPLACEMENT,
                ReplacementAbilityEntry::of);
        appendLandMana(p);
        postProcess(p);
        return buildCardFace(face, p.abilities);
    }

    /**
     * Route each keyword to the ability that renders it.
     *
     * <p>The keyword ordinal is asked of ProvenanceKey rather than counted
     * here, because the collector asks the same question of the same method
     * at runtime and the two answers have to be the same number. A running
     * counter over getKeywords() is not that number: the iteration order is
     * KeywordCollection's, which is hash order over an enum and differs
     * between the convert JVM and the collect JVM. Counted here, the two
     * sides produced keys that were in range on both and named different
     * keywords — a silent wrong join rather than a loud failure. A keyword
     * with no intrinsic ordinal (a granted one) records no key at all,
     * which is the honest answer and the one the collector gives for it.
     */
    private void parseKeywords(FaceParse p) {
        for (KeywordInterface ki : p.card.getKeywords()) {
            Keyword kw = ki.getKeyword();
            int before = p.abilities.size();
            if (kw == Keyword.UNDEFINED) {
                routeUndefinedKeyword(ki, p);
            } else if (kw == Keyword.HAUNT) {
                p.abilities.addAll(HauntKeyword.of(ki, p.card));
            } else if (kw == Keyword.GIFT) {
                p.abilities.add(GiftKeyword.of(ki, p.card));
            } else if (kw == Keyword.COMPANION && ki instanceof Companion comp) {
                p.abilities.add(CompanionKeyword.of(ki, comp));
            } else {
                p.abilities.add(StandardKeyword.of(ki, kw));
            }
            int keywordIndex = ProvenanceKey.keywordIndex(p.card.getCurrentState(), ki);
            p.recorder.attributeKeyword(p.since(before),
                    keywordIndex < 0 ? null : p.key(TraitKind.KEYWORD, keywordIndex),
                    ki.getOriginal());
        }
    }

    /** Where the spell phase sends one of a face's spell abilities. */
    private enum SpellRoute {
        /**
         * A keyword's own spell. It renders as the keyword line, so that line
         * claims this trait's key rather than dropping it.
         */
        KEYWORD_SPELL,
        /**
         * An Adventure or Omen spell met while parsing the main face: it
         * belongs to the Secondary state and is parsed with that face.
         */
        OTHER_FACE,
        CHARM,
        /** A spell cast for an alternative cost ({@code NonBasicSpell}). */
        ALT_COST,
        /**
         * A mana ability of a face with basic land types. The synthetic land
         * mana line renders it once in canonical form; CardFactory's fully
         * resolved card lists it among the spell abilities as well, so
         * rendering it here would count it twice.
         */
        LAND_MANA,
        ACTIVATED,
        /**
         * A spell ability that is also a trigger (ImmediateTrigger) or a
         * replacement effect (ETBReplacement), and is rendered by that phase
         * rather than twice.
         */
        DUPLICATE_TRAIT,
        SPELL,
        /** Anything else: it renders no line, but is still attributed. */
        UNRENDERED
    }

    private void parseSpellAbilities(FaceParse p) {
        boolean isAlternateFace = p.face.getType().hasSubtype("Adventure")
                || p.face.getType().hasSubtype("Omen");
        int spellIndex = -1;
        for (SpellAbility sa : p.card.getSpellAbilities()) {
            spellIndex++;
            // Declared before any route skips the trait, so a trait the parser
            // passes over lands in dropped_keys rather than failing the join:
            // it is still live at runtime and a record can still name it.
            ProvenanceKey spellKey = p.key(TraitKind.SPELL, spellIndex);
            p.recorder.declare(spellKey);
            SpellRoute route = classifySpell(sa, p, isAlternateFace);
            switch (route) {
                case KEYWORD_SPELL ->
                        p.recorder.attributeToKeyword(spellKey, sa, sa.getKeyword().getOriginal());
                case OTHER_FACE, DUPLICATE_TRAIT -> { }
                case LAND_MANA -> {
                    p.landManaAbilities.add(sa);
                    p.landManaKeys.add(spellKey);
                }
                case CHARM, ALT_COST, ACTIVATED, SPELL, UNRENDERED -> {
                    int before = p.abilities.size();
                    renderSpell(route, sa, p.abilities);
                    p.recorder.attribute(p.since(before), spellKey, sa);
                }
            }
        }
    }

    private static SpellRoute classifySpell(SpellAbility sa, FaceParse p,
                                            boolean isAlternateFace) {
        if (sa.getKeyword() != null) return SpellRoute.KEYWORD_SPELL;
        // Both the SA's own Adventure/Omen flag and its state name are asked:
        // the state name is the fallback when the flag cannot resolve the type.
        if (!isAlternateFace && (sa.isAdventure() || sa.isOmen()
                || (sa.getCardState() != null
                        && sa.getCardState().getStateName() == CardStateName.Secondary))) {
            return SpellRoute.OTHER_FACE;
        }
        if (sa.getApi() == ApiType.Charm) return SpellRoute.CHARM;
        if (sa.isSpell() && "True".equals(sa.getParam("NonBasicSpell"))) return SpellRoute.ALT_COST;
        if (sa.isActivatedAbility()) {
            return sa.isManaAbility() && p.landManaDescription != null
                    ? SpellRoute.LAND_MANA : SpellRoute.ACTIVATED;
        }
        if (sa.isSpell()) {
            return sa.getApi() == ApiType.ImmediateTrigger
                    || "ETBReplacement".equals(sa.getParam("Mode"))
                    ? SpellRoute.DUPLICATE_TRAIT : SpellRoute.SPELL;
        }
        return SpellRoute.UNRENDERED;
    }

    /** Append the lines a rendering route makes of {@code sa}. */
    private static void renderSpell(SpellRoute route, SpellAbility sa, List<Ability> out) {
        switch (route) {
            case CHARM -> out.addAll(CharmAbility.fromSpellAbility(sa));
            case ALT_COST -> addIfNotNull(out, AlternateCostSpell.of(sa));
            case ACTIVATED -> addIfNotNull(out, ActivatedAbilityEntry.of(sa));
            case SPELL -> {
                addIfNotNull(out, SpellAdditionalCost.of(sa));
                out.addAll(SpellEffect.fromChain(sa));
                out.addAll(DiceOutcomes.of(sa, text -> new SpellEffect(text, List.of())));
            }
            default -> { }
        }
    }

    /**
     * Wrap each trigger, skipping the secondary half of a pair.
     *
     * <p>When a secondary trigger re-uses its primary's Execute SVar, it is the
     * "blocks" (or "dies") half of an "attacks or blocks" pair — the primary
     * already renders both, so the secondary adds no line. Both runtime objects
     * stay live, so the one line carries both keys.
     */
    private void parseTriggers(FaceParse p) {
        // The ability each Execute SVar's primary rendered, so the secondary
        // half can be attributed to the one line both halves share.
        Set<String> primaryExecuteSVars = new HashSet<>();
        Map<String, Ability> primaryByExecute = new HashMap<>();
        int triggerIndex = -1;
        for (Trigger t : p.card.getTriggers()) {
            triggerIndex++;
            ProvenanceKey key = p.key(TraitKind.TRIGGER, triggerIndex);
            p.recorder.declare(key);
            String exec = t.getParam("Execute");
            boolean secondary = "True".equalsIgnoreCase(t.getParam("Secondary"));
            if (secondary && exec != null && primaryExecuteSVars.contains(exec)) {
                p.recorder.attribute(primaryByExecute.get(exec), key, t);
                continue;
            }
            if (exec != null && !secondary) {
                primaryExecuteSVars.add(exec);
            }
            Ability triggered = TriggeredAbilityEntry.of(t);
            addIfNotNull(p.abilities, triggered);
            attributeTrait(p, triggered, key, t);
            if (triggered != null && exec != null) primaryByExecute.put(exec, triggered);
        }
    }

    /** Wrap each static or replacement trait directly in its entry. */
    private <T extends CardTraitBase> void parseTraits(FaceParse p, Iterable<T> traits,
                                                       TraitKind kind,
                                                       Function<T, Ability> factory) {
        int index = -1;
        for (T trait : traits) {
            index++;
            ProvenanceKey key = p.key(kind, index);
            p.recorder.declare(key);
            Ability entry = factory.apply(trait);
            addIfNotNull(p.abilities, entry);
            attributeTrait(p, entry, key, trait);
        }
    }

    /**
     * Record what a trait rendered: its own entry, or — for a keyword-derived
     * trait that renders none of its own — the keyword's line.
     */
    private static void attributeTrait(FaceParse p, Ability entry, ProvenanceKey key,
                                       CardTraitBase trait) {
        if (entry == null && trait.getKeyword() != null) {
            p.recorder.attributeToKeyword(key, trait, trait.getKeyword().getOriginal());
        } else {
            p.recorder.attribute(entry, key, trait);
        }
    }

    /** The basic land types' mana line, claiming the mana abilities the spell phase skipped. */
    private static void appendLandMana(FaceParse p) {
        if (p.landManaDescription == null) return;
        TextAbility landMana = new TextAbility(AbilityType.ACTIVATED, p.landManaDescription);
        p.abilities.add(landMana);
        for (int i = 0; i < p.landManaAbilities.size(); i++) {
            p.recorder.attribute(landMana, p.landManaKeys.get(i), p.landManaAbilities.get(i));
        }
    }

    private void postProcess(FaceParse p) {
        if (p.face.getType().toString().contains("Class")) {
            p.abilities = applyClassPostProcessing(p.abilities, p.classLevelDescriptions,
                    p.recorder);
        }
        // Remove abilities whose description duplicates an earlier one.
        // This eliminates the spurious second trigger that Forge registers for
        // the "enters or attacks" pattern (two T: lines, identical TriggerDescription).
        p.abilities = deduplicateByDescription(p.abilities, p.recorder);
        p.abilities = sortCostsFirst(p.abilities);
    }

    private static CardFace buildCardFace(ICardFace face, List<Ability> abilities) {
        String name = AbilityDescription.applyCasing(face.getName());
        ManaCost manaCost = face.getManaCost();
        String manaCostStr = (manaCost == null || manaCost == ManaCost.NO_COST)
                ? null : manaCost.getSimpleString();

        String typeLine = formatTypeLine(face);
        String pt = (face.getPower() != null && face.getToughness() != null)
                ? face.getPower() + "/" + face.getToughness() : null;
        String loyalty = Strings.emptyToNull(face.getInitialLoyalty());
        String defense = Strings.emptyToNull(face.getDefense());

        String text = Strings.emptyToNull(face.getNonAbilityText());
        if (text != null) {
            text = DEVELOPER_NOTE.matcher(text).replaceAll("").strip();
            if (text.isEmpty()) text = null;
        }
        if (text != null) text = AbilityDescription.applyCasing(text);

        return new CardFace(name, manaCostStr, typeLine, pt, loyalty, defense, null, text, abilities);
    }

    // An undefined keyword is a script keyword Forge has no Keyword constant
    // for; its rendering depends only on how its original text starts.
    private static final Map<String, BiConsumer<KeywordInterface, FaceParse>>
            UNDEFINED_KEYWORDS = undefinedKeywordHandlers();

    private static Map<String, BiConsumer<KeywordInterface, FaceParse>> undefinedKeywordHandlers() {
        Map<String, BiConsumer<KeywordInterface, FaceParse>> handlers = new LinkedHashMap<>();
        BiConsumer<KeywordInterface, FaceParse> nameStatic = (ki, p) -> p.abilities.add(
                new TextAbility(AbilityType.STATIC, AbilityDescription.applyCasing(ki.getOriginal())));
        handlers.put("CARDNAME ", nameStatic);
        handlers.put("NICKNAME ", nameStatic);
        handlers.put("Chapter:", (ki, p) -> p.abilities.addAll(ChapterAbility.fromKeyword(ki)));
        handlers.put("Class:", (ki, p) -> {
            ClassLevelAbility level = ClassLevelAbility.of(ki);
            if (level != null) {
                p.classLevelDescriptions.add(level.innerDescription());
                p.abilities.add(level);
            }
        });
        BiConsumer<KeywordInterface, FaceParse> etb =
                (ki, p) -> p.abilities.addAll(EtbReplacementAbility.fromKeyword(ki));
        handlers.put("etbCounter:", etb);
        handlers.put("ETBReplacement:", etb);
        handlers.put("AlternateAdditionalCost:", (ki, p) -> p.abilities.add(new TextAbility(
                AbilityType.ADDITIONAL_COST, AbilityDescription.applyCasing(
                        buildAlternateAdditionalCostDescription(ki.getOriginal())))));
        handlers.put("Visit:", (ki, p) -> p.abilities.addAll(visitAbilities(ki)));
        BiConsumer<KeywordInterface, FaceParse> openingEffect =
                (ki, p) -> addIfNotNull(p.abilities, openingEffectAbility(ki));
        handlers.put("MayEffectFromOpeningHand:", openingEffect);
        handlers.put("MayEffectFromOpeningDeck:", openingEffect);
        return handlers;
    }

    private void routeUndefinedKeyword(KeywordInterface ki, FaceParse p) {
        String original = ki.getOriginal();
        for (Map.Entry<String, BiConsumer<KeywordInterface, FaceParse>> handler
                : UNDEFINED_KEYWORDS.entrySet()) {
            if (original.startsWith(handler.getKey())) {
                handler.getValue().accept(ki, p);
                return;
            }
        }
        p.abilities.add(StandardKeyword.of(ki, Keyword.UNDEFINED));
    }

    /**
     * An Attraction's visit lines.
     *
     * <p>For Charm-based visits, the TriggerDescription is the header and the
     * Charm options are OPTION sub-abilities. For non-Charm visits, the header
     * comes from the Execute SA's SpellDescription: Forge generates
     * TriggerDescription = "Visit — " + SpellDescription, so when SpellDescription
     * already starts with "Visit — " the result is doubled. Using SpellDescription
     * directly with an ensured single "Visit — " prefix avoids the doubled header.
     * Children are then empty: the SpellDescription is the complete oracle text
     * and SpellEffect.fromChain would only duplicate it.
     */
    private static List<Ability> visitAbilities(KeywordInterface ki) {
        List<Ability> visits = new ArrayList<>();
        for (Trigger t : ki.getTriggers()) {
            SpellAbility overriding = t.getOverridingAbility();
            String header;
            List<Ability> children = List.of();
            if (overriding != null && overriding.getApi() == ApiType.Charm) {
                header = triggerHeader(t);
                children = CharmAbility.optionsFrom(overriding, t);
            } else {
                String spellDesc = overriding != null ? overriding.getParam("SpellDescription") : null;
                if (spellDesc == null || spellDesc.isEmpty()) {
                    // Fallback: the TriggerDescription, whose prefix may be doubled.
                    header = triggerHeader(t);
                } else {
                    if (!spellDesc.startsWith("Visit — ")) {
                        spellDesc = "Visit — " + spellDesc;
                    }
                    header = AbilityDescription.normalize(spellDesc);
                }
            }
            if (header == null) continue;
            visits.add(new TriggeredAbilityEntry(AbilityType.TRIGGERED, header, children));
        }
        return visits;
    }

    /** A trigger's normalized first description line, or null if it prints none. */
    private static String triggerHeader(Trigger t) {
        String tDesc = t.getParam("TriggerDescription");
        if (tDesc == null || "Blank".equals(tDesc)) return null;
        return AbilityDescription.normalize(AbilityDescription.firstLine(tDesc));
    }

    /**
     * The effect a "MayEffectFromOpeningHand:SvarName" or
     * "MayEffectFromOpeningDeck:SvarName" keyword offers, or null.
     *
     * <p>The Forge game engine processes this keyword at game-start (GameAction.java)
     * rather than registering it as a standard trigger/spell. The description comes
     * from an SA built from the named SVar, falling back to a RevealCard SVar used by
     * Chancellor-cycle cards where the named SVar lacks a SpellDescription.
     */
    private static Ability openingEffectAbility(KeywordInterface ki) {
        String svarName = new KeywordSpec(ki.getOriginal()).part(1);
        Card hostCard = ki.getHostCard();
        if (hostCard == null) return null;
        String desc = null;
        if (hostCard.hasSVar(svarName)) {
            SpellAbility svarSa = AbilityFactory.getAbility(hostCard, svarName);
            if (svarSa != null) desc = svarSa.getParam("SpellDescription");
        }
        if ((desc == null || desc.isEmpty()) && hostCard.hasSVar("RevealCard")) {
            SpellAbility revealSa = AbilityFactory.getAbility(hostCard, "RevealCard");
            if (revealSa != null) desc = revealSa.getParam("SpellDescription");
        }
        String normalized = AbilityDescription.normalize(desc);
        return normalized == null ? null : new TextAbility(AbilityType.TRIGGERED, normalized);
    }

    private List<Ability> applyClassPostProcessing(List<Ability> abilities,
                                                   Set<String> classLevelDescriptions,
                                                   ProvenanceRecorder recorder) {
        List<Ability> result = new ArrayList<>(abilities);

        // A level's effect reaches the parser twice — once through the Class
        // keyword that prints the level, once as the trait itself — and only
        // the keyword's entry survives. It therefore takes over the removed
        // entry's keys, or a record fired by that trait names no line at all.
        Map<String, Ability> levelByDescription = new HashMap<>();
        for (Ability a : result) {
            if (a.type() != AbilityType.LEVEL) continue;
            // A Class level prints its level-up cost ahead of its text, so the
            // description the removal below matches is the level's inner text,
            // not its printed line. Keyed by the printed line, every lookup
            // misses and the removed trait's keys are lost.
            levelByDescription.putIfAbsent(
                    a instanceof ClassLevelAbility level
                            ? level.innerDescription() : a.descriptionText(), a);
        }
        Iterator<Ability> it = result.iterator();
        while (it.hasNext()) {
            Ability a = it.next();
            if (a.type() != AbilityType.LEVEL
                    && classLevelDescriptions.contains(a.descriptionText())) {
                recorder.merge(levelByDescription.get(a.descriptionText()), a);
                it.remove();
            }
        }

        for (int i = 0; i < result.size(); i++) {
            Ability a = result.get(i);
            if (a.type() == AbilityType.STATIC || a.type() == AbilityType.TRIGGERED
                    || a.type() == AbilityType.REPLACEMENT) {
                // The level-one ability is re-typed as a fresh object, and the
                // recorder keys by identity, so its attribution moves with it.
                TextAbility level =
                        new TextAbility(AbilityType.LEVEL, a.descriptionText(), 1);
                recorder.transfer(a, level);
                result.set(i, level);
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

    private static String buildAlternateAdditionalCostDescription(String original) {
        List<String> costs = new ArrayList<>();
        for (String costPart : new KeywordSpec(original).rest(1).split(":")) {
            Cost cost = new Cost(costPart, false);
            String costText = cost.toSimpleString();
            costs.add((cost.isOnlyManaCost() ? "pay " : "")
                    + costText.substring(0, 1).toLowerCase() + costText.substring(1));
        }
        return joinWithOr(costs);
    }

    /**
     * {@code a or b} for two items and {@code a, b, or c} for more — the
     * serial comma Forge's own scripts print, as in Arcane Sanctum's "Add {W},
     * {U}, or {B}." — so a line the converter builds reads like one a script
     * wrote.
     */
    static String joinWithOr(List<String> items) {
        if (items.size() <= 2) return String.join(" or ", items);
        return String.join(", ", items.subList(0, items.size() - 1))
                + ", or " + items.get(items.size() - 1);
    }

    private Card buildFullCard(CardRules rules) {
        PaperCard paperCard = new PaperCard(rules, "UNK", CardRarity.Common);
        return CardFactory.getCard(paperCard, null, nextCardId++, OfflineGame.instance());
    }

    /**
     * Basic land types in colour-wheel order, which is the order the symbols
     * are rendered in. A list rather than a {@code Map.of}: that map's
     * iteration order is salted per JVM, so a dual land rendered
     * {@code {W} or {U}} in one run and {@code {U} or {W}} in the next, and a
     * converted corpus could not be reproduced byte for byte.
     */
    private static final List<Map.Entry<String, String>> LAND_TYPE_MANA = List.of(
            Map.entry("Plains",   "{W}"),
            Map.entry("Island",   "{U}"),
            Map.entry("Swamp",    "{B}"),
            Map.entry("Mountain", "{R}"),
            Map.entry("Forest",   "{G}")
    );

    static String buildLandManaDescription(ICardFace face) {
        List<String> symbols = new ArrayList<>();
        for (Map.Entry<String, String> entry : LAND_TYPE_MANA) {
            if (face.getType().hasSubtype(entry.getKey())) {
                symbols.add(entry.getValue());
            }
        }
        if (symbols.isEmpty()) {
            return null;
        }
        // Shaped like a scripted mana ability's line after casing — "{T}: add
        // {W}, {U}, or {B}." — so a triome and a tri-land print one sentence.
        return "{T}: add " + joinWithOr(symbols) + ".";
    }

    private static String formatTypeLine(ICardFace face) {
        String typeStr = face.getType().toString();
        typeStr = typeStr.replace(" - ", " ");
        return typeStr.toLowerCase();
    }

    private static void addIfNotNull(List<Ability> list, Ability item) {
        if (item != null) list.add(item);
    }

    /**
     * Remove abilities whose descriptionText duplicates an earlier entry.
     *
     * <p>The survivor inherits the discarded duplicate's provenance: one line
     * now carries the text of both traits, and both stay live at runtime, so a
     * record fired by either has to join to that line.
     */
    private static List<Ability> deduplicateByDescription(List<Ability> abilities,
                                                          ProvenanceRecorder recorder) {
        Map<String, Ability> survivors = new HashMap<>();
        List<Ability> result = new ArrayList<>();
        for (Ability a : abilities) {
            String desc = a.descriptionText();
            if (desc == null) {
                result.add(a);
                continue;
            }
            Ability survivor = survivors.putIfAbsent(desc, a);
            if (survivor == null) {
                result.add(a);
            } else {
                recorder.merge(survivor, a);
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

        return card.withFirstFace(first.withAbilities(fallback));
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
        return card.withFirstFace(first.withAbilities(combined));
    }
}
