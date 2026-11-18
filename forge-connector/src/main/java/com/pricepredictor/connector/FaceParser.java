package com.pricepredictor.connector;

import com.google.common.base.Strings;
import com.pricepredictor.connector.ability.ActivatedAbilityEntry;
import com.pricepredictor.connector.ability.AlternateCostSpell;
import com.pricepredictor.connector.ability.ChapterAbility;
import com.pricepredictor.connector.ability.CharmAbility;
import com.pricepredictor.connector.ability.CharmHeader;
import com.pricepredictor.connector.ability.ClassLevelAbility;
import com.pricepredictor.connector.ability.CompanionKeyword;
import com.pricepredictor.connector.ability.DescriptionMarkers;
import com.pricepredictor.connector.ability.EtbReplacementAbility;
import com.pricepredictor.connector.ability.FaceOracle;
import com.pricepredictor.connector.ability.GiftKeyword;
import com.pricepredictor.connector.ability.HauntKeyword;
import com.pricepredictor.connector.ability.KeywordSpec;
import com.pricepredictor.connector.ability.ReplacementAbilityEntry;
import com.pricepredictor.connector.ability.SpellLine;
import com.pricepredictor.connector.ability.StandardKeyword;
import com.pricepredictor.connector.ability.StaticAbilityEntry;
import com.pricepredictor.connector.ability.TextAbility;
import com.pricepredictor.connector.ability.TriggerDescribedKeyword;
import com.pricepredictor.connector.ability.TriggeredAbilityEntry;
import com.pricepredictor.connector.ability.UndescribedSpellText;
import com.pricepredictor.connector.effects.FaceOrder;
import com.pricepredictor.connector.effects.ProvenanceKey;
import com.pricepredictor.connector.effects.ProvenanceRecorder;
import com.pricepredictor.connector.effects.TraitKind;
import forge.card.CardStateName;
import forge.card.ICardFace;
import forge.card.MagicColor;
import forge.card.mana.ManaCost;
import forge.game.CardTraitBase;
import forge.game.ability.AbilityFactory;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.cost.Cost;
import forge.game.keyword.Companion;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.regex.Pattern;

/**
 * One card face's parse, optionally recording provenance.
 *
 * <p>Each phase of {@link #parse} appends to the face's ability list, and a
 * final pass reshapes it; one parser serves one face, and the list, the
 * recorder and whatever a later phase needs from an earlier one are its own
 * state.
 *
 * <p>Recording happens here rather than inside the ability implementations
 * because this is the only place both the runtime trait and its index within
 * its kind are in scope, and {@code index_within_kind} is half the provenance
 * key. Each phase records the lines it rendered for a trait as it renders
 * them, so a factory that returns one ability and one that returns several
 * are handled identically; a keyword's rendering also names the line each
 * trait it built printed ({@link KeywordRendering}).
 */
final class FaceParser {

    /** A {@code [Developer's note: …]} bracket: Forge-internal metadata, not oracle content. */
    private static final Pattern DEVELOPER_NOTE =
            Pattern.compile("(?i)\\[Developer's note:[^]]*]");

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

    /** Cost lines first; within each group, the order the phases produced. */
    private static final Comparator<Ability> COSTS_FIRST =
            Comparator.comparing((Ability a) -> !a.type().isCostType());

    /** A spell ability the spell phase set aside, with the key it is recorded under. */
    private record KeyedSpell(SpellAbility ability, ProvenanceKey key) {}

    private final Card card;
    private final ICardFace face;
    private final ProvenanceRecorder recorder;
    private final int faceIndex;
    /** The tree-prefixed script path keys name, or null when not recording. */
    private final String scriptFile;
    private final List<Ability> abilities = new ArrayList<>();
    /** Each Class level's inner text, so its duplicate trait entry can be removed. */
    /** The lines of secondary triggers whose Execute$ no primary shares (identity set). */
    private final Set<Ability> secondaryLines = Collections.newSetFromMap(new IdentityHashMap<>());
    private final Set<String> classLevelDescriptions = new HashSet<>();
    /** The synthetic mana line of a basic-typed face, or null. */
    private final String landManaDescription;
    /**
     * The mana abilities a basic-typed face's land types build, skipped by the spell
     * phase and rendered once by the synthetic land-mana line. Kept with
     * their keys so that line can claim them: without the claim the land's
     * mana ability is a dropped key, and every activation record that names
     * it joins to nothing.
     */
    private final List<KeyedSpell> landManaTraits = new ArrayList<>();
    /**
     * Spells cast for an alternative cost (cleave), whose cost line declares
     * no key: each is claimed by the face's spell line, whose bracketed words
     * state both versions of the spell.
     */
    private final List<KeyedSpell> alternateCostSpells = new ArrayList<>();
    /**
     * The spells of an instant or sorcery face whose chain states nothing,
     * given their text once every other line of the face is known
     * ({@link #addUndescribedSpells}).
     */
    private final List<KeyedSpell> undescribedSpells = new ArrayList<>();
    /** Where the first of {@link #undescribedSpells} would have printed. */
    private int undescribedAt;
    /** The first line a spell of this face printed, or null. */
    private Ability spellLine;

    /**
     * @param card        the fully built card, set to the state {@code face} shows
     * @param recorder    collects trait-to-ability attribution
     * @param faceIndex   this face's ordinal in the converter's emission order
     * @param scriptFile  the tree-prefixed path of the card's script, or null
     *                    to record nothing
     */
    FaceParser(Card card, ICardFace face, ProvenanceRecorder recorder,
               int faceIndex, String scriptFile) {
        this.card = card;
        this.face = face;
        this.recorder = recorder;
        this.faceIndex = faceIndex;
        this.scriptFile = scriptFile;
        this.landManaDescription = buildLandManaDescription(face);
    }

    /** Run every phase, in order, and build the face from what they produced. */
    CardFace parse() {
        parseKeywords();
        parseSpellAbilities();
        parseTriggers();
        parseTraits(card.getStaticAbilities(), TraitKind.STATIC, StaticAbilityEntry::of);
        parseTraits(card.getReplacementEffects(), TraitKind.REPLACEMENT,
                ReplacementAbilityEntry::of);
        appendLandMana();
        addUndescribedSpells();
        claimAlternateCostSpells();
        List<Ability> finished = postProcessed();
        recorder.claimModesUnderEveryRoot(finished);
        return buildCardFace(face, finished);
    }

    /**
     * The key of this face's {@code index}-th trait of {@code kind}, or null
     * when not recording. Every recorder call takes a null key as "nothing
     * to record", which is what lets the phases record unconditionally.
     */
    private ProvenanceKey key(TraitKind kind, int index) {
        return scriptFile == null ? null
                : new ProvenanceKey(scriptFile, faceIndex, kind, index);
    }

    /**
     * The key of {@code ki}'s line, or null when not recording or when the
     * keyword has no intrinsic ordinal (a granted one).
     */
    private ProvenanceKey keywordKey(KeywordInterface ki) {
        int keywordIndex = FaceOrder.keywordIndex(card.getCurrentState(), ki);
        return keywordIndex < 0 ? null : key(TraitKind.KEYWORD, keywordIndex);
    }

    // -- keywords --------------------------------------------------------

    /**
     * Route each keyword to the ability that renders it.
     *
     * <p>The keyword ordinal is asked of FaceOrder rather than counted
     * here, because the collector asks the same question of the same method
     * at runtime and the two answers have to be the same number. A running
     * counter over getKeywords() is not that number: the iteration order is
     * KeywordCollection's, the script's insertion order only under the
     * {@code -android} Guava Forge pins and an identity-hash order under any
     * other flavour. Counted here, the two
     * sides produced keys that were in range on both and named different
     * keywords — a silent wrong join rather than a loud failure. A keyword
     * with no intrinsic ordinal (a granted one) records no key at all,
     * which is the honest answer and the one the collector gives for it.
     *
     * <p>The keywords are walked in {@link FaceOrder#keywordOrder}, so the
     * lines of a face's keywords come out in the order their keys number them
     * (cost keywords first, by the later sort), whatever the script's order.
     *
     * <p>The traits a keyword built are named through this key alone: the
     * recorder declares it, or the {@code built} keys of those traits, on the
     * keyword's lines, and the later phases skip the traits' kind-indexed keys.
     */
    private void parseKeywords() {
        for (KeywordInterface ki : FaceOrder.keywordOrder(card.getKeywords())) {
            KeywordRendering rendering = renderKeyword(ki);
            abilities.addAll(rendering.lines());
            recorder.attributeKeyword(rendering, keywordKey(ki), ki);
        }
    }

    /** The lines {@code ki} prints. */
    private KeywordRendering renderKeyword(KeywordInterface ki) {
        Keyword kw = ki.getKeyword();
        if (kw == Keyword.UNDEFINED) return renderUndefinedKeyword(ki);
        if (kw == Keyword.HAUNT) return HauntKeyword.of(ki, card);
        if (kw == Keyword.GIFT) return KeywordRendering.of(GiftKeyword.of(ki, card));
        if (kw == Keyword.BACKUP) return backupLine(ki);
        if (kw == Keyword.COMPANION && ki instanceof Companion comp) {
            return KeywordRendering.of(CompanionKeyword.of(ki, comp));
        }
        return KeywordRendering.of(StandardKeyword.of(ki, kw));
    }

    // An undefined keyword is a script keyword Forge has no Keyword constant
    // for; its rendering depends only on how its original text starts.
    private static final Map<String, KeywordRenderer> UNDEFINED_KEYWORDS =
            undefinedKeywordHandlers();

    /** How one kind of undefined keyword renders. */
    private interface KeywordRenderer {
        KeywordRendering render(KeywordInterface ki, FaceParser parser);
    }

    /** The face's Oracle text, for a keyword line whose script states nothing. */
    private FaceOracle oracle() {
        return new FaceOracle(face.getOracleText(), face.getName());
    }

    private static Map<String, KeywordRenderer> undefinedKeywordHandlers() {
        Map<String, KeywordRenderer> handlers = new LinkedHashMap<>();
        KeywordRenderer nameStatic = (ki, p) -> KeywordRendering.of(new TextAbility(
                AbilityType.STATIC, AbilityDescription.applyCasing(ki.getOriginal())));
        handlers.put("CARDNAME ", nameStatic);
        handlers.put("NICKNAME ", nameStatic);
        handlers.put("Chapter:", (ki, p) -> ChapterAbility.fromKeyword(ki, p.oracle()));
        handlers.put("Class:", (ki, p) -> p.renderClassLevel(ki));
        KeywordRenderer etb = (ki, p) -> EtbReplacementAbility.fromKeyword(ki, p.oracle());
        handlers.put("etbCounter:", etb);
        handlers.put("ETBReplacement:", etb);
        handlers.put("AlternateAdditionalCost:", (ki, p) -> KeywordRendering.of(new TextAbility(
                AbilityType.ADDITIONAL_COST, AbilityDescription.applyCasing(
                        buildAlternateAdditionalCostDescription(ki.getOriginal())))));
        handlers.put("Visit:", (ki, p) -> visitAbilities(ki));
        handlers.put("Dungeon:", (ki, p) -> TriggerDescribedKeyword.rooms(ki));
        handlers.put("Prize:", (ki, p) -> TriggerDescribedKeyword.prize(ki));
        KeywordRenderer openingEffect = (ki, p) -> openingEffect(ki);
        handlers.put("MayEffectFromOpeningHand:", openingEffect);
        handlers.put("MayEffectFromOpeningDeck:", openingEffect);
        return handlers;
    }

    private KeywordRendering renderUndefinedKeyword(KeywordInterface ki) {
        String original = ki.getOriginal();
        for (Map.Entry<String, KeywordRenderer> handler : UNDEFINED_KEYWORDS.entrySet()) {
            if (original.startsWith(handler.getKey())) {
                return handler.getValue().render(ki, this);
            }
        }
        return KeywordRendering.of(StandardKeyword.of(ki, Keyword.UNDEFINED));
    }

    /**
     * A Class level's line, its inner text kept so its duplicate trait entry
     * can go; the line is the text of the level-up ability and of the static
     * granting what the level adds, which Forge builds from the keyword.
     */
    private KeywordRendering renderClassLevel(KeywordInterface ki) {
        ClassLevelAbility level = ClassLevelAbility.of(ki);
        if (level == null) return KeywordRendering.of(List.of());
        classLevelDescriptions.add(level.innerDescription());
        KeywordRendering.Builder rendering = new KeywordRendering.Builder();
        rendering.add(level);
        ki.getAbilities().forEach(levelUp -> rendering.printed(levelUp, level));
        ki.getStaticAbilities().forEach(grant -> rendering.chained(grant, level));
        return rendering.build();
    }

    /**
     * Backup's line, the text of the trigger Forge builds for it: its
     * counter, and below it the abilities the keyword's SVar grants.
     */
    private static KeywordRendering backupLine(KeywordInterface ki) {
        Trigger trigger = ki.getTriggers().stream().findFirst().orElse(null);
        return KeywordRendering.ofTrait(StandardKeyword.of(ki, Keyword.BACKUP), trigger);
    }

    /**
     * An Attraction's visit lines, each the text of its trigger.
     *
     * <p>For Charm-based visits, the TriggerDescription is the header and the
     * Charm options are OPTION sub-abilities. Any other visit takes its header
     * from {@link #visitHeader}, and has no children: its SpellDescription is
     * the complete oracle text and SpellEffect.fromChain would only duplicate
     * it.
     */
    private static KeywordRendering visitAbilities(KeywordInterface ki) {
        KeywordRendering.Builder visits = new KeywordRendering.Builder();
        for (Trigger t : ki.getTriggers()) {
            SpellAbility overriding = t.getOverridingAbility();
            boolean charm = overriding != null && overriding.getApi() == ApiType.Charm;
            String header = charm ? charmVisitHeader(overriding) : visitHeader(t, overriding);
            if (header == null) continue;
            List<Ability> children =
                    charm ? CharmAbility.optionsFrom(overriding, t) : List.of();
            Ability visit = new TriggeredAbilityEntry(AbilityType.TRIGGERED, header, children);
            visits.add(visit);
            visits.printed(t, visit);
        }
        return visits.build();
    }

    /**
     * A non-Charm visit's header, or null if it prints none.
     *
     * <p>The header comes from the Execute SA's SpellDescription: Forge
     * generates TriggerDescription = "Visit — " + SpellDescription, so when
     * SpellDescription already starts with "Visit — " the TriggerDescription
     * doubles it. Using SpellDescription directly with an ensured single
     * "Visit — " prefix avoids the doubled header.
     */
    private static String visitHeader(Trigger t, SpellAbility overriding) {
        String spellDesc = overriding != null ? overriding.getParam("SpellDescription") : null;
        if (spellDesc == null || spellDesc.isEmpty()) {
            // Fallback: the TriggerDescription, whose prefix may be doubled.
            return triggerHeader(t);
        }
        return AbilityDescription.normalize(
                spellDesc.startsWith("Visit — ") ? spellDesc : "Visit — " + spellDesc);
    }

    /**
     * A charm visit's header: Forge writes "Visit — " and the charm's
     * description, whose header is the charm's choice sentence.
     */
    private static String charmVisitHeader(SpellAbility charm) {
        return AbilityDescription.normalize("Visit — " + CharmHeader.of(charm));
    }

    /** A trigger's normalized first description line, or null if it prints none. */
    private static String triggerHeader(Trigger t) {
        String tDesc = t.getParam("TriggerDescription");
        if (DescriptionMarkers.printsNoLine(tDesc)) return null;
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
     * The line is the text of the named SVar's ability, the one Forge plays.
     */
    private static KeywordRendering openingEffect(KeywordInterface ki) {
        Card hostCard = ki.getHostCard();
        if (hostCard == null) return KeywordRendering.of(List.of());
        SpellAbility named = svarAbility(hostCard, new KeywordSpec(ki.getOriginal()).part(1));
        SpellAbility described = spellDescription(named) != null
                ? named : svarAbility(hostCard, "RevealCard");
        String normalized = AbilityDescription.normalize(spellDescription(described));
        Ability line = normalized == null ? null : new TextAbility(AbilityType.TRIGGERED, normalized);
        return KeywordRendering.ofTrait(line, named != null ? named : described);
    }

    /** The ability the SVar {@code name} of {@code host} holds, or null. */
    private static SpellAbility svarAbility(Card host, String name) {
        return host.hasSVar(name) ? AbilityFactory.getAbility(host, name) : null;
    }

    /** An ability's own non-empty {@code SpellDescription$}, or null. */
    private static String spellDescription(SpellAbility sa) {
        String description = sa == null ? null : sa.getParam("SpellDescription");
        return description == null || description.isEmpty() ? null : description;
    }

    private static String buildAlternateAdditionalCostDescription(String original) {
        List<String> costs = new ArrayList<>();
        for (String costPart : new KeywordSpec(original).rest(1).split(":")) {
            Cost cost = new Cost(costPart, false);
            String costText = cost.toSimpleString();
            costs.add((cost.isOnlyManaCost() ? "pay " : "")
                    + costText.substring(0, 1).toLowerCase(Locale.ROOT) + costText.substring(1));
        }
        return joinWithOr(costs);
    }

    // -- spell abilities -------------------------------------------------

    /** Where the spell phase sends one of a face's spell abilities. */
    private enum SpellRoute {
        /**
         * A keyword's own spell. It renders as the keyword line, and is named
         * through that line's key: its own key is declared nowhere.
         */
        KEYWORD_SPELL,
        /**
         * An Adventure or Omen spell met while parsing the main face: it
         * belongs to the Secondary state and is parsed and keyed with that
         * face, as the collector keys it by its own state. The main face
         * declares no key of it.
         */
        OTHER_FACE,
        CHARM,
        /** A spell cast for an alternative cost ({@code NonBasicSpell}). */
        ALT_COST,
        /**
         * A mana ability Forge builds from a basic land type of the face. The
         * synthetic land mana line renders it once in canonical form;
         * CardFactory's fully resolved card lists it among the spell
         * abilities as well, so rendering it here would count it twice. A
         * mana ability the script writes is ACTIVATED.
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
        UNRENDERED;

        /** Whether this face declares the spell's kind-indexed key. */
        boolean keyedHere() {
            return this != KEYWORD_SPELL && this != OTHER_FACE;
        }
    }

    private void parseSpellAbilities() {
        boolean isAlternateFace = face.getType().hasSubtype("Adventure")
                || face.getType().hasSubtype("Omen");
        int spellIndex = -1;
        for (SpellAbility sa : card.getSpellAbilities()) {
            spellIndex++;
            SpellRoute route = classifySpell(sa, isAlternateFace);
            if (!route.keyedHere()) continue;
            // Declared before any other route skips the trait, so a trait the
            // parser passes over lands in dropped_keys rather than failing the
            // join: it is still live at runtime and a record can still name it.
            ProvenanceKey spellKey = key(TraitKind.SPELL, spellIndex);
            recorder.declare(spellKey);
            addRouted(route, sa, spellKey);
        }
    }

    /** Send {@code sa}, declared under {@code key}, where {@code route} says. */
    private void addRouted(SpellRoute route, SpellAbility sa, ProvenanceKey key) {
        switch (route) {
            case KEYWORD_SPELL, OTHER_FACE, DUPLICATE_TRAIT -> { }
            case LAND_MANA -> landManaTraits.add(new KeyedSpell(sa, key));
            case ALT_COST -> addAlternateCost(sa, key);
            case SPELL -> addSpell(sa, key);
            case CHARM, ACTIVATED, UNRENDERED -> addSpellLines(renderSpell(route, sa), key, sa);
        }
    }

    /**
     * A spell's lines; an instant's or sorcery's spell whose chain states
     * nothing is set aside for {@link #addUndescribedSpells}.
     */
    private void addSpell(SpellAbility sa, ProvenanceKey key) {
        if (!isInstantOrSorcery() || !SpellLine.statesNothing(sa)) {
            addSpellLines(SpellLine.linesOf(sa), key, sa);
            return;
        }
        if (undescribedSpells.isEmpty()) undescribedAt = abilities.size();
        undescribedSpells.add(new KeyedSpell(sa, key));
    }

    private boolean isInstantOrSorcery() {
        return face.getType().isInstant() || face.getType().isSorcery();
    }

    /**
     * Each set-aside spell's line, where the spell phase would have printed
     * it, from the text {@link UndescribedSpellText} finds against every
     * other line of the face; a spell with no text anywhere prints no line
     * and its key stays dropped.
     */
    private void addUndescribedSpells() {
        Set<String> stated = statedTexts();
        List<Ability> lines = new ArrayList<>();
        for (KeyedSpell spell : undescribedSpells) {
            String text = UndescribedSpellText.of(
                    spell.ability(), face.getOracleText(), face.getName(), stated);
            List<Ability> spellLines = SpellLine.linesOf(spell.ability(), text);
            lines.addAll(spellLines);
            attributeSpellLines(spellLines, spell.key(), spell.ability());
        }
        abilities.addAll(undescribedAt, lines);
    }

    /** The description texts of the face's lines so far, and of its {@code Text:}. */
    private Set<String> statedTexts() {
        Set<String> stated = new HashSet<>();
        for (Ability line : abilities) {
            if (line.descriptionText() != null) stated.add(line.descriptionText());
        }
        String text = AbilityDescription.scriptLine(face.getNonAbilityText(), face.getName());
        if (text != null) stated.add(text);
        return stated;
    }

    /**
     * Add {@code lines}, attributing to {@code key} those that state more
     * than a cost: a line stating only a cost (a permanent's additional
     * cost) declares no spell key.
     */
    private void addSpellLines(List<Ability> lines, ProvenanceKey key, SpellAbility sa) {
        abilities.addAll(lines);
        attributeSpellLines(lines, key, sa);
    }

    /** {@link #addSpellLines}'s attribution, for lines placed elsewhere. */
    private void attributeSpellLines(List<Ability> lines, ProvenanceKey key, SpellAbility sa) {
        List<Ability> stated = lines.stream().filter(line -> !line.type().isCostType()).toList();
        recorder.attribute(stated, key, sa);
        if (spellLine == null) {
            spellLine = stated.stream().filter(SpellLine.class::isInstance).findFirst().orElse(null);
        }
    }

    /** The cleave line, which states only a cost and declares no key. */
    private void addAlternateCost(SpellAbility sa, ProvenanceKey key) {
        addIfNotNull(abilities, AlternateCostSpell.of(sa));
        alternateCostSpells.add(new KeyedSpell(sa, key));
    }

    /** Each spell cast for an alternative cost joins the face's spell line. */
    private void claimAlternateCostSpells() {
        for (KeyedSpell spell : alternateCostSpells) {
            recorder.attribute(spellLine, spell.key(), spell.ability());
        }
    }

    private SpellRoute classifySpell(SpellAbility sa, boolean isAlternateFace) {
        if (sa.getKeyword() != null) return SpellRoute.KEYWORD_SPELL;
        if (!isAlternateFace && isSecondaryStateSpell(sa)) return SpellRoute.OTHER_FACE;
        if (sa.getApi() == ApiType.Charm) return SpellRoute.CHARM;
        if (sa.isActivatedAbility()) {
            return isIntrinsicLandMana(sa) ? SpellRoute.LAND_MANA : SpellRoute.ACTIVATED;
        }
        return sa.isSpell() ? castRoute(sa) : SpellRoute.UNRENDERED;
    }

    /**
     * Whether {@code sa} is an Adventure or Omen spell. Both the SA's own
     * flag and its state name are asked: the state name is the fallback when
     * the flag cannot resolve the type.
     */
    private static boolean isSecondaryStateSpell(SpellAbility sa) {
        if (sa.isAdventure() || sa.isOmen()) return true;
        CardState state = sa.getCardState();
        return state != null && state.getStateName() == CardStateName.Secondary;
    }

    /** The route of a spell that is neither a keyword's, another face's nor a charm. */
    private static SpellRoute castRoute(SpellAbility sa) {
        if ("True".equals(sa.getParam("NonBasicSpell"))) return SpellRoute.ALT_COST;
        boolean anotherTrait = sa.getApi() == ApiType.ImmediateTrigger
                || "ETBReplacement".equals(sa.getParam("Mode"));
        return anotherTrait ? SpellRoute.DUPLICATE_TRAIT : SpellRoute.SPELL;
    }

    /**
     * Whether {@code sa} is a mana ability Forge builds from one of this
     * face's basic land types ({@code CardState.getLandManaForColor}), which
     * the synthetic land-mana line stands for. A mana ability the script
     * writes (Murmuring Bosk's "{T}: Add {W} or {B}.") is not: it prints its
     * own line under its own key.
     */
    private boolean isIntrinsicLandMana(SpellAbility sa) {
        if (landManaDescription == null) return false;
        CardState state = card.getCurrentState();
        return Arrays.stream(MagicColor.Color.values())
                .filter(colour -> colour.getBasicLandType() != null
                        && face.getType().hasSubtype(colour.getBasicLandType()))
                .anyMatch(colour -> state.getLandManaForColor(colour) == sa);
    }

    /** The lines a rendering route makes of {@code sa}. */
    private static List<Ability> renderSpell(SpellRoute route, SpellAbility sa) {
        List<Ability> out = new ArrayList<>();
        switch (route) {
            case CHARM -> out.addAll(CharmAbility.fromSpellAbility(sa));
            case ACTIVATED -> addIfNotNull(out, ActivatedAbilityEntry.of(sa));
            default -> { }
        }
        return out;
    }

    // -- triggers, statics and replacements --------------------------------

    /**
     * Wrap each trigger, skipping the secondary half of a pair.
     *
     * <p>When a secondary trigger re-uses its primary's Execute SVar, it is the
     * "blocks" (or "dies") half of an "attacks or blocks" pair — the primary
     * already renders both, so the secondary adds no line. Both runtime objects
     * stay live, so the one line carries both keys.
     */
    private void parseTriggers() {
        // The ability each Execute SVar's primary rendered, so the secondary
        // half can be attributed to the one line both halves share.
        Set<String> primaryExecuteSVars = new HashSet<>();
        Map<String, Ability> primaryByExecute = new HashMap<>();
        int triggerIndex = -1;
        for (Trigger t : card.getTriggers()) {
            triggerIndex++;
            if (t.getKeyword() != null) {
                addBuiltTrait(TriggeredAbilityEntry.of(t), t);
            } else {
                parseOwnTrigger(t, key(TraitKind.TRIGGER, triggerIndex),
                        primaryExecuteSVars, primaryByExecute);
            }
        }
    }

    /**
     * A trigger the face states itself, under its kind-indexed {@code key}.
     *
     * <p>A secondary trigger that is also static prints no line and its key
     * is dropped: Forge's card view prints no secondary trait
     * ({@code Card.getAbilityText}), and a static one, resolving with no stack,
     * is the script's bookkeeping (Diabolic Servitude's "forget remembered
     * card if it goes into exile") or repeats a static's own text (Serra
     * Paragon).
     */
    private void parseOwnTrigger(Trigger t, ProvenanceKey key, Set<String> primaryExecuteSVars,
                                 Map<String, Ability> primaryByExecute) {
        recorder.declare(key);
        String exec = t.getParam("Execute");
        boolean secondary = t.isSecondary();
        // The set holds no null, so a trigger with no Execute$ is never found.
        if (secondary && primaryExecuteSVars.contains(exec)) {
            recorder.attribute(primaryByExecute.get(exec), key, t);
            return;
        }
        if (isBookkeeping(t)) return;
        if (exec != null && !secondary) {
            primaryExecuteSVars.add(exec);
        }
        Ability triggered = TriggeredAbilityEntry.of(t);
        addIfNotNull(abilities, triggered);
        noteSecondary(triggered, secondary);
        recorder.attribute(triggered, key, t);
        if (triggered != null && exec != null) primaryByExecute.put(exec, triggered);
    }

    private void noteSecondary(Ability line, boolean secondary) {
        if (secondary && line != null) secondaryLines.add(line);
    }

    /** Whether {@code t} is a secondary trigger that is also static: no card text. */
    private static boolean isBookkeeping(Trigger t) {
        return t.isSecondary() && t.isStatic();
    }

    /** Wrap each static or replacement trait directly in its entry. */
    private <T extends CardTraitBase> void parseTraits(Iterable<T> traits, TraitKind kind,
                                                       Function<T, Ability> factory) {
        int index = -1;
        for (T trait : traits) {
            index++;
            Ability entry = factory.apply(trait);
            if (trait.getKeyword() != null) {
                addBuiltTrait(entry, trait);
                continue;
            }
            ProvenanceKey key = key(kind, index);
            recorder.declare(key);
            addIfNotNull(abilities, entry);
            recorder.attribute(entry, key, trait);
        }
    }

    /**
     * A trait a keyword built. Its kind-indexed key is declared nowhere: its
     * text is its keyword's line, keyed in the keyword phase, or, as with
     * Tribute's unpaid-tribute trigger, a line of its own, which declares the
     * keyword's key with the trait's {@code built}.
     */
    private void addBuiltTrait(Ability entry, CardTraitBase trait) {
        addIfNotNull(abilities, entry);
        recorder.attributeBuilt(entry, keywordKey(trait.getKeyword()), trait);
    }

    // -- land mana ---------------------------------------------------------

    /** The basic land types' mana line, claiming the mana abilities the spell phase skipped. */
    private void appendLandMana() {
        if (landManaDescription == null) return;
        TextAbility landMana = new TextAbility(AbilityType.ACTIVATED, landManaDescription);
        abilities.add(landMana);
        for (KeyedSpell trait : landManaTraits) {
            recorder.attribute(landMana, trait.key(), trait.ability());
        }
    }

    private static String buildLandManaDescription(ICardFace face) {
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

    // -- the finished face -------------------------------------------------

    /** The phases' abilities, reshaped into the order and set the face prints. */
    private List<Ability> postProcessed() {
        List<Ability> result = abilities;
        if (face.getType().hasSubtype("Class")) {
            result = ClassLevels.arrange(result, classLevelDescriptions, recorder);
        }
        // Remove abilities whose description duplicates an earlier one.
        // This eliminates the spurious second trigger that Forge registers for
        // the "enters or attacks" pattern (two T: lines, identical TriggerDescription).
        result = deduplicateByDescription(result);
        result.sort(COSTS_FIRST);
        result = withoutRepeatedSecondaries(result);
        return result;
    }

    /**
     * Remove abilities whose descriptionText duplicates an earlier entry.
     *
     * <p>The survivor inherits the discarded duplicate's provenance: one line
     * now carries the text of both traits, and both stay live at runtime, so a
     * record fired by either has to join to that line.
     */
    private List<Ability> deduplicateByDescription(List<Ability> list) {
        Map<String, Ability> survivors = new HashMap<>();
        List<Ability> result = new ArrayList<>();
        for (Ability a : list) {
            String desc = a.descriptionText();
            Ability survivor = desc == null ? null : survivors.putIfAbsent(desc, a);
            if (survivor == null) {
                result.add(a);
            } else {
                mergeDuplicate(survivor, a);
            }
        }
        return result;
    }

    /** The survivor takes the duplicate's keys, and is no secondary line if the duplicate was not. */
    private void mergeDuplicate(Ability survivor, Ability duplicate) {
        if (!secondaryLines.contains(duplicate)) secondaryLines.remove(survivor);
        recorder.merge(survivor, duplicate);
    }

    /**
     * Drop each secondary trigger's line whose text another line of the face
     * restates (Coffin Queen's triggers repeat its activated line). The
     * trigger fires in play, so its keys go to the first line, in line order,
     * that states it, as a deduplicated trait's do. Forge's card view prints
     * no secondary trait, and a line that says something no other line does
     * keeps its place.
     */
    private List<Ability> withoutRepeatedSecondaries(List<Ability> list) {
        List<Ability> result = new ArrayList<>(list);
        for (Ability a : list) {
            if (!secondaryLines.contains(a) || a.descriptionText() == null) continue;
            Ability cover = result.stream().filter(other -> other != a
                    && other.descriptionText() != null
                    && RepeatedLines.repeats(a.descriptionText(), other.descriptionText()))
                    .findFirst().orElse(null);
            if (cover != null) {
                result.removeIf(x -> x == a);
                recorder.merge(cover, a);
            }
        }
        return result;
    }

    private static CardFace buildCardFace(ICardFace face, List<Ability> abilities) {
        String name = AbilityDescription.lowerCase(face.getName());
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
        if (text != null) text = AbilityDescription.lowerCase(text);

        return new CardFace(name, manaCostStr, typeLine, pt, loyalty, defense, text, abilities);
    }

    private static String formatTypeLine(ICardFace face) {
        String typeStr = face.getType().toString();
        typeStr = typeStr.replace(" - ", " ");
        return typeStr.toLowerCase(Locale.ROOT);
    }

    private static void addIfNotNull(List<Ability> list, Ability item) {
        if (item != null) list.add(item);
    }
}
