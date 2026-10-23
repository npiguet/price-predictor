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
import com.pricepredictor.connector.effects.FaceOrder;
import com.pricepredictor.connector.effects.ProvenanceKey;
import com.pricepredictor.connector.effects.ProvenanceRecorder;
import com.pricepredictor.connector.effects.TraitKind;
import forge.card.CardStateName;
import forge.card.ICardFace;
import forge.card.mana.ManaCost;
import forge.game.CardTraitBase;
import forge.game.ability.AbilityFactory;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.cost.Cost;
import forge.game.keyword.Companion;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;
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
 * key. Each phase records the abilities it appended by bracketing the list, so
 * a factory that returns one ability and one that returns several are handled
 * identically.
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

    /** A basic-typed face's runtime mana ability, with the key it is recorded under. */
    private record LandManaTrait(SpellAbility ability, ProvenanceKey key) {}

    private final Card card;
    private final ICardFace face;
    private final ProvenanceRecorder recorder;
    private final int faceIndex;
    /** The tree-prefixed script path keys name, or null when not recording. */
    private final String scriptFile;
    private final List<Ability> abilities = new ArrayList<>();
    /** Each Class level's inner text, so its duplicate trait entry can be removed. */
    private final Set<String> classLevelDescriptions = new HashSet<>();
    /** The synthetic mana line of a basic-typed face, or null. */
    private final String landManaDescription;
    /**
     * Runtime mana abilities of a basic-typed face, skipped by the spell
     * phase and rendered once by the synthetic land-mana line. Kept with
     * their keys so that line can claim them: without the claim the land's
     * mana ability is a dropped key, and every activation record that names
     * it joins to nothing.
     */
    private final List<LandManaTrait> landManaTraits = new ArrayList<>();

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
        return buildCardFace(face, postProcessed());
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

    /** The abilities appended since the list had {@code before} of them. */
    private List<Ability> since(int before) {
        return List.copyOf(abilities.subList(before, abilities.size()));
    }

    // -- keywords --------------------------------------------------------

    /**
     * Route each keyword to the ability that renders it.
     *
     * <p>The keyword ordinal is asked of FaceOrder rather than counted
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
    private void parseKeywords() {
        for (KeywordInterface ki : card.getKeywords()) {
            Keyword kw = ki.getKeyword();
            int before = abilities.size();
            if (kw == Keyword.UNDEFINED) {
                routeUndefinedKeyword(ki);
            } else if (kw == Keyword.HAUNT) {
                abilities.addAll(HauntKeyword.of(ki, card));
            } else if (kw == Keyword.GIFT) {
                abilities.add(GiftKeyword.of(ki, card));
            } else if (kw == Keyword.COMPANION && ki instanceof Companion comp) {
                abilities.add(CompanionKeyword.of(ki, comp));
            } else {
                abilities.add(StandardKeyword.of(ki, kw));
            }
            int keywordIndex = FaceOrder.keywordIndex(card.getCurrentState(), ki);
            recorder.attributeKeyword(since(before),
                    keywordIndex < 0 ? null : key(TraitKind.KEYWORD, keywordIndex),
                    ki.getOriginal());
        }
    }

    // An undefined keyword is a script keyword Forge has no Keyword constant
    // for; its rendering depends only on how its original text starts.
    private static final Map<String, BiConsumer<KeywordInterface, FaceParser>>
            UNDEFINED_KEYWORDS = undefinedKeywordHandlers();

    private static Map<String, BiConsumer<KeywordInterface, FaceParser>> undefinedKeywordHandlers() {
        Map<String, BiConsumer<KeywordInterface, FaceParser>> handlers = new LinkedHashMap<>();
        BiConsumer<KeywordInterface, FaceParser> nameStatic = (ki, p) -> p.abilities.add(
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
        BiConsumer<KeywordInterface, FaceParser> etb =
                (ki, p) -> p.abilities.addAll(EtbReplacementAbility.fromKeyword(ki));
        handlers.put("etbCounter:", etb);
        handlers.put("ETBReplacement:", etb);
        handlers.put("AlternateAdditionalCost:", (ki, p) -> p.abilities.add(new TextAbility(
                AbilityType.ADDITIONAL_COST, AbilityDescription.applyCasing(
                        buildAlternateAdditionalCostDescription(ki.getOriginal())))));
        handlers.put("Visit:", (ki, p) -> p.abilities.addAll(visitAbilities(ki)));
        BiConsumer<KeywordInterface, FaceParser> openingEffect =
                (ki, p) -> addIfNotNull(p.abilities, openingEffectAbility(ki));
        handlers.put("MayEffectFromOpeningHand:", openingEffect);
        handlers.put("MayEffectFromOpeningDeck:", openingEffect);
        return handlers;
    }

    private void routeUndefinedKeyword(KeywordInterface ki) {
        String original = ki.getOriginal();
        for (Map.Entry<String, BiConsumer<KeywordInterface, FaceParser>> handler
                : UNDEFINED_KEYWORDS.entrySet()) {
            if (original.startsWith(handler.getKey())) {
                handler.getValue().accept(ki, this);
                return;
            }
        }
        abilities.add(StandardKeyword.of(ki, Keyword.UNDEFINED));
    }

    /**
     * An Attraction's visit lines.
     *
     * <p>For Charm-based visits, the TriggerDescription is the header and the
     * Charm options are OPTION sub-abilities. Any other visit takes its header
     * from {@link #visitHeader}, and has no children: its SpellDescription is
     * the complete oracle text and SpellEffect.fromChain would only duplicate
     * it.
     */
    private static List<Ability> visitAbilities(KeywordInterface ki) {
        List<Ability> visits = new ArrayList<>();
        for (Trigger t : ki.getTriggers()) {
            SpellAbility overriding = t.getOverridingAbility();
            boolean charm = overriding != null && overriding.getApi() == ApiType.Charm;
            String header = charm ? triggerHeader(t) : visitHeader(t, overriding);
            if (header == null) continue;
            List<Ability> children =
                    charm ? CharmAbility.optionsFrom(overriding, t) : List.of();
            visits.add(new TriggeredAbilityEntry(AbilityType.TRIGGERED, header, children));
        }
        return visits;
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

    // -- spell abilities -------------------------------------------------

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

    private void parseSpellAbilities() {
        boolean isAlternateFace = face.getType().hasSubtype("Adventure")
                || face.getType().hasSubtype("Omen");
        int spellIndex = -1;
        for (SpellAbility sa : card.getSpellAbilities()) {
            spellIndex++;
            // Declared before any route skips the trait, so a trait the parser
            // passes over lands in dropped_keys rather than failing the join:
            // it is still live at runtime and a record can still name it.
            ProvenanceKey spellKey = key(TraitKind.SPELL, spellIndex);
            recorder.declare(spellKey);
            SpellRoute route = classifySpell(sa, isAlternateFace);
            switch (route) {
                case KEYWORD_SPELL ->
                        recorder.attributeToKeyword(spellKey, sa, sa.getKeyword().getOriginal());
                case OTHER_FACE, DUPLICATE_TRAIT -> { }
                case LAND_MANA -> landManaTraits.add(new LandManaTrait(sa, spellKey));
                case CHARM, ALT_COST, ACTIVATED, SPELL, UNRENDERED -> {
                    int before = abilities.size();
                    renderSpell(route, sa, abilities);
                    recorder.attribute(since(before), spellKey, sa);
                }
            }
        }
    }

    private SpellRoute classifySpell(SpellAbility sa, boolean isAlternateFace) {
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
            return sa.isManaAbility() && landManaDescription != null
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
            ProvenanceKey key = key(TraitKind.TRIGGER, triggerIndex);
            recorder.declare(key);
            String exec = t.getParam("Execute");
            boolean secondary = "True".equalsIgnoreCase(t.getParam("Secondary"));
            if (secondary && exec != null && primaryExecuteSVars.contains(exec)) {
                recorder.attribute(primaryByExecute.get(exec), key, t);
                continue;
            }
            if (exec != null && !secondary) {
                primaryExecuteSVars.add(exec);
            }
            Ability triggered = TriggeredAbilityEntry.of(t);
            addIfNotNull(abilities, triggered);
            attributeTrait(triggered, key, t);
            if (triggered != null && exec != null) primaryByExecute.put(exec, triggered);
        }
    }

    /** Wrap each static or replacement trait directly in its entry. */
    private <T extends CardTraitBase> void parseTraits(Iterable<T> traits, TraitKind kind,
                                                       Function<T, Ability> factory) {
        int index = -1;
        for (T trait : traits) {
            index++;
            ProvenanceKey key = key(kind, index);
            recorder.declare(key);
            Ability entry = factory.apply(trait);
            addIfNotNull(abilities, entry);
            attributeTrait(entry, key, trait);
        }
    }

    /**
     * Record what a trait rendered: its own entry, or — for a keyword-derived
     * trait that renders none of its own — the keyword's line.
     */
    private void attributeTrait(Ability entry, ProvenanceKey key, CardTraitBase trait) {
        if (entry == null && trait.getKeyword() != null) {
            recorder.attributeToKeyword(key, trait, trait.getKeyword().getOriginal());
        } else {
            recorder.attribute(entry, key, trait);
        }
    }

    // -- land mana ---------------------------------------------------------

    /** The basic land types' mana line, claiming the mana abilities the spell phase skipped. */
    private void appendLandMana() {
        if (landManaDescription == null) return;
        TextAbility landMana = new TextAbility(AbilityType.ACTIVATED, landManaDescription);
        abilities.add(landMana);
        for (LandManaTrait trait : landManaTraits) {
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
        if (face.getType().toString().contains("Class")) {
            result = ClassLevels.arrange(result, classLevelDescriptions, recorder);
        }
        // Remove abilities whose description duplicates an earlier one.
        // This eliminates the spurious second trigger that Forge registers for
        // the "enters or attacks" pattern (two T: lines, identical TriggerDescription).
        result = deduplicateByDescription(result);
        result.sort(COSTS_FIRST);
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

        return new CardFace(name, manaCostStr, typeLine, pt, loyalty, defense, text, abilities);
    }

    private static String formatTypeLine(ICardFace face) {
        String typeStr = face.getType().toString();
        typeStr = typeStr.replace(" - ", " ");
        return typeStr.toLowerCase();
    }

    private static void addIfNotNull(List<Ability> list, Ability item) {
        if (item != null) list.add(item);
    }
}
