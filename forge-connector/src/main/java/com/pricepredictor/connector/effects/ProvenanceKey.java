package com.pricepredictor.connector.effects;

import forge.ImageKeys;
import forge.StaticData;
import forge.card.CardRules;
import forge.card.CardStateName;
import forge.card.ICardFace;
import forge.game.CardTraitBase;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.card.GamePieceType;
import forge.game.keyword.KeywordInterface;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.Trigger;
import forge.game.trigger.WrappedAbility;
import forge.item.IPaperCard;
import forge.item.PaperToken;
import forge.util.FileSection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Printed provenance of one runtime trait: which script file, which face, which
 * kind of trait, and which position within that kind.
 *
 * <p>This is the join every effect record uses to name the ability it is about.
 * Converted-line ordinals cannot serve: the converter runs one per-face bracket
 * counter across five mixed line kinds, emits keyword-derived lines first though
 * Forge appends them last in its runtime lists, and merges, deduplicates and
 * splits abilities relative to the runtime objects. An ordinal names different
 * things on the two sides; printed provenance is stable.
 *
 * <p>{@code indexWithinKind} indexes that kind's slice of the face's raw trait
 * list, not a global ordinal.
 */
public record ProvenanceKey(
        String scriptFile,
        int face,
        String traitKind,
        int indexWithinKind
) {

    public static final String KIND_KEYWORD = "keyword";
    public static final String KIND_SPELL = "spell";
    public static final String KIND_TRIGGER = "trigger";
    public static final String KIND_STATIC = "static";
    public static final String KIND_REPLACEMENT = "replacement";

    public ProvenanceKey {
        Objects.requireNonNull(scriptFile, "scriptFile");
        Objects.requireNonNull(traitKind, "traitKind");
    }

    /** The trait kind a runtime object belongs to, or null if it is none of them. */
    public static String kindOf(CardTraitBase trait) {
        if (trait instanceof Trigger) return KIND_TRIGGER;
        if (trait instanceof ReplacementEffect) return KIND_REPLACEMENT;
        if (trait instanceof StaticAbility) return KIND_STATIC;
        if (trait instanceof SpellAbility) return KIND_SPELL;
        return null;
    }

    /**
     * A key, or the reason there is none. Exactly one of the two is set.
     *
     * <p>The reason is the whole point of the type. An empty {@code ability}
     * array on a record cannot separate "no printed line exists, correctly" —
     * the Monarch, a dungeon, an engine-built effect card — from "the resolver
     * regressed", and a corpus where 3.1% of resolution records named no line
     * could not be asked which it was. The reason is written to the record's
     * {@code ability_unresolved}, which is collection metadata and never a
     * model input.
     */
    public record Resolved(ProvenanceKey key, String reason) {

        /** The acting-line list a record carries: the key, or nothing. */
        public List<ProvenanceKey> keys() {
            return key == null ? List.of() : List.of(key);
        }
    }

    /** The trait names no state, so there is nothing to index it against. */
    public static final String UNRESOLVED_NO_CARD_STATE = "no_card_state";
    /** The trait is none of the five kinds a printed line can be. */
    public static final String UNRESOLVED_UNKNOWN_KIND = "unknown_kind";
    /** The acting card is engine-built and has no script file in any tree. */
    public static final String UNRESOLVED_ENGINE_EFFECT = "engine_effect";
    /**
     * The trait came from a keyword the card does not print, and the permanent
     * that handed it out is no longer reachable.
     *
     * <p>Truthful rather than alarming: a keyword an anthem or a pump spell
     * granted names no printed line <em>on the recipient</em>, so there is
     * nothing to key even when nothing is broken. The static-granted case does
     * key — through the grantor — so what is left here is the spell-granted
     * one, where Forge keeps no back-reference to the ability that granted it.
     */
    public static final String UNRESOLVED_GRANTED_KEYWORD = "granted_keyword";

    /**
     * The trait was built at runtime from another card's script and lives on a
     * card other than the one whose text it came from.
     *
     * <p>An aura's or an equipment's granted trigger, an {@code Animate}'s
     * granted ability, and an effect card's own traits are all parsed from an
     * SVar of the donor. Forge points them at the donor's {@code CardState},
     * which is enough to name the donor's script file, but the SVar is not a
     * member of any of that state's trait slices and nothing records which
     * printed line named the SVar — so the card is known and the line is not.
     * Distinct from {@link #UNRESOLVED_UNINDEXABLE} because it is a property of
     * how Forge grants traits, not evidence of a resolver that regressed.
     */
    public static final String UNRESOLVED_GRANTED_TRAIT = "granted_trait";

    /**
     * A keyable kind, printed on the state it names, whose position in that
     * state could not be established.
     *
     * <p>The alarm. Every shape that legitimately has no printed line now has a
     * reason of its own, so this one means the resolver regressed.
     */
    public static final String UNRESOLVED_UNINDEXABLE = "unindexable";

    /** How many donor hops {@link #resolve} will take before giving up. */
    private static final int MAX_HOPS = 8;

    /**
     * The key for a live trait, following the donor chain for wrapped, granted
     * and copied traits.
     *
     * <p>Returns null when the trait cannot be attributed to a printed line at
     * all — a record naming no key is still trainable through its state and
     * payload, so this is not an error. Use {@link #resolve} when the caller
     * wants to know <em>why</em> there is no key.
     */
    public static ProvenanceKey of(CardTraitBase trait) {
        Resolved resolved = resolve(trait);
        return resolved == null ? null : resolved.key();
    }

    /**
     * The key for a live trait, or the reason it has none.
     *
     * <p>Forge rarely hands a collector the object that is actually a member of
     * a card state's trait list. The stack replaces every non-mana activated
     * ability with a fresh copy before pushing it ({@code MagicStack.add}), a
     * triggered ability arrives inside a {@code WrappedAbility} around an
     * {@code Execute$} SVar ability that belongs to no slice at all, and
     * alternative- and extra-cost variants are copies with no back-reference
     * whatsoever. Keying by object identity alone therefore fails for most of
     * what resolves, which is why this walks a donor chain and falls back to a
     * structural match rather than trusting {@code isIntrinsic()}.
     *
     * <p>The chain, in order:
     * <ol>
     *   <li><b>Unwrap</b> a {@code WrappedAbility}, then a sub-ability to its
     *       root: neither is ever a member of a slice.</li>
     *   <li><b>Trigger- and replacement-borne abilities</b> key to the trait
     *       that owns them. The {@code Execute$} SVar ability lives in
     *       {@code Trigger.overridingAbility}, not in the state's spell list,
     *       so the {@code Trigger} is the printed line, not the SVar.</li>
     *   <li><b>Granted abilities</b> resolve through the grantor static to the
     *       donor's printed line. This is tried <em>before</em> keying the
     *       trait where it stands, because a granted ability is appended to the
     *       recipient's own trait list and would otherwise be found there by
     *       identity — a confidently wrong key naming the recipient's script.</li>
     *   <li><b>The trait as it stands</b>, by identity first and then by
     *       structural fingerprint, which is what catches the stack's copies.</li>
     *   <li><b>Granted keywords</b> resolve through
     *       {@code KeywordInterface.getStatic()} to the static that handed the
     *       keyword out. A keyword granted by another permanent — Virulent
     *       Sliver's poisonous, Eldrazi Conscription's annihilator — builds its
     *       trigger on the <em>recipient</em>, where it names no printed line;
     *       {@code Card.getKeywordForStaticAbility} records the grantor, and
     *       that grantor's static is the printed line. Tried only when the
     *       keyword is not one the state prints, so an intrinsic keyword still
     *       keys to its own ordinal.</li>
     *   <li><b>Engine-spawned triggers</b> resolve through
     *       {@code Trigger.getSpawningAbility()}. A delayed trigger — the
     *       "sacrifice it at the beginning of the next end step" that
     *       {@code AtEOT$} builds, and the Immediate and Delayed trigger
     *       effects — is assembled in Java onto an LKI copy of its host and is
     *       a member of no slice at all, but it carries a copy of the ability
     *       that spawned it, and that copy fingerprints back to the line.</li>
     *   <li><b>Copied abilities</b> resolve through the original-ability
     *       back-reference, which the stack sets but the cost-variant helpers
     *       do not.</li>
     *   <li><b>Effect cards</b> ({@code Foo's Effect}) key to the ability that
     *       created them, which is where the printed line lives.</li>
     *   <li><b>Traits granted out of a donor's SVar</b> — an aura's
     *       {@code AddTrigger$}, an {@code Animate}'s {@code Triggers$} — key
     *       to the line that named the SVar, found by
     *       {@link #keyForGrantingLine}. Forge keeps no back-reference for
     *       these, so this is the one step that searches rather than follows,
     *       and it is last for that reason.</li>
     * </ol>
     *
     * <p>{@code getOriginalHost()} is deliberately not used to key a granted
     * trait: {@code CardTraitBase.getOriginalHost()} returns
     * {@code getCardState().getCard()}, which for a granted trait is the
     * <em>recipient</em>, not the donor, so keying by it would attribute an
     * anthem's effect to the creature that received it. That is the reason the
     * grantor step reads {@code getGrantorStatic()} instead.
     */
    public static Resolved resolve(CardTraitBase trait) {
        if (trait == null) {
            // Not a resolver failure: a hook whose signature drifted hands the
            // collector something that is not a trait at all, and nothing that
            // is not one of the five kinds can name a printed line.
            return new Resolved(null, UNRESOLVED_UNKNOWN_KIND);
        }
        Set<CardTraitBase> seen =
                Collections.newSetFromMap(new IdentityHashMap<>());
        CardTraitBase t = trait;
        String reason = UNRESOLVED_UNINDEXABLE;
        for (int hop = 0; t != null && hop < MAX_HOPS && seen.add(t); hop++) {
            // 1. Unwrap to something that could be a slice member.
            if (t instanceof WrappedAbility wrapped) {
                t = wrapped.getWrappedAbility();
                continue;
            }
            if (t instanceof SpellAbility sub && sub.getParent() != null) {
                t = sub.getRootAbility();
                continue;
            }
            if (t instanceof SpellAbility sa) {
                // 2. The trigger or replacement that owns this ability.
                Trigger owner = sa.getTrigger();
                ProvenanceKey borne = keyFor(owner);
                if (borne == null) borne = keyFor(sa.getReplacementEffect());
                if (borne != null) return new Resolved(borne, null);
                // A trigger the engine assembled at runtime keys to nothing
                // itself, but it knows what spawned it, so hop to the trigger
                // rather than past it and let step 6 follow that chain. This is
                // the reflexive "you may pay {X}. When you do..." shape, which
                // was the largest population reaching the end of this loop with
                // no key: ImmediateTriggerEffect deliberately nulls the execute
                // ability's parent, so there is no root to climb instead, and
                // the only way back to the printed line is through the trigger
                // that owns it.
                if (owner != null && owner.getSpawningAbility() != null) {
                    t = owner;
                    continue;
                }
                // 3. Granted: the donor's static, not the recipient's list.
                StaticAbility grantor = sa.getGrantorStatic();
                if (grantor != null) {
                    t = grantor;
                    continue;
                }
            }
            // 4. The trait where it stands.
            ProvenanceKey here = keyFor(t);
            if (here != null) return new Resolved(here, null);
            // 5. A granted keyword's printed line is the donor's static.
            KeywordInterface keyword = t.getKeyword();
            if (keyword != null && keyword.getStatic() != null
                    && keywordIndex(t.getCardState(), keyword) < 0) {
                t = keyword.getStatic();
                continue;
            }
            // 6. A trigger the engine spawned keys to the ability that spawned it.
            if (t instanceof Trigger trigger && trigger.getSpawningAbility() != null) {
                t = trigger.getSpawningAbility();
                continue;
            }
            // 7. The copy's original.
            if (t instanceof SpellAbility sa) {
                SpellAbility original = sa.getOriginalAbility();
                if (original != null && original != sa) {
                    t = original;
                    continue;
                }
            }
            // 8. An effect card's printed line belongs to whatever made it.
            CardState state = t.getCardState();
            Card host = state == null ? t.getHostCard() : state.getCard();
            if (host != null && host.getEffectSourceAbility() != null) {
                t = host.getEffectSourceAbility();
                continue;
            }
            // 9. A trait a donor line granted out of one of its own SVars.
            ProvenanceKey granting = keyForGrantingLine(t);
            if (granting != null) return new Resolved(granting, null);
            reason = reasonFor(t, state, host);
            break;
        }
        return new Resolved(null, reason);
    }

    /**
     * The printed line that granted a trait out of one of its own SVars.
     *
     * <p>An aura's or an equipment's {@code AddTrigger$}, an {@code Animate}'s
     * {@code Triggers$} and an {@code Effect}'s traits are all parsed from an
     * SVar of the donor, and Forge points the result at the donor's
     * {@code CardState} — so the card is known, and only the line is missing.
     * Nothing in the engine records which line named the SVar, but the state
     * still holds both halves, so the lookup Forge did on the way out can be
     * done again on the way back: find the SVar whose text parses to exactly
     * these parameters, then find the line that names that SVar.
     *
     * <p>Both halves refuse when more than one candidate matches, for the
     * reason {@link #indexLike} refuses: a record with no key is still
     * trainable through its state and payload, while a record with the wrong
     * key silently trains the wrong line. The parameter comparison is equality
     * between two runs of Forge's own {@code FileSection} parse, not a
     * resemblance test, so a match is the SVar and not something like it.
     *
     * <p>This is the last resort, tried only once every back-reference in the
     * chain has come up empty — which is also why walking four trait lists here
     * costs nothing measurable: in the smoke corpus one record in a thousand
     * ever reached this point.
     */
    private static ProvenanceKey keyForGrantingLine(CardTraitBase trait) {
        CardState state = trait.getCardState();
        if (state == null) return null;
        Map<String, String> want = trait.getOriginalMapParams();
        if (want == null || want.isEmpty()) return null;
        String svar = soleSVarParsingTo(state, want);
        if (svar == null) return null;
        CardTraitBase line = soleLineNaming(state, svar);
        return line == null ? null : keyFor(line);
    }

    /** The one SVar of {@code state} whose text parses to {@code want}, or null. */
    private static String soleSVarParsingTo(CardState state, Map<String, String> want) {
        String hit = null;
        int hits = 0;
        for (Map.Entry<String, String> svar : state.getSVars().entrySet()) {
            String text = svar.getValue();
            if (text == null || text.indexOf('$') < 0) continue;
            if (want.equals(FileSection.parseToMap(
                    text, FileSection.DOLLAR_SIGN_KV_SEPARATOR))) {
                if (hits == 0) hit = svar.getKey();
                hits++;
            }
        }
        return hits == 1 ? hit : null;
    }

    /** The one printed line of {@code state} that names {@code svar}, or null. */
    private static CardTraitBase soleLineNaming(CardState state, String svar) {
        CardTraitBase hit = null;
        int hits = 0;
        for (String kind : List.of(
                KIND_SPELL, KIND_TRIGGER, KIND_STATIC, KIND_REPLACEMENT)) {
            Iterable<? extends CardTraitBase> slice = slice(state, kind);
            if (slice == null) continue;
            for (CardTraitBase candidate : slice) {
                if (namesSVar(candidate, svar)) {
                    if (hits == 0) hit = candidate;
                    hits++;
                }
            }
        }
        return hits == 1 ? hit : null;
    }

    /**
     * Whether a line names an SVar, as a whole word.
     *
     * <p>{@code AddTrigger$ TrigA & TrigB} and {@code Triggers$ TrigA} both
     * name {@code TrigA}; a substring test would also match {@code TrigAttack}.
     */
    private static boolean namesSVar(CardTraitBase trait, String svar) {
        Map<String, String> params = trait.getOriginalMapParams();
        if (params == null) return false;
        for (String value : params.values()) {
            if (value == null) continue;
            for (String token : SVAR_LIST.split(value)) {
                if (token.equals(svar)) return true;
            }
        }
        return false;
    }

    /** How a line separates the SVar names it grants: spaces, commas, ampersands. */
    private static final java.util.regex.Pattern SVAR_LIST =
            java.util.regex.Pattern.compile("[\\s,&]+");

    /**
     * Why a trait that reached the end of the chain names no printed line.
     *
     * <p>The engine-built case is checked first and wins: The Monarch, The
     * Initiative and the dungeons are {@code new Card(...)} with their traits
     * built inline, no paper card and no rules, so no script file exists to
     * name. That is expected rather than a resolver failure, and the two have
     * to be distinguishable — an empty key alone cannot tell them apart, which
     * is exactly what hid this defect for a whole collection run.
     *
     * <p>The two granted cases exist for the same reason one layer down. A
     * corpus that reported 443 {@code unindexable} traits was reporting a
     * resolver bug that was not there: every one of them was a trait Forge
     * grants at runtime — a poisonous trigger on the Sliver that received it, a
     * Genju's trigger on an animated Plains, an {@code AtEOT$} sacrifice on a
     * token copy — and lumping them in with the alarm makes the alarm
     * unreadable, which is the same failure as an empty key with no reason at
     * all. What the two say is checkable: {@code isCopiedTrait()} is Forge's
     * own test for a trait living on a card other than the one whose script
     * text it came from.
     */
    private static String reasonFor(CardTraitBase trait, CardState state, Card host) {
        if (isScriptless(host)) {
            return UNRESOLVED_ENGINE_EFFECT;
        }
        if (state == null) {
            return UNRESOLVED_NO_CARD_STATE;
        }
        if (trait.getKeyword() != null) {
            // keyFor refuses a keyword the state does not print, and the
            // grantor hop has already been tried, so this is the granted one.
            return UNRESOLVED_GRANTED_KEYWORD;
        }
        if (kindOf(trait) == null) {
            return UNRESOLVED_UNKNOWN_KIND;
        }
        if (trait.isCopiedTrait()) {
            return UNRESOLVED_GRANTED_TRAIT;
        }
        return UNRESOLVED_UNINDEXABLE;
    }

    /**
     * Whether a card is an engine-built game piece with no script file anywhere.
     *
     * <p>An effect card's own traits <em>are</em> members of its own state, so
     * without this guard they would key by identity against a script path
     * derived from a name like {@code Foo's Effect} — a file that exists in no
     * tree, which the sidecar join is required to fail loudly on. Refusing here
     * is what lets the chain go on to the ability that created the effect,
     * which is where the printed line actually is, and what makes The Monarch
     * report {@code engine_effect} instead of a fabricated key.
     *
     * <p>Tokens are deliberately not included: a token <em>does</em> have a
     * script, in {@code tokenscripts}, and {@link #scriptFileOf} finds it.
     */
    private static boolean isScriptless(Card host) {
        if (host == null) return false;
        GamePieceType type = host.getGamePieceType();
        return type == GamePieceType.EFFECT || type == GamePieceType.DUNGEON;
    }

    /**
     * One trait, keyed where it stands; null when it is not a member of its own
     * state's slice for its kind.
     *
     * <p>A keyword-derived trait keys to the keyword rather than to the spell
     * slice it was appended to: {@code Card.updateSpellAbilities} appends those
     * in {@code KeywordCollection} order, which is hash order over an enum and
     * therefore differs between the convert JVM and the collect JVM, so a
     * {@code spell} index landing in that tail is not reproducible.
     */
    static ProvenanceKey keyFor(CardTraitBase trait) {
        if (trait == null) return null;
        CardState state = trait.getCardState();
        if (state == null) return null;
        Card host = state.getCard();
        if (host == null) return null;
        if (isScriptless(host)) return null;
        if (trait.getKeyword() != null) {
            int index = keywordIndex(state, trait.getKeyword());
            return index < 0 ? null : new ProvenanceKey(
                    scriptFileOf(host),
                    faceIndex(host, state.getStateName()),
                    KIND_KEYWORD,
                    index);
        }
        String kind = kindOf(trait);
        if (kind == null) return null;
        int index = indexWithin(state, trait, kind);
        if (index < 0) index = indexLike(state, trait, kind);
        return index < 0 ? null : new ProvenanceKey(
                scriptFileOf(host),
                faceIndex(host, state.getStateName()),
                kind,
                index);
    }

    /**
     * The position of the one same-shape member of the kind's slice, or -1.
     *
     * <p>The fingerprint is {@code getOriginalMapParams()}, the script's own
     * parameters as parsed. {@code SpellAbility.copy} preserves it because
     * {@code putParam} writes only the live {@code mapParams}, so a stack copy,
     * an alternative-cost copy and a copied spell all still fingerprint back to
     * the line they came from even though none of them carries a back-reference.
     *
     * <p>Refuses to answer when two members match: a record with no key is
     * still trainable through its state and payload, while a record with the
     * wrong key silently trains the wrong line.
     */
    /**
     * The position of the kind's only member of this trait's class, or -1.
     *
     * <p>The fallback for a trait with no fingerprint at all. A permanent spell
     * is built in Java rather than parsed from a line -- {@code SpellPermanent}
     * is constructed with an empty parameter map -- so a cost variant of one
     * carries nothing to recognise it by: an Adventure half, a kicker, a
     * {@code MayFlashCost} cast all copy the spell and set no back-reference,
     * and the copy's fingerprint is as empty as the original's.
     *
     * <p>What is left is arithmetic. A card state has exactly one permanent
     * spell, and a single candidate of the right class cannot be the wrong one.
     * Two candidates and this refuses, for the reason {@link #indexLike}
     * refuses: a record with no key is still trainable through its state and
     * payload, while a record with the wrong key silently trains the wrong line.
     */
    private static int indexIfUnique(CardState state, CardTraitBase trait, String kind) {
        Iterable<? extends CardTraitBase> slice = slice(state, kind);
        if (slice == null) return -1;
        int index = 0;
        int hit = -1;
        int hits = 0;
        for (CardTraitBase candidate : slice) {
            if (candidate.getClass() == trait.getClass()) {
                hit = index;
                hits++;
            }
            index++;
        }
        return hits == 1 ? hit : -1;
    }

    private static int indexLike(CardState state, CardTraitBase trait, String kind) {
        Iterable<? extends CardTraitBase> slice = slice(state, kind);
        if (slice == null) return -1;
        Map<String, String> want = trait.getOriginalMapParams();
        if (want == null || want.isEmpty()) return indexIfUnique(state, trait, kind);
        int index = 0;
        int hit = -1;
        int hits = 0;
        for (CardTraitBase candidate : slice) {
            if (candidate.getClass() == trait.getClass()
                    && want.equals(candidate.getOriginalMapParams())) {
                if (hit < 0) hit = index;
                hits++;
            }
            index++;
        }
        return hits == 1 ? hit : -1;
    }

    /**
     * The converted script path a live card's traits belong to.
     *
     * <p>Forge is asked first and the name is only a fallback. The strongest
     * answer is {@code CardRules.getPath()}, the file the card reader actually
     * opened: the converter keys each sidecar by that same path relativised
     * against the same root, so taking it here is not a rule that agrees with
     * the tree but the tree's own record, and the two cannot drift. It is also
     * the only answer that is right for the 86 cards the tree does not file
     * under their initial — {@code +2 Mace} is {@code cardsfolder/p/+2_mace.txt}
     * and no derivation reaches that.
     *
     * <p>Failing that, {@code CardRules.getNormalizedName()} is the stem of the
     * file Forge read the card from, and that still beats any sanitizer:
     * Forge's filenames disagree with its card names often enough to matter — a
     * Fallaji Archaeologist lives in {@code fallaji_archeologist.txt} — and a
     * misspelling cannot be derived. It also settles the two-faced cards for
     * free, where {@code getName()} answers whichever face is up while the
     * script is {@code westvale_abbey_ormendahl_profane_prince.txt} either way.
     *
     * <p>A token is filed under what it <em>is</em>, not what it is called:
     * {@code c_1_1_eldrazi_scion_sac.txt} is the Eldrazi Scion. And
     * {@code isToken()} does not mean "came from a token script" — a token copy
     * of a real permanent is a token by that flag while its abilities are the
     * printed card's — so the paper card decides the tree: only a
     * {@link PaperToken} was read from a token script.
     */
    static String scriptFileOf(Card host) {
        if (VariantRegistry.isVariant(host.getName())) {
            return CardFilenames.scriptFile(
                    SourceTree.VARIANT_SCRIPTS, host.getName());
        }
        String stem = tokenScriptStem(host);
        if (stem != null) {
            return CardFilenames.scriptFileForStem(SourceTree.TOKENSCRIPTS, stem);
        }
        CardRules rules = host.getRules();
        if (rules != null) {
            String read = SourceTree.relativePathIn(
                    SourceTree.CARDSFOLDER, rules.getPath());
            if (read != null) {
                return SourceTree.CARDSFOLDER + "/" + read;
            }
            String normalized = rules.getNormalizedName();
            if (normalized != null && !normalized.isEmpty()) {
                return CardFilenames.scriptFileForStem(
                        SourceTree.CARDSFOLDER, normalized);
            }
        }
        return CardFilenames.scriptFile(SourceTree.CARDSFOLDER, host.getName());
    }

    /**
     * A token's script-file stem, or null when the card did not come from a
     * token script — including a token copy of a printed card.
     *
     * <p>Two sources, tried in order. A {@link PaperToken} is the token as
     * Forge read it from {@code tokenscripts}, and its image filename starts
     * with the script stem. A token that was rebuilt by {@code TokenInfo} —
     * every token in a {@code GameCopier} fork — has no paper card at all, but
     * {@code TokenInfo.toCard} copies the image key, and a token image key is
     * {@code t:<stem>[|edition|…]}. The stem read either way is checked
     * against the token database, so a key that names no script (a token copy
     * of a printed card carries the printed card's image key, which is not a
     * token key) yields null rather than a fabricated path.
     */
    private static String tokenScriptStem(Card host) {
        try {
            IPaperCard paper = host.getPaperCard();
            String image = null;
            if (paper instanceof PaperToken token) {
                image = token.getImageFilename(1);
            } else if (host.isToken()) {
                image = ImageKeys.getTokenImageName(host.getImageKey());
            }
            if (image == null || image.isEmpty()) {
                return null;
            }
            int bar = image.indexOf('|');
            String stem = (bar < 0 ? image : image.substring(0, bar)).replace(' ', '_');
            if (stem.isEmpty() || !StaticData.instance().getAllTokens().containsRule(stem)) {
                return null;
            }
            return stem;
        } catch (RuntimeException e) {
            return null;
        }
    }

    /**
     * The state's trait list for one kind, in declaration order.
     *
     * <p>Printed traits occupy the head of every slice — a state builds each
     * list from its own parsed traits and only then appends the keyword-derived
     * and runtime-granted ones — which is why an index into a slice is stable
     * across processes for a scripted line.
     */
    private static Iterable<? extends CardTraitBase> slice(CardState state, String kind) {
        return switch (kind) {
            case KIND_TRIGGER -> state.getTriggers();
            case KIND_REPLACEMENT -> state.getReplacementEffects();
            case KIND_STATIC -> state.getStaticAbilities();
            case KIND_SPELL -> state.getSpellAbilities();
            default -> null;
        };
    }

    /** Index of {@code trait} within its kind's slice of {@code state}'s trait list. */
    private static int indexWithin(CardState state, CardTraitBase trait, String kind) {
        Iterable<? extends CardTraitBase> slice = slice(state, kind);
        if (slice == null) return -1;
        int index = 0;
        for (CardTraitBase candidate : slice) {
            if (candidate == trait) return index;
            index++;
        }
        return -1;
    }

    /**
     * The face ordinal the converter emitted a state under.
     *
     * <p>Shared by the converter and the collectors so the two cannot drift:
     * the converter renders the main part first (LeftSplit for a split card,
     * Original otherwise), then the specialize parts in map order, or the single
     * other part.
     */
    public static int faceIndex(Card card, CardStateName state) {
        if (state == null) return 0;
        List<CardStateName> order = faceOrder(card);
        int index = order.indexOf(state);
        return index < 0 ? 0 : index;
    }

    /** The state names in the order the converter emits faces. */
    public static List<CardStateName> faceOrder(Card card) {
        if (card == null) return List.of(CardStateName.Original);
        if (card.getStates().contains(CardStateName.LeftSplit)) {
            return List.of(CardStateName.LeftSplit, CardStateName.RightSplit);
        }
        java.util.List<CardStateName> order = new java.util.ArrayList<>();
        order.add(CardStateName.Original);
        for (CardStateName name : card.getStates()) {
            if (name != CardStateName.Original && !order.contains(name)) {
                order.add(name);
            }
        }
        return List.copyOf(order);
    }

    /**
     * The printed-keyword ordinal a keyword-derived trait keys to.
     *
     * <p>Shared by the converter and the collectors so the two cannot drift,
     * the same discipline {@link #faceIndex} already enforces for faces.
     *
     * <p>Not the position in {@code getKeywords()}:
     * {@code forge.game.keyword.KeywordCollection} is a
     * {@code MultimapBuilder.hashKeys()} over the {@code Keyword} enum, whose
     * {@code hashCode()} is the identity hash, so that order differs between
     * the convert JVM and the collect JVM and an ordinal taken from it names a
     * different keyword on each side. Sorting the state's own keywords by their
     * printed text is stable in both processes.
     *
     * <p>Restricted to intrinsics, so a <em>granted</em> keyword answers -1 and
     * keys to nothing. That is correct: a keyword an anthem handed out names no
     * printed line on the card that received it, and the alternative — the
     * position it happens to occupy in the recipient's list — is a wrong key
     * dressed as an answer.
     */
    public static int keywordIndex(CardState state, KeywordInterface keyword) {
        if (state == null || keyword == null) return -1;
        String original = keyword.getOriginal();
        if (original == null || original.isEmpty()) return -1;
        List<String> originals = new ArrayList<>();
        for (KeywordInterface intrinsic : state.getIntrinsicKeywords()) {
            if (intrinsic.getOriginal() != null) originals.add(intrinsic.getOriginal());
        }
        Collections.sort(originals);
        return originals.indexOf(original);
    }

    /** The JSON object form used inside a record's {@code ability} array. */
    public String toJson() {
        return "{\"script_file\":" + Json.string(scriptFile)
                + ",\"face\":" + face
                + ",\"trait_kind\":" + Json.string(traitKind)
                + ",\"index_within_kind\":" + indexWithinKind + "}";
    }

    /** The JSON object form used inside a sidecar, where the file is in the header. */
    public String toSidecarJson() {
        return "{\"face\":" + face
                + ",\"trait_kind\":" + Json.string(traitKind)
                + ",\"index_within_kind\":" + indexWithinKind + "}";
    }
}
