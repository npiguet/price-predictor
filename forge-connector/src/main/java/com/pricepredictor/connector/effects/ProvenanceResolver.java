package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.ProvenanceKey.Resolved;
import forge.card.GamePieceType;
import forge.game.CardTraitBase;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.Trigger;
import forge.game.trigger.WrappedAbility;
import forge.util.FileSection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;

/**
 * The printed line a live Forge trait came from, or the reason it has none.
 */
public final class ProvenanceResolver {

    private ProvenanceResolver() {
    }

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
     *
     * <p>A clause resolving inside a chosen mode of a charm keys to the
     * charm's line <em>plus</em> that mode's {@code Choices$} position
     * (FR-029a): the chain climbs to the root as for any sub-ability, and the
     * option is read off the clause before the climb, because the root knows
     * nothing of which clone is resolving. A mode {@link CharmModes} cannot
     * place keeps the root key.
     */
    public static Resolved resolve(CardTraitBase trait) {
        Resolved root = resolveRoot(trait);
        return root.key() == null ? root : root.withOption(CharmModes.optionOf(trait));
    }

    private static Resolved resolveRoot(CardTraitBase trait) {
        if (trait == null) {
            // Not a resolver failure: a hook whose signature drifted hands the
            // collector something that is not a trait at all, and nothing that
            // is not one of the five kinds can name a printed line.
            return Resolved.UNKNOWN;
        }
        Set<CardTraitBase> seen =
                Collections.newSetFromMap(new IdentityHashMap<>());
        CardTraitBase t = trait;
        UnresolvedReason reason = UnresolvedReason.UNINDEXABLE;
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
                    && FaceOrder.keywordIndex(t.getCardState(), keyword) < 0) {
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
        Map.Entry<String, String> hit = sole(state.getSVars().entrySet(), svar -> {
            String text = svar.getValue();
            return text != null && text.indexOf('$') >= 0 && want.equals(
                    FileSection.parseToMap(text, FileSection.DOLLAR_SIGN_KV_SEPARATOR));
        });
        return hit == null ? null : hit.getKey();
    }

    /** The one printed line of {@code state} that names {@code svar}, or null. */
    private static CardTraitBase soleLineNaming(CardState state, String svar) {
        List<CardTraitBase> lines = new ArrayList<>();
        for (TraitKind kind : TraitKind.values()) {
            kind.slice(state).forEach(lines::add);
        }
        return sole(lines, line -> namesSVar(line, svar));
    }

    /**
     * The position of the only item that matches, or -1 when none or several
     * do.
     *
     * <p>Every fallback that recognises a line by resemblance refuses on two
     * candidates: a record with no key is still trainable through its state
     * and payload, while a record with the wrong key silently trains the
     * wrong line.
     */
    private static <T> int soleIndex(Iterable<? extends T> items, Predicate<? super T> matches) {
        int index = 0;
        int hit = -1;
        for (T item : items) {
            if (matches.test(item)) {
                if (hit >= 0) {
                    return -1;
                }
                hit = index;
            }
            index++;
        }
        return hit;
    }

    /** The only item that matches, or null when none or several do; see {@link #soleIndex}. */
    private static <T> T sole(Iterable<? extends T> items, Predicate<? super T> matches) {
        T hit = null;
        boolean found = false;
        for (T item : items) {
            if (matches.test(item)) {
                if (found) {
                    return null;
                }
                hit = item;
                found = true;
            }
        }
        return hit;
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
    private static UnresolvedReason reasonFor(CardTraitBase trait, CardState state, Card host) {
        if (isScriptless(host)) {
            return UnresolvedReason.ENGINE_EFFECT;
        }
        if (state == null) {
            return UnresolvedReason.NO_CARD_STATE;
        }
        if (trait.getKeyword() != null) {
            // keyFor refuses a keyword the state does not print, and the
            // grantor hop has already been tried, so this is the granted one.
            return UnresolvedReason.GRANTED_KEYWORD;
        }
        if (TraitKind.of(trait) == null) {
            return UnresolvedReason.UNKNOWN_KIND;
        }
        if (trait.isCopiedTrait()) {
            return UnresolvedReason.GRANTED_TRAIT;
        }
        return UnresolvedReason.UNINDEXABLE;
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
     * script, in {@code tokenscripts}, and {@link ScriptLocator#scriptFileOf} finds it.
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
            int index = FaceOrder.keywordIndex(state, trait.getKeyword());
            return index < 0 ? null : new ProvenanceKey(
                    ScriptLocator.scriptFileOf(host),
                    FaceOrder.faceIndex(host, state.getStateName()),
                    TraitKind.KEYWORD,
                    index);
        }
        TraitKind kind = TraitKind.of(trait);
        if (kind == null) return null;
        int index = indexWithin(state, trait, kind);
        if (index < 0) index = indexLike(state, trait, kind);
        return index < 0 ? null : new ProvenanceKey(
                ScriptLocator.scriptFileOf(host),
                FaceOrder.faceIndex(host, state.getStateName()),
                kind,
                index);
    }

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
    private static int indexIfUnique(CardState state, CardTraitBase trait, TraitKind kind) {
        return soleIndex(kind.slice(state),
                candidate -> candidate.getClass() == trait.getClass());
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
    private static int indexLike(CardState state, CardTraitBase trait, TraitKind kind) {
        Iterable<? extends CardTraitBase> slice = kind.slice(state);
        Map<String, String> want = trait.getOriginalMapParams();
        if (want == null || want.isEmpty()) return indexIfUnique(state, trait, kind);
        return soleIndex(slice, candidate -> candidate.getClass() == trait.getClass()
                && want.equals(candidate.getOriginalMapParams()));
    }


    /** Index of {@code trait} within its kind's slice of {@code state}'s trait list. */
    private static int indexWithin(CardState state, CardTraitBase trait, TraitKind kind) {
        int index = 0;
        for (CardTraitBase candidate : kind.slice(state)) {
            if (candidate == trait) return index;
            index++;
        }
        return -1;
    }

    /**
     * The printed line a runtime trait came from, as a list of at most one.
     *
     * <p>For the payload fields that carry keys but no reason — a playability
     * candidate, a rewrite's {@code replaced_by}. A record's own acting line
     * goes through {@link #resolve} instead, keeping the reason there is no
     * key: an empty {@code ability} with nothing beside it cannot separate
     * "this card has no printed line anywhere" from "the resolver regressed".
     */
    static List<ProvenanceKey> keysOf(CardTraitBase trait) {
        return resolve(trait).keys();
    }
}
