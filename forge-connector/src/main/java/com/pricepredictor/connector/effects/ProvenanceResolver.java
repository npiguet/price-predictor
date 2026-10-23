package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.ProvenanceKey.Resolved;
import forge.card.GamePieceType;
import forge.game.CardTraitBase;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.trigger.WrappedAbility;
import forge.util.FileSection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.regex.Pattern;

/**
 * The printed line a live Forge trait came from, or the reason it has none.
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
 */
public final class ProvenanceResolver {

    private ProvenanceResolver() {
    }

    /** How many donor hops {@link #resolve} will take before giving up. */
    private static final int MAX_HOPS = 8;

    /** How a line separates the SVar names it grants: spaces, commas, ampersands. */
    private static final Pattern SVAR_LIST = Pattern.compile("[\\s,&]+");

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

    /**
     * The key for a live trait, or the reason it has none, found by walking
     * the {@link #DONOR_CHAIN}.
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
        for (int hop = 0; t != null && hop < MAX_HOPS && seen.add(t); hop++) {
            StepResult result = firstApplying(t);
            if (result instanceof Keyed keyed) return new Resolved(keyed.key(), null);
            if (!(result instanceof Hop next)) return new Resolved(null, reasonFor(t));
            t = next.trait();
        }
        return new Resolved(null, UnresolvedReason.UNINDEXABLE);
    }

    /**
     * What one step of the {@link #DONOR_CHAIN} makes of a trait: the key it
     * names, another trait to start the chain again from, or nothing, which
     * hands the trait to the next step.
     */
    private sealed interface StepResult permits Keyed, Hop, Pass {
    }

    /** The trait keys to this printed line, which ends the chain. */
    private record Keyed(ProvenanceKey key) implements StepResult {
    }

    /**
     * The trait stands for this one, which starts the chain again from its
     * first step; a hop to null ends the chain unresolved.
     */
    private record Hop(CardTraitBase trait) implements StepResult {
    }

    /** The step does not apply to the trait. */
    private enum Pass implements StepResult {
        PASS
    }

    /**
     * The donor chain, in the order its steps are tried; each step's Javadoc
     * says what it follows and why it sits where it does.
     *
     * <p>A step either keys the trait, which ends the chain, or hops to
     * another trait, which starts the chain again from the first step; a trait
     * no step applies to ends it with {@link #reasonFor}.
     */
    private static final List<Function<CardTraitBase, StepResult>> DONOR_CHAIN = List.of(
            ProvenanceResolver::unwrapped,
            ProvenanceResolver::owner,
            ProvenanceResolver::grantor,
            ProvenanceResolver::whereItStands,
            ProvenanceResolver::keywordGrantor,
            ProvenanceResolver::spawner,
            ProvenanceResolver::original,
            ProvenanceResolver::effectSource,
            ProvenanceResolver::grantingLine);

    /** The first step of the {@link #DONOR_CHAIN} that applies to {@code trait}. */
    private static StepResult firstApplying(CardTraitBase trait) {
        for (Function<CardTraitBase, StepResult> step : DONOR_CHAIN) {
            StepResult result = step.apply(trait);
            if (result != Pass.PASS) return result;
        }
        return Pass.PASS;
    }

    /** A step that keys the trait to {@code key}, or passes when it is null. */
    private static StepResult keyed(ProvenanceKey key) {
        return key == null ? Pass.PASS : new Keyed(key);
    }

    /** A step that hops to {@code next}, or passes when it is null. */
    private static StepResult hop(CardTraitBase next) {
        return next == null ? Pass.PASS : new Hop(next);
    }

    /**
     * The trait a wrapper or a sub-ability stands for, neither of which is
     * ever a member of a slice: a {@code WrappedAbility} to the ability it
     * wraps, a sub-ability to its root.
     *
     * <p>Unlike the other hops this hops even to null, because a wrapper
     * around nothing unwraps to nothing, and that ends the chain.
     */
    private static StepResult unwrapped(CardTraitBase trait) {
        if (trait instanceof WrappedAbility wrapped) {
            return new Hop(wrapped.getWrappedAbility());
        }
        if (trait instanceof SpellAbility sub && sub.getParent() != null) {
            return new Hop(sub.getRootAbility());
        }
        return Pass.PASS;
    }

    /**
     * The trigger or replacement that owns an ability: its key, or, when it
     * keys to nothing, a hop to the owning trigger if the engine assembled it
     * at runtime.
     *
     * <p>The {@code Execute$} SVar ability lives in
     * {@code Trigger.overridingAbility}, not in the state's spell list, so the
     * {@code Trigger} is the printed line, not the SVar.
     *
     * <p>A trigger the engine assembled keys to nothing itself, but it knows
     * what spawned it, so the chain hops to the trigger rather than past it
     * and lets {@link #spawner} follow that. This is the reflexive "you may
     * pay {X}. When you do..." shape: {@code ImmediateTriggerEffect}
     * deliberately nulls the execute ability's parent, so there is no root to
     * climb instead, and the only way back to the printed line is through the
     * trigger that owns it.
     */
    private static StepResult owner(CardTraitBase trait) {
        if (!(trait instanceof SpellAbility sa)) return Pass.PASS;
        ProvenanceKey borne = keyFor(sa.getTrigger());
        if (borne == null) borne = keyFor(sa.getReplacementEffect());
        if (borne != null) return new Keyed(borne);
        Trigger owner = sa.getTrigger();
        return owner != null && owner.getSpawningAbility() != null ? new Hop(owner) : Pass.PASS;
    }

    /**
     * The static that granted an ability.
     *
     * <p>Tried <em>before</em> keying the trait where it stands, because a
     * granted ability is appended to the recipient's own trait list and would
     * otherwise be found there by identity — a confidently wrong key naming
     * the recipient's script.
     *
     * <p>{@code getOriginalHost()} is deliberately not used to key a granted
     * trait: {@code CardTraitBase.getOriginalHost()} returns
     * {@code getCardState().getCard()}, which for a granted trait is the
     * <em>recipient</em>, not the donor, so keying by it would attribute an
     * anthem's effect to the creature that received it. That is the reason
     * this step reads {@code getGrantorStatic()} instead.
     */
    private static StepResult grantor(CardTraitBase trait) {
        return trait instanceof SpellAbility sa ? hop(sa.getGrantorStatic()) : Pass.PASS;
    }

    /** The trait keyed where it stands; see {@link #keyFor}. */
    private static StepResult whereItStands(CardTraitBase trait) {
        return keyed(keyFor(trait));
    }

    /**
     * One trait, keyed where it stands; null when it is not a member of its
     * own state's slice for its kind.
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
            return keyAt(state, TraitKind.KEYWORD,
                    FaceOrder.keywordIndex(state, trait.getKeyword()));
        }
        TraitKind kind = TraitKind.of(trait);
        if (kind == null) return null;
        int index = indexWithin(state, trait, kind);
        return keyAt(state, kind, index < 0 ? indexLike(state, trait, kind) : index);
    }

    /**
     * The key of the {@code index}-th {@code kind} trait of {@code state}, or
     * null for a negative index, which is how every index search here says it
     * found nothing.
     */
    private static ProvenanceKey keyAt(CardState state, TraitKind kind, int index) {
        if (index < 0) return null;
        Card host = state.getCard();
        return new ProvenanceKey(
                ScriptLocator.scriptFileOf(host),
                FaceOrder.faceIndex(host, state.getStateName()),
                kind,
                index);
    }

    /**
     * The static that handed out a granted keyword.
     *
     * <p>A keyword granted by another permanent — Virulent Sliver's poisonous,
     * Eldrazi Conscription's annihilator — builds its trigger on the
     * <em>recipient</em>, where it names no printed line;
     * {@code Card.getKeywordForStaticAbility} records the grantor, and that
     * grantor's static is the printed line. Tried only when the keyword is not
     * one the state prints, so an intrinsic keyword still keys to its own
     * ordinal.
     */
    private static StepResult keywordGrantor(CardTraitBase trait) {
        KeywordInterface keyword = trait.getKeyword();
        if (keyword != null && keyword.getStatic() != null
                && FaceOrder.keywordIndex(trait.getCardState(), keyword) < 0) {
            return new Hop(keyword.getStatic());
        }
        return Pass.PASS;
    }

    /**
     * The ability that spawned an engine-built trigger.
     *
     * <p>A delayed trigger — the "sacrifice it at the beginning of the next end
     * step" that {@code AtEOT$} builds, and the Immediate and Delayed trigger
     * effects — is assembled in Java onto an LKI copy of its host and is a
     * member of no slice at all, but it carries a copy of the ability that
     * spawned it, and that copy fingerprints back to the line.
     */
    private static StepResult spawner(CardTraitBase trait) {
        return trait instanceof Trigger trigger ? hop(trigger.getSpawningAbility()) : Pass.PASS;
    }

    /**
     * The ability a copy was made from.
     *
     * <p>The stack sets this back-reference, but the cost-variant helpers do
     * not.
     */
    private static StepResult original(CardTraitBase trait) {
        if (!(trait instanceof SpellAbility sa)) return Pass.PASS;
        SpellAbility original = sa.getOriginalAbility();
        return original != sa ? hop(original) : Pass.PASS;
    }

    /**
     * The ability that created the effect card ({@code Foo's Effect}) a trait
     * sits on; that ability is where the printed line lives.
     */
    private static StepResult effectSource(CardTraitBase trait) {
        Card host = hostOf(trait);
        return host == null ? Pass.PASS : hop(host.getEffectSourceAbility());
    }

    /** The card a trait sits on: its state's card, or its host when it has no state. */
    private static Card hostOf(CardTraitBase trait) {
        CardState state = trait.getCardState();
        return state == null ? trait.getHostCard() : state.getCard();
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
     * <p>Forge keeps no back-reference for these, so this is the one step
     * that searches rather than follows, and it is last for that reason. It
     * runs only once every back-reference in the chain has come up empty,
     * which is also why walking four trait lists here is cheap: few records
     * ever reach this point.
     */
    private static StepResult grantingLine(CardTraitBase trait) {
        CardState state = trait.getCardState();
        if (state == null) return Pass.PASS;
        Map<String, String> want = trait.getOriginalMapParams();
        if (want == null || want.isEmpty()) return Pass.PASS;
        String svar = soleSVarParsingTo(state, want);
        if (svar == null) return Pass.PASS;
        CardTraitBase line = soleLineNaming(state, svar);
        return line == null ? Pass.PASS : keyed(keyFor(line));
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

    /**
     * Why a trait that reached the end of the chain names no printed line.
     *
     * <p>The engine-built case is checked first and wins: The Monarch, The
     * Initiative and the dungeons are {@code new Card(...)} with their traits
     * built inline, no paper card and no rules, so no script file exists to
     * name. That is expected rather than a resolver failure, and the two have
     * to be distinguishable — an empty key alone cannot tell them apart.
     *
     * <p>The two granted cases exist for the same reason one layer down. A
     * trait Forge grants at runtime — a poisonous trigger on the Sliver that
     * received it, a Genju's trigger on an animated Plains, an {@code AtEOT$}
     * sacrifice on a token copy — is not a resolver bug, and lumping it in with
     * {@code unindexable} makes that alarm unreadable, which is the same
     * failure as an empty key with no reason at all. What the two say is
     * checkable: {@code isCopiedTrait()} is Forge's
     * own test for a trait living on a card other than the one whose script
     * text it came from.
     */
    private static UnresolvedReason reasonFor(CardTraitBase trait) {
        if (isScriptless(hostOf(trait))) {
            return UnresolvedReason.ENGINE_EFFECT;
        }
        if (trait.getCardState() == null) {
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
     * script, in {@code tokenscripts}, and {@link ScriptLocator#scriptFileOf}
     * finds it.
     */
    private static boolean isScriptless(Card host) {
        if (host == null) return false;
        GamePieceType type = host.getGamePieceType();
        return type == GamePieceType.EFFECT || type == GamePieceType.DUNGEON;
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
}
