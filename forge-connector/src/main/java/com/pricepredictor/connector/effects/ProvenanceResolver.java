package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.ProvenanceKey.Resolved;
import forge.card.CardStateName;
import forge.card.GamePieceType;
import forge.game.CardTraitBase;
import forge.game.card.Card;
import forge.game.card.CardCloneStates;
import forge.game.card.CardState;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.Trigger;
import forge.game.trigger.WrappedAbility;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

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
    static final int MAX_HOPS = 8;

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
     * (024 FR-029a): the chain climbs to the root as for any sub-ability, and the
     * option is read off the clause before the climb, because the root knows
     * nothing of which clone is resolving. A mode {@link CharmModes} cannot
     * place keeps the root key.
     *
     * <p>A charm cast through an alternative-cost keyword — Flashback — keys
     * its root to the keyword's line, which has no modes: the sidecar writes
     * a mode's line only under the charm's own spell line (FR-039). So a mode
     * of such a cast keys to the printed spell its cast copies, plus the
     * option, when one printed spell is that charm. A saga chapter's mode,
     * whose keyword line does carry modes, is a trigger's and stays keyed to
     * its keyword.
     */
    public static Resolved resolve(CardTraitBase trait) {
        Resolved root = resolveRoot(trait);
        Integer option = CharmModes.optionOf(trait);
        if (root.key() == null || option == null) {
            return root;
        }
        if (root.key().traitKind() == TraitKind.KEYWORD) {
            ProvenanceKey printed = printedSpellOf(trait);
            if (printed != null) {
                return new Resolved(printed.withOption(option), null);
            }
        }
        return root.withOption(option);
    }

    /**
     * The printed spell line a keyword-cast spell copies, by its script
     * parameters ({@link PrintedSlice#indexLike}), or null when the trait is
     * not inside a spell or no single printed spell matches.
     */
    private static ProvenanceKey printedSpellOf(CardTraitBase trait) {
        if (!(trait instanceof SpellAbility clause)) return null;
        SpellAbility root = clause.getRootAbility();
        if (!isKeywordCastSpell(root)) return null;
        CardState state = root.getCardState();
        if (!onScriptedCard(state)) return null;
        return keyAt(state, TraitKind.SPELL, PrintedSlice.of(state, TraitKind.SPELL).indexLike(root));
    }

    /** Whether {@code root} is a spell cast through a keyword (Flashback); false for null. */
    private static boolean isKeywordCastSpell(SpellAbility root) {
        return root != null && root.isSpell() && root.getKeyword() != null;
    }

    /** Whether {@code state} is a state of a card that has a script file; false for null. */
    private static boolean onScriptedCard(CardState state) {
        return state != null && state.getCard() != null && !isScriptless(state.getCard());
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
     * in {@code KeywordCollection} order, which is the script's insertion order
     * only under the {@code -android} Guava Forge pins (an identity-hash order,
     * different per JVM, under any other flavour), so a {@code spell} index
     * landing in that tail is not guaranteed reproducible. The
     * keyword's key carries {@code built}, which of the instance's traits it
     * is ({@link #builtOf}), so a saga's chapter III names its own line and
     * not the saga's first.
     *
     * <p>A trait the card granted itself keys to the line that granted it
     * ({@link #keyInState}).
     */
    static ProvenanceKey keyFor(CardTraitBase trait) {
        return keyFor(trait, 0);
    }

    /**
     * {@link #keyFor}, {@code depth} granting lines into a search for one
     * ({@link GrantingLineSearch#keyOf}): a granted line that names the SVar of
     * another is keyed by its own granting line in turn, as deep as
     * {@link #MAX_HOPS}, and two lines granting each other end there.
     */
    static ProvenanceKey keyFor(CardTraitBase trait, int depth) {
        if (trait == null) return null;
        CardState state = trait.getCardState();
        if (state == null) return null;
        Card host = state.getCard();
        if (host == null) return null;
        if (isScriptless(host)) return null;
        if (trait.getKeyword() != null) {
            return keyOfPrinted(state, trait.getKeyword(), builtOf(trait), 0);
        }
        CardCloneStates cloned = copyStatesOf(state);
        if (cloned != null) return lineOfCopyEffect(cloned, state, trait);
        TraitKind kind = TraitKind.of(trait);
        if (kind == null) return null;
        return keyInState(state, trait, kind, depth);
    }

    /**
     * A trait a copy's state holds itself, not one copied from what it copies
     * (a copied trait keeps the original's state): one the copy effect added
     * (AddTriggers$, AddAbilities$), among the state's own traits, keys to
     * the line that made the copy. One Forge built fresh on the copy's state
     * (a land copy's mana ability, a copied planeswalker's loyalty
     * replacement) is in no list of the state's own and keys to nothing.
     */
    private static ProvenanceKey lineOfCopyEffect(CardCloneStates cloned,
                                                  CardState state, CardTraitBase trait) {
        CardTraitBase source = cloned.getSource();
        if (source == null || source == trait) return null;
        return ownTraitOf(state, trait) ? of(source) : null;
    }

    /** Whether a trait is in its state's own list, through no trait change. */
    private static boolean ownTraitOf(CardState state, CardTraitBase trait) {
        TraitKind kind = TraitKind.of(trait);
        List<? extends CardTraitBase> own = kind == null ? null
                : ForgeFields.traitsThrough(state, kind, List.of());
        return own != null && PrintedSlice.indexIn(own, trait) >= 0;
    }

    /**
     * A trait of a state that copies nothing, keyed by its index in the
     * state's slice, unless the card granted it to itself: then the line
     * that granted it ({@link GrantingLineSearch#keyOf}), and the index only when no
     * line is found.
     *
     * <p>Forge lists a state's traits printed first, then the ones its trait
     * changes granted, then the keyword-built ones. A trait the card's own
     * static or effect granted (a Class level's trigger, Expedition
     * Diviner's "has 'When this creature dies...'", a manland's animated
     * trigger or activated ability) is parsed onto the card's own state, so
     * it stands in that
     * list, past the printed lines, where the converter numbered no line or
     * another one.
     */
    private static ProvenanceKey keyInState(CardState state, CardTraitBase trait, TraitKind kind,
                                            int depth) {
        if (depth < MAX_HOPS && grantedInPlace(state, trait, kind)) {
            ProvenanceKey granting = GrantingLineSearch.keyOf(trait, depth);
            if (granting != null) return granting;
        }
        PrintedSlice slice = PrintedSlice.of(state, kind);
        int index = slice.indexOf(trait);
        return keyAt(state, kind, index < 0 ? slice.indexLike(trait) : index);
    }

    /**
     * Whether a trait stands in its own state's live slice but not among the
     * state's own traits (Forge's {@code get...WithoutChanges()}, the slice
     * through no trait change), or is a stack copy of one that does
     * ({@code getOriginalAbility}): a trait the card granted itself. A
     * printed trait and every keyword-built trait are among its own; a trait
     * another card granted stands in the donor's state, and not in that
     * state's slice. A land trait a type change built is not among its own
     * either, but names no SVar, so {@link GrantingLineSearch#keyOf} finds no line
     * for it.
     *
     * <p>An ability a card's own {@code Animate} granted ({@code Abilities$},
     * a manland's) is one: the effect leaves no back-reference on it, unlike
     * a static's grant, which {@link #grantor} follows first.
     */
    private static boolean grantedInPlace(CardState state, CardTraitBase trait, TraitKind kind) {
        List<? extends CardTraitBase> own = ForgeFields.traitsThrough(state, kind, List.of());
        if (own == null) return false;
        return grantedAmong(own, state, kind, trait) || grantedAmong(own, state, kind, originalOf(trait));
    }

    /** Whether {@code trait} is in its state's live slice and not in {@code own}; false for null. */
    private static boolean grantedAmong(List<? extends CardTraitBase> own, CardState state, TraitKind kind,
                                        CardTraitBase trait) {
        return trait != null && PrintedSlice.indexIn(own, trait) < 0
                && PrintedSlice.indexIn(kind.slice(state), trait) >= 0;
    }

    /**
     * The key of a keyword instance's own line: the keyword {@code state}
     * prints, else the line of the static that granted it, else null.
     *
     * <p>The same answer a trait the keyword built gets from {@link #keyFor}
     * and {@link #keywordGrantor}, less that trait's {@code built}: a board
     * lists one line per keyword instance, and the triggers, statics and
     * abilities the instance carries each name that line plus which of them
     * it is. A keyword with no traits at all, such as flying, has only this
     * to key it.
     */
    static ProvenanceKey keyOfKeyword(CardState state, KeywordInterface keyword) {
        if (state == null || keyword == null) return null;
        Card host = state.getCard();
        if (host != null && !isScriptless(host)) {
            ProvenanceKey printed = keyOfPrinted(state, keyword, null, 0);
            if (printed != null) return printed;
        }
        return grantingLine(keyword.getStatic());
    }

    /**
     * The line of the static that granted a keyword, or null. A static a
     * keyword built (Dash's or Blitz's haste, a Class level's) grants as its
     * keyword's line, bare as that line keys, so a keyword it grants its own
     * card is one slot with the granting keyword's line, not a second one.
     */
    private static ProvenanceKey grantingLine(StaticAbility granting) {
        ProvenanceKey key = granting == null ? null : of(granting);
        return key == null ? null : key.withoutBuilt();
    }

    /** How many copies of copies are followed before giving up. */
    private static final int MAX_COPY_DEPTH = 8;

    /**
     * The keyword line {@code state} prints for {@code keyword}, carrying
     * {@code built} when it is set, or null.
     *
     * <p>A copy's state (a Clone's, a Phantasmal Image's) is read through
     * what it copies: its keywords are the original's, keyed into the
     * original's script file; one the copy effect added has the line that
     * made the copy, and is no keyword line of the copier's own file, so
     * that line's key is returned as it is, without {@code built}, and one
     * another permanent's static granted keys to that static. The
     * original is read as it was when the copy was made
     * ({@link CopyHistory#stateAsOf}), not as it is now: a later copy effect
     * on it (Cytoshape on Moldering Karok) changes the original, not the
     * copy's values (707.2).
     *
     * <p>A mutated permanent's and a token or spell copy's state are no such
     * copy: they are read by {@link #keyOfUncloned}.
     */
    private static ProvenanceKey keyOfPrinted(CardState state, KeywordInterface keyword,
                                              ProvenanceKey.Built built, int depth) {
        Map.Entry<Long, CardCloneStates> copy = CopyHistory.copyHolding(state);
        if (copy == null) {
            return keyOfUncloned(state, keyword, built, depth);
        }
        CardState original = depth < MAX_COPY_DEPTH ? copiedStateHolding(copy, state, keyword) : null;
        if (original != null) {
            return keyOfPrinted(original, keyword, built, depth + 1);
        }
        return keyOfAddedKeyword(copy, keyword);
    }

    /**
     * The key of a keyword the copied state does not hold: the line of the
     * static that granted it when one did (an Aura's, an Equipment's, a lord's:
     * the key it has on a creature that copies nothing), else the line of the
     * copy effect that added it, which carries no static.
     */
    private static ProvenanceKey keyOfAddedKeyword(Map.Entry<Long, CardCloneStates> copy, KeywordInterface keyword) {
        if (keyword.getStatic() != null) {
            return grantingLine(keyword.getStatic());
        }
        CardTraitBase source = copy.getValue().getSource();
        return source == null ? null : of(source);
    }

    /**
     * The state of what a copy copied, as it showed it when the copy was
     * made, that holds {@code keyword} of the copy's {@code state}: its
     * state of the copy's name, else of the face it shows (a copy of a
     * transformed card copies its back face into its own front). Null for
     * none: a keyword the copy effect added.
     */
    private static CardState copiedStateHolding(Map.Entry<Long, CardCloneStates> copy,
                                                CardState state, KeywordInterface keyword) {
        Card origin = CopyHistory.originOf(copy);
        if (origin == null) {
            return null;
        }
        for (CardStateName name : List.of(state.getStateName(), origin.getCurrentStateName())) {
            CardState original = CopyHistory.stateAsOf(origin, name, copy.getKey());
            if (holdsKeyword(original, state, keyword)) {
                return original;
            }
        }
        return null;
    }

    /** Whether {@code original}, a state other than the copy's {@code state}, holds {@code keyword}. */
    private static boolean holdsKeyword(CardState original, CardState state, KeywordInterface keyword) {
        return original != null && original != state && FaceOrder.keywordIndex(original, keyword) >= 0;
    }

    /**
     * The keyword line a state no copy effect's clone states hold prints
     * for {@code keyword}: a mutated permanent's keyword in the file of the
     * merged card that prints it, a token or spell copy's in the original's
     * file at the original's ordinal, a token copy of a Clone's through the
     * printed state of what the Clone copied when the token was made, and
     * none for one its copy effect added ({@link KeywordPrinters}), unless
     * the Clone's own copy effect added it, which keys to that effect's line
     * ({@link KeywordPrinters#copyClauseOf}); any other state's at its own
     * ordinal.
     */
    private static ProvenanceKey keyOfUncloned(CardState state, KeywordInterface keyword,
                                               ProvenanceKey.Built built, int depth) {
        CardState printer = KeywordPrinters.printerOf(state, keyword);
        if (printer == state) {
            return withBuilt(keyAt(state, TraitKind.KEYWORD, KeywordPrinters.indexOf(state, keyword)), built);
        }
        if (printer == null) {
            SpellAbility clause = KeywordPrinters.copyClauseOf(state, keyword);
            return clause == null ? null : of(clause);
        }
        return depth >= MAX_COPY_DEPTH ? null : keyOfPrinted(printer, keyword, built, depth + 1);
    }

    private static ProvenanceKey withBuilt(ProvenanceKey key, ProvenanceKey.Built built) {
        return key == null || built == null ? key : key.withBuilt(built.kind(), built.index());
    }

    /**
     * Which of its keyword instance's traits {@code trait} is: its kind and
     * its position, by identity, in the instance's list of that kind, else
     * the position of the ability it is a stack copy of
     * ({@code getOriginalAbility}, which {@code MagicStack.add} sets on the
     * activated ability it pushes).
     *
     * <p>Null when neither is in the list, as with a copy Forge made and set
     * the keyword on with no back-reference (a flashback cast): the key stays
     * the bare keyword key, which joins the keyword's one line, and no index
     * is guessed.
     */
    static ProvenanceKey.Built builtOf(CardTraitBase trait) {
        TraitKind kind = TraitKind.of(trait);
        if (kind == null) return null;
        Iterable<? extends CardTraitBase> built = kind.builtBy(trait.getKeyword());
        int index = indexOfSame(built, trait);
        if (index < 0) index = indexOfSame(built, originalOf(trait));
        return index < 0 ? null : new ProvenanceKey.Built(kind, index);
    }

    /** The position of {@code trait} itself in {@code items}, or -1; -1 for null. */
    private static int indexOfSame(Iterable<? extends CardTraitBase> items, CardTraitBase trait) {
        return trait == null ? -1 : SoleMatch.indexOf(items, each -> each == trait);
    }

    /** The ability a stack copy was made from, or null. */
    private static CardTraitBase originalOf(CardTraitBase trait) {
        return trait instanceof SpellAbility sa ? sa.getOriginalAbility() : null;
    }

    /**
     * The copy effect's states {@code state} is one of: its card's clone
     * states that hold it, when they are a copy effect's
     * ({@link CopyHistory#copyHolding}, which a mutation's or a prototype's
     * are not), else null.
     */
    static CardCloneStates copyStatesOf(CardState state) {
        Map.Entry<Long, CardCloneStates> copy = CopyHistory.copyHolding(state);
        return copy == null ? null : copy.getValue();
    }

    /**
     * The key of the {@code index}-th {@code kind} trait of {@code state}, or
     * null for a negative index, which is how every index search here says it
     * found nothing, or for a state the converter emitted no face for.
     */
    private static ProvenanceKey keyAt(CardState state, TraitKind kind, int index) {
        if (index < 0) return null;
        Card host = state.getCard();
        Integer face = FaceOrder.faceIndex(host, state.getStateName());
        if (face == null) return null;
        return new ProvenanceKey(ScriptLocator.scriptFileOf(host), face, kind, index);
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
        return grantedByStatic(trait, keyword) ? new Hop(keyword.getStatic()) : Pass.PASS;
    }

    /** Whether {@code keyword}, the trait's, came from a static and is not one the trait's state prints. */
    private static boolean grantedByStatic(CardTraitBase trait, KeywordInterface keyword) {
        return keyword != null && keyword.getStatic() != null
                && FaceOrder.keywordIndex(trait.getCardState(), keyword) < 0;
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
     * The printed line that granted a trait out of one of its own SVars
     * ({@link GrantingLineSearch}).
     *
     * <p>Forge keeps no back-reference for these, so this is the one step
     * that searches rather than follows, and it is last for that reason. It
     * runs only once every back-reference in the chain has come up empty,
     * which is also why walking four trait lists here is cheap: few records
     * ever reach this point. A trait the card granted itself is the
     * exception: {@link #keyInState} searches for its line before its index.
     */
    private static StepResult grantingLine(CardTraitBase trait) {
        return keyed(GrantingLineSearch.keyOf(trait, 0));
    }

    /**
     * Why a trait that reached the end of the chain names no printed line.
     *
     * <p>The engine-built case is checked first and wins: an effect card
     * (The Monarch, The Initiative, a resolved effect's card) is
     * {@code new Card(...)} with its traits built inline, no paper card and
     * no rules, so no script file exists to name. That is expected rather
     * than a resolver failure, and the two have to be distinguishable — an
     * empty key alone cannot tell them apart.
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
     * <p>Tokens and dungeons are deliberately not included: each <em>does</em>
     * have a script, in {@code tokenscripts} (a venture builds its dungeon
     * from the token's paper card), and {@link ScriptLocator#scriptFileOf}
     * finds it, so a room's trigger keys to its dungeon keyword's line.
     */
    private static boolean isScriptless(Card host) {
        return host != null && host.getGamePieceType() == GamePieceType.EFFECT;
    }

    /**
     * Whether a trait is one a type change attached at runtime, missing from
     * its state's printed assembly and so keyed past every printed line
     * ({@link PrintedSlice#addedByTypeChange}).
     */
    static boolean addedByTypeChange(CardTraitBase trait) {
        return PrintedSlice.addedByTypeChange(trait);
    }
}
