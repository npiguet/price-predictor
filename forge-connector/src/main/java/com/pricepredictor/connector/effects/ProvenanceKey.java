package com.pricepredictor.connector.effects;

import forge.card.CardStateName;
import forge.game.CardTraitBase;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.Trigger;

import java.util.List;
import java.util.Objects;

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
     * The key for a live trait, following the donor chain for granted and
     * copied traits.
     *
     * <p>Returns null when the trait cannot be attributed to a printed line at
     * all — a record naming no key is still trainable through its state and
     * payload, so this is not an error.
     *
     * <p>The chain, in order:
     * <ol>
     *   <li><b>Granted abilities</b> resolve through the grantor accessors to
     *       the donor card's printed line.</li>
     *   <li><b>Copied abilities</b> resolve through the original-ability
     *       back-reference, which is set only for copies.</li>
     *   <li><b>Copy-spell effects</b> (Fork, Reverberate) carry only a copied
     *       flag, so they resolve through the stack object's source card.</li>
     * </ol>
     *
     * <p>{@code getOriginalHost()} is deliberately not used to key a granted
     * trait: {@code CardTraitBase.getOriginalHost()} returns
     * {@code getCardState().getCard()}, which for a granted trait is the
     * <em>recipient</em>, not the donor, so keying by it would attribute an
     * anthem's effect to the creature that received it.
     */
    public static ProvenanceKey of(CardTraitBase trait) {
        CardTraitBase printed = resolveToPrinted(trait);
        if (printed == null) return null;
        String kind = kindOf(printed);
        if (kind == null) return null;
        CardState state = printed.getCardState();
        if (state == null) return null;
        Card host = state.getCard();
        if (host == null) return null;
        int index = indexWithin(state, printed, kind);
        if (index < 0) return null;
        return new ProvenanceKey(
                CardFilenames.scriptFile(treeOf(host), host.getName()),
                faceIndex(host, state.getStateName()),
                kind,
                index);
    }

    /** Follow the granted / copied / copy-spell chain back to a printed trait. */
    private static CardTraitBase resolveToPrinted(CardTraitBase trait) {
        if (trait == null) return null;
        if (trait.isIntrinsic() && !trait.isCopiedTrait()) {
            return trait;
        }
        if (trait instanceof SpellAbility sa) {
            // 1. Granted: the grantor static's own trait is the printed one.
            StaticAbility grantor = sa.getGrantorStatic();
            if (grantor != null) {
                return grantor;
            }
            // 2. Copied: the original ability back-reference.
            SpellAbility original = sa.getOriginalAbility();
            if (original != null && original != sa) {
                return resolveToPrinted(original);
            }
            // 3. Copy-spell effect: only a copied flag survives, so fall back to
            //    the stack object's source card and key its own spell trait.
            if (sa.isCopied()) {
                Card source = sa.getHostCard();
                if (source != null) {
                    CardState state = source.getCurrentState();
                    if (state != null) {
                        for (SpellAbility printed : state.getSpellAbilities()) {
                            return printed;
                        }
                    }
                }
            }
        }
        // A granted non-SpellAbility trait has no grantor accessor of its own;
        // the trait as it stands is the best available attribution.
        return trait;
    }

    /** Index of {@code trait} within its kind's slice of {@code state}'s trait list. */
    private static int indexWithin(CardState state, CardTraitBase trait, String kind) {
        Iterable<? extends CardTraitBase> slice = switch (kind) {
            case KIND_TRIGGER -> state.getTriggers();
            case KIND_REPLACEMENT -> state.getReplacementEffects();
            case KIND_STATIC -> state.getStaticAbilities();
            case KIND_SPELL -> state.getSpellAbilities();
            default -> null;
        };
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
     * Which converted tree a live card's script came from.
     *
     * <p>A staged variant is an ordinary non-token card by the time Forge has
     * loaded it, so the tree cannot be read off the card — it comes from the
     * names {@link VariantRegistry} recorded at staging time. Without that a
     * variant's records would be keyed under {@code cardsfolder/}, where no
     * sidecar of that name exists, and the join would fail loudly on a corpus
     * that is in fact correct.
     */
    private static String treeOf(Card card) {
        if (card.isToken()) return SourceTree.TOKENSCRIPTS;
        if (VariantRegistry.isVariant(card.getName())) {
            return SourceTree.VARIANT_SCRIPTS;
        }
        return SourceTree.CARDSFOLDER;
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
