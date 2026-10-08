package com.pricepredictor.connector.effects;

import forge.game.CardTraitBase;
import forge.game.ability.ApiType;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Which chosen mode of a charm a resolving clause belongs to.
 *
 * <p>Forge gives a mode no identity of its own. {@code CharmEffect.chainAbilities}
 * copies each chosen mode, numbers the copy's {@code CharmOrder} SVar with its
 * chain position, appends it to the charm's tail and keeps no back-reference,
 * so by the time a clause resolves the only things that say "this is mode two"
 * are that SVar and the copy's parameters. The mode's stable name is its
 * 0-based position in the charm's {@code Choices$} (research.md: a mode's key
 * is the root key plus its {@code Choices$} position), which is what both the
 * converter and this side can compute.
 *
 * <p>A clone is recognised by an {@code own} {@code CharmOrder}: a trait's
 * SVar lookup falls back to its parent, so a mode's own sub-abilities answer
 * their mode's number too and are told apart by answering the same number as
 * their parent. The position is recovered by fingerprint first — the copy's
 * {@code getOriginalMapParams()}, which {@code copyHelper} preserves across
 * every copy the stack and the chain make — and, where two modes share a
 * fingerprint, by the chosen list's order, which {@code chainAbilities} sorts
 * in place and numbers the clones by. A mode nothing can place keeps the root
 * key rather than guessing: a record with the wrong option trains the wrong
 * line.
 */
final class CharmModes {

    private static final String CHARM_ORDER = "CharmOrder";
    private static final String CHOICES = "Choices";

    private CharmModes() {
    }

    /** A clause that opens a chosen mode, and which mode it is. */
    record ModeStart(SpellAbility charm, SpellAbility opening, Integer option) {
    }

    /**
     * The mode this clause opens, or null when it opens none.
     *
     * <p>Only the clone itself opens a mode: its own sub-abilities resolve
     * inside it and are already covered by the half it opened.
     */
    static ModeStart modeStartOf(SpellAbility clause) {
        if (clause == null || !isModeClone(clause)) {
            return null;
        }
        SpellAbility charm = charmOf(clause);
        if (charm == null) {
            return null;
        }
        return new ModeStart(charm, clause, optionWithin(charm, clause));
    }

    /**
     * The option of the mode {@code trait} resolves inside, or null.
     *
     * <p>Walks up from the clause rather than down from the root, for the
     * reason {@link EventAttribution} does: Forge resolves copies, and the
     * clause in hand is the only object on the resolving chain this side holds.
     */
    static Integer optionOf(CardTraitBase trait) {
        if (!(trait instanceof SpellAbility ability)) {
            return null;
        }
        for (SpellAbility t = ability; t != null; t = t.getParent()) {
            if (isModeClone(t)) {
                SpellAbility charm = charmOf(t);
                return charm == null ? null : optionWithin(charm, t);
            }
        }
        return null;
    }

    /**
     * Whether this sub-ability is a clone {@code chainAbilities} appended.
     *
     * <p>A mode's own sub-abilities inherit its number through the SVar
     * fallback and read the same value as their parent; a clone's parent is
     * the previous mode's tail, or the charm itself, and reads a different one
     * or none.
     */
    private static boolean isModeClone(SpellAbility ability) {
        if (!(ability instanceof AbilitySub) || ability.getParent() == null) {
            return false;
        }
        Integer order = ability.getSVarInt(CHARM_ORDER);
        if (order == null) {
            return false;
        }
        Integer parentOrder = ability.getParent().getSVarInt(CHARM_ORDER);
        return parentOrder == null || !parentOrder.equals(order);
    }

    /** The nearest enclosing charm, which is the one whose modes these are. */
    private static SpellAbility charmOf(SpellAbility clone) {
        for (SpellAbility t = clone.getParent(); t != null; t = t.getParent()) {
            if (t.getApi() == ApiType.Charm
                    && !t.getAdditionalAbilityList(CHOICES).isEmpty()) {
                return t;
            }
        }
        return null;
    }

    /** The clone's 0-based position in the charm's {@code Choices$}, or null. */
    private static Integer optionWithin(SpellAbility charm, SpellAbility clone) {
        List<AbilitySub> choices = charm.getAdditionalAbilityList(CHOICES);
        Map<String, String> want = clone.getOriginalMapParams();
        int hit = -1;
        int hits = 0;
        for (int i = 0; i < choices.size(); i++) {
            if (Objects.equals(choices.get(i).getOriginalMapParams(), want)) {
                hit = i;
                hits++;
            }
        }
        if (hits == 1) {
            return hit;
        }
        // Two modes with one fingerprint: fall back to the order the chain was
        // built in, which is the sorted chosen list's. The chosen entries are
        // the charm's own Choices objects, or the original ability's where the
        // stack copied an activated charm (its copy of the lists is fresh, but
        // the copied chosen list still points at the originals).
        Integer order = clone.getSVarInt(CHARM_ORDER);
        List<AbilitySub> chosen = charm.getChosenList();
        if (order == null || chosen == null || order < 1 || order > chosen.size()) {
            return null;
        }
        AbilitySub picked = chosen.get(order - 1);
        Integer index = indexByIdentity(choices, picked);
        if (index == null && charm.getOriginalAbility() != null) {
            index = indexByIdentity(
                    charm.getOriginalAbility().getAdditionalAbilityList(CHOICES), picked);
        }
        return index;
    }

    private static Integer indexByIdentity(List<AbilitySub> choices, AbilitySub wanted) {
        for (int i = 0; i < choices.size(); i++) {
            if (choices.get(i) == wanted) {
                return i;
            }
        }
        return null;
    }
}
