package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.ability.OptionAbility;
import forge.game.CardTraitBase;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Collects, while a card is parsed, which runtime trait produced which
 * {@link Ability}.
 *
 * <p>Recording happens at the parse sites rather than inside the fifteen
 * {@code Ability} implementations, because that is the only place both the
 * trait and its index within its kind are in scope. Abilities are keyed by
 * identity: two abilities can render identical text and still come from
 * different traits, which is exactly the case the converter deduplicates and
 * the sidecar has to report.
 *
 * <p>A recorder is per face. Keys declared but attached to no surviving ability
 * end up in the sidecar's {@code dropped_keys}, which holds only traits with no
 * text of their own — a permanent's implicit cast spell, say, or a charm whose
 * modes are its only lines. A trait the converter deduplicated, merged or
 * derived from a keyword keeps its attribution on the line that carries its
 * text, because it is still live at runtime and a record naming it must reach
 * the words it produced. A mode key is never declared: it is claimed by the
 * mode's line or it does not exist, so it can never be dropped.
 */
public final class ProvenanceRecorder {

    /** What one parsed ability came from. */
    public record Source(List<ProvenanceKey> keys, TraitScript script) {
    }

    private final Map<Ability, Source> byAbility = new IdentityHashMap<>();
    private final Set<ProvenanceKey> declared = new LinkedHashSet<>();
    /**
     * The abilities each keyword rendered, kept so a trait Forge derived from
     * that keyword can be claimed by the keyword's own line: the derived trait
     * is walked long after the keyword loop has moved on.
     */
    private final Map<String, List<Ability>> byKeyword = new LinkedHashMap<>();
    /** Where a chain's undefined SVars go; shared across every face of a run. */
    private final MissingSVarReport missingSVars;

    public ProvenanceRecorder() {
        this(new MissingSVarReport());
    }

    public ProvenanceRecorder(MissingSVarReport missingSVars) {
        this.missingSVars = missingSVars;
    }

    /**
     * Note that a trait exists on this face, whether or not it produced a line.
     *
     * <p>Called for every trait the parser walks, so the dropped set can be the
     * difference between what the card has and what the file shows.
     */
    public void declare(ProvenanceKey key) {
        if (key != null) declared.add(key);
    }

    /**
     * Attribute one or more parsed abilities to the trait that produced them.
     *
     * <p>A charm's mode lines are attributed here too, to the trait's key
     * extended by the mode's {@code Choices$} position and to the mode's own
     * chain: they arrive either as the charm line's children or, for a charm
     * with no description of its own, as top-level lines in this list. The
     * root key is then claimed by nobody and lands in {@code dropped_keys},
     * which is the honest answer for a line the file does not show. A mode
     * belonging to some other trait — a chapter's options being attributed to
     * a sibling chapter's trigger — is left alone, so it keys to its own.
     */
    public void attribute(List<Ability> abilities, ProvenanceKey key,
                          CardTraitBase trait) {
        if (key == null) return;
        declare(key);
        TraitScript script = TraitScript.of(trait, key, missingSVars);
        for (Ability ability : abilities) {
            if (ability == null) continue;
            if (ability instanceof OptionAbility option) {
                attributeMode(option, key, trait);
                continue;
            }
            claim(ability, key, script);
            for (Ability sub : ability.subAbilities()) {
                if (sub instanceof OptionAbility option) {
                    attributeMode(option, key, trait);
                }
            }
        }
    }

    private void attributeMode(OptionAbility option, ProvenanceKey rootKey,
                               CardTraitBase trait) {
        if (!option.belongsTo(trait)) return;
        ProvenanceKey modeKey = rootKey.withOption(option.modeIndex());
        claim(option, modeKey, TraitScript.ofMode(
                option.mode(), option.label(), modeKey, missingSVars));
    }

    private void claim(Ability ability, ProvenanceKey key, TraitScript script) {
        Source existing = byAbility.get(ability);
        if (existing == null) {
            List<ProvenanceKey> keys = new ArrayList<>();
            keys.add(key);
            byAbility.put(ability, new Source(keys, script));
        } else if (!existing.keys().contains(key)) {
            // A line merged from several traits carries several keys, so
            // the join is many-to-one and never assumed one-to-one.
            existing.keys().add(key);
        }
    }

    /** Attribute a single parsed ability. */
    public void attribute(Ability ability, ProvenanceKey key, CardTraitBase trait) {
        if (ability != null) attribute(List.of(ability), key, trait);
    }

    /** Attribute abilities to a keyword, which carries a script but no trait. */
    public void attributeKeyword(List<Ability> abilities, ProvenanceKey key,
                                 String original) {
        if (key == null) return;
        declare(key);
        for (Ability ability : abilities) {
            if (ability == null) continue;
            byKeyword.computeIfAbsent(original, k -> new ArrayList<>()).add(ability);
        }
        TraitScript script = TraitScript.ofKeyword(original);
        for (Ability ability : abilities) {
            if (ability == null) continue;
            Source existing = byAbility.get(ability);
            if (existing == null) {
                List<ProvenanceKey> keys = new ArrayList<>();
                keys.add(key);
                byAbility.put(ability, new Source(keys, script));
            } else if (!existing.keys().contains(key)) {
                existing.keys().add(key);
            }
        }
    }

    /**
     * Attribute a runtime trait to the abilities its keyword rendered.
     *
     * <p>A keyword-derived trigger, static, replacement effect or spell returns
     * no entry of its own — its text is the keyword line — but Forge gives it a
     * trait index, and a record fired by it names that key. Claiming the key
     * on the keyword's abilities is what keeps such a record joinable instead
     * of leaving it pointing at a dropped key with nothing to say about the
     * ability.
     */
    public void attributeToKeyword(ProvenanceKey key, CardTraitBase trait,
                                   String original) {
        if (key == null) return;
        declare(key);
        List<Ability> abilities = byKeyword.get(original);
        if (abilities == null || abilities.isEmpty()) return;
        attribute(abilities, key, trait);
    }

    /**
     * Fold a discarded duplicate's attribution into the ability that stays.
     *
     * <p>The converter deduplicates two traits that render one description. The
     * survivor carries the text of both, so it has to carry both keys: dropping
     * the duplicate's key would leave every record fired by that live trait
     * with no line to join to.
     */
    public void merge(Ability survivor, Ability duplicate) {
        // Tested before the removal, so a merge that cannot land leaves the
        // duplicate's attribution where it was rather than discarding it.
        if (survivor == null || duplicate == null) return;
        Source from = byAbility.remove(duplicate);
        if (from == null) return;
        Source into = byAbility.get(survivor);
        if (into == null) {
            byAbility.put(survivor, from);
            return;
        }
        for (ProvenanceKey key : from.keys()) {
            if (!into.keys().contains(key)) into.keys().add(key);
        }
    }

    /**
     * Move an ability's attribution to the object that replaces it.
     *
     * <p>Post-processing rebuilds some abilities as fresh objects — a Class
     * card's level lines, for one — and the recorder keys by identity, so
     * without this the replacement would be textless to the model and the
     * original's key would look dropped.
     */
    public void transfer(Ability from, Ability to) {
        Source source = byAbility.remove(from);
        if (source != null && to != null) byAbility.put(to, source);
    }

    /**
     * Give each mode line its mode under every root key its charm line carries.
     *
     * <p>A charm line can carry several keys: two triggers sharing one charm,
     * which the converter merged, or the triggers a keyword generated. Forge
     * resolves a mode through whichever of them fired, so a mode line keyed
     * under one root alone leaves the others' records with no line. Run once
     * every merge and keyword attribution has landed, which no single
     * attribution site can know.
     */
    public void claimModesUnderEveryRoot(Iterable<Ability> surviving) {
        for (Ability ability : surviving) {
            Source root = ability == null ? null : byAbility.get(ability);
            if (root != null) claimModesBelow(ability, root.keys());
        }
    }

    private void claimModesBelow(Ability parent, List<ProvenanceKey> rootKeys) {
        for (Ability sub : parent.subAbilities()) {
            Source mode = byAbility.get(sub);
            if (sub instanceof OptionAbility option && mode != null) {
                for (ProvenanceKey key : rootKeys) {
                    if (key.option() != null) continue;
                    ProvenanceKey modeKey = key.withOption(option.modeIndex());
                    if (!mode.keys().contains(modeKey)) mode.keys().add(modeKey);
                }
            } else if (mode == null) {
                // A line of the charm's own rendering, between root and modes.
                claimModesBelow(sub, rootKeys);
            }
        }
    }

    /** What produced this ability, or null if it came from no runtime trait. */
    public Source sourceOf(Ability ability) {
        return byAbility.get(ability);
    }

    /** Every key declared on this face, in declaration order. */
    public Set<ProvenanceKey> declaredKeys() {
        return declared;
    }

    /** Keys that no surviving ability claims — the deduplicated traits. */
    public List<ProvenanceKey> droppedKeys(Iterable<Ability> surviving) {
        Set<ProvenanceKey> claimed = new LinkedHashSet<>();
        for (Ability ability : surviving) {
            collectClaimed(ability, claimed);
        }
        List<ProvenanceKey> dropped = new ArrayList<>();
        for (ProvenanceKey key : declared) {
            if (!claimed.contains(key)) dropped.add(key);
        }
        return dropped;
    }

    private void collectClaimed(Ability ability, Set<ProvenanceKey> claimed) {
        Source source = byAbility.get(ability);
        if (source != null) claimed.addAll(source.keys());
        for (Ability sub : ability.subAbilities()) {
            collectClaimed(sub, claimed);
        }
    }
}
