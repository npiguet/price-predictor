package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.Ability;
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
 * text of their own — a permanent's implicit cast spell, say. A trait the
 * converter deduplicated, merged or derived from a keyword keeps its
 * attribution on the line that carries its text, because it is still live at
 * runtime and a record naming it must reach the words it produced.
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

    /**
     * Note that a trait exists on this face, whether or not it produced a line.
     *
     * <p>Called for every trait the parser walks, so the dropped set can be the
     * difference between what the card has and what the file shows.
     */
    public void declare(ProvenanceKey key) {
        if (key != null) declared.add(key);
    }

    /** Attribute one or more parsed abilities to the trait that produced them. */
    public void attribute(List<Ability> abilities, ProvenanceKey key,
                          CardTraitBase trait) {
        if (key == null) return;
        declare(key);
        TraitScript script = TraitScript.of(trait);
        for (Ability ability : abilities) {
            if (ability == null) continue;
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
     * <p>A keyword-derived trigger, static or spell returns no entry of its own
     * — its text is the keyword line — but Forge gives it a trait index, and a
     * record fired by it names that key. Claiming the key on the keyword's
     * abilities is what keeps such a record joinable instead of leaving it
     * pointing at a dropped key with nothing to say about the ability.
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
        Source from = byAbility.remove(duplicate);
        if (from == null || survivor == null) return;
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
