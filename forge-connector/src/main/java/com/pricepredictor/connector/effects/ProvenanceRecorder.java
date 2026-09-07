package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.Ability;
import forge.game.CardTraitBase;

import java.util.ArrayList;
import java.util.IdentityHashMap;
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
 * end up in the sidecar's {@code dropped_keys} — the converter merged or
 * deduplicated that trait away, but it is still live at runtime and still
 * produces a key, so a record naming it must be kept rather than failing the
 * join.
 */
public final class ProvenanceRecorder {

    /** What one parsed ability came from. */
    public record Source(List<ProvenanceKey> keys, TraitScript script) {
    }

    private final Map<Ability, Source> byAbility = new IdentityHashMap<>();
    private final Set<ProvenanceKey> declared = new LinkedHashSet<>();

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
