package com.pricepredictor.connector.effects;

import com.google.common.collect.MapMaker;
import forge.game.CardTraitBase;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * An ability's identity across Forge's recomputes: its kind and its script
 * parameters, which a static's re-granted copy keeps while its object and id
 * are new.
 *
 * <p>Each trait's signature is kept by the trait's identity, beside the
 * parameters it was built from, and is used again while the trait's
 * parameters still equal them: Forge rewrites a trait's original parameters
 * in place (an announced variable) or replaces them (an intrinsic text
 * change), so the trait object alone is no key. The keys are weak, so a
 * trait no game holds is dropped with its entry, and an entry holds the
 * parameters' strings, never the trait.
 */
final class TraitSignatures {

    private record Known(Map<String, String> params, String signature) {
    }

    private final Map<CardTraitBase, Known> known = new MapMaker().weakKeys().makeMap();

    /** The trait's signature: its class's simple name and its original parameters, sorted by key. */
    String of(CardTraitBase trait) {
        Map<String, String> params = trait.getOriginalMapParams();
        Known kept = known.get(trait);
        if (kept != null && kept.params().equals(params)) {
            return kept.signature();
        }
        String signature = trait.getClass().getSimpleName() + new TreeMap<>(params);
        known.put(trait, new Known(new HashMap<>(params), signature));
        return signature;
    }
}
