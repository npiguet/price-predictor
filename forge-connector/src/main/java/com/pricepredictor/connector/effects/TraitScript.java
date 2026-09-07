package com.pricepredictor.connector.effects;

import forge.game.CardTraitBase;
import forge.game.ability.ApiType;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * The script behind one runtime trait: its API type, its parameter keys, its
 * text, and the index paths of its sub-abilities.
 *
 * <p>Everything here is read from the trait's own parameter map rather than
 * from the card script on disk, so it is available for a granted or generated
 * trait that no file ever contained. {@code scriptText} is the stage-four
 * primary encoding surface: the model reads the mechanism instead of its
 * description, and every command that encodes finds it in the sidecar rather
 * than needing a Forge cardsfolder path of its own.
 */
public record TraitScript(
        String apiType,
        List<String> paramKeys,
        String scriptText,
        List<SubAbilityLink> subAbilityLinks
) {

    /** One sub-ability below a trait, addressed by its index path. */
    public record SubAbilityLink(List<Integer> path, String label) {
        public String toJson() {
            StringBuilder sb = new StringBuilder("{\"path\":[");
            for (int i = 0; i < path.size(); i++) {
                if (i > 0) sb.append(',');
                sb.append(path.get(i));
            }
            return sb.append("],\"label\":").append(Json.string(label)).append('}')
                    .toString();
        }
    }

    /**
     * The script behind a keyword.
     *
     * <p>A {@code KeywordInterface} is not a {@code CardTraitBase} and carries
     * no parameter map; its original script line is all there is, and it is the
     * only surface a keyword-derived line has.
     */
    public static TraitScript ofKeyword(String original) {
        return new TraitScript("Keyword", List.of(), original, List.of());
    }

    /** Read the script behind a trait; never null, but may be all-empty. */
    public static TraitScript of(CardTraitBase trait) {
        if (trait == null) {
            return new TraitScript(null, List.of(), null, List.of());
        }
        SpellAbility effect = effectOf(trait);
        Map<String, String> params = new TreeMap<>(trait.getMapParams());
        String apiType = apiTypeOf(trait, effect);
        List<SubAbilityLink> links = new ArrayList<>();
        if (effect != null) {
            collectSubAbilities(effect, links);
        }
        return new TraitScript(
                apiType,
                List.copyOf(params.keySet()),
                render(params),
                List.copyOf(links));
    }

    /**
     * The ability a trait executes, where it has one.
     *
     * <p>A trigger's effect is its overriding ability; a spell or activated
     * ability is its own effect; a static or replacement has none of its own.
     */
    private static SpellAbility effectOf(CardTraitBase trait) {
        if (trait instanceof Trigger trigger) {
            return trigger.getOverridingAbility();
        }
        if (trait instanceof SpellAbility sa) {
            return sa;
        }
        return null;
    }

    private static String apiTypeOf(CardTraitBase trait, SpellAbility effect) {
        if (effect != null) {
            ApiType api = effect.getApi();
            if (api != null) return api.name();
        }
        // A static ability's Mode is the closest thing it has to an API type,
        // and it is what the script-API auxiliary head predicts for one.
        String mode = trait.getParam("Mode");
        if (mode != null) return mode;
        return null;
    }

    /**
     * Index paths of the sub-abilities below an ability, in resolution order.
     *
     * <p>Forge chains sub-abilities linearly — each one points at the next
     * through {@code getSubAbility()} — so position {@code i} in the chain is
     * path {@code [i]}. The label is the SVar name the parent referenced it by,
     * which is what a card script writes, falling back to the API type where
     * the reference is anonymous.
     *
     * <p>This list is what an event's {@code attributed_to} names. An event
     * attributed to a link that is not here falls back to the root line rather
     * than being dropped, so attribution is never lossy.
     */
    private static void collectSubAbilities(SpellAbility ability,
                                            List<SubAbilityLink> out) {
        SpellAbility parent = ability;
        AbilitySub sub = ability.getSubAbility();
        int index = 0;
        while (sub != null) {
            String svar = parent.getParam("SubAbility");
            ApiType api = sub.getApi();
            String label = svar != null ? svar : (api != null ? api.name() : "");
            out.add(new SubAbilityLink(List.of(index), label));
            parent = sub;
            sub = sub.getSubAbility();
            index++;
        }
    }

    /** The parameter map as a Forge script line, keys in a stable order. */
    private static String render(Map<String, String> params) {
        if (params.isEmpty()) return null;
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, String> entry : params.entrySet()) {
            if (sb.length() > 0) sb.append(" | ");
            sb.append(entry.getKey()).append("$ ").append(entry.getValue());
        }
        return sb.toString();
    }
}
