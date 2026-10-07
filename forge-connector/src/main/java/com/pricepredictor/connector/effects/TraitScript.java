package com.pricepredictor.connector.effects;

import forge.game.CardTraitBase;
import forge.game.ability.ApiType;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
 *
 * <p>The text is the <em>whole chain</em> the trait owns, not its first line:
 * the root segment, then every ability a chain parameter of an emitted segment
 * names — a trigger's {@code Execute$}, each {@code SubAbility$}, a
 * {@code RepeatSubAbility$}, a replacement's {@code ReplaceWith$}, the
 * {@code Choices$} of a non-modal chooser — each segment its own parameters in
 * key order, joined by {@value #SEGMENT_SEPARATOR} and opened by the label its
 * parent referenced it by. Labels are the script author's own SVar names,
 * 3,488 distinct ones across the card tree and most used once, so they are
 * renamed by position ({@code SV1}, {@code SV2}, …) in order of first
 * appearance reading the text from its start: two cards whose chains differ
 * only in what they called a sub-ability render the same text. An SVar is
 * emitted once per chain, at its first reach; one the script never defines is
 * numbered like any other, left out, and reported.
 *
 * <p>A charm is the exception to inlining: its root segment names its modes
 * ({@code Choices$ SV1,SV2,…}) and stops, because each mode is encoded on the
 * converter's {@code option} line for it, built by {@link #ofMode}.
 */
public record TraitScript(
        String apiType,
        List<String> paramKeys,
        String scriptText,
        List<SubAbilityLink> subAbilityLinks
) {

    /** What separates two segments of a chain; seeded as a special token. */
    public static final String SEGMENT_SEPARATOR = " [SEG] ";

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
        return of(trait, null, null);
    }

    /**
     * Read the script behind a trait, reporting the SVars its chain names but
     * the script never defines.
     *
     * @param key    the trait's provenance key, named in the report
     * @param report where undefined references go, or null to drop them
     */
    public static TraitScript of(CardTraitBase trait, ProvenanceKey key,
                                 MissingSVarReport report) {
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
        String text = params.isEmpty() ? null
                : new ChainRenderer(trait, key, report).render(null, trait, params);
        return new TraitScript(
                apiType,
                List.copyOf(params.keySet()),
                text,
                List.copyOf(links));
    }

    /**
     * The script behind one mode of a charm, for its {@code option} line.
     *
     * <p>The mode's own chain, opened by the label the charm referenced it by —
     * renamed {@code SV1}, since numbering restarts on every line — then its
     * sub-abilities as for any trait. The API type is the mode's own, which is
     * what a record resolving through the mode's key acts as.
     *
     * @param label the mode's item in the charm's {@code Choices$}
     */
    public static TraitScript ofMode(SpellAbility mode, String label, ProvenanceKey key,
                                     MissingSVarReport report) {
        if (mode == null) {
            return new TraitScript(null, List.of(), null, List.of());
        }
        Map<String, String> params = new TreeMap<>(mode.getMapParams());
        ApiType api = mode.getApi();
        List<SubAbilityLink> links = new ArrayList<>();
        collectSubAbilities(mode, links);
        String text = params.isEmpty() ? null
                : new ChainRenderer(mode, key, report).render(label, mode, params);
        return new TraitScript(
                api == null ? null : api.name(),
                List.copyOf(params.keySet()),
                text,
                List.copyOf(links));
    }

    /**
     * The ability a trait executes, where it has one.
     *
     * <p>A trigger's effect is its overriding ability; a spell or activated
     * ability is its own effect; a replacement's is the ability its
     * {@code ReplaceWith$} names; a static has none of its own.
     */
    private static SpellAbility effectOf(CardTraitBase trait) {
        if (trait instanceof Trigger trigger) {
            return trigger.getOverridingAbility();
        }
        if (trait instanceof ReplacementEffect replacement) {
            return replacement.getOverridingAbility();
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

    /**
     * Renders one line's chain, renaming labels as it goes.
     *
     * <p>Segments are emitted breadth-first in the order their references were
     * read, so the opener numbers rise through the text: a reference is
     * numbered the moment it is rendered, and the segment it names is queued
     * right then. One renderer is one line, which is what makes the numbering
     * restart per line.
     */
    private static final class ChainRenderer {

        /** A parameter whose value is one SVar label naming one ability. */
        private static final Set<String> SINGLE_LINK_KEYS =
                Set.of("Execute", "SubAbility", "RepeatSubAbility", "ReplaceWith");
        /** The parameter whose value is a comma-separated list of labels. */
        private static final String CHOICES = "Choices";
        /**
         * The APIs on which {@code Choices$} names abilities. On any other API
         * — {@code PutCounter}, {@code ChangeZone} — the same parameter is a
         * selector ({@code Choices$ Creature.YouCtrl}) and must stay as written;
         * the list is {@code AbilityFactory}'s own.
         */
        private static final Set<ApiType> CHOOSER_APIS = Set.of(
                ApiType.Charm, ApiType.GenericChoice, ApiType.AssignGroup,
                ApiType.VillainousChoice, ApiType.Vote);

        private final CardTraitBase lineTrait;
        private final ProvenanceKey key;
        private final MissingSVarReport report;
        private final Map<String, Integer> numbers = new HashMap<>();
        private final Set<String> reached = new HashSet<>();
        private final Deque<Pending> queue = new ArrayDeque<>();
        private final StringBuilder out = new StringBuilder();

        /** A segment waiting its turn: the label it opens with and what to render. */
        private record Pending(String label, SpellAbility ability) {
        }

        ChainRenderer(CardTraitBase lineTrait, ProvenanceKey key, MissingSVarReport report) {
            this.lineTrait = lineTrait;
            this.key = key;
            this.report = report;
        }

        /** The whole chain below {@code root}, opened by {@code label} when it has one. */
        String render(String label, CardTraitBase root, Map<String, String> params) {
            if (label != null) {
                reached.add(label);
            }
            segment(label, root, params);
            while (!queue.isEmpty()) {
                Pending next = queue.removeFirst();
                out.append(SEGMENT_SEPARATOR);
                segment(next.label(), next.ability(),
                        new TreeMap<>(next.ability().getMapParams()));
            }
            return out.toString();
        }

        private void segment(String label, CardTraitBase trait, Map<String, String> params) {
            if (label != null) {
                out.append(number(label)).append(": ");
            }
            boolean first = true;
            for (Map.Entry<String, String> entry : params.entrySet()) {
                if (!first) out.append(" | ");
                first = false;
                String name = entry.getKey();
                String value = entry.getValue();
                out.append(name).append("$ ");
                if (SINGLE_LINK_KEYS.contains(name) && value != null) {
                    String reference = value.trim();
                    out.append(number(reference));
                    follow(name, reference, linked(name, reference, trait));
                } else if (CHOICES.equals(name) && value != null && isChooser(trait)) {
                    renderChoices(value, (SpellAbility) trait);
                } else {
                    out.append(value);
                }
            }
        }

        /**
         * A charm's modes are named and not followed: each lives on its own
         * {@code option} line. Any other chooser's choices are part of the
         * trait's chain like a sub-ability.
         */
        private static boolean isChooser(CardTraitBase trait) {
            return trait instanceof SpellAbility sa && CHOOSER_APIS.contains(sa.getApi());
        }

        private void renderChoices(String value, SpellAbility chooser) {
            String[] labels = value.split(",");
            boolean charm = chooser.getApi() == ApiType.Charm;
            List<AbilitySub> choices = chooser.getAdditionalAbilityList(CHOICES);
            for (int i = 0; i < labels.length; i++) {
                if (i > 0) out.append(',');
                String label = labels[i].trim();
                out.append(number(label));
                if (!charm) {
                    follow(CHOICES, label, i < choices.size() ? choices.get(i) : null);
                }
            }
        }

        /** The ability a chain parameter names, as the trait resolved it, or null. */
        private static SpellAbility linked(String name, String value, CardTraitBase trait) {
            if (value == null) return null;
            return switch (name) {
                case "Execute" -> trait instanceof Trigger trigger
                        ? trigger.getOverridingAbility()
                        : trait instanceof SpellAbility sa
                                ? sa.getAdditionalAbility("Execute") : null;
                case "ReplaceWith" -> trait instanceof ReplacementEffect replacement
                        ? replacement.getOverridingAbility() : null;
                case "SubAbility" -> trait instanceof SpellAbility sa
                        ? sa.getSubAbility() : null;
                case "RepeatSubAbility" -> trait instanceof SpellAbility sa
                        ? sa.getAdditionalAbility("RepeatSubAbility") : null;
                default -> null;
            };
        }

        /**
         * Queue the ability a reference names, once per label per chain.
         *
         * <p>Forge builds a fresh object for every reference, so the object
         * graph is a tree and nothing here can loop; the once-per-label rule is
         * what keeps a sub-ability two branches share from being rendered
         * twice. A label the script never defines is one the accessor could not
         * resolve: reported, and its segment left out.
         */
        private void follow(String parameter, String label, SpellAbility ability) {
            if (label == null || label.isEmpty() || !reached.add(label)) return;
            if (ability == null) {
                if (report != null) {
                    report.add(cardName(), key, parameter, label);
                }
                return;
            }
            queue.addLast(new Pending(label, ability));
        }

        /** The label's number on this line, assigned at its first appearance. */
        private String number(String label) {
            if (label == null) return "null";
            return "SV" + numbers.computeIfAbsent(label.trim(), l -> numbers.size() + 1);
        }

        private String cardName() {
            try {
                return lineTrait.getHostCard() == null ? null : lineTrait.getHostCard().getName();
            } catch (RuntimeException e) {
                return null;
            }
        }
    }
}
