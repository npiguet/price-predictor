package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
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
import java.util.regex.Pattern;

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
 * {@code Choices$} of a non-modal chooser, each trait a granting parameter
 * names ({@link GrantedTraits}: a static's {@code AddTrigger$}, an Animate's
 * {@code Triggers$}) — each segment its own parameters in
 * key order, joined by {@value #SEGMENT_SEPARATOR} and opened by the label its
 * parent referenced it by. An ability Forge chained with no parameter naming
 * it (a trigger's or replacement's effect set directly, a sub-ability a
 * keyword attached) is a segment too, opened by a label of its own. Labels are the script author's own SVar names,
 * thousands of distinct ones across the card tree and most used once, so they are
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

    /**
     * Every parameter a chain is followed through: one label naming one
     * ability ({@code SubAbility$}, {@code Execute$}, {@code ReplaceWith$}
     * and Forge's other additional abilities), and a chooser's {@code
     * Choices$}.
     */
    static Set<String> chainKeys() {
        Set<String> keys = new HashSet<>(ChainRenderer.SINGLE_LINK_KEYS);
        keys.addAll(ChainRenderer.ADDITIONAL_LINK_KEYS);
        keys.add(ChainRenderer.CHOICES);
        keys.addAll(GrantedTraits.parameters());
        return keys;
    }

    /** What separates two segments of a chain; seeded as a special token. */
    public static final String SEGMENT_SEPARATOR = " [SEG] ";

    /** One sub-ability below a trait, addressed by its index path. */
    public record SubAbilityLink(List<Integer> path, String label) {
        public String toJson() {
            return Json.object()
                    .value("path", path)
                    .value("label", label)
                    .render();
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

    /** The script of a line no trait stands behind. */
    public static final TraitScript EMPTY = new TraitScript(null, List.of(), null, List.of());

    /**
     * Read the script behind a trait, reporting the SVars its chain names but
     * the script never defines.
     *
     * @param key    the trait's provenance key, named in the report
     * @param report where undefined references go, or null to drop them
     */
    public static TraitScript of(CardTraitBase trait, ProvenanceKey key,
                                 MissingSVarReport report) {
        return trait == null ? EMPTY : ofTraits(List.of(trait), key, report);
    }

    /**
     * The script of a line several traits print together (a Class level: its
     * level-up ability and the static that grants the level's traits): the
     * first trait's, its chain followed by each other trait's as a segment of
     * its own, in order.
     */
    public static TraitScript ofTraits(List<? extends CardTraitBase> traits, ProvenanceKey key,
                                 MissingSVarReport report) {
        if (traits.isEmpty()) {
            return EMPTY;
        }
        CardTraitBase trait = traits.get(0);
        SpellAbility effect = effectOf(trait);
        Root root = new Root(trait, effect, apiTypeOf(trait, effect), null);
        return build(root, List.copyOf(traits.subList(1, traits.size())), key, report);
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
            return EMPTY;
        }
        ApiType api = mode.getApi();
        return build(new Root(mode, mode, api == null ? null : api.name(), label),
                List.of(), key, report);
    }

    /**
     * The trait a script is read from: {@code effect} is the ability it
     * executes, or null, and {@code label} what its chain opens with, or null.
     */
    private record Root(CardTraitBase trait, SpellAbility effect, String apiType, String label) {
    }

    /**
     * The script of {@code root}: its parameters, the sub-ability links below
     * its effect, and its chain, followed by each of {@code followers}' chains.
     */
    private static TraitScript build(Root root, List<CardTraitBase> followers,
                                     ProvenanceKey key, MissingSVarReport report) {
        CardTraitBase trait = root.trait();
        Map<String, String> params = new TreeMap<>(trait.getMapParams());
        List<SubAbilityLink> links = new ArrayList<>();
        if (root.effect() != null) {
            collectSubAbilities(root.effect(), links);
        }
        String text = params.isEmpty() ? null
                : new ChainRenderer(trait, key, report).render(root.label(), trait, params,
                        unnamed(unnamedEffect(trait, root.effect(), params), followers));
        return new TraitScript(
                root.apiType(),
                List.copyOf(params.keySet()),
                text,
                List.copyOf(links));
    }

    /** The root's unnamed effect, where it has one, then {@code followers}. */
    private static List<CardTraitBase> unnamed(SpellAbility effect, List<CardTraitBase> followers) {
        List<CardTraitBase> all = new ArrayList<>();
        if (effect != null) all.add(effect);
        all.addAll(followers);
        return all;
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

    /**
     * The effect of a trigger or replacement no parameter of its own names, or
     * null. Forge builds a keyword's chapter trigger, haunt trigger and ETB
     * replacement with the effect set directly rather than through an
     * {@code Execute$} or {@code ReplaceWith$}, and the chain still holds it.
     */
    private static SpellAbility unnamedEffect(CardTraitBase trait, SpellAbility effect,
                                              Map<String, String> params) {
        boolean executes = trait instanceof Trigger || trait instanceof ReplacementEffect;
        boolean named = params.containsKey("Execute") || params.containsKey("ReplaceWith");
        return executes && !named ? effect : null;
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
        /**
         * The other parameters whose value is one label naming one ability,
         * as Forge parses them into its additional abilities ({@code
         * AbilityFactory.additionalAbilityKeys}: a clash's, a coin flip's, a
         * branch's, a pile's, a gift's…), followed when Forge holds that
         * ability; elsewhere the same parameter is written as it is.
         */
        private static final Set<String> ADDITIONAL_LINK_KEYS = additionalLinkKeys();

        private static Set<String> additionalLinkKeys() {
            Set<String> keys = new HashSet<>(forge.game.ability.AbilityFactory.additionalAbilityKeys);
            keys.removeAll(SINGLE_LINK_KEYS);
            return Set.copyOf(keys);
        }

        /** Where a root's unnamed effect or follower is reported from. */
        private static final String EFFECT = "Effect";
        /** Where an unnamed sub-ability is reported from. */
        private static final String SUB_ABILITY = "SubAbility";

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
        private int unnamedLabels;

        /** A segment waiting its turn: the label it opens with and what to render. */
        private record Pending(String label, CardTraitBase trait) {
        }

        ChainRenderer(CardTraitBase lineTrait, ProvenanceKey key, MissingSVarReport report) {
            this.lineTrait = lineTrait;
            this.key = key;
            this.report = report;
        }

        /**
         * The whole chain below {@code root}, opened by {@code label} when it
         * has one; each of {@code unnamed}, traits no parameter of the root
         * names (its effect set directly, the line's other traits), follows
         * the root segment as a segment of its own.
         */
        String render(String label, CardTraitBase root, Map<String, String> params,
                      List<CardTraitBase> unnamed) {
            if (label != null) {
                reached.add(label);
            }
            segment(label, root, params);
            for (CardTraitBase trait : unnamed) {
                followUnnamed(EFFECT, trait);
            }
            while (!queue.isEmpty()) {
                Pending next = queue.removeFirst();
                out.append(SEGMENT_SEPARATOR);
                segment(next.label(), next.trait(), new TreeMap<>(next.trait().getMapParams()));
            }
            return out.toString();
        }

        private void segment(String label, CardTraitBase trait, Map<String, String> params) {
            if (label != null) {
                out.append(number(label)).append(": ");
            }
            String separator = "";
            for (Map.Entry<String, String> entry : params.entrySet()) {
                out.append(separator).append(entry.getKey()).append("$ ");
                separator = " | ";
                renderValue(entry.getKey(), entry.getValue(), trait);
            }
            AbilitySub sub = unnamedSubAbility(trait, params);
            if (sub != null) {
                followUnnamed(SUB_ABILITY, sub);
            }
        }

        /** One parameter's value: a link renamed and followed, or as written. */
        private void renderValue(String name, String value, CardTraitBase trait) {
            if (value == null) {
                out.append((String) null);
            } else if (SINGLE_LINK_KEYS.contains(name)) {
                linkTo(name, value.trim(), linked(name, value, trait));
            } else if (isAdditionalLink(name, trait)) {
                linkTo(name, value.trim(), ((SpellAbility) trait).getAdditionalAbility(name));
            } else if (CHOICES.equals(name) && isChooser(trait)) {
                renderChoices(value, (SpellAbility) trait);
            } else if (GrantedTraits.separator(name, trait) != null) {
                renderGrants(name, value, trait);
            } else {
                out.append(value);
            }
        }

        private void linkTo(String parameter, String reference, CardTraitBase linked) {
            out.append(number(reference));
            follow(parameter, reference, linked);
        }

        /** Whether {@code name} is an additional ability Forge holds on {@code trait}. */
        private static boolean isAdditionalLink(String name, CardTraitBase trait) {
            if (!ADDITIONAL_LINK_KEYS.contains(name)) return false;
            return trait instanceof SpellAbility sa && sa.getAdditionalAbility(name) != null;
        }

        /**
         * The sub-ability Forge chained below {@code trait} with no {@code
         * SubAbility$} naming it, or null: a keyword attaches one directly
         * (Backup's granted abilities below its counter).
         */
        private static AbilitySub unnamedSubAbility(CardTraitBase trait,
                                                    Map<String, String> params) {
            if (!(trait instanceof SpellAbility sa) || params.containsKey(SUB_ABILITY)) return null;
            return sa.getSubAbility();
        }

        /**
         * The SVars a granting parameter lists, each renamed, written with
         * the parameter's own separator, and followed as the trait it holds.
         */
        private void renderGrants(String parameter, String value, CardTraitBase granter) {
            String separator = GrantedTraits.separator(parameter, granter);
            String[] names = value.split(Pattern.quote(separator), -1);
            for (int i = 0; i < names.length; i++) {
                if (i > 0) out.append(separator);
                String name = names[i].trim();
                out.append(number(name));
                if (!reached.contains(name)) {
                    follow(parameter, name, GrantedTraits.build(parameter, name, granter));
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
            if (trait instanceof SpellAbility sa) return linkedBelow(name, sa);
            return switch (name) {
                case "Execute" -> trait instanceof Trigger trigger
                        ? trigger.getOverridingAbility() : null;
                case "ReplaceWith" -> trait instanceof ReplacementEffect replacement
                        ? replacement.getOverridingAbility() : null;
                default -> null;
            };
        }

        /** The ability a chain parameter of an ability names, or null. */
        private static SpellAbility linkedBelow(String name, SpellAbility sa) {
            return switch (name) {
                case "Execute", "RepeatSubAbility" -> sa.getAdditionalAbility(name);
                case SUB_ABILITY -> sa.getSubAbility();
                default -> null;
            };
        }

        /**
         * Queue the trait a reference names, once per label per chain.
         *
         * <p>Forge builds a fresh object for every reference it resolves, so
         * that graph is a tree; a granted trait is built here from its SVar's
         * name, and a grant naming itself or an earlier SVar would loop but
         * for the once-per-label rule, which also keeps a sub-ability two
         * branches share from being rendered twice. A label the script never
         * defines is one the accessor could not resolve: reported, and its
         * segment left out.
         */
        private void follow(String parameter, String label, CardTraitBase trait) {
            if (!firstReach(label)) return;
            if (trait == null) {
                if (report != null) {
                    report.add(cardName(), key, parameter, label);
                }
                return;
            }
            queue.addLast(new Pending(label, trait));
        }

        /** Whether {@code label} is a real label this chain reaches for the first time. */
        private boolean firstReach(String label) {
            boolean named = label != null && !label.isEmpty();
            return named && reached.add(label);
        }

        /**
         * Queue a trait no parameter names, under a label of its own: no
         * script holds it, and like every label it prints as its {@code SV}
         * number, so it only has to differ from every SVar name.
         */
        private void followUnnamed(String parameter, CardTraitBase trait) {
            unnamedLabels++;
            follow(parameter, "<unnamed " + unnamedLabels + ">", trait);
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
