package com.pricepredictor.connector.effects;

import forge.game.CardTraitBase;
import forge.game.card.Card;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Changes to a card's computed characteristics, as events.
 *
 * <p>Forge's bus publishes what an effect <i>did</i> only for the outcomes that
 * have their own event: damage, counters, zone changes, taps. A pump, an anthem
 * recompute, an animation and a colour change all publish the same thing — the
 * card's stats changed — and say nothing about what changed. Read off the bus
 * alone, the whole "gets +2/+2 and gains flying" family, which is most of what
 * a limited deck does, would produce records whose event list is empty or
 * nearly so.
 *
 * <p>So this remembers each card's computed characteristics and describes
 * the difference when the engine says they moved. Computed rather
 * than printed, for the reason every other characteristic in a snapshot is: the
 * model has to predict what happens to the 5/5 an anthem made, not to the 4/4
 * on the card.
 *
 * <p>The remembered value updates on every diff, so two effects in one bracket
 * each report their own share rather than the second reporting both. It also
 * updates whenever a record that collects events takes its snapshot
 * ({@link #rebaseline}): Forge can apply a change well before it publishes
 * it — a state-based check applies the statics on every pass and announces
 * the cards they moved once, at its end — so a change made before a
 * snapshot, and therefore already shown in it, would otherwise be reported
 * again as an event of that record.
 *
 * <p>Keywords, hidden keywords and abilities are multisets, kept sorted: a
 * second instance of a keyword (a second Annihilator, a second Bushido) is a
 * change, and every list an event carries is in one order on every JVM. The
 * face is not read here: {@link BusEventReader} reads it beside these
 * changes and writes its {@code face_change} once.
 */
final class StatDiffer {

    /**
     * One card's computed characteristics, as of the last time it was read.
     *
     * @param keywords       its keywords, lower-cased, sorted, repeats kept
     * @param hiddenKeywords its hidden extrinsic keywords as Forge spells
     *                       them, sorted, repeats kept
     * @param abilities      one signature per ability that is no keyword's
     *                       (a spell, activated or mana ability, trigger,
     *                       static or replacement), sorted, repeats kept
     */
    private record Stats(
            int power, int toughness, int basePower, int baseToughness, List<String> keywords,
            List<String> hiddenKeywords, List<String> abilities,
            Set<String> types, Set<String> colors,
            String name, String manaCost, int manaValue, FaceState face, int copies) {
    }

    private final Map<Integer, Stats> last = new HashMap<>();

    /** Each card's controller as last read, kept beside its stats: {@link #controlChange} reads it alone. */
    private final Map<Integer, String> controllers = new HashMap<>();

    private final TraitSignatures signatures = new TraitSignatures();

    /**
     * What changed on this card since it was last read, as events.
     *
     * <p>A card seen for the first time reports nothing: there is no earlier
     * value to have moved from, and reporting its whole characteristic set as a
     * change would make every card that enters look like it was animated.
     */
    List<EffectEvent> diff(Card card) {
        if (card == null) {
            return List.of();
        }
        Stats now = read(card);
        Stats before = last.put(card.getId(), now);
        EffectEvent control = controlChange(card);
        if (before == null || before.equals(now)) {
            return control == null ? List.of() : List.of(control);
        }
        return changes(card, new Move<>(before, now), control);
    }

    /** A value as last read and as read now. */
    private record Move<T>(T before, T now) {
    }

    /** What a set-valued change is called: its event and the two parameters naming what joined and what left. */
    private record SetChange(EventType type, EventParam added, EventParam removed) {
    }

    /** Every event between two readings of a card that differ, in the order its fields are read. */
    private List<EffectEvent> changes(Card card, Move<Stats> stats, EffectEvent control) {
        Stats before = stats.before();
        Stats now = stats.now();
        String subject = Refs.of(card);
        List<EffectEvent> events = new ArrayList<>();

        EffectEvent pt = ptChange(card, before, now);
        if (pt != null) {
            events.add(pt);
        }
        addKeywordEvents(events, subject, new Move<>(before.keywords(), now.keywords()), false);
        // "Can't block", "doesn't untap during its controller's next untap
        // step": Forge's hidden keywords, which no keyword list holds.
        addKeywordEvents(events, subject, new Move<>(before.hiddenKeywords(), now.hiddenKeywords()), true);
        addAbilityEvents(events, subject, card, new Move<>(before.abilities(), now.abilities()));
        addSetChangeEvent(events, subject,
                new SetChange(EventType.TYPE_CHANGE, EventParam.TYPES_ADDED, EventParam.TYPES_REMOVED),
                new Move<>(before.types(), now.types()));
        // colors_removed is this side's addition to the type's normalization:
        // the head has a colors_lost field and the event carried no way to say
        // a colour went away.
        addSetChangeEvent(events, subject,
                new SetChange(EventType.COLOR_CHANGE, EventParam.COLORS, EventParam.COLORS_REMOVED),
                new Move<>(before.colors(), now.colors()));
        if (control != null) {
            events.add(control);
        }
        addNameAndCostEvents(events, card, stats);
        return events;
    }

    /**
     * A name or mana cost a face change or a copy effect brought is that
     * change's: the face_change and the copy say it already.
     */
    private static void addNameAndCostEvents(List<EffectEvent> events, Card card, Move<Stats> stats) {
        Stats before = stats.before();
        Stats now = stats.now();
        String subject = Refs.of(card);
        boolean ownChange = before.face().equals(now.face()) && before.copies() == now.copies();
        if (ownChange && !java.util.Objects.equals(before.name(), now.name())) {
            events.add(new EffectEvent(EventType.NAME_CHANGE)
                    .subject(subject)
                    .param(EventParam.NAME, now.name()));
        }
        if (ownChange && !java.util.Objects.equals(before.manaCost(), now.manaCost())) {
            events.add(new EffectEvent(EventType.MANA_COST_CHANGE)
                    .subject(subject)
                    .param(EventParam.MANA_COST, CardFacts.manaCostSymbols(card.getManaCost()))
                    .param(EventParam.MANA_VALUE, now.manaValue()));
        }
    }

    /** The {@code pt_change} between two readings of a card, or null when its power and toughness did not move. */
    private static EffectEvent ptChange(Card card, Stats before, Stats now) {
        boolean setsBase = card.isCreature() && (before.basePower() != now.basePower()
                || before.baseToughness() != now.baseToughness());
        boolean moved = before.power() != now.power() || before.toughness() != now.toughness();
        if (!moved && !setsBase) {
            return null;
        }
        EffectEvent pt = new EffectEvent(EventType.PT_CHANGE)
                .subject(Refs.of(card))
                .param(EventParam.POWER_DELTA, now.power() - before.power())
                .param(EventParam.TOUGHNESS_DELTA, now.toughness() - before.toughness());
        // The base a setting effect gave it (layer 7b, or a CDA's 7a), as
        // the snapshot's base_pt spells it: without it a 0/0 artifact
        // animated into a 5/5 reads the same as a 0/0 creature pumped
        // +5/+5. Only on a creature, as base_pt is.
        if (setsBase) {
            pt.param(EventParam.SET_PT, List.of(now.basePower(), now.baseToughness()));
        }
        return pt;
    }

    private static EffectEvent controlChangeEvent(String subject, String controller) {
        return new EffectEvent(EventType.CONTROL_CHANGE).subject(subject).param(EventParam.CONTROLLER, controller);
    }

    /**
     * The change of control a card has had since it was last read, alone,
     * or null: nothing else it changed is read, and the reading it is
     * compared with keeps every other value, so a later {@link #diff} still
     * reports those. For a clause that only gains control, which publishes
     * no stats change.
     */
    EffectEvent controlChange(Card card) {
        if (card == null) {
            return null;
        }
        boolean seen = controllers.containsKey(card.getId());
        String now = Refs.ofNullable(card.getController());
        String before = controllers.put(card.getId(), now);
        return seen && !java.util.Objects.equals(before, now) ? controlChangeEvent(Refs.of(card), now) : null;
    }

    /**
     * Remember a card's characteristics without reporting a change.
     *
     * <p>Used when a card arrives on the battlefield, and for every card on a
     * forked board, so that its first real change is a change rather than
     * nothing at all: Forge publishes a stats change only after applying it.
     */
    void prime(Card card) {
        if (card != null) {
            last.put(card.getId(), read(card));
            controllers.put(card.getId(), Refs.ofNullable(card.getController()));
        }
    }

    /**
     * Take a reading of every card in {@code cards}, as of a snapshot just
     * taken, so that what changes from here is measured from what the
     * snapshot shows.
     *
     * @return the changes the readings swallowed: made, and shown in the
     *         snapshot, but not yet published, so no record reports them
     */
    List<EffectEvent> rebaseline(Iterable<Card> cards) {
        List<EffectEvent> unreported = new ArrayList<>();
        for (Card card : cards) {
            unreported.addAll(diff(card));
        }
        return unreported;
    }

    /** Forget a card, so a re-entering permanent is a new object. */
    void forget(int cardId) {
        last.remove(cardId);
        controllers.remove(cardId);
    }

    /**
     * One {@code keyword_change} for the keywords gained and one for those
     * lost, each list sorted; a hidden keyword's carry {@code hidden: true}
     * and Forge's own text ("CARDNAME can't block.").
     */
    private static void addKeywordEvents(
            List<EffectEvent> events, String subject, Move<List<String>> keywords, boolean hidden) {
        List<String> gained = difference(keywords.now(), keywords.before());
        List<String> lost = difference(keywords.before(), keywords.now());
        if (!gained.isEmpty()) {
            events.add(keywordEvent(subject, gained, false, hidden));
        }
        if (!lost.isEmpty()) {
            events.add(keywordEvent(subject, lost, true, hidden));
        }
    }

    private static EffectEvent keywordEvent(
            String subject, List<String> keywords, boolean removed, boolean hidden) {
        EffectEvent event = new EffectEvent(EventType.KEYWORD_CHANGE)
                .subject(subject)
                .param(EventParam.KEYWORDS, keywords)
                .param(EventParam.REMOVED, removed);
        if (hidden) {
            event.param(EventParam.HIDDEN, true);
        }
        return event;
    }

    /**
     * An {@code ability_change} naming the abilities gained, comma-joined by
     * their provenance keys ({@code ProvenanceKey.describe}, {@code runtime}
     * for one that keys to no line), and one with {@code removed: true} and
     * no names when any was lost — the shape the {@code Animate} clause rule
     * writes, read the same way: a count of what was granted, and whether
     * anything went.
     */
    private void addAbilityEvents(
            List<EffectEvent> events, String subject, Card card, Move<List<String>> abilities) {
        List<String> gained = difference(abilities.now(), abilities.before());
        List<String> lost = difference(abilities.before(), abilities.now());
        if (!gained.isEmpty()) {
            events.add(new EffectEvent(EventType.ABILITY_CHANGE)
                    .subject(subject)
                    .param(EventParam.ABILITIES, String.join(",", namesOf(card, gained)))
                    .param(EventParam.REMOVED, false));
        }
        if (!lost.isEmpty()) {
            events.add(new EffectEvent(EventType.ABILITY_CHANGE)
                    .subject(subject)
                    .param(EventParam.ABILITIES, "")
                    .param(EventParam.REMOVED, true));
        }
    }

    /** The gained abilities' names, in signature order. */
    private List<String> namesOf(Card card, List<String> gained) {
        Map<String, List<CardTraitBase> > bySignature = new HashMap<>();
        for (CardTraitBase trait : abilitiesOf(card)) {
            bySignature.computeIfAbsent(signatures.of(trait), s -> new ArrayList<>()).add(trait);
        }
        List<String> names = new ArrayList<>();
        for (String signature : gained) {
            List<CardTraitBase> traits = bySignature.get(signature);
            CardTraitBase trait = traits == null || traits.isEmpty() ? null : traits.remove(0);
            ProvenanceKey key = trait == null ? null : ProvenanceResolver.of(trait);
            names.add(key == null ? "runtime" : key.describe());
        }
        return names;
    }

    /** One event naming what joined and what left a set, or none when neither did. */
    private static void addSetChangeEvent(
            List<EffectEvent> events, String subject, SetChange kind, Move<Set<String>> values) {
        Set<String> added = new LinkedHashSet<>(values.now());
        added.removeAll(values.before());
        Set<String> removed = new LinkedHashSet<>(values.before());
        removed.removeAll(values.now());
        if (added.isEmpty() && removed.isEmpty()) {
            return;
        }
        events.add(new EffectEvent(kind.type())
                .subject(subject)
                .param(kind.added(), List.copyOf(added))
                .param(kind.removed(), List.copyOf(removed)));
    }

    /** {@code from} less {@code without}, as multisets: each repeat counts. */
    private static List<String> difference(List<String> from, List<String> without) {
        List<String> left = new ArrayList<>(from);
        for (String each : without) {
            left.remove(each);
        }
        return left;
    }

    /** The card's abilities that no keyword built: its lines other than keywords. */
    private static List<CardTraitBase> abilitiesOf(Card card) {
        List<CardTraitBase> abilities = new ArrayList<>();
        addOwn(abilities, card.getSpellAbilities());
        addOwn(abilities, card.getTriggers());
        addOwn(abilities, card.getStaticAbilities());
        addOwn(abilities, card.getReplacementEffects());
        return abilities;
    }

    private static void addOwn(List<CardTraitBase> into, Iterable<? extends CardTraitBase> traits) {
        for (CardTraitBase trait : traits) {
            if (trait.getKeyword() == null) {
                into.add(trait);
            }
        }
    }

    private static List<String> sorted(List<String> values) {
        List<String> sorted = new ArrayList<>(values);
        java.util.Collections.sort(sorted);
        return sorted;
    }

    private Stats read(Card card) {
        List<String> abilities = new ArrayList<>();
        for (CardTraitBase trait : abilitiesOf(card)) {
            abilities.add(signatures.of(trait));
        }
        Set<String> types = new LinkedHashSet<>(TypeLine.names(card.getType()));
        return new Stats(
                card.getNetPower(), card.getNetToughness(),
                card.getNetPowerBreakdown().currentValue,
                card.getNetToughnessBreakdown().currentValue,
                sorted(CardSpelling.keywordNames(card.getKeywords())),
                sorted(StaticSuppression.NONE.hiddenKeywords(card)),
                sorted(abilities), types,
                new LinkedHashSet<>(CardSpelling.colorLetters(card.getColor())),
                card.getName(), CardFacts.manaCostJson(card), card.getCMC(),
                CardFacts.faceState(card), card.getCloneStates().size());
    }
}
