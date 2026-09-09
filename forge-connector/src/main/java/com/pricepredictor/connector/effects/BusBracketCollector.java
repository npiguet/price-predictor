package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import com.google.common.eventbus.Subscribe;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import forge.game.card.CardView;
import forge.game.event.GameEventCardAttachment;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventCardCounters;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventCardStatsChanged;
import forge.game.event.GameEventCardTapped;
import forge.game.event.GameEventCombatEnded;
import forge.game.event.GameEventGameFinished;
import forge.game.event.GameEventScry;
import forge.game.event.GameEventSurveil;
import forge.game.event.GameEventPlayerDamaged;
import forge.game.event.GameEventPlayerLivesChanged;
import forge.game.event.GameEventPlayerPoisoned;
import forge.game.event.GameEventSpellAbilityCast;
import forge.game.event.GameEventSpellResolved;
import forge.game.event.GameEventTurnPhase;
import forge.game.spellability.SpellAbility;

import java.lang.reflect.InvocationHandler;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;

/**
 * Stage-one collection: the public event bus plus a bracket around resolution.
 *
 * <p>No patched Forge. Events arriving on the bus are attributed to whatever the
 * stack said was resolving when the bracket opened — correct for the common case
 * of one thing resolving at a time, and approximate where two resolve together.
 * Every record it writes is stamped {@code degraded} so a corpus never claims
 * more precision than it has.
 *
 * <p>The bracket is delimited by the bus itself: a cast opens one, a resolution
 * closes it. The resolving ability is read from {@code game.getStack()} rather
 * than from the event, because the event carries a view and the provenance key
 * needs the model object.
 *
 * <p>The cost half is <b>held</b> from the cast until the bracket closes, so it
 * can be stamped with what actually became of the spell. Every other record this
 * class writes goes out at the moment it describes; this one cannot, because
 * {@code outcome} is a field of it and a spell that is countered before it
 * resolves never publishes anything else the collector could attach the answer
 * to. A cast still on the stack when the game ends is dropped rather than given
 * one of the five outcomes it did not have — but it is now counted and named
 * rather than vanishing; see {@link #finishGame}.
 *
 * <p>Damage steps get their own brackets, closed at the phase boundary, so a
 * first-strike combat writes two records — the only way a keyword that acts by
 * splitting the step is visible in the corpus at all.
 */
public final class BusBracketCollector {

    private final Game game;
    private final RecordShardWriter writer;
    private final SnapshotBuilder snapshots;
    private final AttributionMode mode;
    private final String gameId;

    /** The ability the open bracket attributes to, or null between brackets. */
    private SpellAbility resolving;
    private ProvenanceKey.Resolved resolvingKeys =
            new ProvenanceKey.Resolved(null, ProvenanceKey.UNRESOLVED_UNKNOWN_KIND);
    private String resolvingActor;
    private String openBracketState;
    private final List<EffectEvent> bracketEvents = new ArrayList<>();
    private final Set<String> bracketEventKeys = new LinkedHashSet<>();

    /**
     * Casts whose outcome is not known yet, oldest first.
     *
     * <p>A list rather than a map because it is the stack: two spells can be
     * waiting at once and the inner one resolves first, so a lookup runs from
     * the newest end. Nothing here has been written yet — see
     * {@link PendingActivation} for why the record has to wait.
     */
    private final List<PendingActivation> pendingActivations = new ArrayList<>();

    /** Combat damage accumulates into its own bracket, closed at the phase end. */
    private final List<EffectEvent> combatEvents = new ArrayList<>();
    private final Set<String> combatEventKeys = new LinkedHashSet<>();
    /** The step's combat, read when its bracket opens rather than at flush. */
    private CombatShape combatShape = CombatShape.empty();
    /**
     * Computed characteristics, so a stats-changed event can say what changed.
     *
     * <p>Lives for the game rather than the bracket: a pump in one bracket and
     * its wearing off in another are both differences from what was there
     * before, and a per-bracket baseline would report the second as nothing.
     */
    private final StatDiffer stats = new StatDiffer();
    /**
     * Assignments already written, for the length of one combat.
     *
     * <p>Forge's assignment table is never cleared between the first-strike and
     * regular steps, so without this the regular step's record repeats first
     * strike's. Cleared when the combat ends, not when a step does.
     */
    private final Set<String> assignmentsSeen = new LinkedHashSet<>();
    private final Set<String> combatParticipants = new LinkedHashSet<>();
    private String combatSubstep;
    private String combatState;

    private long recordsWritten;

    /**
     * Whether the controller turned something down while this bracket was open.
     *
     * <p>Scoped to the bracket rather than matched against the declined
     * ability's id, because Forge resolves a copy of the ability it was handed
     * and an id comparison would silently never match — the same reason
     * {@link EventAttribution} counts a clause upward from the pointer instead
     * of downward from the root. What keeps the flag from over-claiming is the
     * second half of the test in {@link #endBracket}: a resolution that
     * produced any outcome at all is {@code resolved} however many optional
     * clauses inside it were declined.
     */
    private boolean bracketDeclined;

    /** Casts still on the stack when the game ended, over the whole game. */
    private long abandonedActivations;

    /** Told each combat record's id, for a probe that mirrors it. */
    private java.util.function.Consumer<String> onCombatRecord;

    /**
     * @param caps the run's collection caps, whose tier vector every collector
     *             in the worker shares
     *
     * <p>The depth is the run's, not this collector's. Taken as an argument
     * because that is what makes it impossible to pick one here: a hardcoded
     * pair is how {@code combat} and {@code resolution} records came to carry
     * {@code tiers=[1,2]} while every other kind carried {@code [1,2,3]}, which
     * made the tier list say which collector wrote a record rather than what
     * the run collected.
     */
    public BusBracketCollector(
            Game game, RecordShardWriter writer, String gameId,
            PatchedCollectors.CollectionCaps caps) {
        this.game = game;
        this.writer = writer;
        this.gameId = gameId;
        this.snapshots = new SnapshotBuilder(game, caps.snapshotTierArray());
        this.mode = AttributionMode.detect();
        // Installed here and dropped at GameEventGameFinished, because this
        // collector has no other lifecycle: it is subscribed to the bus for
        // exactly one game and the bus tells it when that game is over. On an
        // unpatched checkout install() finds no hook and nothing changes.
        PatchHooks.install(
                PLAYER_CONTROLLER_AI, CONFIRM_LISTENER, confirmHandler());
    }

    /**
     * The same collector for a caller that does not hold the run's caps.
     *
     * <p>Reads the run-level {@code effect.*} properties itself rather than
     * choosing a depth, so the vector is still the run's however the collector
     * was built. It exists only so the caller can be changed separately; a
     * caller that already holds the caps — {@code GamePlayer} reads them once
     * per match — should pass them.
     */
    public BusBracketCollector(
            Game game, RecordShardWriter writer, String gameId) {
        this(game, writer, gameId,
                PatchedCollectors.CollectionCaps.fromSystemProperties());
    }

    /**
     * Be told the id of each {@code combat} record as it is written.
     *
     * <p>A damage-step probe forks before the step and cannot know what it
     * mirrors until the real record exists, so the pairing is handed over here
     * rather than reconstructed. Reconstructing it from board state would not
     * work: two combats in one turn can look identical, and a wrong pairing
     * makes the difference between them meaningless.
     */
    public void onCombatRecord(java.util.function.Consumer<String> listener) {
        this.onCombatRecord = listener;
    }

    public long recordsWritten() {
        return recordsWritten;
    }

    /** The depth this collector's snapshots are built at — the run's, not its own. */
    int[] snapshotTiers() {
        return snapshots.tiers();
    }

    // ── brackets ────────────────────────────────────────────────────────

    /**
     * A cast opens a bracket and holds the cost half until its outcome is known.
     *
     * <p>The costs are read <b>now</b> — {@link EffectRecord#costsJson} says why
     * they are unreadable a moment later — but the record cannot be written yet,
     * because {@code outcome} is a field of the cost half and the answer to it
     * does not exist until the spell resolves or leaves the stack without
     * resolving. Writing it here is why every one of the corpus's 4,989
     * activation records said the literal {@code resolved}, and why 6.1% of
     * link halves had no partner: a fizzled spell issued a {@code link_id} whose
     * effect half never came.
     */
    @Subscribe
    public void onCast(GameEventSpellAbilityCast event) {
        // Read through peek() rather than peekAbility(), which dereferences the
        // top of an empty stack. The bus is not a promise about the stack.
        var top = game.getStack().peek();
        SpellAbility ability = top == null ? null : top.getSpellAbility();
        // A spell taken off the stack without resolving publishes nothing this
        // collector can tell from a resolution, so the absence is noticed
        // whenever the stack is next observed. A new cast is one such moment.
        reconcilePending();
        beginBracket(ability);
    }

    /**
     * Everything a cast does once the ability has been found on the stack.
     *
     * <p>Split from {@link #onCast} at the one line that reads the stack, so the
     * rules that decide what a cost record eventually says can be exercised
     * against a real ability without a game in progress — which is exactly what
     * they were missing while the outcome was a literal.
     */
    void beginBracket(SpellAbility ability) {
        openBracket(ability);
        if (ability == null) {
            return;
        }
        pendingActivations.add(new PendingActivation(
                ability.getId(), writer.nextRecordId(),
                RecordShardWriter.timestamp(), EffectRecord.costsJson(ability),
                actorOf(ability), keysOf(ability), openBracketState,
                gameId + ".link." + ability.getId(), targetCount(ability)));
    }

    /**
     * A resolution closes the bracket, stamps the cost half and writes the
     * effect half.
     *
     * <p>A fizzle writes no effect half at all. {@code MagicStack.resolveStack}
     * skips resolution entirely when the targets are gone, so the alternative is
     * a record claiming an empty event list for something that never ran — and
     * a partnerless {@code link_id} on the cost half, which the schema bars.
     *
     * <p>A decline writes no effect half either, for the same reason and by the
     * same rule: {@code declined} is one of the three outcomes the schema calls
     * partnerless, and the Python loader rejects a {@code declined} activation
     * that reaches resolution.
     */
    @Subscribe
    public void onResolved(GameEventSpellResolved event) {
        endBracket(
                event.spell() == null ? -1 : event.spell().getId(),
                event.hasFizzled());
        // The counterspell that removed something has just resolved, so this is
        // the first moment its victim's absence from the stack is visible.
        reconcilePending();
    }

    /**
     * The half of {@link #onResolved} that does not read the stack.
     *
     * @param abilityId the ability the engine says has finished
     * @param fizzled   whether it was removed for having no legal target left,
     *                  in which case it never ran at all
     */
    void endBracket(int abilityId, boolean fizzled) {
        // The bracket is a single slot, so a spell cast in response to another
        // takes it over: when the outer one finally resolves the events that
        // belong to it were never gathered. The cost half is still stamped with
        // its real outcome; what it must not do is issue a link to an effect
        // half nobody is going to write.
        boolean bracketIsThisAbility = resolving != null && resolving.getId() == abilityId;
        boolean declined = bracketIsThisAbility && !fizzled && wasDeclined();
        boolean writesEffectHalf = bracketIsThisAbility && !fizzled && !declined;

        String link = settleActivation(
                abilityId,
                fizzled
                        ? EffectRecord.OUTCOME_FIZZLED
                        : declined
                                ? EffectRecord.OUTCOME_DECLINED
                                : outcomeOf(heldTargetsAtCast(abilityId),
                                        bracketIsThisAbility ? resolving : null),
                writesEffectHalf);

        if (writesEffectHalf) {
            emit(new EffectRecord(
                    writer.nextRecordId(), writer.runId(), RecordShardWriter.timestamp(),
                    gameId, EffectRecord.KIND_RESOLUTION, mode)
                    .moment(EffectRecord.MOMENT_RESOLUTION)
                    .actor(resolvingActor)
                    .ability(resolvingKeys)
                    .linkId(link)
                    // Modes, X and the named card are set while the ability
                    // resolves, so the block that carries them is re-read here
                    // and spliced into the board captured at the cast.
                    .state(SnapshotBuilder.spliceRefs(
                            openBracketState, snapshots.refsJson(resolving)))
                    .payload(EffectRecord.eventsPayload(bracketEvents)));
        }

        if (bracketIsThisAbility) {
            closeBracket();
        } else if (resolving == null) {
            bracketEvents.clear();
            bracketEventKeys.clear();
        }
    }

    /**
     * A phase change closes the combat bracket, if one is open.
     *
     * <p>One record per damage step: first strike and regular damage are
     * separate phases, so the phase boundary is exactly the step boundary.
     */
    @Subscribe
    public void onPhase(GameEventTurnPhase event) {
        flushCombat();
        // An effect that empties the stack -- ending the turn, ending combat --
        // resolves nothing afterwards, so a phase boundary is the last chance
        // to notice what it swept away.
        reconcilePending();
    }

    @Subscribe
    public void onCombatEnded(GameEventCombatEnded event) {
        flushCombat();
        // The next combat's assignment table starts empty, so what this one
        // wrote must stop suppressing entries in it.
        assignmentsSeen.clear();
    }

    // ── outcome events ──────────────────────────────────────────────────

    /**
     * One damage, and the permanent that dealt it.
     *
     * <p>The cause comes off the event rather than off the bracket, because in
     * a damage step there is no bracket: nothing is resolving, and the
     * attacking creature is the only answer to "what caused this". It is also
     * what makes two attackers each dealing 1 damage to the same blocker two
     * distinguishable events instead of one line rendered twice — 2,193 of the
     * smoke corpus's 3,780 duplicated events are exactly that pair.
     */
    @Subscribe
    public void onCardDamaged(GameEventCardDamaged event) {
        String subject = "E" + event.card().getId();
        EffectEvent damage = EventAttribution.stamp(
                BusEvents.cardDamaged(event, isCombatDamage()), resolving,
                BusEvents.causeOf(event));
        if (isCombatDamage()) {
            openCombatBracket();
            combatParticipants.add(subject);
            combatEvents.add(damage);
        } else {
            bracketEvents.add(damage);
        }
    }

    @Subscribe
    public void onPlayerDamaged(GameEventPlayerDamaged event) {
        String subject = "P" + event.target().getId();
        EffectEvent damage = EventAttribution.stamp(
                BusEvents.playerDamaged(event), resolving,
                BusEvents.causeOf(event));
        if (event.combat()) {
            openCombatBracket();
            combatParticipants.add(subject);
            combatEvents.add(damage);
        } else {
            bracketEvents.add(damage);
        }
    }

    @Subscribe
    public void onLifeChanged(GameEventPlayerLivesChanged event) {
        record(BusEvents.lifeChanged(event));
    }

    @Subscribe
    public void onPoisoned(GameEventPlayerPoisoned event) {
        record(BusEvents.poisoned(event), BusEvents.causeOf(event));
    }

    @Subscribe
    public void onCounters(GameEventCardCounters event) {
        record(BusEvents.counters(event));
    }

    @Subscribe
    public void onTapped(GameEventCardTapped event) {
        record(BusEvents.tapped(event));
    }

    /**
     * A card moving, and the draw, discard or mill it may also be.
     *
     * <p>The workhorse outcome, and it is read here rather than off
     * {@code GameEventZone} because only this event is a <b>move</b>: the
     * per-zone-list notification fires once for the zone left, once for the zone
     * reached and once more for the stack, which is where the 13.68% of events
     * that repeated another event in the same record came from, and it cannot
     * say where a card came from at all.
     *
     * <p>Draws, discards and mills have no event of their own: each is a card
     * moving between two zones and which of the three it is depends on the pair.
     * The head counts all three per player and none of those counters had ever
     * fired.
     */
    @Subscribe
    public void onCardChangeZone(GameEventCardChangeZone event) {
        EffectEvent moved = BusEvents.cardMoved(event);
        if (moved != null) {
            record(moved);
        }
        EffectEvent named = BusEvents.libraryMovement(event);
        if (named != null) {
            record(named);
        }
        if (event.card() != null && event.from() != null
                && event.from().zoneType() == ZoneType.Battlefield) {
            // Off the battlefield a card's computed characteristics stop
            // meaning anything, and one that returns is a new object.
            stats.forget(event.card().getId());
        }
    }

    /**
     * The engine says a card's stats moved; this says what moved.
     *
     * <p>The bus publishes one event for a pump, an anthem recompute, an
     * animation and a colour change alike, and it names only the card. Without
     * the difference, the whole "gets +2/+2 and gains flying" family — most of
     * what a limited deck does — resolved into an empty event list.
     */
    @Subscribe
    public void onStatsChanged(GameEventCardStatsChanged event) {
        if (event.cards() == null) {
            return;
        }
        for (CardView view : event.cards()) {
            if (view == null) {
                continue;
            }
            Card card = cardById(view.getId());
            if (card != null) {
                for (EffectEvent changed : stats.diff(card)) {
                    record(changed);
                }
            }
        }
    }

    @Subscribe
    public void onAttachment(GameEventCardAttachment event) {
        EffectEvent attached = BusEvents.attachment(event);
        if (attached != null) {
            record(attached);
        }
    }

    @Subscribe
    public void onScry(GameEventScry event) {
        for (EffectEvent looked : BusEvents.scry(event)) {
            record(looked);
        }
    }

    @Subscribe
    public void onSurveil(GameEventSurveil event) {
        for (EffectEvent looked : BusEvents.surveil(event)) {
            record(looked);
        }
    }

    /**
     * The battlefield card an id names, or null.
     *
     * <p>The bus hands out views rather than cards, and a view carries the id
     * and not the object. The battlefield is small enough that a scan costs
     * less than an index kept in step with every zone change.
     */
    private Card cardById(int id) {
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            if (card.getId() == id) {
                return card;
            }
        }
        return null;
    }

    // ── plumbing ────────────────────────────────────────────────────────

    /**
     * Route an event into whichever bracket is open.
     *
     * <p>State-based-action deaths arrive with no ability resolving; they
     * attribute to the bracket they follow, which is the one that caused them.
     */
    private void record(EffectEvent event) {
        record(event, null);
    }

    /**
     * The same, for an event the bus itself named a causer for.
     *
     * @param namedCause the ref the engine named, which outranks the bracket's,
     *                   or null to take the bracket's
     */
    private void record(EffectEvent event, String namedCause) {
        // Stamped here rather than where the event is built: which clause
        // produced it, how long it lasts and — absent a name off the event
        // itself — what caused it are properties of the bracket it landed in,
        // and only this side knows that.
        EventAttribution.stamp(event, resolving, namedCause);
        if (isCombatDamage()) {
            openCombatBracket();
            if (fileEvent(event, combatEvents, combatEventKeys)) {
                combatParticipants.addAll(event.subjects());
            }
        } else {
            fileEvent(event, bracketEvents, bracketEventKeys);
        }
    }

    /**
     * Outcomes a second identical report of cannot mean a second occurrence.
     *
     * <p>These describe a transition into a state a thing is either in or not:
     * a card is on the battlefield, a permanent is tapped, an aura is attached.
     * Told twice in one record, the second telling is the engine publishing the
     * same change again — nothing in the record can distinguish it from the
     * first, and nothing downstream can use it.
     *
     * <p>Everything else is deliberately absent, because for a quantitative
     * outcome the repeat <b>is</b> the information: two creatures each dealing
     * one damage to the same blocker render two identical {@code damage_dealt}
     * events, and collapsing them would turn two damage into one.
     */
    private static final Set<String> IDEMPOTENT_EVENTS = Set.of(
            EffectEvent.ZONE_CHANGE, EffectEvent.TAPPED, EffectEvent.UNTAPPED,
            EffectEvent.ATTACHED, EffectEvent.UNATTACHED, EffectEvent.PHASED,
            EffectEvent.FACE_CHANGE, EffectEvent.DESTROYED,
            EffectEvent.SACRIFICED, EffectEvent.REGENERATED);

    /**
     * Add an event to a bracket unless it repeats one already there.
     *
     * @return whether it was added
     */
    static boolean fileEvent(
            EffectEvent event, List<EffectEvent> into, Set<String> seen) {
        if (!IDEMPOTENT_EVENTS.contains(event.type())) {
            into.add(event);
            return true;
        }
        // Compared as the rendered line, so two events are the same when what
        // the record says about them is the same -- subjects, params, duration
        // and the clause they were attributed to, all of it.
        if (!seen.add(event.toJson())) {
            return false;
        }
        into.add(event);
        return true;
    }

    private void openBracket(SpellAbility ability) {
        this.resolving = ability;
        this.bracketDeclined = false;
        this.resolvingKeys = keysOf(ability);
        this.resolvingActor = actorOf(ability);
        this.openBracketState = snapshots.toJson(ability, referencedOf(ability));
        this.bracketEvents.clear();
        this.bracketEventKeys.clear();
    }

    private void closeBracket() {
        this.resolving = null;
        this.bracketDeclined = false;
        this.resolvingKeys =
                new ProvenanceKey.Resolved(null, ProvenanceKey.UNRESOLVED_UNKNOWN_KIND);
        this.resolvingActor = null;
        this.openBracketState = null;
        this.bracketEvents.clear();
        this.bracketEventKeys.clear();
    }

    private static String actorOf(SpellAbility ability) {
        return ability == null || ability.getActivatingPlayer() == null
                ? null
                : SnapshotBuilder.playerId(ability.getActivatingPlayer());
    }

    // ── the outcome of a cast ───────────────────────────────────────────

    /**
     * A cast that has been observed but whose outcome is not settled.
     *
     * <p>Holds the rendered strings rather than the {@link SpellAbility},
     * because what was paid lives on the ability object and its next activation
     * clears it — the reason {@link EffectRecord#costsJson} must be called at
     * the cast and nowhere later. The record id and the timestamp are taken at
     * the cast too, so a held record still says when the spell was cast rather
     * than when it stopped being on the stack.
     */
    private record PendingActivation(
            int abilityId,
            String recordId,
            String timestamp,
            String costsJson,
            String actor,
            ProvenanceKey.Resolved ability,
            String state,
            String linkId,
            int targetsAtCast) {
    }

    /** Take the held cast for an ability id, newest first, or null. */
    private PendingActivation takePending(int abilityId) {
        for (int i = pendingActivations.size() - 1; i >= 0; i--) {
            if (pendingActivations.get(i).abilityId() == abilityId) {
                return pendingActivations.remove(i);
            }
        }
        return null;
    }

    /** How many targets the held cast for an ability had when it was cast. */
    private int heldTargetsAtCast(int abilityId) {
        for (int i = pendingActivations.size() - 1; i >= 0; i--) {
            if (pendingActivations.get(i).abilityId() == abilityId) {
                return pendingActivations.get(i).targetsAtCast();
            }
        }
        return 0;
    }

    /**
     * Write a held cost half now that its outcome is known.
     *
     * <p>{@code linked} is separate from the outcome because they answer
     * different questions: the outcome says what became of the spell, and
     * {@code link_id} promises that an effect half carrying the same id exists.
     * A resolution this collector could not describe -- the outer half of a
     * nested pair -- is a {@code resolved} activation with no link, which is
     * honest; issuing the link anyway is what left 306 halves dangling.
     *
     * @return the link the effect half may claim, or null when there is to be
     *         no effect half or nothing was held for this ability
     */
    String settleActivation(int abilityId, String outcome, boolean linked) {
        PendingActivation activation = takePending(abilityId);
        if (activation == null) {
            return null;
        }
        emitActivation(activation, outcome, linked);
        return linked ? activation.linkId() : null;
    }

    private void emitActivation(
            PendingActivation activation, String outcome, boolean linked) {
        emit(new EffectRecord(
                activation.recordId(), writer.runId(), activation.timestamp(),
                gameId, EffectRecord.KIND_RESOLUTION, mode)
                .moment(EffectRecord.MOMENT_ACTIVATION)
                .actor(activation.actor())
                .ability(activation.ability())
                .linkId(linked ? activation.linkId() : null)
                .state(activation.state())
                .payload(EffectRecord.costPayload(activation.costsJson(), outcome)));
    }

    /**
     * Whether a resolution lost some of its targets on the way.
     *
     * <p>{@code MagicStack.hasFizzled} strips the targets that became illegal
     * from the ability before resolving it, and only then does the ability
     * resolve — so comparing the count with the one read at the cast is the
     * engine's own answer to "did part of this spell do nothing". Nothing else
     * in the record distinguishes a Lightning Helix that hit from one whose
     * creature had already died.
     *
     * <p>Only the root line's targets are compared. A partial fizzle confined to
     * a sub-ability reads as {@code resolved}, which understates rather than
     * invents. Losing <b>every</b> target without the engine calling it a fizzle
     * is the {@code CantFizzle} case (Gilded Drake), where the spell still does
     * what it does: that is {@code resolved} too.
     */
    static String outcomeOf(int targetsAtCast, SpellAbility resolved) {
        if (resolved == null) {
            return EffectRecord.OUTCOME_RESOLVED;
        }
        int surviving = targetCount(resolved);
        return surviving > 0 && surviving < targetsAtCast
                ? EffectRecord.OUTCOME_PARTIALLY_FIZZLED
                : EffectRecord.OUTCOME_RESOLVED;
    }

    private static int targetCount(SpellAbility ability) {
        return ability == null || ability.getTargets() == null
                ? 0 : ability.getTargets().size();
    }

    /**
     * Write off every held cast that has left the stack without resolving.
     *
     * <p>Counterspells, {@code CostExileFromStack}, a spell bounced off the
     * stack and {@code MagicStack.clear} all take an entry away with no event
     * this collector can tell apart from a resolution -- {@code finishResolving}
     * publishes the same removal for a spell that resolved normally. So the
     * question is asked of the stack itself, at every moment the stack can have
     * changed: a held cast that is no longer on it and did not resolve was
     * removed, and {@code countered} is what the schema calls that.
     *
     * <p>It is the nearest of the five outcomes rather than an exact one. A
     * spell exiled off the stack by a cost was not countered in the rules sense;
     * what the corpus needs to know, and what all of these share, is that the
     * cost was paid and no effect followed.
     */
    private void reconcilePending() {
        if (pendingActivations.isEmpty()) {
            return;
        }
        writeOffRemoved(idsOnStack());
    }

    /** The ability ids the stack holds right now. */
    private Set<Integer> idsOnStack() {
        Set<Integer> onStack = new LinkedHashSet<>();
        for (var instance : game.getStack()) {
            SpellAbility sa = instance == null ? null : instance.getSpellAbility();
            if (sa != null) {
                onStack.add(sa.getId());
            }
        }
        return onStack;
    }

    /**
     * The half of {@link #reconcilePending} that has already read the stack.
     *
     * @param onStack the ability ids still on the stack
     * @return how many held casts were written off
     */
    int writeOffRemoved(Set<Integer> onStack) {
        int removed = 0;
        for (Iterator<PendingActivation> held = pendingActivations.iterator();
                held.hasNext();) {
            PendingActivation activation = held.next();
            if (!onStack.contains(activation.abilityId())) {
                held.remove();
                emitActivation(activation, EffectRecord.OUTCOME_COUNTERED, false);
                removed++;
            }
        }
        return removed;
    }

    /**
     * Casts still waiting for an outcome, for a test that has to see the hold.
     *
     * <p>At the end of a game whatever is left here is dropped: a spell on the
     * stack when the last player lost neither resolved nor was removed, and
     * {@code outcome} has no member for it. Writing one of the five would be a
     * claim about a game that stopped, so the count is reported instead — by
     * {@link #abandonedActivations()}, which is the running total of them.
     */
    long unresolvedActivations() {
        return pendingActivations.size();
    }

    // ── the end of the game ─────────────────────────────────────────────

    /**
     * The last thing that happens to a game, and the last chance to write.
     *
     * <p>Nothing subscribed this event, and two channels were being lost at it.
     *
     * <p>The <b>combat bracket</b> is closed at a phase boundary, and a game
     * that ends in combat damage never reaches one:
     * {@code PhaseHandler.mainLoopStep} returns the moment
     * {@code checkStateBasedEffects} says the game is over, so neither
     * {@code GameEventTurnPhase} nor {@code GameEventCombatEnded} follows the
     * lethal damage step. The record for the combat that decided the game was
     * therefore the one combat record no game had.
     *
     * <p>The <b>held casts</b> split in two, and only one half is writable.
     * A spell that left the stack since the last reconcile point — countered,
     * exiled off the stack, swept away by the effect that ended the game — is
     * written off exactly as it would be at any other moment. What is genuinely
     * still on the stack is not: it neither resolved nor was removed, and
     * {@code outcome} has no member for that. Those are counted and named on
     * stderr rather than stamped with one of the five they did not have.
     */
    @Subscribe
    public void onGameFinished(GameEventGameFinished event) {
        finishGame(idsOnStack());
    }

    /**
     * The half of {@link #onGameFinished} that has already read the stack.
     *
     * @param onStack the ability ids the stack still holds
     * @return how many held casts were abandoned rather than written
     */
    long finishGame(Set<Integer> onStack) {
        flushCombat();
        writeOffRemoved(onStack);
        long abandoned = pendingActivations.size();
        if (abandoned > 0) {
            StringJoiner names = new StringJoiner(", ");
            for (PendingActivation activation : pendingActivations) {
                names.add(activation.linkId());
            }
            System.err.println(
                    "effect records: " + gameId + " ended with " + abandoned
                            + " cast(s) still on the stack, dropped because"
                            + " outcome has no member for a game that stopped: "
                            + names);
        }
        pendingActivations.clear();
        abandonedActivations += abandoned;
        // The listener is static and this collector lives for one game, so the
        // game ending is where it has to go.
        PatchHooks.uninstall(PLAYER_CONTROLLER_AI, CONFIRM_LISTENER);
        return abandoned;
    }

    /** How many casts this game abandoned on the stack, over the whole game. */
    long abandonedActivations() {
        return abandonedActivations;
    }

    // ── an offer the controller turned down ─────────────────────────────

    /** The patched class that answers for the AI, and the setter it exposes. */
    private static final String PLAYER_CONTROLLER_AI = "forge.ai.PlayerControllerAi";
    private static final String CONFIRM_LISTENER = "setEffectRecordConfirmListener";

    /**
     * Notice that the controller said no to something.
     *
     * <p>{@code declined} is the one outcome nothing ever wrote, and there is
     * no bus event for it: an optional effect nobody took produces no zone
     * change, no damage and no life total to notice, which is precisely why the
     * schema wanted it — the counterfactual "it was offered and refused" is
     * otherwise indistinguishable from "it was never offered".
     *
     * <p>Only the refusal is kept. The hook is called with both answers, the
     * way the trigger-condition hook is, but an accepted offer is already
     * described by the events it produced.
     *
     * <p>Package-visible so a test can drive it through a proxy of its own. The
     * method name and the argument positions are the whole contract with the
     * patch — {@link PatchHooks} looks both up by string — and a typo in either
     * degrades this channel silently.
     */
    InvocationHandler confirmHandler() {
        return (proxy, method, args) -> {
            if (!"onConfirm".equals(method.getName()) || args == null
                    || args.length < 2) {
                return null;
            }
            if (Boolean.FALSE.equals(args[1])) {
                noteDeclined();
            }
            return null;
        };
    }

    /** Record a refusal against the open bracket, if there is one. */
    void noteDeclined() {
        if (resolving != null) {
            bracketDeclined = true;
        }
    }

    /**
     * Whether the open bracket is a refusal rather than a resolution.
     *
     * <p>Both halves are required. A refusal alone is not enough: a line whose
     * mandatory clause drew a card and whose optional clause was declined did
     * something, and {@code resolved} is the honest word for it. An empty event
     * list alone is not enough either — plenty of resolutions produce nothing
     * this collector can observe. Together they are the shape the schema
     * describes: offered, turned down, and nothing happened.
     */
    private boolean wasDeclined() {
        return bracketDeclined && bracketEvents.isEmpty();
    }

    /**
     * The acting line, or the reason there is none.
     *
     * <p>Resolved rather than keyed, because a record whose {@code ability} is
     * empty and silent cannot be told apart from one the resolver failed on,
     * and this collector writes most of the corpus's resolution records.
     */
    private static ProvenanceKey.Resolved keysOf(SpellAbility ability) {
        if (ability == null) {
            return new ProvenanceKey.Resolved(null, ProvenanceKey.UNRESOLVED_UNKNOWN_KIND);
        }
        return ProvenanceKey.resolve(ability);
    }

    private static List<Card> referencedOf(SpellAbility ability) {
        List<Card> referenced = new ArrayList<>();
        if (ability == null) {
            return referenced;
        }
        if (ability.getHostCard() != null) {
            referenced.add(ability.getHostCard());
        }
        if (ability.getTargets() != null) {
            for (Object target : ability.getTargets()) {
                if (target instanceof Card card) {
                    referenced.add(card);
                }
            }
        }
        return referenced;
    }

    private boolean isCombatDamage() {
        var phase = game.getPhaseHandler();
        if (phase == null || phase.getPhase() == null) {
            return false;
        }
        return phase.getPhase().toString().contains("COMBAT_DAMAGE");
    }

    /**
     * Start a damage step's bracket, and capture the combat it is about.
     *
     * <p>The structure is read here rather than at the flush, because the flush
     * happens at the phase boundary — after the damage has been dealt and the
     * creatures it killed have left. Read there, an attacker that traded with
     * its blocker is in neither list, and the record describes a combat nobody
     * fought. Read here it matches the snapshot beside it, which is taken in
     * the same call.
     */
    private void openCombatBracket() {
        if (combatState != null) {
            return;
        }
        var phase = game.getPhaseHandler();
        combatSubstep = phase != null && phase.getPhase() != null
                && phase.getPhase().toString().contains("FIRST_STRIKE")
                ? "first_strike" : "regular";
        combatState = snapshots.toJson(null, List.of());
        // The damage has been assigned by now -- the first damage event is what
        // opened this bracket -- but nothing has died yet, so this is the one
        // moment both halves of the shape are readable.
        combatShape = CombatShape.of(game.getCombat());
        combatShape.addAssignment(game.getCombat(), assignmentsSeen);
    }

    /**
     * Write the open combat bracket, if any, as one {@code combat} record.
     *
     * <p>{@code ability} is absent: no single line acts in a damage step, and
     * inventing one would attribute the whole step to whichever creature
     * happened to be first.
     */
    private void flushCombat() {
        if (combatState == null) {
            return;
        }
        var phase = game.getPhaseHandler();
        String actor = phase == null || phase.getPlayerTurn() == null
                ? null : SnapshotBuilder.playerId(phase.getPlayerTurn());
        String recordId = writer.nextRecordId();
        emit(new EffectRecord(
                recordId, writer.runId(), RecordShardWriter.timestamp(),
                gameId, EffectRecord.KIND_COMBAT, mode)
                .actor(actor)
                .state(combatState)
                .payload(combatPayload()));
        // A damage-step probe forked before this step and has been holding its
        // branch since: it needs this record's id to say which combat it
        // mirrors, and the id does not exist until the record is written.
        if (onCombatRecord != null) {
            onCombatRecord.accept(recordId);
        }
        combatEvents.clear();
        combatEventKeys.clear();
        combatParticipants.clear();
        combatShape = CombatShape.empty();
        combatState = null;
        combatSubstep = null;
    }

    private String combatPayload() {
        StringJoiner events = new StringJoiner(",", "[", "]");
        for (EffectEvent event : combatEvents) {
            events.add(event.toJson());
        }
        return "{" + combatShape.fields() + ",\"events\":" + events + "}";
    }

    private void emit(EffectRecord record) {
        writer.write(record.toJson());
        recordsWritten++;
    }
}
