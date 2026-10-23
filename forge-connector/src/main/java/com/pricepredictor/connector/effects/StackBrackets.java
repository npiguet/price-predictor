package com.pricepredictor.connector.effects;

import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.Supplier;

/**
 * The live collector's brackets of the stack items it saw cast, and which one
 * the bus files into now.
 *
 * <p>The brackets mirror Forge's stack: one per item, keyed by its ability's
 * id, which is the key {@code MagicStack} itself finds an item by
 * ({@code getInstanceMatchingSpellAbilityID}) and the id both the resolution
 * and the cost half carry.
 *
 * <p>An item's bracket outlives everything put on the stack above it. A cast
 * trigger — prowess, "whenever you cast", heroic, storm, cascade — and any
 * response go above the spell and resolve first; each writes its own halves
 * and is removed, and the spell's bracket is still there when the spell
 * resolves.
 *
 * <p>A cast made <b>while something is resolving</b> waits beside the stack
 * instead of going on top. Forge freezes the stack while an ability resolves
 * and parks what is added to it until the resolution ends; a spell a
 * {@code Play} effect casts goes onto it anyway — cascade, discover, madness,
 * Tetsuo's cast mode and every other "you may cast" are that effect — and so
 * does a copy marked {@code IgnoreFreeze}. Its
 * {@code GameEventSpellAbilityCast} is published in the middle of the
 * resolution; the resolution then finishes its remaining clauses and
 * publishes {@code GameEventSpellResolved}, and the cast spell resolves
 * afterwards. So the resolving item stays the top: what it does after the
 * cast is its own. When it resolves, what waited goes on top in the order it
 * was cast, where Forge put it, below anything the end of the resolution then
 * adds — a parked trigger, a parked copy. The board a waiting cast pairs with
 * is still the one taken at its cast.
 *
 * <p>The <b>open</b> bracket is the one the bus files into now, or none: from
 * a cast until the next action or resolution, the cast's own, and while
 * something resolves, the resolving item's ({@link #followResolving}). It
 * changes only through the transitions below; which events it takes is the
 * collector's cause to say. What is filed while none is open is kept as the
 * orphans.
 */
final class StackBrackets {

    /** Bottom first: the last is the top. */
    private final List<OpenResolution> items = new ArrayList<>();
    /** Cast while something was resolving, oldest first. */
    private final List<OpenResolution> waiting = new ArrayList<>();

    /** The bracket the bus files into now, or null when nothing is. */
    private OpenResolution open;

    /**
     * The ability whose resolution was the last published, while Forge may
     * still be finishing it, or null; see {@link #followResolving}.
     */
    private Integer finishing;

    /**
     * What the bus filed while no bracket was open: the aftermath of a
     * resolution, the state-based check before anything is cast again.
     *
     * <p>The collector rescues it if the game ends before a cast or a
     * resolution empties it again — the ordinary place of a win.
     */
    private final EventBuffer orphans = new EventBuffer();

    /** The open bracket, or null when none is open. */
    OpenResolution open() {
        return open;
    }

    /** The ability the open bracket attributes to, or null when none is open. */
    SpellAbility resolving() {
        return open == null ? null : open.ability;
    }

    /** Where an event filed now goes: the open bracket's buffer, or the orphans. */
    EventBuffer filing() {
        return open == null ? orphans : open.events;
    }

    /**
     * Whether the action in progress is the open bracket's own cast, the rest
     * of whose {@code MagicStack.add} runs after the cast is published. An
     * activation is pushed as a copy of the ability the action names.
     */
    boolean isOpenCast(SpellAbility acting) {
        return open != null
                && (open.ability == acting || open.ability.getOriginalAbility() == acting);
    }

    /** Whether anything was filed while no bracket was open, and is still kept. */
    boolean hasOrphans() {
        return !orphans.isEmpty();
    }

    /** What was filed while no bracket was open, no longer kept. */
    List<EffectEvent> drainOrphans() {
        return orphans.drain();
    }

    /**
     * A cast has gone on top of the stack: its bracket opens. The item it
     * went above keeps its own whole, with what it gathered since its cast —
     * the state-based check its cast set off, a creature it left without
     * toughness — which is that cast's consequence.
     *
     * @param cast the cast's bracket, or null for a cast whose stack top held
     *             no ability, after which no bracket is open
     */
    void openCast(OpenResolution cast) {
        // Nothing is cast before Forge has finished the last resolution, and
        // the id it watched may be cast again from here on.
        finishing = null;
        orphans.clear();
        if (cast != null) {
            items.add(cast);
        }
        open = cast;
    }

    /**
     * A cast made by the resolving ability itself: it waits beside the stack,
     * and the resolving item keeps the open bracket until it ends.
     */
    void waitForResolution(OpenResolution cast) {
        waiting.add(cast);
    }

    /**
     * While the stack resolves, the bracket the bus files into is the
     * resolving item's: the top of the stack's brackets, since what is cast
     * during a resolution waits beside the stack rather than going on top.
     * Only to be asked while the stack resolves.
     *
     * <p>Forge publishes nothing when a resolution starts, so this is asked
     * wherever one first shows — an event filed, a mode beginning, a refusal,
     * a cast made by it — and its resolution's end finds its bracket by id
     * however little it showed.
     *
     * <p>The stack still reads as resolving after a resolution has been
     * published: {@code MagicStack.resolveStack} publishes
     * {@code GameEventSpellResolved} first, then runs the static check and
     * {@code finishResolving}, which moves the card off the stack — to the
     * graveyard, to exile, out of existence for a copy — before it stops
     * resolving. That window is the resolved item's, which has no bracket any
     * more, and not the item below it: while the resolved item is still on
     * the stack ({@link #finishing}), nothing is opened.
     *
     * @param idsOnStack the ability ids the stack holds, read only while a
     *                   resolved item may still be on it
     */
    void followResolving(Supplier<Set<Integer>> idsOnStack) {
        OpenResolution top = top();
        if (top == null || open == top) {
            return;
        }
        if (finishing != null) {
            if (idsOnStack.get().contains(finishing)) {
                return;
            }
            finishing = null;
        }
        open = top;
    }

    /**
     * A resolution has been published: its bracket leaves, found by id
     * wherever it is — on top as a rule, and anywhere for an item resolved by
     * hand. What the resolution cast goes on top, and until the next item
     * resolves or something is cast, nothing is open: what follows is the
     * resolution's aftermath.
     *
     * @return the resolved item's bracket, or null for an item this collector
     *         never saw cast
     */
    OpenResolution closeAfterResolution(int abilityId) {
        OpenResolution resolved = remove(abilityId);
        items.addAll(waiting);
        waiting.clear();
        open = null;
        orphans.clear();
        finishing = abilityId;
        return resolved;
    }

    /**
     * A player's action at priority has begun: it ends the last cast's
     * consequences, and nothing is open until the next cast or resolution.
     */
    void closeForAction() {
        open = null;
    }

    /** The bracket of {@code abilityId} is written off, and closed if it was open. */
    void writeOff(int abilityId) {
        OpenResolution gone = remove(abilityId);
        if (gone != null && open == gone) {
            open = null;
        }
    }

    /**
     * Remove the bracket of every item the stack no longer holds: nothing
     * they gathered is anyone's. One of them may have been open.
     */
    void removeAbsent(Set<Integer> onStack) {
        items.removeIf(item -> !onStack.contains(item.ability.getId()));
        waiting.removeIf(item -> !onStack.contains(item.ability.getId()));
        if (open != null && !items.contains(open) && !waiting.contains(open)) {
            open = null;
        }
    }

    /** The top item's bracket, or null when no item is on the stack. */
    private OpenResolution top() {
        return items.isEmpty() ? null : items.get(items.size() - 1);
    }

    /** The bracket of {@code abilityId}, no longer on the stack, or null. */
    private OpenResolution remove(int abilityId) {
        for (int i = items.size() - 1; i >= 0; i--) {
            if (items.get(i).is(abilityId)) {
                return items.remove(i);
            }
        }
        for (int i = waiting.size() - 1; i >= 0; i--) {
            if (waiting.get(i).is(abilityId)) {
                return waiting.remove(i);
            }
        }
        return null;
    }
}
