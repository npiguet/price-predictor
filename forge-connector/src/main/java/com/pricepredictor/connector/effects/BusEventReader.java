package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardView;
import forge.game.event.GameEventCardAttachment;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventCardCounters;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventCardPhased;
import forge.game.event.GameEventCardRegenerated;
import forge.game.event.GameEventCardStatsChanged;
import forge.game.event.GameEventCardTapped;
import forge.game.event.GameEventDayTimeChanged;
import forge.game.event.GameEventGameOutcome;
import forge.game.event.GameEventPlayerCounters;
import forge.game.event.GameEventPlayerDamaged;
import forge.game.event.GameEventPlayerLivesChanged;
import forge.game.event.GameEventPlayerPoisoned;
import forge.game.event.GameEventPlayerRadiation;
import forge.game.event.GameEventScry;
import forge.game.event.GameEventShuffle;
import forge.game.event.GameEventSpeedChanged;
import forge.game.event.GameEventSurveil;
import forge.game.event.GameEventTurnPhase;
import forge.game.event.GameEventZone;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.CostPaymentStack;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The outcome events of the public bus, read once for every collector.
 *
 * <p>The bracket collector and the fork sink both describe what happened
 * through {@link BusEvents}, and a forced resolution must be described the
 * way an observed one is: the same outcome events, each with the cause the
 * bus names for a damage, a poison or a radiation counter. Two subscription
 * lists drift apart, so the subscriptions live here, once, and a subclass
 * decides only what it alone can decide: whether a card
 * damage is combat damage, where an event goes, and what a permanent leaving
 * the battlefield means to it.
 *
 * <p>An abstract class rather than a reader handed a sink, because Forge's bus
 * finds {@code @Subscribe} methods on a subscriber's superclasses too: every
 * caller keeps subscribing the collector itself.
 */
abstract class BusEventReader {

    /** The game whose bus this reads, or null for a sink with no game to read. */
    final Game game;

    /**
     * Computed characteristics, so a stats-changed event can say what changed.
     *
     * <p>Lives as long as the reader: a pump and its wearing off are both
     * differences from what was there before.
     */
    private final StatDiffer stats = new StatDiffer();

    /**
     * Each player's command-zone marks as last read: whether they have the
     * city's blessing and how many dungeons they have completed, by player
     * id. Read when the reader starts, so a fork's reader takes the forked
     * board's marks as its baseline.
     */
    private final Map<Integer, CommandMarks> marks = new HashMap<>();

    /** The library-to-hand moves filed as draws whose draw counter is still read live. */
    private final DrawCheck.Pending draws = new DrawCheck.Pending();

    /** A player's blessing and completed dungeons. */
    private record CommandMarks(boolean blessed, int dungeonsCompleted) {
        static CommandMarks of(Player player) {
            return new CommandMarks(player.hasBlessing(), player.getCompletedDungeons().size());
        }
    }

    BusEventReader(Game game) {
        this.game = game;
        if (game != null) {
            for (Player player : game.getPlayers()) {
                marks.put(player.getId(), CommandMarks.of(player));
            }
        }
    }

    /**
     * Whether a card damaged now is damaged in combat.
     *
     * <p>{@code GameEventCardDamaged} carries no combat flag, unlike its player
     * sibling, so the reader asks: the bracket collector answers from the
     * step, a damage-step probe always yes, an intervention always no.
     */
    abstract boolean cardDamageIsCombat();

    /**
     * File one damage, card or player.
     *
     * @param namedCause the permanent the bus says dealt it, or null
     * @param combat     whether it is combat damage, which routes it
     */
    abstract void fileDamage(EffectEvent damage, String namedCause, boolean combat);

    /**
     * File any other outcome.
     *
     * @param namedCause the ref the bus named as the causer, which outranks
     *                   the bracket's, or null to take the bracket's
     */
    abstract void fileOutcome(EffectEvent event, String namedCause);

    /** File a completed move; leaving the battlefield can mean a death. */
    abstract void fileMove(GameEventCardChangeZone event, EffectEvent moved);

    /**
     * Whether this move is a permanent leaving the battlefield while a
     * state-based check runs: a death some earlier action left behind.
     *
     * <p>The check is read as the game's view tracker being frozen, which
     * {@code GameAction.checkStateEffects} holds for its whole loop. One
     * reading for the bracket collector and the fork sink alike, so an
     * observed death and a forced one are recognised by the same rule.
     */
    final boolean isStateBasedLeave(GameEventCardChangeZone event) {
        return leavesBattlefield(event) && game != null && game.getTracker().isFrozen();
    }

    /** Whether this move takes a permanent off the battlefield. */
    static boolean leavesBattlefield(GameEventCardChangeZone event) {
        return event.from() != null && event.from().zoneType() == ZoneType.Battlefield;
    }

    /**
     * Whether a player is paying, now, for a spell that {@code resolving}
     * casts — the mana and tapped lands of a madness cast, the sacrifice,
     * discard or life of a cascaded spell's additional cost.
     *
     * <p>That payment is the cast spell's cost, which its own cost half
     * records, and not the resolving ability's effect. Read off {@code
     * game.costPaymentStack}, which {@code CostPayment} keeps for every part
     * being paid, so a mana ability activated to pay is recognised by the
     * spell beneath it. Only a spell cast from a {@code Play} effect reaches
     * the stack in the middle of a resolution; the resolving ability's own
     * payments — an optional cost, an unless cost, which Forge pays through
     * the resolving ability or an empty one — are its effect and stay. One
     * reading for the bracket collector and the fork sink alike, so an
     * observed cascade and a forced one leave the same payment out.
     */
    final boolean payingForACast(SpellAbility resolving) {
        if (game == null) {
            return false;
        }
        for (CostPaymentStack.Entry entry : game.costPaymentStack) {
            var paid = entry.payment() == null ? null : entry.payment().getAbility();
            if (paid != null && paid.isSpell() && paid.isCastFromPlayEffect()
                    && paid != resolving) {
                return true;
            }
        }
        return false;
    }

    /**
     * One damage, and the permanent that dealt it.
     *
     * <p>The cause comes off the event rather than off the bracket, because in
     * a damage step there is no bracket: nothing is resolving, and the
     * attacking creature is the only answer to "what caused this". It is also
     * what makes two attackers each dealing 1 damage to the same blocker two
     * distinguishable events instead of one line rendered twice.
     */
    @Subscribe
    public void onCardDamaged(GameEventCardDamaged event) {
        boolean combat = cardDamageIsCombat();
        fileDamage(BusEvents.cardDamaged(event, combat), BusEvents.causeOf(event), combat);
    }

    @Subscribe
    public void onPlayerDamaged(GameEventPlayerDamaged event) {
        fileDamage(BusEvents.playerDamaged(event), BusEvents.causeOf(event), event.combat());
    }

    @Subscribe
    public void onLifeChanged(GameEventPlayerLivesChanged event) {
        file(BusEvents.lifeChanged(event), null);
    }

    @Subscribe
    public void onPoisoned(GameEventPlayerPoisoned event) {
        file(BusEvents.poisoned(event), BusEvents.causeOf(event));
    }

    @Subscribe
    public void onCounters(GameEventCardCounters event) {
        file(BusEvents.counters(event), null);
    }

    /** Energy, and any other player counter the vocabulary learns to name. */
    @Subscribe
    public void onPlayerCounters(GameEventPlayerCounters event) {
        file(BusEvents.playerCounter(event), null);
    }

    @Subscribe
    public void onRadiation(GameEventPlayerRadiation event) {
        file(BusEvents.radiation(event), BusEvents.causeOf(event));
    }

    @Subscribe
    public void onSpeed(GameEventSpeedChanged event) {
        file(BusEvents.speed(event), null);
    }

    @Subscribe
    public void onDayTime(GameEventDayTimeChanged event) {
        file(BusEvents.dayTime(event), null);
    }

    /**
     * A regeneration shield doing its job. No named cause: see {@link
     * BusEvents#regenerated}.
     */
    @Subscribe
    public void onRegenerated(GameEventCardRegenerated event) {
        file(BusEvents.regenerated(event), null);
    }

    /**
     * A player's command zone changed: the city's blessing arriving, or a
     * completed dungeon leaving it.
     *
     * <p>Neither has an event of its own. {@code Player.setBlessing} puts an
     * effect card named for the blessing into the command zone, and {@code
     * GameAction.completeDungeon} counts the dungeon completed and then takes
     * it out, so the zone's own notification ({@code GameEventZone}) is the
     * one moment both can be read, as the player's state against the state
     * last read: a {@code state_flag_change} {@code city_blessing} or {@code
     * dungeon_completed} (each {@code value} true) on the player. A player
     * gets the blessing once and keeps it, and a dungeon completed stays
     * counted, so neither is ever reported lost.
     */
    @Subscribe
    public void onZone(GameEventZone event) {
        if (game == null || event.zoneType() != ZoneType.Command || event.player() == null) {
            return;
        }
        Player player = game.getPlayer(event.player());
        if (player == null) {
            return;
        }
        CommandMarks now = CommandMarks.of(player);
        CommandMarks before = marks.put(player.getId(), now);
        if (before == null) {
            return;
        }
        if (now.blessed() && !before.blessed()) {
            fileOutcome(commandMark(player, "city_blessing"), null);
        }
        if (now.dungeonsCompleted() > before.dungeonsCompleted()) {
            fileOutcome(commandMark(player, "dungeon_completed"), null);
        }
    }

    private static EffectEvent commandMark(Player player, String flag) {
        return new EffectEvent(EventType.STATE_FLAG_CHANGE)
                .subject(Refs.of(player))
                .param("flag", flag)
                .param("value", true);
    }

    /** A library shuffled. No named cause: see {@link BusEvents#libraryShuffled}. */
    @Subscribe
    public void onShuffle(GameEventShuffle event) {
        file(BusEvents.libraryShuffled(event), null);
    }

    /**
     * The game ended. No named cause: see {@link BusEvents#gameOutcome}. Nothing
     * is filed on a draw, or when the winning name resolves to no player in
     * this game.
     */
    @Subscribe
    public void onGameOutcome(GameEventGameOutcome event) {
        if (game != null) {
            file(BusEvents.gameOutcome(event, game), null);
        }
    }

    @Subscribe
    public void onTapped(GameEventCardTapped event) {
        file(BusEvents.tapped(event), null);
    }

    /** A permanent phasing. No named cause: the bus names none. */
    @Subscribe
    public void onPhased(GameEventCardPhased event) {
        file(BusEvents.phased(event), null);
    }

    /**
     * A card moving, and the draw, discard or mill it may also be.
     *
     * <p>The workhorse outcome, and it is read here rather than off
     * {@code GameEventZone} because only this event is a <b>move</b>: the
     * per-zone-list notification fires once for the zone left, once for the zone
     * reached and once more for the stack, and it cannot say where a card came
     * from at all.
     *
     * <p>Draws, discards and mills have no event of their own: each is a card
     * moving between two zones and which of the three it is depends on the pair.
     */
    @Subscribe
    public void onCardChangeZone(GameEventCardChangeZone event) {
        Player libraryOwner = libraryLeftBy(event);
        if (libraryOwner != null) {
            draws.settle(libraryOwner);
        }
        EffectEvent moved = BusEvents.cardMoved(event);
        if (moved != null) {
            placeInLibrary(event, moved);
            fileMove(event, moved);
        }
        EffectEvent movement = BusEvents.libraryMovement(event);
        if (movement != null && movement.type() == EventType.CARD_DRAWN && libraryOwner != null) {
            movement.onlyIf(draws.open(libraryOwner)::drawn);
        }
        file(movement, null);
        if (event.card() == null) {
            return;
        }
        if (event.from() != null && event.from().zoneType() == ZoneType.Battlefield) {
            // Off the battlefield a card's computed characteristics stop
            // meaning anything, and one that returns is a new object.
            stats.forget(event.card().getId());
        }
        if (event.to() != null && event.to().zoneType() == ZoneType.Battlefield) {
            // Forge publishes a stats change after applying it, so without a
            // reading taken now the first pump finds no earlier value and
            // reports nothing. Read off the battlefield object, which the
            // statics and the counters it entered with have already reached.
            stats.prime(cardById(event.card().getId()));
        }
        if (event.to() != null && event.to().zoneType() == ZoneType.Battlefield) {
            enteredTapped(event);
        }
        if (event.to() != null && event.to().zoneType() == ZoneType.Stack) {
            onStackArrival(event.card().getId(),
                    event.from() == null ? null : CardSpelling.zoneName(event.from().zoneType()));
        }
    }

    /**
     * A permanent that entered the battlefield tapped: a {@code tapped} event
     * after its {@code zone_change}, in the record that moved it (FR-042d).
     *
     * <p>Forge publishes no tap for it. An "enters tapped" replacement and a
     * "put onto the battlefield tapped" effect both set the flag with {@code
     * Card.setTapped} before the card arrives, which publishes nothing, so the
     * entering card's own state after the move is the only place the fact
     * shows, and a library card a record names is rendered as it was in the
     * library (FR-042e), untapped.
     */
    private void enteredTapped(GameEventCardChangeZone event) {
        Card entered = cardById(event.card().getId());
        if (entered != null && entered.isTapped()) {
            fileOutcome(new EffectEvent(EventType.TAPPED).subject(Refs.of(entered)), null);
        }
    }

    /**
     * A move into a library, with where in it the card went
     * ({@code library_position}): 0 the top, -1 the bottom, else its index
     * from the top. Read off the library as the bus publishes the move, which
     * {@code GameAction.changeZone} does after adding the card, so a card put
     * second from the top reads 1 and the bottom of a library of any size
     * reads -1. A card alone in its library is its top. Written nowhere when
     * the card cannot be found there, and taken off again by a shuffle of
     * that library later in the same record ({@link EventBuffer#file}): a
     * card shuffled in has no position.
     *
     * <p>Into a library only. A move out of one says nothing of where the card
     * was: by the time the bus publishes it the card has left.
     */
    private void placeInLibrary(GameEventCardChangeZone event, EffectEvent moved) {
        if (game == null || event.to() == null || event.to().zoneType() != ZoneType.Library
                || event.to().player() == null) {
            return;
        }
        Player owner = game.getPlayer(event.to().player());
        if (owner == null) {
            return;
        }
        List<Card> library = new ArrayList<>();
        owner.getCardsIn(ZoneType.Library).forEach(library::add);
        int id = event.card().getId();
        for (int index = 0; index < library.size(); index++) {
            if (library.get(index).getId() == id) {
                boolean bottom = index > 0 && index == library.size() - 1;
                moved.param("library_position", bottom ? -1 : index);
                moved.intoLibraryOf(Refs.of(owner));
                return;
            }
        }
    }

    /**
     * A card has arrived in the stack zone. Published after the static check
     * its move set off, so what that check did has already been filed.
     *
     * <p>A no-op here: only the bracket collector follows a cast from the
     * card's arrival to its cast event.
     *
     * @param fromZone the zone it came from, as a record spells it, or null
     */
    void onStackArrival(int cardId, String fromZone) {
    }

    /**
     * Take a reading of every card on the battlefield, as of now.
     *
     * <p>For a reader that starts listening to a game already in progress, such
     * as a fork: every card on the board is new to it, and its first change
     * would otherwise be its baseline. And at each snapshot of a record that
     * collects events — a bracket opened at a cast, a mode's half, a damage
     * step — taken just before: a record's stat changes are changes after its
     * own snapshot ({@link StatDiffer#rebaseline}).
     *
     * @return the changes already made and shown on the board but not yet
     *         published, which no record reports
     */
    List<EffectEvent> rebaselineStats() {
        if (game == null) {
            return List.of();
        }
        return stats.rebaseline(game.getCardsIn(ZoneType.Battlefield));
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
                    fileOutcome(changed, null);
                }
            }
        }
    }

    @Subscribe
    public void onAttachment(GameEventCardAttachment event) {
        file(BusEvents.attachment(event), null);
    }

    @Subscribe
    public void onScry(GameEventScry event) {
        for (EffectEvent looked : BusEvents.scry(event)) {
            fileOutcome(looked, null);
        }
    }

    @Subscribe
    public void onSurveil(GameEventSurveil event) {
        for (EffectEvent looked : BusEvents.surveil(event)) {
            fileOutcome(looked, null);
        }
    }

    /**
     * A step boundary: no card is still being drawn, and the cleanup step's
     * reset of the draw counters comes after its boundary, so every draw
     * check is settled here.
     */
    @Subscribe
    public void onStepBoundary(GameEventTurnPhase event) {
        draws.settleAll();
    }

    /** The player whose library this move takes a card out of, or null when it is none or this reader has no game. */
    private Player libraryLeftBy(GameEventCardChangeZone event) {
        if (game == null || event.card() == null || event.card().getOwner() == null
                || event.from() == null || event.from().zoneType() != ZoneType.Library) {
            return null;
        }
        return game.getPlayer(event.card().getOwner());
    }

    /** File an outcome a {@link BusEvents} factory may have answered null for. */
    private void file(EffectEvent event, String namedCause) {
        if (event != null) {
            fileOutcome(event, namedCause);
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
        if (game == null) {
            return null;
        }
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            if (card.getId() == id) {
                return card;
            }
        }
        return null;
    }
}
