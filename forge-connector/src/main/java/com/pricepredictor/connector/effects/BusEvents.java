package com.pricepredictor.connector.effects;

import forge.LobbyPlayer;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardView;
import forge.game.event.GameEventCardAttachment;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventCardCounters;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventCardPhased;
import forge.game.event.GameEventCardRegenerated;
import forge.game.event.GameEventCardSacrificed;
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
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * One Forge bus event as one {@link EffectEvent}.
 *
 * <p>Pure shaping, with no opinion about where the result belongs. Two
 * collectors need it and route the results differently: the bracket collector
 * sorts each event into the resolution or the combat bracket, while the fork
 * collector appends everything one forced resolution produced. Shared because
 * the alternative is two readings of the same bus event that drift — and a
 * drift here is a record that describes the wrong thing while looking fine.
 */
final class BusEvents {

    private BusEvents() {
    }

    /**
     * The {@code mode} of a look at a library, each the engine's own name for
     * it: the trigger modes Forge runs for a scry and a surveil, and the one
     * it runs for a search ({@code ApiEvents}' search rule).
     */
    static final String SCRY = "Scry";
    static final String SURVEIL = "Surveil";
    static final String SEARCHED_LIBRARY = "SearchedLibrary";

    // ── what the bus names as the causer ────────────────────────────────

    /**
     * The permanent the bus says dealt this damage, or null.
     *
     * <p>Read here rather than fabricated by the collector because the event
     * carries it: {@code GameEventCardDamaged} and
     * {@code GameEventPlayerDamaged} each name a source {@code CardView}, and
     * for combat damage that source is the attacking or blocking creature —
     * the one fact a damage step's record could not say.
     *
     * <p>Answered as a ref rather than written onto the event, because which
     * cause an event ends up carrying is a decision about precedence — the
     * source the bus named beats the ability whose bracket the event landed in
     * — and this class deliberately holds no opinion about brackets. The
     * collector combines the two; see {@code BusBracketCollector.fileOutcome}.
     */
    static String causeOf(GameEventCardDamaged event) {
        return Refs.ofNullable(event.source());
    }

    /** The permanent the bus says dealt this damage to a player, or null. */
    static String causeOf(GameEventPlayerDamaged event) {
        return Refs.ofNullable(event.source());
    }

    /**
     * The player the bus says handed out this poison, or null.
     *
     * <p>A player rather than a card: {@code GameEventPlayerPoisoned} names the
     * source as a {@code PlayerView}, which is the infecting player rather than
     * the creature. Both spellings are legal refs and {@code state.entities}
     * carries both, so the channel takes whichever the engine actually names.
     */
    static String causeOf(GameEventPlayerPoisoned event) {
        return Refs.ofNullable(event.source());
    }

    /**
     * The player the bus says irradiated this player, or null.
     *
     * <p>{@code GameEventPlayerRadiation} names its source the same way
     * {@code GameEventPlayerPoisoned} does — both are fired from
     * {@code Player.setCounters}, one line apart, from the same {@code source}
     * local — so it is read the same way.
     */
    static String causeOf(GameEventPlayerRadiation event) {
        return Refs.ofNullable(event.source());
    }

    /**
     * Damage to a creature, with how it was dealt as {@code damage_type}:
     * {@code normal} (marked on the creature), {@code m1m1_counters} (wither
     * or infect, 120.3d: -1/-1 counters, which the event's own
     * {@code counter_change} also tells, and no damage marked) or {@code
     * deathtouch}. Forge's choice: a source with deathtouch reads {@code
     * deathtouch} even when its wither or infect turned the damage into
     * counters. Damage to a planeswalker or a battle publishes no such
     * event, only its counter change; {@code loyalty_loss} is Forge's
     * spelling for it should it ever.
     *
     * <p>{@code excess} is the damage beyond lethal damage (120.4a) the
     * engine measured before the damage was dealt, which the event carries
     * ({@link PatchHooks#DAMAGE_EXCESS}): by the time it is published the
     * damage is marked or the loyalty gone. Absent on a checkout without it.
     */
    static EffectEvent cardDamaged(GameEventCardDamaged event, boolean combat) {
        return new EffectEvent(EventType.DAMAGE_DEALT)
                .subject(Refs.of(event.card()))
                .param(EventParam.AMOUNT, event.amount())
                .param(EventParam.COMBAT, combat)
                .param(EventParam.DAMAGE_TYPE, damageType(event.type()))
                .param(EventParam.EXCESS, PatchHooks.DAMAGE_EXCESS.read(event));
    }

    /** A card damage's type as {@code damage_type} spells it, or null when Forge named none. */
    static String damageType(GameEventCardDamaged.DamageType type) {
        if (type == null) {
            return null;
        }
        return switch (type) {
            case Normal -> "normal";
            case M1M1Counters -> "m1m1_counters";
            case Deathtouch -> "deathtouch";
            case LoyaltyLoss -> "loyalty_loss";
        };
    }

    static EffectEvent playerDamaged(GameEventPlayerDamaged event) {
        return new EffectEvent(EventType.DAMAGE_DEALT)
                .subject(Refs.of(event.target()))
                .param(EventParam.AMOUNT, event.amount())
                .param(EventParam.COMBAT, event.combat());
    }

    static EffectEvent lifeChanged(GameEventPlayerLivesChanged event) {
        return new EffectEvent(EventType.LIFE_CHANGE)
                .subject(Refs.of(event.player()))
                .param(EventParam.DELTA, event.newLives() - event.oldLives());
    }

    static EffectEvent poisoned(GameEventPlayerPoisoned event) {
        return new EffectEvent(EventType.POISON_CHANGE)
                .subject(Refs.of(event.receiver()))
                .param(EventParam.DELTA, event.amount());
    }

    /**
     * A player counter, for the one kind the vocabulary names.
     *
     * <p>Energy, and only energy: poison has its own bus event and its own
     * factory, and a counter this has no reading for is skipped rather than
     * guessed at — an event type invented here is one no reader accepts. A
     * null type is the bulk reset Forge fires when a player's whole counter
     * set is replaced at once — {@code Player.clearCounters} and the
     * {@code Multiset} overload {@code GameCopier} and {@code GameSnapshot}
     * use to copy a player's counters onto a fresh game both name no single
     * counter — which is even less of a reading than an unmapped type, so it
     * is skipped the same way rather than risking a {@code
     * NullPointerException} on {@code event.type()}.
     *
     * <p>The delta is {@code amount - oldValue}, not {@code amount} alone.
     * Despite its name, {@code GameEventPlayerCounters.amount()} is fired with
     * the counter's new total rather than the change: {@code
     * Player.setCounters} computes the real delta as {@code num - old} for the
     * sibling poison and radiation events two lines below the same fire site,
     * from the same two values. Reading {@code amount} alone would make a
     * second Aether Hub tap in the same game record the player's cumulative
     * energy as if it had all arrived at once.
     */
    static Optional<EffectEvent> playerCounter(GameEventPlayerCounters event) {
        if (event.type() == null) {
            return Optional.empty();
        }
        String type = CardSpelling.counterName(event.type());
        if (!"ENERGY".equals(type)) {
            return Optional.empty();
        }
        return Optional.of(new EffectEvent(EventType.ENERGY_CHANGE)
                .subject(Refs.of(event.receiver()))
                .param(EventParam.DELTA, event.amount() - event.oldValue()));
    }

    static EffectEvent radiation(GameEventPlayerRadiation event) {
        return new EffectEvent(EventType.RADIATION_CHANGE)
                .subject(Refs.of(event.receiver()))
                .param(EventParam.DELTA, event.change());
    }

    static EffectEvent speed(GameEventSpeedChanged event) {
        return new EffectEvent(EventType.SPEED_CHANGED)
                .subject(Refs.of(event.player()))
                .param(EventParam.DELTA, event.newValue() - event.oldValue());
    }

    /** Day and night are a property of the game, so this event names no subject. */
    static EffectEvent dayTime(GameEventDayTimeChanged event) {
        return new EffectEvent(EventType.DAY_NIGHT_CHANGED)
                .param(EventParam.TO, event.daytime() ? "day" : "night");
    }

    /**
     * A regeneration shield doing its job: heal, tap, remove from combat.
     *
     * <p>Read off the bus rather than through the run-parameter mode table
     * ({@code RunParamNormalizer}), which has no entry for it: {@code
     * TriggerType} and {@code ReplacementType} spell regeneration's own
     * trigger and replacement modes differently, and no mode a hook reports
     * is the regeneration itself. {@code RegenerationEffect.resolve()}, the
     * only fire site in the engine, publishes this once per card actually
     * regenerated.
     *
     * <p>Multiple subjects in principle — the record type wraps a collection —
     * though the one fire site always passes a single card; read generically
     * rather than assume that stays true.
     */
    static EffectEvent regenerated(GameEventCardRegenerated event) {
        EffectEvent built = new EffectEvent(EventType.REGENERATED);
        for (CardView card : event.cards()) {
            built.subject(Refs.ofNullable(card));
        }
        return built;
    }

    /**
     * A library shuffled, named by the player whose library it was.
     *
     * <p>{@code Player.shuffle(SpellAbility)}, the only fire site in the
     * engine, fires this once, whatever scripted
     * the shuffle — a fetchland, a scripted "shuffle your library" effect, the
     * end-of-search shuffle after a tutor. No cause rides the event because
     * none is named: the record carries only the player, so unlike {@code
     * radiation}/{@code poisoned} this has no {@code causeOf} counterpart.
     */
    static EffectEvent libraryShuffled(GameEventShuffle event) {
        return new EffectEvent(EventType.LIBRARY_SHUFFLED)
                .subject(Refs.ofNullable(event.player()));
    }

    /**
     * The game's own outcome, resolved back to the winning {@link Player}.
     *
     * <p>{@code GameEventGameOutcome} names the winner only by {@code
     * winningPlayerName} -- a {@code LobbyPlayer}'s display name, not a ref --
     * so it is matched back against {@code game.getPlayers()} here rather than
     * carried through as a name: every other subject on this branch is an
     * entity or player ref, and a display name in a ref's place silently drops
     * every row a reader joins to {@code state.entities}.
     *
     * <p>Empty on a draw: {@code winningPlayerName} is null whenever {@code
     * GameOutcome.getWinningLobbyPlayer()} is, and a {@code player_won} with no
     * winner would be a false record -- {@code game_drawn} is the type for
     * that outcome.
     *
     * <p>No {@link #causeOf} counterpart, the same as {@link #regenerated} and
     * {@link #libraryShuffled}: the record carries only the winner, because
     * Forge's own outcome names no separate cause of the win.
     *
     * <p>Two players sharing a lobby name is not a shape this connector's own
     * worker setup produces -- the two seats' lobby names ({@code
     * Side.lobbyName()}) are always distinct strings -- but nothing enforces
     * that in general, so a
     * collision is broken by taking the first match in {@code
     * game.getPlayers()}'s own order rather than guessing further.
     */
    static Optional<EffectEvent> gameOutcome(GameEventGameOutcome event, Game game) {
        String winnerName = event.winningPlayerName();
        if (winnerName == null) {
            return Optional.empty();
        }
        for (Player player : game.getPlayers()) {
            LobbyPlayer lobby = player.getLobbyPlayer();
            if (lobby != null && winnerName.equals(lobby.getName())) {
                return Optional.of(new EffectEvent(EventType.PLAYER_WON)
                        .subject(Refs.of(player)));
            }
        }
        return Optional.empty();
    }

    /**
     * Every player the game's end found lost, each a {@code player_lost}
     * with Forge's {@code GameLossReason} as {@code reason}
     * ({@code LifeReachedZero}, {@code Milled}, {@code Poisoned}, {@code
     * SpellEffect}, {@code Conceded}, {@code CommanderDamage}, {@code
     * OpponentWon}), as the trigger path's run parameters spell it. Read off
     * each player's outcome when the game's outcome is published; a player
     * who agreed to a draw lost nothing.
     */
    static List<EffectEvent> playersLost(Game game) {
        List<EffectEvent> lost = new ArrayList<>();
        // Registered rather than in the game: a player who lost has left it.
        for (Player player : game.getRegisteredPlayers()) {
            lossOf(player).ifPresent(reason -> lost.add(new EffectEvent(EventType.PLAYER_LOST)
                    .subject(Refs.of(player))
                    .param(EventParam.REASON, reason.name())));
        }
        return lost;
    }

    /** Why the player lost, or empty when they lost nothing: no outcome, or a draw. */
    private static Optional<forge.game.player.GameLossReason> lossOf(Player player) {
        var outcome = player.getOutcome();
        if (outcome == null || outcome.lossState == null) {
            return Optional.empty();
        }
        return outcome.lossState == forge.game.player.GameLossReason.IntentionalDraw
                ? Optional.empty() : Optional.of(outcome.lossState);
    }

    /** A permanent sacrificed, or empty when the event names no card. */
    static Optional<EffectEvent> sacrificed(GameEventCardSacrificed event) {
        if (event.card() == null) {
            return Optional.empty();
        }
        return Optional.of(new EffectEvent(EventType.SACRIFICED).subject(Refs.of(event.card())));
    }

    static EffectEvent counters(GameEventCardCounters event) {
        return new EffectEvent(EventType.COUNTER_CHANGE)
                .subject(Refs.of(event.card()))
                .param(EventParam.COUNTER_TYPE, CardSpelling.counterName(event.type()))
                .param(EventParam.DELTA, event.newValue() - event.oldValue());
    }

    /**
     * A permanent phasing in or out, and which: {@code out} is the state it
     * reached, as {@code Card.phase} publishes it once the switch is made.
     *
     * <p>Fired for each permanent that phases, the ones phasing indirectly
     * with what they are attached to included, and whatever phased it — the
     * untap step's turn-based action or a phasing effect resolving.
     */
    static Optional<EffectEvent> phased(GameEventCardPhased event) {
        if (event.card() == null) {
            return Optional.empty();
        }
        return Optional.of(new EffectEvent(EventType.PHASED)
                .subject(Refs.of(event.card()))
                .param(EventParam.OUT, event.phaseState()));
    }

    static EffectEvent tapped(GameEventCardTapped event) {
        return new EffectEvent(
                event.tapped() ? EventType.TAPPED : EventType.UNTAPPED)
                .subject(Refs.of(event.card()));
    }

    /**
     * One completed move, with both ends of it.
     *
     * <p>Read from {@code GameEventCardChangeZone}, which fires once per move
     * and carries the zone the card left as well as the one it reached, rather
     * than from the per-zone-list notification {@code GameEventZone}, which is
     * not a move at all: {@code Zone.remove} publishes one for the zone the
     * card left, {@code Zone.add} another for the zone it reached and
     * {@code MagicStack} a third for the stack, so one cast would render three
     * events, the first naming a destination the card never reached and the
     * third repeating the second verbatim. Reading the move is also the only
     * way {@code from_zone} gets filled at all.
     *
     * <p>Empty when neither end is known, which is a move this collector has
     * nothing to say about rather than a move to nowhere.
     *
     * <p>Also empty for a move <b>onto</b> the stack. That move is a spell being
     * cast, and the cast already has a record of its own — the activation half,
     * with the costs that were paid. Filed as an outcome it is worse than
     * redundant: it lands in whichever bracket happens to be open, so an
     * opponent's instant cast in response reads as something the resolving
     * ability did. A move <i>off</i> the stack is kept, because that one is an
     * outcome — the permanent arriving, or the spell going to the graveyard —
     * and says {@code from_zone=stack}.
     */
    static Optional<EffectEvent> cardMoved(GameEventCardChangeZone event) {
        return cardMoved(new Move(event));
    }

    /** {@link #cardMoved(GameEventCardChangeZone)} for a move already read. */
    static Optional<EffectEvent> cardMoved(Move move) {
        if (move.card() == null) {
            return Optional.empty();
        }
        String from = move.fromName();
        String to = move.toName();
        if (from == null && to == null) {
            return Optional.empty();
        }
        if (move.to(ZoneType.Stack)) {
            return Optional.empty();
        }
        return Optional.of(new EffectEvent(EventType.ZONE_CHANGE)
                .subject(Refs.of(move.card()))
                .param(EventParam.FROM_ZONE, from)
                .param(EventParam.TO_ZONE, to));
    }

    /**
     * A draw, a discard or a mill, named for what it is.
     *
     * <p>Forge publishes no event for any of the three: each is a card moving
     * between two zones, and which of the three it is depends on where it came
     * from. The head counts all three per player, so the movement has to be
     * classified here or those counters never fire.
     *
     * <p>The subject is the <b>player</b>, not the card, because that is whose
     * count it is. A zone-change event for the card itself is recorded
     * separately by {@link #cardMoved}.
     *
     * <p>A library-to-hand move is only a draw candidate here: a tutor or an
     * explore moves a card the same way. The reader keeps it only when
     * Forge's draw counter moved for it ({@link DrawCheck}). A
     * library-to-graveyard move is likewise only a mill candidate, kept when
     * Forge flagged the card milled ({@link MillCheck}): a dig's rest and an
     * Entomb move cards the same way. A hand-to-exile move is a discard when
     * the player's discarded list says that card was just discarded, which is
     * how a madness discard reads.
     */
    static Optional<EffectEvent> libraryMovement(Move move, Player owner) {
        if (move.card() == null || move.owner() == null) {
            return Optional.empty();
        }
        if (move.fromZone() == null || move.toZone() == null) {
            return Optional.empty();
        }
        return movementType(move, owner).map(type -> new EffectEvent(type)
                .subject(Refs.of(move.card().getOwner()))
                .param(EventParam.COUNT, 1));
    }

    private static Optional<EventType> movementType(Move move, Player owner) {
        ZoneType from = move.fromZone();
        return switch (move.toZone()) {
            case Hand -> from == ZoneType.Library
                    ? Optional.of(EventType.CARD_DRAWN) : Optional.empty();
            case Graveyard -> switch (from) {
                case Hand -> Optional.of(EventType.CARD_DISCARDED);
                case Library -> Optional.of(EventType.CARD_MILLED);
                default -> Optional.empty();
            };
            case Exile -> discardedToExile(move, owner)
                    ? Optional.of(EventType.CARD_DISCARDED) : Optional.empty();
            default -> Optional.empty();
        };
    }

    private static boolean discardedToExile(Move move, Player owner) {
        return move.from(ZoneType.Hand) && justDiscarded(owner, move.card());
    }

    /**
     * Whether the card leaving {@code owner}'s hand is the one Forge counted
     * as discarded a moment ago: {@code Player.discard} appends the card to
     * the player's discarded-this-turn list <i>before</i> it moves it, so a
     * discard that a madness replacement sends to exile instead of the
     * graveyard (702.35a) is the list's last card when its move is
     * published. Any other move from the hand to exile is no discard.
     */
    private static boolean justDiscarded(Player owner, CardView card) {
        if (owner == null) {
            return false;
        }
        List<Card> discarded = owner.getDiscardedThisTurn();
        return !discarded.isEmpty()
                && discarded.get(discarded.size() - 1).getId() == card.getId();
    }

    /**
     * An aura, equipment or fortification moving, or empty when it names no
     * card: {@code attached} with {@code attached_to}, the entity it now
     * hangs on, or {@code unattached} with {@code detached_from}, the one it
     * left, as the trigger path's run parameters name them. A move from one
     * host to another is an {@code unattached} from the old host, then an
     * {@code attached} to the new one: {@code Card.attachToEntity} unattaches
     * first.
     */
    static Optional<EffectEvent> attachment(GameEventCardAttachment event) {
        if (event.equipment() == null) {
            return Optional.empty();
        }
        boolean attaching = event.newTarget() != null;
        EffectEvent built = new EffectEvent(
                attaching ? EventType.ATTACHED : EventType.UNATTACHED)
                .subject(Refs.of(event.equipment()));
        String host = entityRef(attaching ? event.newTarget() : event.oldEntity());
        return Optional.of(attaching
                ? built.param(EventParam.ATTACHED_TO, host)
                : built.param(EventParam.DETACHED_FROM, host));
    }

    /** A card's or a player's ref from its view, or null for neither. */
    private static String entityRef(forge.game.GameEntityView entity) {
        if (entity instanceof CardView card) {
            return Refs.of(card);
        }
        if (entity instanceof forge.game.player.PlayerView player) {
            return Refs.of(player);
        }
        return null;
    }

    /**
     * Scry, as the two things it is: cards looked at, and a library reordered.
     *
     * <p>The head's library-event channel is a multi-hot over what happened to
     * a library, and a scry is both. The look carries the engine's name for it
     * as its {@code mode}, {@value #SCRY}, which is what tells it from a
     * surveil that kept every card on top: the two are otherwise the same
     * events.
     */
    static List<EffectEvent> scry(GameEventScry event) {
        int count = event.toTop() + event.toBottom();
        if (event.player() == null || count <= 0) {
            return List.of();
        }
        String player = Refs.of(event.player());
        return List.of(
                new EffectEvent(EventType.CARD_LOOKED_AT)
                        .subject(player).param(EventParam.COUNT, count).param(EventParam.MODE, SCRY),
                new EffectEvent(EventType.LIBRARY_REORDERED)
                        .subject(player).param(EventParam.COUNT, count));
    }

    /**
     * Surveil: cards looked at, and the library reordered. The look's
     * {@code mode} is {@value #SURVEIL} (see {@link #scry}).
     *
     * <p>No mill: each card the surveil bins is its own move from the library
     * to the graveyard, which {@link #libraryMovement} already tells as one
     * {@code card_milled}.
     */
    static List<EffectEvent> surveil(GameEventSurveil event) {
        int count = event.toLibrary() + event.toGraveyard();
        if (event.player() == null || count <= 0) {
            return List.of();
        }
        String player = Refs.of(event.player());
        List<EffectEvent> events = new ArrayList<>();
        events.add(new EffectEvent(EventType.CARD_LOOKED_AT)
                .subject(player).param(EventParam.COUNT, count).param(EventParam.MODE, SURVEIL));
        events.add(new EffectEvent(EventType.LIBRARY_REORDERED)
                .subject(player).param(EventParam.COUNT, count));
        return events;
    }
}
