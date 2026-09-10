package com.pricepredictor.connector.effects;

import forge.LobbyPlayer;
import forge.game.Game;
import forge.game.card.CardView;
import forge.game.event.GameEventCardAttachment;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventCardCounters;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventCardRegenerated;
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
import forge.game.player.PlayerView;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

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
     * collector combines the two; see {@code BusBracketCollector.record}.
     */
    static String causeOf(GameEventCardDamaged event) {
        return cardRef(event.source());
    }

    /** The permanent the bus says dealt this damage to a player, or null. */
    static String causeOf(GameEventPlayerDamaged event) {
        return cardRef(event.source());
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
        return playerRef(event.source());
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
        return playerRef(event.source());
    }

    private static String cardRef(CardView view) {
        return view == null ? null : "E" + view.getId();
    }

    private static String playerRef(PlayerView view) {
        return view == null ? null : "P" + view.getId();
    }

    static EffectEvent cardDamaged(GameEventCardDamaged event, boolean combat) {
        return new EffectEvent(EffectEvent.DAMAGE_DEALT)
                .subject("E" + event.card().getId())
                .param("amount", event.amount())
                .param("combat", combat);
    }

    static EffectEvent playerDamaged(GameEventPlayerDamaged event) {
        return new EffectEvent(EffectEvent.DAMAGE_DEALT)
                .subject("P" + event.target().getId())
                .param("amount", event.amount())
                .param("combat", event.combat());
    }

    static EffectEvent lifeChanged(GameEventPlayerLivesChanged event) {
        return new EffectEvent(EffectEvent.LIFE_CHANGE)
                .subject("P" + event.player().getId())
                .param("delta", event.newLives() - event.oldLives());
    }

    static EffectEvent poisoned(GameEventPlayerPoisoned event) {
        return new EffectEvent(EffectEvent.POISON_CHANGE)
                .subject("P" + event.receiver().getId())
                .param("delta", event.amount());
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
    static EffectEvent playerCounter(GameEventPlayerCounters event) {
        if (event.type() == null) {
            return null;
        }
        String type = event.type().getName().toUpperCase(Locale.ROOT);
        if (!"ENERGY".equals(type)) {
            return null;
        }
        return new EffectEvent(EffectEvent.ENERGY_CHANGE)
                .subject("P" + event.receiver().getId())
                .param("delta", event.amount() - event.oldValue());
    }

    static EffectEvent radiation(GameEventPlayerRadiation event) {
        return new EffectEvent(EffectEvent.RADIATION_CHANGE)
                .subject("P" + event.receiver().getId())
                .param("delta", event.change());
    }

    static EffectEvent speed(GameEventSpeedChanged event) {
        return new EffectEvent(EffectEvent.SPEED_CHANGED)
                .subject("P" + event.player().getId())
                .param("delta", event.newValue() - event.oldValue());
    }

    /** Day and night are a property of the game, so this event names no subject. */
    static EffectEvent dayTime(GameEventDayTimeChanged event) {
        return new EffectEvent(EffectEvent.DAY_NIGHT_CHANGED)
                .param("to", event.daytime() ? "day" : "night");
    }

    /**
     * A regeneration shield doing its job: heal, tap, remove from combat.
     *
     * <p>Read off the bus rather than through the mode table {@code
     * describeParams} consults, because that table's {@code "Regenerated"}
     * entry names no real Forge mode — {@code TriggerType} and {@code
     * ReplacementType} spell regeneration's own trigger and replacement modes
     * differently, so the entry can never match at runtime. It is a dead
     * reference, the same shape as the original {@code damage_prevented} /
     * {@code spell_copied} finding, not a live path serving a different
     * purpose. {@code RegenerationEffect.resolve()} (fire site: {@code
     * RegenerationEffect.java:49}, the only one in the engine) publishes this
     * once per card actually regenerated.
     *
     * <p>Multiple subjects in principle — the record type wraps a collection —
     * though the one fire site always passes a single card; read generically
     * rather than assume that stays true.
     */
    static EffectEvent regenerated(GameEventCardRegenerated event) {
        EffectEvent built = new EffectEvent(EffectEvent.REGENERATED);
        for (CardView card : event.cards()) {
            built.subject(cardRef(card));
        }
        return built;
    }

    /**
     * A library shuffled, named by the player whose library it was.
     *
     * <p>{@code Player.shuffle(SpellAbility)} fires this once (fire site:
     * {@code Player.java:1627}, the only one in the engine), whatever scripted
     * the shuffle — a fetchland, a scripted "shuffle your library" effect, the
     * end-of-search shuffle after a tutor. No cause rides the event because
     * none is named: the record carries only the player, so unlike {@code
     * radiation}/{@code poisoned} this has no {@code causeOf} counterpart.
     */
    static EffectEvent libraryShuffled(GameEventShuffle event) {
        return new EffectEvent(EffectEvent.LIBRARY_SHUFFLED)
                .subject(playerRef(event.player()));
    }

    /**
     * The game's own outcome, resolved back to the winning {@link Player}.
     *
     * <p>{@code GameEventGameOutcome} names the winner only by {@code
     * winningPlayerName} -- a {@code LobbyPlayer}'s display name, not a ref --
     * so it is matched back against {@code game.getPlayers()} here rather than
     * carried through as a name: every other subject on this branch is an
     * entity or player ref, and Task 7 shipped {@code damage_prevented.source}
     * as a display name once, which silently dropped every row a reader joined
     * to {@code state.entities}.
     *
     * <p>Null on a draw: {@code winningPlayerName} is null whenever {@code
     * GameOutcome.getWinningLobbyPlayer()} is, and a {@code player_won} with no
     * winner would be a false record -- {@code game_drawn} is the type for
     * that outcome, and it is not this task's scope.
     *
     * <p>No {@link #causeOf} counterpart, the same as {@link #regenerated} and
     * {@link #libraryShuffled}: the record carries only the winner, because
     * Forge's own outcome names no separate cause of the win.
     *
     * <p>Two players sharing a lobby name is not a shape this connector's own
     * worker setup produces -- {@code GamePlayer.LOBBY_NAME_A}/{@code _B} are
     * always distinct strings -- but nothing enforces that in general, so a
     * collision is broken by taking the first match in {@code
     * game.getPlayers()}'s own order rather than guessing further.
     */
    static EffectEvent gameOutcome(GameEventGameOutcome event, Game game) {
        String winnerName = event.winningPlayerName();
        if (winnerName == null) {
            return null;
        }
        for (Player player : game.getPlayers()) {
            LobbyPlayer lobby = player.getLobbyPlayer();
            if (lobby != null && winnerName.equals(lobby.getName())) {
                return new EffectEvent(EffectEvent.PLAYER_WON)
                        .subject(SnapshotBuilder.playerId(player));
            }
        }
        return null;
    }

    static EffectEvent counters(GameEventCardCounters event) {
        return new EffectEvent(EffectEvent.COUNTER_CHANGE)
                .subject("E" + event.card().getId())
                .param("counter_type", event.type().getName().toUpperCase(Locale.ROOT))
                .param("delta", event.newValue() - event.oldValue());
    }

    static EffectEvent tapped(GameEventCardTapped event) {
        return new EffectEvent(
                event.tapped() ? EffectEvent.TAPPED : EffectEvent.UNTAPPED)
                .subject("E" + event.card().getId());
    }

    /**
     * One completed move, with both ends of it.
     *
     * <p>Read from {@code GameEventCardChangeZone}, which fires once per move
     * and carries the zone the card left as well as the one it reached. It
     * replaces the per-zone-list notification {@code GameEventZone}, which is
     * not a move at all: {@code Zone.remove} publishes one for the zone the
     * card left, {@code Zone.add} another for the zone it reached and
     * {@code MagicStack} a third for the stack, so one cast rendered three
     * events of which the first named a destination the card never reached and
     * the third repeated the second verbatim. That triple fire is where every
     * exact-duplicate event measured in a record came from, and reading the
     * move instead is the only way {@code from_zone} — a documented parameter
     * of the event type that nothing had ever written — gets filled at all.
     *
     * <p>Null when neither end is known, which is a move this collector has
     * nothing to say about rather than a move to nowhere.
     *
     * <p>Also null for a move <b>onto</b> the stack. That move is a spell being
     * cast, and the cast already has a record of its own — the activation half,
     * with the costs that were paid. Filed as an outcome it is worse than
     * redundant: it lands in whichever bracket happens to be open, so an
     * opponent's instant cast in response reads as something the resolving
     * ability did. It was 54% of the whole zone_change channel. A move
     * <i>off</i> the stack is kept, because that one is an outcome — the
     * permanent arriving, or the spell going to the graveyard — and now says
     * {@code from_zone=stack} where before it said nothing.
     */
    static EffectEvent cardMoved(GameEventCardChangeZone event) {
        if (event.card() == null) {
            return null;
        }
        String from = zoneName(event.from());
        String to = zoneName(event.to());
        if (from == null && to == null) {
            return null;
        }
        if (event.to() != null && event.to().zoneType() == ZoneType.Stack) {
            return null;
        }
        return new EffectEvent(EffectEvent.ZONE_CHANGE)
                .subject("E" + event.card().getId())
                .param("from_zone", from)
                .param("to_zone", to);
    }

    private static String zoneName(forge.game.zone.ZoneView zone) {
        if (zone == null || zone.zoneType() == null) {
            return null;
        }
        return zone.zoneType().name().toLowerCase(Locale.ROOT);
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
     */
    static EffectEvent libraryMovement(GameEventCardChangeZone event) {
        if (event.card() == null || event.card().getOwner() == null
                || event.from() == null || event.to() == null) {
            return null;
        }
        ZoneType from = event.from().zoneType();
        ZoneType to = event.to().zoneType();
        if (from == null || to == null) {
            return null;
        }
        String type = switch (to) {
            case Hand -> from == ZoneType.Library ? EffectEvent.CARD_DRAWN : null;
            case Graveyard -> switch (from) {
                case Hand -> EffectEvent.CARD_DISCARDED;
                case Library -> EffectEvent.CARD_MILLED;
                default -> null;
            };
            default -> null;
        };
        if (type == null) {
            return null;
        }
        return new EffectEvent(type)
                .subject("P" + event.card().getOwner().getId())
                .param("count", 1);
    }

    /** An aura or equipment moving, or null when it names no card. */
    static EffectEvent attachment(GameEventCardAttachment event) {
        if (event.equipment() == null) {
            return null;
        }
        boolean attaching = event.newTarget() != null;
        return new EffectEvent(
                attaching ? EffectEvent.ATTACHED : EffectEvent.UNATTACHED)
                .subject("E" + event.equipment().getId());
    }

    /**
     * Scry, as the two things it is: cards looked at, and a library reordered.
     *
     * <p>The head's library-event channel is a multi-hot over what happened to
     * a library, and a scry is both.
     */
    static List<EffectEvent> scry(GameEventScry event) {
        int count = event.toTop() + event.toBottom();
        if (event.player() == null || count <= 0) {
            return List.of();
        }
        String player = "P" + event.player().getId();
        return List.of(
                new EffectEvent(EffectEvent.CARD_LOOKED_AT)
                        .subject(player).param("count", count),
                new EffectEvent(EffectEvent.LIBRARY_REORDERED)
                        .subject(player).param("count", count));
    }

    /** Surveil: cards looked at, and however many went to the graveyard. */
    static List<EffectEvent> surveil(GameEventSurveil event) {
        int count = event.toLibrary() + event.toGraveyard();
        if (event.player() == null || count <= 0) {
            return List.of();
        }
        String player = "P" + event.player().getId();
        List<EffectEvent> events = new ArrayList<>();
        events.add(new EffectEvent(EffectEvent.CARD_LOOKED_AT)
                .subject(player).param("count", count));
        events.add(new EffectEvent(EffectEvent.LIBRARY_REORDERED)
                .subject(player).param("count", count));
        if (event.toGraveyard() > 0) {
            events.add(new EffectEvent(EffectEvent.CARD_MILLED)
                    .subject(player).param("count", event.toGraveyard()));
        }
        return events;
    }
}
