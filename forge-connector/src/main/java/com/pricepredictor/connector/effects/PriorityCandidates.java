package com.pricepredictor.connector.effects;

import forge.ai.ComputerUtilAbility;
import forge.card.CardStateName;
import forge.game.Game;
import forge.game.GameActionUtil;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Both players' decision candidates at each priority stop, each with its
 * verdict, and the {@code decision} records written of them (FR-030, FR-033).
 *
 * <p>The collector enumerates them itself, at the priority-settled signal,
 * once the state-based check is over, rather than through the AI's
 * playability hook: Forge filters a spell on timing before that hook and
 * calls it for the player with priority only, so a sorcery in the hand of the
 * player without priority could never be recorded as unplayable.
 *
 * <p>A candidate is every spell ability of every card the player could play
 * from (the AI's own list of available cards), the unplayable ones included,
 * land plays included (FR-033a), mana abilities aside. Each is sampled in
 * two stages (FR-033b): at {@code --playability-rate} before anything is
 * evaluated, so a candidate the sampler drops costs no rules check, then by
 * its {@link DecisionSample} bucket once evaluated.
 *
 * <p>The random seat reads the same stop (FR-033 with FR-040's filter): when
 * the stop is for the seat's own priority, the seat draws from the candidates
 * the records were written of, by the verdicts they carry, so its plays and
 * its records cannot disagree on {@code can_play}. A seat asking at any
 * other moment, or in a run that records nothing, gets the same enumeration
 * evaluated afresh. The seat draws from every candidate of the stop, written
 * or not, and never from a land play: land drops stay the AI's.
 */
public final class PriorityCandidates {

    /** One candidate of one player, read once. */
    public static final class Candidate {
        private final Player player;
        private final SpellAbility ability;
        private Optional<DecisionCandidate> read;

        Candidate(Player player, SpellAbility ability) {
            this.player = player;
            this.ability = ability;
        }

        public Player player() {
            return player;
        }

        public SpellAbility ability() {
            return ability;
        }

        /**
         * What the rules say of it, read the first time it is asked; empty
         * where the engine cannot evaluate it on this board.
         */
        public Optional<DecisionCandidate> read() {
            if (read == null) {
                read = DecisionCandidate.read(ability, player);
            }
            return read;
        }

        /** Whether the rules let its player play it now. */
        public boolean playable() {
            return read().map(r -> r.verdict().playable()).orElse(false);
        }
    }

    /** The candidates of one priority stop, until the priority holder takes them. */
    private record Stop(Player priorityPlayer, List<Candidate> candidates) {
    }

    private final LiveGame live;
    private final PlayabilityChannel records;
    /** The last stop, or null once its priority holder has taken it. */
    private Stop current;

    PriorityCandidates(LiveGame live, PlayabilityChannel records) {
        this.live = live;
        this.records = records;
    }

    /** Where the game's {@link PhaseSignals} deliver the priority-settled signal. */
    PhaseSignals.Sink sink() {
        return new PhaseSignals.Sink() {
            @Override
            public void prioritySettled(Player priorityPlayer) {
                settle(priorityPlayer);
            }
        };
    }

    /**
     * A priority stop: enumerate both players' candidates, and record each
     * both sampling stages keep, the player without priority included, and
     * each a directed fork keeps whatever they draw, unless the fork has
     * written it failing the same checks at an earlier stop.
     *
     * <p>Run as the collector's own enumeration ({@link LiveGame#enumerate}):
     * the rules queries it makes are not the game's, so the state triggers
     * Forge's static check evaluates and queues inside them reach no channel
     * and leave nothing on the way to the stack. A candidate read later, when
     * the random seat draws, is the seat's own query, as the AI's are.
     */
    void settle(Player priorityPlayer) {
        // Cleared first: a stop whose enumeration throws leaves none, never the stop before.
        current = null;
        live.enumerate(() -> enumerate(priorityPlayer));
    }

    private void enumerate(Player priorityPlayer) {
        Game game = live.game();
        List<Candidate> candidates = new ArrayList<>();
        for (Player player : game.getPlayers()) {
            candidates.addAll(candidatesOf(game, player));
        }
        for (Candidate candidate : candidates) {
            // A candidate a directed fork keeps draws from neither stage, and
            // is written once per (candidate key, failed checks) pair (FR-010c).
            boolean kept = records.kept(candidate.ability());
            if (kept || records.sampled()) {
                candidate.read()
                        .filter(read -> kept ? records.firstKept(read) : records.keeps(read.sample()))
                        .ifPresent(read -> records.recordDecision(
                                candidate.player(), candidate.ability(), read));
            }
        }
        current = new Stop(priorityPlayer, candidates);
    }

    /**
     * {@code player}'s candidates as its priority stands: the current stop's
     * when it is {@code player}'s and not yet taken, which takes it;
     * otherwise a fresh enumeration.
     */
    public List<Candidate> takeFor(Player player) {
        Stop stop = current;
        if (stop == null || stop.priorityPlayer() != player) {
            return candidatesOf(player.getGame(), player);
        }
        current = null;
        return stop.candidates().stream().filter(c -> c.player() == player).toList();
    }

    /**
     * Every candidate of {@code player} on the board as it stands, none read
     * yet: every spell ability of the cards it could play from (the AI's own
     * list of available cards), land plays included, mana abilities aside.
     */
    public static List<Candidate> candidatesOf(Game game, Player player) {
        List<Candidate> candidates = new ArrayList<>();
        for (Card card : ComputerUtilAbility.getAvailableCards(game, player)) {
            for (SpellAbility ability : abilitiesOf(card, player)) {
                if (!ability.isManaAbility()) {
                    candidates.add(new Candidate(player, ability));
                }
            }
        }
        return candidates;
    }

    /**
     * Every ability {@code player} could play from {@code card}, the
     * unplayable ones included: the abilities of
     * {@code Card.getAllPossibleAbilities}, line for line, without its last
     * step.
     *
     * <p>A reimplementation, for a documented reason: Forge's only
     * enumeration ends by dropping every ability that cannot be played now
     * and by setting each one's activating player, and the candidates it
     * drops — a sorcery off turn, a spell a static forbids — are the ones
     * FR-033 records. Each alternative cost is its own candidate, beside the
     * spell it replaces, where the AI keeps it only once the spell itself is
     * unplayable: both are ways the rules let the card be played.
     */
    private static List<SpellAbility> abilitiesOf(Card card, Player player) {
        CardState original = card.getOriginalState(CardStateName.Original);
        List<SpellAbility> abilities = new ArrayList<>();
        addOwnAbilities(card, player, abilities);
        addFaceDownExileCosts(card, original, player, abilities);
        addHiddenAgendas(card, original, abilities);
        addModalBackside(card, player, abilities);
        addUnlocks(card, player, abilities);
        addTurnsFaceUp(card, original, abilities);
        abilities.removeIf(Objects::isNull);
        return abilities;
    }

    /** The card's own abilities, each with its alternative costs; an adventure's not while it is on one. */
    private static void addOwnAbilities(Card card, Player player, List<SpellAbility> abilities) {
        for (SpellAbility sa : card.getSpellAbilities()) {
            if (!onItsAdventure(card, sa)) {
                addWithAlternativeCosts(sa, player, abilities);
            }
        }
    }

    private static boolean onItsAdventure(Card card, SpellAbility sa) {
        return sa.isAdventure() && card.isOnAdventure();
    }

    private static void addWithAlternativeCosts(SpellAbility sa, Player player, List<SpellAbility> abilities) {
        abilities.add(sa);
        abilities.addAll(GameActionUtil.getAlternativeCosts(sa, player, false));
    }

    /** A face-down card in exile (a foretold one): its face's alternative costs alone. */
    private static void addFaceDownExileCosts(
            Card card, CardState original, Player player, List<SpellAbility> abilities) {
        if (!faceDownIn(card, ZoneType.Exile)) {
            return;
        }
        for (SpellAbility sa : original.getSpellAbilities()) {
            abilities.addAll(GameActionUtil.getAlternativeCosts(sa, player, false));
        }
    }

    /** A face-down card in the command zone: its hidden and double agendas' reveals. */
    private static void addHiddenAgendas(Card card, CardState original, List<SpellAbility> abilities) {
        if (!faceDownIn(card, ZoneType.Command)) {
            return;
        }
        for (Keyword agenda : List.of(Keyword.HIDDEN_AGENDA, Keyword.DOUBLE_AGENDA)) {
            addKeywordAbilities(original, agenda, abilities);
        }
    }

    private static void addKeywordAbilities(CardState state, Keyword keyword, List<SpellAbility> abilities) {
        for (KeywordInterface instance : state.getCachedKeyword(keyword)) {
            abilities.addAll(instance.getAbilities());
        }
    }

    private static boolean faceDownIn(Card card, ZoneType zone) {
        return card.isFaceDown() && card.isInZone(zone);
    }

    /** A modal double-faced card's back face: its spells and land plays, each with its alternative costs. */
    private static void addModalBackside(Card card, Player player, List<SpellAbility> abilities) {
        if (!modalWithBackside(card)) {
            return;
        }
        for (SpellAbility sa : card.getState(CardStateName.Backside).getSpellAbilities()) {
            if (castOrPlayed(sa)) {
                addWithAlternativeCosts(sa, player, abilities);
            }
        }
    }

    private static boolean modalWithBackside(Card card) {
        return card.isModal() && card.hasState(CardStateName.Backside);
    }

    private static boolean castOrPlayed(SpellAbility sa) {
        return sa.isSpell() || sa.isLandAbility();
    }

    /** A Room on the battlefield at sorcery speed: the unlock of each door still locked. */
    private static void addUnlocks(Card card, Player player, List<SpellAbility> abilities) {
        if (!mayUnlock(card, player)) {
            return;
        }
        CardStateName state = card.getCurrentStateName();
        if (leftLocked(state)) {
            abilities.add(card.getUnlockAbility(CardStateName.LeftSplit));
        }
        if (rightLocked(state)) {
            abilities.add(card.getUnlockAbility(CardStateName.RightSplit));
        }
    }

    private static boolean mayUnlock(Card card, Player player) {
        return card.isInPlay() && !card.isPhasedOut() && player.canCastSorcery();
    }

    private static boolean leftLocked(CardStateName state) {
        return state == CardStateName.RightSplit || state == CardStateName.EmptyRoom;
    }

    private static boolean rightLocked(CardStateName state) {
        return state == CardStateName.LeftSplit || state == CardStateName.EmptyRoom;
    }

    /**
     * A face-down permanent: its face's turn-face-up abilities (morph,
     * disguise) while it shows the face-down state, then a manifested or
     * cloaked creature's turning up for its mana cost.
     */
    private static void addTurnsFaceUp(Card card, CardState original, List<SpellAbility> abilities) {
        if (!faceDownInPlay(card)) {
            return;
        }
        if (card.getCurrentStateName() == CardStateName.FaceDown) {
            addTurnFaceUpAbilities(original, abilities);
        }
        if (creatureWithManaCost(original)) {
            addManifestedOrCloakedUp(card, original, abilities);
        }
    }

    private static boolean faceDownInPlay(Card card) {
        return card.isInPlay() && card.isFaceDown();
    }

    private static void addTurnFaceUpAbilities(CardState original, List<SpellAbility> abilities) {
        for (SpellAbility sa : original.getNonManaAbilities()) {
            if (sa.isTurnFaceUp()) {
                abilities.add(sa);
            }
        }
    }

    private static boolean creatureWithManaCost(CardState original) {
        return original.getType().isCreature() && original.getManaCost() != null
                && !original.getManaCost().isNoCost();
    }

    private static void addManifestedOrCloakedUp(Card card, CardState original, List<SpellAbility> abilities) {
        if (card.isManifested()) {
            abilities.add(original.getManifestUp());
        }
        if (card.isCloaked()) {
            abilities.add(original.getCloakUp());
        }
    }
}
