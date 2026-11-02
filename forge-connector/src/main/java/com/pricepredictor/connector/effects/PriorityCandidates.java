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
 * lands and mana abilities aside, as the random seat has always left them
 * out. Each is sampled at
 * {@code --playability-rate} before anything is evaluated, so a candidate the
 * sampler drops costs no rules check.
 *
 * <p>The random seat reads the same stop (FR-033 with FR-040's filter): when
 * the stop is for the seat's own priority, the seat draws from the candidates
 * the records were written of, by the verdicts they carry, so its plays and
 * its records cannot disagree on {@code can_play}. A seat asking at any
 * other moment, or in a run that records nothing, gets the same enumeration
 * evaluated afresh.
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
     * the sampler draws, the player without priority included.
     */
    void settle(Player priorityPlayer) {
        Game game = live.game();
        List<Candidate> candidates = new ArrayList<>();
        for (Player player : game.getPlayers()) {
            candidates.addAll(candidatesOf(game, player));
        }
        for (Candidate candidate : candidates) {
            if (records.sampled()) {
                candidate.read().ifPresent(read ->
                        records.recordDecision(candidate.player(), candidate.ability(), read));
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
     * list of available cards), lands and mana abilities aside.
     */
    public static List<Candidate> candidatesOf(Game game, Player player) {
        List<Candidate> candidates = new ArrayList<>();
        for (Card card : ComputerUtilAbility.getAvailableCards(game, player)) {
            for (SpellAbility ability : abilitiesOf(card, player)) {
                if (!ability.isLandAbility() && !ability.isManaAbility()) {
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
        for (SpellAbility sa : card.getSpellAbilities()) {
            if (sa.isAdventure() && card.isOnAdventure()) {
                continue;
            }
            abilities.add(sa);
            abilities.addAll(GameActionUtil.getAlternativeCosts(sa, player, false));
        }
        if (card.isFaceDown() && card.isInZone(ZoneType.Exile)) {
            for (SpellAbility sa : original.getSpellAbilities()) {
                abilities.addAll(GameActionUtil.getAlternativeCosts(sa, player, false));
            }
        }
        if (card.isFaceDown() && card.isInZone(ZoneType.Command)) {
            for (Keyword agenda : List.of(Keyword.HIDDEN_AGENDA, Keyword.DOUBLE_AGENDA)) {
                for (KeywordInterface keyword : original.getCachedKeyword(agenda)) {
                    abilities.addAll(keyword.getAbilities());
                }
            }
        }
        if (card.isModal() && card.hasState(CardStateName.Backside)) {
            for (SpellAbility sa : card.getState(CardStateName.Backside).getSpellAbilities()) {
                if (sa.isSpell() || sa.isLandAbility()) {
                    abilities.add(sa);
                    abilities.addAll(GameActionUtil.getAlternativeCosts(sa, player, false));
                }
            }
        }
        if (card.isInPlay() && !card.isPhasedOut() && player.canCastSorcery()) {
            CardStateName state = card.getCurrentStateName();
            if (state == CardStateName.RightSplit || state == CardStateName.EmptyRoom) {
                abilities.add(card.getUnlockAbility(CardStateName.LeftSplit));
            }
            if (state == CardStateName.LeftSplit || state == CardStateName.EmptyRoom) {
                abilities.add(card.getUnlockAbility(CardStateName.RightSplit));
            }
        }
        if (card.isInPlay() && card.isFaceDown()) {
            if (card.getCurrentStateName() == CardStateName.FaceDown) {
                for (SpellAbility sa : original.getNonManaAbilities()) {
                    if (sa.isTurnFaceUp()) {
                        abilities.add(sa);
                    }
                }
            }
            if (original.getType().isCreature() && original.getManaCost() != null
                    && !original.getManaCost().isNoCost()) {
                if (card.isManifested()) {
                    abilities.add(original.getManifestUp());
                }
                if (card.isCloaked()) {
                    abilities.add(original.getCloakUp());
                }
            }
        }
        abilities.removeIf(Objects::isNull);
        return abilities;
    }
}
