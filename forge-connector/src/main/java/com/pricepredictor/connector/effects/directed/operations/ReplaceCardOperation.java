package com.pricepredictor.connector.effects.directed.operations;

import com.google.common.collect.HashMultiset;
import com.pricepredictor.connector.effects.ForkRefusal;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import forge.GameCommand;
import forge.ai.simulation.GameCommandCopier;
import forge.card.CardRules;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.GameObject;
import forge.game.IEntityMap;
import forge.game.StaticEffect;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.combat.Combat;
import forge.game.phase.Phase;
import forge.game.player.Player;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerHandler;
import forge.game.zone.Zone;

import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.UnaryOperator;

/**
 * {@code replace_card}: put another card from the card database in a
 * permanent's place, as the same permanent in every respect but the card.
 *
 * <p>{@link RemovePermanentOperation}'s removing step, then
 * {@link AddPermanentOperation}'s adding step at the old card's place in the
 * battlefield list, with the old card's state carried across: controller,
 * tapped state, summoning sickness, damage (and whether any was deathtouch),
 * counters, timestamps (so its place in the timestamp order and "entered this
 * turn" are the old card's), the effects applied to it (a resolved pump's
 * power and toughness, keywords, types, colours and abilities granted or
 * removed until end of turn), who controls it and since when (a gain-control
 * effect's controller included), its soulbond pair, its damage history (what
 * it attacked, blocked and dealt damage to this turn), what is attached to it
 * and what it is attached to, and its place in combat. The static abilities
 * that applied to the old card are handed the new one in its place, so the
 * static check the edit ends with takes theirs off it and applies them anew
 * where they still apply. Attachments are relinked directly rather than
 * unattached and attached again, which would fire the attach and unattach
 * triggers; the combat is rebuilt with the new card wherever the old one
 * stood, by the same copy {@code GameCopier} builds a fork's combat with, so
 * its bands, blocks and damage-assignment orders stay as they were. Nothing
 * fires: the old card goes to no zone and the new one arrives with no enters
 * trigger or replacement.
 *
 * <p>What names the old card object names the new one after the edit: every
 * pending command of the fork (a phase's, a state-based check's, a card's,
 * the old card's own moved onto the new one), so an until-end-of-turn pump
 * the new card took over ends on it; every delayed trigger hosted by it or
 * remembering it; every card remembering it. Its goad, monstrous and
 * renowned designations are carried.
 *
 * <p>Some state belongs to the old card's own text, or to a face no other
 * card has, and carrying it would describe a board nobody built: the edit
 * refuses ({@link ForkRefusal}, counted as a refusal, not a failure) a
 * face-down or phased-out permanent, a Class above its first level, a
 * permanent holding chosen values (a player, number, type, colour, card,
 * name or parity its text chose) or remembered, imprinted or exiled-with
 * cards, a merged (mutated) or melded permanent, whose other cards the edit
 * would leave orphaned, a token, which a card in its place would turn into a
 * real card, and a permanent a pending command names that cannot be
 * rewritten. It also refuses, after the edit, a replacing creature the next
 * state-based check would put into a graveyard: the damage and counters it
 * carried, with its boosts and resolved effects, are lethal to it.
 *
 * <p>Summoning sickness is carried as the raw flag: a hasty creature that
 * came under its controller's control this turn is replaced by a card that
 * is sick unless it has haste of its own.
 *
 * @param permanent the fork's permanent to replace
 * @param cardName  the replacing card's name in the card database
 */
public record ReplaceCardOperation(Card permanent, String cardName) implements BoardEdit {

    public ReplaceCardOperation {
        Objects.requireNonNull(cardName, "cardName");
    }

    @Override
    public ForkOperation operation() {
        return ForkOperation.REPLACE_CARD;
    }

    /** Refusals' reasons: what the replaced permanent holds that the new card cannot take over. */
    public static final String FACE_DOWN = "replace_card face-down permanent";
    public static final String PHASED_OUT = "replace_card phased-out permanent";
    public static final String CLASS_LEVEL = "replace_card class level";
    public static final String CHOSEN_VALUES = "replace_card chosen values";
    /** No comma: the tally the supervisor reads is a printed map split on {@code ", "}. */
    public static final String LINKED_CARDS = "replace_card remembered or imprinted or exiled cards";
    public static final String UNREWRITABLE_COMMAND = "replace_card command that cannot be rewritten";
    public static final String DIES = "replace_card replacement dies at the first state-based check";
    public static final String MERGED = "replace_card merged permanent";
    public static final String MELDED = "replace_card melded permanent";
    public static final String TOKEN = "replace_card token";

    /** A refusal: what the old card holds that its replacement cannot take over, and its reason. */
    private record Refusal(Predicate<Card> holds, String reason) {
    }

    /** The refusals before the edit, in the order they are asked; see the class comment. */
    private static final List<Refusal> REFUSALS = List.of(
            new Refusal(Card::isFaceDown, FACE_DOWN),
            new Refusal(Card::isPhasedOut, PHASED_OUT),
            new Refusal(card -> card.getClassLevel() > 1, CLASS_LEVEL),
            new Refusal(ReplaceCardOperation::holdsChosenValues, CHOSEN_VALUES),
            new Refusal(ReplaceCardOperation::holdsLinkedCards, LINKED_CARDS),
            new Refusal(Card::hasMergedCard, MERGED),
            new Refusal(card -> card.getMeldedWith() != null, MELDED),
            new Refusal(Card::isToken, TOKEN));

    @Override
    public void apply(Game fork) {
        Card old = BoardEdit.onBattlefield(permanent);
        refuseWhatCannotBeCarried(old);
        Player controller = old.getController();
        Zone battlefield = old.getZone();
        int index = battlefield.getCards(false).indexOf(old);
        Card replacement = NewCards.create(fork, cardName, old.getOwner());
        UnaryOperator<GameCommand> rewrite = command -> GameCommandCopier.replacing(command, old, replacement);
        if (!rewriteCommands(fork, old, command -> rewrite.apply(command) == null ? null : command)) {
            throw new ForkRefusal(UNREWRITABLE_COMMAND);
        }

        rewriteCommands(fork, old, rewrite);
        replacement.copyCommandsFrom(old, rewrite);
        relinkDelayedTriggers(fork, old, replacement);
        relinkRemembered(fork, old, replacement);
        relinkAttachments(old, replacement);
        relinkPair(old, replacement);
        swapInCombat(fork, old, replacement);
        swapInStaticEffects(fork, old, replacement);
        RemovePermanentOperation.takeOff(fork, old);
        AddPermanentOperation.put(fork, replacement, controller, index);
        carryState(old, replacement);
        BoardEdit.settle(fork);
        if (diesAtTheFirstCheck(replacement)) {
            throw new ForkRefusal(DIES);
        }
    }

    /**
     * Whether the next state-based check puts the replacing card into a
     * graveyard ({@link #diesOf}), read off the card once the edit is made.
     * No variant's point is that death (the rules' preconditions keep such
     * boards out, through {@link #replacementDies}); this is the edit's own
     * guard for every rule using it.
     */
    static boolean diesAtTheFirstCheck(Card replacement) {
        return replacement.isCreature() && diesOf(replacement.getNetToughness(), replacement.getDamage(),
                replacement.hasBeenDealtDeathtouchDamage());
    }

    /**
     * Whether a creature of {@code toughness} with {@code damage} marked on it,
     * deathtouch damage among it when {@code deathtouch}, is put into a
     * graveyard by a state-based check: toughness 0 or less (704.5f), lethal
     * damage (704.5g) or any deathtouch damage (704.5h). The one reading
     * {@link #diesAtTheFirstCheck} and {@link #replacementDies} share.
     */
    static boolean diesOf(int toughness, int damage, boolean deathtouch) {
        return toughness <= 0 || damage >= toughness || damage > 0 && deathtouch;
    }

    /**
     * Whether {@code cardName} put in {@code permanent}'s place would die at
     * the fork's first state-based check, read off the live permanent before
     * any edit: the replacing card's toughness is its printed toughness, or
     * the last toughness a resolved effect set on the permanent, which it
     * carries ({@link #carriedSetPt}), plus the permanent's toughness boosts
     * and counters, which it carries too, against the permanent's marked
     * damage and deathtouch damage. A boost from a static ability is read as it applies to the
     * permanent: the static check the edit ends with applies it to the new
     * card anew, which differs only where the static selects by a
     * characteristic the two cards do not share. A replacing card that is
     * no creature never dies of toughness or damage.
     *
     * @throws IllegalArgumentException when the card database has no {@code cardName}
     */
    public static boolean replacementDies(Card permanent, String cardName) {
        CardRules rules = NewCards.rules(cardName);
        if (!rules.getType().isCreature()) {
            return false;
        }
        int toughness = carriedSetPt(permanent, Pair::getRight).orElse(rules.getMainPart().getIntToughness())
                + permanent.getTempToughnessBoost() + permanent.getToughnessBonusFromCounters();
        return diesOf(toughness, permanent.getDamage(), permanent.hasBeenDealtDeathtouchDamage());
    }

    /**
     * The power {@code cardName} put in {@code permanent}'s place has once
     * the edit is made, read off the live permanent as
     * {@link #replacementDies} reads toughness: the replacing card's printed
     * power, or the last power a resolved effect set on the permanent, plus
     * the permanent's power boosts and counters. 0 for a replacing card that
     * is no creature.
     *
     * @throws IllegalArgumentException when the card database has no {@code cardName}
     */
    public static int replacementPower(Card permanent, String cardName) {
        CardRules rules = NewCards.rules(cardName);
        if (!rules.getType().isCreature()) {
            return 0;
        }
        return carriedSetPt(permanent, Pair::getLeft).orElse(rules.getMainPart().getIntPower())
                + permanent.getTempPowerBoost() + permanent.getPowerBonusFromCounters();
    }

    /**
     * The last power or toughness ({@code part}) a resolved effect set on the
     * permanent: the set-P/T rows the edit carries ({@link #carryState}). A
     * characteristic-defining ability's row (Tarmogoyf's) belongs
     * to the old card's text and is not carried, so it is not read either.
     */
    private static Optional<Integer> carriedSetPt(
            Card permanent, Function<Pair<Integer, Integer>, Integer> part) {
        Integer last = null;
        for (Pair<Integer, Integer> set : permanent.getSetPTTable().values()) {
            Integer value = part.apply(set);
            last = value == null ? last : value;
        }
        return Optional.ofNullable(last);
    }

    /**
     * Whether the edit takes {@code permanent} on at all: no refusal before
     * the edit ({@link #REFUSALS}) holds for it. Read off the live permanent
     * by the preconditions, so they bind no board the edit would refuse.
     */
    public static boolean carries(Card permanent) {
        return REFUSALS.stream().noneMatch(refusal -> refusal.holds().test(permanent));
    }

    /** Refuse a permanent whose state the new card cannot take over ({@link #REFUSALS}). */
    private static void refuseWhatCannotBeCarried(Card old) {
        for (Refusal refusal : REFUSALS) {
            if (refusal.holds().test(old)) {
                throw new ForkRefusal(refusal.reason());
            }
        }
    }

    /** Whether the card holds a value its text chose: a player, number, type, colour, card, name or parity. */
    private static boolean holdsChosenValues(Card card) {
        return card.hasChosenPlayer() || card.hasChosenNumber() || card.hasChosenType() || card.hasChosenColor()
                || card.hasChosenCard() || card.hasNamedCard() || card.hasChosenEvenOdd();
    }

    /** Whether the card remembers, imprints or has exiled cards. */
    private static boolean holdsLinkedCards(Card card) {
        return card.hasRemembered() || card.hasImprintedCard() || card.hasExiledCard();
    }

    /**
     * Every pending command of the fork but the old card's own put through
     * {@code rewrite}: each phase's, the state-based check's, each other
     * card's. Each list is left whole when {@code rewrite} refuses one of its
     * commands, so a dry run returning a command itself or null tells whether
     * the real one can go through.
     *
     * @return whether every command was rewritten
     */
    private static boolean rewriteCommands(Game fork, Card old, UnaryOperator<GameCommand> rewrite) {
        boolean all = true;
        for (Phase phase : List.of(fork.getUntap(), fork.getUpkeep(), fork.getBeginOfCombat(),
                fork.getEndOfCombat(), fork.getEndOfTurn(), fork.getCleanup())) {
            all &= phase.rewriteCommands(rewrite);
        }
        List<GameCommand> checked = new ArrayList<>();
        for (GameCommand command : fork.sbaCheckedCommandList) {
            GameCommand rewritten = rewrite.apply(command);
            if (rewritten == null) {
                return false;
            }
            checked.add(rewritten);
        }
        fork.sbaCheckedCommandList.clear();
        fork.sbaCheckedCommandList.addAll(checked);
        for (Card card : fork.getCardsInGame()) {
            if (card != old) {
                all &= card.rewriteCommands(rewrite);
            }
        }
        if (old.rewriteCommands(command -> rewrite.apply(command) == null ? null : command)) {
            return all;
        }
        return false;
    }

    /** The delayed triggers hosted by or remembering {@code old}, on {@code replacement} instead. */
    private static void relinkDelayedTriggers(Game fork, Card old, Card replacement) {
        TriggerHandler.PendingDelayedTriggers pending = fork.getTriggerHandler().getPendingDelayedTriggers();
        List<Trigger> triggers = new ArrayList<>(pending.delayed());
        triggers.addAll(pending.playerDefined().values());
        for (Trigger trigger : triggers) {
            if (trigger.getHostCard() == old) {
                trigger.setHostCard(replacement);
            }
            List<Object> remembered = trigger.getTriggerRemembered();
            if (remembered.stream().anyMatch(o -> o == old)) {
                List<Object> relinked = new ArrayList<>();
                for (Object o : remembered) {
                    relinked.add(o == old ? replacement : o);
                }
                trigger.setTriggerRemembered(relinked);
            }
        }
    }

    /** Every card remembering {@code old} remembers {@code replacement} instead. */
    private static void relinkRemembered(Game fork, Card old, Card replacement) {
        for (Card card : fork.getCardsInGame()) {
            if (card != old && card.isRemembered(old)) {
                card.removeRemembered(old);
                card.addRemembered(replacement);
            }
        }
    }

    /** The old card's board state, onto the card that replaced it. */
    private static void carryState(Card old, Card replacement) {
        replacement.setGameTimestamp(old.getGameTimestamp());
        replacement.setLayerTimestamp(old.getLayerTimestamp());
        replacement.setTurnInZone(old.getTurnInZone());
        replacement.setTurnInController(old.getTurnInController());
        replacement.copyControllersFrom(old, player -> player);
        replacement.setTapped(old.isTapped());
        replacement.setSickness(old.isFirstTurnControlled());
        replacement.setCounters(HashMultiset.create(old.getCounters()));
        replacement.setDamage(old.getDamage());
        replacement.setHasBeenDealtDeathtouchDamage(old.hasBeenDealtDeathtouchDamage());
        replacement.setDamageHistory(old.getDamageHistory());
        // Every layer row, as GameCopier carries a card's: a resolved
        // effect's (static id 0) stays; a static's is taken off and put
        // back by the static check the edit ends with (swapInStaticEffects).
        replacement.setPTTable(old.getSetPTTable());
        replacement.setPTBoost(old.getPTBoostTable());
        replacement.copyFrom(old);
        replacement.copyResolvedRowsFrom(old);
        // The permanent's designations, which stay with it whatever card it is.
        for (Map.Entry<Long, Player> goad : old.getGoadMap().entrySet()) {
            replacement.addGoad(goad.getKey(), goad.getValue());
        }
        replacement.setMonstrous(old.isMonstrous());
        replacement.setRenowned(old.isRenowned());
        replacement.updateKeywordsCache();
    }

    /** {@code old}'s soulbond partner, paired with {@code replacement} instead. */
    private static void relinkPair(Card old, Card replacement) {
        Card partner = old.getPairedWith();
        if (partner == null) {
            return;
        }
        old.setPairedWith(null);
        replacement.setPairedWith(partner);
        partner.setPairedWith(replacement);
    }

    /**
     * {@code replacement} in {@code old}'s place among the cards each static
     * ability of the fork applies to, so the next static check takes the
     * static's rows off the card that now carries them.
     */
    private static void swapInStaticEffects(Game fork, Card old, Card replacement) {
        for (StaticEffect effect : fork.getStaticEffects().getEffects()) {
            if (!effect.getAffectedCards().contains(old)) {
                continue;
            }
            CardCollection affected = new CardCollection();
            for (Card card : effect.getAffectedCards()) {
                affected.add(old.equals(card) ? replacement : card);
            }
            effect.setAffectedCards(affected);
        }
    }

    /**
     * Move what is attached to {@code old}, and what {@code old} is attached
     * to, onto {@code replacement}, through the attachment links themselves:
     * no attach or unattach trigger, no new timestamp for the attachments.
     */
    private static void relinkAttachments(Card old, Card replacement) {
        for (Card attached : List.copyOf(old.getAttachedCards())) {
            old.removeAttachedCard(attached);
            attached.setEntityAttachedTo(replacement);
            replacement.addAttachedCard(attached);
        }
        GameEntity host = old.getEntityAttachedTo();
        if (host != null) {
            host.removeAttachedCard(old);
            old.setEntityAttachedTo(null);
            replacement.setEntityAttachedTo(host);
            host.addAttachedCard(replacement);
        }
    }

    /** The fork's combat with {@code replacement} wherever {@code old} stood. */
    private static void swapInCombat(Game fork, Card old, Card replacement) {
        Combat combat = fork.getCombat();
        if (combat == null || !(combat.isAttacking(old) || combat.isBlocking(old)
                || combat.getDefenders().contains(old))) {
            // Left alone otherwise: the copy does not carry which creatures
            // already dealt first-strike damage.
            return;
        }
        fork.getPhaseHandler().setCombat(new Combat(combat, new IEntityMap() {
            @Override
            public Game getGame() {
                return fork;
            }

            @Override
            public GameObject map(GameObject o) {
                return old.equals(o) ? replacement : o;
            }
        }));
    }
}
