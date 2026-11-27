package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.TargetChoices;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The uniform draws (FR-022b, FR-022c), on real charms and real targets.
 *
 * <p>Each draw is checked for its range over many seeds rather than for one
 * seed's answer: what the random seat promises is that every legal option is
 * reachable and nothing illegal is, and one seed can show neither.
 */
@ExtendWith(ForgeExtension.class)
class RandomChoicesTest {

    private static final int SEEDS = 200;

    /**
     * The draws for one seed, scrambled through {@code scramble} (the choices' own seeding is
     * {@code SeedStreams}, not this): a
     * linear congruential generator's first output is nearly linear in its
     * seed, so consecutive raw seeds all make nearly the same first draw and
     * a range test over them sees one value.
     */
    private static RandomChoices drawsFor(int seed) {
        return new RandomChoices(new Random(GameRecording.scramble(seed)));
    }

    // ── modes ───────────────────────────────────────────────────────────

    /**
     * A "choose one" charm draws exactly one mode, and every mode is
     * reachable: with a creature out for its double-strike mode to target.
     */
    @Test
    void aChooseOneCharmDrawsOneModeAndReachesEveryMode() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        TestBoard.inPlay(game, caster, "Grizzly Bears");
        SpellAbility charm = TestBoard.charmOf(TestBoard.card(game, caster, "Boros Charm"), caster);
        List<AbilitySub> choices = charm.getAdditionalAbilityList("Choices");

        Set<Integer> seen = new HashSet<>();
        for (int seed = 0; seed < SEEDS; seed++) {
            List<AbilitySub> modes = drawsFor(seed).drawModes(charm).orElseThrow();
            assertEquals(1, modes.size(), "CharmNum 1: " + modes);
            seen.add(choices.indexOf(modes.get(0)));
        }
        assertEquals(Set.of(0, 1, 2), seen);
    }

    /**
     * A charm draws only among the modes whose targets exist: Abzan Charm in
     * hand over a board with no creature draws its untargeted mode every
     * time, and once a 3-power creature is out its targeted modes too.
     */
    @Test
    void aCharmDrawsOnlyAmongModesWhoseTargetsExist() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        SpellAbility charm = TestBoard.charmOf(TestBoard.inHand(game, caster, "Abzan Charm"), caster);
        List<AbilitySub> choices = charm.getAdditionalAbilityList("Choices");

        assertEquals(Set.of(1), modeIndexesDrawn(charm, choices), "no creature: only the mode that targets nothing");

        TestBoard.inPlay(game, game.getPlayers().get(1), "Hill Giant");
        assertEquals(Set.of(0, 1, 2), modeIndexesDrawn(charm, choices), "a 3-power creature: every mode");
    }

    /** The indexes among {@code choices} of every mode a charm draws over all the seeds. */
    private static Set<Integer> modeIndexesDrawn(SpellAbility charm, List<AbilitySub> choices) {
        Set<Integer> seen = new HashSet<>();
        for (int seed = 0; seed < SEEDS; seed++) {
            for (AbilitySub mode : drawsFor(seed).drawModes(charm).orElseThrow()) {
                seen.add(choices.indexOf(mode));
            }
        }
        return seen;
    }

    /**
     * Whether a forced charm has its choices asks of every legal mode set,
     * not of one draw's: with no creature out, the untargeted mode is one.
     * And a forced charm's real draw then always finds its targets.
     */
    @Test
    void aCharmWithOneModeLeftHasItsChoices() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card host = TestBoard.inHand(game, caster, "Abzan Charm");
        SpellAbility charm = TestBoard.charmOf(host, caster);
        InterventionFork forks = new InterventionFork(null, null, null, 0L);

        assertTrue(forks.hasChoices(charm, caster));
        for (int seed = 0; seed < SEEDS; seed++) {
            SpellAbility trial = charm.copy(host, caster, true);
            trial.setActivatingPlayer(caster);
            assertTrue(drawsFor(seed).chooseModes(trial), "seed " + seed);
            assertTrue(drawsFor(seed).chooseTargets(trial), "seed " + seed + ": " + trial.getChosenList());
        }
    }

    /**
     * A charm whose every mode targets, over a board with nothing to target,
     * has no choices: Boros Charm's player mode always has one, so a charm
     * built of creature modes alone.
     */
    @Test
    void aCharmWithNoTargetableModeHasNoChoices() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card host = TestBoard.inHand(game, caster, "Grizzly Bears");
        host.setSVar("A", "DB$ Pump | ValidTgts$ Creature | NumAtt$ +1 | SpellDescription$ a");
        host.setSVar("B", "DB$ Destroy | ValidTgts$ Creature | SpellDescription$ b");
        SpellAbility charm = AbilityFactory.getAbility("SP$ Charm | Choices$ A,B", host);
        charm.setActivatingPlayer(caster);

        assertTrue(drawsFor(1).drawModes(charm).isEmpty());
        assertFalse(new InterventionFork(null, null, null, 0L).hasChoices(charm, caster));

        TestBoard.inPlay(game, caster, "Grizzly Bears");
        assertTrue(drawsFor(1).drawModes(charm).isPresent());
        assertTrue(new InterventionFork(null, null, null, 0L).hasChoices(charm, caster));
    }

    /**
     * A mode whose target is in a later clause of its own chain has no
     * targets either when that clause has none: Forge's own check of a
     * mode's legality (603.3c, {@code CharmEffect.makePossibleOptions}) reads
     * the mode's first clause alone, so the draw reads the whole chain. Over
     * a board with no creature, "draw a card, then target creature gets +1/+0"
     * is never drawn, and the charm still has its choices in the other mode.
     */
    @Test
    void aModeWhoseLaterClauseHasNoTargetIsNeverDrawn() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card host = TestBoard.inHand(game, caster, "Grizzly Bears");
        host.setSVar("A", "DB$ Draw | NumCards$ 1 | SubAbility$ A2 | SpellDescription$ a");
        host.setSVar("A2", "DB$ Pump | ValidTgts$ Creature | NumAtt$ +1");
        host.setSVar("B", "DB$ GainLife | Defined$ You | LifeAmount$ 2 | SpellDescription$ b");
        SpellAbility charm = AbilityFactory.getAbility("SP$ Charm | Choices$ A,B", host);
        charm.setActivatingPlayer(caster);
        List<AbilitySub> choices = charm.getAdditionalAbilityList("Choices");

        Set<Integer> seen = new HashSet<>();
        for (int seed = 0; seed < SEEDS; seed++) {
            for (AbilitySub mode : drawsFor(seed).drawModes(charm).orElseThrow()) {
                seen.add(choices.indexOf(mode));
            }
        }
        assertEquals(Set.of(1), seen);
        assertTrue(new InterventionFork(null, null, null, 0L).hasChoices(charm, caster));
    }

    /**
     * Targets the AI's evaluation left on a mode do not narrow the count of
     * what the mode can reach (finding 8.3): "two target creatures controlled
     * by different players" over two creatures of one player has two
     * candidates counted alone, but with one of them left chosen the other is
     * barred by the different-controllers rule. The mode is drawable, and its
     * leftover choices come back as the same object.
     */
    @Test
    void aModeWithLeftoverTargetsIsJudgedWithNoneChosen() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card first = TestBoard.inPlay(game, game.getPlayers().get(1), "Hill Giant");
        TestBoard.inPlay(game, game.getPlayers().get(1), "Hill Giant");
        Card host = TestBoard.inHand(game, caster, "Grizzly Bears");
        host.setSVar("A", "DB$ Pump | ValidTgts$ Creature | TargetMin$ 2 | TargetMax$ 2"
                + " | TargetsWithDifferentControllers$ True | NumAtt$ +1 | SpellDescription$ a");
        host.setSVar("B", "DB$ GainLife | Defined$ You | LifeAmount$ 2 | SpellDescription$ b");
        SpellAbility charm = AbilityFactory.getAbility("SP$ Charm | Choices$ A,B", host);
        charm.setActivatingPlayer(caster);
        List<AbilitySub> choices = charm.getAdditionalAbilityList("Choices");
        AbilitySub mode = choices.get(0);
        mode.getTargets().add(first);
        TargetChoices leftover = mode.getTargets();

        boolean drawn = false;
        for (int seed = 0; seed < SEEDS && !drawn; seed++) {
            drawn = drawsFor(seed).drawModes(charm).orElseThrow().contains(mode);
        }
        assertTrue(drawn, "the mode is a legal choice");
        assertSame(leftover, mode.getTargets());
        assertEquals(List.of(first), List.copyOf(mode.getTargets()));
    }

    /**
     * A mode that counters a spell is drawn once the spell is on the stack
     * (finding 3.1): Ojutai's Command over a creature spell draws its
     * "counter target creature spell" mode, which no card of the stack
     * zone could make choosable, and the targets draw then takes the spell.
     */
    @Test
    void aModeThatCountersASpellIsDrawnWhenTheSpellIsOnTheStack() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        SpellAbility bears = TestBoard.castUntargeted(game, game.getPlayers().get(1), "Grizzly Bears");
        SpellAbility charm = TestBoard.charmOf(TestBoard.inHand(game, caster, "Ojutai's Command"), caster);
        AbilitySub counter = charm.getAdditionalAbilityList("Choices").get(2);

        boolean drawn = false;
        for (int seed = 0; seed < SEEDS && !drawn; seed++) {
            drawn = drawsFor(seed).drawModes(charm).orElseThrow().contains(counter);
        }
        assertTrue(drawn, "the counter mode is a legal choice");
        assertTrue(drawsFor(0).chooseTargetsFor(counter));
        assertEquals(List.of(bears), List.copyOf(counter.getTargets()));
    }

    /** A count uniform in [MinCharmNum, CharmNum], repeats only where allowed. */
    @Test
    void aMinimumOfZeroAndRepeatsAreHonoured() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card host = TestBoard.card(game, caster, "Grizzly Bears");
        host.setSVar("A", "DB$ GainLife | Defined$ You | LifeAmount$ 1 | SpellDescription$ a");
        host.setSVar("B", "DB$ GainLife | Defined$ You | LifeAmount$ 2 | SpellDescription$ b");
        SpellAbility charm = AbilityFactory.getAbility(
                "SP$ Charm | Choices$ A,B | CharmNum$ 3 | MinCharmNum$ 0 | CanRepeatModes$ True",
                host);
        charm.setActivatingPlayer(caster);

        Set<Integer> counts = new TreeSet<>();
        boolean repeated = false;
        for (int seed = 0; seed < SEEDS; seed++) {
            List<AbilitySub> modes = drawsFor(seed).drawModes(charm).orElseThrow();
            counts.add(modes.size());
            repeated |= new HashSet<>(modes).size() < modes.size();
        }
        assertEquals(Set.of(0, 1, 2, 3), counts, "every count between the minimum and maximum");
        assertTrue(repeated, "a charm that may repeat modes sometimes does");

        SpellAbility noRepeat = AbilityFactory.getAbility(
                "SP$ Charm | Choices$ A,B | CharmNum$ 2", host);
        noRepeat.setActivatingPlayer(caster);
        for (int seed = 0; seed < SEEDS; seed++) {
            List<AbilitySub> modes = drawsFor(seed).drawModes(noRepeat).orElseThrow();
            assertEquals(2, modes.size());
            assertEquals(2, new HashSet<>(modes).size(), "no repeat without CanRepeatModes");
        }
    }

    /** A Pawprint charm spends its budget one mode at a time and never overspends. */
    @Test
    void aPawprintCharmDrawsUntilNothingFits() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        SpellAbility charm = TestBoard.charmOf(
                TestBoard.card(game, caster, "Season of Gathering"), caster);
        int budget = 5;

        Set<Integer> spent = new TreeSet<>();
        for (int seed = 0; seed < SEEDS; seed++) {
            List<AbilitySub> modes = drawsFor(seed).drawModes(charm).orElseThrow();
            int total = 0;
            for (AbilitySub mode : modes) {
                total += Integer.parseInt(mode.getParam("Pawprint"));
            }
            assertTrue(total <= budget, "overspent: " + total);
            // Nothing fits any more: the cheapest mode costs one, so a draw
            // stops only at a budget of zero.
            assertEquals(budget, total, "stopped with budget left: " + modes);
            spent.add(modes.size());
        }
        assertTrue(spent.size() > 1, "different mode mixes reach the same budget: " + spent);
    }

    /** The fork's form sets the chosen list and chains, so a direct push resolves them. */
    @Test
    void chooseModesSetsTheChosenListAndChains() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        SpellAbility charm = TestBoard.charmOf(TestBoard.card(game, caster, "Boros Charm"), caster);

        assertTrue(new RandomChoices(new Random(1)).chooseModes(charm));
        assertNotNull(charm.getChosenList());
        assertEquals(1, charm.getChosenList().size());
        assertNotNull(charm.getSubAbility(), "the chosen mode is chained");
    }

    @Test
    void aNonCharmNeedsNoModes() {
        SpellAbility gain = AbilityFactory.getAbility(
                "AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1",
                TestCards.build("Fountain of Youth"));
        assertTrue(new RandomChoices(new Random(1)).drawModes(gain).isEmpty());
        assertTrue(new RandomChoices(new Random(1)).chooseModes(gain));
    }

    // ── targets ─────────────────────────────────────────────────────────

    /**
     * A clause that divides its amount gets it split over its targets, every
     * target a share and the shares summing to the amount: Arc
     * Lightning's resolution reads a share per target and threw on a null.
     */
    @Test
    void aDividedAmountIsSplitOverTheDrawnTargets() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        TestBoard.inPlay(game, other, "Grizzly Bears");
        TestBoard.inPlay(game, other, "Hill Giant");
        SpellAbility arc = TestBoard.card(game, caster, "Arc Lightning").getFirstSpellAbility();
        arc.setActivatingPlayer(caster);

        Set<Integer> counts = new TreeSet<>();
        for (int seed = 0; seed < SEEDS; seed++) {
            assertTrue(drawsFor(seed).chooseTargets(arc));
            int total = 0;
            for (Object target : arc.getTargets().getTargetEntities()) {
                Integer share = arc.getDividedValue((forge.game.GameObject) target);
                assertNotNull(share, "every target has its share");
                assertTrue(share >= 1, "at least one each");
                total += share;
            }
            assertEquals(3, total, "the shares make the amount");
            counts.add(arc.getTargets().size());
        }
        assertEquals(Set.of(1, 2, 3), counts);
    }

    /**
     * A fixed amount smaller than the targets the clause allows is drawn
     * over no more targets than the amount, each taking at least one
     * (601.2d): 2 damage over up to four targets.
     */
    @Test
    void aDividedAmountSmallerThanTheMaximumLimitsTheTargets() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        for (String name : List.of("Grizzly Bears", "Hill Giant", "Gray Ogre", "Craw Wurm")) {
            TestBoard.inPlay(game, other, name);
        }
        SpellAbility split = AbilityFactory.getAbility(
                "SP$ DealDamage | ValidTgts$ Creature | NumDmg$ 2 | TargetMin$ 1 | TargetMax$ 4"
                        + " | DividedAsYouChoose$ 2",
                TestBoard.card(game, caster, "Forked Bolt"));
        split.setActivatingPlayer(caster);

        Set<Integer> counts = new TreeSet<>();
        for (int seed = 0; seed < SEEDS; seed++) {
            assertTrue(drawsFor(seed).chooseTargets(split));
            int total = 0;
            for (Object target : split.getTargets().getTargetEntities()) {
                int share = split.getDividedValue((forge.game.GameObject) target);
                assertTrue(share >= 1, "seed " + seed + ": every target takes one");
                total += share;
            }
            assertEquals(2, total);
            counts.add(split.getTargets().size());
        }
        assertEquals(Set.of(1, 2), counts);
    }

    /**
     * A constraint across a clause's targets is honoured: Cannibalize
     * takes two creatures one player controls, never one of each player's.
     */
    @Test
    void aConstraintAcrossTargetsIsHonoured() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        TestBoard.inPlay(game, caster, "Grizzly Bears");
        TestBoard.inPlay(game, other, "Hill Giant");
        TestBoard.inPlay(game, other, "Gray Ogre");
        SpellAbility cannibalize = TestBoard.card(game, caster, "Cannibalize").getFirstSpellAbility();
        cannibalize.setActivatingPlayer(caster);

        for (int seed = 0; seed < SEEDS; seed++) {
            if (!drawsFor(seed).chooseTargets(cannibalize)) {
                continue;
            }
            Set<Player> controllers = new HashSet<>();
            cannibalize.getTargets().getTargetCards().forEach(c -> controllers.add(c.getController()));
            assertEquals(Set.of(other), controllers, "seed " + seed);
        }
    }

    /** "Up to two" over three candidates: a count in {0, 1, 2}, distinct targets. */
    @Test
    void targetCountIsUniformBetweenMinimumAndMaximumAndTargetsAreDistinct() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        TestBoard.inPlay(game, other, "Grizzly Bears");
        TestBoard.inPlay(game, other, "Runeclaw Bear");
        TestBoard.inPlay(game, other, "Hill Giant");
        SpellAbility pump = AbilityFactory.getAbility(
                "SP$ Pump | ValidTgts$ Creature | TargetMin$ 0 | TargetMax$ 2 | NumAtt$ +1",
                TestBoard.card(game, caster, "Giant Growth"));
        pump.setActivatingPlayer(caster);

        Set<Integer> counts = new TreeSet<>();
        for (int seed = 0; seed < SEEDS; seed++) {
            assertTrue(drawsFor(seed).chooseTargets(pump));
            int chosen = pump.getTargets().size();
            counts.add(chosen);
            Set<Object> distinct = new HashSet<>();
            pump.getTargets().getTargetEntities().forEach(distinct::add);
            assertEquals(chosen, distinct.size(), "targets are distinct");
        }
        assertEquals(Set.of(0, 1, 2), counts);
    }

    /**
     * A clause that targets a spell draws the spell on the stack, never its
     * card (findings 3.1, 3.2): the effect reads spell targets only, so a
     * card drawn there resolved into nothing. Essence Scatter takes a
     * creature spell, which no card candidate ever offered, and Negate an
     * instant, whose card it used to take.
     */
    @Test
    void aClauseThatTargetsASpellDrawsTheSpellNeverItsCard() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        SpellAbility bears = TestBoard.castUntargeted(game, other, "Grizzly Bears");
        SpellAbility bolt = TestBoard.cast(
                game, TestBoard.inHand(game, other, "Lightning Bolt"), other, caster);
        SpellAbility scatter = TestBoard.spellOf(TestBoard.inHand(game, caster, "Essence Scatter"), caster);
        SpellAbility negate = TestBoard.spellOf(TestBoard.inHand(game, caster, "Negate"), caster);

        for (int seed = 0; seed < 20; seed++) {
            assertTrue(drawsFor(seed).chooseTargetsFor(scatter), "seed " + seed);
            assertEquals(List.of(bears), List.copyOf(scatter.getTargets()), "seed " + seed);
            assertTrue(drawsFor(seed).chooseTargetsFor(negate), "seed " + seed);
            assertEquals(List.of(bolt), List.copyOf(negate.getTargets()), "seed " + seed);
        }
    }

    /** A clause that cannot reach its minimum is a failed draw, not a silent resolution. */
    @Test
    void aClauseShortOfCandidatesFailsTheDraw() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        SpellAbility bolt = AbilityFactory.getAbility(
                "SP$ DealDamage | ValidTgts$ Creature | NumDmg$ 3",
                TestBoard.card(game, caster, "Lightning Bolt"));
        bolt.setActivatingPlayer(caster);

        assertFalse(new RandomChoices(new Random(1)).chooseTargets(bolt));
    }

    /** A line that targets nothing needs nothing chosen. */
    @Test
    void aLineThatTargetsNothingIsReadyToResolve() {
        SpellAbility gain = AbilityFactory.getAbility(
                "AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1",
                TestCards.build("Fountain of Youth"));

        assertTrue(new RandomChoices(new Random(1)).chooseTargets(gain));
    }

    // ── X ───────────────────────────────────────────────────────────────

    /**
     * Feature 024 story 3 scenario 15: with five Mountains and a cost of {X}{R}, X
     * falls uniformly between zero and four — every value reached, none past
     * what the mana can pay.
     */
    @Test
    void xIsUniformBetweenTheSmallestLegalValueAndWhatTheManaCanPay() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        for (int i = 0; i < 5; i++) {
            TestBoard.inPlay(game, caster, "Mountain");
        }
        SpellAbility drain = AbilityFactory.getAbility(
                "AB$ DealDamage | Cost$ X R | ValidTgts$ Any | NumDmg$ X",
                TestBoard.card(game, caster, "Fireball"));
        drain.setActivatingPlayer(caster);

        Set<Integer> seen = new TreeSet<>();
        for (int seed = 0; seed < SEEDS; seed++) {
            drain.setXManaCostPaid(null);
            drawsFor(seed).announceX(drain, caster, 0);
            seen.add(drain.getXManaCostPaid());
        }
        assertEquals(Set.of(0, 1, 2, 3, 4), seen);
    }

    /** With nobody paying, the floor is the whole answer: the fork's one. */
    @Test
    void withNoPayerTheFloorIsAnnounced() {
        SpellAbility drain = AbilityFactory.getAbility(
                "AB$ LoseLife | Cost$ X B | Defined$ Player.Opponent | LifeAmount$ X",
                TestCards.build("Fountain of Youth"));
        new RandomChoices(new Random(1)).announceX(drain, null, 1);
        assertEquals(1, drain.getXManaCostPaid());
    }

    @Test
    void anAbilityWithNoXIsLeftAlone() {
        Player caster = TestBoard.twoPlayerGame(PhaseType.MAIN1).getPlayers().get(0);
        SpellAbility gain = AbilityFactory.getAbility(
                "AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1",
                TestCards.build("Fountain of Youth"));

        new RandomChoices(new Random(1)).announceX(gain, caster, 1);

        assertNull(gain.getXManaCostPaid());
    }
}
