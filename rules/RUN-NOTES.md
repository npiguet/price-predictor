# Rules catalogue run notes

What the scenario writer flagged and what the verifier fixed, per chunk of units. "Concerns" are mostly scenarios whose outcome depends on how Forge implements an interaction, and rules skipped as unreachable; read them before playing the scenarios in Forge. "Fixed" summarizes the verifier's changes, which are listed one by one in each scenario file's `review` array.

101 chunks, 1724 scenarios written, 729 verifier changes.

## 101 103

8 scenarios, 2 verifier changes.

**Concerns.** 103.1c and 103.6a are pre-game rules, so the schema has no field for the command zone or the opening-hand step. I put the setup in notes and used step "upkeep". For 101.2-2, the "does_not_fire" variant is a close situation rather than a true non-application of the rule.

**Fixed.** The only real error was a rule triaged as mechanical that belongs out of scope. 103.1c (Power Play choosing the starting player) can only happen in Conspiracy Draft, a chapter 9 casual variant, and it is part of starting the game. I reclassified it in rules/triage/103.jsonl and dropped its scenario, then set both files to verified; `validate 101 103` reports 0 errors.

## 104 105

12 scenarios, 4 verifier changes.

**Concerns.** 104.3f (simultaneous win and loss) is skipped: no two-player Forge game holds both outcomes pending at once. The 104.2a Platinum Angel variant follows the CR, but Forge's hasWon() may disagree because it returns false while cantWin() holds.

**Fixed.** The most common errors were empty libraries, which made a 'game continues' variant end in a deck-out or broke Karn's restart under rule 727.3. One scenario had a wrong mana count (Phage costs seven, not six). Another missed that Divine Intervention's draw is a separate triggered ability.

## 106

11 scenarios, 3 verifier changes.

**Concerns.** 106.8 to 106.11 (hybrid, Phyrexian, generic and snow mana added to a pool) are skipped because no Forge card adds mana of those symbol types. Forge may differ from the rules text in two places: Mana Reflection may also double Ashnod's Altar's mana (106.12b-1), and Drain Power first taps the target's lands (106.13-1); both are noted in forge_notes.

**Fixed.** One variant changed two facts at once (106.6-1 switched both the color chosen and the spell cast), so Raging Goblin replaced Elite Vanguard and red is now chosen in both variants. Healing Salve's mode was not fixed in the decisions, and a control variant whose restriction still applies (106.6a-1 second_creature) was mislabelled does_not_fire, so it is now polarity other.

## 107

21 scenarios, 15 verifier changes.

**Concerns.** 107.2 (an undeterminable number) and 107.17a (ticket cost) are skipped. For 107.2, Forge turns every such case into last-known information or a plain 0. For 107.17a, no Forge card has a ticket symbol in a cost. Several scenarios (107.1b-1, 107.3e-1, 107.3h-1, 107.3j-1) depend on Forge's X and clamping behaviour, and the scenario's forge_notes flags each one.

**Fixed.** The most common error was a does_not_fire variant that flipped something other than the rule under test. Two variants flipped a trigger condition instead of the X value (Lightning Bolt for Zaxara, Raging Goblin for Storm King's Thunder), one tested colored payment instead of the X = 0 treatment, and one never activated the ability at all; these were replaced with X = 0 variants or dropped. Triage had also classed 107.3b and the {C}, hybrid and Phyrexian payment rules (107.4c, 107.4e, 107.4f) as decision or definitional although each has an observable legality or resolution outcome; they were reclassified and five scenarios were added. validate 107 reports 0 errors and 4 warnings: 107.3e-1, 107.3h-1, 107.3j-1 and 107.3n-1 have no does_not_fire variant, because none of those rules has a reachable does_not_fire case, and each scenario's forge_notes says so.

## 109 110

9 scenarios, 8 verifier changes.

**Concerns.** 110.4c is skipped: no Forge script strips every permanent type without adding another, so a typeless permanent cannot be reached. 110.5c depends on Forge keeping the flipped state through a Cytoshape copy of a flip card, which I did not verify (noted in forge_notes).

**Fixed.** The most common fault was a contrast variant that changed whether the effect happened at all (the trigger fires, the cost can be paid, a creature is revealed) rather than the fact the listed rule decides. I dropped those variants or rebuilt them as minimal pairs. The other faults were a land play filed under the resolution record kind, though a land play produces no resolution record, and a triage outcome for 110.2b that said the opposite of what the rule says.

## 111

35 scenarios, 6 verifier changes.

**Concerns.** Rule 111.5's second sentence (a token copy of an instant or sorcery card is not created) has no simple Forge card, so only the "can't enter the battlefield" half is illustrated, with a forge_note saying so. Prepared Woodwork Prodigy (111.10x), Mimic Vat with nothing exiled (111.12) and the Hatchery Sliver replicate copy under Parallel Lives (111.13) may be hard for Forge to reach or may behave differently, and each has a forge_note.

**Fixed.** The most common error was a base variant with a non-empty delta, where the base setup was written as one of the variants (111.10d-1, 111.10h-1, 111.10t-1). The second was a variant pair that flipped a fact unrelated to the rule: a counterspell, the prepare mechanic, or a removal target in place of the token's own definition (111.2-2, 111.10x-1, 111.10d-1). One setup was incomplete: a multi-turn setup with empty libraries, so a player would lose on drawing. One expected outcome left out the reach that Tangled Skyline gives the transformed Phyrexian token.

## 112

8 scenarios, 1 verifier changes.

**Concerns.** 112.2a has no outcome that depends on who owns a created copy, because a copy ceases to exist when it leaves the stack. The scenario therefore shows the copy's controller (A draws from a copy of B's Divination). 112.4 and 112.3 rely on Forge's ChangeText (Sleight of Mind on a spell) and Painter's Servant's AddColor reaching the stack, which I did not check in the engine.

**Fixed.** One error: in 112.3-1, Painter's Servant set to red also makes itself a red permanent. Blue Elemental Blast's destroy mode could then target it, so the base verdict that the spell is castable did not depend on the spell's color. The scenario now asks about the counter mode's legal targets instead.

## 113

23 scenarios, 11 verifier changes.

**Concerns.** 113.7a-1 has no does_not_fire variant (validate warns): an ability on the stack always survives its source's removal, so only fires and other are possible. 113.6c is skipped because no Forge card states zones an ability does not function in, and 113.6i can't show the as-it-enters half because Forge's CantPutCounter static is limited to the battlefield.

**Fixed.** The most common problems were variants that tested a different rule from the one listed (113.2c-1's land variant, and 113.10b-1's timestamp-order and no-op variants) and setups that were incomplete: implicit loyalty counters, missing summoning-sickness flags, decisions that re-cast a spell already on the stack, and one Fog Patch with too little mana to cast. Rule 113.12 was also misclassified as definitional even though it has an observable outcome, so it now has a new Muraganda Petroglyphs scenario.

## 114

3 scenarios, 0 verifier changes.

**Concerns.** Setups that start with an emblem or loyalty counters rely on the "counters" key (loyalty) and setup notes, since the schema has no emblem zone; 114.4-2's second variant reads the opponent's creature rather than changing a board fact.

**Fixed.** I found no errors in unit 114. The Forge scripts confirm the card facts for Gideon, Ally of Zendikar (-4 emblem, Creatures.YouCtrl +1/+1), Chandra, Torch of Defiance (-7 emblem that triggers on spell cast), Stifle (it can target an emblem's ability), Shock and Grizzly Bears. The expected outcomes follow the state-based check and the order the stack resolves in. The one weak spot: the does_not_fire variant of 114.4-2 (the static ability reading B's creature) checks the ability's wording, not the command-zone rule itself. I accepted it because 114.4 has no real negative case. The file is marked verified and validate reports 0 errors.

## 115

16 scenarios, 3 verifier changes.

**Concerns.** 115.9a, 115.9b and 115.9c are the weakest: the triage expects trigger records, but only Spinerock Tyrant (9a) and Mirrorwing Dragon (9c) are real triggers, while 9b is shown through Rebuff the Wicked's target restriction as a resolution record. In 115.9c Forge may copy a two-target Agony Warp with only the first target set, so the copy's outcome is flagged in forge_notes.

**Fixed.** The worst error was in 115.8-1: its contrast variant used a new target that no mode of Izzet Charm could take, so it never tested that retargeting keeps the chosen mode; I rebuilt the scenario around Rakdos Charm. Two smaller fixes: 115.7f-1 had a variant marked does_not_fire where the rule does not apply (now marked other), and 115.7a-1 said a destroyed creature's survivor was "undamaged" (now "stays on the battlefield"). validate 115 gives 0 errors and one accepted warning: 115.7f-1 has no does_not_fire variant because a division cannot be changed, so none can exist.

## 116

11 scenarios, 3 verifier changes.

**Concerns.** 116.2c and 116.2k are the weakest. For 116.2c, only the Licid half is illustrated; no Forge card covers stopping a delayed trigger, and Forge may implement the Licid revert on the stack. For 116.2k, rule 116.2k and 702.170a disagree on timing, so I left out the upkeep case.

**Fixed.** I fixed three problems in unit 116. One variant changed two facts at once (116.2d-1 not-paid also changed which lands were tapped). One variant left out the observable result (116.2b-1 opponent-tries did not say the face-down creature dies). One variant took no special action, so it did not test the rule at all (116.3-1 illegal-land, now dropped). Validate reports 0 errors and one warning: 116.3-1 has no does_not_fire variant, because the rule applies after every special action, and the scenario's forge_notes say so.

## 117

11 scenarios, 5 verifier changes.

**Concerns.** 117.3a and 117.3b (who receives priority) have no field in today's records, so those scenarios assert priority holders that the harness may need new fields to expose. The untap-step variants (no priority) may not be reachable as Forge puzzle states.

**Fixed.** Two timing rules were filed as definitional even though they have outcomes. 117.1c is the legality of special-action timing, now skipped because unit 116's scenarios cover it. 117.3c is "the caster receives priority again", now mechanical with a new scenario, 117.3c-1. One variant's delta gave a player an action while they did not hold priority (117.3b-1 mana-ability).

## 118

20 scenarios, 10 verifier changes.

**Concerns.** Skipped 118.7d-g (no Forge card reduces by colorless, hybrid, Phyrexian or snow symbols) and 118.8c (no card forces a cast with a mandatory hidden-zone additional cost). The 118.7c, 118.6a, 118.9a and 118.9d scenarios depend on Forge applying stacked or free-cast cost modifiers the way the rules do, flagged in forge_notes.

**Fixed.** The most common error was a base variant that carried a non-empty delta, with the choice under test written into the delta instead of the decisions. This was fixed in six scenarios: 118.8d-1, 118.9a-1, 118.9c-1, 118.12-1, 118.12a-1 and 118.12b-1. The other errors were a scenario that illustrated the wrong sibling rule (118.7c-1 really showed 118.7b; it is now rebuilt on Eluge's single {U}{U} reduction) or the wrong permission (118.14-1 used Chromatic Orrery's 'any color' wording; it now uses Vizier of the Menagerie's 'any type'), a minimal pair that flipped on land color rather than on the rule (118.7a-1), and a 118.3b-1 variant that let a 0-life player draw a card before the state-based check ended the game.

## 119

12 scenarios, 4 verifier changes.

**Concerns.** 119.8 has no pure can't-lose-life card in Forge, so Platinum Emperion (life total can't change) stands in. The 119.4 and 119.4b playability scenarios depend on Forge exposing Phyrexian mana and X-life payments as legal or illegal verdicts rather than hiding the option.

**Fixed.** The main problems were weak minimal pairs. One variant had the same outcome as its base, so I dropped it. One scenario had no does_not_fire variant, so I relabelled a variant from "other" to does_not_fire. I also added a missing decision that resolves a trigger, and a variant for the "0 or less" half of 119.6, which nothing had tested.

## 120

24 scenarios, 1 verifier changes.

**Concerns.** The clause in 120.4a that damage above 1 from a deathtouch source is excess is not illustrated, because the card pool has no vanilla deathtouch trampler for Ram Through (noted in forge_notes of 120.4a-2). The battle (120.3h-1) and Spirit Link on an opponent's creature (120.9-1) may be hard for Forge to set up directly.

**Fixed.** The only real gap was coverage: 120.4a-2's forge_notes claimed the deathtouch clause of 120.4a could not be illustrated, so I added 120.4a-3 (Ram Through from Argothian Swine equipped with Gorgon's Head into Hill Giant, with a variant where the equipment is unattached). The other 23 scenarios matched the Comprehensive Rules and the Forge scripts (P/T, keywords, Flame Spill and Ram Through ExcessDamage, Magmatic Galleon's noncombat excess trigger, Invasion of Dominaria with Defense:5), so they needed no changes. C:/Users/nicol/IdeaProjects/price-predictor/.claude/worktrees/gen3-rules-coverage/rules/scenarios/120.json is now marked verified and validate reports 0 errors.

## 121

16 scenarios, 5 verifier changes.

**Concerns.** 121.7 (unreplaced parts of a replaced event) is illustrated with Swans of Bryn Argoll plus a trampling attacker, where the unreplaced damage to the player is a separate Forge event, so the fit is approximate. 121.8 (face-down card drawn mid-cast) relies on Shadowblood Egg, and Forge likely does not model the face-down state.

**Fixed.** The fixes were mostly to expected outcomes that skipped part of what the rules then do. One variant had B ending at 19 life, but B loses at the state-based check before the Underworld Dreams trigger can resolve. Another said the spell "cannot be cast" without the 733.1 reversal, under which Magmatic Insight returns to hand but the Egg's sacrifice and draw stand. The rest were bookkeeping slips: hand counts that left out an uncast spell, a delta that repeated what the base already did, a missing card in the cards list, and a forge note that contradicted the decisions.

## 122

19 scenarios, 2 verifier changes.

**Concerns.** 122.6a (the controller puts the counters) can only be shown through Vorinclex, Monstrous Raider and Forge may not key its replacement on the entering permanent's controller. 122.7 only exercises single-counter crossings, and its 12-to-13 variant needs a Midnight Clock sitting at 12 counters, which arises only if the twelfth-counter trigger was countered earlier.

**Fixed.** The setup contradicted the expected outcome: the library order meant the rad trigger milled a land, not the nonland cards the expected text named. The Siege defeat trigger's optional cast was left open, and the variant never stated that the trigger exiles the battle.

## 201

4 scenarios, 6 verifier changes.

**Concerns.** 201.2c is skipped: I found no Forge card that checks whether a nameless object has a different name than a group. The 201.2b scenario relies on Forge counting a face-down Conspiracy-typed Demon as nameless in Liliana's Contract, which I could not confirm and noted in forge_notes.

**Fixed.** The most common problem was a variant whose delta changed a fact the listed rule does not decide: 201.5a-1 swapped a trigger's controller condition, so it was rebuilt as a one-fact pair (which Gutter Grime created the Ooze: a 1/1 that dies against a 3/3 that survives). There were also mislabeled polarities and wrong outcome text, such as 'removed from combat' when no combat takes place. One triage line was misclassified: 201.2a was marked definitional although its nameless-object clause decides Guardian Project's trigger, so it is now mechanical with a new scenario, 201.2a-1.

## 202 204

16 scenarios, 2 verifier changes.

**Concerns.** Forge's handling of mana value for transformed, melded and copied-melded permanents (202.3b, 202.3c) and for a split card in the library under Counterbalance (202.3d) is unconfirmed, and the transformed and melded states must be set up as puzzle states, which is noted in forge_notes. The no-mana-cost rule (202.3a) is shown through the revealed Dryad Arbor, not a suspended spell cast for free.

**Fixed.** Variants that were not true minimal pairs: one changed two facts at once (spell and target) and had the wrong polarity, and one scenario lacked a does_not_fire variant that isolates the rule (hybrid symbols counting their largest component rather than the mana paid). Card facts, mana, timing and the expected outcomes were otherwise correct against the Forge scripts and the CR.

## 205

9 scenarios, 4 verifier changes.

**Concerns.** 205.1a's clause that an instant or sorcery keeps its type is not illustrated: no Forge card sets the card type of an instant or sorcery. 205.3d relies on Forge dropping changeling's all-creature-types flag when Song of the Dryads removes the Creature type, which I did not verify in the engine.

**Fixed.** The polarities in 205.1b-1 were the wrong way round: the variant where prior types are not kept was labelled other, and the variant with no type-changing effect at all was labelled does_not_fire. One expected text gave a reason instead of only the outcome, one mana payment was left unspecified, and 205.1a's rule that counters stay on an object whose card type is set had no variant, so I added one. Validation reports 0 errors.

## 208 209 210

8 scenarios, 4 verifier changes.

**Concerns.** 208.5 (a creature with no power or toughness value) is skipped because every Forge creature-making effect also sets P/T. In the 208.4b-1 copy variant, the "Curie became a copy of Juggernaut" state cannot be expressed in the setup schema and is described in notes only.

**Fixed.** The most common error was a variant that changed more than one fact, or changed a fact without changing the outcome (208.2a opponent_swamp, 208.2b countered, 208.4b copy variant). Other errors were setups that missed a fact needed for the rule or for the game to continue: 209.2 had empty libraries, so the Jace +2 variant would end the game by decking, and the 208.4b copy variant had no exiled Juggernaut, so it could not be reached. All three files are now verified and validate reports 0 errors.

## 300 301 302

27 scenarios, 12 verifier changes.

**Concerns.** 301.5b and 301.5e are hard to reach through untargeted effects: the Brass Squire scenario (301.5b-2) makes the creature target illegal via protection, so Forge may report a fizzle rather than a refused attachment. The Unfinished Business scenario (301.5e-1) depends on Forge honouring protection from artifacts when it returns an Equipment attached.

**Fixed.** The most serious errors were confounded mechanisms. In 300.2a, Dryad Arbor's missing mana cost already explains why it cannot be cast, so I added Omniscience to remove that explanation. In 301.5b-2, protection from artifacts made the target illegal (608.2b) rather than showing the 'cannot be equipped' clause, so I rewrote it with Sigarda's Aid. The rest were a variant whose expected outcome did not follow from its delta (301.5c-3), wrong polarity labels, declare-attackers steps used with no attackers, and 300.2 being triaged as definitional when it is a targeting legality rule (reclassified, and scenario 300.2-1 added).

## 303 304

16 scenarios, 7 verifier changes.

**Concerns.** Three skips: 303.4d (no Forge card makes an Aura a creature or lets it enchant itself), 303.4h (every AttachedTo$ script selects only Auras, Equipment or Roles) and 304.4 (no script was found that puts an unrestricted card onto the battlefield). The 303.4j scenario needs an illegal attach choice (Black Knight) that Forge may not offer, and 303.4g depends on how Forge's Replenish handles an Aura with nothing to enchant; both are noted in forge_notes.

**Fixed.** The most common error was a wrong skip reason: 303.4d can be reached in Forge with Mycosynth Lattice plus March of the Machines, and 304.4 with a manifested instant flickered by Cloudshift, so I added scenarios for both and removed the skips. Two scenarios did not exercise the rule they listed. Aura Graft only lets you pick a permanent the Aura can legally enchant, so 303.4j now uses Enchantment Alteration. Retether is a 303.4g case, so 303.4i now uses Academy Researchers. 304.5-1 also had setup notes that contradicted its own decisions.

## 305 306

19 scenarios, 3 verifier changes.

**Concerns.** 306.5b has no clean does_not_fire variant (the entering replacement always applies), so it uses fires plus an "other" Doubling Season variant. Legality scenarios use fires for the rule's own effect (illegal for restrictions, legal for 305.1), which is not uniform across the catalogue.

**Fixed.** The most common problem was a variant that changed more than one fact: 305.3-1 changed both the active player and the stack contents. I also fixed a delta that contradicted the setup's loyalty counters (306.5d-1) and an unclear base decision about whether the opponent responds (306.2-1). Both files are now verified, and `python scripts/rules_catalogue.py validate 305 306` reports 0 errors.

## 307 308 309

14 scenarios, 8 verifier changes.

**Concerns.** 309.2d (an effect other than venturing bringing a dungeon into the game) is skipped because no Forge card can do it, and 309.2c's "cannot leave the command zone" half has no testable card. Several scenarios (dungeon marker position, Archon's chosen type, a pre-filled stack, the Commander-format 309.2c) depend on setup Forge may only reach through puzzle files or forks, and I did not verify that Forge offers a Kindred Sorcery as a choice in Harald's chapter I (307.4).

**Fixed.** The most common problem was a setup that can never happen in a game: three 309 scenarios started with the venture marker on the bottommost room and nothing on the stack, but rule 309.6 removes that dungeon before any player gets priority. I rebuilt 309.5b-1 so the second venture (Immovable Rod untapped by Twiddle) is made while Temple of Dumathoin's room ability is still on the stack, made the reachable variant the base of 309.7-1, and dropped 309.5a-1, folding 309.5a into 309.5b-1. The other fixes were variants that did not form minimal pairs (307.5a-1 changed two facts; 309.2c-1's contrast flipped nothing), plus a wrong skip reason for 309.2d, missing creature-stat observables in 309.6-1, and playability added to triage 308.1's record kinds; `validate 307 308 309` reports 0 errors.

## 310

11 scenarios, 2 verifier changes.

**Concerns.** 310.9d and 310.9g are weakly illustrable: in a two-player game a Siege's protector always equals the attacked player, so the 310.9d variants give the same life totals, and 310.9g is skipped as unobservable. The Doubling Season variant in 310.4b-1 may not match Forge's implementation, and I flagged that in forge_notes.

**Fixed.** One variant described the board while a trigger was still on the stack, even though its decisions let that trigger resolve. Another variant flipped a neighbouring rule (a tapped creature can't block, 509.1a) when it should have flipped the rule under test (310.9c: only the protector may block). Two warnings remain and are left on purpose: 310.4b-1 has no does_not_fire variant because a battle always enters with defense counters, and 310.9d-1 has none because a Siege's protector always differs from its controller in a two-player game.

## 400

15 scenarios, 6 verifier changes.

**Concerns.** 400.7c (Circle of Protection: Red choosing a spell on the stack) and 400.7k (madness with Containment Construct) may not match Forge's implementation, so both carry forge_notes. 400.7h and 400.7i have no Forge card whose later effect part acts on the cast spell or the played land, so they are skipped. 400.8, 400.9, 400.10, 400.11b and 400.11c are skipped as unreachable in the collected two-player games, and 400.4b and 400.7m for the same reason.

**Fixed.** The main errors were wrong rules: 400.4a-1 used Tezzeret's face-down -7, which does put instants and sorceries onto the battlefield, and the 400.7h skip was wrong because Torrential Gearhulk covers that rule. The rest were weak contrasts: some variants did not test the rule they claimed (400.6-1, 400.7e-1). There were also small expected-outcome and note mistakes (Tormod's Crypt left out of the graveyard in 400.12-1, and a wrong color in the 400.7f-1 note).

## 401 402 403 404 405

24 scenarios, 8 verifier changes.

**Concerns.** 401.6 (a revealed top card becoming a new object) is skipped: no Forge card makes the old and new object differ on the board. 401.5 is only observable as reveal timing, which Forge may not model and no existing playability field captures.

**Fixed.** The most common errors were expected outcomes that left out consequences: a sacrificed Mountain going to the graveyard, a player at 1 life dropping to 0 and losing, and the end result after a response. Two setups activated Prodigal Pyromancer without saying it was free of summoning sickness. One polarity was wrong: a sacrificed token, which does reach the graveyard under 404.1, was labelled does_not_fire, so a Leyline of the Void variant was added as the real does_not_fire case.

## 406 407 408

7 scenarios, 4 verifier changes.

**Concerns.** 406.7 (an already-exiled card exiled again) is skipped: a search of Forge's cardsfolder found no script that moves a card from exile to exile. 407.4 needs a game played for ante, which collection decks never use, so it relies on a hand-built state (noted in forge_notes).

**Fixed.** The most common error was a contrasting variant whose outcome changed for a reason outside the listed rule: casting timing, the source card's own one-spell limit, or a loyalty cost that couldn't be paid. I fixed those by rebuilding the variants around the rule itself: a face-up versus face-down morph cast for 406.3a, Vivien's creature-only permission for 406.3b, and Stifle countering the emblem ability for 408.2. 407.4's rule that only an object's owner can ante it was never tested, so I added a Jeweled Bird scenario that contrasts an owned Bird with one stolen by Steal Artifact.

## 500 502 503 504

22 scenarios, 10 verifier changes.

**Concerns.** 500.10a is skipped because Obeka's "you get" trigger cannot reach a turn other than its controller's in a two-player game. 500.7 and 500.8 cannot show that the most recently created extra phase comes first, since two identical Relentless Assaults give the same phase order. 500.6-1 and 500.6-2 also assume a state-based setup (day/night history, phased-out start) that the schema has no field for, so those are fixed in notes.

**Fixed.** The most common fix was setup that could not finish: several unit 500 scenarios ran into a draw step with an empty library, so a player would lose before the outcome could be observed. I also fixed minimal pairs that broke the schema: a base variant with a non-empty delta (503.2-1), a variant that changed two facts at once (504.1-1), and a contrast that only moved the observation point (500.4-1). The last kind was decisions left open, such as a trigger order or a timing.

## 505

6 scenarios, 2 verifier changes.

**Concerns.** 505.5 (Attraction visit roll) needs a fixed die result and a specific Attraction variant, so Forge may not reach it without a test hook. 505.4's trigger record kind is approximate, because the lore counter itself is a turn-based action and only the chapter trigger reaches the stack.

**Fixed.** One error was a polarity mix-up: in 505.5-1, a roll that misses the lights was marked as the case where the rule does not apply, but the roll still happens there. The case with no Attraction is the real one. The other was a decisions list in 505.6b-3 that did not match one of its variants. Card facts, rules outcomes and triage were all correct.

## 506

26 scenarios, 20 verifier changes.

**Concerns.** 506.3d-1 has no does_not_fire variant (validate warns): the only contrast, a creature entering attacking before blockers, needs a second card and would change more than one fact. The skips (506.3b, 3c, 3e, 3f, 3g, 4d, 4e) rest on my grep of Forge's cardsfolder for put-attacking and put-blocking effects, so a missed card would invalidate them.

**Fixed.** The most common fault was a base variant with a non-empty delta, often with a setup step that came after the attack decision. I moved the base choice into the decisions, and each other variant now changes one fact. Two scenarios lacked a real minimal pair (506.3a-1 changed three facts; 506.3d-1 had no does_not_fire variant and a draw from an empty library), and 506.8c-1 had its polarities inverted. The 506.3c skip was false (ninjutsu into a planeswalker that died in response illustrates it), so I replaced it with the new scenario 506.3c-1.

## 508

26 scenarios, 12 verifier changes.

**Concerns.** 508.4a and 508.4b are skipped: they need a creature entering attacking a defender that has since left the game or the battlefield, which two-player Forge play with real cards does not reach. 508.4d-1 has no fires/does_not_fire pair that isolates the rule, so it uses ninjutsu with "other" and "does_not_fire" contrast variants, and the Propaganda and Portal Mage interactions (508.1d-3, 508.7b-1) depend on how Forge handles optional attack costs and reselection.

**Fixed.** The most common errors were variants that changed two facts at once, or whose contrast did not flip anything observable (508.7b with Propaganda, 508.1k's observation-time delta, 508.8's extra block). Card facts were also wrong: Shock cannot kill the 3/3 Crypt Cobra, and tapped creatures were described as 'starting the turn tapped', which the untap step undoes. One skip was invalid: 508.4a is reachable through ninjutsu (702.49c) plus removing the attacked planeswalker, so scenario 508.4a-1 now covers it.

## 509

21 scenarios, 5 verifier changes.

**Concerns.** Forge may not fire the "becomes blocked by a creature" trigger for a token put onto the battlefield blocking (509.3d), and may fire the single "blocks" trigger once per blocked attacker (509.3a). The menace plus must-block interaction (509.1c-2) also depends on Forge's requirement validation; I flagged all three in forge_notes.

**Fixed.** The most serious error was a card-control mistake: in 509.3e-2 the defending player controlled Seifer, Balamb Rival, so its trigger could never fire for the attacker. In 509.3g-1 a variant was not a minimal pair because it changed two facts from the base. Smaller fixes covered a wrong polarity label on 509.4-1, a damage assignment the 509.3e-1 variant no longer matched, and an illegal-declaration outcome in 509.1b-3 that never said the declaration was undone.

## 510 511 513 514

17 scenarios, 5 verifier changes.

**Concerns.** 510.1c/510.1d's divided-damage branch and 510.1b's "attacking nothing" case (Forge may redirect or drop the attacker) are the least certain, noted in forge_notes. 513.2 rests on Forge not re-collecting a Phase delayed trigger created during the end step, also noted; 511.3 is shown only through a Condemn playability verdict.

**Fixed.** The most common error was a setup that ran into the opponent's turn with an empty library for B. B would lose the game at their draw step before the rule could be observed (513.2-1, 513.2-2, 514.2-1). The other two fixes were decisions left open: a cleanup discard choice that was never fixed (514.3a-1), and a misstated priority order in which the non-active player was said to hold priority first (511.3-1).

## 601

24 scenarios, 4 verifier changes.

**Concerns.** Several scenarios (601.2-1, 601.2e-1, 601.3e-1, 601.6-1, 601.6a-1) depend on Forge never offering an illegal cast or on Forge modelling Adventure halves in library MayPlay, X-dependent prohibitions or Altar sacrifice payments, so they may be unreachable as written (flagged in forge_notes). 601.3f is skipped because no Forge card grants cast permission without look permission.

**Fixed.** One setup left a decision open: Painter's Servant asks for a color as it enters, so it was swapped for the vanilla Bronze Sable. One scenario's steps contradicted each other: Giant Killer was cast and then Chop Down was cast from the same library card, so both are now checked from the same game state. One scenario showed a different mechanism than its rule (601.8): it is rebuilt so Shock is already on the stack when Sphere of Resistance is cast, and a skip reason that called Gonti the only such card was corrected.

## 602

10 scenarios, 2 verifier changes.

**Concerns.** 602.5e has only mana-ability cards in Forge (Charmed Pendant), so the scenario shows the cannot-pay-during-cost-payment restriction rather than a stack-using ability. 602.5c depends on Forge keeping a separate once-only count per Touch of Vitae copy, and 602.1e on Forge applying the Suppression Field increase before offering Heirloom Epic's creature taps.

**Fixed.** A scenario (602.5c-1) had no base variant because its first variant carried a delta, which I restructured into a proper base plus a one-fact flip. Another expected outcome (602.5e-1 during_payment) asserted a playability verdict for a different spell that the rules do not decide, which I narrowed to the Pendant activation being illegal and the cast being reversed. Card facts, triage and coverage otherwise checked out against the Forge scripts and CR 602/601/605.

## 603

32 scenarios, 11 verifier changes.

**Concerns.** Skipped 603.1a, 603.1b, 603.2f, 603.7f, 603.7g, 603.7h, 603.9 and 603.10f: no Forge card implements the first six, and the last two need a lose-the-game trigger that cannot go on the stack in a two-player game. Several scenarios depend on Forge details worth checking: Felidar Sovereign's re-check on resolution, how Sneak Attack schedules its sacrifice when activated in the end step, Living Plane applying its type change as the land enters, Jackal Pup's trigger when it dies to the damage, and the phased-out Teferi's Imp setup.

**Fixed.** The most common errors were broken setups: B's library was empty in scenarios that pass through B's draw step, so B would lose the game, and one decision list never resolved a trigger the expected outcome depends on. Next came wrong rule mapping and non-minimal pairs: Morbid Opportunist's "triggers only once each turn" was used for 603.2h's "Do this only once each turn" (rewritten with Ancient Cornucopia), and one variant changed two facts (a bounce and a recast, replaced by a single Cloudshift flicker). Two skip reasons did not hold: 603.9 and 603.10f were reclassified out_of_scope, and the 603.7g reason falsely said no Forge card has that shape.

## 604 605 606

15 scenarios, 7 verifier changes.

**Concerns.** 605.3c needs Forge to refuse letting Initiates of the Ebon Hand pay its own {1} cost, and 606.5/606.6-2 depend on Forge netting Carth the Lion's extra AddCounter cost against the SubCounter part; both are in forge_notes. 604.6 uses Berserk, which Forge implements as an ActivationPhases spell restriction and not as a static ability.

**Fixed.** The most common faults were expected outcomes that stopped too early and left out state-based results: a planeswalker at 0 loyalty still on the battlefield, a missing dies trigger, and an unresolved pump. Other faults were variants that changed two facts at once or flipped the outcome by a different mechanism than the listed rule, and one variant that described an illegal cast still tapping a land and dealing damage. The 605.3c scenario turned on an unpayable cost rather than on the no-reactivation rule, so it was dropped and the rule was skipped with a reason.

## 607

14 scenarios, 5 verifier changes.

**Concerns.** 607.2f is skipped: Forge's only rules-less-word choosers are anchor-word cards, which 607.2m-1 covers. 607.5, 607.3 and 607.2q depend on how Forge tracks ExiledWithSource across copied triggers, acquired abilities and cost exiles, and are flagged in forge_notes.

**Fixed.** The most common fault was a contrast variant that never reached the linked ability: 607.2a-1's "no ring leaves" case did not test the rule, and Stifling the first ring's trigger now replaces it. One setup was impossible: Jeweled Amulet's first ability taps it, so 607.2e-1 is rebuilt across two turns and uses Scorching Shot, so the noted mana type, not the land, decides whether the spell can be cast. 607.2f was skipped as covered by 607.2m-1 but that scenario did not list it, so it is now listed there and the skip is gone.

## 608

19 scenarios, 7 verifier changes.

**Concerns.** Skipped 608.2k, 608.2m and 608.2p because no simple Forge-reachable card illustrates them (no "when this resolves" trigger event, no mid-resolution departure from the stack, no instant-speed type removal). The 608.3e (Clone with Worms of the Earth), 608.3d (bouncing the mutate target) and 608.2a (re-check of the intervening-if condition on resolution) scenarios depend on Forge implementing those interactions as the rules describe, and are flagged in forge_notes.

**Fixed.** Most fixes were variants labelled does_not_fire although the rule never applied in them (a 603.4 trigger that never fired, a 608.2b fizzle inside a 608.2h scenario); those are now labelled other. Two skips were false: Forge does have a trigger for an ability resolving (Mode$ AbilityResolves on Narci, Fable Singer), and 608.2k can be shown with ordinary cards (Heliod plus Wall of Tears), so both rules now have scenarios. 608.2j-2 checked whether a target was legal rather than what an effect does, so it was rewritten around Hellfire; validate shows 0 errors and 2 warnings, which are intended because 608.2h and 608.2k have no does_not_fire case.

## 609 610

22 scenarios, 8 verifier changes.

**Concerns.** Several 610.x scenarios (3, 3c, 3d, 4, 4a, 4d) need an until-link between a card in exile or phased out and its source, which the zone lists cannot express, so the setup notes and forge_notes say how to build the state by play. The 609.7a-2 permanent-spell shield, 609.4b-2 hybrid payment through Chromatic Orrery, and 610.2-1 end-step variant depend on Forge implementation details I could not confirm from the scripts.

**Fixed.** Most of the fixes removed rules reasoning from the "expected" fields, which the README says must state only the observable outcome. Two scenarios (609.4b-2 and 610.3c-1) had no does_not_fire variant because their contrast variant was labelled "other", so I relabelled it does_not_fire. The rules, the card facts from the Forge scripts, the mana and timing, and the triage all checked out, and `python scripts/rules_catalogue.py validate 609 610` now reports 0 errors.

## 611

11 scenarios, 5 verifier changes.

**Concerns.** 611.3d is skipped because no Forge script combines MayPlay with a granted battlefield-only ability on the same object. Scenarios 611.2c-2 (Holy Day) and 611.2d-1 (Chorus of Might) have no does_not_fire variant, since the contrast is a different branch, so they carry "other" and validate warns.

**Fixed.** The most common problems were variants that changed more than one fact or tested a different rule (611.2a-2's new-object variant really illustrated 400.7, not 611.2a), and wording that was ambiguous or showed an impossible sequence (an unclear "it", passing with an empty stack before the board is read, a stray "instead of Purelace"). The 611.3d skip was wrong: Henzie "Toolbox" Torre gives creature spells blitz, whose battlefield-only parts last after Henzie leaves, so I added scenario 611.3d-1. Validate shows 0 errors; two warnings remain because 611.2c-2 and 611.2d-1 have no does_not_fire variant, which these rules cannot produce.

## 612

9 scenarios, 4 verifier changes.

**Concerns.** Forge may not honor several of these, and each has a forge_notes entry. The open questions are whether Forge applies text changes to keywords granted by a static ability (612.3), whether a token's name follows its changed subtype (612.2a), and whether Echoing Truth's sharesNameWith counts Spy Kit's added names (612.7). Rule 612.9 is out of scope in triage, so it has no scenario.

**Fixed.** The main errors were in token names: under rule 111.4 an unnamed token is called "<subtypes> Token", and a text change to an existing token changes its subtype but not its name, so 612.2a-1 and 612.4-1 were corrected. Other fixes: one expected outcome left out a state-based consequence (in 612.8-1 the Aura goes to the graveyard when its creature leaves), and in 612.7-1 the Equipment's attachment was written only in the notes instead of as attached_to in the setup.

## 613

23 scenarios, 7 verifier changes.

**Concerns.** 613.7n (an entering object's own static ability versus an effect that sets its characteristics) and 613.8c (dependency re-evaluation) are skipped because no simple Forge cards produce them. Several scenarios (dependency, keyword-counter timestamps, copying a face-down permanent, timestamps on turn-face-up and transform) rely on Forge matching the rules, and each carries a forge_notes warning.

**Fixed.** The most common errors were contrasts that mixed two mechanisms or never exercised the rule they listed. Turn to Frog was used as a pure color change, Clone copying a face-down permanent was filed under sublayer order, and choosing Land for protection made the type-changing layer irrelevant. The rest were setup gaps: a timestamp order the outcome depends on was missing, the 613.6 "same set of objects" sentence had no scenario, and the base decisions included the variant's actions.

## 614

29 scenarios, 11 verifier changes.

**Concerns.** 614.10b (skip, then an action that is first in the next step that occurs) is skipped: no Forge script has that pattern. 614.13c (Ashiok with Breeding Pool) and 614.7 (regeneration against indestructible) depend on Forge ordering that I assumed and did not test; both are noted in forge_notes.

**Fixed.** The most common error was a base variant with a non-empty delta, where the base's own action was written as a change (614.7-1, 614.12-1, 614.17a-1, 614.17c-1, and the old Avizoa scenario); I moved those actions into the base decisions. Some variants did not test the rule they were filed under. 614.12-2 had a variant whose outcome matched the base, 614.9-1's giant_gone removed the redirection effect's own source, and Avizoa's skipped untap step is not something "scheduled for" a skipped step. I replaced these with a Daughter of Autumn recipient-gone scenario (614.9-2) and an Eon Hub plus Phyrexian Arena scenario (614.10a-2). The 614.10b skip reason was also wrong, because Time Vault does act after a skip; I reworded it to say the timing cannot be observed.

## 615 616

17 scenarios, 33 verifier changes.

**Concerns.** rules/triage/615.jsonl and 616.jsonl do not exist, so validate checks no coverage for these units; I judged which rules are mechanical or legality myself and covered or skipped every rule except purely definitional ones. Forge may not reproduce the rules outcome in 616.1c-1 (Essence of the Wild vs Rusted Sentinel enters-tapped), 616.1g-1 and 616.2-1 (Pariah plus Daunting Defender chain); each has a forge_notes entry.

**Fixed.** The triage files rules/triage/615.jsonl and 616.jsonl were missing, so validate had silently skipped coverage checks for both units; I wrote both. Several decisions were wrong: in one, A cast Fog in the declare blockers step before blockers were declared; in others, an order the rules impose (616.1a, 616.1c, 616.1g) was offered as a player's choice. Some polarities and minimal pairs were also wrong: two 615.8-1 variants each changed two facts, and polarities were mislabelled in 615.11-1 and 616.1-1.

## 700 701.2

21 scenarios, 4 verifier changes.

**Concerns.** 700.5a is weakly illustrated: Altar of the Pantheon shows that a devotion modifier counts, but no practical card pair shows the copy/control/text-change ordering. 700.3c can only be observed through final zones, since nothing can read the piles mid-split, and 700.10's clause that the permanent counts even after losing the ability is not exercised.

**Fixed.** The most common error was a scenario whose variants never made the rule's distinctive outcome observable. In 700.3c every variant ended the same whichever zone the piles were in, and in 700.12a the variants only showed the outlaw definition (700.12), not the permanents-only restriction. Smaller problems were a minimal pair that changed two facts (701.2a-2) and a 700.2d scenario with no variant where the mode is not repeated.

## 701.3 701.4 701.5 701.7 701.8 701.9

16 scenarios, 3 verifier changes.

**Concerns.** 701.3b's branch for a non-attachable object and 701.9c's illegal-cost branch (Library of Leng replaces only effect-caused discards) have no reachable card, so they appear in forge_notes only; 701.4b, 701.7b and 701.3d rely on Forge implementing hidden-zone and timing details that forge_notes flag as unconfirmed.

**Fixed.** I checked all six units against the rules and the Forge scripts and made three fixes; all six files are marked verified and validation passes with 0 errors. One variant flipped its outcome through a card's own condition rather than the rule it illustrates: in 701.7b-1, moving Grumgully to B's control was replaced by removing March of the Machines. The other two fixes were incomplete decisions: 701.3b-1 had no priority pass for the becomes-unattached trigger to resolve, and 701.3a-1 named an attach choice that does not exist in its protected-host variant.

## 701.6 702.2 702.19

15 scenarios, 5 verifier changes.

**Concerns.** 702.2d (deathtouch damage from a non-battlefield zone) has no simple Forge card to show it, so it is skipped with a reason. The 702.19e and 702.19f scenarios assume Forge keeps a creature attacking after its planeswalker leaves; forge_notes flags that Forge may instead remove the creature from combat.

**Fixed.** The most common error was a variant that changed two facts at once: it added a card to the setup and also declared that card as a blocker. I fixed these by putting the card in the base setup without blocking. The other errors were polarity labels: does_not_fire was used where the rule still applies (702.19f-1) or where the rule has no does_not_fire branch at all (701.6b, since costs are never refunded). 702.19f-1 also gained a Thrasta variant so it has a real does_not_fire branch.

## 701.10 701.11 701.12 701.13

23 scenarios, 5 verifier changes.

**Concerns.** Validate passes with 0 errors, but 701.10g-2 and 701.12d-1 have no fires/does_not_fire pair (warnings only). 701.12d's ownership clause can't be shown with Forge's ExchangeZone cards, so 701.12d-1 covers only the empty-zone half; the Everlasting Torment can't-gain-life variants and the tiered Tifa's Limit Break are the cases most likely to behave differently in Forge.

**Fixed.** The worst error was in the life-exchange variants (701.12c-1 and 701.12g-1): they applied half an exchange when a player could not gain life. Under 701.12a nothing happens in that case, and Forge's LifeExchangeEffect and LifeExchangeVariantEffect do the same, so both expected outcomes now change nothing. The other fixes: a mana payment left open (Doubling Cube could have spent the {R} in the pool), a scenario with no does_not_fire variant (701.10g-2), and the 701.12d ownership clause not being exercised, now covered by the new scenario 701.12d-2 (Aura swap on an Aura owned by the opponent).

## 701.14 701.15 701.16 701.17 701.18 701.19 701.21 701.22

20 scenarios, 7 verifier changes.

**Concerns.** 701.14c (a creature fighting itself) has no Forge card, so it is skipped. 701.15c (several goaders) cannot be told apart from a single goad in a two-player game, so it is skipped. The 701.17c/d (Charmed Pendant, Bruvac), 701.17b cost and 701.22d empty-library scenarios depend on Forge implementation details and carry forge_notes.

**Fixed.** The most common errors were skip reasons that were not true. 701.14c said no Forge card can make a creature fight itself, but Nightfall Predator can, and 701.15c said there was nothing observable, but Laser Screwdriver plus Silent Arbiter makes it observable; both are now real scenarios. Other fixes covered a variant that showed no difference (the old 701.15d-1), a missing sub-clause (a land put onto the battlefield by a spell is not a land play, 701.18a-2), and deltas that changed more than one fact or left the timing open (701.18a opponents_turn, 701.19b one_shield_only).

## 701.23 701.24 701.25 701.26

15 scenarios, 4 verifier changes.

**Concerns.** 701.24f (simultaneous shuffles) and 701.23e (reveal) are weakly illustrable: Forge likely counts two sequential Darksteel Colossus shuffles, and no record field or trigger observes a reveal. 701.24c's Guile-exiled-in-response case may diverge in Forge, which could skip the shuffle when the card has left the graveyard; I noted this in forge_notes.

**Fixed.** Timing slips: actions were placed before the trigger or spell that performs them resolved (a surveil done before its trigger resolved, a Snitch trigger resolving ahead of the spell's own draw). One variant duplicated another instead of testing the case it was named for: the base of 701.24e-1 shuffled an empty library, not a one-card one. A triage line was missing the trigger record kind its own outcome names.

## 701.27 701.28 701.29 701.30 701.34 701.35 701.36

18 scenarios, 9 verifier changes.

**Concerns.** 701.27d, 701.28c and 701.28d are skipped because no Forge effect reaches them (no non-self transform targets a permanent with an instant or sorcery face, and every convert effect names its own double-faced card). The scenarios for 701.27e (Cult trigger on Arcee's convert) and 701.28e/f rely on Forge-specific behaviour of Arcee, Sharpshooter and are flagged in forge_notes.

**Fixed.** The errors I fixed most often were base variants with a non-empty delta, and decisions or expected text that contradicted the variant (choosing a token or counter that did not exist, or reasoning written into expected). Two skip reasons were false: a Clone copying Arcee, Sharpshooter makes rule 701.28c reachable, so I added scenario 701.28c-1, and the 701.27d reason listed too few transform effects. I also replaced variants that showed a different mechanism or changed two facts: Negate countering populate, and a detain check that changed both the turn and attack versus block. I reclassified triage rule 701.30b from decision to mechanical; all seven files validate with 0 errors.

## 701.37 701.39

5 scenarios, 2 verifier changes.

**Concerns.** The monstrous designation has no setup field, so the already_monstrous variants (701.37a-1, 701.37a-2, 701.37c-1) rely on setup notes and forge_notes saying the permanent became monstrous earlier. Bolster ties for least toughness need a player choice, so no scenario covers them.

**Fixed.** The only problem found was a triage misclassification. 701.37b, which says monstrous lasts until the permanent leaves the battlefield, was marked definitional. The reset is observable, so I reclassified it as mechanical and added scenario 701.37b-1, where Cloudshift flickers a monstrous Fleecemane Lion and monstrosity then works on it again. Every other scenario matched the Forge scripts and the Comprehensive Rules, and validate reports 0 errors.

## 701.40 701.41 701.42 701.43 701.44 701.46 701.47

20 scenarios, 14 verifier changes.

**Concerns.** 701.43c (exerting an object not on the battlefield) is skipped because Forge only offers exert as an attack cost or an ability cost on the battlefield. 701.43b shows only the not-yet-tapped branch, since a second exert before one untap step is not reachable with these cards. Forge may not prevent a manifest of an instant under Grafdigger's Cage, and the 701.44c last-known-information explore is flagged in forge_notes.

**Fixed.** The most common fixes were to variants that changed two facts, or changed a fact the listed rule does not decide: a base variant carrying a delta, a Hill Giant variant that added a creature and a target, and a Wildgrowth Walker variant that flipped its "you control" condition rather than whether the creature explored. I also reclassified 701.40b, c and d from decision to legality (whether a manifested permanent may be turned face up, and by which cost) and added playability scenarios for them. The rest were overclaimed or loosely ordered outcomes: a zero-target support trigger said to be "not put on the stack", and amass's 0/0 token dying before the reflexive trigger goes on the stack.

## 701.48 701.49 701.50 701.51 701.52 701.53 701.54

22 scenarios, 14 verifier changes.

**Concerns.** 701.50c (APNAP) only illustrates same-controller sequencing, since no Forge card makes both players' permanents connive at once, and the replaced/prevented-entry branch of 701.51c and the outside-the-game Lesson branch of 701.48a are unreachable in Forge. 701.50e (connive 0 still fires the trigger in ConniveEffect) deviates from the rules, so its expected text follows 701.50e and the divergence is noted in forge_notes.

**Fixed.** The most common faults were incomplete setups and variants that tested the wrong rule. In 701.54c-3 and c-4, A's library was empty while the Ring's loot trigger also fires at three or more temptations, so A would have drawn from an empty library and lost. Several does_not_fire variants (701.50f-1, 701.54d-1) changed a trigger's controller condition instead of the listed rule, and one delta changed two facts; the remaining gaps were branches the triage had wrongly ruled out or left uncovered (701.48a Lesson-from-sideboard, which Forge supports, and 701.54b non-copiable Ring-bearer, now covered by a Clone scenario).

## 701.55 701.56 701.57 701.58

13 scenarios, 6 verifier changes.

**Concerns.** 701.55d is skipped because no two-player Forge card makes both players face a villainous choice at once, so there is no APNAP order to show. 701.58e (cloaking several cards one at a time) and 701.58f (Grafdigger's Cage blocking the face-down creature) are weakly observable, and Forge's handling of both is uncertain; I put notes in forge_notes.

**Fixed.** The two Davros scenarios (701.55a-1, 701.55b-1) passed through combat without saying whether A attacks, and a 3/4 menace attacker could change B's life total. I added "A declares no attackers" to both. In 701.58, triage had put 701.58b/c/d (which ways to turn a cloaked card face up are legal) under decision. I moved them to legality, matching the already-verified 701.40 manifest unit, and added three playability scenarios (701.58b-1, 701.58c-1, 701.58d-1); validate reports 0 errors.

## 701.59 701.60 701.62 701.63 701.64 701.65 701.66 701.67

21 scenarios, 32 verifier changes.

**Concerns.** rules/triage/<unit>.jsonl does not exist for any of these eight units, so validate checked no rule coverage or card names. I covered every lettered rule from the CR text and checked card names and coverage with a separate script, and all passed. 701.60d (a suspected permanent can't become suspected again) is skipped because nothing observable differs.

**Fixed.** The most common problem was that none of the eight units had a rules/triage/<unit>.jsonl file, so validate silently skipped checking their scenarios; I wrote all eight triage files. Some variants failed the minimal-pair rule: in 701.66a-1 the does_not_fire variant worked through a fizzle that rule 608.2b decides, not 701.66a, and in 701.60b-1 a variant whose outcome came from an ability was labelled does_not_fire. There was also one false skip reason: 701.60d does change an outcome, through Frantic Scapegoat's "if you do", so I replaced the skip with a new scenario, 701.60d-1; in that scenario Forge's setSuspected likely diverges from the rules. Smaller fix: 701.59a-1 had a mana ability going on the stack.

## 701.61

3 scenarios, 2 verifier changes.

**Concerns.** The Food token is not a cardsfolder card, so it is left out of "cards" and described in notes and forge_notes. The scenarios rely on Treetop Sentries and Thornvault Forager implementing the forage cost the way their scripts suggest, which I did not check in the Forge engine code.

**Fixed.** The card facts checked out against the Forge scripts (Treetop Sentries, Thornvault Forager, Grizzly Bears and the Food Token script), and every expected outcome was correct. The only gaps were missing does_not_fire variants for the "your" part of 701.61a. I added one where the three cards sit in the opponent's graveyard, and one where the Food is controlled by the opponent. Validation passes with 0 errors and the file is marked verified.

## 701.68 701.69 701.70 701.71 702.1 702.3 702.4 702.5 702.6

23 scenarios, 43 verifier changes.

**Concerns.** rules/triage/ has no file for any of the nine units (it stops at 701.58, and 702.1, 702.3 to 702.6 and 701.68 to 701.71 are missing), so validate skips the coverage check and I chose which rules to cover from the rule text alone. 701.71 (empower Jace), 701.68d, 702.5c and 702.4e/702.3c cannot be illustrated in Forge and are skipped with reasons, and the echo-cost reevaluation (702.1b) and Luxior's equipped-planeswalker P/T (702.6e) depend on Forge internals I did not run.

**Fixed.** None of the nine units had a triage file, so validate never compared their scenarios against triage. I wrote all nine files in rules/triage/, following the conventions of the existing triage files. The scenario fixes were wrong setups: Volcano Hellion's notes made it ineligible for its echo trigger, Jace Beleren had no loyalty counters so it would die as a 0/0, and blight was chosen too early, when Grub's trigger went on the stack rather than as it resolved. I also swapped the inverted polarities on 701.68b-1, added 702.6c-2 for 702.6c's last sentence, which had no scenario, and stated the library order for 701.70a-1. validate now reports 0 errors and one accepted warning for 702.6c-2, whose second variant is 'other' because that rule never removes an attachment.

## 702.7 702.8 702.9 702.10 702.11 702.12

19 scenarios, 54 verifier changes.

**Concerns.** rules/triage/<unit>.jsonl does not exist for any of the six units, so validate reported 0 errors without checking coverage; I judged categories from the rule text and checked card names separately against Forge. Several scenarios (702.7c-1, 702.7c-2, 702.11b-2, 702.11e-1) rest on Forge behaviours I could not confirm: re-selecting second-step damage dealers, hexproof re-checked at resolution, and Shadowspear removing 'Hexproof:Black'. The 702.11b-2 and 702.12b-1/702.12b-2 scenarios have no does_not_fire variant.

**Fixed.** The most common error was a base variant with a non-empty delta, which left other variants differing from the base in two or more facts; polarities were also mislabeled, so several scenarios had no does_not_fire variant. There were two rules or card errors: 702.11b-2 claimed Doom Blade still resolves after its target gains hexproof (it fizzles under 608.2b), and 702.11f-1 used Doom Blade against the black Raddic, Tal Zealot (now Last Gasp). The triage files for all six units were missing, so validate had passed without checking anything; I wrote them (redundancy and static-marker rules are definitional, as in 702.2f), dropped the 702.7d-1 scenario, whose variants changed no outcome, and removed the skip entries for definitional rules.

## 702.13 702.14 702.15 702.16 702.17 702.18 702.20

26 scenarios, 4 verifier changes.

**Concerns.** 702.16j and 702.16k name only damage prevention, targeting and blocking, not enchanting or equipping; I did not illustrate those two clauses. In 702.16e-1 and 702.16j-3 the variant swaps the board creature rather than changing a decision, and 702.16p-1 follows Benevolent Blessing's card text (only Auras and Equipment its controller controls are kept), which is narrower than the rule text.

**Fixed.** The most serious fix was an illegal action: in 702.16n-1, the variant cast a white Gods Willing on a creature that already had protection from white, so that cast could never happen. The second protection now comes from equipping the colorless Sword of Light and Shadow. In 702.15c-1, the base and its variant gave the same result whether last known information applied or not, because the creature had lifelink printed on the card; its lifelink now comes from an Aura. In 702.18a-1, a variant changed three facts at once (the spell, the land and the target), and it now changes only the target.

## 702.21 702.22 702.23

14 scenarios, 4 verifier changes.

**Concerns.** 702.22b is weakly illustrated: Tolaria's Debuff names both banding and bands-with-other, so the scenario cannot separate "loses banding implies loses bands-with-other" from a direct removal. 702.23c is skipped because no Forge card gives one creature two rampage instances; 702.22d and the band declarations in the other 702.22 scenarios depend on how Forge lets a band be declared.

**Fixed.** In 702.22, two of the cards had Forge placeholder scripts with no Oracle text, so I swapped them for vanilla red legendary creatures. 702.22c's band-composition limits had been triaged as a player decision; I reclassified them as legality and added a scenario for them. Some response timings were only implied and some expected outcomes left out B's life total, so I filled both in. Separately, I accidentally re-ran another session's stale scratchpad fix.py, which duplicated review entries in 701.55 and 701.58 and three scenarios in 701.58. I removed the duplicates and 701.55 and 701.58 now validate with 0 errors.

## 702.24 702.25 702.26 702.27 702.28

19 scenarios, 2 verifier changes.

**Concerns.** 702.26h (an object phasing out both directly and indirectly) is skipped because phase-in timing is identical either way in a two-player game, so no outcome separates the two cases. The 702.26e, f and i setups rely on a phased-out permanent that the schema can only express in setup notes, and 702.26j covers only the unattach trigger, not the attach trigger.

**Fixed.** I found few errors, and all of them were about wording, not rules. In 702.24a-1, the base decision said A pays by tapping both Islands, which the one_island_tapped variant makes impossible; I reworded it as a choice to pay. In 702.26p-1, the identical-outcome control for a redundancy rule was labelled fires; I relabelled it other, because that rule cannot have a does_not_fire branch.

## 702.29 702.30 702.31 702.32 702.33 702.34 702.35 702.36

23 scenarios, 2 verifier changes.

**Concerns.** 702.35c is skipped: no card in a two-player Forge game looks up a discarded madness card after the madness trigger resolves (cards like Chandra Ablaze resolve before the trigger goes on the stack), and I did not search every candidate. Several scenarios depend on how Forge implements a rule: Tsabo's Web treating Lonely Sandbar's cycling as a non-mana activated ability (702.29b), Stabilizer blocking typecycling (702.29f), and the echo-already-paid state (702.30a); each has a forge_notes entry.

**Fixed.** I found two errors and fixed both; all eight files are now marked verified and validate reports 0 errors. In 702.29d-1, the does_not_fire variant moved Drake Haven to B's battlefield, which tests the trigger's "you" condition rather than 702.29d's once-per-cycle count; it is now a variant that discards the card through Tormenting Voice, with a note that a counting rule has no does_not_fire branch. 702.35c had been skipped as impossible to illustrate, but Containment Construct's "whenever you discard" trigger can resolve after the madness trigger, so I removed the skip and added scenario 702.35c-1 with three variants (scripts in C:/Users/nicol/AppData/Local/Temp/claude/C--Users-nicol-IdeaProjects-price-predictor/0bf1cb33-f5ed-4f6d-9e19-6f4cecb1b732/scratchpad/rev_702_29_36.py).

## 702.37 702.38 702.39 702.40 702.41 702.42 702.43

14 scenarios, 8 verifier changes.

**Concerns.** The "b" subrules (702.38b, 702.39b, 702.40b, 702.41b, 702.43b) are skipped because no Forge card has two instances of amplify, provoke, storm, affinity or modular. 702.37e's clause that enters-the-battlefield abilities don't trigger on turning face up isn't isolated, since every morph card with an ETB trigger in Forge also has a turned-face-up trigger.

**Fixed.** The most common error was skipping a "multiple instances" rule (702.39b, 702.40b, 702.41b, 702.43b) on the claim that no Forge card can have the keyword twice. Cards that grant the keyword make it reachable: Hunter Sliver for provoke, Prismari, the Inspiration for storm, Mycosynth Golem for affinity and Blaster, Combat DJ for modular. I replaced each of those skips with a scenario. I also fixed a wrong forge_note that said the enters-the-battlefield clause of 702.37e cannot be tested (new scenario 702.37e-2 tests it with Soul Warden) and a mana-count mistake in 702.42b-1 (seven Mountains for a six-mana cost).

## 702.44 702.45 702.46 702.47 702.48 702.49 702.50

21 scenarios, 7 verifier changes.

**Concerns.** 702.44d (multiple instances of sunburst) has no Forge card or effect that reaches it, and 702.49d (commander ninjutsu) needs the command zone, so both are skipped. Splice (702.47) and offering (702.48) are the least certain against Forge's implementation: protection against a spliced spell's color, splicing the same card twice, and a creature's changeling type counting as Goblin for an offering sacrifice.

**Fixed.** The most common faults were incomplete setups and decisions: the opponent's response was left open or written into the base variant's delta or expected text, and in 702.50a-1 the opponent's library was empty, so they would lose the game in their own draw step. Two scenarios got trigger targeting wrong. A soulshift trigger with no legal target was described as removed from the stack, when it is never put there (603.3d). A second soulshift trigger was said to have no target, when both triggers can target the same card and the second fizzles on resolution. Polarities were inverted or a variant tested a different rule (702.47e-1, 702.49c-1). Those two rules can have no does_not_fire case, so `validate` still prints a warning for each, with 0 errors.

## 702.51 702.52 702.53 702.54 702.55 702.56 702.57

18 scenarios, 4 verifier changes.

**Concerns.** 702.54c and 702.56b (multiple instances) are skipped because no Forge card has two bloodthirst or two replicate instances. 702.53b (Tolaria West under Tsabo's Web) and 702.55b (haunted permanent turned into a land by Imprisoned in the Moon) may not match Forge's implementation, and both carry forge_notes.

**Fixed.** The most common error was a false skip reason. 702.54c and 702.56b were skipped as impossible to build, but Bloodlord of Vaasgoth and Djinn Illuminatus each grant a second instance of the keyword, so I added scenarios 702.54c-1 and 702.56b-1 in their place. I also fixed one setup that could not be reached (702.53b-1 observed an untap step that the setup placed after upkeep) and dropped one variant that changed a fact without changing the outcome (702.55b-1 still_creature).

## 702.58 702.59 702.60 702.61 702.62 702.63 702.64 702.65 702.66

24 scenarios, 5 verifier changes.

**Concerns.** 702.58b and 702.63c are skipped because no Forge card gives a permanent two instances of graft or vanishing. Maelstrom Djinn's gained vanishing (702.63b-1), the split-second special action (702.61b-1) and Arcanum Wings owned by B but controlled by A (702.65b-2) need setups the board schema cannot express directly, so those scenarios explain them in setup notes and forge_notes.

**Fixed.** The most common faults were in the absorb scenarios. One variant was labelled does_not_fire although the rule still applied and only the total damage changed. Another used two Shocks, which are two different sources, to illustrate damage from the same source at a different time (I rewrote it as a double-strike attack against a single 4-power attacker). The others were a skip reason that falsely claimed a second instance of vanishing cannot be reached (Maelstrom Djinn plus Backslide reaches it) and rule 702.62d triaged as a decision, though it decides whether a cost increase makes the free cast from suspend possible.

## 702.67 702.68 702.69 702.70 702.71 702.72 702.73 702.74 702.75 702.76 702.77 702.78 702.79

22 scenarios, 2 verifier changes.

**Concerns.** 702.67c (multiple fortify) and 702.69b (multiple gravestorm) are skipped because no Forge card has two instances. 702.68b, 702.70b and 702.77b rely on Forge keeping two granted trigger or keyword instances separate (Frenzy Sliver, Virulent Sliver) and on hasNonManaActivatedAbility counting reinforce for a card in the graveyard (Tazri); I noted these in forge_notes.

**Fixed.** I checked all 13 units against the rules and the Forge scripts, and they were mostly right. I fixed two problems. In 702.74a-2, the only does_not_fire variant took away a land, so it tested whether the mana could be paid rather than the evoke alternative cost; I added a variant that swaps in Air Elemental, a card with no evoke. In 702.75a-1, the expected library order only holds if libraries are listed top first, which the setup did not say, so I added that to its notes. All files are now marked verified and validate reports 0 errors.

## 702.80 702.81 702.82 702.83 702.84 702.85 702.86 702.87 702.88 702.89 702.90

25 scenarios, 5 verifier changes.

**Concerns.** 702.90e (infect from a non-battlefield zone) is skipped because no Forge card gives infect to a spell or to a card outside the battlefield; 702.80b and 702.90d rely on Forge using last known information for a sacrificed or dead source, which I could not check in the engine. Two multi-instance scenarios (702.83a-2, 702.86b-1) pair a "one instance" variant, marked does_not_fire, against the base, since there is no natural does_not_fire for "each instance triggers".

**Fixed.** The most common error was a base variant with a non-empty delta. In 702.85c-1, 702.89a-1, 702.90b-1 and 702.90c-1 the base cast or attacker was named in the delta rather than fixed in decisions, so I moved it into decisions and emptied the base delta. Coverage also had a gap: no scenario tested rebound's "cast from your hand" condition, so I added 702.88a-3, which compares Staggershock cast from exile through cascade with Staggershock cast from hand. All 11 files are now verified and validate reports 0 errors.

## 702.91 702.92 702.93 702.94 702.95 702.96 702.97 702.98 702.99 702.100 702.101

24 scenarios, 1 verifier changes.

**Concerns.** Multiple-instance rules (702.91b, 702.100d, 702.101b) depend on Forge keeping duplicate keyword instances granted by Pump or static effects, and soulbond pairing (702.95e) and an opponent-owned Young Wolf (702.93a-2) cannot be written directly in a setup, so those are described in notes and forge_notes.

**Fixed.** I found one error and fixed it. In 702.95e-1, the stops_being_creature variant tapped four lands to cast Song of the Dryads, which costs {2}{G}; it now taps Forest, Island and Mountain. I checked every other scenario against the Comprehensive Rules and the Forge card scripts and found it correct. All 11 files are now marked verified, and validate reports 0 errors.

## 702.102 702.103 702.104 702.105 702.107 702.108 702.109 702.110

24 scenarios, 4 verifier changes.

**Concerns.** 702.103g (bestowed Aura phasing in unattached) needs Teferi's Protection to phase the Aura out while its host stays in play, so it spans two turns and may not be reachable in Forge. 702.103c relies on Lithoform Engine copying a bestowed spell, and 702.104 and 702.105b depend on Forge honouring a forced tribute choice and a duplicated dethrone keyword. All of these carry forge_notes.

**Fixed.** The most common problems were incomplete setups. In 702.103f-2, B scried with an empty library. In 702.107a-2, a delta put Shock on the stack while leaving it in hand and its Mountain untapped. One rule was also misclassified: 702.110b, which decides whether "exploits a creature" triggers fire, was marked definitional, so I reclassified it as mechanical and covered it with the new scenario 702.110b-1 (Bloodthrone Vampire sacrifice contrast).

## 702.111 702.112 702.113 702.114 702.115 702.117 702.118

12 scenarios, 7 verifier changes.

**Concerns.** 702.115b (multiple ingest instances) is skipped because no Forge card prints ingest twice or grants it. 702.112b (renowned lost on leaving the battlefield) needs a two-turn flicker line and a renowned flag the harness may not expose, and 702.114a-3 relies on Forge evaluating Pyroblast's color check against the colorless spell on the stack.

**Fixed.** The most common error was in blocked variants, where a 2/2 is blocked by Grizzly Bears: the expected outcome left out that both creatures die (702.112a-1, 702.115a-1). Some variants were not minimal pairs: one changed two facts at once (702.113b-1), and another did not change the outcome compared with its base (702.111b-1). Smaller fixes covered timing that did not match the setup step (an opponent's spell cast in the upkeep of a main1 scenario in 702.117a-1), a wording error about an intervening-if trigger, and an overstated skip reason for 702.115b.

## 702.119 702.120 702.121 702.122 702.123 702.124 702.125 702.126 702.127 702.128 702.129

22 scenarios, 7 verifier changes.

**Concerns.** 702.125c is skipped because no Forge card gives one spell two instances of undaunted. 702.126c is illustrated with Inspiring Statuary granting a second improvise to Reverse Engineer, but redundancy leaves the outcome unchanged, so its variants only show that each generic mana is paid once. Several scenarios rely on Forge keeping granted keyword instances (Fang of the Pack melee, Cayth fabricate) distinct from printed ones.

**Fixed.** The most common errors were setups that did not finish the job: Abundant Maw's cast trigger and Crabomination's enters trigger were left unresolved, and A's library was empty while Reverse Engineer's expected outcome draws three cards. Some variants were not true minimal pairs: one changed two facts at once (Islands and Ornithopters), and the undaunted scenario's only does_not_fire variant still had undaunted applying. One gap in rule coverage was closed: the intervening-if half of 702.122e had no scenario, so 702.122e-2 now tests it with Mighty Servant of Leuk-o.

## 702.130 702.131 702.132 702.133 702.134 702.135 702.136 702.137 702.138

20 scenarios, 2 verifier changes.

**Concerns.** 702.131d (continuous effects reapplied before blessing-conditioned triggers) was skipped, because no Forge card has a blessing-conditioned trigger whose event also changes the permanent count. Assist (702.132a) is hard to reach in Forge: in a two-player game the opponent must be the assisting player, and the playability verdict may not count the assister's lands.

**Fixed.** The 702.131d skip gave a false reason: a Gnome token from Illustrious Wanderglyph can be the tenth permanent and also meet Garruk's Packleader's power-3 condition once the blessing applies. I removed the skip and added scenario 702.131d-1. I also dropped a 702.133a-1 variant that only tested ordinary mana payment rather than jump-start. Every other scenario matched the rules and its cards' Forge scripts, and validate reports 0 errors.

## 702.140 702.141

8 scenarios, 4 verifier changes.

**Concerns.** 702.140f is illustrated with Thundermane Dragon granting haste to a library-cast mutating spell, which depends on Forge allowing a mutate cast from the library and carrying the haste over to the merged permanent. The 702.141a-2 "opponent-turn" variant is awkward because encore needs sorcery timing and Forge may not give A priority in that state.

**Fixed.** The main problems were setups that left decisions open or did not match a variant. One variant never said whether the mutating spell goes over or under its target, and one set of decisions declared an attack that a variant's attack tax made impossible. One variant also changed two facts at once (whose turn it is and whether the stack is empty), so I split it into two variants and added one with the card in hand to test the graveyard-only restriction.

## 702.142 702.143 702.144 702.145 702.146 702.147 702.148 702.149 702.150 702.151

24 scenarios, 3 verifier changes.

**Concerns.** 702.145g is hard to reach: a back-face-up nightbound permanent while it is neither day nor night needs a transformed double-faced card on the battlefield before day or night has begun, so that scenario relies on the harness setting the state directly. The day/night state and the face of a transformed permanent have no setup field, so they are described in setup.notes.

**Fixed.** The most common problem was a rule with more than one clause where only one clause had a scenario. Foretell's rule that a card can be foretold only during its owner's own turn had none, and neither did daybound's ability that the permanent can't transform any other way, so I added scenario 702.143a-2 and scenario 702.145b-4 (Moonmist on Hound Tamer against Gatstaf Shepherd). The one other fix was wording: 702.144a-1's expected outcome called the original spell a copy, and I corrected it. All ten files are now marked verified and validate reports 0 errors.

## 702.152 702.153 702.154 702.155 702.156 702.157 702.159 702.160 702.161 702.162 702.163 702.164 702.165

17 scenarios, 4 verifier changes.

**Concerns.** 702.153b, 702.154d and 702.157b are skipped because no Forge card has two instances of casualty, enlist or squad. 702.159a depends on the die roll being forced, and 702.161a on placing Goldbug already converted, which Forge's puzzle format may not reach easily. The validate run reported 0 errors and 0 warnings.

**Fixed.** Two variants failed to isolate the rule they listed. In 702.165c-1, the ability was gained after the backup trigger was already on the stack, so 702.165d decided the outcome and 702.165c did not; it was rebuilt with Herald of Dromoka and Sigiled Sentinel. In 702.165d-1, a fizzle variant illustrated 608.2b instead, so it was dropped, which leaves the one remaining validate warning (that scenario has no does_not_fire variant, explained in its forge_notes). Smaller fixes: a delta in 702.153 named two facts at once, and 702.155 described a chapter ability with no legal target as resolving when it is removed from the stack under 603.3d.

## 702.166 702.167 702.168 702.169 702.170 702.171

22 scenarios, 6 verifier changes.

**Concerns.** 702.171b (saddled stays saddled until end of turn or until it leaves, not copiable) is illustrated only through the end-of-turn persistence; the leave-the-battlefield and not-copied parts have no scenario, because a copy or a recast Mount is summoning sick and cannot attack to show the designation. 702.168e relies on Forge carrying the X paid for the turn-face-up special action into the trigger, and 702.167c on Forge not counting the crafted card itself among the exiled cards; both are flagged in forge_notes.

**Fixed.** The most common error was a board that could not be reached: four spell_on_stack variants had B cast Shock with no land, so I gave B a Mountain and made the delta tap it. In 702.171b-1, the last-turn variant still showed Grizzly Bears tapped, although the untap step would have untapped it. One expected field in 702.168e-1 also contained reasoning, which I removed. I checked every card fact against its Forge script and found no errors, and validate reports 0 errors for all six units, now marked verified.

## 702.172 702.173 702.174 702.175 702.176 702.177

19 scenarios, 4 verifier changes.

**Concerns.** 702.175b is skipped because no Forge card has two offspring instances. Several scenarios (702.176a-2, 702.177a-1, 702.177b-1 and the Perch Protection extra-turn scenario) need prior-turn state or a multi-turn observation, and the gift triggered ability may not raise Forge's GiveGift trigger (702.174c-2); both are flagged in forge_notes.

**Fixed.** The most common errors were incomplete or inaccurate setups and expected outcomes. One scenario played on through later turns with empty libraries, which would make B lose before the result could be seen. One observation point fell after the end-step trigger, and two expected outcomes misstated card effects: Dawn's Truce protects every permanent A controls, not only Grizzly Bears, and a noncreature Overlord still has flying. A skip for 702.175b was wrong, because Zinnia, Valley's Voice gives creature spells a second offspring, so a scenario was added in its place.

## 702.178 702.179 702.180 702.181 702.182 702.183

15 scenarios, 5 verifier changes.

**Concerns.** 702.179c is skipped because no card in Forge's corpus tells a player to increase speed by a value. Player speed has no setup field, so the scenarios state it in setup.notes.

**Fixed.** Most fixes were missing setup facts: the speed scenarios never said that A's once-per-turn speed trigger had not already fired that turn, so I added it to their notes. The rest were "expected" strings that explained the reasoning instead of stating only the outcome, and a Mobilize X = 0 variant that did not say the trigger still fires. All six files (rules/scenarios/702.178.json to 702.183.json) are now marked verified and validate reports 0 errors.

## 702.184 702.185 702.186 702.187 702.188 702.189 702.190 702.191 702.192 702.193 702.194

23 scenarios, 5 verifier changes.

**Concerns.** 702.191b (two instances of increment) is skipped because no Forge card gives a creature a second instance. 702.193b's excess-reduces-generic branch is unreachable, since Hulk, Gamma Goliath is the only power-up card; it is flagged in forge_notes.

**Fixed.** The most common fault was a variant that did not change what the rule decides. 702.184a-2's summoning-sick variant left the activation legal, and 702.193b-1's two-Forests variant flipped legality only by taking mana away, while its Forests would have paid the cost even if the colored reduction were wrong. Other fixes: 702.194a-1 had assumed the spell quietly falls back to a cast without teamwork when the payment is too small (now the payment is illegal), and 702.194b-1's expected token was missing flying; in triage, 702.186b gained the trigger record kind.

## 702.195 703

15 scenarios, 12 verifier changes.

**Concerns.** 703.4g depends on a random die roll, so the expected outcome covers both branches. 703.4b-2 needs a night-state setup with the back face on the battlefield. 703.3-1 relies on Forge ordering state-based actions before the draw-step trigger goes on the stack, which I did not confirm in the engine.

**Fixed.** Most fixes removed reasoning from "expected" strings so they state only the observable outcome, and changed record_kind to resolution where the outcome is the effect of a resolving trigger. The setup gaps were empty libraries that would make the active player lose at the next draw step. In 703.4g-1 the die roll and the visit target had been left undetermined, so the roll is now fixed at 5 and the target is set.

## 704

27 scenarios, 15 verifier changes.

**Concerns.** 704.5v-y (battles) are skipped: no isolatable Forge setup exists for them. Only the creature clause of 704.5p, the spell-copy sentence of 704.5e and the non-tie branch of 704.5k are illustrated, and 704.5aa has no existing record field to show speed.

**Fixed.** The skip reasons for 704.5v and 704.5y were wrong. Both cases can happen in a two-player game (countering a Siege's defeat trigger with Stifle; taking control of a Siege with Confiscate), so I replaced each skip with a new scenario, and I reclassified 704.5t (dungeons) from out_of_scope to mechanical with a new scenario. The other fixes were setup steps that did not match the decisions (704.5h, 704.5s), a Pacifism sent to the wrong owner's graveyard (704.5m-2), a record kind of resolution on poison dealt by combat damage (704.5c), and a trigger that players passed on before it was put on the stack (704.5r).

## 705 706

9 scenarios, 3 verifier changes.

**Concerns.** 706.8a and 706.8b are only observable through the Centaur of Attention's power and toughness, and 706.8b needs a prearranged stored-results state that Forge can only reach by running the enter trigger with forced dice. Every coin and die result has to be forced by the harness, and Forge never asks a player to call a coin, so 705.2's call is implicit.

**Fixed.** The worst error was in 706.6-1. Berserker's Frenzy always ignores the lower roll, so the scenario's contrast variant flipped Steel Squirrel's own 5-or-higher condition, not the ignore rule; I rebuilt it so having Wyll, Blade of Frontiers on the battlefield or not decides whether the roll is ignored, and Complaints Clerk's roll-a-1 trigger shows the result. The other fixes were small: the roll-again choice was fixed only after the first roll in one decision list, and one variant's delta named a consequence as if it were a second change.

## 707

34 scenarios, 20 verifier changes.

**Concerns.** Several scenarios (707.2b-1, 707.2c-1, 707.9e-1, 707.9g-1, 707.10-2, 707.9f-1) depend on Forge modelling edge cases (copying face-down permanents, a second copy effect on a copy of Clone, spent-mana on copies, conditional exceptions), flagged in forge_notes; 707.3's face-down-becomes-a-copy part is not exercised; 707.10a-1 has no natural does_not_fire variant, leaving one validator warning.

**Fixed.** Thirteen scenarios started with a variant that carried a delta, with no empty-delta base; I moved that choice into decisions. Three setups passed a turn through a draw step with an empty library, so a player would lose; I added library cards. Two card choices broke the rule being shown, and I swapped them: Banisher Priest has no linked abilities, so Fiend Hunter replaces it in 707.7-1, and Ambush Viper's deathtouch would kill Centaur Courser, so Ashcoat Bear replaces it in 707.11-1.

## 708

8 scenarios, 3 verifier changes.

**Concerns.** 708.12 and 708.7 (the face-down-spell-on-stack variant) rely on Forge reaching manifested or stack face-down states and on its reveal check ignoring continuous effects, so they may be hard to set up or may differ from the rules. In 708.2b the polarity is read from the turn-face-down effect (fires = it turns face down), and in 708.3 from the trigger (fires = it triggers).

**Fixed.** Variant polarities were inverted: the face-up cast, where 708.4 does not apply, was labelled other, and the Essence Scatter check, where the face-down characteristics still apply, was labelled does_not_fire. One variant tested cost payment rather than the rule it was listed under, so I dropped it. One expected outcome left out a continuous effect that still applies after the permanent is turned face up (rule 708.8).

## 709

19 scenarios, 15 verifier changes.

**Concerns.** 709.5b (copiable values of a Room spell on the stack) is skipped because its observable effect depends on rulings outside the rule text. 709.3c (cast copy of a split card), 709.4a (name matching on a two-name card) and the non-cast Room in 709.5d-2 may not behave as the rules say in Forge, so each carries a forge_notes warning.

**Fixed.** The most common problem was variants that did not test the rule. Some gave the same outcome as base, such as the Shock copy in 709.3c-1. Others would also have happened under the misreading, such as three Forests in 709.3a-1 or a library search that already sees combined types in 709.5a-1. Some variants changed two facts at once (709.2-1, 709.5g-1), and the forge_notes in five scenarios named the wrong split card. One variant also missed a side effect: in 709.5f-1 Locker Room's draw trigger would have drawn from an empty library.

## 710 711

12 scenarios, 10 verifier changes.

**Concerns.** The scenario schema has no field for a flipped permanent, so each flip scenario records "flipped" in setup.notes. Forge may offer only the front-face name when a player chooses a card name, so 710.5-1 might not be playable as written (flagged in forge_notes).

**Fixed.** The most common error was in invariance rules (710.1c, 711.4, 711.6), which say that something stays the same and so can never fail. Their variants were labelled does_not_fire although they turned on some other rule or did not test the rule at all (anthem controller, sorcery timing, cost payment, a leveler illegal under either reading of its power). Those variants were relabelled 'other' or dropped, and 710.1c-3 was rebuilt so that a Giant Growth cast before the flip carries over after it. In 710's triage, observability was understated: a continuous or playability record field shows the outcome directly.

## 712

29 scenarios, 12 verifier changes.

**Concerns.** 712.14a's non-double-faced branch, 712.10, 712.13a and 712.14b cannot be reached with any Forge card, so the first is only partly covered and the other three are skipped. Several scenarios (712.4b naming a back face, 712.16 Ixidron on a double-faced card, the 712.21 melded-permanent family, 712.20 day/night) depend on Forge behavior I did not confirm and are flagged in forge_notes.

**Fixed.** The most common fault was variant pairs that flipped nothing the rule decides: both variants gave the same outcome, or the change only affected affordability or whether a trigger fired. Two skip reasons were false, and an untested route exists for each: The Golden-Gear Colossus with Mycosynth Lattice for 712.10, and Ghostly Flicker on a sorcery-front MDFC land for 712.14b. Two base variants also carried a non-empty delta, and some decisions were left open (a missing target, an unstated destination for a countered disturb spell).

## 714 715 716

17 scenarios, 5 verifier changes.

**Concerns.** The validator reports 0 errors. Four scenarios (714.3a-1, 715.2c-1, 715.4-1, 716.4-1) have no fires/does_not_fire pair, because the rule has no natural near-miss that Forge can reach. 715.2b (copiable Adventure values) is skipped and pointed at 715.3c-1, and the part of 716.2b about a permanent keeping its level after it stops being a Class is not illustrated (no Forge card removes the Class type).

**Fixed.** The most common faults were setup details: three Saga scenarios started in the draw step with an empty library, and the targets for chapter I and chapter III were never chosen. The other faults were wrong polarity on variants: the contrast the rule actually decides was labelled other, while a mana-only variant was labelled does_not_fire. One 716.3 variant also changed three facts at once and was rewritten so it changes only one. No card facts or expected outcomes were wrong. Validate reports 0 errors and three remaining warnings (715.2c-1, 715.4-1, 716.4-1), left as is because those rules state an identity or a non-interaction and cannot have a does_not_fire case.

## 717 718 719

17 scenarios, 1 verifier changes.

**Concerns.** 718.2a (alternative values as copiable values) has no separate observable, so it is skipped with a reason naming 718.3c-1 and 718.3d-1. Dice results, the Attraction deck, the solved designation and the prototype cast choice cannot be set in the setup schema, so the scenarios state them in setup.notes and forge_notes.

**Fixed.** The only error was in 718.3a-1: its seven_plains variant changed two facts at once against the base (how many lands and what type), so I dropped it; the other variants still cover the rule. In units 717 and 719, every scenario's rules outcome, card facts (checked against the Forge scripts) and mana setup were correct, so all three files are now marked verified and validate reports 0 errors.

## 720

8 scenarios, 2 verifier changes.

**Concerns.** 720.2b (copiable values of the Omen face) has no observable difference in Forge, since no card refers to "has an Omen", so it is skipped and points to 720.3c-1. 720.3c-1 depends on Forge copying the spell in its alternate state, which I could not check.

**Fixed.** Two does_not_fire variants did not test the rule. In 720.3c-1 the counterspell meant no copy ever existed, so the copy rule never applied. In 720.4-2 the variant was an illegal cast, not a resolution outcome. I rebuilt both: 720.3c-1 now uses See Double to copy either the Omen or the same card cast as a creature, and 720.4-2 uses Exhume so both variants resolve.

## 721 722 723

12 scenarios, 6 verifier changes.

**Concerns.** The 723.x scenarios (player control) vary mostly who makes B's decisions, so their physical outcomes are identical across variants, because decisions are fixed and the record schema has no controller field. Prepared status and the exiled prepare copy have no setup field and are described in setup.notes.

**Fixed.** Most of the fixes were to decision sequences whose steps happened in an illegal order or left out steps: B drawing before its upkeep, B declaring blockers before Jump resolved, and the skipped turn and the turn between not being spelled out. One variant flipped an unrelated mechanism instead of the listed rule (a tapped station creature changes whether the cost can be paid, which is not what 721.4 decides), so I dropped it. I also corrected one wrong card-state description and a triage record_kinds list that did not match the scenario's record kind. Validation passes with 0 errors and one warning I left on purpose: 721.4-1 has no does_not_fire variant because the rule is unconditional.

## 724 725

15 scenarios, 3 verifier changes.

**Concerns.** 724.2a, 724.2c and 724.2g are skipped because Mandate of Peace, the only card that ends the combat phase, has no step where a trigger or a state-based action could occur. Several 724 scenarios also depend on Forge firing and ordering triggers correctly during or after EndTurnEffect and endCombatPhaseByEffect, which forge_notes flag but I did not verify in the engine.

**Fixed.** The most common fault was contrast variants that changed more than one fact. In 724.1f-1 and 724.2f-1 the variant switched the active player, which changed the whole turn and combat setup, so each now moves only Stonebinder's Familiar to B's side. In 725.5-1, Dawnglade Regent's "as long as you're the monarch" was only an ordinary false condition and never tested 725.5, so the scenario now uses Fealty to the Realm ("the monarch controls enchanted creature").

## 726

5 scenarios, 4 verifier changes.

**Concerns.** The initiative designation and the player's dungeon progress have no field in the setup schema, so they are stated in setup.notes and flagged in forge_notes as needing direct state setup. The built-in Undercity venture rooms (Secret Entrance, Forge) are not Forge card scripts, so I could not check their effects against a script.

**Fixed.** Some setups gave a player the initiative while that player had never entered a dungeon. That position can't come up in a real game, because taking the initiative always makes the player venture into Undercity. Some variants also said nobody held the initiative even though dungeon progress implied someone had taken it, so I gave each holder a marker on Secret Entrance and described those variants as games where the initiative never came up.

## 727 728 729

14 scenarios, 6 verifier changes.

**Concerns.** Subgame rules (729.4a, 729.4b, 729.6, 727.6) can only be reached from mid-subgame snapshots, because the subgame's random first player and shuffled libraries rule out a fixed decision list. Each of those scenarios lists the subgame zones in setup and puts the main game in notes, and Forge's handling of Wish, Karn and nested Shahrazad inside a subgame is untested. 729.2a and 729.5a are skipped because the supplementary-deck shuffle gives no assertable outcome.

**Fixed.** Most fixes were card-count bookkeeping. Cards the notes mention were left out of every zone (Living Wish in 727.2-1, Swords to Plowshares in 727.5-1), so the hand and library totals came out wrong. Other fixes: one delta changed two facts at once (727.6-1), one polarity label was wrong (728.1-1), and two skip reasons for 729.2a/729.5a did not hold, because Forge's SubgameEffect moves Attraction decks; I replaced those skips with a new scenario, 729.2a-1.

## 730

16 scenarios, 3 verifier changes.

**Concerns.** Several scenarios (730.2g, 730.2h, 730.2i, 730.2j, and the token-on-top cases of 730.2d, 730.2f and 730.3e) depend on Forge's mutate implementation handling manifested instants, flip or double-faced components, and token-ness of merged permanents. They follow the rules text; each one's forge_notes flags the uncertainty. For trigger-record scenarios (730.2b, 730.2e, 730.2g) polarity follows whether the observed trigger fires, so the base variant is does_not_fire.

**Fixed.** The worst error was in 730.2g-1: the manifest rule 701.40g produced the outcome by itself, so 730.2g decided nothing, and the scenario was rebuilt so that only 730.2g keeps the permanent face down. In two other scenarios an expected outcome left out abilities that a lower component gives under 702.140e, and a variant was marked does_not_fire although the rule's effect does happen.

## 731

5 scenarios, 3 verifier changes.

**Concerns.** Neither-designation states with Brimstone Vandal already on the battlefield (731.1a-1 neither variant, 731.2c-1) need a puzzle-style placement, since the Vandal's enters-as-day replacement would normally set day. The designation is not an expressible setup field, so it is stated in setup notes.

**Fixed.** One scenario put A's discard and draw for Into the Night after Brimstone Vandal's trigger resolved, but both happen while Into the Night itself resolves, so I reordered the decisions. Opt's scry choice was left open in two variants, so the variants now say B keeps the card on top. In 731.2c-1 a variant was labelled does_not_fire although the day/night designation changes in neither case, so the day variant, where day becomes night, is now the contrast.
