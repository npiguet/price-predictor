# Verifier report: GAP-1, GAP-2 and GAP-3

I checked the tree at `ebee9154` (HEAD; the GAP-3 code is `d4f8269b` plus a docs-only commit). I left it clean: every mutation was restored with `git checkout` and touched. `git status` shows only `grep.exe.stackdump` and the untracked diaries, which were there before. No java process is running, and the lock `HEAVY.lock` is gone. I committed nothing. This report file is untracked.

Scratch files are in `C:\Users\nicol\AppData\Local\Temp\claude\01313e7323e07cb5\68ea1cf9-196c-43df-9df3-18c467a74c8b\scratchpad\`, prefixed `GV-`:
- `GV-dump\`: my own `DirectedForkPlayedForwardTest` run with `-Ddirected.records.dump`. It ran 45 cases, all passed, and wrote 58 games.
- `GV-dump2\`: two temporary played-forward cases I added, for 702.1c and 702.98a, the two listed rules that have no case of their own.
- `GV-dump3\`: the 118.12b forks of a temporary `ForkIsolationTest` match.
- `GV-tag.py` and `GV-tag.out`: every fork record tagged with the real Python `tag_record`. Each fork record is matched to its rule by its own `aimed_at`, not by the file name, and set beside the side the rule's counts lack.
- `GV-stale.py` and `GV-stale.out`: each listed rule's detector declarations on the tree the counts were made from, compared with HEAD.
- `GV-sealed.py`: the scenario cards that no booster set prints.
- `GV-mutate.py` and `GV-mut*.log`, `GV-iso.log`: the mutations, the temporary tests and their Maven runs.

"Lacking" below is what `scripts/rules_catalogue.py` computes from `rules/directed-fork-counts.json`. A rule lacks `fires` when it has fewer than 100 `fires` records. It lacks `does_not_fire` when it has none.

## Findings, most severe first

### 1. 702.1c's fork cannot yield the side its counts lack, and its counted `fires` side is mostly other cards' keyword grants
Severity: should fix before the pilot. Confidence: high (played and tagged).

Rule 702.1c is "the same is true for": Concerted Effort gives your creatures each listed keyword that one of them has. The counts lack its `does_not_fire` side (55,564 fires, 0 does_not_fire). Its variant `no_keywords` puts Hill Giant in White Knight's place, so the upkeep trigger grants nothing.

I played this fork forward with a temporary case. The fork's Concerted Effort resolution has no events, and it is tagged on neither side. The declaration in 702.1c-1 is `outcome_event` with `keyword_change`, `removed: false` and no `api` or `keyword`. An unscoped `outcome_event` declaration never tags a near miss (`outcome_event.py`: `if not scoped ...: return None`). So no fork, and no live record, can ever be tagged `does_not_fire` for 702.1c.

The `fires` side is not scoped either. In the same game, Sneak Attack's resolution, which gives haste, was tagged 702.1c `fires`, both live and in the fork. So the 55,564 fires count any resolution that grants a keyword.

Fix: scope the declaration to the shared-keyword grant: `api: PumpAll` plus a `script_param` on `SharedKeywordsZone`, or a qualifier that reads the shared keywords. Then recount, or take the rule off the list with that reason.

### 2. The counts the list is decided from predate most of the current declarations: 510.2 is listed only because of that, and 702.133a's lacking side is unknown
Severity: should fix before the pilot (a user decision: the recount is a heavy job). Confidence: high on the facts, medium on what a recount would show for 702.133a.

The implementer's claim (finding H) holds, and it reaches further than 19 rules. I checked it from the files and git history only, and ran no recount.
- `rules/directed-fork-counts.json` equals `output/effects/rule-counts-gen2.json` exactly, in its `rules`, `not_countable` and `none` members. That file was written on 2026-11-06 at 20:20, on the tree at about `1a0d4dc1`.
- 23 of the 49 listed rules have scenario declarations at HEAD that differ from that tree (`GV-stale.out`). 16 rules have a scenario whose declaration changed: 104.4a, 118.12b, 205.1a, 405.6d, 510.2, 601.2c, 607.2k, 702.53a, 702.84a, 702.118b, 702.126a, 702.127a, 702.133a, 702.187b, 702.188a and 718.3a. Seven rules have a scenario added since, which the counts never tagged: 701.23f, 701.44d, 702.19c, 702.103e, 704.5b, 712.4a and 712.14c.
- About twenty commits changed detector family code in the same period (`git log 1a0d4dc1..HEAD -- src/effects/domain/rule_detectors/`).

Rule 510.2 says combat damage is dealt all at once. It is listed only because its old declaration could not count a near miss. On 2026-11-06, 510.2-1 was `damage_dealt` with `min_count: 2` and no scope, so it tagged no near miss: hence its 0 `does_not_fire`. The current declaration, with `source_dies`, makes every combat damage step whose damage dealers all survive a near miss. That covers almost every unblocked attack. My played games show it: the live game of 702.19f's board, an ordinary unblocked attack, is tagged 510.2 `does_not_fire`. A recount would show hundreds of thousands of `does_not_fire` records, and 510.2 would leave the list. Its fork (`knight_in_the_bears_place`) adds one more record of that same population.

Rule 702.133a says a spell cast with jump-start is exiled as it leaves the stack. On 2026-11-06, 702.133a-1 read every spell's cost check and 702.133a-2 read every move from the stack to exile, with no keyword and no near miss. Its counts (2,779 fires, 0 does_not_fire) are the same as 702.127a's to the record, so they measure neither keyword. The current declarations filter on Jump-start, and 702.133a-2 now counts a near miss: a jump-start card cast from the hand, which goes to the graveyard. Self-play probably makes that common, and the graveyard casts rare. If a recount shows that, 702.133a lacks `fires`, and its fork (`cast_from_hand`, `does_not_fire`) makes the side the counts hold. That is the defect the earlier verifier's finding 1 named.

Fix: the user launches a gen-2 `rule-counts` with the current catalogue, then `directed-rules` regenerates the list, the wording cards and the counts file. Expect 510.2 to leave the list, and check 702.133a and the other 21 rules. For 510.2, a near miss that reads the rule's point would be a creature that dies in a combat damage step having dealt no damage (a first striker's victim), instead of "every damage dealer survives".

### 3. Three listed rules can never bind in a sealed game, because their cards are in no booster set
Severity: should fix, or a user decision. Confidence: high (read from Forge's editions and the preconditions).

`match-outcomes` opens pools only from editions with a Draft booster (`BoosterTemplates.eligibleSealedSets`). Commander products have none.
- 111.10p (a Virtuous Role gives +1/+1 for each enchantment): `VirtuousRole` needs a creature card whose enters trigger attaches a Virtuous Role. The only one in Forge is Ellivere of the Wild Court, which is printed only in Wilds of Eldraine Commander.
- 701.44d (permanents told to explore at once explore one at a time): `SeveralExplore` needs a phase trigger that makes every creature of a type explore. The only one is Hakbal of the Surging Soul, printed only in The Lost Caverns of Ixalan Commander.
- 701.50f (a permanent connives even when its controller cannot draw or discard): `ConniveWithoutDraw` needs a "whenever a creature connives" trigger. The only ones are Glorious Purpose, Iron Monger and Ultron, and no booster set holds any of them. GAP did not touch this rule.

Their forks can only be taken where a deck names these cards: a `collect-variants` steered round, whose decks hold scenario cards. `collect-coverage` holds them through `--wording-cards`, but it runs no directed fork. No budget is wasted, because the precondition never binds. But in the full `match-outcomes` collection these rules' lacking sides stay as short as they are.

The other GAP rules can bind in sealed play. 205.1a can through Imprisoned in the Moon (Eldritch Moon, Foundations). 701.23f can through Archivist of Oghma (Commander Legends: Battle for Baldur's Gate, a Draft set; its `AI:RemoveDeck:Random` does not drop it from limited decks) or Ob Nixilis, Unshackled (Magic 2015). 608.2p can through Tom Bombadil (Tales of Middle-earth). 702.133a can through Guilds of Ravnica and Modern Horizons 2. 118.12b, 702.19c and 702.103e add or replace their key card.

Fix: give each a reason in `NOT_DIRECTED` ("no sealed pool holds the card"), as 712.16 has, or say in FR-014b that they bind only in a steered `collect-variants` round.

### 4. Five precondition guards stay green when reverted
Severity: should fix (tests); the third is the one that can spend budget. Confidence: high (mutation seen green).

I reverted 10 guards in one batch, choosing different guards from the ones the implementers reverted, and ran `RegisteredRulesTest`. These stayed green:
- `VirtuousRole`: the enters trigger that makes a Virtuous Role (`makesAVirtuousRoleAsItEnters`) was removed, and every test stayed green. `virtuousRoleBindsNothingForACreatureWithoutTheRoleTrigger` puts Craw Wurm (six mana) in hand over four lands. The board fails the affordability guard and never reaches the role-trigger guard. Use a four-mana creature such as Centaur Courser. Without this guard, any affordable creature would bind, its cast would be forced, and no Role would appear.
- `TypeSettingAura`: the `canTarget` filter on the creature was removed. No test puts the only creature out of the Aura's reach, for example under hexproof or protection.
- `SearchUnderMindcensor`: the "not sorcery speed" condition was removed. The live dry run checks no timing, so a sorcery-speed search on the searcher's permanent would bind. The fork would then be taken, spending budget, and refused as `forced play not made`. No test covers it.
- `JumpStartFromHand`: the "spell needs no target" condition was removed. No test covers it.
- `OptionalSearch`: `isBasicLand` was replaced by `isLand`. The only negative board's library holds no land at all.

These went red, as named: `trampleOverPlaneswalkersBindsNothingWhenThrastaWouldHaveNoExcess`, `severalExploreBindsNothingWithOneCardInTheLibrary`, `cantGainLifeExchangeBindsNothingWhenTheExchangeWouldLowerTheLife` (GAP-1), `finalChapterResolvesBindsNothingWhenTheNextCounterReachesTheLastChapter` (GAP-1) and `plottedFromHandBindsNothingWhenBothCanBeCast` (GAP-1).

### 5. A role the edit makes switches off the whole live dry run, not just that role's check
Severity: minor (latent). Confidence: high (read).

`ForcedPlayRun.dryRun` builds a `ForcedPlayRun` on the live binding. The constructor throws at the first role the live binding lacks, which is a role bound through `Application.making`. The catch then returns "no refusal". So for any variant with a made role, the live board's check of the line, the source and the named targets is skipped entirely, and every refusal is found in the fork, after the budget is spent.

Today this changes nothing. The only user is 118.12b, whose play is an answer, and the dry run only checks that an answer's source exists. The same broad catch also hides a registry error, as it did before GAP-3.

Fix: check the roles the live binding holds, and skip only the made ones. Or have the registry refuse `making` on a cast's or activation's targets, and say so in forks.md.

### 6. FR-010b ties `making` to casts and activations, and its only user is an answer
Severity: minor. Confidence: high (read).

FR-010b says a forced cast or activation may use "a line of a card the edit puts on the battlefield, bound on the edited copy". Its "you may" clause speaks only of "a named line of a bound source". The only made card, 118.12b's Elfhame Sanctuary, is the source of an answer. The contract and forks.md describe `making` for any role, so only the spec sentence is narrower than the code.

## Per listed rule: what its fork yields (finding A)

Counts are (fires, does_not_fire) from the counts file. "Fork tags" are the real `tag_record`'s tags of the rule on the fork's records in my played games. Every listed rule except 702.1c and 702.98a has a case in `DirectedForkPlayedForwardTest`. Those two I played with temporary cases.

| Rule | Counts | Lacking | Variant | Fork tags | Yields lacking side |
|---|---|---|---|---|---|
| 104.2b | 0 / 12 | fires | forty | fires | yes |
| 104.3d | 47 / 1390 | fires | nine | fires | yes |
| 104.4a | 0 / 0 | both | b_has_more | does_not_fire | yes |
| 111.10p | 45 / 0 | both | two_enchantments (forced cast) | fires 2 | yes; never binds in sealed (finding 3) |
| 111.10v | 92 / 0 | both | mana_available | fires | yes |
| 118.12b | 0 / 36006 | fires | sanctuary_added (forced yes) | fires; also fires in 4 of 4 isolation matches | yes |
| 119.7 | 3 / 42 | fires | leyline | fires | yes |
| 205.1a | 0 / 1245716 | fires | counter-kept (`other`, forced cast) | fires | yes |
| 405.6d | 36267 / 0 | does_not_fire | cannot_pay | does_not_fire 64 | yes |
| 510.2 | 456396 / 0 | does_not_fire | knight_in_the_bears_place | does_not_fire | yes on stale counts; not lacking on current declarations (finding 2) |
| 601.2c | 25382367 / 0 | does_not_fire | two-creatures | does_not_fire 18 | yes |
| 607.2k | 13884 / 0 | does_not_fire | no_creature_to_champion | does_not_fire 2 | yes |
| 608.2p | 1 / 1 | fires | final-chapter | fires 2, does_not_fire 4 | yes |
| 614.17d | 16 / 96978 | fires | cage | fires | yes |
| 701.12e | 0 / 7 | fires | aura_in_hand | fires | yes |
| 701.12g | 42 / 3 | fires | opponent_above | fires | yes |
| 701.23f | 2 / 0 | both | mindcensor_added (forced activation) | fires | yes |
| 701.24e | 16 / 0 | both | zero-cards | fires | yes |
| 701.39a | 70 / 8 | fires | has_creature | fires | yes |
| 701.42a | 17 / 3 | fires | meld_pair | fires | yes |
| 701.44b | 90 / 37 | fires | land_on_top | fires | yes |
| 701.44d | 56 / 2643 | fires | branchwalker_in_the_bears_place | fires | yes; never binds in sealed (finding 3) |
| 701.50f | 6 / 0 | both | nothing_to_discard | fires | yes; never binds in sealed (finding 3) |
| 701.57b | 10 / 5 | fires | card_found | fires | yes |
| 702.1c | 55564 / 0 | does_not_fire | no_keywords | fires (Sneak Attack's haste); the Concerted Effort resolution is untagged | no (finding 1) |
| 702.19c | 0 / 2454 | fires | thrasta_in_the_dreadmaws_place | fires | yes |
| 702.19f | 44 / 0 | both | trample_over_planeswalkers | does_not_fire; also 702.19e fires | yes |
| 702.53a | 218095 / 0 | does_not_fire | no_match | does_not_fire | yes |
| 702.65a | 0 / 7 | fires | aura_in_hand | fires | yes |
| 702.65b | 7 / 0 | both | can_enchant | does_not_fire | yes |
| 702.71a | 12 / 0 | both | no_matching_value | does_not_fire | yes |
| 702.84a | 903 / 0 | does_not_fire | card_in_hand | does_not_fire 16, fires 8 | yes |
| 702.98a | 3169 / 8 | none (always listed) | no_counter | does_not_fire | nothing lacking; the fork gives its rare side |
| 702.103e | 0 / 1050 | fires | skulking_ghost | fires | yes |
| 702.105a | 91 / 66 | fires | tied | fires 2 | yes |
| 702.118b | 0 / 0 | both | lesser_power | does_not_fire | yes |
| 702.124j | 19 / 671 | fires | named_card_in_library | fires | yes |
| 702.126a | 2044 / 0 | does_not_fire | one_artifact_tapped | does_not_fire 24 | yes |
| 702.127a | 2779 / 0 | does_not_fire | from_hand | does_not_fire 24 | yes |
| 702.131a | 0 / 1458 | fires | ten_permanents | fires | yes |
| 702.133a | 2779 / 0 | does_not_fire | cast_from_hand (forced cast) | does_not_fire | yes on stale counts; unknown on current declarations (finding 2) |
| 702.170c | 42 / 9 | fires | small_card_in_hand | fires | yes |
| 702.187b | 8 / 0 | both | one_mountain | does_not_fire 8 | yes |
| 702.188a | 78 / 0 | both | creature_untapped | does_not_fire 8 | yes |
| 704.5b | 0 / 0 | both | empty | fires | yes |
| 704.5f | 6814 / 15221 | none (always listed) | tougher | does_not_fire | nothing lacking |
| 712.4a | 17 / 3 | fires | meld_pair | fires | yes |
| 712.14c | 17 / 3 | fires | meld_pair | fires (through 712.4a-1) | yes |
| 718.3a | 303 / 0 | does_not_fire | two_mountains | does_not_fire 24 | yes |

## Checked and found correct

- **C, 205.1a's type-set reading.** `LayerContributions.typeTokens` writes `-all-<kebab RemoveType>` for each `RemoveType` of a `CardChangedType` and `=` only for a `StateChangedType`. Song of the Dryads and Imprisoned in the Moon are `RemoveCardTypes$ True` plus `AddType$`. So `-all-card-types` is the token a type set is written as, and the rule's own base is its `fires` side. The played fork is tagged `fires`. The closed list now holds every class the collector can write (super, card, sub, land, creature, artifact, enchantment). No other declaration uses the new tokens. The gen-1 contract `specs/023-ability-effect-model/contracts/record-schema.md` already documents them.
- **C, `source_dies`.** It reads `source`, or `cause` on gen-2 records, against a battlefield-to-graveyard `zone_change` of the same id in the same record, which is what the collector writes: the live trade is tagged `fires`, the first striker's fork `does_not_fire`. Only 510.2-1 and its `same_as` 703.4m-1 use it, and 703.4m-1 still states 510.2's fact. The qualifier is documented in the family's docstring, where the other qualifiers live. The fault is the breadth of the near miss (finding 2), not the reading.
- **C, tests.** `tests/unit/effects/domain/rule_detectors/`, `test_rules_catalogue.py` and `test_directed_fork_flags.py`: 459 passed.
- **D, isolation.** `ForkIsolationTest` with `RecordCharacterizationTest` (28 tests) stays green with my temporary match added. That match runs four burn mirrors with 118.12b's directed fork on, two with plain AI seats and two with random seats. The live records are byte-identical with forks off and on. Each of the four forks was kept, wrote a Sanctuary search with `phase_skipped`, and was tagged `fires`. `afterEdit` reads only the copy: the binding is the copy's and `added` reads the copy's battlefield. The registry twin check calls `afterEdit` and the forced-line check.
- **D, budget.** For 118.12b's answer, nothing is lost by skipping the live dry run (finding 5). An in-fork refusal is still a counted `ForkRefusal`, thrown before `writeForkLines`.
- **E, GAP-2 machinery.** Each guard went red under its named test:
  - the scheduler's dry run: `aRefusalFoundBeforeTheForkSpendsNoBudgetAndLeavesTheRuleItsFork`;
  - `forced.finish()`: `anAnswerNeverAskedIsRefusedOnceTheTurnIsOver` and `aRefusalFlaggedInTheLastStepIsThrownOnceTheTurnIsOver`;
  - the AI-affordability check: `aLineTheAiCannotPayForIsRefusedBeforeItsCardMoves`;
  - the forced-yes refusal: `aForcedYesThatCannotBePaidIsRefused`;
  - the attack re-target: `anAttackTheSeatAimsElsewhereIsMovedToTheNamedDefender`;
  - the answer matching its line: `anAnswerIsGivenOnlyToItsLine`;
  - the modal refusal: `aForcedCharmHasNoLegalLine`;
  - `keeping` combining: `keptRecordsCombine`;
  - the Python `same_as` of a scenario's own rule: `test_a_same_as_naming_a_scenario_of_its_own_rule_is_refused`.
- **E, the forced yes pays.** `ForcedPlayRun.pay` is Forge's `PlayerControllerAi.payCostToPreventEffect` without the `willPayUnlessCost` question. The fork's records of the forced Preferred Selection payment show the Selection sacrificed and four Forests tapped, each with its `mana_produced` record, as a live payment writes them. A random seat routes the question the same way.
- **E, removed options.** Nothing in the code, the spec, the contracts, the `CLAUDE.md` files or the effects docs still names `withX`, `ForcedPlay.at` or a later-step option.
- **F, removed rules' reasons.** Each is true against the code and Forge.
  - 118.11 (a cost's sacrifice moved by a replacement): same mechanism as 400.6.
  - 122.1d (a stun counter replaces the untap): Forge's stun replacement in `Card.java` is unconditional.
  - 702.19e (trample over planeswalkers, the planeswalker gone): the 702.19f fork's records are tagged 702.19e `fires` in my played game.
  - 701.42b (a meld fails when one partner is a token): no operation makes a token; `AddPermanentOperation` and `ReplaceCardOperation` put cards.
  - 702.70b (each poisonous instance triggers): no Forge card has `K:Poisonous`; Virulent Sliver and Snake Cult Initiation grant it through a static.
  - 701.42c: its scenario is `same_as` 701.42b-1.
- **G, the list regenerates.** `directed-rules --counts rules/directed-fork-counts.json` rewrites the list, the wording cards and the counts unchanged: git clean, 49 rules (36 rare, 24 one-sided, 2 always).
- **G, spec and docs agree with the code and each other**, apart from finding 6:
  - FR-014b's `NOT_DIRECTED` list, all 43 rules, and its count of 49;
  - FR-010b and FR-014a, which says that 205.1a's and 111.10p's variants of polarity `other` make the `fires` side through their forced cast;
  - `contracts/catalogue-extensions.md` (the forced-play catalogue, `making`, the seed sentence);
  - `contracts/cli.md`, forks.md and rules-catalogue.md (the `same_as` refusal).
- **G, the experiment record.** The known-limit entry adds up: four groups of 6, 5, 4 and 6 rules, plus 702.19e, gives 22, which matches the "no single board edit" part of `NOT_DIRECTED`. Its header keeps "nine known limits", and the Future plans bullet matches. The edited prose keeps to the `feature-workflow` rules. One nit: the 702.19e sentence sits between "They fall into four groups" and the groups themselves.
- **G, named cards.** `DirectedRulesContractTest.TheCardsARuleNames` checks that every card a rule class names is in its scenario's `cards`, and it passed in the full suite. `aac582dc` does not compile alone (it deletes three rule classes that `DirectedRules` still names until `398d1c82`), as known.
- **H.** See finding 2.

## Not checked, and why

- **A gen-2 recount with the current declarations.** It is a heavy job the user launches, so finding 2's conclusion for 702.133a and the other 21 rules is a reasoned expectation.
- **How often each precondition binds in real sealed games.** I checked only whether its cards exist in booster sets (finding 3). There is no game-state corpus to replay.
- **A full prose second pass over the whole experiment record.** I read only the passages the GAP commits edited.
- **The guards the GAP-3 implementer reverted** (one per new class, in its scratch `GAP3-mutate.py`): I chose other guards in the same classes rather than repeat theirs.
- **The GAP-1 rules' negative tests beyond the three I sampled** (119.7, 608.2p, 702.170c), and the GAP-2 thread-safety argument (read, not reproduced).
