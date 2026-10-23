# Rules catalogue findings

Implementation notes drawn from the catalogue, written for whoever implements gen-3's rule
coverage. The conclusions and recommendations are in the gen-3 design record,
`experiments/2026-XX-XX-effect-model-gen3-design.md`, in its Result. This
file holds the detail behind them: which rules need which new record fields, why rules were skipped,
what the scenario writers suspect about Forge, and what kinds of errors the verifier fixed.

Counts in sections 3 and 4 are approximate. They come from reading the 762 `forge_notes` and the
566 `review` entries and grouping them by hand, so they can be off by about a tenth. Every other
count is exact and comes from the sidecars; `python scripts/rules_catalogue.py status` regenerates
the totals.

## 1. Rules that need new record fields

210 mechanical or legality rules are triaged `needs_new_fields`: a game shows the outcome, but no
gen-2 record field captures it. Grouped by what the snapshot or record would have to add:

| what is missing | rules | gen-3 field it suggests |
|---|---|---|
| game-level designations | day/night 502.2, 702.145b–g, 703.4b, 731.1–731.2c; monarch 725.2, 725.3, 725.5; initiative 726.2, 726.3, 726.5; dungeon and venture marker 309.4a, 309.5a, 309.5b, 704.5t; speed 704.5aa; city's blessing 702.131b, 702.131d; enduring story 702.195a, 702.195c | a global block: day/night, monarch, initiative holder, each player's dungeon and room, speed, city's blessing, enduring story |
| per-permanent designations and state | battle protector 310.9d, 310.9f, 310.9g; soulbond pairing 702.95a–e; renowned 702.112b; solved 719.3b; prepared 722.3a–c; Room unlocked halves 709.5, 709.5b, 709.5d, 709.5f, 709.5g; class level 716.2b, 716.2d, 716.4; merged components 730.2e–j; stored die results 706.8a, 706.8b; flipped 110.5c; loyalty 306.5c; stickers 400.7m | per-entity fields: designations as a set, paired-with, protector, class level, unlocked halves, merged component list, loyalty |
| characteristics the snapshot does not show | mana value 202.3–202.3g; X 107.3c–107.3n, 702.37f; cost riders 118.8d, 118.9c; base P/T 208.4b; characteristic-defining abilities off the battlefield 113.6a; alternative characteristics on the stack: Adventure 715.2b, 715.3b, 715.3c, 715.4, Omen 720.2b, 720.3b, 720.3c, 720.4, prototype 718.2a, 718.3c, split 709.3c, 709.4a; face-down 406.3a, 708.4, 708.7, 708.11, 708.12, 121.8; meld and transform 712.4b, 712.20 | per stack object and permanent: mana value, the X chosen, the face or alternative characteristics in force, base P/T beside current P/T |
| layers, text changes, timestamps, dependency | 613.1a, 613.1c, 613.2, 613.4, 613.7c, 613.7f, 613.7g, 613.8a, 613.8c; 612.2–612.8; 701.12h, 702.96c; devotion 700.5, 700.5a; 701.3c; 604.7; 707.7 | layer attribution per contribution. Expensive, and the `continuous` record already shows each layer's result |
| linked abilities and remembered information | 607.2d, 607.2e, 607.2f, 607.2g, 607.2j, 607.2m, 607.2q; 614.12c, 614.14 | per permanent: the objects "exiled with" it, and the value or word its linked ability chose |
| what happened earlier in the turn | 700.10 (activated this turn), 700.11 (descended), 702.185c (warped), 702.194b (teamwork), 307.5a (cast at instant speed), 701.4b (beheld), 702.35c | per-turn counters on the global block: spells cast, permanents put into a graveyard, abilities activated per permanent |
| turn structure, priority, durations | 500.5, 500.5a, 500.7–500.11, 502.3, 505.5, 703.4q; priority 117.3a, 117.3b, 117.3c; special actions 116.2b–116.3; mana pool 106.6, 106.6a, 118.3a; 702.26f; game start 103.6a, 103.8a | step and turn markers already exist; extra or skipped turns and who holds priority are the gaps (priority is also in the turn-rule section of the design record) |
| subgames, restarts, controlling another player, endless loops | 727.1–727.6, 729.1b–729.6, 723.1–723.7, 104.4b, 104.6 | none recommended; these are rare enough to leave out |
| other | emblems 114.2; counters on entry 122.6a; draw replacements 121.2a, 121.6c; command zone 400.4b, 400.9, 400.10; zone identity 400.7a, 400.7b, 401.5, 401.6; reveals 701.23e; splice 702.47a–e; copy of a permanent spell 608.3f | case by case |

`grep '"needs_new_fields"' rules/triage/*.jsonl` lists each rule with its one-line outcome and note.

## 2. Rules skipped instead of illustrated

95 rules are listed under `skipped` in their unit's scenario file, each with a reason. They are of
four kinds:

- **No Forge card produces the situation**, the large majority. Examples are two instances of the
  same keyword on one object (702.23c rampage, 702.38b amplify, 702.58b graft, 702.153b casualty,
  702.154d enlist), mana produced as a hybrid, Phyrexian, generic or snow symbol (106.8–106.11),
  cost reductions by such symbols (118.7d–g), a battle that is not a Siege (310.8, 704.5w), and a
  deathtouch or infect source dealing damage from outside the battlefield (702.2d, 702.90e).
- **The situation exists only outside a two-player game without a sideboard**: the command zone
  (400.9, 400.10, 702.49d), cards from outside the game (309.2d, 400.11b, 400.11c), and "you get"
  extra steps on another player's turn (500.10a).
- **No observable outcome distinguishes the cases**: redundant keyword instances (702.3c, 702.4e),
  phasing directly and indirectly at once (702.26h), and a prevention effect applying only once
  (615.12a).
- **Another scenario in the unit covers it**, which the reason names (for example 117.1c, 615.2,
  715.2b).

The "no Forge card" reasons rest on the writer's grep of `forge-gui/res/cardsfolder`. A card added
to Forge later can make a skipped rule reachable.

## 3. What the scenario writers flagged about Forge

762 of the 1,763 scenarios carry `forge_notes`. They are of eight kinds:

| kind of note | about | examples |
|---|---:|---|
| engine behaviour the writer could not confirm: timing, durations, last known information, trigger counts | 190 | 500.4-1, 702.101b-1, 603.7c-1 |
| how the card is scripted (informational, not a risk) | 170 | 607.2a-1, 701.10b-2, 614.15-1 |
| scenario structure: where cards sit per variant, a rule without a `does_not_fire` branch | 110 | 105.2-1, 301.5b-1, 701.51c-1 |
| setup state a puzzle file may not express: designations, counters, history, phased out, face down, day/night, monarch, initiative, dungeon progress | 90 | 702.103f-1, 726.2-2, 702.26e-1 |
| a suspected Forge deviation from the rules | 75 | 400.7c-1, 701.60d-1, 702.55b-1 |
| a choice, die roll or AI decision the harness must force | 55 | 702.104a-1, 703.4g-1, 727.1-1 |
| card naming and faces: split, Room, flip, meld and back faces appear under another name | 45 | 709.4-1, 202.2e-1, 701.61a-2 |
| an outcome that no record field shows | 25 | 309.2a-1, 310.5-1, 701.23e-1 |

**Suspected Forge deviations to test first.** Each is a writer's reading of Forge's scripts or source,
and none has been confirmed by running Forge:

1. 400.7c-1: `ChooseSource` may not follow a spell to the battlefield.
2. 407.4-2: a ChangeZone to the ante zone has no owner check, so a stolen card is anted into the wrong player's zone.
3. 701.60d-1: `setSuspected` returns true for an already-suspected card, so Frantic Scapegoat's suspicion ends wrongly.
4. 701.50e-1: `ConniveEffect` has no zero-count check, so connive 0 fires the "connives" trigger.
5. 106.12b-1: Mana Reflection may double Ashnod's Altar's mana although the Altar has no tap cost.
6. 702.55b-1: the haunt trigger (`Creature.HauntedBy`) may not fire when the haunted card is a non-creature.
7. 701.17d-1: Charmed Pendant may read only one milled card when milling is doubled.
8. 506.4c-1, 702.19e-1, 510.1b-1: an attacker may be removed from combat, or its damage redirected, when the planeswalker it attacks leaves.
9. 702.7c-2: the creatures that deal damage in the second step may be re-selected, so a creature whose first strike is removed between the steps strikes twice.
10. 104.2a-1: `Player.hasWon()` returns false while `cantWin()` holds, so Forge may not report the game as won in the Platinum Angel scenario.

## 4. What the verifier fixed

The Opus verifier made about 570 changes, logged one per line in each scenario file's `review`
array. About 30 to 40 of them, 6% to 7%, changed a rules-level expected outcome. The rest fixed the
scenario's form.

| error type | about | examples |
|---|---:|---|
| skip reason false, or triage wrong (rule reclassified, record kinds added, triage file missing) | 90 | 608.2k skip false, scenario added; 701.37b definitional → mechanical |
| base variant carried a delta, or a variant's action missing from the decisions | 85 | 506.4a-1, 707.9c-1 |
| setup cannot complete: a draw from an empty library (about 30), missing mana or land, summoning sickness | 75 | 513.2-1, 702.170a-1 |
| variant tests another rule, or changes nothing | 70 | 107.3h-1 tested colored payment, not X = 0; 119.4b-1 no difference |
| decision left open or in the wrong order, priority passes wrong | 55 | 503.1a-1 trigger order open; 611.3b-1 passing on an empty stack ends the step |
| polarity label wrong | 50 | 205.1b-1 inverted; 701.6b-1 relabelled `other` |
| expected text carried reasoning or left out a consequence | 45 | 609.4-1, 702.112a-1 |
| variant changes more than one fact | 40 | 208.2a-1, 724.2f-1 |
| wrong rules outcome | 30 | 702.11b-2: a fizzle on recheck (608.2b) was written as a resolution |
| wrong card fact: cost, color, ownership, loyalty | 20 | 104.3e-1 Phage's cost; 704.5m-2 Pacifism's owner |

A tool that writes more scenarios, or one that turns these into Forge states, should check the
same list mechanically where it can. The checks that can be automated are every library non-empty
when a draw step lies between the setup and the outcome, mana available for each cast, an empty
`delta` on the base, and one fact changed per variant.

## 5. Limits of the scenarios as written

- **Deltas and decisions are prose.** A directed fork needs each `delta` classified into a fork
  operation: change power or toughness, add or remove a keyword, change a colour or type, add or
  remove counters, tap or untap, change life, replace a card. Only a puzzle-style setup, the last
  resort, needs the `decisions` turned into structured actions (cast, activate, target, pay,
  attack, block, assign, pass), since mined and forked records take their decisions from the game.
- **Part of the setup lives in `notes`.** Counters, damage and attachments have fields; designations,
  history and anything unusual are in free text.
- **Tokens are not in `cards`.** The validator checks names against `cardsfolder` only, so a token
  such as a Food appears in `notes` and its script lives under `tokenscripts`.
- **No scenario has been played.** Every expected outcome is the writer's reading, checked by the
  verifier against the rules text and the cards' Forge scripts, not against the engine.
- **A few cards dominate.** Grizzly Bears appears in about half the scenarios, and the basic lands,
  Shock, Hill Giant, Lightning Bolt and Counterspell follow. 1,382 distinct cards appear in all.

## 6. What the model can learn: the heads pass

`rules/heads/` maps each of the 1,647 mechanical and legality rules to the output trained on its
outcome and the deciding facts the model does not read, against `MODEL-FIELDS.md`. The totals and
the families are in the gen-3 design record's Result. To list the rules behind any name:
`grep -l 'new:chosen_values' rules/heads/*.jsonl`, or `grep '"output": "missing"' rules/heads/*.jsonl`.

**Names the families group.** The design record's tables count each rule once per family:

| family | names in `inputs_missing` or `outputs` |
|---|---|
| values chosen earlier | `new:chosen_values`, `linked_choice`, `x_paid_on_permanent`, `x_of_other_object`, `optional_costs_paid`, `alternative_cost_paid`, `mana_spent`, `die_result`, `flip_result`, `stored_results`, `cast_from_zone` |
| designations and game-level state | `new:designations`, `day_night`, `monarch`, `initiative`, `dungeon_room`, `speed`, `city_blessing`, `class_level`, `room_unlocked`, `soulbond_pair`, `protector`, `radiation`, `phased_out_flag`, `band_membership` |
| earlier in the turn or game | `new:this_turn_history`, `game_history`, `damage_sources_this_turn`, `last_known_information` |
| faces and copies | `snapshot:face`, `snapshot:copy_source`, `new:face`, `copiable_values`, `merged_components` |
| exiled-with and hidden zones | `new:exiled_with`, `exile_contents`, `library_contents`, `library_top` |
| pending shields and delayed triggers | `new:replacement_shield`, `regeneration_shield`, `delayed_triggers`, `active_effects`, `active_restrictions`, `cast_permission` |
| attachments, attack and stack targets | `snapshot:attached_to_target`, `snapshot:attack_target`, `snapshot:stack_targets`, `planned:block_pairing` |
| zone destinations merged | `record:event:zone_change`, `new:library_zone_outcome`, `exile_zone_outcome`, `zone_outcome_battlefield`, `zone_outcome_command`, `library_position`, `entered_objects` |
| game-level state changing | `new:designations_changed`, `dungeon_progress`, `day_night_change`, `speed_change`, `city_blessing_gained`, `monarch_change`, `initiative_change`, `class_level_change`, `room_unlocked`, `soulbond_pair_change`, `protector_change`, `record:event:day_night_changed`, `dungeon_ventured`, `state_flag_change`, `ring_tempts` |

About 30 input names and 50 output names fall in no family, each on five rules or fewer, for
example `new:attraction_deck`, `new:spliced_text`, `new:base_pt_change` and
`record:event:counter_change`. The scratch script that made the tables is not kept; the groups
above are its whole configuration.

**Where packets disagree.** Each packet was judged separately, so a few neighbouring rules read
the same mechanism differently. The known case was a planeswalker's or battle's entering
counters. 306.5b and 310.4b said they reach `counters_delta_loyalty` and `counters_delta_defense`,
and 209.1 filed them under the refused permanent-spell resolution. The refusal decides it: a cast
planeswalker or battle enters through its own permanent spell's record, which `build-corpus`
drops. The two lines now say `recorded`. Other disagreements may remain, so read the note on a line
before relying on it alone.
