# Contract: record schema, gen-3 (delta)

**Feature**: `025-ability-effect-model-gen3`
Authority: [`../../2026-10-30-ability-effect-model-gen3.md`](../../2026-10-30-ability-effect-model-gen3.md) § 8.
Base: [`../../023-ability-effect-model/contracts/record-schema.md`](../../023-ability-effect-model/contracts/record-schema.md),
[`../../024-ability-effect-model-gen2/contracts/record-schema-delta.md`](../../024-ability-effect-model-gen2/contracts/record-schema-delta.md).

The text below becomes the base contract's "Gen-3" section (FR-019). The frozen-schema rule holds
within gen-3. Relative to gen-2 a field may change meaning or go away.

## Version

- Every gen-3 record carries `"generation": 3` as its first envelope member after `record_id`.
- Readers decide a record's generation by `generation`, else by `random_seat` (gen-2), else gen-1.
- `build-corpus`, `train-effect-model`, `evaluate-effect-model` and the probes accept gen-3 only.
  `rule-counts` accepts gen-2 or gen-3, not mixed.

## Envelope

| Field | Shape | Present on |
|---|---|---|
| `generation` | `3` | every record |
| `aimed_at` | `["704.5f", …]` | aimed records only: every `collect-coverage` and `collect-variants` record (the run's `--directed-fork-rules`, `[]` when it names none) and every directed fork (`[rule]`); never another `match-outcomes` record |
| `directed` | `{"variant": "tougher", "operation": "pt"}` | directed-fork records only |

Flag signatures, added row:

| record | `interventional` | `fork` | `directed` | `aimed_at` |
|---|---|---|---|---|
| directed fork | `false` | `true` | set | `[rule]` |
| protection fork (FR-058) | `true` | `true` | absent | the run's, as a live record's |

A protection fork's records are of any kind, played on to the end of the turn, and its cost halves
carry `link_id` as live halves do. An interventional resolution's forced effect half has no
partner, and neither has a protection fork's mana record (every event `mana_produced`), as a live
one has none; no flag tells the two apart. Readers accept an interventional record of every kind, with
or without `link_id`.

`mirror_of` on a directed-fork record names the live record open at the fork point, and is absent
when none was open.

## Kinds

`turn_action`, with `subkind` one of `draw`, `cleanup_discard`, `untap`, `phasing`, `day_night`,
`lore_counter`; payload `{"events": [...]}`. No `ability`. Never written to a curated stratum.

## Snapshot additions

Shapes follow `rules/gen3-fields.json`; the collector writes the field names below.

| Where | Field | Shape |
|---|---|---|
| entity | `owner` | player ref |
| card | `lines` | list of `{keys, kind, granted, active?, cast_with?, chosen?, fragment?, affected?}`; `kind` ∈ `spell`, `activated`, `trigger`, `static`, `replacement`, `keyword`; `active` on statics, and `false` on every line of a face not up |
| card | `face_state` | `front` \| `back` \| `flipped` \| `melded` \| `face_down` |
| card | `base_pt`, `pt_boosts` | `[p, t]`; boosts `{temporary: [p, t], counters: [p, t]}`; both `null` on a non-creature; base + temporary + counters = net P/T |
| card | `counters` | `{type: n}` over every named type |
| card | `designations`, `x_paid`, `chosen`, `costs_paid`, `history`, `order_rank`, `real_face`, `printed_loyalty`, `printed_defense`, `mana_cost` | below, "A card's facts" |
| entity | `role` | `null` on an entity of the four tiers; else `library_card`, `exiled_with`, `mutate_component` or `last_known` (below, "Entities beyond the tiers") |
| player | `max_land_plays`, `speed`, `city_blessing`, `rad`, `monarch`, `initiative`, `turn_counts` | below, "A player's facts" |
| global | `day_night` | `"day"` \| `"night"` \| `null` |
| global | `floating_effects`, `delayed_triggers` | `{id, source_lines, affected, remembered, duration, shield_left}`; `duration` ∈ `end_of_turn`, `until_leaves`, `permanent` (below, "Command-zone objects") |
| global | `dungeons` | `{player, dungeon, room}` |
| refs | `random` | `{dice: [n, …], coin_flips: {won, lost}}` on a resolution's effect half, live or interventional; `null` on every other refs block |
| snapshot | `links` | `[{from, to, type, index?}]`; `type` a § 10.3 family name; `index` only for `targets` (below, "Links") |

A card's `lines` members:

| member | shape | present on |
|---|---|---|
| `keys` | list of 0 or 1 provenance key | every line; `[]` when the line resolves to no printed line |
| `kind` | one of the six kinds | every line |
| `granted` | bool: the line is not printed on this card (a static's, an attachment's, an effect's, an emblem's or a splice's, or a basic land type's mana ability a type change gave a land not printed with that type, such as a Mountain's under Blood Moon, keyed past every line the land's sidecar declares) | every line |
| `active` | bool | statics of the face up (`checkConditions()`); `false` on every line of a face not up |
| `cast_with` | int ≥ 1: times the spell this card was cast as paid the line's cost (1 for an alternative cost or kicker, the count for multikicker) | `keyword` lines of a card on the stack or a permanent cast so |
| `chosen` | list of `Choices$` positions the ability on the stack chose, in chain order | the line of a modal ability on the stack |
| `fragment` | the keyword instance's own fragment as Forge writes it (`Ward:2`, `Annihilator:2`, `Protection:Goblin`) | `keyword` lines |
| `affected` | list of refs: the permanents the static applies to, then the players (`P…`), from Forge's static effect (`StaticEffect.getAffectedCards`, `getAffectedPlayers`) | statics of the face up that Forge applies as a continuous effect; absent on every line of a `continuous` record's board, whose answer it would be |

A `keyword` line's keys name the keyword line the card prints, else the static that granted the
keyword; a keyword a resolved effect granted has `keys: []` and is read by its `fragment`. The
triggers, statics and abilities a keyword instance builds are its line, not lines of their own; a
`Secondary$ True` trait is listed under its own key and joins its primary's line through the
sidecar. A land's play and a permanent's plain cast are not lines. A face not up (a face-down
card's real face, a double-faced card's other face) lists its printed lines with `active: false`;
a split card lists no face not up.

Entity members are written in this order: `id, name, zone, owner, controller, face, face_state,
copy_source, token_script_id, types, subtypes, supertypes, colors, mana_value, mana_cost, base_pt,
pt_boosts, printed_loyalty, printed_defense, tapped, sick, damage, counters, combat, attached_to,
face_down, real_face, designations, x_paid, chosen, costs_paid, history, order_rank,
granted_temporary, lines, stack_extras, role`.

A `continuous` record's board (the board without the record's static) lists no line the removed
static granted, including a trait built by a keyword instance it granted, and lists every line it
removed, keyed to its printed line. Every other entity member reads that board too: `types`,
`subtypes`, `supertypes`, `colors`, `controller`, `sick`, `base_pt` and `pt_boosts` (its set
base, layer 7a and 7b, and its boost left out; `null` when the permanent is no creature without
it), `granted_temporary`. Forge's `...Without` reads (`Card.getTypeWithout`, `getColorWithout`,
`getKeywordsWithout`, `getNetPowerBreakdownWithout`, `getControllerWithout`;
`CardState.getSpellAbilitiesWithout` and its three siblings) compute it.

### A card's facts

| member | shape |
|---|---|
| `owner` | player ref |
| `face_state` | `front` \| `back` \| `flipped` \| `melded` \| `face_down`; a split card, a prepared spell and a specialized card read `front` |
| `mana_cost` | `{symbol: count}` over the non-zero symbols: `generic`, and each other symbol by Forge's short spelling (`W`, `C`, `X`, `S`, `W/U`, `2/W`, `G/P`); `{}` for {0}; `null` for a card with no mana cost |
| `printed_loyalty`, `printed_defense` | int, the face up's printed value; `null` when it prints none or prints X |
| `real_face` | `{name, types, subtypes, colors, mana_value, base_pt}` of the front a face-down card would turn up as (`base_pt` `null` on a non-creature); `null` unless face down |
| `designations` | sorted list over `bestowed`, `cloaked`, `commander`, `foretold`, `goaded`, `harnessed`, `manifested`, `monstrous`, `phased_out`, `plotted`, `renowned`, `ring_bearer`, `saddled`, `solved`, `specialized`, `suspected`, `unearthed`, `warped` |
| `x_paid` | int: the X the spell this card is or was cast as paid; `null` when not cast, or not with X |
| `chosen` | object, each member only when set: `colors` (list), `type`, `type2`, `number`, `player` (ref), `cards` (refs), `names` (list), `direction`, `mode`, `even_odd`, `sector`, `stored_rolls` (list); `{}` when none |
| `costs_paid` | `{mana: {W, U, B, R, G, C}, cast_from: zone or null, mana_rider: bool}` for the spell the card is or was cast as (its spell on the stack, or a permanent's cast spell); `null` when not cast. `mana_rider`: some mana spent carried a rider (can't be countered, adds a counter or a keyword, triggers when spent) |
| `history` | `{entered_this_turn, attacked_this_turn, blocked_this_turn, loyalty_activated}`, bools |
| `order_rank` | int, 0 first: battlefield by timestamp (oldest 0, across both players); stack from the top; library from the top; graveyard, hand and exile in Forge's zone order (earliest put there 0); `null` in the command zone or outside a zone |

### A player's facts

| member | shape |
|---|---|
| `max_land_plays` | int: the land plays this player's turn allows, Forge's `Player.getMaxLandPlays()` (1 plus every `AdjustLandPlays$` raise in force); `null` when `getMaxLandPlaysInfinite()` holds. Written right after `this_turn`, whose `lands_played` it bounds |
| `speed`, `rad` | int |
| `untapped_production` | changed meaning: per mana symbol, what the player's untapped permanents' mana abilities that can be activated now produce, each ability read with that player as its activator; a summoning-sick creature's {T} ability and an ability whose restrictions forbid activation count for nothing |
| `city_blessing`, `monarch`, `initiative` | bool |
| `turn_counts` | `{cards_drawn, life_gained, life_lost, damage_taken, permanents_to_graveyard, spells_cast_last_turn, dungeons_completed, descended}`, ints. `permanents_to_graveyard` counts cards put into this player's graveyard from the battlefield this turn; `descended` and `dungeons_completed` are whole-game counts, as Forge keeps them |

### Command-zone objects

| list | one object per | `id` | `source_lines` | `affected` | `remembered` | `duration` | `shield_left` |
|---|---|---|---|---|---|---|---|
| `floating_effects` | effect card in a command zone (not an emblem, not a dungeon) | its entity id `E<card id>` | the key of the ability that made it; `[]` when none did (the monarch, the initiative, the city's blessing) | the cards its statics apply to, then the players; `[]` on a `continuous` record's board | its remembered cards and players | its maker's `Duration$`, bucketed; `permanent` when no ability made it | the `Number$` SVar its replacement's `Amount$` spends, else `null` |
| `floating_effects` | timestamp of a static-id-0 layer-table row on a snapshot entity (P/T set and boost, keywords, traits, types, colours, names) drawn while a noted live clause resolved | `F<timestamp>` | the key of that clause | the entities carrying a row of that timestamp | `[]` | the clause's `Duration$`, else its root's, bucketed | `null` |
| `delayed_triggers` | pending delayed trigger, each once | `D<trigger id>` | the key of the ability that spawned it | `[]` | the trigger's remembered cards and players | `end_of_turn` for a this-turn trigger, else `permanent` | `null` |

Effect cards come first, then timestamps in ascending order. A timestamped row no noted clause
covers is not listed. A note is taken at each live clause's `ON_CLAUSE_RESOLVING` and closed at
its `ON_CLAUSE_RESOLVED`; a fork's snapshot reads the live game's notes for every timestamp at or
before the copy, and the notes of its own clauses (a played-forward fork's) after it. `duration` buckets
`Duration$`: `Permanent` is `permanent`; `UntilHostLeavesPlay`, `UntilLoseControlOfHost`,
`AsLongAs…`, `…Untaps`, `UntilUnattached` and `UntilFacedown` are `until_leaves`; everything else
is `end_of_turn`, including an absent `Duration$` and any `…OrEOT` or `…OrEndOfCombat`.
`dungeons`: `player` the dungeon's owner, `dungeon` its entity ref, `room` its current room's
name. Delayed triggers need the patched `getPendingDelayedTriggers`; on an unpatched Forge the list
is `[]`.

### Entities beyond the tiers

Appended to `state.entities` after the four tiers, each card once:

| `role` | which cards | written as |
|---|---|---|
| `library_card` | each library's top card; every library card the pending event names; every library card named by the events of a resolution's effect half (live or interventional) or of a `turn_action` record | `zone: library`, `order_rank` its position when the board was captured |
| `exiled_with` | the cards exiled with a permanent of the tiers (`getExiledCards`) | the card as it is |
| `mutate_component` | a mutated permanent's merged cards | the card as it is |
| `last_known` | the pending event's subject as it last existed (the run parameters' `CardLKI`) | id `L<card id>`; `zone` the zone it was last in |

A library card an event names, and each library's top card, is rendered as it was in the library: `tapped` false, no counters,
no `attached_to`, no `combat`, `controller` its owner, face up, `entered_this_turn` false, nothing
`chosen`; its name, types and lines are its library face's. A card put onto the battlefield is the
library's own object, so these fields are not read from it after the move.

### Links

`links` sits in `state` after `entities`. Each link is `{from, to, type, index?}`, written once in
its family's forward direction; the reader takes the reverse type from the other end. `type` is a
family name of `relation_types.FAMILIES` (`FE/LinkFamily.java`, pinned by `LinkFamilyContractTest`
and `test_relation_types.py`). `index` is on `targets` only: the target's position across the
stack object's chain, the root's targets first. Links are listed by family in table order, each
once.

An end is a ref (`E<id>`, `P<id>`, `L<id>`, `F<timestamp>`, `D<id>`) or a line
`{entity, key}`: a link an ability made ends at that ability's line. Where Forge keeps no record of
the ability, or the trait keys to no line, the end is the card. An end the snapshot carries no
entity for is still named. A card that has left the game (a token that died) is no end: a link
to it is not written, though Forge's lists (`getGainControlTargets`, `getDamageHistory`) may
still hold it.

| `type` | from | to | Forge source |
|---|---|---|---|
| `merged_with` | a mutated permanent's other card; the card a melded permanent melded with | the permanent | `getMergedCards`, `getMeldedWith` |
| `attached_to` | the attached card | the card or player | `getAttachedTo`, `getPlayerAttachedTo` |
| `attacks` | the attacker | the player, planeswalker or battle | `Combat.getDefenderByAttacker` |
| `blocks` | the blocker | each attacker it blocks | `Combat.getAttackersBlockedBy` |
| `in_band_with` | the lower card id of two attackers of one band | the other | `Combat.getAttackingBands` |
| `must_block` | the creature | the attacker it must block | `getMustBlockCards` |
| `goaded_by` | the goaded creature | each player who goaded it | `getGoaded` |
| `targets` | a stack object's line (the acting ability's too, before it is on the stack) | each target: a card or player; a spell by its card; an ability by its line | each clause's targets |
| `source_of` | the stack object's host card | its line | the stack object's host |
| `triggered_by` | a triggered ability's line on the stack | each card or player among its triggering objects | `getTriggeringObjects` |
| `affects` | a static line, a command-zone effect card's included; a timestamped floating effect `F<ts>` | each permanent it applies to, and each player (`StaticEffect.getAffectedPlayers`); an `F<ts>` reaches cards only | the static's `affected`; the floating effect's `affected` |
| `created_by` | a command-zone effect card or emblem; `F<ts>`; `D<id>` | the line that made it, else the effect's source card | `getEffectSourceAbility`, `getEffectSource`; the clause note; the delayed trigger's own key |
| `pending_on` | `D<id>` | each card or player it holds | `getTriggerRemembered` |
| `linked_by_text` | the line that exiled the object (`getExiledSA`), else the card | each exiled, imprinted, until-leaves or lasting remembered card or player | `getExiledCards`, `getExiledWith`, `getImprintedCards`, `getUntilLeavesBattlefield`, `getRemembered` |
| `bound_to` | the exiled cipher or haunt card | the permanent | `getEncodingCard`, `getEncodedCards`, `getHaunting`, `getHauntedBy` |
| `controls`, `owns` | the player | each card entity | `getController`, `getOwner` |
| `control_granted_by` | the card whose control was gained | the card that gained it | `getGainControlTargets` |
| `copy_of` | the copy | what it copies | `getCopiedPermanent`; the latest clone state's origin |
| `paired_with` | the lower card id of a pair | the other | `getPairedWith` |
| `protects` | the battle's protector | the battle | `getProtectingPlayer` |
| `chose` | the card | each chosen card, the chosen player, the gift's player | `getChosenCards`, `getChosenPlayer`, `getPromisedGift` |
| `damaged_this_turn` | the damage source | each card or player it damaged this turn | `getDamageHistory` with `Game.getDamageLKI` |
| `fought_this_turn` | the blocker | each attacker it blocked this turn | `getBlockedThisTurn`, `getBlockedByThisTurn` |
| `crewed_by` | the Vehicle or Mount | each creature that crewed or saddled it this turn | `getCrewedByThisTurn`, `getSaddledByThisTurn` |
| `in_dungeon` | the dungeon | its owner | the command zone's dungeon |
| `last_known_state_of` | `L<id>` | the subject's card | the pending event's last-known copy |

A remembered entry is a link only when it is a card or a player and no ability of the remembering
card is on the stack or resolving (C8). A `continuous` record's board writes no `affects` from a
static, as it writes no `affected`. `line_of`, `line_of_the_same_card`, `mode_of` and
`has_the_same_name` are never written: the reader builds them from its own layout (T116).

## Payload additions

| Payload | Field | Shape |
|---|---|---|
| resolution, cost half | `events` | before `costs`: the cast card's `zone_change` to the stack, from the zone it was cast from (`attributed_to: unresolved`, `cause` the card); `[]` on an activated ability's cost half. The payload is `{events, costs}`. Every cost half carries `link_id`; a triggered ability and a copy of a spell or ability, which nobody casts, write no cost half, and their effect halves no `link_id` |
| resolution, effect half | `outcome` | after `conditions_met`: `resolved`, `fizzled`, `partially_fizzled` or `declined`; a `fizzled` or `declined` half has `events: []`; `null` on a mana record and the game-end rescue record, which resolve no stack item |
| resolution | `conditions_met` | `[{segment, met}]`, after `events`, on every effect half: one entry per conditional clause that resolved in the half, in resolution order; a clause is conditional when it carries a parameter `SpellAbilityCondition.setConditions` reads (`Condition$`, `ConditionPresent$`, `ConditionCheckSVar$`, …; not `ConditionDescription$`); `segment` is spelled as the clause's events' `attributed_to` (`root`, or the chain index as a string, counted down the line the record keys to: a triggered ability's root clause is `root`, and a chosen charm mode's clauses are counted down its option line, the mode's own first clause `root`), `met` is Forge's own answer, the clause hook's `ClauseRun` read when the clause's own effect is over (`CONDITIONS_UNMET` false, `RAN` and `UNLESS_COST_DECIDED` true; no entry for a clause that threw); `[]` where no conditional clause resolved, and always on a mana ability's record and the game-end rescue record |
| decision candidate | `verdict.timing` | bool |
| decision candidate | `failed_checks` | `[{check, responsible_static}]`, after `responsible_static` in the candidate, one entry per failed check in the order listed here; `responsible_static` a key list, one key where a static decided the check (`static_prohibition`), else empty; `check` ∈ `split_second`, `casting_speed`, `whose_turn`, `phase`, `combat`, `activation_limit`, `land_play_limit`, `script_condition`, `zone`, `static_prohibition`, `cost`, `script_restriction` |
| decision candidate | `land_play` | bool, after `failed_checks`, on every candidate: `true` on a land-play candidate (`SpellAbility.isLandAbility()`), else `false` |
| decision candidate | `cost_after_adjustment.mana` | the mana Forge would charge now, after every increase and reduction (601.2f): the `Cost` raised by `CostAdjustment.adjust(Cost, sa, false)` (commander tax, `RaiseCost`), its mana part (`getManaCostFor`) reduced by `CostAdjustment.adjust(ManaCostBeingPaid, sa, payer, null, test=true, false)` (`ReduceCost`, set costs, and the delve, convoke and improvise mana the AI would pay), the host's cast-from zone set to where it is meanwhile; X stays an unpaid `{X}`; rendered as a `ManaCost` (`{1}{R}`); absent on a land play, a candidate with no costs, and where the adjustment throws. Gen-3 before T123a wrote the unadjusted `getTotalMana()` |
| decision candidate | `legal_targets` | union over every targeting clause, uncapped |
| decision candidate | `responsible_static` | the key of the static forbidding the play (its `static_prohibition` entry's), else empty |
| `attackers` | `anchor_attacker`, `legal_defenders`, `must_attack`, `tapped_by_attack?` | one record per creature the attacking player controls, these four first and `legal_attackers`, `forbidden` after them; `anchor_attacker` a card ref; `legal_defenders` the refs (`P`/`E`) of the declaration's defenders (players, planeswalkers, battles) the anchor may attack, in Forge's defender order, `[]` for a creature that may attack none; `must_attack` whether an attack requirement applies to the anchor (`AttackRequirement.hasRequirement`), whether or not it can attack; `tapped_by_attack` on a declared anchor only: whether a `GameEventCardTapped` for it arrived between the engine's first question and `GameEventAttackersDeclared`. `legal_attackers` and `forbidden` keep their gen-2 shape and are the declaration's, the same on each of its records: a creature is legal when it may attack at least one defender, and a forbidden one's `responsible_static` is the first static, in defender order, keeping it from a defender |
| `blockers` | `must_block` | after `forbidden`: the refs of the listed creatures, legal or forbidden, in candidate order, that a requirement makes block the anchor; a listed creature not in it is false. Requirements: the creature's `getMustBlockCards`, the anchor's `MustBeBlockedBy` (while no creature of that kind blocks it) and `MustBeBlockedByAll`, a lure ("All creatures able to block CARDNAME do so."), and a `MustBlock` static naming the creature (names every attacker); a requirement whose block costs something is none (509.1c); recorded whether or not the creature can obey it |
| `continuous` | contributions' `set_pt`, `controller` | `set_pt` after `pt_boost`: `[power, toughness]`, the base the static sets (layer 7a, then 7b), a half `null` where it sets only the other; `controller` last: the player ref the static gives control to (a `GainControl$` static, when the permanent's controller without it differs). Each present only when set. A static whose only change is one of these is a `continuous` record |
| `combat`, observed and probe | damage events' `source` | entity ref, after `cause`: the dealer the bus names as `cause`, else the one source in `assignment_choices` that assigned damage to the event's subject, or among several the one that assigned exactly its amount; absent when none can be named. Written on the observed record too, so a probe branch's damage reads as the record it mirrors |

## Event additions

No new event type and no new parameter outside a type's `EVENT_PARAMS` row: each fact rides a
parameter the row already declares, a `characteristics` member, or the provenance `mode` any event
may carry. `state_flag_change` gains flags outside `DESIGNATIONS`, listed below.

| Event | Field | Shape |
|---|---|---|
| `tapped` | after a `zone_change` onto the battlefield | subjects the card, written when it entered tapped, in the record that moved it |
| `tapped` | mana record | subjects each permanent the mana ability's cost tapped, before `mana_produced`. A triggered mana ability (Wild Growth), never paid for, writes no record of its own: its `mana_produced` follows in the record of the activation that set it off |
| `zone_change` off the stack | last event of a spell's resolution in its effect half | the resolved spell's card to the graveyard (or wherever `finishResolving` puts it), after the resolution's own events and before the state-based deaths the half holds; a permanent spell's move onto the battlefield is made while it resolves. A forced resolution's half takes it too |
| `zone_change` into a library | `library_position` | int, after `to_zone`: where the card sits in the library as the bus publishes the move (after the add): `0` the top, `-1` the bottom of a library holding other cards, else its index from the top; a card alone in its library is `0`. Absent on every other move, where the card is not found in the library, and on a move into a library that a `library_shuffled` of that library's owner follows in the same event buffer (the shuffle takes it off). Never written for a move out of a library |
| `pt_change` | `set_pt` | `[power, toughness]`, after `toughness_delta`: the base the change set (`getNetPowerBreakdown().currentValue`, the snapshot's `base_pt`), written when the base moved and the card is a creature after the change: a layer-7b setting effect, a CDA, a face change, an animation of a noncreature. A `pt_change` is written when the base moved even if the total did not (deltas `0`) |
| `card_looked_at` | `mode` | `Scry` on a scry's look, `Surveil` on a surveil's look (bus events; a surveil keeping every card on top writes no `card_milled`, so only `mode` tells the two apart), `SearchedLibrary` on a search: `subjects` the searching player, `from_zone` `library`, no `count`; written by the `ChangeZone` clause rule when the player's `Player.getLibrarySearched()` rose across the clause, so only a search of one's own library (Forge counts no other) |
| `token_created` | `characteristics.controller` | player ref of the sample token's controller (null where it has none) |
| `token_created` | `characteristics.counters` | `{counter name: count}` the sample entered with, names as an entity's `counters` spells them, `{}` for none. Forge batches tokens by controller and counters as well as by script and characteristics, so the sample speaks for its whole batch |
| `ability_change` (`Animate`, `AnimateAll`) | `subjects` | refs of the animated cards still in play after the clause: `Animate` the clause's `getCardsfromTargets`, `AnimateAll` the cards of its `Zone$` (battlefield by default) of the targeted or defined players, or of everyone, matching `ValidCards$`, read before the clause |
| `ability_change` | `abilities` | comma-joined SVar names of every trait granted: `Abilities$`, then `Triggers$`, `Replacements$`, `staticAbilities$` (gen-3 before T123a: `Abilities$` alone); `""` where none |
| `state_flag_change` | `flag` `entered_attacking` / `entered_blocking` | `value` true; subjects the attackers (else the blockers) in combat after a `Token`, `CopyPermanent`, `ChangeZone` or `ChangeZoneAll` clause that were not before it. One event per clause: a `ChangeZone` clause that also searched reports the combat entry |
| `became_blocked` | `subjects`, `blockers` | from a `BecomesBlocked` or `Block` clause: the attackers unblocked before it and blocked after, `blockers` the refs now blocking them (`[]` for `BecomesBlocked`) |
| `state_flag_change` | `flag` `class_level` | subject the Class, `value` the level reached (int), from `ClassLevelUp` when the level rose; one event per level gained |
| `state_flag_change` | `flag` `door_unlocked` | `value` true; subjects the battlefield permanents with more unlocked rooms after an `UnlockDoor` clause than before |
| `state_flag_change` | `flag` `paired` | `value` true; subjects every battlefield permanent paired after a `Bond` clause and not before (both creatures of the pair) |
| `state_flag_change` | `flag` `merged` | `value` true; subject the permanent a `Mutate` clause's spell merged into (`getMergedToCard`) |
| `state_flag_change` | `flag` `city_blessing` / `dungeon_completed` | `value` true; subject the player, read on the player's command-zone notification (`GameEventZone`) as the state against the state last read: `hasBlessing()` turned true, `getCompletedDungeons().size()` rose. Filed wherever the bus event lands (a resolution's effect half, the held half across the state-based loop) |

Not collected: a battle's protector changing (`Card.setProtectingPlayer` publishes nothing), the
position a card left a library from, and a search of another player's library.

## Behaviour changes readers rely on

- Each record describes one moment; its snapshot is the board just before it (root § 8.14):
  - a spell's cost half: before the card moves to the stack, the card where it was cast from and its
    costs unpaid;
  - an activated ability's cost half: before its costs are paid;
  - an effect half: when the resolution starts, before its first clause (a modal half: before its
    mode's first clause);
  - a mana record: before the mana ability's costs are paid and its mana is added.
- A cost half is written when its costs are paid. It does not wait for the stack item's fate.
- What a card's move onto the stack set off, and what an item's bracket gathered before its
  resolution started, precede the effect half's board and reach no record: each is counted as
  dropped.
- A rewrite record's snapshot is read before the replacement runs (`onReplacing`); replacements
  nest, and each record takes the board read when its own replacement started.
- A forced (interventional) resolution reads its board when its resolution starts, as a live one.
- A record's referenced card (tier 1) is the game's own object for it: a dies trigger's host and
  the host of an ability that sacrificed it are the card in the graveyard, written once as
  `E<id>` with its zone; the last-known copy is only the `L<id>` entity.
- `refs.x` is the X paid into the mana cost, else the ability's `SVar:X` when a parameter of its
  chain names X (`NumDmg$ X`, `NumAtt$ +X`, `cmcLEX`), else `null`; a `Count$Random` X is never
  read. It is read once: when a cast item goes on the stack, its costs paid, and when the
  resolution starts for a triggered ability or a copy. The cost half and every effect half of the
  item carry that one reading, never one taken after the resolution.
- A countered stack item writes no effect half. The countering clause's effect half holds the
  countered spell's `spell_countered` event and its card's `zone_change` off the stack.
- A combat record's snapshot precedes every damage event of its step.
- An attackers record's snapshot is the board before the declaration, and the record is written when
  the declaration is made, at the next declaration, or at the game's end; only the first gives
  `tapped_by_attack`.
- A resolution's effect half holds every death of the state-based loop after it. A trigger record
  never holds one.
- A prevention event with amount 0 is not written.
- A record's stat-change events (`pt_change`, `keyword_change`, `type_change`, `color_change`,
  `control_change`) are changes after its own snapshot: the differ re-reads the battlefield when a
  cost half, an effect half at its resolution's start, a mode's half or a damage step takes its
  snapshot. A change Forge
  applied before such a snapshot and published after it reaches no record and is counted as
  dropped.
- `can_play = canPlay ∧ timing ∧ no static prohibition`.
- Decision candidates exist for both players at every priority stop.
- A decision candidate is every spell ability of every card its player could play from, unplayable
  ones included, land plays included and mana abilities aside; an alternative cost is a candidate of
  its own.
- A land-play candidate (`land_play: true`) is judged by `Player.canPlayLand`, split into its parts:
  `whose_turn` when it is not the player's turn (rule 305.3); `casting_speed` whenever `timing` is
  false, `timing` being the player's turn with `canCastSorcery()`, or the play's flash; `zone` when
  the land is on the battlefield, or outside the hand with no effect letting it be played from
  there, or owned by another player without one; `static_prohibition`, with its static, when
  `StaticAbilityCantBeCast.cantPlayLandAbility` forbids it; `land_play_limit` when the player's
  `this_turn.lands_played` has reached `max_land_plays`. `can_play` is `canPlayLand`'s answer;
  `affordable` and `has_legal_target` are `true`, `legal_targets` `[]` and `cost_after_adjustment`
  carries no mana. Its `ability` is the card's land-play trait, which the sidecar lists in
  `dropped_keys`, as it does a permanent's plain cast. The random seat never draws one.
- `land_play_limit` is an identifier of its own, not `activation_limit`: `activation_limit` counts
  one ability's activations per turn or per game (`getLimitToCheck`, `getGameLimitToCheck`), while
  the land-play limit counts the turn's land plays against `max_land_plays`.
- `whose_turn`, `phase` and `combat` name the parts of Forge's timing restrictions,
  `activation_limit` and `script_condition` the parts of its other restrictions that fail;
  `script_restriction` is a failed activator check, or a failed restriction none of those names.

## Removed

- `outcome` on the cost half (the effect half carries it).
- `what_if` on `attackers` records (none is written).
- A card's `printed` and `granted_attached` (both now in `lines`) and `pt` (now `base_pt` and `pt_boosts`).
- The AI hook's decision records (replaced by the enumerator's).
