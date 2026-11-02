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
| `aimed_at` | `["704.5f", …]` | aimed records only; `[]` allowed |
| `directed` | `{"variant": "tougher", "operation": "pt"}` | directed-fork records only |

Flag signatures, added row:

| record | `interventional` | `fork` | `directed` | `aimed_at` |
|---|---|---|---|---|
| directed fork | `false` | `true` | set | `[rule]` |

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
| player | `speed`, `city_blessing`, `rad`, `monarch`, `initiative`, `turn_counts` | below, "A player's facts" |
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
| `granted` | bool: the line is not printed on this card (a static's, an attachment's, an effect's, an emblem's or a splice's) | every line |
| `active` | bool | statics of the face up (`checkConditions()`); `false` on every line of a face not up |
| `cast_with` | int ≥ 1: times the spell this card was cast as paid the line's cost (1 for an alternative cost or kicker, the count for multikicker) | `keyword` lines of a card on the stack or a permanent cast so |
| `chosen` | list of `Choices$` positions the ability on the stack chose, in chain order | the line of a modal ability on the stack |
| `fragment` | the keyword instance's own fragment as Forge writes it (`Ward:2`, `Annihilator:2`, `Protection:Goblin`) | `keyword` lines |

| `affected` | list of entity refs: the permanents the static applies to, from Forge's static effect | statics of the face up that Forge applies as a continuous effect; absent on every line of a `continuous` record's board, whose answer it would be |

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
static granted, including a trait built by a keyword instance it granted.

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
| `speed`, `rad` | int |
| `city_blessing`, `monarch`, `initiative` | bool |
| `turn_counts` | `{cards_drawn, life_gained, life_lost, damage_taken, permanents_to_graveyard, spells_cast_last_turn, dungeons_completed, descended}`, ints. `permanents_to_graveyard` counts cards put into this player's graveyard from the battlefield this turn; `descended` and `dungeons_completed` are whole-game counts, as Forge keeps them |

### Command-zone objects

| list | one object per | `id` | `source_lines` | `affected` | `remembered` | `duration` | `shield_left` |
|---|---|---|---|---|---|---|---|
| `floating_effects` | effect card in a command zone (not an emblem, not a dungeon) | its entity id `E<card id>` | the key of the ability that made it; `[]` when none did (the monarch, the initiative, the city's blessing) | the cards its statics apply to | its remembered cards and players | its maker's `Duration$`, bucketed; `permanent` when no ability made it | the `Number$` SVar its replacement's `Amount$` spends, else `null` |
| `floating_effects` | timestamp of a static-id-0 layer-table row on a snapshot entity (P/T set and boost, keywords, traits, types, colours, names) drawn while a noted live clause resolved | `F<timestamp>` | the key of that clause | the entities carrying a row of that timestamp | `[]` | the clause's `Duration$`, else its root's, bucketed | `null` |
| `delayed_triggers` | pending delayed trigger, each once | `D<trigger id>` | the key of the ability that spawned it | `[]` | the trigger's remembered cards and players | `end_of_turn` for a this-turn trigger, else `permanent` | `null` |

Effect cards come first, then timestamps in ascending order. A timestamped row no noted clause
covers is not listed. A note is taken at each live clause's `ON_CLAUSE_RESOLVING` and closed at
its `ON_CLAUSE_RESOLVED`; a fork's snapshot reads the live game's notes. `duration` buckets
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

A library card an event names is rendered at the write from the object the library held. A card
put onto the battlefield is that same object, so its board state (tapped, counters) reads as after
the move; any other move leaves the library's object as it was.

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
entity for is still named.

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
| `affects` | a static line; a timestamped floating effect `F<ts>` | each permanent it applies to | the static's `affected`; the floating effect's `affected` |
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
| resolution | `conditions_met` | `[{segment, met}]`, after `events`, on every effect half: one entry per conditional clause that resolved in the half, in resolution order; a clause is conditional when it carries a parameter `SpellAbilityCondition.setConditions` reads (`Condition$`, `ConditionPresent$`, `ConditionCheckSVar$`, …; not `ConditionDescription$`); `segment` is spelled as the clause's events' `attributed_to` (`root`, or the chain index as a string), `met` is `metConditions()` read at the clause hook, before the engine reads it; `[]` where no conditional clause resolved, and always on a mana ability's record and the game-end rescue record |
| decision candidate | `verdict.timing` | bool |
| decision candidate | `failed_checks` | `[{check, responsible_static}]`, after `responsible_static` in the candidate, one entry per failed check in the order listed here; `responsible_static` a key list, one key where a static decided the check (`static_prohibition`), else empty; `check` ∈ `split_second`, `casting_speed`, `whose_turn`, `phase`, `combat`, `activation_limit`, `script_condition`, `zone`, `static_prohibition`, `cost`, `script_restriction` |
| decision candidate | `legal_targets` | union over every targeting clause, uncapped |
| decision candidate | `responsible_static` | the key of the static forbidding the play (its `static_prohibition` entry's), else empty |
| `attackers` | `anchor_attacker`, `legal_defenders`, `must_attack`, `tapped_by_attack?` | one record per creature the attacking player controls, these four first and `legal_attackers`, `forbidden` after them; `anchor_attacker` a card ref; `legal_defenders` the refs (`P`/`E`) of the declaration's defenders (players, planeswalkers, battles) the anchor may attack, in Forge's defender order, `[]` for a creature that may attack none; `must_attack` whether an attack requirement applies to the anchor (`AttackRequirement.hasRequirement`), whether or not it can attack; `tapped_by_attack` on a declared anchor only: whether a `GameEventCardTapped` for it arrived between the engine's first question and `GameEventAttackersDeclared`. `legal_attackers` and `forbidden` keep their gen-2 shape and are the declaration's, the same on each of its records: a creature is legal when it may attack at least one defender, and a forbidden one's `responsible_static` is the first static, in defender order, keeping it from a defender |
| `blockers` | `must_block` | after `forbidden`: the refs of the listed creatures, legal or forbidden, in candidate order, that a requirement makes block the anchor; a listed creature not in it is false. Requirements: the creature's `getMustBlockCards`, the anchor's `MustBeBlockedBy` (while no creature of that kind blocks it) and `MustBeBlockedByAll`, a lure ("All creatures able to block CARDNAME do so."), and a `MustBlock` static naming the creature (names every attacker); a requirement whose block costs something is none (509.1c); recorded whether or not the creature can obey it |
| probe `combat` | damage events' `source` | entity ref |

## Behaviour changes readers rely on

- A combat record's snapshot precedes every damage event of its step.
- An attackers record's snapshot is the board before the declaration, and the record is written when
  the declaration is made, at the next declaration, or at the game's end; only the first gives
  `tapped_by_attack`.
- A resolution's effect half holds every death of the state-based loop after it. A trigger record
  never holds one.
- A prevention event with amount 0 is not written.
- `can_play = canPlay ∧ timing ∧ no static prohibition`.
- Decision candidates exist for both players at every priority stop.
- A decision candidate is every spell ability of every card its player could play from, unplayable
  ones included, lands and mana abilities aside; an alternative cost is a candidate of its own.
- `whose_turn`, `phase` and `combat` name the parts of Forge's timing restrictions,
  `activation_limit` and `script_condition` the parts of its other restrictions that fail;
  `script_restriction` is a failed activator check, or a failed restriction none of those names.

## Removed

- `what_if` on `attackers` records (none is written).
- A card's `printed` and `granted_attached` (both now in `lines`) and `pt` (now `base_pt` and `pt_boosts`).
- The AI hook's decision records (replaced by the enumerator's).
