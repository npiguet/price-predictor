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
- A `generation` other than the integer `3` (`4`, `"3"`, `true`, `null`) names no generation:
  readers refuse the record.
- `build-corpus`, `train-effect-model`, `evaluate-effect-model` and the probes accept gen-3 only.
  `rule-counts` accepts gen-2 or gen-3, not mixed.

## Provenance key

A gen-3 key is `(script_file, face, trait_kind, index_within_kind[, option][, built])`. The member
`built` is new; the others are as in the base contract.

| Member | Shape | Present on |
|---|---|---|
| `built` | `{"kind": "trigger", "index": 2}`: `kind` one of `trigger`, `static`, `replacement`, `spell`; `index` the trait's 0-based position in the keyword instance's list of that kind (`KeywordInterface.getTriggers()`, `getReplacements()`, `getStaticAbilities()`, `getAbilities()`) | a key of `trait_kind` `keyword` that names a trait the keyword instance built; absent on every other key |

- The member is written after `option`, in records and in sidecars alike:
  `{"script_file":"cardsfolder/b/binding_the_old_gods.txt","face":0,"trait_kind":"keyword","index_within_kind":0,"built":{"kind":"trigger","index":2}}`
  is that saga's chapter III trigger. Java `ProvenanceKey.toJson` and Python `ProvenanceKey.as_dict`
  write this string byte for byte, and a record without the member reads `built` as absent.
- Two keys differing only in `built` are different keys. `ProvenanceKey.withoutBuilt()` (Java) and
  `ProvenanceKey.without_built()` (Python) give the keyword's own key.
- A record names a keyword-built trait by its keyword's key with `built`, and never by the
  kind-indexed key (`trigger n`, `static n`, `spell n`, `replacement n`) the trait occupies in the
  card's runtime lists: those keys are declared by no sidecar and emitted by no collector.
- Which lines a sidecar declares `built` keys on, and how a `built` key and a bare keyword key
  join: [`../../023-ability-effect-model/contracts/provenance-sidecar.md`](../../023-ability-effect-model/contracts/provenance-sidecar.md) § Rules.

## Envelope

| Field | Shape | Present on |
|---|---|---|
| `generation` | `3` | every record |
| `fork_kind` | `"probe"` | fork records only, right after `fork`: the kind of fork that wrote the record, one of `intervention` (an interventional resolution), `probe` (a damage-step probe), `protection` (a protection fork) and `directed` (a directed fork); absent on every live record |
| `aimed_at` | `["704.5f", …]` | aimed records only: every `collect-coverage` and `collect-variants` record (the run's `--directed-fork-rules`, `[]` when it names none) and every directed fork (`[rule]`); never another `match-outcomes` record |
| `directed` | `{"variant": "tougher", "operation": "pt"}` | directed-fork records only |
| `copy_dropped` | `{"remembered": 1, "command": 2}` | fork records only (interventional resolution, damage-step probe, directed fork, protection fork), after `directed`: what the copier left out of the fork's copy for want of a counterpart in it, kind to count, kinds in name order, each count at least 1; absent on every live record and on a fork whose copy left nothing out |

Flag signatures, added rows and the `fork_kind` column:

| record | `interventional` | `fork` | `fork_kind` | `directed` | `aimed_at` |
|---|---|---|---|---|---|
| observed | `false` | `false` | absent | absent | the run's on `collect-coverage` and `collect-variants`, else absent |
| interventional resolution | `true` | `true` | `intervention` | absent | the run's, as a live record's |
| damage-step probe | `false` | `true` | `probe` | absent | the run's, as a live record's |
| directed fork | `false` | `true` | `directed` | set | `[rule]` |
| protection fork (FR-058) | `true` | `true` | `protection` | absent | the run's, as a live record's |

A protection fork's records and an interventional resolution's half share `interventional: true`
and `fork: true`; `fork_kind` tells them apart, the premise record (`fork_edit`) included.

A protection fork's records are of any kind, played on to the end of the turn, and its cost halves
carry `link_id` as live halves do. An interventional resolution's forced effect half has no
partner, and neither has a protection fork's mana record (every event `mana_produced`), as a live
one has none; only `fork_kind` tells the two apart. Readers accept an interventional record of every kind, with
or without `link_id`.

`mirror_of` is `null` on a directed-fork record: a directed fork is taken at a phase boundary,
which no live record stays open across. Directed-fork records are paired with the live turn by
`game_id`, turn and `directed`.

A directed fork's records may hold the one play its variant forced its seat to make (FR-010b): a
cast or activation paid for through the engine's own payment, an answer to a "you may", or an
attack. No member marks it, and the stamps stay `fork: true`, `interventional: false`,
`aimed_at: [rule]`, `directed: {variant, operation}`; which variants force what is fixed per rule
in the directed-rule registry. A reader that must tell a seat's own decision from a forced one
reads the variant there.

`copy_dropped` kinds, the values of Forge's `GameCopier.NoCounterpart` lower-cased:

| Kind | What the copy left out |
|---|---|
| `command` | a pending command (end of turn, end of combat, leaves the battlefield, a state-based check's) |
| `delayed_trigger` | a pending delayed trigger, or an extra turn's or extra phase's trigger |
| `remembered` | an object a card or a trigger remembers |
| `link` | a link between cards: exiled with, held until it leaves, gained control of, merged, encoded, haunting, devoured, exploited, delved |
| `attack` | an attack made this turn, by defender |
| `ability` | an ability a copied object names (what made an effect card, what exiled a card) |
| `activation` | a key of a card's activation, resolution or chosen-mode tables |

`copy_dropped` is collection metadata: it decides nothing at collection (the fork is kept and
written as any other) and nothing in training.

## Kinds

`turn_action`, with `subkind` one of `draw`, `cleanup_discard`, `untap`, `phasing`, `day_night`,
`lore_counter`, `fork_edit`, `game_end`; payload `{"events": [...]}`. No `ability`. Never written to a
curated stratum.

`turn_action` / `fork_edit` is a fork's premise, on protection-fork and directed-fork records only:

- The state-based check the fork's edit makes (a protection fork's grant, a directed fork's board
  edit), run once the fork's collectors listen and before the fork begins its phase.
- Its events are those the bracket routes while the check runs (an Aura's `unattached` and its
  `zone_change` to the graveyard, a creature's death at 0 toughness), each `attributed_to:
  unresolved` with `cause` the object the engine names, if any. Its board is read after the edit
  and before the check. `actor_player` is the active player.
- The fork's first record, carrying the fork's flags. A check that publishes nothing writes none.

`turn_action` / `game_end` is the game-end marker:

- One per game, written as the game's last record: after the bracket's end-of-game records, the
  trigger and mana reservoirs' records, the rewrite records still waiting on a buffered ability,
  and every kept fork's records.
- Written only for a game that ended (`Game.isGameOver()`) with every end-of-game write done: the
  live record holding the outcome (`player_lost`, `player_won`), the fired trigger records, the mana
  records and the waiting rewrite records, none of which the cap withholds. A game that threw
  mid-match, a game whose end-of-game writes threw, a game whose outcome was published and reached
  no written record, a game open while the shard lost a block flush (the append is cut back and
  the block kept for the next flush), a game cut by a killed worker, and a game whose marker could
  not be written
  (its snapshot threw; counted as `game-end marker` in the worker's failure report) have none.
- Never withheld by `--max-records-per-game`.

A record counts toward what the cap bounds, and toward the worker's rewrite tally, only when it
reaches the shard: a rewrite record the cap withholds is not in the tally, and a not-fired trigger
record the cap withholds spends neither its mode's negative balance nor its line's negative cap. A
fired trigger enters its line's reservoir count when Forge runs it, never when it is evaluated.

What `--max-records-per-game` (the cap) bounds: the records of a game written as it is played, its
forks' included, so a game whose forks wrote records reaches the cap sooner and withholds the live
records past it. Past the cap the shard still takes the game's end-of-game records, each bounded
on its own:

| Passes the cap | At most, per game |
|---|---|
| fired `trigger` records | 24 per trigger line (`lineKey`: its provenance key, or reason, host, mode and parameters for a line that keys to none) |
| mana records | `--mana-cap` per mana key (the ability's text and the mana it produced) |
| rewrite records still waiting on a buffered ability's run | 1 per replacement the game ended on, none of them withheld |
| the live record holding the outcome | 1 |
| the marker | 1 |

A game therefore writes at most the cap + 24 × its fired trigger lines + `--mana-cap` × its mana
keys + its waiting rewrite records + 2 records. At the defaults (cap 5000, `--mana-cap` 1), a game whose 20 trigger lines fired
and whose lands made 10 kinds of mana writes at most 5000 + 480 + 10 + 2 = 5492.

A fork's record holding `player_lost` or `player_won` (a probe branch whose stripped keyword ended
the copy's game, a forced resolution or a played-forward fork that ended it) holds the fork's
outcome, not the game's. Past the cap it is withheld as any other record, and it never stands for
the live record holding the outcome that the marker requires.
- Payload `{"events": [], "records_withheld": n}`: `records_withheld` the records of the game the
  cap kept out of the shard (`0` for none).
- `actor_player` the player whose turn the game ended in, else the first registered player;
  `state` the board at the game's end.
- Readers treat a game whose `game_id` has no marker in its shard as cut: its records are
  incomplete (no fired-trigger or mana records, possibly a missing tail).

## Snapshot additions

Shapes follow `rules/gen3-fields.json`; the collector writes the field names below.

| Where | Field | Shape |
|---|---|---|
| entity | `owner` | player ref |
| card | `lines` | list of `{keys, kind, granted, grant?, active?, cast_with?, chosen?, fragment?, affected?}`; `kind` ∈ `spell`, `activated`, `trigger`, `static`, `replacement`, `keyword`; `active` on statics, `Class:N` keyword lines and the lines their levels grant, Room door lines, and `false` on every line of a face not up |
| card | `face_state` | `front` \| `back` \| `flipped` \| `melded` \| `face_down` |
| card | `base_pt`, `pt_boosts` | `[p, t]`; boosts `{temporary: [p, t], counters: [p, t]}`; both `null` on a non-creature; base + temporary + counters = net P/T |
| card | `counters` | `{type: n}` over every named type, in the order of the names as written (`+1/+1` before `CHARGE`) |
| card | `designations`, `x_paid`, `chosen`, `costs_paid`, `history`, `order_rank`, `real_face`, `printed_loyalty`, `printed_defense`, `mana_cost` | below, "A card's facts" |
| entity | `role` | `null` on an entity of the four tiers; else `library_card`, `exiled_with`, `mutate_component` or `last_known` (below, "Entities beyond the tiers"); `last_known` too on a referenced card the game no longer holds (below, the referenced tier) |
| entity | `combat.attacking` | changed meaning: the defender's ref, as the `attacks` link names it: `P<id>`, or `E<id>` for a planeswalker or battle; `E-1` for the placeholder an attacker attacks once its planeswalker or battle has left (506.4c, Forge's `Combat.isRemovedDefender`), which no entity carries; `null` for a creature not attacking |
| player | `this_turn.creatures_died` | changed meaning: the creatures that died under this player's control this turn, put into a graveyard from the battlefield (700.4), as morbid counts them: each graveyard's cards put there from the battlefield this turn (Forge's `Zone.getCardsAddedThisTurn`), judged as they last were on the battlefield. A token that died counts; a stolen creature counts for the player who controlled it; a bounced, blinked or exiled creature does not |
| player | `max_land_plays`, `speed`, `city_blessing`, `rad`, `monarch`, `initiative`, `turn_counts` | below, "A player's facts" |
| global | `day_night` | `"day"` \| `"night"` \| `null` |
| global | `floating_effects`, `delayed_triggers` | `{id, source_lines, affected, remembered, duration, shield_left}`; `duration` ∈ `end_of_turn`, `until_leaves`, `permanent` (below, "Command-zone objects") |
| global | `dungeons` | `{player, dungeon, room}` |
| refs | `targets` | changed meaning: every target the acting item's chain holds (the root, each sub-ability, a charm's chosen modes), root first, each ref once; a targeted spell or ability by its card's `E` ref. On an effect half, the targets of the half's own links (the root's and, on a mode's half, its mode's) as its resolution started, after Forge's check for illegal targets removed the illegal ones |
| refs | `targets_at_cast` | after `targets`: on a resolution's effect half, live or interventional, the same links' targets before that check (608.2b), spelled as `targets`, except that a card that has ceased to exist (a destroyed token) or become a new object (a flickered creature; Forge's `Card.equalsWithGameTimestamp`) is `L<card id>`, an entity of role `last_known` with no zone and no `order_rank`, and never the new object's `E<card id>`; `null` where Forge could not copy them, on a worker without the resolution hook, and on every other refs block. Each card in it is an entity of the half's board |
| refs | `targets_on_stack` | after `targets_at_cast`, on every refs block: the refs of `targets` and `targets_at_cast` that name a spell or an ability on the stack, by its host card's `E` ref (the card's entity is on the board, whatever zone it is in); `[]` when none. A ref in `targets` and not here is the card or player itself |
| refs | `random` | `{dice: [n, …], coin_flips: {won, lost}}` on a resolution's effect half, live or interventional; `null` on every other refs block |
| card | `granted_temporary.keywords` | changed order: the keywords sorted by name, as `token_created`'s `characteristics.keywords` |
| card | `granted_temporary.hidden_keywords` | after `abilities`: Forge's hidden keywords on the card as Forge writes them (`CARDNAME can't block.`, `This card doesn't untap during your next untap step.`), in timestamp order; a `continuous` record's board leaves out its static's |
| snapshot | `links` | `[{from, to, type, index?}]`; `type` a § 10.3 family name; `index` only for `targets` (below, "Links") |

A card's `lines` members:

| member | shape | present on |
|---|---|---|
| `keys` | list of 0 or 1 provenance key | every line; `[]` when the line resolves to no printed line |
| `kind` | one of the six kinds | every line |
| `granted` | bool: the line is not printed on this card (a static's, an attachment's, an effect's, an emblem's, a keyword counter's or a splice's, or a basic land type's mana ability a type change gave a land not printed with that type, such as a Mountain's under Blood Moon, keyed past every line the land's sidecar declares). A copy prints what it copies and the lines its copy effect adds; a card prints a line its own static grants it | every line |
| `grant` | why the line is granted: `static` (a permanent's static ability grants it: an anthem, an attachment, an emblem, a type change such as Blood Moon's, the suspected designation), `effect` (a resolved effect grants it: a pump, an animation, the card's own resolved effect too (a manland's animated trigger or activated ability), a copy effect another card made, an emblem's own line; it lasts as long as that effect's duration, which the line does not carry), `counter` (a keyword counter on the card, such as a trample counter) or `splice` (a spliced card's spell, for one resolution) | after `granted`, on a granted line only |
| `active` | bool: the line is in force | statics of the face up (`checkConditions()`; on a `continuous` record's board, read with every card's type line, colours and controller as on that board, the board without the record's static); every `Class:N` keyword line, `true` once the Class has reached level N (`getClassLevel()`), else `false`, and every trigger, ability or replacement that level grants (keyed to its `Class:N` keyword), `true` (a static it grants carries its own `checkConditions()`); every line of a Room's door, `true` while the door is unlocked, `false` while it is locked; `false` on every line of a face not up |
| `cast_with` | int ≥ 1: times the spell this card was cast as paid the line's cost (1 for an alternative cost or kicker, the count for multikicker) | `keyword` lines of a card on the stack or a permanent cast so |
| `chosen` | list of `Choices$` positions the ability on the stack chose, in chain order | the line of a modal ability on the stack |
| `fragment` | the keyword instance's own fragment as Forge writes it (`Ward:2`, `Annihilator:2`, `Protection:Goblin`) | `keyword` lines |
| `affected` | list of refs: the permanents the static applies to, then the players (`P…`), from Forge's static effect (`StaticEffect.getAffectedCards`, `getAffectedPlayers`) | statics of the face up that Forge applies as a continuous effect; absent on every line of a `continuous` record's board, whose answer it would be |

A `keyword` line's keys name the keyword line the card prints, else the static that granted the
keyword, a static a keyword built as that keyword's bare key (a dashed Mardu Scout's haste keys as
its Dash line); a keyword a resolved effect granted has `keys: []` and is read by its `fragment`. A copy
(a Clone's, a Phantasmal Image's) prints what it copies: its keyword lines key into the copied
card's script file, through a copy of a copy to the original; a keyword, trigger or ability the copy
effect itself adds (`AddTriggers$`, `AddAbilities$`) keys to the line that made the copy. None of
these lines is `granted`: the copy prints the script of what it copies (its latest copy effect's
origin, through a copy of a copy) and its own, and a line its copy effect adds is a copiable
exception (707.9b); one a copy effect from another card adds is `granted`, `grant: effect`. A trait
Forge builds on a copy's state (a land copy's basic land type mana ability, a copied planeswalker's
loyalty replacement) has no key. A token or spell copy is built from the copied card's own card, so
its lines key into that card's script; its keywords are numbered against the face that card prints,
not its own state, and a keyword its copy effect added (Kiki-Jiki's haste) has `keys: []` and is
`granted`, `grant: effect`. A mutated permanent's keyword keys into the script of the merged card
that prints it, and every merged card's lines are printed; a keyword two merged cards print, which
Forge keeps once, keys into the card on top, and one Forge keeps twice keys into each card in merge
order, top first. A
trait a card granted itself (a Class level's, a "has ... as long as" trigger, a manland's
animated trigger or activated ability, or a stack copy of one) keys to the line that granted it,
a Class level's to its
`Class:N` keyword's bare key, and past every declared line when none is found. Such a line is
listed beside its granting line under the same key. One a static of the card's grants (a Class
level's, a "has ... as long as" trigger) is printed, not `granted`, and so is a keyword a card's own
static grants it (Iymrith's ward); one the card's own resolved effect grants (a manland's animated
trigger or activated ability, until end of turn) is `granted`, `grant: effect`. A line another permanent's static
grants is `granted` even when it keys to the same line of the same script (a second Fungus
Sliver's trigger). The
triggers, statics and abilities a keyword instance builds are its line, not lines of their own; a
`Secondary$ True` trait is listed under its own key and joins its primary's line through the
sidecar. A land's play and a permanent's plain cast are not lines. A face not up (a face-down
card's real face, a double-faced card's other face) lists its printed lines with `active: false`;
a split card lists no face not up, except a Room. A Room lists both doors' lines: an unlocked door's
with `active: true` (a static's `checkConditions()`), a locked door's with `active: false`, keyed as
an unlocked door's are. A door is unlocked while its half is the state up: on the battlefield the
doors unlocked, in another zone both halves (709.4), on the stack the half cast; a face-down Room
lists both doors locked.

Entity members are written in this order: `id, name, zone, owner, controller, face, face_state,
copy_source, token_script_id, types, subtypes, supertypes, colors, mana_value, mana_cost, base_pt,
pt_boosts, printed_loyalty, printed_defense, tapped, sick, damage, counters, combat, attached_to,
face_down, real_face, designations, x_paid, chosen, costs_paid, history, order_rank,
granted_temporary, lines, stack_extras, role`.

A `continuous` record's board (the board without the record's static) lists no line the removed
static granted, including a trait built by a keyword instance it granted, and lists every line it
removed, keyed to its printed line. Every other entity member reads that board too: `name`,
`mana_value` and `mana_cost` (the static's layer-3 name or mana cost change left out), `types`,
`subtypes`, `supertypes`, `colors`, `controller`, `sick` (a permanent the static took control
of is summoning sick only if its controller without the static had not controlled it since their
most recent turn began), `base_pt` and `pt_boosts` (its set
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
| `untapped_production` | changed meaning: per mana symbol, what the player's untapped permanents' mana abilities that can be activated now produce, each ability read with its permanent's controller as its activator and with no X announced (X = 0), whatever an earlier activation or AI evaluation left on it; a summoning-sick creature's {T} ability and an ability whose restrictions forbid activation count for nothing. On a `continuous` record's board, the board without its static: the permanents the player controls there, their mana abilities there (none the static granted, every one it removed), each ability read with its permanent's controller there as its activator and with every card's type line, colours and controller as there, so "permanents you control" means the permanents that player controls there, metalcraft counts the artifacts by their type lines there (an artifact the static took control of still counts on the battlefield Forge moved it to, since no zone is lent) and "each colour among permanents you control" reads their colours there; a creature the static took control of is as summoning sick as it was for its controller there, a creature that came under its controller's control this turn counts nothing when its haste is the static's, and a {T} ability of a permanent that is a creature only by the static is never held back by summoning sickness |
| `city_blessing`, `monarch`, `initiative` | bool |
| `turn_counts` | `{cards_drawn, life_gained, life_lost, damage_taken, permanents_to_graveyard, spells_cast_last_turn, dungeons_completed, descended}`, ints. `permanents_to_graveyard` counts cards put into this player's graveyard from the battlefield this turn; `descended` and `dungeons_completed` are whole-game counts, as Forge keeps them |

### Command-zone objects

| list | one object per | `id` | `source_lines` | `affected` | `remembered` | `duration` | `shield_left` |
|---|---|---|---|---|---|---|---|
| `floating_effects` | effect card in a command zone (not an emblem, not a dungeon) | its entity id `E<card id>` | the key of the ability that made it; `[]` when none did (the monarch, the initiative, the city's blessing) | the cards its statics apply to, then the players; `[]` on a `continuous` record's board | its remembered cards and players | its maker's `Duration$`, bucketed; `permanent` when no ability made it | the `Number$` SVar its replacement's `Amount$` spends, else `null` |
| `floating_effects` | timestamp of a static-id-0 layer-table row on a snapshot entity (P/T set and boost, keywords, hidden keywords, can't-have keywords, traits, types, colours, names), or of a resolved control change (Forge's temporary controllers, a static's left out), drawn while a noted live clause resolved | `F<timestamp>` | the key of that clause | the entities carrying a row of that timestamp | `[]` | the clause's `Duration$`, else its root's, bucketed | `null` |
| `delayed_triggers` | pending delayed trigger, each once | `D<trigger id>` | the key of the ability that spawned it | `[]` | the trigger's remembered cards and players | `end_of_turn` for a this-turn trigger, else `permanent` | `null` |

Effect cards come first, then timestamps in ascending order. A timestamped row no noted clause
covers is not listed. A note is taken at each live clause's `ON_CLAUSE_RESOLVING` and closed at
its `ON_CLAUSE_RESOLVED`; a fork's snapshot reads the live game's notes for every timestamp at or
before the copy, and the notes of its own clauses (a played-forward fork's) after it. `duration` buckets
`Duration$`: `Permanent` is `permanent`; `UntilHostLeavesPlay`, `UntilLoseControlOfHost`,
`AsLongAs…`, `…Untaps`, `UntilUnattached` and `UntilFacedown` are `until_leaves`; everything else
is `end_of_turn`, including an absent `Duration$` and any `…OrEOT` or `…OrEndOfCombat`. A
`GainControl` clause with no `Duration$` lasts as its `LoseControl$` says: `EOT`
`end_of_turn`, an object's change (`LeavesPlay`, `LoseControl`, `Untap`, …) `until_leaves`, none
`permanent`.
`dungeons`: `player` the dungeon's owner, `dungeon` its entity ref, `room` its current room's
name. Delayed triggers need the patched `getPendingDelayedTriggers`; on an unpatched Forge the list
is `[]`.

### Entities beyond the tiers

Appended to `state.entities` after the four tiers, each card once:

| `role` | which cards | written as |
|---|---|---|
| `library_card` | each library's top card; every library card the pending event names; every library card named by the events of a resolution's effect half (live or interventional) or of a `turn_action` record | `zone: library`, `order_rank` its position when the board was captured; its owner's `controls` and `owns` links, in their families' places |
| `exiled_with` | the cards exiled with a permanent of the tiers (`getExiledCards`) | the card as it is |
| `mutate_component` | a mutated permanent's merged cards | the card as it is |
| `last_known` | the pending event's subject as it last existed (the run parameters' `CardLKI`); on an effect half, a target as cast that has ceased to exist or become a new object (`refs.targets_at_cast`), with `zone` and `order_rank` null | id `L<card id>`; `zone` the zone it was last in |

A library card an event names, and each library's top card, is rendered as it was in the library: `tapped` false, no counters,
no `attached_to`, no `combat`, `controller` its owner, face up, `entered_this_turn` false, nothing
`chosen`; its name, types and lines are its library face's. A card put onto the battlefield is the
library's own object, so these fields are not read from it after the move.

### Links

`links` sits in `state` after `entities`. Each link is `{from, to, type, index?}`, written once in
its family's forward direction; the reader takes the reverse type from the other end. `type` is a
family name of `relation_types.FAMILIES` (`FE/LinkFamily.java`, pinned by `LinkFamilyContractTest`
and `test_relation_types.py`). `index` is on `targets` only: the target's position among the
stack object's targets, root first, each ref once, as a refs block lists them (an acting item's
targets are its refs' list: § Behaviour changes readers rely on). A target two links share is
one link. Links are listed by family in table order, each once.

An end is a ref (`E<id>`, `P<id>`, `L<id>`, `F<timestamp>`, `D<id>`) or a line
`{entity, key}`: a link an ability made ends at that ability's line. Where Forge keeps no record of
the ability, or the trait keys to no line, the end is the card. An end the snapshot carries no
entity for is still named. A card that has left the game (a token that died) is no end unless the snapshot writes it
as an entity (a referenced token as it last existed): a link to it is not written, though Forge's
lists (`getGainControlTargets`, `getDamageHistory`) may still hold it.

| `type` | from | to | Forge source |
|---|---|---|---|
| `merged_with` | a mutated permanent's other card; the card a melded permanent melded with | the permanent | `getMergedCards`, `getMeldedWith` |
| `attached_to` | the attached card | the card or player | `getAttachedTo`, `getPlayerAttachedTo` |
| `attacks` | the attacker | the player, planeswalker or battle | `Combat.getDefenderByAttacker` |
| `blocks` | the blocker | each attacker it blocks | `Combat.getAttackersBlockedBy` |
| `in_band_with` | the lower card id of two attackers of one band | the other | `Combat.getAttackingBands` |
| `must_block` | the creature | each attacker it must block, by card id | `getMustBlockCards` |
| `goaded_by` | the goaded creature | each player who goaded it | `getGoaded` |
| `targets` | a stack object's line (the acting ability's too, before it is on the stack) | each target: a card or player; a spell by its card; an ability by its line | each clause's targets, each ref once |
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
| resolution, cost half | `costs` | what the cast or activation paid, as paid: `mana_by_color`, `tapped`, `life`, `sacrificed`, `discarded`, `exiled`, then `returned`, `revealed`, `exerted`, `untapped`, `convoked`, `delved`, `counters`. `life` is the sum of the life payments (two Phyrexian symbols paid with life read 4, not the last 2). `counters` is a signed net count per counter type, upper case and in type order: counters removed negative (a planeswalker's -2 is `{"LOYALTY": -2}`), counters put positive (its +1 is `{"LOYALTY": 1}`), `{}` when none moved. `returned`, `revealed`, `exerted`, `untapped` (the host of an untap symbol included), `convoked` (what a convoke cost tapped, beside `tapped`) and `delved` (what a delve exiled, beside `exiled`) are entity ids as `tapped` and `sacrificed` are. Counters and life are read from the events the payment published; the entity channels from the paid lists, the ability's convoke list and the cast card's delve list |
| resolution, effect half | `outcome` | after `conditions_met`: `resolved`, `fizzled`, `partially_fizzled` or `declined`; a `fizzled` or `declined` spell's half holds exactly one event, its card's `zone_change` off the stack to wherever Forge puts it (608.2b; the last half of a modal spell takes it); a `fizzled` or `declined` ability's half, and a copy of a spell's, which ceases to exist with no move (707.10a), has `events: []`; `partially_fizzled` when some of the targets of the item's chain, every link's counted, became illegal and some remain; `null` on a mana record and the game-end rescue record, which resolve no stack item |
| resolution | `conditions_met` | `[{segment, met}]`, after `events`, on every effect half: one entry per conditional clause that resolved in the half, in resolution order; a clause is conditional when it carries a parameter `SpellAbilityCondition.setConditions` reads (`Condition$`, `ConditionPresent$`, `ConditionCheckSVar$`, …; not `ConditionDescription$`); `segment` is spelled as the clause's events' `attributed_to` (`root`, or the chain index as a string, counted down the line the record keys to: a triggered ability's root clause is `root`, and a chosen charm mode's clauses are counted down its option line, the mode's own first clause `root`), `met` is Forge's own answer, the clause hook's `ClauseRun` read when the clause's own effect is over (`CONDITIONS_UNMET` false, `RAN` and `UNLESS_COST_DECIDED` true; no entry for a clause that threw); `[]` where no conditional clause resolved, and always on a mana ability's record and the game-end rescue record |
| decision candidate | `verdict.timing` | bool |
| decision candidate | `failed_checks` | `[{check, responsible_static}]`, after `responsible_static` in the candidate, one entry per failed check in the order listed here; `responsible_static` a key list, one key where a static decided the check (`static_prohibition`), else empty; `check` ∈ `split_second`, `casting_speed`, `whose_turn`, `phase`, `combat`, `activation_limit`, `land_play_limit`, `script_condition`, `zone`, `static_prohibition`, `cost`, `script_restriction` |
| decision candidate | `land_play` | bool, after `failed_checks`, on every candidate: `true` on a land-play candidate (`SpellAbility.isLandAbility()`), else `false` |
| decision candidate | `cost_after_adjustment.mana` | the mana Forge would charge now, after every increase and reduction (601.2f): the `Cost` raised by `CostAdjustment.adjust(Cost, sa, false)` (commander tax, `RaiseCost`), its mana part (`getManaCostFor`) reduced by `CostAdjustment.adjust(ManaCostBeingPaid, sa, payer, null, test=true, false)` (`ReduceCost`, set costs, and the delve, convoke and improvise mana the AI would pay), the host's cast-from zone set to where it is meanwhile; X stays an unpaid `{X}`; rendered as a `ManaCost` (`{1}{R}`); absent on a land play, a candidate with no costs, and where the adjustment throws. Gen-3 before T123a wrote the unadjusted `getTotalMana()` |
| decision candidate | `legal_targets` | union over every targeting clause, uncapped, counted with no target chosen on any clause (targets left on the ability narrow no cross-target rule); a clause whose target zone is the stack alone contributes the stack items it can target, each as its card's ref |
| decision candidate | `responsible_static` | the key of the static forbidding the play (its `static_prohibition` entry's), else empty |
| `attackers` | `anchor_attacker`, `legal_defenders`, `must_attack`, `tapped_by_attack?` | one record per creature the attacking player controls, these four first and `legal_attackers`, `forbidden` after them; `anchor_attacker` a card ref; `legal_defenders` the refs (`P`/`E`) of the declaration's defenders (players, planeswalkers, battles) the anchor may attack, in Forge's defender order, `[]` for a creature that may attack none; `must_attack` whether an attack requirement applies to the anchor (`AttackRequirement.hasRequirement`), whether or not it can attack; `tapped_by_attack` on a declared anchor only: whether a `GameEventCardTapped` for it arrived between the engine's first question and `GameEventAttackersDeclared`. `legal_attackers` and `forbidden` keep their gen-2 shape and are the declaration's, the same on each of its records: a creature is legal when it may attack at least one defender, and a forbidden one's `responsible_static` is the first static, in defender order, keeping it from a defender, else its own keyword line Forge reads as a keyword and no static ("CARDNAME can't attack or block.", "CARDNAME can't attack."); `responsible_own` and `responsible_unresolved` as on a blockers entry |
| `blockers` | `forbidden` entries | `{entity, responsible_static, responsible_own?, responsible_unresolved?, required_elsewhere?}`: `responsible_static` the key of the line keeping the creature out: the static keeping it from blocking the anchor (`cantBlockByStatic`), else the static keeping it from blocking at all (`cantBlockStatic`: Pacifism), else the creature's own keyword line Forge reads as a keyword and no static ("CARDNAME can't attack or block.", "CARDNAME can't block.", and "CARDNAME can't attack or block alone." / "CARDNAME can't block alone." while its controller has no other creature in play), else empty. `responsible_own: true` when that line is the creature's own (a static the creature hosts, a keyword line it prints), absent when it is another card's (Pacifism, a flying anchor's keyword line, a granting static): a consumer tells the creature's own line from a static elsewhere by this member alone. `responsible_unresolved` when a responsible line exists and has no key (an engine-built effect card's static, a keyword a spell granted): `responsible_static` is then `[]` and this member names why, in the `ability_unresolved` vocabulary; absent when a key is written or nothing is responsible. `required_elsewhere: true`, last, when the restrictions alone would let the creature block the anchor and Forge refuses the block for a requirement to block another attacker of the combat (`CombatUtil.mustBlockAnAttacker`, read through `canBlock(anchor, blocker, combat)`): a lure, provoke, "must be blocked by", and another attacker that "must be blocked if able", "by exactly one creature if able" or "by two or more creatures if able" while that is unmet (509.1c); absent otherwise, and on a creature the restrictions keep from the anchor anyway (a flying anchor, whose keyword line is then the `responsible_static`). A creature kept out by a hidden keyword (Panic's "can't block this turn", read off its `granted_temporary.hidden_keywords`) or by the rules alone (tapped) has `responsible_static: []` and neither `responsible_own` nor `responsible_unresolved` |
| `blockers` | `must_block` | after `forbidden`: the refs of the listed creatures, legal or forbidden, in candidate order, that a requirement makes block the anchor; a listed creature not in it is false. Requirements: the creature's `getMustBlockCards`, the anchor's `MustBeBlockedBy` (while no creature of that kind blocks it) and `MustBeBlockedByAll`, a lure ("All creatures able to block CARDNAME do so."), and a `MustBlock` static naming the creature (names every attacker); a requirement whose block costs something is none (509.1c); recorded whether or not the creature can obey it |
| `continuous` | contributions' `abilities`, `hidden_keywords` | after `colors`, on every contribution: `abilities` a token per layer-6 change the static makes to the entity, a granted ability's line key as `ProvenanceKey.describe` spells it (`(cardsfolder/c/cryptolith_rite.txt) face0/static[0]`; `runtime` for one that keys to no line), and `-` before the key of an ability it takes away; `hidden_keywords` the hidden keywords the static grants (`AddHiddenKeyword$`), as Forge writes them. A static whose only change is one of these is a `continuous` record. `keywords` starts with a lone `=` when the static removes every keyword (`RemoveAllAbilities$`), as `types` and `colors` mark a line set outright |
| `continuous` | contributions' `set_pt`, `controller` | `set_pt` after `pt_boost`: `[power, toughness]`, the base the static sets (layer 7a, then 7b), a half `null` where it sets only the other; `controller` last: the player ref the static gives control to (a `GainControl$` static, when the permanent's controller without it differs). Each present only when set. A static whose only change is one of these is a `continuous` record |
| `rewrite` | `incoming_unknown` | after `replaced_by`, only when true: Forge could not copy the event as it arrived (`before` null); `incoming` is then the event as it left and `outgoing` is `null`, which says nothing about an in-place edit |
| `rewrite` | `buffered`, `buffered_ran`, `buffered_events` | after `replaced_by` (and `incoming_unknown`), only on a damage replacement that set an ability aside (Forge's buffered ability: Swans of Bryn Argoll's draw, Rock Hydra's counter removal, a `Prevent$ True` rider, the sub-ability of a `ReplaceDamage` run in place such as Forcefield's exile, whose keys are then `replaced_by`'s too): `buffered` the ability's provenance keys (`[]` where it keys to no printed line); `buffered_ran` whether Forge ran it; `buffered_events` the events published during its run that this record took, each with `attributed_to` counted on the buffered ability's chain. Forge sums the damage of several replacements naming one ability into one run: the oldest record waiting on that ability takes the run's events, the others carry `buffered_ran: true` and `buffered_events: []`. A record whose ability never ran carries `buffered_ran: false` and `buffered_events: []` |
| resolution, mana record | `events` | `tapped` per permanent its cost tapped; then the other events its own payment published (a creature sacrificed for mana: `sacrificed`, `zone_change`); then `mana_produced`; then the events its own sub-abilities published (a pain land's `damage_dealt` and `life_change` to its controller), `attributed_to` counted on the mana ability's chain, except the record's own `tapped` events (`unresolved`) and its `mana_produced` (`root`), all with `duration` `instant`; a mana ability that produces no mana writes no `mana_produced` |
| `combat`, observed and probe | damage events' `source` | entity ref, after `cause`: the dealer the bus names as `cause`, else the one source in `assignment_choices` that assigned damage to the event's subject, or among several the one that assigned exactly its amount; absent when none can be named. Written on the observed record too, so a probe branch's damage reads as the record it mirrors |

## Event additions

Two new event types, `name_change` and `mana_cost_change`. Two new parameters outside a type's
`EVENT_PARAMS` row, `damage_dealt`'s `damage_type` and `keyword_change`'s `hidden`; every other fact rides a parameter the row already declares, a `characteristics`
member, or the provenance `mode` any event may carry. `state_flag_change` gains flags outside
`DESIGNATIONS`, listed below.

| Event | Field | Shape |
|---|---|---|
| `tapped` | after a `zone_change` onto the battlefield | subjects the card, written when it entered tapped, in the record that moved it |
| `tapped` | mana record | subjects each permanent the mana ability's cost tapped, before `mana_produced`. A triggered mana ability whose payment Forge reports (Wild Growth) is an activation of its own: a mana record keyed to its own line (`wild_growth.txt`), read before its payment; one whose payment Forge does not report adds its `mana_produced` to the record of the activation that set it off, while that record is being built |
| `zone_change` off the stack | last event of a spell's resolution in its effect half | the resolved spell's card to the graveyard (or wherever `finishResolving` puts it), after the resolution's own events and before the state-based deaths the half holds; a permanent spell's move onto the battlefield is made while it resolves. A forced resolution's half takes it too |
| `zone_change` into a library | `library_position` | int, after `to_zone`: where the card sits in the library as the bus publishes the move (after the add): `0` the top, `-1` the bottom of a library holding other cards, else its index from the top; a card alone in its library is `0`. Absent on every other move, where the card is not found in the library, and on a move into a library that a `library_shuffled` of that library's owner follows in the same event buffer (the shuffle takes it off). Never written for a move out of a library |
| `pt_change` | `set_pt` | `[power, toughness]`, after `toughness_delta`: the base the change set (`getNetPowerBreakdown().currentValue`, the snapshot's `base_pt`), written when the base moved and the card is a creature after the change: a layer-7b setting effect, a CDA, a face change, an animation of a noncreature. A `pt_change` is written when the base moved even if the total did not (deltas `0`) |
| `card_looked_at` | `mode` | `Scry` on a scry's look, `Surveil` on a surveil's look (bus events; a surveil keeping every card on top writes no `card_milled`, so only `mode` tells the two apart), `SearchedLibrary` on a search: `subjects` the searching player, `from_zone` `library`, no `count`; written by the `ChangeZone` clause rule when the player's `Player.getLibrarySearched()` rose across the clause, so only a search of one's own library (Forge counts no other) |
| `token_created` | `characteristics.keywords` | changed order: sorted by name, spelled as an entity's `granted_temporary.keywords` |
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
| `damage_dealt` | `damage_type` | on damage to a creature (bus `GameEventCardDamaged`), after `combat`: `normal` (marked damage), `m1m1_counters` (wither or infect, 120.3d: the same record's `counter_change` adds the -1/-1 counters and no damage is marked), `deathtouch` (a deathtouch source, whatever else it has: a deathtouch source with wither or infect reads `deathtouch`), `loyalty_loss` (Forge's value, never published). Absent on damage to a player, and on a `damage_dealt` read off a trigger's or replacement's map |
| `attached` | `attached_to` | on the bus event too: the ref (`E`, or `P` for a player) of what the Aura, Equipment or Fortification now hangs on. A move from one host to another is an `unattached` from the old host, then an `attached` to the new one |
| `unattached` | `detached_from` | on the bus event too: the ref of what it left. One the state-based check makes while a resolution's effect half is held (an Aura falling off the creature the resolution killed) is in that held half, with the half that names the permanent it left |
| `destroyed` | resolution, combat and `turn_action` records | subjects the permanent, `cause` the host of the line the record attributes it to (as every bracket event's), written just before its `zone_change` off the battlefield and filed where that move is filed (a state-based destruction joins the held half or the damage step's record its death joins). No `regenerable` (Forge's bus event does not say); a destruction a replacement or regeneration stopped writes nothing |
| `sacrificed` | resolution, combat and `turn_action` records | subjects the permanent, `cause` as on `destroyed`, just before its `zone_change` off the battlefield, filed where that move is filed. A sacrifice paid as a cost is the payment's and reaches no record |
| `face_change` | resolution, combat and `turn_action` records | subjects the permanent, `to_state` the face now up: `front`, `back`, `flipped`, `melded` or `face_down` (an entity's `face_state` values), written when a battlefield permanent's face differs from the one last read (transformed, flipped, turned face up or face down), beside the stat changes the new face makes |
| `player_lost` | the record the game's outcome is written in | subjects the player, `reason` Forge's `GameLossReason` name (`LifeReachedZero`, `Milled`, `Poisoned`, `SpellEffect`, `Conceded`, `CommanderDamage`, `OpponentWon`), one per losing player, before `player_won` when there is a winner. The record is the effect half of the `LosesGame` or `WinsGame` ability whose resolution ended the game (Door to Nothingness, a "you win the game"), which holds `player_lost` and `player_won`, then a spell's card's move off the stack; the combat record, when the game ends in a combat damage step whose record is open; for any other end of the game, the game-end record, never the bracket of a cast or activation still on the stack (a "pay life" cost paid to 0). The game-end record is an effect half with no `ability`, no `link_id` and `outcome` null, its board read at the game's end, and holds `player_lost` and `player_won` only; every other event filed with no bracket open since the last cast or resolution is counted as dropped. Each is written once per game among the live records; a fork that ends its copy's game files its own in its records as a live game does, marked by their `fork_kind`. A draw (every player lost at once) writes a `player_lost` per player and no `player_won`; a player who agreed to a draw writes none |
| `keyword_change` | `keywords`, `hidden` | a stat change's `keywords` are sorted, a second instance of a keyword counted as a keyword gained or lost; `hidden: true`, last, on a change of Forge's hidden keywords, whose `keywords` are spelled as Forge writes them (`CARDNAME can't block.`), absent on any other |
| `ability_change` | from a stat change | the abilities the card gained, other than a keyword's, as `abilities` comma-joined line keys (`ProvenanceKey.describe`, `runtime` for one that keys to no line) with `removed: false`; any it lost as `abilities: ""` with `removed: true`; not written while an `Animate` or `AnimateAll` clause resolves, whose own rule writes its grants |
| `name_change` | `name` | new type, from a stat change: subjects the permanent, `name` the name it has now; written when the name changes with no face change and no copy effect (an `Animate`'s `Name$`, a `SetName$` static) |
| `mana_cost_change` | `mana_cost`, `mana_value` | new type, from a stat change: subjects the permanent, `mana_cost` the cost now as an entity's `mana_cost` spells it (absent for no cost), `mana_value` its mana value; written when the cost changes with no face change and no copy effect (a perpetual `ManaCost$`, an `Incorporate$`) |
| `card_revealed` | `from_zone`, `subjects` | `from_zone` a zone as an entity's `zone` spells it (`library`, `hand`, …), absent when Forge names none; for a hidden search Forge shows after the move ("search your library for a land card, reveal it, put it into your hand") the zone searched, `library`, not the zone the card was put into; subjects the player who reveals, from any zone and whether or not a clause resolves: `GameAction.reveal`'s `cardOwner` (a Dig's library owner, Duress's target opponent, a reveal cost's payer), and for a clash each clashing player. This replaces the base schema's library owners during a clause. One note per reveal the rules make: a Dig's showing every player the cards a player picked from cards it revealed already (Murmurs from Beyond's opponent's pick) writes none, while a pick among cards only looked at (Commune with Nature's "you may reveal a creature card from among them") is the reveal, by the picker. A move out of a public zone (graveyard, battlefield, stack, command, a face-up card in exile) writes none, though Forge still shows the players the moved cards (Grapple with the Past's return from the graveyard); a hidden zone (library, hand, a face-down card in exile) is noted as above. `library_events.reveal` is set only by a reveal from `library`; a reveal from another zone sets no output |
| `card_made` | `to_zone` | a zone as an entity's `zone` spells it; absent when Forge names none |
| `choice_made` | `choice_kind`, `value` | read from the accessor of the clause's own API: `ChooseColor` `color`, `ChooseType` `type`, `ChooseNumber` `number`, `NameCard` `card_name`, `ChoosePlayer` `player`, `ChooseCard` and `ChooseSource` `cards` (the count chosen), `ChooseDirection` `direction`, `ChooseEvenOdd` `even_odd`, `GenericChoice` `mode`; no event when that accessor holds nothing |
| `phase_skipped` | `phase` | the phase or step a `SkipPhase` clause names (`Phase$`, else `Step$`), spelled as the snapshot's `phase` (`draw`, `combat_begin`); a script phase covering several (`Main`) joins them with commas (`main1,main2`); `?` where the clause names none Forge reads |
| `continuous_effect_created` | one per effect | one event per effect card an `Effect` clause put into a command zone (one per `EffectOwner$` player), none when it made none; `layers` the clause's `StaticAbilities$` |
| `damage_healed` | `subjects`, `amount` | one event per permanent a `HealDamage` clause removed damage from: subject the permanent, `amount` the damage removed (the damage before the clause less the damage after); none when it removed none |
| `ownership_change` | `subjects` | refs of the cards a `GainOwnership` clause hands over, read before it |
| `targets_changed` | `subjects` | refs of the spells a `ChangeTargets` clause targets, by their cards |
| `removed_from_combat` | from a clause, or from a control change's zone correction | from a clause: subjects every attacker or blocker in combat before the clause and not after it: a `RemoveFromCombat` clause's targets, a `ChangeCombatants` clause's reselected attackers, and the creature a `Regeneration`, `GainControl` or `Phases` clause regenerated, took control of or phased out. From a zone correction: one event per attacker or blocker a control change Forge applies outside a `GainControl` clause (`ExchangeControl`, `GainControlVariant`, a control-giving static starting or ending, a steal running out) moves to its new controller's battlefield, read on that battlefield's `GameEventZone` and written only if the creature is out of combat when its record is written; filed wherever the bus event lands, so a resolution's static check puts it in that resolution's half, attributed to the clause the check runs under, else to the resolution's last clause that ran. One removal is told once: while a `Regeneration`, `GainControl` or `Phases` clause runs, its clause rule tells every removal and the zone correction tells none. A steal ending at the end of combat (`LoseControl$ EndOfCombat`) is filed with no bracket open and reaches no record |
| `state_flag_change` | `flag` `city_blessing` / `dungeon_completed` | `value` true; subject the player, read on the player's command-zone notification (`GameEventZone`) as the state against the state last read: `hasBlessing()` turned true, `getCompletedDungeons().size()` rose. Filed wherever the bus event lands (a resolution's effect half, the held half across the state-based loop) |

Not collected: a battle's protector changing (`Card.setProtectingPlayer` publishes nothing), the
position a card left a library from, and a search of another player's library.

## Behaviour changes readers rely on

- Each record describes one moment; its snapshot is the board just before it (root § 8.14):
  - a spell's cost half: before the card moves to the stack, the card where it was cast from and its
    costs unpaid; its acting item's `targets` links excepted (below);
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
- A rewrite record's `replaced_by` names only an ability that ran in the event's place during the
  replacement. A damage replacement whose ability Forge buffers until every replacement of the
  damage event has applied (Swans of Bryn Argoll's draw, Rock Hydra's counter removal, a
  `Prevent$ True` rider such as Gatta and Luzzu's) writes `replaced_by: []` and names the ability in
  `buffered`. A damage replacement whose own `ReplaceDamage`, `ReplaceSplitDamage` or `ReplaceEffect`
  ability runs in the event's place and has a sub-ability (Forcefield's exile) names the ability that
  ran in `replaced_by` and the sub-ability, which Forge buffers, in `buffered`: one resolver keys a
  sub-ability to its root's line, so both lists hold the same keys, and `buffered_ran` is false when
  the sub-ability did not run. The shapes a rewrite record can carry:

  | `result` | `replaced_by` | `buffered` |
  |---|---|---|
  | `prevented`, `skipped` | `[]` | absent, or the `Prevent$ True` rider's keys (`prevented`) |
  | `replaced`, `updated`, `not_replaced`, buffering a damage ability | `[]` | the buffered ability's keys |
  | `replaced`, `updated`, `not_replaced`, an ability run in place with its sub-ability buffered | the ability's keys | the same keys |
  | any other | the ability run in place, or `[]` | absent |

  No other combination is written: a record whose `replaced_by` and `buffered` are both non-empty
  and differ, or whose `replaced_by` is non-empty beside `buffered: []`, is a contradiction
  (`validate-corpus`). Its record is held until the ability's run ends, or until the state-based loop next
  begins or the game ends when it never runs; its board is still the one read before the
  replacement ran. The events of the run are the record's `buffered_events` and reach no other
  record.
- A `trigger` record with `fired: true` is a trigger Forge ran (`onTriggerRun`): its condition held
  and the trigger went on the stack, or a static trigger's ability was played. Its board is read
  when the condition was evaluated. An evaluation whose condition held and that Forge never ran
  writes no record. Several runs of one evaluation (Panharmonicon) write one record.
- A game's fired `trigger` records and its mana records are written at its end, after its other
  records and before its marker, and are never withheld by `--max-records-per-game`.
- A `coin_flipped` event names the player who flipped as its subject. No coin flipped writes no
  event. Forge's outcome hook also reports the coins a cost flips (cumulative upkeep): one flipped
  paying an unless cost while the asking ability resolves is that resolution's event, one flipped
  paying for a cast or an activation is the payment's.
- An event a spell's or an activated ability's cost payment publishes (a tap, a sacrifice, a
  counter removed) reaches no record and is counted as dropped; what was paid is read off the
  payment itself, the cost half's `costs`. A mana ability's activation is its mana record's: the
  events its own payment publishes (a tap as its `tapped` events, a sacrifice as `sacrificed` and
  `zone_change`) and those its own sub-abilities publish (a pain land's damage to its controller).
- A bestowed Aura spell or a mutating creature spell whose target is illegal when it resolves
  resolves as a creature spell (702.103e, 702.140b): its effect half reads `resolved` and holds its
  events, the permanent's arrival included.
- An effect that ends the turn or the combat phase (`EndTurn`, `EndCombatPhase`) writes its effect
  half, live or forced: its `turn_ended` or `combat_ended` (subject: the attacking player; told at the clause's end,
  after the step it skips to has begun, and still the effect's), the exile of every card on the stack
  (its own included), and what its own state-based check does (a death there). What follows its own
  check while it still resolves is the step's it skips to, recorded as in an ordinary step: the cleanup's discard to hand size is a
  `cleanup_discard` `turn_action` record; an "until end of turn" effect ending, damage wearing off,
  an "until end of combat" effect ending after an `EndCombatPhase` effect's own check, and a death in
  the phase end's own check reach no record and are counted as dropped. `EndTurn` removes creatures
  from combat before its own check, so an "until end of combat" effect it ends is its half's.
- A refusal made between a cast and the start of its resolution does not make the effect half
  `declined`.
- `conditions_met` lists the conditional clauses of the line the effect half keys to; a clause of
  another line of the same card resolving meanwhile (an enters-the-battlefield replacement's) is
  not listed.
- An event's `duration` buckets a declared `Duration$` as a floating effect's `duration` is
  bucketed (above, "Command-zone objects"): `permanent` and `until_leaves` read `permanent`, every
  other reads `end_of_turn`; an undeclared one stays what Forge applies to the resolving clause's
  API (end of turn for `Pump`, `PumpAll`, `Animate`, `AnimateAll`, `Debuff`, `Protection`,
  `ProtectionAll`, `Effect`, `ChangeText` and `ExchangePower`, permanent for any other). Only the
  resolving clause's own `Duration$` is read: a sub-ability with none never takes its line's.
- The bus tells a card's move between zones as a discard or a mill only where Forge counted one.
  A hand-to-graveyard move is a `card_discarded`, and so is a hand-to-exile move of the card
  Forge's `Player.discard` has just listed as discarded (a discard a madness replacement sends to
  exile). A library-to-graveyard move is a `card_milled` only when Forge flagged the card milled
  (`Player.mill`) or surveilled (`Player.surveil`); a dig's rest or an Entomb writes the
  `zone_change` alone.
- A state transition (`zone_change`, `tapped`, `untapped`, `attached`, `unattached`, `phased`,
  `face_change`, `destroyed`, `sacrificed`, `regenerated`) that repeats, line for line, the last
  transition a record holds for the same subjects is written once; the same transition after
  another of those subjects is written again (a creature flickered twice, a permanent tapped twice).
- Interventional resolutions are spread over a game's turns: each game turn draws once, from the
  game's seed, whether it takes one of `--interventions-per-game`, with chance
  `left / (20 - turn + 1)` (at least the whole of it from game turn 20 on; `left` is the budget
  neither forced nor drawn yet), and the step of upkeep to end of turn it opens at; one not forced
  is carried to the next boundary with a line to force. The 20 is a constant
  (`InterventionScheduler.TYPICAL_GAME_TURNS`), not a flag.
- A line is forced at a step's first priority with the stack empty, after the step's turn-based
  action, never at the step's phase event: its record's `phase` is a step the rules gave priority
  in, a draw step's board holds the card drawn, and a declare-blockers or combat-damage step's
  board holds an attacker. It is forced only where its own restrictions let its holder play it,
  its cost aside: its timing (a sorcery-speed line in a main phase of its holder's turn), the
  steps and turns its script names (`ActivationPhases$`, `PlayerTurn$`), its script's conditions
  (`IsPresent$`), its activation limits and a static forbidding it. A spell whose mana cost is no
  cost and a special action are never forced, nor a line whose resolution reads what its cost
  paid (a sacrificed or discarded card, the mana spent, converge), which no forced resolution
  pays. Among the lines a step can force, the one forced is
  drawn at random from the fork's stream. Each printed line is forced once a game per pass,
  whoever holds it.
- A forced (interventional) resolution's effect halves are filed by the live collector's own rules,
  in a recording bound to the fork: the board read when its resolution starts, one half per chosen
  mode, the events the engine hooks report, the deaths of the state-based check after it and what
  they take off (`unattached`). What a live half leaves to another record is left out of a forced
  one too: the events of the step an effect ending the turn skips to, and the mana abilities an
  unless cost was paid with. The forced item's effect halves are the only records the fork writes,
  without `link_id`: not its cost half, nor the records of what its cast triggered and what its
  resolution caused.
- A damage-step probe's combat record holds the events a live record takes from the engine hooks
  rather than the bus: the events a clause's effect API promises (`turn_added`, `choice_made`,
  `continuous_effect_created`, …) and the outcomes an effect reports (`token_created`,
  `coin_flipped`, `dice_rolled`, `damage_prevented`, `spell_countered`, …).
- A forced resolution that began to resolve is written whatever it did, as a live one is: one that
  did nothing has empty halves (`events: []`, `outcome: resolved`); a fizzled spell's halves hold
  nothing but its card's move off the stack, in the last half, a fizzled ability's halves nothing,
  all `outcome: fizzled`, one half per chosen mode of a modal item. One that
  never began to resolve writes nothing. One whose controller turned down a "you may" while it
  resolved and that did nothing else reads `outcome: declined`, as a live one does.
- A forced spell's X is announced before its targets are drawn, so a clause dividing X among its
  targets divides the X announced.
- A forced bestowed Aura or mutating spell whose target is illegal as it resolves is `resolved`, as
  a live one is: it resolves as a creature spell.
- A phased-out permanent is an entity of the board, its `designations` holding `phased_out`.
- An entity's `copy_source` is the ref of what it is a copy of: the permanent or spell a copy was
  made from, else the card a copy effect copied (a Clone's original); `null` on any other card.
- A decision candidate's board and a cost half's show the acting ability's host, when it is off the
  battlefield and the stack, on the face the ability is on: an Adventure's spell, a split card's
  half, a modal double-faced card's back face; its `zone` and `order_rank` are the card's own.
- A record's referenced card (tier 1) is the game's own object for it: a dies trigger's host and
  the host of an ability that sacrificed it are the card in the graveyard, written once as
  `E<id>` with its zone; the last-known copy is only the `L<id>` entity. A referenced card the
  game no longer holds, a token that ceased to exist (a Food sacrificed for its own ability),
  is written as it last existed under its `E<id>`, the
  ref `refs.source`, its lines and the events name it by: role `last_known`, `zone` and
  `order_rank` null, and an end of links as any entity is (its `source_of`, `controls`, `owns`).
- `refs.x` is the X paid into the mana cost, else the ability's `SVar:X` when a parameter of its
  chain names X (`NumDmg$ X`, `NumAtt$ +X`, `cmcLEX`), else `null`; a `Count$Random` X is never
  read. It is read once: when a cast item goes on the stack, its costs paid, and when the
  resolution starts for a triggered ability or a copy. The cost half and every effect half of the
  item carry that one reading, never one taken after the resolution.
- A countered stack item writes no effect half. The countering clause's effect half holds the
  countered spell's `spell_countered` event and its card's `zone_change` off the stack.
- The acting line's `targets` links are the refs the record's refs block lists for them, in that
  order, `index` the position there:
  - an effect half: its targets as cast, the refs `refs.targets_at_cast` lists (an `L<id>` ref for
    one that ceased to exist), never those left after Forge's check for illegal targets; a modal
    half's are its own: the charm's root's and its mode's, a fizzled charm's halves too;
  - a cost half: the chain's targets at its cast event, the refs `refs.targets` lists, after its
    modes and any deferred targets are chosen (Forge chains a charm's modes, and the random seat
    draws a charm's targets, after the card moves to the stack). A target chosen then that the
    board read before the move does not hold is added to its entities, as read at the cast event;
  - any other record: the targets the acting chain holds.

  Every other stack object links the targets it holds. A targeted ability ends at its line on every
  record.
- An entity's `stack_extras` describe its item's whole chain, root first: `targets` lists every
  link's targets spelled as `refs.targets` spells them (a targeted spell or ability by its card's
  `E` ref), each once; `per_target_amounts` holds each divided link's share per target, summed
  over links; `up_to_counts.chosen` counts every link's targets, a target two links share twice,
  and is present when a link of the chain targets. On an effect half the acting item's entity
  reads them from its targets as cast, the refs `refs.targets_at_cast` lists, never from those
  left after Forge's check for illegal targets. `per_target_amounts` on such an entity holds
  every target's share as announced when the item went on the stack, the targets the check
  removed included, however many a link lost. Every other entity, and every record that is no
  effect half, reads what its item holds.
- Forge deals a chain's damage from one damage map when its last clause resolves (Arc Trail's
  `DamageResolve`). A `damage_dealt` whose subject the resolving clause does not target, but an
  earlier link of its chain does, carries that link's `attributed_to`; so does every other event
  a `DamageResolve` clause publishes about such a subject (the damaged player's `life_change`).
- An event a resolution publishes after its last clause, while no clause is resolving (the static
  check before "spell resolved": an Equip's `pt_change`), carries the last clause of the line that
  ran: its `attributed_to`, its `duration` (its own `Duration$`, else its API's default) and its
  `cause`, as an event published under that clause's pointer does, but without the `DamageResolve`
  link above: it keeps that clause's own `attributed_to`. Brass Squire's Attach
  sub-ability gives its `pt_change` `attributed_to: "0"` and `duration: permanent`, not the
  Pump root's `end_of_turn`. `attributed_to` is `unresolved` only when no clause of the line has
  run, and `duration` and `cause` are then the root's.
- A combat record's snapshot precedes every damage event of its step.
- An attackers record's snapshot is the board before the declaration, and the record is written when
  the declaration is made, at the next declaration, or at the game's end; only the first gives
  `tapped_by_attack`.
- A resolution's effect half holds every death of the state-based loop after it. A trigger record
  never holds one.
- A prevention event with amount 0 is not written.
- An event read off a trigger's or replacement's map (one carrying a provenance `mode`) is read
  by its mode's row, each row a mode Forge's `TriggerType` or `ReplacementType` declares. Its
  `delta` is signed as a bus event's: positive a gain, negative a loss.

  | mode (trigger / replacement) | event | read |
  |---|---|---|
  | `LifeGained` / `GainLife` | `life_change` | `delta` +the amount |
  | `LifeLost`, `PayLife` / `LifeReduced`, `PayLife` | `life_change` | `delta` −the amount |
  | `CounterAdded` | `counter_change` | `delta` +1: one event per counter |
  | `CounterRemoved` | `counter_change` | `delta` −1: one event per counter |
  | `CounterAddedOnce` / `AddCounter` | `counter_change` | `delta` +the counters put on (`CounterAmount`; the replacement's `CounterMap` summed over its players, per counter type as below) |
  | `CounterRemovedOnce` | `counter_change` | `delta` −`CounterAmount` |
  | `RemoveCounter` (replacement) | `counter_change` | `delta` the count the removal leaves (`Result`) less the card's count: 0 or negative |
  | `Drawn`, `Discarded`, `Milled`, `TokenCreated` / `Draw` | `card_drawn`, `card_discarded`, `card_milled`, `token_created` | `count` 1: one event per card |
  | `TokenCreatedOnce` | `token_created` | subjects the tokens made (`Cards`), `count` how many |
  | `DrawCards`, `Mill` (replacement) | `card_drawn`, `card_milled` | `count` `Number` |
  | `CopySpell` (replacement) | `spell_copied` | `count` `Amount` |
  | `DamagePreventedOnce` | `damage_prevented` | subject the player or permanent, `amount` the damage prevented |
  | `TurnFaceUp`, `Transformed` / `TurnFaceUp`, `Transform` | `face_change` | `to_state` the face the card shows, an entity's `face_state` value |
  | `Attached` / `Attached` | `attached` | subject the Aura or Equipment, `attached_to` what it hangs on |
  | `Unattached` | `unattached` | subject the Aura or Equipment, `detached_from` what it left |
  | `DamageDoneOnce` | `damage_dealt` | subject the damaged, `amount` the summed `DamageMap`, `source` when one source dealt it |
  | `DamageDealtOnce` | `damage_dealt` | subjects the damaged (the `DamageMap`'s keys), `amount` the map's amounts summed, `combat`, `source` the `DamageSource` |
  | `DamageAll` | `damage_dealt` | subjects the damaged (the `CardDamageTable`'s targets), `amount` its amounts summed, `combat`, `source` when one source dealt it all |
  | `ChangesZoneAll` | `zone_change` | subjects the batch's cards, its moves taken in the order of their origin's and then their destination's zone name; `from_zone` and `to_zone` when every card made one move |
  | `Exploited` | `sacrificed` | subject the exploited creature as it last was |
  | `DiscardedAll` | `card_discarded` | subjects the cards, `count` how many |
  | `RolledDie`, `RolledDieOnce` | `dice_rolled` | `sides`, `results` (`[{natural, final}]` for one die; `[{final}]` per die of a roll) |
  | `CounterPlayerAddedAll` | `counter_change` | subject the object, `delta` the counters put, summed over their types, `counter_type` when one type; a trigger record is one firing, never written once per type |
  | `Scry` / `Scry` | `card_looked_at` | subject the player, `count` the scry's number (`ScryNum` on the trigger, `Num` on the replacement) |
  | `Surveil` | `card_looked_at` | subject the player, no `count`: Forge puts no number in the map |
  | `Blocks` | `blockers_declared` | subject the blocker (`Blocker`) in every map Forge builds, `blocked` the attackers (`Attackers`) or, in a switch-block effect's map, its one `Attacker` |
  | `CreateToken` (replacement) | `token_created` | subject the affected player, `count` the tokens its table makes for that player (the table's `Affected` row): Forge runs the replacement once per player with one table |
  | `LoseMana` (replacement) | `mana_lost` | subject the affected player, no `mana_by_color`: the map's `Mana` is the replacement's default answer, never the mana lost |

  A mode's row (`RunParamRows`) names its event type, the keys its subject is read from (the first
  that names an entity), and every parameter it reads; a mode reads nothing its row does not name. A replacement's subject is `Affected`. The rows not in the table above give their
  event type its plain reading: `ChangesZone` / `Moved` a `zone_change` (`from_zone`, `to_zone`),
  `DamageDone` / `DamageDone`, `DealtDamage` a `damage_dealt` (`amount`, `combat`, `source`),
  `TapsForMana` / `ProduceMana` a `mana_produced` (`mana_by_color`), and the rest their type with
  the slot that type declares. Every Forge trigger and replacement mode with no row carries no
  magnitude a declared event holds (`Explore` and `Proliferate` among them) and is a
  `state_flag_change` with its `mode`, about the first entity its map names under `Affected`,
  `Card`, `Player`, `DamageTarget`, `Attacker`, `Blocker`, `Attackers`, `Attached`, `AttachSource`
  or `CardLKI`.

  An `AddCounter` replacement whose `CounterMap` names several counter types, on either side
  of the call, writes one rewrite record per counter type whose count the call changed, in
  the order of the types' names. Each record's `incoming` and `outgoing` carry that type as
  `counter_type` and its count before and after as `delta`, summed over the map's players; a
  type on one side only counts 0 on the other (a type the call removed has `outgoing` delta
  0). Each carries the call's `result` and `replaced_by`, and its snapshot's `pending_event`
  is its own `incoming`. A type the call left unchanged writes no record. A call that changed
  no type's count, a map of one type, and a call whose `before` is null (`incoming_unknown`)
  write the one record of the whole map: their summed `delta`, and `counter_type` only when
  the map names one type. This holds as well for an `AddCounter` replacement Forge runs on a
  permanent's move to the battlefield, over the move's `CounterMap`.

  A trigger record's pending `face_change` is that record's event; the same turn is written once
  more, by the bus, in the record the face turned in.
- A zone is Forge's `ZoneType` name lower-cased: `hand`, `library`, `graveyard`, `battlefield`,
  `exile`, `flashback`, `command`, `stack`, `sideboard`, `ante`, `merged` (a mutated permanent's
  merged card), `schemedeck`, `planardeck`, `attractiondeck`, `junkyard`, `contraptiondeck`,
  `subgame`, `extrahand`, `none` (no zone at all). An entity's `zone` is a zone, or null for a
  last-known card no zone holds; a `zone_change`'s `from_zone` and `to_zone` are zones or null.
  The `phase` is Forge's `PhaseType` name lower-cased, or null where the game has none yet.
- Readers decode a value outside its member's closed list as written and count it: an entity's
  `zone`, `face_state` and `role`, a line's `kind`, a link's `type`, the `phase`,
  `combat_substep`, `day_night`, a command-zone object's `duration`, a `zone_change`'s
  `from_zone` and `to_zone`, an event's `attributed_to`, and the envelope's `mode`, which every
  gen-3 record carries.
- A record's stat-change events (`pt_change`, `keyword_change`, `type_change`, `color_change`,
  `control_change`) are changes after its own snapshot: the differ re-reads the battlefield when a
  cost half, an effect half at its resolution's start, a mode's half or a damage step takes its
  snapshot. A cast started while something resolves and a trigger the state-based loop puts on the
  stack between two passes re-read nothing: the resolving item and the damage step keep their
  changes. A change Forge
  applied before such a snapshot and published after it reaches no record and is counted as
  dropped.
- `can_play = canPlay ∧ timing ∧ no static prohibition`.
- `affordable` is false for a spell whose mana cost is "no cost" (Ancestral Vision), cast other than
  through an effect that casts it without paying (118.6).
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
