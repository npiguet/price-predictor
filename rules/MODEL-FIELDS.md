# What the gen-2 effect model reads and predicts

Reference for the heads pass (`rules/heads/`), compiled from the code at the gen-2 input layout.
A rule is learnable only when the model **reads** what decides it and an output is **trained on**
the outcome. The catalogue's triage asked only whether a record **shows** the outcome; this file
is what the heads pass checks against instead.

Sources: `src/effects/domain/effect_head_input.py` (inputs), `effect_model.py` (outputs),
`effect_targets.py` (which record fields become targets), `state_snapshot.py`, `records.py`,
`event_schema.py`.

## Inputs: the token surface

One record becomes a sequence of slots. The trunk attends over all of them.

| slot | features |
|---|---|
| `[GLOBAL]` | turn number, stack size, phase (one-hot over untap, upkeep, draw, main1, begin_combat, declare_attackers, declare_blockers, first_strike_damage, combat_damage, end_combat, main2, end_of_turn, cleanup), combat substep (first_strike or regular), whether the acting player is active, whether the acting player holds priority, number of emblems, record kind, moment, playability subkind |
| `[ACT]` | the acting ability's `e` vector; X announced and whether there is one; number of modes, targets and choices; the resolution outcome flag (resolved, fizzled, partially_fizzled, declined, countered), which comes from the cost half and is an **input**, never a target. Empty on combat and on attackers/blockers records |
| player, one per player | life, hand size, library size, graveyard size, poison, energy; this-turn counts of creatures died, spells cast, lands played; floating mana per colour; untapped mana production per colour; whether this is the acting player; whether targeted; a thermometer of life |
| card, one per entity in the snapshot (battlefield, stack, and hands and graveyards) | card types (artifact, battle, creature, enchantment, instant, land, planeswalker, sorcery, kindred, emblem), supertypes (basic, legendary, snow, world), colours, mana value, base P/T, P/T boosts, P/T from counters; zone (battlefield, stack, hand, graveyard, library, exile, command); tapped, summoning sick, face down, damage marked; counters (P1P1, M1M1, LOYALTY, CHARGE, TIME, STUN, SHIELD, OIL, POISON, EXPERIENCE, KI, LEVEL, LORE, DEFENSE, other); attacking, blocking, became blocked, number of blockers; attached to something; keywords granted temporarily (first strike, double strike, deathtouch, lifelink, trample, indestructible, wither, infect, flying, reach, vigilance, haste, menace, hexproof, shroud, defender, protection, ward, flash, banding, fear, intimidate, horsemanship, skulk, shadow, unblockable, annihilator, landwalk); number of temporarily granted abilities; controlled by the acting player; targeted by the acting ability; is the acting ability's source; on the stack: number of targets, amounts per target, "up to" counts; combat record: is attacker, damage-assignment choices; rewrite/trigger: a pending event exists and names this entity; thermometers of power, toughness, toughness minus damage |
| ability, after each card | the `e` vector of each printed or attachment-granted line of that card: spell and activated abilities only (statics, triggers and replacements are left out, see the gen-3 snapshot section), with charm modes as extra slots |

**In the snapshot but not read by the model:** subtypes (the hook `subtype_tokens_for` exists and no
caller passes it, so "Goblin", "Aura", "Equipment" and "Saga" are invisible except through ability
text), card name, face index, copy source, token script id, controller other than "mine or not",
owner (not even recorded), the attached-to target (only a flag), who blocks whom (only flags),
the emblems' abilities (only their count).

**Not in the snapshot at all:** owner; day/night; monarch; initiative; dungeon and room;
speed; city's blessing; designations on a permanent (renowned, monstrous, suspected, saddled,
solved, ring-bearer, goaded, prepared …); soulbond pair; Class level; Room unlocked halves;
the objects exiled with a card; a linked ability's remembered choice; what happened earlier in
the turn beyond the three counts; extra turns and phases pending; who has priority beyond
"acting player or not".

## Outputs: what is trained

### Per-entity head (one output vector per card and per player slot)

Index 0 is the **gate**: whether any event of the record names the entity.

| scope | fields |
|---|---|
| permanent / stack object | `zone_outcome` (stayed, died, exiled, to_hand, library_top, library_bottom, transformed, face_changed, phased_out, blinked, countered); `tap_state`; `damage_taken`; `power_delta`, `toughness_delta`, `pt_duration` (instant, end_of_turn, permanent); `counters_delta_<type>` for each counter type above plus other; `types_gained`, `types_lost`, `colors_gained`, `colors_lost`, `type_color_duration`; `keywords_gained`, `keywords_lost` (over the keyword list above); `control_change`; `attached_to_change` |
| player | `life_delta`; `cards_drawn`, `cards_discarded`, `cards_milled`; `library_events` (scry, surveil, tutor, reveal, reorder); `poison_delta`; `mana_delta_<colour>` |
| legality (playability records) | `target_legal` per entity (decision records); `attacker_legal`, `blocker_legal` per entity; `min_blockers` at the anchored attacker |

### Verdict head, at `[ACT]`

`can_play`, `affordable`, `has_legal_target` (decision records); mana paid per colour (cost halves);
`trigger_fired` (trigger records, about half of them negatives).

### Created-objects head, at `[GLOBAL]`

Up to four groups of created tokens: present, scripted, count, power, toughness, card types,
keywords; plus an overflow flag.

### What records carry and no output is trained on

- **Resolution outcome** of a spell or ability: `fizzled`, `partially_fizzled`, `declined`,
  `resolved`. An input on the effect half; a fizzled spell writes no effect half, so nothing
  predicts that it fizzles. Only `countered` reaches an output, as the stack object's
  `zone_outcome`.
- **Event types with no target path** (they still set the gate): `regenerated`,
  `permanent_copied`, `card_made`, `player_removed` (a player losing), `damage_prevented`,
  `energy_change`, `radiation_change`, `library_shuffled`, `text_change`, `ability_change`,
  `ownership_change`, `state_flag_change`, `restriction_change`, `became_blocked`,
  `removed_from_combat`, `spell_copied`, `targets_changed`, `x_changed`,
  `delayed_trigger_created`, `replacement_applied`, `continuous_effect_created`, `phase_added`,
  `phase_skipped`, `turn_added`, `turn_skipped`, `turn_order_reversed`, `day_night_changed`.
  The damage-dealt event's **source** is not a target either: `damage_taken` says how much, not
  from what.
- **Rewrite records:** `result` (what the replacement handler reported) and `replaced_by` are not
  targets, and most rewrites carry no outgoing event, so a replacement or prevention effect is
  learned only where its outcome reaches another record's events.
- **Playability:** `cost_after_adjustment` (cost reductions and increases), `legal_targets`
  beyond the per-entity `target_legal` bit, and `responsible_static`.
- **Keywords outside the list:** a keyword gained or lost that is not one of the listed keywords
  reaches no output.
- **Types:** subtypes and supertypes gained or lost.

## Already planned in the gen-3 design record

- Input: `owner` for every entity; active player and priority holder for every player and card,
  not only relative to the acting player (turn-rule section).
- Verdict head: a `timing` column and one `failed_checks` column per rules check (split second,
  casting speed, whose turn, phase or step, combat, activation limit, script condition, zone,
  static prohibition, cost) (turn-rule section).
- Input: which attacker each blocker blocks (block-pairing section).
- Input: every line of a permanent, statics, triggers and replacements included (snapshot
  section).
- Record: the global designation block and the per-card fields (rules catalogue, part 2). These
  have **no** input slot or output planned yet; the heads pass decides which they need.
