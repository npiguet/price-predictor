# Contract: effect record schema

**Feature**: `023-ability-effect-model`
Authority: [`../../2026-09-05-ability-effect-model.md`](../../2026-09-05-ability-effect-model.md) § Corpus.

This schema is **fixed before stage-one collection begins**. Later stages widen the corpus — new
`kind` values become reachable, new snapshot tiers appear — but never redefine a field. That is what
lets a stage-one corpus stay trainable alongside stage-four records.

## File layout

```
output/effects/records/
  {run_id}.{worker}-{lifetime}.jsonl    one JSON record per line, append-only;
                                        one file per worker JVM lifetime
```

Readers load every `*.jsonl` in the directory and skip a trailing partial (non-newline-terminated)
final line — a JVM crash mid-write is expected, not exceptional.

## Envelope

```jsonc
{
  "record_id":      "3f2a…-uuid.4-mk3p9x2q.10237",  // {run_id}.{worker}-{lifetime}.{counter}
  "run_id":         "3f2a…-uuid",
  "timestamp":      "2026-09-06T15:31:12.152018Z",
  "game_id":        "3f2a…-uuid.4-mk3p9x2q.812",    // {run_id}.{worker}-{lifetime}.{game counter}
  "kind":           "resolution",           // resolution|rewrite|continuous|combat|trigger|playability
  "moment":         "resolution",           // resolution kind only: activation|resolution
  "subkind":        null,                   // playability kind only: decision|attackers|blockers
  "link_id":        "…",                    // joins a resolution pair; absent where no partner
  "mirror_of":      null,                   // fork records: the real record mirrored
  "variant_of":     null,                   // synthetic records: the source card name
  "mode":           "degraded",             // patched|degraded
  "interventional": false,
  "fork":           false,
  "synthetic":      false,
  "actor_player":   "P0",
  "ability":        [ /* ProvenanceKey[] */ ],
  "ability_unresolved": null,               // why `ability` is empty, where a line was sought
  "state":          { /* StateSnapshot */ },
  "payload":        { /* per-kind */ }
}
```

### Field rules

| Field | Rule |
|---|---|
| `record_id` | unique across the run's shards; carries the worker slot because workers count independently, and the JVM lifetime because each worker JVM counts from zero |
| `game_id` | same construction; **the join key for a checkpoint's recorded split** |
| `link_id` | absent on a half with no partner: a `fizzled`, `countered`, or `declined` cost record, or an interventional effect half |
| `mirror_of` | set only when `fork = true` and a same-game real counterpart exists |
| `variant_of` | set only when `synthetic = true` |
| `ability` | absent where no single line acts (`combat`, `playability`); the chosen `option` line on modal resolutions; several keys where the rendered line merged several traits |
| `ability_unresolved` | set only where `ability` is an empty array on a kind that does name a line; a closed vocabulary (`engine_effect` \| `no_card_state` \| `unknown_kind` \| `unindexable`) saying why no printed line was found. An empty `ability` alone cannot separate "the Monarch has no printed line in any tree" from "the resolver regressed", which is the ambiguity that hid a broken resolver for a whole collection run |
| `mode`, `interventional`, `fork`, `synthetic`, `ability_unresolved` | **collection metadata; never model inputs** |

### Flag signatures

| Record | `kind` | `interventional` | `fork` |
|---|---|---|---|
| ordinary observation | any | false | false |
| interventional resolution | `resolution` | true | true |
| damage-step probe | `combat` | false | true |

The probe's `interventional = false` is what distinguishes it from an intervention by flags alone.

## State snapshot

```jsonc
"state": {
  "global":  { "turn": 7, "phase": "…", "active": "P0", "priority": "P1",
               "stack_size": 1, "combat_substep": null, "emblems": [] },
  "players": [ { "id": "P0", "life": 14, "hand": 3, "library": 21, "graveyard": 9,
                 "poison": 0, "energy": 0,
                 "this_turn": { "creatures_died": 1, "spells_cast": 2, "lands_played": 1 },
                 "floating_mana": {…}, "untapped_production": {…} } ],
  "entities": [ { "id": "E12", "name": "…", "face": "…", "copy_source": null,
                  "token_script_id": null, "zone": "battlefield", "controller": "P0",
                  "types": {…}, "colors": [...], "mana_value": 3,
                  "pt": { "base": [2,2], "boosts": [1,1], "counters": [0,0] },
                  "tapped": false, "sick": false, "damage": 0, "counters": {…},
                  "combat": {…}, "attached_to": null, "face_down": false,
                  "granted_attached": [ /* ProvenanceKey[] */ ],
                  "granted_temporary": { "keywords": ["flying"],
                                         "abilities": [ /* ProvenanceKey[] */ ] },
                  "stack_extras": null } ],
  "refs":    { "targets": [...], "source": "E12", "modes": [...], "x": 3, "choices": {…} },
  "pending_event": null
}
```

**Rules**

- Characteristics are computed (post-layer), never printed. Exception: a `continuous` record's snapshot
  has the acting static's own contributions removed from every layer channel it wrote — the board is
  **recomputed with that static's layer entries dropped**, not stripped of the tokens it contributed.
  So a contribution can still appear in its own snapshot where a second static or the card's own
  printed text supplies it: an anthem granting trample to a creature printed with trample leaves
  trample in the board, which is the board that anthem acted on.
- Inclusion tiers, in order: (1) referenced objects — every entity-valued ref appears as an entity in
  whatever zone it sits, with that zone recorded; (2) core — global, battlefield, and command-zone
  effect cards; (3) unreferenced stack; (4) unreferenced hand and graveyard. Tiers 1–2 from stage one,
  3 from stage two, 4 from stage three. **An absent tier means uncollected, not empty.**
- Perspective is not stored. Controllers are absolute; mine/opponent derives at training time from
  `actor_player`.
- **The two grant sources are separate fields and must stay separate.** `granted_attached` holds
  attachment-granted abilities as provenance keys; `granted_temporary` holds what the timestamped
  change tables report, as provenance keys where a grant resolves to a printed line and as bare keyword
  strings where it does not (`gains flying until end of turn` names no line). They cannot be merged,
  because the model reads them differently: an entity's ability tokens are its printed and
  attachment-granted lines only, while temporary grants ride the overlay — and gate 2's perturbation
  removes a keyword from the ability token *or* from the temporarily-granted-keywords channel depending
  on which one carries it.

## Events

```jsonc
{ "type": "…", "subjects": ["E12"], "params": {…}, "duration": null, "attributed_to": "…" }
```

The type vocabulary is the union of Forge trigger types, bus events, and bracket diffs. The canonical
member list and per-type field normalization live in `src/effects/domain/event_schema.py`. A checked-in
completeness test maps every Forge effect API class to a covered type or an explicit exclusion.

### The vocabulary's reachability promise

Three declared types are retired rather than wired, because each names a fact a payload field already
carries under another name: `cost_adjusted` (`playability/decision` payload,
`candidates[].cost_after_adjustment`), `damage_assignment_ordered` (`combat` payload,
`assignment_choices`), and `name_change` (`continuous` payload, `contributions[].name`).
`SUPERSEDED_EVENT_TYPES` in `event_schema.py` names all three and where the information lives; wiring
them as events too would give one fact two spellings, and a corpus without them would read as incomplete
rather than correct. Removing a declared type is safe in exactly one direction — no corpus has ever
contained one, because nothing could emit it.

Every other declared type is reachable by one of two mechanisms. A **bus subscription** covers the six
types Forge already broadcasts on its own game-event bus with nothing more than a listener:
`energy_change`, `radiation_change`, `speed_changed`, `day_night_changed` from the original plan, and
`regenerated`/`library_shuffled`, added by a Task 10 fix round once a review found the engine already
broadcasting both (`GameEventCardRegenerated`, `GameEventShuffle`) into a void — see "Four declared types
found not firing, now wired" below for why those two moved here rather than staying documented gaps. `energy_change`
is the channel's canary, since Aether Hub is the cheapest sealed-legal card that exercises it. A **clause hook
plus an API-emitter table** covers the rest: `ApiEvents` in `forge-connector` wraps every `SpellAbility`
resolution, and a table keyed by the effect's `ApiType` name — not its effect class, so a Forge rename
costs a missing event rather than a compile error — says which event, if any, that API produces and how
to build it from the clause's own parameters and whatever a memo captured before the clause ran. Where
the fact an event carries exists only inside the effect's own resolution logic and not in its before/
after parameters — a coin's actual result, a vote's ballot, which cards a restriction landed on, a
reveal, a prevented amount — the effect reports it with an explicit call into the same per-clause
channel instead of through the table (`EffectRecordOutcomes.note`, called from `FlipCoinEffect`,
`ClashEffect`, `VoteEffect`, `DetainEffect`, `GoadEffect`, `MustBlockEffect`, `MakeCardEffect`,
`(Multiple)Piles/TwoPilesEffect`, `VentureEffect`, `Clone`/`CopyPermanentEffect`,
`CopySpellAbilityEffect`, and the two engine choke points `GameAction.reveal` and
`ReplacementHandler.runSingleReplaceDamageEffect`). Both branches of this mechanism together account for
22 of the 29 types that were unreachable for the life of the project before this plan; the bus
subscription accounts for the other 4, and the three retirements above for the last 3 — the full 29 are
what "previously unreachable" means throughout this contract.

**A declared type nothing emits is indistinguishable, in a corpus, from one that is merely rare — which
is exactly the state those 29 types were in, silently, before it was noticed.** Two checks stand where
that silence was, and neither is sufficient alone — and the first of them had its own version of the same
blind spot, discovered by running the second against a real collection.

- `test_no_declared_type_is_unreachable_by_surprise` (`test_event_schema_completeness.py`) statically
  scans the connector's Java source for a reference to each type's constant and fails if the declared
  vocabulary's set of unreferenced types changes unexpectedly. **It now checks against the full 78-member
  `EventType` vocabulary; until a Task 10 fix round it checked against `set(EVENT_PARAMS)` — 57 of the 78,
  every type with its own per-type params row — so the 21 types that carry no params beyond their
  subjects were never compared at all, referenced or not.** That was this plan's own founding failure
  mode ("the guard scans for references, not fires") recurring one level up, inside the guard itself: a
  check that never looks at a fifth of the vocabulary gives false assurance about exactly the thing it
  exists to assure. Widening it surfaced 15 types that were unreachable and unchecked for the entire life
  of this plan, not broken by widening it — `KNOWN_UNEMITTED` (in the test file, not empty as an earlier
  draft of this contract said) named all 15 with the producing Forge effect API and a card count for
  each, from `ability_activated` (2 cards) and `combat_ended` (1 card) up to `delayed_trigger_created`
  (757 cards, `DelayedTrigger`/`ImmediateTrigger` combined — the largest count of any type in
  `KNOWN_UNEMITTED`) and `monarch_changed` (60 cards, a whole named mechanic). Even at its full width, this scan
  **cannot** catch a type referenced from code that never runs on the path a reader would expect, in either
  of two distinct sub-shapes: see "Four declared types found not firing, now wired" below.

  Task 12 wired all eight highest-reach of those 15 — `delayed_trigger_created` and `replacement_applied`
  (an `ApiEvents.RULES` entry per API: `DelayedTrigger`/`ImmediateTrigger`, and the six `Replace*` APIs
  respectively), `monarch_changed`, `ring_tempts`, `removed_from_combat` (`ChangeCombatants`/
  `RemoveFromCombat`), `initiative_taken`, `text_change` (an `ApiEvents.RULES` entry for `ChangeText`, an
  `EffectRecordOutcomes.note` call for `ExchangeTextBox`), and `turn_ended` (`EffectRecordOutcomes.note`
  from `EndTurnEffect`, not the tempting but wrong `GameEventTurnEnded` bus event, which fires at the
  natural end of every turn and would have drowned the nine-card `EndTurn` signal in noise). Together
  these eight account for ~84% of the 15's combined card count. `replacement_applied` is a strict subset of
  the `rewrite` record this connector already writes for every replacement — see the comment on
  `EventType.REPLACEMENT_APPLIED` in `event_schema.py`, which states the relationship in the same words
  ruling R12 used for `damage_prevented`/`rewrite`.

  A Task 12 fix round found that two of the eight, `turn_ended` and `text_change`, needed to go back into
  `KNOWN_UNEMITTED` under a reason the dict had never carried before: the emitter for each is wired and
  correct — referenced right there in the connector source, unlike every other entry in the dict — but no
  AI-only corpus can ever reach it, because the AI refuses every card that carries these APIs before it
  resolves one. `forge-ai`'s `EndTurnAi.canPlay`/`.chkDrawback` both return `CantPlayAi` unconditionally
  (and for the three scripts that bury `EndTurn` in a `SubAbility$` chain, `chkDrawbackWithSubs` propagates
  that refusal to the whole enclosing spell), so none of the nine `turn_ended` cards is ever played.
  `ChangeText`/`ExchangeTextBox` have no `SpellApiToAi` entry at all and fall back to `CannotPlayAi`, which
  refuses the same way, so none of the fourteen `text_change` cards is ever played either. `KNOWN_UNEMITTED`'s
  `AI_CAPABILITY_GAP` constant names exactly these two, and the guard test that reads the dict is carved
  around it in both directions — once to stop a live reference from permanently reading as "newly wired,"
  and once to catch the reference silently disappearing, which would mean the wiring itself had been
  removed. Both go live the moment the AI gains these APIs or a human-play corpus is collected; the wiring
  is not provisional, only what plays it is.

  So six of Task 12's eight are fully live: `delayed_trigger_created`, `replacement_applied`,
  `monarch_changed`, `ring_tempts`, `removed_from_combat`, `initiative_taken`.

  Nine remain in `KNOWN_UNEMITTED`, split by two kinds. Seven have no reference at all:
  `ability_activated` (2 cards), `combat_ended` (1 card), `game_drawn` (2 cards), `game_restarted`
  (1 card), `player_removed` (2 cards), `trigger_fired` (no producing Forge effect API at all — see its
  own entry), and `turn_order_reversed` (3 cards, the largest of the seven). Two are wired and referenced
  but AI-unreachable, per `AI_CAPABILITY_GAP` above: `turn_ended` (9 cards) and `text_change` (14 cards).
- `event_type_coverage` (`validate_corpus.py`) measures, over an actually collected window, which
  declared types the corpus's records contain at all. It is watched, not judged: a short window
  legitimately misses types that are rare or depend on which decks were drawn, and a floor nobody has
  calibrated would fail every smoke run. What it must not do, and does not do, is stay silent — every
  run's report names exactly which declared types it did and did not observe, which is where the gap the
  static guard cannot see, however wide, has somewhere to show up. This is why it is a permanent
  counterpart to the static guard rather than a one-off measurement: no widening of the guard's *domain*
  closes the *reference-vs-fire* gap, because a reference is a reference regardless of what it is for.

### Four declared types found not firing, now wired

`damage_prevented` and `spell_copied` were the first two types found referenced by the static guard while
firing zero times across a 10.1M-record corpus — the reference was to *dead* code in `PatchedCollectors.java`
that never runs. A Task 10 fix-round measurement against a fresh 54,016-record run found four more types
with a reference that is real, live code, but still not an outcome emitter for the resolving ability's own
announcement — in **two distinct sub-shapes**, not one, which an earlier draft of this contract conflated:

- **Reactive-trigger mode, not the ability's own outcome.** `spell_countered` and `token_created` are
  referenced by a live *trigger* mode name — `"Countered"`, `"TokenCreated"`/`"TokenCreatedOnce"`, all real
  `TriggerType` members — reached from `PatchedCollectors.triggerRecord` (`:862-865`), which fires for a
  card that *reacts* to a counter or a token appearing elsewhere, never for the countering/token-making
  ability's own resolution.
- **Replacement mode, not a trigger at all.** `dice_rolled` and `player_won` are referenced by a live
  *replacement* mode name instead — `"RollDice"` (`ReplacementType.java:46`) and `"GameWin"`
  (`ReplacementType.java:34`) — reached from `PatchedCollectors.rewriteRecord` (`:461-467`), not
  `triggerRecord` at all, despite both mode-name spellings sitting in the same flat table (divided only by
  a comment, `PatchedCollectors.java:2832`). Firing needs a replacement effect that intercepts the roll or
  the win already in progress; there is no reactive-trigger path for these two, and no ability's-own-outcome
  path either.

Two further candidates surfaced by the same measurement, `regenerated` and `library_shuffled`, turned out
**not** to belong in this table on closer reading, in two different directions. `"Regenerated"` in the mode
table (`PatchedCollectors.java:2831`) matches no real `TriggerType` or `ReplacementType` member at all —
`grep -ic regen` returns 0 in both enums — so it is not "live code, different purpose" but a **dead
reference**, the same defect shape as `damage_prevented`/`spell_copied`. `"Shuffled"` (for
`library_shuffled`) *is* a real trigger mode, so that half of the original claim held; the type was
genuinely in this category. Both are now moot for this table regardless of which correction applied,
because both are wired directly off the bus instead (see "The vocabulary's reachability promise" above) —
Task 4's pattern, not new scope, since neither type carries an `EVENT_PARAMS` row and Forge was already
broadcasting both.

At the time of that measurement none of the remaining four had an entry in `ApiEvents.RULES` or an
`EffectRecordOutcomes.note` call, the two mechanisms this contract describes above as covering "every
other declared type". `token_created` and `spell_countered` were, and remain, among the most
commonly-scripted effect APIs in the entire card pool (`Token` alone scripted by 3,345 cards, `Counter` by
514), which is exactly why their absence from a corpus was the kind of gap a reader would not think to
check for.

**Task 11 wired all four**, closing the gap this section used to document rather than leaving it as a
to-do:

- `spell_countered` — `EffectRecordOutcomes.note` from `CounterEffect.resolve()`, fired once per spell
  `removeFromStack` actually reports removed, never for one merely targeted (uncounterable, or a target
  already off the stack). Subjects-only, the countered card as an entity ref.
- `token_created` — `EffectRecordOutcomes.note` from `TokenEffectBase.makeTokenTable(TokenCreateTable, …)`,
  the method shared by all six token-making effects (`Amass`, `Endure`, `Incubate`, `Investigate`,
  `Recruit`, `Token` itself, and `CopyPermanent`'s token copies) rather than `TokenEffect` alone — script
  id and characteristics are read off the created `Card` itself (the same convention
  `SnapshotBuilder.entityToJson` already uses: `token_script_id = card.isToken() ? card.getName() : null`),
  which is in scope at that one shared site for every caller, not "one frame away" the way it would be from
  any single effect. `count` is `allTokens.size()`, the engine's own tally of what actually entered play —
  a `CreateToken` replacement (Doubling Season) can change it from what was requested, the same
  `cards.size()` vs `madeCards.size()` shape Task 7 shipped once already.
- `dice_rolled` — `EffectRecordOutcomes.note` from `RollDiceEffect.rollDiceForPlayer()`, once `resultsList`
  is fully settled. `results` carries **both** the natural and the final value per die, paired rather than
  as two parallel lists, because a die can be modified after it is rolled (Xenosquirrel, Night Shift, the
  Vedalken Exchange swap, a uniform `Modifier$`) and a reader given only one reading could never recover
  the other.
- `player_won` — a bus subscription (`GameEventGameOutcome`), Task 4's pattern and not new hook scope,
  wired on **both** `BusBracketCollector` and `ForkEventSink` (`ForkEventSinkTest`'s reflection guard
  catches wiring only one). Emits nothing on a draw (`winningPlayerName` is null exactly when
  `GameOutcome.getWinningLobbyPlayer()` is). The event names the winner only by lobby-player display name,
  not a ref, so `BusEvents.gameOutcome` resolves it back to the real `Player` by matching
  `getLobbyPlayer().getName()` against `game.getPlayers()` — the same `damage_prevented.source` defect
  shape Task 7 shipped once, a name where every sibling subject is a ref. No `causeOf` overload: like
  `regenerated`/`library_shuffled`, Forge's own outcome names no separate cause of the win.

The taxonomy above (dead reference vs. reactive-trigger vs. replacement-mode) is unchanged and still
accurate as *why the static guard's reference could not be trusted* — none of the mode-table entries it
describes were touched, because the four types now fire through an entirely different, correct path.
`regenerated` and `library_shuffled` were not the same shape of scope as these four and moved out of this
table for a different reason (no new hook, only a bus subscription already documented above); these four
needed new hooks and call sites, which is what Task 11 added.

**Idempotent dedup for `regenerated`.** `EffectEvent.REGENERATED` is also in
`BusBracketCollector.IDEMPOTENT_EVENTS` (`:645-649`) — a set where a second byte-identical report within
one bracket is treated as the engine re-publishing the same change rather than a second occurrence. That
reference was inert while `regenerated` had no emitter and is load-bearing now. The decision is to keep it
there; an earlier draft of this paragraph justified that by treating `regenerated` as one more
subjects-only, state-transition type like its neighbours (`tapped`, `destroyed`, `sacrificed`, `phased`),
and **that comparison is wrong, not merely imprecise.**

`destroyed`/`sacrificed` are one-shot by construction — the object changes zones and stops being the same
object — and `tapped` has a real re-fire guard (`Card.tap`: `if (tapped) { return false; }`). Regeneration
has neither. `RegenerationEffect.resolve()` runs *inside* a `Destroy` replacement: it is the shield being
**consumed**, not the ability granting one. Shields are granted separately, by
`RegenerateEffect.createRegenerationEffect` (`RegenerateEffect.java:66-118`), which builds one independent
command-zone `Effect` + `ReplacementEffect` per activation and calls `c.incShieldCount()` for each — two
activations on one creature make two independent shields, each able to fire its own
`GameEventCardRegenerated`. The engine itself treats repeated regeneration as a real, distinguishable
fact: `Card.regeneratedThisTurn` (`Card.java:291`, `:3536-3543`) is an *incrementing counter*, exposed to
card scripts as `RegeneratedThisTurn` X-math (`AbilityUtils.java:2329-2330`). So this dedup **can**, in
principle, drop a genuine second occurrence rather than only a duplicate publication of the first — the
premise the other members of `IDEMPOTENT_EVENTS` actually satisfy does not hold for this one.

**The decision to keep the dedup stands anyway, for reasons that hold regardless of that fact rather than
because a repeat is impossible.** A combat bracket is one per damage step — first strike and the regular
step are different brackets, flushed at the phase boundary (`BusBracketCollector.java:onPhase`) — so the
textbook "regenerate through first strike, then again in the regular step" lands in two brackets and is
never compared. And a second destroy check driven by a separately-resolving ability changes
`attributed_to` (`EventAttribution.stamp`, called from `record`), which breaks the rendered-line equality
`fileEvent` dedups on. Nobody has constructed, or ruled out, a same-bracket, same-attribution double from
reasoning about the code alone — it would need two purely-SBA-driven destroy passes on the same card with
nothing resolving in between.

**That is an open question static reasoning cannot settle, and it is now a question a corpus run can
answer — the channel that would let anyone measure it did not exist before this task wired it.** The
method is this branch's own, turned on itself: over a real collection, group `regenerated` events by
bracket and check whether any single card's entity ref appears twice within one. A future pass with
`--probe-keywords` or a plain `collect-coverage` window large enough to gather a few hundred `regenerated`
events would settle this empirically rather than by further argument.

### `spell_countered` cannot see a counter that removed the stack instance but then failed to exile

`Counter | Destination$ Exile` (21 real cards: `dissipate`, `void_shatter`, `syncopate`,
`force_of_negation`, …) is a known limit of the channel, found in fix round 1 and not fixed: `removeFromStack`
removes the target's `SpellAbilityStackInstance` from the stack first, then attempts the `Exile` destination,
and can still return `false` if `canExiledBy` refuses (e.g. an exile-protected target) — after the stack
instance is already gone. `CounterEffect.resolve()`'s note is gated on that boolean, so this narrow case
skips `spell_countered` even though the counter genuinely happened: the spell is gone from the stack, just
not exiled where the script asked. Fixing it would mean deciding what "countered" means when the requested
destination fails, which is a real question and not this task's to answer alone; documented here so a reader
of a corpus with these 21 cards in it knows the undercount exists, rather than assuming `spell_countered` is
exhaustive.

### `card_made`, `permanent_copied` and `token_created` describe only the first object a clause makes

All three follow the same convention when one clause creates several objects at once: `card_made.card_name`
(`MakeCardEffect.java`), `permanent_copied.copy_source` (`CopyPermanentEffect.java`) and
`token_created.token_script_id`/`characteristics` (`TokenEffectBase.java`) are all read off the *first*
created object, while `count` totals every one the engine actually made. A clause that creates more than one
*distinct* kind in a single resolution — `MakeCard` naming several `Name$` values, `CopyPermanent` copying
several different permanents, or `Token` with a multi-valued `TokenScript$` (26 real cards, e.g.
`bestial_menace`: Snake + Wolf + Elephant) — reports the true total count alongside characteristics that
describe only the one sampled, not all of them.

This is schema-forced, not an oversight: `EVENT_PARAMS` gives each of these a scalar name/characteristics
field, not a per-kind breakdown, so a clause making three distinct things cannot be described exactly without
widening the schema. Kept as-is (fix round 1, M2) — a reader counting distinct token or card *kinds* from
these fields, rather than the total `count`, will undercount whenever a clause makes more than one kind.

### `attributed_to` is tri-state

| Value | Means |
|---|---|
| a chain index, e.g. `"2"` | the sub-ability at that position down the acting line's chain produced the event |
| `"root"` | the acting line's own top-level clause produced it |
| `"unresolved"` | a producing clause was sought and the pointer named nothing on this chain |
| `null` | **unknown** — what a writer that predates the sentinels left behind |

The first three are the tri-state; `null` is not one of them. One spelling covered "the root acted"
and "the pointer did not land" for a whole collection run, so 96.8% of resolution events said nothing
at all and a dead attribution channel was indistinguishable from a working one. The corpus is
append-only, so shards written before the sentinels keep their `null` and can never be disambiguated
— which is why the validator counts `absent` apart from the three rather than folding it into `root`.
Adding the two sentinels **widens the value set and does not redefine the field** (compatibility rule
1): it stays a nullable string, and a reader that predates them parses them as the strings they are.

### `zone_change` says where the card came from

`from_zone` is part of `zone_change`'s normalized params, not an optional extra. An event naming only
`to_zone` says a card arrived and not what left, and nothing then separates a graveyard recursion from
a token entering the battlefield. It is legitimately absent only where nothing left — a card *made*
rather than moved. `validate-corpus` measures the share that carries it, and the share whose `to_zone`
is `stack`, because a channel that is mostly casts is reporting the stack rather than the board.

`library_position` joins the row for the destination `from_zone`/`to_zone` cannot describe: where in
the library the card landed, counted from the top. A `Moved` replacement that puts a card second from
the top instead of into the graveyard changes only that, so without the slot the two halves of the
rewrite serialize identically — which is why `Moved` was 91 of 110 dropped rewrites in the smoke run
and had never once produced a written one. The identity drop that measured this is gone (see § A
`rewrite` says which of five results happened), but the slot is still what the change needs: a real
in-place edit whose only edited param has no slot writes an `outgoing` indistinguishable from
`incoming`, which is the unreadable shape under a different name. Absent wherever the destination is
not a library. Adding a param to a type's row **widens the row and redefines
nothing** (compatibility rule 1).

### `cause` names the object that caused the event

`cause` is a provenance param: *any* event may carry it, and it holds the entity or player ref of the
causing object, or is absent where the hook names none. Three types declare it in their own params
row — `zone_change`, `destroyed`, `sacrificed` — and those are the ones `validate-corpus` measures,
because a rate over the whole vocabulary would read near zero on a perfectly healthy corpus.

It is also what keeps two real outcomes from reading as one written twice: two attackers dealing 1
damage to the same player produce events that differ in nothing else, so the duplicate-event check
counts them as a repeat while the collector is right not to dedupe them. `damage_dealt` names the
same fact as `source`; the two spellings are deliberate and not interchangeable, and a collector
fills the one its type's row declares.

### `coin_flipped` speaks two dialects, and `FlipUntilYouLose` reports exactly one loss

`results` speaks one of two vocabularies, and `called` — a boolean in `coin_flipped`'s own `EVENT_PARAMS`
row — says which: `heads`/`tails` for a flip nobody called (`NoCall$ True`: there is no caller, so
nothing to win or lose against), `win`/`loss` for every other flip. The two are different facts about the
same event type, not a formatting inconsistency, so both are kept rather than one being normalized away.
A reader computing a win rate from `results == "win"` without filtering on `called` silently drops every
uncalled flip and has no signal that it did.

Under `FlipUntilYouLose`, `coin_flipped` reports **exactly one loss**, however many wins preceded it.
Forge computes `countLosses = |countWins - amount|` (`FlipCoinEffect.java`) as the gate that decides
whether `LoseSubAbility` runs at all, not as a count of how many flips went against the caller; the event
reports `1` in this branch specifically because reading `countLosses` as a tally would invent losses the
flip sequence never produced.

### `damage_prevented` is a second view of its `rewrite` sibling, not independent evidence

`damage_prevented.source` is an entity ref (e.g. `"E17"`), converted from the raw `Card`/`Player` the
effect reported, exactly like `damage_dealt.source` and every other producer of a `source` key. A card
name cannot be joined against `state.entities` and cannot distinguish two permanents that share a name;
the connector's outcome handler normalizes it the same way it normalizes `restriction_change.subjects`,
below, and for the same reason.

Prevention in this engine is implemented entirely as a replacement effect, so a `damage_prevented` event
and the `rewrite` record for the same replacement are two views of **one** engine event, one call frame
apart: the `rewrite` record is written from inside `executeReplacement`, before the engine computes the
prevented amount the `damage_prevented` row carries. They are not independent confirmation of the same
prevention — a reader that treats the pair as two confirming signals will double-count it.

### `restriction_change` names who was restricted, not who restricted them

`restriction_change`'s subjects (`Event.subjects`, not a `params` key) name only the cards or players the
restriction landed on — never the card or ability that applied it. `subjects` means *who the event is
about*; the actor rides in the acting ability's own provenance (the record's `ability` field), and naming
it again as a subject would double it into a list a reader counts against. It can also carry
`value: false`: the `Goad` API and its un-goad form (`NoLonger$ True`) both write a `restriction_change`
record, and only `value` distinguishes granting the restriction from lifting it. `Detain` always writes
`true`, since nothing un-detains through that API.

### `vote_taken.options` is the ballot, not the tally's keys

`options` is the script's own `VoteType` ballot, read before any vote is cast — not derived from the
tally's keys after the fact. In 1v1 play the tally is almost always size 1, since the second player tends
to vote however the outcome is already decided, so computing `options` from the tally would read the
number of choices on the ballot as 1 on nearly every real game: a different and far less useful fact than
what the card actually offered.

### Two `choice_made`-family keys are not spelled like their effect classes

`EFFECT_API_EVENTS` is keyed by `ApiType` name, and two of the `choice_made` family's keys are not
spelled like the effect class that implements them: `NameCard` (not `ChooseCardName`) and
`GenericChoice` (not `ChooseGenericEffect`). A rule keyed by the class stem instead of the `ApiType` name
matches nothing and fails silently, which is exactly the mistake this note exists to head off.

`ChooseSector` is deliberately **not** covered. It is excluded (`EXCLUDED_EFFECT_APIS`,
`event_schema.py:651`) as an Unfinity attraction not reachable in sealed or draft, the only formats this
corpus collects from — not an oversight, however much an uncovered API in this family looks like one.

### Clash's `card_revealed` is Clash's own reveal

`ClashEffect` calls `GameAction.revealTo`, not the six-argument `reveal(...)` overload this plan's engine
hook wraps, so nothing about Clash's reveal reaches the generic `card_revealed` path on its own —
`ClashEffect` reports it itself, with an explicit outcome call at the point where it already knows
exactly which cards were revealed and that they came from the library. This is a deliberate, narrow gap
and not an oversight: `revealTo` has other real callers (`ChooseCardEffect`'s `Secretly` path,
`Player.java:1166` and `:3891`) that are look-at-shaped rather than reveal-shaped, and recording one of
those as a `card_revealed` would misreport it worse than omitting it does. Clash's own use is the one
`revealTo` call this corpus can say, without qualification, is a genuine reveal — always from the
library, always to everyone.

### The two lobby seats never run simulated, and forks that do are kept out of real records

This section previously claimed collection as a whole never runs against a simulated game. That was
wrong (ruling R31) and is corrected here.

What is still true, unconditionally: the two lobby seats' own game never runs under Forge's AI full- or
hybrid-simulation modes. `AiController` only enters those modes when an `AIOption` is set, and the
connector's player setup builds both seats with a null `Set<AIOption>` — a test reads that off the actual
player instances the connector builds, not off a re-declaration of the constant, so a future edit that
starts passing an `AIOption` here fails a test rather than silently poisoning every collection channel at
once.

What is not true: that no simulated resolution ever happens during collection. Stage three's
`ForkCollector` copies the live game (`GameCopier`) two ways, both taken by default:

- **An interventional resolution** (`--interventions-per-game` defaults to 2): `ForkCollector
  .forceResolution` forces a candidate ability onto the copy's stack and hands the copy to
  `GameSimulator.resolveStack`, which builds its own `PlayerControllerAi` and unconditionally calls
  `setUseSimulation(AIOption.USE_FULL_SIMULATION)`, regardless of how the two lobby seats were
  registered. So a forced ability resolves through the exact same `AbilityUtils`/`EffectRecordOutcomes`
  machinery a real resolution does, under full simulation, every time collection takes an intervention.
- **A damage-step probe** (`--probe-keywords`, off unless named): `ForkCollector.probe` strips one
  keyword from a copy and calls `Combat.assignCombatDamage`/`dealAssignedDamage` on it directly — a
  turn-based action, not a resolution, so it never touches `GameSimulator` or an `AIOption` at all, but it
  is exactly as much a fork as the first kind.

Every effect-record listener this plan installs — the clause hook, the outcome hook, the rewrite listener,
the trigger-fire listener, the mana listener, the playability listener, the combat-legality listener and
the confirm listener, eight in total — is a plain JVM-wide static with no `Game` reference of its own, so
by construction none of them can tell a fork's resolution from the live game's just by being called. Three
mechanisms are what actually keep a fork's events out of the live game's records, and all eight listeners
above are covered by one of them:

- **Bus-derived events**: `ForkEventSink` subscribes to the *fork's own* `Game` object's event bus, not
  the live game's, so a fork's `GameEventCardDamaged` and friends never reach the live game's
  `BusBracketCollector` in the first place — there is no shared channel for them to leak through.
- **An ability-, replacement- or trigger-bearing argument**: `PatchedCollectors.clauseHandler()`,
  `outcomeHandler()` (when the reporting ability is non-null), `rewriteHandler()`, `triggerFireHandler()`,
  `manaHandler()`, `playabilityHandler()` and `combatLegalityHandler()`, and `BusBracketCollector
  .confirmHandler()`, each check the acting argument's own game identity — `belongsToLiveGame`, one
  overload keyed on `CardTraitBase.getHostCard()` (covers `SpellAbility`, `ReplacementEffect` and
  `Trigger`, which is why the rewrite and trigger-fire hooks share it with the clause/outcome hooks rather
  than needing their own), one keyed on `GameEntity` directly (the combat-legality hook's
  attacker/defender) — before ever handing an event to the bracket or the writer. `GameCopier` builds
  every copied `Card` against a new `Game` object, so a fork's card carries a different `Game` than the
  live one by construction, and each check compares the two by reference and drops anything that does not
  match. What happens when an argument cannot be identified splits along the two overloads, not evenly
  across all eight listeners (final-fix-4.md item 4 corrects this section, which previously said the
  opposite): for the six handlers keyed on `CardTraitBase.getHostCard()`, an argument of the *wrong type*
  (not a `SpellAbility`/`ReplacementEffect`/`Trigger` at all) is left alone, but a `CardTraitBase` of the
  right type with *no host card* is dropped — `belongsToLiveGame` returns `false` for a null host exactly
  as it does for a different game, and the surrounding `instanceof` pattern only lets the wrong-type case
  through uncompared. `combatLegalityHandler`'s `GameEntity` overload has no equivalent right-type-but-
  unidentifiable case at all — every real `Card`/`Player` carries a `Game` — so it fails open only on the
  wrong type. None of the eight listeners' real callers ever hand over either unidentifiable shape, so
  which direction each one degrades in costs nothing today; it is documented here because the direction
  itself, not just its unreachability, must be stated correctly.
- **A null ability, resolved another way**: `outcomeHandler()`'s two production null-ability sources
  (`GameAction.reveal`'s `card_revealed`, ruling R10; `ReplacementHandler
  .runSingleReplaceDamageEffect`'s two `damage_prevented` sites, rulings R11/R12) report with no
  `SpellAbility` at all, on the live game exactly as on a fork, so the argument-identity check above has
  nothing to compare. Simply admitting every null-ability outcome would readmit exactly what the checks
  above exist to keep out: `ForkCollector.intervene`'s forced ability can itself reveal or prevent damage
  with a null ability, and `ForkCollector.probe`'s damage step always does (`Combat`'s own cause is null
  for combat damage, live or forked). Two signals, tried in this order (reversed from final-fix-3.md's
  first pass — final-fix-4.md item 2):
  1. `ForkCollector.isForkRunningOnThisThread()` — a thread flag set around the *entirety* of both
     `intervene` and `probe`, not only `probe`'s own damage-dealing call as first shipped. The narrower
     scope left every segment of a fork outside `AbilityUtils.resolve` uncovered — `intervene`'s own
     `GameCopier`/score-check setup, `forceResolution`'s pre-push `chooseModes`/`chooseTargets`/
     `announceX`, and `GameSimulator.resolveStack`'s own `checkStateEffects`/
     `addAllTriggeredAbilitiesToStack` — latent only because no null-ability emitter happened to sit in any
     of them yet. Checked first because, widened this way, it is by itself sufficient for every fork
     mechanism this class has today.
  2. The resolving-clause pointer (`AbilityUtils.getEffectRecordSubAbility`), when the flag is false. Kept
     as a second, independent signal rather than removed — insurance against a future fork mechanism that
     resolves an ability without going through `intervene`/`probe` — not because the flag needs it: it
     names the fork's own ability during `intervene`'s forced resolution (a forced ability resolves through
     the normal resolution path, which sets it) but names nothing during `probe`'s damage step, a
     turn-based action nested inside no resolution, so it cannot by itself tell a probe's combat damage
     from the live game's own.

  A `"source"` Card carried by `damage_prevented`'s own params (raw, before `outcomeRefOf` converts it) was
  considered as a third, narrower signal and not added: with the flag now covering both fork mechanisms
  wholesale, there is no remaining gap it would close, and it could not cover `card_revealed` at all (that
  type's params carry no card).

Without these checks, a default-configuration collection run would mix a fork's hypothetical `vote_taken`
or `coin_flipped` into the real game's records, or a probe's hypothetical `damage_prevented` into it —
indistinguishably, since nothing marks a leaked event as hypothetical. Task 7's `GamePlayerTest` AIOption
guard is still correct, and stays; it is just narrower than this section used to claim: it proves the two
lobby seats never simulate, not that collection as a whole never does.

## Payloads

| Kind / moment | Payload |
|---|---|
| `resolution` / `activation` | `{costs: {mana_by_color, tapped, life, sacrificed, discarded, exiled}, outcome}` where `outcome` ∈ `resolved` \| `fizzled` \| `partially_fizzled` \| `declined` \| `countered`. Only `resolved` and `partially_fizzled` have a linked effect half |
| `resolution` / `resolution` | `{events: Event[]}`; attribution granularity is the sub-ability, and the three states an event's `attributed_to` may report are above |
| `rewrite` | `{incoming: Event, outgoing: Event\|null, result, replaced_by: ProvenanceKey[]}` (parameter maps deep-copied at the hook). `result` ∈ `replaced` \| `not_replaced` \| `prevented` \| `updated` \| `skipped`; `outgoing` is null where the event was not rewritten in place; `replaced_by` keys the ability that ran instead, empty where none. One record per `executeReplacement` call, `not_replaced` included — see below |
| `continuous` | `{contributions: [{entity, pt_boost, keywords, types, colors, name}], board_hash}` — one record per stable board |
| `combat` | `{attackers, blocks, assignment_choices, events: Event[], probed_keyword, probed_entity}` — one record per damage step; the two probe fields are set only on a damage-step probe fork and name what was stripped and from whom |
| `trigger` | `{event: Event, fired: bool}`; non-fired negatives drawn from same-event-type evaluations at ~1:1 |
| `playability` / `decision` | `{candidates: [{ability, verdict: {can_play, affordable, has_legal_target}, legal_targets, cost_after_adjustment, responsible_static}]}` |
| `playability` / `attackers` | `{legal_attackers, forbidden: [{entity, responsible_static}]}` |
| `playability` / `blockers` | `{anchor_attacker, legal_blockers, forbidden: [{entity, responsible_static}], min_blockers}` |

Verdicts are rules-level only. The AI's policy judgments ("another time", "life in danger") are never
recorded.

### A `rewrite` says which of five results happened, not just what changed

```jsonc
"payload": {
  "incoming":    { /* Event */ },            // always
  "outgoing":    { /* Event */ } | null,     // the rewritten event; null when the map was not edited
  "result":      "replaced" | "not_replaced" | "prevented" | "updated" | "skipped",
                                             // null only on a shard written before this contract
  "replaced_by": [ /* ProvenanceKey[] */ ]   // the ability that ran instead; empty when none
}
```

The original pair `{incoming, outgoing}` assumed a replacement edits the event's parameter map in
place. Forge mostly does not. `ReplacementHandler.executeReplacementInternal` runs the `ReplaceWith$`
ability — `playSpellAbilityNoStack(effectSA, true)` — and then writes only a `ReplacementResult` into
the map; for `Prevented`, `Skipped` and `NotReplaced` it returns without touching the map at all. So
"enters tapped", "if it would die, exile it instead" and "prevent that damage" are substitutions
carried out by running a *different ability*, and the before/after maps the hook deep-copied were
byte-identical. Only genuine in-place edits — `ReplaceCounterEffect.setCount`, damage amounts — ever
differed; Orim's Cure cutting 5 combat damage to 1 is one, and reads correctly under either shape.

What that cost is measurable: **34 written records in a 1.88M-record corpus**, against the trainer's
7% `--kind-mix` share, with **87% of candidates dropped as identity rewrites**. The drop was correct
under the old shape — a record whose two halves serialize alike carries no information — but the
information was never in the payload to begin with. Which of the five results happened, and which
ability ran instead, is available at the hook and was simply not passed.

Hence:

- **`outgoing` is null when nothing was rewritten in place**, not a copy of `incoming`. A pair of
  identical halves is what made this channel unreadable; null says "not rewritten" honestly, and a
  reader no longer has to compare two events to learn a negative fact.
- **Every `executeReplacement` call writes a record**, `not_replaced` included. These are the rewrite
  channel's negatives — the same role the `trigger` channel's non-fired evaluations play — and they
  are what teaches *when* a replacement applies rather than only what it does. The identity-drop rule
  is therefore gone: with `result` recorded there is no uninformative rewrite record.
- `result` is the lower_snake_case spelling of Forge's `ReplacementResult` enum
  (`{Replaced, NotReplaced, Prevented, Updated, Skipped}`), so `NotReplaced` becomes `not_replaced`.
  The vocabulary is closed by that enum.
- `replaced_by` keys `effectSA` — the substituted ability — through the same provenance resolver that
  keys the record's own `ability` field, so the two are joinable against the sidecar identically.

What each outcome may carry follows from where in `executeReplacementInternal` it returns, and that is
what the validator judges:

| Result | `outgoing` | `replaced_by` |
|---|---|---|
| `prevented`, `skipped` | **must be null** — both are bare returns *above* the `playSpellAbilityNoStack` call | **must be empty** — no ability ran |
| `replaced`, `updated` | null unless the substituted ability edited a parameter in place | the substituted ability, **but legitimately empty** where the replacement is scripted with `ReplacementResult$`, which returns the outcome having run nothing. So the naming rate has a ceiling below 1 and is watched, not judged |
| `not_replaced` | usually null, but **may be non-null**: the prevention branch writes `PreventedAmount` into the map before returning | empty |

`outgoing` is never a copy of `incoming`. A record whose two halves are byte-identical cannot be told
from a replacement that set a parameter back to its own value, and there is no such replacement — so
the identity case is a defect in the writer, judged at zero.

**`result` absent is not `not_replaced`.** A record with no `result` is one written before this
contract, where the collector had no opinion; `not_replaced` is the collector watching a replacement
decline to apply. Conflating them would silently reinterpret every pre-contract record as a negative,
which is why the reader keeps the field nullable and the validator judges only records that carry one.

**Why this is legal against a schema fixed before collection.** Compatibility rule 1 forbids changing
an existing field's meaning or type; it permits adding fields. `incoming` keeps its meaning and type
exactly. `result` and `replaced_by` are additions. `outgoing` is the one judgment call: it gains
`null` as a value, which widens a value set the way the `attributed_to` sentinels did rather than
redefining the field — it stays an Event-or-absent slot, and the reading a pre-change consumer would
have given a null (`no rewritten event here`) is the reading the null now carries explicitly, because
under the old shape that same case arrived as a copy that the collector then dropped. Corpora written
before the change stay readable: they hold only genuine in-place edits, for which `outgoing` is
non-null and `result` is absent. The change is not cosmetic and is not a repair of a bug — the old
pair could not *express* a substitution, and substitution is how Forge implements most replacements.

### A contribution's `types` and `colors`

Each is one flat list of tokens, so the operation rides on the token rather than on a field:

| Token | Means |
|---|---|
| `creature`, `W` | the static adds this type or colour |
| `-creature`, `-W` | it removes it |
| `=` | the tokens after it are a line the static **sets** rather than adds to |
| `=`, `C` | it sets the colour to none, which is what makes a permanent colourless |
| `all-creature-types` | it grants every creature type at once |
| `-all-creature-types` | it removes a whole class; likewise `-all-card-types`, `-all-super-types`, `-all-sub-types`, `-all-land-types`, `-all-artifact-types`, `-all-enchantment-types` |

Types are spelled as the snapshot spells an entity's own, and include subtypes and supertypes.
`types_gained` / `types_lost` cover the core types only, so the rest stay in the record without
reaching a head field.

A `=` reports what it sets and nothing lost. The displaced types are not in the record — the snapshot
beside it is the board after the static applied — so `types_lost` / `colors_lost` go unset, which reads
as "this record does not say" rather than as an empty set.

## Compatibility rules

### Why the id carries a lifetime

The supervisor recycles the longest-running worker every status interval and restarts crashed ones, so
an eight-hour run is hundreds of JVMs per worker slot and both counters behind an id restart at zero in
each. Stage one shipped `{run_id}.{worker}.{counter}`, which cannot satisfy this table's own uniqueness
rule under that condition: the first collected corpus reused 71–75% of its record ids and held 804
distinct `game_id` values for 31,662 games. The lifetime segment is what makes the documented rule true.

It is a change to how the id is *spelled*, not to what the field means or its type, and the number of
`.`-separated segments is unchanged — the token goes inside the middle segment, joined by `-`. So
`link_id`'s `{game_id}.link.{n}` suffix and every parse counting segments from the right are
unaffected, including the codebase's only structural parse, `EffectRecord.worker`. Corpora written
before the segment stay readable and report an empty lifetime.

### The rules

1. A field may be **added**; existing fields may not change meaning or type.
2. A new `kind` or `subkind` value may be introduced; existing values may not be repurposed.
3. Snapshot tiers are additive; absence is uncollected.
4. Nothing in this schema may become a model input that is listed above as collection metadata.
