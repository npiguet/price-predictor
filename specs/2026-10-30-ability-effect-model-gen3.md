# Ability effect model — rule-balanced training (generation 3)

Normative spec for the third generation of the ability effect model: the rules catalogue
and its scenarios, per-rule detectors and directed forks, the map of rules to the model's
inputs and outputs, a new record schema and the collection changes it carries, the
relation-aware trunk, the new inputs and outputs, per-rule curation and the rule mask,
the training loss, and checkpoint selection by probes and rule tests. Rationale, evidence
and rejected alternatives live in
[`../experiments/2026-10-30-effect-model-gen3-design.md`](../experiments/2026-10-30-effect-model-gen3-design.md)
(the gen-3 record).

This spec amends [`2026-09-05-ability-effect-model.md`](2026-09-05-ability-effect-model.md)
(the base spec) and [`2026-10-06-ability-effect-model-gen2.md`](2026-10-06-ability-effect-model-gen2.md)
(the gen-2 spec), and the record-schema contract
[`023-ability-effect-model/contracts/record-schema.md`](023-ability-effect-model/contracts/record-schema.md).
Everything it does not mention stays as those documents specify. Where they disagree, this
spec governs.

Generations

| gen | what |
|---|---|
| gen-1 | the checkpoint of run `09.17f`, trained under the base spec |
| gen-2 | the arms of the gen-2 sweep, trained under the gen-2 spec |
| gen-3 | this feature: a corpus collected from scratch under a new record schema, balanced rule by rule, and a model with a relation-aware trunk |

# 1. What it does

Gen-3 indexes the effect-record corpus by the rules of the game and balances training rule
by rule. A catalogue of the Comprehensive Rules says which rules a record can show.
Detectors count the records that show each rule, and scenarios say which situations
illustrate it. Every rule is mapped to the input that decides it and the output trained on
its outcome, and the missing inputs and outputs are added together with a new record
schema.

The model reads every line of a permanent, tied to its card, and reads each link between
two objects as a typed relation in attention. A record drawn for a rule trains only that
rule's outputs. Checkpoints are selected by knowledge probes and rule tests.

The deliverable is a gen-3 corpus, a curated dataset built from it, and a gen-3 ability
encoder with its `e` cache, for the game-playing agent. The trunk and heads train the
encoder and score it; no consumer reads them.

# 2. Scope

In scope

- The rules catalogue: units, triage, scenarios, and their review against Forge (§ 4).
- Per-rule detectors, the per-rule index in the corpus manifest, and the sources that
  illustrate a rule (§ 5).
- Directed forks of live games (§ 6).
- The rules-to-fields map and its rerun per layout (§ 7).
- Collection: the gen-3 record schema, every line of a permanent, keyword lines, the new
  snapshot fields, the combat snapshot, state-based checks, the playability verdict and its
  checks, resolution branches, new record coverage, and the dropped-event count (§ 8).
- The model input: widths, line flags, command-zone and library slots, card, player and
  global inputs, and keyword flags (§ 9).
- The relation-aware trunk and its relation types (§ 10).
- The new outputs, the extensions of existing ones, and the legality outputs (§ 11).
- Curation by rule, condition balance, protection examples and the strata (§ 12).
- Training: the rule mask, the per-field loss scale, the negative binomial, and the
  averaged weights (§ 13).
- The control score, checkpoint selection and the new probes (§ 14).

Reused unchanged

- The collectors, hooks, fork budgets and shard format of the base and gen-2 specs, except
  where § 8 changes them.
- The gen-2 encoding text, script tokenization and keyword expansion ([gen-2 spec](2026-10-06-ability-effect-model-gen2.md) § 4,
  § 5).
- The gen-2 masked-template holdout and its coverage round ([gen-2 spec](2026-10-06-ability-effect-model-gen2.md) § 6.4, § 7).
- Gen-2's lethality copies ([gen-2 spec](2026-10-06-ability-effect-model-gen2.md) § 9).
- `build-corpus`'s class budgets, water-filling allocator, reuse cap and repacking, beside
  which § 12 adds rule budgets.
- The random seat and its share of matches ([gen-2 spec](2026-10-06-ability-effect-model-gen2.md) § 6.1).
- The knowledge-probe suite and its scorecards ([gen-2 spec](2026-10-06-ability-effect-model-gen2.md) § 11.2), which § 14 extends.

Out of scope

- A controller that draws more of the records a weak output field needs during training:
  the gen-4 record.
- A thresholded or focal loss on a rule-drawn record's other outputs (§ 13.1 names the two
  mask modes only).
- Puzzle-style starting states as a source of rule records: considered once detectors and
  directed forks show which rules remain.
- A larger random-seat share, or a softer random seat: decided from the per-rule detector
  counts (§ 5.2).
- Lethality copies for combat: decided once the gen-3 combat changes (§ 8.6) are measured.
- Rule-labelled copies other than gen-2's lethality copies and the protection copies of
  § 12.3.
- Layer attribution.
- Subgames, restarting the game, and controlling another player.
- Casual mechanics and anything outside a two-player game without a sideboard:
  Attractions, stickers, tickets, companions, concession, the opening hand.
- Banding.
- Engine procedure in the middle of an action: an activation or payment in progress, the
  order of pending extra turns, effect expiry, mana emptying, event and stack order.
- Characteristics the input already shows with no change to predict, such as a colour
  derived from a mana cost.
- Which entries of Forge's remembered list are links: only an entry that outlives its
  resolution is one (§ 10.3), and the rule that tells them apart is settled in planning.
- Per-clause target legality for a spell with several targeting clauses (§ 11.4).

# 3. How it works

```
Comprehensive Rules ──► catalogue: units, triage, scenarios (§ 4)
                              │
                              ├──► detectors ──► per-rule counts on gen-2's corpus (§ 5)
                              │                         │
                              │                         ▼
                              │        directed forks and coverage rounds (§ 6, § 8.11)
                              │                         │
                              └──► rules-to-fields map ─┤──► new inputs and outputs (§ 9–§ 11)
                                       (§ 7)            ▼
                              gen-3 collection, new record schema (§ 8)
                                                        │
                                                        ▼
                              build-corpus: rule budgets, rule tags (§ 12)
                                                        │
                                                        ▼
                              training: rule mask, one loss scale, averaged weights (§ 13)
                                                        │
                                                        ▼
                              control score from probes and rule tests (§ 14)
```

# 4. Rules catalogue

The catalogue lives in `rules/`. `rules/README.md` specifies its file formats. It reads the
2026-09-25 edition of the Comprehensive Rules.

## 4.1 Units

- Each numbered rule is one line, with its text, examples and cross-references.
- A unit is one three-digit section, such as 704. In sections 701 and 702, keyword
  actions and keyword abilities, each keyword is its own unit, such as 702.2 for
  deathtouch.
- Chapters 8 and 9, multiplayer and casual variants, are out of scope without further
  reading.

## 4.2 Triage

A triage pass sorts every rule into one category. A language model makes the pass, and a
second pass checks the first. Decisions are out of scope throughout: the model predicts
what the engine does once they are made.

| category | the rule |
|---|---|
| mechanical | the engine decides an observable outcome: a zone change, damage, a resolution or fizzle, a trigger firing. Legality rechecked at resolution, as in 608.2b, counts here |
| legality | the engine decides whether an action is allowed when it is taken: casting timing, legal targets, legal blocks |
| decision | a choice a player makes, such as ordering triggers or assigning combat damage |
| definitional | a term or characteristic with no outcome of its own |
| out of scope | cannot occur in a Forge two-player game, or is not game mechanics |

A mechanical or legality rule also carries its observability, the record kinds that could
show it, and its outcome in one sentence.

| observability | the rule shows |
|---|---|
| direct | in a record field, such as a resolution's outcome or a playability verdict |
| indirect | through a detector over a record's events, such as a state-based death after a resolution |
| needs new fields | in a game, but in no record field |
| none | nowhere |

## 4.3 Scenarios

Every mechanical and legality rule that some Forge card can produce gets a scenario: a
minimal pair of boards on which changing one fact flips the rule's outcome.

- A scenario sets a board, lists every decision already taken, and gives at least two
  variants.
- Each variant has a polarity: `fires` (the rule's effect happens), `does not fire`, or
  `other` for a third branch.
- Distinct mechanisms get distinct scenarios, even where their outcomes look alike.
- Cards are named as Forge names them. A vanilla card is preferred over one with other
  abilities.
- Expected outcomes are observable only: zone moves, life, damage, counters, power and
  toughness, whether a stack object resolved, fizzled or was countered, whether a trigger
  fired, whether an action was legal.
- Each scenario is checked against the rules text and against the Forge script of every
  card it uses.
- A scenario names the record kind and observable outcome its detector looks for, and its
  contrasting variant gives the near miss the detector rejects.

## 4.4 Review against Forge

- A record whose outcome differs from its scenario's expected outcome sends the scenario
  to review. The review rereads the rule, the card scripts and the record, and corrects the
  scenario where it was wrong.
- A disagreement that survives review is a suspected Forge departure from the rules.
- A confirmed departure is recorded on the scenario. The rule's records stay out of the
  rule tests (§ 14.4) until Forge is patched on the `effect-record-hooks` branch or the
  departure is accepted.

# 5. Rule detectors and the per-rule index

## 5.1 Detectors

- A detector recognizes a rule in a record. Each scenario says what its detector looks
  for.
- Detectors range from an existing field, through a derivation over a record's events, to
  none:

  | rule | how a record shows it | detector |
  |---|---|---|
  | 702.2, deathtouch | a combat with a deathtouch creature, and the probe forks that strip it | the combat family and the probe keyword |
  | 704.5f, a creature with toughness 0 dies | a zone change during the state-based check after a resolution | derived from the effect half's events |
  | 613, layer order | the `continuous` record's result only | none |

- A detector tags each record with the rules it shows. The same tags feed the per-rule
  counts, the rule budgets (§ 12.1) and the rule mask (§ 13.1).

## 5.2 Per-rule counts

- The detectors run over gen-2's corpus before any gen-3 collection, through
  `rule-counts` (§ 15).
- Per rule, the counts separate records in which the rule's effect happens from records
  in which it does not, and random-seat games from games between two Forge AIs.
- The corpus manifest of every gen-3 build carries the same per-rule counts.

## 5.3 Sources of rule records

A rule's records come from three sources, in this order of preference:

1. the collected corpus, found by the rule's detector;
2. directed forks of live games (§ 6);
3. puzzle-style starting states, out of scope (§ 2).

Records from directed forks and from collection aimed at a rule never enter the rule
tests, wherever their game is placed (§ 14.4).

# 6. Directed forks

A directed fork waits for a live board that meets a rule's precondition and changes the
one fact that flips its outcome. The real record and the forked one form a minimal pair
from a real game.

- The scenario's base variant is the situation the detector searches for on the live
  board. Each other variant's change is the perturbation the fork applies. The decisions
  are the ones the game made.
- Each variant's change is classified into one fork operation:

  | operation |
  |---|
  | change a creature's power or toughness |
  | add or remove a keyword |
  | change a colour or type |
  | add or remove counters |
  | tap or untap a permanent |
  | change a life total |
  | replace one card with another |

- An operation is available to directed forks once it is built. The damage-step probe is
  the existing "remove a keyword" operation.
- The collector runs each enabled rule's detector on the board at every phase boundary.
  When the board meets the precondition, it forks, applies the operation and plays the
  fork on.
- Each enabled rule is named in `--directed-fork-rules`, and directed forks take their own
  per-game budget, `--directed-forks-per-game`.
- The base spec's score check against the live game before any change applies.

# 7. Rules-to-fields map

The map checks every mechanical and legality rule against the model's input layout and
output heads as the code builds them.

- A rule is learnable when an output is trained on its outcome and the model reads every
  fact that decides it.
- Each rule gets one output verdict:

  | verdict | meaning |
  |---|---|
  | trained | an existing output is trained on the outcome |
  | recorded | a record carries the outcome, and no output is trained on it |
  | missing | no record carries the outcome |
  | planned | the output is the `timing` or a failed-check column of the verdict head (§ 11.5) |

- Each rule also lists the deciding facts the model does not read, and the gen-3 fields
  that close each gap.
- `rules/heads/` holds one line per rule. `rules/MODEL-FIELDS.md` lists what the model
  reads and predicts.
- `rules/gen3-fields.json` is the list of record for gen-3's new inputs, outputs,
  extensions and fixes, and `rules/GEN3-FIELDS.md` renders it. § 9 and § 11 summarize it.
  Where its conventions disagree with this spec, this spec governs: relations are
  specified by § 10.
- The map is a pure reading of the layout and the scenarios. It reruns over each new
  layout, and each rerun reports the rules every change made learnable.

# 8. Collection

## 8.1 Record schema version

- Gen-3 collects its corpus from scratch under a new record-schema version. A field may
  change meaning or go away relative to gen-2. The base spec's frozen-schema rule holds
  within gen-3 only.
- Every gen-3 record carries the envelope field `generation`, with the value `3`. A
  record without it is gen-1 or gen-2, told apart as the gen-2 spec specifies.
  `generation` is collection metadata and never reaches the model.
- `build-corpus`, `train-effect-model`, `evaluate-effect-model` and the knowledge probes
  refuse a records set holding a gen-1 or gen-2 record. `rule-counts` reads gen-2
  records.

## 8.2 Every line of a permanent

- A permanent's card slot lists its live triggers, static abilities and replacement
  effects (`getTriggers()`, `getStaticAbilities()`, `getReplacementEffects()`) beside its
  spell and activated abilities. The same holds for the lines an attachment grants.
- The lists are Forge's live ones, so a creature that has lost its abilities lists none.
- A line a permanent gains from another permanent's static is listed too.
- Spliced lines, an emblem's lines, and lines as text-changing effects leave them are
  listed.
- Each listed static carries `active`: Forge's answer to whether its condition holds now.

## 8.3 Keyword lines

- Every keyword whose fragment Forge's keyword factory produces gets a line in its card
  slot: the instance's own fragment, with its value, such as annihilator 2 or ward {2}.
  The fragment need not be printed on the card.
- The collector keeps a trait Forge marks `Secondary$ True`. Its provenance key names the
  keyword line the card prints, so the sidecar joins it.
- A protection line is read from the instance's fragment, which names its quality. The
  keyword file's template, which leaves the quality blank, is not used.

## 8.4 New snapshot fields

The snapshot gains the facts `rules/gen3-fields.json` marks `new`:

- per player: speed, the city's blessing, rad counters, the monarch, the initiative, and
  the per-turn counts of `in.history`;
- global: day, night or neither;
- per entity: `owner`;
- per card: designations, X paid, chosen values, costs paid, history bits, order rank,
  the real face of a face-down card, printed loyalty and defense, mana cost symbols;
- floating effects and pending delayed triggers, each as a command-zone object with the
  source lines that created it, the objects it affects or remembers, its duration and
  the amount left on a counting-down shield;
- the dungeon each player is in and its current room;
- the top card of each library and every library card the record's events name;
- the cards exiled with a permanent;
- the components of a mutated permanent;
- the pending event's subject as it last existed;
- the resolving ability's die results and coin flips;
- for each static line, the permanents it affects.

The snapshot reports a creature's set base power and toughness as its base, and its
boosts apart from it.

## 8.5 Link sources

Each relation type of § 10.3 is collected from the Forge source its table row names. A
link made by an ability is recorded from that ability's line, not from its card.

## 8.6 Combat records

- A combat record's snapshot is taken at the start of the damage step, before any damage
  is dealt.
- The combat shape is read at the first damage event, and the assignment at any point of
  the step.

## 8.7 State-based checks

This section replaces the base spec's rule that state-based deaths attribute to the
bracket they follow.

- State-based actions have no record of their own. Their outcomes join the record of the
  action that left the board in that state. A combat damage step's record holds its own.
- A resolution's held effect half takes every death until Forge's state-based check is
  over: while the game's view tracker is frozen, every pass included.
- A trigger that reaches the stack while the tracker is frozen opens its own bracket
  without closing the held half. The held half is written at the first action after the
  check ends.
- The trigger's own record holds none of the check's deaths. A death filed in a held half
  stays marked as no clause's doing.
- A state-based death with no held resolution and no open damage step reaches no record,
  and the dropped-event count (§ 8.12) counts it.

## 8.8 Playability verdicts

- The AI's playability hook passes the candidate before the AI evaluates it, ahead of its
  `canPlay` filter.
- The collector's `PlayabilityVerdict` computes `can_play`, `affordable` and
  `has_legal_target` for both seats, from the rules alone:
  - mana and additional costs without the AI's cost preferences;
  - each targeting clause's candidates counted against the clause's minimum;
  - a charm judged by how many of its offered modes have reachable targets.
- The random seat filters its own plays on the same verdict.
- `legal_targets` lists every card and player the rules allow as a target of the
  candidate, whatever the AI would choose.
- Each candidate carries a fourth verdict bit, `timing`: Forge's `canCastTiming`.
- `can_play` is `canPlay` and `canCastTiming` together.
- Each candidate carries `failed_checks`, the list of checks Forge found unmet:

  | identifier | check |
  |---|---|
  | `split_second` | a split-second spell is on the stack |
  | `casting_speed` | sorcery timing, for a spell without flash or an ability with `SorcerySpeed$` or a loyalty cost |
  | `whose_turn` | the turn a restriction requires |
  | `phase` | the phase or step a restriction names |
  | `combat` | a "during combat" restriction |
  | `activation_limit` | activations per turn or per game |
  | `script_condition` | a script's own condition on the board, such as "only if you control an artifact" |
  | `zone` | the zone the card must be in |
  | `static_prohibition` | a static that forbids the play |
  | `cost` | the cost cannot be paid |
  | `script_restriction` | several of the above, where Forge evaluates them together and cannot report which failed |

  A check Forge evaluates apart from the others is reported under its own identifier.
- Each `failed_checks` entry carries `responsible_static` where a static decided it. A
  candidate whose `timing` bit is false carries a `casting_speed` entry. The candidate's own
  `responsible_static` keeps its gen-2 meaning.
- The collector evaluates both players' candidates at every priority stop, the player
  without priority included.

## 8.9 Resolution branches

The resolution payload carries `conditions_met`: for each chain segment with a
`Condition$`, whether the condition held, keyed by the segment index `attributed_to`
uses.

## 8.10 Event notes

- A prevention that prevented nothing writes no note.
- A probe fork's damage events carry their source, recovered from the payload's
  `assignment_choices`.

## 8.11 New record coverage

- Turn-based actions get a record of kind `turn_action`, its `subkind` naming the
  action: `draw`, `cleanup_discard`, `untap`, `phasing`, `day_night`, `lore_counter`,
  `attack_tap`. Its payload is the action's events.
- Attackers records are anchored to one attacker, as blockers records are. Beside the
  anchor, the payload lists the players, planeswalkers and battles the anchor may attack,
  and whether it must attack.
- A coverage round targets cards with an activated "any target" damage ability.

## 8.12 Keying, random seat and dropped events

- A charm mode cast through Flashback is keyed to a sidecar line.
- An ability a clone gains from the card it copies is keyed to that card's line.
- A variant record's key names its file under `variant-scripts/`.
- The random seat does not repeat a free activation whose resolution changes nothing.
- Under `--count-dropped-events`, the collector counts each event that arrives while no
  record is open, by event type and engine phase.
- The first gen-3 pilot runs with the dropped-event count on, and reports the number of
  `mana_produced` events whose `mana_by_color` map is empty.

## 8.13 Record-schema changes

| field | where | shape |
|---|---|---|
| `generation` | envelope | integer, `3` |
| `owner` | each snapshot entity | player ref |
| `timing` | each decision candidate's `verdict` | bool |
| `failed_checks` | each decision candidate | list of `{check, responsible_static}`, `check` an identifier of § 8.8 |
| `conditions_met` | resolution payload | list of `{segment, met}`, `segment` the chain index `attributed_to` uses |
| `active` | each listed static line | bool |
| `anchor_attacker`, `legal_defenders`, `must_attack` | `attackers` payload | entity ref; list of player or entity refs; bool |
| `must_block` | `blockers` payload, per listed creature | bool: the creature is required to block the anchor |
| `turn_action` | `kind` | with the subkinds of § 8.11 |
| the fields of § 8.4 | snapshot | as `rules/gen3-fields.json` gives each field's shape |

# 9. Model input

## 9.1 Widths

The ability encoder, `e` and the trunk are each 256 wide.

## 9.2 Lines and line flags

- What rules text describes reaches the model as lines carrying `e`, never as a
  vocabulary of effect kinds.
- Every ability slot carries line flags:

  | flag | set on |
  |---|---|
  | active | a line in force: its face is up, its Class level is reached, its Room door is unlocked, its static's condition holds |
  | granted temporarily | a temporary grant, read as a line; a bare keyword is read through its reminder text |
  | cast with | the alternative-cost or kicker line a spell was cast with, with a count for multikicker |
  | chosen | the mode a modal spell chose |

- A temporarily granted keyword is a line flagged granted temporarily, in place of a count
  and a keyword multi-hot.
- The lines of a face that is not up stay as ability slots flagged inactive. A face-down
  card carries its real face that way.

## 9.3 Command-zone and library slots

- A floating effect or a pending delayed trigger is a card slot in the command zone, read
  like a permanent. Its ability slots carry the `e` of the source lines that created it.
  Structured inputs hold only what text cannot say: its links to the objects it affects or
  remembers (§ 10.3), its duration over the three duration values, and the amount left on
  a counting-down shield.
- A dungeon is a command-zone card slot linked to its player by `in dungeon`. Its rooms
  are lines, the current room flagged active.
- The pending event's subject as it last existed gets a card slot, linked to the subject by
  `last known state of`.
- The library cards of § 8.4 get card slots, with their library position as order rank.
- The cards exiled with a permanent and the components of a mutated permanent get card
  slots.

## 9.4 New inputs

The 33 inputs of `rules/gen3-fields.json` go into these places:

| where | inputs |
|---|---|
| card slot | subtypes, face state, is a copy, is a token, pending-event details, mana cost, printed loyalty and defense, designations, X paid, chosen values, costs paid, history, order rank |
| ability slot | line flags (§ 9.2) |
| relations (§ 10) | attached to, attacks, targets, source of a stack object, soulbond pair, a battle's protector, exiled with, merged with, same name |
| player slot | speed, city's blessing, rad counters, monarch, initiative |
| global slot | day or night |
| command-zone and library card slots | floating effects, dungeons, library cards an effect touches |
| acting-ability slot | die results and coin flips |

- Amounts are scalars as raw plus log1p, with a thermometer where the model compares
  them with damage or toughness.
- Printed loyalty and defense each carry a thermometer, as toughness does.
- Subtypes are one learned embedding per subtype, summed into the card slot. The same
  embeddings serve chosen creature types.
- Face state is one-hot over front, back, flipped, melded and face down, and replaces the
  face-down flag.

## 9.5 Turn, priority and control

| input | slot |
|---|---|
| is the active player | each player |
| holds priority | each player |
| controller is the active player | each card |
| controller holds priority | each card |
| owner is the acting player | each card |
| owner differs from controller | each card |

The acting-player inputs of gen-2 stay.

## 9.6 Keyword flags

- A keyword keeps its flag where a selector in Forge's card scripts names it, or where
  Forge's engine checks it by name. Protection keeps its flag.
- A scripted keyword that no selector names and the engine never checks loses its flag;
  its line carries it. The list follows from the selectors in Forge's card scripts and the
  keyword checks in its engine code.
- A keyword with neither a flag nor a fragment expands to its reminder text, as an
  unknown keyword does.
- The keyword flags gain `split_second`.
- A keyword counter reads as a granted keyword line. Every other named counter gets a
  scalar input of its own.

# 10. Relation-aware trunk

This section replaces the gen-2 spec's rule that no other positional scheme is added.

## 10.1 Attention

- Each relation type is directed and has a learned key vector and a learned value vector.
- When slot i attends to slot j, the key of j gains the key vector of the relation from i
  to j before the score is taken, and the value of j gains its value vector before it is
  passed to i.
- Several relations on one pair add as vectors.
- The trunk's layer replaces `nn.TransformerEncoderLayer`. With both relation tables at
  zero it computes what the stock layer computes.
- Every gen-3 checkpoint is trained with this trunk.

## 10.2 Relation tables

- Each layer and each head has its own key table and value table, one vector per relation
  type, each one head wide.
- The tables start at zero.
- Under `--relation-values off`, the value tables stay at zero and are not trained.

## 10.3 Relation types

- A card's lines are tied to it by `line of`, in both directions, and the lines of one
  card are tied to each other by `line of the same card`. A charm's mode slots are tied to
  their root by `mode of`.
- A link gets a relation type of its own when the rules give it behaviour no line states,
  or when the text that gives it meaning sits at the other end. Every other link shares
  its family's type.
- A link made by an ability points from that ability's line.
- A mutated or melded permanent's cards contribute their lines through `line of`.
  `merged with` records which cards leave the battlefield with it.
- Cost history, such as devour, delve, convoke, emerge and exploit, is a count on the
  card, not a relation.
- Two slots whose names match are related by `has the same name`, and so are a slot and
  a name chosen earlier that it matches. No name embedding is learned.
- An entry of Forge's remembered list is a link only when it outlives its resolution.

| family | examples | Forge source | relation, one type per direction |
|---|---|---|---|
| a card's own lines | a card and its abilities, a charm's modes, the cards of a mutated or melded permanent | the entity each ability slot is built from, `getMergedCards`, `getMeldedWith` | line of, line of the same card, mode of, merged with |
| attachment | Aura, Equipment, Fortification, bestow, reconfigure, a Curse on a player | `getAttachedTo`, `getPlayerAttachedTo` | attached to |
| combat | attacks, blocks | the combat status | attacks, blocks |
| | banding | `Combat.getBandOfAttacker` | in band with |
| | a lure, "must block" | `getMustBlockCards` | must block |
| | goad | `getGoaded` | goaded by, from a card to a player |
| the stack | targets, a source, the object a trigger is about | each stack object's targets and source, the triggered object | targets (one type per target index), source of, triggered by |
| statics and floating effects | an anthem, "creatures can't block this turn" | the per-static affected list (§ 8.4), `getEffectSource` | affects, created by |
| delayed effects | "return it at the beginning of the next end step", Sneak Attack's sacrifice | the objects a delayed trigger holds | pending on |
| linked by a line's text, rule 607 | Oblivion Ring, Banisher Priest, champion, imprint, hideaway, lasting remembered objects | `getExiledWith`, `getImprintedCards`, `getUntilLeavesBattlefield`, `getRemembered` | linked by text, from the line |
| an exiled card acting through a permanent | cipher, haunt | `getEncodingCard`, `getHaunting` | bound to, from the exiled card to the permanent |
| control and ownership | controls, owns | `getController`, `getOwner` | controls, owns |
| | Control Magic, "for as long as" | `getGainControlTargets` | control granted by |
| copies | a Clone, a token copy, a copied spell | `getCopiedPermanent`, `getCloneOrigin` | copy of |
| pairs and protectors | soulbond, a battle's protector | `getPairedWith`, `getProtectingPlayer` | paired with, protects |
| an object chosen | "choose a creature", Stuffy Doll's chosen player, the player a gift is promised to | `getChosenCard`, `getChosenPlayer`, `getPromisedGift` | chose, from the line |
| this turn's history | Sengir Vampire, a creature that blocked or was blocked this turn, crewed or saddled this turn | `getDamageHistory`, `getBlockedThisTurn`, `getCrewedByThisTurn`, `getSaddledByThisTurn` | damaged this turn, fought this turn, crewed by |
| names | the legend rule | the names in the snapshot | has the same name |
| dungeons | the dungeon a player is in | the player's dungeon | in dungeon, from the dungeon to the player |
| pending events | the subject of a pending event as it last existed | the pending event's subject | last known state of |

## 10.4 Gen-2 link features

| gen-2 feature | relation | the feature |
|---|---|---|
| card is attacking | attacks, to the player, planeswalker or battle attacked | kept |
| card is blocking | blocks | kept |
| attacker became blocked | blocks | kept |
| number of blockers | blocks | kept |
| card is attached | attached to | replaced by the relation |
| card or player is a target of the acting ability | targets, from `[ACT]` | kept |
| card is the acting ability's source | source of | kept |
| controller is the acting player | controls, from each player slot | kept |
| stack object's number of targets, divided amounts and up-to counts | targets, from the stack object | kept |
| charm-mode flag | mode of | kept |

# 11. Outputs

## 11.1 Field types

- A new output is a binary, a count, a signed delta, a categorical or a multi-binary,
  supervised only by the records that carry it.
- Per-card and per-player outputs join the per-entity head. Outputs on the acting ability
  join the verdict head.
- A global head beside the created-objects head carries outputs that belong to no card or
  player.

## 11.2 New outputs

The 22 outputs of `rules/gen3-fields.json`:

| output | slot | shape |
|---|---|---|
| copies | stack object and card | count of copies of a spell; binary for a permanent becoming a copy |
| designations changed | card | multi-binary gained and lost over the designation list; count of Class levels gained; binary each for a door unlocked, paired, protector changed, merged |
| face after | card | categorical over the face states, where a face change names the card or it enters |
| cost adjustment | verdict head, decision records | signed delta per mana symbol kind against the printed cost |
| player state change | player | binary each for became the monarch, took the initiative, gained the city's blessing, ventured, completed a dungeon; count of speed gained |
| game lost | player | binary per player |
| combat status change | card | binary each for removed from combat, became blocked, put onto the battlefield attacking or blocking |
| subtypes changed | card | multi-binary gained and lost over a closed list the manifest records; supertypes as a second pair |
| abilities changed | card | count of non-keyword abilities gained; binary for abilities lost; binary for text changed |
| resolution outcome | verdict head, cost halves | categorical: resolved, fizzled, partially fizzled, countered, declined |
| turn structure change | global head | counts of extra turns, turns skipped, extra steps or phases, steps or phases skipped |
| day/night change | global head | categorical: became day, became night, unchanged |
| base P/T delta | card | signed delta of base power and of base toughness, against the previous base |
| effects created | verdict head | count of effect objects and delayed triggers the resolution created |
| targets changed | stack object | binary |
| replacement result | verdict head, rewrite records | categorical over Forge's `ReplacementResult` |
| attack target legal | legality, card and player | binary per defender, on an anchored attackers record |
| attack/block requirement | legality, card | binary: must attack, on anchored attackers records; must block the anchored attacker, on blockers records |
| damage prevented | card and player | count |
| regenerated | card | binary |
| rad delta | player | signed delta |
| energy delta | player | signed delta |

- A player losing is a binary per player: in a two-player game a win is the other
  player's loss, and a draw is both players losing.
- The resolution outcome is an output only, not an input on the acting-ability slot.
- The replacement result supervises `rewrite` records, which the base spec leaves
  unsupervised.

## 11.3 Extensions of existing outputs

- `zone_outcome` gains shuffled into library, to command zone, entered the battlefield
  apart from a blink, and ceased to exist. `library_bottom` is set where it applies.
  Phasing in and phasing out are distinct outcomes.
- `library_events` sets `scry` for a scry only, `surveil` for a surveil and `tutor` for a
  tutor, and gains `shuffle`.
- Each created-objects group carries a binary for being controlled by a player other than
  the actor, its subtypes, its counters per type, and a group kind for emblems, effect
  objects and cards made outside the game.
- `keywords_gained` and `keywords_lost` cover every keyword an effect in the corpus grants
  or removes, a closed list the manifest records.
- Every named counter other than a keyword counter gets a signed-delta output.

## 11.4 Legality outputs

This section replaces how the base spec derives targets from playability records.

- On a decision record, `target_legal` is supervised on every card and player in the
  record: 1 for those in the candidate's `legal_targets`, 0 for the others. It trains
  independently of the affected gate. A candidate with no targeting clause leaves it
  unsupervised.
- For a candidate with several targeting clauses, `target_legal` labels the union of the
  clauses' legal targets.
- On a blockers record, `blocker_legal` is supervised on every creature the record lists:
  1 for the legal blockers, 0 for the forbidden ones. `attacker_legal` is supervised the
  same way on an attackers record.
- On an anchored attackers record, attack target legal is supervised on every player,
  planeswalker and battle: 1 for those in `legal_defenders`, 0 for the others. The
  attack/block requirement is supervised on the anchor.
- These outputs train independently of the affected gate.
- Playability records do not supervise the affected gate.

## 11.5 Verdict head columns

The verdict head gains a column for `timing` and one for each identifier of
`failed_checks`, `script_restriction` included (§ 8.8).

# 12. Curation

## 12.1 Rule budgets

- A record's natural share is the number of copies gen-2's class, family and signature
  budgets give it.
- A rule's share is taken within each sampling class: the class budget divided by the
  number of rules tagged on the class's records (§ 5.1).
- Within a class, `build-corpus` adds copies of a rule's records beyond their natural
  share until the rule's written records reach its share, through the gen-2 water-filling
  allocator. The reuse cap and `--text-cap` bound the added copies.
- Added copies come on top of the class budget.
- A record tagged with several rules counts toward each. Each added copy is drawn for the
  rule furthest below its share. A record with no tag keeps its natural share.
- An added copy carries the rule it was drawn for. A copy within the natural share carries
  none.
- The manifest records, per class and rule, the records available, the share, the records
  written and the shortfall.

## 12.2 Condition balance

- For a record whose acting line, candidate or listed static carries a turn or step
  condition, the outcome signature gains whether the condition held, read from
  `failed_checks`, `conditions_met` or the static's `active`.
- Met and unmet then draw equal budgets, and a short side repeats up to the reuse cap.
- The manifest names every condition that stays one-sided.
- Interventional forks resolve a text whose branch is seen on one player's turn only at a
  phase boundary of the other player's turn, under the existing fork rules and budgets.

## 12.3 Protection examples

Rare protection qualities, from a creature type, a card type, a condition or everything,
get examples from three sources, in amounts per-rule rebalancing sets:

- Rule-labelled copies: a copy of a real record grants a creature a protection quality and
  relabels the outcomes by rule 702.16: damage prevented, a block or target made illegal,
  an Aura put into the graveyard.
- Variant scripts: `collect-variants` swaps a protection keyword's quality as a
  whitelisted selector.
- Interventional forks: a fork grants protection mid-game and plays the turn on.

## 12.4 Strata and refusals

- The validation strata keep games between two Forge AIs.
- A permanent spell's resolution record survives `build-corpus`, keyed to the permanent's
  card.
- A decision record whose candidate's key is unjoinable is refused as `unjoinable-key`.

# 13. Training

## 13.1 Rule mask

For a record, n is its natural share and d the number of its copies written, natural and
added.

- Under `--rule-mask hard`, an added copy trains the outputs the rules-to-fields map names
  for the rule it was drawn for, and the affected gate on the rule's own entities where the
  record supervises the gate. Every other output is masked. A copy within the natural share
  trains every output.
- Under `--rule-mask soft`, every copy trains every output. Outputs other than those of a
  rule an added copy was drawn for weigh n / d on each copy.
- Under both, the outputs of every rule an added copy of the record was drawn for weigh
  n / d on each copy of that record.
- A gen-2 lethality copy trains every output at full weight.

## 13.2 One loss scale

- Each output field's loss is divided by the loss of its constant prediction on the same
  records, with that denominator clamped below at `--deviance-floor`.
- The constant prediction is computed under the same distribution as the field's own
  loss.

## 13.3 Counts and amounts

Count and amount fields are scored by a negative binomial likelihood, in place of the base
spec's Poisson. Gate 1's deviance margin reads the negative binomial deviance.

## 13.4 Averaged weights

- Training keeps an exponential moving average of the weights, spanning `--ema-epochs`
  epochs: each step moves the average toward the current weights by
  1 / (`--steps-per-epoch` × `--ema-epochs`).
- The average starts from the weights at the end of the learning-rate warmup.
- Validation, stopping and the saved checkpoints use the averaged weights. An epoch that
  ends before the warmup does is validated on the current weights, and is neither a best
  candidate nor counted by `--patience`.
- `resume.pt` carries the averaged weights and the step they started from.
- A resume refuses a change to `--rule-mask`, `--deviance-floor`, `--ema-epochs`,
  `--stratum-weight` or `--relation-values`.

# 14. Checkpoint selection and evaluation

This section replaces the base spec's best checkpoint by card-disjoint validation loss,
and the gen-2 spec's patience and `resume-best.pt` rules, which read that loss.

## 14.1 Control score

- Each validation stratum has a stratum score, computed on the averaged weights each
  epoch from forward passes only, over that stratum's records. Its components:
  - the knowledge probes read at the model's own prediction, rung 3 of the gen-2 ladder:
    each probe type one score, averaged into one component;
  - the damage grid, the combat grid and the evasion grid, each scored against the rule's
    outcome in every cell, averaged into one component;
  - the rule tests (§ 14.4), each rule counting equally;
  - the mean of every output field's explained deviance, each field counting equally. A
    field whose constant-prediction loss is at or below `--deviance-floor` is left out of
    the mean.
- Each score is rescaled so that 0 is no better than its baseline and 1 is perfect, then
  clipped below at −1:
  - an AUC as 2 × AUC − 1;
  - an R² and an explained deviance as they are;
  - a grid as one minus the ratio of its error to the error of predicting the grid's mean
    outcome in every cell, its error being the mean absolute difference between the
    predicted probability and the rule's outcome over its cells.
- The stratum score is the mean of the four rescaled components, each counting equally.
- The control score is the card-disjoint stratum score weighted by `--stratum-weight`,
  plus the game-disjoint stratum score weighted by one minus it.
- A run keeps one best checkpoint, the epoch with the best control score. Encoder, trunk
  and heads are saved together from that epoch.
- `--patience` counts epochs without a new best control score. A new best copies
  `resume.pt` to `resume-best.pt`.
- Calibration stays a reported number.

## 14.2 Selection and reporting halves

- Each probe set and the rule tests are split in two by game. The selection half feeds the
  control score. The reporting half scores the selected checkpoint.
- Each half of a probe set is at least as large as gen-2's whole probe set.
- A probe that selects training records drops repeated rows of one record. Such a probe
  is reported and never feeds a stratum score.

## 14.3 Gate 1

Gate 1 reads the negative binomial deviance (§ 13.3). Its other margins are unchanged.

## 14.4 Rule tests

- A rule's test records are taken from games placed in the validation strata, from the
  records its detector finds there.
- The rule tests are frozen per curated corpus, with the knowledge-probe set, and exclude
  every rule with a confirmed Forge departure (§ 4.4).
- Records from probe forks, interventional resolutions, directed forks and collection aimed
  at a rule are not rule-test records.
- Records matching a variant of polarity `other` are not rule-test records.
- For each output the map names for the rule, a rule test reads the model's prediction at
  the entity the scenario's outcome names, or at the record for an output on the acting
  ability: the probability of the scenario's outcome for a binary or categorical output,
  the expected value for a count or a delta.
- A rule test's score is the AUC with which that prediction ranks the records where the
  outcome happens above those where it does not, averaged over the rule's outputs.
- A rule whose test records in a stratum hold one polarity only is left out of that
  stratum's rule-test component.

## 14.5 New probes and measures

| probe | reads |
|---|---|
| turn and step conditions | records whose acting line, candidate or board carries a turn or step condition, with whose turn it is and the phase edited; reads the matching verdict column, the predicted outcome or branch, or the affected creature's keywords or legality; subsets for casting speed and a split-second spell on the stack |
| anthem | a record whose board holds an anthem keeping damaged creatures alive, with the anthem destroyed; reads the predicted deaths of each creature |
| combat grid, several blockers | gen-2's combat grid extended to combats with several blocked attackers |
| evasion grid, protection rows | gen-2's protection rows extended to every protection quality, from a creature type to everything |
| toughness sweep, activated damage | activated damage records read with "any target" and with "target creature" |
| board-abilities ablation | gen-2's board-abilities ablation, on gen-3 checkpoints |
| ability-slot noise | the gen-2 ablation that replaces the card blocks' ability-slot `e` with noise |
| target legality | the target-legality probe reads `target_legal` |

A run with `--relation-values off` is scored on the anthem probe and the combat grids
beside a run with the value tables on.

# 15. Operator surface

```
python scripts/rules_catalogue.py split | packets | premark | validate | status | gen3-fields

python -m effects rule-counts
    --records-dir DIR                    records to count; gen-2 or gen-3
    [--output PATH]                      default output/effects/rule-counts.json

python -m sealed match-outcomes
    [--count-dropped-events]             off; counts events arriving with no open record
    [--directed-fork-rules LIST]         default none; rule ids whose directed forks run
    [--directed-forks-per-game N]        default 2

python -m effects collect-coverage
    [--wording-cards PATH]               cards for the "any target" coverage round (§ 8.11)

python -m effects train-effect-model
    [--rule-mask hard|soft]              default hard
    [--deviance-floor F]                 default measured from gen-2, below
    [--ema-epochs E]                     default 1; span of the averaged weights
    [--relation-values on|off]           default on
    [--stratum-weight W]                 default 0.5; card-disjoint share of the control score
    [--patience N]                       default 5; epochs without a new best control score
```

`collect-coverage` and `collect-variants` take `--count-dropped-events` as
`match-outcomes` does.

`collect-variants` takes `--directed-fork-rules` and `--directed-forks-per-game` as
`match-outcomes` does.

The `--deviance-floor` default is measured once. The noisy fields are those whose explained
deviance, as gen-2's 256-wide continuation runs logged it over their last five epochs, moves
by more than 5 points between consecutive epochs. The default is the largest
constant-prediction loss among them, computed on gen-2's card-disjoint validation sample
under gen-3's distributions.

Startup validation

- `build-corpus`, `train-effect-model`, `evaluate-effect-model` and the knowledge probes
  check that every record carries `generation: 3`.
- `train-effect-model` checks that the corpus manifest carries per-rule counts.
- `match-outcomes` checks that every rule in `--directed-fork-rules` has a detector and a
  built fork operation.

Exit codes

| code | meaning |
|---|---|
| 0 | success |
| 1 | a startup check failed |

Printed output

- A collecting run prints its dropped-event counts by event type and engine phase when it
  ends, under `--count-dropped-events`.
- `rule-counts` prints, per rule, the records in which its effect happens and does not,
  split by random-seat games and games between two Forge AIs.
- Each training epoch line carries each stratum's score and its components, and the
  control score.

# 16. Records and artifacts

| file | written | purpose |
|---|---|---|
| `rules/` | by the catalogue scripts | catalogue, triage, scenarios, findings; formats in `rules/README.md` |
| `rules/heads/<unit>.jsonl` | by the map | per-rule verdict, missing facts, gen-3 fields, learnable after gen-3 |
| `rules/gen3-fields.json`, `rules/GEN3-FIELDS.md` | by the map's merge | gen-3's inputs, outputs, extensions and fixes |
| effect-record shards | by collection | gen-3 records, `generation: 3` (§ 8.1) |
| `manifest.json` | by `build-corpus` | adds per-rule counts (§ 5.2), per-rule budgets (§ 12.1), one-sided conditions (§ 12.2), and the closed subtype and keyword lists (§ 11.2, § 11.3) |
| `output/effects/rule-counts.json` | by `rule-counts` | per-rule counts over a records set (§ 5.2) |
| frozen rule tests | with the knowledge-probe set | the rule tests' records and their selection and reporting halves, with a content digest |
| checkpoints | by `train-effect-model` | add the relation tables, the averaged weights, `--rule-mask`, `--deviance-floor`, `--ema-epochs`, `--stratum-weight` and `--relation-values` |
| `resume.pt` | every epoch | adds the averaged weights |
| knowledge-probe scorecards | per checkpoint | add the probes of § 14.5 and the rule tests, per half |

Run results land in the gen-3 record's Result section, never in this spec.
