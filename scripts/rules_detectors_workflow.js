export const meta = {
  name: 'rules-detectors',
  description: 'T030/T031 pass: declare detectors and operations on rules scenarios (all of a unit, or only its open requests), then verify',
  phases: [
    { title: 'Declare', detail: 'one Sonnet agent per unit writes detector and operation fields', model: 'sonnet' },
    { title: 'Verify', detail: 'Opus re-checks each unit and fixes in place', model: 'opus' },
  ],
}

const WT = 'C:/Users/nicol/IdeaProjects/price-predictor'
// Batches of units, one declaring and one verifying agent each: args.batches as
// [{id, units}], or args.units, one unit per batch.
const BATCHES = args && args.batches
  ? args.batches
  : (args && args.units ? args.units : ['704', '608', '701.6', '702.2', '702.19']).map(u => ({ id: u, units: [u] }))
// 'all': every scenario of the unit. 'requests': only the scenarios with no detector yet.
const MODE = args && args.mode ? args.mode : 'all'

function limiter(n) {
  let active = 0
  const q = []
  const next = () => {
    if (active >= n || !q.length) return
    active++
    const { fn, res, rej } = q.shift()
    fn().then(res, rej).finally(() => { active--; next() })
  }
  return fn => new Promise((res, rej) => { q.push({ fn, res, rej }); next() })
}
const opusSlot = limiter(3)

const COMMON = `Working directory: ${WT}. All paths are relative to it. Do not cd elsewhere. Write only rules/scenarios/<unit>.json and rules/detector-requests/<unit>.jsonl for your units. Do not commit. Run Python as .venv/Scripts/python. Run nothing heavy: no test suites, no Maven, no JVM.`

const SCOPE = MODE === 'requests'
  ? `SCOPE: in each of your units, only the scenarios that have NO "detector" field yet. Each has a line in rules/detector-requests/<unit>.jsonl from an earlier pass, written when fewer families existed. Families have been added since: re-decide every such scenario from scratch with the families now registered. Leave every scenario that already has a detector, and every variant that already has an operation, exactly as it is, unless you find it plainly wrong (say so in your notes). When you finish, rewrite rules/detector-requests/<unit>.jsonl to hold only the requests still open (one line per scenario still without a detector); delete the file if none is left.`
  : `SCOPE: every scenario of each of your units.`

const TASK = `
## What you are doing

Each scenario in rules/scenarios/<unit>.json is a minimal pair illustrating a Comprehensive Rules rule: a setup, a base variant (the FIRST variant, empty delta), and variants that each change one fact. Gen-3 of the effect model needs two new fields on every scenario. The format is specified in rules/README.md, section "Gen-3 fields: detector, variants[].operation, departure". Read that section first and in full.

### 1. detector (every scenario)

A detector finds, in the self-play effect-record corpus, the records that show this scenario's rule, and says which side of the rule each record is on (fires / does_not_fire). Detectors are a fixed set of FAMILIES, one Python module each under src/effects/domain/rule_detectors/. Read every family module there before declaring anything: each module's docstring states what record shape it matches, its parameters and their allowed values (look at CHECKS and similar constants too). The registered families are: sba_zone_move, sba_game_loss, trigger_mode, keyword_combat, verdict_bit, continuous_contribution, rewrite_event, legality_membership, outcome_event, resolution_outcome. The record format the families read is in specs/023-ability-effect-model/contracts/record-schema.md (gen-2 records, the corpus today) and specs/025-ability-effect-model-gen3/contracts/record-schema-delta.md (fields gen-3 adds). Every event type a record can hold, with its parameters, is EVENT_PARAMS in src/effects/domain/event_schema.py.

THE MOST IMPORTANT RULE: a detector describes the rule's condition as it would show in ANY game, never the scenario's particular board. The scenario's cards only illustrate the rule. 704.5f's detector tags any creature at toughness 0 or less dying in the state-based check after any resolution, not Grizzly Bears after Shock. A detector written from the scenario's specifics matches that one board and almost nothing in the corpus. So "outcome" and "near_miss" are general sentences about the rule, and "params" hold only the family's declared parameters with general values (a check, a Mode$, a keyword, a verdict bit, an event type, an API type); never a card name, never a selector naming a card.

Facts earlier passes got wrong, check them yourself where they matter:
- Before declaring that no event shows an outcome, look it up in EVENT_PARAMS: player_won and player_lost exist, token_created, delayed_trigger_created, counter_change and many more.
- A gen-2 damage_dealt event names its source in the "cause" parameter.
- A keyword-derived trigger (Undying, Persist, Afflict ...) acts through the keyword's own line, which carries no Mode$: declare trigger_mode with "keyword" (and no mode). outcome_event, verdict_bit and rewrite_event also take "keyword" for an ability acting through a keyword's line. Shared reads live in src/effects/domain/rule_detectors/acting_line.py.
- A requested parameter may exist under another name now: verdict_bit's "source_keyword" (the candidate's card prints the keyword) and "cost_part" (a Cost$ part such as Sac<…>, PayLife<…>). No parameter name may contain "card" (the registry refuses it).
- A resolution's cost half holds no events (only payload.costs: mana, tapped, life, sacrificed, discarded, exiled); a cost that puts or removes counters, such as a loyalty cost, shows on no gen-2 record.
- A continuous record has no refs.source; its snapshot omits the static from its host's printed lines.
- sba_zone_move sees one resolution's effect half: the snapshot before it and its events. It cannot tell which pass of the state-based check a death came from.
- outcome_event's near miss is counted only with "api" set (the acting line's script API type, e.g. ChangeZone, Token, Destroy); without it the detector tags only the fires side, which is acceptable when no clean near miss exists.

For each scenario choose exactly one of:
- a declaration: {"family", "record_kind" (= the scenario's record_kind), "outcome", "near_miss", "params", "needs"}. "needs" lists gen-3 record fields the detector reads (e.g. "timing", "failed_checks" for verdict_bit's gen-3 bits); empty when it runs on gen-2 records.
- {"none": reason} when no record field or event shows the rule's outcome at all, in gen-2 or gen-3 records. Name what you looked for and where.
- a REQUEST, when a record does show the outcome but no existing family (or no existing parameter value) can detect it. Do not force an ill-fitting family and do not invent families or parameters. Leave that scenario's "detector" field absent and put one JSON line in rules/detector-requests/<unit>.jsonl:
  {"scenario": id, "rules": [...], "record_kind": kind, "what_to_detect": general description of the fires side, "near_miss": general description, "closest_family": family id or null, "missing": "new family" | "new parameter" | "new parameter value", "proposed": {"family": ..., "params": {...}}, "record_fields": [record fields/events that show it]}

### 2. operation (every variant but the base)

A directed fork reproduces a variant on a live game by applying one of seven operations to a copy of the game: pt (set a creature's base power/toughness), keyword (add or remove one keyword), colour_type (add or remove a colour, a card type or a subtype), counters (set a counter count, on a permanent or a player, poison included), tap (tap or untap a permanent), life (set a player's life total), replace_card (put a different card in a permanent's place, e.g. "B's creature is Centaur Courser instead of Grizzly Bears"). Classify each non-base variant's delta into the one operation that produces it from the base, or {"unclassifiable": reason} when none does: adding or removing a permanent, a different decision or choice by a player, a different spell cast, a different target, changing a hand or library, changing the phase or turn, more than one fact. Put it in the variant as "operation". The base variant gets no operation.

### 3. Editing and checking

Edit the JSON with a Python script that loads the file, sets the fields, and writes it back with json.dumps(doc, indent=2, ensure_ascii=False) + "\\n" (the file's existing format; nothing else may change). Then run:
  .venv/Scripts/python scripts/rules_catalogue.py validate <unit> --no-cards
Every error must be fixed except "no detector" on a scenario you filed a request for.
`

const DECLARE_SCHEMA = {
  type: 'object',
  properties: {
    units: { type: 'array', items: { type: 'string' } },
    declared: { type: 'integer', description: 'scenarios in scope given a family declaration' },
    none: { type: 'integer', description: 'scenarios in scope declared none' },
    requests: { type: 'integer', description: 'scenarios in scope left as requests' },
    operations: { type: 'integer', description: 'variants given one of the seven operations' },
    unclassifiable: { type: 'integer' },
    validate_errors_left: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string', description: 'anything the reviewer should know: families that almost fit, doubts' },
  },
  required: ['units', 'declared', 'none', 'requests', 'operations', 'unclassifiable', 'validate_errors_left', 'notes'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    units: { type: 'array', items: { type: 'string' } },
    changes: { type: 'array', items: { type: 'string' }, description: 'one line per change made, naming scenario/variant and why' },
    too_specific_found: { type: 'integer', description: 'declarations that were tied to the scenario board and had to be generalized' },
    requests_kept: { type: 'integer' },
    validate_errors_left: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['units', 'changes', 'too_specific_found', 'requests_kept', 'validate_errors_left', 'notes'],
}

const files = units => units.map(u => `rules/scenarios/${u}.json (rule texts: rules/cr/${u}.md)`).join(', ')

const declarePrompt = batch => `${COMMON}

Your units: ${batch.units.join(', ')}. Files: ${files(batch.units)}. Other agents work beside you in the same scratchpad: name every scratch file you write with the prefix ${batch.id}-declare-, and never run a script you did not write. Work through them one unit at a time; each unit has its own request file and its own validate run.
${SCOPE}
${TASK}
Return the counts (for the scenarios in scope) and any notes.`

const verifyPrompt = (batch, declared) => `${COMMON}

Your units: ${batch.units.join(', ')}. Other agents work beside you in the same scratchpad: name every scratch file you write with the prefix ${batch.id}-verify-, and never run a script you did not write. Another agent has just worked on ${files(batch.units)} and their rules/detector-requests/<unit>.jsonl files. ${SCOPE} Its summary: ${JSON.stringify(declared)}.

Verify every declaration, operation and request in that scope against the rules below and fix what is wrong in place. Check in particular:
- generality: a detector tied to the scenario's cards or board (a parameter, an outcome or near_miss sentence naming the scenario's cards, a selector naming a card) is wrong; rewrite it to the rule's general condition;
- the family really matches the rule's record shape and the params are ones the family's module declares, with values it accepts (read the module);
- the near_miss is the rule's does_not_fire side, and "needs" is right for the fields read;
- "none" is used only where no record field shows the outcome; a request only where no family fits;
- each operation produces exactly that variant's delta from the base, and unclassifiable has a concrete reason;
- the request file holds exactly the scenarios still without a detector.
${TASK}
Return what you changed.`

phase('Declare')
const results = await pipeline(
  BATCHES,
  batch => agent(declarePrompt(batch), { label: `declare:${batch.id}`, phase: 'Declare', model: 'sonnet', effort: 'medium', schema: DECLARE_SCHEMA }),
  (declared, batch) => declared && opusSlot(() => agent(verifyPrompt(batch, declared), { label: `verify:${batch.id}`, phase: 'Verify', model: 'opus', effort: 'medium', schema: VERIFY_SCHEMA }))
    .then(verified => ({ batch: batch.id, units: batch.units, declared, verified })),
)
return results.filter(Boolean)
