export const meta = {
  name: 'rules-detectors',
  description: 'T030/T031 pass: declare detectors and operations on rules scenarios (all of a unit, only its open requests, or only its unclassifiable variants), then verify',
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
// 'variants': only the non-base variants marked unclassifiable, re-decided over the ten
// fork operations (T031b); detectors and classified variants are left alone.
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

const WRITES = MODE === 'variants'
  ? 'Write only rules/scenarios/<unit>.json for your units.'
  : 'Write only rules/scenarios/<unit>.json and rules/detector-requests/<unit>.jsonl for your units.'
const COMMON = `Working directory: ${WT}. All paths are relative to it. Do not cd elsewhere. ${WRITES} Never stage and never commit: no git add, no git commit, no git stash, no git checkout, no git restore. Run Python as .venv/Scripts/python. Run nothing heavy: no test suites, no Maven, no JVM.`

const SCOPES = {
  all: `SCOPE: every scenario of each of your units.`,
  requests: `SCOPE: in each of your units, only the scenarios that have NO "detector" field yet. Each has a line in rules/detector-requests/<unit>.jsonl from an earlier pass, written when fewer families existed. Families have been added since: re-decide every such scenario from scratch with the families now registered. Leave every scenario that already has a detector, and every variant that already has an operation, exactly as it is, unless you find it plainly wrong (say so in your notes). When you finish, rewrite rules/detector-requests/<unit>.jsonl to hold only the requests still open (one line per scenario still without a detector); delete the file if none is left.`,
  variants: `SCOPE: in each of your units, only the non-base variants whose "operation" is currently {"unclassifiable": ...}. Why this rerun: the first pass classified against seven operations and left most variants unclassifiable, mostly for adding or removing a permanent or for changing a card in a hand, library or graveyard. Three operations now cover those (remove_permanent, add_permanent, move_card). Re-decide every variant in scope from scratch against all ten operations; do not keep an old reason because it is there. Change nothing else: every "detector", every other field of a scenario, every base variant, every variant that already names an operation, and every variant's label, polarity, delta and expected stay exactly as they are, even where you would word them differently (put doubts in your notes). Do not touch rules/detector-requests/.`,
}
if (!SCOPES[MODE]) throw new Error(`unknown mode ${MODE}: one of ${Object.keys(SCOPES).join(', ')}`)
const SCOPE = SCOPES[MODE]

// The ten fork operations (root spec § 6, FR-011, FR-011a), shared by the
// detector pass's section 2 and the variants rerun.
const OPERATIONS = `A directed fork reproduces a variant on a live game: it waits for a live board that matches the base variant, copies the game, and applies ONE operation to the copy. Classify each non-base variant's delta into the one operation that produces it from the base, or {"unclassifiable": reason} when none does. Put it in the variant as "operation". The base variant gets no operation. The ten operations, by wire spelling:
- pt: set one creature's base power and/or toughness ("B's creature is 3/3 instead of 2/2"). The creature is otherwise the same card.
- keyword: add or remove one keyword on one permanent ("Grizzly Bears has flying").
- colour_type: add or remove one colour, card type or subtype on one permanent ("the creature is also an artifact", "the Bear is blue").
- counters: set the count of one kind of counter on one permanent or one player, poison counters included ("the creature has two +1/+1 counters", "B has 9 poison counters"). Marked damage is not a counter.
- tap: tap or untap one permanent.
- life: set one player's life total.
- replace_card: put a different card in one permanent's place on the battlefield, same controller ("B's creature is Centaur Courser instead of Grizzly Bears").
- remove_permanent: take one permanent off the battlefield ("Painter's Servant is not on the battlefield", "A controls no Island"). It leaves combat, is unattached from what it was attached to and from what was attached to it, and goes to NO zone: no leaves-the-battlefield trigger, no graveyard. A delta that also needs the card to be in some zone (a graveyard, a hand, exile) for the outcome is two facts.
- add_permanent: put one new card from the card database onto the battlefield under the variant's controller, tapped or untapped, summoning-sick or not, and with counters, as the delta states ("A also controls Glorious Anthem", "B controls a 1/1 white Soldier token"). No enters-the-battlefield trigger and no replacement effect applies. It attaches the card to nothing: an Aura or Equipment that must be attached to something is two facts.
- move_card: move one card between a hand, a library (at a stated position) and a graveyard, or put a different card from the card database at one position of a hand, library or graveyard in place of the card there ("A's top library card is Island instead of Forest", "the Bear is in A's graveyard instead of A's hand"). No draw, discard or mill event happens, and the players' per-turn counts (cards drawn, lands played, spells cast this turn) stay as the live game left them. Exile, the stack, the command zone and the battlefield are not among its zones.

Every edit is silent: it raises no event the change would raise in a game, no trigger, no replacement. A variant whose outcome depends on such an event having happened (an enters trigger on the stack, a card drawn this turn, a creature that died this turn) is made by no operation.

What stays unclassifiable. Start the reason with its category, then a colon, then the concrete difference:
- "choice": a different choice or decision by a player (a mode, an X, a cost paid, an answer to a question, a different spell cast or ability activated, a response, a different attack or block declared);
- "target": a different target;
- "timing": a different timing (when a spell is cast or an ability used, the order things happen in, what is on the stack);
- "phase or turn": a different phase, step or turn, or whose turn it is;
- "history": something that happened earlier in the game (entered or died this turn, under control since the turn began, cards drawn or lands played this turn, damage marked);
- "more than one fact": the delta needs two or more operations (a permanent changing controller is a removal plus an addition; a card leaving the battlefield for a graveyard is a removal plus a move);
- "not a board edit": anything else no operation edits (exile, the stack, the command zone, a game rule or format, a property no operation sets, a continuous effect with no permanent behind it). Name it.`

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

${OPERATIONS}

### 3. Editing and checking

Edit the JSON with a Python script that loads the file, sets the fields, and writes it back with json.dumps(doc, indent=2, ensure_ascii=False) + "\\n" (the file's existing format; nothing else may change). Then run:
  .venv/Scripts/python scripts/rules_catalogue.py validate <unit> --no-cards
Every error must be fixed except "no detector" on a scenario you filed a request for.
`

const VARIANTS_TASK = `
## What you are doing

Each scenario in rules/scenarios/<unit>.json is a minimal pair illustrating a Comprehensive Rules rule: a setup, a base variant (the FIRST variant, empty delta), and variants that each change one fact. Every non-base variant carries an "operation": the change a directed fork applies to a live board matching the base to make that variant. The format is in rules/README.md, section "Gen-3 fields: detector, variants[].operation, departure"; read its "operation" bullet first. Read each in-scope variant's delta against the scenario's setup and base variant: the delta says what differs, the setup says what the base board is, and "expected" says what outcome the difference must produce.

## The operations

${OPERATIONS}

## Editing and checking

Edit the JSON with a Python script that loads the file, sets the "operation" of the in-scope variants, and writes it back with json.dumps(doc, indent=2, ensure_ascii=False) + "\\n" (the file's existing format; nothing else may change). Check with git diff rules/scenarios/<unit>.json that only "operation" values of in-scope variants changed (other agents edit other units at the same time: their files are not yours to check or touch). Then run:
  .venv/Scripts/python scripts/rules_catalogue.py validate <unit> --no-cards
Every error must be fixed except "no detector" (detectors are another pass's work).
`
const TASKS = { all: TASK, requests: TASK, variants: VARIANTS_TASK }

const DECLARE_SCHEMA = {
  type: 'object',
  properties: {
    units: { type: 'array', items: { type: 'string' } },
    declared: { type: 'integer', description: 'scenarios in scope given a family declaration' },
    none: { type: 'integer', description: 'scenarios in scope declared none' },
    requests: { type: 'integer', description: 'scenarios in scope left as requests' },
    operations: { type: 'integer', description: 'variants given one of the ten operations' },
    unclassifiable: { type: 'integer' },
    validate_errors_left: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string', description: 'anything the reviewer should know: families that almost fit, doubts' },
  },
  required: ['units', 'declared', 'none', 'requests', 'operations', 'unclassifiable', 'validate_errors_left', 'notes'],
}

const OPERATION_NAMES = ['pt', 'keyword', 'colour_type', 'counters', 'tap', 'life', 'replace_card',
  'remove_permanent', 'add_permanent', 'move_card']
const UNCLASSIFIABLE_CATEGORIES = ['choice', 'target', 'timing', 'phase or turn', 'history',
  'more than one fact', 'not a board edit']
const counts = (names, description) => ({
  type: 'object', description,
  properties: Object.fromEntries(names.map(name => [name, { type: 'integer' }])),
})

const VARIANTS_DECLARE_SCHEMA = {
  type: 'object',
  properties: {
    units: { type: 'array', items: { type: 'string' } },
    in_scope: { type: 'integer', description: 'variants that were unclassifiable when you started' },
    classified: counts(OPERATION_NAMES, 'in-scope variants now given each operation; omit operations with none'),
    unclassifiable: counts(UNCLASSIFIABLE_CATEGORIES, 'in-scope variants still unclassifiable, per reason category'),
    validate_errors_left: { type: 'array', items: { type: 'string' }, description: 'errors other than "no detector"' },
    notes: { type: 'string', description: 'doubtful calls, and variants whose delta reads ambiguously' },
  },
  required: ['units', 'in_scope', 'classified', 'unclassifiable', 'validate_errors_left', 'notes'],
}

const VARIANTS_VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    units: { type: 'array', items: { type: 'string' } },
    changes: { type: 'array', items: { type: 'string' }, description: 'one line per change made, naming scenario/variant and why' },
    classified: counts(OPERATION_NAMES, 'in-scope variants given each operation after your fixes; omit operations with none'),
    unclassifiable: counts(UNCLASSIFIABLE_CATEGORIES, 'in-scope variants still unclassifiable after your fixes, per reason category'),
    validate_errors_left: { type: 'array', items: { type: 'string' }, description: 'errors other than "no detector"' },
    notes: { type: 'string' },
  },
  required: ['units', 'changes', 'classified', 'unclassifiable', 'validate_errors_left', 'notes'],
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
${TASKS[MODE]}
Return the counts (for the ${MODE === 'variants' ? 'variants' : 'scenarios'} in scope) and any notes.`

const DETECTOR_CHECKS = `Verify every declaration, operation and request in that scope against the rules below and fix what is wrong in place. Check in particular:
- generality: a detector tied to the scenario's cards or board (a parameter, an outcome or near_miss sentence naming the scenario's cards, a selector naming a card) is wrong; rewrite it to the rule's general condition;
- the family really matches the rule's record shape and the params are ones the family's module declares, with values it accepts (read the module);
- the near_miss is the rule's does_not_fire side, and "needs" is right for the fields read;
- "none" is used only where no record field shows the outcome; a request only where no family fits;
- each operation produces exactly that variant's delta from the base, and unclassifiable has a concrete reason;
- the request file holds exactly the scenarios still without a detector.`

const VARIANTS_CHECKS = `The scope is the variants unclassifiable in the committed file (git show HEAD:rules/scenarios/<unit>.json), which holds the state before this rerun; git diff rules/scenarios/<unit>.json shows what the other agent changed. Verify every variant in that scope against the rules below and fix what is wrong in place. Check in particular:
- each operation produces exactly that variant's delta from the base by ONE silent edit: the right operation (a different card in a permanent's place is replace_card, not a removal plus an addition), and no second fact hiding in the delta (a controller change, a card that must also land in a graveyard, an attached Aura or Equipment);
- the outcome the variant's "expected" describes does not depend on an event the silent edit does not raise (an enters trigger, a draw, a death this turn); if it does, the variant is unclassifiable;
- every variant still unclassifiable fits its reason's category, and its reason names the concrete difference;
- nothing outside the scope changed: git diff rules/scenarios/<unit>.json shows only "operation" values of in-scope variants, no detector and no other field.`

const verifyPrompt = (batch, declared) => `${COMMON}

Your units: ${batch.units.join(', ')}. Other agents work beside you in the same scratchpad: name every scratch file you write with the prefix ${batch.id}-verify-, and never run a script you did not write. Another agent has just worked on ${files(batch.units)}${MODE === 'variants' ? '' : ' and their rules/detector-requests/<unit>.jsonl files'}. ${SCOPE} Its summary: ${JSON.stringify(declared)}.

${MODE === 'variants' ? VARIANTS_CHECKS : DETECTOR_CHECKS}
${TASKS[MODE]}
Return what you changed.`

phase('Declare')
const results = await pipeline(
  BATCHES,
  batch => agent(declarePrompt(batch), { label: `declare:${batch.id}`, phase: 'Declare', model: 'sonnet', effort: 'medium', schema: MODE === 'variants' ? VARIANTS_DECLARE_SCHEMA : DECLARE_SCHEMA }),
  (declared, batch) => declared && opusSlot(() => agent(verifyPrompt(batch, declared), { label: `verify:${batch.id}`, phase: 'Verify', model: 'opus', effort: 'medium', schema: MODE === 'variants' ? VARIANTS_VERIFY_SCHEMA : VERIFY_SCHEMA }))
    .then(verified => ({ batch: batch.id, units: batch.units, declared, verified })),
)
return results.filter(Boolean)
