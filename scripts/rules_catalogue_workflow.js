export const meta = {
  name: 'rules-catalogue',
  description: 'Triage Comprehensive Rules units, write minimal-pair scenarios for mechanical rules, adversarially verify them',
  phases: [
    { title: 'Triage', detail: 'one Sonnet agent per packet classifies every rule', model: 'sonnet' },
    { title: 'Scenarios', detail: 'Sonnet writes minimal-pair scenarios per chunk of units', model: 'sonnet' },
    { title: 'Verify', detail: 'independent reviewer checks rules, card facts and minimality, fixes in place' },
  ],
}

const WT = 'C:/Users/nicol/IdeaProjects/price-predictor/.claude/worktrees/gen3-rules-coverage'
const FORGE = 'C:/Users/nicol/IdeaProjects/forge/forge-gui/res/cardsfolder'
const VERIFY_MODEL = args.verifyModel || 'opus'
const MAX_NEEDS = args.maxNeeds || 20

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
const slot = limiter(4)
const opusSlot = limiter(3)
const run = (prompt, opts) => {
  const go = () => agent(prompt, opts)
  return slot(() => (opts.model === 'opus' ? opusSlot(go) : go()))
}

const COMMON = `Working directory: ${WT} (a git worktree). All paths below are relative to it. Do not cd anywhere else and do not touch files outside rules/ except reading. Do not commit. Forge card scripts live in ${FORGE}/<first letter>/<name lowercased, spaces to underscores, punctuation such as apostrophes and commas removed>.txt; use Glob there if a name does not resolve.`

const TRIAGE_SCHEMA = {
  type: 'object',
  properties: {
    units: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          unit: { type: 'string' },
          rules: { type: 'integer' },
          needs_scenario: { type: 'integer', description: 'rules with category mechanical or legality and observability other than none' },
        },
        required: ['unit', 'rules', 'needs_scenario'],
      },
    },
  },
  required: ['units'],
}
const SCEN_SCHEMA = {
  type: 'object',
  properties: {
    units: { type: 'array', items: { type: 'object', properties: { unit: { type: 'string' }, scenarios: { type: 'integer' }, skipped: { type: 'integer' } }, required: ['unit', 'scenarios', 'skipped'] } },
    concerns: { type: 'string', description: 'at most two sentences: rules you found hard to illustrate, or empty' },
  },
  required: ['units', 'concerns'],
}
const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    units: { type: 'array', items: { type: 'object', properties: { unit: { type: 'string' }, checked: { type: 'integer' }, changes: { type: 'integer' }, triage_changes: { type: 'integer' } }, required: ['unit', 'checked', 'changes', 'triage_changes'] } },
    common_errors: { type: 'string', description: 'at most three sentences on the kinds of errors you fixed most' },
  },
  required: ['units', 'common_errors'],
}

const triagePrompt = units => `${COMMON}

You are triaging Magic: The Gathering Comprehensive Rules for a coverage catalogue. Read rules/README.md (the introduction and "Triage schema" are binding). Then, for each unit in [${units.join(', ')}]: read rules/cr/<unit>.md and write rules/triage/<unit>.jsonl with exactly one line per bold rule id in that file, parents included, in document order. Read rules/cr/glossary.md or other rules/cr/ files only when a rule cannot be classified without them.

If rules/triage/<unit>.jsonl already exists and validates, keep it and only report its counts.

Classify by what the rule makes the engine do once every player decision is taken. A rule that names a choice and also its observable consequence is mechanical. Legality re-checked at resolution is mechanical; legality at the time of the action is legality.

Forge implements nearly every printed card, including dungeons, sieges, attractions and many digital-only mechanics. Mark a rule out_of_scope for being unsupported only after grepping ${FORGE} for the mechanic and finding nothing.

When done, run \`python scripts/rules_catalogue.py validate ${units.join(' ')} --no-cards\` and fix every error. Return per-unit counts.`

const scenPrompt = units => `${COMMON}

You write rule-illustration scenarios for a Magic: The Gathering coverage catalogue. They will become training and test examples for a model that predicts the engine's mechanical outcome from a board plus decisions already taken. Read rules/README.md in full; its "Scenario schema" is binding.

For each unit in [${units.join(', ')}]: read rules/cr/<unit>.md and rules/triage/<unit>.jsonl, then write rules/scenarios/<unit>.json. Every rule whose category is mechanical or legality and whose observability is not none needs a scenario listing it in "rules", or an entry in "skipped" whose reason names the scenario that covers it or says why it cannot be illustrated. If the file already exists and validates, keep it.

How to write a good scenario:
- It isolates the rule. The variants differ by exactly one fact, and the rule is what makes their outcomes differ. Everything else on the board is inert. A card that only one variant uses still belongs in the base setup, so that the variant's delta is a single fact (declaring a block, not adding a creature and declaring it).
- Positive and negative: at least one variant where the rule's effect happens ("fires") and one close situation where the rule applies but its effect does not happen ("does_not_fire"): toughness 2 versus 3 against 2 damage; a target still legal versus made illegal in response; the trigger condition met versus barely missed. When the rule simply is not involved in a variant (another rule decides it), the polarity is "other", not "does_not_fire".
- Do not skip a rule because no simple card comes to mind. Grep ${FORGE} for the script parameter or trigger mode that implements it (for example Mode$ AbilityResolves) and use what you find. Skip only a rule that cannot occur in a two-player Forge game, or that another scenario in the file fully covers (name it).
- A rule with distinct branches gets a scenario per branch, or a variant with polarity "other". Example: a spell whose only target became illegal fizzles (608.2b), while a spell whose resolution-time condition is false resolves and does nothing (Pyroblast on a non-blue spell).
- Prefer one scenario per rule. Group sibling subrules into one scenario only when the same variants exercise each of them.
- Decisions are all fixed in "decisions", in order, including payments, targets, modes, attacks, blocks, damage assignment order and every response. Mana must be available for every cast.
- Cards are real, spelled exactly as Forge's Name: line. Open the Forge script of every card you use and check its types, colors, cost, power/toughness and abilities match what the scenario assumes. Prefer vanilla creatures, basic lands and simple, well-known spells so nothing else interferes.
- "expected" is concrete and observable only: zone moves, life totals, damage marked, counters, power/toughness, resolved / fizzled / countered, trigger fired or not, legal or not. A legality rule uses record_kind "playability" and states the verdict.
- The situation must be reachable in a two-player Forge game.

When done, run \`python scripts/rules_catalogue.py validate ${units.join(' ')}\` and fix every error (unknown card names included). Return the summary.`

const verifyPrompt = units => `${COMMON}

You are an independent, adversarial reviewer of rule-illustration scenarios for a Magic: The Gathering coverage catalogue. Another agent wrote them; assume it made mistakes. Read rules/README.md in full (the schemas are binding).

For each unit in [${units.join(', ')}]: read rules/cr/<unit>.md, rules/triage/<unit>.jsonl and rules/scenarios/<unit>.json; a unit whose scenario file already has "verified": true is done, so skip it. Check every scenario and variant:
1. Rules correctness. Is each "expected" exactly what the Comprehensive Rules produce? When the outcome depends on other rules (state-based actions 704, resolution 608, combat damage 510, layers 613, replacement 614/615, triggers 603), read them in rules/cr/ and apply them.
2. Card facts. Open the Forge script of every card in "cards" and confirm type, color, cost, power/toughness and abilities are what the scenario assumes, and that no ability of any card on the board interferes with the outcome.
3. Minimal pairs. Each non-base variant changes exactly one fact, its "delta" names it, and that change is what flips the outcome the listed rule decides. Each scenario has a "fires" and a "does_not_fire" variant unless the rule genuinely cannot have one.
4. Completeness of the setup. Every decision is fixed, mana is available for every cast, timing is legal for every action, and the board is reachable.
5. Coverage. The scenario really exercises each rule it lists. Every skip reason holds: the named covering scenario truly covers that rule. Triage lines are plausible; fix a misclassified rule in rules/triage/<unit>.jsonl and then cover or skip it.

Fix every problem in place by editing the files: rewrite variants, swap cards, add a missing scenario, or drop one that cannot be fixed. Append one short string per change to "review" (scenario id, what was wrong, what you changed). Set "verified": true once the whole file is checked. Then run \`python scripts/rules_catalogue.py validate ${units.join(' ')}\` and fix every error. Return the summary.`

function chunk(triUnits) {
  const out = []
  let cur = [], n = 0
  for (const u of triUnits) {
    if (!u.needs_scenario) continue
    if (cur.length && n + u.needs_scenario > MAX_NEEDS) { out.push(cur); cur = []; n = 0 }
    cur.push(u.unit); n += u.needs_scenario
  }
  if (cur.length) out.push(cur)
  return out
}

const short = us => us.length === 1 ? us[0] : `${us[0]}..${us[us.length - 1]}`

const results = await pipeline(
  args.packets,
  units => run(triagePrompt(units), { label: `triage:${short(units)}`, phase: 'Triage', model: 'sonnet', effort: 'medium', schema: TRIAGE_SCHEMA }),
  async tri => {
    if (!tri) return null
    const chunks = chunk(tri.units)
    const done = await Promise.all(chunks.map(c =>
      run(scenPrompt(c), { label: `scen:${short(c)}`, phase: 'Scenarios', model: 'sonnet', effort: 'high', schema: SCEN_SCHEMA })
        .then(s => s && run(verifyPrompt(c), { label: `verify:${short(c)}`, phase: 'Verify', model: VERIFY_MODEL, effort: 'medium', schema: VERIFY_SCHEMA })
          .then(v => ({ chunk: c, scen: s, verify: v })))))
    return done.filter(Boolean).map(d => ({
      chunk: d.chunk.join(' '),
      scenarios: d.scen.units.reduce((n, u) => n + u.scenarios, 0),
      changes: d.verify ? d.verify.units.reduce((n, u) => n + u.changes + u.triage_changes, 0) : null,
      concerns: d.scen.concerns,
      errors: d.verify ? d.verify.common_errors : 'VERIFY MISSING',
    }))
  },
)
const flat = results.filter(Boolean).flat()
const failed = args.packets.filter((_, i) => !results[i]).map(p => p.join(' '))
if (failed.length) log(`packets with no result: ${failed.join(' | ')}`)
return { failed, chunks: flat }
