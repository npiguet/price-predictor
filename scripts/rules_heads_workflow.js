export const meta = {
  name: 'rules-heads',
  description: 'Map every mechanical and legality rule to the model output trained on it and the inputs it needs, then verify',
  phases: [
    { title: 'Map', detail: 'one Sonnet agent per packet writes rules/heads/<unit>.jsonl', model: 'sonnet' },
    { title: 'Verify', detail: 'Opus re-checks each packet against MODEL-FIELDS.md and fixes in place', model: 'opus' },
  ],
}

const WT = 'C:/Users/nicol/IdeaProjects/price-predictor/.claude/worktrees/gen3-rules-coverage'
const VERIFY_MODEL = args.verifyModel || 'opus'

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

const COMMON = `Working directory: ${WT} (a git worktree). All paths are relative to it. Do not cd elsewhere, do not write outside rules/heads/, do not commit.`

const SCHEMA_TEXT = `
## The question

For each rule, decide whether the gen-2 effect model can LEARN it: whether it READS every fact that decides the rule's outcome, and whether some output is TRAINED on that outcome. The rule's scenarios (rules/scenarios/<unit>.json) make this concrete: the difference in "expected" between a scenario's fires and does_not_fire variants is the outcome that must be trained, and each variant's "delta" is a fact the model must read. rules/MODEL-FIELDS.md is the authority on what the model reads and what is trained; read it first and in full. When it leaves a doubt, src/effects/domain/effect_targets.py (record -> targets) and src/effects/domain/effect_head_input.py (record -> inputs) are the code.

Judge where the outcome actually lands as records carry it today. Examples:
- A creature dying in the state-based check after a burn spell resolves: the effect half carries the zone change, so zone_outcome (died) is trained.
- A replacement such as "if it would die, exile it instead": the destroy spell's effect half carries the zone change to exile, so zone_outcome is trained, even though the rewrite record itself carries no outcome.
- A spell fizzling under 608.2b because its only target became illegal: no effect half is written and the outcome is only an input, so output "recorded" with "record:resolution_outcome".
- A spell that resolves but does nothing because a condition fails on resolution: the effect half exists with no events, so the gate (nothing affected) is trained.
- Damage prevented: "record:event:damage_prevented" unless the outcome is fully visible as damage_taken staying 0, in which case damage_taken/gate is trained.
- Casting legality: can_play/affordable/has_legal_target; timing failures are "planned:timing" and specific failed checks "planned:failed_checks" (already planned in the gen-3 design).
- A trigger firing or not: trigger_fired; what the trigger then does is a separate resolution of its own.
- A static ability on a permanent other than the acting one (an anthem, "creatures can't block", a cost reducer), or a triggered or replacement ability on a non-acting permanent: the model does NOT read those lines today (only spell and activated abilities are ability tokens), so the input is incomplete with "planned:every_line".
- Subtypes (Goblin, Aura, Equipment, Saga, Siege ...) are not read: "snapshot:subtypes".

## Output format: rules/heads/<unit>.jsonl

One JSON object per line, one line for EVERY rule of the unit whose triage category is mechanical or legality and whose observability is not none (rules/triage/<unit>.jsonl), in rule order, and for no other rule:

{"id": "704.5f", "output": "trained", "outputs": ["zone_outcome"], "input": "complete", "inputs_missing": [], "note": ""}

- "output", exactly one of:
  - "trained": an existing output named in MODEL-FIELDS.md is trained on this outcome by the records collected today.
  - "recorded": a record carries the outcome, but no output is trained on it (an input-only field, a payload field, or an event type with no target path).
  - "missing": no record carries the outcome at all.
  - "planned": the gen-3 design record already plans the output (timing, failed_checks).
- "outputs": non-empty list. Existing output names exactly as MODEL-FIELDS.md spells them (gate, zone_outcome, tap_state, damage_taken, power_delta, toughness_delta, pt_duration, counters_delta_<type> e.g. counters_delta_p1p1, types_gained, types_lost, colors_gained, colors_lost, type_color_duration, keywords_gained, keywords_lost, control_change, attached_to_change, life_delta, cards_drawn, cards_discarded, cards_milled, library_events, poison_delta, mana_delta_<colour>, target_legal, attacker_legal, blocker_legal, min_blockers, can_play, affordable, has_legal_target, mana_paid, trigger_fired, created_objects). For "recorded", name what the record carries as "record:<field or event:type>" (record:resolution_outcome, record:event:damage_prevented, record:event:player_removed, record:rewrite_result, record:cost_after_adjustment, record:event:regenerated ...) and also the "new:<output>" that would train it. For "missing", name "new:<output>". For "planned", "planned:timing" or "planned:failed_checks". A "trained" rule may also list a "new:" output when the existing one captures the outcome only partly; say why in note.
- "input": "complete" when the model reads every fact that decides the outcome (every variant's delta), otherwise "incomplete".
- "inputs_missing": empty when complete; otherwise tagged names: "snapshot:<field>" (in the snapshot, not read: subtypes, name, owner_of_record, attached_to_target, copy_source, face, emblem_abilities, block_pairing ...), "planned:<name>" (planned in gen-3: owner, active_player, priority_holder, block_pairing, every_line), "new:<name>" (not recorded at all).
- "note": one short sentence when a judgement was borderline, else "".

Reuse these names wherever they fit, so that the pass can count them; invent a new snake_case name only when none fits.
- new outputs: resolution_outcome, damage_source, damage_prevented, rewrite_result, game_result, regenerated, designations_changed, monarch_change, initiative_change, day_night_change, dungeon_progress, speed_change, city_blessing_gained, class_level_change, room_unlocked, soulbond_pair_change, extra_turn, extra_phase, phase_skipped, turn_skipped, spell_copied, permanent_copied, cost_after_adjustment, legal_targets, keywords_other, subtypes_gained, subtypes_lost, supertypes_changed, ownership_change, text_change, abilities_gained, abilities_lost, removed_from_combat, became_blocked, x_value, delayed_trigger_created, library_shuffled, energy_delta, radiation_delta, phased_in, mana_value_change, face_change, emblem_created, game_restart
- new inputs: subtypes, name, owner, active_player, priority_holder, block_pairing, every_line, attached_to_target, designations, day_night, monarch, initiative, dungeon_room, speed, city_blessing, soulbond_pair, class_level, room_unlocked, exiled_with, linked_choice, this_turn_history, copy_source, face, emblem_abilities, damage_sources_this_turn, pending_extra_turns, chosen_values, x_paid_on_permanent, mana_spent, timestamps, game_history
`

const MAP_SCHEMA = {
  type: 'object',
  properties: {
    units: { type: 'array', items: { type: 'object', properties: { unit: { type: 'string' }, lines: { type: 'integer' } }, required: ['unit', 'lines'] } },
    validate_errors: { type: 'integer', description: 'errors reported by the validate command at the end' },
  },
  required: ['units', 'validate_errors'],
}
const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    changed: { type: 'integer', description: 'heads lines changed' },
    fixes: { type: 'string', description: 'at most three sentences: what kinds of mistakes were fixed, with one example rule id each' },
    validate_errors: { type: 'integer' },
  },
  required: ['changed', 'fixes', 'validate_errors'],
}

const validateCmd = units => `python scripts/rules_catalogue.py validate ${units.join(' ')} --no-cards --heads`

const mapPrompt = units => `${COMMON}

Units: ${units.join(', ')}.
${SCHEMA_TEXT}
## Steps

1. Read rules/MODEL-FIELDS.md in full.
2. For each unit: read rules/triage/<unit>.jsonl and rules/scenarios/<unit>.json; read rules/cr/<unit>.md for a rule's text when the triage outcome and scenarios do not settle it. A rule listed under the scenario file's "skipped" still gets a heads line, judged from its text and triage outcome.
3. Write rules/heads/<unit>.jsonl for each unit, overwriting any existing file only if it fails validation.
4. Run: ${validateCmd(units)}
   Fix every reported error and re-run until it reports 0 errors. Report the final count.`

const verifyPrompt = units => `${COMMON}

Units: ${units.join(', ')}.

Another agent wrote rules/heads/<unit>.jsonl for these units. Check every line independently, as a skeptical reviewer, and fix what is wrong in place. The usual mistakes: calling an outcome "trained" when it reaches no output (a fizzle, a prevented event, a payload field, an event type with no target path, a keyword outside the overlay list); missing an input the model does not read (subtypes, a static, triggered or replacement line on a non-acting permanent, a designation, owner, who blocks whom); naming outputs or inputs off the shared list when a listed name fits; and "complete" inputs on a rule whose variant delta the model cannot see.
${SCHEMA_TEXT}
## Steps

1. Read rules/MODEL-FIELDS.md in full, then for each unit rules/triage/<unit>.jsonl, rules/scenarios/<unit>.json and rules/heads/<unit>.jsonl; the rule text in rules/cr/<unit>.md where needed; the two source files named above where MODEL-FIELDS.md leaves a doubt.
2. Correct each wrong line in place. Leave correct lines byte-identical.
3. Run: ${validateCmd(units)} and fix until 0 errors.`

const short = units => units.length > 3 ? `${units[0]}..${units[units.length - 1]}` : units.join(',')

const results = await pipeline(
  args.packets,
  units => run(mapPrompt(units), { label: `map:${short(units)}`, phase: 'Map', model: 'sonnet', effort: 'medium', schema: MAP_SCHEMA }),
  (m, units) => m && run(verifyPrompt(units), { label: `verify:${short(units)}`, phase: 'Verify', model: VERIFY_MODEL, effort: 'medium', schema: VERIFY_SCHEMA })
    .then(v => ({ units, map: m, verify: v })),
)
return results
