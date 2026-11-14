# Forced plays in directed forks: what a read-only trace found (2026-11-14)

A source-only trace over the 69 rules of `rules/directed-fork-rules.txt` (as of `c34e7653`), asked for by the user. Nothing here is built. The AI behaviour comes from reading `../forge/forge-ai`; no game was played. The user decides what is built (see "Resume here" in `HANDOFF.md`).

## The answer

- About a third of the 69 rules yield a tagged record from their fork as registered.
- About 22 need a play Forge's AI makes rarely or never (or that cannot be told from source).
- 12 reach their point with no choice, but the record that shows it is a decision candidate written only at a sampling rate (0.01, 0.001 or 0.0001); 3 more depend on a trigger's "evaluated, did not fire" record, which a fork's fresh recording almost never writes (`TriggerNegativeSampler.offer` wants an earlier firing of the same mode in that fork).
- For about 16 rules the fork's variant writes NO record the detector can tag, whatever anyone plays. Forcing a play fixes none of these: the variant must be inverted or the rule leaves the list.
- Each listed rule takes an equal share of the directed-fork budget, so about a third of it is spent on forks that write nothing.
- Building forced plays reverses a recorded decision: the 2026-11-02 clarification and FR-011 ("No operation forces a choice, a target or a timing on a seat") and FR-010 (every seat decides afresh). The spec changes first.

## Classes (how the rule's point is reached in the fork)

| Class | Count | Rules |
|---|---|---|
| No choice (turn-based or state-based action, trigger, the engine's own question) | 16 | 104.2b, 104.4a, 122.1d, 400.6, 510.2, 608.2p, 701.42a, 701.42b, 701.44d, 702.1c, 702.19c, 702.98a, 702.118b, 704.5b, 712.4a, 712.14c |
| Sampled only (playability candidate) | 12 | 405.6d, 601.2c, 702.62c, 702.84a, 702.102c, 702.126a, 702.127a, 702.133a, 702.187b, 702.188a, 712.11d, 718.3a |
| AI choice, likely | 16 | 111.10n, 111.10p, 119.8, 614.17d, 701.23f, 701.24e, 701.24g, 701.44b, 701.50f, 701.57b, 702.85a, 702.103e, 702.131a, 702.154a, 702.167a, 702.170c |
| AI choice, unlikely or never | 20 | 104.6, 111.10v, 118.11, 118.12b, 119.7, 607.2k, 701.12e, 701.12g, 701.12h, 701.39a, 702.35a, 702.53a, 702.65a, 702.65b, 702.71a, 702.124j, 704.5f, 704.5v, 707.10d, 730.2g |
| Cannot tell from source | 5 | 104.3d, 205.1a, 702.70b, 702.105a, 704.7 |

Trigger-negative dependent (besides their class): 608.2p, 702.105a, 702.154a.

## Forks that write nothing the detector can tag (16, plus 2 probable)

104.2b, 104.6, 111.10n, 111.10p, 111.10v, 118.11, 400.6, 614.17d, 701.24g, 701.42a, 702.35a, 702.167a, 704.7, 712.4a, 712.14c, 712.11d; probable: 702.84a (variant fails the zone check, detector reads the casting-speed check), 119.8 (scenario 119.8-2's detector is `none`).

- A replacement taken away (400.6, 614.17d, 704.7, 702.35a): Forge's hooks fire only for a replacement that applies, so no replacement means no rewrite record.
- The trigger never triggers (104.2b Felidar `LifeAmount$ GE40`; 701.42a, 712.4a, 712.14c Graf Rats `IsPresent2`): no trigger, no resolution record.
- Nothing left to play (104.6, 111.10v, 118.11, 702.167a).
- 111.10n, 111.10p: no Role is created; the near miss is another source's static.
- 701.24g: its detector counts no near miss. 712.11d: the variant removes the Disturb candidate the detector reads.

Variants of polarity `other` (count on neither side): 205.1a, 701.12h, 701.23f, 702.70b.

## Variants that point the wrong way (bind on the rare board and remove; should bind on the common board and add)

- 118.11: add Rest in Peace, force the sacrifice activation with its cost paid.
- 104.2b: Felidar under 40 life, set life to 40 (no choice needed).
- 104.3d: bind at 8 poison, set 9 (gen-2: fires 36, does_not_fire 1032).
- 702.85a: move a cheaper card into the library (gen-2: fires 0, does_not_fire 168).
- 702.124j: move Toothy into the library (the AI will not cast Pir when Toothy is absent).
- 702.131a: add a tenth permanent (gen-2: fires 0, does_not_fire 1121).
- 614.17d: add Grafdigger's Cage (the AI casts Zombify whatever the Cage).
- 111.10v: a Mutagen token with no spare mana, add a land, force the activation.

Detectors that read what a COST did (so the forced play must really pay its cost; intervention-style forcing bypasses costs): 118.11, 702.53a, 702.71a, 607.2k.

## The build, smallest first

1. **Keep the record, no play forced (small, about a day).** In a directed fork keep the bound card's decision candidates at rate 1 and let the bound trigger's "evaluated, did not fire" record through. A predicate handed to `GameRecording.forkRecording`, read by `PlayabilityChannel` and `TriggerNegativeSampler.offer`. Fixes about 13 rules: 405.6d, 601.2c, 702.62c, 702.102c, 702.126a, 702.127a, 702.133a, 702.187b, 702.188a, 718.3a, and the trigger negatives 608.2p, 702.105a, 702.154a.
2. **Invert the wrong-way variants (small to medium; scenarios plus rule classes).** No forcing needed for 104.2b, 702.85a, 702.124j, 614.17d, and 702.131a for a sorcery.
3. **Force one cast or activation, with a "you may" answered (medium, 3-5 days with step 2).** A `ForcedPlay` value on the variant's `Application`: actor (a `Binding` role), step (the fork's own or a later one of the same turn), kind (cast / activate with a source role and a line selector; answer a "you may"; keep-record), targets by role (unnamed clauses drawn by `RandomChoices`), cost choices by role, X and modes. Executed by a forcing seat controller put on the actor's player in the COPY only: a `PlayerControllerAi` subclass on the pattern of `RandomSeatController`, overriding `chooseSpellAbilityToPlay` / `playChosenSpellAbility` (play through `RandomSpellPlayer.play`, the engine's own PAID cast path, with a role-aware cost decision), `confirmTrigger` / `confirmAction` / `payCostToPreventEffect`; everything else stays the AI's. `InterventionFork`'s forced resolution is NOT reusable (it pushes the ability on the stack with `addAndUnfreeze`, bypassing costs, and the fork's own recorders would write a cost record for a payment that never happened). A dry run at precondition time (the `hasChoices` idea) keeps the budget off impossible forks. Fixes about 13: 118.11 (inverted), 111.10v (inverted), 119.7, 607.2k, 701.12e, 701.12g, 701.39a, 702.53a, 702.65a, 702.65b, 702.71a, 704.5f, 702.131a's instants; answering a "you may" adds 118.12b; and it makes about 10 "likely" rules certain.
4. **Force an attack (small, on top of 3).** 702.105a, 702.154a (each also needs step 1), 104.3d.

Not covered: a play made in response, with something on the stack (707.10d, 702.103e's fires side, 115.7a).

Refusals (counted, `ForkRefusal`): no legal line or target after the edit; forced play not made by the end of the turn; cost not paid. The controller flags, `ForkPlayForward.play` throws after its loop and before `writeForkLines`.

Records and wire: a paid forced play is written by the fork's normal recorders as a live play is. Stamps stay `fork=true`, `interventional=false`, `aimed_at=[rule]`, `directed={variant, operation}`. NO wire change is needed (which variants force is fixed per rule in the registry). `interventional=true` is not an option: Python refuses a directed record that is interventional (`src/effects/domain/records.py` ~965). Consumers: rule tests exclude every fork and aimed record; `protection_copies`, `lethality_counterfactuals`, `gate_two`, `grid_edits` skip fork records; `rule_mask` treats aimed records as natural (FR-061); `rule-counts` counts them under "aimed". A paid forced play fits the random seat's convention (an off-policy play with no per-record mark).

Size of steps 2-3: new `ForcedPlay`, a forcing seat controller (300-400 lines), a role-aware cost decision; touched `ForkSeats`, `ForkPlayForward`, `RandomSeatController`, the refusal reasons, about 14 rule classes, about 8 scenario variants. Spec: amend the 2026-11-02 clarification, FR-010, FR-011, a new FR, FR-014b's reasons; `contracts/catalogue-extensions.md`; `data-model.md`; `src/effects/CLAUDE.md`. Risks: the AI's mana payment failing on a play the rules allow (refusals); forcing at first priority changes how the AI sequences the rest of its turn; a random seat's fork needs `RandomSeatController` to consult `ForcedPlay`.

## Hopeless as registered (off the list, or a new variant)

- 104.6: its variant writes nothing, AND a hazard: its precondition binds the boards where the fork's AI may restart the game; Forge's restart resets the static `Trigger` id counter the whole JVM shares, and `Forks.LiveMarks` does not read it.
- 400.6, 704.7: the missing side needs a replacement that applies and does not replace; no edit produces it.
- 701.24g: the detector counts no near miss.
- 701.42a, 712.4a, 712.14c: Graf Rats' trigger condition.
- 702.35a: no replacement without madness (and the AI's discard choice ignores madness).
- 702.167a: no material, no activation.
- 111.10n, 111.10p: the near miss is another source's static.
- 704.5v: `DamageDealAi` never targets a battle; a defeated Siege's trigger exiles it anyway.
- 730.2g: needs a turn-face-up effect nothing in the turn provides.
- 707.10d: needs a two-spell response sequence.
- 712.11d: the variant removes the candidate the detector reads.
- 205.1a, 701.12h, 701.23f, 702.70b: `other` polarity (701.12h also needs a trigger's targets forced: `ExchangeTextBox` has no AI).

## Only a played game settles

Whether the champion leaves trigger fires after a failed-champion sacrifice (607.2k); whether a forced Elfhame search finding nothing writes `phase_skipped` (118.12b); whether gen-2's 702.85a "fires 0" comes from the AI or the detector; whether any rewrite record reads "met and not replaced" (400.6, 704.7); how 119.8's fork records are tagged; whether the AI's own mana payment completes forced activations; the enters-trigger checks of Merfolk Branchwalker, Echo Inspector, Geological Appraiser; how often each precondition binds.
