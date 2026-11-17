# golden-024: pinned card scripts and the converted text they must produce

Feature 024 changes the sidecar's `script_text` to the whole chained ability (FR-001) and keys
charm modes (FR-005a) while leaving the converted `.txt` byte-identical (FR-004, SC-001).
This directory holds a fixed set of card scripts and the converted text expected of them, so
the byte-identity test (`BatchConverterTest`) has something fixed to compare against, and the
tests that read the sidecar side of a conversion (`TraitScriptChainTest`, `KeywordLineKeysTest`,
`ProvenanceSidecarTest`) run on scripts an upstream card edit cannot move.

Layout:

- `cardsfolder/<letter>/<name>.txt` — source scripts; copies of `../forge/forge-gui/res/cardsfolder/`
  as of 2026-10-07 (History of Benalia, Blind Hunter, Censor, Ranger Class, Balloon
  Stand, Harrow, Abeyance, Contact Other Plane, Dig Up, Selvala's Charge, Crashing Wave,
  Garbage Fire and Tyranid Invasion as of 2026-11-17; Pick-a-Beeble, Lost Mine of Phandelver and Undercity as of 2026-11-18), plus the four synthetic scripts marked below. The tree is letter-keyed like
  Forge's so every provenance key reads `cardsfolder/<letter>/<name>.txt`.
- `tokenscripts/<name>.txt` — token scripts, flat like Forge's token tree.
- `expected/cardsfolder/…`, `expected/tokenscripts/…` — the `.txt` that `python -m price_predictor convert`
  writes for them: the byte-identity reference. The gen-1 sidecars are not kept: no test
  compares them, and the gen-2 sidecars differ by design.

The `expected/` tree was first captured with the pre-024 converter and is maintained by hand
since: a converter change that is meant to alter the converted text edits the affected
expected lines in the same commit, and any other difference is a regression. There is no
regeneration switch, because regenerating would accept whatever the converter now writes.

One line was re-captured by hand: `expected/cardsfolder/t/tundra.txt`'s synthetic mana line.
The pre-024 converter rendered a dual land's colours in the iteration order of a `Map.of`,
which the JVM salts per run, so the first capture read `{U} or {W}` while the shipped
`output/cardsfolder/` corpus (and every other dual land in it) reads `{W} or {U}`. The
converter now renders colour-wheel order deterministically, and the golden carries the
corpus's order.

The synthetic mana lines of `m/mountain.txt` and `t/tundra.txt` were later edited by hand to
end in a period, when the synthesized land-mana line took the shape of a scripted mana line
(`{T}: add {W} or {U}.`, with the serial comma from three colours on).

When a spell came to print one line (task A04), the expected text of `l/label_a.txt` and
`l/label_b.txt` (two sentences, once two `spell` lines) and `r/ranger_class.txt` (level 3's
continuation line folded into the level line) was edited to match.

When a face's keyword lines came to print in the sorted order of the keywords' original text (task A16), the expected text of `b/blast_from_the_past.txt` was edited to match: only its line order changed.

When a sentence the description already holds stopped being appended again (task A14, finding 1.12), the expected text of `a/aberrant_mind_sorcerer.txt` was edited to match: its trigger line says "roll a d20." once. When a haunt line came to name its card as CARDNAME, `b/blind_hunter.txt`'s haunted-dies line was edited likewise.

| Script | Why it is here |
|---|---|
| `a/anticausal_vestige.txt` | trigger with `Execute$` → `SubAbility$` chain (story 1 scenario 1); `Warp` keyword |
| `a/acidic_soil.txt` | `RepeatEach` with `RepeatSubAbility$` |
| `l/leader_super_genius.txt` | `ReplaceWith$ DBDraw` whose ability has a sub-ability (scenario 3); plus a trigger |
| `s/swans_of_bryn_argoll.txt` | replacement with a single replacing segment; `Flying` |
| `c/cryptic_command.txt` | spell charm, four modes (scenarios 2 and 11) |
| `a/abiding_grace.txt` | triggered charm (`TriggerDescription$ … ABILITY`) |
| `l/life_of_toshiro_umezawa_memory_of_toshiro.txt` | chapter charm on a Saga; double-faced |
| `s/season_of_gathering.txt` | Pawprint charm; one mode is a `GenericChoice` with its own `Choices$` |
| `d/doomsday_confluence.txt` | descriptionless charm: option lines are the card's only lines |
| `a/aberrant_mind_sorcerer.txt` | die-roll `option` lines that are not charm modes |
| `l/lonely_end.txt` | two modes naming the same sub-ability |
| `b/blood_on_the_snow.txt` | charm root with its own `SubAbility$`; modes sharing a sub-ability |
| `a/aboleth_spawn.txt` | `Ward:2`, `Flash` keyword lines |
| `s/serra_angel.txt` | `Flying`, `Vigilance` |
| `p/pacifism.txt` | `Enchant:Creature` |
| `b/blast_from_the_past.txt` | five keyword lines (keyword ordinal coupling) |
| `b/blightsteel_colossus.txt` | `K:` lines in reverse alphabetical order: the converted lines come out sorted (task A16) |
| `g/glorious_anthem.txt` | static ability |
| `f/fountain_of_youth.txt` | activated ability |
| `a/ajanis_pridemate.txt` | simple trigger (the sidecar contract's worked example) |
| `w/wall_of_omens.txt` | ETB trigger |
| `l/lightning_bolt.txt`, `c/counterspell.txt` | plain spells |
| `m/mogg_war_marshal.txt` | echo keyword trigger; "enters or dies" trigger pair |
| `s/stadium_tidalmage.txt` | "enters or attacks" trigger pair |
| `h/hunters_talent.txt`, `b/bard_class.txt` | Class cards (level lines; keyword-derived replacement) |
| `d/daybreak_ranger_nightfall_predator.txt` | double-faced card, two faces of keys |
| `f/fire_ice.txt` | split card |
| `p/paralyze.txt` | aura with a trigger and a replacement |
| `m/mountain.txt`, `t/tundra.txt` | synthetic land mana line claiming runtime mana abilities |
| `h/history_of_benalia.txt` | saga whose "I, II" line prints two chapter triggers (keys of a multi-line keyword) |
| `b/blind_hunter.txt` | haunt creature: five haunt triggers over two lines |
| `c/censor.txt` | `Cycling`: a single-line keyword that builds a spell |
| `r/ranger_class.txt` | Class level 3 whose text runs onto a second line, folded into one |
| `h/harrow.txt` | spell with an additional cost: the cost sentence opens the spell's one line |
| `a/abeyance.txt` | spell whose chain states two sentences: one line |
| `c/contact_other_plane.txt` | spell die roll: unkeyed `option` result rows |
| `d/dig_up.txt` | cleave: an alternate-cost line declaring no key, its spell claimed by the spell line |
| `s/selvalas_charge.txt` | `PrecostDesc$ Parley —`: an ability word Forge reports as a cost description, which is no additional cost |
| `b/balloon_stand.txt` | Attraction `Visit` charm: a single-line keyword whose built trigger has modes |
| `c/crashing_wave.txt` | spell with no `SpellDescription$`: its line is the oracle sentence the waterbend static's line does not state |
| `g/garbage_fire.txt` | spell with no `SpellDescription$`: its line is its own `Description$` |
| `t/tyranid_invasion.txt` | spell with no `SpellDescription$` and no other line: its keyed line is the whole oracle, not an unkeyed `text:` line |
| `tokenscripts/c_a_food_sac.txt` | token script, flat tree |
| `tokenscripts/lost_mine_of_phandelver.txt`, `tokenscripts/undercity.txt` | dungeons: one line per room trigger, each declaring its `built` key (Undercity also has a static keyword) |
| `p/pick_a_beeble.txt` | Attraction `Prize`: a single-line keyword whose line is its built trigger's description |
| `u/undefined_svar.txt` | **synthetic** — `TrigDraw` names `SubAbility$ DBMissing`, which no `SVar:` defines (FR-003). The task asked for an undefined `Execute$`, but Forge fails the whole card at parse for that shape (`Error in Trigger for Card`), so no converter can produce output for it; `SubAbility$` is the one undefined reference Forge tolerates |
| `l/label_a.txt`, `l/label_b.txt` | **synthetic** — identical chains naming their sub-ability `DBDraw` and `DBDrawCard` (scenario 12) |
| `r/repeat_loop.txt` | **synthetic** — a `RepeatSubAbility$` chain that reaches an SVar the root chain also names, so the once-per-chain rule is exercised |
