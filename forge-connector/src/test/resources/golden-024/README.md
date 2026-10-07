# golden-024: converter output captured before the gen-2 chain renderer

Feature 024 changes the sidecar's `script_text` to the whole chained ability (FR-001) and keys
charm modes (FR-005a) while leaving the converted `.txt` byte-identical (FR-004, SC-001).
This directory holds the inputs and the outputs of the converter **as it was before that
change**, so the byte-identity test (`BatchConverterTest`) has something fixed to compare
against and the chain tests (`TraitScriptChainTest`) run on the same scripts.

Layout:

- `cardsfolder/<letter>/<name>.txt` — source scripts; copies of `../forge/forge-gui/res/cardsfolder/`
  as of 2026-10-07, plus the four synthetic scripts marked below. The tree is letter-keyed
  like Forge's so every provenance key reads `cardsfolder/<letter>/<name>.txt`.
- `tokenscripts/<name>.txt` — one token script, flat like Forge's token tree.
- `expected/cardsfolder/…`, `expected/tokenscripts/…` — what `python -m price_predictor convert`
  wrote for them before the change: the `.txt` (the byte-identity reference) and the gen-1
  `.provenance.json` (kept for comparison; the gen-2 sidecars differ by design).

Regenerate the `expected/` tree only with the pre-024 converter; after it, the `.txt` files
must still match and the sidecars are the new contract.

One line was re-captured by hand: `expected/cardsfolder/t/tundra.txt`'s synthetic mana line.
The pre-024 converter rendered a dual land's colours in the iteration order of a `Map.of`,
which the JVM salts per run, so the first capture read `{U} or {W}` while the shipped
`output/cardsfolder/` corpus (and every other dual land in it) reads `{W} or {U}`. The
converter now renders colour-wheel order deterministically, and the golden carries the
corpus's order.

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
| `tokenscripts/c_a_food_sac.txt` | token script, flat tree |
| `u/undefined_svar.txt` | **synthetic** — `TrigDraw` names `SubAbility$ DBMissing`, which no `SVar:` defines (FR-003). The task asked for an undefined `Execute$`, but Forge fails the whole card at parse for that shape (`Error in Trigger for Card`), so no converter can produce output for it; `SubAbility$` is the one undefined reference Forge tolerates |
| `l/label_a.txt`, `l/label_b.txt` | **synthetic** — identical chains naming their sub-ability `DBDraw` and `DBDrawCard` (scenario 12) |
| `r/repeat_loop.txt` | **synthetic** — a `RepeatSubAbility$` chain that reaches an SVar the root chain also names, so the once-per-chain rule is exercised |
