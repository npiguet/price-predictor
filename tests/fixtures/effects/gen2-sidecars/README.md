Gen-2 sidecars and converted texts, copied from a full `python -m price_predictor convert`
run of the feature-024 converter: chained, label-renamed `script_text`
(`seg0 [SEG] SV1: seg1 [SEG] SV2: seg2`) and keyed charm `option` lines. Real output, never
hand-built. Shared by the holdout, value-target and option-row tests.

Cards several tests lean on:

- `c/cryptic_command` — a charm root (`Choices$ SV1,SV2,SV3,SV4`) and four keyed mode lines
- `d/doomsday_confluence` — a descriptionless charm
- `a/arc_trail` — damage stated in two segments (2 + 1)
- `a/abraded_bluffs` — a trigger whose `Execute$` segment deals the damage
- `b/bone_splinters` — an additional sacrifice cost with mana in `Cost$`
- `p/prodigal_sorcerer` — a tap-cost activated ability

The tree also holds the sidecars every record of `../gen2-records.jsonl.gz` names. That fixture is
cut from the real gen-2 shards by `python scripts/make_fixture_records.py --generation 2`, three
records per category: the shared kinds and subkinds, plus a 704.5f state-based death at toughness 0
or less (Saproling Burst's tokens, Barrenton Medic, Decree of Pain), a creature lowered to toughness
1 that stays, a death by damage, the legend rule, a planeswalker at loyalty 0, an Aura whose host
left, triggers a keyword generates (fired and not), a predefined token kind made (111.10) and an
ability whose source is one, a damage-step probe stripping deathtouch, records from
random-seat and Forge-AI games, and the shapes the detector families' parameters read: a
resolution and a decision candidate acting through a keyword's own line, a candidate whose card
prints a keyword, a candidate whose `Cost$` has a part (`Sac<…>`), a candidate listing legal
targets, a damage rewrite a prevention shield replaced or reduced, a creature leaving the
battlefield while attacking or blocking, a token's static (the Monster Role's) and a static whose
`IsPresent$` condition reads counters (Captain America's). Rerunning the script rewrites the records file and adds sidecars here; it never edits a
sidecar already present.

Fifteen cards were since replaced by the current converter's output (sidecar and converted text),
because their old sidecars declared one key on two lines of a face, which a sidecar reader now
refuses: the sagas `d/death_in_heaven`, `s/summon_shiva`, `v/vault_21_house_gambit` and
`h/huatli_poet_of_unity_roar_of_the_fifth_people` (its face 1), whose chapter lines now declare
`built` keys, and eleven spells whose additional-cost line declared the spell's key beside the
spell line (`a/abjure`, `b/bone_splinters`, `c/charge_of_the_forever_beast`, `e/endemic_plague`,
`f/final_flare`, `i/infernal_plunge`, `n/new_blood`, `r/reckless_abandon`,
`r/ritual_of_the_machine`, `s/soulblast`, `t/tormented_thoughts`), now one spell line. The records
naming these cards keep their old keys: a saga's bare `keyword 0` acting key now resolves as
ambiguous.
