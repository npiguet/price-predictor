# Textless abilities: design record

Written 2026-09-17 from an audit of the curated corpus at `output/effects/corpus/`
(built 02:44 the same day) and the trainer's warning about unresolved token
scripts. Extends spec `specs/023-ability-effect-model/spec.md`; the FR amendments
are listed at the end. One plan implements it:

- `docs/superpowers/plans/2026-09-17-textless-abilities.md`

## What the audit showed

Every finding below was measured on random samples of the curated training shards
with the sidecars under `output/cardsfolder/` and `output/tokenscripts/`, using
the trainer's own `SidecarCache`.

1. **The head reads a zero-vector ability token on almost every entity.** Over 4
   shards (6,000 records, 185,719 entities), 248,676 of 308,621 ability slots
   (81%) carried a zero vector, and 89% of entities carried at least one. All but
   three were `spell` keys the converter lists as dropped. The cause is Forge's
   implicit "cast this permanent" object (`SpellPermanent`), which the engine
   attaches to every non-land permanent, tokens included, and which the converter
   correctly renders as no line. Lands carry two: the play-land object and the
   mana ability. The surface builder emits a token for every printed key without
   asking whether the key has a line, so the phantom rides along in every zone
   the snapshot records: battlefield 184,937 times, hand 35,934, graveyard 22,820,
   stack 1,622.

2. **Three quarters of resolution records act with no text.** Over 10 shards,
   4,378 of 6,045 resolution records had an acting key that maps to no line:
   3,578 were permanent spells resolving (a creature moving from the stack to the
   battlefield), 752 were basic lands tapping for mana, about 40 were real
   abilities whose key the converter dropped, and 4 named a key the sidecar
   does not describe at all. FR-148 refuses a resolution record with no acting
   *key*; these all have a key. The sampler then weights a textless record at
   1.0, the weight of a text seen in one game, so the resolution class's draws
   are dominated by "creature spell enters the battlefield" with an empty acting
   slot.

3. **Basic-land mana text exists but is unreachable.** The converted Mountain
   carries the line `activated[1]: {T}: add {R}` with an empty provenance list,
   and its sidecar lists both `spell 0` (play land) and `spell 1` (the mana
   ability) as dropped. `RulesParser` skips the runtime mana ability for a face
   with basic-land subtypes and appends a synthetic line that claims no key.

4. **The converter drops keys for real abilities in three more ways.** A
   trigger Forge registers twice for "enters or dies" / "enters or attacks"
   renders once, and only the first object's key is claimed (Mogg War Marshal,
   Stadium Tidalmage, Gathering Stone, Shrine of Loyal Legions). A
   keyword-derived trigger or static (echo's upkeep trigger, for instance)
   returns no entry, so its key is dropped although the keyword line carries the
   text. A Class card's level lines are rebuilt as fresh `TextAbility` objects
   after attribution, so every ability on Hunter's Talent or Scavenger's Talent
   is textless.

5. **A key in neither list is swallowed rather than raised.** The batcher's
   `text_of`, the trainer's `text_for_key` and the corpus builder's fold all
   catch `KeyError` around the sidecar lookup, so the contract's fail-loudly
   case (the sidecar does not describe the card the record was collected
   against) reads as "no text". Four records in the 10-shard sample were in
   that state (Exit Specialist, Dog Walker, Rising Chicane).

The token warning that started the audit (`307 ability scripts the converted
corpus does not hold`) is finding 1 seen from the other side: the old collector
filed a token's implicit cast-spell key under a card-tree path that does not
exist, so the lookup is counted instead of silently dropped. The key has no text
in either tree. No recollection recovers anything.

## Decisions

1. **The surface builder skips a key that resolves to no rendered line.** A
   phantom is not board content. The zero token stays only for a line a control
   variant masks out, so the `state-only` and `no-state` variants keep their
   geometry. (`build_effect_head_input` gains a `has_line` predicate.)

2. **A sidecar mismatch fails loudly everywhere.** The batcher, the trainer's
   key-to-text fold and the corpus builder catch only the two expected
   no-text cases (an unconverted script, an unconfigured tree) and let the
   neither-list `KeyError` propagate. `SidecarCache.path_for` raises a distinct
   `UnconfiguredTree` so the two can be told apart from a mismatch.

3. **`build-corpus` refuses a resolution record whose acting keys all map to no
   line.** A key the sidecar lists as dropped, or naming an unconverted script,
   is "no text". A key past what the face declared (level up, bestow, scavenge)
   keeps the record, because that key is a runtime addition rather than a
   contentless object and its text lives on a keyword line the join does not
   reach yet. The count lands in `quality_dropped["no-acting-text"]`, and the
   manifest records the refused count per script file so a converter regression
   that drops a whole card family shows up in the build output.

4. **The converter claims every key whose text it renders.** The synthetic
   land-mana line claims the runtime mana abilities it stands for; the survivor
   of a deduplicated description absorbs the duplicate's keys; the primary of
   an "attacks or blocks" pair absorbs the secondary's key; a keyword-derived
   trigger or static attributes its key to the keyword's line; a Class card's
   rebuilt level line inherits the attribution of the ability it replaces. The
   rendered text does not change, which keeps the sidecar's byte-identity
   guarantee.

5. **Reconvert and rebuild, do not recollect.** Keys are runtime ordinals, so
   existing records join to the regenerated sidecars. The corpus is rebuilt
   after decisions 3 and 4 because the resolution class shrinks by more than
   half and the rarity table is keyed by text.

## Order of operations

Decision 3 refuses on "no text", so it must run against sidecars that already
carry decision 4's joins. Applied the other way round, the builder throws away
every mana activation, an eighth of the resolution class, and nothing warns.
The runbook at the end of the plan pins the order: rebuild the JAR, reconvert,
audit, rebuild the corpus, audit again, train.

## Out of scope

- Mapping a runtime-only key (level up, bestow, scavenge) to its keyword line.
  Kept records, counted separately; a later join improvement.
- The gate-one stratum and the rarity table already exclude textless records by
  construction (they key on text), so neither changes.

## FR amendments

- **FR-073** gains: a key that resolves to no rendered line contributes no
  token; a line a variant masks keeps its token.
- **FR-148** gains a third refusal reason, `no-acting-text`, with the
  runtime-only exception, and a manifest field `no_acting_text_scripts`.
- **FR-152** (new): the converter claims, on a rendered line, every runtime
  trait whose text that line carries — synthetic land mana, deduplicated
  descriptions, secondary triggers of a pair, keyword-derived traits, and
  Class level lines — and `dropped_keys` holds only traits with no text.
- The provenance-sidecar contract's rule table is updated to match: a
  deduplicated trait joins the surviving line; `dropped_keys` is the
  contentless set; `build-corpus` refuses a resolution record acting only
  through dropped keys.
