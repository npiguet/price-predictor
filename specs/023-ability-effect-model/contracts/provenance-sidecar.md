# Contract: provenance sidecar

**Feature**: `023-ability-effect-model`
Authority: [`../../2026-09-05-ability-effect-model.md`](../../2026-09-05-ability-effect-model.md) § Ability identity.

The sidecar is the join between runtime Forge trait objects and converted ability lines. It is the
single most load-bearing contract in the feature: every effect record names an ability through it, and
every cache row is aligned by it.

## Why converted-line ordinals cannot be the key

The converter's bracket numbers run one per-face counter across five mixed line kinds; keyword-derived
lines are emitted first in the file but appended last in Forge's runtime lists; and the converter
merges, deduplicates, and splits abilities relative to the runtime objects. An ordinal therefore names
different things on the two sides. The stable key is **printed provenance**.

## Key

```
ProvenanceKey = (script_file, face, trait_kind, index_within_kind, option?, built?)
```

- `script_file` includes its tree **and that tree's own layout**, which differ: `cardsfolder/` is
  letter-keyed and `tokenscripts/` is flat, so Ajani's Pridemate is
  `cardsfolder/a/ajanis_pridemate.txt` on one side and `tokenscripts/ajanis_pridemate.txt` on the
  other. The key is the path as written, produced identically by the Java writer and the Python
  reader — a mismatch is the fail-loudly case below.
- `index_within_kind` is the index within that kind's slice of the face's raw trait list, not a
  global ordinal.
- `option` (feature 024, FR-005a) names one mode of a charm: its 0-based position in the root's
  `Choices$`. It is written only when set (`{"face":0,"trait_kind":"spell","index_within_kind":0,
  "option":2}`), and a key carrying it never equals the root key without it. Records' `ability`
  lists and sidecars' `provenance` lists both carry it.
- `built` (feature 025) names one trait a keyword instance built: `{"kind", "index"}`, the trait's
  kind (`trigger`, `static`, `replacement` or `spell`, never `keyword`) and its 0-based position in
  that instance's list of that kind (`KeywordInterface.getTriggers()`, `getReplacements()`,
  `getStaticAbilities()`, `getAbilities()`). It is set only on a key whose `trait_kind` is
  `keyword`, and written only when set, after `option`
  (`{"face":0,"trait_kind":"keyword","index_within_kind":0,"built":{"kind":"trigger","index":2}}`,
  a saga's chapter III). A key carrying it never equals the keyword's own key without it, nor a key
  with another `built`. Records and sidecars carry it wherever they carry a key.

## File

`<name>.provenance.json`, written beside the converted `<name>.txt` — the same pairing convention the
`.npz` embedding files already use. **Writing it never alters the converted text.**

```jsonc
{
  "card": "ajani's pridemate",                // the first face's name as the converted text writes it
  "script_file": "cardsfolder/a/ajanis_pridemate.txt",
  "lines": [
    {
      "line_index": 3,                       // index into the converted file's rendered lines
      "line_kind": "triggered",
      "prose_offset": 11,                    // where the prose starts, after "triggered: "
      "provenance": [                        // several keys where the line merged several traits
        { "face": 0, "trait_kind": "trigger", "index_within_kind": 0 }
      ],
      "sub_ability_links": [                 // index paths below the trait
        { "path": [0], "label": "DBPutCounter" }
      ],
      "script_api_type": "PutCounter",
      "script_param_keys": ["Defined", "CounterType", "CounterNum"],
      "script_text": "DB$ PutCounter | Defined$ Self | CounterType$ P1P1 | CounterNum$ 1",
      "role_spans": [
        { "start": 0, "end": 34, "role": "trigger-condition" },
        { "start": 35, "end": 71, "role": "effect" }
      ]
    }
  ],
  "dropped_keys": [                          // traits that map to no rendered line
    { "face": 0, "trait_kind": "static", "index_within_kind": 2 }
  ]
}
```

`card` is the first face's name, lower-cased as the converted text writes it: `history of benalia`,
and a split card's left half alone (`fire` for Fire // Ice).

`dropped_keys` exists because the collector computes keys from runtime trait accessors, never from the
sidecar. A trait the converter deduplicated away is still live at runtime and still produces a key, so
without this list an expected dedup and a genuine corpus/sidecar mismatch would look identical at the
join — and the mismatch is the one condition that must fail loudly.

## Rules

| Rule | Consequence |
|---|---|
| A rendered line merged from several runtime traits carries **several** provenance keys | the join is many-to-one, never assumed one-to-one |
| A trait whose text a rendered line carries — merged, deduplicated, a secondary of a pair, keyword-derived, or a Class level — is claimed by that line | the join is many-to-one. A keyword-built trait is claimed through its keyword's key, by the rows below |
| `dropped_keys` holds the traits no line states | traits with no text (the implicit permanent spell on every permanent) and traits whose text the converter does not render (Orcish Farmer's activated ability, Sutured Ghoul's `ETBReplacement` keyword, a description-less sibling static such as Clutch of Undeath `static 1`) |
| A spell prints one line | its additional cost's sentence first (`as an additional cost to cast this spell, sacrifice a land.`), under a `cost` role span, then every sentence of its sub-ability chain in chain order; a die roll's outcomes are `option` lines below it. An instant's or sorcery's spell whose chain states no sentence states its own `Description$` (Garbage Fire), else the face's oracle lines no other line of the face states (Crashing Wave: the line after its waterbend cost); with neither (Into the Time Vortex, whose oracle is its cascade and rebound lines) it prints no line and its key is in `dropped_keys`. The line declares the spell's key. An Adventure or Omen spell is keyed on its own face only, never on the main face that lists it |
| A line that states only a cost (`alternate cost:`, `additional cost:`, `cost reduction:`) declares no `spell` key | a cleave line keeps its text and declares no key; the spell cast for cleave is claimed by the face's spell line, whose bracketed words state both versions. A permanent whose spell states nothing but an additional cost prints that cost on an `additional cost:` line, and its spell key is in `dropped_keys` with every permanent's |
| No key is declared by two lines of one face | a key names at most one line; a join never picks among lines. A reader refuses a sidecar that declares one (`SidecarError`, naming the file and the key) |
| A keyword that renders one line declares its bare keyword key on that line | its built traits join that line through the bare key |
| A keyword that renders several lines (a saga's chapters, haunt's two lines) declares on each line the `built` keys of the traits that line renders, and its bare keyword key on no line | each chapter joins its own line. A keyword whose built traits all lack a line declares its bare key on its first line |
| A keyword-built trait that renders a line of its own (Tribute's unpaid-tribute trigger) | that line declares the keyword's key with the trait's `built` |
| The kind-indexed keys (`trigger n`, `static n`, `spell n`, `replacement n`) of keyword-built traits | declared on no line and in no `dropped_keys`, and never emitted by the collector: a keyword-built trait is named only through its keyword's key |
| A key a line declares | resolves to that line |
| A `built` key no line declares | resolves as the same key without `built`: the keyword renders one line, and that line holds its text |
| A bare keyword key no line declares, on a keyword whose lines declare `built` keys | on a board line, resolves to all of those lines in line order, one slot each; as an acting key, resolves as `AMBIGUOUS`: no text, and counted |
| A record whose acting keys all sit in `dropped_keys` (or name an unconverted script) | `build-corpus` refuses it as `no-acting-text`; on the board, the surface builder emits no token for such a key. A runtime-only key is neither: it keeps the record and gets no token |
| A key names a script file with **no sidecar at all** under its tree | not an error: the converted corpus never held that card, and the ability contributes no text. Token entities do this — the collector files a token's provenance as `cardsfolder/<letter>/<sanitized token name>.txt` while tokens convert into `output/tokenscripts/` under Forge's own script filenames, so the path resolves to nothing. Readers raise `UnconvertedScript` and count it; `SidecarCache.line_for` returns None. Distinct from the row below, where the sidecar exists and disagrees |
| A record names a provenance key **inside** a declared `(face, trait_kind)` range that is in neither `lines` nor `dropped_keys` | fail loudly — the sidecar does not describe the card the record was collected against, i.e. a reconversion between collection and training. `dropped_keys` is declared-minus-claimed, so every index the converter parsed sits in one of the two lists and a gap between them can only come from a different trait list |
| A key whose face or trait kind the sidecar never declared, or whose index is past the highest it declared for that pair | runtime-only: a trait Forge attaches to a live card — level up, bestow, scavenge; a trigger granted by another card's static, keyed to the granting card's script; a trait a card granted itself whose granting line is not found; a disguise creature's face-down face. No line, the record is kept, and the surface builder emits no token |
| A record whose `ability` is empty and whose `ability_unresolved` is `engine_effect` | expected, not an error, and **not** a `dropped_keys` case: the acting card is engine-built (The Monarch, The Initiative, dungeons, emblems, speed) and has no script file in any tree, so there is no sidecar to drop from. A sentinel `script_file` would turn that honest answer into a hard join failure, inverting the fail-loudly rule above |
| An event attributed to a sub-ability link absent from `sub_ability_links` falls back to the **root line** | attribution never drops an event |
| `script_text` is the stage-four primary encoding surface | every command that encodes reads it here and needs no Forge-cardsfolder path of its own |
| `script_text` is the whole chain of the trait that first claimed the line (feature 024, FR-001–003) | on a line merged from several traits, the first claimant's chain alone (a dual land's mana line carries its first colour's mana ability); on a one-line keyword's line, the keyword's own script text (`Flying`, `Ward:2`), with `script_api_type` `Keyword`, except where the line is the text of a trait the keyword built (an `ETBReplacement` or `etbCounter` replacement, a saga whose chapters print one line, a Visit's or Backup's trigger, a `MayEffectFromOpeningHand` effect), which carries the first such trait's chain alone, and a Class level's line, which carries its level-up ability's chain followed by the chain of the static granting what the level adds; on a line a keyword's built traits claim with `built` keys, the first such trait's chain. A chain reads `seg₀ [SEG] SV1: seg₁ [SEG] SV2: seg₂ …`: the root segment, then the trigger's `Execute$` ability, every sub-ability in chain order including `RepeatSubAbility`, and a replacement's `ReplaceWith$` ability with its chain. A trigger or replacement Forge builds with its effect set and no `Execute$` or `ReplaceWith$` naming it (a chapter, a haunt trigger, an ETB replacement, a planeswalker's or saga's entering-with-counters replacement) carries that effect's chain as the segment after the root, opened by the next label; an ability with a sub-ability Forge attached and no `SubAbility$` naming it (Backup's granted abilities below its counter) carries that sub-ability's chain the same way. A parameter naming SVars that hold traits (a static's `AddTrigger$`, `AddStaticAbility$`, `AddReplacementEffect$`, `AddAbility$`; Animate's `Abilities$`, `Triggers$`, `Replacements$`, `staticAbilities$`; Effect's `Abilities$`, `Triggers$`, `StaticAbilities$`, `ReplacementEffects$`; a copy's `AddTriggers$`, `AddAbilities$`, `GainTextAbilities$`, `AddStaticAbilities$`) names each as a label, and each named trait is a segment of the chain with its own sub-abilities, in the order the parameter lists them. Each segment is its trait's parameters in key order and opens, after the root, with the label its parent referenced it by. Chain labels (values of `Execute$`, `SubAbility$`, `RepeatSubAbility$`, `ReplaceWith$`, each `Choices$` item of a chooser API, each SVar a granting parameter names, and segment openers) are renamed `SV1…` by first appearance, restarting per line; amount SVars are not renamed. Each SVar appears once per chain; one the script does not define is reported by `convert` and left out |
| A charm root line carries its own segment only | `Choices$ SV1,SV2,…`, no mode inlined |
| A charm root line is the kind of line its root is | a modal spell's is `spell`; an activated charm's is `activated` (or `planeswalker`) and opens with its cost under a `cost` role span (`activated[1]: {1}{G}, {T}: choose one`); its prose is the charm's header with the restriction Forge words (`{1}: choose one that hasn't been chosen this turn`), then its `AdditionalDescription$` whether or not that opens with a period (`choose one. if you control a commander…`), never a bare continuation; a charm whose count is an expression and that has no description prints the header "choose one" |
| A trigger executing a charm prints one `option` line per mode, with or without an `ABILITY` placeholder in its description | each mode line carries every root key the trigger line carries, plus its `option` (The Fantastic Four: `trigger 0 option m` and `trigger 1 option m`) |
| A charm-mode `option` line follows its root, in `Choices$` order | `provenance: [root key + option]`; `script_text` is the mode's chain opened `SV1:`; `script_api_type` is the mode's |
| A non-charm `option` line (a die-roll outcome of a spell, an activated or a triggered ability) | `provenance: []` and no `script_text`; no record acts through it |
| A chapter executing a charm prints one `option` line per mode below its chapter line | each mode line carries every root key the chapter line carries, plus its `option` |
| A modal spell with no description prints its synthesized root line | the line declares the spell's key, so no charm root is in `dropped_keys`; its option lines carry their mode keys |
| `role_spans` are character ranges over the **converted prose** | roles are `cost` \| `effect` \| `trigger-condition` \| `target-spec`; the line's kind picks them: a loyalty or level cost and a cost-only line (additional cost, alternate cost, cost reduction) are `cost`, and a trigger's condition starts after a leading ability word |
| The converted prose is the rendered line from `prose_offset` on | `prose_offset` is the length of the line's `kind[n]: ` prefix (ASCII). A description that runs onto several lines is folded into one, each line break read as a space, so every line has its prefix. Readers take the split from the field and never re-derive it from the text, whose prose may hold a `": "` of its own (`{T}: add {G}.`). A sidecar written before the fold, whose continuation lines repeat their first line's keys, is refused as declaring a key twice. A sidecar without the field is read by cutting at the first `": "` |
| Row `i` of the ability cache corresponds to `lines[i]` | rendered lines for a converted tree, script lines for the variant tree |

## Runtime-side key computation

Keys come from the trait accessors, verified present in Forge 2.0.15-SNAPSHOT
(`forge-game/src/main/java/forge/game/CardTraitBase.java`, and `SpellAbility` for the last):

| Accessor | Use |
|---|---|
| `getCardState()` | the printed face |
| `isIntrinsic()` | printed vs granted |
| `getKeyword()` | the originating keyword, where the trait came from one |
| `getOriginalHost()` | the card whose state hosts the trait. **For a granted trait this is the recipient, not the donor** — `CardTraitBase.getOriginalHost()` returns `getCardState().getCard()` — so it must not be used to key granted abilities; use the grantor accessors below |
| `isCopiedTrait()` | copy detection |
| `getGrantorStatic()` | **`SpellAbility` only**, and does not survive copies |
| `getTrigger()` | the trigger a wrapped or `Execute$`-SVar ability belongs to |
| `getReplacementEffect()` | the replacement effect an ability belongs to |
| `getRootAbility()` | the head of a sub-ability chain |
| `getOriginalMapParams()` | the structural fingerprint a `copy()` preserves, since `putParam` writes only `mapParams` |
| `Card.getEffectSourceAbility()` | the ability that created an effect card, which owns its printed line |

Fallback chain, in order. Each step is tried only after the ones above it miss, and the structural
match refuses to answer when it is ambiguous — **no key beats a wrong key**:

1. **Unwrap** a triggered ability's wrapper, then a sub-ability to its root.
2. **Trigger- and replacement-borne abilities** key to the trait that owns them (`getTrigger()`,
   `getReplacementEffect()`), not to the `Execute$` SVar ability, which is a member of no trait slice.
3. **Keyword-derived traits** key to their keyword's ordinal (see below).
4. **The trait where it stands**, located by identity in its own `CardState` slice. A trait the slice holds
   and the state's own traits (`get...WithoutChanges()`) do not, or a stack copy of one (`getOriginalAbility()`),
   is one the card granted itself (a manland's `Animate` grants an activated ability so): it keys to its
   granting line, the line naming the SVar it was parsed from, and to its index
   only when no such line is found. Several lines naming the SVar are accepted when they key alike, or differ only in `built`, which
   gives the bare key: a Class level's trait keys to its `Class:N` keyword's bare key.
5. **The trait where it stands**, located by *structure*: the unique member of the slice with the same
   class and the same `getOriginalMapParams()`. This is what reaches an alternative-cost or
   extra-keyword-cost copy, which carries no back-reference at all.
6. **Granted abilities** resolve through the grantor accessors to the donor card's printed line.
7. **Copied abilities** resolve through the original-ability back-reference.
8. **Effect cards** resolve through `Card.getEffectSourceAbility()` to the ability that created them.
9. Give up, and record *why* in the record's `ability_unresolved`.

Step 4 alone was the whole chain's identity test in stage one, and it fails for every activated ability
the stack hands back: `MagicStack.add` replaces a non-mana activated ability with a fresh copy before
pushing it, so identity finds nothing and the key is null. Mana abilities return before that copy,
which is why land mana abilities keyed and Fountain of Youth did not.

### Keyword ordinals

`index_within_kind` for `trait_kind = keyword` is the position of the keyword's printed text in the
trait's own `CardState.getIntrinsicKeywords()` **sorted by that text**. Not the position in
`getKeywords()`: `KeywordCollection` is a `MultimapBuilder.hashKeys()` over the `Keyword` enum, whose
`hashCode()` is the identity hash, so that order differs between the convert JVM and the collect JVM —
an ordinal neither side could reproduce. Restricting it to intrinsics also correctly refuses a *granted*
keyword, which names no printed line on the card that received it.

Both sides compute it from one shared function, `FaceOrder.keywordIndex`, the same discipline
`faceIndex`/`faceOrder` already enforce for faces. The two are coupled: the moment the runtime emits
`keyword@N`, the sidecars must have been rebuilt by the fixed converter, or every such record trips the
fail-loudly rule below.

## Trees

| Tree | Source | Sidecar written by |
|---|---|---|
| `cardsfolder` | `../forge/forge-gui/res/cardsfolder/` → `output/cardsfolder/` | `price_predictor convert` |
| `tokenscripts` | `../forge/forge-gui/res/tokenscripts/` → `output/tokenscripts/` | `price_predictor convert` |
| `variant-scripts` | perturbed scripts → `output/effects/variant-scripts/` | `effects collect-variants` (stage four) |

Token scripts stay out of `output/cardsfolder/` because converted token and card filenames collide and
the sealed pipeline treats that tree as its card corpus.

## Rejected alternative

**Normalized-text matching** between runtime description text and converted lines was considered as a
no-converter-change fallback and rejected: it silently fails exactly where the converter merged or
deduplicated lines, which is precisely the case the sidecar exists to record.
