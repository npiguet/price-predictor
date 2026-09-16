# Castability consult: why every instant and sorcery came back uncastable

Branch `effects-coverage-deck-play`, `forge-connector/src/main/java/com/pricepredictor/connector/CastabilityMain.java`,
against the sibling Forge checkout at `../forge` (branch `effect-record-hooks`).

## Verdict: case 3 — the card id, not the accessor and not the owner or game as such

`CardFactory.getCard(IPaperCard, Player, Game)` derives the card id from the owner:

```java
// forge-game/src/main/java/forge/game/card/CardFactory.java:169-171
public static Card getCard(final IPaperCard cp, final Player owner, final Game game) {
    return getCard(cp, owner, owner == null ? -1 : owner.getGame().nextCardId(), game);
}
```

and `readCardFace` reads **none** of a card script's traits when that id is negative:

```java
// CardFactory.java:396-416 (readCardFace)
// Negative card Id's are for view purposes only
if (c.getId() >= 0) {
    ... setSVar / addReplacementEffect / addStaticAbility / addTrigger / addIntrinsicKeywords ...
    // add spells only after
    CardFactoryUtil.addAbilityFactoryAbilities(c, face.getAbilities());   // the A: lines
}
```

So `getCard(paper, null, null)` builds a display-only card: no SVars, no replacement effects, no statics, no
triggers, no keywords, and no `A:` lines — which is where an instant's or sorcery's `A:SP$ …` spell lives.
The only spell abilities such a card ever has are the ones `CardState.updateSpellAbilities`
(`CardState.java:482-540`) synthesises **at query time from the type line**, independent of the id:
`LandAbility` for a land (`isSpell()` is false), the aura spell for an aura, and `SpellPermanent` for every
other permanent. `Card.getSpellAbilities()` is just `currentState.getSpellAbilities()` (`Card.java:3416`),
which is the synthesised set plus the state's `abilities` list — empty for a negative id.

That is why the old rule was exactly "permanent ⇒ castable, everything else ⇒ uncastable", with no exception
thrown: `SpellPermanent extends Spell` (`isSpell()` true), an instant has nothing at all.

Forge itself uses the negative-id path deliberately — `Card.fromPaperCard(pc, null)` and
`Card.getCardForUi(pc)` (`Card.java:7619-7633`) are the deck-editor/card-detail view, which shows Oracle text
and never needs abilities.

### Why not case 1 (wrong accessor)

Every accessor — `getSpellAbilities`, `getNonManaAbilities`, `getSpells`, `getBasicSpells`,
`getFirstSpellAbility`, `getAllSpellAbilities` (all states), `CardState.getIntrinsicSpellAbilities` — reads the
same per-state `abilities` collection plus the synthesised ones. On an id −1 card that collection is empty for
every state, so no accessor can find an instant's spell. The probe's `abilities=0` before any filter already
said as much.

### Why not case 2 as stated (needs a real owner and/or game)

The owner's only role in `getCard` is to supply an id (`c.setOwner(owner)` is otherwise cosmetic here). The
abilities themselves are parsed by `AbilityFactory.getAbility(String, Card)`, which never touches the game, and
`buildAbilities` / `setupKeywordedAbilities` run regardless of owner. So an ownerless card is fine **given an
explicit non-negative id** through the public four-argument overload
`getCard(IPaperCard, Player, int cardId, Game)`.

A game is *almost* optional: with `game == null`, ability setup NPEs for exactly 3 of 33,232 cards
(Bloom Tender, Faeburrow Elder, Tarnation Vista — their setup calls `Game.getCardsIn(ZoneType)`), which the
`catch (RuntimeException)` would have turned into false uncastables. With a player-less dummy `Game` there are
zero exceptions. This repo already had the recipe: `RulesParser.buildFullCard` runs
`CardFactory.getCard(paperCard, null, nextCardId++, DummyGameHolder.INSTANCE)` over the whole corpus for
`convert`, where the dummy is `new Game(List.of(), rules, new Match(rules, List.of(), "DummyMatch"))`.

## The fix

`verdictFor` now builds the card as `CardFactory.getCard(paper, null, game.nextCardId(), game)` against a
lazily-created player-less `Game` (`ConsultGame.INSTANCE`, the `RulesParser` recipe) and answers
`card.getSpells().isEmpty() ? UNCASTABLE : CASTABLE` — `getSpells()` being the same "non-mana abilities with
`isSpell()`" filter the old loop hand-rolled. `consult`, `render`, the CLI and the `{name: verdict}` JSON the
Python side reads are unchanged, so `castability_connector.py` and `collect_coverage.py` were not touched.

## Census: before vs after, every unique card Forge knows

Throwaway probe (deleted after the run) over the 33,232 unique names in
`FModel.getMagicDb().getCommonCards().getUniqueCards()`, categorised by `CardRules.getType()`. "Before" is
the old code path replicated inline; "after" is the new one.

| category        |      n | before                     | after, dummy game            | after, null game                        |
|-----------------|-------:|----------------------------|------------------------------|-----------------------------------------|
| instant/sorcery |  7,296 | 7,296 uncastable           | **7,296 castable**           | 7,296 castable                          |
| land            |  1,156 | 1,156 uncastable           | 1,148 uncastable, 8 castable | 1,147 uncastable, 8 castable, 1 threw   |
| permanent       | 24,780 | 24,780 castable            | 24,780 castable              | 24,778 castable, 2 threw                |

- No card went castable → uncastable.
- Time over all 33,232 cards: old path 1.4 s, new path 4.3 s. The consult's "about a minute" is Forge start-up;
  the fix adds roughly three seconds to it.
- No "odd" category appears: planes, schemes, vanguards and conspiracies live in Forge's **variant** `CardDb`,
  not `getCommonCards()`, so `getRules(name, true)` returns null for them and they are omitted from the
  verdicts rather than judged — they are (part of) the 433 nulls the Python run saw. The 782 "odd" cards in the
  converted corpus were therefore never judged uncastable by this consult; they rank as `unknown` = castable.
- The 8 lands the engine calls castable are genuinely castable: **Zoetic Cavern** (`K:Morph:2`) and
  **Branch of Vitu-Ghazi** (`K:Disguise:3`) can be cast face down as 2/2 creature spells, and the six
  Final Fantasy Town lands (Ishgard, Jidoor, Lindblum, Midgar, Zanarkand, Value Town) are
  `AlternateMode:Adventure` lands whose sorcery/instant half goes on the stack.
- The Python run's population differs slightly (the 811 held-out cards are excluded; its names come from the
  converted files), so its exact counts will differ, but every consulted instant and sorcery flips to castable
  and about eight lands do too.

Raw output: `.superpowers/castability-census.txt`.

## The type-rule alternative, weighed

Rule: castable iff not a land and not an oddity type (conspiracy, scheme, plane, phenomenon, vanguard,
dungeon, emblem, contraption, attraction, sticker). Against the fixed engine consult over the same 33,232 cards:

- Agreement on 33,224 cards. The only disagreements are the 8 lands above (type rule: uncastable; engine:
  castable), and the engine is right about them.
- **Zero** cards where the type rule says castable and the engine says uncastable. The javadoc's case — "a card
  whose only spell is a variant Forge does not implement" — does not exist in Forge's common card DB. The
  nearest real things are (a) `CardRules.getUnsupportedCardNamed`, a placeholder for names *absent* from the
  DB, which `getRules` never returns (so such a name is omitted, not judged), and (b) functional variants of
  Un-cards, where an unsupported variant is simply not added to the set (`CardDb.java:492`). Neither yields an
  uncastable verdict. The javadoc example has been replaced with the real one (Adventure/Morph lands).
- A cheaper-still "does the script have an `A:SP$` line" rule would be wrong on 3 cards: Benediction of Moons,
  Cry of Contrition and Seize the Soul have no `A:SP$` line — their spell is synthesised from `K:Haunt` by
  `setupKeywordedAbilities`. Only the engine gets those right; the new test pins Benediction of Moons.
- Cost: the type rule could run Python-side from the converted text and skip the JVM entirely, saving the
  consult's ~1 minute of Forge start-up per `collect-coverage` run. The engine consult costs that minute plus
  ~3 s and is exact on 8 more cards today — and stays exact as Forge adds mechanics that put lands on the stack
  (Adventure lands only arrived with Final Fantasy in 2025).

**Recommendation: keep the engine consult, fixed as above.** A `collect-coverage` run is hours of games; a
minute of JVM start-up is noise, the consult is the question the collector actually asks, the contract and
connector already exist, and the type rule buys nothing but that minute while being wrong on the cards where
the two differ.

## Tests

- `CastabilityMainTest` (`@Tag("integration")`, `@ExtendWith(ForgeExtension.class)`), 9 tests: instant
  (Lightning Bolt), sorcery (Wrath of God), Haunt-synthesised spell (Benediction of Moons), creature, noncreature
  permanent, basic land, nonbasic land, unknown name omitted, and every recognised name gets exactly one verdict
  in request order (the "ranks, never drops" side of the contract). **Red first**: 3 failures against the old code
  (instant, sorcery, Haunt — `expected: <castable> but was: <uncastable>`), 6 passes; all 9 green after the fix.
- `CastabilityRenderTest` (Forge-free), 3 tests pinning the `{name: verdict}` JSON shape the Python side reads.
- Default suite `mvn -o test`: 781 tests, 0 failures. Python untouched (contract unchanged).
- The untracked throwaway `CastabilityProbeTest.java` and my census probe were deleted, not committed.

## Not settled

- The consult still does not ask whether the spell's *cost* is payable: a card with no mana cost (Ancestral
  Vision, Living End, Evermind, …) has a spell ability and is judged castable, yet only reaches the stack through
  suspend, cascade or splice. That is a false castable in the direction the class javadoc calls cheap ("a few
  wasted decks"), and the run's "castable but short" residue is where it surfaces. Not measured here.
- A back-face name resolves (via `allowAltNames`) to the front face's verdict — right for every DFC I can think
  of (the back is never cast), but not verified card by card.
- `.superpowers/` (this report, the census output, the two Maven logs) is untracked and left out of the commit.
