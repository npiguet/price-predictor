# Research: CardMarket EUR Pricing

**Feature**: 004-cardmarket-eur-pricing | **Date**: 2026-03-01

## R-001: Existing CardMarket EUR Implementation

**Decision**: The codebase already implements CardMarket EUR price extraction. No changes needed for FR-001–FR-006, FR-009, FR-010.

**Rationale**: Code audit confirms:
- `build_price_map()` navigates `paper → cardmarket → retail → {normal, foil}` (lines 101–112 of `mtgjson_loader.py`)
- `get_cheapest_price()` uses the identical path (lines 65–76)
- No references to TCGPlayer, USD, or other vendors exist anywhere in `src/`
- Domain entities already use EUR field names: `actual_price_eur`, `predicted_price_eur`, `price_range_min_eur`, `price_range_max_eur`
- Feature 001's spec was already clarified to use EUR/CardMarket (session 2026-02-27)

**Alternatives considered**: Full reimplementation with explicit vendor selection config — rejected as unnecessary complexity (violates Constitution II: Simplicity First).

## R-002: Exclusion Count Computation (FR-008)

**Decision**: Compute exclusion count as `len(name_to_uuids) - len(result)` after the price map loop completes, and log it as an INFO message.

**Rationale**: `build_price_map()` receives `name_to_uuids` (all card names with UUIDs) and produces `result` (only cards with valid prices). The difference is exactly the count of cards excluded due to missing CardMarket prices. This requires no additional data structures — just a subtraction of two already-available values.

**Alternatives considered**:
- Tracking excluded cards by name in a separate list — rejected as unnecessary (we only need the count, not the names)
- WARNING-level log — rejected per clarification (INFO is consistent with feature 003's logging pattern)

## R-003: Feature 003 Price Floor Interaction

**Decision**: FR-007 (zero-price inclusion) is compatible with feature 003's €0.01 floor. Zero-priced cards are included in the price map (not excluded as "missing"), then clamped to €0.01 by the existing floor logic.

**Rationale**: The `price is not None and price >= 0` filter (from feature 003) already allows zero prices through. The `max(min(all_prices), 0.01)` floor then clamps them. This is exactly the behavior FR-007 requires: "included in training set" with "final training label €0.01." No code change needed.

**Alternatives considered**: None — this interaction was resolved in the clarification session.

## R-004: Test Fixture Coverage

**Decision**: The existing test fixtures (`allprices_sample.json`) contain only CardMarket EUR data. No TCGPlayer entries exist in fixtures. For US3 (exclusion testing), a card entry in `allprintings_sample.json` that has no corresponding price entry in `allprices_sample.json` will serve as the "excluded card" test case.

**Rationale**: The fixture already contains `eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee` (Swords to Plowshares) with an empty price object `{}`. However, this card is filtered out by `build_name_to_uuids()` because it has `isOnlineOnly: true`. To test FR-008, we need a card that passes the name-to-UUID filter but has no price in the price data. We can either add a new fixture entry or use an existing card whose UUID is not in the price file.

**Alternatives considered**: Adding TCGPlayer entries to fixtures to test FR-003 — rejected because the code never reads TCGPlayer paths, so there's nothing to test. The absence of TCGPlayer handling IS the correct behavior.
