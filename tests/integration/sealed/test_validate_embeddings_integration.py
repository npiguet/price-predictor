"""T010 — Integration test for ValidateEmbeddingsUseCase."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sealed.application.validate_embeddings import ValidateEmbeddingsUseCase
from sealed.domain.embedding_probe import ValidationResult

# Use small embedding dim so linear probes are well-conditioned on small fixture data
EMBED_DIM = 16

# Embedding layout (structured fixtures):
#  dim 0:  is_land signal
#  dim 1:  W pip count (actual value * scale)
#  dim 2:  U pip count
#  dim 3:  B pip count
#  dim 4:  R pip count
#  dim 5:  G pip count
#  dim 6:  C pip count
#  dim 7:  mana value
#  dim 8:  produces W signal
#  dim 9:  produces U signal
#  dim 10: produces B signal
#  dim 11: produces R signal
#  dim 12: produces G signal
#  dim 13: produces C signal
#  dim 14-15: noise

_SIGNAL = 10.0  # signal amplitude — dominates any noise


def _write_card(base_dir: Path, stem: str, text: str, embedding: np.ndarray) -> None:
    subdir = base_dir / stem[0]
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / f"{stem}.txt").write_text(text, encoding="utf-8")
    np.savez_compressed(subdir / f"{stem}.npz", embedding=embedding)


def _make_emb(
    rng: np.random.Generator,
    *,
    is_land: bool,
    pip_w: float = 0, pip_u: float = 0, pip_b: float = 0,
    pip_r: float = 0, pip_g: float = 0, pip_c: float = 0,
    mv: float = 0,
    prod_w: bool = False, prod_u: bool = False, prod_b: bool = False,
    prod_r: bool = False, prod_g: bool = False, prod_c: bool = False,
) -> np.ndarray:
    """Build a structured 16-dim embedding with strong feature signals."""
    emb = rng.standard_normal(EMBED_DIM).astype(np.float32) * 0.01
    emb[0]  = _SIGNAL if is_land else -_SIGNAL
    emb[1]  = pip_w * _SIGNAL
    emb[2]  = pip_u * _SIGNAL
    emb[3]  = pip_b * _SIGNAL
    emb[4]  = pip_r * _SIGNAL
    emb[5]  = pip_g * _SIGNAL
    emb[6]  = pip_c * _SIGNAL
    emb[7]  = mv * _SIGNAL * 0.2   # scale down to stay in reasonable range
    emb[8]  = _SIGNAL if prod_w else -_SIGNAL
    emb[9]  = _SIGNAL if prod_u else -_SIGNAL
    emb[10] = _SIGNAL if prod_b else -_SIGNAL
    emb[11] = _SIGNAL if prod_r else -_SIGNAL
    emb[12] = _SIGNAL if prod_g else -_SIGNAL
    emb[13] = _SIGNAL if prod_c else -_SIGNAL
    return emb


def _make_structured_cards(base_dir: Path, rng: np.random.Generator) -> None:
    """Write 150+ cards with structured embeddings that strongly encode mana features.

    Each card's embedding faithfully represents its ground truth labels so that
    linear probes can achieve near-perfect cross-validated scores.
    """
    cards: list[tuple[str, str, np.ndarray]] = []

    # Basic lands (10 copies each = 60 land cards)
    basic_lands = [
        ("plains",   "types: basic land plains",  dict(prod_w=True)),
        ("island",   "types: basic land island",  dict(prod_u=True)),
        ("swamp",    "types: basic land swamp",   dict(prod_b=True)),
        ("mountain", "types: basic land mountain",dict(prod_r=True)),
        ("forest",   "types: basic land forest",  dict(prod_g=True)),
        ("wastes",   "types: basic land",         dict(prod_c=True)),
    ]
    for name, types, prod in basic_lands:
        color = name[0].upper() if name != "wastes" else "C"
        add_sym = {"plains": "W", "island": "U", "swamp": "B",
                   "mountain": "R", "forest": "G", "wastes": "C"}[name]
        for i in range(10):
            stem = f"{name}_{i:02d}"
            text = (
                f"name: {name} {i}\n"
                f"{types}\n"
                f"activated[1]: {{T}}: add {{{add_sym}}}.\n"
            )
            emb = _make_emb(rng, is_land=True, mv=0, **prod)
            cards.append((stem, text, emb))

    # Single-color spells — varied pip counts (1, 2, 3 pips) for good R²
    for color_name, pip_key in [("W","pip_w"),("U","pip_u"),("B","pip_b"),
                                  ("R","pip_r"),("G","pip_g"),("C","pip_c")]:
        for pip_count in [1, 2, 3]:
            for i in range(6):
                generic = pip_count  # add generic to vary mana value
                cost = "{" + f"{generic}" + "}" + ("{" + color_name + "}" * pip_count)
                mv = generic + pip_count
                stem = f"spell_{color_name.lower()}_{pip_count}pip_{i:02d}"
                text = (
                    f"name: {color_name} spell {pip_count}pip {i}\n"
                    f"mana cost: {cost}\n"
                    f"types: instant\n"
                )
                emb = _make_emb(rng, is_land=False, mv=mv, **{pip_key: float(pip_count)})
                cards.append((stem, text, emb))

    # Mana dorks (creatures that produce mana)
    dork_data = [
        ("llanowar_elves", "mana cost: {G}\ntypes: creature elf druid\n"
         "activated[1]: {T}: add {G}.\n",
         dict(pip_g=1, mv=1, prod_g=True)),
        ("sol_ring",       "mana cost: {1}\ntypes: artifact\n"
         "activated[1]: {T}: add {C}{C}.\n",
         dict(mv=1, prod_c=True)),
        ("birds_of_paradise", "mana cost: {G}\ntypes: creature bird\n",
         dict(pip_g=1, mv=1)),
    ]
    for name, body, kw in dork_data:
        for i in range(8):
            stem = f"{name}_{i:02d}"
            text = f"name: {name} {i}\n{body}"
            emb = _make_emb(rng, is_land=False, **kw)
            cards.append((stem, text, emb))

    # Write all cards
    for stem, text, emb in cards:
        _write_card(base_dir, stem, text, emb)


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestValidateEmbeddingsIntegration:
    def test_structured_embeddings_return_20_results(self, tmp_path):
        rng = np.random.default_rng(42)
        _make_structured_cards(tmp_path, rng)

        use_case = ValidateEmbeddingsUseCase()
        result = use_case.execute(tmp_path)

        assert isinstance(result, ValidationResult)
        assert len(result.probe_results) == 20

    def test_structured_embeddings_reports_card_counts(self, tmp_path):
        rng = np.random.default_rng(42)
        _make_structured_cards(tmp_path, rng)

        result = ValidateEmbeddingsUseCase().execute(tmp_path)

        assert result.n_cards >= 50
        assert result.n_lands >= 6  # at least the basic lands

    def test_structured_embeddings_score_higher_than_random(self, tmp_path):
        """Structured embeddings must score significantly higher than random ones.

        This verifies the probes are meaningful: structured embeddings (which
        directly encode the target features) should consistently outscore random
        noise. We check the mean score across all 20 probes is ≥ 0.10 higher
        than random, confirming the validation is not a rubber stamp.
        """
        rng = np.random.default_rng(42)
        structured_dir = tmp_path / "structured"
        random_dir = tmp_path / "random"
        structured_dir.mkdir()
        random_dir.mkdir()

        # Build structured cards
        _make_structured_cards(structured_dir, rng)

        # Build random-embedding cards with the same texts
        rng_rand = np.random.default_rng(99)
        structured_cards = sorted(structured_dir.rglob("*.txt"))
        for txt_path in structured_cards:
            npz_path = structured_dir / txt_path.relative_to(structured_dir).with_suffix(".npz")
            rand_dir = random_dir / txt_path.parent.name
            rand_dir.mkdir(exist_ok=True)
            emb = rng_rand.standard_normal(EMBED_DIM).astype(np.float32)
            np.savez_compressed(rand_dir / npz_path.name, embedding=emb)
            (rand_dir / txt_path.name).write_text(txt_path.read_text(encoding="utf-8"), encoding="utf-8")

        use_case = ValidateEmbeddingsUseCase()
        structured_result = use_case.execute(structured_dir)
        random_result = use_case.execute(random_dir)

        structured_mean = sum(r.score for r in structured_result.probe_results) / len(structured_result.probe_results)
        random_mean = sum(r.score for r in random_result.probe_results) / len(random_result.probe_results)

        assert structured_mean >= random_mean + 0.10, (
            f"Structured embeddings (mean={structured_mean:.3f}) should score "
            f"≥ 0.10 above random (mean={random_mean:.3f})"
        )

    def test_random_embeddings_fail_majority_of_categories(self, tmp_path):
        """SC-002: random embeddings should fail, confirming validation isn't a rubber stamp."""
        rng = np.random.default_rng(42)

        # Generate the same structured card texts as the structured test, but with random embeddings
        basic_lands = [
            ("plains",   "types: basic land plains",   "W"),
            ("island",   "types: basic land island",   "U"),
            ("swamp",    "types: basic land swamp",    "B"),
            ("mountain", "types: basic land mountain", "R"),
            ("forest",   "types: basic land forest",   "G"),
            ("wastes",   "types: basic land",          "C"),
        ]
        colors = [("W", "pip_w"), ("U", "pip_u"), ("B", "pip_b"),
                  ("R", "pip_r"), ("G", "pip_g"), ("C", "pip_c")]

        cards = []
        for name, types, add_sym in basic_lands:
            for i in range(10):
                stem = f"{name}_{i:02d}"
                text = f"name: {name} {i}\n{types}\nactivated[1]: {{T}}: add {{{add_sym}}}.\n"
                cards.append((stem, text))

        for color_name, _ in colors:
            for pip_count in [1, 2, 3]:
                for i in range(6):
                    generic = pip_count
                    cost = "{" + f"{generic}" + "}" + ("{" + color_name + "}" * pip_count)
                    stem = f"spell_{color_name.lower()}_{pip_count}pip_{i:02d}"
                    text = (f"name: {color_name} spell {pip_count}pip {i}\n"
                            f"mana cost: {cost}\ntypes: instant\n")
                    cards.append((stem, text))

        for stem, text in cards:
            emb = rng.standard_normal(EMBED_DIM).astype(np.float32)  # no signal
            _write_card(tmp_path, stem, text, emb)

        result = ValidateEmbeddingsUseCase().execute(tmp_path)

        assert not result.all_passed, "Random embeddings should not pass all probes"

        # Count failed categories (not just probes)
        categories = {"Is land", "Card color", "Pip counts", "Mana value", "Mana produced"}
        failed_categories = set()
        for r in result.probe_results:
            if not r.passed:
                for cat in categories:
                    if r.feature_name.startswith(cat):
                        failed_categories.add(cat)
                        break

        assert len(failed_categories) >= 3, (
            f"Expected ≥3 failing categories (SC-002), got {len(failed_categories)}: "
            f"{failed_categories}"
        )

    def test_insufficient_cards_raises_value_error(self, tmp_path):
        """Fewer than 50 paired cards should raise ValueError."""
        rng = np.random.default_rng(0)
        # Write only 5 cards
        for i in range(5):
            stem = f"card_{i:03d}"
            text = f"name: card {i}\nmana cost: {{R}}\ntypes: instant\n"
            emb = rng.standard_normal(EMBED_DIM).astype(np.float32)
            _write_card(tmp_path, stem, text, emb)

        with pytest.raises(ValueError, match="Insufficient data"):
            ValidateEmbeddingsUseCase().execute(tmp_path)
