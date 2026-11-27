"""Builders the geometry-check tests share: converted sources with cache rows."""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from effects.domain.ability_cache_layout import ARRAY_KEY
from effects.domain.line_query import LineQuery
from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
from effects.domain.ward_twins import BARE_KEYWORDS, WARD, WARD_TWINS
from effects.infrastructure.sidecar_io import sidecar_path_for
from tests.unit.effects.file_builders import write_sidecar
from tests.unit.effects.selection_builders import read_cache


@dataclass(frozen=True)
class SourceRows:
    """One converted source's rows: the card, its stem and each line's text and vector.

    ``texts`` are the lines' script; ``prose``, when given, is written into the
    converted ``.txt`` beside the sidecar, one ``static:`` line per ability, so
    the two surfaces can differ the way they do on a real card.
    """

    card: str
    stem: str
    texts: list[str]
    vectors: list[list[float]]
    variant: str = "full"
    prose: list[str] | None = None
    tree: str = "cardsfolder"


def _write_source(cards_root, abilities_root, rows: SourceRows):
    """One converted source plus its cache rows, row-aligned."""
    card, stem, texts, vectors = rows.card, rows.stem, rows.texts, rows.vectors
    variant, prose, tree = rows.variant, rows.prose, rows.tree
    script_file = f"{tree}/{stem}.txt"
    write_sidecar(
        ProvenanceSidecar(
            card=card,
            script_file=script_file,
            lines=tuple(
                SidecarLine(
                    line_index=i, line_kind="static",
                    provenance=(ProvenanceKey(script_file, 0, "static", i),),
                    script_text=text, prose_offset=len("static: "),
                )
                for i, text in enumerate(texts)
            ),
        ),
        sidecar_path_for(cards_root / f"{stem}.txt"),
    )
    if prose is not None:
        (cards_root / f"{stem}.txt").write_text(
            "\n".join(f"static: {line}" for line in prose) + "\n",
            encoding="utf-8",
        )
    suffix = ".npz" if variant == "full" else f".{variant}.npz"
    target = abilities_root / tree / f"{stem}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target, **{ARRAY_KEY: np.array(vectors, dtype=np.float32)},
    )


def _two_cards_one_prose(tmp_path):
    """Two cards printing one prose line that compiles to two scripts."""
    cards = tmp_path / "cardsfolder"
    (cards / "a").mkdir(parents=True)
    abilities = tmp_path / "abilities"
    prose = [TAX_PROSE]
    _write_source(
        cards, abilities,
        SourceRows("one", "a/one", ["RaiseCost | Amount$ 2"], [[1.0, 0.0]], prose=prose),
    )
    _write_source(
        cards, abilities,
        SourceRows(
            "two",
            "a/two",
            ["RaiseCost | Amount$ 2 | Type$ Spell"],
            [[0.0, 1.0]],
            prose=prose,
        ),
    )
    return cards, abilities


TAX_PROSE = "spells your opponents cast that target CARDNAME cost {2} more to cast."


def _ward_cache(tmp_path, *, missing: tuple[LineQuery, ...] = ()):
    """Ward, its twins and the bare keywords, as the converter spells them."""
    cards = tmp_path / "cardsfolder"
    (cards / "w").mkdir(parents=True)
    abilities = tmp_path / "abilities"
    rng = np.random.default_rng(1)
    _write_source(
        cards, abilities,
        SourceRows(
            "Aboleth Spawn",
            "w/aboleth_spawn",
            ["Ward:2"],
            [[1.0, 0.0, 0.0]],
            prose=[WARD.prose],
        ),
    )
    for number, twin in enumerate(WARD_TWINS):
        if twin in missing:
            continue
        _write_source(
            cards, abilities,
            SourceRows(
                twin.card or f"twin {number}",
                f"w/twin_{number}",
                [f"Trigger | Twin$ {number}"],
                [[0.99, 0.14, 0.0]],
                prose=[twin.prose],
            ),
        )
    for number, keyword in enumerate(BARE_KEYWORDS):
        _write_source(
            cards, abilities,
            SourceRows(
                f"bare {number}",
                f"w/bare_{number}",
                [keyword.prose.title()],
                [rng.normal(size=3).tolist()],
                prose=[keyword.prose],
            ),
        )
    return read_cache(abilities, cards, surface="script")
