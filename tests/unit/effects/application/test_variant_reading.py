"""A variant line reads its script on the prose surface (finding 1.5, D4).

A variant card has no prose, so it is encoded from its script. Read through
the prose grammar against a vocabulary that never saw a script, every API name
and parameter key was ``[UNK]`` (19.1% of a variant's tokens), and its
``SpellDescription$`` stated the original number in readable words after the
perturbed one. ``build-vocab --surface prose`` now stages the trees' script
lines, and a line with no prose reads its script, free text dropped, through
the script grammar.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from effects.application.build_vocab import BuildVocabConfig, run
from effects.domain.ability_encoder import encoding_text
from effects.domain.ability_tokenizer import Surface
from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
from effects.domain.script_text import RELABELLED_KEYS
from effects.domain.tokenizer_files import TokenizerFiles
from effects.infrastructure.sidecar_io import sidecar_path_for
from effects.infrastructure.tokenizer_loading import load_tokenizer
from tests.unit.effects.builders import write_sidecar

_FORGE_CARDS = Path(__file__).resolve().parents[4].parent / "forge/forge-gui/res/cardsfolder"

#: Burn spells as the gen-3 converter writes them: Shock's 2 and two cards
#: stating the 4 its variant is perturbed to.
_BURN = {"s/shock": 2, "l/lightning_blast": 4, "f/fireblast": 4}


def _script(amount: int) -> str:
    return (
        f"NumDmg$ {amount} | SP$ DealDamage | "
        f"SpellDescription$ CARDNAME deals {amount} damage to any target. | ValidTgts$ Any"
    )


def _write_card(cards: Path, stem: str, prose: list[str], scripts: list[str]) -> None:
    """One converted card: its ``.txt`` with ``prose`` as ``spell`` lines, and
    a sidecar whose lines carry ``scripts``."""
    name = stem.split("/")[-1].replace("_", " ")
    text = f"name: {name}\ntypes: instant\n" + "\n".join(
        f"spell[{row + 1}]: {line}" for row, line in enumerate(prose))
    path = cards / f"{stem}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    script_file = f"cardsfolder/{stem}.txt"
    write_sidecar(
        ProvenanceSidecar(card=name, script_file=script_file, lines=tuple(
            SidecarLine(
                line_index=2 + row, line_kind="spell",
                provenance=(ProvenanceKey(script_file, 0, "spell", row),),
                script_text=script, prose_offset=len(f"spell[{row + 1}]: "),
            )
            for row, script in enumerate(scripts)
        )),
        sidecar_path_for(path),
    )


def _keywords(tmp_path: Path) -> Path:
    path = tmp_path / "keyword-definitions.json"
    path.write_text(json.dumps({}), encoding="utf-8")
    return path


def _prose_tokenizer(cards: Path, tmp_path: Path):
    """``build-vocab --surface prose`` over ``cards``, and the tokenizer that
    reads it."""
    keywords = _keywords(tmp_path)
    vocab = tmp_path / "vocab.txt"
    run(BuildVocabConfig(cards_folders=(cards,), vocab_path=vocab,
                         keyword_definitions=keywords, surface=Surface.PROSE))
    return load_tokenizer(TokenizerFiles.of(vocab, keywords))


class _Variant:
    """A variant's sidecar line: a script and no prose."""

    role_spans = ()

    def __init__(self, script_text: str) -> None:
        self.script_text = script_text


class TestAShockVariant:
    def test_it_reads_its_perturbed_number_and_no_unknown_token(self, tmp_path):
        cards = tmp_path / "cardsfolder"
        for stem, amount in _BURN.items():
            _write_card(cards, stem, [f"CARDNAME deals {amount} damage to any target."],
                        [_script(amount)])
        tokenizer = _prose_tokenizer(cards, tmp_path)
        variant = _script(2).replace("NumDmg$ 2", "NumDmg$ 4")

        text = encoding_text(_Variant(variant), None, Surface.PROSE)
        tokens = tokenizer.encoding_tokens(text)
        words = [token.text for token in tokens]

        unknown = [word for word, token in zip(words, tokens, strict=True)
                   if token.token_id == tokenizer.unk_id]
        assert not unknown, f"the variant reads [UNK] for {unknown}: {words}"
        assert "4" in words, f"the perturbed number is not read: {words}"
        pairs = list(zip(words, words[1:]))
        assert ("2", "damage") not in pairs and "deals" not in words, (
            f"the variant still states the original's 2 damage in its description: {words}"
        )


# ── 500 real script lines ───────────────────────────────────────────────

#: An ability line of a Forge source script, its prefix cut.
_ABILITY_LINE = re.compile(r"^(?:A|T|S|R):(.*\$.*)$|^SVar:[^:]+:(.*\$.*)$")
_RELABELLED = re.compile(
    rf"(?<![A-Za-z])({'|'.join(sorted(RELABELLED_KEYS))})\$[ \t]*([^|]+)")
_SAMPLED_SCRIPTS = 2500
_SAMPLED_LINES = 500


def _relabelled(line: str) -> str:
    """``line`` with every chain label renamed ``SV#``, as the converter writes it."""
    labels: dict[str, str] = {}

    def rename(match: re.Match) -> str:
        items = [labels.setdefault(item.strip(), f"SV{len(labels) + 1}")
                 for item in match.group(2).split(",") if item.strip()]
        return f"{match.group(1)}$ {','.join(items)} "

    return _RELABELLED.sub(rename, line).strip()


def _forge_lines(paths: list[Path]) -> dict[Path, list[str]]:
    """Each script's ability lines, relabelled."""
    found: dict[Path, list[str]] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = [_relabelled(match.group(1) or match.group(2))
                 for match in map(_ABILITY_LINE.match, text.splitlines()) if match]
        if lines:
            found[path] = lines
    return found


@pytest.fixture(scope="module")
def forge_sample() -> dict[Path, list[str]]:
    if not _FORGE_CARDS.is_dir():
        pytest.skip(f"no Forge checkout at {_FORGE_CARDS}")
    rng = random.Random(20261117)
    paths = rng.sample(sorted(_FORGE_CARDS.rglob("*.txt")), _SAMPLED_SCRIPTS)
    return _forge_lines(paths)


class TestRealScriptLines:
    def test_under_one_percent_of_their_tokens_are_unknown(self, forge_sample, tmp_path):
        """The lines of a sample of Forge's scripts, staged as their sidecars
        would carry them, then 500 of them read as variant lines."""
        cards = tmp_path / "cardsfolder"
        for number, lines in enumerate(forge_sample.values()):
            _write_card(cards, f"f/card_{number}", ["CARDNAME deals damage."] * len(lines), lines)
        tokenizer = _prose_tokenizer(cards, tmp_path)
        every = [line for lines in forge_sample.values() for line in lines]
        drawn = random.Random(1305).sample(every, _SAMPLED_LINES)

        total = unknown = 0
        for line in drawn:
            for token in tokenizer.encoding_tokens(
                    encoding_text(_Variant(line), None, Surface.PROSE)):
                total += 1
                unknown += token.token_id == tokenizer.unk_id
        share = unknown / total
        assert share < 0.01, (
            f"{share:.2%} of the tokens of {_SAMPLED_LINES} real script lines are [UNK] "
            f"({unknown} of {total})"
        )
