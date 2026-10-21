"""A provenance key's one-string form, the spelling manifests and surveys key on."""

from __future__ import annotations

from effects.domain.provenance import ProvenanceKey, parse_keys, render_keys

_ROOT = ProvenanceKey("cardsfolder/c/cryptic_command.txt", 0, "spell", 0)
_MODE = ProvenanceKey("cardsfolder/c/cryptic_command.txt", 0, "spell", 0, option=2)
_TOKEN = ProvenanceKey("tokenscripts/role_wicked.txt", 1, "static", 3)


def test_a_key_renders_to_the_frozen_spelling():
    assert _ROOT.render() == "cardsfolder/c/cryptic_command.txt|0|spell|0"
    assert _MODE.render() == "cardsfolder/c/cryptic_command.txt|0|spell|0|2"


def test_a_key_tuple_renders_its_keys_joined_by_semicolons():
    assert render_keys((_TOKEN, _MODE)) == (
        "tokenscripts/role_wicked.txt|1|static|3;"
        "cardsfolder/c/cryptic_command.txt|0|spell|0|2"
    )


def test_a_rendered_key_parses_back_to_the_same_key():
    for key in (_ROOT, _MODE, _TOKEN):
        assert ProvenanceKey.parse(key.render()) == key
    assert parse_keys(render_keys((_ROOT, _MODE, _TOKEN))) == (_ROOT, _MODE, _TOKEN)


def test_a_mode_key_and_its_root_never_parse_equal():
    assert parse_keys(_MODE.render()) != parse_keys(_ROOT.render())
