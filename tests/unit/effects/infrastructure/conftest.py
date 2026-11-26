import pytest

from tests.unit.effects.domain.conftest import ability_key, make_record, snapshot  # noqa: F401


@pytest.fixture
def files(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n", encoding="utf-8")
    keywords = tmp_path / "keyword-definitions.json"
    keywords.write_text("{}", encoding="utf-8")
    return vocab, keywords
