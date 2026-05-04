"""Verify the missing-default-file guard message wording (FR-026)."""

from __future__ import annotations

from pathlib import Path

from sealed.infrastructure.cli import _missing_sealed_encoder_message


class TestMissingSealedEncoderMessage:
    def test_message_names_path_and_train_encoder_command(self):
        path = Path("models/sealed/encoder/latest.pt")
        msg = _missing_sealed_encoder_message(path)
        assert "Sealed encoder not found" in msg
        assert str(path) in msg
        assert "python -m sealed train-encoder" in msg
        assert "--encoder-checkpoint" in msg
