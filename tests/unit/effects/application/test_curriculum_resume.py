"""A run interrupted inside the curriculum's re-warm resumes to the run it
interrupted (finding 15.5, D10).

The re-warmed rate is a function of the step and of flags a resume compares,
and so is the averaged weights' start, so ``resume.pt`` stores nothing new
for them. Three fixture epochs of 7 steps, the switch at step 7 and a 10-step
re-warm: epoch 2 ends at step 14, inside the re-warm, where the average starts
(one epoch after the switch). Runs a real training loop: about five minutes.
"""

from __future__ import annotations

import pytest
import torch

from tests.unit.effects.application.test_training_loop_workers import (
    _assert_same_average,
    _run,
)

#: Steps 7 to 17 re-warm; epoch 2 ends at step 14.
REWARM = {"epochs": 3, "curriculum_rewarm_steps": 10}


def test_a_run_interrupted_inside_the_re_warm_resumes_to_the_same_weights(
    monkeypatch, corpus, tmp_path, caplog,
):
    whole = _run(monkeypatch, corpus, tmp_path / "whole", caplog, workers=0, **REWARM)
    out = tmp_path / "interrupted"
    first = _run(monkeypatch, corpus, out, caplog, workers=0, interrupt_after=2, **REWARM)
    assert first["average"]["start_step"] == 14
    second = _run(monkeypatch, corpus, out, caplog, workers=0, resume=True, **REWARM)

    assert first["losses"] + second["losses"] == whole["losses"]
    for name, value in whole["parameters"].items():
        assert torch.equal(value, second["parameters"][name]), name
    _assert_same_average(whole["average"], second["average"])
    # Epoch 3 moved the average away from the weights it started at.
    assert any(
        not torch.equal(value, whole["parameters"][f"{owner}.{name}"])
        for owner, tensors in whole["average"]["state"].items()
        for name, value in tensors.items()
    )


def test_a_resume_file_recording_no_re_warm_is_refused_by_name(
    monkeypatch, corpus, tmp_path, caplog,
):
    """A ``resume.pt`` written before the flag existed (D19)."""
    from effects.infrastructure.effect_model_store import (
        ResumeMismatchError,
        load_resume_state,
        save_resume_state,
    )

    out = tmp_path / "older"
    _run(monkeypatch, corpus, out, caplog, workers=0, interrupt_after=1, **REWARM)
    state = load_resume_state(out)
    del state["settings"]["curriculum_rewarm_steps"]
    save_resume_state(out, state)
    with pytest.raises(ResumeMismatchError, match="curriculum_rewarm_steps"):
        _run(monkeypatch, corpus, out, caplog, workers=0, resume=True, **REWARM)
